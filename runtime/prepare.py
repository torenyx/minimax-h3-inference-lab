"""Prepare reusable native components, adapter caches and TP2 weight shards."""
import json
import os
from pathlib import Path
import runpy
import sys

from runtime.accelerator_compat import install
install()
from runtime.compat import enable
enable()

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from runtime import adapters
from runtime.checkpoint_io import Reader


def embedding_grid(model, output):
    from lightx2v.models.networks.minimax_h3.infer.pre_infer import timestep_embedding
    reader = Reader(sorted((model / 'transformer').glob('*.safetensors')))
    w1, b1, w2, b2 = [reader.tensor('time_embedder.' + key).float()
                      for key in ['proj_in.weight', 'proj_in.bias', 'proj_out.weight', 'proj_out.bias']]
    t = torch.linspace(0, 1, 1025)
    hidden = F.linear(timestep_embedding(t, w1.shape[1]).float(), w1, b1)
    grid = F.silu(F.linear(F.silu(hidden), w2, b2))
    save_file({'silu_t_emb_grid': grid.to(torch.bfloat16).contiguous()}, str(output))
    reader.close()


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    out, cache, model = (Path(spec[k]) for k in ['output_dir', 'cache_dir', 'model_dir'])
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    fl = spec['profile'] == 'fl2va-turbo'
    config = json.loads((Path(__file__).parents[1] / 'configs/lightx2v.json').read_text())
    native = cache / 'native-model'
    native.mkdir(exist_ok=True)
    for component in ['transformer', 'text_encoder', 'tokenizer', 'processor', 'audio_vae']:
        link = native / component
        if not link.exists():
            link.symlink_to(model / component, target_is_directory=True)
    vae = Path(spec['vae_dir']) if spec.get('vae_dir') else model / 'vae'
    if not vae.exists():
        os.environ.update(H3_CACHE_DIR=str(cache), H3_ORIGINAL_MODEL=str(model))
        runpy.run_module('runtime.prepare_vae', run_name='__main__')
        vae = cache / 'video-vae'
    if not (native / 'vae').exists():
        (native / 'vae').symlink_to(vae, target_is_directory=True)
    repack = cache / 'repacked'
    repack.mkdir(exist_ok=True)
    weight = Path(spec['int8_checkpoint']) if spec.get('int8_checkpoint') else repack / 'int8-convrot.safetensors'
    if spec.get('int8_checkpoint') and not weight.exists():
        raise FileNotFoundError(weight)
    if not weight.exists():
        os.environ.update(H3_PRUNED_CHECKPOINT=spec['pruned_checkpoint'], H3_LIGHT_REPACK_DIR=str(repack))
        runpy.run_module('runtime.repack', run_name='__main__')
    config.update(model_variant='fl2av' if fl else 'ref2av', task=spec['task'] if fl else 'ref2av',
                  infer_steps=spec['steps'], num_frames=spec['num_frames'], size=[spec['height'], spec['width']],
                  adaln_cache_dir=str(out / 'adaln-cache'), dit_quantized_ckpt=str(weight),
                  dit_original_ckpt=str(repack), lora_dynamic_apply=True,
                  lora_merge_device='cpu', lora_merge_force_fp32=True,
                  tensor_parallel=True, seq_parallel=spec['gpus'] == 4,
                  parallel={'tensor_p_size': 2, 'seq_p_size': spec['gpus'] // 2, 'seq_p_attn_type': 'ulysses'},
                  use_fused_qkv_norm_rope=True, qkv_norm_rope_type='triton',
                  video_flow_shift=6.0 if spec['profile'] == 'ref2va-turbo' else 12.0,
                  attn_type=spec['attention'])
    backbone = out / ('turbo-backbone.safetensors' if fl else 'pdd-backbone.safetensors')
    adapter = Path(spec['adapter'])
    if spec['profile'] == 'ref2va-turbo':
        backbone = adapter
    config['lora_configs'] = [{'path': str(backbone), 'strength': 1.0, 'alpha': spec['alpha']}]
    (repack / 'config.json').write_text(json.dumps(config, indent=2))
    adapters.KIND = 'pdd' if spec['profile'] == 'ref2va-pdd' else 'turbo'
    adapters.QUANTIZED = True
    adapters.PRUNED = Path(spec['pruned_checkpoint'])
    adapters.ORIGINAL_TRANSFORMER = model / 'transformer'
    if fl:
        grid = Path(spec['embedding_grid']) if spec.get('embedding_grid') else cache / 'h3_silu_temb_grid.safetensors'
        if not grid.exists():
            embedding_grid(model, grid)
        adapters.LORA, adapters.BACKBONE_LORA, adapters.EGRID = adapter, backbone, grid
        adapters.prepare_backbone_lora()
        adapters.build_turbo_adaln_cache(config)
    else:
        if adapters.KIND == 'pdd':
            adapters.pdd_backbone(adapter, backbone)
        adapters.build_pruned_cache(config, adapter)
    os.environ.update(H3_INT8_TP_SOURCE=str(weight), H3_INT8_TP_SHARD_CACHE=str(cache / 'tp2-shards'))
    try:
        runpy.run_module('runtime.prepare_shards', run_name='__main__')
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'prompt.txt').write_text(Path(spec['prompt_file']).read_text(encoding='utf-8'), encoding='utf-8')
    print('H3_PREPARATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
