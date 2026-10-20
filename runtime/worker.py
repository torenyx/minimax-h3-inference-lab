"""One torchrun worker; patches the pinned LightX2V implementation in-process."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

spec = json.loads(Path(sys.argv[1]).read_text())
out = Path(spec['output_dir'])
os.environ.update(H3_INT8_TP_SHARD_DIR=str(Path(spec['cache_dir']) / 'tp2-shards'),
                  H3_FF_CHUNK_TOKENS='8192', H3_RMS_CHUNK_ROWS='4096', H3_OUTPUT_DIR=str(out))
from runtime.accelerator_compat import install
install()
from runtime.compat import enable
enable()
from runtime import pre_sharded, memory

if spec['profile'] != 'ref2va-turbo':
    from runtime import resident
if spec['profile'] == 'fl2va-turbo':
    from runtime import fl2va_decode
elif spec['profile'] == 'ref2va-pdd':
    from runtime.adapters import attach_pdd_heads
    from lightx2v.models.runners.minimax_h3 import minimax_h3_runner
    original_build = minimax_h3_runner.build_minimax_h3_model_with_lora

    def build(config, model_kwargs, lora_configs):
        model = original_build(config, model_kwargs, lora_configs)
        attach_pdd_heads(model, spec['adapter'])
        return model

    minimax_h3_runner.build_minimax_h3_model_with_lora = build

argv = ['lightx2v.infer', '--model_cls', 'minimax_h3',
        '--model-variant', 'fl2av' if spec['profile'] == 'fl2va-turbo' else 'ref2av',
        '--task', spec['task'] if spec['profile'] == 'fl2va-turbo' else 'ref2av',
        '--model_path', str(Path(spec['cache_dir']) / 'native-model'),
        '--config_json', str(out / 'config.json'), '--prompt', (out / 'prompt.txt').read_text(encoding='utf-8'),
        '--save_result_path', str(out / 'trial.mp4'), '--seed', str(spec['seed']),
        '--size', str(spec['height']), str(spec['width']), '--num_frames', str(spec['num_frames'])]
for key, option in [('image', '--image_path'), ('video', '--video_path'), ('audio', '--audio_path')]:
    if spec[key]:
        argv.extend([option, ','.join(spec[key])])
if spec.get('last_frame'):
    argv.extend(['--last_frame_path', spec['last_frame']])
sys.argv = argv
from lightx2v.infer import main
main()
