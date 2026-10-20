import inspect
import json
import os
import runpy
from pathlib import Path

import torch
from safetensors import safe_open

from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from runtime.checkpoint_names import normalize_final_norm


SHARD_DIR = Path(os.environ['H3_INT8_TP_SHARD_DIR']).resolve()
MANIFEST = json.loads((SHARD_DIR / 'manifest.json').read_text())
SIGNATURE = MANIFEST['signature']
SOURCE = Path(SIGNATURE['source']['path'])
SPLITTER = Path(inspect.getsourcefile(MiniMaxH3Model)).resolve()


def current_file(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


expected = {
    'schema': 'h3-int8-convrot-tp2-v1',
    'tp_size': 2,
    'source': current_file(SOURCE),
    'splitter': current_file(SPLITTER),
}
if SIGNATURE != expected:
    raise RuntimeError('Pre-sharded INT8 TP2 cache is stale; rebuild it before inference')

original_load = MiniMaxH3Model._load_quant_ckpt


def load_pre_sharded(self, unified_dtype, sensitive_layer):
    if not self.use_tp:
        return original_load(self, unified_dtype, sensitive_layer)
    if self.tp_size != 2:
        raise RuntimeError(f'Pre-sharded INT8 checkpoint requires TP2, got {self.tp_size}')

    path = SHARD_DIR / f'rank{self.tp_rank}.safetensors'
    weight_dict = {}
    load_device = self._checkpoint_load_device()
    remove_keys = self.remove_keys if hasattr(self, 'remove_keys') else []
    with safe_open(path, framework='pt', device='cpu') as source:
        metadata = source.metadata() or {}
        if metadata.get('h3_tp2_signature') != json.dumps(SIGNATURE, sort_keys=True):
            raise RuntimeError(f'Rank {self.tp_rank} shard metadata does not match its manifest')
        if metadata.get('tp_rank') != str(self.tp_rank):
            raise RuntimeError(f'Rank {self.tp_rank} received the wrong pre-sharded checkpoint')
        for key in source.keys():
            if any(remove_key in key for remove_key in remove_keys):
                continue
            tensor = source.get_tensor(key)
            if torch.device(load_device).type != 'cpu':
                tensor = tensor.to(load_device)
            weight_dict[key] = tensor
    normalize_final_norm(weight_dict)
    self._validate_checkpoint_devices(weight_dict, load_device)
    print(f'H3_TP2_PRESHARDED_LOADED rank={self.tp_rank} tensors={len(weight_dict)} device={load_device}', flush=True)
    return weight_dict


MiniMaxH3Model._load_quant_ckpt = load_pre_sharded
print(f'H3_INT8_TP_SHARD_DIR={SHARD_DIR}', flush=True)
