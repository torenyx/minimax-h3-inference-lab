import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model


SOURCE = Path(os.environ['H3_INT8_TP_SOURCE']).resolve()
DEST = Path(os.environ['H3_INT8_TP_SHARD_CACHE']).resolve()
SPLITTER = Path(inspect.getsourcefile(MiniMaxH3Model)).resolve()
SIGNATURE = {
    'schema': 'h3-int8-convrot-tp2-v1',
    'tp_size': 2,
    'source': {
        'path': str(SOURCE),
        'size': SOURCE.stat().st_size,
        'mtime_ns': SOURCE.stat().st_mtime_ns,
    },
    'splitter': {
        'path': str(SPLITTER),
        'size': SPLITTER.stat().st_size,
        'mtime_ns': SPLITTER.stat().st_mtime_ns,
    },
}
SIGNATURE_JSON = json.dumps(SIGNATURE, sort_keys=True)


def shard_is_current(rank):
    path = DEST / f'rank{rank}.safetensors'
    if not path.is_file():
        return False
    with safe_open(path, framework='pt', device='cpu') as checkpoint:
        metadata = checkpoint.metadata() or {}
        return metadata.get('h3_tp2_signature') == SIGNATURE_JSON and metadata.get('tp_rank') == str(rank)


DEST.mkdir(parents=True, exist_ok=True)
if all(shard_is_current(rank) for rank in range(2)):
    manifest = DEST / 'manifest.json'
    if manifest.is_file() and json.loads(manifest.read_text())['signature'] == SIGNATURE:
        print('H3_TP2_SHARDS_REUSED', flush=True)
        raise SystemExit(0)

for rank in range(2):
    model = SimpleNamespace(
        config={'tensor_parallel': True},
        tp_size=2,
        tp_rank=rank,
        _tp_split_type=MiniMaxH3Model._tp_split_type,
    )
    tensors = {}
    elements = 0
    with safe_open(SOURCE, framework='pt', device='cpu') as source:
        keys = list(source.keys())
        for index, key in enumerate(keys, 1):
            tensor = source.get_tensor(key)
            local = MiniMaxH3Model._select_tensor_parallel_shard(model, key, tensor)
            tensors[key] = local.contiguous()
            elements += local.numel()
            if index % 100 == 0:
                print(f'H3_TP2_SHARD_PROGRESS rank={rank} tensors={index}/{len(keys)}', flush=True)

    temporary = DEST / f'.rank{rank}.{os.getpid()}.partial.safetensors'
    output = DEST / f'rank{rank}.safetensors'
    save_file(tensors, str(temporary), metadata={
        'h3_tp2_signature': SIGNATURE_JSON,
        'tp_rank': str(rank),
    })
    with safe_open(temporary, framework='pt', device='cpu') as saved:
        if set(saved.keys()) != set(tensors):
            raise RuntimeError(f'Rank {rank}: saved tensor keys do not match')
        for key, expected in tensors.items():
            actual = saved.get_tensor(key)
            if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
                raise RuntimeError(f'Rank {rank}: shard verification failed for {key}')
    temporary.replace(output)
    print(f'H3_TP2_SHARD_READY rank={rank} tensors={len(tensors)} elements={elements} bytes={output.stat().st_size}', flush=True)

manifest_data = {
    'signature': SIGNATURE,
    'tensors_per_rank': len(keys),
    'shards': [
        {'rank': rank, 'file': f'rank{rank}.safetensors', 'size': (DEST / f'rank{rank}.safetensors').stat().st_size}
        for rank in range(2)
    ],
}
temporary_manifest = DEST / f'.manifest.{os.getpid()}.partial'
temporary_manifest.write_text(json.dumps(manifest_data, indent=2))
temporary_manifest.replace(DEST / 'manifest.json')
if not all(shard_is_current(rank) for rank in range(2)):
    raise RuntimeError('TP2 shard metadata verification failed')
print('H3_TP2_SHARDS_VERIFIED ' + json.dumps(manifest_data, sort_keys=True), flush=True)
