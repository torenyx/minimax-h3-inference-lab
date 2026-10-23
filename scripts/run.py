#!/usr/bin/env python3
"""Launch the packaged LightX2V profiles with user-provided inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['fl2va-turbo', 'ref2va-pdd', 'ref2va-turbo'], required=True)
    p.add_argument('--gpus', type=int, choices=[2, 4], default=2)
    p.add_argument('--paths', type=Path, help='JSON with model_dir, pruned_checkpoint, adapter and embedding_grid')
    p.add_argument('--model-dir', type=Path, help='Official FL2VA or Ref2VA component directory')
    p.add_argument('--pruned-checkpoint', type=Path, help='Comfy-layout pruned INT8 ConvRot checkpoint')
    p.add_argument('--int8-checkpoint', type=Path, help='Optional already-repacked LightX2V INT8 checkpoint')
    p.add_argument('--adapter', type=Path, help='Released Turbo or PDD safetensors')
    p.add_argument('--embedding-grid', type=Path, help='Larry silu_t_emb_grid; generated from original weights if omitted')
    p.add_argument('--vae-dir', type=Path, help='Optional converted native LightX2V video VAE directory')
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--image', action='append', default=[], help='Reference image; repeat for Ref2VA')
    p.add_argument('--video', action='append', default=[], help='Reference video; repeat for Ref2VA')
    p.add_argument('--audio', action='append', default=[], help='Reference audio; repeat for Ref2VA')
    p.add_argument('--last-frame', type=Path)
    p.add_argument('--task', choices=['t2av', 'i2av', 'l2av', 'fl2av'], default='i2av')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, help='Reusable converted weights and TP2 shards')
    p.add_argument('--num-frames', type=int, default=362)
    p.add_argument('--height', type=int)
    p.add_argument('--width', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--attention', choices=['sage_attn2', 'torch_sdpa'], default='sage_attn2')
    p.add_argument('--dry-run', action='store_true', help='Print the run specification without importing GPU packages')
    return p


def specification(args):
    values = vars(args).copy()
    if args.paths:
        stored = json.loads(args.paths.read_text(encoding='utf-8'))
        for key in ['model_dir', 'pruned_checkpoint', 'int8_checkpoint', 'adapter', 'embedding_grid', 'vae_dir', 'cache_dir']:
            if values[key] is None and stored.get(key):
                values[key] = Path(stored[key]).expanduser()
    for key in ['model_dir', 'pruned_checkpoint', 'adapter']:
        if values[key] is None:
            raise ValueError('--' + key.replace('_', '-') + ' is required (or supply it through --paths)')
    fl = args.profile == 'fl2va-turbo'
    if fl:
        if args.task in ['i2av', 'fl2av'] and len(args.image) != 1:
            raise ValueError('i2av/fl2av requires exactly one --image')
        if args.task in ['l2av', 'fl2av'] and not args.last_frame:
            raise ValueError('l2av/fl2av requires --last-frame')
        if args.video or args.audio:
            raise ValueError('Use a Ref2VA profile for reference videos or audio')
    elif not args.image and not args.video:
        raise ValueError('Ref2VA requires --image or --video')
    values['height'] = args.height or (768 if fl else 1344)
    values['width'] = args.width or (1344 if fl else 768)
    values['seed'] = args.seed if args.seed is not None else (101 if fl else 20261013)
    values['steps'] = 6 if fl else 8
    values['alpha'] = 8.0 if args.profile == 'ref2va-turbo' else 64.0
    values['cache_dir'] = values['cache_dir'] or values['output_dir'] / 'cache'
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value.resolve())
    for key in ['image', 'video', 'audio']:
        values[key] = [str(Path(v).resolve()) for v in values[key]]
    return values


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        spec = specification(args)
    except ValueError as e:
        p.error(str(e))
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return 0
    for key in ['model_dir', 'pruned_checkpoint', 'adapter', 'prompt_file']:
        if not Path(spec[key]).exists():
            p.error(f'{key} does not exist: {spec[key]}')
    out = Path(spec['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / 'run.json'
    spec_path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), env.get('PYTHONPATH', '')])
    for key, value in {'DTYPE': 'BF16', 'SENSITIVE_LAYER_DTYPE': 'BF16', 'OMP_NUM_THREADS': '8',
                       'PYTHONUNBUFFERED': '1',
                       'PYTORCH_CUDA_ALLOC_CONF': 'backend:native,expandable_segments:True'}.items():
        env.setdefault(key, value)
    started = time.perf_counter()
    commands = [[sys.executable, '-m', 'runtime.prepare', str(spec_path)],
                [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                 '--nproc_per_node', str(args.gpus), str(ROOT / 'runtime/worker.py'), str(spec_path)]]
    with (out / 'run.log').open('w', encoding='utf-8') as log:
        for command in commands:
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code:
                return code
    (out / 'timing.json').write_text(json.dumps({'launch_seconds': time.perf_counter() - started,
                                               'includes_weight_and_adapter_preparation': True}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
