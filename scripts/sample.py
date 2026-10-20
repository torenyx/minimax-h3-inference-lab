#!/usr/bin/env python3
"""Prepare a profile's checkpoint once, then generate its included sample."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from download_models import PROFILES, paths_for

ROOT = Path(__file__).resolve().parents[1]


def commands_for(profile, model_root, gpus, frames, output):
    paths = paths_for(profile, model_root)
    settings = model_root.expanduser().resolve() / f'{profile}.json'
    commands = []
    if not Path(paths['pruned_checkpoint']).is_file():
        commands.append([sys.executable, str(ROOT / 'scripts/convert.py'), '--variant', PROFILES[profile][0],
                         '--model-dir', paths['model_dir'], '--output-dir', str(Path(paths['pruned_checkpoint']).parent)])
    family = 'fl2va' if profile == 'fl2va-turbo' else 'ref2va'
    prompt = 'ref2va-5s.txt' if profile == 'ref2va-turbo' else family + '-15s.txt'
    image = 'fl2va-first-frame.jpg' if family == 'fl2va' else 'ref2va-reference.png'
    commands.append([sys.executable, str(ROOT / 'scripts/run.py'), '--profile', profile,
                     '--paths', str(settings), '--gpus', str(gpus), '--num-frames', str(frames),
                     '--prompt-file', str(ROOT / 'benchmarks/inputs' / prompt),
                     '--image', str(ROOT / 'benchmarks/inputs' / image), '--output-dir', str(output.resolve())])
    return paths, settings, commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='fl2va-turbo')
    parser.add_argument('--model-root', type=Path, default=Path('models'))
    parser.add_argument('--gpus', type=int, choices=[2, 4], default=2)
    parser.add_argument('--num-frames', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    frames = args.num_frames if args.num_frames is not None else (124 if args.profile == 'ref2va-turbo' else 362)
    output = args.output_dir or ROOT / 'outputs' / f'{args.profile}-{args.gpus}gpu'
    paths, settings, commands = commands_for(args.profile, args.model_root, args.gpus, frames, output)
    if not args.dry_run:
        if not Path(paths['model_dir']).is_dir() or not Path(paths['adapter']).is_file():
            parser.error('Download this profile first with scripts/download_models.py')
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps(paths, indent=2) + '\n', encoding='utf-8')
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
