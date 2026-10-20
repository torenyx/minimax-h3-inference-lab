#!/usr/bin/env python3
"""Download one H3 profile with aria2 and write its reusable paths file."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import quote
from urllib.request import urlopen

BASE_REPO = 'MiniMaxAI/MiniMax-H3'
BASE_REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
PROFILES = {
    'fl2va-turbo': ('FL2VA', 'larryvrh/MiniMax-H3-Turbo-Lora',
                    '43a74557ac3f6539db8e0f2a959d03feb7a81480',
                    'minimax_h3_turbo_v4_step600_ema.safetensors'),
    'ref2va-pdd': ('Ref2VA', 'alibaba-pai/MiniMax-H3-Acc-LoRAs',
                   '335001fb9e5455d68a0caa18ec2e319072150328',
                   'MiniMax-H3-Ref2VA-Acc-8Step.safetensors'),
    'ref2va-turbo': ('Ref2VA', 'lightx2v/Minimax-h3-Turbo',
                     '30f485ed874aa44c8183348ad361c5eabad318da',
                     'minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors'),
}


def model_files(repo, revision):
    with urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=60) as response:
        return json.load(response)['siblings']


def paths_for(profile, root):
    variant, _, _, adapter = PROFILES[profile]
    root = root.expanduser().resolve()
    return {
        'model_dir': str(root / 'MiniMax-H3' / variant),
        'pruned_checkpoint': str(root / 'derived' / variant / 'pruned_int8_convrot.safetensors'),
        'adapter': str(root / 'adapters' / adapter),
        'cache_dir': str(root / 'cache' / profile),
    }


def complete(item):
    path = Path(item['destination'])
    return path.is_file() and not Path(str(path) + '.aria2').exists() and path.stat().st_size == item['size']


def download_plan(profile, root, fetch=model_files):
    variant, adapter_repo, adapter_revision, adapter_name = PROFILES[profile]
    plan = []
    sources = [(BASE_REPO, BASE_REVISION, root / 'MiniMax-H3'),
               (adapter_repo, adapter_revision, root / 'adapters')]
    for repo, revision, destination in sources:
        selected = []
        for entry in fetch(repo, revision):
            name = entry['rfilename']
            wanted = (name.startswith(variant + '/') or name == 'LICENSE') if repo == BASE_REPO else name == adapter_name
            if wanted:
                selected.append(name)
                plan.append({'repo': repo, 'revision': revision, 'file': name,
                             'size': entry['size'], 'destination': str(destination / name),
                             'url': f'https://huggingface.co/{repo}/resolve/{revision}/{quote(name)}'})
        if repo == BASE_REPO and not any(name.startswith(variant + '/transformer/') for name in selected):
            raise ValueError(f'{repo} is missing {variant}/transformer/')
        if repo != BASE_REPO and adapter_name not in selected:
            raise ValueError(f'{repo} is missing {adapter_name}')
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--model-root', type=Path, default=Path('models'))
    parser.add_argument('--dry-run', action='store_true', help='List pinned sources and sizes without downloading')
    args = parser.parse_args(argv)
    root = args.model_root.expanduser().resolve()
    plan = download_plan(args.profile, root)
    pending = [item for item in plan if not complete(item)]
    print(f"{args.profile}: {len(plan)} files, {sum(item['size'] for item in plan) / 1e9:.2f} GB")
    print(f"Remaining: {len(pending)} files, {sum(item['size'] for item in pending) / 1e9:.2f} GB")
    if args.dry_run:
        for item in plan:
            print(f"{item['size']:>13}  {item['repo']}@{item['revision'][:12]}/{item['file']}")
        return 0
    if pending and not shutil.which('aria2c'):
        parser.error('aria2c is required; install aria2, then rerun this command')
    root.mkdir(parents=True, exist_ok=True)
    if pending:
        with tempfile.TemporaryDirectory(prefix='h3-download-') as temp:
            manifest = Path(temp) / 'aria2.txt'
            entries = []
            for item in pending:
                path = Path(item['destination'])
                path.parent.mkdir(parents=True, exist_ok=True)
                entries.append(f"{item['url']}\n  dir={path.parent}\n  out={path.name}\n")
            manifest.write_text('\n'.join(entries), encoding='utf-8')
            subprocess.run(['aria2c', '--input-file', str(manifest), '--continue=true',
                            '--split=16', '--max-connection-per-server=16', '--min-split-size=1M',
                            '--max-concurrent-downloads=4', '--auto-file-renaming=false',
                            '--summary-interval=10'], check=True)
    for item in plan:
        if not complete(item):
            raise RuntimeError(f"Incomplete download: {item['destination']}")
    (root / f'{args.profile}-sources.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    settings = root / f'{args.profile}.json'
    settings.write_text(json.dumps(paths_for(args.profile, root), indent=2) + '\n', encoding='utf-8')
    print(f'Paths: {settings}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
