"""Backbone mapping and AdaLN cache construction for released H3 adapters."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from runtime.pdd_schedule import pdd_sampling_plan,pdd_time_grid

def pdd_backbone(source_path, target_path):
    """Normalize actual PDD suffixes; AdaLN is merged in its exact cache builder."""
    tensors = {}
    with safe_open(source_path, framework='pt', device='cpu') as source:
        for key in source.keys():
            if '.adaln_proj.linear.' in key or '.lora_' not in key:
                continue
            target = key.replace('.lora_down', '.lora_A.weight').replace('.lora_up', '.lora_B.weight')
            tensors[target] = source.get_tensor(key)
    assert len(tensors) == 624, len(tensors)
    save_file(tensors, target_path, metadata={'source': str(source_path), 'alpha': '64'})


def attach_pdd_heads(model, checkpoint):
    # Fuse on the 32-interval training grid; the 8-step Euler schedule visits
    # the same every-fourth grid boundaries independently for each modality.
    with safe_open(checkpoint, framework='pt', device='cpu') as source:
        for name, shift in (('proj_out', 12.0), ('audio_proj_out', 3.0)):
            dt = pdd_time_grid(shift, 32).diff()
            plans = torch.cat([pdd_sampling_plan(dt, step * 4, 4).float() for step in range(8)])
            stacks = []
            for suffix in ('weight', 'bias'):
                heads = source.get_tensor(name + '.' + suffix).float()
                expression = 'pn,noi->poi' if suffix == 'weight' else 'pn,no->po'
                stacks.append(torch.einsum(expression, plans, heads))

            def apply(hidden, stacks=stacks):
                index = int(model.scheduler.step_index)
                assert 0 <= index < 8, index
                weight, bias = (stack[index].to(hidden.device) for stack in stacks)
                return F.linear(hidden.float(), weight, bias)

            getattr(model.post_weight, name).apply = apply
    print('H3_PDD_HEADS_ATTACHED grid=32 block=4 nfe=8', flush=True)


def build_pruned_cache(config, checkpoint):
    from runtime.checkpoint_io import Reader
    from lightx2v.models.networks.minimax_h3.adaln_cache import (
        _block_key, _build_spec, _cache_path, _expected_table_shape,
        _norm_out_key, _timesteps_from_bits, _validate_cache,
    )
    from lightx2v.models.networks.minimax_h3.infer.pre_infer import timestep_embedding

    spec = _build_spec(config)
    cache = _cache_path(config)
    if _validate_cache(cache, spec):
        return
    pruned_path = PRUNED
    pruned = Reader([pruned_path])
    table = pruned.tensor('adaln_t_table').float()
    coordinates = {}
    for entry in spec['entries']:
        timesteps = _timesteps_from_bits(entry['timestep_bits'])
        position = timesteps * (table.shape[0] - 1)
        lower = position.floor().long().clamp(0, table.shape[0] - 2)
        coordinates[entry['name']] = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))

    inputs = {}
    if KIND == 'pdd':
        original = Reader(sorted(ORIGINAL_TRANSFORMER.glob('*.safetensors')))
        params = [original.tensor('time_embedder.' + name).float()
                  for name in ('proj_in.weight', 'proj_in.bias', 'proj_out.weight', 'proj_out.bias')]
        for entry in spec['entries']:
            embedded = timestep_embedding(_timesteps_from_bits(entry['timestep_bits']), spec['freq_dim']).float()
            hidden = F.linear(embedded, params[0], params[1])
            hidden = F.linear(F.silu(hidden), params[2], params[3])
            inputs[entry['name']] = F.silu(hidden).to(torch.bfloat16).float()
        original.close()
        del params

    tensors = {}
    delta_peak = 0.0
    with safe_open(checkpoint, framework='pt', device='cpu') as adapter_source:
        for block in range(51):
            prefix = f'blocks.{block}.adaln_proj.linear' if block < 50 else 'final_layer.adaln_proj.linear'
            weight, bias = (pruned.tensor(prefix + suffix).float() for suffix in ('.weight', '.bias'))
            down = up = None
            if KIND == 'pdd' and block < 50:
                lora_prefix = f'transformer_blocks.{block}.adaln_proj.linear'
                down = adapter_source.get_tensor(lora_prefix + '.lora_down').float()
                up = adapter_source.get_tensor(lora_prefix + '.lora_up').float()
            for entry in spec['entries']:
                value = F.linear(coordinates[entry['name']], weight, bias)
                if down is not None:
                    # Keep the pruned base modulation; add the released update
                    # in its original full time-embedding coordinate system.
                    delta = F.linear(F.linear(inputs[entry['name']], down), up)
                    delta_peak = max(delta_peak, delta.abs().max().item())
                    value = value + delta
                assert torch.isfinite(value).all(), (block, entry['name'])
                value = value.to(torch.bfloat16)
                key = _block_key(block, entry) if block < 50 else _norm_out_key(entry)
                if block < 50:
                    value = value.view(_expected_table_shape(spec, entry))
                tensors[key] = value.contiguous()
            if block % 10 == 0:
                print('H3_PRUNED_CACHE_BLOCK', block, flush=True)
    pruned.close()
    assert KIND != 'pdd' or delta_peak > 0, delta_peak
    cache.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(cache / 'adaln_cache.safetensors'))
    (cache / 'manifest.json').write_text(json.dumps(spec, indent=2))
    assert _validate_cache(cache, spec)
    print('H3_PRUNED_CACHE_READY pdd_delta_peak=' + str(delta_peak), flush=True)


def prepare_backbone_lora():
    from safetensors.torch import save_file

    tensors = {}
    lora_alpha = 64.0
    with safe_open(LORA, framework="pt", device="cpu") as source:
        source_modules = {key.rsplit(".lora_", 1)[0] for key in source.keys() if ".adaln_proj." not in key}
        if len(source_modules) != 208:
            raise RuntimeError(f"Turbo backbone adapter mismatch: modules={len(source_modules)}")
        for module in sorted(source_modules):
            down = source.get_tensor(module + ".lora_A.weight")
            up = source.get_tensor(module + ".lora_B.weight")
            # Larryvrh exports B@A directly; alpha must equal rank to preserve it.
            assert down.shape[0] == lora_alpha == 64.0
            if module.startswith("token_refiner.blocks."):
                target = module.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
            elif module.startswith("blocks."):
                target = module.replace("blocks.", "transformer_blocks.", 1)
            else:
                raise RuntimeError(f"Unexpected Turbo backbone module: {module}")

            if ".attn.qkv_proj" in module:
                if up.shape[0] % 3:
                    raise RuntimeError(f"Turbo fused QKV LoRA has invalid shape: {module} {tuple(up.shape)}")
                for projection, projection_up in zip(("q", "k", "v"), up.chunk(3, dim=0)):
                    target_projection = target.replace(".attn.qkv_proj", ".attn.to_" + projection)
                    tensors[target_projection + ".lora_A.weight"] = down.contiguous().clone()
                    tensors[target_projection + ".lora_B.weight"] = projection_up.contiguous()
            else:
                target = target.replace(".attn.out_proj", ".attn.to_out.0")
                target = target.replace(".mlp.fc1", ".ff.net.0.proj")
                target = target.replace(".mlp.fc2", ".ff.net.2")
                if ".mlp.fc1" in module:
                    first, second = up.chunk(2, dim=0)
                    up = torch.cat((second, first), dim=0)
                tensors[target + ".lora_A.weight"] = down.contiguous()
                tensors[target + ".lora_B.weight"] = up.contiguous()

    modules = {key.rsplit(".lora_", 1)[0] for key in tensors}
    if len(modules) != 312 or len(tensors) != 624:
        raise RuntimeError(f"Turbo LightX2V adapter mismatch: modules={len(modules)}, tensors={len(tensors)}")
    save_file(tensors, str(BACKBONE_LORA), metadata={"source": str(LORA), "strength": "1.0", "alpha": "64"})
    print("H3_TURBO_BACKBONE_LORA_READY source_modules=208 lightx_modules=312 tensors=624", flush=True)


def build_turbo_adaln_cache(config):
    from runtime.checkpoint_io import Reader
    from lightx2v.models.networks.minimax_h3.adaln_cache import (
        _block_key,
        _build_spec,
        _cache_path,
        _expected_table_shape,
        _norm_out_key,
        _timesteps_from_bits,
        _validate_cache,
    )

    spec = _build_spec(config)
    cache = _cache_path(config)
    if _validate_cache(cache, spec):
        print("H3_TURBO_ADALN_CACHE_REUSED", flush=True)
        return

    pruned = Reader([PRUNED])
    base_table = pruned.tensor("adaln_t_table").float()
    grid = None
    with safe_open(EGRID, framework="pt", device="cpu") as source:
        grid = source.get_tensor("silu_t_emb_grid").float()
    if tuple(grid.shape) != (1025, 2688):
        raise RuntimeError(f"Unexpected Turbo e-grid shape: {tuple(grid.shape)}")

    entries = {}
    for entry in spec["entries"]:
        times = _timesteps_from_bits(entry["timestep_bits"]).float()
        position = times * (base_table.shape[0] - 1)
        lo = position.floor().long().clamp(0, base_table.shape[0] - 2)
        base = torch.lerp(base_table[lo], base_table[lo + 1], (position - lo).unsqueeze(1))
        epos = times.clamp(0, 1) * (grid.shape[0] - 1)
        elo = epos.floor().long().clamp(0, grid.shape[0] - 2)
        entries[entry["name"]] = (base, torch.lerp(grid[elo], grid[elo + 1], (epos - elo).unsqueeze(1)))

    tensors = {}
    delta_peak = 0.0
    adaln_count = 0
    with safe_open(LORA, framework="pt", device="cpu") as adapter:
        for block in range(51):
            prefix = f"blocks.{block}.adaln_proj.linear" if block < 50 else "final_layer.adaln_proj.linear"
            lora_prefix = prefix
            weight, bias = (pruned.tensor(prefix + suffix).float() for suffix in (".weight", ".bias"))
            down = adapter.get_tensor(lora_prefix + ".lora_A.weight").float()
            up = adapter.get_tensor(lora_prefix + ".lora_B.weight").float()
            adaln_count += 1
            for entry in spec["entries"]:
                base, embedding = entries[entry["name"]]
                value = F.linear(base, weight, bias)
                delta = F.linear(F.linear(embedding, down), up)
                delta_peak = max(delta_peak, delta.abs().max().item())
                value.add_(delta)
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Non-finite AdaLN cache at block={block}, entry={entry['name']}")
                value = value.to(torch.bfloat16)
                key = _block_key(block, entry) if block < 50 else _norm_out_key(entry)
                if block < 50:
                    value = value.view(_expected_table_shape(spec, entry))
                tensors[key] = value.contiguous()
            if block % 10 == 0:
                print(f"H3_TURBO_ADALN_BLOCK={block}", flush=True)
    pruned.close()

    if adaln_count != 51 or delta_peak <= 0:
        raise RuntimeError(f"Turbo AdaLN injection incomplete: count={adaln_count}, delta_peak={delta_peak}")
    cache.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(cache / "adaln_cache.safetensors"))
    (cache / "manifest.json").write_text(json.dumps(spec, indent=2))
    if not _validate_cache(cache, spec):
        raise RuntimeError("Turbo AdaLN cache validation failed")
    print(f"H3_TURBO_ADALN_READY count={adaln_count} delta_peak={delta_peak}", flush=True)

