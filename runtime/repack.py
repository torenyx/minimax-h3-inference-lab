"""Repack the FL2VA pruned INT8 checkpoint into LightX2V tensor names."""
import json
import os
import struct
from pathlib import Path

import torch

from runtime.checkpoint_io import Reader, make_catalog, start_output, write_tensor

torch.set_num_threads(12)
SOURCE = Path(os.environ["H3_PRUNED_CHECKPOINT"])
DEST = Path(os.environ["H3_LIGHT_REPACK_DIR"])
DEST.mkdir(parents=True, exist_ok=True)
WEIGHTS = DEST / "int8-convrot.safetensors"
reader = Reader([SOURCE])


def rename(name):
    for old, new in (
        ("video_patch_proj", "proj_in"),
        ("audio_patch_proj", "audio_proj_in"),
        ("condition_proj", "context_embedder"),
        ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
        ("final_layer.norm_final", "norm_out.norm"),
        ("final_layer.video_out", "proj_out"),
        ("final_layer.audio_out", "audio_proj_out"),
        (".attn.q_norm.", ".attn.norm_q."),
        (".attn.k_norm.", ".attn.norm_k."),
        (".attn.out_proj.", ".attn.to_out.0."),
        (".mlp.fc1.", ".ff.net.0.proj."),
        (".mlp.fc2.", ".ff.net.2."),
    ):
        name = name.replace(old, new)
    if name.startswith("blocks."):
        name = "transformer_blocks." + name[len("blocks.") :]
    return name


specs = []
operations = []
for name, entry in reader.catalog.items():
    if name == "adaln_t_table" or ".adaln_proj." in name or name.startswith(("time_embedder.", "rotary_pos_emb.")):
        continue
    if name.endswith(".comfy_quant"):
        prefix = name.removesuffix(".comfy_quant")
        if prefix.endswith(".attn.qkv_proj"):
            targets = [rename(prefix.replace(".qkv_proj", ".to_" + part)) + ".convrot_groupsize" for part in ("q", "k", "v")]
        else:
            targets = [rename(prefix + ".").rstrip(".") + ".convrot_groupsize"]
        for target in targets:
            specs.append((target, "I64", []))
            operations.append((target, name, "group", None))
    elif ".attn.qkv_proj." in name:
        rows = entry["shape"][0]
        if rows % 3:
            raise ValueError(f"Invalid fused QKV shape for {name}: {entry['shape']}")
        for index, part in enumerate(("q", "k", "v")):
            target = rename(name.replace(".qkv_proj.", ".to_" + part + "."))
            specs.append((target, entry["dtype"], [rows // 3, *entry["shape"][1:]]))
            operations.append((target, name, "qkv", index))
    else:
        target = rename(name)
        specs.append((target, entry["dtype"], entry["shape"]))
        operations.append((target, name, "swap" if ".mlp.fc1." in name else "copy", None))

if not WEIGHTS.exists():
    catalog, size = make_catalog(specs)
    stream, temporary, start = start_output(
        WEIGHTS,
        catalog,
        {"source": str(SOURCE), "transform": "rename/split-QKV/swap-SwiGLU; no requantization"},
    )
    with stream:
        for index, (target, name, operation, part) in enumerate(operations):
            if operation == "copy":
                reader.copy(name, stream)
            elif operation == "group":
                stream.write(struct.pack("<q", 256))
            else:
                tensor = reader.tensor(name)
                if operation == "qkv":
                    tensor = tensor.chunk(3, dim=0)[part]
                else:
                    gate, value = tensor.chunk(2, dim=0)
                    tensor = torch.cat((value, gate), dim=0)
                write_tensor(stream, tensor.contiguous())
            if index % 100 == 0:
                print(f"H3_FL2VA_REPACK {index}/{len(operations)} {target}", flush=True)
        if stream.tell() != start + size:
            raise RuntimeError("LightX2V checkpoint repack ended at an unexpected byte offset")
    temporary.replace(WEIGHTS)

reader.close()
(DEST / "repack.json").write_text(json.dumps({
    "state": "passed",
    "source": str(SOURCE),
    "weights": str(WEIGHTS),
    "bytes": WEIGHTS.stat().st_size,
    "transform": "LightX2V name mapping, fused QKV split, SwiGLU row swap",
    "quantization_preserved": True,
}, indent=2))
print(f"H3_FL2VA_LIGHTX2V_REPACK_READY {WEIGHTS}", flush=True)
