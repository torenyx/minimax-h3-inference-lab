# Install and run

## Environment

Linux, NVIDIA CUDA GPUs, Python 3.10+, Git and a CUDA toolkit with `nvcc`.
The published runs used two or four RTX 3090s. Select cards with
`CUDA_VISIBLE_DEVICES`.

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

The installer creates `.venv`, fetches LightX2V at
`8e86555809efd20138d3a4e1f937ee306cdf0ee0`, applies our small H3 patch,
and builds SageAttention v2.2.0. `requirements-runtime.txt` pins the main
libraries from the measured environment: Torch 2.5.1 / CUDA 12.4,
Transformers 4.57.6, Diffusers 0.35.2, Triton 3.1.0, and Comfy Kitchen 0.2.37.
Set `H3_VENV` or `H3_LIGHTX2V_SOURCE` to choose the installation directories.

For an existing PyTorch environment, install the requirements and pinned
LightX2V manually. To skip the SageAttention build, set `H3_SKIP_SAGE=1` and
run with `--attention torch_sdpa`. Published performance uses SageAttention.

## Download and run a sample

Install `aria2` with your system package manager. The downloader uses 16
connections per file, four concurrent files, and resumes interrupted transfers.

```bash
python scripts/download_models.py --profile fl2va-turbo --model-root /models/h3 --dry-run
python scripts/download_models.py --profile fl2va-turbo --model-root /models/h3
python scripts/sample.py --profile fl2va-turbo --model-root /models/h3 --gpus 4
```

The FL2VA source download is 144.83 GB. Converted checkpoints and runtime
caches occupy additional disk space. `--dry-run` lists the exact files, pinned
revisions, and remaining download size for each profile.

`sample.py` converts the checkpoint once, then uses the included prompt and
image. It reuses converted weights on later runs. Replace the profile in both
commands with `ref2va-pdd` or `ref2va-turbo` for the reference-image samples.
FL2VA/PDD default to 362 frames; Ref2VA Turbo defaults to 124 frames.
Add `--num-frames 124` for a short FL2VA/PDD run. Set `--output-dir` to choose
the destination; otherwise outputs go to `outputs/<profile>-<gpus>gpu/`.

The downloader also writes `/models/h3/<profile>.json` for direct use with
`scripts/run.py --paths`. Model files shared by the two Ref2VA profiles are
reused. Download revisions and file sizes are saved alongside the paths file.

## Use existing weights

The launcher reads the official component directories from
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
Each task model directory contains `transformer/`, `text_encoder/`,
`tokenizer/`, `processor/`, `audio_vae/`, and `video_vae/`.

| Profile | Base directory | Adapter |
|---|---|---|
| `fl2va-turbo` | `FL2VA` | Larryvrh `minimax_h3_turbo_v4_step600_ema.safetensors` |
| `ref2va-pdd` | `Ref2VA` | Alibaba PAI `MiniMax-H3-Ref2VA-Acc-8Step.safetensors` |
| `ref2va-turbo` | `Ref2VA` | LightX2V `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` |

Adapter sources:
[Larryvrh Turbo](https://huggingface.co/larryvrh/Minimax-H3-Turbo-LoRA),
[Alibaba PAI PDD](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs),
[LightX2V Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo).
Use the LightX2V Diffusers adapter file listed above, rather than the
separate ComfyUI adapter variant.

Create the pruned INT8 ConvRot checkpoint once per base model:

```bash
python scripts/convert.py --variant FL2VA \
  --model-dir /models/MiniMax-H3/FL2VA \
  --output-dir /models/derived/FL2VA --device cuda:0

python scripts/convert.py --variant Ref2VA \
  --model-dir /models/MiniMax-H3/Ref2VA \
  --output-dir /models/derived/Ref2VA --device cuda:0
```

This streams unchanged tensors, constructs a rank-8 time-conditioning basis,
and quantizes the backbone with INT8 ConvRot. Outputs include
`pruned_bf16.safetensors`, `pruned_int8_convrot.safetensors` and a conversion
report. Keep the original component directory: PDD cache construction and
Larry's time-embedding grid also use its original time-embedder weights.

The inference launcher then prepares the LightX2V tensor layout, native video
VAE, mapped backbone adapter, AdaLN cache, and reusable TP2 shards. An existing
native video VAE can be passed with `--vae-dir`; an existing LightX2V-layout
INT8 checkpoint with `--int8-checkpoint`.

## FL2VA: Larry Turbo, six steps

```bash
python scripts/run.py --profile fl2va-turbo --gpus 2 \
  --model-dir /models/MiniMax-H3/FL2VA \
  --pruned-checkpoint /models/derived/FL2VA/pruned_int8_convrot.safetensors \
  --adapter /models/minimax_h3_turbo_v4_step600_ema.safetensors \
  --prompt-file benchmarks/inputs/fl2va-15s.txt \
  --image benchmarks/inputs/fl2va-first-frame.jpg \
  --cache-dir /models/cache/fl2va-turbo \
  --output-dir outputs/fl2va-turbo-2gpu
```

Change `--gpus 2` to `--gpus 4` for TP2 × SP2. Default output geometry is
1344 × 768, 362 frames, seed 101. Add `--embedding-grid` to use the author's
`h3_silu_temb_grid.safetensors`; otherwise it is generated once from the
original FL2VA time embedder.

The same entry accepts `--task t2av` without an image, `--task l2av` with
`--last-frame`, and `--task fl2av` with both `--image` and `--last-frame`.
The published FL2VA results use `i2av`.

## Ref2VA: PDD or Turbo, eight steps

```bash
python scripts/run.py --profile ref2va-pdd --gpus 4 \
  --model-dir /models/MiniMax-H3/Ref2VA \
  --pruned-checkpoint /models/derived/Ref2VA/pruned_int8_convrot.safetensors \
  --adapter /models/MiniMax-H3-Ref2VA-Acc-8Step.safetensors \
  --prompt-file benchmarks/inputs/ref2va-15s.txt \
  --image benchmarks/inputs/ref2va-reference.png \
  --cache-dir /models/cache/ref2va-pdd \
  --output-dir outputs/ref2va-pdd-4gpu
```

For the LightX2V Ref2VA Turbo adapter, select `--profile ref2va-turbo` and
its adapter path. Both profiles use eight steps; their native adapter scaling
and video flow shifts are kept separate. Use `--gpus 2` for TP2. Repeat
`--image`, `--video`, or `--audio` to supply several references; at least one
image or video is required. Default Ref2VA geometry is 768 × 1344, 362 frames.
The five-second published Turbo result uses `--num-frames 124`.

## Inputs, outputs and checks

`--prompt-file` reads Context-IR verbatim. `--height`, `--width`,
`--num-frames` and `--seed` override the profile defaults. Store reusable model
paths in a JSON file based on `configs/paths.example.json` and pass `--paths`.
Use a separate cache directory for each base-model/adapter combination.

`--dry-run` prints the resolved settings without loading PyTorch or weights.
Successful runs write `trial.mp4`, `config.json`, `run.json`, `prompt.txt`,
`run.log` and `timing.json` in the output directory. `launch_seconds` includes
weight/cache preparation; the structured `H3_RESIDENT_DENOISE` log record
measures denoising separately for the resident profiles.

```bash
python -m unittest discover -s tests
```

CPU checks cover launch contracts. The [benchmark records](../benchmarks/records/)
contain the completed GPU runs' stage timings.
