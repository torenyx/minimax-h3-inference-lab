# Measurements

RTX 3090 24GB. One completed run per final configuration. Times are seconds.

## Timing

- **Denoising:** accumulated denoiser evaluations.
- **Framework total:** loading, condition encoding, denoising, VAE decoding, and saving.
- Weight conversion, first-time cache preparation, process launch, and scheduler queue time are separate.

## 15-second resident runs

| Profile | GPUs | Layout | Load | Denoise | Total | Size |
|---|---:|---|---:|---:|---:|---|
| FL2VA Larry 6-step | 2 | TP2 | 50.30 | 656.49 | 806.85 | 1344×768 |
| FL2VA Larry 6-step | 4 | TP2×SP2 | 50.36 | 450.77 | 568.95 | 1344×768 |
| Ref2VA PDD 8-step | 2 | TP2 | 23.73 | 838.13 | 952.81 | 768×1344 |
| Ref2VA PDD 8-step | 4 | TP2×SP2 | 43.21 | 571.82 | 678.55 | 768×1344 |

362 frames, 24 fps. FL2VA uses seed 101 and a first-frame image; Ref2VA uses seed 20261013 and a reference image. All four outputs contain native stereo audio.

The four-card layouts reduce denoising time by 31.34% for FL2VA and 31.78% for Ref2VA relative to their two-card layouts, using twice as many GPUs.

## Same-card PDD optimization

Same Ref2VA input, two GPUs, eight steps:

| Configuration | Load | Denoise | Total |
|---|---:|---:|---:|
| Block offload | 64.63 | 892.41 | 1052.08 |
| Resident + prepared shards + memory optimizations | 23.73 | 838.13 | 952.81 |

The combined configuration reduces total time by 9.44%. Four-card VAE decoding takes 39.62 seconds; two-card decoding takes 69.54 seconds.

## 5-second adapter runs

Two GPUs, TP2, 768×1344, 124 frames, eight denoiser evaluations, seed 20261013. Video/audio shifts are 6/3 for the LightX2V Ref2VA Turbo adapter and 12/3 for PDD.

| Adapter | Backbone | Denoise | Load | Total |
|---|---|---:|---:|---:|
| LightX2V Ref2VA Turbo | BF16 | 229.15 | 102.03 | 376.15 |
| LightX2V Ref2VA Turbo | INT8 pruned | 191.14 | 258.87 | 515.04 |
| PDD | BF16 | 226.92 | 102.08 | 376.20 |
| PDD | INT8 pruned | 191.57 | 247.17 | 495.03 |

These configurations use offload and different loading paths. Their total times describe complete recipes.

## Base framework runs

Same reference image, prompt and seed; 768×1344, 124 frames, 20 evaluations, pruned INT8 ConvRot, SageAttention 2.

| Framework | GPUs | Parallel | Denoise | Script/framework total |
|---|---:|---|---:|---:|
| LightX2V | 2 | TP2 + U1 | 417.30 | 524.20 |
| LightX2V | 4 | TP2 + U2 | 353.87 | 452.06 |
| SGLang | 2 | TP2 + U1 | 425.18 | 607.93 |
| SGLang | 4 | TP2 + U2 | 317.62 | 442.76 |
| Raylight | 2 | FSDP + U2 | 594.54 | 794.08 |
| Raylight | 4 | FSDP + U4 | 362.62 | 650.38 |
| vLLM-Omni | 2 | TP2 + U1 | ~590 | 820.48 |
| vLLM-Omni | 4 | TP2 + U2 | ~401 | 612.36 |

vLLM-Omni denoising values come from progress bars. Raylight two-card execution reuses conditioning. Total timing boundaries follow each framework's benchmark script.

## Data

- [Successful adapter runs · JSON](successful-runs.json) / [CSV](successful-runs.csv)
- [Base framework runs · JSON](base-frameworks.json) / [CSV](base-frameworks.csv)
- [Inputs and prompts](inputs/) · [Per-step timing records](records/)
- [Generated clips and storyboards](../media/)
