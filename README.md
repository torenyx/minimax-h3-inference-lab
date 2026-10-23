# MiniMax H3 on 24GB GPUs

[Simplified Chinese](README.zh-CN.md) · [Video gallery](https://torenyx.github.io/minimax-h3-inference-lab/docs/gallery.html) · [Measurements](benchmarks/README.md) · [Setup](docs/setup.md)

Generate video and native stereo audio with **two or four RTX 3090 or RTX 4090 GPUs**. Three LightX2V profiles cover first-frame generation with Larry Turbo and reference-conditioned generation with PDD or Turbo.

## Choose a profile

| Your input | Profile | Included sample |
|---|---|---|
| First frame | `fl2va-turbo` · Larry v4, 6 steps | 15 s · 1344×768 |
| Reference image | `ref2va-pdd` · PDD, 8 steps | 15 s · 768×1344 |
| Reference image | `ref2va-turbo` · LightX2V Turbo, 8 steps | 5 s · 768×1344 |

## Watch the results

<table>
<tr>
<td width="50%" valign="top">
<h3>FL2VA · Larry Turbo</h3>
<p align="center"><a href="media/fl2va-turbo-4gpu.mp4"><img src="media/fl2va-turbo-4gpu-poster.jpg" alt="FL2VA four-GPU sample — click to play" height="270"></a></p>
<p>15 s · 1344×768 · 6 steps<br><b>4 GPUs: 9:29 total · 450.77 s denoising</b><br>2 GPUs: 13:27 total · 656.49 s denoising</p>
<p><a href="media/fl2va-turbo-4gpu.mp4">Play four-GPU clip</a> · <a href="media/fl2va-turbo-2gpu.mp4">Play two-GPU clip</a></p>
</td>
<td width="50%" valign="top">
<h3>Ref2VA · PDD</h3>
<p align="center"><a href="media/ref2va-pdd-4gpu.mp4"><img src="media/ref2va-pdd-4gpu-poster.jpg" alt="Ref2VA four-GPU sample — click to play" height="270"></a></p>
<p>15 s · 768×1344 · 8 steps<br><b>4 GPUs: 11:19 total · 571.82 s denoising</b><br>2 GPUs: 15:53 total · 838.13 s denoising</p>
<p><a href="media/ref2va-pdd-4gpu.mp4">Play four-GPU clip</a> · <a href="media/ref2va-pdd-2gpu.mp4">Play two-GPU clip</a></p>
</td>
</tr>
</table>

RTX 3090 24GB, pruned INT8 ConvRot, SageAttention 2, 362 frames at 24 fps. Each two/four-GPU pair uses the same input and seed. Total includes model loading, encoding, denoising, decoding, and saving; converted weights and caches are prepared beforehand.

[All five clips](https://torenyx.github.io/minimax-h3-inference-lab/docs/gallery.html) · [Stage timings and inputs](benchmarks/README.md) · [CSV](benchmarks/successful-runs.csv)

## Run the included sample

Linux, CUDA toolkit, Python 3.10+, and `aria2c`. The installer uses PyTorch 2.5.1 / CUDA 12.4 and SageAttention 2.2.0.

```bash
git clone https://github.com/torenyx/minimax-h3-inference-lab.git
cd minimax-h3-inference-lab
bash scripts/setup.sh
source .venv/bin/activate

python scripts/download_models.py --profile fl2va-turbo --model-root /models/h3
python scripts/sample.py --profile fl2va-turbo --model-root /models/h3 --gpus 4
```

The downloader selects pinned model revisions, resumes transfers with 16 connections per file, and writes reusable path settings. The sample command converts the checkpoint on first use and generates `outputs/fl2va-turbo-4gpu/trial.mp4` from the included prompt and first frame.

Use `--gpus 2` for two cards. Download and run `ref2va-pdd` or `ref2va-turbo` the same way. Add `--num-frames 124` for a short FL2VA/PDD run. [Model sizes, existing weights, and your own inputs →](docs/setup.md)

## Runtime changes

- **Resident DiT:** TP2 on two cards, TP2×SP2 on four; release DiT before VAE decoding.
- **Bounded temporaries:** chunk FFN, RMSNorm, AdaLN, and LoRA updates; release Q/K/V buffers after attention.
- **Reusable preparation:** persist TP2 INT8 shards and AdaLN schedules; decode video VAE tiles across GPUs.

On the same two cards, the Ref2VA PDD resident recipe reduces total time from **1052.08 to 952.81 seconds (−9.44%)**. [Full measurements →](benchmarks/README.md)

## More measurements

[Ref2VA Turbo/PDD, BF16 vs INT8](benchmarks/README.md#5-second-adapter-runs) · [LightX2V / SGLang / Raylight / vLLM-Omni Base](benchmarks/README.md#base-framework-runs) · [Inputs](benchmarks/inputs/) · [Per-step records](benchmarks/records/)

## Built on

[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3), [LightX2V](https://github.com/ModelTC/LightX2V), [Larryvrh Turbo](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora), [Alibaba PAI PDD](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs), [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun), and [SageAttention](https://github.com/thu-ml/SageAttention). Rank-8 time-conditioning preparation follows [Wan2GP](https://github.com/deepbeepmeep/Wan2GP).
