# 在 24GB 显卡上运行 MiniMax H3

[English](README.md) · [视频画廊](https://torenyx.github.io/minimax-h3-inference-lab/docs/gallery.html) · [完整实测](benchmarks/README.md) · [安装说明](docs/setup.md)

用**两张或四张 RTX 3090** 生成视频与原生立体声音频。三个 LightX2V 配置覆盖 Larry Turbo 首帧生成，以及 PDD、Turbo 参考图生成。

## 选一条路线

| 你的输入 | 运行配置 | 自带样例 |
|---|---|---|
| 首帧图片 | `fl2va-turbo` · Larry v4，六步 | 15 秒 · 1344×768 |
| 参考图片 | `ref2va-pdd` · PDD，八步 | 15 秒 · 768×1344 |
| 参考图片 | `ref2va-turbo` · LightX2V Turbo，八步 | 5 秒 · 768×1344 |

## 看成片

<table>
<tr>
<td width="50%" valign="top">
<h3>FL2VA · Larry Turbo</h3>
<p align="center"><a href="media/fl2va-turbo-4gpu.mp4"><img src="media/fl2va-turbo-4gpu-poster.jpg" alt="FL2VA 四卡成片，点击播放" height="270"></a></p>
<p>15 秒 · 1344×768 · 六步<br><b>四卡：总耗时 9:29 · 去噪 450.77 秒</b><br>两卡：总耗时 13:27 · 去噪 656.49 秒</p>
<p><a href="media/fl2va-turbo-4gpu.mp4">播放四卡成片</a> · <a href="media/fl2va-turbo-2gpu.mp4">播放两卡成片</a></p>
</td>
<td width="50%" valign="top">
<h3>Ref2VA · PDD</h3>
<p align="center"><a href="media/ref2va-pdd-4gpu.mp4"><img src="media/ref2va-pdd-4gpu-poster.jpg" alt="Ref2VA 四卡成片，点击播放" height="270"></a></p>
<p>15 秒 · 768×1344 · 八步<br><b>四卡：总耗时 11:19 · 去噪 571.82 秒</b><br>两卡：总耗时 15:53 · 去噪 838.13 秒</p>
<p><a href="media/ref2va-pdd-4gpu.mp4">播放四卡成片</a> · <a href="media/ref2va-pdd-2gpu.mp4">播放两卡成片</a></p>
</td>
</tr>
</table>

RTX 3090 24GB、INT8 ConvRot 裁剪主干、SageAttention 2、362 帧、24 fps。同一路线的两卡/四卡使用相同输入与种子。总耗时包括加载、编码、去噪、解码与保存；权重转换与缓存提前完成。

[全部五段成片](https://torenyx.github.io/minimax-h3-inference-lab/docs/gallery.html) · [阶段耗时与输入](benchmarks/README.md) · [CSV](benchmarks/successful-runs.csv)

## 跑自带样例

Linux、CUDA toolkit、Python 3.10+、`aria2c`。安装脚本使用 PyTorch 2.5.1 / CUDA 12.4 和 SageAttention 2.2.0。

```bash
git clone https://github.com/torenyx/minimax-h3-inference-lab.git
cd minimax-h3-inference-lab
bash scripts/setup.sh
source .venv/bin/activate

python scripts/download_models.py --profile fl2va-turbo --model-root /models/h3
python scripts/sample.py --profile fl2va-turbo --model-root /models/h3 --gpus 4
```

下载器按固定模型版本取文件，每个文件使用 16 路连接并支持续传，同时生成路径配置。样例入口首次运行会转换权重，再用自带提示词和首帧生成 `outputs/fl2va-turbo-4gpu/trial.mp4`。

两卡使用 `--gpus 2`。将下载和运行命令中的配置换为 `ref2va-pdd` 或 `ref2va-turbo`，即可运行对应样例。FL2VA/PDD 加 `--num-frames 124` 可先跑短片。[模型大小、已有权重与自定义输入 →](docs/setup.md)

## 运行优化

- **DiT 常驻**：两卡 TP2，四卡 TP2×SP2；去噪后释放 DiT，为 VAE 解码腾出显存。
- **限制临时张量**：FFN、RMSNorm、AdaLN、LoRA 增量分块，attention 后及时释放 Q/K/V。
- **复用准备结果**：持久化 TP2 INT8 分片与 AdaLN 日程，按空间 tile 并行解码视频 VAE。

相同两张卡上，Ref2VA PDD 常驻配置将总耗时从 **1052.08 秒降至 952.81 秒，减少 9.44%**。[完整实测 →](benchmarks/README.md)

## 更多对照

[Ref2VA Turbo/PDD：BF16 与 INT8](benchmarks/README.md#5-second-adapter-runs) · [LightX2V / SGLang / Raylight / vLLM-Omni Base](benchmarks/README.md#base-framework-runs) · [测试输入](benchmarks/inputs/) · [逐步记录](benchmarks/records/)

## 上游项目

[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)、[LightX2V](https://github.com/ModelTC/LightX2V)、[Larryvrh Turbo](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)、[Alibaba PAI PDD](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)、[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)、[SageAttention](https://github.com/thu-ml/SageAttention)。Rank-8 时间条件准备沿用 [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) 的构造方法。
