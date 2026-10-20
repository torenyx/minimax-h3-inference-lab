# 在 24GB 显卡上运行 MiniMax H3

[English](README.md) · [完整实测](benchmarks/README.md) · [安装与模型准备](docs/setup.md)

两张或四张 RTX 3090，生成视频与同步音频。仓库包含 FL2VA Larry Turbo、Ref2VA Turbo/PDD 的运行入口、显存优化、配置与成功成片。

**生成 15 秒音视频：四卡 FL2VA 六步含加载约 9 分 29 秒；四卡 Ref2VA PDD 八步约 11 分 19 秒。**

## 成功结果

INT8 ConvRot 裁剪主干、SageAttention 2、362 帧、24 fps，去噪期间 DiT 权重常驻 GPU。

| 任务 | 卡数 | 步数 | 去噪 | 框架总耗时 | 成片 |
|---|---:|---:|---:|---:|---|
| FL2VA · Larry Turbo | 2 | 6 | 656.49 s | 806.85 s | [播放](media/fl2va-turbo-2gpu.mp4) |
| FL2VA · Larry Turbo | 4 | 6 | **450.77 s** | **568.95 s** | [播放](media/fl2va-turbo-4gpu.mp4) |
| Ref2VA · PDD | 2 | 8 | 838.13 s | 952.81 s | [播放](media/ref2va-pdd-2gpu.mp4) |
| Ref2VA · PDD | 4 | 8 | **571.82 s** | **678.55 s** | [播放](media/ref2va-pdd-4gpu.mp4) |

FL2VA 使用首帧条件，分辨率 1344×768；Ref2VA 使用参考图像，分辨率 768×1344。每条路线内两卡、四卡保持输入和种子一致。总耗时包括加载、编码、去噪、解码与保存；权重和缓存提前准备。

[![FL2VA 四卡成片](media/fl2va-turbo-4gpu.png)](media/fl2va-turbo-4gpu.mp4)

[![Ref2VA 四卡成片](media/ref2va-pdd-4gpu.png)](media/ref2va-pdd-4gpu.mp4)

## 开始运行

Linux、CUDA、两张或四张 NVIDIA GPU。实测环境为 PyTorch 2.5.1 + CUDA 12.4、SageAttention 2.2.0。

```bash
git clone https://github.com/torenyx/minimax-h3-inference-lab.git
cd minimax-h3-inference-lab
bash scripts/setup.sh
source .venv/bin/activate
```

两卡 FL2VA Larry Turbo：

```bash
python scripts/run.py \
  --profile fl2va-turbo --gpus 2 \
  --model-dir /models/h3/fl2va \
  --pruned-checkpoint /models/h3/fl2va/pruned_int8_convrot.safetensors \
  --adapter /models/h3/larry-v4_step600_ema.safetensors \
  --embedding-grid /models/h3/h3_silu_temb_grid.safetensors \
  --prompt-file examples/prompt.txt --image /inputs/first-frame.png \
  --output-dir outputs/fl2va
```

四卡改为 `--gpus 4`。Ref2VA PDD 选择 `--profile ref2va-pdd` 并提供对应主干及 PDD 权重；Ref2VA Turbo 选择 `--profile ref2va-turbo` 并提供 LightX2V Ref2VA 八步权重。

模型目录、权重准备与路径复用方式见[安装说明](docs/setup.md)。

## 优化内容

- **常驻去噪**：两卡 TP2，四卡 TP2×SP2；完成去噪后释放 DiT，为 VAE 解码腾出显存。
- **降低临时张量峰值**：FFN、RMSNorm、AdaLN、LoRA 增量分块，及时释放 Q/K/V 和通信缓冲。
- **复用准备结果**：持久化 TP2 INT8 分片及 AdaLN 日程，复用节点本地权重和模型组件。
- **并行视频解码**：按空间 tile 分配 VAE 工作，同时保留原生音频解码。

相同两卡、相同 Ref2VA PDD 输入，常驻优化配置将去噪从 **892.41 秒降至 838.13 秒**，框架总耗时从 **1052.08 秒降至 952.81 秒**。四卡去噪为 571.82 秒。[阶段数据](benchmarks/README.md)

## 四框架 Base 对照

Ref2VA、INT8 ConvRot 裁剪主干、SageAttention 2、768×1344、124 帧、20 次去噪前向。下表单位为秒。

| 框架 | 两卡去噪 | 四卡去噪 |
|---|---:|---:|
| LightX2V | 417.30 | 353.87 |
| SGLang | 425.18 | 317.62 |
| Raylight | 594.54 | 362.62 |
| vLLM-Omni | 约 590 | 约 401 |

本仓库运行入口使用 LightX2V；其他框架的配置与结果收录于[对照数据](benchmarks/base-frameworks.json)。

## Ref2VA Turbo 与 PDD

两张 RTX 3090、768×1344、124 帧、八步：

| 加速权重 | 主干 | 去噪 | 框架总耗时 |
|---|---|---:|---:|
| LightX2V Turbo | BF16 | 229.15 s | 376.15 s |
| LightX2V Turbo | INT8 裁剪 | 191.14 s | 515.04 s |
| PDD | BF16 | 226.92 s | 376.20 s |
| PDD | INT8 裁剪 | 191.57 s | 495.03 s |

[Turbo 成片](media/ref2va-turbo-int8-2gpu-5s.mp4) · [CSV](benchmarks/successful-runs.csv) · [JSON](benchmarks/successful-runs.json)

[测试输入](benchmarks/inputs/)与[逐步计时记录](benchmarks/records/)随成片一同提供。

## 上游项目

[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)、[LightX2V](https://github.com/ModelTC/LightX2V)、[Larryvrh Turbo](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)、[Alibaba PAI PDD](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)、[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)、[SageAttention](https://github.com/thu-ml/SageAttention)。

Base 对照还使用了 [SGLang](https://github.com/sgl-project/sglang)、[Raylight](https://github.com/komikndr/raylight)、[vLLM-Omni](https://github.com/vllm-project/vllm-omni)。

Rank-8 时间条件准备沿用 [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) 的构造方法。
