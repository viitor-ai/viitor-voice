<h1 align="center">🚀 ViiTor Voice TTS</h1>
<p align="center">基于 transformers 与 vLLM 的快速灵活语音克隆，支持批量与时长控制。</p>
<p align="center"><a href="README.md">English</a> · <a href="https://viitor-ai.github.io/viitor-voice/">在线 Demo</a> · <a href="https://github.com/viitor-ai/viitor-voice/">GitHub</a> · <a href="https://huggingface.co/ZzWater/ViiTor-voice-2.0-base">Hugging Face</a></p>

## 🍀 方案简介
ViiTor Voice 是一个三阶段语音克隆流程：
- 阶段 1：Prompt + 文本 → 语义 token。
- 阶段 2：Prompt 的声学/语义 + 预测语义 → 预测声学 token。
- 阶段 3：声学 token → 波形。

## ✨ 模型亮点
- **无文本 Prompt**：更强跨语言克隆，降低 ASR 依赖，原始语音即可。
- **相似度增强**：InfoNCE + condition encoder 作为相似度约束，在噪声/背景复杂场景也稳健。
- **内置时长控制**：LLM 主干包含时长预测；可强制时长，精度约 0.5s。
- **LoRA 情绪控制**：通过 LoRA 适配器调节情绪/风格，无需全量微调。

`cli.py` 覆盖 transformers 与 vLLM 后端、两种批处理模式，以及单文本可选时长提示。

## ⚡ 快速开始（Linux）
### 1) 环境
使用提供的脚本（PyTorch、vLLM 0.12.0 CUDA 12.8、requirements、dualcodec）：
```
bash create_env.sh
source .venv/bin/activate
```
说明：
- `create_env.sh` 使用 Python 3.12 的 `uv venv`，如有需要可调整。
- vLLM 安装目标为 CUDA 12.8（`--torch-backend=cu128`），可按实际 CUDA/Toolkit 修改。

### 2) 模型
通过脚本下载（默认使用 Hugging Face 镜像）：
```
bash download_checkpoints.sh
```
默认路径（可在命令行覆盖）：
- SoundStorm: `checkpoints/viitor/soundstorm`
- DualCodec:  `checkpoints/dualcodec`
- wav2vec:    `checkpoints/w2v`
- LLM:        `checkpoints/viitor/llm/zh-en`

## 🎯 Demo 用法
### 🖥️ Gradio Demo
启动 Web 界面（监听 `0.0.0.0`，关闭 Gradio share）：
```
python gradio_demo.py \
  --soundstorm-model-path checkpoints/viitor/soundstorm \
  --dualcodec-model-path checkpoints/dualcodec \
  --w2v-path checkpoints/w2v \
  --llm-model-path checkpoints/viitor/llm/zh-en \
  --server-port 7860
```
在界面中上传 Prompt 音频，输入文本，可选填时长（秒），点击 “Synthesize” 预览生成音频。
如需减少口音泄露、跨语言时希望淡化原始口音，可勾选 “Enable two-pass speaker refinement (prompt + generated speech)”。

### 💻 命令行 Demo
基础命令（transformers 后端 + 默认 checkpoint）：
```
python cli.py \
  --prompt /path/to/prompt.wav \
  --text "你好，ViiTorVoice！" \
  --output outputs/out.wav
```
常用参数：
- `--use-vllm`：切换到 vLLM 后端。
- `--duration <秒>`：时长提示，仅单条文本时生效。
- `--speaker-windowed`：开启双阶段说话人表示优化（Prompt + 生成语音求平均，可减少口音泄露，跨语言时按需开启）。

### 🧪 场景示例
1) 单条推理（transformers）
```
python cli.py \
  --prompt data/prompt.wav \
  --text "欢迎使用 ViiTorVoice。" \
  --output outputs/single.wav
```

2) vLLM 后端
```
python cli.py \
  --use-vllm \
  --prompt data/prompt.wav \
  --text "这是 vLLM 推理示例。" \
  --output outputs/vllm.wav
```

3) 时长提示（仅单文本）
```
python cli.py \
  --prompt data/prompt.wav \
  --text "请将这句话控制在三秒左右。" \
  --duration 3.0 \
  --output outputs/with_duration.wav
```

4) 批处理：Prompt 与文本一一对应
```
python cli.py \
  --prompt data/p1.wav data/p2.wav \
  --text "第一条文本" "第二条文本" \
  --output outputs/pair_batch/
```
按顺序配对，输出自动命名。

5) 批处理：单个 Prompt，多条文本
```
python cli.py \
  --prompt data/prompt.wav \
  --text "第一条" "第二条" "第三条" \
  --output outputs/multi_text_batch/
```
生成多条音频，自动命名 `000_prompt_t0.wav` 等。

### 📣 输出日志
```
Saved -> path | text='...' | prompt='...' | set/predicted duration=3.00s | actual duration=2.95s
```
- `set/predicted duration`：指定或模型预测的时长（未指定则为预测）。
- `actual duration`：实际生成音频的时长。

## 🧭 Tips
- 确保 CUDA 与 PyTorch/vLLM 版本匹配；需更换 CUDA Wheel 可修改 `create_env.sh`。
- vLLM 需要相对充足的显存；显存紧张可改用 transformers。
- 时长提示需设定在合理范围，过于极端可能导致生成异常音频。

## 📌 TODO
- ✅ 开源中英 Base 模型
- ✅ 推理代码（本仓库与 Demo）
- ⏳ SoundStorm 训练流程
- ⏳ LLM 训练流程
- ✅ Gradio Demo
- ⏳ 情绪控制 LoRA
- ⏳ 日语、韩语、粤语模型权重
- ⏳ 基于 Flow Matching 的 semantic-to-wav 模块

## 🙌 致谢
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Amphion](https://github.com/open-mmlab/Amphion)
- [soundstorm-pytorch](https://github.com/lucidrains/soundstorm-pytorch)
- [IndexTTS](https://github.com/index-tts/index-tts)

## 🌟 产品
官网: [ViiTor AI](https://www.viitor.com/)
