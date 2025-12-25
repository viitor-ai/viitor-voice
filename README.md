<h1 align="center">🚀 ViiTor Voice TTS</h1>
<p align="center">Fast, flexible speech cloning with transformers or vLLM — batch-friendly and duration-aware.</p>
<p align="center"><a href="README_zh.md">中文文档</a> · <a href="https://viitor-ai.github.io/viitor-voice/">Demo page</a> · <a href="https://github.com/viitor-ai/viitor-voice/">GitHub</a> · <a href="https://huggingface.co/ZzWater/ViiTor-voice-2.0-base">Hugging Face</a></p>

## 🍀 What it is
ViiTor Voice is a three-stage speech cloning stack:
- Stage 1: prompt + text → semantic tokens.
- Stage 2: prompt acoustic/semantic + predicted semantic → predicted acoustic tokens.
- Stage 3: acoustic tokens → waveform.

## ✨ Why it shines
- **Text-free prompts**: stronger cross-lingual cloning, less ASR dependency—raw prompts are welcome.
- **Similarity boost**: InfoNCE + condition encoder as a similarity constraint; robust even with noisy/background prompts.
- **Built-in duration control**: duration prediction in the LLM trunk; force duration with ~0.5s precision.
- **LoRA-based emotion control**: plug in LoRA adapters to steer emotion/style without full finetuning.

`cli.py` covers both backends, two batch modes, and an optional duration hint (single-text only).

## ⚡ Quickstart (Linux)
### 1) Environment
Use the provided script (PyTorch, vLLM 0.12.0 CUDA 12.8, requirements, dualcodec):
```
bash create_env.sh
source .venv/bin/activate
```
Notes:
- `create_env.sh` uses `uv venv` with Python 3.12—adjust if needed.
- vLLM install targets CUDA 12.8 (`--torch-backend=cu128`); adapt to your CUDA/toolkit.

### 2) Checkpoints
Fetch required models (Hugging Face mirror by default):
```
bash download_checkpoints.sh
```
Default paths (override via CLI if you store elsewhere):
- SoundStorm: `checkpoints/viitor/soundstorm`
- DualCodec:  `checkpoints/dualcodec`
- wav2vec:    `checkpoints/w2v`
- LLM:        `checkpoints/viitor/llm/zh-en`

## 🎯 Demo usage
### 🖥️ Gradio demo
Launch a web UI (hosted on `0.0.0.0`, Gradio share disabled):
```
python gradio_demo.py \
  --soundstorm-model-path checkpoints/viitor/soundstorm \
  --dualcodec-model-path checkpoints/dualcodec \
  --w2v-path checkpoints/w2v \
  --llm-model-path checkpoints/viitor/llm/zh-en \
  --server-port 7860
```
Upload a prompt audio file in the UI, type text, optionally set a duration (seconds), then click “Synthesize” to preview the generated audio.
Toggle “Enable two-pass speaker refinement (prompt + generated speech)” to reduce accent leakage; helpful for cross-language cloning when you want lighter source accent.

### 💻 CLI demo
Base command (transformers backend + default checkpoints):
```
python cli.py \
  --prompt /path/to/prompt.wav \
  --text "Hello ViiTorVoice!" \
  --output outputs/out.wav
```
Common flags:
- `--use-vllm` switch to vLLM.
- `--duration <seconds>` duration hint; honored only when exactly one text.
- `--speaker-windowed` enable two-pass speaker refinement (average prompt embedding with generated-speech embedding; reduces accent leakage, useful for cross-language cloning).

### 🧪 Cases
1) Single inference (transformers)
```
python cli.py \
  --prompt data/prompt.wav \
  --text "Welcome to ViiTorVoice." \
  --output outputs/single.wav
```

2) vLLM backend
```
python cli.py \
  --use-vllm \
  --prompt data/prompt.wav \
  --text "This runs with vLLM." \
  --output outputs/vllm.wav
```

3) Duration hint (single text)
```
python cli.py \
  --prompt data/prompt.wav \
  --text "Keep this around three seconds." \
  --duration 3.0 \
  --output outputs/with_duration.wav
```

4) Batch: prompts and texts 1:1
```
python cli.py \
  --prompt data/p1.wav data/p2.wav \
  --text "First line" "Second line" \
  --output outputs/pair_batch/
```
Paired by order; outputs auto-named in the directory.

5) Batch: one prompt, many texts
```
python cli.py \
  --prompt data/prompt.wav \
  --text "Line 1" "Line 2" "Line 3" \
  --output outputs/multi_text_batch/
```
Generates multiple files, auto-named `000_prompt_t0.wav`, etc.

### 📣 Output log
```
Saved -> path | text='...' | prompt='...' | set/predicted duration=3.00s | actual duration=2.95s
```
- `set/predicted duration`: provided duration (or model-predicted if none)
- `actual duration`: measured from generated audio

## 🧭 Tips
- Ensure CUDA driver/toolkit matches the PyTorch/vLLM build; edit `create_env.sh` if you need a different CUDA wheel.
- vLLM prefers generous GPU memory; fall back to transformers if constrained.
- Set duration hints reasonably; extreme values can produce abnormal audio.

## 📌 TODO
- ✅ Open-sourced Chinese/English base model
- ✅ Inference code (this repo and demo)
- ⏳ SoundStorm training recipe
- ⏳ LLM training recipe
- ✅ Gradio demo
- ⏳ Emotion-control LoRA
- ⏳ Japanese, Korean, Cantonese model weights
- ⏳ Flow matching–based semantic-to-wav module

## 🙌 Acknowledgments
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Amphion](https://github.com/open-mmlab/Amphion)
- [soundstorm-pytorch](https://github.com/lucidrains/soundstorm-pytorch)
- [IndexTTS](https://github.com/index-tts/index-tts)

## 🌟 Product
Official site: [ViiTor AI](https://www.viitor.com/)
