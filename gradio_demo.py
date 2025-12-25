#!/usr/bin/env python3
"""
Gradio demo for ViiTorVoice TTS.

Usage example:
python gradio_demo.py --soundstorm-model-path checkpoints/viitor/soundstorm \\
    --dualcodec-model-path checkpoints/dualcodec \\
    --w2v-path checkpoints/w2v \\
    --llm-model-path checkpoints/viitor/llm/zh-en
"""

import argparse
from typing import Optional, Tuple

import gradio as gr
import numpy as np

from viitor_voice.viitor_voice_tts import ViiTorVoiceTTS as HfViiTorVoiceTTS
from viitor_voice.viitor_voice_tts_vllm import ViiTorVoiceTTS as VllmViiTorVoiceTTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for ViiTorVoice TTS.")
    parser.add_argument(
        "--soundstorm-model-path",
        default="checkpoints/viitor/soundstorm",
        help="Path to soundstorm model (default: checkpoints/viitor/soundstorm).",
    )
    parser.add_argument(
        "--dualcodec-model-path",
        default="checkpoints/dualcodec",
        help="Path to dualcodec model (default: checkpoints/dualcodec).",
    )
    parser.add_argument(
        "--w2v-path",
        default="checkpoints/w2v",
        help="Path to wav2vec model (default: checkpoints/w2v).",
    )
    parser.add_argument(
        "--llm-model-path",
        default="checkpoints/viitor/llm/zh-en",
        help="Path to the text-to-semantic LLM (default: checkpoints/viitor/llm/zh-en).",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM backend instead of transformers generate.",
    )
    parser.add_argument(
        "--speaker-windowed",
        action="store_true",
        help="Compute speaker embeddings with sliding-window averaging instead of full prompt.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Output sample rate, default 24000.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port for Gradio server (default: 7860).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP for Gradio server (default: 0.0.0.0).",
    )
    return parser.parse_args()


class TtsRunner:
    def __init__(
        self,
        use_vllm: bool,
        soundstorm_model_path: str,
        dualcodec_model_path: str,
        w2v_path: str,
        llm_model_path: str,
        sample_rate: int,
        speaker_windowed: bool,
    ):
        tts_cls = VllmViiTorVoiceTTS if use_vllm else HfViiTorVoiceTTS
        self.tts = tts_cls(
            soundstorm_model_path=soundstorm_model_path,
            dualcodec_model_path=dualcodec_model_path,
            w2v_path=w2v_path,
            llm_model_path=llm_model_path,
            speaker_windowed=speaker_windowed,
        )
        self.use_vllm = use_vllm
        self.sample_rate = sample_rate
        self.speaker_windowed = speaker_windowed

    def __call__(
        self,
        prompt_audio_path: Optional[str],
        text: Optional[str],
        duration: Optional[float],
        windowed: bool,
    ) -> Tuple[Tuple[int, np.ndarray], str]:
        if not prompt_audio_path:
            raise gr.Error("Please upload a prompt audio file.")
        if text is None or not text.strip():
            raise gr.Error("Please enter text to synthesize.")

        if windowed != self.speaker_windowed:
            # Update mode on the fly without reinstantiating models.
            try:
                self.tts.llm.set_speaker_windowed(windowed)
            except AttributeError:
                pass
            self.speaker_windowed = windowed

        text_clean = text.strip()
        duration_val = float(duration) if duration not in (None, "", np.nan) else None
        if duration_val is not None and duration_val <= 0:
            duration_val = None

        if self.use_vllm:
            outputs = self.tts.clone_batch([prompt_audio_path], [[text_clean]], [[duration_val]])[0]
            if not outputs:
                raise gr.Error("生成失败：没有返回结果。")
            audio, set_duration, real_duration, _ = outputs[0]
        else:
            audio, set_duration, real_duration = self.tts.clone(
                prompt_audio_path, text_clean, duration=duration_val
            )

        waveform = audio.squeeze(0).cpu().numpy()
        details = (
            f"Text: {text_clean}\n"
            f"Requested duration: {set_duration if set_duration is not None else '-'}s | "
            f"Actual duration: {real_duration if real_duration is not None else '-'}s"
        )
        return (self.sample_rate, waveform.astype(np.float32)), details


def build_interface(runner: TtsRunner, sample_rate: int) -> gr.Blocks:
    with gr.Blocks(title="ViiTorVoice TTS") as demo:
        gr.Markdown("## ViiTorVoice TTS Gradio Demo")

        prompt_audio = gr.Audio(
            label="Prompt audio (upload)",
            type="filepath",
            sources=["upload"],
            waveform_options=gr.WaveformOptions(sample_rate=sample_rate),
        )
        text_input = gr.Textbox(label="Text to synthesize", lines=4, placeholder="Enter text here")
        duration_input = gr.Number(
            label="Duration (seconds, optional)",
            value=None,
            precision=1,
            minimum=0,
            maximum=None,
        )
        windowed_checkbox = gr.Checkbox(
            label="Enable two-pass speaker refinement (prompt + generated speech, reduces accent leakage)",
            value=runner.speaker_windowed,
        )
        generate_btn = gr.Button("Synthesize")

        output_audio = gr.Audio(label="Generated audio", type="numpy")
        info_box = gr.Textbox(label="Generation info", interactive=False)

        generate_btn.click(
            fn=runner,
            inputs=[prompt_audio, text_input, duration_input, windowed_checkbox],
            outputs=[output_audio, info_box],
            api_name=False,
        )

    return demo


def main():
    args = parse_args()
    runner = TtsRunner(
        use_vllm=args.use_vllm,
        soundstorm_model_path=args.soundstorm_model_path,
        dualcodec_model_path=args.dualcodec_model_path,
        w2v_path=args.w2v_path,
        llm_model_path=args.llm_model_path,
        sample_rate=args.sample_rate,
        speaker_windowed=args.speaker_windowed,
    )

    demo = build_interface(runner, sample_rate=args.sample_rate)
    demo.launch(server_name=args.host, server_port=args.server_port, share=False)


if __name__ == "__main__":
    main()
