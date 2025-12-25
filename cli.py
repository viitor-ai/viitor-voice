#!/usr/bin/env python3
"""
Small ViiTorVoice TTS demo that can run with either the vanilla or vLLM backend.

Besides model paths, it takes:
1) --use-vllm toggle
2) --prompt prompt audio path(s)
3) --text text(s) to synthesize
4) --output output audio path(s)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torchaudio

from viitor_voice.viitor_voice_tts import ViiTorVoiceTTS as HfViiTorVoiceTTS
from viitor_voice.viitor_voice_tts_vllm import ViiTorVoiceTTS as VllmViiTorVoiceTTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ViiTorVoice TTS demo (HF or vLLM backend). "
                    "Supports batch modes: "
                    "1) number of prompts == number of texts "
                    "2) single prompt with multiple texts."
    )
    parser.add_argument("--soundstorm-model-path", default="checkpoints/viitor/soundstorm",
                        help="Path to soundstorm model (default: checkpoints/viitor/soundstorm).")
    parser.add_argument("--dualcodec-model-path", default="checkpoints/dualcodec",
                        help="Path to dualcodec model (default: checkpoints/dualcodec).")
    parser.add_argument("--w2v-path", default="checkpoints/w2v",
                        help="Path to wav2vec model (default: checkpoints/w2v).")
    parser.add_argument("--llm-model-path", default="checkpoints/viitor/llm/zh-en",
                        help="Path to the text-to-semantic LLM (default: checkpoints/viitor/llm/zh-en).")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM backend instead of transformers generate.")
    parser.add_argument("--speaker-windowed", action="store_true",
                        help="Two-pass speaker refinement: average prompt embedding with generated-speech embedding.")
    parser.add_argument("--prompt", nargs="+", required=True, help="Prompt audio path(s).")
    parser.add_argument("--text", nargs="+", required=True, help="Text(s) to synthesize.")
    parser.add_argument("--output", nargs="+", required=True,
                        help="Output audio path. Provide one file for single result, "
                             "one directory for multiple results, or one file per result.")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Output sample rate, default 24000.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Optional duration (seconds). Only honored when exactly one text is provided.")
    return parser.parse_args()


def prepare_jobs(prompts: List[str], texts: List[str]) -> Tuple[List[str], List[List[str]], List[Dict]]:
    if len(prompts) == len(texts):
        prompt_audios = prompts
        texts_per_prompt = [[text] for text in texts]
        jobs = [{"prompt": p, "text": t, "prompt_idx": idx, "text_idx": 0} for idx, (p, t) in
                enumerate(zip(prompts, texts))]
    elif len(prompts) == 1 and len(texts) > 1:
        prompt_audios = prompts
        texts_per_prompt = [texts]
        jobs = [{"prompt": prompts[0], "text": t, "prompt_idx": 0, "text_idx": idx} for idx, t in enumerate(texts)]
    else:
        raise ValueError("Batch not supported: use either equal counts of prompts/texts or one prompt with many texts.")
    return prompt_audios, texts_per_prompt, jobs


def resolve_output_paths(output_args: List[str], jobs: List[Dict]) -> List[Path]:
    output_paths = [Path(p) for p in output_args]
    num_results = len(jobs)
    if num_results == 1:
        target = output_paths[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        return [target]

    if len(output_paths) == 1:
        out_dir = output_paths[0]
        if out_dir.suffix:
            raise ValueError("When producing multiple results, --output should be a directory or a list of files.")
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved = []
        for idx, job in enumerate(jobs):
            prompt_stem = Path(job["prompt"]).stem
            resolved.append(out_dir / f"{idx:03d}_{prompt_stem}_t{job['text_idx']}.wav")
        return resolved

    if len(output_paths) == num_results:
        for p in output_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
        return output_paths

    raise ValueError("Output paths do not match result count. Provide one dir or one file per result.")


def run_inference(use_vllm: bool,
                  prompt_audios: List[str],
                  texts_per_prompt: List[List[str]],
                  jobs: List[Dict],
                  durations: Optional[List[List[Optional[float]]]],
                  soundstorm_model_path: str,
                  dualcodec_model_path: str,
                  w2v_path: str,
                  llm_model_path: str,
                  speaker_windowed: bool):
    tts_cls = VllmViiTorVoiceTTS if use_vllm else HfViiTorVoiceTTS
    tts = tts_cls(
        soundstorm_model_path=soundstorm_model_path,
        dualcodec_model_path=dualcodec_model_path,
        w2v_path=w2v_path,
        llm_model_path=llm_model_path,
        speaker_windowed=speaker_windowed,
    )

    if use_vllm:
        raw_outputs = tts.clone_batch(prompt_audios, texts_per_prompt, durations)
        order_map = {(job["prompt_idx"], job["text_idx"]): idx for idx, job in enumerate(jobs)}
        results = [None] * len(jobs)
        for prompt_idx, prompt_outputs in enumerate(raw_outputs):
            # Sort by text_idx to align with user input ordering.
            for audio, duration, real_duration, text_idx in sorted(prompt_outputs, key=lambda x: x[3]):
                order_idx = order_map[(prompt_idx, text_idx)]
                job = jobs[order_idx]
                results[order_idx] = {
                    "audio": audio,
                    "duration": duration,
                    "real_duration": real_duration,
                    "prompt": job["prompt"],
                    "text": job["text"],
                }
    else:
        results = []
        for job in jobs:
            # Duration is only honored when there is a single text overall (validated earlier).
            duration_arg = durations[0][0] if durations else None
            audio, duration, real_duration = tts.clone(job["prompt"], job["text"], duration=duration_arg)
            results.append({
                "audio": audio,
                "duration": duration,
                "real_duration": real_duration,
                "prompt": job["prompt"],
                "text": job["text"],
            })
    return results


def main():
    args = parse_args()
    if args.duration is not None and len(args.text) != 1:
        raise ValueError("Duration is only supported when exactly one text is provided.")

    prompt_audios, texts_per_prompt, jobs = prepare_jobs(args.prompt, args.text)
    durations = None
    if args.duration is not None:
        # Only applies when text count is 1; shape to align with clone_batch signature.
        durations = [[args.duration]]

    results = run_inference(
        use_vllm=args.use_vllm,
        prompt_audios=prompt_audios,
        texts_per_prompt=texts_per_prompt,
        jobs=jobs,
        durations=durations,
        soundstorm_model_path=args.soundstorm_model_path,
        dualcodec_model_path=args.dualcodec_model_path,
        w2v_path=args.w2v_path,
        llm_model_path=args.llm_model_path,
        speaker_windowed=args.speaker_windowed,
    )

    output_paths = resolve_output_paths(args.output, jobs)
    for output_path, result in zip(output_paths, results):
        torchaudio.save(str(output_path), result["audio"].cpu(), args.sample_rate)
        set_duration = result["duration"]
        real_duration = result["real_duration"]
        set_str = f"{set_duration:.2f}s" if set_duration is not None else "-"
        real_str = f"{real_duration:.2f}s" if real_duration is not None else "-"
        print(
            f"Saved -> {output_path} | text='{result['text']}' | prompt='{result['prompt']}' "
            f"| set/predicted duration={set_str} | actual duration={real_str}"
        )


if __name__ == "__main__":
    main()
