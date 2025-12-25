from typing import List, Optional, Tuple

from viitor_voice.s2a.dualcodec_soundstorm import SemanticToWav
from viitor_voice.t2s.model_vllm import ViiTorVoiceVllm
import torch


class ViiTorVoiceTTS:
    def __init__(self,
                 soundstorm_model_path: str,
                 dualcodec_model_path: str,
                 w2v_path: str,
                 llm_model_path: str,
                 speaker_windowed: bool = False,
                 device='cuda'):
        self.semantic_to_wav = SemanticToWav(
            soundstorm_model_path=soundstorm_model_path,
            dualcodec_model_path=dualcodec_model_path,
            w2v_path=w2v_path,
            device=device
        )
        # Use vLLM-based inference for the language model while keeping the rest of the flow unchanged.
        # Only sample once per text; diversity now comes from providing multiple texts.
        self.llm = ViiTorVoiceVllm(llm_model_path, tensor_parallel_size=1, max_tokens=1024, n=1, temperature=0.3,
                                   speaker_windowed=speaker_windowed)

    def _synthesize_single(self,
                           prompt_semantic_tokens,
                           prompt_acoustic_tokens,
                           ref_audio,
                           sample: dict,
                           prompt_tokens_list: List[int]) -> Tuple[torch.Tensor, float, float]:
        z = sample['speech_ids']
        duration_curr = sample['duration']
        real_duration = len(z) / 25
        # Keep a small slice of the prompt to help the decoder start smoothly.
        z = prompt_tokens_list[-25:] + z
        generated_semantic_tokens = torch.tensor([z]).to(prompt_semantic_tokens[0][0].dtype).to(
            prompt_semantic_tokens[0][0].device)
        generated_audio = self.semantic_to_wav.token_to_wav(
            generated_semantic_tokens,
            prompt_semantic_tokens,
            prompt_acoustic_tokens,
            ref_audio,
        )
        generated_audio = generated_audio[:, int(-24000 * real_duration):]
        return generated_audio, duration_curr, real_duration

    def clone_batch(self,
                    prompt_audios: List[str],
                    texts_per_prompt: List[List[str]],
                    durations: Optional[List[List[Optional[float]]]] = None) -> List[
        List[Tuple[torch.Tensor, float, float, int]]]:
        """
        Run vLLM in batch to exploit parallel decoding.

        texts_per_prompt: list of texts for each prompt audio; each text generates once.
        """
        if durations is None:
            durations = [[None] * len(texts) for texts in texts_per_prompt]

        prompt_features = []
        prompt_tokens_list = []
        for prompt_audio in prompt_audios:
            prompt_semantic_tokens, prompt_acoustic_tokens, ref_audio = self.semantic_to_wav.preprocess(prompt_audio)
            prompt_features.append((prompt_semantic_tokens, prompt_acoustic_tokens, ref_audio))
            prompt_tokens_list.append(prompt_semantic_tokens[0][0].cpu().tolist())

        flat_texts = []
        flat_durations = []
        flat_prompt_tokens = []
        prompt_indices = []
        text_indices = []

        for prompt_idx, (texts, duration_list, prompt_tokens) in enumerate(
                zip(texts_per_prompt, durations, prompt_tokens_list)):
            if len(texts) != len(duration_list):
                raise ValueError(f'Prompt {prompt_idx} has {len(texts)} texts but {len(duration_list)} durations.')
            for text_idx, (text, duration) in enumerate(zip(texts, duration_list)):
                flat_texts.append(text)
                flat_durations.append(duration)
                flat_prompt_tokens.append(prompt_tokens)
                prompt_indices.append(prompt_idx)
                text_indices.append(text_idx)

        batch_results = self.llm.batch_clone(
            text_list=flat_texts,
            prompt_speech_token_list=flat_prompt_tokens,
            duration_list=flat_durations
        )

        outputs = [[] for _ in prompt_features]
        for samples, prompt_idx, text_idx in zip(batch_results, prompt_indices, text_indices):
            features = prompt_features[prompt_idx]
            prompt_tokens = prompt_tokens_list[prompt_idx]
            for sample in samples:
                generated_audio, duration_curr, real_duration = self._synthesize_single(
                    *features, sample, prompt_tokens
                )
                outputs[prompt_idx].append((generated_audio, duration_curr, real_duration, text_idx))
        return outputs

    def clone(self, prompt_audio: str, text: str, duration: float = None):
        return self.clone_batch([prompt_audio], [[text]], [[duration]])[0]
