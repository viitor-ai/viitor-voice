from viitor_voice.s2a.dualcodec_soundstorm import SemanticToWav
from viitor_voice.t2s.model import ViiTorVoiceForCausalLM
from transformers import AutoTokenizer
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
        self.llm = ViiTorVoiceForCausalLM.from_pretrained(llm_model_path, torch_dtype=torch.bfloat16).to('cuda')
        self.llm.set_speaker_windowed(speaker_windowed)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

    def format_condition(self, prompt_speech_token, text, device='cuda', duration: float = None):
        input_text = '<|cos_bos|>' + ''.join(['<|condition|>'] * 32) + text + '<|task|>'
        if duration is not None:
            duration = round(round(duration * 2) * 0.5, 1)
            input_text += '<|duration-{}|>'.format(duration)

        input_ids = self.tokenizer(input_text, add_special_tokens=False).input_ids
        condition_input_ids = prompt_speech_token + list(
            range(self.llm.config.start_condition_token_id,
                  self.llm.config.start_condition_token_id + self.llm.config.num_condition_tokens))

        return {"input_ids": torch.tensor([input_ids]).to(device),
                'condition_input_ids': torch.tensor([condition_input_ids]).to(device),
                'duration': duration}

    def clone(self, prompt_audio: str, text: str, duration: float = None):
        prompt_semantic_tokens, prompt_acoustic_tokens, ref_audio = self.semantic_to_wav.preprocess(prompt_audio)
        inputs = self.format_condition(prompt_semantic_tokens[0][0].cpu().tolist(), text, duration=duration)
        # First pass with prompt-based speaker embedding
        self.llm.set_speaker_override(None)
        output = self.llm.generate(input_ids=inputs['input_ids'],
                                   condition_input_ids=inputs['condition_input_ids'],
                                   eos_token_id=self.llm.config.eos_token_id,
                                   max_new_tokens=2048,
                                   repetition_penalty=1.5,
                                   # no_repeat_ngram_size=5,
                                   temperature=0.3, do_sample=True, top_k=50)
        z = output.tolist()[0][len(inputs['input_ids'][0]):-1]
        if self.llm.windowed_speaker:
            # Refine: compute speaker embedding from generated speech tokens, average, rerun generate.
            gen_speech_ids = z[1:] if duration is None else z
            gen_speech_ids = [x - self.llm.speech_start_idx for x in gen_speech_ids]
            prompt_emb = self.llm.get_speaker_embedding(inputs['condition_input_ids'])
            gen_emb = self.llm.speaker_from_speech_ids(gen_speech_ids)
            avg_emb = (prompt_emb + gen_emb) / 2.0
            self.llm.set_speaker_override(avg_emb)
            output = self.llm.generate(input_ids=inputs['input_ids'],
                                       condition_input_ids=inputs['condition_input_ids'],
                                       eos_token_id=self.llm.config.eos_token_id,
                                       max_new_tokens=2048,
                                       repetition_penalty=1.5,
                                       temperature=0.3, do_sample=True, top_k=50)
            z = output.tolist()[0][len(inputs['input_ids'][0]):-1]
            self.llm.set_speaker_override(None)
        if duration is None:
            duration = self.tokenizer.decode(z[0])
            duration = duration.replace('<|duration-', '').replace('|>', '')
            duration = float(duration)
            z = z[1:]
        else:
            duration = inputs['duration']

        z = [x - self.llm.speech_start_idx for x in z]
        real_duration = len(z) / 25
        z = prompt_semantic_tokens[0][0].cpu().tolist()[-25:] + z
        generated_semantic_tokens = torch.tensor([z]).to(prompt_semantic_tokens[0][0].dtype).to(
            prompt_semantic_tokens[0][0].device)
        generated_audio = self.semantic_to_wav.token_to_wav(generated_semantic_tokens, prompt_semantic_tokens,
                                                            prompt_acoustic_tokens, ref_audio)
        generated_audio = generated_audio[:, int(-24000 * real_duration):]
        return generated_audio, duration, real_duration
