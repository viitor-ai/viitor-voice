from typing import List
from typing import Optional, Union, Iterable

import torch
from torch import nn
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model as HfQwen2Model, Qwen2PreTrainedModel
from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import maybe_prefix, PPMissingLayer, AutoWeightsLoader
from vllm.sequence import IntermediateTensors
from collections import Counter
import torch.nn.functional as F


def prefix_similarity(
        emb_a: torch.Tensor,  # [32, D]
        emb_b: torch.Tensor,  # [32, D]
        eps: float = 1e-8,
):
    """
    输入两个 [32, D] 的 embedding，输出：
    - pos_sim: 每个位置的 cosine 相似度 [32]
    - mean_sim: 所有 32 个位置的平均相似度（标量）
    """

    assert emb_a.shape == emb_b.shape, "两个 embedding 形状必须一致"
    assert emb_a.dim() == 2, "期望输入 shape 为 [32, D]"

    # 归一化到单位向量
    a_norm = F.normalize(emb_a, dim=-1)
    b_norm = F.normalize(emb_b, dim=-1)

    # 逐位置 cosine，相当于对每一行做点积
    pos_sim = (a_norm * b_norm).sum(dim=-1)  # [32]

    # 整体平均相似度
    mean_sim = pos_sim.mean()

    return mean_sim


class ViiTorVoiceLogitsProcessor(LogitsProcessor):
    def set_idx(self, speech_start_idx, speech_end_idx):
        self.speech_start_idx = speech_start_idx
        self.speech_end_idx = speech_end_idx

    def _get_logits(
            self,
            hidden_states: torch.Tensor,
            lm_head: VocabParallelEmbedding,
            embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head,
                                            hidden_states,
                                            bias=embedding_bias)
        # Gather logits for TP
        logits = self._gather_logits(logits)

        s, _ = logits.shape
        new_logits = torch.full(
            (s, self.vocab_size),
            fill_value=-1e9,  # 极小值
            device=logits.device,
            dtype=logits.dtype
        )
        new_logits[:, self.speech_start_idx: self.speech_end_idx + 1] = logits[
            :,
            :self.speech_end_idx - self.speech_start_idx + 1]

        # Remove paddings in vocab (if any).
        if new_logits is not None:
            new_logits = new_logits[..., :self.org_vocab_size]
        return new_logits


class ViitorVoiceForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))
        self.speech_start_idx = config.speech_start_idx
        self.speech_end_idx = config.speech_end_idx
        self.speech_vocab_size = self.speech_end_idx - self.speech_start_idx + 1
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(self.speech_vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config,
                                          prefix=maybe_prefix(
                                              prefix, "lm_head"),
                                          bias=True)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = ViiTorVoiceLogitsProcessor(config.vocab_size)
        self.logits_processor.set_idx(self.speech_start_idx, self.speech_end_idx)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        # vLLM v0.12 exposes embed_input_ids instead of get_input_embeddings.
        return self.model.embed_input_ids(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
    torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["speaker_condition."]),
        )
        return loader.load_weights(weights)


ModelRegistry.register_model("ViitorVoiceForCausalLM", ViitorVoiceForCausalLM)


def extract_pattern_hidden(condition_input_ids, condition_hidden_states, start_condition_token_id,
                           num_condition_tokens):
    """
    input_ids:     LongTensor, shape (B, L)
    hidden_states: FloatTensor, shape (B, L, D)
    pattern:       LongTensor,  shape (n,)
    verify:        True 时严格检查窗口 == pattern；False 时仅以 p1 作为起点

    return: FloatTensor, shape (b * n, D)
    """
    B, L = condition_input_ids.shape
    D = condition_hidden_states.size(-1)
    device = condition_input_ids.device

    # 1) 找到所有等于 p1 的起点 (batch_idx, pos)
    starts = (condition_input_ids == start_condition_token_id).nonzero(as_tuple=False)  # (cand, 2)
    if starts.numel() == 0:
        return condition_hidden_states.new_zeros((0, D))

    # 4) 从 hidden_states 取对应窗口，再 reshape 成 (b*n, D)
    offs = torch.arange(num_condition_tokens, device=device)  # (n,)
    idx = starts[:, 1].unsqueeze(1) + offs  # (b, n)
    hs = condition_hidden_states[starts[:, 0].unsqueeze(1), idx]  # (b, n, D)
    return hs.reshape(-1, D)


class EmbeddingOnly(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class ViiTorVoiceConditionModel(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = EmbeddingOnly(config)
        self.vocab_size = config.vocab_size
        self.speech_start_idx = config.speech_start_idx
        self.speech_end_idx = config.speech_end_idx
        self.speech_vocab_size = self.speech_end_idx - self.speech_start_idx + 1

        # TODO: config中需要新增
        self.speaker_condition = HfQwen2Model(
            Qwen2Config(**config.speaker_condition))

        self.start_condition_token_id = config.start_condition_token_id
        self.num_condition_tokens = config.num_condition_tokens
        self.condition_placeholder = config.condition_placeholder
        self.windowed_speaker: bool = False
        self.speaker_window_size: int = getattr(config, "speaker_window_size", 75)
        self.speaker_window_stride: int = getattr(config, "speaker_window_stride", 25)
        self.speaker_hidden_override: Optional[torch.Tensor] = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_speaker_windowed(self, enabled: bool):
        self.windowed_speaker = bool(enabled)

    def set_speaker_override(self, speaker_hidden: Optional[torch.Tensor]):
        self.speaker_hidden_override = speaker_hidden

    def speaker_from_speech_ids(self, speech_ids: List[int]) -> torch.Tensor:
        cond_ids = speech_ids + list(range(self.start_condition_token_id,
                                           self.start_condition_token_id + self.num_condition_tokens))
        cond_tensor = torch.tensor([cond_ids], device=self.model.device, dtype=torch.long)
        return self._compute_speaker_hidden_states(cond_tensor)

    def _windowed_speaker_hidden_states(
            self,
            condition_input_ids: torch.LongTensor,
            condition_position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        raise RuntimeError("Windowed speaker embedding is deprecated in this flow.")

    def _compute_speaker_hidden_states(
            self,
            condition_input_ids: torch.LongTensor,
            condition_position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if self.speaker_hidden_override is not None:
            return self.speaker_hidden_override
        condition_hidden_states = self.speaker_condition(input_ids=condition_input_ids,
                                                         position_ids=condition_position_ids,
                                                         use_cache=False)[0]
        return extract_pattern_hidden(condition_input_ids, condition_hidden_states,
                                      self.start_condition_token_id,
                                      self.num_condition_tokens)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            condition_input_ids: Optional[torch.LongTensor] = None,
            condition_position_ids: Optional[torch.LongTensor] = None,
            **kwargs,
    ):
        # 1. Extract the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and speaker embedding
        speaker_hidden_states = self._compute_speaker_hidden_states(condition_input_ids, condition_position_ids)

        special_audio_mask = (input_ids == self.condition_placeholder).to(inputs_embeds.device)
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
        speaker_hidden_states = speaker_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, speaker_hidden_states)
        return inputs_embeds, speaker_hidden_states.view(-1, self.num_condition_tokens, speaker_hidden_states.size(
            -1))  # batch, num_condition_tokens, dim

    def get_similar(
            self,
            speaker_hidden_states: torch.Tensor,
            condition_input_ids: torch.Tensor,
            **kwargs,
    ):
        # 2. Merge text and speaker embedding
        target_speaker_hidden_states = self._compute_speaker_hidden_states(condition_input_ids, None)
        target_speaker_hidden_states = target_speaker_hidden_states.to(speaker_hidden_states.device)
        return prefix_similarity(speaker_hidden_states, target_speaker_hidden_states)


def memory_usage_ratio(reserve_gb: float = 8.0) -> float:
    """
    计算指定显存占用比例

    Args:
        reserve_gb (float): 需要占用的显存大小 (GB)，默认为 8GB

    Returns:
        float: 占用比例 (0~1 之间)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("当前没有检测到可用的 GPU")

    device = torch.device("cuda:0")
    total_mem = torch.cuda.get_device_properties(device).total_memory  # 单位: byte
    reserve_bytes = reserve_gb * 1024 ** 3
    ratio = reserve_bytes / total_mem
    return min(ratio, 1.0)


def select_dtype() -> str:
    """
    根据 GPU compute capability 决定使用的精度:
      - compute capability <= 7.5 → float16
      - 否则 → bfloat16

    Returns:
        torch.dtype: torch.float16 或 torch.bfloat16
    """
    if not torch.cuda.is_available():
        raise RuntimeError("当前没有检测到可用的 GPU")

    device = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(device)
    capability = major + minor / 10.0

    if capability <= 7.5:
        return 'float16'
    else:
        return 'bfloat16'


class ViiTorVoiceVllm:
    def __init__(self, model_path, tensor_parallel_size=1, max_tokens=1024, n=1, temperature=0.1,
                 speaker_windowed: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        stop = self.tokenizer.eos_token
        gpu_memory_utilization = memory_usage_ratio(12.0)
        dtype = select_dtype()

        self.model = LLM(model_path,
                         tensor_parallel_size=tensor_parallel_size,
                         dtype=dtype,
                         max_model_len=max_tokens,
                         gpu_memory_utilization=gpu_memory_utilization,
                         enable_prompt_embeds=True, enforce_eager=False
                         )
        self.condition_model = ViiTorVoiceConditionModel.from_pretrained(model_path,
                                                                         torch_dtype=torch.bfloat16 if dtype == 'bfloat16' else torch.float16).to(
            'cuda')
        self.condition_model.set_speaker_windowed(speaker_windowed)
        self.windowed_speaker = speaker_windowed
        self.num_condition_tokens = self.condition_model.num_condition_tokens
        self.speech_start_idx = self.condition_model.speech_start_idx
        self.start_condition_token_id = self.condition_model.start_condition_token_id
        self.speech_end_idx = self.condition_model.speech_end_idx

        self.sampling_params = SamplingParams(temperature=temperature, stop=stop,
                                              max_tokens=max_tokens, repetition_penalty=1.5, n=n)

    def set_speaker_windowed(self, enabled: bool):
        self.windowed_speaker = bool(enabled)
        self.condition_model.set_speaker_windowed(self.windowed_speaker)

    @torch.no_grad()
    def forward(self,
                input_ids: List[List[int]],
                condition_input_ids: List[List[int]]):
        merged_input_ids = []
        merged_condition_input_ids = []
        merged_condition_position_ids = []

        for x, y in zip(input_ids, condition_input_ids):
            merged_input_ids.extend(x)
            merged_condition_input_ids.extend(y)
            merged_condition_position_ids.extend(range(len(y)))

        merged_input_ids = torch.tensor([merged_input_ids], dtype=torch.long, device='cuda:0')
        merged_condition_input_ids = torch.tensor([merged_condition_input_ids], dtype=torch.long, device='cuda:0')
        merged_condition_position_ids = torch.tensor([merged_condition_position_ids], dtype=torch.long, device='cuda:0')
        prompt_embeds, speaker_hidden_states = self.condition_model(input_ids=merged_input_ids,
                                                                    condition_input_ids=merged_condition_input_ids,
                                                                    condition_position_ids=merged_condition_position_ids)
        prompt_embeds = prompt_embeds[0]
        prompt_embeds_list = []
        start_ids = 0
        for x in input_ids:
            prompt_embeds_list.append({'prompt_embeds': prompt_embeds[start_ids:start_ids + len(x)]})
            start_ids += len(x)

        outputs = self.model.generate(prompt_embeds_list, sampling_params=self.sampling_params)

        return outputs, speaker_hidden_states

    def speaker_from_speech_ids(self, speech_ids: List[int]) -> torch.Tensor:
        cond_ids = speech_ids + list(range(self.start_condition_token_id,
                                           self.start_condition_token_id + self.num_condition_tokens))
        device = next(self.condition_model.parameters()).device
        cond_tensor = torch.tensor([cond_ids], device=device, dtype=torch.long)
        return self.condition_model._compute_speaker_hidden_states(cond_tensor)

    @torch.no_grad()
    def parse_results(self, outputs, speaker_hidden_states, duration_list: List[float]):
        results = []
        for output, duration, emb in zip(outputs, duration_list, speaker_hidden_states):
            tmp = []
            for i, output_curr in enumerate(output.outputs):
                generated_ids = output_curr.token_ids[:-1]
                if duration is not None:
                    speech_ids = [x - self.speech_start_idx for x in generated_ids]
                    duration_curr = duration
                else:
                    duration_curr = float(self.tokenizer.decode(generated_ids[0]).split('-')[-1].replace('|>', ''))
                    speech_ids = [x - self.speech_start_idx for x in generated_ids[1:]]

                r = {
                    "duration": duration_curr,
                    "speech_ids": speech_ids,
                    "duration_diff": (len(speech_ids) / 25.0 - duration_curr) / duration_curr
                }
                condition_input_ids = speech_ids[1:-1] + list(
                    range(self.start_condition_token_id, self.start_condition_token_id + self.num_condition_tokens))
                r['similar'] = self.condition_model.get_similar(emb,
                                                                torch.tensor([condition_input_ids], device='cuda:0',
                                                                             dtype=torch.long)).item()
                tmp.append(r)

            results.append(tmp)
        return results

    def _format(self, text: str, prompt_speech_token: List[int], duration: float = None):

        tmp = '<|cos_bos|>' + ''.join(['<|condition|>'] * self.num_condition_tokens) + text + '<|task|>'
        if duration is not None:
            tmp += '<|duration-{}|>'.format(round(duration, 2))

        tmp_input_ids = self.tokenizer(tmp, add_special_tokens=False).input_ids
        tmp_condition_input_ids = prompt_speech_token + list(
            range(self.start_condition_token_id, self.start_condition_token_id + self.num_condition_tokens))
        return tmp_input_ids, tmp_condition_input_ids

    def batch_clone(self,
                    text_list: List[str],
                    prompt_speech_token_list: List[List[int]],
                    duration_list: List[float] = None):
        if duration_list is None:
            duration_list = [None] * len(text_list)
        if self.condition_model.windowed_speaker:
            return self._batch_clone_refine(text_list, prompt_speech_token_list, duration_list)
        input_ids = []
        condition_input_ids = []
        for text, duration, prompt_speech_token in zip(text_list, duration_list, prompt_speech_token_list):
            tmp_input_ids, tmp_condition_input_ids = self._format(text, prompt_speech_token, duration)
            input_ids.append(tmp_input_ids)
            condition_input_ids.append(tmp_condition_input_ids)
        outputs, speaker_hidden_states = self.forward(input_ids, condition_input_ids)
        results = self.parse_results(outputs, speaker_hidden_states, duration_list)
        return results

    def _batch_clone_refine(self,
                            text_list: List[str],
                            prompt_speech_token_list: List[List[int]],
                            duration_list: List[float]):
        results = []
        for text, duration, prompt_speech_token in zip(text_list, duration_list, prompt_speech_token_list):
            tmp_input_ids, tmp_condition_input_ids = self._format(text, prompt_speech_token, duration)
            # First pass with prompt-based embedding
            self.condition_model.set_speaker_override(None)
            outputs, speaker_hidden_states = self.forward([tmp_input_ids], [tmp_condition_input_ids])
            first_res = self.parse_results(outputs, speaker_hidden_states, [duration])[0]
            # Build speech ids from first pass
            speech_ids = first_res[0]["speech_ids"] if isinstance(first_res, list) and first_res else []
            # Refine embedding
            if speech_ids:
                prompt_emb = speaker_hidden_states[0]
                gen_emb = self.speaker_from_speech_ids(speech_ids)
                avg_emb = (prompt_emb + gen_emb) / 2.0
                self.condition_model.set_speaker_override(avg_emb)
                outputs, speaker_hidden_states = self.forward([tmp_input_ids], [tmp_condition_input_ids])
                refined_res = self.parse_results(outputs, speaker_hidden_states, [duration])[0]
                results.append(refined_res)
            else:
                results.append(first_res)
        self.condition_model.set_speaker_override(None)
        return results
