from typing_extensions import Optional, List, Union, Tuple, Unpack

from torch import nn
import torch
from transformers import Qwen2PreTrainedModel, GenerationMixin, Qwen2Model, Cache, Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import can_return_tuple
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


class ViiTorVoiceForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.speech_start_idx = config.speech_start_idx
        self.speech_end_idx = config.speech_end_idx
        self.speech_vocab_size = self.speech_end_idx - self.speech_start_idx + 1
        self.lm_head = nn.Linear(config.hidden_size, self.speech_vocab_size, bias=True)

        if config.speaker_condition['hidden_size'] != config.hidden_size:
            self.aligner = nn.Linear(config.speaker_condition['hidden_size'], config.hidden_size)
        else:
            self.aligner = None

        self.speaker_condition = Qwen2Model(Qwen2Config(**config.speaker_condition))
        self.start_condition_token_id: int = config.start_condition_token_id
        self.num_condition_tokens: int = config.num_condition_tokens
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

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_speaker_similarity(self, condition_input_ids, target_input_ids):

        speaker_hidden_states = self._compute_speaker_hidden_states(condition_input_ids)
        target_speaker_hidden_states = self._compute_speaker_hidden_states(target_input_ids)

        return prefix_similarity(speaker_hidden_states, target_speaker_hidden_states)

    def get_speaker_embedding(self, condition_input_ids):
        return self._compute_speaker_hidden_states(condition_input_ids)

    @staticmethod
    def compare(emb1, emb2):
        return prefix_similarity(emb1, emb2)

    def set_speaker_windowed(self, enabled: bool):
        self.windowed_speaker = bool(enabled)

    def set_speaker_override(self, speaker_hidden: Optional[torch.Tensor]):
        self.speaker_hidden_override = speaker_hidden

    def speaker_from_speech_ids(self, speech_ids: List[int]) -> torch.Tensor:
        """
        Build condition_input_ids from speech ids (semantic tokens) and compute speaker embedding.
        """
        cond_ids = speech_ids + list(range(self.start_condition_token_id,
                                           self.start_condition_token_id + self.num_condition_tokens))
        device = next(self.parameters()).device
        cond_tensor = torch.tensor([cond_ids], device=device, dtype=torch.long)
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
                                                         position_ids=condition_position_ids)[0]
        return extract_pattern_hidden(condition_input_ids, condition_hidden_states,
                                      self.start_condition_token_id,
                                      self.num_condition_tokens)

    @can_return_tuple
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            condition_input_ids: Optional[torch.LongTensor] = None,
            condition_position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and speaker embedding
            if input_ids.shape[-1] != 1:
                speaker_hidden_states = self._compute_speaker_hidden_states(condition_input_ids, condition_position_ids)
                if self.aligner is not None:
                    speaker_hidden_states = self.aligner(speaker_hidden_states)

                special_audio_mask = (input_ids == self.condition_placeholder).to(inputs_embeds.device)
                special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
                speaker_hidden_states = speaker_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, speaker_hidden_states)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            adjusted_labels = labels - self.speech_start_idx
            adjusted_labels[adjusted_labels < 0] = -100
            loss = self.loss_function(logits=logits, labels=adjusted_labels, vocab_size=self.speech_vocab_size,
                                      **kwargs)

        b, s, _ = logits.shape
        new_logits = torch.full(
            (b, s, self.vocab_size),
            fill_value=-1e9,  # 极小值
            device=logits.device,
            dtype=logits.dtype
        )
        new_logits[:, :, self.speech_start_idx: self.speech_end_idx + 1] = logits
        return CausalLMOutputWithPast(
            loss=loss,
            logits=new_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
