# This file includes code copied/adapted from:
#   open-mmlab/Amphion
#   https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct/maskgct_s2a.py
#
# The original project is licensed under the MIT License.
# You must retain the copyright and permission notice when redistributing.

import hashlib
import math

import dualcodec
import numpy as np
import onnxruntime as ort
import pyloudnorm as pyln
import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from omegaconf import OmegaConf
from transformers import SeamlessM4TFeatureExtractor
import os


def _to_numpy_for_pyloudnorm(x: torch.Tensor):
    """
    x: torch.Tensor, shape (T,) 或 (C, T)
    返回: np.ndarray, shape (T,) 或 (T, C)
    """
    if x.ndim == 1:
        return x.detach().cpu().numpy()
    elif x.ndim == 2:
        # (C, T) -> (T, C)
        return x.detach().cpu().transpose(0, 1).numpy()
    else:
        raise ValueError("只支持 1D (T,) 或 2D (C, T) 音频张量")


def _from_numpy_like_torch(x_np: np.ndarray, like: torch.Tensor):
    """
    x_np: shape (T,) 或 (T, C)
    like: 原始 torch.Tensor, 用来恢复 shape 和 dtype
    返回: torch.Tensor, shape 与 like 相同
    """
    x_t = torch.from_numpy(x_np).to(like.device)
    if like.ndim == 1:
        return x_t.to(like.dtype)
    elif like.ndim == 2:
        # (T, C) -> (C, T)
        return x_t.transpose(0, 1).to(like.dtype)
    else:
        raise ValueError("只支持 1D (T,) 或 2D (C, T) 音频张量")


def match_loudness_lufs(
        target: torch.Tensor,
        ref: torch.Tensor,
        sample_rate: int,
        peak_limit: float = 0.99,
        max_gain_db: float = 12.0,  # 最大允许放大（+dB）
        max_cut_db: float = 12.0,  # 最大允许衰减（-dB）
        eps: float = 1e-12,
):
    """
    使用 pyloudnorm（LUFS）将 target 的响度匹配到 ref。

    参数:
        target: 目标音频, shape (T,) 或 (C, T), float32/float64, 一般范围 [-1, 1]
        ref:    参考音频, shape 同上（长度可以不同）
        sample_rate: 采样率（Hz）
        peak_limit: 峰值绝对值上限（防止削波），例如 0.99
        max_gain_db: 最大允许放大值（dB），防止从极小音量暴力拉升
        max_cut_db: 最大允许衰减值（dB），防止过度变小
        eps: 数值稳定项

    返回:
        matched: 匹配响度后的 target (torch.Tensor)，shape 与 target 相同
        gain_db: 实际使用的增益（dB, float）
    """
    # ---- 1. 转成 numpy，适配 pyloudnorm ----
    tgt_np = _to_numpy_for_pyloudnorm(target).astype(np.float64)
    ref_np = _to_numpy_for_pyloudnorm(ref).astype(np.float64)

    # ---- 2. 创建 LUFS meter（ITU-R BS.1770）----
    meter = pyln.Meter(sample_rate)  # 默认就是 BS.1770

    # ---- 3. 计算参考和目标的 integrated loudness (LUFS) ----
    ref_lufs = meter.integrated_loudness(ref_np)
    tgt_lufs = meter.integrated_loudness(tgt_np)

    # ---- 4. 计算理论需要的增益 (dB) ----
    #     gain_db > 0 -> 放大；gain_db < 0 -> 衰减
    gain_db = ref_lufs - tgt_lufs

    # 限制增益范围，避免拉升/衰减太狠
    gain_db = float(np.clip(gain_db, -max_cut_db, max_gain_db))

    # ---- 5. 防削波：保证应用增益后峰值不超过 peak_limit ----
    peak = float(np.max(np.abs(tgt_np)))  # 当前峰值
    if peak < eps:
        # 完全静音，直接返回原始 target
        return target.clone(), 0.0

    current_peak_db = 20.0 * np.log10(max(peak, eps))
    max_peak_db = 20.0 * np.log10(max(peak_limit, eps))

    # 新峰值 dB = 当前峰值 dB + gain_db，必须 <= max_peak_db
    allowed_gain_from_peak_db = max_peak_db - current_peak_db

    # 实际使用的增益 = 既要满足目标 LUFS，又不能削波
    gain_db = min(gain_db, allowed_gain_from_peak_db)

    # ---- 6. dB -> 线性增益，应用到波形上 ----
    gain_lin = 10.0 ** (gain_db / 20.0)
    matched_np = tgt_np * gain_lin

    # 再保险：极端情况下做一次 clamp
    matched_np = np.clip(matched_np, -peak_limit, peak_limit)

    # ---- 7. 转回 torch，保持原来的 dtype / shape ----
    matched = _from_numpy_like_torch(matched_np, target)

    return matched, gain_db


TYPING_MAP = {
    "TYPE_BOOL": {
        "torch": torch.bool,
        "numpy": np.bool_
    },
    "TYPE_UINT8": {
        "torch": torch.uint8,
        "numpy": np.uint8
    },
    "TYPE_UINT16": {
        "torch": torch.uint16,
        "numpy": np.uint16
    },
    "TYPE_UINT32": {
        "torch": torch.uint32,
        "numpy": np.uint32
    },
    "TYPE_UINT64": {
        "torch": torch.uint64,
        "numpy": np.uint64
    },
    "TYPE_INT8": {
        "torch": torch.int8,
        "numpy": np.int8
    },
    "TYPE_INT16": {
        "torch": torch.int16,
        "numpy": np.int16
    },
    "TYPE_INT32": {
        "torch": torch.int32,
        "numpy": np.int32
    },
    "TYPE_INT64": {
        "torch": torch.int64,
        "numpy": np.int64
    },
    "TYPE_FP16": {
        "torch": torch.float16,
        "numpy": np.float16
    },
    "TYPE_FP32": {
        "torch": torch.float32,
        "numpy": np.float32
    },
    "TYPE_FP64": {
        "torch": torch.float64,
        "numpy": np.float64
    }
}


class DiffusionOrt:
    def __init__(
            self,
            onnx_model_path: str,
            seq_min: int = 128,
            seq_opt: int = 1024,
            seq_max: int = 4096,
    ):
        self.inputs = {
            "x": "TYPE_FP32",
            "diffusion_step": "TYPE_FP32",
            "cond": "TYPE_FP32",
            "mask": "TYPE_FP32",
        }
        self.outputs = {
            "output": "TYPE_FP32",
        }

        self.device_id = torch.cuda.current_device()
        self.device = f"cuda:{self.device_id}"

        trt_profile_min = (
            f"x:1x{seq_min}x1024,"
            f"cond:1x{seq_min}x1024,"
            f"mask:1x{seq_min},"
            f"diffusion_step:1"
        )
        trt_profile_opt = (
            f"x:1x{seq_opt}x1024,"
            f"cond:1x{seq_opt}x1024,"
            f"mask:1x{seq_opt},"
            f"diffusion_step:1"
        )
        trt_profile_max = (
            f"x:1x{seq_max}x1024,"
            f"cond:1x{seq_max}x1024,"
            f"mask:1x{seq_max},"
            f"diffusion_step:1"
        )

        trt_providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": self.device_id,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_cache_{}".format(
                        hashlib.md5(onnx_model_path.encode("utf-8")).hexdigest()),
                    "trt_profile_min_shapes": trt_profile_min,
                    "trt_profile_opt_shapes": trt_profile_opt,
                    "trt_profile_max_shapes": trt_profile_max,
                    "trt_layer_norm_fp32_fallback": True,
                },
            )
        ]

        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=trt_providers,
        )
        self.io_binding = self.session.io_binding()

    def __call__(
            self,
            x: torch.Tensor,
            diffusion_step: torch.Tensor,
            cond: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:             (1, seq_len, 1024)
        cond:          (1, seq_len, 1024)
        mask:          (1, seq_len)
        diffusion_step:(1,)
        return :
        output:        (1, seq_len, 1024)
        """
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()

        x = x.to(self.device, non_blocking=True)
        cond = cond.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        diffusion_step = diffusion_step.to(self.device, non_blocking=True)

        input_tensors = {
            "x": x,
            "diffusion_step": diffusion_step,
            "cond": cond,
            "mask": mask,
        }

        for name, tensor in input_tensors.items():
            np_dtype = TYPING_MAP[self.inputs[name]]["numpy"]
            self.io_binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=np_dtype,
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        output_tensor = torch.empty_like(x, device=self.device)
        self.io_binding.bind_output(
            name="output",
            device_type="cuda",
            device_id=self.device_id,
            element_type=TYPING_MAP[self.outputs["output"]]["numpy"],
            shape=tuple(output_tensor.shape),
            buffer_ptr=output_tensor.data_ptr(),
        )

        self.session.run_with_iobinding(self.io_binding)

        return output_tensor


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class MaskGCT_S2A_infer(nn.Module):
    def __init__(
            self,
            num_quantizer=12,
            hidden_size=1024,
            num_layers=16,
            num_heads=16,
            codebook_size=1024,
            cfg_scale=0.15,
            mask_layer_schedule="linear",
            cond_codebook_size=1024,
            cond_dim=1024,
            predict_layer_1=True,
            mask_all_tokens=False,
            cfg=None,
            onnx_model_path='model.onnx'
    ):
        super().__init__()

        num_quantizer = (
            cfg.num_quantizer
            if cfg is not None and hasattr(cfg, "num_quantizer")
            else num_quantizer
        )
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        num_layers = (
            cfg.num_layers
            if cfg is not None and hasattr(cfg, "num_layers")
            else num_layers
        )
        num_heads = (
            cfg.num_heads
            if cfg is not None and hasattr(cfg, "num_heads")
            else num_heads
        )
        codebook_size = (
            cfg.codebook_size
            if cfg is not None and hasattr(cfg, "codebook_size")
            else codebook_size
        )
        cfg_scale = (
            cfg.cfg_scale
            if cfg is not None and hasattr(cfg, "cfg_scale")
            else cfg_scale
        )
        mask_layer_schedule = (
            cfg.mask_layer_schedule
            if cfg is not None and hasattr(cfg, "mask_layer_schedule")
            else mask_layer_schedule
        )
        cond_codebook_size = (
            cfg.cond_codebook_size
            if cfg is not None and hasattr(cfg, "cond_codebook_size")
            else cond_codebook_size
        )
        cond_dim = (
            cfg.cond_dim if cfg is not None and hasattr(cfg, "cond_dim") else cond_dim
        )
        predict_layer_1 = (
            cfg.predict_layer_1
            if cfg is not None and hasattr(cfg, "predict_layer_1")
            else predict_layer_1
        )

        self.num_quantizer = num_quantizer
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.mask_layer_schedule = mask_layer_schedule
        self.cond_codebook_size = cond_codebook_size
        self.cond_dim = cond_dim
        self.predict_layer_1 = predict_layer_1

        self.layer_emb = nn.Embedding(self.num_quantizer, self.hidden_size)
        self.mask_emb = nn.Embedding(1, self.hidden_size)

        self.token_emb = torch.nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, self.hidden_size)
                for _ in range(self.num_quantizer)
            ]
        )

        self.to_logits = torch.nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.codebook_size)
                for _ in range(self.num_quantizer)
            ]
        )

        self.cond_emb = nn.Embedding(cond_codebook_size, self.hidden_size)

        self.diff_estimator = DiffusionOrt(onnx_model_path, 5, 2048, 4096)
        self.mask_all_tokens = mask_all_tokens

    def mask_prob(self, t):
        return torch.sin(t * np.pi / 2).to(t.device)

    @torch.no_grad()
    def reverse_diffusion(
            self,
            cond,
            prompt,
            x_mask=None,
            prompt_mask=None,
            temp=1.5,
            filter_thres=0.98,
            max_layer=None,
            gt_code=None,
            n_timesteps=[10, 4, 4, 4, 4, 4, 4, 4],
            cfg=1.0,
            rescale_cfg=1.0,
    ):

        assert (
                len(n_timesteps) == self.num_quantizer
        )  # each layer has a number of steps

        prompt_code = prompt  # (B, prompt_len, num_quantizer)
        prompt_len = prompt_code.shape[1]
        target_len = cond.shape[1] - prompt_len

        if x_mask == None:
            x_mask = torch.ones(cond.shape[0], target_len).to(cond.device)  # (B, T)
        if prompt_mask == None:
            prompt_mask = torch.ones(cond.shape[0], prompt_len).to(
                cond.device
            )  # (B, prompt_len)

        cum = torch.zeros(x_mask.shape[0], x_mask.shape[1], self.hidden_size).to(
            x_mask.device
        )  # (B, T, hidden_size)

        bsz, seq_len, _ = cum.shape

        choice_temp = 1.0
        start_temp = temp  # temperature for sampling
        start_choice_temp = choice_temp  # temperature for choicing mask tokens

        if max_layer is None:
            max_layer = self.num_quantizer

        xt = torch.LongTensor(bsz, seq_len, max_layer).to(x_mask.device)

        if gt_code is not None:
            gt_layer = gt_code.shape[-1]
            xt[:, :, :gt_layer] = gt_code
            for i in range(gt_layer):
                cum += self.token_emb[i](xt[:, :, i])
        else:
            gt_layer = 0

        for mask_layer in range(gt_layer, max_layer):
            steps = n_timesteps[mask_layer]
            to_logits = self.to_logits[mask_layer]
            token_emb = self.token_emb[mask_layer]
            mask_layer = torch.tensor(mask_layer).to(x_mask.device).long().unsqueeze(0)
            mask_layer_cond = self.layer_emb(mask_layer).unsqueeze(
                1
            )  # (1,) -> (1, 1, hidden_size)
            temp_cond = cond + mask_layer_cond  # (B, T, hidden_size)

            mask_token = self.mask_emb(torch.zeros_like(mask_layer))  # (1, hidden_size)
            mask = torch.full((bsz, seq_len, 1), True).to(x_mask.device)  # (B, T, 1)
            seq = torch.full((bsz, seq_len), 0).to(x_mask.device)

            h = 1.0 / steps

            # prompt_code: (B, prompt_len, num_quantizer)
            cur_prompt = 0
            for idx, emb in enumerate(self.token_emb):
                cur_prompt = cur_prompt + emb(
                    prompt_code[:, :, idx]
                )  # (B, prompt_len, hidden_size)

            t_list = [1.0 - i * h for i in range(steps)]
            t_list.append(0.0)
            for i in range(steps):
                t = t_list[i] * torch.ones(bsz).to(x_mask.device)
                token = token_emb(seq)  # (B, T, hidden_size)
                cur = cum + mask * mask_token[:, None, :] + (~mask) * token
                cur = cur + mask_token[:, None, :] * (max_layer - 1 - mask_layer)

                xt_input = torch.cat([cur_prompt, cur], dim=1)  # (B, T, hidden_size)
                xt_mask = torch.cat(
                    [prompt_mask, x_mask], dim=1
                )  # (B, T), mask is 0 for padding

                # torch.cuda.synchronize()
                embeds = self.diff_estimator(xt_input, t, temp_cond, xt_mask)
                # torch.cuda.synchronize()
                embeds = embeds[:, prompt_len:, :]

                # cfg
                if cfg > 0:
                    mask_embeds = self.diff_estimator(
                        cur, t, temp_cond[:, prompt_len:, :], x_mask
                    )
                    pos_emb_std = embeds.std()  # std(g_cond)
                    embeds = embeds + cfg * (embeds - mask_embeds)  # g_cfg
                    rescale_embeds = embeds * pos_emb_std / embeds.std()  # g_final
                    embeds = rescale_cfg * rescale_embeds + (1 - rescale_cfg) * embeds

                logits = to_logits(embeds)  # (B, T, codebook_size)
                annealing_scale = t_list[i]

                choice_temp = start_choice_temp * annealing_scale
                temp = start_temp * annealing_scale
                logits = top_k(logits, filter_thres)

                if i == steps - 1:
                    # greedy
                    if steps == 1:
                        temp = 0.2
                        sampled_ids = gumbel_sample(logits, temperature=max(temp, 1e-3))
                    else:
                        sampled_ids = logits.argmax(dim=-1)

                else:
                    # sampling
                    sampled_ids = gumbel_sample(logits, temperature=max(temp, 1e-3))

                seq = torch.where(mask.squeeze(-1), sampled_ids, seq)

                scores = logits.softmax(dim=-1)
                scores = scores.gather(2, rearrange(sampled_ids, "b n -> b n 1"))
                scores = rearrange(scores, "b n 1 -> b n")

                scores = choice_temp * gumbel_noise(scores) + scores
                scores = 1 - scores

                next_t = t_list[i + 1] * torch.ones(bsz).to(x_mask.device)

                next_mask_num = (self.mask_prob(next_t) * seq_len).long()[0].item()

                if next_mask_num == 0:
                    break
                scores = scores.masked_fill(
                    ~mask.squeeze(-1), -torch.finfo(scores.dtype).max
                )

                mask_indices = scores.topk(next_mask_num, dim=-1).indices
                mask = torch.zeros_like(scores, dtype=torch.bool).scatter(
                    1, mask_indices, True
                )
                seq = seq.masked_fill(mask, 0)

                mask = mask.unsqueeze(-1)

            cum = cum + token_emb(seq)
            xt[..., mask_layer.squeeze(0).item()] = seq

        return xt


def match_loudness_to_ref(
        target: torch.Tensor,
        ref: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-8,
        clamp: bool = True,
):
    """
    将 target 音频的响度（RMS）匹配到 ref 音频。

    参数:
        target: 目标音频张量，形状 (..., T)
        ref:    参考音频张量，形状 (..., T) 或可广播到 target
        dim:    时间维，一般是 -1
        eps:    避免除零的数值稳定项
        clamp:  是否把输出限制在 [-1, 1]（常见 wave 规范）

    返回:
        scaled_target: 已经匹配响度后的 target，形状同 target
        scale:         实际使用的缩放因子（可用于调试/记录）
    """

    # 计算参考音频的 RMS
    ref_rms = ref.pow(2).mean(dim=dim, keepdim=True).sqrt()

    # 计算目标音频的 RMS
    tgt_rms = target.pow(2).mean(dim=dim, keepdim=True).sqrt()

    # 计算缩放因子，避免除零
    scale = (ref_rms + eps) / (tgt_rms + eps)

    # 如果目标几乎是静音，可以选择不缩放（防止爆炸）
    # 这里给个简单的保护：当目标 RMS 非常小，就把 scale 限制一下
    very_small = tgt_rms < eps
    if very_small.any():
        # 这些位置我们就不调整响度（scale=1）
        scale = torch.where(very_small, torch.ones_like(scale), scale)

    scaled_target = target * scale

    # 如果是标准音频波形，通常范围在 [-1, 1]，可以选择性 clamp
    if clamp:
        scaled_target = scaled_target.clamp(-1.0, 1.0)

    return scaled_target, scale


def volume_norm_rms(
        wav: torch.Tensor,
        target_db: float = -16.0,
        eps: float = 1e-8,
):
    """
    使用 RMS 把音量归一到 target_db (单位: dBFS)

    输入:
        wav: (..., time)  任意前置维度都可以 (batch, ch, time) / (ch, time) / (time,)
    输出:
        same shape 的张量，音量已经归一
    """
    # 确保是浮点
    if not wav.is_floating_point():
        wav = wav.float()

    # 以最后一维 time 为轴计算 RMS
    rms = torch.sqrt(torch.mean(wav ** 2, dim=-1, keepdim=True) + eps)  # (..., 1)

    # 当前 dB
    current_db = 20.0 * torch.log10(rms + eps)  # (..., 1)

    # 需要调整的增益（dB）
    gain_db = target_db - current_db  # (..., 1)

    # dB -> 线性
    gain = 10.0 ** (gain_db / 20.0)  # (..., 1)

    # 应用增益
    wav_norm = wav * gain

    # 可选：防止超过 [-1, 1]，可以夹一下
    wav_norm = torch.clamp(wav_norm, -1.0, 1.0)
    return wav_norm


def rms_db(x: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    """
    计算 RMS 对应的 dB 值（类似 dBFS, 不做绝对标定）
    返回 shape: 与 x 在 dim 维上压缩后的形状（keepdim=True）
    """
    rms = x.pow(2).mean(dim=dim, keepdim=True).sqrt()
    db = 20.0 * torch.log10(rms + eps)
    return db


def match_loudness_rms_db(
        target: torch.Tensor,
        ref: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-12,
        max_gain_db: float = 12.0,  # 允许的最大放大
        max_cut_db: float = 12.0,  # 允许的最大衰减
        peak_limit: float = 0.99,  # 峰值不超过的绝对值
):
    """
    用 RMS 的 dB 值，把 target 的响度匹配到 ref，带 dB 限制及防削波。

    target, ref: (..., T)
    dim: 时间维。
    """

    # 1) 计算参考 & 目标的 RMS(dB)
    ref_db = rms_db(ref, dim=dim, eps=eps)  # (..., 1)
    tgt_db = rms_db(target, dim=dim, eps=eps)

    # 2) 理论需要的增益（dB）
    #    gain_db > 0 表示放大，< 0 表示衰减
    gain_db = ref_db - tgt_db

    # 3) 限制增益范围（防止从极小音量暴力拉升）
    gain_db = torch.clamp(gain_db, -max_cut_db, max_gain_db)

    # 4) 防止削波：保证 |target * gain| 不超过 peak_limit
    peak = target.abs().amax(dim=dim, keepdim=True)  # (..., 1)

    # 如果 peak 为 0（完全静音），就不做增益调整
    silent_mask = peak < eps

    # 允许的最大增益（dB），保证 peak * gain <= peak_limit
    # peak * g <= peak_limit  => g <= peak_limit / peak
    allowed_gain_db_from_peak = 20.0 * torch.log10(
        (peak_limit / (peak + eps)).clamp(min=eps)
    )

    # 实际使用的增益 = 理想增益 和 防削波允许增益 取较小值
    gain_db = torch.minimum(gain_db, allowed_gain_db_from_peak)

    # 完全静音的位置就不改变增益（置 0dB）
    gain_db = torch.where(silent_mask, torch.zeros_like(gain_db), gain_db)

    # 5) dB -> 线性
    gain_linear = 10.0 ** (gain_db / 20.0)

    matched = target * gain_linear
    return matched, gain_db, gain_linear


class DualCodecProcessor:
    def __init__(self, model_path='/mnt/data0/xialang/latent-flow-matching/thirdparty_models/w2v-bert-2.0'):
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            model_path)

    def __call__(self, audio: torch.Tensor, sr: int):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        audio_16k = list(audio_16k.cpu().split(1, dim=0))
        inputs = self.feature_extractor(
            audio_16k, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        return {'input_features': input_features, 'attention_mask': attention_mask}


class SemanticToWav:
    def __init__(self,
                 soundstorm_model_path: str,
                 dualcodec_model_path: str,
                 w2v_path: str,
                 device='cuda'):
        config_path = os.path.join(soundstorm_model_path, 'model.yaml')
        ckpt_path = os.path.join(soundstorm_model_path, 'ckpt.pt')
        onnx_path = os.path.join(soundstorm_model_path, 'model.onnx')
        self.full_model = self.load_from_config(config_path, ckpt_path, onnx_path).to(device)
        self.device = device
        self.dualcodec_model = dualcodec.get_model('25hz_v1', dualcodec_model_path).eval().to(device)
        self.dualcodec_inference = dualcodec.Inference(dualcodec_model=self.dualcodec_model,
                                                       dualcodec_path=dualcodec_model_path,
                                                       w2v_path=w2v_path, device=device)
        self.processor = DualCodecProcessor(w2v_path)

    @staticmethod
    def load_from_config(config_path, ckpt_path, onnx_path):
        cfg = OmegaConf.load(config_path)
        model = MaskGCT_S2A_infer(**cfg.model, onnx_model_path=onnx_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        key_to_remove = []
        for key in ckpt['model_state_dict']:
            if key.startswith('diff_estimator'):
                key_to_remove.append(key)
        for key in key_to_remove:
            ckpt['model_state_dict'].pop(key)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.eval()

    @torch.no_grad()
    def semantic_to_acoustic(
            self,
            semantic_code,
            acoustic_code,
            n_timesteps=(10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            cfg=2.5,
            rescale_cfg=0.75,
    ):
        cond = self.full_model.cond_emb(semantic_code)
        prompt = acoustic_code[:, :, :]
        predict_full = self.full_model.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
            gt_code=None,
        )
        return predict_full

    @staticmethod
    def _trim_long_silences(
        audio: torch.Tensor,
        sample_rate: int,
        max_silence_sec: float = 0.5,
        silence_threshold: float = 1e-4,
    ) -> torch.Tensor:
        """
        If a silent span is longer than max_silence_sec, trim its middle so the
        remaining silence is capped at max_silence_sec (keeping edges for smooth joins).
        """
        num_samples = audio.shape[-1]
        max_silence_samples = int(max_silence_sec * sample_rate)
        if max_silence_samples <= 0 or num_samples == 0:
            return audio

        # Use the maximum absolute value across channels to detect silence.
        energy = torch.max(audio.abs(), dim=0).values
        silent = energy < silence_threshold

        keep = torch.ones(num_samples, dtype=torch.bool, device=audio.device)
        idx = 0
        while idx < num_samples:
            if not silent[idx]:
                idx += 1
                continue
            start = idx
            while idx < num_samples and silent[idx]:
                idx += 1
            end = idx  # exclusive
            length = end - start
            if length > max_silence_samples:
                # Keep the front/back edges, drop the middle to fit the cap.
                front_keep = max_silence_samples // 2
                back_keep = max_silence_samples - front_keep
                drop_start = start + front_keep
                drop_end = end - back_keep
                if drop_end > drop_start:
                    keep[drop_start:drop_end] = False

        return audio[:, keep]

    def preprocess(self, prompt_audio: str):
        audio, sr = torchaudio.load(prompt_audio)
        # Ensure mono to keep downstream assumptions consistent.
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != 24000:
            audio = torchaudio.functional.resample(audio, sr, 24000)
            sr = 24000
        audio = self._trim_long_silences(audio, sample_rate=sr)
        audio_norm = volume_norm_rms(audio)
        semantic_codes, acoustic_codes = self.dualcodec_inference.encode(audio_norm[None],
                                                                         n_quantizers=12)
        return semantic_codes, acoustic_codes, audio

    def token_to_wav(self,
                     generated_semantic_codes: torch.Tensor,
                     prompt_semantic_codes: torch.Tensor,
                     prompt_acoustic_codes: torch.Tensor,
                     ref_audio: torch.Tensor):
        '''

        :param generated_semantic_codes: (batch, seq_length)
        :param prompt_semantic_codes: (batch, 1, seq_length)
        :param prompt_acoustic_codes: (batch, 11, seq_length)
        :param ref_audio: (batch, seq_length)
        :return:
        '''

        semantic_codes = torch.cat([prompt_semantic_codes[:, 0], generated_semantic_codes], dim=-1)
        prompt = prompt_acoustic_codes.transpose(1, 2)  # batch, seq_length, num_quantizers
        restore_acoustic_codes = self.semantic_to_acoustic(
            semantic_code=semantic_codes,
            acoustic_code=prompt)
        restore_semantic_codes = generated_semantic_codes[:, None, :]
        restore_acoustic_codes = restore_acoustic_codes.transpose(1, 2)
        restore_audio = self.dualcodec_inference.decode(semantic_codes=restore_semantic_codes,
                                                        acoustic_codes=restore_acoustic_codes).cpu()[0]

        restore_audio, scale = match_loudness_lufs(restore_audio, ref_audio, 24000)
        return restore_audio
