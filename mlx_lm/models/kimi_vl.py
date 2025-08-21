# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs
from .deepseek_v3 import DeepseekV3Model


@dataclass
class TextArgs(BaseModelArgs):
    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Dict = None
    attention_bias: bool = False


@dataclass
class VisionArgs(BaseModelArgs):
    hidden_size: int
    merge_kernel_size: List[int]
    patch_size: int
    init_pos_emb_height: int
    init_pos_emb_width: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: NotImplementedError


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextArgs, dict]
    vision_config: Union[VisionArgs, dict]
    model_type: str
    media_placeholder_token_id: int

    def __post_init__(self):
        self.text_config = TextArgs.from_dict(self.text_config)
        self.vision_config = VisionArgs.from_dict(self.vision_config)


class Learnable2DInterpPosEmb(nn.Module):
    def __init__(
        self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic"
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = mx.random.normal(shape=(height, width, dim))

    def __call__(self, x: mx.array, grid_hws: mx.array) -> mx.array:
        pos_embs = []
        for shape in grid_hws.tolist():
            if shape == self.weight.shape[:-1]:
                pos_embs.append(self.weight.flatten(end_dim=1))
            else:
                scale_factor = (shape[0] / self.height, shape[1] / self.width)
                pos_embs.append(
                    nn.Upsample(scale_factor=scale_factor, mode="cubic")(
                        self.weight[None]
                    )
                    .squeeze(0)
                    .flatten(end_axis=1)
                )
        out = x + mx.concatenate(pos_embs)
        return out


class MoonVisionPatchEmbed(nn.Module):

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: Union[int, Tuple[int, int]] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
    ):
        super().__init__()
        assert isinstance(
            patch_size, (int, Sequence)
        ), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert (
            len(patch_size) == 2
        ), f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_emb = Learnable2DInterpPosEmb(
            height=pos_emb_height, width=pos_emb_width, dim=out_dim
        )

    def __call__(self, x: mx.array, grid_hws: mx.array) -> mx.array:
        """
        Args:
            x (image_token_nums, channel_num, patch_size, patch_size): input tensor
            grid_hws (N, 2): grid height and width
        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).reshape(x.shape[0], -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_hws)
        return x


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding with multi-resolution support.
    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each __call__ pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the __call__ pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.
    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py
    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

        self.freqs_cis = None

    def _precompute_freqs_cis(self) -> mx.array:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """

        def polar(abs, angle):
            return abs * mx.cos(angle) + abs * mx.sin(angle) * 1j

        N = self.max_height * self.max_width
        flat_pos = mx.arange(0, N).astype(mx.float32)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = mx.arange(0, self.dim, 4)[: (self.dim // 4)].astype(
            mx.float32
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = mx.outer(x_pos, freqs).astype(mx.float32)  # N, C/4
        y_freqs = mx.outer(y_pos, freqs).astype(mx.float32)  # N, C/4
        x_cis = polar(mx.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = polar(mx.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = mx.concatenate([x_cis[..., None], y_cis[..., None]], axis=-1)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_hws: mx.array) -> mx.array:
        """
        Args:
            grid_hws (mx.array): grid height and width
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_freqs_cis()

        shapes = grid_hws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = mx.concatenate(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes],
            axis=0,
        )
        return freqs_cis


def trunc_normal(shape, std=1.0, mean=0.0):
    # sample normal and clip at 2 stddev (like PyTorch's default)
    t = mx.random.normal(shape) * std + mean
    return mx.clip(t, mean - 2 * std, mean + 2 * std)


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            m.weight = trunc_normal(
                m.weight.shape, std=math.sqrt(2 / m.weight.shape[1])
            )
            if m.bias is not None:
                m.bias = mx.zeros(m.bias.shape)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == mx.complex64, freqs_cis.dtype


def view_as_complex(x: mx.array) -> mx.array:
    # assumes x.shape[-1] == 2, last dim = (real, imag)
    real = x[..., 0]
    imag = x[..., 1]
    return real + 1j * imag


def view_as_real(z: mx.array) -> mx.array:
    return mx.stack([mx.real(z), mx.imag(z)], axis=-1)


def apply_rope(
    xq: mx.array, xk: mx.array, freqs_cis: mx.array
) -> tuple[mx.array, mx.array]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis[..., None, :]  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = view_as_complex(xq.astype(mx.float32).reshape(*xq.shape[:-1], -1, 2))
    xk_ = view_as_complex(xk.astype(mx.float32).reshape(*xq.shape[:-1], -1, 2))
    xq_out = view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


class MoonVitEncoderLayer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        activation=nn.GELU,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.scale = self.hidden_size_per_attention_head**-0.5

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: mx.array,
        cu_seqlens: mx.array,
        rope_freqs_cis: Optional[mx.array] = None,
    ):
        """
        Args:
            x (mx.array): (batch_size * seqlen, hidden_dim)
            cu_seqlens (mx.array):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.shape[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size * seqlen, 3, nheads, headdim)
        xqkv = xqkv.reshape(*qkv_shape)
        # xq, xk, xv: (batch_size * seqlen, nheads, headdim)
        xq, xk, xv = [
            _x.squeeze(-3) for _x in mx.split(xqkv, indices_or_sections=3, axis=-3)
        ]

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        # attention
        seq_length = xq.shape[0]
        attention_mask = mx.zeros([1, seq_length, seq_length], dtype=mx.bool_)
        for i in range(1, len(cu_seqlens)):
            st = cu_seqlens[i - 1].item()
            ed = cu_seqlens[i].item()
            attention_mask[
                ...,
                st:ed,
                st:ed,
            ] = True
        xq = xq.swapaxes(0, 1)[None]
        xk = xk.swapaxes(0, 1)[None]
        xv = xv.swapaxes(0, 1)[None]
        attn_out = mx.fast.scaled_dot_product_attention(
            xq, xk, xv, scale=self.scale, mask=attention_mask
        ).squeeze(0)
        attn_out = attn_out.swapaxes(0, 1)
        attn_out = attn_out.reshape(seq_length, -1)

        attn_out = self.wo(attn_out)
        return attn_out

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        rope_freqs_cis: Union[mx.array, None] = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: non-packed (B, N, D) or packed (L, D). if non-packed, seqlens should be None, if packed, seqlens should be set
        Returns:
            output: same shape of input, non-packed (B, N, D) for non-packed input, (L, D) for packed input
        """
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        attn_out = self.attention_qkvpacked(
            hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.norm1(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class MoonVitEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
    ) -> None:
        super().__init__()
        self.rope_2d = Rope2DPosEmb(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512
        )
        self.blocks = [MoonVitEncoderLayer(**block_cfg) for _ in range(num_layers)]
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def __call__(self, hidden_states: mx.array, grid_hws: mx.array) -> mx.array:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_hws=grid_hws)

        lengths = mx.concatenate(
            (
                mx.zeros(1, dtype=grid_hws.dtype),
                grid_hws[:, 0] * grid_hws[:, 1],
            )
        )
        cu_seqlens = lengths.cumsum(axis=0).astype(mx.int32)

        for _, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


def patch_merger(
    x: mx.array,
    grid_hws: mx.array,
    merge_kernel_size: list[int, int] = (2, 2),
) -> List[mx.array]:
    d_model = x.shape[-1]

    outputs = []
    pre_sum = 0
    for x_shape in grid_hws.tolist():
        height, width = x_shape[0], x_shape[1]
        # Get the current sequence
        seq = x[pre_sum : pre_sum + height * width]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = height // kernel_height, width // kernel_width
        reshaped_seq = seq.reshape(
            new_height, kernel_height, new_width, kernel_width, d_model
        )
        reshaped_seq = reshaped_seq.transpose(0, 2, 1, 3, 4)
        padded_seq = reshaped_seq.reshape(
            new_height * new_width, kernel_height * kernel_width, -1
        )
        outputs.append(padded_seq)
        pre_sum += height * width

    return outputs


class MoonViT(nn.Module):
    def __init__(self, config: VisionArgs):
        super().__init__()
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.patch_embed = MoonVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
        )
        self.encoder = MoonVitEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": nn.GELU(approx="precise"),
                "attn_bias": True,
            },
        )

    def __call__(self, pixel_values: mx.array, grid_hws: mx.array) -> mx.array:
        """
        Args:
            pixel_values (mx.array): The input pixel values.
            grid_hws (mx.array): The grid height and width.
        Returns:
            mx.array: The output tokens.
        """
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states = self.encoder(hidden_states, grid_hws)
        hidden_states = patch_merger(
            hidden_states, grid_hws, merge_kernel_size=self.merge_kernel_size
        )
        return hidden_states


class KimiVLMultiModalProjector(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = (
            args.vision_config.hidden_size
            * args.vision_config.merge_kernel_size[0]
            * args.vision_config.merge_kernel_size[1]
        )

        self.pre_norm = nn.LayerNorm(args.vision_config.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            self.hidden_size, args.text_config.hidden_size, bias=True
        )

    def __call__(self, image_features: mx.array) -> mx.array:
        image_features = mx.concatenate(image_features, axis=0)
        hidden_states = self.pre_norm(image_features).reshape(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states


class LanguageModel(nn.Module):
    def __init__(self, config: TextArgs):
        super().__init__()
        self.args = config
        self.model = DeepseekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, mask, input_embeddings=input_embeddings)
        return self.lm_head(out)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.vision_tower = MoonViT(config.vision_config)
        self.multi_modal_projector = KimiVLMultiModalProjector(config)
        self.language_model = LanguageModel(config.text_config)

    def _merge_with_image_features(
        self,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        image_features: mx.array,
    ):
        """
        Args:
            inputs_embeds (:obj:`mx.array` of shape :obj:`(batch_size, sequence_length, input_embed_dim)`):
                The input embeddings.
            input_ids (:obj:`mx.array` of shape :obj:`(batch_size, sequence_length)`):
                The input ids.
            image_features (:obj:`mx.array` of shape :obj:`(image_token_nums, image_feature_dim)`):
                The image features to merge with the input embeddings.
        """
        image_token_index: int = self.args.media_placeholder_token_id

        batch_size, sequence_length, input_embed_dim = inputs_embeds.shape
        image_feature_nums, image_feature_dim = image_features.shape

        assert image_feature_dim == input_embed_dim

        image_token_nums = (input_ids == image_token_index).sum().item()
        assert image_feature_nums == image_token_nums

        # (batch_size, sequence_length, input_embed_dim) -> (batch_size * sequence_length, input_embed_dim)
        inputs_embeds = inputs_embeds.reshape(-1, input_embed_dim)

        # (batch_size, sequence_length) -> (batch_size * sequence_length)
        input_ids = input_ids.flatten()
        image_indices = np.where(input_ids == image_token_index)[0].tolist()
        inputs_embeds[image_indices] = image_features

        inputs_embeds = inputs_embeds.reshape(
            (batch_size, sequence_length, input_embed_dim)
        )

        return inputs_embeds

    def _extract_image_features(self, pixel_values: mx.array, image_grid_hws: mx.array):
        """
        Args:
            pixel_values (:obj:`mx.array` of shape :obj:`(image_token_nums, 3, patch_size, patch_size)`):
                The pixel values of the images processed by image processor.
        Returns:
            image_features (:obj:`mx.array` of shape :obj:`(image_token_nums, image_feature_dim)`):
                The selected image features to use as input to the projector head.
        """
        # [(image_token_nums_0, image_feature_dim), (image_token_nums_1, image_feature_dim), ...]
        image_features = self.vision_tower(pixel_values, image_grid_hws)
        # (image_token_nums_0 + image_token_nums_1 + ..., image_feature_dim)
        image_features = self.multi_modal_projector(image_features)
        return image_features

    def __call__(
        self,
        inputs: mx.array,
        pixel_values: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(inputs)
        if pixel_values is not None:
            pixel_values = pixel_values.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            image_features = self._extract_image_features(
                pixel_values, kwargs["image_grid_hws"]
            )
            inputs_embeds = inputs_embeds.astype(image_features[0].dtype)
            inputs_embeds = self._merge_with_image_features(
                inputs_embeds, inputs, image_features
            )

        outputs = self.language_model(
            inputs,
            cache,
            mask,
            input_embeddings=inputs_embeds,
        )

        return outputs

    def sanitize(self, weights):
        weights = {k: v for k, v in weights.items() if "rotary_emb" not in k}

        vision_tower_keys = [k for k in weights.keys() if k.startswith("vision_tower.")]
        for k in vision_tower_keys:
            new_key = k.replace(".attn.", ".")
            new_key = new_key.replace(".blocks.", ".encoder.blocks.")
            new_key = new_key.replace(".final_layernorm.", ".encoder.final_layernorm.")
            weights[new_key] = weights.pop(k)

        # Stack experts
        for l in range(self.args.text_config.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}"
            for m in [("gate_proj"), ("down_proj"), ("up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.text_config.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        return weights

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate
