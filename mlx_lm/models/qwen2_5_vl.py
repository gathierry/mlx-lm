import base64
import copy
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests
from PIL import Image

from mlx_lm.processor_utils import ProcessorWrapper

from .base import BaseModelArgs
from .qwen2 import Qwen2Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    image_token_id: int
    video_token_id: int
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    vision_config: Optional[Dict[str, Union[bool, int, str, List[int]]]] = None


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)

        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen.item(), dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.astype(mx.float32), k.astype(mx.float32)
    cos = mx.expand_dims(cos, axis=-2).astype(mx.float32)
    sin = mx.expand_dims(sin, axis=-2).astype(mx.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.astype(orig_q_dtype)
    k_embed = k_embed.astype(orig_k_dtype)
    return q_embed, k_embed


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self,
        x: mx.array,
        cu_seqlens: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
    ) -> mx.array:
        seq_length = x.shape[0]
        q, k, v = mx.split(
            self.qkv(x)
            .reshape(seq_length, 3, self.num_heads, -1)
            .transpose(1, 0, 2, 3),
            3,
        )
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = mx.full(
            [1, seq_length, seq_length], mx.finfo(q.dtype).min, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            start = int(cu_seqlens[i - 1])
            end = int(cu_seqlens[i])
            attention_mask[..., start:end, start:end] = 0

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attention_mask
        )
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(seq_length, -1)
        return self.proj(output)


# Activation function mapping
_ACT2FN = {
    "silu": nn.silu,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "gelu_new": nn.gelu_approx,
    "gelu_fast": nn.gelu_approx,
}


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, args, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.vision_config["hidden_size"],
            args.vision_config["intermediate_size"],
            bias=bias,
        )
        self.up_proj = nn.Linear(
            args.vision_config["hidden_size"],
            args.vision_config["intermediate_size"],
            bias=bias,
        )
        self.down_proj = nn.Linear(
            args.vision_config["intermediate_size"],
            args.vision_config["hidden_size"],
            bias=bias,
        )
        self.act_fn = _ACT2FN[args.vision_config["hidden_act"]]

    def __call__(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(args.vision_config["hidden_size"], eps=1e-6)
        self.norm2 = nn.RMSNorm(args.vision_config["hidden_size"], eps=1e-6)

        self.attn = Qwen2_5_VLVisionAttention(
            dim=args.vision_config["hidden_size"],
            num_heads=args.vision_config["num_heads"],
        )
        self.mlp = Qwen2_5_VLMLP(args, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.RMSNorm(context_dim, eps=1e-6)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class Qwen2_5_VisionTransformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.spatial_merge_size = args.vision_config["spatial_merge_size"]
        self.patch_size = args.vision_config["patch_size"]
        self.fullatt_block_indexes = args.vision_config["fullatt_block_indexes"]
        self.window_size = args.vision_config["window_size"]
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=args.vision_config["temporal_patch_size"],
            in_channels=args.vision_config["in_chans"],
            embed_dim=args.vision_config["hidden_size"],
        )

        head_dim = args.vision_config["hidden_size"] // args.vision_config["num_heads"]
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [
            Qwen2_5_VLVisionBlock(args) for _ in range(args.vision_config["depth"])
        ]
        self.merger = Qwen2_5_VLPatchMerger(
            dim=args.vision_config["out_hidden_size"],
            context_dim=args.vision_config["hidden_size"],
            spatial_merge_size=args.vision_config["spatial_merge_size"],
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            hpos_ids = mx.expand_dims(mx.arange(h), 1)
            hpos_ids = mx.repeat(hpos_ids, w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = mx.expand_dims(mx.arange(w), 0)
            wpos_ids = mx.repeat(wpos_ids, h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()

            stacked_pos_ids = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids.append(mx.tile(stacked_pos_ids, (t, 1)))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = mx.max(grid_thw[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids]

        return rotary_pos_emb.reshape(pos_ids.shape[0], -1)

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw.tolist():
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = mx.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

            index_padded = mx.pad(
                index,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=-100,
            )

            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = mx.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )

            seqlens = mx.sum(index_padded != -100, axis=(2, 3)).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index = np.where(index_padded != -100)[0].tolist()
            index_new = index_padded[index]

            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                mx.cumsum(seqlens, axis=0) * self.spatial_merge_unit
                + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += int(grid_t * llm_grid_h * llm_grid_w)
        window_index = mx.concatenate(window_index, axis=0)
        cu_window_seqlens = mx.array(cu_window_seqlens)

        return window_index, cu_window_seqlens

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
    ) -> mx.array:
        """
        Args:
            hidden_states (`mx.array` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`mx.array` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `mx.array`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        # Get indices of first occurrence of each unique value. In torch:
        # cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seen = set()
        idx = []
        for i, x in enumerate(cu_window_seqlens):
            if x not in seen:
                seen.add(x)
                idx.append(i)
        idx = mx.array(idx, dtype=mx.int32)
        cu_window_seqlens = cu_window_seqlens[idx]

        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = mx.concatenate((rotary_pos_emb, rotary_pos_emb), axis=-1)
        position_embeddings = (mx.cos(emb), mx.sin(emb))

        # torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        batch_size = grid_thw.shape[0]
        cu_seqlens = []
        for i in range(batch_size):
            seq_len = grid_thw[i, 1] * grid_thw[i, 2]
            cu_seqlens.append(mx.repeat(seq_len, grid_thw[i, 0]))
        cu_seqlens = mx.concatenate(cu_seqlens)

        cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = mx.argsort(window_index, axis=0)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # change type from 'mrope' to 'default' because `mrope` does default RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        if args.rope_scaling is not None and "type" in args.rope_scaling:
            if args.rope_scaling["type"] == "mrope":
                args.rope_scaling["type"] = "default"
            args.rope_scaling["rope_type"] = args.rope_scaling["type"]

        self.args = args
        self.model = Qwen2Model(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(
            inputs, mask=mask, cache=cache, input_embeddings=input_embeddings
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vision_tower = Qwen2_5_VisionTransformer(args)
        self.language_model = LanguageModel(args)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
    ):

        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states = self.vision_tower(pixel_values, image_grid_thw)

        if hidden_states.ndim == 2:
            hidden_states = hidden_states[None, :, :]

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            hidden_states, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_id = self.args.image_token_id
        video_token_id = self.args.video_token_id
        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        image_indices = np.where(image_positions)[1].tolist()
        inputs_embeds[:, image_indices, :] = image_features
        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        grid_thw = image_grid_thw

        input_embeddings = self.get_input_embeddings(input_ids, pixel_values, grid_thw)

        out = self.language_model(
            None, mask=mask, cache=cache, input_embeddings=input_embeddings
        )
        return out

    @property
    def layers(self):
        return self.language_model.model.layers


# from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(
            pil_image, mask=pil_image.split()[3]
        )  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(
    ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR
) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]["url"]

    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    image = to_rgb(image_obj)

    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[list[Image.Image]] | None]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            raise NotImplementedError
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs


class Processor(ProcessorWrapper):
    def load_multimodalities(self, conversation, **kwargs):
        image_inputs, video_inputs = process_vision_info(conversation)
        results = dict()
        if image_inputs is not None:
            results["images"] = image_inputs
        if video_inputs is not None:
            results["videos"] = video_inputs
        return results
