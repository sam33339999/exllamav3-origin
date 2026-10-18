"""
MiniCPM-V 4.7.

The language model is Qwen3.5 MoE (``model.language_model.*``). The vision tower is a
SigLIP encoder with a 2x2 window merger after ``insert_layer_id``, then a second 2x2
merger that projects into the LM hidden size. ``4x`` skips only the window merger.
Canvas positions are not Qwen's single-grid MRoPE; see ``mm_processing.minicpmv4_7``.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from typing_extensions import override

from ..model.config import no_default
from ..model.model import Model
from ..modules import Attention, Conv, LayerNorm, Linear, MLP, Module, TransformerBlock
from ..tokenizer.mm_embedding import MMEmbedding
from .mm_processing.minicpmv4_7 import (
    MiniCPMCanvasConfig,
    canvas_position_ids,
    merge_spatial,
    nearest_position_ids,
    window_index,
)
from .qwen3_5 import Qwen3_5VLMoeBaseConfig, Qwen3_5VLMoeModel
from .qwen3_5_mtp import Qwen3_5MoeMTPModel


class MiniCPMV4_7Config(Qwen3_5VLMoeBaseConfig):
    arch_string = "MiniCPMV4_7ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            "text_config",
            MiniCPMV4_7TextModel,
            None,
            Qwen3_5MoeMTPModel,
            **kwargs,
        )

        vcfg = self.read_cfg(dict, "vision_config", {})
        hidden_act = vcfg.get("hidden_act", "gelu_pytorch_tanh")
        if hidden_act != "gelu_pytorch_tanh":
            raise ValueError(
                f"MiniCPM-V vision hidden_act must be gelu_pytorch_tanh, got {hidden_act!r}"
            )

        image_size = vcfg.get("image_size", 224)
        patch_size = vcfg.get("patch_size", 16)
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        image_size = int(image_size)
        patch_size = int(patch_size)

        hidden_size = int(vcfg.get("hidden_size", 768))
        intermediate_size = int(vcfg.get("intermediate_size", 3072))
        num_heads = int(vcfg.get("num_attention_heads", 12))
        window = tuple(int(x) for x in vcfg.get("window_kernel_size", (2, 2)))
        merge = tuple(int(x) for x in self.read_cfg(list, "merge_kernel_size", [2, 2]))
        # Official json keeps insert_layer_id on the outer config. transformers copies it
        # onto the vision config in __post_init__, overwriting whatever vision_config had.
        insert_layer_id = self.read_cfg(int, "insert_layer_id", 6)

        # deepstack_visual_indexes stays unset so the reused Qwen3.5 text stack does not
        # insert Qwen3-VL deepstack embeddings.
        self.vision = SimpleNamespace(
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
            num_hidden_layers = int(vcfg.get("num_hidden_layers", 12)),
            num_attention_heads = num_heads,
            num_channels = int(vcfg.get("num_channels", 3)),
            image_size = image_size,
            patch_size = patch_size,
            num_patches_per_side = image_size // patch_size,
            layer_norm_eps = float(vcfg.get("layer_norm_eps", 1e-6)),
            insert_layer_id = int(insert_layer_id),
            window_kernel_size = window,
            merge_kernel_size = merge,
            merger_times = int(self.read_cfg(int, "merger_times", 1)),
            downsample_mode = self.read_cfg(str, "downsample_mode", "16x"),
            head_dim = hidden_size // num_heads,
            window_hidden_size = hidden_size * window[0] * window[1],
            window_intermediate_size = intermediate_size * window[0] * window[1],
            deepstack_visual_indexes = None,
        )
        self.model_classes["vision"] = MiniCPMV4_7VisionModel

        self.image_token_id = self.read_cfg(int, "image_token_id", no_default)
        self.video_token_id = self.read_cfg(int, "video_token_id", no_default)
        self.image_start_id = self.read_cfg(int, "image_start_id", no_default)
        self.image_end_id = self.read_cfg(int, "image_end_id", no_default)
        self.slice_start_id = self.read_cfg(int, "slice_start_id", no_default)
        self.slice_end_id = self.read_cfg(int, "slice_end_id", no_default)
        self.newline_id = self.read_cfg(int, "newline_id", no_default)

        # mtp_num_hidden_layers can be 1 while the checkpoint has no mtp.* tensors
        # (the abliterated release). Drop the component instead of failing at load.
        if "mtp" in self.model_classes and not self.stc.list_tensors(prefix = "mtp."):
            del self.model_classes["mtp"]

    def canvas_markers(self) -> MiniCPMCanvasConfig:
        return MiniCPMCanvasConfig(
            image_start_id = self.image_start_id,
            image_end_id = self.image_end_id,
            slice_start_id = self.slice_start_id,
            slice_end_id = self.slice_end_id,
            newline_id = self.newline_id,
            downsample_mode = self.vision.downsample_mode,
        )


class MiniCPMV4_7TextModel(Qwen3_5VLMoeModel):
    config_class = MiniCPMV4_7Config


def _vision_attention(
    config: MiniCPMV4_7Config,
    key: str,
    layer_idx: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
) -> Attention:
    """Non-causal attention. Linears use pad_to=1 so widths such as 17216 stay exact."""
    qkv_out = num_heads * head_dim

    def proj(name: str, in_features: int, out_features: int, qmap: str, qgroup: str) -> Linear:
        return Linear(
            config = config,
            key = f"{key}.{name}",
            in_features = in_features,
            out_features = out_features,
            qmap = qmap,
            pad_to = 1,
            qgroup = qgroup,
            trim_padded_out = True,
        )

    attn = Attention(
        config = config,
        key = key,
        layer_idx = layer_idx,
        hidden_size = hidden_size,
        head_dim = head_dim,
        num_q_heads = num_heads,
        num_kv_heads = num_heads,
        rope_settings = None,
        q_proj = proj("q_proj", hidden_size, qkv_out, "block.attn.input", key + ".qkv"),
        k_proj = proj("k_proj", hidden_size, qkv_out, "block.attn.input", key + ".qkv"),
        v_proj = proj("v_proj", hidden_size, qkv_out, "block.attn.input", key + ".qkv"),
        o_proj = proj("out_proj", qkv_out, hidden_size, "block.attn.o", key + ".o"),
        use_cu_seqlens = True,
    )
    # Packed vision attention has no KV cache. The base class opts into one by default.
    attn.caps["kv_cache"] = False
    return attn


def _ungated_mlp(
    config: MiniCPMV4_7Config,
    key: str,
    hidden_size: int,
    intermediate_size: int,
    out_size: int,
    activation_fn: str,
) -> MLP:
    return MLP(
        config = config,
        key = key,
        hidden_size = hidden_size,
        intermediate_size = intermediate_size,
        out_size = out_size,
        key_up = "linear_1",
        key_down = "linear_2",
        activation_fn = activation_fn,
        qmap = "block.mlp",
        pad_to = 1,
        out_dtype = torch.half,
    )


def _cu_seqlens(target_sizes: torch.Tensor) -> tuple[torch.Tensor, int]:
    lengths = (target_sizes[:, 0] * target_sizes[:, 1]).to(torch.int32)
    cu = F.pad(torch.cumsum(lengths, dim = 0, dtype = torch.int32), (1, 0))
    if lengths.numel() == 0:
        return cu, 0
    return cu, int((cu[1:] - cu[:-1]).max().item())


class MiniCPMGatheredPos(Module):
    """Add NaViT nearest-neighbor position embeddings and publish packed cu_seqlens."""

    def __init__(
        self,
        config: MiniCPMV4_7Config,
        key: str,
        num_patches_per_side: int,
    ):
        super().__init__(config, key, None)
        self.num_patches_per_side = num_patches_per_side
        self.weight = None
        self._numel = None

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(f"{self.key}.weight", self.device, float2half = True)
        self._numel = weight.numel()
        self.weight = weight

    @override
    def unload(self):
        self.device = None
        self.weight = None

    @override
    def get_tensors(self):
        return {f"{self.key}.weight": self.weight.contiguous()}

    @override
    def weights_numel(self):
        return self._numel

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        target_sizes = params["target_sizes"].to(device = x.device)
        params["target_sizes"] = target_sizes
        pos_ids = nearest_position_ids(target_sizes, self.num_patches_per_side)
        if pos_ids.numel() != x.shape[1]:
            raise ValueError(
                f"Position ids ({pos_ids.numel()}) do not match patch tokens ({x.shape[1]})"
            )
        pos = self.weight.index_select(0, pos_ids.to(self.weight.device))
        x = x + pos.to(dtype = x.dtype, device = x.device).unsqueeze(0)
        cu, max_seqlen = _cu_seqlens(target_sizes)
        params["cu_seqlens"] = cu
        params["max_seqlen"] = max_seqlen
        params["causal"] = False
        return x.to(out_dtype) if out_dtype is not None else x


class MiniCPMViTWindowMerger(Module):
    """2x2 window attention + tanh-GELU MLP after ``insert_layer_id``. Identity in ``4x``."""

    def __init__(self, config: MiniCPMV4_7Config, key: str):
        super().__init__(config, key, None)
        v = config.vision
        window_h, window_w = v.window_kernel_size
        if window_h != window_w:
            raise ValueError(f"window_kernel_size must be square, got {(window_h, window_w)}")
        self.window_kernel_size = (window_h, window_w)
        eps = v.layer_norm_eps

        self.layer_norm1 = LayerNorm(config, f"{key}.layer_norm1", eps)
        self.attn = _vision_attention(
            config, f"{key}.self_attn", v.insert_layer_id,
            v.hidden_size, v.num_attention_heads, v.head_dim,
        )
        self.pre_norm = LayerNorm(config, f"{key}.pre_norm", eps)
        # Concatenated 2x2 window is tanh-GELU. The outer merger uses erf GELU instead.
        self.mlp = MLP(
            config = config,
            key = key,
            hidden_size = v.window_hidden_size,
            intermediate_size = v.window_intermediate_size,
            out_size = v.hidden_size,
            key_up = "linear_1",
            key_down = "linear_2",
            activation_fn = "gelu",
            qmap = "block.mlp",
            pad_to = 1,
            out_dtype = torch.half,
        )
        self.register_submodule(self.layer_norm1)
        self.register_submodule(self.attn)
        self.register_submodule(self.pre_norm)
        self.register_submodule(self.mlp)

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        if not params.get("use_vit_merger", True):
            return x

        target_sizes = params["target_sizes"]
        window_h, window_w = self.window_kernel_size
        order, window_cu = window_index(target_sizes, window_size = window_h)
        order = order.to(x.device)

        residual = x
        y = self.layer_norm1.forward(x, params, out_dtype = torch.half)
        y = y[:, order, :]
        params["cu_seqlens"] = window_cu
        window_lengths = window_cu[1:] - window_cu[:-1]
        params["max_seqlen"] = int(window_lengths.max().item()) if window_lengths.numel() else 0
        params["causal"] = False
        y = self.attn.forward(y, params)
        y = residual[:, order, :].to(y.dtype) + y

        window_area = window_h * window_w
        if y.shape[1] % window_area:
            raise ValueError(
                f"ViT merger expected a multiple of {window_area} patches, got {y.shape[1]}. "
                f"Grids must be divisible by window_kernel_size {self.window_kernel_size}."
            )
        patch = y.reshape(-1, window_area, y.shape[-1])
        flat = patch.flatten(1).unsqueeze(0)
        patch_residual = patch.mean(dim = 1)
        flat = self.pre_norm.forward(flat, params, out_dtype = torch.half)
        flat = self.mlp.forward(flat, params)
        flat = flat + patch_residual.to(dtype = flat.dtype, device = flat.device).unsqueeze(0)

        target_sizes = target_sizes // 2
        params["target_sizes"] = target_sizes
        cu, max_seqlen = _cu_seqlens(target_sizes)
        params["cu_seqlens"] = cu
        params["max_seqlen"] = max_seqlen
        return flat if out_dtype is None else flat.to(out_dtype)


class MiniCPMSpatialMerger(Module):
    """Outer 2x2 pack and erf-GELU projection into LM hidden size. Runs in both 4x and 16x."""

    def __init__(self, config: MiniCPMV4_7Config, key: str):
        super().__init__(config, key, None)
        v = config.vision
        self.merge_kernel_size = tuple(v.merge_kernel_size)
        merge_h, merge_w = self.merge_kernel_size
        merged_hidden = v.hidden_size * merge_h * merge_w
        eps = v.layer_norm_eps
        self.rounds = []
        for i in range(v.merger_times):
            out_size = config.hidden_size if i == v.merger_times - 1 else v.hidden_size
            prefix = f"{key}.mlp.{i}"
            norm = LayerNorm(config, f"{prefix}.pre_norm", eps)
            mlp = _ungated_mlp(
                config, prefix, merged_hidden, merged_hidden, out_size, "gelu_exact",
            )
            self.rounds.append((norm, mlp))
            self.register_submodule(norm)
            self.register_submodule(mlp)

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        target_sizes = params["target_sizes"]
        merge_h, merge_w = self.merge_kernel_size
        hidden = x[0]
        start = 0
        outputs = []
        for height, width in target_sizes.tolist():
            height, width = int(height), int(width)
            num_patches = height * width
            crop = hidden[start : start + num_patches]
            for round_i, (norm, mlp) in enumerate(self.rounds):
                crop = merge_spatial(crop, height, width, merge_h, merge_w)
                crop = norm.forward(crop.unsqueeze(0), params, out_dtype = torch.half)
                crop = mlp.forward(crop, params)[0]
                if round_i + 1 < len(self.rounds):
                    height //= merge_h
                    width //= merge_w
            outputs.append(crop)
            start += num_patches
        if start != hidden.shape[0]:
            raise ValueError(
                f"Merger consumed {start} vision tokens, hidden state has {hidden.shape[0]}"
            )
        y = torch.cat(outputs, dim = 0).unsqueeze(0)
        return y if out_dtype is None else y.to(out_dtype)


class MiniCPMV4_7VisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: MiniCPMV4_7Config) -> dict:
        tensors = {}
        tensors.update(config.stc.list_tensors(prefix = "model.vision_tower"))
        tensors.update(config.stc.list_tensors(prefix = "model.merger"))
        return tensors

    def __init__(self, config: MiniCPMV4_7Config, **kwargs):
        super().__init__(config, **kwargs)
        self.caps.update({
            "image_input": True,
            "supports_tp": False,
        })
        v = config.vision
        if v.head_dim * v.num_attention_heads != v.hidden_size:
            raise ValueError(
                f"vision hidden_size {v.hidden_size} is not divisible by "
                f"{v.num_attention_heads} heads"
            )
        eps = v.layer_norm_eps
        tower = "model.vision_tower"

        self.modules += [
            Conv(
                config = config,
                key = f"{tower}.embeddings.patch_embedding",
                in_channels = v.num_channels,
                out_channels = v.hidden_size,
                kernel_size = (v.patch_size, v.patch_size),
            ),
            MiniCPMGatheredPos(
                config = config,
                key = f"{tower}.embeddings.position_embedding",
                num_patches_per_side = v.num_patches_per_side,
            ),
        ]

        for idx in range(v.insert_layer_id + 1):
            self.modules.append(self._encoder_block(config, idx))

        self.modules.append(MiniCPMViTWindowMerger(config, f"{tower}.vit_merger"))

        for idx in range(v.insert_layer_id + 1, v.num_hidden_layers):
            self.modules.append(self._encoder_block(config, idx))

        self.modules += [
            LayerNorm(
                config = config,
                key = f"{tower}.post_layernorm",
                layernorm_eps = eps,
            ),
            MiniCPMSpatialMerger(config, "model.merger"),
        ]

    @override
    def default_load_shape_dtype(self, chunk_size):
        # Autosplit measures VRAM with one forward. The text default is a token
        # row, which Conv2d rejects. Four patches on a side is the smallest grid
        # the 2x2 window merger and the outer merger can both consume.
        del chunk_size
        v = self.config.vision
        side = v.patch_size * 4
        return (1, v.num_channels, side, side), torch.half

    @override
    def default_load_params(self, chunk_size):
        del chunk_size
        return {
            "target_sizes": torch.tensor([[4, 4]], dtype = torch.long),
            "use_vit_merger": True,
            "causal": False,
            "downsample_mode": self.config.vision.downsample_mode,
        }

    def _encoder_block(self, config: MiniCPMV4_7Config, idx: int) -> TransformerBlock:
        v = config.vision
        key = f"model.vision_tower.encoder.layers.{idx}"
        eps = v.layer_norm_eps
        return TransformerBlock(
            config = config,
            key = key,
            layer_idx = idx,
            attn_norm = LayerNorm(config, f"{key}.layer_norm1", eps),
            attn = _vision_attention(
                config, f"{key}.self_attn", idx,
                v.hidden_size, v.num_attention_heads, v.head_dim,
            ),
            mlp_norm = LayerNorm(config, f"{key}.layer_norm2", eps),
            mlp = MLP(
                config = config,
                key = f"{key}.mlp",
                hidden_size = v.hidden_size,
                intermediate_size = v.intermediate_size,
                key_up = "fc1",
                key_down = "fc2",
                activation_fn = "gelu",
                qmap = "block.mlp",
                pad_to = 1,
                out_dtype = torch.half,
            ),
        )

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        mode = params.get("downsample_mode") or self.config.vision.downsample_mode
        if mode not in ("4x", "16x"):
            raise ValueError(f"downsample_mode must be '4x' or '16x', got {mode!r}")
        params["downsample_mode"] = mode
        params["use_vit_merger"] = mode != "4x"
        params["causal"] = False
        return input_ids

    def prepare_multimodal(
        self,
        tokenizer,
        messages,
        images,
        downsample_mode: str | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> dict:
        """Run the checkpoint processor and the vision tower for one chat turn.

        Returns ``input_ids`` of shape ``(1, seq)`` with image placeholders already
        replaced by MMEmbedding ids, plus a one-element embeddings list that carries
        the canvas MRoPE table. ``tokenizer`` is accepted for call-site symmetry; the
        processor loaded from the checkpoint produces the ids.
        """
        del tokenizer  # processor tokenizer is the one that matches this checkpoint
        from transformers import AutoProcessor

        if self.modules[0].device is None:
            raise RuntimeError("Load the vision model before prepare_multimodal")

        processor = AutoProcessor.from_pretrained(self.config.directory)
        template_kwargs = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        text = processor.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = add_generation_prompt,
            **template_kwargs,
        )
        if isinstance(text, str):
            text = [text]
        if not isinstance(images, list):
            images = [images]

        mode = downsample_mode or self.config.vision.downsample_mode
        batch = processor(
            text = text,
            images = images,
            return_tensors = "pt",
            return_mm_token_type_ids = True,
            downsample_mode = mode,
        )
        if batch["input_ids"].shape[0] != 1:
            raise ValueError("prepare_multimodal supports one sequence at a time")

        device = self.modules[0].device
        params = {
            "target_sizes": batch["target_sizes"],
            "downsample_mode": mode,
        }
        features = self.forward(batch["pixel_values"].to(device), params = params)
        features = features.squeeze(0).to(dtype = torch.half).cpu().contiguous()

        input_ids = batch["input_ids"].to(dtype = torch.long, device = "cpu")
        mm_token_type_ids = batch["mm_token_type_ids"].to(dtype = torch.long, device = "cpu")
        token_string = input_ids.clone()
        placeholder = token_string == self.config.image_token_id
        if self.config.video_token_id is not None:
            placeholder = placeholder | (token_string == self.config.video_token_id)
        n_placeholder = int(placeholder.sum().item())
        if n_placeholder != features.shape[0]:
            raise ValueError(
                f"Image placeholder count {n_placeholder} does not match vision tokens {features.shape[0]}"
            )
        token_string[placeholder] = -1

        position_ids, rope_deltas = canvas_position_ids(
            input_ids,
            mm_token_type_ids,
            batch["target_sizes"],
            batch.get("target_sizes_videos"),
            mode,
            self.config.canvas_markers(),
            attention_mask = batch.get("attention_mask"),
        )
        if position_ids.shape[-1] != token_string.shape[-1]:
            raise ValueError("Canvas length does not match the prompt")

        embedding = MMEmbedding(
            embeddings = features,
            token_string = token_string,
            prompt_mrope_pos_ids = position_ids[:, 0, :].contiguous(),
            rope_delta = int(rope_deltas.view(-1)[0].item()),
        )
        embedding.metadata["downsample_mode"] = mode
        embedding.metadata["model_architecture"] = self.config.architecture
        return {
            "input_ids": embedding.token_string,
            "embeddings": [embedding],
        }
