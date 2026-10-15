"""Weight-free checks for MiniCPM-V 4.7 canvas positions and the vision graph.

Canvas ids are compared with ``MiniCPMV4_7Model.get_rope_index`` from the installed
transformers. No checkpoint and no GPU are required.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from exllamav3.architecture.architectures import ARCHITECTURES
from exllamav3.architecture.minicpmv4_7 import (
    MiniCPMV4_7TextModel,
    MiniCPMV4_7VisionModel,
)
from exllamav3.architecture.mm_processing.minicpmv4_7 import (
    MiniCPMCanvasConfig,
    canvas_position_ids,
    merge_spatial,
    nearest_position_ids,
    visual_token_count,
    window_index,
    group_visual_frames,
)
from exllamav3.architecture.qwen3_5 import Qwen3_5VLMoeModel
from exllamav3.modules import Attention, Linear, MLP
from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX, MMEmbedding
from exllamav3.util.rope import RoPE, RopeSettings, RopeStyle
from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7Model
from transformers.vision_utils import get_vision_nearest_position_ids, get_vision_window_index


MARKERS = MiniCPMCanvasConfig(
    image_start_id = 1,
    image_end_id = 2,
    slice_start_id = 3,
    slice_end_id = 4,
    newline_id = 5,
    downsample_mode = "16x",
)


def _hf(mode = "16x"):
    model = MiniCPMV4_7Model.__new__(MiniCPMV4_7Model)
    model.config = SimpleNamespace(
        downsample_mode = mode,
        slice_start_id = MARKERS.slice_start_id,
        slice_end_id = MARKERS.slice_end_id,
        image_start_id = MARKERS.image_start_id,
        image_end_id = MARKERS.image_end_id,
        newline_id = MARKERS.newline_id,
    )
    return model


def _batch(ids, mm, sizes = None, videos = None, mode = "16x", mask = None):
    ids_t = torch.tensor(ids, dtype = torch.long)
    mm_t = torch.tensor(mm, dtype = torch.long)
    if ids_t.dim() == 1:
        ids_t = ids_t.unsqueeze(0)
        mm_t = mm_t.unsqueeze(0)
    sizes_t = None if sizes is None else torch.tensor(sizes, dtype = torch.long)
    videos_t = None if videos is None else torch.tensor(videos, dtype = torch.long)
    mask_t = None if mask is None else torch.tensor(mask, dtype = torch.long)
    if mask_t is not None and mask_t.dim() == 1:
        mask_t = mask_t.unsqueeze(0)
    ours, our_delta = canvas_position_ids(
        ids_t, mm_t, sizes_t, videos_t, mode, MARKERS, attention_mask = mask_t,
    )
    hf_pos, hf_delta = _hf(mode).get_rope_index(
        ids_t, mm_t, sizes_t, videos_t, mode, mask_t,
    )
    assert torch.equal(ours, hf_pos)
    assert torch.equal(our_delta, hf_delta)
    return ours, our_delta


def _row_major_windows(patches, height, width, merge_h, merge_w):
    """Pack windows with an explicit loop, independent of reshape/permute."""
    dim = patches.shape[-1]
    grid = patches.reshape(height, width, dim)
    windows = []
    for i in range(height // merge_h):
        for j in range(width // merge_w):
            flat = []
            for r in range(merge_h):
                for c in range(merge_w):
                    flat.append(grid[i * merge_h + r, j * merge_w + c])
            windows.append(torch.stack(flat).reshape(-1))
    return torch.stack(windows)


def test_canvas_module_does_not_import_the_extension():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "exllamav3", "architecture", "mm_processing", "minicpmv4_7.py",
    )
    source = open(path, encoding = "utf8").read()
    assert "exllamav3_ext" not in source
    assert "exllamav3.ext" not in source


def test_visual_token_count():
    assert visual_token_count(8, 8, "16x") == 4
    assert visual_token_count(8, 8, "4x") == 16
    assert visual_token_count(4, 4, "4x") == 4
    with pytest.raises(ValueError):
        visual_token_count(3, 5, "16x")
    with pytest.raises(ValueError):
        visual_token_count(4, 4, "8x")


def test_merge_spatial_matches_row_major_loop():
    height, width, dim = 4, 6, 3
    patches = torch.arange(height * width * dim, dtype = torch.float32).reshape(height * width, dim)
    got = merge_spatial(patches, height, width, 2, 2)
    ref = _row_major_windows(patches, height, width, 2, 2)
    assert torch.equal(got, ref)
    assert got.shape == (6, 2 * 2 * dim)
    with pytest.raises(ValueError):
        merge_spatial(patches, height, width, 3, 2)


def test_nearest_and_window_match_transformers():
    sizes = torch.tensor([[3, 5], [4, 4]], dtype = torch.long)
    assert torch.equal(
        nearest_position_ids(sizes, 70),
        get_vision_nearest_position_ids(sizes, 70),
    )

    windows = torch.tensor([[4, 6], [4, 4]], dtype = torch.long)
    order, cu = window_index(windows, window_size = 2)
    hf_order, hf_cu = get_vision_window_index(
        F.pad(windows, (1, 0), value = 1),
        spatial_merge_size = 1,
        window_size = 2,
        patch_size = 1,
    )
    assert torch.equal(order, hf_order)
    assert torch.equal(cu, hf_cu)
    assert torch.equal(order.sort().values, torch.arange(4 * 6 + 4 * 4))


def test_text_only_is_arange_with_zero_delta():
    ids = [10, 11, 12, 13]
    mm = [0, 0, 0, 0]
    pos, delta = _batch(ids, mm)
    expected = torch.arange(4).view(1, 1, 4).expand(3, 1, 4)
    assert torch.equal(pos, expected)
    assert torch.equal(delta, torch.zeros(1, 1, dtype = torch.long))


def test_thumbnail_canvas_matches_hand_fixture_and_hf():
    # 8x8 patches, 16x -> 2x2 LLM tokens. One text token before the frame, so the
    # canvas origin is 1 and the halo on <image> is 0.
    ids = [10, 1, 9, 9, 9, 9, 2, 11]
    mm = [0, 0, 1, 1, 1, 1, 0, 0]
    pos, delta = _batch(ids, mm, sizes = [[8, 8]])
    expected = torch.tensor([
        [0, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 1, 1, 2, 2, 3, 4],
        [0, 0, 1, 2, 1, 2, 3, 4],
    ], dtype = torch.long).unsqueeze(1)
    assert torch.equal(pos, expected)
    assert int(delta[0, 0]) == -3


def test_slice_newline_canvas_matches_hand_fixture_and_hf():
    # Thumbnail plus two slice rows (one column). The newline sits on the right edge.
    ids = [10, 1, 9, 2, 3, 9, 4, 5, 3, 9, 4, 11]
    mm = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    groups = group_visual_frames(ids, mm, MARKERS)
    assert len(groups) == 1
    assert len(groups[0][2]) == 1
    assert len(groups[0][2][0][4]) == 2
    pos, delta = _batch(ids, mm, sizes = [[4, 4], [4, 4], [4, 4]])
    expected = torch.tensor([
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 1, 3, 1, 1, 1, 1, 2, 2, 2, 4],
        [0, 0, 1, 2, 1, 1, 1, 2, 1, 1, 1, 4],
    ], dtype = torch.long).unsqueeze(1)
    assert torch.equal(pos, expected)
    assert int(delta[0, 0]) == 4 + 1 - len(ids)


def test_4x_and_video_group_match_hf():
    ids = [10, 1, 9, 9, 9, 9, 2, 11]
    mm = [0, 0, 1, 1, 1, 1, 0, 0]
    _batch(ids, mm, sizes = [[4, 4]], mode = "4x")

    # Each 8x8 frame keeps 4 LLM tokens in 16x. Only markers sit between the frames,
    # so they share one canvas group.
    video_ids = [10, 1, 9, 9, 9, 9, 2, 1, 9, 9, 9, 9, 2, 11]
    video_mm = [0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0]
    groups = group_visual_frames(video_ids, video_mm, MARKERS)
    assert len(groups) == 1
    assert [frame[0] for frame in groups[0][2]] == [2, 2]
    _batch(video_ids, video_mm, videos = [[8, 8], [8, 8]])


def test_attention_mask_and_batch_match_hf():
    ids = [10, 1, 9, 9, 9, 9, 2, 11, 0]
    mm = [0, 0, 1, 1, 1, 1, 0, 0, 0]
    mask = [1, 1, 1, 1, 1, 1, 1, 1, 0]
    pos, _ = _batch(ids, mm, sizes = [[8, 8]], mask = mask)
    assert torch.equal(pos[:, 0, -1], torch.zeros(3, dtype = torch.long))

    batch_ids = [
        [10, 11, 12, 13, 0, 0, 0, 0],
        [10, 1, 9, 9, 9, 9, 2, 11],
    ]
    batch_mm = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ]
    pos, delta = _batch(batch_ids, batch_mm, sizes = [[8, 8]])
    assert torch.equal(pos[:, 0, :], torch.arange(8).view(1, 8).expand(3, 8))
    assert int(delta[0, 0]) == 0
    assert int(delta[1, 0]) == -3


def test_rope_freqs_from_canvas_positions():
    rope = RoPE(
        device = "cpu",
        rope_settings = RopeSettings(
            head_dim = 256,
            partial_rotary_factor = 0.25,
            rope_theta = 1e7,
            rope_scaling = {
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
                "rope_type": "default",
            },
            rope_style = RopeStyle.NEOX,
        ),
    )
    seq = 8
    pos = torch.arange(seq)
    ids = pos.unsqueeze(0).expand(3, -1).contiguous()
    freqs = rope.freqs_from_mrope_pos_ids(ids)
    outer = torch.outer(pos.float(), rope.inv_freq.float())
    assert freqs.shape == (1, seq, rope.inv_freq.numel())
    torch.testing.assert_close(freqs[0], outer)
    batched = ids.unsqueeze(1)
    torch.testing.assert_close(rope.freqs_from_mrope_pos_ids(batched), freqs)

    shifted = ids.clone()
    shifted[1] = shifted[1] + 3
    freqs_h = rope.freqs_from_mrope_pos_ids(shifted)
    h_idx = slice(1, 33, 3)
    w_idx = slice(2, 30, 3)
    t_idx = [i for i in range(freqs.shape[-1]) if i % 3 == 0]
    assert not torch.equal(freqs_h[0, :, h_idx], freqs[0, :, h_idx])
    torch.testing.assert_close(freqs_h[0, :, w_idx], freqs[0, :, w_idx])
    torch.testing.assert_close(freqs_h[0, :, t_idx], freqs[0, :, t_idx])

    plain = RoPE("cpu", RopeSettings(head_dim = 64, rope_theta = 10000.0))
    with pytest.raises(ValueError):
        plain.freqs_from_mrope_pos_ids(ids)


def test_mm_embedding_stores_canvas_and_accepts_old_payloads():
    token_string = torch.tensor([[10, -1, -1, 11]])
    canvas = torch.arange(24, dtype = torch.int32).reshape(3, 8)[:, ::2]
    assert not canvas.is_contiguous()
    embedding = MMEmbedding(
        embeddings = torch.zeros(2, 8),
        token_string = token_string,
        prompt_mrope_pos_ids = canvas,
        rope_delta = -3,
    )
    assert embedding.token_string is token_string
    assert int(token_string[0, 0]) == 10
    assert int(token_string[0, 3]) == 11
    assert int(token_string[0, 1]) >= FIRST_MM_EMBEDDING_INDEX
    assert int(token_string[0, 2]) == int(token_string[0, 1]) + 1
    assert embedding.rope_delta == -3
    assert embedding.prompt_mrope_pos_ids.dtype == torch.long
    assert embedding.prompt_mrope_pos_ids.device.type == "cpu"
    assert embedding.prompt_mrope_pos_ids.is_contiguous()
    assert torch.equal(embedding.prompt_mrope_pos_ids, canvas.to(dtype = torch.long).contiguous())

    legacy = MMEmbedding(imp = {
        "metadata": {},
        "full_length": 1,
        "mm_length": 1,
        "first_index": 1,
        "last_index": 2,
        "text_alias": "x",
        "grid_thw": None,
        "mrope_merge_size": None,
        "embeddings": torch.zeros(1, 4),
        "deepstack_embeddings": None,
    })
    assert legacy.prompt_mrope_pos_ids is None
    assert legacy.rope_delta is None


def _vision_config():
    hidden = 1152
    intermediate = 4304
    heads = 16
    window = (2, 2)
    return SimpleNamespace(
        hidden_size = 2048,
        vision = SimpleNamespace(
            hidden_size = hidden,
            intermediate_size = intermediate,
            num_hidden_layers = 27,
            num_attention_heads = heads,
            num_channels = 3,
            image_size = 980,
            patch_size = 14,
            num_patches_per_side = 980 // 14,
            layer_norm_eps = 1e-6,
            insert_layer_id = 6,
            window_kernel_size = window,
            merge_kernel_size = (2, 2),
            merger_times = 1,
            downsample_mode = "16x",
            head_dim = hidden // heads,
            window_hidden_size = hidden * 4,
            window_intermediate_size = intermediate * 4,
            deepstack_visual_indexes = None,
        ),
    )


def test_vision_graph_without_weights():
    model = MiniCPMV4_7VisionModel(_vision_config())
    assert model.caps["image_input"] is True
    assert model.caps["supports_tp"] is False

    keys = [module.key for module in model.modules]
    assert keys.index("model.vision_tower.encoder.layers.6") \
        < keys.index("model.vision_tower.vit_merger") \
        < keys.index("model.vision_tower.encoder.layers.7")

    leaves = {module.key for module in model if len(module.modules) == 0}
    for key in (
        "model.vision_tower.embeddings.patch_embedding",
        "model.vision_tower.embeddings.position_embedding",
        "model.vision_tower.post_layernorm",
        "model.merger.mlp.0.pre_norm",
        "model.merger.mlp.0.linear_1",
        "model.merger.mlp.0.linear_2",
    ):
        assert key in leaves, key

    linears = [module for module in model if isinstance(module, Linear)]
    assert linears
    assert all(module.qmap for module in linears)
    for width in (4304, 17216):
        matched = [module for module in linears if module.out_features_unpadded == width]
        assert matched, width
        assert all(module.out_features == width for module in matched)

    mlps = [module for module in model if isinstance(module, MLP)]
    by_key = {module.key: module for module in mlps}
    assert by_key["model.vision_tower.encoder.layers.0.mlp"].activation_fn == "gelu"
    assert by_key["model.vision_tower.vit_merger"].activation_fn == "gelu"
    assert by_key["model.merger.mlp.0"].activation_fn == "gelu_exact"
    assert all(module.num_slices == 1 for module in mlps)

    attns = [module for module in model if isinstance(module, Attention)]
    assert attns
    assert all(module.caps.get("kv_cache") is False for module in attns)
    assert all(module.q_proj.qmap == "block.attn.input" for module in attns)
    assert all(module.o_proj.qmap == "block.attn.o" for module in attns)

    shape, dtype = model.default_load_shape_dtype(2048)
    assert shape == (1, 3, 56, 56)
    assert dtype == torch.half
    load_params = model.default_load_params(2048)
    assert torch.equal(load_params["target_sizes"], torch.tensor([[4, 4]]))
    assert load_params["use_vit_merger"] is True

    merger = next(module for module in model.modules if module.key == "model.vision_tower.vit_merger")
    hidden = torch.randn(1, 16, 1152)
    target = torch.tensor([[4, 4]])
    params = {"use_vit_merger": False, "target_sizes": target}
    assert merger.forward(hidden, params) is hidden
    assert torch.equal(params["target_sizes"], target)


def test_text_class_reuses_qwen35_and_is_registered():
    assert MiniCPMV4_7TextModel.config_class.arch_string == "MiniCPMV4_7ForConditionalGeneration"
    assert MiniCPMV4_7TextModel.__init__ is Qwen3_5VLMoeModel.__init__
    entry = ARCHITECTURES["MiniCPMV4_7ForConditionalGeneration"]
    assert entry["model_class"] is MiniCPMV4_7TextModel
    assert entry["config_class"] is MiniCPMV4_7TextModel.config_class
