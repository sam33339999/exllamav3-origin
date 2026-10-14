"""
Canvas M-RoPE and NaViT helpers for MiniCPM-V 4.7.

The position rules match ``MiniCPMV4_7Model.get_rope_index`` in transformers 5.18.
Qwen's ``gen_mrope_pos_ids`` describes one grid between vision-start and vision-end.
MiniCPM lays a thumbnail, slice rows, and marker tokens onto one canvas, so the
ids are computed here and handed to RoPE as a finished ``(3, seq)`` table.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MiniCPMCanvasConfig:
    image_start_id: int
    image_end_id: int
    slice_start_id: int
    slice_end_id: int
    newline_id: int
    downsample_mode: str = "16x"


def visual_token_count(height: int, width: int, downsample_mode: str) -> int:
    """LLM tokens for one crop. ``16x`` keeps 1/16 of the patches, ``4x`` keeps 1/4."""
    if downsample_mode not in ("4x", "16x"):
        raise ValueError(f"downsample_mode must be '4x' or '16x', got {downsample_mode!r}")
    divisor = 4 if downsample_mode == "4x" else 16
    patches = height * width
    if patches % divisor:
        raise ValueError(
            f"Patch grid {height}x{width} is not divisible by {divisor} ({downsample_mode})"
        )
    return patches // divisor


def nearest_position_ids(target_sizes: torch.Tensor, num_patches_per_side: int) -> torch.Tensor:
    """Flat indices into a learned ``side x side`` position table, one per patch.

    Fractional coordinates ``i / h`` are bucketed onto the fixed table. Same rule as
    ``transformers.vision_utils.get_vision_nearest_position_ids``.
    """
    device = target_sizes.device
    step = 1 / num_patches_per_side
    boundaries = torch.arange(step, 1.0, step, device = device)
    pos_ids = []
    for height, width in target_sizes.tolist():
        height, width = int(height), int(width)
        h_coords = torch.arange(height, device = device) / height
        w_coords = torch.arange(width, device = device) / width
        bucket_h = torch.bucketize(h_coords, boundaries, right = True)
        bucket_w = torch.bucketize(w_coords, boundaries, right = True)
        pos_ids.append((bucket_h[:, None] * num_patches_per_side + bucket_w).flatten())
    return torch.cat(pos_ids)


def window_index(
    target_sizes: torch.Tensor,
    window_size: int,
    spatial_merge_size: int = 1,
    patch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder patches into windows.

    ``target_sizes`` is ``(num_crops, 2)`` of ``(h, w)``. A leading temporal size of 1 is
    added, matching the ViT merger's call into ``get_vision_window_index``. Returns
    ``(window_index, cu_window_seqlens)``.
    """
    grid_thw = F.pad(target_sizes.to(dtype = torch.long), (1, 0), value = 1)
    return _window_index(grid_thw, spatial_merge_size, window_size, patch_size)


def _window_index(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    window_index_parts: list[torch.Tensor] = []
    cu_window_seqlens: list[int] = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size
    spatial_merge_unit = spatial_merge_size ** 2

    for grid_t, grid_h, grid_w in grid_thw.tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index_parts.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w

    order = torch.cat(window_index_parts, dim = 0)
    cu = torch.tensor(cu_window_seqlens, device = grid_thw.device, dtype = torch.int32)
    cu = torch.unique_consecutive(cu)
    return order, cu


def merge_spatial(
    patches: torch.Tensor,
    height: int,
    width: int,
    merge_h: int,
    merge_w: int,
) -> torch.Tensor:
    """Pack a row-major ``(height * width, dim)`` grid into row-major windows.

    Inside each window the patches stay row-major. This is the view/permute used by
    ``MiniCPMV4_7Merger``.
    """
    if height % merge_h or width % merge_w:
        raise ValueError(
            f"Patch grid ({height}, {width}) is not divisible by merge kernel ({merge_h}, {merge_w})"
        )
    dim = patches.shape[-1]
    merged_h = height // merge_h
    merged_w = width // merge_w
    return (
        patches.reshape(merged_h, merge_h, merged_w, merge_w, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(merged_h * merged_w, merge_h * merge_w * dim)
    )


def _frame_end_idx(last_crop_end: int, input_ids: list[int], markers: tuple[int, ...], limit: int) -> int:
    end_idx = last_crop_end
    while end_idx < limit and input_ids[end_idx] in markers:
        end_idx += 1
    return end_idx


def group_visual_frames(
    input_ids: list[int],
    mm_token_type_ids: list[int],
    markers: MiniCPMCanvasConfig,
) -> list[tuple[int, int, list]]:
    """Split one sequence into groups of thumbnail-plus-slice frames.

    Each group is ``(group_start, group_end, frames)``. A frame is
    ``(modality, thumb_start, thumb_end, frame_end, slices)`` and each slice is
    ``(start, end)``. Indices follow the transformers grouping.
    """
    marker_ids = (
        markers.slice_start_id,
        markers.slice_end_id,
        markers.image_start_id,
        markers.image_end_id,
        markers.newline_id,
    )
    slice_start_id = marker_ids[0]
    seq_len = len(input_ids)
    groups = []
    open_slices = None
    previous_end_idx = 0
    for modality, run in itertools.groupby(enumerate(mm_token_type_ids), lambda item: item[1]):
        if modality == 0:
            continue
        run = list(run)
        crop = (run[0][0], run[-1][0] + 1)
        if open_slices is not None and input_ids[crop[0] - 1] == slice_start_id:
            open_slices.append(crop)
        else:
            open_slices = []
            frame = (modality, crop[0], crop[1], open_slices)
            gap = input_ids[previous_end_idx : crop[0] - 1]
            adjacent = bool(groups) and all(token_id in marker_ids for token_id in gap)
            if adjacent:
                groups[-1][1].append(frame)
            else:
                groups.append((crop[0] - 1, [frame]))
        previous_end_idx = crop[1]

    visual_groups = []
    for group_start_idx, frames in groups:
        frames = [
            (
                modality,
                thumb_start,
                thumb_end,
                _frame_end_idx(
                    slices[-1][1] if slices else thumb_end,
                    input_ids,
                    marker_ids,
                    seq_len,
                ),
                slices,
            )
            for modality, thumb_start, thumb_end, slices in frames
        ]
        visual_groups.append((group_start_idx, frames[-1][3], frames))
    return visual_groups


def vision_position_ids(
    start_position: int,
    grid_hw: tuple[int, int] | list[int],
    canvas_height: int = 0,
    canvas_width: int = 0,
    h_offset: int = 0,
    w_offset: int = 0,
    spatial_merge_size: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``(3, llm_h * llm_w)`` positions for one crop. A zero canvas falls back to the crop itself."""
    llm_grid_h = int(grid_hw[0]) // spatial_merge_size
    llm_grid_w = int(grid_hw[1]) // spatial_merge_size
    canvas_height = canvas_height or llm_grid_h
    canvas_width = canvas_width or llm_grid_w
    h_coords = torch.linspace(0, canvas_height - 1, llm_grid_h, device = device).round().long() + h_offset
    w_coords = torch.linspace(0, canvas_width - 1, llm_grid_w, device = device).round().long() + w_offset
    h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing = "ij")
    t_grid = torch.zeros_like(h_grid)
    position_ids = torch.stack([t_grid, h_grid, w_grid], dim = 0).reshape(3, -1)
    position_ids += start_position
    return position_ids


def canvas_position_ids(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    target_sizes: torch.Tensor | None,
    target_sizes_videos: torch.Tensor | None,
    downsample_mode: str | None,
    markers: MiniCPMCanvasConfig,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(3, batch, seq)`` canvas positions and ``(batch, 1)`` rope deltas.

    ``16x`` merges 4x4 patches into one LLM token. ``4x`` skips the ViT window merger
    and merges 2x2. Text-only rows are ``arange`` on all three axes and have delta 0.
    """
    downsample_mode = downsample_mode or markers.downsample_mode
    if downsample_mode not in ("4x", "16x"):
        raise ValueError(f"downsample_mode must be '4x' or '16x', got {downsample_mode!r}")
    merge_factor = 2 if downsample_mode == "4x" else 4

    device = input_ids.device
    position_ids = torch.zeros(3, *input_ids.shape, dtype = torch.long, device = device)
    grid_iters = {
        1: iter(target_sizes.tolist()) if target_sizes is not None else None,
        2: iter(target_sizes_videos.tolist()) if target_sizes_videos is not None else None,
    }

    for batch_idx in range(input_ids.shape[0]):
        current_input_ids = input_ids[batch_idx]
        current_mm = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            valid_mask = attention_mask[batch_idx].bool()
            current_input_ids = current_input_ids[valid_mask]
            current_mm = current_mm[valid_mask]
        else:
            valid_mask = slice(None)

        current_input_ids = current_input_ids.tolist()
        current_mm = current_mm.tolist()
        seq_len = len(current_input_ids)
        curr_position_ids = torch.arange(seq_len, device = device, dtype = torch.long).expand(3, -1).clone()

        current_pos = 0
        current_idx = 0
        for group_start_idx, group_end_idx, frames in group_visual_frames(
            current_input_ids, current_mm, markers
        ):
            if group_start_idx > current_idx:
                text_len = group_start_idx - current_idx
                curr_position_ids[:, current_idx:group_start_idx] = (
                    torch.arange(text_len, device = device) + current_pos
                )
                current_pos += text_len

            frame_cursor_idx = group_start_idx
            for modality, thumb_start, thumb_end, frame_end, slices in frames:
                target_sizes_thumb = next(grid_iters[modality])
                frame_start_idx = thumb_start - 1
                if frame_start_idx > frame_cursor_idx:
                    gap_len = frame_start_idx - frame_cursor_idx
                    curr_position_ids[:, frame_cursor_idx:frame_start_idx] = (
                        torch.arange(gap_len, device = device) + current_pos
                    )
                    current_pos += gap_len + 1

                frame_end = min(frame_end, group_end_idx)
                canvas_origin = current_pos
                halo_before_canvas = max(canvas_origin - 1, 0)

                num_rows, num_cols = 0, 0
                canvas_height = target_sizes_thumb[0] // merge_factor
                canvas_width = target_sizes_thumb[1] // merge_factor
                llm_slice_h, llm_slice_w = 0, 0
                target_sizes_first_slice = None
                if slices:
                    num_cols = len(slices)
                    for k in range(len(slices) - 1):
                        if slices[k + 1][0] - slices[k][1] > 2:
                            num_cols = k + 1
                            break
                    num_rows = len(slices) // num_cols if num_cols > 0 else 1
                    if num_rows * num_cols != len(slices):
                        num_rows, num_cols = 1, len(slices)
                    target_sizes_first_slice = next(grid_iters[modality])
                    llm_slice_h = target_sizes_first_slice[0] // merge_factor
                    llm_slice_w = target_sizes_first_slice[1] // merge_factor
                    canvas_height = num_rows * llm_slice_h
                    canvas_width = num_cols * llm_slice_w

                curr_position_ids[:, frame_start_idx:frame_end] = canvas_origin
                curr_position_ids[1:, frame_start_idx] = halo_before_canvas

                if thumb_end < frame_end:
                    curr_position_ids[1, thumb_end] = canvas_origin + canvas_height
                    curr_position_ids[2, thumb_end] = canvas_origin + canvas_width

                curr_position_ids[:, thumb_start:thumb_end] = vision_position_ids(
                    start_position = canvas_origin,
                    grid_hw = target_sizes_thumb,
                    canvas_height = canvas_height,
                    canvas_width = canvas_width,
                    spatial_merge_size = merge_factor,
                    device = device,
                )

                for k, (slice_start, slice_end) in enumerate(slices):
                    h_off = (k // num_cols) * llm_slice_h
                    w_off = (k % num_cols) * llm_slice_w
                    target_sizes_slice = target_sizes_first_slice if k == 0 else next(grid_iters[modality])
                    slice_h = target_sizes_slice[0] // merge_factor
                    slice_w = target_sizes_slice[1] // merge_factor

                    slice_start_marker = slice_start - 1
                    if slice_start_marker >= frame_start_idx:
                        curr_position_ids[1, slice_start_marker] = canvas_origin + h_off
                        curr_position_ids[2, slice_start_marker] = canvas_origin + w_off
                    if slice_end < frame_end:
                        curr_position_ids[1, slice_end] = canvas_origin + h_off + slice_h - 1
                        curr_position_ids[2, slice_end] = canvas_origin + w_off + slice_w - 1

                    curr_position_ids[:, slice_start:slice_end] = vision_position_ids(
                        start_position = canvas_origin,
                        grid_hw = target_sizes_slice,
                        h_offset = h_off,
                        w_offset = w_off,
                        spatial_merge_size = merge_factor,
                        device = device,
                    )

                for k in range(len(slices) - 1):
                    gap_start, gap_end = slices[k][1], slices[k + 1][0]
                    if gap_end - gap_start <= 2:
                        continue
                    boundary_h = (k // num_cols + 1) * llm_slice_h - 1
                    right_edge_w = num_cols * llm_slice_w
                    for newline_idx in range(gap_start + 1, gap_end - 1):
                        curr_position_ids[1, newline_idx] = canvas_origin + boundary_h
                        curr_position_ids[2, newline_idx] = canvas_origin + right_edge_w

                current_pos = canvas_origin + max(canvas_height, canvas_width) + 1
                frame_cursor_idx = frame_end

            if frame_cursor_idx < group_end_idx:
                trail_len = group_end_idx - frame_cursor_idx
                curr_position_ids[:, frame_cursor_idx:group_end_idx] = (
                    torch.arange(trail_len, device = device) + current_pos
                )
                current_pos += trail_len
            current_idx = group_end_idx

        if current_idx < seq_len:
            curr_position_ids[:, current_idx:seq_len] = (
                torch.arange(seq_len - current_idx, device = device) + current_pos
            )
        position_ids[:, batch_idx, valid_mask] = curr_position_ids

    if attention_mask is not None:
        seq_lens = attention_mask.sum(-1)
    else:
        seq_lens = torch.full(
            (input_ids.shape[0],), input_ids.shape[1], device = input_ids.device, dtype = torch.long
        )
    rope_deltas = position_ids.amax(dim = (0, 2)).unsqueeze(1) + 1 - seq_lens.unsqueeze(1)
    return position_ids, rope_deltas.long()
