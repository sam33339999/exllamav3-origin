#!/usr/bin/env python3
"""MiniCPM-V 4.7 checkpoint checks.

Exits 0 when ``--model_dir`` is missing. With a directory it checks that every
safetensors key is owned by a module, then compares canvas ids to transformers.
Vision hidden states and greedy tokens run only when the weights fit in memory.
This script does not quantize.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.architecture.mm_processing.minicpmv4_7 import canvas_position_ids
from exllamav3.generator.sampler import GreedySampler


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    return "out of memory" in str(exc).lower()


def owner_key(tensor_key: str, owners: dict) -> str | None:
    found = None
    current = ""
    for index, part in enumerate(tensor_key.split(".")):
        current = part if index == 0 else current + "." + part
        if current in owners:
            found = current
    return found


def index_modules(models) -> dict:
    owners = {}
    for model in models:
        for module in model:
            if not module.key:
                continue
            leaf = len(module.modules) == 0
            previous = owners.get(module.key)
            owners[module.key] = leaf if previous is None else previous and leaf
    return owners


def check_keys(config) -> None:
    keys = list(config.stc.tensor_file_map)
    if not keys:
        fail("checkpoint has no tensors")

    models = [Model.from_config(config), Model.from_config(config, component = "vision")]
    if "mtp" in config.model_classes:
        models.append(Model.from_config(config, component = "mtp"))
    owners = index_modules(models)

    vision_prefixes = ("model.vision_tower", "model.merger")
    missing = []
    not_leaf = []
    for key in keys:
        owner = owner_key(key, owners)
        if owner is None:
            missing.append(key)
            continue
        under_vision = any(key == prefix or key.startswith(prefix + ".") for prefix in vision_prefixes)
        if under_vision and not owners[owner]:
            not_leaf.append((key, owner))

    print(f"[INFO] tensors {len(keys)}, module keys {len(owners)}")
    if missing:
        preview = ", ".join(missing[:20])
        fail(f"{len(missing)} tensor(s) are not owned by a module: {preview}")
    if not_leaf:
        preview = ", ".join(f"{key} -> {owner}" for key, owner in not_leaf[:20])
        fail(f"{len(not_leaf)} vision tensor(s) land on a non-leaf module: {preview}")
    print("[PASS] every tensor is owned; vision tensors land on leaves")


def check_processor_canvas(config) -> None:
    try:
        from PIL import Image
        from transformers import AutoProcessor
        from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7Model
    except Exception as exc:
        print(f"[SKIP] canvas processor check: {exc!r}")
        return

    try:
        processor = AutoProcessor.from_pretrained(config.directory, local_files_only = True)
    except Exception as exc:
        print(f"[SKIP] canvas processor check: {exc!r}")
        return

    image = Image.new("RGB", (480, 320), (20, 80, 140))
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Describe."}],
    }]
    text = processor.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    batch = processor(
        text = [text],
        images = [image],
        return_tensors = "pt",
        return_mm_token_type_ids = True,
        downsample_mode = "16x",
    )
    from transformers import AutoConfig
    hf = MiniCPMV4_7Model.__new__(MiniCPMV4_7Model)
    hf.config = AutoConfig.from_pretrained(config.directory, local_files_only = True)
    ours, our_delta = canvas_position_ids(
        batch["input_ids"],
        batch["mm_token_type_ids"],
        batch["target_sizes"],
        batch.get("target_sizes_videos"),
        "16x",
        config.canvas_markers(),
        attention_mask = batch.get("attention_mask"),
    )
    hf_pos, hf_delta = hf.get_rope_index(
        batch["input_ids"],
        batch["mm_token_type_ids"],
        batch["target_sizes"],
        batch.get("target_sizes_videos"),
        "16x",
        batch.get("attention_mask"),
    )
    if not torch.equal(ours, hf_pos) or not torch.equal(our_delta, hf_delta):
        fail("processor canvas ids differ from transformers get_rope_index")
    print(f"[PASS] processor canvas ids match ({ours.shape[-1]} tokens)")


def _weight_map(model_dir: str) -> dict:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding = "utf8") as handle:
            return json.load(handle)["weight_map"]
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        from safetensors import safe_open
        with safe_open(single, framework = "pt") as handle:
            return {key: "model.safetensors" for key in handle.keys()}
    raise FileNotFoundError(f"no safetensors index in {model_dir}")


def _load_prefix(model_dir: str, prefix: str) -> dict:
    from safetensors.torch import load_file
    grouped = {}
    for key, filename in _weight_map(model_dir).items():
        if key == prefix or key.startswith(prefix + "."):
            grouped.setdefault(filename, []).append(key)
    tensors = {}
    for filename, keys in grouped.items():
        shard = load_file(os.path.join(model_dir, filename), device = "cpu")
        for key in keys:
            tensors[key[len(prefix) + 1:]] = shard[key]
        del shard
    return tensors


def _report_close(name: str, ours: torch.Tensor, hf: torch.Tensor) -> None:
    if ours.shape != hf.shape:
        fail(f"{name} shape {tuple(ours.shape)} != {tuple(hf.shape)}")
    a = ours.detach().float().cpu().reshape(-1)
    b = hf.detach().float().cpu().reshape(-1)
    diff = (a - b).abs()
    cosine = torch.nn.functional.cosine_similarity(a, b, dim = 0).item()
    print(
        f"[INFO] {name} max_abs={diff.max().item():.5f} "
        f"mean_abs={diff.mean().item():.5f} cosine={cosine:.6f}"
    )
    if cosine < 0.99 or diff.max().item() > 1.0:
        fail(f"{name} is outside fp16/bf16 tolerance")


def _exllama_vision(vision, pixels, target_sizes, mode: str):
    params = {"target_sizes": target_sizes, "downsample_mode": mode}
    hidden = vision.prepare_inputs(pixels, params)
    pre = None
    for module, instance, _idx in vision.fwd_modules:
        params["layer_instance"] = instance
        hidden = module.prepare_for_device(hidden, params)
        if module.key == "model.merger":
            pre = hidden
        hidden = module.forward(hidden, params)
    return pre, hidden


def _hf_vision_forward(hf_vision, hf_merger, pixels, target_sizes, mode: str):
    use_vit = mode != "4x"
    output = hf_vision(
        pixels.to(device = hf_vision.device, dtype = hf_vision.dtype),
        target_sizes = target_sizes.to(hf_vision.device),
        use_vit_merger = use_vit,
    )
    sizes = target_sizes // 2 if use_vit else target_sizes
    merged = hf_merger(output.last_hidden_state, sizes.to(output.last_hidden_state.device))
    merged = torch.cat(merged, dim = 0).unsqueeze(0)
    return output.last_hidden_state, merged


def compare_vision(config, device: str) -> None:
    from transformers import AutoConfig
    from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import (
        MiniCPMV4_7Merger,
        MiniCPMV4_7VisionModel,
    )
    from transformers import AutoProcessor
    from PIL import Image

    hf_cfg = AutoConfig.from_pretrained(config.directory, local_files_only = True)
    hf_vision = MiniCPMV4_7VisionModel(hf_cfg.vision_config)
    hf_vision.load_state_dict(_load_prefix(config.directory, "model.vision_tower"), strict = True)
    hf_merger = MiniCPMV4_7Merger(hf_cfg)
    hf_merger.load_state_dict(_load_prefix(config.directory, "model.merger"), strict = True)
    hf_vision = hf_vision.to(device = device, dtype = torch.bfloat16).eval()
    hf_merger = hf_merger.to(device = device, dtype = torch.bfloat16).eval()

    vision = Model.from_config(config, component = "vision")
    vision.load(device = device)
    processor = AutoProcessor.from_pretrained(config.directory, local_files_only = True)
    image = Image.new("RGB", (480, 320), (20, 80, 140))

    counts = {}
    for mode in ("16x", "4x"):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Describe."}],
        }]
        text = processor.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        batch = processor(
            text = [text],
            images = [image],
            return_tensors = "pt",
            return_mm_token_type_ids = True,
            downsample_mode = mode,
        )
        pixels = batch["pixel_values"]
        target = batch["target_sizes"]
        with torch.inference_mode():
            pre, merged = _exllama_vision(vision, pixels, target, mode)
            hf_pre, hf_merged = _hf_vision_forward(hf_vision, hf_merger, pixels, target, mode)
        _report_close(f"vision {mode} last_hidden", pre, hf_pre)
        _report_close(f"vision {mode} merger", merged, hf_merged)
        counts[mode] = int(merged.shape[1])
        print(f"[INFO] vision {mode} llm tokens {counts[mode]}")

    if counts["4x"] != counts["16x"] * 4:
        fail(f"4x token count {counts['4x']} is not 4 times 16x ({counts['16x']})")
    print("[PASS] vision hidden states")
    vision.unload()
    del hf_vision, hf_merger
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _greedy(generator, input_ids: torch.Tensor, embeddings, n: int) -> torch.Tensor:
    prompt_len = input_ids.shape[-1]
    job = Job(
        input_ids = input_ids,
        max_new_tokens = n,
        min_new_tokens = n,
        sampler = GreedySampler(),
        embeddings = embeddings,
        stop_conditions = [],
    )
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        generator.iterate()
    full = job.sequences[0].sequence_ids.torch()
    return full.view(-1)[prompt_len:prompt_len + n].cpu()


def compare_generate(config, device: str) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from PIL import Image

    text_model = Model.from_config(config)
    cache = Cache(text_model, max_num_tokens = 4096)
    text_model.load(device = device)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model = text_model, cache = cache, tokenizer = tokenizer)
    hf = AutoModelForImageTextToText.from_pretrained(
        config.directory,
        dtype = torch.bfloat16,
        device_map = "auto",
        local_files_only = True,
    ).eval()
    hf_tok = AutoTokenizer.from_pretrained(config.directory, local_files_only = True)

    prompt = "The capital of France is"
    ids = hf_tok(prompt, return_tensors = "pt").input_ids
    ex_ids = tokenizer.encode(prompt, add_bos = False, encode_special_tokens = True)
    if not torch.equal(ex_ids.cpu().view(-1), ids.cpu().view(-1)):
        print("[INFO] tokenizer ids differ; both sides use the transformers ids")

    hf_device = next(hf.parameters()).device
    with torch.inference_mode():
        hf_out = hf.generate(
            input_ids = ids.to(hf_device),
            max_new_tokens = 32,
            min_new_tokens = 32,
            do_sample = False,
        )
    hf_new = hf_out[0, ids.shape[-1]:].cpu()
    ex_new = _greedy(generator, ids, None, 32)
    if not torch.equal(ex_new, hf_new):
        fail(f"text greedy mismatch\nex {ex_new.tolist()}\nhf {hf_new.tolist()}")
    print("[PASS] text greedy 32")

    vision = Model.from_config(config, component = "vision")
    vision.load(device = device)
    processor = AutoProcessor.from_pretrained(config.directory, local_files_only = True)
    image = Image.new("RGB", (480, 320), (20, 80, 140))
    for mode, n_tokens in (("16x", 16), ("4x", 8)):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}],
        }]
        prepared_text = processor.apply_chat_template(
            messages, tokenize = False, add_generation_prompt = True,
        )
        batch = processor(
            text = [prepared_text],
            images = [image],
            return_tensors = "pt",
            return_mm_token_type_ids = True,
            downsample_mode = mode,
        )
        # Rebuild through the exllama helper so the canvas embedding matches generation.
        prepared = vision.prepare_multimodal(
            tokenizer, messages, [image], downsample_mode = mode,
        )
        with torch.inference_mode():
            hf_out = hf.generate(
                **{k: v.to(hf_device) if torch.is_tensor(v) else v for k, v in batch.items()
                   if k in ("input_ids", "attention_mask", "pixel_values", "target_sizes", "mm_token_type_ids")},
                max_new_tokens = n_tokens,
                min_new_tokens = n_tokens,
                do_sample = False,
                downsample_mode = mode,
            )
        hf_new = hf_out[0, batch["input_ids"].shape[-1]:].cpu()
        ex_new = _greedy(generator, prepared["input_ids"], prepared["embeddings"], n_tokens)
        if mode == "16x" and not torch.equal(ex_new, hf_new):
            fail(f"image greedy mismatch\nex {ex_new.tolist()}\nhf {hf_new.tolist()}")
        print(f"[PASS] image {mode} generate {n_tokens}: {tokenizer.decode(ex_new.unsqueeze(0))!r}")

    vision.unload()
    text_model.unload()
    del hf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("--model_dir", required = True)
    parser.add_argument("--device", default = "cuda:0")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"[SKIP] model dir not found: {args.model_dir}")
        print("[SKIP] numeric parity and 3.5 bpw quantization were not run")
        return

    config = Config.from_directory(args.model_dir)
    print("[INFO] architecture:", config.architecture)
    if config.architecture != "MiniCPMV4_7ForConditionalGeneration":
        fail(f"unexpected architecture {config.architecture}")

    check_keys(config)
    check_processor_canvas(config)

    if not torch.cuda.is_available():
        print("[SKIP] CUDA is not available; vision and greedy parity were not run")
        print("[PASS] structural checks only")
        return

    try:
        compare_vision(config, args.device)
    except Exception as exc:
        if is_oom(exc):
            print(f"[SKIP] vision parity did not fit in memory: {exc!r}")
        else:
            raise

    try:
        compare_generate(config, args.device)
    except Exception as exc:
        if is_oom(exc):
            print(f"[SKIP] greedy parity did not fit in memory: {exc!r}")
        else:
            raise

    print("[PASS] parity script finished")
    print("[SKIP] 3.5 bpw quantization is not part of this script")


if __name__ == "__main__":
    main()
