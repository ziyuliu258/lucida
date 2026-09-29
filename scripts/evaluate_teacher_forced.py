#!/usr/bin/env python3
"""Evaluate exact next-action recall with the gold GizmoAct history.

Unlike ``evaluate_closed_loop.py``, this script never feeds a model action
back into the prompt.  Each supervised turn receives exactly the same history
as ``TurnDataset`` used during SFT: every prior user observation and its gold
assistant action, including deliberately injected and unsupervised error
actions.  It reports both greedy action recall and teacher-forced token CE, so
that a low aggregate loss cannot be confused with correct action recall.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from lucida_mini.actions import PermuteAxis, Stop, SwitchObs, UpdatePose, parse_action
from lucida_mini.schema import DatasetManifest
from lucida_mini.serialization import action_to_text
from lucida_mini.prompts import SYSTEM_PROMPT, instruction_for_turn
from lucida_mini.vision import (
    image_paths_in_messages,
    validate_image_files,
    validate_processed_image_grids,
)


def action_type(action: object) -> str:
    """Return the GizmoAct XML tag for a parsed action."""
    if isinstance(action, UpdatePose):
        return "update_pose"
    if isinstance(action, Stop):
        return "stop"
    if isinstance(action, SwitchObs):
        return "switch_obs"
    if isinstance(action, PermuteAxis):
        return "permute_axis"
    raise TypeError(f"unsupported action type: {type(action)!r}")


def canonicalize_action(text: str) -> tuple[str, str]:
    """Strictly parse an action and return its canonical XML plus its tag."""
    action = parse_action(text)
    return action_to_text(action), action_type(action)


def iter_supervised_turns(
    manifest: DatasetManifest, dataset_root: Path
) -> Iterator[dict]:
    """Yield records with precisely the history constructed by ``TurnDataset``."""
    contexts = {context.context_id: context for context in manifest.contexts}
    for trajectory in manifest.trajectories:
        context = contexts[trajectory.context_id]
        history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        injected_errors_before = 0
        for turn_index, turn in enumerate(trajectory.turns):
            image_content = [
                {"type": "image", "image": str(dataset_root / path)}
                for path in turn.observation_paths
            ]
            history.append({
                "role": "user",
                "content": image_content
                + [{
                    "type": "text",
                    "text": instruction_for_turn(context, turn, dataset_root),
                }],
            })
            if turn.supervise:
                yield {
                    "trajectory_id": trajectory.trajectory_id,
                    "context_id": trajectory.context_id,
                    "turn_index": turn_index,
                    "turn_step": turn.step,
                    "injected_errors_before": injected_errors_before,
                    "messages": list(history),
                    "answer": turn.action,
                }
            elif turn.injected_error:
                injected_errors_before += 1

            # This is deliberately unconditional.  In particular, a masked
            # injected-error action remains part of every later supervised
            # prompt, exactly as it did in TurnDataset during training.
            history.append({"role": "assistant", "content": turn.action})


def encode_supervised_turn(
    processor: object,
    messages: list[dict],
    answer: str,
    max_sequence_length: int,
    vision_max_pixels: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Mirror GizmoCollator's prompt/full encoding and supervision mask."""
    from qwen_vl_utils import process_vision_info

    full_messages = messages + [{"role": "assistant", "content": answer}]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    image_paths = image_paths_in_messages(full_messages)
    image_processor = processor.image_processor
    patch_size = int(getattr(image_processor, "patch_size", 16))
    merge_size = int(getattr(image_processor, "spatial_merge_size", 2))
    image_sizes = validate_image_files(
        image_paths,
        vision_max_pixels,
        patch_size=patch_size,
        spatial_merge_size=merge_size,
    )
    images, videos = process_vision_info(full_messages)
    full_inputs = processor(
        text=[full_text],
        images=images,
        videos=videos,
        padding=False,
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
    )
    validate_processed_image_grids(full_inputs, image_sizes, patch_size=patch_size)
    prompt_inputs = processor(
        text=[prompt_text],
        images=images,
        videos=videos,
        padding=False,
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
    )
    validate_processed_image_grids(prompt_inputs, image_sizes, patch_size=patch_size)
    if full_inputs["input_ids"].shape[1] >= max_sequence_length:
        raise ValueError(
            "sample reached --max-sequence-length; refusing to score a truncated action"
        )
    prefix_length = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prefix_length] = -100
    if not torch.any(labels != -100):
        raise ValueError("no supervised tokens remain after prompt masking")
    return full_inputs, prompt_inputs, labels


def move_to_device(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def teacher_forced_ce(
    model: object,
    full_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, int]:
    """Return mean CE, summed NLL, and count for the SFT-supervised tokens."""
    inputs = move_to_device(full_inputs, device)
    labels = labels.to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(-100)
    token_count = int(mask.sum().item())
    if token_count == 0:
        raise ValueError("no next-token labels remain after causal shift")
    nll = F.cross_entropy(
        shifted_logits[mask], shifted_labels[mask], reduction="sum"
    )
    nll_sum = float(nll.item())
    return nll_sum / token_count, nll_sum, token_count


def greedy_action(
    model: object,
    processor: object,
    prompt_inputs: dict[str, torch.Tensor],
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Greedily decode one action from the gold history prompt."""
    inputs = move_to_device(prompt_inputs, device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated = output[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--adapter", type=Path,
        help="Optional LoRA adapter; omit to evaluate the pretrained base as-is.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=131072)
    parser.add_argument("--vision-max-pixels", type=int, default=786432)
    parser.add_argument("--vision-min-pixels", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_sequence_length <= 0:
        raise ValueError("--max-sequence-length must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard-count)")

    manifest = DatasetManifest.model_validate_json(args.manifest.read_text())
    if not manifest.frozen:
        raise ValueError("evaluation requires a frozen manifest")
    missing = manifest.validate_files(args.dataset_root)
    if missing:
        raise FileNotFoundError(
            f"manifest references {len(missing)} missing files; first: {missing[0]}"
        )
    manifest.validate_corrected_inputs(args.dataset_root)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.image_processor.max_pixels = args.vision_max_pixels
    processor.image_processor.min_pixels = args.vision_min_pixels
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = (
        PeftModel.from_pretrained(base, args.adapter)
        if args.adapter is not None else base
    ).eval().to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    total_nll = 0.0
    total_tokens = 0
    type_counts: Counter[str] = Counter()
    for global_index, example in enumerate(iter_supervised_turns(manifest, args.dataset_root)):
        if global_index % args.shard_count != args.shard_index:
            continue
        example_number = len(rows) + 1
        full_inputs, prompt_inputs, labels = encode_supervised_turn(
            processor,
            example["messages"],
            example["answer"],
            args.max_sequence_length,
            args.vision_max_pixels,
        )
        gold_canonical, gold_type = canonicalize_action(example["answer"])
        token_ce, token_nll, token_count = teacher_forced_ce(
            model, full_inputs, labels, device
        )
        raw_prediction = greedy_action(
            model, processor, prompt_inputs, device, args.max_new_tokens
        )
        try:
            predicted_canonical, predicted_type = canonicalize_action(raw_prediction)
            valid_action = 1.0
            parse_error = ""
        except (TypeError, ValueError) as exc:
            predicted_canonical = ""
            predicted_type = "invalid"
            valid_action = 0.0
            parse_error = str(exc)

        row = {
            "step": args.step,
            "trajectory_id": example["trajectory_id"],
            "context_id": example["context_id"],
            "turn_index": example["turn_index"],
            "turn_step": example["turn_step"],
            "injected_errors_before": example["injected_errors_before"],
            "gold_action": example["answer"],
            "gold_canonical_action": gold_canonical,
            "gold_action_type": gold_type,
            "prediction_raw": raw_prediction,
            "prediction_canonical_action": predicted_canonical,
            "prediction_action_type": predicted_type,
            "valid_gizmoact_xml": valid_action,
            "action_type_match": float(predicted_type == gold_type),
            "canonical_exact_match": float(predicted_canonical == gold_canonical),
            "teacher_forced_token_ce": token_ce,
            "teacher_forced_nll_sum": token_nll,
            "teacher_forced_token_count": token_count,
            "parse_error": parse_error,
        }
        rows.append(row)
        total_nll += token_nll
        total_tokens += token_count
        type_counts[gold_type] += 1
        print(
            f"[{example_number:03d}] {example['trajectory_id']} turn={example['turn_index']} "
            f"CE={token_ce:.4f} valid={int(valid_action)} "
            f"type={int(row['action_type_match'])} exact={int(row['canonical_exact_match'])}"
        )

    if not rows:
        raise ValueError("manifest has no supervised turns")
    summary = {
        "step": args.step,
        "supervised_turns": len(rows),
        "teacher_forced_token_count": total_tokens,
        "teacher_forced_mean_ce": total_nll / total_tokens,
        "mean_per_turn_ce": sum(row["teacher_forced_token_ce"] for row in rows) / len(rows),
        "valid_gizmoact_xml_rate": sum(row["valid_gizmoact_xml"] for row in rows) / len(rows),
        "action_type_match_rate": sum(row["action_type_match"] for row in rows) / len(rows),
        "canonical_exact_match_rate": sum(row["canonical_exact_match"] for row in rows) / len(rows),
    }
    summary_json = {
        "summary": summary,
        "gold_action_type_counts": dict(sorted(type_counts.items())),
        "config": {
            "base_model": args.base_model,
            "adapter": str(args.adapter) if args.adapter is not None else None,
            "max_new_tokens": args.max_new_tokens,
            "max_sequence_length": args.max_sequence_length,
            "vision_max_pixels": args.vision_max_pixels,
            "vision_min_pixels": args.vision_min_pixels,
            "device": str(device),
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        },
    }

    per_turn_fields = list(rows[0])
    write_csv(args.output_dir / "per_turn.csv", rows, per_turn_fields)
    write_csv(args.output_dir / "summary.csv", [summary], list(summary))
    (args.output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2) + "\n")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
