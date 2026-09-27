#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from lucida_mini.schema import DatasetManifest


SYSTEM_PROMPT = """You control a 9-DoF object gizmo. Inspect the images and action history, then output exactly one valid GizmoAct XML action and no other text."""


def semantic_action_spans(
    action: str,
    opening_tag_weight: float = 2.0,
    payload_value_weight: float = 4.0,
) -> list[tuple[int, int, float]]:
    """Return character spans whose action semantics deserve extra loss weight.

    GizmoAct serializes a compact XML tag around either a compact JSON object or
    the literal ``permute_axis``.  Weighting parser-level spans avoids the
    incorrect alternative of weighting tokenizer IDs: tokens such as ``x`` and
    ``y`` occur in both JSON keys and action values.
    """
    opening_end = action.find(">")
    closing_start = action.rfind("<")
    if opening_end < 0 or closing_start <= opening_end:
        raise ValueError(f"invalid GizmoAct serialization: {action!r}")
    spans = [(0, opening_end + 1, opening_tag_weight)]
    body_start = opening_end + 1
    body = action[body_start:closing_start]
    if body == "permute_axis":
        spans.append((body_start, closing_start, payload_value_weight))
        return spans
    # Values in the project's canonical JSON serializers are either quoted
    # strings or finite decimal numbers.  Only values, never field names or
    # punctuation, receive the stronger semantic weight.
    for match in re.finditer(r':(?:"[^"\\]*"|-?\d+(?:\.\d+)?)', body):
        value_start = body_start + match.start() + 1
        spans.append((value_start, body_start + match.end(), payload_value_weight))
    return spans


def focus_actions_from_csv(path: Path) -> set[str]:
    """Read the gold actions still missed by a teacher-forced evaluation."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"gold_action", "canonical_exact_match"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"focus CSV must contain {sorted(required)}: {path}")
    actions = {
        row["gold_action"]
        for row in rows
        if float(row["canonical_exact_match"]) < 1.0
    }
    if not actions:
        raise ValueError(f"focus CSV has no missed actions: {path}")
    return actions


def example_weights_for_answers(
    answers: list[str],
    action_weighting: dict | None,
    focus_actions: set[str] | None = None,
) -> list[float]:
    """Return one static loss scale per supervised complete action.

    ``power_law`` is the original capped inverse-frequency rule and remains
    the default for existing configs.  ``empirical_plus_uniform_action_band``
    mixes ordinary per-turn SFT with a uniform-over-complete-actions term for
    a frequency band.  It is intentionally static: its only inputs are the
    frozen answer strings and their corpus counts, never model errors.
    """
    if not answers:
        return []
    if not action_weighting or not action_weighting.get("enabled", False):
        return [1.0] * len(answers)

    frequency = Counter(answers)
    mode = action_weighting.get("example_weight_mode", "power_law")
    if mode == "power_law":
        # Preserve the original action-weighting behavior for all existing
        # configs that do not declare an explicit example_weight_mode.
        max_frequency = max(frequency.values())
        exponent = float(action_weighting["action_frequency_exponent"])
        cap = float(action_weighting["action_frequency_cap"])
        raw_weights = [
            min(cap, (max_frequency / frequency[answer]) ** exponent)
            for answer in answers
        ]
    elif mode == "empirical_plus_uniform_action_band":
        empirical_mass = float(action_weighting["empirical_mass"])
        uniform_mass = float(action_weighting["uniform_action_band_mass"])
        if (
            not math.isfinite(empirical_mass)
            or not math.isfinite(uniform_mass)
            or empirical_mass < 0.0
            or uniform_mass < 0.0
            or not math.isclose(empirical_mass + uniform_mass, 1.0, abs_tol=1e-8)
        ):
            raise ValueError(
                "empirical_mass and uniform_action_band_mass must be finite, "
                "non-negative, and sum to 1"
            )
        minimum = int(action_weighting["action_band_min_frequency"])
        maximum = int(action_weighting["action_band_max_frequency"])
        if minimum < 1 or maximum < minimum:
            raise ValueError("action frequency band must satisfy 1 <= min <= max")
        band_actions = {
            answer
            for answer, count in frequency.items()
            if minimum <= count <= maximum
        }
        if not band_actions:
            raise ValueError(
                f"no complete actions have frequency in [{minimum}, {maximum}]"
            )
        # L = empirical_mass * mean_over_turns(L_i) + uniform_mass *
        # mean_over_actions_in_band(mean_over_turns_for_that_action(L_i)).
        # Expressing this as a per-example scale keeps the dataset fixed and
        # works with the existing micro-batch-one DDP setup.  The mean scale is
        # exactly one when no focus multiplier is applied.
        number_of_answers = len(answers)
        number_of_band_actions = len(band_actions)
        raw_weights = [
            empirical_mass
            + (
                uniform_mass
                * number_of_answers
                / (number_of_band_actions * frequency[answer])
                if answer in band_actions
                else 0.0
            )
            for answer in answers
        ]
    else:
        raise ValueError(f"unsupported example_weight_mode: {mode!r}")

    focus = focus_actions or set()
    focus_multiplier = float(action_weighting.get("focus_action_multiplier", 1.0))
    if focus:
        raw_weights = [
            weight * focus_multiplier if answer in focus else weight
            for answer, weight in zip(answers, raw_weights)
        ]
    if action_weighting.get("normalize_example_weights", True):
        mean_weight = sum(raw_weights) / len(raw_weights)
        return [weight / mean_weight for weight in raw_weights]
    return raw_weights


def supervised_prediction_positions(labels: torch.Tensor) -> torch.Tensor:
    """Positions whose next tokens are supervised, for micro-batch one."""
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("supervised-logit selection requires micro-batch one")
    positions = torch.nonzero(labels[0, 1:].ne(-100), as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("sample contains no supervised next tokens")
    return positions


def weighted_causal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_token_weights: torch.Tensor,
    prediction_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the same weighted CE with full or supervised-position logits."""
    shifted_labels = labels[:, 1:].contiguous()
    shifted_weights = action_token_weights[:, 1:].to(logits.dtype)
    if prediction_positions is None:
        shifted_logits = logits[:, :-1, :].contiguous()
    else:
        shifted_logits = logits.contiguous()
        shifted_labels = shifted_labels.index_select(1, prediction_positions)
        shifted_weights = shifted_weights.index_select(1, prediction_positions)
    valid = shifted_labels.ne(-100)
    per_token = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    effective_weights = shifted_weights * valid
    return (per_token * effective_weights).sum() / effective_weights.sum().clamp_min(1)


class OverfitLoggingCallback(TrainerCallback):
    def __init__(self, output_dir: Path, threshold: float, patience: int):
        self.path = output_dir / "train_metrics.csv"
        self.threshold = threshold
        self.patience = patience
        self.below_threshold = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # Trainer restores checkpoint intervals after initially parsing the new
        # args. Reapply this continuation's intervals without touching optimizer,
        # scheduler, RNG, global_step, or sampler state.
        state.compute_steps(args, state.max_steps)
        if state.is_world_process_zero:
            print(f"training intervals: save_steps={state.save_steps}, logging_steps={state.logging_steps}", flush=True)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs or "loss" not in logs:
            return control
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists()
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "loss", "learning_rate", "epoch"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "step": state.global_step,
                "loss": logs["loss"],
                "learning_rate": logs.get("learning_rate", float("nan")),
                "epoch": logs.get("epoch", float("nan")),
            })
        # A single sampled micro-batch is useful for plotting, but it is not a
        # defensible convergence criterion for a 50-trajectory overfit run.
        # Keep optional early stopping available for explicit experiments while
        # disabling it when the config supplies no positive patience.
        if self.patience > 0 and self.threshold >= 0:
            self.below_threshold = self.below_threshold + 1 if logs["loss"] <= self.threshold else 0
            if self.below_threshold >= self.patience:
                control.should_save = True
                control.should_training_stop = True
        return control


class TurnDataset(Dataset):
    def __init__(
        self,
        manifest: DatasetManifest,
        root: Path,
        action_weighting: dict | None = None,
        focus_actions: set[str] | None = None,
    ):
        self.root = root
        self.examples: list[dict] = []
        for trajectory in manifest.trajectories:
            history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            for turn in trajectory.turns:
                image_content = [
                    {"type": "image", "image": str(root / path)}
                    for path in turn.observation_paths
                ]
                history.append(
                    {
                        "role": "user",
                        "content": image_content
                        + [{"type": "text", "text": "Predict the next GizmoAct action."}],
                    }
                )
                if turn.supervise:
                    self.examples.append({
                        "messages": list(history),
                        "answer": turn.action,
                        "trajectory_id": trajectory.trajectory_id,
                        "step": turn.step,
                    })
                history.append({"role": "assistant", "content": turn.action})
        self._set_example_weights(action_weighting, focus_actions or set())

    def _set_example_weights(self, action_weighting: dict | None, focus_actions: set[str]) -> None:
        weights = example_weights_for_answers(
            [example["answer"] for example in self.examples],
            action_weighting,
            focus_actions,
        )
        for example, weight in zip(self.examples, weights):
            example["example_weight"] = weight

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


@dataclass
class GizmoCollator:
    processor: object
    max_length: int
    action_weighting: dict | None = None

    def _token_weights(
        self,
        answer: str,
        encoded: dict[str, torch.Tensor],
        labels: torch.Tensor,
        prefix: int,
    ) -> torch.Tensor:
        """Build per-token semantic weights and verify answer-token alignment."""
        tokenized = self.processor.tokenizer(
            answer,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        answer_ids = torch.tensor(tokenized["input_ids"], dtype=encoded["input_ids"].dtype)
        answer_offsets = tokenized["offset_mapping"]
        answer_end = prefix + len(answer_ids)
        actual_ids = encoded["input_ids"][0, prefix:answer_end].cpu()
        if not torch.equal(actual_ids, answer_ids):
            raise ValueError("answer tokenization does not align with the supervised suffix")

        weights = torch.zeros_like(labels, dtype=torch.float32)
        valid = labels.ne(-100)
        weights[valid] = float(self.action_weighting.get("syntax_token_weight", 1.0))
        spans = semantic_action_spans(
            answer,
            float(self.action_weighting.get("opening_tag_weight", 2.0)),
            float(self.action_weighting.get("payload_value_weight", 4.0)),
        )
        for token_index, (start, end) in enumerate(answer_offsets):
            token_weight = 1.0
            for span_start, span_end, span_weight in spans:
                if start < span_end and end > span_start:
                    token_weight = max(token_weight, span_weight)
            weights[0, prefix + token_index] = token_weight

        if self.action_weighting.get("normalize_token_weights_per_turn", True):
            mean = weights[valid].mean()
            weights[valid] /= mean.clamp_min(torch.finfo(weights.dtype).eps)
        return weights

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        # Default training uses micro-batch one. Supporting larger batches here would
        # require padding variable image grids and prefix boundaries independently.
        if len(examples) != 1:
            raise ValueError("GizmoCollator requires per_device_train_batch_size=1")
        from qwen_vl_utils import process_vision_info

        example = examples[0]
        prompt_messages = example["messages"]
        full_messages = prompt_messages + [{"role": "assistant", "content": example["answer"]}]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        images, videos = process_vision_info(full_messages)
        encoded = self.processor(
            text=[full_text], images=images, videos=videos,
            padding=False, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_encoded = self.processor(
            text=[prompt_text], images=images, videos=videos,
            padding=False, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        prefix = prompt_encoded["input_ids"].shape[1]
        if encoded["input_ids"].shape[1] >= self.max_length:
            raise ValueError("sample reached max_sequence_length; refusing silent action truncation")
        labels = encoded["input_ids"].clone()
        labels[:, :prefix] = -100
        encoded["labels"] = labels
        if self.action_weighting and self.action_weighting.get("enabled", False):
            encoded["action_token_weights"] = self._token_weights(
                example["answer"], encoded, labels, prefix
            )
            encoded["example_weight"] = torch.tensor([example["example_weight"]], dtype=torch.float32)
        return encoded


class ActionWeightedTrainer(Trainer):
    """Trainer that emphasizes action choices while retaining standard CE units."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        token_weights = inputs.pop("action_token_weights")
        example_weight = inputs.pop("example_weight")
        positions = supervised_prediction_positions(labels)
        # Keep the full multimodal history in the transformer, but project only
        # supervised prediction positions through the vocabulary head. Prompt
        # logits have zero loss weight and otherwise consume several GiB.
        outputs = model(**inputs, logits_to_keep=positions)
        loss = weighted_causal_cross_entropy(
            outputs.logits, labels, token_weights, prediction_positions=positions
        )
        # The current collator intentionally uses a micro-batch of one.  Apply
        # the per-example reweighting after normalizing its token loss, so it is
        # not cancelled by the token-weight denominator.
        loss = loss * example_weight.to(loss.dtype).mean()
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sft_lora.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--focus-errors-csv", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=str)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    train_config = config["training"]
    # Bind before RNG initialization/model setup to avoid every DDP rank
    # creating an unnecessary CUDA context on the first visible GPU.
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    set_seed(train_config["seed"])
    manifest = DatasetManifest.model_validate_json(args.manifest.read_text())
    if not manifest.frozen:
        raise ValueError("training requires a frozen manifest")
    missing = manifest.validate_files(args.dataset_root)
    if missing:
        raise FileNotFoundError(f"manifest references {len(missing)} missing files; first: {missing[0]}")

    model_path = args.model or config["model"]["name_or_path"]
    processor = AutoProcessor.from_pretrained(model_path)
    if "vision_max_pixels" in config["model"]:
        processor.image_processor.max_pixels = config["model"]["vision_max_pixels"]
    if "vision_min_pixels" in config["model"]:
        processor.image_processor.min_pixels = config["model"]["vision_min_pixels"]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if config["model"]["freeze_vision_encoder"]:
        for parameter in model.visual.parameters():
            parameter.requires_grad = False
    if config["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        lora = config["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                target_modules=lora["target_modules"], task_type="CAUSAL_LM",
            ),
        )
    if int(os.environ.get("RANK", "0")) == 0:
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        max_steps=train_config["max_steps"],
        warmup_ratio=train_config["warmup_ratio"],
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        bf16=train_config["bf16"],
        remove_unused_columns=False,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        save_only_model=False,
    )
    action_weighting = config.get("action_weighting")
    focus_actions = (
        focus_actions_from_csv(args.focus_errors_csv) if args.focus_errors_csv else set()
    )
    if focus_actions and int(os.environ.get("RANK", "0")) == 0:
        print(f"focusing on {len(focus_actions)} missed complete actions")
    trainer_class = ActionWeightedTrainer if action_weighting and action_weighting.get("enabled", False) else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=TurnDataset(
            manifest,
            args.dataset_root,
            action_weighting,
            focus_actions,
        ),
        data_collator=GizmoCollator(
            processor,
            train_config["max_sequence_length"],
            action_weighting,
        ),
        callbacks=[OverfitLoggingCallback(
            args.output_dir,
            train_config.get("overfit_loss_threshold", -1.0),
            train_config.get("overfit_loss_patience", 20),
        )],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    if trainer.is_world_process_zero():
        (args.output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))
        (args.output_dir / "run_metadata.json").write_text(json.dumps({
            "init_adapter": str(args.init_adapter) if args.init_adapter else None,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "focus_errors_csv": str(args.focus_errors_csv) if args.focus_errors_csv else None,
            "focused_complete_action_count": len(focus_actions),
            "base_model": str(model_path),
            "dataset_manifest": str(args.manifest.resolve()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "peak_cuda_reserved_mib": (
                round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1)
                if torch.cuda.is_available() else None
            ),
        }, indent=2))


if __name__ == "__main__":
    main()
