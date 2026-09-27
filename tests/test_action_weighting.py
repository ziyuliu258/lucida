import csv

import pytest
import torch
import torch.nn.functional as F

from scripts.train_sft import (
    example_weights_for_answers,
    focus_actions_from_csv,
    semantic_action_spans,
    supervised_prediction_positions,
    ActionWeightedTrainer,
    weighted_causal_cross_entropy,
)


def test_semantic_spans_weight_values_not_json_keys():
    action = '<permute_axis>{"x":"-y","z":"x"}</permute_axis>'
    spans = semantic_action_spans(action)
    weighted_text = [action[start:end] for start, end, weight in spans if weight == 4.0]
    assert weighted_text == ['"-y"', '"x"']
    assert '"x"' not in [action[start:end] for start, end, _ in spans[:1]]


def test_unit_weights_match_standard_shifted_cross_entropy():
    torch.manual_seed(7)
    logits = torch.randn(1, 5, 11)
    labels = torch.tensor([[-100, 2, 3, -100, 5]])
    weights = torch.ones_like(labels, dtype=torch.float32)
    actual = weighted_causal_cross_entropy(logits, labels, weights)
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(actual, expected)


def test_focus_csv_only_selects_missed_gold_actions(tmp_path):
    path = tmp_path / "per_turn.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gold_action", "canonical_exact_match"])
        writer.writeheader()
        writer.writerows([
            {"gold_action": "<stop>{}</stop>", "canonical_exact_match": "1.0"},
            {"gold_action": "<switch_obs>permute_axis</switch_obs>", "canonical_exact_match": "0.0"},
        ])
    assert focus_actions_from_csv(path) == {"<switch_obs>permute_axis</switch_obs>"}


def test_band_balanced_example_weights_mix_empirical_and_action_uniform_terms():
    # N=30; only the two 4--19-count complete actions are in the uniform
    # action term.  The common and singleton actions retain the empirical half.
    answers = ["common"] * 20 + ["band4"] * 4 + ["band5"] * 5 + ["singleton"]
    config = {
        "enabled": True,
        "example_weight_mode": "empirical_plus_uniform_action_band",
        "empirical_mass": 0.5,
        "uniform_action_band_mass": 0.5,
        "action_band_min_frequency": 4,
        "action_band_max_frequency": 19,
        "normalize_example_weights": False,
    }
    weights = example_weights_for_answers(answers, config)

    by_answer = {
        answer: {weight for seen_answer, weight in zip(answers, weights) if seen_answer == answer}
        for answer in set(answers)
    }
    assert by_answer["common"] == {0.5}
    assert by_answer["singleton"] == {0.5}
    assert len(by_answer["band4"]) == 1
    assert len(by_answer["band5"]) == 1
    assert next(iter(by_answer["band4"])) == pytest.approx(2.375)
    assert next(iter(by_answer["band5"])) == pytest.approx(2.0)
    # 0.5 * empirical mean + 0.5 * uniform-action mean has unit average scale.
    assert sum(weights) == pytest.approx(len(answers))


def test_default_example_weight_mode_preserves_legacy_capped_power_law():
    answers = ["common"] * 8 + ["target"] * 2
    config = {
        "enabled": True,
        # Deliberately omit example_weight_mode: existing configs use this path.
        "action_frequency_exponent": 0.5,
        "action_frequency_cap": 4.0,
        "focus_action_multiplier": 3.0,
        "normalize_example_weights": True,
    }
    weights = example_weights_for_answers(answers, config, {"target"})
    raw_common = 1.0
    # The legacy path caps before applying the hard-mined focus multiplier.
    raw_target = min(4.0, (8 / 2) ** 0.5) * 3.0
    normalizer = (8 * raw_common + 2 * raw_target) / len(answers)

    assert weights[:8] == pytest.approx([raw_common / normalizer] * 8)
    assert weights[8:] == pytest.approx([raw_target / normalizer] * 2)


@pytest.mark.parametrize("labels", [
    [[-100, -100, -100, 2, 5, 7, 3]],
    [[-100, 2, -100, 4, -100, -100, 3]],
])
def test_supervised_logits_preserve_weighted_loss_and_all_parameter_gradients(labels):
    from copy import deepcopy
    from types import SimpleNamespace

    class TinyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(13, 8)
            self.head = torch.nn.Linear(8, 11)
            self.positions = None

        def forward(self, input_ids, logits_to_keep=None):
            # Cumulative history also checks gradients reaching prompt tokens.
            hidden = self.embedding(input_ids).cumsum(dim=1)
            self.positions = logits_to_keep
            if logits_to_keep is not None:
                hidden = hidden[:, logits_to_keep, :]
            return SimpleNamespace(logits=self.head(hidden))

    torch.manual_seed(19)
    full_model = TinyHead().double()
    compact_model = deepcopy(full_model)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    targets = torch.tensor(labels)
    weights = torch.tensor([[0., 2., 4., 1., 3., 2., 1.]])
    example_weight = torch.tensor([1.7], dtype=torch.float64)
    full_logits = full_model(ids).logits
    expected = weighted_causal_cross_entropy(full_logits, targets, weights) * example_weight.mean()
    expected.backward()
    inputs = dict(input_ids=ids, labels=targets, action_token_weights=weights,
                  example_weight=example_weight)
    actual = ActionWeightedTrainer.compute_loss(None, compact_model, inputs)
    actual.backward()
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert compact_model.positions.numel() == targets[:, 1:].ne(-100).sum()
    for original, compact in zip(full_model.parameters(), compact_model.parameters()):
        assert torch.allclose(original.grad, compact.grad, atol=1e-12, rtol=1e-12)
    assert compact_model.embedding.weight.grad[1].abs().sum() > 0


def test_supervised_logits_reject_empty_supervision():
    with pytest.raises(ValueError, match="no supervised"):
        supervised_prediction_positions(torch.full((1, 5), -100))


def test_resume_honors_new_save_interval_without_resetting_training_state(tmp_path):
    from transformers import TrainingArguments, TrainerState, TrainerControl
    from transformers.trainer_callback import DefaultFlowCallback
    from scripts.train_sft import OverfitLoggingCallback

    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
                             max_steps=2000, save_steps=100, logging_steps=1)
    state = TrainerState(global_step=1100, max_steps=2000, epoch=26.19,
                         save_steps=200, logging_steps=1)
    flow = DefaultFlowCallback()
    assert not flow.on_step_end(args, state, TrainerControl()).should_save
    callback = OverfitLoggingCallback(tmp_path, threshold=-1, patience=0)
    callback.on_train_begin(args, state, TrainerControl())
    assert state.global_step == 1100 and state.epoch == 26.19
    assert flow.on_step_end(args, state, TrainerControl()).should_save
