# Lucida: fixed-data GizmoAct SFT prototype

Lucida is a research prototype for training and evaluating a vision-language policy on a frozen set of GizmoAct pose-editing trajectories. The experiment studies fitting this fixed set; it does not measure generalization to new scenes or initial states.

## Experiment snapshot

The latest local run completed 2,000 steps on 50 fixed trajectories. These figures describe that run; the dataset, pretrained model and run artifacts are not included in this repository.

| Metric | Step 1,000 | Step 2,000 |
|---|---:|---:|
| Exact supervised actions | 175/207 | 205/207 (99.03%) |
| Trajectories with all supervised actions exact | 19/50 | 48/50 |
| Success under the fixed error-injection schedule | 29/50 | 48/50 |
| Teacher-forced token cross-entropy | 0.0263025 | 0.00155847 |

The run did not reach a perfect 50/50 result. The execution metric replays prescribed masked error injections and lets the model generate the other actions, so it is not an unassisted rollout or a test on unseen data. See [the experiment protocol](docs/implementation_plan.md) and [data provenance notes](docs/data_source_status.md).

## Repository layout

```text
configs/              Training and evaluation configurations
src/lucida_mini/      Core actions, state, geometry and evaluation code
scripts/              Data preparation, training, evaluation and plotting entry points
tests/                Unit and protocol checks
docs/                 Experiment protocol and data provenance
pyproject.toml         Package metadata and optional dependency groups
data/README.md         Local dataset and asset placement
models/README.md       Local pretrained model placement
```

Datasets, downloaded weights, environments, caches, generated outputs, vendored dependencies and the source-paper PDF are intentionally excluded. See the `data/` and `models/` notes before attempting to run training scripts.

## Setup

Use Python 3.11. Install the package and the test dependencies with:

```bash
python -m pip install -e '.[test]'
pytest -q
```

For training, first install a PyTorch build that matches the machine's CUDA runtime, then install the training extras:

```bash
python -m pip install -e '.[train]'
```

The training configs expect the Qwen3-VL-2B-Instruct checkpoint at `models/Qwen3-VL-2B-Instruct` by default. Obtain the model from its publisher and follow its access and license terms. The code also requires the matching frozen dataset and generated observation assets; none are bundled here. See [data/README.md](data/README.md).

## Scope and reproducibility

The fixed manifest describes five contexts and 50 trajectories (207 supervised expert-action turns and 45 masked injected-error turns). The protocol, metrics and implementation choices are documented in [docs/implementation_plan.md](docs/implementation_plan.md). Configurations under `configs/` capture experiment variants; check each config before reusing it because some are smoke-test, resume or historical settings.

Local machine setup is not portable. Do not use a checked-in environment snapshot; install from `pyproject.toml` and configure paths for your machine. Training outputs and checkpoints stay under the ignored `outputs/` directory.

## License and redistribution

This repository does not yet include a project license. Choose one before publishing if you want to grant reuse rights. Separately review the terms for the research paper, source datasets, pretrained model and upstream code before redistributing any of them; the current Git ignore rules keep those local assets out of the repository.
