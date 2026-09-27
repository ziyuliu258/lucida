# Experiment protocol

## Objective and scope

This project implements a fixed-data GizmoAct pose-editing experiment using supervised fine-tuning (SFT). It is designed to measure whether a policy can fit the generated training trajectories. It does not evaluate held-out scenes, unseen initial states, transfer, generalization or reinforcement learning.

The data snapshot used for the reported local run contains five fixed contexts and ten trajectories per context (50 total). The source distribution is two MesaTask / 3D-FRONT contexts, two FoundationPose contexts and one CA-1M / SAM 3D Objects context. The dataset and model weights are not included in this repository; see [data provenance](data_source_status.md).

## Environment and actions

The interaction state records object position, rotation and scale. The policy receives rendered observations and action history, then emits one of the supported GizmoAct operations:

- `update_pose` applies rotation, translation and scale increments using the configured coordinate conventions.
- `switch_obs` changes between scene and six-axis views without changing pose.
- `permute_axis` applies a legal right-handed signed axis permutation.
- `stop` ends the rollout.

The action parser validates operation names, values, axis combinations and positive resulting scales. The implementation and tests live in `src/lucida_mini/` and `tests/`.

## Training setup

The reported experiment used `Qwen/Qwen3-VL-2B-Instruct` with LoRA SFT, BF16 and distributed data parallel training. The visual backbone was frozen; LoRA modules were applied to visual and language projections. These are implementation choices for this experiment and are not attributed to the source paper.

The manifest has 252 turns: 207 supervised expert-action turns and 45 injected-error turns. Injected errors remain in the history but their output labels are masked. Training and evaluation use the same frozen set. Config files capture the run variants; inspect the selected config and supply the corresponding local dataset and model paths before launching a run.

## Evaluation protocol

Teacher-forced evaluation supplies the prescribed observations and gold action history for each supervised turn, then compares the model's generated action with the expert action. It reports exact action matches and token cross-entropy over 207 turns.

The complementary execution evaluation starts from each of the same 50 initial states. It replays the prescribed masked error injections; on other turns it executes the model's generated actions and renders the resulting observation. It allows up to 12 turns. This is a fixed error-schedule evaluation, not an unassisted rollout.

Reported geometry metrics include ADD-SB and its object-diameter thresholds, oriented 3D IoU, and rotation error. Full-loop success additionally requires valid actions, an explicit stop, ADD-SB divided by object diameter below 0.05, and rotation error below 5 degrees.

## Reported local snapshot

The latest run recorded in the project notes completed 2,000 steps. It matched 205/207 supervised actions (99.03%) and succeeded on 48/50 cases under the fixed error schedule. The two remaining exact-action errors were the initial `update_pose` actions for `foundationpose_01_09` and `front_02_09`. This is not a perfect fit. Run artifacts and the dataset used to produce these figures are local and are not part of this repository.
