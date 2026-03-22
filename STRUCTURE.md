# CLRerNet Repository Structure and File Responsibilities

This document is a full repository walkthrough intended to brief another coding agent (or teammate) on what exists in the repo and what each major file/folder does.

---

## 1) Top-Level Repository Overview

```text
CLRerNet/
├── LICENSE
├── README.md
├── STRUCTURE.md
├── clrernet_culane_dla34_ema.pth
├── configs/
├── dataset/
├── demo/
├── docker/
├── docker-compose.yaml
├── docs/
├── libs/
├── reproduce.md
├── requirements.txt
├── result.png
├── tests/
├── tools/
└── work_dirs/
```

### Top-level files

- `README.md`
  - Main project description, paper link, performance table, install/inference/train/test commands.

- `reproduce.md`
  - Practical reproducibility workflow, especially Docker-based setup + benchmark reproduction.

- `requirements.txt`
  - Python dependency list.

- `docker-compose.yaml`
  - Containerized development/runtime setup.

- `clrernet_culane_dla34_ema.pth`
  - Pretrained EMA model checkpoint (CULane).

- `result.png`
  - Example generated output visualization.

- `LICENSE`
  - Repository license terms.

---

## 2) `configs/` (Model + runtime configuration)

```text
configs/
├── _base_/
│   └── default_runtime.py
└── clrernet/
    ├── base_clrernet.py
    └── culane/
        ├── clrernet_culane_dla34.py
        ├── clrernet_culane_dla34_ema.py
        └── dataset_culane_clrernet.py
```

- `configs/_base_/default_runtime.py`
  - Shared runtime defaults (logging/checkpoint intervals/hooks style in mmdet config ecosystem).

- `configs/clrernet/base_clrernet.py`
  - Base CLRerNet architecture/training settings inherited by concrete configs.

- `configs/clrernet/culane/clrernet_culane_dla34.py`
  - Main CULane training/eval config for DLA34 backbone.

- `configs/clrernet/culane/clrernet_culane_dla34_ema.py`
  - EMA variant config (typically best benchmark numbers).

- `configs/clrernet/culane/dataset_culane_clrernet.py`
  - Dataset-specific pipeline and dataloader/eval setup for CULane.

---

## 3) `dataset/` (Local data mount point / prepared data)

```text
dataset/
└── culane/
    ├── driver_100_30frame/
    ├── driver_193_90frame/
    ├── driver_37_30frame/
    └── list/
```

- `dataset/culane/...`
  - Expected location for CULane raw sequences and list files.
  - Data is not fully tracked in git; this folder is where users place downloaded dataset files.

- `dataset/culane/list/`
  - Holds split/list metadata (`train.txt`, `test.txt`, etc. depending on preparation) and frame-diff files used by training.

---

## 4) `demo/` (Quick inference examples)

```text
demo/
├── demo.jpg
├── image_demo.py
└── result.png
```

- `demo/image_demo.py`
  - Single-image inference entry script.
  - Typical usage: load config + checkpoint, run lane prediction, save visualization.

- `demo/demo.jpg`
  - Sample input image.

- `demo/result.png`
  - Sample rendered detection output from demo script.

---

## 5) `docker/` and container setup

```text
docker/
└── Dockerfile
```

- `docker/Dockerfile`
  - Builds the recommended runtime/dev environment for reproducibility.

- `docker-compose.yaml` (root)
  - One-command compose orchestration for the Dockerized workflow from README/docs.

---

## 6) `docs/` (Project docs)

```text
docs/
├── DATASETS.md
├── INSTALL.md
└── figures/
    ├── clrernet.jpg
    └── laneiou.jpg
```

- `docs/INSTALL.md`
  - Installation guidance and practical environment tips.

- `docs/DATASETS.md`
  - Dataset preparation instructions (where to place files, expected list structure, etc.).

- `docs/figures/*.jpg`
  - Method illustrations used in README/docs.

---

## 7) `libs/` (Main source code)

This is the core implementation.

```text
libs/
├── api/
│   └── inference.py
├── core/
│   ├── anchor/
│   ├── bbox/
│   └── hook/
├── datasets/
│   ├── __init__.py
│   ├── culane_dataset.py
│   ├── metrics/
│   └── pipelines/
├── models/
│   ├── __init__.py
│   ├── backbones/
│   ├── dense_heads/
│   ├── detectors/
│   ├── layers/
│   ├── losses/
│   └── necks/
└── utils/
    ├── lane_utils.py
    └── visualizer.py
```

### 7.1 `libs/api/`

- `libs/api/inference.py`
  - Inference helper API layer.
  - Usually wraps model init + forward + postprocess calls for scripts/tools.

### 7.2 `libs/core/`

- `libs/core/anchor/anchor_generator.py`
  - Generates lane anchors / priors used by detection head.

- `libs/core/bbox/assigners/dynamic_topk_assigner.py`
  - Prediction-to-GT assignment logic using dynamic top-k policy.

- `libs/core/bbox/match_costs/match_cost.py`
  - Matching costs used in assignment; includes LaneIoU-related cost computation (core novelty area).

- `libs/core/hook/logger.py`
  - Custom logging hook integration for training runtime.

### 7.3 `libs/datasets/`

- `libs/datasets/culane_dataset.py`
  - Dataset class for CULane: sample loading, annotation parsing, and sample formatting.

- `libs/datasets/metrics/culane_metric.py`
  - Evaluation logic for CULane metrics (precision/recall/F1 style reporting under the benchmark protocol).

- `libs/datasets/pipelines/compose.py`
  - Transform composition utility for dataset pipeline.

- `libs/datasets/pipelines/alaug.py`
  - Data augmentation steps (albumentations-style pipeline glue).

- `libs/datasets/pipelines/lane_formatting.py`
  - Converts raw lane annotations into model-ready tensor/target formats.

### 7.4 `libs/models/`

- `libs/models/backbones/dla.py`
  - DLA backbone implementation used by config variants.

- `libs/models/necks/clrernet_fpn.py`
  - FPN-like neck for multi-scale feature aggregation.

- `libs/models/dense_heads/clrernet_head.py`
  - Main lane detection head (classification/regression outputs).

- `libs/models/dense_heads/seg_decoder.py`
  - Segmentation decoder branch (auxiliary task supervision).

- `libs/models/detectors/clrernet.py`
  - Top-level detector module wiring backbone/neck/head/training/inference flow.

- `libs/models/losses/iou_loss.py`
  - LaneIoU loss implementation (the paper’s key method component).

- `libs/models/losses/focal_loss.py`
  - Focal loss for confidence/classification behavior.

- `libs/models/losses/seg_loss.py`
  - Segmentation-related loss term.

- `libs/models/layers/nms/`
  - Custom NMS extension (includes CUDA/C++ source and build scripts) for lane proposal filtering.

### 7.5 `libs/utils/`

- `libs/utils/lane_utils.py`
  - Lane geometry helper utilities and conversions.

- `libs/utils/visualizer.py`
  - Rendering utility to draw detected lanes on images.

---

## 8) `tools/` (Operational scripts)

```text
tools/
├── calculate_frame_diff.py
├── speed_test.py
├── test.py
└── train.py
```

- `tools/train.py`
  - Main training entrypoint.
  - Expects a config path (`python tools/train.py <config.py>`).

- `tools/test.py`
  - Evaluation/inference-on-dataset script using config + checkpoint.

- `tools/speed_test.py`
  - Throughput/FPS benchmark script.

- `tools/calculate_frame_diff.py`
  - Preprocessing utility to compute frame difference values used to filter redundant adjacent frames during training.

---

## 9) `tests/` (Unit/integration checks)

```text
tests/
├── data/
│   ├── culane_dummy_list.txt
│   └── test_iou_data.npz
├── test_core/
│   └── test_match_costs.py
├── test_datasets/
│   └── test_culane_dataset.py
└── test_models/
    ├── test_forward.py
    └── test_losses.py
```

- `tests/test_core/test_match_costs.py`
  - Validates matching cost computations (important for assignment correctness).

- `tests/test_datasets/test_culane_dataset.py`
  - Ensures dataset parsing/pipeline outputs are correct.

- `tests/test_models/test_forward.py`
  - Sanity checks for model forward behavior.

- `tests/test_models/test_losses.py`
  - Loss computation checks.

- `tests/data/*`
  - Small fixtures for deterministic test execution.

---

## 10) `work_dirs/` (Experiment outputs)

```text
work_dirs/
└── clrernet_culane_dla34_ema/
    ├── 20260320_032201/
    └── clrernet_culane_dla34_ema.py
```

- `work_dirs/<experiment_name>/`
  - Training/eval run artifacts:
  - logs, JSON metric summaries, config snapshots, intermediate outputs.

- `work_dirs/.../clrernet_culane_dla34_ema.py`
  - Resolved/copied config used in that run.

This is the first place to inspect for what actually happened in a specific experiment.

---

## 11) Key workflow entry points (quick lookup)

- Train model:
  - `tools/train.py`

- Evaluate on dataset:
  - `tools/test.py`

- Single-image inference:
  - `demo/image_demo.py`

- Speed benchmark:
  - `tools/speed_test.py`

- Frame-diff preprocessing:
  - `tools/calculate_frame_diff.py`

- Core model class:
  - `libs/models/detectors/clrernet.py`

- Core method novelty:
  - `libs/models/losses/iou_loss.py`
  - `libs/core/bbox/match_costs/match_cost.py`

---

## 12) Suggested agent briefing (copy/paste)

If you are giving this repo to another agent, this concise prompt works well:

> This is CLRerNet (lane detection, WACV 2024) built on MMDetection 3.3-style configs.  
> Start with `README.md`, then inspect `configs/clrernet/culane/` for experiment definitions.  
> Core implementation lives in `libs/`: model graph in `libs/models/`, assignment/cost logic in `libs/core/`, data loading in `libs/datasets/`, inference helper in `libs/api/inference.py`, and visualization in `libs/utils/visualizer.py`.  
> Use `tools/train.py`, `tools/test.py`, and `demo/image_demo.py` for train/eval/demo workflows.  
> Validate changes with tests in `tests/` and inspect run outputs under `work_dirs/`.

---

## 13) Notes

- Some folders (e.g., `dataset/culane/*`) can be large and partially local-only depending on setup.
- `__pycache__` and build artifacts may appear in source subfolders during execution.
- The repo includes a pretrained checkpoint at root for quick demo/eval start.

