#!/usr/bin/env python3
"""
CLRerNet Trajectory Ensemble Evaluation
========================================
Loads two model checkpoints (standard 15-epoch + EMA 60-epoch), runs both on
every CULane test image, merges predictions using coordinate-based NMS, then
evaluates with the standard CULane metric.

Novelty: "Trajectory Ensemble" — two checkpoints of the same architecture
trained along different optimisation trajectories (standard vs. EMA) produce
complementary errors. Merging them reduces False Negatives without retraining.

Usage (inside Docker container, from /work):
    python tools/test_ensemble.py \
        configs/clrernet/culane/clrernet_culane_dla34.py \
        clrernet_culane_dla34.pth \
        configs/clrernet/culane/clrernet_culane_dla34_ema.py \
        clrernet_culane_dla34_ema.pth

Optional flags:
    --data-root       dataset/culane          (default)
    --data-list       dataset/culane/list/test.txt  (default)
    --dist-threshold  20.0                    pixels; lanes closer than this are duplicates
    --device          cuda:0                  (default)
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from mmengine.config import Config
from mmengine.logging import MMLogger

# ── Register all custom libs before any mmdet model is built ──────────────────
import libs.models       # noqa: F401  registers CLRerNet, CLRerNetHead, DLA, etc.
import libs.datasets     # noqa: F401  registers CULaneDataset, CULaneMetric, pipelines
import libs.core.bbox    # noqa: F401  registers assigners + match costs
import libs.core.anchor  # noqa: F401  registers anchor generator
import libs.core.hook    # noqa: F401  registers custom hooks

from mmdet.apis import init_detector
from libs.api.inference import inference_one_image
from libs.datasets.metrics.culane_metric import eval_predictions


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
    """Load a CLRerNet model from config + checkpoint."""
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


# ── Lane-level NMS helpers ─────────────────────────────────────────────────────

def mean_lane_distance(lane_a, lane_b) -> float:
    """
    Mean horizontal pixel distance between two lanes sampled at their
    overlapping y-range.  Returns inf if there is no y-overlap.

    Args:
        lane_a, lane_b: list of (x, y) tuples in pixel coordinates.
    """
    if len(lane_a) < 2 or len(lane_b) < 2:
        return float('inf')

    arr_a = np.array(lane_a, dtype=np.float32)   # (N, 2)  col0=x  col1=y
    arr_b = np.array(lane_b, dtype=np.float32)

    # Sort by y so np.interp works correctly
    arr_a = arr_a[arr_a[:, 1].argsort()]
    arr_b = arr_b[arr_b[:, 1].argsort()]

    # Find overlapping y range
    y_min = max(arr_a[:, 1].min(), arr_b[:, 1].min())
    y_max = min(arr_a[:, 1].max(), arr_b[:, 1].max())

    if y_min >= y_max:
        return float('inf')

    # Sample 30 evenly-spaced y values in the overlap
    sample_ys = np.linspace(y_min, y_max, num=30)

    xs_a = np.interp(sample_ys, arr_a[:, 1], arr_a[:, 0])
    xs_b = np.interp(sample_ys, arr_b[:, 1], arr_b[:, 0])

    return float(np.mean(np.abs(xs_a - xs_b)))


def ensemble_nms(lanes_m1, lanes_m2, dist_threshold: float = 20.0):
    """
    Merge predictions from two models using greedy distance-based NMS.

    Strategy:
        - Start with all lanes from model 2 (EMA, higher baseline quality).
        - For each lane from model 1 (standard), add it only if it is NOT
          a near-duplicate of any lane already in the merged set.

    This means the EMA model's lanes always take priority; the standard model
    contributes lanes that the EMA model missed (reducing False Negatives).

    Args:
        lanes_m1: list of (x,y)-tuple lanes from model 1 (standard).
        lanes_m2: list of (x,y)-tuple lanes from model 2 (EMA).
        dist_threshold: pixel distance below which two lanes are duplicates.

    Returns:
        merged: deduplicated list of (x,y)-tuple lanes.
    """
    merged = list(lanes_m2)   # EMA predictions are the base set

    for lane1 in lanes_m1:
        is_dup = False
        for lane_existing in merged:
            if mean_lane_distance(lane1, lane_existing) < dist_threshold:
                is_dup = True
                break
        if not is_dup:
            merged.append(lane1)

    return merged


# ── Prediction writer ──────────────────────────────────────────────────────────

def write_prediction(lanes, dst_path: Path) -> None:
    """Write lane predictions to a CULane-format txt file (x y x y ...)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for lane in lanes:
        if len(lane) < 2:
            continue
        lane_str = " ".join(f"{x:.5f} {y:.5f}" for x, y in lane)
        lines.append(lane_str)
    with open(str(dst_path), 'w') as f:
        if lines:
            f.write("\n".join(lines))


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='CLRerNet Trajectory Ensemble Evaluation'
    )
    parser.add_argument('config1',      help='Config file for model 1 (standard)')
    parser.add_argument('checkpoint1',  help='Checkpoint for model 1')
    parser.add_argument('config2',      help='Config file for model 2 (EMA)')
    parser.add_argument('checkpoint2',  help='Checkpoint for model 2')
    parser.add_argument(
        '--data-root', default='dataset/culane',
        help='CULane dataset root directory'
    )
    parser.add_argument(
        '--data-list', default='dataset/culane/list/test.txt',
        help='Test split list file'
    )
    parser.add_argument(
        '--dist-threshold', type=float, default=20.0,
        help='Mean horizontal pixel distance threshold for duplicate NMS (default: 20)'
    )
    parser.add_argument(
        '--device', default='cuda:0',
        help='Inference device (default: cuda:0)'
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger = MMLogger.get_current_instance()

    # ── Load both models ───────────────────────────────────────────────────────
    print("=" * 60)
    print("[1/4] Loading model 1 — standard CLRerNet (15-epoch)")
    print("=" * 60)
    model1 = load_model(args.config1, args.checkpoint1, args.device)

    print("=" * 60)
    print("[2/4] Loading model 2 — CLRerNet EMA (60-epoch)")
    print("=" * 60)
    model2 = load_model(args.config2, args.checkpoint2, args.device)

    # ── Read test image list ───────────────────────────────────────────────────
    with open(args.data_list, 'r') as f:
        img_rel_paths = [line.strip().lstrip('/') for line in f if line.strip()]

    print(f"\n[3/4] Running ensemble inference on {len(img_rel_paths)} images...")
    print(f"      dist_threshold = {args.dist_threshold} px\n")

    result_dir = tempfile.mkdtemp(prefix='clrernet_ensemble_')
    print(f"      Prediction txts will be written to: {result_dir}\n")

    for img_rel_path in tqdm(img_rel_paths, desc='Ensemble inference'):
        img_full_path = os.path.join(args.data_root, img_rel_path)

        # Run both models (inference_one_image handles pipeline + forward)
        _, preds1 = inference_one_image(model1, img_full_path)
        _, preds2 = inference_one_image(model2, img_full_path)

        # Merge with coordinate-based NMS
        merged = ensemble_nms(preds1, preds2, dist_threshold=args.dist_threshold)

        # Write to txt
        dst_path = Path(result_dir) / Path(img_rel_path).with_suffix('.lines.txt')
        write_prediction(merged, dst_path)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating merged predictions...")
    categories_dir = str(Path(args.data_root) / 'list' / 'test_split')

    results = eval_predictions(
        pred_dir=result_dir,
        anno_dir=args.data_root,
        list_path=args.data_list,
        categories_dir=categories_dir,
        iou_thresholds=[0.5],    # match paper's primary metric
        logger=logger,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS  (compare with single-model baseline: 81.55%)")
    print("=" * 60)
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:<35s}: {val * 100:.4f}%")
        else:
            print(f"  {key:<35s}: {val}")
    print("=" * 60)


if __name__ == '__main__':
    main()