#!/usr/bin/env python3
"""
CLRerNet Test-Time Augmentation (Horizontal Flip) Evaluation
=============================================================
The paper explicitly states: "At the test time only the crop and resize are
adopted and no test-time augmentations are applied."
This script exploits that gap.

Method:
  For each test image:
    1. Run the EMA model on the ORIGINAL image  → preds_orig
    2. Horizontally flip the image, run the SAME model → preds_flip
    3. Un-flip the x-coordinates of preds_flip back to original space:
           x_unflipped_norm = 1.0 - x_flipped_norm
    4. Merge preds_orig + preds_unflipped via distance-based NMS:
       - Keep all original predictions (base set)
       - Add a flipped prediction ONLY if its mean horizontal distance
         to every existing lane exceeds the threshold (not a duplicate)

Why this works better than the ensemble approach:
  - SAME model quality for both passes (no weaker-model FP injection)
  - The flip genuinely changes which lanes are easy vs hard to detect
    (e.g. a right-side lane in the original = left-side lane in the flip,
     which may be detected with higher confidence from that perspective)
  - Particularly effective for Curve, Night, and asymmetric Crowd scenes

Usage (inside Docker container, from /work):
    python tools/test_tta.py \
        configs/clrernet/culane/clrernet_culane_dla34_ema.py \
        clrernet_culane_dla34_ema.pth

Optional flags:
    --data-root        dataset/culane
    --data-list        dataset/culane/list/test.txt
    --dist-threshold   30.0   (pixel distance for duplicate NMS)
    --device           cuda:0
"""

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from mmengine.logging import MMLogger

# ── Register all custom libs ───────────────────────────────────────────────────
import libs.models       # noqa: F401
import libs.datasets     # noqa: F401
import libs.core.bbox    # noqa: F401
import libs.core.anchor  # noqa: F401
import libs.core.hook    # noqa: F401

from mmdet.apis import init_detector
from libs.datasets.pipelines import Compose
from libs.datasets.metrics.culane_metric import interp, eval_predictions


# ── Single-image inference (accepts pre-loaded image array) ───────────────────

def run_inference_on_array(model, img: np.ndarray, img_path: str):
    """
    Run CLRerNet inference on a pre-loaded (possibly transformed) image array.
    Mirrors inference_one_image but accepts img array directly so we can
    pass the flipped image without saving to disk.

    Args:
        model:    Loaded CLRerNet model.
        img:      BGR image array (H, W, 3).
        img_path: Original image path (used only for pipeline metadata).

    Returns:
        preds: List of (x, y) pixel-coordinate tuples per lane.
    """
    ori_shape = img.shape
    data = dict(
        filename=img_path,
        sub_img_name=None,
        img=img,
        gt_points=[],
        id_classes=[],
        id_instances=[],
        img_shape=ori_shape,
        ori_shape=ori_shape,
    )

    cfg = model.cfg
    model.bbox_head.test_cfg.as_lanes = False

    test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
    data = test_pipeline(data)
    data_ = dict(
        inputs=[data["inputs"]],
        data_samples=[data["data_samples"]],
    )

    with torch.no_grad():
        results = model.test_step(data_)

    lanes = results[0]['lanes']
    preds = get_prediction(lanes, ori_shape[0], ori_shape[1])
    return preds


def get_prediction(lanes, ori_h, ori_w):
    """Convert model lane output to pixel-coordinate tuples."""
    preds = []
    for lane in lanes:
        lane = lane.cpu().numpy()
        xs = lane[:, 0]
        ys = lane[:, 1]
        valid_mask = (xs >= 0) & (xs < 1)
        xs = xs * ori_w
        lane_xs = xs[valid_mask]
        lane_ys = ys[valid_mask] * ori_h
        lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
        pred = [(x, y) for x, y in zip(lane_xs, lane_ys)]
        interp_pred = interp(pred, n=5)
        preds.append(interp_pred)
    return preds


# ── Un-flip lane x-coordinates ─────────────────────────────────────────────────

def unflip_lanes(lanes_flipped, ori_w: int):
    """
    Mirror x-coordinates of lanes detected on a horizontally-flipped image
    back to the original image coordinate space.

    For a normalized x in [0,1]:  x_orig = 1.0 - x_flipped
    In pixel coords:               x_orig = ori_w - x_flipped

    Args:
        lanes_flipped: List of lanes as (x, y) pixel-coord tuples.
        ori_w:         Original image width in pixels.

    Returns:
        List of lanes with x-coordinates mirrored.
    """
    unflipped = []
    for lane in lanes_flipped:
        mirrored = [(ori_w - x, y) for x, y in lane]
        unflipped.append(mirrored)
    return unflipped


# ── Distance-based NMS for lane merging ────────────────────────────────────────

def mean_lane_distance(lane_a, lane_b) -> float:
    """
    Mean horizontal pixel distance between two lanes at their overlapping
    y-range. Returns inf if there is no y-overlap.
    """
    if len(lane_a) < 2 or len(lane_b) < 2:
        return float('inf')

    arr_a = np.array(lane_a, dtype=np.float32)
    arr_b = np.array(lane_b, dtype=np.float32)

    arr_a = arr_a[arr_a[:, 1].argsort()]
    arr_b = arr_b[arr_b[:, 1].argsort()]

    y_min = max(arr_a[:, 1].min(), arr_b[:, 1].min())
    y_max = min(arr_a[:, 1].max(), arr_b[:, 1].max())

    if y_min >= y_max:
        return float('inf')

    sample_ys = np.linspace(y_min, y_max, num=30)
    xs_a = np.interp(sample_ys, arr_a[:, 1], arr_a[:, 0])
    xs_b = np.interp(sample_ys, arr_b[:, 1], arr_b[:, 0])

    return float(np.mean(np.abs(xs_a - xs_b)))


def tta_nms(lanes_orig, lanes_flip_unflipped, dist_threshold: float = 30.0):
    """
    Merge original and un-flipped TTA predictions.

    Strategy:
        - Keep all original predictions (primary set).
        - Add a flipped prediction only if it is NOT a near-duplicate of
          any lane already in the merged set (distance > dist_threshold).

    The dist_threshold default (30px) matches the CULane metric lane width,
    so anything closer than the IoU-relevant width is considered duplicate.

    Args:
        lanes_orig:           Predictions from the original image.
        lanes_flip_unflipped: Predictions from the flipped image, un-mirrored.
        dist_threshold:       Pixel distance below which lanes are duplicates.

    Returns:
        merged: List of merged lane predictions.
    """
    merged = list(lanes_orig)

    for flip_lane in lanes_flip_unflipped:
        is_dup = False
        for existing in merged:
            if mean_lane_distance(flip_lane, existing) < dist_threshold:
                is_dup = True
                break
        if not is_dup:
            merged.append(flip_lane)

    return merged


# ── Prediction writer ──────────────────────────────────────────────────────────

def write_prediction(lanes, dst_path: Path) -> None:
    """Write lane list (pixel-coord tuples) to CULane txt format."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out_lines = []
    for lane in lanes:
        if len(lane) < 2:
            continue
        lane_str = " ".join(f"{x:.5f} {y:.5f}" for x, y in lane)
        out_lines.append(lane_str)
    with open(str(dst_path), 'w') as f:
        if out_lines:
            f.write("\n".join(out_lines))


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='CLRerNet TTA (Horizontal Flip) Evaluation'
    )
    parser.add_argument('config',     help='Config file (EMA recommended)')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--data-root', default='dataset/culane')
    parser.add_argument('--data-list', default='dataset/culane/list/test.txt')
    parser.add_argument(
        '--dist-threshold', type=float, default=30.0,
        help='Mean pixel distance below which two lanes are duplicates (default: 30)'
    )
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger = MMLogger.get_current_instance()

    print("=" * 60)
    print("Loading CLRerNet EMA model...")
    print("=" * 60)
    model = init_detector(args.config, args.checkpoint, args.device)
    model.eval()
    print(f"  dist_threshold = {args.dist_threshold} px (CULane metric width = 30px)\n")

    with open(args.data_list, 'r') as f:
        img_rel_paths = [line.strip().lstrip('/') for line in f if line.strip()]

    result_dir = tempfile.mkdtemp(prefix='clrernet_tta_')
    print(f"Running TTA on {len(img_rel_paths)} images...")
    print(f"Prediction txts → {result_dir}\n")

    n_added = 0   # lanes recovered by the flip pass

    for img_rel_path in tqdm(img_rel_paths, desc='TTA inference'):
        img_full_path = os.path.join(args.data_root, img_rel_path)
        ori_w = 1640   # CULane standard width (used for un-flip only)

        # ── Pass 1: original image ─────────────────────────────────────────────
        img_bgr = cv2.imread(img_full_path)
        if img_bgr is None:
            # write empty prediction and continue
            dst_path = Path(result_dir) / Path(img_rel_path).with_suffix('.lines.txt')
            write_prediction([], dst_path)
            continue

        preds_orig = run_inference_on_array(model, img_bgr, img_full_path)

        # ── Pass 2: horizontally flipped image ─────────────────────────────────
        img_flipped = cv2.flip(img_bgr, 1)           # flip code 1 = horizontal
        preds_flip  = run_inference_on_array(model, img_flipped, img_full_path)

        # ── Un-flip x-coords back to original space ────────────────────────────
        preds_flip_unflipped = unflip_lanes(preds_flip, ori_w)

        # ── Merge with NMS ─────────────────────────────────────────────────────
        n_before = len(preds_orig)
        merged   = tta_nms(preds_orig, preds_flip_unflipped, args.dist_threshold)
        n_added += len(merged) - n_before

        # ── Write prediction ───────────────────────────────────────────────────
        dst_path = Path(result_dir) / Path(img_rel_path).with_suffix('.lines.txt')
        write_prediction(merged, dst_path)

    avg_added = n_added / len(img_rel_paths)
    print(f"\nFlip pass added {n_added} lanes total ({avg_added:.3f} per image on average)")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\nEvaluating TTA predictions...")
    categories_dir = str(Path(args.data_root) / 'list' / 'test_split')

    results = eval_predictions(
        pred_dir=result_dir,
        anno_dir=args.data_root,
        list_path=args.data_list,
        categories_dir=categories_dir,
        iou_thresholds=[0.5],
        logger=logger,
    )

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TTA RESULTS  (baseline EMA static: 81.55%)")
    print("=" * 60)
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:<40s}: {val * 100:.4f}%")
        else:
            print(f"  {key:<40s}: {val}")

    print("\nKey categories vs baseline:")
    baseline = {
        'F1_test0_normal_0.5': 94.36,
        'F1_test1_crowd_0.5':  80.85,
        'F1_test2_hlight_0.5': 75.17,
        'F1_test3_shadow_0.5': 84.55,
        'F1_test4_noline_0.5': 56.75,
        'F1_test5_arrow_0.5':  90.99,
        'F1_test6_curve_0.5':  78.83,
        'F1_test8_night_0.5':  76.85,
        'F1_0.5':              81.55,
    }
    for key, base_val in baseline.items():
        if key in results:
            new_val = results[key] * 100
            delta = new_val - base_val
            arrow = "▲" if delta > 0 else "▼"
            print(f"  {key:<40s}: {new_val:.4f}%  "
                  f"({arrow}{abs(delta):.4f}% vs {base_val:.2f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()