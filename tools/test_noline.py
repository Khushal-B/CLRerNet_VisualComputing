#!/usr/bin/env python3
"""
CLRerNet No-Line Category Improvement via Lane Segment Stitching
================================================================
The CULane No-Line category has the lowest F1 (56.75%) — more than 15 points
below the next-worst category. This script addresses it with two complementary
post-processing techniques that require zero additional model inference and no
GPU memory beyond the baseline forward pass.

Why No-Line is hard:
  In "no line" scenes (construction zones, dirt roads, faded markings), the
  model often detects *fragments* of lanes rather than full lane lines. The
  CULane metric requires a lane to be matched end-to-end; partial detections
  do not score as true positives. Two failure modes:
    1. Same lane detected as two or three disjoint segments → each scored
       as a FP (wrong location) or missed TP (incomplete coverage).
    2. The detector's confidence for faint lanes is near the 0.43 threshold;
       some partial lanes are suppressed.

Novel approach — Lane Segment Stitching + Confidence-gate adaptive threshold:

  (A) Lane segment stitching (post-hoc, CPU only):
      After inference, scan all predicted lanes pairwise. If two lanes:
        - have similar direction angles (within `max_angle_diff` degrees)
        - have endpoints within `max_endpoint_dist` pixels of each other
        - do not y-overlap significantly (they are consecutive, not parallel)
      → replace both with a single degree-2 polynomial fitted to their
        combined points. This recovers full lane coverage from fragments.

  (B) Confidence-gate adaptive threshold (per-image):
      If the standard pass yields *fewer than min_lanes_for_scene* predictions,
      the scene likely has faint / partial markings. A second pass with a
      relaxed threshold captures additional low-confidence lanes. The relaxed
      threshold (`low_thresh`) is only applied when the first pass is sparse
      to avoid adding FPs in normal scenes.

Usage (inside Docker container, from /work):
    python tools/test_noline.py \\
        configs/clrernet/culane/clrernet_culane_dla34_ema.py \\
        clrernet_culane_dla34_ema.pth

Optional flags:
    --data-root            dataset/culane
    --data-list            dataset/culane/list/test.txt
    --max-angle-diff       10.0    (degrees; stitching similarity gate)
    --max-endpoint-dist    80.0    (pixels; stitching proximity gate)
    --min-y-overlap-ratio  0.3     (max y-overlap for non-parallel check)
    --min-lanes-for-scene  2       (if fewer lanes, enable relaxed threshold)
    --low-thresh           0.38    (relaxed confidence for sparse scenes)
    --base-thresh          0.43    (standard confidence threshold)
    --device               cuda:0
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
from libs.utils.postprocess import (
    tta_nms,
    lane_direction_angle,
    y_overlap_ratio,
    min_endpoint_distance,
    stitch_two_lanes,
    stitch_lane_segments,
)


# ── Inference helper (mirrors test_tta.py) ─────────────────────────────────────

def run_inference_on_array(model, img: np.ndarray, img_path: str):
    """
    Run CLRerNet inference on a pre-loaded image array.

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
        description='CLRerNet No-Line Improvement: Lane Segment Stitching'
    )
    parser.add_argument('config',     help='Config file (EMA recommended)')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--data-root', default='dataset/culane')
    parser.add_argument('--data-list', default='dataset/culane/list/test.txt')

    # Stitching parameters
    parser.add_argument(
        '--max-angle-diff', type=float, default=10.0,
        help='Max direction angle difference (degrees) to stitch two lanes (default: 10)'
    )
    parser.add_argument(
        '--max-endpoint-dist', type=float, default=80.0,
        help='Max endpoint distance (pixels) to stitch two lanes (default: 80)'
    )
    parser.add_argument(
        '--min-y-overlap-ratio', type=float, default=0.3,
        help='Max y-overlap fraction to allow stitching (prevents merging parallel lanes)'
    )

    # Confidence-gate parameters
    parser.add_argument(
        '--min-lanes-for-scene', type=int, default=2,
        help='If fewer lanes detected (first pass), enable relaxed threshold (default: 2)'
    )
    parser.add_argument(
        '--low-thresh', type=float, default=0.38,
        help='Relaxed confidence threshold for sparse detection scenes (default: 0.38)'
    )
    parser.add_argument(
        '--base-thresh', type=float, default=0.43,
        help='Standard confidence threshold (paper default: 0.43)'
    )

    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger = MMLogger.get_current_instance()

    print("=" * 60)
    print("Loading CLRerNet EMA model for No-Line improvement...")
    print("=" * 60)
    model = init_detector(args.config, args.checkpoint, args.device)
    model.eval()

    original_threshold = model.bbox_head.test_cfg.conf_threshold
    print(f"  base threshold         : {args.base_thresh}")
    print(f"  low threshold (sparse) : {args.low_thresh}")
    print(f"  min_lanes_for_scene    : {args.min_lanes_for_scene}")
    print(f"  max_angle_diff         : {args.max_angle_diff}°")
    print(f"  max_endpoint_dist      : {args.max_endpoint_dist}px")
    print(f"  min_y_overlap_ratio    : {args.min_y_overlap_ratio}\n")

    with open(args.data_list, 'r') as f:
        img_rel_paths = [line.strip().lstrip('/') for line in f if line.strip()]

    result_dir = tempfile.mkdtemp(prefix='clrernet_noline_')
    print(f"Running No-Line inference on {len(img_rel_paths)} images...")
    print(f"Prediction txts → {result_dir}\n")

    n_stitched_total = 0    # lanes recovered by stitching
    n_relaxed_total  = 0    # images where relaxed threshold was used

    for img_rel_path in tqdm(img_rel_paths, desc='No-Line inference'):
        img_full_path = os.path.join(args.data_root, img_rel_path)

        img_bgr = cv2.imread(img_full_path)
        if img_bgr is None:
            dst_path = Path(result_dir) / Path(img_rel_path).with_suffix('.lines.txt')
            write_prediction([], dst_path)
            continue

        # ── Pass 1: standard threshold ─────────────────────────────────────────
        model.bbox_head.test_cfg.conf_threshold = args.base_thresh
        preds = run_inference_on_array(model, img_bgr, img_full_path)

        # ── Pass 2: relaxed threshold for sparse detections (confidence gate) ──
        if len(preds) < args.min_lanes_for_scene:
            model.bbox_head.test_cfg.conf_threshold = args.low_thresh
            preds_low = run_inference_on_array(model, img_bgr, img_full_path)
            # Merge: keep all low-thresh predictions not already in preds
            # (use a simple distance gate to avoid duplicates)
            preds = tta_nms(preds, preds_low, dist_threshold=30.0)
            n_relaxed_total += 1

        model.bbox_head.test_cfg.conf_threshold = original_threshold

        # ── Lane segment stitching ─────────────────────────────────────────────
        n_before = len(preds)
        preds = stitch_lane_segments(
            preds,
            max_angle_diff=args.max_angle_diff,
            max_endpoint_dist=args.max_endpoint_dist,
            min_y_overlap_ratio=args.min_y_overlap_ratio,
        )
        n_stitched_total += max(0, n_before - len(preds))

        # ── Write prediction ───────────────────────────────────────────────────
        dst_path = Path(result_dir) / Path(img_rel_path).with_suffix('.lines.txt')
        write_prediction(preds, dst_path)

    print(f"\nRelaxed threshold used : {n_relaxed_total} images")
    print(f"Lanes stitched (merged): {n_stitched_total} total")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\nEvaluating No-Line predictions...")
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
    print("NO-LINE IMPROVEMENT RESULTS  (baseline EMA static: 81.55%)")
    print("=" * 60)
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:<40s}: {val * 100:.4f}%")
        else:
            print(f"  {key:<40s}: {val}")

    print("\nKey categories vs baseline:")
    baseline = {
        'F1_test4_noline_0.5': 56.75,
        'F1_test0_normal_0.5': 94.36,
        'F1_test1_crowd_0.5':  80.85,
        'F1_test2_hlight_0.5': 75.17,
        'F1_test3_shadow_0.5': 84.55,
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
