#!/usr/bin/env python3
"""
CLRerNet Combined TTA + Cross Suppression + Lane Stitching
===========================================================
This script is the "full novelty" evaluation that combines the two strongest
improvements identified through iterative experimentation:

  (1) Horizontal-flip TTA with geometric Cross FP suppression  (test_tta.py)
  (2) Lane segment stitching for faint/fragmented lane recovery (test_noline.py)

Why combining them works:
  - TTA flip recovers missed lanes in Curve, Night, and asymmetric Crowd scenes.
  - Cross suppression (y-extent filter) keeps the Cross FP count near baseline.
  - Stitching then merges the augmented lane set's fragments into complete lanes,
    further boosting recall without adding extra FPs.
  - Each component is complementary: TTA operates on the raw model pass, while
    stitching operates on the merged output.

Expected improvements over baseline (F1=81.55%):
  - Curve  : +0.96% (from TTA flip, preserved by Cross suppression)
  - Night  : +0.12% (from TTA flip, preserved by Cross suppression)
  - No-Line: potential +% (from lane stitching closing fragmented detection gaps)
  - Cross FP: maintained near baseline (from y-extent filter)
  - Overall F1: targeted > 81.55%

Usage (inside Docker container, from /work):
    python tools/test_combined.py \\
        configs/clrernet/culane/clrernet_culane_dla34_ema.py \\
        clrernet_culane_dla34_ema.pth

Optional flags:
    --data-root            dataset/culane
    --data-list            dataset/culane/list/test.txt
    --dist-threshold       40.0    (TTA NMS: pixel distance for duplicate suppression)
    --min-y-extent         30.0    (Cross suppression: minimum vertical lane span)
    --max-angle-diff       10.0    (stitching: direction similarity gate, degrees)
    --max-endpoint-dist    80.0    (stitching: endpoint proximity gate, pixels)
    --min-y-overlap-ratio  0.3     (stitching: y-overlap threshold)
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
from libs.utils.postprocess import (
    unflip_lanes,
    tta_nms,
    filter_horizontal_lanes,
    stitch_lane_segments,
)
from libs.datasets.metrics.culane_metric import eval_predictions

# Inference helpers (need torch/model, so imported from test_tta)
from tools.test_tta import run_inference_on_array, write_prediction


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='CLRerNet Combined TTA + Cross Suppression + Lane Stitching'
    )
    parser.add_argument('config',     help='Config file (EMA recommended)')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--data-root', default='dataset/culane')
    parser.add_argument('--data-list', default='dataset/culane/list/test.txt')

    # TTA / Cross suppression parameters
    parser.add_argument(
        '--dist-threshold', type=float, default=40.0,
        help='TTA NMS pixel distance threshold (default: 40). Higher values classify '
             'more flip-pass predictions as duplicates, admitting fewer and reducing FPs.'
    )
    parser.add_argument(
        '--min-y-extent', type=float, default=30.0,
        help='Cross suppression: minimum vertical span in pixels (default: 30)'
    )

    # Lane stitching parameters
    parser.add_argument(
        '--max-angle-diff', type=float, default=10.0,
        help='Stitching: max direction angle difference in degrees (default: 10)'
    )
    parser.add_argument(
        '--max-endpoint-dist', type=float, default=80.0,
        help='Stitching: max endpoint proximity in pixels (default: 80)'
    )
    parser.add_argument(
        '--min-y-overlap-ratio', type=float, default=0.3,
        help='Stitching: max y-overlap fraction to allow merge (default: 0.3)'
    )

    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger = MMLogger.get_current_instance()

    print("=" * 60)
    print("CLRerNet Combined: TTA + Cross Suppression + Lane Stitching")
    print("=" * 60)
    model = init_detector(args.config, args.checkpoint, args.device)
    model.eval()

    print(f"  [TTA] dist_threshold   = {args.dist_threshold} px")
    print(f"  [TTA] min_y_extent     = {args.min_y_extent} px  (Cross suppression)")
    print(f"  [STI] max_angle_diff   = {args.max_angle_diff}°")
    print(f"  [STI] max_endpoint_dist= {args.max_endpoint_dist} px")
    print(f"  [STI] min_y_overlap    = {args.min_y_overlap_ratio}\n")

    with open(args.data_list, 'r') as f:
        img_rel_paths = [line.strip().lstrip('/') for line in f if line.strip()]

    result_dir = tempfile.mkdtemp(prefix='clrernet_combined_')
    print(f"Running combined inference on {len(img_rel_paths)} images...")
    print(f"Prediction txts → {result_dir}\n")

    n_added_by_flip   = 0   # lanes recovered by TTA flip
    n_suppressed_cross = 0  # lanes removed by Cross suppression
    n_stitched        = 0   # lane pairs merged by stitching

    for img_rel_path in tqdm(img_rel_paths, desc='Combined inference'):
        img_full_path = os.path.join(args.data_root, img_rel_path)
        ori_w = 1640   # CULane standard width

        img_bgr = cv2.imread(img_full_path)
        if img_bgr is None:
            dst_path = Path(result_dir) / Path(img_rel_path).with_suffix('.lines.txt')
            write_prediction([], dst_path)
            continue

        # ── Stage 1a: Original image inference ────────────────────────────────
        preds_orig = run_inference_on_array(model, img_bgr, img_full_path)

        # ── Stage 1b: Flipped image inference + un-flip ───────────────────────
        img_flipped          = cv2.flip(img_bgr, 1)
        preds_flip           = run_inference_on_array(model, img_flipped, img_full_path)
        preds_flip_unflipped = unflip_lanes(preds_flip, ori_w)

        # ── Stage 2: TTA NMS merge ────────────────────────────────────────────
        n_before_flip = len(preds_orig)
        merged = tta_nms(preds_orig, preds_flip_unflipped, args.dist_threshold)
        n_added_by_flip += len(merged) - n_before_flip

        # ── Stage 3: Cross FP suppression (y-extent filter) ───────────────────
        n_before_suppress = len(merged)
        merged = filter_horizontal_lanes(merged, min_y_extent=args.min_y_extent)
        n_suppressed_cross += n_before_suppress - len(merged)

        # ── Stage 4: Lane segment stitching ───────────────────────────────────
        n_before_stitch = len(merged)
        merged = stitch_lane_segments(
            merged,
            max_angle_diff=args.max_angle_diff,
            max_endpoint_dist=args.max_endpoint_dist,
            min_y_overlap_ratio=args.min_y_overlap_ratio,
        )
        n_stitched += max(0, n_before_stitch - len(merged))

        # ── Write prediction ───────────────────────────────────────────────────
        dst_path = Path(result_dir) / Path(img_rel_path).with_suffix('.lines.txt')
        write_prediction(merged, dst_path)

    n_imgs = len(img_rel_paths)
    print(f"\nFlip TTA added    : {n_added_by_flip} lanes total "
          f"({n_added_by_flip/n_imgs:.3f} per image)")
    print(f"Cross FP removed  : {n_suppressed_cross} lanes total "
          f"({n_suppressed_cross/n_imgs:.3f} per image)")
    print(f"Segments stitched : {n_stitched} merges total "
          f"({n_stitched/n_imgs:.3f} per image)")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\nEvaluating combined predictions...")
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
    print("COMBINED RESULTS  (baseline EMA static: 81.55%)")
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
