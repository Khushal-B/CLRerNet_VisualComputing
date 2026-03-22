#!/usr/bin/env python3
"""
CLRerNet Adaptive Confidence Thresholding Evaluation
=====================================================
Replaces the paper's static confidence threshold (0.43) with a per-image
dynamic threshold derived from the image's mean brightness.

Motivation (from the paper itself):
  "The F1 metric employed in the lane detection evaluation is utterly
   sensitive to the detector's lane confidence threshold."
  The paper uses a single threshold tuned via cross-validation on the
  *average* across all scene types. This is suboptimal: dark Night and
  Dazzle images produce weaker activations, so their faint-but-correct
  lanes get killed by a threshold tuned for bright Normal scenes.

Method:
  1. Compute grayscale mean brightness of each input image (< 1ms, CPU).
  2. Map brightness → threshold via a piecewise linear function:
       brightness < LOW_BRIGHT  → threshold = LOW_THRESH  (dark:  relax)
       brightness > HIGH_BRIGHT → threshold = HIGH_THRESH (bright: tighten)
       in-between               → linear interpolation
  3. Patch model.bbox_head.test_cfg.conf_threshold before each forward pass.
  4. Restore original threshold after (clean, no side-effects).

Usage (inside Docker container, from /work):
    python tools/test_adaptive.py \
        configs/clrernet/culane/clrernet_culane_dla34_ema.py \
        clrernet_culane_dla34_ema.pth

Optional flags:
    --data-root     dataset/culane
    --data-list     dataset/culane/list/test.txt
    --low-bright    50      mean pixel value below which threshold is relaxed
    --high-bright   120     mean pixel value above which threshold is tightened
    --low-thresh    0.37    threshold for dark images  (Night, Dazzle)
    --high-thresh   0.46    threshold for bright images (Normal, Arrow)
    --base-thresh   0.43    threshold for mid-range images (paper default)
    --device        cuda:0
"""

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from mmengine.config import Config
from mmengine.logging import MMLogger

# ── Register all custom libs ───────────────────────────────────────────────────
import libs.models       # noqa: F401
import libs.datasets     # noqa: F401
import libs.core.bbox    # noqa: F401
import libs.core.anchor  # noqa: F401
import libs.core.hook    # noqa: F401

from mmdet.apis import init_detector
from libs.api.inference import inference_one_image
from libs.datasets.metrics.culane_metric import eval_predictions


# ── Brightness → threshold mapping ────────────────────────────────────────────

def compute_adaptive_threshold(
    img_bgr: np.ndarray,
    low_bright: float,
    high_bright: float,
    low_thresh: float,
    high_thresh: float,
    base_thresh: float,
) -> float:
    """
    Map image brightness to a confidence threshold.

    The mapping is piecewise linear:
        mean < low_bright   →  low_thresh   (dark scene: relax to catch faint lanes)
        mean > high_bright  →  high_thresh  (bright scene: tighten to reduce FP)
        in-between          →  linear interpolation through base_thresh

    Args:
        img_bgr:      Raw BGR image (any resolution).
        low_bright:   Brightness lower bound.
        high_bright:  Brightness upper bound.
        low_thresh:   Threshold for dark images.
        high_thresh:  Threshold for bright images.
        base_thresh:  Threshold at mid-point (paper default = 0.43).

    Returns:
        threshold (float): Per-image confidence threshold.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())

    if mean_brightness <= low_bright:
        return low_thresh
    elif mean_brightness >= high_bright:
        return high_thresh
    else:
        # piecewise: [low_bright, mid_bright] → [low_thresh, base_thresh]
        #            [mid_bright, high_bright] → [base_thresh, high_thresh]
        mid_bright = (low_bright + high_bright) / 2.0
        if mean_brightness <= mid_bright:
            t = (mean_brightness - low_bright) / (mid_bright - low_bright)
            return low_thresh + t * (base_thresh - low_thresh)
        else:
            t = (mean_brightness - mid_bright) / (high_bright - mid_bright)
            return base_thresh + t * (high_thresh - base_thresh)


# ── Prediction writer ──────────────────────────────────────────────────────────

def write_prediction(lanes, dst_path: Path, ori_h: int = 590, ori_w: int = 1640,
                     y_step: int = 2) -> None:
    """
    Write lane predictions to a CULane-format txt file.
    Accepts Lane objects (from culane_metric.py pipeline) directly.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ys = np.arange(0, ori_h, y_step) / ori_h
    out_lines = []
    for lane in lanes:
        xs = lane(ys)                          # Lane.__call__ interpolates at ys
        valid = (xs >= 0) & (xs < 1)
        xs_px = xs[valid] * ori_w
        ys_px = ys[valid] * ori_h
        xs_px, ys_px = xs_px[::-1], ys_px[::-1]
        if len(xs_px) < 2:
            continue
        lane_str = " ".join(f"{x:.5f} {y:.5f}" for x, y in zip(xs_px, ys_px))
        out_lines.append(lane_str)
    with open(str(dst_path), 'w') as f:
        if out_lines:
            f.write("\n".join(out_lines))


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='CLRerNet Adaptive Threshold Evaluation'
    )
    parser.add_argument('config',     help='Config file (EMA recommended)')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--data-root', default='dataset/culane')
    parser.add_argument(
        '--data-list', default='dataset/culane/list/test.txt')
    # Brightness knobs
    parser.add_argument('--low-bright',  type=float, default=50.0,
        help='Brightness below which threshold is LOW_THRESH (default: 50)')
    parser.add_argument('--high-bright', type=float, default=120.0,
        help='Brightness above which threshold is HIGH_THRESH (default: 120)')
    # Threshold knobs
    parser.add_argument('--low-thresh',  type=float, default=0.37,
        help='Threshold for dark images, e.g. Night (default: 0.37)')
    parser.add_argument('--high-thresh', type=float, default=0.46,
        help='Threshold for bright images, e.g. Normal (default: 0.46)')
    parser.add_argument('--base-thresh', type=float, default=0.43,
        help='Threshold for mid-brightness images (paper default: 0.43)')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger = MMLogger.get_current_instance()

    # ── Load model ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading CLRerNet EMA model...")
    print("=" * 60)
    model = init_detector(args.config, args.checkpoint, args.device)
    model.eval()

    original_threshold = model.bbox_head.test_cfg.conf_threshold
    print(f"  Original static threshold : {original_threshold}")
    print(f"  Dark  image threshold     : {args.low_thresh}  (brightness < {args.low_bright})")
    print(f"  Base  threshold           : {args.base_thresh} (paper default)")
    print(f"  Bright image threshold    : {args.high_thresh} (brightness > {args.high_bright})")

    # ── Read test list ──────────────────────────────────────────────────────────
    with open(args.data_list, 'r') as f:
        img_rel_paths = [line.strip().lstrip('/') for line in f if line.strip()]

    result_dir = tempfile.mkdtemp(prefix='clrernet_adaptive_')
    print(f"\nRunning adaptive inference on {len(img_rel_paths)} images...")
    print(f"Prediction txts → {result_dir}\n")

    # Track threshold distribution for analysis
    thresh_log = []

    for img_rel_path in tqdm(img_rel_paths, desc='Adaptive threshold inference'):
        img_full_path = os.path.join(args.data_root, img_rel_path)

        # 1. Compute adaptive threshold from raw image brightness
        img_bgr = cv2.imread(img_full_path)
        if img_bgr is None:
            # fallback to static threshold if image missing
            adaptive_thresh = args.base_thresh
        else:
            adaptive_thresh = compute_adaptive_threshold(
                img_bgr,
                low_bright=args.low_bright,
                high_bright=args.high_bright,
                low_thresh=args.low_thresh,
                high_thresh=args.high_thresh,
                base_thresh=args.base_thresh,
            )
        thresh_log.append(adaptive_thresh)

        # 2. Patch threshold (single attribute write, no model recompile)
        model.bbox_head.test_cfg.conf_threshold = adaptive_thresh

        # 3. Run inference (same pipeline as standard test.py)
        _, preds = inference_one_image(model, img_full_path)

        # 4. Restore (clean)
        model.bbox_head.test_cfg.conf_threshold = original_threshold

        # 5. Write predictions
        dst_path = Path(result_dir) / Path(img_rel_path).with_suffix('.lines.txt')
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # preds from inference_one_image are already (x,y) pixel tuples
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        out_lines = []
        for lane in preds:
            if len(lane) < 2:
                continue
            lane_str = " ".join(f"{x:.5f} {y:.5f}" for x, y in lane)
            out_lines.append(lane_str)
        with open(str(dst_path), 'w') as f:
            if out_lines:
                f.write("\n".join(out_lines))

    # ── Threshold distribution summary ─────────────────────────────────────────
    thresh_arr = np.array(thresh_log)
    print(f"\nThreshold distribution across {len(thresh_arr)} images:")
    print(f"  min={thresh_arr.min():.4f}  max={thresh_arr.max():.4f}  "
          f"mean={thresh_arr.mean():.4f}  std={thresh_arr.std():.4f}")
    print(f"  Images using low  threshold (<{args.base_thresh}): "
          f"{(thresh_arr < args.base_thresh).sum()} "
          f"({100*(thresh_arr < args.base_thresh).mean():.1f}%)")
    print(f"  Images using high threshold (>{args.base_thresh}): "
          f"{(thresh_arr > args.base_thresh).sum()} "
          f"({100*(thresh_arr > args.base_thresh).mean():.1f}%)")

    # ── Evaluate ────────────────────────────────────────────────────────────────
    print("\nEvaluating adaptive predictions...")
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
    print("ADAPTIVE THRESHOLD RESULTS  (baseline EMA static: 81.55%)")
    print("=" * 60)
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:<40s}: {val * 100:.4f}%")
        else:
            print(f"  {key:<40s}: {val}")
    print("=" * 60)
    print("\nKey categories to watch vs baseline:")
    baseline = {
        'F1_test8_night_0.5':  76.85,
        'F1_test2_hlight_0.5': 75.17,
        'F1_test4_noline_0.5': 56.75,
        'F1_test1_crowd_0.5':  80.85,
        'F1_0.5':              81.55,
    }
    for key, base_val in baseline.items():
        if key in results:
            new_val = results[key] * 100
            delta = new_val - base_val
            arrow = "▲" if delta > 0 else "▼"
            print(f"  {key:<40s}: {new_val:.4f}%  ({arrow} {abs(delta):.4f}% vs {base_val:.2f}%)")


if __name__ == '__main__':
    main()