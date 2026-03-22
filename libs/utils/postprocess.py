"""
Post-processing helpers for CLRerNet inference pipelines.

This module contains pure NumPy/Python utility functions used by the
test-time augmentation (TTA), No-Line stitching, and combined evaluation
scripts. Keeping these helpers here (separate from the tool entry-points)
enables lightweight unit testing without importing cv2, torch, or mmdet.
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# TTA / Merge helpers
# ──────────────────────────────────────────────────────────────────────────────

def mean_lane_distance(lane_a: list, lane_b: list) -> float:
    """
    Mean horizontal pixel distance between two lanes at their overlapping
    y-range.  Returns inf if there is no y-overlap.

    Args:
        lane_a, lane_b: Lists of (x, y) pixel-coordinate tuples.

    Returns:
        Mean absolute horizontal distance (float), or float('inf').
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


def unflip_lanes(lanes_flipped: list, ori_w: int) -> list:
    """
    Mirror x-coordinates of lanes detected on a horizontally-flipped image
    back to the original image coordinate space.

    For pixel coordinates: x_orig = ori_w - x_flipped

    Args:
        lanes_flipped: List of lanes, each a list of (x, y) pixel-coord tuples.
        ori_w:         Original image width in pixels.

    Returns:
        List of lanes with x-coordinates mirrored.
    """
    unflipped = []
    for lane in lanes_flipped:
        mirrored = [(ori_w - x, y) for x, y in lane]
        unflipped.append(mirrored)
    return unflipped


def tta_nms(
    lanes_orig: list,
    lanes_flip_unflipped: list,
    dist_threshold: float = 40.0,
) -> list:
    """
    Merge original and un-flipped TTA predictions via distance-based NMS.

    Strategy:
        - Keep all original predictions (primary set).
        - Add a flipped prediction only if it is NOT a near-duplicate of
          any lane already in the merged set (distance > dist_threshold).

    A higher dist_threshold classifies more flip-pass predictions as duplicates,
    admitting fewer flip lanes and thereby reducing FP injection. The default
    (40px) is higher than the original TTA value (30px) to suppress the
    Cross-category FP increase observed in prior experiments.

    Args:
        lanes_orig:           Predictions from the original image.
        lanes_flip_unflipped: Predictions from the flipped image, un-mirrored.
        dist_threshold:       Pixel distance below which lanes are duplicates.
                              Higher values admit fewer flip-pass predictions.

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


def filter_horizontal_lanes(lanes: list, min_y_extent: float = 30.0) -> list:
    """
    Remove nearly-horizontal lane predictions that are likely zebra crossings
    (CULane Cross category) rather than true lane lines.

    In the CULane coordinate system, true lane lines run roughly top-to-bottom
    (large y-extent). Zebra crossing markings run horizontally (small y-extent).
    This filter directly addresses the Cross FP increase observed in TTA
    experiments, where the flip pass detected crossing markings in flipped
    images as lane candidates.

    Args:
        lanes:         List of lanes, each a list of (x, y) pixel-coord tuples.
        min_y_extent:  Minimum vertical span (y_max - y_min) in pixels.
                       Lanes with smaller y-extent are discarded. Default: 30px.

    Returns:
        filtered: List of lanes that satisfy the y-extent criterion.
    """
    filtered = []
    for lane in lanes:
        if len(lane) < 2:
            continue
        ys = [pt[1] for pt in lane]
        if (max(ys) - min(ys)) >= min_y_extent:
            filtered.append(lane)
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Lane segment stitching helpers (No-Line improvement)
# ──────────────────────────────────────────────────────────────────────────────

def lane_direction_angle(lane: list) -> float:
    """
    Compute the direction angle of a lane (degrees from vertical).

    Fits a line x = a*y + b and returns arctan(a) in degrees.
    A perfectly vertical lane returns 0°; a 45° lane returns ±45°.

    Args:
        lane: List of (x, y) pixel-coordinate tuples.

    Returns:
        angle (float): Direction angle in degrees, or float('nan') if fit fails.
    """
    if len(lane) < 2:
        return float('nan')
    arr = np.array(lane, dtype=np.float32)
    ys = arr[:, 1]
    xs = arr[:, 0]
    if ys.max() == ys.min():
        return float('nan')
    coeffs = np.polyfit(ys, xs, 1)   # x = coeffs[0]*y + coeffs[1]
    return float(np.degrees(np.arctan(coeffs[0])))


def lane_y_range(lane: list) -> tuple[float, float]:
    """Return (y_min, y_max) for a lane."""
    arr = np.array(lane, dtype=np.float32)
    return float(arr[:, 1].min()), float(arr[:, 1].max())


def y_overlap_ratio(lane_a: list, lane_b: list) -> float:
    """
    Compute the ratio of y-range overlap to the shorter lane's y-span.

    Returns 0.0 if there is no overlap, 1.0 if one range fully contains the
    other.  Used to distinguish consecutive segments (low overlap) from
    parallel co-located lanes (high overlap).

    Args:
        lane_a, lane_b: Lists of (x, y) pixel-coordinate tuples.

    Returns:
        Overlap ratio in [0, 1].
    """
    a_min, a_max = lane_y_range(lane_a)
    b_min, b_max = lane_y_range(lane_b)
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    shorter_span = min(a_max - a_min, b_max - b_min)
    if shorter_span <= 0:
        return 0.0
    return overlap / shorter_span


def min_endpoint_distance(lane_a: list, lane_b: list) -> float:
    """
    Minimum Euclidean distance between any endpoint pair of two lanes.

    Considers both (start, end) of each lane → 4 candidate pairs.

    Args:
        lane_a, lane_b: Lists of (x, y) pixel-coordinate tuples.

    Returns:
        Minimum endpoint distance in pixels.
    """
    arr_a = np.array(lane_a, dtype=np.float32)
    arr_b = np.array(lane_b, dtype=np.float32)
    endpoints_a = [arr_a[0], arr_a[-1]]
    endpoints_b = [arr_b[0], arr_b[-1]]
    min_d = float('inf')
    for pa in endpoints_a:
        for pb in endpoints_b:
            d = float(np.linalg.norm(pa - pb))
            if d < min_d:
                min_d = d
    return min_d


def stitch_two_lanes(lane_a: list, lane_b: list) -> list:
    """
    Merge two lane segments into a single degree-2 polynomial lane.

    Combines all points from both lanes, fits a degree-2 polynomial
    x = f(y), and resamples at 50 evenly-spaced y values spanning the
    full y-range of the merged pair.

    Args:
        lane_a, lane_b: Lists of (x, y) pixel-coord tuples.

    Returns:
        merged: List of (x, y) tuples representing the stitched lane.
    """
    combined = list(lane_a) + list(lane_b)
    arr = np.array(combined, dtype=np.float32)
    ys = arr[:, 1]
    xs = arr[:, 0]
    degree = min(2, len(arr) - 1)
    coeffs = np.polyfit(ys, xs, degree)
    ys_new = np.linspace(ys.min(), ys.max(), num=50)
    xs_new = np.polyval(coeffs, ys_new)
    return [(float(x), float(y)) for x, y in zip(xs_new, ys_new)]


def stitch_lane_segments(
    lanes: list,
    max_angle_diff: float = 10.0,
    max_endpoint_dist: float = 80.0,
    min_y_overlap_ratio: float = 0.3,
) -> list:
    """
    Iteratively stitch compatible lane fragments into complete lanes.

    Two lanes are stitched if they satisfy all three criteria:
      1. Similar direction: |angle_a - angle_b| < max_angle_diff (degrees)
      2. Close endpoints: min_endpoint_distance < max_endpoint_dist (pixels)
      3. Not significantly y-overlapping: y_overlap_ratio < min_y_overlap_ratio
         (prevents merging two parallel lanes running alongside each other)

    The algorithm runs iteratively until no more pairs can be stitched.

    This approach specifically targets the CULane No-Line category (F1=56.75%)
    where the model often detects *fragments* of faint lanes rather than
    complete lane lines. By stitching consecutive fragments, their combined
    coverage qualifies as a true positive under the CULane metric.

    Args:
        lanes:               Input lane list, each entry a list of (x,y) tuples.
        max_angle_diff:      Maximum angle difference in degrees for stitching.
        max_endpoint_dist:   Maximum endpoint proximity in pixels for stitching.
        min_y_overlap_ratio: Maximum y-overlap fraction to allow stitching.

    Returns:
        stitched: List of lane predictions after segment stitching.
    """
    if len(lanes) <= 1:
        return list(lanes)

    current = list(lanes)
    current_angles = [lane_direction_angle(l) for l in current]

    changed = True
    while changed:
        changed = False
        used = [False] * len(current)
        next_lanes = []
        next_angles = []

        for i in range(len(current)):
            if used[i]:
                continue
            lane_i = current[i]
            angle_i = current_angles[i]

            for j in range(i + 1, len(current)):
                if used[j]:
                    continue
                angle_j = current_angles[j]

                # Gate 1: similar direction
                if np.isnan(angle_i) or np.isnan(angle_j):
                    continue
                if abs(angle_i - angle_j) > max_angle_diff:
                    continue

                # Gate 2: endpoints close enough
                if min_endpoint_distance(lane_i, current[j]) > max_endpoint_dist:
                    continue

                # Gate 3: not parallel / co-located (low y-overlap)
                if y_overlap_ratio(lane_i, current[j]) > min_y_overlap_ratio:
                    continue

                # All gates passed → stitch
                merged_lane = stitch_two_lanes(lane_i, current[j])
                lane_i = merged_lane
                angle_i = lane_direction_angle(lane_i)
                used[j] = True
                changed = True

            next_lanes.append(lane_i)
            next_angles.append(angle_i)
            used[i] = True

        current = next_lanes
        current_angles = next_angles

    return current
