"""
Unit tests for post-processing helper functions in libs/utils/postprocess.py.

These tests validate the core geometry/filtering logic without requiring a GPU,
pretrained weights, or the CULane dataset.
"""

import sys
import os

# Ensure project root is on the path when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

# ── Helpers being tested ───────────────────────────────────────────────────────
from libs.utils.postprocess import (
    mean_lane_distance,
    unflip_lanes,
    tta_nms,
    filter_horizontal_lanes,
    lane_direction_angle,
    y_overlap_ratio,
    min_endpoint_distance,
    stitch_two_lanes,
    stitch_lane_segments,
)


# ──────────────────────────────────────────────────────────────────────────────
# filter_horizontal_lanes
# ──────────────────────────────────────────────────────────────────────────────

class TestFilterHorizontalLanes:
    """Tests for the Cross-FP suppression filter."""

    def _make_vertical_lane(self, x=100, y_start=100, y_end=500, n=50):
        """A near-vertical lane (large y-extent)."""
        ys = np.linspace(y_start, y_end, n)
        return [(x, y) for y in ys]

    def _make_horizontal_lane(self, y=300, x_start=50, x_end=500, n=50):
        """A near-horizontal lane (small y-extent, like a zebra crossing)."""
        xs = np.linspace(x_start, x_end, n)
        return [(x, y) for x in xs]

    def test_vertical_lane_passes(self):
        """A vertical lane (y-extent > min_y_extent) should be kept."""
        lanes = [self._make_vertical_lane(y_start=100, y_end=500)]
        result = filter_horizontal_lanes(lanes, min_y_extent=30.0)
        assert len(result) == 1

    def test_horizontal_lane_suppressed(self):
        """A horizontal lane (y-extent < min_y_extent) should be removed."""
        # y_extent = 5, which is < 30 default
        lanes = [self._make_horizontal_lane(y=300)]
        result = filter_horizontal_lanes(lanes, min_y_extent=30.0)
        assert len(result) == 0

    def test_mixed_lanes(self):
        """Vertical lanes pass; horizontal lanes are suppressed."""
        lanes = [
            self._make_vertical_lane(x=200, y_start=50, y_end=580),
            self._make_horizontal_lane(y=100),   # crossing — should be removed
            self._make_vertical_lane(x=400, y_start=100, y_end=490),
        ]
        result = filter_horizontal_lanes(lanes, min_y_extent=30.0)
        assert len(result) == 2

    def test_empty_input(self):
        """Empty lane list returns empty list."""
        assert filter_horizontal_lanes([], min_y_extent=30.0) == []

    def test_single_point_lane_dropped(self):
        """Lanes with fewer than 2 points are always dropped."""
        lanes = [[(100, 200)]]
        result = filter_horizontal_lanes(lanes, min_y_extent=0.0)
        assert len(result) == 0

    def test_threshold_boundary(self):
        """Lane with y-extent exactly equal to min_y_extent should pass (>=)."""
        lane = [(100, 100), (100, 130)]   # y-extent = 30
        result = filter_horizontal_lanes([lane], min_y_extent=30.0)
        assert len(result) == 1

    def test_custom_threshold(self):
        """Custom min_y_extent is respected."""
        lane = [(100, 100), (100, 150)]   # y-extent = 50
        # Should pass threshold of 30 but fail threshold of 60
        assert len(filter_horizontal_lanes([lane], min_y_extent=30.0)) == 1
        assert len(filter_horizontal_lanes([lane], min_y_extent=60.0)) == 0


# ──────────────────────────────────────────────────────────────────────────────
# mean_lane_distance
# ──────────────────────────────────────────────────────────────────────────────

class TestMeanLaneDistance:
    """Tests for the TTA duplicate-detection distance metric."""

    def test_identical_lanes_zero_distance(self):
        """Distance between a lane and itself should be zero."""
        lane = [(x, y) for x, y in zip(range(10, 100, 10), range(100, 500, 40))]
        assert mean_lane_distance(lane, lane) == pytest.approx(0.0, abs=1e-4)

    def test_parallel_lanes_known_distance(self):
        """Two perfectly parallel vertical lines 50px apart should return 50."""
        lane_a = [(100, float(y)) for y in range(100, 500, 10)]
        lane_b = [(150, float(y)) for y in range(100, 500, 10)]
        dist = mean_lane_distance(lane_a, lane_b)
        assert dist == pytest.approx(50.0, abs=1.0)

    def test_no_y_overlap_returns_inf(self):
        """Lanes with no y-overlap return infinity."""
        lane_a = [(100, float(y)) for y in range(100, 200)]
        lane_b = [(100, float(y)) for y in range(300, 400)]
        assert mean_lane_distance(lane_a, lane_b) == float('inf')

    def test_short_lane_returns_inf(self):
        """Lanes with fewer than 2 points return infinity."""
        assert mean_lane_distance([(100, 200)], [(200, 300), (200, 400)]) == float('inf')
        assert mean_lane_distance([], [(200, 300), (200, 400)]) == float('inf')


# ──────────────────────────────────────────────────────────────────────────────
# unflip_lanes
# ──────────────────────────────────────────────────────────────────────────────

class TestUnflipLanes:
    """Tests for horizontal-flip coordinate inversion."""

    def test_unflip_symmetry(self):
        """Flipping twice should return the original coordinates."""
        ori_w = 1640
        lane = [(200.0, 100.0), (300.0, 200.0), (400.0, 300.0)]
        double_unflipped = unflip_lanes(unflip_lanes([lane], ori_w), ori_w)
        for (x_orig, y_orig), (x_back, y_back) in zip(lane, double_unflipped[0]):
            assert x_orig == pytest.approx(x_back, abs=1e-6)
            assert y_orig == pytest.approx(y_back, abs=1e-6)

    def test_centre_lane_unchanged(self):
        """A lane at x = ori_w/2 should be unchanged after un-flip."""
        ori_w = 1640
        lane = [(820.0, float(y)) for y in range(100, 500, 50)]
        result = unflip_lanes([lane], ori_w)
        for (x_orig, _), (x_res, _) in zip(lane, result[0]):
            assert x_orig == pytest.approx(x_res, abs=1e-6)

    def test_y_coordinates_unchanged(self):
        """Un-flipping should not affect y-coordinates."""
        ori_w = 1640
        lane = [(200.0, 100.0), (500.0, 300.0), (700.0, 500.0)]
        result = unflip_lanes([lane], ori_w)
        for (_, y_orig), (_, y_res) in zip(lane, result[0]):
            assert y_orig == pytest.approx(y_res, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# tta_nms
# ──────────────────────────────────────────────────────────────────────────────

class TestTtaNms:
    """Tests for the TTA merge with distance-based NMS."""

    def _make_lane(self, x_offset=0, y_start=100, y_end=500, n=40):
        ys = np.linspace(y_start, y_end, n)
        return [(x_offset + 0.1 * y, y) for y in ys]

    def test_duplicate_not_added(self):
        """A flip lane identical (≈0px away) to an existing lane is not added."""
        lane = self._make_lane(x_offset=0)
        result = tta_nms([lane], [lane], dist_threshold=40.0)
        assert len(result) == 1

    def test_distant_lane_added(self):
        """A flip lane far away (>dist_threshold) from all existing is added."""
        lane_orig = self._make_lane(x_offset=0)
        lane_flip = self._make_lane(x_offset=200)   # 200px away
        result = tta_nms([lane_orig], [lane_flip], dist_threshold=40.0)
        assert len(result) == 2

    def test_orig_lanes_always_kept(self):
        """All original lanes are always in the merged set."""
        orig_lanes = [self._make_lane(x_offset=i * 150) for i in range(3)]
        result = tta_nms(orig_lanes, [], dist_threshold=40.0)
        assert len(result) == 3

    def test_empty_flip(self):
        """Empty flip list returns the original lane list unchanged."""
        orig_lanes = [self._make_lane(x_offset=100)]
        result = tta_nms(orig_lanes, [], dist_threshold=40.0)
        assert len(result) == 1

    def test_empty_orig(self):
        """Empty original list: all flip lanes are added (no existing to compare)."""
        flip_lanes = [self._make_lane(x_offset=100), self._make_lane(x_offset=300)]
        result = tta_nms([], flip_lanes, dist_threshold=40.0)
        assert len(result) == 2


# ──────────────────────────────────────────────────────────────────────────────
# lane_direction_angle
# ──────────────────────────────────────────────────────────────────────────────

class TestLaneDirectionAngle:
    """Tests for the lane direction angle helper."""

    def test_vertical_lane_zero_angle(self):
        """A perfectly vertical lane (x constant) should return ~0°."""
        lane = [(100.0, float(y)) for y in range(100, 500, 10)]
        angle = lane_direction_angle(lane)
        assert abs(angle) < 1.0

    def test_diagonal_lane_positive_angle(self):
        """A diagonal lane (x increases as y increases) has a positive angle."""
        lane = [(float(y), float(y)) for y in range(100, 500, 10)]
        angle = lane_direction_angle(lane)
        assert angle == pytest.approx(45.0, abs=2.0)

    def test_single_point_returns_nan(self):
        """Single-point lane returns NaN (cannot fit a line)."""
        angle = lane_direction_angle([(100, 200)])
        assert np.isnan(angle)


# ──────────────────────────────────────────────────────────────────────────────
# y_overlap_ratio
# ──────────────────────────────────────────────────────────────────────────────

class TestYOverlapRatio:
    """Tests for the y-range overlap fraction helper."""

    def test_no_overlap_zero(self):
        """Non-overlapping ranges return 0."""
        lane_a = [(100, float(y)) for y in range(100, 200)]
        lane_b = [(100, float(y)) for y in range(300, 400)]
        assert y_overlap_ratio(lane_a, lane_b) == pytest.approx(0.0)

    def test_full_overlap_one(self):
        """One lane contained entirely in the other returns 1."""
        lane_a = [(100, float(y)) for y in range(100, 500)]
        lane_b = [(100, float(y)) for y in range(200, 400)]
        assert y_overlap_ratio(lane_a, lane_b) == pytest.approx(1.0)

    def test_partial_overlap(self):
        """Partial overlap returns ratio in (0, 1)."""
        lane_a = [(100, float(y)) for y in range(100, 300)]  # span 200
        lane_b = [(100, float(y)) for y in range(200, 400)]  # span 200, overlap 100
        ratio = y_overlap_ratio(lane_a, lane_b)
        assert 0.0 < ratio < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# min_endpoint_distance
# ──────────────────────────────────────────────────────────────────────────────

class TestMinEndpointDistance:
    """Tests for the endpoint proximity helper."""

    def test_touching_endpoints_zero(self):
        """Lanes sharing an endpoint return distance ≈ 0."""
        lane_a = [(100.0, 100.0), (100.0, 300.0)]
        lane_b = [(100.0, 300.0), (100.0, 500.0)]
        assert min_endpoint_distance(lane_a, lane_b) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        """Endpoints separated by exactly 50px are detected correctly."""
        lane_a = [(0.0, 0.0), (0.0, 100.0)]
        lane_b = [(0.0, 150.0), (0.0, 250.0)]
        assert min_endpoint_distance(lane_a, lane_b) == pytest.approx(50.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# stitch_lane_segments
# ──────────────────────────────────────────────────────────────────────────────

class TestStitchLaneSegments:
    """Tests for the lane segment stitching algorithm."""

    def _vertical_segment(self, x, y_start, y_end, n=20):
        ys = np.linspace(y_start, y_end, n)
        return [(float(x), float(y)) for y in ys]

    def test_collinear_consecutive_stitched(self):
        """Two consecutive collinear vertical segments should be stitched into one."""
        seg_a = self._vertical_segment(x=200, y_start=100, y_end=200)
        seg_b = self._vertical_segment(x=200, y_start=210, y_end=310)
        result = stitch_lane_segments(
            [seg_a, seg_b],
            max_angle_diff=5.0,
            max_endpoint_dist=20.0,
            min_y_overlap_ratio=0.3,
        )
        # Two segments merged into one
        assert len(result) == 1

    def test_far_apart_not_stitched(self):
        """Two segments far apart should not be stitched."""
        seg_a = self._vertical_segment(x=200, y_start=100, y_end=200)
        seg_b = self._vertical_segment(x=200, y_start=400, y_end=500)
        result = stitch_lane_segments(
            [seg_a, seg_b],
            max_angle_diff=5.0,
            max_endpoint_dist=20.0,   # endpoints 200px apart → won't stitch
            min_y_overlap_ratio=0.3,
        )
        assert len(result) == 2

    def test_parallel_lanes_not_stitched(self):
        """Two parallel lanes (high y-overlap) should NOT be stitched."""
        seg_a = self._vertical_segment(x=200, y_start=100, y_end=500)
        seg_b = self._vertical_segment(x=400, y_start=100, y_end=500)
        result = stitch_lane_segments(
            [seg_a, seg_b],
            max_angle_diff=5.0,
            max_endpoint_dist=500.0,   # endpoints close enough
            min_y_overlap_ratio=0.3,   # but y-overlap > 0.3 → won't stitch
        )
        assert len(result) == 2

    def test_different_direction_not_stitched(self):
        """Lanes pointing in very different directions should not be stitched."""
        seg_vert = self._vertical_segment(x=200, y_start=100, y_end=300)
        # Diagonal lane going 45°
        seg_diag = [(200.0 + float(i) * 10, 310.0 + float(i) * 10) for i in range(20)]
        result = stitch_lane_segments(
            [seg_vert, seg_diag],
            max_angle_diff=5.0,       # strict angle gate
            max_endpoint_dist=30.0,
            min_y_overlap_ratio=0.3,
        )
        assert len(result) == 2

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert stitch_lane_segments([]) == []

    def test_single_lane_unchanged(self):
        """Single lane returns unchanged."""
        lane = self._vertical_segment(x=200, y_start=100, y_end=400)
        result = stitch_lane_segments([lane])
        assert len(result) == 1


# ──────────────────────────────────────────────────────────────────────────────
# stitch_two_lanes
# ──────────────────────────────────────────────────────────────────────────────

class TestStitchTwoLanes:
    """Tests for the single-pair stitching primitive."""

    def test_output_spans_combined_y_range(self):
        """Stitched lane y-range covers the union of both input y-ranges."""
        seg_a = [(100.0, float(y)) for y in range(100, 200)]
        seg_b = [(100.0, float(y)) for y in range(300, 400)]
        stitched = stitch_two_lanes(seg_a, seg_b)
        ys = [pt[1] for pt in stitched]
        assert min(ys) <= 100.0
        assert max(ys) >= 399.0

    def test_output_has_points(self):
        """Stitched lane must be non-empty."""
        seg_a = [(100.0, float(y)) for y in range(100, 150)]
        seg_b = [(100.0, float(y)) for y in range(200, 250)]
        stitched = stitch_two_lanes(seg_a, seg_b)
        assert len(stitched) > 0
