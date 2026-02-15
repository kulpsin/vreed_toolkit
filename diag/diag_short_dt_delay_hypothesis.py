#!/usr/bin/env python3
"""Diagnostic: Test the delayed-timestamp hypothesis for short-dt frames.

Hypothesis: Short-dt frames (dt < 8ms) are caused by a delayed timestamp on
the PREVIOUS frame (N-1). If frame N-1's timestamp is recorded later than it
should be, then dt(N-2 -> N-1) is inflated and dt(N-1 -> N) is deflated.
The combined 2-frame span dt(N-2 -> N) should still equal ~2x the normal
frame interval.

This script tests the hypothesis across all users and videos.
"""

import sys
import os
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)
os.chdir(_project_root)

import warnings
warnings.filterwarnings("ignore")

import glob
import numpy as np
import preprocess

DATA_PATH = "dataset/04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)"
SHORT_DT_THRESHOLD = 0.008  # 8 ms in seconds
VERY_SHORT_DT_THRESHOLD = 0.004  # 4 ms in seconds


def load_all_users():
    """Load and preprocess all users. Returns dict {user_id: data_list}."""
    files = sorted(glob.glob(os.path.join(DATA_PATH, "*_EyeTracking_PreProcessed.dat")))
    all_data = {}
    for f in files:
        basename = os.path.basename(f)
        user_id = int(basename.split("_")[0])
        labels, data = preprocess.load_eye_tracking_data(f)
        preprocess.replace_missing_values_with_nan(data)
        preprocess.convert_timestamps_to_duration(data, convert_to_seconds=True)
        preprocess.fix_swapped_channel_issue(data)
        preprocess.add_empty_data(labels, data)
        all_data[user_id] = data
    return all_data


def collect_short_dt_events(all_data, threshold):
    """For every short-dt frame, collect the three-frame context.

    Returns arrays:
        dt_before  : dt(N-2 -> N-1)
        dt_short   : dt(N-1 -> N)   (the short interval)
        dt_after   : dt(N -> N+1)
        dt_normal  : median dt for the video this event came from
    """
    dt_before_list = []
    dt_short_list = []
    dt_after_list = []
    dt_normal_list = []

    for user_id, data in all_data.items():
        for vid_idx, vdata in enumerate(data):
            if vdata is None:
                continue
            timestamps = vdata[0]
            N = len(timestamps)
            if N < 4:
                continue

            dt = np.diff(timestamps)  # shape (N-1,)
            median_dt = np.median(dt)

            # Find short-dt frames: dt[i] is the interval from frame i to frame i+1
            # So the "short interval" dt[i] means frame i is N-1, frame i+1 is N
            # We need dt[i-1] (dt_before), dt[i] (dt_short), dt[i+1] (dt_after)
            # and i-1 >= 0, i+1 < len(dt)
            short_indices = np.where(dt < threshold)[0]

            for i in short_indices:
                # Need i-1 >= 0 and i+1 < len(dt)
                if i < 1 or i + 1 >= len(dt):
                    continue
                # Skip if any involved timestamp is NaN
                if np.any(np.isnan(timestamps[i-1:i+3])):
                    continue

                dt_before_list.append(dt[i - 1])
                dt_short_list.append(dt[i])
                dt_after_list.append(dt[i + 1])
                dt_normal_list.append(median_dt)

    return (np.array(dt_before_list), np.array(dt_short_list),
            np.array(dt_after_list), np.array(dt_normal_list))


def collect_normal_dt_events(all_data, lo=0.013, hi=0.016, max_per_video=50):
    """Collect dt_before for 'normal' frames (control group).

    Samples frames where dt is between lo and hi seconds.
    Returns the same tuple format as collect_short_dt_events.
    """
    rng = np.random.default_rng(42)
    dt_before_list = []
    dt_short_list = []
    dt_after_list = []
    dt_normal_list = []

    for user_id, data in all_data.items():
        for vid_idx, vdata in enumerate(data):
            if vdata is None:
                continue
            timestamps = vdata[0]
            N = len(timestamps)
            if N < 4:
                continue

            dt = np.diff(timestamps)
            median_dt = np.median(dt)

            normal_indices = np.where((dt >= lo) & (dt <= hi))[0]
            # Filter to those with valid neighbors
            valid = normal_indices[(normal_indices >= 1) & (normal_indices + 1 < len(dt))]
            if len(valid) == 0:
                continue
            # Sample
            sample_size = min(max_per_video, len(valid))
            chosen = rng.choice(valid, size=sample_size, replace=False)

            for i in chosen:
                if np.any(np.isnan(timestamps[i-1:i+3])):
                    continue
                dt_before_list.append(dt[i - 1])
                dt_short_list.append(dt[i])
                dt_after_list.append(dt[i + 1])
                dt_normal_list.append(median_dt)

    return (np.array(dt_before_list), np.array(dt_short_list),
            np.array(dt_after_list), np.array(dt_normal_list))


def print_header(title):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_stats(values, label):
    """Print summary statistics for an array."""
    print(f"  {label}:")
    print(f"    N       = {len(values)}")
    print(f"    Mean    = {np.mean(values)*1000:.3f} ms")
    print(f"    Median  = {np.median(values)*1000:.3f} ms")
    print(f"    Std     = {np.std(values)*1000:.3f} ms")
    print(f"    P5      = {np.percentile(values, 5)*1000:.3f} ms")
    print(f"    P25     = {np.percentile(values, 25)*1000:.3f} ms")
    print(f"    P75     = {np.percentile(values, 75)*1000:.3f} ms")
    print(f"    P95     = {np.percentile(values, 95)*1000:.3f} ms")


def print_histogram(values_ms, bins, label):
    """Print a text histogram with given bin edges (in ms)."""
    print(f"\n  {label}")
    counts, _ = np.histogram(values_ms, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    bar_width = 40
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        c = counts[i]
        bar = "#" * int(c / max_count * bar_width)
        pct = c / len(values_ms) * 100 if len(values_ms) > 0 else 0
        print(f"    {lo:5.0f}-{hi:5.0f} ms: {c:6d} ({pct:5.1f}%) {bar}")


def analyze_group(dt_before, dt_short, dt_after, dt_normal, label):
    """Run the full analysis for a group of events."""
    print_header(f"{label}: Basic Statistics (N={len(dt_short)})")

    if len(dt_short) == 0:
        print("  No events found.")
        return

    dt_span = dt_before + dt_short
    dt_before_excess = dt_before - dt_normal
    dt_span_excess = dt_span - 2 * dt_normal

    print_stats(dt_short, "dt_short (the short interval)")
    print_stats(dt_before, "dt_before (interval before the short one)")
    print_stats(dt_after, "dt_after (interval after the short one)")
    print_stats(dt_normal, "dt_normal (median dt for the video)")

    # --- Compensation test ---
    print_header(f"{label}: Compensation Test")

    print_stats(dt_before_excess, "dt_before - dt_normal (excess on previous interval)")
    frac_before_gt_normal = np.mean(dt_before > dt_normal)
    print(f"\n  Fraction with dt_before > dt_normal: {frac_before_gt_normal:.4f} "
          f"({frac_before_gt_normal*100:.1f}%)")
    print(f"  (If hypothesis holds, this should be high, e.g. >70%)")

    print_stats(dt_span, "dt_span = dt_before + dt_short (2-frame span)")
    print_stats(dt_span_excess, "dt_span - 2*dt_normal (span excess over expected)")

    frac_span_within_2ms = np.mean(np.abs(dt_span - 2 * dt_normal) < 0.002)
    print(f"\n  Fraction with |dt_span - 2*dt_normal| < 2ms: {frac_span_within_2ms:.4f} "
          f"({frac_span_within_2ms*100:.1f}%)")
    print(f"  (If hypothesis holds, this should be high)")

    # --- Correlation ---
    print_header(f"{label}: Correlations")

    if len(dt_short) > 2:
        corr_before = np.corrcoef(dt_short, dt_before)[0, 1]
        corr_after = np.corrcoef(dt_short, dt_after)[0, 1]
        print(f"  Correlation(dt_short, dt_before) = {corr_before:.4f}")
        print(f"    (Should be negative if delay hypothesis holds)")
        print(f"  Correlation(dt_short, dt_after)  = {corr_after:.4f}")
        print(f"    (Should be ~0 if only the previous frame is affected)")
    else:
        print("  Not enough data for correlation.")


def main():
    print("Loading all users...")
    all_data = load_all_users()
    print(f"Loaded {len(all_data)} users.")

    # ======================================================================
    # 1. Collect short-dt events (<8ms)
    # ======================================================================
    dt_before, dt_short, dt_after, dt_normal = collect_short_dt_events(
        all_data, SHORT_DT_THRESHOLD)

    analyze_group(dt_before, dt_short, dt_after, dt_normal,
                  "SHORT-DT FRAMES (dt < 8ms)")

    # ======================================================================
    # 4. Distribution of dt_before
    # ======================================================================
    if len(dt_short) > 0:
        print_header("SHORT-DT (dt < 8ms): Distribution of dt_before")
        bins_before = [0, 5, 10, 12, 14, 16, 18, 20, 22, 25, 30, 40, 100]
        print_histogram(dt_before * 1000, bins_before,
                        "dt_before for short-dt events (ms)")

        # Also show overall dt distribution for comparison
        all_dt = []
        for user_id, data in all_data.items():
            for vdata in data:
                if vdata is None:
                    continue
                all_dt.extend(np.diff(vdata[0]).tolist())
        all_dt = np.array(all_dt)
        print_histogram(all_dt * 1000, bins_before,
                        "Overall dt distribution (all frames, for comparison)")

    # ======================================================================
    # 5. Distribution of dt_span (2-frame span)
    # ======================================================================
    if len(dt_short) > 0:
        print_header("SHORT-DT (dt < 8ms): Distribution of dt_span = dt_before + dt_short")
        dt_span = dt_before + dt_short
        bins_span = [0, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 60, 100]
        print_histogram(dt_span * 1000, bins_span,
                        "dt_span for short-dt events (ms)")

        # Expected 2*median_dt
        median_normal = np.median(dt_normal)
        print(f"\n  For reference: 2 * overall median dt_normal = "
              f"{2*median_normal*1000:.2f} ms")

    # ======================================================================
    # 6. Stronger test: very short dt (<4ms)
    # ======================================================================
    dt_before_v, dt_short_v, dt_after_v, dt_normal_v = collect_short_dt_events(
        all_data, VERY_SHORT_DT_THRESHOLD)

    analyze_group(dt_before_v, dt_short_v, dt_after_v, dt_normal_v,
                  "VERY SHORT-DT FRAMES (dt < 4ms)")

    if len(dt_short_v) > 0:
        print_header("VERY SHORT-DT (dt < 4ms): Distribution of dt_span")
        dt_span_v = dt_before_v + dt_short_v
        bins_span = [0, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 60, 100]
        print_histogram(dt_span_v * 1000, bins_span,
                        "dt_span for very-short-dt events (ms)")

    # ======================================================================
    # 7. Control comparison: normal frames (13-16ms)
    # ======================================================================
    dt_before_c, dt_short_c, dt_after_c, dt_normal_c = collect_normal_dt_events(all_data)

    analyze_group(dt_before_c, dt_short_c, dt_after_c, dt_normal_c,
                  "CONTROL: NORMAL FRAMES (dt 13-16ms)")

    # ======================================================================
    # Final Verdict
    # ======================================================================
    print_header("FINAL VERDICT")

    if len(dt_short) == 0:
        print("  No short-dt events found. Cannot assess hypothesis.")
        return

    dt_span = dt_before + dt_short
    frac_before_gt = np.mean(dt_before > dt_normal)
    frac_span_ok = np.mean(np.abs(dt_span - 2 * dt_normal) < 0.002)
    corr = np.corrcoef(dt_short, dt_before)[0, 1] if len(dt_short) > 2 else 0

    # Control comparison
    if len(dt_before_c) > 0:
        frac_ctrl_gt = np.mean(dt_before_c > dt_normal_c)
    else:
        frac_ctrl_gt = float('nan')

    print(f"  Key metrics for short-dt (<8ms) frames:")
    print(f"    Total events:                              {len(dt_short)}")
    print(f"    Fraction dt_before > dt_normal:            {frac_before_gt:.1%}")
    print(f"    Fraction dt_span within 2ms of 2*normal:   {frac_span_ok:.1%}")
    print(f"    Correlation(dt_short, dt_before):          {corr:.4f}")
    print()
    print(f"  Control (normal frames 13-16ms):")
    print(f"    Fraction dt_before > dt_normal:            {frac_ctrl_gt:.1%}")
    print()

    # Decision logic
    supports = 0
    total_tests = 4

    if frac_before_gt > 0.65:
        supports += 1
        print("  [PASS] dt_before is usually longer than normal before short-dt frames.")
    else:
        print("  [FAIL] dt_before is NOT consistently longer than normal.")

    if frac_span_ok > 0.50:
        supports += 1
        print("  [PASS] dt_span clusters around 2*normal, supporting timestamp shift.")
    else:
        print("  [FAIL] dt_span does NOT cluster tightly around 2*normal.")

    if corr < -0.3:
        supports += 1
        print("  [PASS] Negative correlation between dt_short and dt_before "
              "(shorter dt -> longer previous dt).")
    else:
        print("  [FAIL] Correlation between dt_short and dt_before is not strongly negative.")

    if frac_before_gt > frac_ctrl_gt + 0.10:
        supports += 1
        print("  [PASS] dt_before excess is significantly higher than control group.")
    else:
        print("  [FAIL] dt_before excess is NOT significantly higher than control group.")

    print()
    if supports >= 3:
        print(f"  VERDICT: HYPOTHESIS SUPPORTED ({supports}/{total_tests} tests passed).")
        print("  Short-dt frames are likely caused by delayed timestamps on the previous frame.")
    elif supports >= 2:
        print(f"  VERDICT: HYPOTHESIS PARTIALLY SUPPORTED ({supports}/{total_tests} tests passed).")
        print("  There is some evidence for the delayed-timestamp mechanism, but it does not")
        print("  fully explain all short-dt events.")
    else:
        print(f"  VERDICT: HYPOTHESIS NOT SUPPORTED ({supports}/{total_tests} tests passed).")
        print("  The data does not support the delayed-timestamp explanation.")


if __name__ == "__main__":
    main()
