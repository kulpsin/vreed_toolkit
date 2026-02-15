#!/usr/bin/env python3
"""Diagnostic: investigate short frame intervals (<8ms) and multi-frame
tracking artifacts (island plateaus where gaze jumps away and returns).

For every short-dt frame, prints context (5 frames before/after) and
checks for round-trip artifacts.
"""

import sys
import os
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)
os.chdir(_project_root)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging
logging.disable(logging.CRITICAL)  # suppress preprocessing log spam

import numpy
import preprocess
import mapping
import gazelib

DATA_PATH = "dataset/04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)"
USER_IDS = list(range(101, 135))
DT_THRESHOLD = 0.008  # 8 ms in seconds
CONTEXT = 5           # frames before/after to show
LARGE_DIST_THRESHOLD = 2.0   # degrees
ROUND_TRIP_JUMP = 5.0         # minimum jump in azimuth to consider (degrees)
ROUND_TRIP_RETURN = 3.0       # max distance from pre-jump position to count as "returned"
ROUND_TRIP_WINDOW = 10        # frames after the short-dt frame to look for return


def load_user(user_id):
    filename = f"{DATA_PATH}/{user_id}_EyeTracking_PreProcessed.dat"
    labels, data = preprocess.load_eye_tracking_data(filename)
    preprocess.replace_missing_values_with_nan(data)
    preprocess.convert_timestamps_to_duration(data, convert_to_seconds=True)
    preprocess.fix_swapped_channel_issue(data)
    preprocess.add_empty_data(labels, data)
    return data


def compute_gaze(vdata):
    """Return az, pol arrays (nanmean of left/right)."""
    az = numpy.nanmean([vdata[1], vdata[4]], axis=0)
    pol = numpy.nanmean([vdata[2], vdata[5]], axis=0)
    return az, pol


def is_nan_frame(az, pol, idx):
    """Check if a frame has NaN gaze coordinates."""
    return numpy.isnan(az[idx]) or numpy.isnan(pol[idx])


def print_context(vdata, az, pol, dt_arr, dist_arr, event_idx, user_id, vid_idx, vid_name):
    """Print context around a short-dt frame."""
    N = len(az)
    start = max(0, event_idx - CONTEXT)
    end = min(N - 1, event_idx + CONTEXT)

    print(f"  User {user_id}  Video {vid_idx} ({vid_name})  Frame {event_idx}/{N}")
    print(f"  {'idx':>6s}  {'time_s':>10s}  {'az':>10s}  {'pol':>10s}  {'dt_ms':>8s}  {'dist_deg':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")

    for i in range(start, end + 1):
        t = vdata[0, i]
        a = az[i]
        p = pol[i]
        # dt and dist are for transition FROM i-1 TO i, stored at index i-1
        if i > 0 and (i - 1) < len(dt_arr):
            dt_val = dt_arr[i - 1] * 1000  # to ms
            d_val = dist_arr[i - 1]
            dt_str = f"{dt_val:8.2f}"
            d_str = f"{d_val:10.4f}"
        else:
            dt_str = f"{'---':>8s}"
            d_str = f"{'---':>10s}"

        marker = " >>>" if i == event_idx else "    "
        nan_marker = " [NaN]" if numpy.isnan(a) or numpy.isnan(p) else ""
        print(f"{marker}{i:6d}  {t:10.4f}  {a:10.4f}  {p:10.4f}  {dt_str}  {d_str}{nan_marker}")

    print()


def check_round_trip(az, pol, event_idx):
    """Check if the short-dt event is part of a round-trip artifact.

    Logic: find the "pre-jump" stable position (last non-NaN frame
    at or before event_idx - 1 that is at least 2 frames before the event,
    to get a stable reference). Then check if azimuth jumped by >ROUND_TRIP_JUMP
    at the event. Then look forward up to ROUND_TRIP_WINDOW frames for gaze
    returning within ROUND_TRIP_RETURN degrees of pre-jump position.
    """
    N = len(az)

    # Pre-jump reference: use frame event_idx - 1 (the frame just before)
    # But we need it to be valid.  Walk back to find a valid frame.
    pre_idx = None
    for k in range(event_idx - 1, max(-1, event_idx - CONTEXT - 1), -1):
        if k >= 0 and not numpy.isnan(az[k]) and not numpy.isnan(pol[k]):
            pre_idx = k
            break

    if pre_idx is None:
        return False

    pre_az = az[pre_idx]
    pre_pol = pol[pre_idx]

    # Check that the event frame itself has valid data and shows a big jump
    if numpy.isnan(az[event_idx]) or numpy.isnan(pol[event_idx]):
        return False

    jump_dist = gazelib.great_circle_distance(pre_az, pre_pol, az[event_idx], pol[event_idx])
    if numpy.isnan(jump_dist) or jump_dist < ROUND_TRIP_JUMP:
        return False

    # Look forward for a return to within ROUND_TRIP_RETURN of pre-jump
    for k in range(event_idx + 1, min(N, event_idx + ROUND_TRIP_WINDOW + 1)):
        if numpy.isnan(az[k]) or numpy.isnan(pol[k]):
            continue
        return_dist = gazelib.great_circle_distance(pre_az, pre_pol, az[k], pol[k])
        if not numpy.isnan(return_dist) and return_dist < ROUND_TRIP_RETURN:
            return True

    return False


def main():
    # Aggregate counters
    total_short_dt = 0
    total_large_dist = 0
    total_adjacent_nan = 0
    total_round_trip = 0
    dt_histogram = numpy.zeros(8, dtype=int)  # bins: 0-1, 1-2, ..., 7-8 ms

    total_frames = 0
    total_videos = 0

    for user_id in USER_IDS:
        try:
            data = load_user(user_id)
        except FileNotFoundError:
            print(f"[SKIP] User {user_id}: file not found")
            continue

        for vid_idx in range(12):
            vdata = data[vid_idx]
            if vdata is None:
                continue

            vid_name = mapping.get_str_code(vid_idx, "eye")
            N = vdata.shape[1]
            total_frames += N
            total_videos += 1

            az, pol = compute_gaze(vdata)
            timestamps = vdata[0]

            # Compute dt and distances for consecutive frames
            dt_arr = numpy.diff(timestamps)                 # shape (N-1,)
            dist_arr = gazelib.great_circle_distance(
                az[:-1], pol[:-1], az[1:], pol[1:]
            )  # shape (N-1,)

            # Find short-dt frames: dt_arr[i] < threshold means transition
            # from frame i to frame i+1 is short.  The "short-dt frame" is i+1.
            short_dt_indices = numpy.where(dt_arr < DT_THRESHOLD)[0]

            for arr_idx in short_dt_indices:
                frame_idx = arr_idx + 1  # the frame that arrived too quickly
                dt_val = dt_arr[arr_idx]
                dist_val = dist_arr[arr_idx]
                dt_ms = dt_val * 1000

                total_short_dt += 1

                # Histogram bin
                bin_idx = int(dt_ms)
                if 0 <= bin_idx < 8:
                    dt_histogram[bin_idx] += 1

                # Large distance?
                has_large_dist = (not numpy.isnan(dist_val) and dist_val > LARGE_DIST_THRESHOLD)
                if has_large_dist:
                    total_large_dist += 1

                # Adjacent to NaN gap?
                adjacent_nan = False
                for check_idx in range(max(0, frame_idx - 2), min(N, frame_idx + 3)):
                    if is_nan_frame(az, pol, check_idx):
                        adjacent_nan = True
                        break
                if adjacent_nan:
                    total_adjacent_nan += 1

                # Round-trip artifact?
                is_round_trip = check_round_trip(az, pol, frame_idx)
                if is_round_trip:
                    total_round_trip += 1

                # Print context
                flags = []
                if has_large_dist:
                    flags.append("LARGE_DIST")
                if adjacent_nan:
                    flags.append("NEAR_NAN")
                if is_round_trip:
                    flags.append("ROUND_TRIP")
                flag_str = f"  [{', '.join(flags)}]" if flags else ""

                print(f"=== Short dt: {dt_ms:.2f} ms  dist: {dist_val:.4f} deg{flag_str} ===")
                print_context(vdata, az, pol, dt_arr, dist_arr,
                              frame_idx, user_id, vid_idx, vid_name)

    # =====================================================================
    # Summary
    # =====================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total users scanned:         {len(USER_IDS)}")
    print(f"Total videos processed:      {total_videos}")
    print(f"Total frames:                {total_frames}")
    print()
    print(f"Short-dt frames (< 8 ms):    {total_short_dt}")
    if total_frames > 0:
        print(f"  as % of all frames:        {100.0 * total_short_dt / total_frames:.4f}%")
    print()
    print(f"  with large distance (> {LARGE_DIST_THRESHOLD} deg): {total_large_dist}  "
          f"({100.0 * total_large_dist / max(1, total_short_dt):.1f}% of short-dt)")
    print(f"  adjacent to NaN gap:       {total_adjacent_nan}  "
          f"({100.0 * total_adjacent_nan / max(1, total_short_dt):.1f}% of short-dt)")
    print(f"  round-trip artifacts:       {total_round_trip}  "
          f"({100.0 * total_round_trip / max(1, total_short_dt):.1f}% of short-dt)")
    print()
    print("dt distribution (ms bins):")
    for i in range(8):
        count = dt_histogram[i]
        pct = 100.0 * count / max(1, total_short_dt)
        bar = "#" * int(pct / 2)
        print(f"  {i:d}-{i+1:d} ms:  {count:6d}  ({pct:5.1f}%)  {bar}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
