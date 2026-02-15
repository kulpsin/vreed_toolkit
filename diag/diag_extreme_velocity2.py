#!/usr/bin/env python3
"""Follow-up diagnostic: analyze the 'other' category pattern in detail.

The first pass revealed that the dominant pattern in the 'other' category
is: a stable gaze position, then a sudden jump to a different position
(~10-20 deg), with both positions appearing to be valid gaze data (not noise).

This script investigates:
1. Are these "level shifts" — the gaze jumps between two stable plateaus?
2. Why doesn't the triplet detector catch them? (Because they're NOT
   single-frame outliers — they're genuine position changes.)
3. Is the small dt (4-6ms) the main amplifier?
"""

import sys
import os
from pathlib import Path
import numpy

_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)
os.chdir(_project_root)

import preprocess
import gazelib
import mapping

DATA_PATH = "dataset/04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)"
VIDEO_CODES = mapping.EYE_TRACKING_VIDEO_LIST_STR
USER_IDS = list(range(101, 135))
EXTREME_THRESHOLD = 1500.0  # deg/s


def load_user(user_id):
    filename = f"{DATA_PATH}/{user_id}_EyeTracking_PreProcessed.dat"
    labels, data = preprocess.load_eye_tracking_data(filename)
    preprocess.replace_missing_values_with_nan(data)
    preprocess.convert_timestamps_to_duration(data, convert_to_seconds=True)
    preprocess.fix_swapped_channel_issue(data)
    preprocess.add_empty_data(labels, data)
    return data


def main():
    print("=" * 80)
    print("FOLLOW-UP: ANATOMY OF EXTREME VELOCITY EVENTS")
    print("=" * 80)
    print()

    all_events = []

    for user_id in USER_IDS:
        data = load_user(user_id)
        for vid_idx in range(12):
            vdata = data[vid_idx]
            if vdata is None:
                continue

            vid_code = VIDEO_CODES[vid_idx]
            ts_mid, velocity, outliers = gazelib.gaze_velocity_robust(
                vdata, threshold=3.0, fill='split')

            timestamps = vdata[0]
            az = numpy.nanmean([vdata[1], vdata[4]], axis=0)
            pol = numpy.nanmean([vdata[2], vdata[5]], axis=0)
            is_nan = numpy.isnan(az) | numpy.isnan(pol)

            extreme_mask = (~numpy.isnan(velocity)) & (velocity > EXTREME_THRESHOLD)
            for vel_idx in numpy.where(extreme_mask)[0]:
                fa, fb = vel_idx, vel_idx + 1
                if is_nan[fa] or is_nan[fb]:
                    continue

                dt = timestamps[fb] - timestamps[fa]
                dist = gazelib.great_circle_distance(az[fa], pol[fa], az[fb], pol[fb])

                # Measure stability before A: average velocity over 5 frames before
                pre_vels = []
                for k in range(max(0, fa - 5), fa):
                    if not is_nan[k] and not is_nan[k + 1]:
                        d = gazelib.great_circle_distance(az[k], pol[k], az[k+1], pol[k+1])
                        t = timestamps[k+1] - timestamps[k]
                        if t > 0:
                            pre_vels.append(d / t)

                # Measure stability after B: average velocity over 5 frames after
                post_vels = []
                N = len(az)
                for k in range(fb, min(N - 1, fb + 5)):
                    if not is_nan[k] and not is_nan[k + 1]:
                        d = gazelib.great_circle_distance(az[k], pol[k], az[k+1], pol[k+1])
                        t = timestamps[k+1] - timestamps[k]
                        if t > 0:
                            post_vels.append(d / t)

                pre_mean = numpy.mean(pre_vels) if pre_vels else numpy.nan
                post_mean = numpy.mean(post_vels) if post_vels else numpy.nan

                all_events.append({
                    'user': user_id,
                    'vid_idx': vid_idx,
                    'vid_code': vid_code,
                    'vel_idx': vel_idx,
                    'velocity': velocity[vel_idx],
                    'dt_ms': dt * 1000,
                    'distance_deg': dist,
                    'pre_mean_vel': pre_mean,
                    'post_mean_vel': post_mean,
                    'az_before': az[fa],
                    'pol_before': pol[fa],
                    'az_after': az[fb],
                    'pol_after': pol[fb],
                    'outlier_a': outliers[fa],
                    'outlier_b': outliers[fb],
                })

    print(f"Total extreme events: {len(all_events)}")
    print()

    # ===== ANALYSIS 1: dt distribution =====
    print("=" * 80)
    print("ANALYSIS 1: TIME INTERVAL (dt) AT EXTREME VELOCITY EVENTS")
    print("=" * 80)
    dts = [e['dt_ms'] for e in all_events]
    bins_dt = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50)]
    for lo, hi in bins_dt:
        count = sum(1 for d in dts if lo <= d < hi)
        if count:
            print(f"  {lo:3d}-{hi:3d} ms: {count:3d}  ({'*' * count})")
    print(f"  Mean dt: {numpy.mean(dts):.1f} ms, Median: {numpy.median(dts):.1f} ms")
    print()

    # ===== ANALYSIS 2: Distance distribution =====
    print("=" * 80)
    print("ANALYSIS 2: ANGULAR DISTANCE AT EXTREME VELOCITY EVENTS")
    print("=" * 80)
    dists = [e['distance_deg'] for e in all_events]
    bins_dist = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 60)]
    for lo, hi in bins_dist:
        count = sum(1 for d in dists if lo <= d < hi)
        if count:
            print(f"  {lo:3d}-{hi:3d} deg: {count:3d}")
    print(f"  Mean dist: {numpy.mean(dists):.1f} deg, Median: {numpy.median(dists):.1f} deg")
    print()

    # ===== ANALYSIS 3: Stability around event =====
    print("=" * 80)
    print("ANALYSIS 3: SURROUNDING STABILITY (is this a level-shift?)")
    print("=" * 80)
    print("  'Level shift' = pre and post mean velocities are < 100 deg/s")
    print("  (i.e., gaze was stable before and after the jump)")
    print()
    level_shifts = [e for e in all_events
                    if not numpy.isnan(e['pre_mean_vel'])
                    and not numpy.isnan(e['post_mean_vel'])
                    and e['pre_mean_vel'] < 100
                    and e['post_mean_vel'] < 100]
    print(f"  Level-shift events: {len(level_shifts)} / {len(all_events)}")

    non_level = [e for e in all_events if e not in level_shifts]
    print(f"  Non-level-shift events: {len(non_level)}")
    print()

    # Show details of non-level-shift events
    if non_level:
        print("  Non-level-shift events (smooth saccade or ramp):")
        for e in sorted(non_level, key=lambda x: -x['velocity']):
            print(f"    User {e['user']:3d} {e['vid_code']} vel_idx={e['vel_idx']:5d}: "
                  f"{e['velocity']:7.1f} deg/s, dist={e['distance_deg']:.1f}deg, "
                  f"dt={e['dt_ms']:.0f}ms, "
                  f"pre_vel={e['pre_mean_vel']:.0f}, post_vel={e['post_mean_vel']:.0f} deg/s")
        print()

    # ===== ANALYSIS 4: Level-shift details =====
    if level_shifts:
        print("=" * 80)
        print("ANALYSIS 4: LEVEL-SHIFT DETAILS")
        print("=" * 80)
        print("  These are cases where gaze jumps suddenly between two stable positions.")
        print("  The triplet detector can NOT catch these because the frame is NOT an outlier —")
        print("  the gaze genuinely moves to a new stable location.")
        print()
        print(f"  {'User':>4s} {'Video':>5s} {'Vel(deg/s)':>10s} {'Dist(deg)':>10s} "
              f"{'dt(ms)':>7s} {'Az_bef':>8s} {'Az_aft':>8s} {'Pre_v':>7s} {'Post_v':>7s}")
        print("  " + "-" * 75)
        for e in sorted(level_shifts, key=lambda x: -x['velocity']):
            print(f"  {e['user']:4d} {e['vid_code']:>5s} {e['velocity']:10.1f} "
                  f"{e['distance_deg']:10.2f} {e['dt_ms']:7.1f} "
                  f"{e['az_before']:8.2f} {e['az_after']:8.2f} "
                  f"{e['pre_mean_vel']:7.1f} {e['post_mean_vel']:7.1f}")

    # ===== ANALYSIS 5: Would velocity be normal at typical dt? =====
    print()
    print("=" * 80)
    print("ANALYSIS 5: VELOCITY IF DT WERE MEDIAN SAMPLE INTERVAL (~14.4ms)")
    print("=" * 80)
    print("  This shows whether the extreme velocity is due to short dt or large distance.")
    print()
    typical_dt = 0.0144  # ~69.5 Hz
    for e in sorted(all_events, key=lambda x: -x['velocity'])[:15]:
        hypothetical_vel = e['distance_deg'] / typical_dt
        print(f"  User {e['user']:3d} {e['vid_code']:>5s}: "
              f"actual={e['velocity']:7.1f} deg/s (dt={e['dt_ms']:.0f}ms, d={e['distance_deg']:.1f}deg) "
              f"| at 14.4ms would be {hypothetical_vel:.1f} deg/s "
              f"{'[STILL >%d]'%EXTREME_THRESHOLD if hypothetical_vel > EXTREME_THRESHOLD else '[would be OK]'}")

    # ===== ANALYSIS 6: Left vs Right eye disagreement? =====
    print()
    print("=" * 80)
    print("ANALYSIS 6: LEFT vs RIGHT EYE DISAGREEMENT AT EXTREME EVENTS")
    print("=" * 80)
    print("  Check if one eye reports a different position than the other.")
    print()

    for e in sorted(all_events, key=lambda x: -x['velocity'])[:10]:
        uid, vidx, vel_idx = e['user'], e['vid_idx'], e['vel_idx']
        data = load_user(uid)
        vdata = data[vidx]
        fa, fb = vel_idx, vel_idx + 1

        lx_a, ly_a = vdata[1, fa], vdata[2, fa]
        rx_a, ry_a = vdata[4, fa], vdata[5, fa]
        lx_b, ly_b = vdata[1, fb], vdata[2, fb]
        rx_b, ry_b = vdata[4, fb], vdata[5, fb]

        # Distance for each eye separately
        d_left = gazelib.great_circle_distance(lx_a, ly_a, lx_b, ly_b)
        d_right = gazelib.great_circle_distance(rx_a, ry_a, rx_b, ry_b)

        # Inter-eye distance at each frame
        inter_a = gazelib.great_circle_distance(lx_a, ly_a, rx_a, ry_a)
        inter_b = gazelib.great_circle_distance(lx_b, ly_b, rx_b, ry_b)

        l_nan = numpy.isnan(lx_a) or numpy.isnan(lx_b)
        r_nan = numpy.isnan(rx_a) or numpy.isnan(rx_b)

        print(f"  User {uid:3d} {e['vid_code']:>5s} vel_idx={vel_idx}: "
              f"vel={e['velocity']:.0f} deg/s")
        print(f"    Frame A: L=({lx_a:.2f},{ly_a:.2f}) R=({rx_a:.2f},{ry_a:.2f}) "
              f"inter-eye={inter_a:.2f}deg {'[L-NaN]' if numpy.isnan(lx_a) else ''}"
              f"{'[R-NaN]' if numpy.isnan(rx_a) else ''}")
        print(f"    Frame B: L=({lx_b:.2f},{ly_b:.2f}) R=({rx_b:.2f},{ry_b:.2f}) "
              f"inter-eye={inter_b:.2f}deg {'[L-NaN]' if numpy.isnan(lx_b) else ''}"
              f"{'[R-NaN]' if numpy.isnan(rx_b) else ''}")
        print(f"    d_left={d_left:.2f}deg, d_right={d_right:.2f}deg, "
              f"ratio={max(d_left,d_right)/max(min(d_left,d_right),0.001):.1f}x")
        print()

    print("=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)


if __name__ == "__main__":
    main()
