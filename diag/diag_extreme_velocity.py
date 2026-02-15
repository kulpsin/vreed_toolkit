#!/usr/bin/env python3
"""Diagnostic script to investigate extreme velocity values (>1000 deg/s)
that survive the robust velocity filter in the VREED eye tracking data."""

import sys
import os
from pathlib import Path
import numpy

# Ensure we can import project modules
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)
os.chdir(_project_root)

import preprocess
import gazelib
import mapping

DATA_PATH = "dataset/04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)"
USER_IDS = list(range(101, 135))
VIDEO_CODES = mapping.EYE_TRACKING_VIDEO_LIST_STR
EXTREME_THRESHOLD = 1500.0  # deg/s
CONTEXT_FRAMES = 5


def load_user(user_id):
    filename = f"{DATA_PATH}/{user_id}_EyeTracking_PreProcessed.dat"
    labels, data = preprocess.load_eye_tracking_data(filename)
    preprocess.replace_missing_values_with_nan(data)
    preprocess.convert_timestamps_to_duration(data, convert_to_seconds=True)
    preprocess.fix_swapped_channel_issue(data)
    preprocess.add_empty_data(labels, data)
    return data


def analyze_frame_context(vdata, vel_idx, velocity_val):
    """Analyze frames surrounding an extreme velocity sample.
    
    vel_idx is the index in the velocity array; it corresponds to the
    transition from frame vel_idx to frame vel_idx+1.
    """
    timestamps = vdata[0]
    az = numpy.nanmean([vdata[1], vdata[4]], axis=0)
    pol = numpy.nanmean([vdata[2], vdata[5]], axis=0)
    is_nan = numpy.isnan(az) | numpy.isnan(pol)
    N = len(az)
    
    # The extreme velocity is between frame vel_idx and vel_idx+1
    frame_a = vel_idx
    frame_b = vel_idx + 1
    
    start = max(0, frame_a - CONTEXT_FRAMES)
    end = min(N, frame_b + CONTEXT_FRAMES + 1)
    
    lines = []
    lines.append(f"  Velocity[{vel_idx}] = {velocity_val:.1f} deg/s  (frame {frame_a} -> {frame_b})")
    lines.append(f"  {'Idx':>6s}  {'Time(s)':>10s}  {'Az':>8s}  {'Pol':>8s}  {'NaN?':>5s}  {'Dist_prev':>10s}  {'dt_prev':>10s}  {'Mark':>6s}")
    lines.append(f"  {'-'*70}")
    
    for i in range(start, end):
        t = timestamps[i]
        a = az[i]
        p = pol[i]
        nan_flag = "YES" if is_nan[i] else "no"
        
        if i > start and not is_nan[i] and not is_nan[i-1]:
            dist = gazelib.great_circle_distance(az[i-1], pol[i-1], az[i], pol[i])
            dt = timestamps[i] - timestamps[i-1]
            dist_str = f"{dist:.4f}"
            dt_str = f"{dt*1000:.2f}ms"
        elif i > start:
            dist_str = "NaN"
            dt_str = f"{(timestamps[i] - timestamps[i-1])*1000:.2f}ms"
        else:
            dist_str = "-"
            dt_str = "-"
        
        mark = ""
        if i == frame_a:
            mark = "<<< A"
        elif i == frame_b:
            mark = "<<< B"
        
        lines.append(f"  {i:6d}  {t:10.4f}  {a:8.3f}  {p:8.3f}  {nan_flag:>5s}  {dist_str:>10s}  {dt_str:>10s}  {mark}")
    
    return lines


def categorize_extreme(vdata, vel_idx):
    """Categorize why this extreme velocity occurred.
    
    Returns a category string.
    """
    timestamps = vdata[0]
    az = numpy.nanmean([vdata[1], vdata[4]], axis=0)
    pol = numpy.nanmean([vdata[2], vdata[5]], axis=0)
    is_nan = numpy.isnan(az) | numpy.isnan(pol)
    N = len(az)
    
    frame_a = vel_idx
    frame_b = vel_idx + 1
    
    dt = timestamps[frame_b] - timestamps[frame_a]
    
    # Check if either frame is NaN (shouldn't produce a velocity, but check)
    if is_nan[frame_a] or is_nan[frame_b]:
        return "nan_frame"
    
    # Check if dt is abnormally small (< 1ms = 0.001s)
    if dt < 0.001:
        return "small_dt"
    
    # Check if adjacent to a NaN gap (within 2 frames)
    near_nan = False
    for offset in range(-2, 3):
        idx = frame_a + offset
        if 0 <= idx < N and is_nan[idx]:
            near_nan = True
            break
        idx = frame_b + offset
        if 0 <= idx < N and is_nan[idx]:
            near_nan = True
            break
    
    if near_nan:
        return "near_nan_gap"
    
    # Check distance — is it a genuinely large distance?
    dist = gazelib.great_circle_distance(az[frame_a], pol[frame_a], az[frame_b], pol[frame_b])
    if dist > 20.0:
        return "large_distance"
    
    # Check if this is part of a cluster of high-velocity frames
    # (consecutive bad frames that the triplet detector can't catch)
    cluster = False
    for offset in [-1, 1]:
        neighbor_idx = vel_idx + offset
        if 0 <= neighbor_idx < N - 1:
            if not is_nan[neighbor_idx] and not is_nan[neighbor_idx + 1]:
                d = gazelib.great_circle_distance(
                    az[neighbor_idx], pol[neighbor_idx],
                    az[neighbor_idx + 1], pol[neighbor_idx + 1])
                neighbor_dt = timestamps[neighbor_idx + 1] - timestamps[neighbor_idx]
                if neighbor_dt > 0:
                    neighbor_vel = d / neighbor_dt
                    if neighbor_vel > 500:
                        cluster = True
    
    if cluster:
        return "consecutive_bad_frames"
    
    return "other"


def main():
    print("=" * 80)
    print("EXTREME VELOCITY DIAGNOSTIC (>1000 deg/s after robust filter)")
    print("=" * 80)
    print()
    
    all_extremes = []  # (user_id, vid_idx, vid_code, vel_idx, velocity, category)
    
    # Also collect all small-dt cases
    small_dt_cases = []  # (user_id, vid_idx, vid_code, frame_idx, dt_ms)
    
    for user_id in USER_IDS:
        print(f"\rLoading user {user_id}...", end="", flush=True)
        try:
            data = load_user(user_id)
        except Exception as e:
            print(f"\n  ERROR loading user {user_id}: {e}")
            continue
        
        for vid_idx in range(12):
            vdata = data[vid_idx]
            if vdata is None:
                continue
            
            vid_code = VIDEO_CODES[vid_idx]
            
            # Compute robust velocity
            ts_mid, velocity, outliers = gazelib.gaze_velocity_robust(
                vdata, threshold=3.0, fill='split')
            
            # Find extreme velocities
            extreme_mask = velocity > EXTREME_THRESHOLD
            extreme_indices = numpy.where(extreme_mask)[0]
            
            for vel_idx in extreme_indices:
                vel_val = velocity[vel_idx]
                if numpy.isnan(vel_val):
                    continue
                category = categorize_extreme(vdata, vel_idx)
                all_extremes.append((user_id, vid_idx, vid_code, vel_idx, vel_val, category))
            
            # Check for small dt (< 1ms)
            timestamps = vdata[0]
            dts = numpy.diff(timestamps)
            small_dt_mask = dts < 0.001  # < 1ms in seconds
            small_dt_indices = numpy.where(small_dt_mask)[0]
            for idx in small_dt_indices:
                dt_ms = dts[idx] * 1000
                small_dt_cases.append((user_id, vid_idx, vid_code, idx, dt_ms))
    
    print("\r" + " " * 40 + "\r", end="")
    
    # ========== SUMMARY ==========
    print()
    print("=" * 80)
    print(f"TOTAL EXTREME VELOCITY SAMPLES (>{EXTREME_THRESHOLD} deg/s): {len(all_extremes)}")
    print("=" * 80)
    print()
    
    # Categorize
    categories = {}
    for item in all_extremes:
        cat = item[5]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)
    
    print("CATEGORY BREAKDOWN:")
    print("-" * 40)
    for cat in sorted(categories.keys()):
        items = categories[cat]
        print(f"  {cat:30s}: {len(items):5d}")
    print()
    
    # ========== BY VIDEO ==========
    print("BY VIDEO:")
    print("-" * 40)
    video_counts = {}
    for item in all_extremes:
        vc = item[2]
        video_counts[vc] = video_counts.get(vc, 0) + 1
    for vc in VIDEO_CODES:
        count = video_counts.get(vc, 0)
        if count > 0:
            print(f"  {vc}: {count}")
    print()
    
    # ========== SMALL DT ==========
    print("=" * 80)
    print(f"SMALL DT CASES (< 1ms): {len(small_dt_cases)}")
    print("=" * 80)
    if small_dt_cases:
        # Show distribution
        dt_values = [x[4] for x in small_dt_cases]
        print(f"  dt range: {min(dt_values):.4f} ms to {max(dt_values):.4f} ms")
        print(f"  dt == 0:  {sum(1 for v in dt_values if v == 0.0)}")
        print(f"  dt < 0.5: {sum(1 for v in dt_values if v < 0.5)}")
        print()
        # Show first 20 cases
        print("  First 20 small-dt cases:")
        for i, (uid, vidx, vc, fidx, dt_ms) in enumerate(small_dt_cases[:20]):
            print(f"    User {uid}, {vc} (vid {vidx}), frame {fidx}: dt = {dt_ms:.4f} ms")
        if len(small_dt_cases) > 20:
            print(f"    ... and {len(small_dt_cases) - 20} more")
    print()
    
    # ========== Check overlap: small dt that also produce extreme velocity ==========
    small_dt_set = set((x[0], x[1], x[3]) for x in small_dt_cases)
    overlap = []
    for item in all_extremes:
        uid, vidx, vc, vel_idx, vel_val, cat = item
        # vel_idx corresponds to frames vel_idx -> vel_idx+1
        if (uid, vidx, vel_idx) in small_dt_set:
            overlap.append(item)
    print(f"EXTREME VELOCITIES AT SMALL-DT FRAMES: {len(overlap)}")
    if overlap:
        for uid, vidx, vc, vel_idx, vel_val, cat in overlap[:20]:
            print(f"  User {uid}, {vc}, vel_idx {vel_idx}: {vel_val:.1f} deg/s [{cat}]")
    print()
    
    # ========== DETAILED DIAGNOSTICS ==========
    # Print detailed context for each category, limited to keep output manageable
    MAX_PER_CAT = 5
    
    for cat in sorted(categories.keys()):
        items = categories[cat]
        print("=" * 80)
        print(f"CATEGORY: {cat}  ({len(items)} total, showing up to {MAX_PER_CAT})")
        print("=" * 80)
        
        # Sort by velocity descending to show worst cases first
        items_sorted = sorted(items, key=lambda x: -x[4])
        
        for item in items_sorted[:MAX_PER_CAT]:
            uid, vidx, vc, vel_idx, vel_val, _ = item
            print(f"\n--- User {uid}, Video {vc} (idx {vidx}) ---")
            
            data = load_user(uid)
            vdata = data[vidx]
            
            # Also check: was this frame detected as outlier by the robust filter?
            _, _, outliers = gazelib.gaze_velocity_robust(vdata, threshold=3.0, fill='split')
            # The vel_idx involves frames vel_idx and vel_idx+1
            out_a = outliers[vel_idx] if vel_idx < len(outliers) else False
            out_b = outliers[vel_idx + 1] if vel_idx + 1 < len(outliers) else False
            print(f"  Outlier flag: frame_A[{vel_idx}]={out_a}, frame_B[{vel_idx+1}]={out_b}")
            
            context_lines = analyze_frame_context(vdata, vel_idx, vel_val)
            for line in context_lines:
                print(line)
        print()
    
    # ========== SPECIAL FOCUS: JNG and ZMZ ==========
    print("=" * 80)
    print("FOCUS ON JNG AND ZMZ")
    print("=" * 80)
    jng_zmz = [item for item in all_extremes if item[2] in ("JNG", "ZMZ")]
    print(f"Total extreme samples in JNG/ZMZ: {len(jng_zmz)}")
    print()
    
    for vc in ("JNG", "ZMZ"):
        items = [x for x in jng_zmz if x[2] == vc]
        if not items:
            print(f"  {vc}: no extreme velocities")
            continue
        
        cat_counts = {}
        for x in items:
            cat_counts[x[5]] = cat_counts.get(x[5], 0) + 1
        
        print(f"  {vc}: {len(items)} extreme velocities")
        for cat, count in sorted(cat_counts.items()):
            print(f"    {cat}: {count}")
        
        # Max velocity
        worst = max(items, key=lambda x: x[4])
        print(f"    Max velocity: {worst[4]:.1f} deg/s (User {worst[0]}, vel_idx {worst[3]})")
    print()
    
    # ========== VELOCITY DISTRIBUTION NEAR THRESHOLD ==========
    print("=" * 80)
    print("VELOCITY DISTRIBUTION OF EXTREMES")
    print("=" * 80)
    vel_values = [x[4] for x in all_extremes]
    if vel_values:
        bins = [1000, 1500, 2000, 2500, 3000, 5000, 10000, 50000, float('inf')]
        for i in range(len(bins) - 1):
            count = sum(1 for v in vel_values if bins[i] <= v < bins[i+1])
            if count > 0:
                high = f"{bins[i+1]}" if bins[i+1] != float('inf') else "inf"
                print(f"  {bins[i]:>6.0f} - {high:>6s} deg/s: {count}")
    print()
    
    print("Done.")


if __name__ == "__main__":
    main()
