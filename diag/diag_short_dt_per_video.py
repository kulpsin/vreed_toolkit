#!/usr/bin/env python3
"""Diagnostic: Do short-dt frame rates differ systematically between videos?

Analyzes inter-frame time intervals (dt) across all 12 VREED videos,
pooling data from all 34 users, to determine whether certain videos
have systematically different frame-rate characteristics.
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
from scipy import stats

import preprocess
import mapping


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "dataset/04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)"
USER_IDS = list(range(101, 135))
N_VIDEOS = 12
SHORT_DT_THRESHOLD = 8  # ms


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
def load_all():
    """Return dict: user_id -> list of 12 video arrays (or None)."""
    all_data = {}
    for uid in USER_IDS:
        fname = os.path.join(DATA_DIR, f"{uid}_EyeTracking_PreProcessed.dat")
        if not os.path.exists(fname):
            continue
        labels, data = preprocess.load_eye_tracking_data(fname)
        preprocess.replace_missing_values_with_nan(data)
        preprocess.convert_timestamps_to_duration(data, convert_to_seconds=False)  # keep ms
        preprocess.fix_swapped_channel_issue(data)
        preprocess.add_empty_data(labels, data)
        all_data[uid] = data
    return all_data


def compute_dt(vdata):
    """Return array of inter-frame intervals in ms."""
    if vdata is None:
        return np.array([])
    timestamps = vdata[0]
    dt = np.diff(timestamps)
    return dt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data from all users...")
    all_data = load_all()
    print(f"Loaded {len(all_data)} users.\n")

    # Collect dt arrays per video and per (user, video)
    # dt_by_video[v] = pooled dt array for video v
    # dt_by_user_video[uid][v] = dt array for that user-video pair
    dt_by_video = {v: [] for v in range(N_VIDEOS)}
    dt_by_user_video = {}
    durations_by_video = {v: [] for v in range(N_VIDEOS)}

    for uid, data in all_data.items():
        dt_by_user_video[uid] = {}
        for v in range(N_VIDEOS):
            vdata = data[v]
            dt = compute_dt(vdata)
            dt_by_user_video[uid][v] = dt
            if len(dt) > 0:
                dt_by_video[v].append(dt)
                durations_by_video[v].append(vdata[0][-1])  # last timestamp = duration in ms

    # Concatenate pooled arrays
    dt_pooled = {}
    for v in range(N_VIDEOS):
        if dt_by_video[v]:
            dt_pooled[v] = np.concatenate(dt_by_video[v])
        else:
            dt_pooled[v] = np.array([])

    # ==================================================================
    # 1. Per-video summary table
    # ==================================================================
    print("=" * 100)
    print("SECTION 1: Per-Video Summary Table")
    print("=" * 100)
    header = (f"{'Idx':>3} {'Video':>5} {'Total dt':>10} "
              f"{'<8ms #':>8} {'<8ms%':>7} "
              f"{'<6ms #':>8} {'<6ms%':>7} "
              f"{'<4ms #':>8} {'<4ms%':>7} "
              f"{'Mean':>8} {'Median':>8} {'Std':>8} "
              f"{'Dur(s)':>8}")
    print(header)
    print("-" * len(header))

    for v in range(N_VIDEOS):
        dt = dt_pooled[v]
        name = mapping.get_str_code(v, "eye")
        total = len(dt)
        if total == 0:
            print(f"{v:>3} {name:>5}  (no data)")
            continue
        n8 = np.sum(dt < 8)
        n6 = np.sum(dt < 6)
        n4 = np.sum(dt < 4)
        mean_dt = np.mean(dt)
        med_dt = np.median(dt)
        std_dt = np.std(dt)
        dur_vals = durations_by_video[v]
        mean_dur_s = np.mean(dur_vals) / 1000.0 if dur_vals else 0.0
        print(f"{v:>3} {name:>5} {total:>10,} "
              f"{n8:>8,} {100*n8/total:>6.2f}% "
              f"{n6:>8,} {100*n6/total:>6.2f}% "
              f"{n4:>8,} {100*n4/total:>6.2f}% "
              f"{mean_dt:>8.2f} {med_dt:>8.2f} {std_dt:>8.2f} "
              f"{mean_dur_s:>8.1f}")
    print()

    # ==================================================================
    # 2. Per-video dt distribution (histogram as table)
    # ==================================================================
    print("=" * 100)
    print("SECTION 2: Per-Video dt Distribution (% in each bin)")
    print("=" * 100)
    bin_edges = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 50, 100, np.inf]
    bin_labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-14",
                  "14-16", "16-18", "18-20", "20-25", "25-30", "30-50",
                  "50-100", "100+"]

    header2 = f"{'Video':>5} " + " ".join(f"{b:>6}" for b in bin_labels)
    print(header2)
    print("-" * len(header2))

    for v in range(N_VIDEOS):
        dt = dt_pooled[v]
        name = mapping.get_str_code(v, "eye")
        if len(dt) == 0:
            print(f"{name:>5}  (no data)")
            continue
        counts, _ = np.histogram(dt, bins=bin_edges)
        pcts = 100.0 * counts / len(dt)
        row = f"{name:>5} " + " ".join(f"{p:>6.2f}" for p in pcts)
        print(row)
    print()

    # ==================================================================
    # 3. Per-video boxplot stats
    # ==================================================================
    print("=" * 100)
    print("SECTION 3: Per-Video Boxplot Stats of dt (ms)")
    print("=" * 100)
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    pct_labels = ["min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max"]
    header3 = f"{'Video':>5} " + " ".join(f"{l:>8}" for l in pct_labels)
    print(header3)
    print("-" * len(header3))

    for v in range(N_VIDEOS):
        dt = dt_pooled[v]
        name = mapping.get_str_code(v, "eye")
        if len(dt) == 0:
            print(f"{name:>5}  (no data)")
            continue
        vals = np.percentile(dt, percentiles)
        row = f"{name:>5} " + " ".join(f"{x:>8.2f}" for x in vals)
        print(row)
    print()

    # ==================================================================
    # 4. Statistical test
    # ==================================================================
    print("=" * 100)
    print("SECTION 4: Statistical Test — Are video-level short-dt rates different?")
    print("=" * 100)

    # Per-user short-dt rate for each video
    video_user_rates = {v: [] for v in range(N_VIDEOS)}
    for uid in all_data:
        for v in range(N_VIDEOS):
            dt = dt_by_user_video[uid][v]
            if len(dt) == 0:
                continue
            rate = 100.0 * np.sum(dt < SHORT_DT_THRESHOLD) / len(dt)
            video_user_rates[v].append(rate)

    print(f"\nMean short-dt (<{SHORT_DT_THRESHOLD}ms) rate per video (averaged across users):\n")
    print(f"{'Idx':>3} {'Video':>5} {'Mean%':>8} {'SE%':>8} {'N users':>8} {'Min%':>8} {'Max%':>8}")
    print("-" * 55)

    means_for_test = []
    for v in range(N_VIDEOS):
        rates = video_user_rates[v]
        name = mapping.get_str_code(v, "eye")
        if len(rates) == 0:
            print(f"{v:>3} {name:>5}  (no data)")
            means_for_test.append(np.nan)
            continue
        m = np.mean(rates)
        se = np.std(rates, ddof=1) / np.sqrt(len(rates)) if len(rates) > 1 else 0
        means_for_test.append(m)
        print(f"{v:>3} {name:>5} {m:>8.3f} {se:>8.3f} {len(rates):>8} "
              f"{np.min(rates):>8.3f} {np.max(rates):>8.3f}")

    # Kruskal-Wallis test across videos
    groups = [np.array(video_user_rates[v]) for v in range(N_VIDEOS) if len(video_user_rates[v]) > 0]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis test across videos: H = {kw_stat:.4f}, p = {kw_p:.6f}")
        if kw_p < 0.05:
            print("  -> Statistically significant difference between videos (p < 0.05).")
        else:
            print("  -> No statistically significant difference between videos (p >= 0.05).")

    # One-way ANOVA as well
    if len(groups) >= 2:
        f_stat, f_p = stats.f_oneway(*groups)
        print(f"One-way ANOVA: F = {f_stat:.4f}, p = {f_p:.6f}")

    # Identify highest and lowest
    valid = [(v, means_for_test[v]) for v in range(N_VIDEOS) if not np.isnan(means_for_test[v])]
    valid.sort(key=lambda x: x[1], reverse=True)
    print(f"\nHighest short-dt rate: video {valid[0][0]} ({mapping.get_str_code(valid[0][0], 'eye')}) = {valid[0][1]:.3f}%")
    print(f"Lowest  short-dt rate: video {valid[-1][0]} ({mapping.get_str_code(valid[-1][0], 'eye')}) = {valid[-1][1]:.3f}%")
    spread = valid[0][1] - valid[-1][1]
    print(f"Spread (max - min): {spread:.3f} percentage points")
    print()

    # ==================================================================
    # 5. Temporal pattern within videos (top 3 highest short-dt rate)
    # ==================================================================
    print("=" * 100)
    print("SECTION 5: Temporal Pattern — Short-dt rate by time segment")
    print("         (for the 3 videos with highest short-dt rates)")
    print("=" * 100)

    top3 = [x[0] for x in valid[:3]]
    N_SEGMENTS = 10

    for v in top3:
        name = mapping.get_str_code(v, "eye")
        print(f"\nVideo {v} ({name}):")
        # Per-user segment rates
        seg_rates = {s: [] for s in range(N_SEGMENTS)}

        for uid in all_data:
            dt = dt_by_user_video[uid][v]
            if len(dt) == 0:
                continue
            vdata = all_data[uid][v]
            timestamps = vdata[0]
            # Midpoints between consecutive timestamps (where dt lives)
            t_mid = (timestamps[:-1] + timestamps[1:]) / 2.0
            duration = timestamps[-1]
            if duration == 0:
                continue
            seg_boundaries = np.linspace(0, duration, N_SEGMENTS + 1)
            for s in range(N_SEGMENTS):
                mask = (t_mid >= seg_boundaries[s]) & (t_mid < seg_boundaries[s + 1])
                if s == N_SEGMENTS - 1:
                    mask = (t_mid >= seg_boundaries[s]) & (t_mid <= seg_boundaries[s + 1])
                n_in_seg = np.sum(mask)
                if n_in_seg == 0:
                    continue
                n_short = np.sum((dt < SHORT_DT_THRESHOLD) & mask)
                seg_rates[s].append(100.0 * n_short / n_in_seg)

        print(f"  {'Segment':>8} {'Time%':>7} {'Mean short-dt%':>15} {'SE':>8} {'N':>4}")
        print(f"  " + "-" * 48)
        for s in range(N_SEGMENTS):
            rates = seg_rates[s]
            lo = s * 10
            hi = (s + 1) * 10
            if len(rates) == 0:
                print(f"  {s:>8} {lo:>3}-{hi:<3}%  (no data)")
                continue
            m = np.mean(rates)
            se = np.std(rates, ddof=1) / np.sqrt(len(rates)) if len(rates) > 1 else 0
            print(f"  {s:>8} {lo:>3}-{hi:<3}% {m:>14.3f}% {se:>8.3f} {len(rates):>4}")
    print()

    # ==================================================================
    # 6. Cross-tabulation: user x video matrix of short-dt %
    # ==================================================================
    print("=" * 100)
    print("SECTION 6: User x Video Matrix — Short-dt (<8ms) Percentage")
    print("=" * 100)

    video_names = [mapping.get_str_code(v, "eye") for v in range(N_VIDEOS)]
    header6 = f"{'User':>5} " + " ".join(f"{n:>6}" for n in video_names) + f" {'Mean':>6}"
    print(header6)
    print("-" * len(header6))

    sorted_uids = sorted(all_data.keys())
    for uid in sorted_uids:
        vals = []
        for v in range(N_VIDEOS):
            dt = dt_by_user_video[uid][v]
            if len(dt) == 0:
                vals.append("   ---")
            else:
                rate = 100.0 * np.sum(dt < SHORT_DT_THRESHOLD) / len(dt)
                vals.append(f"{rate:>6.2f}")
        # Compute user mean
        numeric_vals = []
        for v in range(N_VIDEOS):
            dt = dt_by_user_video[uid][v]
            if len(dt) > 0:
                numeric_vals.append(100.0 * np.sum(dt < SHORT_DT_THRESHOLD) / len(dt))
        user_mean = np.mean(numeric_vals) if numeric_vals else float('nan')
        row = f"{uid:>5} " + " ".join(vals) + f" {user_mean:>6.2f}"
        print(row)

    # Column means
    col_means = []
    for v in range(N_VIDEOS):
        rates = video_user_rates[v]
        col_means.append(f"{np.mean(rates):>6.2f}" if rates else "   ---")
    print("-" * len(header6))
    print(f"{'Mean':>5} " + " ".join(col_means))
    print()

    print("Done.")


if __name__ == "__main__":
    main()
