#!/usr/bin/env python3
"""
Diagnostic: Investigate regularity / periodicity of short frame intervals
(<8 ms and <6 ms) in the VREED eye tracking data.

Sections:
  1. Prevalence per user and video
  2. Inter-arrival regularity (gap histogram in frame-count space)
  3. Periodic pattern detection (autocorrelation + modulo-N analysis)
  4. Time-domain regularity (time between consecutive short-dt events)
  5. Burst detection (consecutive short-dt frames)
  6. Per-user summary table
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
import mapping

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = "dataset/04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)"
THRESHOLD_8 = 0.008   # 8 ms in seconds
THRESHOLD_6 = 0.006   # 6 ms in seconds
VIDEO_CODES = mapping.EYE_TRACKING_VIDEO_LIST_STR  # length 12


# ── Helpers ──────────────────────────────────────────────────────────────

def load_user(uid):
    fname = os.path.join(DATA_DIR, f"{uid}_EyeTracking_PreProcessed.dat")
    labels, data = preprocess.load_eye_tracking_data(fname)
    preprocess.replace_missing_values_with_nan(data)
    preprocess.convert_timestamps_to_duration(data, convert_to_seconds=True)
    preprocess.fix_swapped_channel_issue(data)
    preprocess.add_empty_data(labels, data)
    return data


def dt_array(vdata):
    """Return array of frame-to-frame time intervals (seconds)."""
    return np.diff(vdata[0])


# ── Collect all user IDs ─────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*_EyeTracking_PreProcessed.dat")))
user_ids = [int(os.path.basename(f).split("_")[0]) for f in files]

# ── Storage ──────────────────────────────────────────────────────────────
# per-user-video records
records = []          # list of dicts
all_gaps_frames = []  # gap in frame-count between consecutive short-dt frames (pooled)
all_gaps_time = []    # gap in seconds between consecutive short-dt events (pooled)
autocorr_results = [] # (uid, vidx, best_lag, best_acorr)
modN_results = []     # (uid, vidx, best_N, best_frac)
burst_global_short = 0
burst_global_adjacent = 0

print("=" * 80)
print("  VREED  —  Short frame-interval regularity diagnostic")
print("=" * 80)

# ──────────────────────────────────────────────────────────────────────────
# SECTION 1 — Prevalence per user and video
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SECTION 1 — Prevalence per user and video")
print("=" * 80)

header = f"{'User':>6}  {'Video':<5}  {'Frames':>8}  {'<8ms':>7}  {'<6ms':>7}  {'%<8ms':>7}  {'%<6ms':>7}"
print(header)
print("-" * len(header))

for uid in user_ids:
    data = load_user(uid)
    for vidx in range(12):
        vdata = data[vidx]
        if vdata is None:
            continue
        dt = dt_array(vdata)
        n_frames = len(vdata[0])
        n_dt = len(dt)
        n8 = int(np.sum(dt < THRESHOLD_8))
        n6 = int(np.sum(dt < THRESHOLD_6))
        pct8 = 100.0 * n8 / n_dt if n_dt > 0 else 0
        pct6 = 100.0 * n6 / n_dt if n_dt > 0 else 0

        rec = dict(uid=uid, vidx=vidx, n_frames=n_frames, n_dt=n_dt,
                   n8=n8, n6=n6, pct8=pct8, pct6=pct6)
        records.append(rec)

        print(f"{uid:>6}  {VIDEO_CODES[vidx]:<5}  {n_frames:>8}  {n8:>7}  {n6:>7}  {pct8:>7.2f}  {pct6:>7.2f}")

# Summary by user
print("\n--- Per-user totals ---")
header2 = f"{'User':>6}  {'TotFrames':>10}  {'Tot<8ms':>8}  {'Tot<6ms':>8}  {'%<8ms':>7}  {'%<6ms':>7}  {'Free?':>6}"
print(header2)
print("-" * len(header2))

user_summaries = {}  # uid -> dict
for uid in user_ids:
    recs = [r for r in records if r["uid"] == uid]
    tot_dt = sum(r["n_dt"] for r in recs)
    tot8 = sum(r["n8"] for r in recs)
    tot6 = sum(r["n6"] for r in recs)
    pct8 = 100.0 * tot8 / tot_dt if tot_dt > 0 else 0
    pct6 = 100.0 * tot6 / tot_dt if tot_dt > 0 else 0
    free = "YES" if tot8 == 0 else "no"
    user_summaries[uid] = dict(tot_dt=tot_dt, tot8=tot8, tot6=tot6,
                               pct8=pct8, pct6=pct6)
    print(f"{uid:>6}  {tot_dt:>10}  {tot8:>8}  {tot6:>8}  {pct8:>7.2f}  {pct6:>7.2f}  {free:>6}")

# Variability across videos within a user
print("\n--- Within-user variability (std of %<8ms across videos) ---")
for uid in user_ids:
    recs = [r for r in records if r["uid"] == uid]
    pcts = [r["pct8"] for r in recs]
    if len(pcts) > 1:
        print(f"  User {uid}: mean={np.mean(pcts):.2f}%  std={np.std(pcts):.2f}%  "
              f"range=[{np.min(pcts):.2f}%, {np.max(pcts):.2f}%]")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 2 — Inter-arrival regularity (gaps between short-dt frames)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SECTION 2 — Inter-arrival regularity (gaps between short-dt frames)")
print("=" * 80)

for uid in user_ids:
    data = load_user(uid)
    for vidx in range(12):
        vdata = data[vidx]
        if vdata is None:
            continue
        dt = dt_array(vdata)
        short_idx = np.where(dt < THRESHOLD_8)[0]
        if len(short_idx) < 2:
            continue
        gaps = np.diff(short_idx)
        all_gaps_frames.extend(gaps.tolist())

# Global histogram
if all_gaps_frames:
    gaps_arr = np.array(all_gaps_frames)
    bins_edges = [1, 2, 3, 4, 5, 6, 11, 21, 51, 101, max(int(gaps_arr.max()) + 1, 102)]
    bin_labels = ["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "51-100", "100+"]
    counts, _ = np.histogram(gaps_arr, bins=bins_edges)

    print(f"\nGlobal gap histogram ({len(gaps_arr)} gaps total):")
    print(f"  {'Bin':<10} {'Count':>8}  {'%':>7}")
    print(f"  {'-'*10} {'-'*8}  {'-'*7}")
    for lab, cnt in zip(bin_labels, counts):
        print(f"  {lab:<10} {cnt:>8}  {100*cnt/len(gaps_arr):>7.2f}")

    print(f"\n  Global gap stats: mean={np.mean(gaps_arr):.2f}  median={np.median(gaps_arr):.1f}  "
          f"std={np.std(gaps_arr):.2f}  min={int(np.min(gaps_arr))}  max={int(np.max(gaps_arr))}")
else:
    print("  No short-dt gaps found.")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 3 — Periodic pattern detection
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SECTION 3 — Periodic pattern detection")
print("=" * 80)

MAX_LAG = 50
MOD_N_RANGE = range(2, 21)

print("\n--- Autocorrelation analysis (binary signal, lags 1-50) ---")
print(f"{'User':>6}  {'Video':<5}  {'#Short':>7}  {'BestLag':>8}  {'AutoCorr':>9}")
print("-" * 50)

for uid in user_ids:
    data = load_user(uid)
    for vidx in range(12):
        vdata = data[vidx]
        if vdata is None:
            continue
        dt = dt_array(vdata)
        short_mask = (dt < THRESHOLD_8).astype(np.float64)
        n_short = int(short_mask.sum())
        if n_short <= 100:
            continue

        # Autocorrelation of mean-centered binary signal
        sig = short_mask - short_mask.mean()
        var = np.dot(sig, sig)
        if var == 0:
            continue
        acorrs = []
        for lag in range(1, MAX_LAG + 1):
            if lag >= len(sig):
                acorrs.append(0.0)
            else:
                acorrs.append(np.dot(sig[:-lag], sig[lag:]) / var)
        best_lag = int(np.argmax(acorrs)) + 1
        best_ac = acorrs[best_lag - 1]
        autocorr_results.append((uid, vidx, best_lag, best_ac))
        print(f"{uid:>6}  {VIDEO_CODES[vidx]:<5}  {n_short:>7}  {best_lag:>8}  {best_ac:>9.4f}")

# Modulo-N analysis
print("\n--- Modulo-N analysis (fraction of short-dt at multiples of N) ---")
print(f"{'User':>6}  {'Video':<5}  {'#Short':>7}  {'BestN':>6}  {'Frac':>7}  {'Expected':>9}")
print("-" * 55)

for uid in user_ids:
    data = load_user(uid)
    for vidx in range(12):
        vdata = data[vidx]
        if vdata is None:
            continue
        dt = dt_array(vdata)
        short_idx = np.where(dt < THRESHOLD_8)[0]
        n_short = len(short_idx)
        if n_short <= 100:
            continue

        first = short_idx[0]
        best_N = None
        best_frac = 0.0
        for N in MOD_N_RANGE:
            at_mult = np.sum((short_idx - first) % N == 0)
            frac = at_mult / n_short
            if frac > best_frac:
                best_frac = frac
                best_N = N
        expected = 1.0 / best_N if best_N else 0
        modN_results.append((uid, vidx, best_N, best_frac))
        print(f"{uid:>6}  {VIDEO_CODES[vidx]:<5}  {n_short:>7}  {best_N:>6}  {best_frac:>7.3f}  {expected:>9.3f}")

# Summary of autocorrelation results
if autocorr_results:
    lags = [r[2] for r in autocorr_results]
    acs  = [r[3] for r in autocorr_results]
    print(f"\nAutocorrelation summary across {len(autocorr_results)} user-video pairs with >100 short-dt:")
    from collections import Counter
    lag_counts = Counter(lags)
    print(f"  Most common best-lag values: {lag_counts.most_common(10)}")
    print(f"  Mean best autocorrelation: {np.mean(acs):.4f}")
    print(f"  Max best autocorrelation:  {np.max(acs):.4f}")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 4 — Time-domain regularity
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SECTION 4 — Time-domain regularity (time between short-dt events)")
print("=" * 80)

per_user_time_gaps = {}  # uid -> list of gap times in ms

for uid in user_ids:
    data = load_user(uid)
    ugaps = []
    for vidx in range(12):
        vdata = data[vidx]
        if vdata is None:
            continue
        dt = dt_array(vdata)
        timestamps = vdata[0]
        short_idx = np.where(dt < THRESHOLD_8)[0]
        if len(short_idx) < 2:
            continue
        # Time of each short-dt event: use the timestamp of the frame *after* the short interval
        event_times = timestamps[short_idx + 1]
        time_gaps = np.diff(event_times) * 1000  # seconds -> ms
        ugaps.extend(time_gaps.tolist())
        all_gaps_time.extend(time_gaps.tolist())
    per_user_time_gaps[uid] = ugaps

print(f"\n--- Per-user time-gap stats (ms between consecutive short-dt events) ---")
print(f"{'User':>6}  {'N_gaps':>7}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'Min':>8}  {'Max':>10}")
print("-" * 70)
for uid in user_ids:
    gaps = per_user_time_gaps[uid]
    if len(gaps) == 0:
        print(f"{uid:>6}  {'---':>7}")
        continue
    g = np.array(gaps)
    print(f"{uid:>6}  {len(g):>7}  {np.mean(g):>8.1f}  {np.median(g):>8.1f}  "
          f"{np.std(g):>8.1f}  {np.min(g):>8.1f}  {np.max(g):>10.1f}")

# Global time-domain histogram
if all_gaps_time:
    gt = np.array(all_gaps_time)
    time_bins = [0, 10, 20, 30, 50, 100, 200, 500, 1000, 5000, 10000, max(int(gt.max()) + 1, 10001)]
    time_labels = ["0-10", "10-20", "20-30", "30-50", "50-100",
                   "100-200", "200-500", "500-1k", "1k-5k", "5k-10k", "10k+"]
    counts, _ = np.histogram(gt, bins=time_bins)
    print(f"\nGlobal time-gap histogram ({len(gt)} gaps):")
    print(f"  {'Bin (ms)':<12} {'Count':>8}  {'%':>7}")
    print(f"  {'-'*12} {'-'*8}  {'-'*7}")
    for lab, cnt in zip(time_labels, counts):
        print(f"  {lab:<12} {cnt:>8}  {100*cnt/len(gt):>7.2f}")

    print(f"\n  Global time-gap stats (ms): mean={np.mean(gt):.1f}  median={np.median(gt):.1f}  "
          f"std={np.std(gt):.1f}  min={np.min(gt):.2f}  max={np.max(gt):.1f}")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 5 — Burst detection
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SECTION 5 — Burst detection")
print("=" * 80)

for uid in user_ids:
    data = load_user(uid)
    for vidx in range(12):
        vdata = data[vidx]
        if vdata is None:
            continue
        dt = dt_array(vdata)
        short_mask = dt < THRESHOLD_8
        n_short = int(short_mask.sum())
        if n_short == 0:
            continue

        # A short-dt frame is "adjacent" if the next or previous dt is also short
        adjacent = 0
        for i in range(len(short_mask)):
            if not short_mask[i]:
                continue
            is_adj = False
            if i > 0 and short_mask[i - 1]:
                is_adj = True
            if i < len(short_mask) - 1 and short_mask[i + 1]:
                is_adj = True
            if is_adj:
                adjacent += 1

        burst_global_short += n_short
        burst_global_adjacent += adjacent

# Burst runs analysis
print("\nBurst-run analysis (consecutive short-dt runs):")
all_run_lengths = []
for uid in user_ids:
    data = load_user(uid)
    for vidx in range(12):
        vdata = data[vidx]
        if vdata is None:
            continue
        dt = dt_array(vdata)
        short_mask = dt < THRESHOLD_8
        # Find runs of consecutive True values
        if not np.any(short_mask):
            continue
        changes = np.diff(short_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        # Handle edge cases
        if short_mask[0]:
            starts = np.concatenate(([0], starts))
        if short_mask[-1]:
            ends = np.concatenate((ends, [len(short_mask)]))
        for s, e in zip(starts, ends):
            all_run_lengths.append(e - s)

if all_run_lengths:
    rl = np.array(all_run_lengths)
    print(f"  Total short-dt runs: {len(rl)}")
    print(f"  Run length stats: mean={np.mean(rl):.2f}  median={np.median(rl):.1f}  "
          f"max={int(np.max(rl))}  std={np.std(rl):.2f}")
    # Histogram of run lengths
    max_rl = min(int(np.max(rl)), 20)
    print(f"\n  Run-length distribution:")
    print(f"  {'Length':<8} {'Count':>8}  {'%':>7}")
    print(f"  {'-'*8} {'-'*8}  {'-'*7}")
    for length in range(1, max_rl + 1):
        cnt = int(np.sum(rl == length))
        if cnt > 0:
            print(f"  {length:<8} {cnt:>8}  {100*cnt/len(rl):>7.2f}")
    cnt_over = int(np.sum(rl > max_rl))
    if cnt_over > 0:
        print(f"  {'>' + str(max_rl):<8} {cnt_over:>8}  {100*cnt_over/len(rl):>7.2f}")

if burst_global_short > 0:
    frac = burst_global_adjacent / burst_global_short
    isolated = burst_global_short - burst_global_adjacent
    print(f"\n  Global burst summary:")
    print(f"    Total short-dt frames:    {burst_global_short}")
    print(f"    Adjacent to another short: {burst_global_adjacent}  ({100*frac:.1f}%)")
    print(f"    Isolated (not adjacent):   {isolated}  ({100*(1-frac):.1f}%)")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 6 — Per-user summary table
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SECTION 6 — Per-user summary")
print("=" * 80)

# Collect dominant period and burst fraction per user
user_dom_lag = {}
for uid, vidx, lag, ac in autocorr_results:
    user_dom_lag.setdefault(uid, []).append((lag, ac))

user_burst_frac = {}
for uid in user_ids:
    data = load_user(uid)
    total_short = 0
    total_adj = 0
    for vidx in range(12):
        vdata = data[vidx]
        if vdata is None:
            continue
        dt = dt_array(vdata)
        sm = dt < THRESHOLD_8
        ns = int(sm.sum())
        if ns == 0:
            continue
        adj = 0
        for i in range(len(sm)):
            if not sm[i]:
                continue
            is_adj = False
            if i > 0 and sm[i - 1]:
                is_adj = True
            if i < len(sm) - 1 and sm[i + 1]:
                is_adj = True
            if is_adj:
                adj += 1
        total_short += ns
        total_adj += adj
    user_burst_frac[uid] = total_adj / total_short if total_short > 0 else 0.0

print(f"\n{'User':>6}  {'Tot<8ms':>8}  {'%Frames':>8}  {'DomLag':>7}  {'MeanAC':>7}  {'BurstFrac':>10}")
print("-" * 55)
for uid in user_ids:
    s = user_summaries[uid]
    tot8 = s["tot8"]
    pct8 = s["pct8"]
    bf = user_burst_frac[uid]

    if uid in user_dom_lag:
        lags_acs = user_dom_lag[uid]
        # Weighted average lag by autocorrelation
        mean_lag = np.mean([l for l, _ in lags_acs])
        mean_ac = np.mean([a for _, a in lags_acs])
        dom_str = f"{mean_lag:>5.1f}"
        ac_str = f"{mean_ac:>7.4f}"
    else:
        dom_str = "    ---"
        ac_str = "    ---"

    print(f"{uid:>6}  {tot8:>8}  {pct8:>7.2f}%  {dom_str}  {ac_str}  {bf:>9.1%}")

# ──────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  FINAL SUMMARY — Key findings about short-dt regularity")
print("=" * 80)

total_8 = sum(s["tot8"] for s in user_summaries.values())
total_6 = sum(s["tot6"] for s in user_summaries.values())
total_dt = sum(s["tot_dt"] for s in user_summaries.values())
users_free = [uid for uid in user_ids if user_summaries[uid]["tot8"] == 0]
users_with = [uid for uid in user_ids if user_summaries[uid]["tot8"] > 0]

print(f"""
1. PREVALENCE:
   - {total_8} of {total_dt} frame intervals ({100*total_8/total_dt:.2f}%) are <8 ms.
   - {total_6} of {total_dt} ({100*total_6/total_dt:.2f}%) are <6 ms.
   - {len(users_free)} user(s) have ZERO short-dt frames: {users_free if users_free else 'none'}
   - {len(users_with)} user(s) have at least some short-dt frames.
""")

if all_gaps_frames:
    g = np.array(all_gaps_frames)
    most_common_gap = int(np.bincount(g.astype(int)).argmax()) if g.max() < 100000 else 'N/A'
    print(f"""2. INTER-ARRIVAL (frame-count):
   - Median gap: {np.median(g):.0f} frames, mean: {np.mean(g):.1f}, std: {np.std(g):.1f}
   - Most common gap: {most_common_gap} frame(s)
   - Gap=1 (consecutive short-dt): {int(np.sum(g==1))} ({100*np.sum(g==1)/len(g):.1f}% of gaps)
""")

if autocorr_results:
    from collections import Counter
    lags = [r[2] for r in autocorr_results]
    acs = [r[3] for r in autocorr_results]
    strength = 'Strong' if np.max(acs) > 0.1 else 'Weak/no'
    print(f"""3. PERIODICITY:
   - Autocorrelation analysis ran on {len(autocorr_results)} user-video pairs (>100 short-dt).
   - Most common dominant lag: {Counter(lags).most_common(3)}
   - Mean best autocorrelation: {np.mean(acs):.4f} (values near 0 = no periodicity)
   - {strength} periodic pattern detected (max AC = {np.max(acs):.4f}).
""")

if all_gaps_time:
    gt = np.array(all_gaps_time)
    cv = np.std(gt)/np.mean(gt)
    variability = 'High' if np.std(gt) > np.mean(gt) else 'Low'
    regularity = 'irregular' if np.std(gt) > 0.5*np.mean(gt) else 'somewhat regular'
    print(f"""4. TIME-DOMAIN:
   - Median time between short-dt events: {np.median(gt):.1f} ms
   - Mean: {np.mean(gt):.1f} ms, std: {np.std(gt):.1f} ms
   - {variability} variability (CV = {cv:.2f}) — {regularity} spacing.
""")

if burst_global_short > 0:
    bf = burst_global_adjacent / burst_global_short
    if bf > 0.5:
        burst_desc = 'Short-dt frames tend to come in bursts/clusters.'
    elif bf < 0.3:
        burst_desc = 'Short-dt frames are mostly isolated.'
    else:
        burst_desc = 'Mixed — some bursting, some isolated.'
    print(f"""5. BURSTS:
   - {burst_global_adjacent} of {burst_global_short} short-dt frames ({100*bf:.1f}%) are adjacent to another short-dt frame.
   - {burst_desc}
""")

print("=" * 80)
print("  END OF DIAGNOSTIC")
print("=" * 80)
