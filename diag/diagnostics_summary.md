# VREED Eye Tracking Data — Diagnostic Findings

Summary of diagnostic analyses performed on gaze velocity distributions and frame timing in the VREED eye tracking dataset (34 participants, 12 videos each, ~69.5 Hz sample rate).

## 1. Gaze Velocity Outliers

### Problem

Gaze velocity histograms showed extreme values (>2500 deg/s) that far exceed physiological saccade velocity (~300–700 deg/s). Two sources of error were identified.

### 1a. Single-Frame Tracking Errors (Triplet Detection)

**Pattern**: A single frame reports a gaze position far from the true position, then the next frame snaps back. This produces two velocity spikes (entering and leaving the erroneous frame).

**Detection**: The triplet detour ratio test compares the path through a frame against the direct skip:

```
ratio = (d(i-1,i) + d(i,i+1)) / d(i-1,i+1)
```

For smooth gaze movement, ratio ≈ 1. For a single-frame outlier, ratio >> 1.

**Implementation**: `gazelib.detect_outlier_frames()` and `gazelib.gaze_velocity_robust()` with configurable threshold (default 3.0) and fill mode (`'nan'` or `'split'`).

**Results across all 34 users at threshold 3.0**:

| Metric | Original | Robust (split) |
|---|---|---|
| Typical p99 reduction | — | 5–25% lower |
| Example: User 119 JNG p99 | 195.3 deg/s | 155.6 deg/s |
| Example: User 108 RPW p99 | 262.1 deg/s | 187.3 deg/s |

The algorithm correctly preserves real saccades (step changes where gaze moves and stays) while removing spike-and-return glitches.

### 1b. Blink Boundary Errors

**Pattern**: Frames immediately before or after a blink (NaN gap) report incorrect gaze positions due to partial eye closure. The tracker captures the eyelid rather than the pupil.

**Detection**: For each NaN gap boundary, compare the distance from the boundary frame to its valid neighbour against a context distance one step further from the gap. Flag when `d_boundary > distance_ratio × d_context` and `d_boundary > min_distance`.

**Implementation**: `preprocess.clean_blink_boundaries()` with configurable `distance_ratio` (default 5.0) and `min_distance` (default 0.5°).

**Results**: 3,870 boundary frames flagged across all users (ranging from 8 to 434 per user). This extends existing NaN gaps by one frame on each affected side.

### 1c. Surviving Extreme Velocities (>1000 deg/s)

After applying both the triplet filter and blink boundary cleaning, 48 velocity samples >1000 deg/s survive across the entire dataset. Investigation (`diag_extreme_velocity.py`) categorised them:

| Category | Count | % | Description |
|---|---|---|---|
| Real saccades + short dt | 31 | 65% | Normal 10–20° saccades sampled at an unusually short frame interval (3–6 ms) |
| Blink-boundary artifacts | 11 | 23% | Erroneous positions at blink edges (catchable by `clean_blink_boundaries`) |
| Sustained fast saccades | 4 | 8% | Genuine large head/eye movements spanning multiple frames |
| Consecutive bad frames | 2 | 4% | Multi-frame tracking errors (not caught by single-frame triplet test) |

**Key finding**: The dominant cause of extreme surviving velocities is **normal saccades amplified by short frame intervals**, not tracking errors.

## 2. Multi-Frame Tracking Artifacts

### Pattern

A short "island" of frames (typically 3–10 frames, ~50–150 ms) reports gaze at a completely different position, then returns to almost exactly the original position. Example (User 119, JNG, frames 7679–7686):

```
7679: az=123.5°  ← stable at ~124°
7680: az=140.9°  ← jump TO ~141°
7681–7684: ~141° (plateau, 5 frames, ~82 ms)
7685: az=123.9°  ← jump BACK to ~124°
7686: az=124.0°  ← stable at ~124° again
```

The round-trip return rules out a real saccade. Likely caused by the tracker momentarily locking onto a wrong feature (lens reflection, wrong pupil edge).

### Prevalence

Detection (`diag_short_dt.py`) of round-trip artifacts (jump >5°, return within 3° within 10 frames) found only **14 cases** across the entire dataset. These are rare.

The triplet detector cannot catch these because each frame within the plateau has consistent neighbours — no single frame looks anomalous.

## 3. Short Frame Intervals (dt < 8 ms)

### Prevalence

**154,713 frames** (4.22% of all 3.67M frame intervals) have dt < 8 ms (`diag_short_dt.py`).

| dt range | Count | % of short-dt |
|---|---|---|
| 0–1 ms | 1 | 0.0% |
| 1–2 ms | 187 | 0.1% |
| 2–3 ms | 606 | 0.4% |
| 3–4 ms | 9,296 | 6.0% |
| 4–5 ms | 20,751 | 13.4% |
| 5–6 ms | 8,932 | 5.8% |
| 6–7 ms | 54,564 | 35.3% |
| 7–8 ms | 60,376 | 39.0% |

The vast majority (74.3%) are in the 6–8 ms range. Very short intervals (<3 ms) are rare.

### Short dt vs. Tracking Errors

Only **601** short-dt frames (0.4%) co-occur with a large gaze distance (>2°). Short dt frames are overwhelmingly benign timing noise — the gaze position is correct, only the timestamp spacing is unusual.

### Per-Video Differences (`diag_short_dt_per_video.py`)

Short-dt rates differ significantly between videos (Kruskal-Wallis H = 316.65, p < 0.000001):

| Video | <8 ms rate | Video | <8 ms rate |
|---|---|---|---|
| **BNS** | **6.71%** | PRS | 3.71% |
| **JNG** | **5.44%** | RST | 3.54% |
| BOT | 4.21% | DST | 3.16% |
| RPW | 4.18% | ZMZ | 3.13% |
| RFS | 3.72% | TRT | 1.74% |
| | | BRZ | 1.34% |
| | | **EXR** | **0.89%** |

This is consistent across all users (everyone has highest rates for BNS/JNG, lowest for EXR/BRZ). The effect is a property of the recording, not the participant.

### Temporal Pattern Within Videos

Short-dt rates **ramp up** from the start toward the middle/second half of each video:
- BNS: 4.3% → 8.0%
- JNG: 3.0% → 6.4%
- BOT: 0.8% → 6.1%

This gradual increase suggests a hardware/firmware behaviour (possibly thermal drift or adaptive clock correction) rather than a content-driven effect.

### Regularity (`diag_short_dt_regularity.py`)

- **Weak periodicity** at ~14 frames (~200 ms): autocorrelation r ≈ 0.10 at lag 14 (and harmonic at 28). Not a strict clock.
- **Mostly isolated**: 85.6% of short-dt frames are not adjacent to another short-dt frame. Bursts are rare (7.7% pairs, max run length 3).
- **Highly irregular timing**: Median inter-arrival 195 ms, CV = 1.64.

One outlier: User 122 × RPW has 15.4% short-dt rate, far above all other user-video combinations.

### Timestamp Delay Hypothesis (`diag_short_dt_delay_hypothesis.py`)

**Hypothesis**: Short dt is caused by the previous frame's timestamp being stored late, shifting time from the next interval to the previous one.

| Test | Result |
|---|---|
| dt_before > dt_normal for short-dt frames | **82.4%** (vs 47.1% control) — **PASS** |
| 2-frame span ≈ 2 × dt_normal | Mean 26.1 ms, only 26.2% within ±2 ms of 28.8 ms — **FAIL** |
| Correlation(dt_short, dt_before) | -0.14 (weak) — **FAIL** |
| Very short subset (dt < 4 ms) | Pattern breaks down, spans often < 20 ms — **FAIL** |

**Verdict**: Partially supported. Two mechanisms are at play:

1. **Timestamp jitter** (majority of 6–8 ms cases): The previous frame's timestamp is recorded late, stealing time from the next interval. The 2-frame span is roughly normal. This covers the bulk of short-dt frames.

2. **Actual sampling irregularity** (rare, <4 ms cases): A sample genuinely arrives early or is inserted. The 2-frame span is also abnormally short. This produces the most extreme velocity artifacts.

## Diagnostic Scripts

| Script | Purpose |
|---|---|
| `diag_extreme_velocity.py` | Categorise velocity samples >1000 deg/s surviving robust filter |
| `diag_short_dt.py` | Short-dt prevalence, co-occurrence with gaze jumps, round-trip artifacts |
| `diag_short_dt_regularity.py` | Inter-arrival patterns, periodicity, bursts per user |
| `diag_short_dt_per_video.py` | Per-video short-dt rates, temporal ramp-up, user×video matrix |
| `diag_short_dt_delay_hypothesis.py` | Test whether short dt is caused by delayed previous timestamp |
