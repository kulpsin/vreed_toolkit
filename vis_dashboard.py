#!/usr/bin/env python3

import logging

import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import preprocess
import mapping
import gazelib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "dataset/04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)"
USER_RANGE = range(101, 135)
NUM_VIDEOS = 12


def load_all_data():
    """Load and preprocess eye tracking data for all users.

    Returns dict[user_id] -> list of 12 entries (ndarray or None).
    """
    all_data = {}
    for user_id in USER_RANGE:
        filename = f"{DATA_PATH}/{user_id}_EyeTracking_PreProcessed.dat"
        labels, data = preprocess.load_eye_tracking_data(filename)
        preprocess.replace_missing_values_with_nan(data)
        preprocess.normalize_coordinates(data)
        preprocess.convert_timestamps_to_duration(data, convert_to_seconds=True)
        preprocess.fix_swapped_channel_issue(data)
        preprocess.add_empty_data(labels, data)
        preprocess.clean_blink_boundaries(data)
        all_data[user_id] = data
        logger.info(f"Loaded user {user_id}")
    return all_data


def plot_gaze_heatmaps(all_data):
    """Chart 1: 2D gaze position heatmaps per video (3x4 grid)."""
    fig, axes = plt.subplots(3, 4, figsize=(12.8, 7.2), dpi=150)
    fig.suptitle("Gaze Position Heatmaps (all users pooled)")

    for video_idx in range(NUM_VIDEOS):
        ax = axes[video_idx // 4, video_idx % 4]
        all_x = []
        all_y = []
        for user_id in USER_RANGE:
            vdata = all_data[user_id][video_idx]
            if vdata is None:
                continue
            gaze_x = gazelib.circular_nanmean_pair(vdata[1], vdata[4])
            gaze_y = numpy.nanmean([vdata[2], vdata[5]], axis=0)
            valid = ~numpy.isnan(gaze_x) & ~numpy.isnan(gaze_y)
            all_x.append(gaze_x[valid])
            all_y.append(gaze_y[valid])

        all_x = numpy.concatenate(all_x)
        all_y = numpy.concatenate(all_y)
        ax.hist2d(all_x, all_y, bins=50, range=[[0, 360], [0, 180]], cmap="hot")
        ax.set_title(mapping.get_str_code(video_idx, "eye"), fontsize=9)
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig("vis5_gaze_heatmaps.png")
    plt.close(fig)
    logger.info("Saved vis5_gaze_heatmaps.png")


def plot_blink_rate_over_time(all_data):
    """Chart 2: Blink rate over time with percentile bands, one plot per video."""
    window_sec = 30.0
    step_sec = 1.0

    for video_idx in range(NUM_VIDEOS):
        video_code = mapping.get_str_code(video_idx, "eye")
        user_rates = []

        for user_id in USER_RANGE:
            vdata = all_data[user_id][video_idx]
            if vdata is None:
                continue

            timestamps = vdata[0]
            both_blinking = (vdata[3] + vdata[6]) == 2.0
            state_changes = numpy.diff(both_blinking.astype(numpy.int8))
            # Blink starts: transition from not-blinking to blinking
            blink_start_indices = numpy.where(state_changes == 1)[0] + 1
            blink_times = timestamps[blink_start_indices]

            duration = timestamps[-1] - timestamps[0]
            if duration < window_sec:
                continue

            centers = numpy.arange(window_sec / 2, duration - window_sec / 2 + step_sec, step_sec)
            rates = numpy.empty(len(centers))
            for i, center in enumerate(centers):
                t_start = center - window_sec / 2
                t_end = center + window_sec / 2
                count = numpy.sum((blink_times >= t_start) & (blink_times < t_end))
                rates[i] = count * (60.0 / window_sec)  # blinks per minute

            user_rates.append((centers, rates))

        if not user_rates:
            continue

        # Align to common time grid
        max_len = max(len(r[0]) for r in user_rates)
        common_t = user_rates[0][0]
        for c, r in user_rates:
            if len(c) > len(common_t):
                common_t = c

        rate_matrix = numpy.full((len(user_rates), len(common_t)), numpy.nan)
        for i, (c, r) in enumerate(user_rates):
            rate_matrix[i, :len(r)] = r

        p25 = numpy.nanpercentile(rate_matrix, 25, axis=0)
        p50 = numpy.nanpercentile(rate_matrix, 50, axis=0)
        p75 = numpy.nanpercentile(rate_matrix, 75, axis=0)

        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
        ax.fill_between(common_t, p25, p75, alpha=0.4, color="b", label="p25–p75")
        ax.plot(common_t, p50, linewidth=1, color="b", label="Median")
        ax.set_title(f"Blink Rate Over Time — {video_code}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Blinks per minute")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"vis5_blink_rate_{video_idx:02d}_({video_code}).png")
        plt.close(fig)
        logger.info(f"Saved vis5_blink_rate_{video_idx:02d}_({video_code}).png")


def plot_vergence_boxplots(all_data):
    """Chart 3: Left vs Right eye vergence box plots per video."""
    x_verg_per_video = [[] for _ in range(NUM_VIDEOS)]
    y_verg_per_video = [[] for _ in range(NUM_VIDEOS)]

    for video_idx in range(NUM_VIDEOS):
        for user_id in USER_RANGE:
            vdata = all_data[user_id][video_idx]
            if vdata is None:
                continue
            dx =  vdata[4] - vdata[1]  #  Right_X - Left_X
            dy = vdata[5] - vdata[2]  #  Right_Y - Left_Y
            valid = ~numpy.isnan(dx)
            x_verg_per_video[video_idx].extend(dx[valid].tolist())
            valid = ~numpy.isnan(dy)
            y_verg_per_video[video_idx].extend(dy[valid].tolist())

    labels = [mapping.get_str_code(i, "eye") for i in range(NUM_VIDEOS)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=150)
    fig.suptitle("Left–Right Eye Vergence by Video")

    ax1.boxplot(x_verg_per_video, showfliers=False)
    ax1.set_xticklabels(labels, rotation=45, fontsize=8)
    ax1.set_title("Horizontal Vergence (Left_X − Right_X)")
    ax1.set_ylabel("Raw coordinate difference")

    ax2.boxplot(y_verg_per_video, showfliers=False)
    ax2.set_xticklabels(labels, rotation=45, fontsize=8)
    ax2.set_title("Vertical Vergence (Left_Y − Right_Y)")
    ax2.set_ylabel("Raw coordinate difference")

    fig.tight_layout()
    fig.savefig("vis5_vergence_boxplots.png")
    plt.close(fig)
    logger.info("Saved vis5_vergence_boxplots.png")


def plot_vergence_per_user(all_data):
    """Chart 3b: Left vs Right eye vergence box plots per user (all videos pooled)."""
    user_ids = list(USER_RANGE)
    x_verg_per_user = []
    y_verg_per_user = []

    for user_id in user_ids:
        ux = []
        uy = []
        for video_idx in range(NUM_VIDEOS):
            vdata = all_data[user_id][video_idx]
            if vdata is None:
                continue
            dx = vdata[1] - vdata[4]  # Left_X - Right_X
            dy = vdata[2] - vdata[5]  # Left_Y - Right_Y
            valid_x = ~numpy.isnan(dx)
            valid_y = ~numpy.isnan(dy)
            ux.append(dx[valid_x])
            uy.append(dy[valid_y])
        if ux:
            x_verg_per_user.append(numpy.concatenate(ux))
            y_verg_per_user.append(numpy.concatenate(uy))
        else:
            x_verg_per_user.append(numpy.array([]))
            y_verg_per_user.append(numpy.array([]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8, 7.2), dpi=150)
    fig.suptitle("Left–Right Eye Vergence by User (all videos pooled)")

    ax1.boxplot(x_verg_per_user, showfliers=False)
    ax1.set_xticklabels([str(uid) for uid in user_ids], rotation=90, fontsize=6)
    ax1.set_title("Horizontal Vergence (Left_X − Right_X)")
    ax1.set_ylabel("Raw coordinate difference")

    ax2.boxplot(y_verg_per_user, showfliers=False)
    ax2.set_xticklabels([str(uid) for uid in user_ids], rotation=90, fontsize=6)
    ax2.set_title("Vertical Vergence (Left_Y − Right_Y)")
    ax2.set_ylabel("Raw coordinate difference")

    fig.tight_layout()
    fig.savefig("vis5_vergence_per_user.png")
    plt.close(fig)
    logger.info("Saved vis5_vergence_per_user.png")


def plot_user_gaze_boxplots(all_data):
    """Chart 4: Per-user gaze range box plots (all videos pooled)."""
    user_ids = list(USER_RANGE)
    gaze_x_per_user = []
    gaze_y_per_user = []

    for user_id in user_ids:
        ux = []
        uy = []
        for video_idx in range(NUM_VIDEOS):
            vdata = all_data[user_id][video_idx]
            if vdata is None:
                continue
            gx = gazelib.circular_nanmean_pair(vdata[1], vdata[4])
            gy = numpy.nanmean([vdata[2], vdata[5]], axis=0)
            ux.append(gx[~numpy.isnan(gx)])
            uy.append(gy[~numpy.isnan(gy)])
        if ux:
            gaze_x_per_user.append(numpy.concatenate(ux))
            gaze_y_per_user.append(numpy.concatenate(uy))
        else:
            gaze_x_per_user.append(numpy.array([]))
            gaze_y_per_user.append(numpy.array([]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8, 7.2), dpi=150)
    fig.suptitle("Per-User Gaze Range (all videos pooled)")

    ax1.boxplot(gaze_x_per_user, showfliers=False)
    ax1.set_xticklabels([str(uid) for uid in user_ids], rotation=90, fontsize=6)
    ax1.set_title("Gaze X (raw coordinates)")

    ax2.boxplot(gaze_y_per_user, showfliers=False)
    ax2.set_xticklabels([str(uid) for uid in user_ids], rotation=90, fontsize=6)
    ax2.set_title("Gaze Y (raw coordinates)")

    fig.tight_layout()
    fig.savefig("vis5_user_gaze_boxplots.png")
    plt.close(fig)
    logger.info("Saved vis5_user_gaze_boxplots.png")


def plot_velocity_histograms(all_data):
    """Chart 5: Gaze velocity histograms per video (3x4 grid, log-Y)."""
    fig, axes = plt.subplots(3, 4, figsize=(12.8, 7.2), dpi=150)
    fig.suptitle("Gaze Velocity Distributions (all users pooled)")

    for video_idx in range(NUM_VIDEOS):
        ax = axes[video_idx // 4, video_idx % 4]
        all_vel = []

        for user_id in USER_RANGE:
            vdata = all_data[user_id][video_idx]
            if vdata is None:
                continue
            _, vel, _ = gazelib.gaze_velocity_robust(vdata, threshold=3.0, fill='split', min_dt=0.010)
            vel = vel[~numpy.isnan(vel) & numpy.isfinite(vel)]
            all_vel.append(vel)

        all_vel = numpy.concatenate(all_vel)
        ax.hist(all_vel, bins=100, log=True, color="steelblue", edgecolor="none")
        ax.set_title(mapping.get_str_code(video_idx, "eye"), fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("vel (deg/s)", fontsize=6)

    fig.tight_layout()
    fig.savefig("vis5_velocity_histograms.png")
    plt.close(fig)
    logger.info("Saved vis5_velocity_histograms.png")


def plot_gaze_percentile_bands(all_data):
    """Chart 6: Gaze position over time with percentile bands, one plot per video."""
    for video_idx in range(NUM_VIDEOS):
        video_code = mapping.get_str_code(video_idx, "eye")
        user_traces_x = []
        user_traces_y = []

        for user_id in USER_RANGE:
            vdata = all_data[user_id][video_idx]
            if vdata is None:
                continue
            gaze_x = gazelib.circular_nanmean_pair(vdata[1], vdata[4])
            gaze_y = numpy.nanmean([vdata[2], vdata[5]], axis=0)
            user_traces_x.append((vdata[0], gaze_x))
            user_traces_y.append((vdata[0], gaze_y))

        if not user_traces_x:
            continue

        # Align to the longest timestamp vector
        longest_t = max(user_traces_x, key=lambda tr: len(tr[0]))[0]
        n = len(longest_t)

        for axis_label, traces in [("X", user_traces_x), ("Y", user_traces_y)]:
            matrix = numpy.full((len(traces), n), numpy.nan)
            for i, (t, vals) in enumerate(traces):
                matrix[i, :len(vals)] = vals

            def pct(p):
                return numpy.nanpercentile(matrix, p, axis=0)

            fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
            ax.fill_between(longest_t, pct(0), pct(25), alpha=0.05, linewidth=0, color="b", label="0–25 / 75–100%")
            ax.fill_between(longest_t, pct(25), pct(40), alpha=0.4, linewidth=0, color="b", label="25–40 / 60–75%")
            ax.fill_between(longest_t, pct(40), pct(60), alpha=0.65, linewidth=0, color="b", label="40–60%")
            ax.fill_between(longest_t, pct(60), pct(75), alpha=0.4, linewidth=0, color="b")
            ax.fill_between(longest_t, pct(75), pct(100), alpha=0.05, linewidth=0, color="b")
            ax.plot(longest_t, pct(50), linewidth=0.5, color="b", label="Median")
            ax.set_title(f"Gaze {axis_label} Over Time — {video_code}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"Gaze {axis_label} (degrees)")
            ax.legend()
            fig.tight_layout()
            fname = f"vis5_gaze_{axis_label.lower()}_percentile_{video_idx:02d}_({video_code}).png"
            fig.savefig(fname)
            plt.close(fig)
            logger.info(f"Saved {fname}")


if __name__ == "__main__":
    logger.info("Loading all user data...")
    all_data = load_all_data()
    logger.info("Data loaded. Generating charts...")

    plot_gaze_heatmaps(all_data)
    plot_blink_rate_over_time(all_data)
    plot_vergence_boxplots(all_data)
    plot_vergence_per_user(all_data)
    plot_user_gaze_boxplots(all_data)
    plot_velocity_histograms(all_data)
    plot_gaze_percentile_bands(all_data)

    logger.info("All charts generated.")
