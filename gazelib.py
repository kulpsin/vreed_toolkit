#!/usr/bin/env python3

"""Spherical geometry functions for gaze data.

Eye tracking coordinates are spherical: X is azimuth (0-360 deg) and Y is
polar angle (0-180 deg).  Naive numpy.diff() is wrong because:
  1. Azimuth wraps at 0/360 -> produces +-360 artifacts
  2. Near the poles (Y~0 or Y~180), large azimuth changes correspond to
     tiny actual movement

The correct metric is great-circle distance between consecutive gaze
vectors on the unit sphere.
"""

import numpy


def circular_nanmean_pair(a, b):
    """Circular-aware nanmean of two azimuth arrays (degrees, period 360).

    When the two values straddle the 0/360 boundary (|a - b| > 180),
    the shorter arc is used so that e.g. mean(359, 1) = 0 instead of 180.

    Parameters
    ----------
    a, b : array_like
        Two arrays of azimuth values in degrees.

    Returns
    -------
    ndarray
        Element-wise circular mean, in [0, 360).  NaN where both inputs
        are NaN; uses the non-NaN value where only one is NaN.
    """
    a = numpy.asarray(a, dtype=float)
    b = numpy.asarray(b, dtype=float)

    diff = a - b
    # Where the gap exceeds 180°, shift the smaller value up by 360
    shift = numpy.abs(diff) > 180
    a_shifted = numpy.where(shift & (a < b), a + 360, a)
    b_shifted = numpy.where(shift & (b < a), b + 360, b)

    result = numpy.nanmean([a_shifted, b_shifted], axis=0) % 360
    return result


def to_cartesian(azimuth_deg, polar_deg):
    """Convert spherical coordinates (degrees) to unit Cartesian vectors.

    Convention: azimuth = X channel (0-360), polar = Y channel (0-180).

    Returns (x, y, z) arrays.  NaN where input is NaN.
    """
    az = numpy.deg2rad(azimuth_deg)
    pol = numpy.deg2rad(polar_deg)
    sin_pol = numpy.sin(pol)
    x = sin_pol * numpy.cos(az)
    y = sin_pol * numpy.sin(az)
    z = numpy.cos(pol)
    return x, y, z


def great_circle_distance(az1, pol1, az2, pol2):
    """Angular distance in degrees between two gaze directions.

    Uses the Vincenty formula (stable for both small and large angles).
    All inputs in degrees.  NaN in either point -> NaN distance.
    """
    az1 = numpy.deg2rad(numpy.asarray(az1, dtype=float))
    pol1 = numpy.deg2rad(numpy.asarray(pol1, dtype=float))
    az2 = numpy.deg2rad(numpy.asarray(az2, dtype=float))
    pol2 = numpy.deg2rad(numpy.asarray(pol2, dtype=float))

    d_az = az2 - az1
    cos_d_az = numpy.cos(d_az)
    sin_pol1 = numpy.sin(pol1)
    sin_pol2 = numpy.sin(pol2)
    cos_pol1 = numpy.cos(pol1)
    cos_pol2 = numpy.cos(pol2)

    # Vincenty numerator (polar angle convention: sin↔cos swapped vs latitude)
    term1 = sin_pol2 * numpy.sin(d_az)
    term2 = sin_pol1 * cos_pol2 - cos_pol1 * sin_pol2 * cos_d_az
    numer = numpy.sqrt(term1**2 + term2**2)

    # Vincenty denominator (polar angle convention)
    denom = cos_pol1 * cos_pol2 + sin_pol1 * sin_pol2 * cos_d_az

    return numpy.rad2deg(numpy.arctan2(numer, denom))


def gaze_distance(vdata):
    """Great-circle distance between consecutive frames.

    Parameters
    ----------
    vdata : ndarray, shape (7, N)
        Single video data array (channels: Timestamp, Left_X, Left_Y,
        Left_Blink, Right_X, Right_Y, Right_Blink).

    Returns
    -------
    dt : ndarray, shape (N-1,)
        Time differences between consecutive frames.
    distances : ndarray, shape (N-1,)
        Great-circle distance in degrees between consecutive frames.
    """
    timestamps = vdata[0]
    az = circular_nanmean_pair(vdata[1], vdata[4])       # mean of Left_X, Right_X
    pol = numpy.nanmean([vdata[2], vdata[5]], axis=0)   # mean of Left_Y, Right_Y

    dt = numpy.diff(timestamps)
    distances = great_circle_distance(az[:-1], pol[:-1], az[1:], pol[1:])
    return dt, distances


def gaze_velocity(vdata, min_dt=None):
    """Gaze angular velocity in degrees per second.

    Parameters
    ----------
    vdata : ndarray, shape (7, N)
    min_dt : float or None
        If set, clamp dt to at least this value (seconds) before dividing.
        Prevents inflated velocities from anomalously short inter-frame
        intervals (e.g. 4 ms glitches).  Typical default: 0.010 (10 ms).

    Returns
    -------
    timestamps_mid : ndarray, shape (N-1,)
        Midpoints of consecutive timestamp pairs.
    velocity : ndarray, shape (N-1,)
        Angular velocity (deg/s).
    """
    dt, distances = gaze_distance(vdata)
    timestamps = vdata[0]
    timestamps_mid = (timestamps[:-1] + timestamps[1:]) / 2.0

    # Avoid division by zero
    dt_safe = numpy.where(dt == 0, numpy.nan, dt)
    # Clamp small dt values (NaN propagates through numpy.maximum)
    if min_dt is not None:
        dt_safe = numpy.maximum(dt_safe, min_dt)
    velocity = distances / dt_safe
    return timestamps_mid, velocity


def detect_outlier_frames(vdata, threshold=3.0):
    """Detect single-frame gaze outliers using the triplet detour ratio.

    For each interior frame i, computes:
        ratio = (d(i-1,i) + d(i,i+1)) / d(i-1,i+1)

    Smooth movement gives ratio ~1 (triangle equality for collinear points).
    A single erroneous frame produces ratio >> 1 (large detour and back).

    Parameters
    ----------
    vdata : ndarray, shape (7, N)
    threshold : float
        Detour ratio above which a frame is flagged.

    Returns
    -------
    outliers : ndarray of bool, shape (N,)
        True for frames detected as outliers.
    """
    az = circular_nanmean_pair(vdata[1], vdata[4])
    pol = numpy.nanmean([vdata[2], vdata[5]], axis=0)

    N = len(az)
    outliers = numpy.zeros(N, dtype=bool)
    if N < 3:
        return outliers

    # Consecutive distances: d[i] = distance(frame i, frame i+1)
    d_consec = great_circle_distance(az[:-1], pol[:-1], az[1:], pol[1:])

    # Skip distances: d_skip[i] = distance(frame i, frame i+2)
    d_skip = great_circle_distance(az[:-2], pol[:-2], az[2:], pol[2:])

    # For interior frame i (index 1..N-2):
    #   d_in  = d_consec[i-1]   (frame i-1 -> i)
    #   d_out = d_consec[i]     (frame i -> i+1)
    #   d_skip[i-1]             (frame i-1 -> i+1)
    d_in = d_consec[:-1]
    d_out = d_consec[1:]
    detour = d_in + d_out

    with numpy.errstate(divide='ignore', invalid='ignore'):
        ratio = detour / d_skip

    # Flag where ratio exceeds threshold (NaN ratios stay unflagged)
    outliers[1:-1] = numpy.where(numpy.isnan(ratio), False, ratio > threshold)

    return outliers


def gaze_velocity_robust(vdata, threshold=3.0, fill='nan', min_dt=None):
    """Gaze angular velocity with single-frame outlier tolerance.

    Detects single erroneous gaze frames via the triplet detour ratio
    and replaces the two affected velocity samples.

    Parameters
    ----------
    vdata : ndarray, shape (7, N)
    threshold : float
        Detour ratio threshold for outlier detection.
    fill : {'nan', 'split'}
        How to handle velocity samples adjacent to an outlier frame.
        'nan'  : set both to NaN.
        'split': assign d_skip/2 to each interval, so
                 v[i-1] = (d_skip/2) / dt_left,
                 v[i]   = (d_skip/2) / dt_right.
    min_dt : float or None
        If set, clamp dt to at least this value (seconds) before dividing.
        Prevents inflated velocities from anomalously short inter-frame
        intervals.  Passed through to gaze_velocity() and applied to
        the split fill computation.

    Returns
    -------
    timestamps_mid : ndarray, shape (N-1,)
    velocity : ndarray, shape (N-1,)
    outliers : ndarray of bool, shape (N,)
        Which frames were detected as outliers.
    """
    timestamps_mid, velocity = gaze_velocity(vdata, min_dt=min_dt)
    outliers = detect_outlier_frames(vdata, threshold)

    if not numpy.any(outliers):
        return timestamps_mid, velocity, outliers

    timestamps = vdata[0]
    az = circular_nanmean_pair(vdata[1], vdata[4])
    pol = numpy.nanmean([vdata[2], vdata[5]], axis=0)

    for i in numpy.where(outliers)[0]:
        # Affected velocity indices: i-1 (frame i-1 -> i) and i (frame i -> i+1)
        if fill == 'nan':
            if i > 0:
                velocity[i - 1] = numpy.nan
            if i < len(velocity):
                velocity[i] = numpy.nan
        elif fill == 'split':
            d_skip = great_circle_distance(
                az[i - 1], pol[i - 1], az[i + 1], pol[i + 1]
            )
            half_d = d_skip / 2.0

            dt_left = timestamps[i] - timestamps[i - 1]
            dt_right = timestamps[i + 1] - timestamps[i]
            dt_left = dt_left if dt_left != 0 else numpy.nan
            dt_right = dt_right if dt_right != 0 else numpy.nan
            if min_dt is not None:
                dt_left = max(dt_left, min_dt) if not numpy.isnan(dt_left) else dt_left
                dt_right = max(dt_right, min_dt) if not numpy.isnan(dt_right) else dt_right

            if i > 0:
                velocity[i - 1] = half_d / dt_left
            if i < len(velocity):
                velocity[i] = half_d / dt_right

    return timestamps_mid, velocity, outliers


if __name__ == "__main__":
    import preprocess

    DATA_PATH = "dataset/04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)"

    def load_user(user_id):
        filename = f"{DATA_PATH}/{user_id}_EyeTracking_PreProcessed.dat"
        labels, data = preprocess.load_eye_tracking_data(filename)
        preprocess.replace_missing_values_with_nan(data)
        preprocess.convert_timestamps_to_duration(data, convert_to_seconds=True)
        preprocess.fix_swapped_channel_issue(data)
        preprocess.add_empty_data(labels, data)
        return data

    # Quick demo: compare original vs robust on two recordings
    test_cases = [
        (118, 5, "PRS"),
        (119, 4, "JNG"),
    ]

    for user_id, vid_idx, vid_name in test_cases:
        data = load_user(user_id)
        vdata = data[vid_idx]
        if vdata is None:
            print(f"User {user_id} video {vid_idx} ({vid_name}): no data")
            continue

        _, gc_vel = gaze_velocity(vdata)
        print(f"User {user_id} video {vid_idx} ({vid_name}):  {vdata.shape[1]} frames")
        print(f"  Original — p99: {numpy.nanpercentile(gc_vel, 99):.1f} deg/s")

        for thresh in [2.0, 3.0, 5.0]:
            _, vel_spl, out = gaze_velocity_robust(vdata, threshold=thresh, fill='split')
            n_out = numpy.sum(out)
            print(f"  thresh={thresh:.1f} — outliers: {n_out:3d}  "
                  f"p99(split): {numpy.nanpercentile(vel_spl, 99):.1f}")
        print()
