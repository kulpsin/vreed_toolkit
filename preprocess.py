#!/usr/bin/env python3

import logging
import pickle

import numpy

import mapping

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    pass

def validate_eye_tracking_data(raw_data):
    """Validate the raw data"""
    
    assert isinstance(raw_data, dict)  # Data should be dictionary
    assert "Data" in raw_data  # Data should contain "Data"
    assert "Labels" in raw_data  # Data should contain "Labels"

    # Data should not contain any other keys
    assert tuple(raw_data) == ("Data", "Labels") or tuple(raw_data) == ("Labels", "Data")

    ## Data validation
    data = raw_data["Data"]
    assert isinstance(data, list)  # Data format: list
    assert len(data) <= 12  # Maximum of 12 data items: All subjects did not finish all videos
    assert all(isinstance(item, numpy.ndarray) for item in data)  # Data in numpy.ndarray format
    for item in data:
        # The actual cells should contain values as numpy.float64
        assert all(isinstance(i, numpy.float64) for i in item.flatten())  
    assert all(item.shape[0] == 7 for item in data)  # Shape must be (7, x)
    assert all(item.ndim == 2 for item in data)  # Data dimension must be 2

    ## Label validation
    labels = raw_data["Labels"]
    assert isinstance(labels, numpy.ndarray)  # Label format: numpy.ndarray
    assert labels.size == 12

    return True


def load_eye_tracking_data(filename: str, validate: bool = False):
    """Loads the Eye Tracking data to memory"""

    with open(filename,  'rb')  as  FID:
        mp  = pickle.Unpickler(FID)
        raw_data = mp.load()
    if validate:
        if not validate_eye_tracking_data(raw_data):
            raise ValidationError("Eye data invalid")
    
    return raw_data["Labels"], raw_data["Data"]


def convert_timestamps_to_duration(data: list, convert_to_seconds: bool = False) -> None:
    """Inplace converts unix timestamps to duration as milliseconds
    starting from 0.0"""
    for video_idx, video_data in enumerate(data):
        # Remove first timestamp from all timestamps
        video_data[0] -= video_data[0][0]
        #
        if convert_to_seconds:
            # Divide the durations with 1000 (ms -> s)
            video_data[0] /= 1000


def replace_missing_values_with_nan(data: list) -> None:
    """Inplace replaces all missing values with numpy.NaN
    On this dataset, the missing values for X and Y coordinates 
    are 0.0. Changing these to NaN helps when calculating averages
    etc.
    """

    for video_idx, video_data in enumerate(data):
        # Left X
        video_data[1][video_data[1] == 0.0] = numpy.nan
        # Left Y
        video_data[2][video_data[2] == 0.0] = numpy.nan
        # Right X
        video_data[4][video_data[4] == 0.0] = numpy.nan
        # Right Y
        video_data[5][video_data[5] == 0.0] = numpy.nan
    logger.debug("All the missing X, Y values have been replaced with NaN")


def normalize_coordinates(data: list) -> None:
    """Inplace normalize spherical coordinates to azimuth [0,360), polar [0,180).

    The eye tracker occasionally reports values slightly outside the
    canonical ranges (e.g. negative polar when looking past the north
    pole, or azimuth > 360).  A negative polar angle of -2° means the
    gaze crossed 2° past the pole, which is equivalent to polar = 2°
    with azimuth flipped by 180°.

    Transformation:
      1. Reduce polar modulo 360 into [0, 360).
      2. If polar is in (180, 360), reflect: polar = 360 - polar,
         azimuth += 180.
      3. Reduce azimuth modulo 360 into [0, 360).

    NaN values are preserved.
    """
    for video_data in data:
        if video_data is None:
            continue
        for x_ch, y_ch in [(1, 2), (4, 5)]:
            az = video_data[x_ch]
            pol = video_data[y_ch]

            # Step 1: reduce polar into [0, 360)
            pol[:] = pol % 360

            # Step 2: reflect polar values in (180, 360) back into [0, 180)
            reflect = pol > 180
            if numpy.any(reflect):
                pol[reflect] = 360 - pol[reflect]
                az[reflect] = az[reflect] + 180

            # Step 3: reduce azimuth into [0, 360)
            az[:] = az % 360


def fix_swapped_channel_issue(data: list) -> None:
    """Inplace handle the swapped channel issue.
    Basically "sometimes" the left_blink and right_blink
    were swapped and this swaps them back.
    """
    for video_idx, video_data in enumerate(data):
        left_x_nan = numpy.isnan(video_data[1])
        left_y_nan = numpy.isnan(video_data[2])
        if numpy.count_nonzero(numpy.logical_xor(left_x_nan, left_y_nan)) != 0:
            raise Exception("Issue with left X and Y channels... Investigate!")
        left_is_blinking = video_data[3] == 1.0
        right_x_nan = numpy.isnan(video_data[4])
        right_y_nan = numpy.isnan(video_data[5])
        if numpy.count_nonzero(numpy.logical_xor(right_x_nan, right_y_nan)) != 0:
            raise Exception("Issue with right X and Y channels... Investigate!")
        right_is_blinking = video_data[6] == 1.0

        # Check that current channel is not correct:
        left_blink_xor = numpy.logical_xor(left_x_nan, left_is_blinking)
        left_blink_xor_n = numpy.count_nonzero(left_blink_xor)
        right_blink_xor = numpy.logical_xor(right_x_nan, right_is_blinking)
        right_blink_xor_n = numpy.count_nonzero(right_blink_xor)
        # Continue if everything seems ok
        if left_blink_xor_n == 0 and right_blink_xor_n == 0:
            continue
        # Unexpected case:
        if left_blink_xor_n != right_blink_xor_n:
            raise Exception("The blink channels have not swapped, but something else is wrong... Investigate!")

        # Check that the channels have indeed been swapped:
        if (left_x_nan == right_is_blinking).all() and (right_x_nan == left_is_blinking).all():
            pass
        else:
            raise Exception("The blink channels have not swapped, but something else is wrong2... Investigate!")
        
        # Make copy of left blink data:
        left_blink = numpy.copy(video_data[3])
        # Overwrite left blink data with right blink data:
        video_data[3] = video_data[6]
        # Overwrite right blink data with the copy of left blink data:
        video_data[6] = left_blink

        logger.info(f"blink left-right mixup have been fixed for video index {video_idx}")


def add_empty_data(labels: numpy.ndarray, data: list) -> None:
    """Users 118 and 130 did not watch everything.
    Add empty data so that indices are correct for these users' data."""
    original = numpy.array(
        [[1.0], [0.0], [0.0], [3.0], [1.0], [2.0], [2.0], [2.0], [0.0], [1.0], [3.0], [3.0]],
        dtype=numpy.float32,
        ndmin=2,
    )
    # Check if 118
    a = numpy.array(
        [[1.0], [0.0], [0.0], [1.0], [2.0], [2.0], [2.0], [0.0], [1.0], [3.0], [3.0], [0.0]],
        dtype=numpy.float32,
        ndmin=2,
    )
    if (labels == a).all():
        # Did not watch EXR
        for index in range(labels.size):
            labels[index][0] = original[index][0]
        data.insert(3, None)

        logger.info("Aligned the data for user 118")
        #print([type(d) for d in data])
        

    # Check if 130:
    a = numpy.array(
        [[2.0], [2.0], [2.0], [3.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        dtype=numpy.float32,
        ndmin=2,
    )
    if (labels == a).all():
        # PRS (index 5), BOT (index 6), RFS (index 7) and TNT (index 10)
        for index in range(labels.size):
            labels[index][0] = original[index][0]
        
        data.insert(0, None)
        data.insert(1, None)
        data.insert(2, None)
        data.insert(3, None)
        data.insert(4, None)
        data.insert(8, None)
        data.insert(9, None)
        data.append(None)

        logger.info("Aligned the data for user 130")
        #print([type(d) for d in data])


def clean_blink_boundaries(data, distance_ratio=5.0, min_distance=0.5):
    """NaN out gaze values adjacent to blinks if they show a velocity spike.

    During partial eye closure the tracker may report incorrect gaze
    positions.  For each NaN gap in gaze coordinates this function checks
    the last valid frame before the gap and the first valid frame after it.
    A boundary frame is flagged when:
      1. Its distance to the next valid neighbour exceeds *distance_ratio*
         times the distance one step further from the gap, AND
      2. That distance exceeds *min_distance* degrees (to avoid flagging
         measurement noise during fixations).

    Flagged frames have their X/Y channels (1, 2, 4, 5) set to NaN.
    Modifies data in-place.

    Parameters
    ----------
    data : list of ndarray or None
    distance_ratio : float
        Ratio threshold for boundary vs. context distance.
    min_distance : float
        Minimum great-circle distance (degrees) for a boundary frame to
        be considered erroneous.  Prevents false positives when the gaze
        is nearly stationary.
    """
    import gazelib

    total_flagged = 0

    for vdata in data:
        if vdata is None:
            continue

        az = gazelib.circular_nanmean_pair(vdata[1], vdata[4])
        pol = numpy.nanmean([vdata[2], vdata[5]], axis=0)
        is_nan = numpy.isnan(az) | numpy.isnan(pol)
        N = len(az)

        if N < 3:
            continue

        # Detect transitions: +1 = valid→NaN, -1 = NaN→valid
        changes = numpy.diff(is_nan.astype(numpy.int8))

        # valid→NaN at position j means frame j is last valid before gap
        before_gap = numpy.where(changes == 1)[0]
        # NaN→valid at position j means frame j+1 is first valid after gap
        after_gap = numpy.where(changes == -1)[0] + 1

        to_nan = set()

        # Frames before gaps: check triplet (a-2, a-1, a) where a is boundary
        for a in before_gap:
            if a < 2 or is_nan[a - 1] or is_nan[a - 2]:
                continue
            d_boundary = gazelib.great_circle_distance(
                az[a - 1], pol[a - 1], az[a], pol[a])
            if d_boundary <= min_distance:
                continue
            d_context = gazelib.great_circle_distance(
                az[a - 2], pol[a - 2], az[a - 1], pol[a - 1])
            if d_context > 0 and d_boundary > distance_ratio * d_context:
                to_nan.add(a)

        # Frames after gaps: check triplet (b, b+1, b+2) where b is boundary
        for b in after_gap:
            if b + 2 >= N or is_nan[b + 1] or is_nan[b + 2]:
                continue
            d_boundary = gazelib.great_circle_distance(
                az[b], pol[b], az[b + 1], pol[b + 1])
            if d_boundary <= min_distance:
                continue
            d_context = gazelib.great_circle_distance(
                az[b + 1], pol[b + 1], az[b + 2], pol[b + 2])
            if d_context > 0 and d_boundary > distance_ratio * d_context:
                to_nan.add(b)

        for idx in to_nan:
            vdata[1, idx] = numpy.nan
            vdata[2, idx] = numpy.nan
            vdata[4, idx] = numpy.nan
            vdata[5, idx] = numpy.nan

        total_flagged += len(to_nan)

    if total_flagged:
        logger.info(f"clean_blink_boundaries: flagged {total_flagged} frames")


if __name__ == "__main__":
    # Test that all preprocessing functions work
    pass
