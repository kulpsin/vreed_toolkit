import numpy as np
import pytest
import preprocess


def make_video_data(n_frames=10, left_xy=(180.0, 90.0), right_xy=(180.0, 90.0)):
    """Create a single video ndarray of shape (7, n_frames) with constant gaze."""
    arr = np.zeros((7, n_frames))
    arr[0] = np.linspace(1000, 2000, n_frames)  # timestamps
    arr[1] = left_xy[0]   # Left_X (azimuth)
    arr[2] = left_xy[1]   # Left_Y (polar)
    arr[3] = 0.0          # Left_Blink (not blinking)
    arr[4] = right_xy[0]  # Right_X
    arr[5] = right_xy[1]  # Right_Y
    arr[6] = 0.0          # Right_Blink
    return arr


def make_data_list(n_videos=12, **kwargs):
    """Create a list of 12 video arrays."""
    return [make_video_data(**kwargs) for _ in range(n_videos)]


# --- replace_missing_values_with_nan ---

class TestReplaceMissingValuesWithNan:
    def test_zeros_become_nan(self):
        data = make_data_list()
        # Set some XY coords to 0.0 (missing)
        data[0][1, 3] = 0.0
        data[0][2, 3] = 0.0
        data[0][4, 5] = 0.0
        data[0][5, 5] = 0.0
        preprocess.replace_missing_values_with_nan(data)
        assert np.isnan(data[0][1, 3])
        assert np.isnan(data[0][2, 3])
        assert np.isnan(data[0][4, 5])
        assert np.isnan(data[0][5, 5])

    def test_nonzero_values_preserved(self):
        data = make_data_list()
        preprocess.replace_missing_values_with_nan(data)
        # All values were 180.0/90.0, should still be there
        assert data[0][1, 0] == 180.0
        assert data[0][2, 0] == 90.0

    def test_timestamps_not_affected(self):
        data = make_data_list()
        # Set timestamp to 0 — should NOT become NaN
        data[0][0, 0] = 0.0
        preprocess.replace_missing_values_with_nan(data)
        assert data[0][0, 0] == 0.0

    def test_blink_channels_not_affected(self):
        data = make_data_list()
        # Blink channels (3, 6) with 0.0 should stay 0.0
        data[0][3, :] = 0.0
        data[0][6, :] = 0.0
        preprocess.replace_missing_values_with_nan(data)
        assert not np.any(np.isnan(data[0][3]))
        assert not np.any(np.isnan(data[0][6]))

    def test_modifies_inplace(self):
        data = make_data_list()
        data[0][1, 0] = 0.0
        original_ref = data[0]
        preprocess.replace_missing_values_with_nan(data)
        assert data[0] is original_ref


# --- convert_timestamps_to_duration ---

class TestConvertTimestampsToDuration:
    def test_starts_at_zero(self):
        data = make_data_list()
        preprocess.convert_timestamps_to_duration(data)
        for vdata in data:
            assert vdata[0, 0] == 0.0

    def test_duration_values(self):
        data = make_data_list(n_frames=5)
        data[0][0] = [1000, 1100, 1200, 1300, 1400]
        preprocess.convert_timestamps_to_duration(data)
        np.testing.assert_array_almost_equal(
            data[0][0], [0, 100, 200, 300, 400]
        )

    def test_convert_to_seconds(self):
        data = make_data_list(n_frames=3)
        data[0][0] = [5000, 6000, 7000]
        preprocess.convert_timestamps_to_duration(data, convert_to_seconds=True)
        np.testing.assert_array_almost_equal(data[0][0], [0.0, 1.0, 2.0])

    def test_without_seconds_keeps_milliseconds(self):
        data = make_data_list(n_frames=3)
        data[0][0] = [5000, 6000, 7000]
        preprocess.convert_timestamps_to_duration(data, convert_to_seconds=False)
        np.testing.assert_array_almost_equal(data[0][0], [0, 1000, 2000])

    def test_does_not_affect_other_channels(self):
        data = make_data_list()
        original_xy = data[0][1].copy()
        preprocess.convert_timestamps_to_duration(data)
        np.testing.assert_array_equal(data[0][1], original_xy)


# --- normalize_coordinates ---

class TestNormalizeCoordinates:
    def test_already_valid_unchanged(self):
        data = make_data_list(left_xy=(180.0, 90.0), right_xy=(180.0, 90.0))
        preprocess.normalize_coordinates(data)
        assert data[0][1, 0] == 180.0
        assert data[0][2, 0] == 90.0

    def test_negative_polar_reflected(self):
        """Negative polar should become positive with azimuth shifted by 180."""
        data = make_data_list(left_xy=(10.0, -5.0), right_xy=(10.0, -5.0))
        preprocess.replace_missing_values_with_nan(data)
        preprocess.normalize_coordinates(data)
        # -5 mod 360 = 355, which is in (180, 360), so polar = 360 - 355 = 5, az += 180
        np.testing.assert_almost_equal(data[0][2, 0], 5.0)
        np.testing.assert_almost_equal(data[0][1, 0], 190.0)

    def test_azimuth_wraps(self):
        """Azimuth >= 360 should wrap."""
        data = make_data_list(left_xy=(370.0, 90.0), right_xy=(370.0, 90.0))
        preprocess.replace_missing_values_with_nan(data)
        preprocess.normalize_coordinates(data)
        np.testing.assert_almost_equal(data[0][1, 0], 10.0)

    def test_nan_preserved(self):
        data = make_data_list()
        data[0][1, 0] = np.nan
        data[0][2, 0] = np.nan
        preprocess.normalize_coordinates(data)
        assert np.isnan(data[0][1, 0])
        assert np.isnan(data[0][2, 0])

    def test_polar_above_180_reflected(self):
        """Polar in (180, 360) should be reflected: polar = 360 - polar, az += 180."""
        data = make_data_list(left_xy=(0.0, 200.0), right_xy=(0.0, 200.0))
        # Don't call replace_missing_values_with_nan — 0.0 is a valid azimuth
        # in our synthetic data, not a missing value marker.
        preprocess.normalize_coordinates(data)
        np.testing.assert_almost_equal(data[0][2, 0], 160.0)  # 360 - 200
        np.testing.assert_almost_equal(data[0][1, 0], 180.0)  # 0 + 180

    # Parametrized edge cases for azimuth normalization.
    # Only azimuth is out of range; polar is a valid 90.0.
    @pytest.mark.parametrize("az_in, az_out", [
        (-10.0, 350.0),       # negative azimuth wraps
        (-180.0, 180.0),      # negative half-turn
        (-360.0, 0.0),        # negative full turn
        (-350.0, 10.0),       # nearly full negative turn
        (360.0, 0.0),         # exactly 360 wraps to 0
        (361.0, 1.0),         # just over 360
        (720.0, 0.0),         # double full turn
        (540.0, 180.0),       # one and a half turns
    ])
    def test_azimuth_edge_cases(self, az_in, az_out):
        data = make_data_list(left_xy=(az_in, 90.0), right_xy=(az_in, 90.0))
        preprocess.normalize_coordinates(data)
        np.testing.assert_almost_equal(data[0][1, 0], az_out)
        np.testing.assert_almost_equal(data[0][4, 0], az_out)
        # Polar should remain 90
        np.testing.assert_almost_equal(data[0][2, 0], 90.0)

    # Parametrized edge cases for polar normalization.
    # Azimuth is 100.0 (non-zero to avoid NaN issues with 0.0).
    # When polar reflects, azimuth gets +180.
    @pytest.mark.parametrize("pol_in, pol_out, az_in, az_out", [
        # Polar in valid range — no reflection, azimuth unchanged
        (0.0, 0.0, 100.0, 100.0),          # polar = 0 (north pole)
        (1.0, 1.0, 100.0, 100.0),          # just above 0
        (179.0, 179.0, 100.0, 100.0),      # just below 180
        # Polar at/above 180 — reflection triggers
        (180.0, 180.0, 100.0, 100.0),      # exactly 180: mod→180, not in (180,360)
        (181.0, 179.0, 100.0, 280.0),      # just over 180: reflect, az += 180
        (270.0, 90.0, 100.0, 280.0),       # three-quarter turn
        (359.0, 1.0, 100.0, 280.0),        # nearly 360
        # Negative polar — mod 360 first, then reflect if needed
        (-1.0, 1.0, 100.0, 280.0),         # -1 mod 360 = 359 → reflect → 1
        (-10.0, 10.0, 100.0, 280.0),       # -10 mod 360 = 350 → reflect → 10
        (-90.0, 90.0, 100.0, 280.0),       # -90 mod 360 = 270 → reflect → 90
        (-179.0, 179.0, 100.0, 280.0),     # -179 mod 360 = 181 → reflect → 179
        (-180.0, 180.0, 100.0, 100.0),     # -180 mod 360 = 180, not in (180,360)
        # Large values
        (360.0, 0.0, 100.0, 100.0),        # full turn mod → 0
        (361.0, 1.0, 100.0, 100.0),        # 361 mod 360 = 1, valid
        (540.0, 180.0, 100.0, 100.0),      # 540 mod 360 = 180, not reflected
        (541.0, 179.0, 100.0, 280.0),      # 541 mod 360 = 181 → reflect
    ])
    def test_polar_edge_cases(self, pol_in, pol_out, az_in, az_out):
        data = make_data_list(left_xy=(az_in, pol_in), right_xy=(az_in, pol_in))
        preprocess.normalize_coordinates(data)
        np.testing.assert_almost_equal(data[0][2, 0], pol_out)
        np.testing.assert_almost_equal(data[0][5, 0], pol_out)
        np.testing.assert_almost_equal(data[0][1, 0], az_out)

    def test_both_eyes_normalized_independently(self):
        """Left and right eye can have different out-of-range values."""
        data = [make_video_data(left_xy=(370.0, -5.0), right_xy=(-10.0, 200.0))]
        data += [make_video_data() for _ in range(11)]
        preprocess.normalize_coordinates(data)
        # Left eye: az 370→10, polar -5 mod 360=355→reflect→5, az += 180→190
        np.testing.assert_almost_equal(data[0][1, 0], 190.0)
        np.testing.assert_almost_equal(data[0][2, 0], 5.0)
        # Right eye: az -10→350, polar 200→reflect→160, az += 180→170
        np.testing.assert_almost_equal(data[0][4, 0], 170.0)
        np.testing.assert_almost_equal(data[0][5, 0], 160.0)

    def test_mixed_valid_and_invalid_frames(self):
        """Different frames in the same video can have different issues."""
        data = [make_video_data(n_frames=3)]
        data[0][1, :] = [180.0, 370.0, -20.0]  # azimuth: valid, >360, negative
        data[0][2, :] = [90.0, 90.0, 90.0]     # polar: all valid
        data[0][4, :] = [180.0, 370.0, -20.0]
        data[0][5, :] = [90.0, 90.0, 90.0]
        data += [make_video_data() for _ in range(11)]
        preprocess.normalize_coordinates(data)
        np.testing.assert_almost_equal(data[0][1, 0], 180.0)
        np.testing.assert_almost_equal(data[0][1, 1], 10.0)
        np.testing.assert_almost_equal(data[0][1, 2], 340.0)


# --- fix_swapped_channel_issue ---

class TestFixSwappedChannelIssue:
    def test_correct_channels_unchanged(self):
        """When blink flags match the NaN pattern, nothing should change."""
        data = make_data_list(n_frames=20)
        # Simulate a left blink: NaN in left XY, blink flag = 1
        data[0][1, 5:8] = np.nan
        data[0][2, 5:8] = np.nan
        data[0][3, 5:8] = 1.0
        preprocess.replace_missing_values_with_nan(data)
        left_blink_before = data[0][3].copy()
        right_blink_before = data[0][6].copy()
        preprocess.fix_swapped_channel_issue(data)
        np.testing.assert_array_equal(data[0][3], left_blink_before)
        np.testing.assert_array_equal(data[0][6], right_blink_before)

    def test_swapped_channels_fixed(self):
        """When blink flags are swapped relative to NaN pattern, they should be swapped back."""
        data = make_data_list(n_frames=20)
        # Left eye has NaN (blink) but RIGHT blink channel is set (swapped)
        data[0][1, 5:8] = np.nan
        data[0][2, 5:8] = np.nan
        data[0][6, 5:8] = 1.0  # wrong channel
        data[0][3, 5:8] = 0.0  # should have been 1
        preprocess.replace_missing_values_with_nan(data)
        preprocess.fix_swapped_channel_issue(data)
        # After fix, left blink (ch 3) should be 1 during the NaN frames
        np.testing.assert_array_equal(data[0][3, 5:8], [1.0, 1.0, 1.0])


# --- validate_eye_tracking_data ---

class TestValidateEyeTrackingData:
    def test_valid_data_passes(self):
        raw_data = {
            "Data": make_data_list(),
            "Labels": np.arange(12),
        }
        preprocess.validate_eye_tracking_data(raw_data)

    def test_missing_keys_raises(self):
        with pytest.raises(Exception):
            preprocess.validate_eye_tracking_data({"Data": []})

    def test_wrong_channels_raises(self):
        bad_arr = np.zeros((5, 10))  # 5 channels instead of 7
        raw_data = {
            "Data": [bad_arr] * 12,
            "Labels": np.arange(12),
        }
        with pytest.raises(Exception):
            preprocess.validate_eye_tracking_data(raw_data)


# --- add_empty_data ---

class TestAddEmptyData:
    def test_normal_user_unchanged(self):
        """For a normal user (not 118/130), data should remain 12 elements."""
        labels = np.arange(12)
        data = make_data_list()
        preprocess.add_empty_data(labels, data)
        assert len(data) == 12
        assert all(d is not None for d in data)

    def test_result_always_12_elements(self):
        labels = np.arange(12)
        data = make_data_list()
        preprocess.add_empty_data(labels, data)
        assert len(data) == 12
