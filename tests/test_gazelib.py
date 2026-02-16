import numpy as np
import pytest
import gazelib
import preprocess


def normalize(az, pol):
    """Normalize a single (az, pol) pair via preprocess.normalize_coordinates."""
    arr = np.zeros((7, 1))
    arr[0, 0] = 1000.0  # timestamp
    arr[1, 0] = az       # Left_X
    arr[2, 0] = pol      # Left_Y
    arr[4, 0] = az       # Right_X
    arr[5, 0] = pol      # Right_Y
    preprocess.normalize_coordinates([arr])
    return arr[1, 0], arr[2, 0]


class TestGreatCircleDistanceBasic:
    def test_same_point_zero_distance(self):
        d = gazelib.great_circle_distance(180.0, 90.0, 180.0, 90.0)
        np.testing.assert_almost_equal(d, 0.0)

    def test_antipodal_points(self):
        """Opposite points on the sphere should be 180 degrees apart."""
        d = gazelib.great_circle_distance(0.0, 90.0, 180.0, 90.0)
        np.testing.assert_almost_equal(d, 180.0)

    def test_pole_to_pole(self):
        """North pole (pol=0) to south pole (pol=180) = 180 degrees."""
        d = gazelib.great_circle_distance(0.0, 0.0, 0.0, 180.0)
        np.testing.assert_almost_equal(d, 180.0)

    def test_known_90_degree_distance(self):
        """Equator (pol=90) to north pole (pol=0) = 90 degrees."""
        d = gazelib.great_circle_distance(0.0, 90.0, 0.0, 0.0)
        np.testing.assert_almost_equal(d, 90.0)

    def test_symmetric(self):
        d1 = gazelib.great_circle_distance(30.0, 45.0, 200.0, 120.0)
        d2 = gazelib.great_circle_distance(200.0, 120.0, 30.0, 45.0)
        np.testing.assert_almost_equal(d1, d2)

    def test_nan_input_gives_nan(self):
        assert np.isnan(gazelib.great_circle_distance(np.nan, 90.0, 180.0, 90.0))
        assert np.isnan(gazelib.great_circle_distance(180.0, np.nan, 180.0, 90.0))
        assert np.isnan(gazelib.great_circle_distance(180.0, 90.0, np.nan, 90.0))
        assert np.isnan(gazelib.great_circle_distance(180.0, 90.0, 180.0, np.nan))


class TestGreatCircleDistanceAzimuthEdgeCases:
    """Test that distance is correct with out-of-range azimuth values,
    both raw and after normalization."""

    @pytest.mark.parametrize("az_raw", [
        -10.0, -180.0, -360.0, -350.0,
        360.0, 361.0, 720.0, 540.0,
    ])
    def test_raw_vs_normalized_point1(self, az_raw):
        """Distance from an out-of-range az point to a reference should match
        the distance after normalizing the out-of-range point."""
        ref_az, ref_pol = 100.0, 60.0
        pol = 90.0
        d_raw = gazelib.great_circle_distance(az_raw, pol, ref_az, ref_pol)
        az_norm, pol_norm = normalize(az_raw, pol)
        d_norm = gazelib.great_circle_distance(az_norm, pol_norm, ref_az, ref_pol)
        np.testing.assert_almost_equal(d_raw, d_norm)

    @pytest.mark.parametrize("az_raw", [
        -10.0, -180.0, -360.0, -350.0,
        360.0, 361.0, 720.0, 540.0,
    ])
    def test_raw_vs_normalized_point2(self, az_raw):
        """Same test but with the out-of-range value as the second argument."""
        ref_az, ref_pol = 100.0, 60.0
        pol = 90.0
        d_raw = gazelib.great_circle_distance(ref_az, ref_pol, az_raw, pol)
        az_norm, pol_norm = normalize(az_raw, pol)
        d_norm = gazelib.great_circle_distance(ref_az, ref_pol, az_norm, pol_norm)
        np.testing.assert_almost_equal(d_raw, d_norm)

    @pytest.mark.parametrize("az_raw", [
        -10.0, -180.0, -360.0, -350.0,
        360.0, 361.0, 720.0, 540.0,
    ])
    def test_equivalent_to_canonical(self, az_raw):
        """Distance from out-of-range az to itself-normalized should be 0."""
        pol = 90.0
        az_norm, pol_norm = normalize(az_raw, pol)
        d = gazelib.great_circle_distance(az_raw, pol, az_norm, pol_norm)
        np.testing.assert_almost_equal(d, 0.0)


class TestGreatCircleDistancePolarEdgeCases:
    """Test that distance is correct with out-of-range polar values,
    both raw and after normalization."""

    @pytest.mark.parametrize("pol_raw", [
        -1.0, -10.0, -90.0, -179.0, -180.0,
        180.0, 181.0, 270.0, 359.0,
        360.0, 361.0, 540.0, 541.0,
    ])
    def test_raw_vs_normalized_point1(self, pol_raw):
        """Distance from an out-of-range polar point to a reference should match
        the distance after normalizing."""
        ref_az, ref_pol = 100.0, 60.0
        az = 100.0
        d_raw = gazelib.great_circle_distance(az, pol_raw, ref_az, ref_pol)
        az_norm, pol_norm = normalize(az, pol_raw)
        d_norm = gazelib.great_circle_distance(az_norm, pol_norm, ref_az, ref_pol)
        np.testing.assert_almost_equal(d_raw, d_norm)

    @pytest.mark.parametrize("pol_raw", [
        -1.0, -10.0, -90.0, -179.0, -180.0,
        180.0, 181.0, 270.0, 359.0,
        360.0, 361.0, 540.0, 541.0,
    ])
    def test_raw_vs_normalized_point2(self, pol_raw):
        """Same test but with the out-of-range value as the second argument."""
        ref_az, ref_pol = 100.0, 60.0
        az = 100.0
        d_raw = gazelib.great_circle_distance(ref_az, ref_pol, az, pol_raw)
        az_norm, pol_norm = normalize(az, pol_raw)
        d_norm = gazelib.great_circle_distance(ref_az, ref_pol, az_norm, pol_norm)
        np.testing.assert_almost_equal(d_raw, d_norm)

    @pytest.mark.parametrize("pol_raw", [
        -1.0, -10.0, -90.0, -179.0, -180.0,
        180.0, 181.0, 270.0, 359.0,
        360.0, 361.0, 540.0, 541.0,
    ])
    def test_equivalent_to_canonical(self, pol_raw):
        """Distance from out-of-range polar to itself-normalized should be 0."""
        az = 100.0
        az_norm, pol_norm = normalize(az, pol_raw)
        d = gazelib.great_circle_distance(az, pol_raw, az_norm, pol_norm)
        np.testing.assert_almost_equal(d, 0.0)


class TestGreatCircleDistanceCombinedEdgeCases:
    """Test with both azimuth and polar out of range simultaneously."""

    @pytest.mark.parametrize("az_raw, pol_raw", [
        (370.0, -5.0),
        (-10.0, 200.0),
        (-180.0, -90.0),
        (720.0, 541.0),
        (540.0, -179.0),
        (-350.0, 359.0),
    ])
    def test_raw_vs_normalized(self, az_raw, pol_raw):
        ref_az, ref_pol = 100.0, 60.0
        d_raw = gazelib.great_circle_distance(az_raw, pol_raw, ref_az, ref_pol)
        az_norm, pol_norm = normalize(az_raw, pol_raw)
        d_norm = gazelib.great_circle_distance(az_norm, pol_norm, ref_az, ref_pol)
        np.testing.assert_almost_equal(d_raw, d_norm)

    @pytest.mark.parametrize("az_raw, pol_raw", [
        (370.0, -5.0),
        (-10.0, 200.0),
        (-180.0, -90.0),
        (720.0, 541.0),
        (540.0, -179.0),
        (-350.0, 359.0),
    ])
    def test_equivalent_to_canonical(self, az_raw, pol_raw):
        """Distance from out-of-range point to itself-normalized should be 0."""
        az_norm, pol_norm = normalize(az_raw, pol_raw)
        d = gazelib.great_circle_distance(az_raw, pol_raw, az_norm, pol_norm)
        np.testing.assert_almost_equal(d, 0.0)

    @pytest.mark.parametrize("az_raw, pol_raw", [
        (370.0, -5.0),
        (-10.0, 200.0),
        (-180.0, -90.0),
        (720.0, 541.0),
        (540.0, -179.0),
        (-350.0, 359.0),
    ])
    def test_both_points_out_of_range(self, az_raw, pol_raw):
        """Distance between two out-of-range points should match their normalized versions."""
        az2_raw, pol2_raw = -az_raw, -pol_raw
        d_raw = gazelib.great_circle_distance(az_raw, pol_raw, az2_raw, pol2_raw)
        az1_n, pol1_n = normalize(az_raw, pol_raw)
        az2_n, pol2_n = normalize(az2_raw, pol2_raw)
        d_norm = gazelib.great_circle_distance(az1_n, pol1_n, az2_n, pol2_n)
        np.testing.assert_almost_equal(d_raw, d_norm)
