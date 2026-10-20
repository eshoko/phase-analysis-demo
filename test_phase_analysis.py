"""Tests for phase_analysis.py (synthetic data only).

Deterministic correctness checks use a 1e-8 degree absolute circular-error
tolerance for the specific cases below; this is not a claim about every
accepted input. wrap_deg and circular_error_deg cases are exact arithmetic on
small integers, so they are compared exactly.
"""

import math

import numpy as np
import pytest

from phase_analysis import (
    circular_error_deg,
    estimate_phase_difference,
    make_signals,
    predicted_std_deg,
    wrap_deg,
)

TOL_DEG = 1e-8
F, FS, T, A = 10.0, 1000.0, 2.0, 1.0  # N = 2000, k = 20


def noiseless_estimate(true_deg):
    _, x, y = make_signals(F, FS, T, A, true_deg, sigma=0.0, seed=0)
    return estimate_phase_difference(x, y, F, FS)


# ---------------------------------------------------------------- wrapping


@pytest.mark.parametrize(
    "angle, expected",
    [(-180, 180), (180, 180), (270, -90), (0, 0), (360, 0), (190, -170), (-190, 170)],
)
def test_wrap_deg_hard_coded(angle, expected):
    assert wrap_deg(angle) == expected


def test_wrap_deg_array():
    angles = np.array([-180.0, 180.0, 270.0, 0.0, 360.0, 190.0, -190.0, 540.0])
    expected = np.array([180.0, 180.0, -90.0, 0.0, 0.0, -170.0, 170.0, 180.0])
    out = wrap_deg(angles)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_wrap_deg_non_finite_returns_nan(value):
    # Requested behaviour: non-finite input gives nan (not an exception).
    with np.errstate(invalid="ignore"):
        assert math.isnan(wrap_deg(value))


@pytest.mark.parametrize(
    "angle",
    [
        -179.99999999999997,  # previously returned 180.00000000000003
        np.nextafter(180.0, np.inf),
        np.nextafter(-180.0, -np.inf),
        2.0**53 + 180.0,
        8.29161717e17,  # previously landed outside the range
        -4.07729686e17,
        1.7e308,
        -1.7e308,
        5e-324,
    ],
)
def test_regression_wrap_deg_range_and_exact_congruence(angle):
    # Regression (review close-out): the ceil-based formula could leave
    # (-180, 180] by rounding. Output must be in range and exactly congruent
    # to the input mod 360 (checked with exact rational arithmetic).
    from fractions import Fraction

    out = wrap_deg(angle)
    assert -180.0 < out <= 180.0
    assert (Fraction(float(angle)) - Fraction(out)) % 360 == 0


@pytest.mark.parametrize("estimate, truth, expected", [(-179, 179, 2), (179, -179, -2)])
def test_signed_circular_error(estimate, truth, expected):
    assert circular_error_deg(estimate, truth) == expected


# ---------------------------------------------------------------- noiseless


@pytest.mark.parametrize("true_deg", [0.0, 90.0, -90.0])
def test_noiseless_offsets(true_deg):
    est = noiseless_estimate(true_deg)
    assert abs(circular_error_deg(est, true_deg)) <= TOL_DEG
    assert -180 < est <= 180


def test_sign_independent_of_make_signals():
    # y = -sin(wt) = cos(wt + 90 deg), so y leads x by +90 degrees.
    t = np.arange(int(FS * T)) / FS
    x = np.cos(2 * np.pi * F * t)
    y = -np.sin(2 * np.pi * F * t)
    est = estimate_phase_difference(x, y, F, FS)
    assert abs(circular_error_deg(est, 90.0)) <= TOL_DEG


@pytest.mark.parametrize("amplitude", [1e152, 1e160])
@pytest.mark.parametrize("true_deg", [30.0, -120.0])
def test_regression_large_amplitude_no_overflow(amplitude, true_deg):
    # Regression (review #1): angle(z_y * conj(z_x)) overflowed, giving nan at
    # 1e160 and wrong finite values (45, -135) at 1e152.
    _, x, y = make_signals(F, FS, T, amplitude, true_deg, sigma=0.0)
    est = estimate_phase_difference(x, y, F, FS)
    assert abs(circular_error_deg(est, true_deg)) <= TOL_DEG


@pytest.mark.parametrize("amplitude", [1e306, 1e307, 1.7e308])
@pytest.mark.parametrize("true_deg", [30.0, -120.0])
def test_regression_float_limit_amplitude(amplitude, true_deg):
    # Regression (review close-out reproducer): near the float64 maximum the
    # projection sums overflowed. 1e306 raised a misleading "undefined" error;
    # 1e307 and 1.7e308 silently returned nan (the record mean overflowed).
    _, x, y = make_signals(F, FS, T, amplitude, true_deg, sigma=0.0)
    with np.errstate(all="raise"):  # any overflow/invalid op fails the test
        est = estimate_phase_difference(x, y, F, FS)
    assert abs(circular_error_deg(est, true_deg)) <= TOL_DEG


@pytest.mark.parametrize("dc", [1e10, 1e11])
def test_large_dc_offset_accepted_after_mean_removal(dc):
    # Review #2: before mean removal these were rejected as "undefined".
    # Tolerance is a rounding bound, not 1e-8: storing dc + cos and removing
    # the mean perturbs each sample by up to ~1 ulp(dc), so |dZ| <= N*ulp and
    # each phase moves by <= 2*ulp/A rad; the difference by <= 4*ulp/A rad.
    t = np.arange(int(FS * T)) / FS
    x = dc + A * np.cos(2 * np.pi * F * t)
    y = dc + A * np.cos(2 * np.pi * F * t + np.deg2rad(45.0))
    tol_deg = math.degrees(4 * np.spacing(dc) / A)
    est = estimate_phase_difference(x, y, F, FS)
    assert abs(circular_error_deg(est, 45.0)) <= tol_deg


@pytest.mark.parametrize("a, b", [(270.0, -90.0), (190.0, -170.0), (180.0, -180.0)])
def test_wrap_equivalent_inputs(a, b):
    est_a, est_b = noiseless_estimate(a), noiseless_estimate(b)
    assert abs(circular_error_deg(est_a, est_b)) <= TOL_DEG
    assert abs(circular_error_deg(est_a, b)) <= TOL_DEG
    assert -180 < est_a <= 180 and -180 < est_b <= 180


# ---------------------------------------------------------------- rejections


@pytest.mark.parametrize("side", ["x", "y"])
def test_rejects_zero_signal(side):
    _, sig, _ = make_signals(F, FS, T, A, 0.0, sigma=0.0)
    args = (np.zeros_like(sig), sig) if side == "x" else (sig, np.zeros_like(sig))
    with pytest.raises(ValueError, match=f"phase of {side} is undefined"):
        estimate_phase_difference(*args, F, FS)


def test_rejects_constant_signal():
    # A pure DC signal has no component at f once its mean is removed.
    _, x, _ = make_signals(F, FS, T, A, 0.0, sigma=0.0)
    with pytest.raises(ValueError, match="phase of y is undefined"):
        estimate_phase_difference(x, np.full_like(x, 5.0), F, FS)


@pytest.mark.parametrize("side", ["x", "y"])
def test_rejects_signal_with_no_component_at_f(side):
    # Scale-aware check: a pure 2f sinusoid projects to ~0 at f.
    t = np.arange(int(FS * T)) / FS
    at_f = np.cos(2 * np.pi * F * t)
    at_2f = np.cos(2 * np.pi * 2 * F * t)
    args = (at_2f, at_f) if side == "x" else (at_f, at_2f)
    with pytest.raises(ValueError, match=f"phase of {side} is undefined"):
        estimate_phase_difference(*args, F, FS)


def test_rejects_non_integer_sample_count():
    with pytest.raises(ValueError, match="integer sample count"):
        make_signals(F, FS, 2.0005, A, 0.0, sigma=0.0)  # N = 2000.5


def test_rejects_non_integer_cycles():
    with pytest.raises(ValueError, match="not an integer"):
        make_signals(10.25, FS, T, A, 0.0, sigma=0.0)  # K = 20.5


def test_rejects_too_few_samples():
    with pytest.raises(ValueError, match="N >= 3"):
        make_signals(0.25, 1.0, 2.0, A, 0.0, sigma=0.0)  # N = 2
    with pytest.raises(ValueError, match="N >= 3"):
        estimate_phase_difference([1.0, -1.0], [1.0, -1.0], 0.25, 1.0)


def test_rejects_cycle_count_rounding_to_zero():
    with pytest.raises(ValueError, match="at least 1"):
        make_signals(1e-13, FS, T, A, 0.0, sigma=0.0)  # K = 2e-13 -> k = 0


def test_rejects_cycle_count_rounding_to_nyquist():
    f = FS / 2 - 1e-12  # below fs/2, but K = 1000 - 2e-12 rounds to N/2
    assert f < FS / 2
    with pytest.raises(ValueError, match="Nyquist"):
        make_signals(f, FS, T, A, 0.0, sigma=0.0)


@pytest.mark.parametrize("f", [0.0, -10.0, FS / 2, 600.0])
def test_rejects_frequency_out_of_range(f):
    with pytest.raises(ValueError, match="0 < f < fs/2"):
        make_signals(f, FS, T, A, 0.0, sigma=0.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(amplitude=0.0), "amplitude"),
        (dict(amplitude=-1.0), "amplitude"),
        (dict(sigma=-0.1), "sigma"),
        (dict(phase_diff_deg=math.nan), "finite"),
        (dict(frequency=math.inf), "finite"),
        (dict(fs=0.0), "fs"),
        (dict(duration=0.0), "duration"),
    ],
)
def test_rejects_bad_parameters(kwargs, match):
    params = dict(frequency=F, fs=FS, duration=T, amplitude=A, phase_diff_deg=0.0, sigma=0.0)
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        make_signals(**params)


@pytest.mark.parametrize("side", ["x", "y"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_non_finite_signal_values(side, bad):
    _, x, y = make_signals(F, FS, T, A, 0.0, sigma=0.0)
    (x if side == "x" else y)[5] = bad
    with pytest.raises(ValueError, match=f"{side} contains non-finite"):
        estimate_phase_difference(x, y, F, FS)


def test_rejects_2d_signal():
    _, x, y = make_signals(F, FS, T, A, 0.0, sigma=0.0)
    with pytest.raises(ValueError, match="1-D"):
        estimate_phase_difference(x.reshape(2, -1), y.reshape(2, -1), F, FS)


@pytest.mark.parametrize(
    "f, fs, match",
    [
        (0.0, FS, "0 < f < fs/2"),
        (FS / 2, FS, "0 < f < fs/2"),
        (F, 0.0, "fs must be > 0"),
        (F, -FS, "fs must be > 0"),
        (10.25, FS, "not an integer"),
        (math.nan, FS, "finite"),
    ],
)
def test_estimator_rejects_bad_frequency_or_fs_directly(f, fs, match):
    _, x, y = make_signals(F, FS, T, A, 0.0, sigma=0.0)
    with pytest.raises(ValueError, match=match):
        estimate_phase_difference(x, y, f, fs)


def test_rejects_mismatched_lengths():
    _, x, y = make_signals(F, FS, T, A, 0.0, sigma=0.0)
    with pytest.raises(ValueError, match="same length"):
        estimate_phase_difference(x, y[:-1], F, FS)


def test_rejects_complex_input():
    _, x, y = make_signals(F, FS, T, A, 0.0, sigma=0.0)
    with pytest.raises(ValueError, match="real"):
        estimate_phase_difference(x.astype(complex), y, F, FS)


# ---------------------------------------------------------------- noisy example


def test_seeded_noisy_example_reproducible_and_broadly_accurate():
    sigma, true_deg = 0.5, 45.0
    t1, x1, y1 = make_signals(F, FS, T, A, true_deg, sigma=sigma, seed=0)
    t2, x2, y2 = make_signals(F, FS, T, A, true_deg, sigma=sigma, seed=0)
    # Reproducibility: exact equality, claimed only within the same environment.
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    # Noise for x and y is the first and second draw from one generator.
    rng = np.random.default_rng(0)
    noise_x, noise_y = rng.standard_normal(t1.size), rng.standard_normal(t1.size)
    assert np.array_equal(x1, A * np.cos(2 * np.pi * F * t1) + sigma * noise_x)
    assert np.array_equal(y1, A * np.cos(2 * np.pi * F * t1 + np.deg2rad(true_deg)) + sigma * noise_y)
    est1 = estimate_phase_difference(x1, y1, F, FS)
    est2 = estimate_phase_difference(x2, y2, F, FS)
    assert est1 == est2
    # Broad accuracy: 5 x small-error approximate std (~6.4 deg); not a guaranteed bound.
    threshold = 5 * predicted_std_deg(A, sigma, t1.size)
    assert abs(circular_error_deg(est1, true_deg)) < threshold
