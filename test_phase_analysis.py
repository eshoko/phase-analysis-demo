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


@pytest.mark.parametrize("a, b", [(270.0, -90.0), (190.0, -170.0), (180.0, -180.0)])
def test_wrap_equivalent_inputs(a, b):
    est_a, est_b = noiseless_estimate(a), noiseless_estimate(b)
    assert abs(circular_error_deg(est_a, est_b)) <= TOL_DEG
    assert abs(circular_error_deg(est_a, b)) <= TOL_DEG
    assert -180 < est_a <= 180 and -180 < est_b <= 180


# ---------------------------------------------------------------- rejections


def test_rejects_zero_signal():
    _, x, _ = make_signals(F, FS, T, A, 0.0, sigma=0.0)
    with pytest.raises(ValueError, match="undefined"):
        estimate_phase_difference(x, np.zeros_like(x), F, FS)


def test_rejects_signal_with_no_component_at_f():
    # Scale-aware check: a pure 2f sinusoid projects to ~0 at f.
    t = np.arange(int(FS * T)) / FS
    x = np.cos(2 * np.pi * F * t)
    y = np.cos(2 * np.pi * 2 * F * t)
    with pytest.raises(ValueError, match="undefined"):
        estimate_phase_difference(x, y, F, FS)


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


def test_rejects_non_finite_signal_values():
    _, x, y = make_signals(F, FS, T, A, 0.0, sigma=0.0)
    x[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        estimate_phase_difference(x, y, F, FS)


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
    # Separate noise draws for x and y from one generator.
    assert not np.array_equal(x1 - A * np.cos(2 * np.pi * F * t1),
                              y1 - A * np.cos(2 * np.pi * F * t1 + np.deg2rad(true_deg)))
    est1 = estimate_phase_difference(x1, y1, F, FS)
    est2 = estimate_phase_difference(x2, y2, F, FS)
    assert est1 == est2
    # Broad accuracy: 5 x small-error approximate std (~6.4 deg); not a guaranteed bound.
    threshold = 5 * predicted_std_deg(A, sigma, t1.size)
    assert abs(circular_error_deg(est1, true_deg)) < threshold
