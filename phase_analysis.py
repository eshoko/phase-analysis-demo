"""Phase difference between two synthetic same-frequency sinusoids.

Method: coherent complex projection at a known frequency f. Each signal is
projected onto exp(-i 2 pi f t) over the whole record, giving one complex
coefficient per signal; the angle of Z_y * conj(Z_x) is a single, record-wide
phase-difference estimate. No time-resolved phase is computed.

Conventions
-----------
- Sample times: t[n] = n / fs for n = 0, ..., N-1 (the endpoint T is excluded).
- Phase difference: Delta = phi_y - phi_x. Positive means y leads x.
- Angles are reported in degrees, wrapped to the interval (-180, 180].
- Signed error: wrap_deg(estimate - truth).

Synthetic data only. Run ``python phase_analysis.py`` for the demonstration.
"""

import math

import numpy as np

# Chosen numerical guards (not universal error guarantees).
# INTEGER_TOL: how close fs*T and f*N/fs must be to an integer to be accepted.
# PROJECTION_RTOL: |Z| <= PROJECTION_RTOL * sum(|x|) is treated as "no
# measurable component at f", so the phase is undefined and we raise.
INTEGER_TOL = 1e-9
PROJECTION_RTOL = 1e-10

# Default demonstration settings.
DEFAULTS = dict(
    frequency=10.0,  # Hz
    fs=1000.0,  # Hz
    duration=2.0,  # s  -> N = 2000 samples, k = 20 cycles
    amplitude=1.0,
    phase_diff_deg=45.0,
    sigma=0.5,
    seed=0,
)


# ---------------------------------------------------------------- angles


def wrap_deg(angle_deg):
    """Wrap angle(s) in degrees to the interval (-180, 180]."""
    a = np.asarray(angle_deg, dtype=float)
    wrapped = a - 360.0 * np.ceil((a - 180.0) / 360.0)
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def circular_error_deg(estimate_deg, truth_deg):
    """Signed circular error estimate - truth, wrapped to (-180, 180]."""
    return wrap_deg(np.asarray(estimate_deg, dtype=float) - np.asarray(truth_deg, dtype=float))


# ---------------------------------------------------------------- checks


def _finite_scalar(name, value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _sample_count(fs, duration):
    """Return N = fs*T as an int, requiring it to be (nearly) an integer and >= 3."""
    n_float = fs * duration
    n = round(n_float)
    if abs(n_float - n) > INTEGER_TOL:
        raise ValueError(
            f"fs*T = {n_float!r} is not an integer sample count "
            f"(tolerance {INTEGER_TOL}); refusing to truncate"
        )
    if n < 3:
        raise ValueError(f"need N >= 3 samples, got N = {n}")
    return int(n)


def _cycle_count(frequency, fs, n):
    """Check 0 < f < fs/2 and that K = f*N/fs is an integer k with 1 <= k, 2k < N."""
    if not 0.0 < frequency < fs / 2.0:
        raise ValueError(f"need 0 < f < fs/2, got f = {frequency}, fs = {fs}")
    cycles = frequency * n / fs
    k = round(cycles)
    if abs(cycles - k) > INTEGER_TOL:
        raise ValueError(
            f"record holds K = f*N/fs = {cycles!r} cycles, not an integer "
            f"(tolerance {INTEGER_TOL})"
        )
    if k < 1:
        raise ValueError(f"K = {cycles!r} rounds to {k} cycles; need at least 1")
    if 2 * k >= n:
        raise ValueError(f"K = {cycles!r} rounds to the Nyquist cycle count (2k >= N = {n})")
    return int(k)


def _signal_array(name, values):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


# ---------------------------------------------------------------- signals


def make_signals(frequency, fs, duration, amplitude, phase_diff_deg, sigma, seed=None):
    """Generate x(t) = A cos(2 pi f t) + noise_x and y(t) = A cos(2 pi f t + Delta) + noise_y.

    Noise is white Gaussian with standard deviation ``sigma``. One random
    generator is created per call; noise for x is drawn first, then for y.

    Returns (t, x, y).
    """
    frequency = _finite_scalar("frequency", frequency)
    fs = _finite_scalar("fs", fs)
    duration = _finite_scalar("duration", duration)
    amplitude = _finite_scalar("amplitude", amplitude)
    phase_diff_deg = _finite_scalar("phase_diff_deg", phase_diff_deg)
    sigma = _finite_scalar("sigma", sigma)
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")
    if amplitude <= 0:
        raise ValueError(f"amplitude must be > 0, got {amplitude}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")

    n = _sample_count(fs, duration)
    _cycle_count(frequency, fs, n)

    t = np.arange(n) / fs
    omega_t = 2.0 * np.pi * frequency * t
    rng = np.random.default_rng(seed)
    noise_x = rng.standard_normal(n)
    noise_y = rng.standard_normal(n)
    x = amplitude * np.cos(omega_t) + sigma * noise_x
    y = amplitude * np.cos(omega_t + np.deg2rad(phase_diff_deg)) + sigma * noise_y
    return t, x, y


# ---------------------------------------------------------------- estimate


def _projection(name, signal, frequency, fs):
    t = np.arange(signal.size) / fs
    z = np.sum(signal * np.exp(-2j * np.pi * frequency * t))
    scale = np.sum(np.abs(signal))
    if scale == 0.0 or abs(z) <= PROJECTION_RTOL * scale:
        raise ValueError(
            f"phase of {name} is undefined: projection at f is zero or negligible "
            f"(|Z| = {abs(z):.3g}, sum|{name}| = {scale:.3g})"
        )
    return z


def estimate_phase_difference(x, y, frequency, fs):
    """Record-wide phase difference phi_y - phi_x (degrees, in (-180, 180]).

    Uses coherent complex projection at the known frequency over the whole
    record, sampled at t[n] = n / fs. Requires an integer number of cycles.
    """
    frequency = _finite_scalar("frequency", frequency)
    fs = _finite_scalar("fs", fs)
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")
    x = _signal_array("x", x)
    y = _signal_array("y", y)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same length, got {x.size} and {y.size}")
    n = x.size
    if n < 3:
        raise ValueError(f"need N >= 3 samples, got N = {n}")
    _cycle_count(frequency, fs, n)

    z_x = _projection("x", x, frequency, fs)
    z_y = _projection("y", y, frequency, fs)
    return wrap_deg(np.rad2deg(np.angle(z_y * np.conj(z_x))))


# ---------------------------------------------------------------- reporting


def snr(amplitude, sigma):
    """Per-sample SNR = signal power / noise power = A**2 / (2 sigma**2)."""
    if sigma == 0:
        return math.inf
    return amplitude**2 / (2.0 * sigma**2)


def predicted_std_deg(amplitude, sigma, n):
    """Small-error approximation of std(Delta_est): 2 sigma / (A sqrt(N)) rad, in degrees."""
    return math.degrees(2.0 * sigma / (amplitude * math.sqrt(n)))


def plot_signals(t, x, y, frequency, true_deg, est_deg, err_deg, path, n_cycles=3):
    """Save a plot of the first few cycles of both signals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = t < n_cycles / frequency
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.plot(t[show], x[show], color="#2a78d6", linewidth=1.0, label="x (reference)")
    ax.plot(t[show], y[show], color="#eb6834", linewidth=1.0, label="y")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(
        f"True Δ = {true_deg:+.2f}°   estimated Δ = {est_deg:+.2f}°   error = {err_deg:+.2f}°",
        fontsize=10,
    )
    ax.grid(True, color="#e0e0e0", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(path="phase_demo.png"):
    p = DEFAULTS
    t, x, y = make_signals(**p)
    est = estimate_phase_difference(x, y, p["frequency"], p["fs"])
    err = circular_error_deg(est, p["phase_diff_deg"])
    true_wrapped = wrap_deg(p["phase_diff_deg"])

    print(f"Settings: f = {p['frequency']} Hz, fs = {p['fs']} Hz, T = {p['duration']} s, "
          f"N = {t.size}, A = {p['amplitude']}, sigma = {p['sigma']}, seed = {p['seed']}")
    print(f"True phase difference (y - x):     {true_wrapped:+.6f} deg")
    print(f"Estimated phase difference:        {est:+.6f} deg")
    print(f"Signed error (estimate - true):    {err:+.6f} deg")
    print(f"SNR = A^2 / (2 sigma^2):           {snr(p['amplitude'], p['sigma']):.3f}")
    print(f"Small-error approx. std of est.:   {predicted_std_deg(p['amplitude'], p['sigma'], t.size):.3f} deg")

    plot_signals(t, x, y, p["frequency"], true_wrapped, est, err, path)
    print(f"Plot written to {path}")


if __name__ == "__main__":
    main()
