# phase-analysis-demo
Scientific coding-agent practice using synthetic signals with known phase relationships.

Two sinusoids of the same known frequency are generated with a known phase
offset and adjustable Gaussian noise. Their phase difference is estimated,
compared with the known value, and plotted. Synthetic data only.

## Files

| File | Purpose |
|---|---|
| `phase_analysis.py` | Signal generation, phase-difference estimate, demonstration and plot |
| `test_phase_analysis.py` | Automated checks (pytest) |
| `requirements.txt` | Exact package versions frozen from the project `.venv` |
| `phase_demo.png` | Output of the default demonstration, committed for review |

## Method: coherent complex projection at a known frequency

- Sample times: `t[n] = n / fs`, `n = 0, ..., N-1` (endpoint `T` excluded).
- Signals: `x = A cos(2πft) + σ·noise_x`, `y = A cos(2πft + Δ) + σ·noise_y`.
  One random generator per call; `noise_x` is drawn first, then `noise_y`.
- Mean removal: each signal's record mean is subtracted before projection.
  Over an integer number of cycles the mean contributes nothing to `Z` in
  exact arithmetic, so this does not change the estimate for a sinusoid; it
  keeps a large DC offset from dominating rounding error and the
  undefined-phase threshold.
- Projection: `Z_x = Σ (x[n] − mean(x)) exp(-i2πf t[n])` (likewise `Z_y`),
  i.e. the record's Fourier coefficient at the known frequency `f`.
- Estimate: `Δ_est = wrap(angle(Z_y) − angle(Z_x))`. This equals
  `angle(Z_y · conj(Z_x))` mathematically but avoids overflow of the product
  for very large signals. It is one record-wide phase difference; no
  time-resolved phase is computed.
- Sign: `Δ = φ_y − φ_x`; positive means **y leads x**. Angles are in degrees,
  wrapped to (−180, 180].
- Signed error: `wrap(Δ_est − Δ_true)`, a circular difference
  (estimate −179°, truth +179° → +2°).
- SNR is reported per sample as signal power / noise power = `A² / (2σ²)`.
- Small-error approximation of the estimate's spread: `2σ / (A√N)` rad.

### Input requirements (violations raise `ValueError`; nothing is truncated)

- Finite parameters; `fs > 0`, `T > 0`, `A > 0`, `σ ≥ 0`.
- `N = fs·T` within 1e-9 of an integer, and `N ≥ 3`.
- `0 < f < fs/2`; `K = f·N/fs` within 1e-9 of an integer `k`, with `1 ≤ k`
  and `2k < N`.
- `x`, `y`: real, 1-D, finite, equal length.
- Undefined phase: with `x₀ = x − mean(x)`, if `Σ|x₀| = 0` or
  `|Z_x| ≤ 1e-10 · Σ|x₀|` (same for `y`). A constant (pure DC) signal is
  therefore rejected.

The 1e-9 integer tolerance and 1e-10 relative projection threshold are
chosen numerical guards, not universal error guarantees. They do not
establish adequate SNR: a noise-only signal passes the projection check and
yields an essentially random phase. Because the threshold is relative to the
mean-removed signal, a large DC offset no longer causes rejection, but very
large offsets still cost precision: storing `DC + signal` in float64 limits
the signal's resolution to about one ulp of `DC`.

Estimates near ±180° may land on either side of the boundary by rounding
(true +180° can return −179.99999999999991). Compare such values with the
circular error, not by raw value.

## Tests

- Deterministic (noiseless) cases use a 1e-8° absolute circular-error
  tolerance: offsets 0, +90, −90; an independent sign check
  (`x = cos`, `y = −sin` → +90°); wrap-equivalent inputs (270/−90,
  190/−170, 180/−180). This tolerance applies to these cases, not to every
  accepted input.
- `wrap_deg` (scalar and array) and signed circular error against
  hard-coded values. `wrap_deg` of a non-finite value returns `nan`; this is
  not tested.
- Regression tests for overflow at very large amplitude (1e152, 1e160), which
  previously returned `nan` or wrong finite values.
- Large DC offsets (1e10, 1e11) are accepted and accurate within a derived
  rounding bound of `4·ulp(DC)/A` rad.
- Rejection tests: zero, constant, and no-component-at-f signals (x and y
  sides); non-finite values (x and y); 2-D input; mismatched lengths; complex
  input; N < 3; non-integer N or K; K rounding to 0 or to the Nyquist count;
  f out of range, fs ≤ 0 and non-finite parameters (through both
  `make_signals` and the estimator for f and fs); A ≤ 0; σ < 0.
- Seeded noisy example (σ = 0.5, seed 0): exact reproducibility within the
  same environment; the noise for x and y is checked to be the first and
  second draws of `default_rng(0)`; error below 5 × the small-error
  approximate std (≈ 6.4°). That threshold comes from an approximation, not
  a guaranteed bound, and one seed cannot detect a bias of a few degrees.

## Running

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -v test_phase_analysis.py
.venv/bin/python phase_analysis.py
```

Default demonstration: f = 10 Hz, fs = 1000 Hz, T = 2 s (N = 2000, k = 20),
A = 1, Δ = +45°, σ = 0.5 (SNR = 2), seed 0.

## Recorded environment

Versions observed in the project `.venv` where the tests and demonstration
were run: Python 3.13.16 (Linux x86_64, glibc 2.39), numpy 2.5.4,
matplotlib 3.11.2, pytest 9.1.1. Full list in `requirements.txt`.

## Limitations

- Assumes `f` is known exactly, is identical in both signals, and that the
  phase relationship is constant across the record.
- Requires an integer number of cycles in the record; no windowing.
- Noise model: independent white Gaussian noise, equal σ in both signals.
- No Monte Carlo characterization of noise performance or bias yet
  (deferred). The default seed's +2.77° error is about 2.2 × the
  approximate std.
- The plot shows three noisy cycles with the noiseless signals overlaid
  (dashed) so the offset is visible; it is illustrative only.
