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
- Scaling: each signal is first multiplied by an exact power of two so its
  peak is ≤ 1. Phase does not depend on scale and this scaling is exact, so
  it changes no result; it prevents overflow for amplitudes up to the
  float64 maximum (tested at 1e306, 1e307 and 1.7e308).
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

`wrap_deg` uses an exact remainder, so for any finite input its output is
in (−180, 180] and exactly congruent to the input mod 360 (an earlier
formula could leave that range by rounding). Non-finite input (`inf`,
`−inf`, `nan`) returns `nan`.

## Tests

- Deterministic (noiseless) cases use a 1e-8° absolute circular-error
  tolerance: offsets 0, +90, −90; an independent sign check
  (`x = cos`, `y = −sin` → +90°); wrap-equivalent inputs (270/−90,
  190/−170, 180/−180). This tolerance applies to these cases, not to every
  accepted input.
- `wrap_deg` (scalar and array) and signed circular error against
  hard-coded values; `wrap_deg(inf / −inf / nan)` returns `nan`.
- Regression tests for `wrap_deg` range and exact congruence at values that
  previously left (−180, 180] (e.g. −179.99999999999997 → 180.00000000000003,
  inputs near 1e17, ±1.7e308).
- Regression tests for overflow at very large amplitude (1e152, 1e160), which
  previously returned `nan` or wrong finite values, and at the float64 limit
  (1e306, 1e307, 1.7e308), which previously raised a misleading "undefined"
  error or returned `nan`; these run with floating-point errors raised.
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

## Multi-seed check (recorded result, not a test)

A one-off measurement, separate from the automated tests, using the default
settings (true Δ = +45°, σ = 0.5, N = 2000) and seeds 0–399, run in the
project `.venv`:

| Quantity | Value |
|---|---|
| Mean signed error | +0.0082° |
| Standard error of the mean | 0.0623° |
| Approximate 95% CI (mean ± 1.96 SE) | [−0.114°, +0.130°] |
| SD of errors | 1.245° (small-error prediction 1.281°) |
| Seeds with \|error\| ≥ 2.772° (seed 0's error) | 3.0% |

No bias was detected at this resolution (about ±0.12°). This does not prove
the estimator is unbiased, and it covers only these settings. The
single-seed noisy test remains a limited check.

## Recorded environments

| Environment | Python | numpy | Role |
|---|---|---|---|
| Project `.venv` (this container) | 3.13.16 | **2.5.4** | Runs all tests and the demonstration; `requirements.txt` is frozen from it |
| Container system `python3` | 3.13.16 | 2.5.3 | Not used for tests or results |
| Independent reviewer's environment | not recorded | 2.5.3 (as reported in the review) | Reproduced the demo result (+47.7723°); not verified here |

`.venv` also has matplotlib 3.11.2 and pytest 9.1.1 (Linux x86_64,
glibc 2.39). Full list in `requirements.txt`.

## Limitations

- Assumes `f` is known exactly, is identical in both signals, and that the
  phase relationship is constant across the record.
- Requires an integer number of cycles in the record; no windowing.
- Noise model: independent white Gaussian noise, equal σ in both signals.
- Bias and noise performance were checked only once, at the default
  settings over 400 seeds (see above); no larger simulation was run. The
  default seed's +2.77° error is about 2.2 × the approximate std.
- The plot shows three noisy cycles with the noiseless signals overlaid
  (dashed) so the offset is visible; it is illustrative only.
