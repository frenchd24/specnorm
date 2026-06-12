# specnorm

Interactive continuum normalization for astronomical spectra, tailored to
**HST/STIS** and **HST/COS** data but flexible enough for generic FITS and
ASCII spectra.

The spectrum is presented one wavelength window at a time (default 20 A). You
mask out contaminated regions (geocoronal Ly-alpha airglow, etc.), click where
the continuum should be anchored, choose a model (cubic spline, polynomial up
to degree 5, or Chebyshev series), inspect the fit, and accept it to move on.
Accepted windows are blended smoothly across their overlaps, and the result is
written out with `WAVELENGTH`, `FLUX`, `NORM_FLUX`, and `ERROR` columns. An
intermediate masked spectrum is saved separately as you work.

## Installation

```bash
pip install .            # from this directory
```

Requires Python >= 3.9 with numpy, scipy, matplotlib, and astropy.

## Quick start

```bash
specnorm oc8c11020_x1d.fits --airglow -o oc8c11020_norm.fits
```

`--airglow` pre-masks Ly-alpha (1213-1218.5), the OI triplet (1301-1307), and
OI] 1355 before the GUI opens. Add custom regions with repeated
`--mask W0:W1` flags. Data are binned x2 by default (`-b 1` to disable,
`-b N` for other factors).

or from Python:

```python
from specnorm import read_spectrum, normalize_interactive, write_spectrum

from specnorm import bin_spectrum, AIRGLOW_REGIONS

spec = bin_spectrum(read_spectrum("lcr301010_x1d.fits"), 2)   # COS x1d, binned x2
result = normalize_interactive(
    spec, window=20.0, fitter="cheb", degree=4,
    low_rej=1.5, high_rej=3.5, niterate=20, grow=6, min_pix=3,  # clipping
    mask_regions=AIRGLOW_REGIONS,
    masked_path="lcr301010_masked.fits",       # intermediate masked file
)
write_spectrum(result, "lcr301010_norm.fits")
```

## What it reads

`read_spectrum` auto-detects the layout:

- **STIS / COS x1d, sx1 files** — bintables with one row per echelle order or
  detector segment and vector `WAVELENGTH` / `FLUX` / `ERROR` / `DQ` columns.
  Rows are concatenated and sorted in wavelength; non-zero DQ pixels are
  masked from node placement (disable with `--no-dq-mask`). Fixed-length
  vector rows are trimmed to their `NELEM` valid elements, and any remaining
  fill/padding pixels with wavelength <= 0 (zero or sentinel values in
  unused row ends or dead orders) are dropped at read time and reported, so
  they never reach the fit or the output file.
- **Generic FITS bintables** with flexible, case-insensitive column names:
  `WAVE`, `WAVELENGTH`, `LAMBDA`, `LOGLAM` (converted from log10),
  `FLUX`, `FLAM`, `FNU`, `ERR`, `ERROR`, `SIGMA`, `STDEV`, `IVAR`
  (converted to sigma), `DQ`, `QUALITY`, ...
- **FITS image HDUs** with a linear wavelength WCS (`CRVAL1`/`CDELT1`/`CRPIX1`).
- **ASCII / CSV** files with 2-3 columns (wave, flux[, error]), with or
  without a header row.

Use `--ext N` to force a specific FITS extension.

## The GUI

| Action | Effect |
|---|---|
| `m` | toggle **mask mode**: two left-clicks bracket a region to mask (shaded red); right-click **undoes masks LIFO** (most recent first, cursor position irrelevant; a pending first edge is cancelled instead) |
| left click | add a continuum node (flux = median of *unmasked* points in a small box around the click) |
| right click | delete nearest node |
| `u` | undo last node; with no nodes placed, step **back to the previous window and un-accept it** (rescues an accidental `enter`) |
| `r` | reset all nodes in this window |
| `f` | fit and overplot the continuum |
| `s` | switch to cubic spline (interpolates through your nodes) |
| `c` | switch to sigma-clipped Chebyshev (fits the data, rejects lines) |
| `1`–`5` | set the degree of the current model (switches spline to polynomial) |
| `enter` or `a` | accept this window's fit, advance to the next |
| `b` | go back one window |
| `q` | stop early and write whatever was accepted |

Matplotlib's zoom/pan toolbar works as usual; clicks are ignored while a
toolbar tool is active (matplotlib's default key shortcuts are disabled so
they don't collide with the bindings above). Each window remembers its own
nodes and model, so you can go back and revise.

**Continuum error band.** Every fit shows a 1-sigma uncertainty band
(dashed lines + faint fill): node-based models propagate the uncertainty of
each node (standard error of the median in the click box) — interpolated for
splines, through the weighted least-squares covariance for polynomials —
while the Chebyshev data fit uses its full parameter covariance scaled by
the reduced chi-square. The band is blended across windows like the
continuum itself, carried on the result as `result.cont_err`, and drawn in
the overview plot. A wide band means your continuum placement is poorly
constrained there (few nodes, or few unclipped pixels).

**Masking and y-scaling.** The y-axis autoscales to the *unmasked* data only,
so a masked airglow spike no longer flattens the rest of the window. Masking a
region drops any nodes inside it and un-accepts affected windows so they get
refit. All mask regions are recorded in the output headers.

The model menu per window offers two genuinely different philosophies:

- **Node-based — "trust my clicks."** **Spline** passes exactly through your
  nodes (best for wiggly continua); **polynomial (deg 1-5)** least-squares
  fits through them. A degree-*n* polynomial needs at least *n*+1 nodes.
- **Data-based — "fit the pixels, reject the lines."** **Chebyshev (deg 1-5)**
  fits all unmasked pixels in the window directly, then iteratively
  sigma-clips outliers IRAF `continuum`-style: pixels more than `low_rej`
  sigma *below* the fit (absorption lines; default 1.5) or `high_rej` sigma
  *above* it (emission lines, cosmic rays; default 3.5) are rejected and the
  fit repeated, up to `--niterate` times (default 20). `--grow N` (default 6)
  also rejects N neighbors of each clipped pixel to catch line wings.
  Two safeguards keep the aggressive defaults from eating noise: low-side
  rejection requires at least `--min-pix` *consecutive* below-threshold
  pixels (default 3) — real lines are resolved into runs of adjacent low
  pixels, isolated noise dips are not — and iteration stops automatically
  once the residual sigma stops improving. (`--min-pix` doesn't apply to the
  high side, so single-pixel cosmic rays are still clipped; runs may include
  already-rejected neighbors, so line wings beside a clipped core still
  count.) Clipped pixels are shown as orange crosses.

  **Nodes guide the clipping.** A blind first fit fails when lines cover a
  large fraction of the window: the fit lands well below the continuum and
  the inflated residual sigma means nothing ever gets rejected. Any nodes
  you place are used as the *reference continuum level* (interpolated
  between them) for a robust MAD-based first rejection pass, which removes
  the lines before the first Chebyshev fit. So for line-dense windows, click
  a few nodes on clean continuum patches first, then press `f`. With zero
  nodes the whole window is blind-fit (fine for sparse lines); with two or
  more nodes the fitted range is also restricted to [first node, last node].

## Windowing and blending

`--window` sets the width of each fitting window in wavelength units (default
20 Å, suited to medium/high-resolution STIS/COS data; use `0` to fit the whole
spectrum at once). Consecutive windows overlap by `--overlap` (default 10%), and accepted
continua are combined with linear ramp weights across the overlaps, so there
are no jumps at window boundaries.

## Output

`write_spectrum` writes a FITS bintable (or ASCII, depending on the
extension). The `--full` format has columns:

- `WAVELENGTH`
- `FLUX` (original)
- `NORM_FLUX` (flux / continuum)
- `ERROR` (original 1-sigma error)
- `NORM_ERROR`, `CONTINUUM` (bonus columns; pass
  `include_continuum=False` to omit)

**The default output is VoigtFit-ready** (`--style voigt`): four columns —
wavelength, normalized flux, normalized error, mask — following the
[VoigtFit](https://voigtfit.readthedocs.io) input conventions. **The mask is
an inclusion mask: 1 = include in fit, 0 = exclude**, matching VoigtFit /
fitsutil. Pixels are excluded (0) if user-masked (airglow), lacking a fitted
continuum, or carrying a non-positive error (VoigtFit weights by 1/error, so
such pixels are masked rather than passed with sigma = 0); NaN and negative
values are written as 0. ASCII files parse with plain `np.loadtxt`, exactly
how VoigtFit reads them, and FITS files put recognized column names
(`WAVE`, `FLUX`, `ERR`, `MASK`) in the first table extension. Since the flux
is normalized, remember the `norm` keyword in your VoigtFit parameter file:

```
data  'target_norm.tab'  <resolution>  norm
```

**All output formats now share one mask convention: 0 = masked/excluded,
1 = good** — including the intermediate masked file. `--style minimal` (or
`--minimal`) gives the 4-column `WAVELENGTH, NORM_FLUX, ERROR, MASK` format;
`--style full` writes only the 6-column full-info file; and the `--full`
*flag* writes **both** the VoigtFit-style output and a full-info companion
file at `<output>_full.<ext>`.

Add `--plain` (or `write_spectrum(..., plain=True)`) for bare ASCII output:
values separated by a single space, no header lines, no column padding —
each row looks like `1200.000700 0.806969 0.023594 0`. NaN/inf values
(e.g. pixels in windows you never accepted) and negative fluxes/errors are
written as 0, so the file feeds straight into line-fitting tools. Works with
either style; ignored for FITS, and the non-plain formats keep their NaNs
for transparency.

ASCII output uses fixed-point notation with 6 decimals (e.g. `1215.123456`)
for wavelengths and all normalized quantities; raw flux-unit columns in
`--full` style keep exponent notation, since STIS/COS fluxes of ~1e-13 are
unreadable in fixed point. FITS output stores binary doubles, so no
precision is lost either way.

A record of the windows, models, node counts, mask regions, and binning used
is stored in the primary header comments.

### Overview plot

By default the CLI saves an overview figure next to the output
(`<output>_overview.pdf`) showing the data and the fitted continuum
together, with masked regions shaded. The spectrum is split into panels of
**3x the fitting-window width** (so `-w 20` gives 60 A panels), up to four
panels per PDF page. Tune the panel width with `--overview-zoom` (a
multiplier of the window width) or skip the plot with `--no-overview`.
From Python: `plot_overview(result, "overview.pdf", zoom=3.0)` — a `.png`
path gives a single tall figure instead of a paged PDF.

### Intermediate masked file

The CLI also writes `<input>_masked.fits` (override with `--masked-output`,
or `--masked-output none` to skip) with columns `WAVELENGTH`, `FLUX`,
`ERROR`, `MASK` (1 = masked) and the mask regions in the header. It is
updated each time you accept a window, so the masking work is preserved even
if you quit before finishing the fits. `specnorm.write_masked(spec, path)`
does the same from Python.

### Binning

`bin_spectrum(spec, n)` averages wavelength and flux over `n`-pixel bins,
propagates errors as `sqrt(sum(err^2))/n`, OR-combines DQ flags, and masks a
bin if any input pixel is masked. The CLI applies `-b/--bin` (default 2)
before the GUI opens.

## Example with synthetic data

```python
import numpy as np
from specnorm import Spectrum, normalize_interactive, write_spectrum

wave = np.linspace(1150, 1700, 8000)
cont = 1e-13 * (wave / 1400) ** -1.5
flux = cont * (1 - 0.8 * np.exp(-0.5 * ((wave - 1215.67) / 4) ** 2))
flux += np.random.normal(0, 3e-15, wave.size)
spec = Spectrum(wave, flux, error=np.full_like(wave, 3e-15))

result = normalize_interactive(spec, window=25.0)
write_spectrum(result, "synthetic_norm.dat")
```

## Limitations / notes

- Echelle orders are merged into a single sorted array; per-order
  normalization isn't supported yet (planned).
- The GUI requires an interactive matplotlib backend (Qt, Tk, macOS, ...).
  On a headless machine, set up X forwarding or use a different backend.
