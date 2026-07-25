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
OI] 1355 before the GUI opens. Add custom regions with repeated `--mask W0:W1` (fit-only) or
`--exclude W0:W1` (also masked in the output) flags. Data are binned x2 by default (`-b 1` to disable,
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
| `m` | toggle **fit-mask mode**: two left-clicks bracket a region to hide from the continuum fit (shaded orange); right-click **undoes LIFO** (most recent first; a pending first edge is cancelled instead) |
| `x` | toggle **exclude mode**: same interaction (shaded red), but those pixels are also flagged as masked in the output |
| left click | add a continuum node (flux = median of *unmasked* points in a small box around the click) |
| right click | delete nearest node |
| `u` | undo last node; with no nodes placed, step **back to the previous window and un-accept it** (rescues an accidental `enter`) |
| `r` | reset all nodes in this window |
| `f` | fit and overplot the continuum |
| `s` | switch to cubic spline (interpolates through your nodes) |
| `c` | switch to sigma-clipped Chebyshev (fits the data, rejects lines) |
| `1`–`5` | set the degree of the current model (switches spline to polynomial) |
| `+` / `-` | zoom the current window in / out (geometric, x1.5 per press, about the window centre) |
| `←` / `→` | pan the window left / right by a quarter of its width, keeping the width |
| `↑` / `↓` | zoom the flux axis in / out, about the continuum level so it stays in view |
| `0` | reset the window to the default width and the flux scaling |
| click the coverage strip | jump to the window at that wavelength |
| `enter` or `a` | accept this window's fit, advance to the next |
| `b` | go back one window |
| `q` | stop early and write whatever was accepted |

Matplotlib's zoom/pan toolbar works as usual; clicks are ignored while a
toolbar tool is active (matplotlib's default key shortcuts are disabled so
they don't collide with the bindings above). Each window remembers its own
nodes and model, so you can go back and revise.

**The plot always shows the knitted continuum.** The blue curve is not
the current window's model in isolation — it is every accepted fit blended
together exactly as it will be written out. Zoom out and you see the real
continuum across all those windows, so you can judge which regions need
redoing and spot joins that knitted badly. The window you are editing is
previewed in that curve at top precedence (so you see what accepting will
produce), and where its own model differs from the knitted result it is
drawn separately as a dashed purple line. Stretches with no fit yet appear
as breaks in the curve, marked along the bottom axis.

**Continuum error band.** Every fit shows a 1-sigma uncertainty band
(dashed lines + faint fill): node-based models propagate the uncertainty of
each node (standard error of the median in the click box) — interpolated for
splines, through the weighted least-squares covariance for polynomials —
while the Chebyshev data fit uses its full parameter covariance scaled by
the reduced chi-square. The band is blended across windows like the
continuum itself, carried on the result as `result.cont_err`, and drawn in
the overview plot. A wide band means your continuum placement is poorly
constrained there (few nodes, or few unclipped pixels).

**Adjustable window (broad features).** For a wide absorption feature such
as the Galactic Ly-alpha damping trough at 1215 A — far wider than a normal
20 A window — press `-` to zoom out until the window spans the whole
feature, place continuum nodes on the clean shoulders either side, and fit a
flat/linear continuum straight across (a degree-1 polynomial or a spline
through the two shoulders). Because node placement and the applied continuum
both operate on the current window, zooming out is what lets a single fit
bridge the trough. `+` zooms back in, the left/right arrows pan the window
without changing its width, the up/down arrows zoom the flux axis, and `0`
restores the default view. Zoom is geometric
(x1.5 per press) about the window centre, so repeated presses scale smoothly
in both directions.

The fit follows the view: it is re-evaluated from the fitted model whenever
you zoom, pan, or move between windows, so a continuum you have fitted stays
visible (and the title reports `no fit yet` / `fitted, not accepted` /
`ACCEPTED` so you always know where a window stands). Changing a model to
one that cannot be fitted with the nodes available — say pressing `5` with
three nodes placed — reverts to the previous model rather than discarding
your fit.

**Knitting.** Accepting a window fits it in among the ones already there
rather than competing with them. The window you just accepted takes
precedence over anything it overlaps: older accepted fits are *trimmed back*
to its edge — keeping a blend zone so the join stays smooth — an older fit
that spanned right across it is split into the parts either side, and only
windows it completely covers are retired. Going back to re-fit a region
therefore never destroys the neighbouring work, and zooming into an
already-accepted window preserves its surrounding coverage as separate
wings.

Where fits do overlap, they are combined with weights built from three
factors: an edge taper (so joins are smooth), **recency** (a fit accepted
later dominates one accepted earlier, so a re-fit supersedes what was
there), and **reliability**. The last one guards against a failure mode of
high-order fits: a polynomial or Chebyshev series can run away at the edge
of its window, and if that edge overlaps a neighbour it would otherwise drag
the knitted continuum with it. A fit is therefore down-weighted where it
extrapolates beyond its own nodes or fitted pixels, and where it strays much
further from the local flux level than a competing fit does — so in an
overlap the fit that stays closer to the data wins. In testing, a degree-5
fit diverging to −142x the continuum level is suppressed to leave the
knitted result within 0.03% of the true continuum.

When you accept a window whose span you changed, the rest of the spectrum
re-tiles at the default width from its right edge, preserving any downstream
windows you had already fitted. Because zooming and panning can leave a
stretch behind unfitted, a **coverage strip** under the plot shows the whole
spectrum with accepted fits in green, masked regions in red, and a box
around the current window; it reports how many unfitted regions remain, a
message warns you if accepting leaves a gap behind, and clicking the strip
jumps straight to any window. The whole-spectrum mode (`-w 0`) has a single
window and disables zooming and panning.

**Two kinds of mask.** Both hide pixels from the continuum fit, from node
placement and from the y-axis autoscale (so a masked airglow spike no longer
flattens the window), but they differ in what reaches the output:

- **Fit masks** (`m`, `--mask W0:W1`) shape the fit only. Use them for *real
  spectral features* — absorption lines, damped troughs — that you do not
  want dragging the continuum down. The data are written out unflagged
  (`MASK = 1`) and remain fully available for later analysis.
- **Exclusions** (`x`, `--exclude W0:W1`) are for *genuinely bad data* —
  geocoronal airglow, detector artefacts. These are ignored by the fit *and*
  written with `MASK = 0` in every output file, so downstream tools skip
  them. `--airglow` adds its regions as exclusions, since airglow-filled
  pixels are contamination rather than signal.

Masking a region drops any nodes inside it and un-accepts affected windows so
they get refit. Both region lists are recorded in the output headers, and the
coverage strip shows fit masks in orange and exclusions in red.

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
  also rejects N neighbours of each clipped pixel to catch line wings.
  Two safeguards keep the aggressive defaults from eating noise: low-side
  rejection requires at least `--min-pix` *consecutive* below-threshold
  pixels (default 3) — real lines are resolved into runs of adjacent low
  pixels, isolated noise dips are not — and iteration stops automatically
  once the residual sigma stops improving. (`--min-pix` doesn't apply to the
  high side, so single-pixel cosmic rays are still clipped; runs may include
  already-rejected neighbours, so line wings beside a clipped core still
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
spectrum at once). Consecutive windows overlap by `--overlap` (default 15%), and accepted
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
panels per PDF page. Consecutive panels **overlap by 15%** and the shared
stretch is tinted grey in both rows with a dotted boundary, so you can see
how the right edge of one row joins the left edge of the next. Tune the
panel width with `--overview-zoom`, the repeat with `--overview-overlap`, or
skip the plot with `--no-overview`. From Python:
`plot_overview(result, "overview.pdf", zoom=3.0, overlap=0.15)` — a `.png`
path gives a single tall figure instead of a paged PDF.

### Dead pixels and finishing

Detector edges are usually padded with zero flux and zero error. Those
pixels cannot anchor a node (a node dropped there would drag the fit to
zero), they are never reported as unfitted gaps, and the continuum across
them is filled by holding the nearest fitted value — so a spectrum never
ends with a stray "missing window" you cannot fit.

Accepting the last window no longer ends the session while fittable regions
remain: you are sent back to the first one still outstanding, with a count of
what is left. Press **`q`** whenever you want to stop and write out what you
have.

### Saving and resuming a session

Every acceptance writes `<output>_session.json` (change it with `--session`,
or `--session none` to skip) recording the input file, binning, all settings,
both mask lists, and every window with its nodes, model and acceptance state.
To carry on later, or to go back and edit a region after you have already
written the output:

```bash
specnorm --resume myspectrum_norm_session.json
```

The spectrum is reopened, the masks reapplied and every window refit from its
saved nodes, reproducing the previous continuum exactly, and the GUI opens on
the first region still needing work. From Python, `load_session(path)` returns
the rebuilt `(gui, native_spectrum)` pair.

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

**Binning affects the fit, not the output.** You interact with the binned
spectrum (better S/N for placing the continuum), but the output is written
at **native resolution**: the fitted models are re-evaluated exactly on the
original pixels (no interpolation of binned arrays), blended with the same
weights, with masks re-applied and the error band recomputed per pixel.
Pass `--write-binned` to write the binned spectrum instead, or from Python
use `normalize_interactive(binned, ..., output_on=native)` /
`gui.assemble_on(native)`. The metadata records both grids
(`binning` = output, `fit_binning` = fitting).

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
