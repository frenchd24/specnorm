"""Overview plot of the full normalization result.

Shows the data and fitted continuum together, split into panels of
``zoom`` times the fitting-window width (default 3x), with masked
regions shaded and un-fitted regions left without a continuum line.
Saved as a multi-page PDF (or a single-page PNG) for record keeping.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .spectrum import NormalizedSpectrum

PANELS_PER_PAGE = 4


def _infer_window(result: NormalizedSpectrum) -> Optional[float]:
    """Fitting-window width from the result metadata, if available."""
    try:
        w0, w1 = result.meta["specnorm"]["windows"][0]["range"]
        return float(w1) - float(w0)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _bin_for_display(result: NormalizedSpectrum, nbin: int):
    """Bin the arrays used for plotting (display only, never the output)."""
    nbin = int(nbin)
    n = (result.wavelength.size // nbin) * nbin
    if nbin <= 1 or n == 0:
        return (result.wavelength, result.flux, result.continuum,
                result.cont_err, result.mask)

    def _mean(a):
        if a is None:
            return None
        block = np.asarray(a, dtype=float)[:n].reshape(-1, nbin)
        with np.errstate(invalid="ignore"):
            out = np.full(block.shape[0], np.nan)
            good = np.isfinite(block)
            any_good = good.any(axis=1)
            if any_good.any():
                out[any_good] = (np.nansum(np.where(good, block, 0.0), axis=1)
                                 [any_good]
                                 / good.sum(axis=1)[any_good])
        return out

    wave = np.asarray(result.wavelength)[:n].reshape(-1, nbin).mean(axis=1)
    mask = None
    if result.mask is not None:
        mask = np.asarray(result.mask)[:n].reshape(-1, nbin).any(axis=1)
    return (wave, _mean(result.flux), _mean(result.continuum),
            _mean(result.cont_err), mask)


def plot_overview(result: NormalizedSpectrum, path: str,
                  zoom: float = 3.0, window: Optional[float] = None,
                  overlap: float = 0.15, bin: int = 1) -> str:
    """Save an overview figure of flux + continuum.

    Parameters
    ----------
    result : NormalizedSpectrum
    path : str
        Output figure path; '.pdf' gives a multi-page document with up
        to four panels per page, other extensions give one tall figure.
    zoom : float
        Panel width as a multiple of the continuum fitting-window width
        (default 3.0).
    window : float, optional
        Fitting-window width.  By default it is read from the result
        metadata; pass it explicitly for results built by hand.
        If neither is available (or the value is <= 0), the whole
        spectrum is shown in a single panel.
    overlap : float
        Fraction of each panel repeated on the next one (default 0.15),
        so the right edge of one row can be matched to the left edge of
        the row below.  The shared stretch is tinted in both panels.
    bin : int
        Bin the spectrum by this many pixels *for display only*, so the
        overview shows the same signal-to-noise you fitted against.
        Pass the binning used while fitting; the written output is
        unaffected.

    Returns
    -------
    str : the path written.
    """
    import matplotlib.pyplot as plt

    nbin = max(int(bin), 1)
    wave, flux_d, cont_d, cerr_d, mask_d = _bin_for_display(result, nbin)
    if window is None:
        window = _infer_window(result)

    overlap = float(np.clip(overlap, 0.0, 0.5))
    if window and window > 0:
        chunk = zoom * window
        step = chunk * (1.0 - overlap)
        chunks = []
        w0 = wave[0]
        while True:
            w1 = min(w0 + chunk, wave[-1])
            chunks.append((w0, w1))
            if w1 >= wave[-1]:
                break
            w0 += step
        share = chunk * overlap
    else:
        chunks = [(wave[0], wave[-1])]
        share = 0.0

    sn = result.meta.get("specnorm", {})
    exclude_regions = sn.get("exclude_regions",
                             result.meta.get("exclude_regions", []))
    fit_regions = sn.get("fit_mask_regions",
                         result.meta.get("fit_mask_regions", []))

    def _draw_panel(ax, w0, w1, first=False, last=False):
        sel = (wave >= w0) & (wave <= w1)
        w, f, c = wave[sel], flux_d[sel], cont_d[sel]
        good = np.isfinite(f)
        if mask_d is not None:
            good &= ~mask_d[sel].astype(bool)
        for (m0, m1) in fit_regions:   # fit masks shape the y-scale too
            good &= ~((w >= m0) & (w <= m1))
        ax.plot(w, f, color="0.35", lw=0.6, drawstyle="steps-mid",
                label="flux" + (f" (binned x{nbin})" if nbin > 1 else ""))
        ax.plot(w, c, color="tab:blue", lw=1.6, label="continuum")
        if cerr_d is not None:
            ce = cerr_d[sel]
            band = np.isfinite(ce) & np.isfinite(c)
            if band.any():
                ax.plot(w[band], (c + ce)[band], color="tab:blue", lw=0.8,
                        ls="--", alpha=0.7)
                ax.plot(w[band], (c - ce)[band], color="tab:blue", lw=0.8,
                        ls="--", alpha=0.7)
                ax.fill_between(w[band], (c - ce)[band], (c + ce)[band],
                                color="tab:blue", alpha=0.12, zorder=1)
        for (m0, m1) in fit_regions:
            if m1 >= w0 and m0 <= w1:
                ax.axvspan(max(m0, w0), min(m1, w1), color="darkorange",
                           alpha=0.10, zorder=0)
        for (m0, m1) in exclude_regions:
            if m1 >= w0 and m0 <= w1:
                ax.axvspan(max(m0, w0), min(m1, w1), color="red",
                           alpha=0.13, zorder=0)
        # Tint the stretches shared with the neighbouring panels so the
        # rows can be matched up by eye.
        if share > 0:
            if not first:
                ax.axvspan(w0, min(w0 + share, w1), color="0.55",
                           alpha=0.13, zorder=0)
                ax.axvline(min(w0 + share, w1), color="0.45", lw=0.7,
                           ls=":", zorder=2)
            if not last:
                ax.axvspan(max(w1 - share, w0), w1, color="0.55",
                           alpha=0.13, zorder=0)
                ax.axvline(max(w1 - share, w0), color="0.45", lw=0.7,
                           ls=":", zorder=2)
        # y-limits from unmasked flux + finite continuum (mask-immune).
        ref = f[good]
        finite_c = c[np.isfinite(c)]
        if finite_c.size:
            ref = np.concatenate([ref, finite_c])
        if ref.size:
            lo, hi = float(np.min(ref)), float(np.max(ref))
            pad = 0.08 * (hi - lo) if hi > lo else (abs(hi) * 0.1 or 1.0)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(w0, w1)
        ax.set_ylabel("Flux")

    title = "specnorm overview — {}".format(
        result.meta.get("rootname") or result.meta.get("targname")
        or result.meta.get("source_file", "spectrum"))

    if path.lower().endswith(".pdf"):
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(path) as pdf:
            for start in range(0, len(chunks), PANELS_PER_PAGE):
                page = chunks[start:start + PANELS_PER_PAGE]
                fig, axes = plt.subplots(len(page), 1,
                                         figsize=(11, 2.6 * len(page) + 0.8),
                                         squeeze=False)
                for k, (ax, (w0, w1)) in enumerate(zip(axes[:, 0], page)):
                    _draw_panel(ax, w0, w1,
                                first=(start + k == 0),
                                last=(start + k == len(chunks) - 1))
                axes[0, 0].legend(loc="upper right", fontsize=8)
                axes[0, 0].set_title(title + ("   (grey stripes are repeated "
                                              "on the neighbouring row)"
                                              if share > 0 else ""),
                                     fontsize=10)
                axes[-1, 0].set_xlabel("Wavelength")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    else:
        fig, axes = plt.subplots(len(chunks), 1,
                                 figsize=(11, 2.6 * len(chunks) + 0.8),
                                 squeeze=False)
        for k, (ax, (w0, w1)) in enumerate(zip(axes[:, 0], chunks)):
            _draw_panel(ax, w0, w1, first=(k == 0),
                        last=(k == len(chunks) - 1))
        axes[0, 0].legend(loc="upper right", fontsize=8)
        axes[0, 0].set_title(title + ("   (grey stripes are repeated on the "
                                      "neighbouring row)" if share > 0 else ""),
                             fontsize=10)
        axes[-1, 0].set_xlabel("Wavelength")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path
