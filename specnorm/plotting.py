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


def plot_overview(result: NormalizedSpectrum, path: str,
                  zoom: float = 3.0, window: Optional[float] = None,
                  overlap: float = 0.15) -> str:
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

    Returns
    -------
    str : the path written.
    """
    import matplotlib.pyplot as plt

    wave = result.wavelength
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
        w, f, c = wave[sel], result.flux[sel], result.continuum[sel]
        good = np.isfinite(f)
        if result.mask is not None:
            good &= ~result.mask[sel].astype(bool)
        for (m0, m1) in fit_regions:   # fit masks shape the y-scale too
            good &= ~((w >= m0) & (w <= m1))
        ax.plot(w, f, color="0.35", lw=0.6, drawstyle="steps-mid",
                label="flux")
        ax.plot(w, c, color="tab:blue", lw=1.6, label="continuum")
        if result.cont_err is not None:
            ce = result.cont_err[sel]
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
