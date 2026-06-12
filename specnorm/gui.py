"""Interactive, windowed continuum fitting with matplotlib.

The spectrum is split into wavelength windows (default 20 Angstroms with
10% overlap).  For each window the user can **mask** contaminated regions
(geocoronal Ly-alpha airglow, detector gaps, ...) and place continuum
nodes:

* **m** toggles mask mode.  In mask mode, two left-clicks bracket a
  region to mask (shaded); a right-click on a shaded region unmasks it.
  The y-axis autoscale ignores masked points, so an airglow spike no
  longer flattens the rest of the window.
* **left-click** (normal mode) drops a continuum node — the node's flux
  is the median *unmasked* flux within a small box around the click;
* **right-click** (normal mode) deletes the nearest node;
* **f** (re)fits and overplots the continuum;
* **s** / **c** / **1-5** switch model: spline, Chebyshev, or polynomial
  of that degree;
* **enter** (or **a**) accepts the fit and advances to the next window;
* **b** goes back, **r** clears nodes, **u** undoes the last node,
  **q** finishes early.

If ``masked_path`` is given, an intermediate masked spectrum file
(WAVELENGTH, FLUX, ERROR, MASK) is written/updated every time a window
is accepted and when the session ends, so the masking work is saved
before/independently of the continuum fit.

Accepted windows are blended together with linear ramps across the
overlap regions to give a smooth global continuum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .spectrum import Spectrum, NormalizedSpectrum
from .fitters import BaseFitter, make_fitter

HELP_TEXT = (
    "left/right click: add/delete node   m: mask mode (two clicks = mask "
    "region, right click = undo last mask)   u: undo node (or un-accept previous "
    "window)   r: reset nodes\n"
    "f: fit   s: spline (thru nodes)   c: Chebyshev (fits data, sigma-clips "
    "lines; nodes set ref level)   1-5: degree   enter/a: accept & next   "
    "b: back   q: quit & save"
)

# Default geocoronal airglow regions (Angstroms): Ly-alpha and the OI
# triplet, the usual contaminants in COS/STIS far-UV spectra.
AIRGLOW_REGIONS = [
    (1213.0, 1218.5),   # Ly-alpha 1215.67
    (1301.0, 1307.0),   # OI 1302.2 / 1304.9 / 1306.0
    (1354.5, 1356.5),   # OI] 1355.6
]


@dataclass
class WindowState:
    """Per-window user state."""
    w0: float
    w1: float
    nodes_x: List[float] = field(default_factory=list)
    nodes_y: List[float] = field(default_factory=list)
    nodes_e: List[float] = field(default_factory=list)
    fitter_kind: str = "spline"
    degree: int = 3
    continuum: Optional[np.ndarray] = None  # evaluated on window pixels
    cont_err: Optional[np.ndarray] = None   # 1-sigma continuum uncertainty
    rejected: Optional[np.ndarray] = None   # sigma-clipped pixels (window sel)
    accepted: bool = False


def _build_windows(wmin: float, wmax: float, window: float, overlap_frac: float):
    """Return list of (w0, w1) covering [wmin, wmax] with overlaps."""
    if window <= 0:
        return [(wmin, wmax)]
    step = window * (1.0 - overlap_frac)
    edges = []
    w0 = wmin
    while True:
        w1 = min(w0 + window, wmax)
        edges.append((w0, w1))
        if w1 >= wmax:
            break
        w0 += step
    return edges


class ContinuumGUI:
    """Matplotlib-based interactive continuum fitter.

    Parameters
    ----------
    spectrum : Spectrum
    window : float
        Window width in wavelength units (0 = whole spectrum at once).
        Default 20 (Angstroms), suited to medium/high-resolution data.
    overlap : float
        Fractional overlap between consecutive windows (0-0.5).
    fitter : str
        Initial model: 'spline', 'poly', or 'cheb'.
    degree : int
        Initial polynomial/Chebyshev degree (1-5).
    node_box : float
        Half-width (in wavelength units) of the median box used to set a
        node's flux when the user clicks.  Default: 0.5% of window width.
    mask_dq : bool
        If True, points with non-zero DQ are excluded from node medians
        and shown in grey.
    mask_regions : list of (w0, w1), optional
        Regions to mask before the GUI opens (e.g. ``AIRGLOW_REGIONS``).
    masked_path : str, optional
        If given, an intermediate masked-spectrum file is written here
        whenever a window is accepted and at the end of the session.
    low_rej, high_rej, niterate, grow, min_pix :
        Sigma-clipping parameters for the Chebyshev (data-fit) model;
        see :class:`specnorm.fitters.ChebyshevFitter`.
    """

    def __init__(self, spectrum: Spectrum, window: float = 20.0,
                 overlap: float = 0.10, fitter: str = "spline",
                 degree: int = 3, node_box: Optional[float] = None,
                 mask_dq: bool = True, mask_regions=None,
                 masked_path: Optional[str] = None,
                 low_rej: float = 1.5, high_rej: float = 3.5,
                 niterate: int = 20, grow: int = 6, min_pix: int = 3):
        if len(spectrum) < 2:
            raise ValueError("Spectrum has fewer than 2 points")
        self.spec = spectrum
        self.mask_dq = mask_dq
        self.masked_path = masked_path
        self.clip = dict(low_rej=low_rej, high_rej=high_rej,
                         niterate=niterate, grow=grow, min_pix=min_pix)
        for (m0, m1) in (mask_regions or []):
            self.spec.mask_region(m0, m1)
        self._refresh_good()

        self.window_edges = _build_windows(spectrum.wmin, spectrum.wmax,
                                           window, np.clip(overlap, 0.0, 0.5))
        self.states = [WindowState(w0, w1, fitter_kind=fitter, degree=degree)
                       for (w0, w1) in self.window_edges]
        self.idx = 0
        width = self.window_edges[0][1] - self.window_edges[0][0]
        self.node_box = node_box if node_box is not None else 0.005 * width

        self._fig = None
        self._ax = None
        self._finished = False
        self._mask_mode = False
        self._mask_start: Optional[float] = None  # first edge of pending mask

    def _refresh_good(self):
        self.good = self.spec.good_mask(use_dq=self.mask_dq, use_mask=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> NormalizedSpectrum:
        """Open the interactive figure; blocks until the user finishes."""
        import matplotlib.pyplot as plt

        self._fig, self._ax = plt.subplots(figsize=(12, 6))
        self._fig.canvas.manager.set_window_title("specnorm — continuum fitting")
        self._fig.subplots_adjust(bottom=0.18)
        # Disable matplotlib's default key bindings (s=save, f=fullscreen,
        # q=quit, c=back, r=home, ...) which collide with ours.
        try:
            self._fig.canvas.mpl_disconnect(
                self._fig.canvas.manager.key_press_handler_id)
        except Exception:
            pass
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()
        plt.show()  # blocks until window closed
        self._write_masked()
        return self._assemble()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _on_click(self, event):
        if event.inaxes is not self._ax or self._finished:
            return
        # Ignore clicks while zoom/pan tools are active.
        toolbar = getattr(self._fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        if event.xdata is None:
            return

        if self._mask_mode:
            self._mask_click(event)
            return

        st = self.states[self.idx]
        if event.button == 1:
            sample = self._node_sample(event.xdata)
            if sample is not None:
                st.nodes_x.append(float(event.xdata))
                st.nodes_y.append(float(sample[0]))
                st.nodes_e.append(float(sample[1]))
                st.continuum = None
                st.rejected = None
                self._draw()
            else:
                self._draw(message="No unmasked data near that click")
        elif event.button == 3 and st.nodes_x:
            i = int(np.argmin(np.abs(np.array(st.nodes_x) - event.xdata)))
            st.nodes_x.pop(i)
            st.nodes_y.pop(i)
            st.nodes_e.pop(i)
            st.continuum = None
            st.rejected = None
            self._draw()

    def _mask_click(self, event):
        x = float(event.xdata)
        if event.button == 1:
            if self._mask_start is None:
                self._mask_start = x
                self._draw(message=f"Mask start at {x:.2f} — click the other edge")
            else:
                w0, w1 = sorted((self._mask_start, x))
                self._mask_start = None
                self.spec.mask_region(w0, w1)
                self._refresh_good()
                self._invalidate_fits(w0, w1)
                self._draw(message=f"Masked [{w0:.2f}, {w1:.2f}]")
        elif event.button == 3:
            # Undo: first cancel a pending first edge, otherwise remove
            # masks LIFO — most recently added first — regardless of
            # cursor position.
            if self._mask_start is not None:
                self._mask_start = None
                self._draw(message="Pending mask edge cancelled")
                return
            removed = self.spec.pop_mask_region()
            if removed is None:
                self._draw(message="No mask regions to undo")
                return
            m0, m1 = removed
            self._refresh_good()
            self._invalidate_fits(m0, m1)
            n_left = len(self.spec.meta.get("mask_regions", []))
            self._draw(message=f"Removed mask [{m0:.2f}, {m1:.2f}] "
                               f"({n_left} remaining)")

    def _invalidate_fits(self, w0: float, w1: float):
        """Drop nodes inside a (un)masked region and clear affected fits."""
        for st in self.states:
            if st.w1 < w0 or st.w0 > w1:
                continue
            keep = [i for i, x in enumerate(st.nodes_x) if not (w0 <= x <= w1)]
            if len(keep) != len(st.nodes_x):
                st.nodes_x = [st.nodes_x[i] for i in keep]
                st.nodes_y = [st.nodes_y[i] for i in keep]
                st.nodes_e = [st.nodes_e[i] for i in keep]
            st.continuum = None
            st.cont_err = None
            st.rejected = None
            st.accepted = False

    def _on_key(self, event):
        import matplotlib.pyplot as plt
        if self._finished:
            return
        st = self.states[self.idx]
        key = (event.key or "").lower()

        if key == "m":
            self._mask_mode = not self._mask_mode
            self._mask_start = None
            self._draw()
        elif key == "f":
            self._fit_current()
        elif key == "s":
            st.fitter_kind = "spline"
            self._fit_current(quiet=True)
        elif key == "c":
            st.fitter_kind = "cheb"
            self._fit_current(quiet=True)
        elif key in "12345":
            if st.fitter_kind == "spline":
                st.fitter_kind = "poly"  # spline has no degree; switch
            st.degree = int(key)
            self._fit_current(quiet=True)
        elif key == "u":
            if st.nodes_x:
                st.nodes_x.pop()
                st.nodes_y.pop()
                st.nodes_e.pop()
                st.continuum = None
                st.rejected = None
                self._draw()
            elif self.idx > 0:
                # Nothing to undo here: step back to the previous window
                # and un-accept it (covers an accidental enter/a).
                self.idx -= 1
                prev = self.states[self.idx]
                prev.accepted = False
                self._mask_mode = False
                self._mask_start = None
                self._draw(message=f"Window {self.idx + 1} un-accepted — "
                                   "edit and re-accept")
        elif key == "r":
            st.nodes_x.clear()
            st.nodes_y.clear()
            st.nodes_e.clear()
            st.continuum = None
            st.rejected = None
            self._draw()
        elif key in ("enter", "a"):
            if st.continuum is None:
                self._fit_current()
            if st.continuum is None:
                return  # fit failed; message already shown
            st.accepted = True
            self._write_masked()
            if self.idx + 1 < len(self.states):
                self.idx += 1
                self._mask_mode = False
                self._mask_start = None
                self._draw()
            else:
                self._finished = True
                plt.close(self._fig)
        elif key == "b" and self.idx > 0:
            self.idx -= 1
            self._draw()
        elif key == "q":
            self._finished = True
            plt.close(self._fig)

    # ------------------------------------------------------------------
    # Fitting / drawing
    # ------------------------------------------------------------------
    def _window_sel(self, st: WindowState) -> np.ndarray:
        return (self.spec.wavelength >= st.w0) & (self.spec.wavelength <= st.w1)

    def _node_sample(self, x: float):
        """Return (median flux, 1-sigma uncertainty) near x, or None."""
        sel = (np.abs(self.spec.wavelength - x) <= self.node_box) & self.good
        if not sel.any():
            sel = (np.abs(self.spec.wavelength - x) <= 5 * self.node_box) & self.good
        if not sel.any():
            return None
        fbox = self.spec.flux[sel]
        ebox = self.spec.error[sel]
        n = int(sel.sum())
        y = float(np.median(fbox))
        # Standard error of the median: 1.2533 * sigma / sqrt(N).
        if np.any(ebox > 0):
            sig = float(np.median(ebox[ebox > 0]))
        elif n >= 3:
            sig = float(np.std(fbox))
        else:
            sig = 0.0
        e = 1.2533 * sig / np.sqrt(max(n, 1))
        return y, e

    def _node_flux(self, x: float) -> Optional[float]:
        sample = self._node_sample(x)
        return None if sample is None else sample[0]

    def _make_fitter(self, st: WindowState) -> BaseFitter:
        return make_fitter(st.fitter_kind, st.degree, **self.clip)

    def _fit_current(self, quiet: bool = False):
        st = self.states[self.idx]
        fitter = self._make_fitter(st)
        sel = self._window_sel(st)
        st.rejected = None

        if getattr(fitter, "fits_data", False):
            # Chebyshev: fit the unmasked data directly, sigma-clipping
            # outliers.  Nodes (if >= 2) restrict the fitted range.
            use = sel & self.good
            if len(st.nodes_x) >= 2:
                x0, x1 = min(st.nodes_x), max(st.nodes_x)
                use &= (self.spec.wavelength >= x0) & (self.spec.wavelength <= x1)
            n_use = int(use.sum())
            if n_use < st.degree + 2:
                self._draw(message=f"Need >= {st.degree + 2} unmasked points "
                                   f"for {fitter.label()} ({n_use} available)")
                return
            err = self.spec.error[use]
            init_x = st.nodes_x if st.nodes_x else None
            init_y = st.nodes_y if st.nodes_x else None
            fitter.fit_data(self.spec.wavelength[use], self.spec.flux[use],
                            err if np.any(err > 0) else None,
                            init_x=init_x, init_y=init_y)
            st.continuum = fitter(self.spec.wavelength[sel])
            st.cont_err = fitter.uncertainty(self.spec.wavelength[sel])
            # Map clipped points back onto the window for display.
            rejected_global = np.zeros(len(self.spec), dtype=bool)
            rejected_global[np.flatnonzero(use)] = ~fitter.keep
            st.rejected = rejected_global[sel]
        else:
            if len(st.nodes_x) < fitter.min_nodes:
                self._draw(message=f"Need >= {fitter.min_nodes} nodes for "
                                   f"{fitter.label()} ({len(st.nodes_x)} placed)")
                return
            fitter.fit(st.nodes_x, st.nodes_y, st.nodes_e)
            st.continuum = fitter(self.spec.wavelength[sel])
            st.cont_err = fitter.uncertainty(self.spec.wavelength[sel])
        self._draw()

    def _draw(self, message: str = ""):
        ax, st = self._ax, self.states[self.idx]
        ax.clear()
        sel = self._window_sel(st)
        w = self.spec.wavelength[sel]
        f = self.spec.flux[sel]
        g = self.good[sel]
        masked = self.spec.mask[sel]

        ax.plot(w[g], f[g], color="0.2", lw=0.8, drawstyle="steps-mid",
                label="flux")
        bad = ~g & ~masked
        if bad.any():
            ax.plot(w[bad], f[bad], ".", color="0.75", ms=3, label="bad (DQ)")
        if masked.any():
            ax.plot(w[masked], f[masked], color="lightcoral", lw=0.6,
                    alpha=0.6, drawstyle="steps-mid", label="masked")
        for (m0, m1) in self.spec.meta.get("mask_regions", []):
            if m1 >= st.w0 and m0 <= st.w1:
                ax.axvspan(max(m0, st.w0), min(m1, st.w1),
                           color="red", alpha=0.10, zorder=0)
        if self._mask_start is not None:
            ax.axvline(self._mask_start, color="red", ls="--", lw=1)
        if st.nodes_x:
            ax.plot(st.nodes_x, st.nodes_y, "o", color="tab:red", ms=8,
                    mec="k", zorder=5, label="nodes")
        if st.continuum is not None:
            ax.plot(w, st.continuum, color="tab:blue", lw=2, zorder=4,
                    label="continuum")
            if st.cont_err is not None:
                hi_b = st.continuum + st.cont_err
                lo_b = st.continuum - st.cont_err
                ax.plot(w, hi_b, color="tab:blue", lw=0.9, ls="--",
                        alpha=0.7, zorder=4)
                ax.plot(w, lo_b, color="tab:blue", lw=0.9, ls="--",
                        alpha=0.7, zorder=4, label=r"$\pm1\sigma$")
                ax.fill_between(w, lo_b, hi_b, color="tab:blue",
                                alpha=0.12, zorder=3)
        if st.continuum is not None and st.rejected is not None \
                and st.rejected.any():
            ax.plot(w[st.rejected], f[st.rejected], "x", color="tab:orange",
                    ms=4, zorder=3,
                    label=f"clipped ({int(st.rejected.sum())})")

        # --- y autoscale ignoring masked / bad points -------------------
        ax.set_xlim(st.w0, st.w1)
        ref = f[g]
        if st.continuum is not None:
            ref = np.concatenate([ref, st.continuum[np.isfinite(st.continuum)]])
        if ref.size:
            lo, hi = float(np.min(ref)), float(np.max(ref))
            pad = 0.07 * (hi - lo) if hi > lo else (abs(hi) * 0.1 or 1.0)
            ax.set_ylim(lo - pad, hi + pad)

        model = {"spline": "spline (nodes)",
                 "poly": f"poly deg {st.degree} (nodes)",
                 "cheb": f"cheb deg {st.degree} (data, clipped)",
                 }.get(st.fitter_kind, st.fitter_kind)
        n_acc = sum(s.accepted for s in self.states)
        mode = "   *** MASK MODE ***" if self._mask_mode else ""
        ax.set_title(f"Window {self.idx + 1}/{len(self.states)}   "
                     f"[{st.w0:.1f}–{st.w1:.1f}]   model: {model}   "
                     f"accepted: {n_acc}/{len(self.states)}{mode}",
                     color="crimson" if self._mask_mode else "black")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        if message:
            ax.text(0.5, 0.95, message, transform=ax.transAxes, ha="center",
                    color="crimson", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        self._fig.text(0.5, 0.02, HELP_TEXT, ha="center", fontsize=8,
                       family="monospace")
        # Remove duplicate help text from prior draws.
        for txt in self._fig.texts[:-1]:
            txt.remove()
        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _write_masked(self):
        if self.masked_path is None:
            return
        from .writer import write_masked
        try:
            write_masked(self.spec, self.masked_path)
        except Exception as exc:  # don't let I/O kill the session
            print(f"Warning: could not write masked file: {exc}")

    def _assemble(self) -> NormalizedSpectrum:
        wave = self.spec.wavelength
        cont = np.zeros_like(wave)
        cerr = np.zeros_like(wave)
        weight = np.zeros_like(wave)

        for st in self.states:
            if not st.accepted or st.continuum is None:
                continue
            sel = self._window_sel(st)
            w = wave[sel]
            # Linear ramp weights -> smooth blending in overlap regions.
            span = max(st.w1 - st.w0, 1e-30)
            ramp = np.minimum(w - st.w0, st.w1 - w) / span + 1e-3
            cont[sel] += st.continuum * ramp
            if st.cont_err is not None:
                cerr[sel] += np.nan_to_num(st.cont_err) * ramp
            weight[sel] += ramp

        covered = weight > 0
        with np.errstate(invalid="ignore"):
            denom = np.where(covered, weight, 1.0)
            cont = np.where(covered, cont / denom, np.nan)
            cerr = np.where(covered, cerr / denom, np.nan)

        meta = dict(self.spec.meta)
        meta["specnorm"] = {
            "mask_regions": list(self.spec.meta.get("mask_regions", [])),
            "binning": self.spec.meta.get("binning", 1),
            "windows": [
                {"range": [s.w0, s.w1], "model": s.fitter_kind,
                 "degree": s.degree, "n_nodes": len(s.nodes_x),
                 "n_clipped": int(s.rejected.sum()) if s.rejected is not None else 0,
                 "accepted": s.accepted}
                for s in self.states
            ],
        }
        return NormalizedSpectrum(wave, self.spec.flux, cont,
                                  self.spec.error,
                                  mask=self.spec.mask.copy(),
                                  cont_err=cerr, meta=meta)


def normalize_interactive(spectrum: Spectrum, window: float = 20.0,
                          overlap: float = 0.10, fitter: str = "spline",
                          degree: int = 3, **kwargs) -> NormalizedSpectrum:
    """Convenience wrapper: run the GUI and return the result."""
    gui = ContinuumGUI(spectrum, window=window, overlap=overlap,
                       fitter=fitter, degree=degree, **kwargs)
    return gui.run()
