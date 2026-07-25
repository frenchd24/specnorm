"""Interactive, windowed continuum fitting with matplotlib.

The spectrum is split into wavelength windows (default 20 Angstroms with
10% overlap).  For each window the user can **mask** contaminated regions
(geocoronal Ly-alpha airglow, detector gaps, ...) and place continuum
nodes:

* **m** toggles *fit-mask* mode: two left-clicks bracket a region to
  hide from the continuum fit (shaded orange), a right-click removes
  the most recent one.  Fit masks keep the data in the output — use
  them for real spectral features you do not want pulling the fit.
* **x** toggles *exclude* mode, which works the same way (shaded red)
  but also flags those pixels as masked in every output file — use it
  for genuinely bad data such as geocoronal airglow.
  The y-axis autoscale ignores both, so an airglow spike no longer
  flattens the rest of the window.
* **left-click** (normal mode) drops a continuum node — the node's flux
  is the median *unmasked* flux within a small box around the click;
* **right-click** (normal mode) deletes the nearest node;
* **+** / **-** zoom the window out / in (geometrically, about its
  centre), and the **left/right arrows** pan it while keeping its width;
  **0** restores the default width.  Zooming out across a broad damped
  feature (Galactic Ly-alpha at 1215 A) lets a single fit bridge it;
* a coverage strip under the plot shows which parts of the spectrum
  already have accepted fits (green), which are masked (red), and where
  the current window sits; click it to jump to any window;
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
    "left/right click: add/delete node   m: fit-mask mode (hide from fit "
    "only)   x: exclude mode (also masked in output)   u: undo node   "
    "r: reset nodes\n"
    "f: fit   s: spline (thru nodes)   c: Chebyshev (fits data, sigma-clips "
    "lines; nodes set ref level)   1-5: degree   +/-: zoom out/in   "
    "arrows: pan window   0: reset width   enter/a: accept & next   "
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
    fitter: Optional[BaseFitter] = None     # the fitted model itself
    rejected: Optional[np.ndarray] = None   # sigma-clipped pixels (window sel)
    resized: bool = False                   # width changed from its tile
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
        Fit-only masks applied before the GUI opens: hidden from the
        continuum fit, but not flagged in the output.
    exclude_regions : list of (w0, w1), optional
        Exclusion masks applied before the GUI opens (e.g.
        ``AIRGLOW_REGIONS``): ignored by the fit *and* flagged as
        masked in the output.
    masked_path : str, optional
        If given, an intermediate masked-spectrum file is written here
        whenever a window is accepted and at the end of the session.
    low_rej, high_rej, niterate, grow, min_pix :
        Sigma-clipping parameters for the Chebyshev (data-fit) model;
        see :class:`specnorm.fitters.ChebyshevFitter`.
    """

    def __init__(self, spectrum: Spectrum, window: float = 20.0,
                 overlap: float = 0.15, fitter: str = "spline",
                 degree: int = 3, node_box: Optional[float] = None,
                 mask_dq: bool = True, mask_regions=None,
                 exclude_regions=None,
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
            self.spec.mask_region(m0, m1, kind="fit")
        for (m0, m1) in (exclude_regions or []):
            self.spec.mask_region(m0, m1, kind="exclude")
        self._refresh_good()

        self.overlap = float(np.clip(overlap, 0.0, 0.5))
        self.default_window = float(window)
        self.init_fitter = fitter
        self.init_degree = degree
        self.window_edges = _build_windows(spectrum.wmin, spectrum.wmax,
                                           window, self.overlap)
        self.states = [WindowState(w0, w1, fitter_kind=fitter, degree=degree)
                       for (w0, w1) in self.window_edges]
        self.idx = 0
        width = self.window_edges[0][1] - self.window_edges[0][0]
        self.node_box = node_box if node_box is not None else 0.005 * width
        # Zoom is geometric: each press scales the width by this factor,
        # so zooming in always works (a fixed step could never shrink a
        # default-width window) and zooming out accelerates sensibly.
        self.zoom_factor = 1.5
        self.pan_frac = 0.25   # pan step as a fraction of current width
        self.min_window = max(5 * self.node_box,
                              0.05 * self.default_window)

        self._fig = None
        self._ax = None
        self._finished = False
        self._mask_mode: Optional[str] = None  # None | "fit" | "exclude"
        self._mask_start: Optional[float] = None  # first edge of pending mask
        self._ax_map = None                       # coverage strip axes

    @staticmethod
    def _clear_fit(st: WindowState):
        """Discard a window's fit (it is stale, e.g. nodes changed)."""
        st.continuum = None
        st.cont_err = None
        st.rejected = None
        st.fitter = None

    def _refresh_good(self):
        self.good = self.spec.good_mask(use_dq=self.mask_dq, use_mask=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> NormalizedSpectrum:
        """Open the interactive figure; blocks until the user finishes."""
        import matplotlib.pyplot as plt

        self._fig, (self._ax, self._ax_map) = plt.subplots(
            2, 1, figsize=(12, 6.6),
            gridspec_kw={"height_ratios": [14, 1], "hspace": 0.45})
        self._fig.canvas.manager.set_window_title("specnorm — continuum fitting")
        self._fig.subplots_adjust(bottom=0.20)
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
        if self._finished:
            return
        if event.inaxes is self._ax_map and event.xdata is not None:
            self._jump_to(float(event.xdata))
            return
        if event.inaxes is not self._ax:
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
                self._clear_fit(st)
                self._draw()
            else:
                self._draw(message="No unmasked data near that click")
        elif event.button == 3 and st.nodes_x:
            i = int(np.argmin(np.abs(np.array(st.nodes_x) - event.xdata)))
            st.nodes_x.pop(i)
            st.nodes_y.pop(i)
            st.nodes_e.pop(i)
            self._clear_fit(st)
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
                kind = self._mask_mode
                self.spec.mask_region(w0, w1, kind=kind)
                self._refresh_good()
                self._invalidate_fits(w0, w1)
                label = ("fit-masked" if kind == "fit" else "EXCLUDED")
                self._draw(message=f"{label} [{w0:.2f}, {w1:.2f}]")
        elif event.button == 3:
            # Undo: first cancel a pending first edge, otherwise remove
            # masks LIFO — most recently added first — regardless of
            # cursor position.
            if self._mask_start is not None:
                self._mask_start = None
                self._draw(message="Pending mask edge cancelled")
                return
            kind = self._mask_mode
            removed = self.spec.pop_mask_region(kind=kind)
            if removed is None:
                self._draw(message=f"No {kind} mask regions to undo")
                return
            m0, m1 = removed
            self._refresh_good()
            self._invalidate_fits(m0, m1)
            key = ("fit_mask_regions" if kind == "fit" else "exclude_regions")
            n_left = len(self.spec.meta.get(key, []))
            self._draw(message=f"Removed {kind} mask [{m0:.2f}, {m1:.2f}] "
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
            self._clear_fit(st)
            st.accepted = False

    def _on_key(self, event):
        import matplotlib.pyplot as plt
        if self._finished:
            return
        st = self.states[self.idx]
        key = (event.key or "").lower()

        if key in ("m", "x"):
            kind = "fit" if key == "m" else "exclude"
            self._mask_mode = None if self._mask_mode == kind else kind
            self._mask_start = None
            self._draw()
        elif key == "f":
            self._fit_current()
        elif key == "s":
            self._switch_model(st, "spline", st.degree)
        elif key == "c":
            self._switch_model(st, "cheb", st.degree)
        elif key in "12345":
            # Digits set the degree of the current model; a spline has
            # no degree, so it becomes a polynomial.
            kind = "poly" if st.fitter_kind == "spline" else st.fitter_kind
            self._switch_model(st, kind, int(key))
        elif key == "u":
            if st.nodes_x:
                st.nodes_x.pop()
                st.nodes_y.pop()
                st.nodes_e.pop()
                self._clear_fit(st)
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
            self._clear_fit(st)
            self._draw()
        elif key in ("enter", "a"):
            if st.fitter is None:
                self._fit_current()
            if st.fitter is None:
                return  # fit failed; message already shown
            st.accepted = True
            if st.resized:
                # Zooming or panning changed this window's span, so
                # re-tile the remaining spectrum from its right edge.
                self._retile_tail()
            self._write_masked()
            # Warn if adjusting the window left an unfitted gap behind.
            behind = self._coverage_gaps(upto=st.w1)
            note = ""
            if behind:
                a, b = behind[0]
                note = (f"Gap left unfitted at [{a:.1f}, {b:.1f}] — "
                        "press b or click the coverage bar to go back")
            if self.idx + 1 < len(self.states):
                self.idx += 1
                self._mask_mode = None
                self._mask_start = None
                self._draw(message=note)
            else:
                self._finished = True
                plt.close(self._fig)
        elif key in ("+", "="):
            self._resize_window(+1)
        elif key in ("-", "_"):
            self._resize_window(-1)
        elif key == "left":
            self._pan_window(-1)
        elif key == "right":
            self._pan_window(+1)
        elif key == "0":
            self._reset_window()
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

    # ------------------------------------------------------------------
    # Window resizing (zoom out across broad features, then back in)
    # ------------------------------------------------------------------
    def _reshape_window(self, w0: float, w1: float, note: str):
        """Apply a new window span, keeping the fit visible.

        The fitted model is retained (it is defined by the nodes, not
        the view), so the continuum is simply re-evaluated over the new
        span.  Data-driven models (sigma-clipped Chebyshev) are refit
        because their input pixels changed; if that refit fails the
        previous model is kept rather than leaving the window blank.
        """
        st = self.states[self.idx]
        st.w0, st.w1 = w0, w1
        st.resized = True
        st.continuum = st.cont_err = None
        st.rejected = None
        if st.fitter is not None and getattr(st.fitter, "fits_data", False):
            previous = st.fitter
            self._fit_current(quiet=True)
            if st.fitter is None:          # refit failed: keep the old one
                st.fitter = previous
        self._draw(message=note)

    def _resize_window(self, direction: int):
        """Zoom the current window out (direction>0) or in (direction<0).

        Zooming is geometric about the window centre, so repeated
        presses scale smoothly and zooming in always works.
        """
        if self.default_window <= 0:
            self._draw(message="Zoom is unavailable when fitting the "
                               "whole spectrum at once (-w 0)")
            return
        st = self.states[self.idx]
        factor = self.zoom_factor if direction > 0 else 1.0 / self.zoom_factor
        centre = 0.5 * (st.w0 + st.w1)
        half = 0.5 * (st.w1 - st.w0) * factor
        new_w0 = max(centre - half, self.spec.wmin)
        new_w1 = min(centre + half, self.spec.wmax)
        if direction < 0 and new_w1 - new_w0 < self.min_window:
            self._draw(message=f"Minimum window width is "
                               f"{self.min_window:.2f}")
            return
        if (new_w0, new_w1) == (st.w0, st.w1):
            self._draw(message="Already showing the full spectrum")
            return
        self._reshape_window(new_w0, new_w1,
                             f"Window width {new_w1 - new_w0:.2f}")

    def _pan_window(self, direction: int):
        """Shift the window left (direction<0) or right (direction>0).

        The width is preserved; the shift is a fraction of the current
        width per press, clamped to the spectrum bounds.
        """
        if self.default_window <= 0:
            return
        st = self.states[self.idx]
        width = st.w1 - st.w0
        shift = direction * self.pan_frac * width
        new_w0 = st.w0 + shift
        new_w1 = st.w1 + shift
        if new_w0 < self.spec.wmin:
            new_w0, new_w1 = self.spec.wmin, self.spec.wmin + width
        if new_w1 > self.spec.wmax:
            new_w0, new_w1 = self.spec.wmax - width, self.spec.wmax
        if abs(new_w0 - st.w0) < 1e-12:
            self._draw(message="At the edge of the spectrum")
            return
        self._reshape_window(new_w0, new_w1,
                             f"Window [{new_w0:.2f}, {new_w1:.2f}]")

    def _reset_window(self):
        """Restore the current window to the default width (recentred)."""
        if self.default_window <= 0:
            return
        st = self.states[self.idx]
        centre = 0.5 * (st.w0 + st.w1)
        half = 0.5 * self.default_window
        self._reshape_window(max(centre - half, self.spec.wmin),
                             min(centre + half, self.spec.wmax),
                             "Window reset to default width")

    def _jump_to(self, wavelength: float):
        """Navigate to the window containing a wavelength (map click)."""
        for i, s in enumerate(self.states):
            if s.w0 <= wavelength <= s.w1:
                self.idx = i
                self._mask_mode = None
                self._mask_start = None
                self._draw(message=f"Jumped to window {i + 1}")
                return
        nearest = int(np.argmin([min(abs(s.w0 - wavelength),
                                     abs(s.w1 - wavelength))
                                 for s in self.states]))
        self.idx = nearest
        self._draw(message=f"Jumped to nearest window {nearest + 1}")

    def _covered_spans(self):
        """Sorted (w0, w1) spans of windows with an accepted fit."""
        return sorted((s.w0, s.w1) for s in self.states
                      if s.accepted and s.fitter is not None)

    def _coverage_gaps(self, upto: Optional[float] = None):
        """Wavelength ranges with no accepted fit, below ``upto``."""
        limit = self.spec.wmax if upto is None else upto
        gaps = []
        cursor = self.spec.wmin
        for (a, b) in self._covered_spans():
            if a > cursor:
                gaps.append((cursor, min(a, limit)))
            cursor = max(cursor, b)
            if cursor >= limit:
                break
        if cursor < limit:
            gaps.append((cursor, limit))
        tol = 1e-6 * max(self.spec.wmax - self.spec.wmin, 1.0)
        return [(a, b) for (a, b) in gaps if b - a > tol]

    def _retile_tail(self):
        """Rebuild the windows after the current one from its right edge.

        Called when an adjusted (zoomed or panned) window is accepted so
        the rest of the spectrum is tiled at the default width with no
        gap.  Downstream windows that already carry work (a fit or an
        acceptance) and lie beyond the accepted span are preserved; only
        the region they do not cover is re-tiled.
        """
        cur = self.states[self.idx]
        if self.default_window <= 0:
            return
        step_back = self.overlap * self.default_window
        kept = [s for s in self.states[self.idx + 1:]
                if (s.accepted or s.fitter is not None) and s.w1 > cur.w1]
        kept.sort(key=lambda s: s.w0)

        # Walk left to right, tiling every stretch not covered by a
        # preserved window, then everything past the last one.
        sliver = 0.1 * self.default_window
        tail = []
        cursor = cur.w1 - step_back
        for s in kept:
            gap = s.w0 - cursor
            if gap > sliver:
                tail += [WindowState(w0, w1, fitter_kind=self.init_fitter,
                                     degree=self.init_degree)
                         for (w0, w1) in _build_windows(
                             cursor, s.w0, self.default_window, self.overlap)]
            elif gap > 0:
                # Too small for its own window: let the preserved one
                # start earlier rather than leave a sliver uncovered.
                s.w0 = cursor
            tail.append(s)
            cursor = max(cursor, s.w1 - step_back)
        if self.spec.wmax - cursor > sliver:
            tail += [WindowState(w0, w1, fitter_kind=self.init_fitter,
                                 degree=self.init_degree)
                     for (w0, w1) in _build_windows(
                         cursor, self.spec.wmax, self.default_window,
                         self.overlap)]
        tail.sort(key=lambda s: s.w0)
        self.states[self.idx + 1:] = tail

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

    def _switch_model(self, st: WindowState, kind: str, degree: int):
        """Change this window's model, reverting if the new one can't fit.

        Losing a good fit to a mistyped key would be unkind, so if the
        requested model cannot be fitted with the nodes available the
        previous model and its fit are restored.
        """
        prev = (st.fitter_kind, st.degree, st.fitter, st.continuum,
                st.cont_err, st.rejected)
        st.fitter_kind, st.degree = kind, degree
        self._fit_current(quiet=True)
        if st.fitter is None and prev[2] is not None:
            (st.fitter_kind, st.degree, st.fitter, st.continuum,
             st.cont_err, st.rejected) = prev
            self._draw(message=f"Kept {prev[2].label()} — "
                               f"not enough nodes for the requested model")

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
                self._clear_fit(st)
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
            st.fitter = fitter
            # Map clipped points back onto the window for display.
            rejected_global = np.zeros(len(self.spec), dtype=bool)
            rejected_global[np.flatnonzero(use)] = ~fitter.keep
            st.rejected = rejected_global[sel]
        else:
            if len(st.nodes_x) < fitter.min_nodes:
                self._clear_fit(st)
                self._draw(message=f"Need >= {fitter.min_nodes} nodes for "
                                   f"{fitter.label()} ({len(st.nodes_x)} placed)")
                return
            fitter.fit(st.nodes_x, st.nodes_y, st.nodes_e)
            st.continuum = fitter(self.spec.wavelength[sel])
            st.cont_err = fitter.uncertainty(self.spec.wavelength[sel])
            st.fitter = fitter
        self._draw()

    def _draw(self, message: str = ""):
        ax, st = self._ax, self.states[self.idx]
        ax.clear()
        sel = self._window_sel(st)
        w = self.spec.wavelength[sel]
        f = self.spec.flux[sel]
        g = self.good[sel]
        masked = self.spec.mask[sel]
        excluded = self.spec.exclude[sel]

        ax.plot(w[g], f[g], color="0.2", lw=0.8, drawstyle="steps-mid",
                label="flux")
        bad = ~g & ~masked
        if bad.any():
            ax.plot(w[bad], f[bad], ".", color="0.75", ms=3, label="bad (DQ)")
        if masked.any():
            ax.plot(w[masked], f[masked], color="darkorange", lw=0.6,
                    alpha=0.7, drawstyle="steps-mid",
                    label="fit-masked (kept in output)")
        if excluded.any():
            ax.plot(w[excluded], f[excluded], color="lightcoral", lw=0.6,
                    alpha=0.6, drawstyle="steps-mid",
                    label="excluded (masked in output)")
        for (m0, m1) in self.spec.meta.get("fit_mask_regions", []):
            if m1 >= st.w0 and m0 <= st.w1:
                ax.axvspan(max(m0, st.w0), min(m1, st.w1),
                           color="darkorange", alpha=0.10, zorder=0)
        for (m0, m1) in self.spec.meta.get("exclude_regions", []):
            if m1 >= st.w0 and m0 <= st.w1:
                ax.axvspan(max(m0, st.w0), min(m1, st.w1),
                           color="red", alpha=0.13, zorder=0)
        if self._mask_start is not None:
            ax.axvline(self._mask_start, color="red", ls="--", lw=1)
        if st.nodes_x:
            ax.plot(st.nodes_x, st.nodes_y, "o", color="tab:red", ms=8,
                    mec="k", zorder=5, label="nodes")
        # Evaluate the fit from the stored model so it stays visible
        # after zooming, panning or navigating between windows.
        if st.fitter is not None and w.size:
            st.continuum = st.fitter(w)
            st.cont_err = st.fitter.uncertainty(w)
        elif st.continuum is not None and st.continuum.size != w.size:
            st.continuum = st.cont_err = None
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
                and st.rejected.size == w.size and st.rejected.any():
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
        mode = ("   *** FIT-MASK MODE ***" if self._mask_mode == "fit"
                else "   *** EXCLUDE MODE ***" if self._mask_mode == "exclude"
                else "")
        if st.accepted:
            status = "ACCEPTED"
        elif st.fitter is not None:
            status = "fitted, not accepted"
        else:
            status = "no fit yet"
        ax.set_title(f"Window {self.idx + 1}/{len(self.states)}   "
                     f"[{st.w0:.1f}–{st.w1:.1f}]  ({st.w1 - st.w0:.1f} wide)   "
                     f"model: {model}   [{status}]   "
                     f"accepted: {n_acc}/{len(self.states)}{mode}",
                     color=("crimson" if self._mask_mode
                            else "darkgreen" if st.accepted else "black"))
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
        self._draw_map()
        self._fig.canvas.draw_idle()

    def _draw_map(self):
        """Coverage strip: which parts of the spectrum have accepted fits."""
        ax = self._ax_map
        if ax is None:
            return
        ax.clear()
        lo, hi = self.spec.wmin, self.spec.wmax
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.axhspan(0, 1, color="0.88", zorder=0)
        for (a, b) in self._covered_spans():
            ax.axvspan(a, b, color="tab:green", alpha=0.55, lw=0, zorder=1)
        for (m0, m1) in self.spec.meta.get("fit_mask_regions", []):
            ax.axvspan(m0, m1, color="darkorange", alpha=0.35, lw=0, zorder=2)
        for (m0, m1) in self.spec.meta.get("exclude_regions", []):
            ax.axvspan(m0, m1, color="red", alpha=0.40, lw=0, zorder=2)
        cur = self.states[self.idx]
        ax.axvspan(cur.w0, cur.w1, facecolor="none", edgecolor="k",
                   lw=1.8, zorder=3)
        n_gap = len(self._coverage_gaps())
        ax.set_xlabel(
            "coverage: green = accepted fit, orange = fit-masked, "
            "red = excluded, box = current "
            f"window   ({n_gap} unfitted region{'' if n_gap == 1 else 's'} "
            "left; click to jump)", fontsize=8)
        ax.tick_params(labelsize=7)

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
        return self.assemble_on(self.spec)

    def assemble_on(self, spectrum: Spectrum) -> NormalizedSpectrum:
        """Evaluate the accepted continuum fits on an arbitrary spectrum.

        This is how native-resolution output is produced when the fit
        was done on binned data: the fitted models themselves are
        re-evaluated on the target wavelength grid (exact — no
        interpolation of binned arrays), blended with the same ramp
        weights, and the interactive mask regions are re-applied to the
        target pixels.
        """
        wave = spectrum.wavelength
        cont = np.zeros_like(wave)
        cerr = np.zeros_like(wave)
        weight = np.zeros_like(wave)

        for st in self.states:
            if not st.accepted or st.fitter is None:
                continue
            sel = (wave >= st.w0) & (wave <= st.w1)
            # Binned fitting grids start/end inside the native range
            # (bin centers), so windows touching the ends of the
            # fitting grid also cover the sub-bin overhang beyond it.
            if st.w0 <= self.spec.wmin:
                sel |= wave < st.w0
            if st.w1 >= self.spec.wmax:
                sel |= wave > st.w1
            if not sel.any():
                continue
            w = wave[sel]
            c = st.fitter(w)
            e = st.fitter.uncertainty(w)
            # Linear ramp weights -> smooth blending in overlap regions.
            span = max(st.w1 - st.w0, 1e-30)
            ramp = np.clip(np.minimum(w - st.w0, st.w1 - w) / span,
                           0.0, None) + 1e-3
            cont[sel] += c * ramp
            if e is not None:
                cerr[sel] += np.nan_to_num(e) * ramp
            weight[sel] += ramp

        covered = weight > 0
        with np.errstate(invalid="ignore"):
            denom = np.where(covered, weight, 1.0)
            cont = np.where(covered, cont / denom, np.nan)
            cerr = np.where(covered, cerr / denom, np.nan)

        # Only *exclusions* reach the output: fit masks hide features
        # from the continuum fit but leave the data intact for analysis.
        exclude_regions = list(self.spec.meta.get("exclude_regions", []))
        fit_regions = list(self.spec.meta.get("fit_mask_regions", []))
        mask = spectrum.exclude.copy()
        for (m0, m1) in exclude_regions:
            mask |= (wave >= m0) & (wave <= m1)

        meta = dict(spectrum.meta)
        meta["specnorm"] = {
            "fit_mask_regions": fit_regions,
            "exclude_regions": exclude_regions,
            "binning": spectrum.meta.get("binning", 1),
            "fit_binning": self.spec.meta.get("binning", 1),
            "windows": [
                {"range": [s.w0, s.w1], "model": s.fitter_kind,
                 "degree": s.degree, "n_nodes": len(s.nodes_x),
                 "n_clipped": int(s.rejected.sum()) if s.rejected is not None else 0,
                 "accepted": s.accepted}
                for s in self.states
            ],
        }
        return NormalizedSpectrum(wave, spectrum.flux, cont,
                                  spectrum.error, mask=mask,
                                  cont_err=cerr, meta=meta)


def normalize_interactive(spectrum: Spectrum, window: float = 20.0,
                          overlap: float = 0.15, fitter: str = "spline",
                          degree: int = 3,
                          output_on: Optional[Spectrum] = None,
                          **kwargs) -> NormalizedSpectrum:
    """Convenience wrapper: run the GUI and return the result.

    If ``output_on`` is given (e.g. the unbinned spectrum when
    ``spectrum`` is binned for fitting), the accepted fits are
    evaluated on that spectrum's wavelength grid instead.
    """
    gui = ContinuumGUI(spectrum, window=window, overlap=overlap,
                       fitter=fitter, degree=degree, **kwargs)
    result = gui.run()
    if output_on is not None:
        return gui.assemble_on(output_on)
    return result
