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
* **+** / **-** zoom the *view* in / out and the **left/right arrows**
  pan it; the **up/down arrows** zoom the flux axis about the continuum
  level and **0** resets the view.  Looking around never changes which
  stretch a fit governs, so the continuum on screen stays put: the view
  becomes the window's range only when you refit (**f**).  Zoom out
  across a broad damped feature (Galactic Ly-alpha at 1215 A), place
  nodes on the shoulders and press **f**, and that single fit takes over
  the whole visible range;
* the blue curve is the **knitted continuum** — every accepted fit
  blended together, exactly as it will be written out — so a join that
  did not knit well is visible from any window and at any zoom.  The
  window you are editing is previewed in it at top precedence until you
  accept, and its own model is drawn separately as a dashed purple line
  whenever the two differ.  Stretches with no fit yet appear as gaps in
  the curve, marked along the bottom;
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
    "lines; nodes set ref level)   1-5: degree   d: drop overlapping fits   "
    "+/-: zoom in/out   "
    "left/right: pan   up/down: y-zoom   0: reset view   "
    "enter/a: accept & next   b: back   q: quit & save"
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
    accept_seq: int = -1                    # order of acceptance (recency)


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
    session_path : str, optional
        If given, the full session (settings, masks, every window with
        its nodes and model) is saved here as JSON each time a window
        is accepted, so the work can be resumed or edited later.
    source : dict, optional
        How the spectrum was loaded (input path, extension, binning),
        recorded in the session file so it can be reopened.
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
                 session_path: Optional[str] = None,
                 low_rej: float = 1.5, high_rej: float = 3.5,
                 niterate: int = 20, grow: int = 6, min_pix: int = 3,
                 source: Optional[dict] = None):
        if len(spectrum) < 2:
            raise ValueError("Spectrum has fewer than 2 points")
        self.spec = spectrum
        self.mask_dq = mask_dq
        self.masked_path = masked_path
        self.session_path = session_path
        self.source = dict(source or {})
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
        self._accept_seq = 0
        # A fresh run may close itself once the whole spectrum is
        # covered; a resumed one never does, since you are editing and
        # decide yourself when to stop (q).
        self.finish_when_complete = True
        self._knit_note = ""
        self._y_zoom = 1.0        # >1 zooms in on the continuum level
        # The displayed range is tracked separately from each window's
        # governing span: looking around must never change which region
        # a fit controls in the knitted continuum.
        self._view_override = None
        self._view_idx = -1
        self.y_zoom_factor = 1.4
        # In an overlap the more recently accepted fit dominates by this
        # factor per generation, so re-fitting a region supersedes what
        # was there before without a discontinuity.
        self.recency_base = 3.0
        # Precedence is capped at this many generations: an unbounded
        # ratio would push the crossover hard against a window edge and
        # turn a knit into a step.
        self.recency_levels = 1
        self._mask_mode: Optional[str] = None  # None | "fit" | "exclude"
        self._mask_start: Optional[float] = None  # first edge of pending mask
        self._ax_map = None                       # coverage strip axes

    @staticmethod
    def _clear_fit(st: WindowState):
        """Discard a window's fit (it is stale, e.g. nodes changed).

        The window also stops counting as accepted: without a model it
        contributes nothing to the knitted continuum, so saying it was
        accepted would misreport the coverage.
        """
        st.continuum = None
        st.cont_err = None
        st.rejected = None
        st.fitter = None
        st.accepted = False

    def _refresh_good(self):
        self.good = (self.spec.good_mask(use_dq=self.mask_dq, use_mask=True)
                     & ~self._no_data(self.spec))
        self._dead_cache = None
        self._ref_cache = None

    @staticmethod
    def _no_data(spectrum: Spectrum) -> np.ndarray:
        """Pixels that carry no measurement at all.

        Detector edges are commonly padded with exactly zero flux and
        zero error.  Such pixels must not anchor a continuum node — a
        node dropped there would drag the fit to zero — and they need no
        fit of their own.
        """
        nodata = ~np.isfinite(spectrum.flux)
        if np.any(spectrum.error > 0):
            nodata |= (spectrum.flux == 0) & (spectrum.error <= 0)
        else:
            nodata |= (spectrum.flux == 0)
        return nodata

    def _dead_mask(self, spectrum: Optional[Spectrum] = None) -> np.ndarray:
        """Pixels that cannot anchor a fit and need none.

        Spectrum edges frequently hold zero-flux or fully masked pixels;
        they should not be reported as unfitted gaps, and the continuum
        there is filled by extending the nearest fit.
        """
        if spectrum is None:
            if getattr(self, "_dead_cache", None) is not None:
                return self._dead_cache
            spectrum = self.spec
            cache = True
        else:
            cache = spectrum is self.spec
        dead = (~spectrum.good_mask(use_dq=self.mask_dq, use_mask=True)
                | self._no_data(spectrum))
        if cache:
            self._dead_cache = dead
        return dead

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
        self._save_session()
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
                self._detach_accepted(st)
                st.nodes_x.append(float(event.xdata))
                st.nodes_y.append(float(sample[0]))
                st.nodes_e.append(float(sample[1]))
                self._clear_fit(st)
                self._draw()
            else:
                self._draw(message="No unmasked data near that click")
        elif event.button == 3 and st.nodes_x:
            self._detach_accepted(st)
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
                self._detach_accepted(st)
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
            self._detach_accepted(st)
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
            had_gaps = bool(self._coverage_gaps())
            st.accepted = True
            st.accept_seq = self._accept_seq
            self._accept_seq += 1
            self._knit_accepted()
            st = self.states[self.idx]      # knitting may reorder the list
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
            if self._knit_note:
                note = (note + "   " if note else "") + self._knit_note
                self._knit_note = ""
            nxt = self._next_needing_work(self.idx + 1)
            if nxt is None and self.finish_when_complete and had_gaps:
                # Reached full coverage by working through the spectrum.
                self._finished = True
                self._save_session()
                plt.close(self._fig)
            elif nxt is None:
                # Everything is covered but you are editing: stay open.
                self._save_session()
                if self.idx + 1 < len(self.states):
                    self.idx += 1
                self._mask_mode = None
                self._mask_start = None
                self._draw(message=(note + "   " if note else "")
                           + "All regions covered — press q to save and quit")
            else:
                wrapped = nxt <= self.idx
                self.idx = nxt
                self._mask_mode = None
                self._mask_start = None
                if wrapped and not note:
                    remaining = len(self._coverage_gaps())
                    note = (f"Back to an unfitted region "
                            f"({remaining} left) — press q to save and stop")
                self._save_session()
                self._draw(message=note)
        elif key in ("+", "="):
            self._resize_window(-1)      # '+' = zoom in (narrower window)
        elif key in ("-", "_"):
            self._resize_window(+1)      # '-' = zoom out (wider window)
        elif key == "up":
            self._zoom_y(+1)
        elif key == "down":
            self._zoom_y(-1)
        elif key == "d":
            self._drop_overlapping_fits()
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
            self._save_session()
            plt.close(self._fig)

    # ------------------------------------------------------------------
    # Fitting / drawing
    # ------------------------------------------------------------------
    def _view_range(self):
        """The wavelength range on display.

        Defaults to the current window's governing span, but zooming and
        panning override it without touching that span.
        """
        st = self.states[self.idx]
        if self._view_override is not None and self._view_idx == self.idx:
            return self._view_override
        return (st.w0, st.w1)

    def _set_view(self, v0: float, v1: float):
        self._view_override = (max(v0, self.spec.wmin),
                               min(v1, self.spec.wmax))
        self._view_idx = self.idx

    def _view_sel(self) -> np.ndarray:
        v0, v1 = self._view_range()
        return ((self.spec.wavelength >= v0) & (self.spec.wavelength <= v1))

    def _window_sel(self, st: WindowState) -> np.ndarray:
        return (self.spec.wavelength >= st.w0) & (self.spec.wavelength <= st.w1)

    # ------------------------------------------------------------------
    # Window resizing (zoom out across broad features, then back in)
    # ------------------------------------------------------------------
    def _resize_window(self, direction: int):
        """Widen the view (direction>0) or narrow it (direction<0).

        Only the display changes: the window keeps governing the same
        stretch of spectrum, so the knitted continuum on screen stays
        put while you look around.  Fitting adopts the view as the new
        span (see :meth:`_fit_current`).
        """
        if self.default_window <= 0:
            self._draw(message="Zoom is unavailable when fitting the "
                               "whole spectrum at once (-w 0)")
            return
        v0, v1 = self._view_range()
        factor = self.zoom_factor if direction > 0 else 1.0 / self.zoom_factor
        centre = 0.5 * (v0 + v1)
        half = 0.5 * (v1 - v0) * factor
        new_v0 = max(centre - half, self.spec.wmin)
        new_v1 = min(centre + half, self.spec.wmax)
        if direction < 0 and new_v1 - new_v0 < self.min_window:
            self._draw(message=f"Minimum view width is "
                               f"{self.min_window:.2f}")
            return
        if (new_v0, new_v1) == (v0, v1):
            self._draw(message="Already showing the full spectrum")
            return
        self._set_view(new_v0, new_v1)
        self._draw(message=f"View {new_v1 - new_v0:.2f} wide "
                           "(f refits over this range)")

    def _pan_window(self, direction: int):
        """Shift the view left or right, keeping its width."""
        if self.default_window <= 0:
            return
        v0, v1 = self._view_range()
        width = v1 - v0
        shift = direction * self.pan_frac * width
        new_v0, new_v1 = v0 + shift, v1 + shift
        if new_v0 < self.spec.wmin:
            new_v0, new_v1 = self.spec.wmin, self.spec.wmin + width
        if new_v1 > self.spec.wmax:
            new_v0, new_v1 = self.spec.wmax - width, self.spec.wmax
        if abs(new_v0 - v0) < 1e-12:
            self._draw(message="At the edge of the spectrum")
            return
        self._set_view(new_v0, new_v1)
        self._draw(message=f"View [{new_v0:.2f}, {new_v1:.2f}]")

    def _zoom_y(self, direction: int):
        """Zoom the flux axis in (direction>0) or out, about the continuum."""
        if direction > 0:
            self._y_zoom *= self.y_zoom_factor
        else:
            self._y_zoom = max(self._y_zoom / self.y_zoom_factor, 0.05)
        self._draw(message=f"y-zoom x{self._y_zoom:.2f}"
                           + ("  (0 resets)" if self._y_zoom != 1.0 else ""))

    def _reset_window(self):
        """Drop any zoom or pan: show exactly what this window governs."""
        self._y_zoom = 1.0
        self._view_override = None
        self._draw(message="View reset to this window's range")

    def _detach_accepted(self, st: WindowState) -> bool:
        """Keep an accepted fit alive as its own window before it is edited.

        Editing the nodes of a window that already has an accepted fit,
        or refitting it over a different range, means you are redoing
        it.  The fit that was already agreed is detached into a window
        of its own first, so the coverage it provided survives until the
        replacement is accepted (knitting retires it if the new fit
        covers the same ground).
        """
        if not (st.accepted and st.fitter is not None):
            return False
        preserved = WindowState(
            w0=st.w0, w1=st.w1, nodes_x=list(st.nodes_x),
            nodes_y=list(st.nodes_y), nodes_e=list(st.nodes_e),
            fitter_kind=st.fitter_kind, degree=st.degree,
            fitter=st.fitter, resized=True, accepted=True,
            accept_seq=st.accept_seq)
        st.accepted = False
        st.accept_seq = -1
        self.states.append(preserved)
        self.states.sort(key=lambda s: (s.w0, s.w1))
        self.idx = next(i for i, s in enumerate(self.states) if s is st)
        self._view_idx = self.idx
        return True

    def _adopt_view(self, st: WindowState) -> bool:
        """Make the view the window's governing span before fitting.

        If the window already had an accepted fit over a different
        stretch, that fit is detached as a window of its own first, so
        re-fitting a sub-region does not silently drop coverage that was
        already agreed.  Returns True if the span changed.
        """
        v0, v1 = self._view_range()
        if abs(v0 - st.w0) < 1e-12 and abs(v1 - st.w1) < 1e-12:
            return False
        self._detach_accepted(st)
        st.w0, st.w1 = v0, v1
        st.resized = True
        st.continuum = st.cont_err = st.rejected = None
        self._view_idx = self.idx
        return True

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

    def _uncovered(self) -> np.ndarray:
        """Pixels of the fitting spectrum still wanting a continuum."""
        wave = self.spec.wavelength
        need = ~self._dead_mask()
        for (a, b) in self._covered_spans():
            need &= ~((wave >= a) & (wave <= b))
        return need

    def _next_needing_work(self, start: int) -> Optional[int]:
        """Index of the next window with unfitted, fittable pixels.

        Searches forward from ``start`` and then wraps, so reaching the
        end of the spectrum sends you back to anything skipped rather
        than ending the session.  If a gap is not covered by any window
        (panning can leave one), a window is created for it.
        """
        need = self._uncovered()
        if not need.any():
            return None
        wave = self.spec.wavelength
        order = list(range(start, len(self.states))) + list(range(0, start))
        for i in order:
            s = self.states[i]
            sel = (wave >= s.w0) & (wave <= s.w1)
            if sel.any() and need[sel].any():
                return i
        gaps = self._coverage_gaps()
        if not gaps:
            return None
        a, _b = gaps[0]
        width = (self.default_window if self.default_window > 0
                 else self.spec.wmax - self.spec.wmin)
        fresh = WindowState(w0=max(a, self.spec.wmin),
                            w1=min(a + width, self.spec.wmax),
                            fitter_kind=self.init_fitter,
                            degree=self.init_degree, resized=True)
        self.states.append(fresh)
        self.states.sort(key=lambda s: (s.w0, s.w1))
        return next(i for i, s in enumerate(self.states) if s is fresh)

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
        gaps = [(a, b) for (a, b) in gaps if b - a > tol]
        # Drop stretches that hold no fittable data at all (dead edges,
        # fully masked ranges): there is nothing there to fit.
        dead = self._dead_mask()
        alive = []
        for (a, b) in gaps:
            sel = (self.spec.wavelength >= a) & (self.spec.wavelength <= b)
            if sel.any() and not dead[sel].all():
                alive.append((a, b))
        return alive

    def _drop_overlapping_fits(self):
        """Discard other accepted fits overlapping the current window.

        Use when you have come back to redo a region and want the old
        fit gone rather than knitted around: the leftovers are cleared,
        so the region is yours alone once you refit.  Any coverage this
        removes is reported as a gap, so nothing is lost silently.
        """
        cur = self.states[self.idx]
        dropped = []
        keep = []
        for s in self.states:
            if s is cur or not (s.accepted or s.fitter is not None):
                keep.append(s)
                continue
            if s.w1 <= cur.w0 or s.w0 >= cur.w1:
                keep.append(s)
                continue
            dropped.append((s.w0, s.w1))
            if s.w0 >= cur.w0 and s.w1 <= cur.w1:
                continue                      # wholly inside: remove it
            s.accepted = False
            self._clear_fit(s)
            keep.append(s)
        if not dropped:
            self._draw(message="No other fits overlap this window")
            return
        self.states = keep
        self.idx = next(i for i, s in enumerate(self.states) if s is cur)
        ranges = ", ".join(f"{a:.1f}-{b:.1f}" for (a, b) in dropped)
        self._draw(message=f"Dropped {len(dropped)} overlapping fit(s) "
                           f"({ranges}) — refit and accept")

    def _knit_accepted(self):
        """Fit the newly accepted window in among the existing ones.

        The window just accepted takes precedence over anything it
        overlaps: older accepted fits are *trimmed back* to its edge
        (keeping a blend zone so the join stays smooth) rather than
        being thrown away, an older fit that spanned right across the
        new one is split into the parts either side, and windows the
        new fit completely covers are retired.  Nothing else loses its
        work.
        """
        cur = self.states[self.idx]
        blend = (self.overlap * self.default_window
                 if self.default_window > 0 else 0.0)
        survivors = []
        retired = 0
        wings = []
        for s in self.states:
            if s is cur:
                survivors.append(s)
                continue
            has_work = s.accepted or s.fitter is not None
            # Anything the new window swallows whole is redundant.
            if s.w0 >= cur.w0 and s.w1 <= cur.w1:
                retired += 1
                continue
            if not has_work:
                survivors.append(s)
                continue
            if s.w0 < cur.w0 and s.w1 > cur.w1:
                # The old fit spans across the new one: keep both wings.
                right = WindowState(
                    w0=max(cur.w1 - blend, cur.w1 - 0.5 * (s.w1 - cur.w1)),
                    w1=s.w1, nodes_x=list(s.nodes_x), nodes_y=list(s.nodes_y),
                    nodes_e=list(s.nodes_e), fitter_kind=s.fitter_kind,
                    degree=s.degree, fitter=s.fitter, resized=True,
                    accepted=s.accepted, accept_seq=s.accept_seq)
                s.w1 = min(cur.w0 + blend, s.w1)
                survivors.append(s)
                survivors.append(right)
                wings.append((s.w0, s.w1))
                wings.append((right.w0, right.w1))
                continue
            if s.w0 < cur.w0 < s.w1:          # old fit laps in from the left
                s.w1 = max(min(s.w1, cur.w0 + blend), s.w0)
            elif s.w0 < cur.w1 < s.w1:        # old fit laps in from the right
                s.w0 = min(max(s.w0, cur.w1 - blend), s.w1)
            survivors.append(s)

        survivors.sort(key=lambda s: (s.w0, s.w1))
        self.states = survivors
        self.idx = next(i for i, s in enumerate(survivors) if s is cur)
        notes = []
        if wings:
            ranges = ", ".join(f"{a:.1f}-{b:.1f}" for (a, b) in wings)
            notes.append(f"older fit kept either side ({ranges}); "
                         "press d to drop it and refit")
        if retired:
            notes.append(f"{retired} covered window(s) retired")
        self._knit_note = "   ".join(notes)

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
                st.cont_err, st.rejected, st.accepted)
        st.fitter_kind, st.degree = kind, degree
        self._fit_current(quiet=True)
        if st.fitter is None and prev[2] is not None:
            (st.fitter_kind, st.degree, st.fitter, st.continuum,
             st.cont_err, st.rejected, st.accepted) = prev
            self._draw(message=f"Kept {prev[2].label()} — "
                               f"not enough nodes for the requested model")

    def _fit_window(self, st: WindowState) -> Optional[str]:
        """Fit one window in place.  Returns an error message or None.

        Kept free of drawing so sessions can be restored by refitting
        every window without a figure.
        """
        fitter = self._make_fitter(st)
        sel = (self.spec.wavelength >= st.w0) & (self.spec.wavelength <= st.w1)
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
                return (f"Need >= {st.degree + 2} unmasked points "
                        f"for {fitter.label()} ({n_use} available)")
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
                return (f"Need >= {fitter.min_nodes} nodes for "
                        f"{fitter.label()} ({len(st.nodes_x)} placed)")
            fitter.fit(st.nodes_x, st.nodes_y, st.nodes_e)
            st.continuum = fitter(self.spec.wavelength[sel])
            st.cont_err = fitter.uncertainty(self.spec.wavelength[sel])
            st.fitter = fitter
        return None

    def _fit_current(self, quiet: bool = False):
        st = self.states[self.idx]
        widened = self._adopt_view(st)
        st = self.states[self.idx]
        message = self._fit_window(st)
        if message is None and widened:
            message = f"Fit now covers {st.w0:.2f}-{st.w1:.2f}"
        self._draw(message=message or "")

    def _draw(self, message: str = ""):
        ax, st = self._ax, self.states[self.idx]
        ax.clear()
        v0, v1 = self._view_range()
        sel = self._view_sel()
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
            if m1 >= v0 and m0 <= v1:
                ax.axvspan(max(m0, v0), min(m1, v1),
                           color="darkorange", alpha=0.10, zorder=0)
        for (m0, m1) in self.spec.meta.get("exclude_regions", []):
            if m1 >= v0 and m0 <= v1:
                ax.axvspan(max(m0, v0), min(m1, v1),
                           color="red", alpha=0.13, zorder=0)
        if self._mask_start is not None:
            ax.axvline(self._mask_start, color="red", ls="--", lw=1)
        # Mark where this window's fit is extrapolating beyond its own
        # nodes: joins made in these stretches are the ones that knit
        # badly, and placing a node here is the cure.
        if st.fitter is not None:
            support = st.fitter.support
            if support is not None:
                shaded = False
                if support[0] > st.w0 + 1e-9:
                    ax.axvspan(st.w0, min(support[0], st.w1), color="0.5",
                               alpha=0.10, hatch="//", lw=0, zorder=0)
                    shaded = True
                if support[1] < st.w1 - 1e-9:
                    ax.axvspan(max(support[1], st.w0), st.w1, color="0.5",
                               alpha=0.10, hatch="//", lw=0,
                               zorder=0, label="extrapolated" if not shaded
                               else None)
                    shaded = True
                if shaded:
                    ax.plot([], [], color="0.5", alpha=0.5, lw=6,
                            label="extrapolated (no node)")
        if st.nodes_x:
            ax.plot(st.nodes_x, st.nodes_y, "o", color="tab:red", ms=8,
                    mec="k", zorder=5, label="nodes")
        # Evaluate this window's own model so it stays visible after
        # zooming, panning or navigating between windows.
        if st.fitter is not None and w.size:
            st.continuum = st.fitter(w)
            st.cont_err = st.fitter.uncertainty(w)
        elif st.continuum is not None and st.continuum.size != w.size:
            st.continuum = st.cont_err = None

        # The headline curve is the *knitted* continuum — every accepted
        # fit blended together, with this window's fit previewed at top
        # precedence.  That is what will be written out, so joins that
        # did not knit well are visible from any window and at any zoom.
        knit = knit_err = anchor_curve = None
        if w.size:
            # An accepted window is already part of the knit; an
            # unaccepted one is previewed only where nothing else covers.
            knit, knit_err, knit_ok = self.knitted(
                w, preview=None if st.accepted else st)
            if not np.isfinite(knit).any():
                knit = knit_err = None
            # The axis is anchored to the accepted continuum alone, so a
            # half-finished fit can never rescale the view.
            if st.accepted:
                anchor_curve = knit
            else:
                accepted_only, _ae, _ao = self.knitted(w)
                anchor_curve = (accepted_only
                                if np.isfinite(accepted_only).any() else None)
        if knit is not None:
            ax.plot(w, knit, color="tab:blue", lw=2, zorder=4,
                    label="continuum")
            if knit_err is not None and np.isfinite(knit_err).any():
                hi_b, lo_b = knit + knit_err, knit - knit_err
                ax.plot(w, hi_b, color="tab:blue", lw=0.9, ls="--",
                        alpha=0.7, zorder=4)
                ax.plot(w, lo_b, color="tab:blue", lw=0.9, ls="--",
                        alpha=0.7, zorder=4, label=r"$\pm1\sigma$")
                ax.fill_between(w, lo_b, hi_b, color="tab:blue",
                                alpha=0.12, zorder=3)
            # Show where nothing is fitted yet, so unfinished stretches
            # inside the view are unmistakable.
            hole = ~np.isfinite(knit)
            if hole.any():
                ylo_h = np.nanmin(f[g]) if f[g].size else 0.0
                ax.plot(w[hole], np.full(int(hole.sum()), ylo_h), "|",
                        color="0.6", ms=6, zorder=2, label="not fitted")
        # This window's own model, drawn thin over its own span, so you
        # can see what you are editing against the final result.  It is
        # not allowed to influence the axis limits.
        if st.continuum is not None:
            same = (knit is not None
                    and np.allclose(np.nan_to_num(st.continuum),
                                    np.nan_to_num(knit), rtol=1e-6, atol=0))
            if not same:
                ax.plot(w, st.continuum, color="tab:purple", lw=1.0,
                        ls="--", alpha=0.85, zorder=5,
                        label=("this window's fit"
                               if st.accepted else
                               "this window's fit (not accepted)"))
            elif knit is None:
                ax.plot(w, st.continuum, color="tab:blue", lw=2, zorder=4,
                        label="continuum")
        if st.continuum is not None and st.rejected is not None \
                and st.rejected.size == w.size and st.rejected.any():
            ax.plot(w[st.rejected], f[st.rejected], "x", color="tab:orange",
                    ms=4, zorder=3,
                    label=f"clipped ({int(st.rejected.sum())})")

        # --- y autoscale ignoring masked / bad points -------------------
        ax.set_xlim(v0, v1)
        ref = f[g]
        if anchor_curve is not None:
            finite = anchor_curve[np.isfinite(anchor_curve)]
            if finite.size:
                ref = np.concatenate([ref, finite])
        if ref.size:
            lo, hi = float(np.min(ref)), float(np.max(ref))
            pad = 0.07 * (hi - lo) if hi > lo else (abs(hi) * 0.1 or 1.0)
            lo, hi = lo - pad, hi + pad
            if self._y_zoom != 1.0:
                # Zoom about the continuum level, not the middle of the
                # range, so the continuum stays in view as you zoom in.
                if anchor_curve is not None and np.isfinite(anchor_curve).any():
                    anchor = float(np.nanmedian(anchor_curve))
                elif st.continuum is not None and np.isfinite(st.continuum).any():
                    anchor = float(np.nanmedian(st.continuum))
                elif f[g].size:
                    anchor = float(np.median(f[g]))
                else:
                    anchor = 0.5 * (lo + hi)
                anchor = min(max(anchor, lo), hi)
                lo = anchor - (anchor - lo) / self._y_zoom
                hi = anchor + (hi - anchor) / self._y_zoom
            ax.set_ylim(lo, hi)

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
        scope = (f"[{st.w0:.1f}–{st.w1:.1f}]"
                 if abs(v0 - st.w0) < 1e-9 and abs(v1 - st.w1) < 1e-9
                 else f"view [{v0:.1f}–{v1:.1f}], fit covers "
                      f"[{st.w0:.1f}–{st.w1:.1f}]")
        ax.set_title(f"Window {self.idx + 1}/{len(self.states)}   {scope}   "
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
        v0, v1 = self._view_range()
        if abs(v0 - cur.w0) > 1e-9 or abs(v1 - cur.w1) > 1e-9:
            ax.axvspan(v0, v1, facecolor="none", edgecolor="0.35",
                       lw=1.0, ls=":", zorder=3)
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
    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------
    def session_state(self) -> dict:
        """Everything needed to rebuild this session later."""
        from . import __version__
        return {
            "specnorm_version": __version__,
            "source": self.source,
            "settings": {
                "window": self.default_window,
                "overlap": self.overlap,
                "fitter": self.init_fitter,
                "degree": self.init_degree,
                "mask_dq": self.mask_dq,
                "node_box": self.node_box,
                "clip": dict(self.clip),
            },
            "fit_mask_regions": [list(r) for r in
                                 self.spec.meta.get("fit_mask_regions", [])],
            "exclude_regions": [list(r) for r in
                                self.spec.meta.get("exclude_regions", [])],
            "accept_seq": self._accept_seq,
            "windows": [
                {"w0": float(s.w0), "w1": float(s.w1),
                 "nodes_x": [float(v) for v in s.nodes_x],
                 "nodes_y": [float(v) for v in s.nodes_y],
                 "nodes_e": [float(v) for v in s.nodes_e],
                 "model": s.fitter_kind, "degree": int(s.degree),
                 "accepted": bool(s.accepted),
                 "accept_seq": int(s.accept_seq),
                 "resized": bool(s.resized)}
                for s in self.states
            ],
        }

    def save_session(self, path: str) -> str:
        """Write the session to JSON so it can be resumed or edited."""
        import json
        with open(path, "w") as fh:
            json.dump(self.session_state(), fh, indent=1)
        return path

    def _save_session(self):
        if self.session_path is None:
            return
        try:
            self.save_session(self.session_path)
        except Exception as exc:      # never let I/O kill the session
            print(f"Warning: could not write session file: {exc}")

    def restore_session(self, state: dict, refit: bool = True):
        """Rebuild windows, masks and fits from :meth:`session_state`."""
        for key, kind in (("fit_mask_regions", "fit"),
                          ("exclude_regions", "exclude")):
            for (m0, m1) in state.get(key, []):
                self.spec.mask_region(m0, m1, kind=kind)
        self._refresh_good()

        states = []
        for w in state.get("windows", []):
            st = WindowState(
                w0=float(w["w0"]), w1=float(w["w1"]),
                nodes_x=[float(v) for v in w.get("nodes_x", [])],
                nodes_y=[float(v) for v in w.get("nodes_y", [])],
                nodes_e=[float(v) for v in w.get("nodes_e", [])],
                fitter_kind=w.get("model", self.init_fitter),
                degree=int(w.get("degree", self.init_degree)),
                accepted=bool(w.get("accepted", False)),
                accept_seq=int(w.get("accept_seq", -1)),
                resized=bool(w.get("resized", False)))
            states.append(st)
        if states:
            self.states = states
        self._accept_seq = int(state.get("accept_seq", len(self.states)))
        if refit:
            for st in self.states:
                if st.nodes_x or st.fitter_kind == "cheb":
                    if self._fit_window(st) is not None:
                        st.accepted = False      # could not be rebuilt
                else:
                    st.accepted = False
        first = self._next_needing_work(0)
        self.idx = 0 if first is None else first
        return self

    def _write_masked(self):
        if self.masked_path is None:
            return
        from .writer import write_masked
        try:
            write_masked(self.spec, self.masked_path)
        except Exception as exc:  # don't let I/O kill the session
            print(f"Warning: could not write masked file: {exc}")

    def _reference_curve(self):
        """Coarse (wavelength, level) sampling of the local flux level.

        A high percentile of the good flux in bands the width of a
        fitting window: close to the continuum where there is one and
        never wild, so it can anchor the comparison when two overlapping
        fits disagree.  Cached, since it only changes when masks do.
        """
        if getattr(self, "_ref_cache", None) is not None:
            return self._ref_cache
        spectrum = self.spec
        wave, flux = spectrum.wavelength, spectrum.flux
        good = self.good & np.isfinite(flux)
        width = (self.default_window if self.default_window > 0
                 else max((wave[-1] - wave[0]) / 10.0, 1e-30))
        nbin = max(int(np.ceil((wave[-1] - wave[0]) / width)), 1)
        edges = np.linspace(wave[0], wave[-1], nbin + 1)
        centres, levels = [], []
        for i in range(nbin):
            sel = (wave >= edges[i]) & (wave <= edges[i + 1]) & good
            if int(sel.sum()) >= 3:
                centres.append(0.5 * (edges[i] + edges[i + 1]))
                levels.append(float(np.percentile(flux[sel], 80)))
        if not centres:
            fallback = float(np.nanmedian(flux[good])) if good.any() else 1.0
            if not np.isfinite(fallback) or fallback <= 0:
                fallback = 1.0
            centres, levels = [float(wave[0]), float(wave[-1])], [fallback] * 2
        levels = np.asarray(levels, dtype=float)
        positive = levels[levels > 0]
        floor = (float(np.median(positive)) * 1e-3 if positive.size else 1e-30)
        self._ref_cache = (np.asarray(centres, dtype=float), levels,
                           max(floor, 1e-30))
        return self._ref_cache

    def _reference_level(self, spectrum: Spectrum) -> np.ndarray:
        """Local flux level evaluated on a spectrum's wavelength grid."""
        return self._ref_at(spectrum.wavelength)

    def _ref_at(self, wave: np.ndarray) -> np.ndarray:
        centres, levels, floor = self._reference_curve()
        return np.maximum(np.interp(wave, centres, levels), floor)

    def _blend(self, wave: np.ndarray, entries):
        """Knit fits together on a wavelength grid.

        ``entries`` is a sequence of ``(window, priority)`` pairs.  The
        priority is a constant per window — never derived from which
        fits happen to cover a given pixel, since that would make the
        weights jump wherever a neighbour's coverage begins.  Returns
        ``(continuum, uncertainty, weight)``, with weight 0 where
        nothing covers a pixel.  See :meth:`assemble_on` for the
        weighting.
        """
        cont = np.zeros_like(wave)
        cerr = np.zeros_like(wave)
        weight = np.zeros_like(wave)
        pieces = []
        for (st, priority) in entries:
            if st.fitter is None:
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
            pieces.append((st, priority, sel, w, st.fitter(w),
                           st.fitter.uncertainty(w)))
        if not pieces:
            return cont, cerr, weight

        ref = self._ref_at(wave)
        # The closest any candidate comes to the local flux level, used
        # to spot a fit that has run away.
        best_dev = np.full(wave.size, np.inf)
        for (st, _p, sel, w, c, _e) in pieces:
            best_dev[sel] = np.minimum(best_dev[sel], np.abs(c - ref[sel]))
        blend = (self.overlap * self.default_window
                 if self.default_window > 0 else 0.0)

        for (st, priority, sel, w, c, e) in pieces:
            span = max(st.w1 - st.w0, 1e-30)
            # Partition-of-unity taper: each fit ramps in and out over
            # the overlap width with a smoothstep (zero value *and* zero
            # slope at its own edges) and is flat across its interior.
            # Neighbouring ramps are complementary, so the handover is a
            # gradual crossover instead of a step.  The tiny floor keeps
            # a fit from being completely defenceless at its own edge
            # against a diverging neighbour, at a cost of ~0.1% in the
            # blend — invisible, unlike the 5% steps a large floor gave.
            width = min(max(blend, 0.05 * span), 0.5 * span)
            u_in = np.clip((w - st.w0) / width, 0.0, 1.0)
            u_out = np.clip((st.w1 - w) / width, 0.0, 1.0)
            taper = (u_in ** 2 * (3.0 - 2.0 * u_in)
                     * u_out ** 2 * (3.0 - 2.0 * u_out)) + 1.0e-3
            recency = priority
            # How much further than the best available candidate this
            # fit strays from the local flux level, in units of that
            # level: 0 for the closest fit, large for a runaway one.
            excess = (np.abs(c - ref[sel]) - best_dev[sel]) / ref[sel]
            # Quartic so a fit that has run away by orders of magnitude
            # is suppressed decisively, while a fit that merely differs
            # a little is barely touched.
            agreement = 1.0 / (1.0 + np.maximum(excess, 0.0) ** 2) ** 2
            reliability = np.maximum(
                self._support_weight(st, w) * agreement, 1e-6)
            wgt = taper * recency * reliability
            cont[sel] += c * wgt
            if e is not None:
                cerr[sel] += np.nan_to_num(e) * wgt
            weight[sel] += wgt
        return cont, cerr, weight

    def knitted(self, wave: np.ndarray, preview: Optional[WindowState] = None):
        """The knitted continuum on ``wave``, as it would be written out.

        This is always the *final* continuum: every accepted fit blended
        together.  ``preview`` is a window whose fit has not been
        accepted yet; it is added at the **lowest** precedence, so it
        fills stretches nothing accepted covers but never overrides the
        accepted result.  That keeps the displayed continuum stable
        while you zoom and pan — changing the view cannot change what
        the final fit looks like.  A preview is also confined to the
        range its nodes actually constrain, so a narrow fit seen from a
        zoomed-out window is not smeared across the view.
        """
        accepted = [s for s in self.states
                    if s.accepted and s.fitter is not None and s is not preview]
        seq_max = max((s.accept_seq for s in accepted), default=0)
        entries = [
            (s, self.recency_base ** -min(max(seq_max - s.accept_seq, 0),
                                          self.recency_levels))
            for s in accepted
        ]
        if preview is not None and preview.fitter is not None:
            p0, p1 = preview.w0, preview.w1
            support = preview.fitter.support
            if support is not None:
                margin = self.overlap * (self.default_window
                                         if self.default_window > 0
                                         else (p1 - p0))
                lo = max(p0, support[0] - margin)
                hi = min(p1, support[1] + margin)
                if hi > lo:
                    p0, p1 = lo, hi
            shim = WindowState(w0=p0, w1=p1, fitter=preview.fitter,
                               fitter_kind=preview.fitter_kind,
                               degree=preview.degree)
            # Negligible priority: a fit that has not been accepted
            # fills stretches nothing covers, but never measurably
            # affects the accepted continuum anywhere.
            entries.append((shim, 1.0e-12))
        cont, cerr, weight = self._blend(wave, entries)
        covered = weight > 0
        with np.errstate(invalid="ignore"):
            denom = np.where(covered, weight, 1.0)
            cont = np.where(covered, cont / denom, np.nan)
            cerr = np.where(covered, cerr / denom, np.nan)
        return cont, cerr, covered

    @staticmethod
    def _support_weight(st: WindowState, w: np.ndarray) -> np.ndarray:
        """Down-weight a fit where it is extrapolating past its nodes/data."""
        support = st.fitter.support if st.fitter is not None else None
        if support is None:
            support = (st.w0, st.w1)
        s0, s1 = support
        beyond = np.maximum(np.maximum(s0 - w, w - s1), 0.0)
        span = max(s1 - s0, 1e-30)
        return 1.0 / (1.0 + (beyond / (0.25 * span)) ** 2)

    def _assemble(self) -> NormalizedSpectrum:
        return self.assemble_on(self.spec)

    def assemble_on(self, spectrum: Spectrum) -> NormalizedSpectrum:
        """Evaluate and knit the accepted continuum fits onto a spectrum.

        This is also how native-resolution output is produced when the
        fit was done on binned data: the fitted models themselves are
        re-evaluated on the target wavelength grid (exact — no
        interpolation of binned arrays).

        Where windows overlap the fits are combined with weights built
        from three factors:

        * an **edge taper**, so joins are smooth rather than stepped;
        * **recency** — a fit accepted later dominates one accepted
          earlier, so re-fitting a region supersedes what was there;
        * **reliability** — a fit is down-weighted where it extrapolates
          beyond its own nodes, and where it strays much further from
          the local flux level than a competing fit does.  This is what
          stops a high-order fit that runs away at its edge from
          dragging the knitted continuum with it.
        """
        wave = spectrum.wavelength
        cont, cerr, covered = self.knitted(wave)

        # Pixels with no measurement — the zero-flux, zero-error padding
        # at detector edges — get the nearest fit extended over them
        # instead of whatever the model happens to do out there.
        # np.interp holds its end values, so edges come out flat and any
        # interior data-less stretch is bridged linearly.  Masked pixels
        # that *do* carry data keep the continuum drawn across them.
        n_filled = 0
        nodata = self._no_data(spectrum)
        usable = covered & ~nodata
        if nodata.any() and usable.any():
            src = np.flatnonzero(usable)
            cont[nodata] = np.interp(wave[nodata], wave[src], cont[src])
            cerr[nodata] = np.interp(wave[nodata], wave[src], cerr[src])
            n_filled = int(nodata.sum())
            covered = covered | nodata

        # Only *exclusions* reach the output: fit masks hide features
        # from the continuum fit but leave the data intact for analysis.
        exclude_regions = list(self.spec.meta.get("exclude_regions", []))
        fit_regions = list(self.spec.meta.get("fit_mask_regions", []))
        mask = spectrum.exclude.copy()
        for (m0, m1) in exclude_regions:
            mask |= (wave >= m0) & (wave <= m1)

        meta = dict(spectrum.meta)
        meta["specnorm"] = {
            "dead_pixels_filled": n_filled,
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


def load_session(path: str):
    """Read a session file and rebuild the spectrum and GUI from it.

    Returns ``(gui, native_spectrum)`` where ``native_spectrum`` is the
    unbinned spectrum to write output on (None if no binning was used).
    """
    import json
    from .io import read_spectrum
    from .spectrum import bin_spectrum

    with open(path) as fh:
        state = json.load(fh)
    src = state.get("source", {})
    infile = src.get("input")
    if not infile:
        raise ValueError(f"{path} does not record which file it came from")
    spec = read_spectrum(infile, ext=src.get("ext"))
    native = None
    nbin = int(src.get("bin", 1) or 1)
    if nbin > 1:
        native = spec
        spec = bin_spectrum(spec, nbin)

    cfg = state.get("settings", {})
    gui = ContinuumGUI(
        spec,
        window=float(cfg.get("window", 20.0)),
        overlap=float(cfg.get("overlap", 0.15)),
        fitter=cfg.get("fitter", "spline"),
        degree=int(cfg.get("degree", 3)),
        node_box=cfg.get("node_box"),
        mask_dq=bool(cfg.get("mask_dq", True)),
        session_path=path,
        source=src,
        **cfg.get("clip", {}))
    gui.restore_session(state)
    # You are editing: decide yourself when to stop.
    gui.finish_when_complete = False
    return gui, native


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
