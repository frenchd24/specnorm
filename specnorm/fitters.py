"""Continuum models anchored on user-selected nodes.

Each fitter consumes a set of (wavelength, flux) *nodes* — anchor points
placed by the user on the continuum — and produces a callable continuum
over an arbitrary wavelength grid.

* SplineFitter      : cubic spline passing exactly through the nodes
                      (falls back to lower order for few nodes).
* PolynomialFitter  : least-squares polynomial through the nodes,
                      degree 1..5.
* ChebyshevFitter   : least-squares Chebyshev series through the nodes,
                      degree 1..5 by default (numerically better behaved
                      than raw polynomials over wide wavelength spans).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from scipy.interpolate import CubicSpline, interp1d

MAX_POLY_DEGREE = 5


class BaseFitter:
    """Common interface: fit(nodes_x, nodes_y) then __call__(wave)."""

    name = "base"
    min_nodes = 2
    fits_data = False  # True for models fit to the data, not the nodes

    def __init__(self):
        self._model = None
        self.nodes_x: Optional[np.ndarray] = None
        self.nodes_y: Optional[np.ndarray] = None
        self.nodes_err: Optional[np.ndarray] = None

    def fit(self, nodes_x, nodes_y, nodes_err=None) -> "BaseFitter":
        x = np.asarray(nodes_x, dtype=float)
        y = np.asarray(nodes_y, dtype=float)
        if x.size != y.size:
            raise ValueError("nodes_x and nodes_y must be the same length")
        if x.size < self.min_nodes:
            raise ValueError(
                f"{self.label()} needs at least {self.min_nodes} nodes, got {x.size}"
            )
        e = None
        if nodes_err is not None:
            e = np.asarray(nodes_err, dtype=float)
            if e.size != x.size or not np.any(e > 0):
                e = None
        order = np.argsort(x)
        x, y = x[order], y[order]
        if e is not None:
            e = e[order]
        if np.any(np.diff(x) <= 0):
            # Merge duplicate-x nodes by averaging.
            ux, inverse = np.unique(x, return_inverse=True)
            uy = np.zeros_like(ux)
            counts = np.zeros_like(ux)
            np.add.at(uy, inverse, y)
            np.add.at(counts, inverse, 1)
            if e is not None:
                ue = np.zeros_like(ux)
                np.add.at(ue, inverse, e)
                e = ue / counts
            x, y = ux, uy / counts
        self.nodes_x, self.nodes_y, self.nodes_err = x, y, e
        self._fit_impl(x, y)
        return self

    def uncertainty(self, wave) -> Optional[np.ndarray]:
        """1-sigma uncertainty of the continuum, or None if unavailable."""
        return None

    def _fit_impl(self, x, y):  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, wave) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before evaluating the continuum")
        return np.asarray(self._model(np.asarray(wave, dtype=float)))

    def label(self) -> str:
        return self.name


class SplineFitter(BaseFitter):
    """Cubic spline through the nodes (k adapts down for <4 nodes)."""

    name = "spline"
    min_nodes = 2

    def _fit_impl(self, x, y):
        if x.size >= 4:
            self._model = CubicSpline(x, y, bc_type="natural", extrapolate=True)
        elif x.size == 3:
            self._model = interp1d(x, y, kind="quadratic", fill_value="extrapolate")
        else:
            self._model = interp1d(x, y, kind="linear", fill_value="extrapolate")

    def uncertainty(self, wave):
        if self.nodes_err is None:
            return None
        return np.interp(np.asarray(wave, dtype=float),
                         self.nodes_x, self.nodes_err)

    def label(self):
        return "cubic spline"


class PolynomialFitter(BaseFitter):
    """Least-squares polynomial of degree 1-5."""

    name = "poly"

    def __init__(self, degree: int = 3):
        super().__init__()
        if not 1 <= degree <= MAX_POLY_DEGREE:
            raise ValueError(f"degree must be 1..{MAX_POLY_DEGREE}, got {degree}")
        self.degree = degree
        self.min_nodes = degree + 1

    def _fit_impl(self, x, y):
        # Center/scale x for numerical stability.
        x0, sx = x.mean(), max(np.ptp(x) / 2.0, 1e-30)
        xs = (x - x0) / sx
        self._cov = None
        try:
            if self.nodes_err is not None:
                coeffs, cov = np.polyfit(xs, y, self.degree,
                                         w=1.0 / self.nodes_err,
                                         cov="unscaled")
            else:
                coeffs, cov = np.polyfit(xs, y, self.degree, cov=True)
            self._cov = cov
        except (ValueError, np.linalg.LinAlgError):
            coeffs = np.polyfit(xs, y, self.degree)
        self._x0, self._sx = x0, sx
        self._model = lambda w: np.polyval(coeffs, (np.asarray(w) - x0) / sx)

    def uncertainty(self, wave):
        if self._cov is None:
            return None
        u = (np.asarray(wave, dtype=float) - self._x0) / self._sx
        V = np.vander(u, self.degree + 1)  # columns: u^deg ... u^0
        var = np.einsum("ij,jk,ik->i", V, self._cov, V)
        return np.sqrt(np.clip(var, 0, None))

    def label(self):
        return f"degree-{self.degree} polynomial"


class ChebyshevFitter(BaseFitter):
    """Sigma-clipped least-squares Chebyshev fit to the *data*.

    Unlike SplineFitter / PolynomialFitter, which pass through the
    user-placed nodes, this fits all unmasked data points directly and
    iteratively rejects outliers, IRAF ``continuum``-style: points more
    than ``low_rej`` sigma *below* the fit (absorption lines) or
    ``high_rej`` sigma *above* it (emission lines, cosmic rays) are
    discarded and the fit repeated, up to ``niterate`` times.  The
    asymmetric defaults (low 2.0, high 3.5) bite hard on absorption.

    In the GUI, nodes are optional for this model but very useful: one
    or more nodes provide an *initial continuum estimate* (interpolated
    between them) used for a robust first rejection pass — essential
    when many narrow lines would otherwise drag the blind first fit
    down and inflate the residual sigma.  Two or more nodes also
    restrict the fitted range to [first node, last node].

    Parameters
    ----------
    degree : int
        Chebyshev series degree, 1-5.
    low_rej, high_rej : float
        Rejection thresholds in units of the residual standard
        deviation, below/above the fit.  Set <= 0 to disable that side.
    niterate : int
        Maximum number of reject-refit iterations.
    grow : int
        Also reject this many pixels on each side of every rejected
        pixel (helps remove line wings).
    min_pix : int
        Minimum number of *consecutive* pixels below the ``low_rej``
        threshold for them to be rejected.  Real absorption lines are
        resolved into runs of adjacent low pixels; isolated noise dips
        are not, so requiring a run (default 3) keeps the clipper from
        flagging random noise.  Runs may include already-rejected
        neighbours (so line wings next to a clipped core still count).
        Does not apply to ``high_rej`` (cosmic rays are single pixels).
        Set to 1 to disable.
    """

    name = "cheb"
    fits_data = True

    def __init__(self, degree: int = 3, low_rej: float = 1.5,
                 high_rej: float = 3.5, niterate: int = 20, grow: int = 6,
                 min_pix: int = 3):
        super().__init__()
        if not 1 <= degree <= MAX_POLY_DEGREE:
            raise ValueError(f"degree must be 1..{MAX_POLY_DEGREE}, got {degree}")
        self.degree = degree
        self.min_nodes = 0  # nodes only bound the range in the GUI
        self.low_rej = float(low_rej)
        self.high_rej = float(high_rej)
        self.niterate = int(niterate)
        self.grow = int(grow)
        self.min_pix = max(int(min_pix), 1)
        self.keep: Optional[np.ndarray] = None  # final accepted-point mask

    @staticmethod
    def _runs(mask: np.ndarray):
        """Yield (start, stop) index pairs of consecutive True runs."""
        padded = np.concatenate(([False], mask, [False]))
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
        return zip(edges[::2], edges[1::2])

    def _reject(self, keep: np.ndarray, resid: np.ndarray,
                sigma: float) -> np.ndarray:
        """One rejection pass: asymmetric thresholds, run-length, grow."""
        new = keep.copy()
        if self.low_rej > 0:
            below = resid < -self.low_rej * sigma
            if self.min_pix > 1:
                # Only reject low pixels that belong to a run of at least
                # min_pix consecutive "bad" pixels, where already-rejected
                # neighbours count toward the run (so wings adjacent to a
                # clipped line core can still be removed).
                cand = below | ~keep
                allowed = np.zeros_like(below)
                for start, stop in self._runs(cand):
                    if stop - start >= self.min_pix:
                        allowed[start:stop] = True
                new &= ~(below & allowed)
            else:
                new &= ~below
        if self.high_rej > 0:
            new &= resid <= self.high_rej * sigma
        if self.grow > 0:
            rejected = keep & ~new
            kernel = np.ones(2 * self.grow + 1, dtype=bool)
            grown = np.convolve(rejected, kernel, mode="same") > 0
            new &= ~grown
        return new

    def fit_data(self, x, y, err=None, init_x=None,
                 init_y=None) -> "ChebyshevFitter":
        """Iteratively sigma-clipped fit to data points (not nodes).

        If ``init_x``/``init_y`` are given (the user's continuum nodes),
        the flux level interpolated between them is used as the initial
        continuum estimate for a robust (MAD-based) first rejection
        pass, before any Chebyshev fit.  This keeps dense narrow lines
        from poisoning the first fit and its sigma estimate.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < self.degree + 2:
            raise ValueError(
                f"degree-{self.degree} Chebyshev needs at least "
                f"{self.degree + 2} data points, got {int(keep.sum())}"
            )
        w = None
        if err is not None:
            err = np.asarray(err, dtype=float)
            if np.any(err > 0):
                with np.errstate(divide="ignore"):
                    w = np.where(err > 0, 1.0 / err, 0.0)

        # --- node-guided pre-rejection -------------------------------
        if init_x is not None and len(np.atleast_1d(init_x)) >= 1:
            ix = np.atleast_1d(np.asarray(init_x, dtype=float))
            iy = np.atleast_1d(np.asarray(init_y, dtype=float))
            order = np.argsort(ix)
            init_cont = np.interp(x, ix[order], iy[order])
            resid0 = y - init_cont
            med = np.median(resid0[keep])
            sigma0 = 1.4826 * np.median(np.abs(resid0[keep] - med))  # MAD
            if sigma0 > 0:
                pre = self._reject(keep, resid0 - med, sigma0)
                if pre.sum() >= self.degree + 2:
                    keep = pre

        # --- iterative fit-and-reject --------------------------------
        domain = [float(np.min(x[keep])), float(np.max(x[keep]))]
        series = None
        prev_sigma = None
        for _ in range(max(self.niterate, 1)):
            series = _cheb.Chebyshev.fit(
                x[keep], y[keep], deg=self.degree, domain=domain,
                w=w[keep] if w is not None else None)
            resid = y - series(x)
            sigma = float(np.std(resid[keep]))
            if sigma <= 0:
                break
            # Convergence: genuine line rejection slashes sigma between
            # iterations; once sigma stops improving substantially, any
            # further "rejection" is just clipping the noise floor (and
            # grow would amplify it into eating the whole window).
            if prev_sigma is not None and sigma > 0.8 * prev_sigma:
                break
            new = self._reject(keep, resid, sigma)
            if new.sum() < self.degree + 2 or np.array_equal(new, keep):
                break
            keep = new
            prev_sigma = sigma

        self._model = series
        self.keep = keep

        # Parameter covariance for the 1-sigma continuum band:
        # cov = s^2 (A^T W A)^-1 with A the Chebyshev design matrix on
        # the fitted domain and s^2 the (reduced-chi^2) residual scale.
        self._cov = None
        self._domain = domain
        try:
            u = 2.0 * (x[keep] - domain[0]) / max(domain[1] - domain[0],
                                                  1e-30) - 1.0
            A = _cheb.chebvander(u, self.degree)
            r = y[keep] - series(x[keep])
            dof = int(keep.sum()) - (self.degree + 1)
            if dof > 0:
                if w is not None:
                    Aw = A * w[keep][:, None]
                    M = Aw.T @ Aw
                    s2 = float(np.sum((r * w[keep]) ** 2)) / dof
                else:
                    M = A.T @ A
                    s2 = float(np.sum(r ** 2)) / dof
                self._cov = np.linalg.pinv(M) * s2
        except np.linalg.LinAlgError:
            pass
        return self

    def uncertainty(self, wave):
        if self._cov is None:
            return None
        d0, d1 = self._domain
        u = 2.0 * (np.asarray(wave, dtype=float) - d0) / max(d1 - d0,
                                                             1e-30) - 1.0
        A = _cheb.chebvander(u, self.degree)
        var = np.einsum("ij,jk,ik->i", A, self._cov, A)
        return np.sqrt(np.clip(var, 0, None))

    def _fit_impl(self, x, y):
        # Plain least-squares through nodes (API fallback; the GUI uses
        # fit_data for this model).
        self._model = _cheb.Chebyshev.fit(x, y, deg=self.degree)

    def label(self):
        return f"degree-{self.degree} Chebyshev (sigma-clipped)"


def make_fitter(kind: str, degree: int = 3, **clip) -> BaseFitter:
    """Factory: kind in {'spline', 'poly', 'cheb'}.

    Extra keyword arguments (``low_rej``, ``high_rej``, ``niterate``,
    ``grow``) configure the Chebyshev sigma-clipping and are ignored for
    the other kinds.
    """
    kind = kind.lower()
    if kind.startswith("spl"):
        return SplineFitter()
    if kind.startswith("pol"):
        return PolynomialFitter(degree)
    if kind.startswith("che"):
        return ChebyshevFitter(degree, **clip)
    raise ValueError(f"Unknown fitter kind: {kind!r} (use spline/poly/cheb)")
