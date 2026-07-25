"""Data containers for spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Spectrum:
    """A 1-D spectrum.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength array (typically Angstroms for STIS/COS).
    flux : np.ndarray
        Flux array (any units; normalization is unit-agnostic).
    error : np.ndarray, optional
        1-sigma flux uncertainty.  If absent, a zero array is used.
    dq : np.ndarray, optional
        Data-quality flags (e.g. STIS/COS DQ column). Non-zero values can
        be masked out before fitting.
    mask : np.ndarray, optional
        *Fit* mask (True = ignore when fitting).  Use it to hide real
        spectral features — absorption lines, broad troughs — from the
        continuum fit.  These pixels are excluded from node placement,
        sigma-clipping and y-axis autoscaling, but they are **not**
        flagged in the output: the data survive for later analysis.
    exclude : np.ndarray, optional
        *Exclusion* mask (True = bad data).  Use it for pixels that are
        unusable rather than merely inconvenient — geocoronal airglow,
        detector artefacts.  These are ignored by the fit **and**
        flagged as masked in every output file.
    meta : dict
        Free-form metadata (FITS header cards of interest, source file, ...).
    """

    wavelength: np.ndarray
    flux: np.ndarray
    error: Optional[np.ndarray] = None
    dq: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    exclude: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.wavelength = np.asarray(self.wavelength, dtype=float).ravel()
        self.flux = np.asarray(self.flux, dtype=float).ravel()
        if self.error is None:
            self.error = np.zeros_like(self.flux)
        else:
            self.error = np.asarray(self.error, dtype=float).ravel()
        if self.dq is not None:
            self.dq = np.asarray(self.dq).ravel()
        if self.mask is None:
            self.mask = np.zeros(self.flux.size, dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool).ravel()
        if self.exclude is None:
            self.exclude = np.zeros(self.flux.size, dtype=bool)
        else:
            self.exclude = np.asarray(self.exclude, dtype=bool).ravel()

        n = self.wavelength.size
        if (self.flux.size != n or self.error.size != n
                or self.mask.size != n or self.exclude.size != n):
            raise ValueError(
                f"Array length mismatch: wavelength={n}, "
                f"flux={self.flux.size}, error={self.error.size}, "
                f"mask={self.mask.size}"
            )

        # Sort by wavelength and drop NaNs / non-finite points.
        order = np.argsort(self.wavelength)
        self.wavelength = self.wavelength[order]
        self.flux = self.flux[order]
        self.error = self.error[order]
        self.mask = self.mask[order]
        self.exclude = self.exclude[order]
        if self.dq is not None:
            self.dq = self.dq[order]

        good = np.isfinite(self.wavelength) & np.isfinite(self.flux)
        if not good.all():
            self.wavelength = self.wavelength[good]
            self.flux = self.flux[good]
            self.error = self.error[good]
            self.mask = self.mask[good]
            self.exclude = self.exclude[good]
            if self.dq is not None:
                self.dq = self.dq[good]

    def __len__(self):
        return self.wavelength.size

    @property
    def wmin(self) -> float:
        return float(self.wavelength[0])

    @property
    def wmax(self) -> float:
        return float(self.wavelength[-1])

    def good_mask(self, use_dq: bool = True, use_mask: bool = True) -> np.ndarray:
        """Boolean mask of points considered usable for fitting."""
        good = np.isfinite(self.flux)
        if use_dq and self.dq is not None:
            good &= (self.dq == 0)
        if use_mask:
            good &= ~self.mask & ~self.exclude
        return good

    # Region bookkeeping ------------------------------------------------
    # kind='fit'     -> self.mask,    meta['fit_mask_regions']
    # kind='exclude' -> self.exclude, meta['exclude_regions']
    _KINDS = {"fit": ("mask", "fit_mask_regions"),
              "exclude": ("exclude", "exclude_regions")}

    def _kind(self, kind: str):
        if kind not in self._KINDS:
            raise ValueError(f"kind must be 'fit' or 'exclude', got {kind!r}")
        attr, key = self._KINDS[kind]
        return getattr(self, attr), self.meta.setdefault(key, [])

    def mask_region(self, w0: float, w1: float, kind: str = "fit"):
        """Mask w0 <= wavelength <= w1 (in place).

        ``kind='fit'`` (default) hides the range from the continuum fit
        only; ``kind='exclude'`` also flags it as bad in the output.
        """
        arr, regions = self._kind(kind)
        w0, w1 = sorted((float(w0), float(w1)))
        arr |= (self.wavelength >= w0) & (self.wavelength <= w1)
        regions.append([w0, w1])

    def pop_mask_region(self, kind: str = "fit"):
        """Remove the most recently added region of ``kind`` (LIFO undo).

        The pixel mask is rebuilt from the remaining regions, so
        overlapping regions are handled correctly.  Returns the removed
        (w0, w1) pair, or None if no regions are defined.
        """
        arr, regions = self._kind(kind)
        if not regions:
            return None
        removed = regions.pop()
        arr[:] = False
        for (m0, m1) in regions:
            arr |= (self.wavelength >= m0) & (self.wavelength <= m1)
        return tuple(removed)

    def unmask_region(self, w0: float, w1: float, kind: str = "fit"):
        """Clear a mask of ``kind`` between w0 and w1 (in place)."""
        arr, regions = self._kind(kind)
        w0, w1 = sorted((float(w0), float(w1)))
        arr[(self.wavelength >= w0) & (self.wavelength <= w1)] = False
        _, key = self._KINDS[kind]
        self.meta[key] = [r for r in regions
                          if not (r[0] >= w0 and r[1] <= w1)]

    @property
    def mask_regions(self):
        """All masked ranges, fit-only and excluded, for display."""
        return (list(self.meta.get("fit_mask_regions", []))
                + list(self.meta.get("exclude_regions", [])))

    def slice(self, w0: float, w1: float) -> "Spectrum":
        """Return the sub-spectrum with w0 <= wavelength <= w1."""
        sel = (self.wavelength >= w0) & (self.wavelength <= w1)
        return Spectrum(
            self.wavelength[sel],
            self.flux[sel],
            self.error[sel],
            self.dq[sel] if self.dq is not None else None,
            self.mask[sel],
            self.exclude[sel],
            dict(self.meta),
        )


def bin_spectrum(spec: Spectrum, nbin: int = 2) -> Spectrum:
    """Bin a spectrum by an integer number of pixels.

    Flux and wavelength are averaged; errors are propagated as
    ``sqrt(sum(err**2)) / nbin``; DQ flags are OR-combined; a bin is
    masked (or excluded) if *any* of its pixels is.  Trailing pixels that don't
    fill a complete bin are dropped.
    """
    nbin = int(nbin)
    if nbin <= 1:
        return spec
    n = (len(spec) // nbin) * nbin
    if n == 0:
        raise ValueError(f"Spectrum too short ({len(spec)} px) to bin by {nbin}")

    def _r(a):
        return np.asarray(a)[:n].reshape(-1, nbin)

    wave = _r(spec.wavelength).mean(axis=1)
    flux = _r(spec.flux).mean(axis=1)
    err = np.sqrt((_r(spec.error) ** 2).sum(axis=1)) / nbin
    mask = _r(spec.mask).any(axis=1)
    exclude = _r(spec.exclude).any(axis=1)
    dq = None
    if spec.dq is not None:
        dq = np.bitwise_or.reduce(_r(spec.dq).astype(np.int64), axis=1)

    meta = dict(spec.meta)
    meta["binning"] = nbin * meta.get("binning", 1)
    return Spectrum(wave, flux, err, dq, mask, exclude, meta)


@dataclass
class NormalizedSpectrum:
    """Result of a continuum normalization run."""

    wavelength: np.ndarray
    flux: np.ndarray
    continuum: np.ndarray
    error: np.ndarray
    mask: Optional[np.ndarray] = None  # excluded pixels, True = masked
    cont_err: Optional[np.ndarray] = None  # 1-sigma continuum uncertainty
    meta: dict = field(default_factory=dict)

    @property
    def norm_flux(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.continuum != 0, self.flux / self.continuum, np.nan)

    @property
    def norm_error(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.continuum != 0, self.error / self.continuum, np.nan)
