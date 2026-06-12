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
        User mask (True = masked / excluded), e.g. geocoronal Ly-alpha
        airglow.  Masked points are excluded from node placement, from
        y-axis autoscaling in the GUI, and flagged in the output.
    meta : dict
        Free-form metadata (FITS header cards of interest, source file, ...).
    """

    wavelength: np.ndarray
    flux: np.ndarray
    error: Optional[np.ndarray] = None
    dq: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
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

        n = self.wavelength.size
        if self.flux.size != n or self.error.size != n or self.mask.size != n:
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
        if self.dq is not None:
            self.dq = self.dq[order]

        good = np.isfinite(self.wavelength) & np.isfinite(self.flux)
        if not good.all():
            self.wavelength = self.wavelength[good]
            self.flux = self.flux[good]
            self.error = self.error[good]
            self.mask = self.mask[good]
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
            good &= ~self.mask
        return good

    def mask_region(self, w0: float, w1: float):
        """Mask all points with w0 <= wavelength <= w1 (in place)."""
        w0, w1 = sorted((float(w0), float(w1)))
        self.mask |= (self.wavelength >= w0) & (self.wavelength <= w1)
        self.meta.setdefault("mask_regions", []).append([w0, w1])

    def pop_mask_region(self):
        """Remove the most recently added mask region (LIFO undo).

        The pixel mask is rebuilt from the remaining regions, so
        overlapping regions are handled correctly.  Returns the removed
        (w0, w1) pair, or None if no regions are defined.
        """
        regions = self.meta.get("mask_regions", [])
        if not regions:
            return None
        removed = regions.pop()
        self.mask[:] = False
        for (m0, m1) in regions:
            self.mask |= (self.wavelength >= m0) & (self.wavelength <= m1)
        return tuple(removed)

    def unmask_region(self, w0: float, w1: float):
        """Clear the user mask between w0 and w1 (in place)."""
        w0, w1 = sorted((float(w0), float(w1)))
        sel = (self.wavelength >= w0) & (self.wavelength <= w1)
        self.mask[sel] = False
        regions = self.meta.get("mask_regions", [])
        self.meta["mask_regions"] = [
            r for r in regions if not (r[0] >= w0 and r[1] <= w1)
        ]

    def slice(self, w0: float, w1: float) -> "Spectrum":
        """Return the sub-spectrum with w0 <= wavelength <= w1."""
        sel = (self.wavelength >= w0) & (self.wavelength <= w1)
        return Spectrum(
            self.wavelength[sel],
            self.flux[sel],
            self.error[sel],
            self.dq[sel] if self.dq is not None else None,
            self.mask[sel],
            dict(self.meta),
        )


def bin_spectrum(spec: Spectrum, nbin: int = 2) -> Spectrum:
    """Bin a spectrum by an integer number of pixels.

    Flux and wavelength are averaged; errors are propagated as
    ``sqrt(sum(err**2)) / nbin``; DQ flags are OR-combined; a bin is
    masked if *any* of its pixels is masked.  Trailing pixels that don't
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
    dq = None
    if spec.dq is not None:
        dq = np.bitwise_or.reduce(_r(spec.dq).astype(np.int64), axis=1)

    meta = dict(spec.meta)
    meta["binning"] = nbin * meta.get("binning", 1)
    return Spectrum(wave, flux, err, dq, mask, meta)


@dataclass
class NormalizedSpectrum:
    """Result of a continuum normalization run."""

    wavelength: np.ndarray
    flux: np.ndarray
    continuum: np.ndarray
    error: np.ndarray
    mask: Optional[np.ndarray] = None  # user mask, True = masked
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
