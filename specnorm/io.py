"""Flexible spectrum readers.

Handles, in rough order of attempt:

1. HST x1d/sx1-style FITS bintables (STIS, COS): one row per echelle order
   or detector segment, with vector columns WAVELENGTH / FLUX / ERROR / DQ.
   Rows are concatenated and sorted in wavelength.
2. Generic FITS bintables with scalar rows and a variety of column names
   (WAVE, WAVELENGTH, LAMBDA, FLUX, ERR, ERROR, SIGMA, ...).
3. FITS image HDUs: flux in the data array, wavelength reconstructed from
   the WCS keywords (CRVAL1 / CDELT1 or CD1_1 / CRPIX1).
4. Plain-text / CSV files with 2+ columns (wave, flux[, error]).

astropy is imported lazily so the rest of the package works without it
(e.g. for ASCII spectra).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from .spectrum import Spectrum

# Case-insensitive aliases for each physical quantity.
WAVE_ALIASES = (
    "wavelength", "wave", "lambda", "lam", "wavelen", "wl",
    "loglam",  # SDSS-style log10(lambda); handled specially
)
FLUX_ALIASES = ("flux", "flux_density", "fluxdensity", "fnu", "flam", "spec", "sci", "net")
ERROR_ALIASES = (
    "error", "err", "sigma", "stdev", "std", "flux_err", "flux_error",
    "fluxerr", "uncertainty", "unc", "ivar", "e_flux", "noise",
)
DQ_ALIASES = ("dq", "quality", "flag", "flags", "mask", "and_mask")
NELEM_ALIASES = ("nelem", "npix", "npts", "npoints")


def _make_spectrum(wave, flux, error, dq, meta) -> Spectrum:
    """Assemble a Spectrum, dropping unphysical (<= 0) wavelengths.

    HST x1d-style tables pad their fixed-length vector rows beyond NELEM
    with fill values (0, -1, ...); any that survive NELEM trimming — or
    appear in files without an NELEM column — are removed here so they
    can't end up sorted to the front of the spectrum and written out.
    """
    wave = np.asarray(wave, dtype=float).ravel()
    flux = np.asarray(flux, dtype=float).ravel()
    error = np.asarray(error, dtype=float).ravel() if error is not None else None
    dq = np.asarray(dq).ravel() if dq is not None else None

    with np.errstate(invalid="ignore"):
        unphys = ~(wave > 0)  # negative, zero (NaN handled by Spectrum)
    unphys &= np.isfinite(wave)
    n_bad = int(np.count_nonzero(unphys))
    if n_bad:
        keepers = ~unphys
        wave, flux = wave[keepers], flux[keepers]
        if error is not None:
            error = error[keepers]
        if dq is not None:
            dq = dq[keepers]
        meta["n_unphysical_dropped"] = n_bad
    return Spectrum(wave, flux, error, dq, meta=meta)


def _find_column(colnames: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    """Return the actual column name matching one of the aliases.

    Exact (case-insensitive) matches are preferred; then prefix matches
    such as ``WAVELENGTH_VAC`` for the alias ``wavelength``.
    """
    lower = {name.lower(): name for name in colnames}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    for alias in aliases:
        for low, orig in lower.items():
            if low.startswith(alias):
                return orig
    return None


def _column_data(value, transform_loglam: bool = False) -> np.ndarray:
    arr = np.asarray(value)
    arr = arr.astype(float, copy=False)
    if transform_loglam:
        arr = 10.0 ** arr
    return arr


def _from_table(colnames, getcol, meta) -> Spectrum:
    """Build a Spectrum from any table-like object.

    Parameters
    ----------
    colnames : sequence of str
    getcol : callable(str) -> array
    meta : dict
    """
    wname = _find_column(colnames, WAVE_ALIASES)
    fname = _find_column(colnames, FLUX_ALIASES)
    if wname is None or fname is None:
        raise ValueError(
            f"Could not identify wavelength/flux columns among {list(colnames)}"
        )
    ename = _find_column(colnames, ERROR_ALIASES)
    dqname = _find_column(colnames, DQ_ALIASES)

    is_loglam = wname.lower() == "loglam"
    wave = _column_data(getcol(wname), transform_loglam=is_loglam)
    flux = _column_data(getcol(fname))

    error = None
    if ename is not None:
        error = _column_data(getcol(ename))
        if ename.lower() == "ivar":  # inverse variance -> sigma
            with np.errstate(divide="ignore"):
                error = np.where(error > 0, 1.0 / np.sqrt(error), 0.0)

    dq = None
    if dqname is not None:
        dq = np.asarray(getcol(dqname))

    # x1d-style files store fixed-length vectors per row; the NELEM
    # column gives the number of *valid* elements in each row.  Trim the
    # fill values beyond NELEM before concatenating rows.
    nelem_name = _find_column(colnames, NELEM_ALIASES)
    if nelem_name is not None and wave.ndim == 2:
        nelem = np.asarray(getcol(nelem_name)).astype(int).ravel()
        if nelem.size == wave.shape[0] and np.any(nelem < wave.shape[1]):
            def _trim(arr):
                if arr is None:
                    return None
                arr = np.asarray(arr)
                if arr.ndim != 2 or arr.shape[0] != nelem.size:
                    return arr
                return np.concatenate(
                    [row[:max(n, 0)] for row, n in zip(arr, nelem)])
            wave = _trim(wave)
            flux = _trim(flux)
            error = _trim(error)
            dq = _trim(dq)

    meta = dict(meta)
    meta["columns"] = {"wavelength": wname, "flux": fname,
                       "error": ename, "dq": dqname}
    return _make_spectrum(wave, flux, error, dq, meta)


# --------------------------------------------------------------------------
# FITS
# --------------------------------------------------------------------------

def _read_fits(path: str, ext: Optional[int | str] = None) -> Spectrum:
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading FITS files requires astropy: pip install astropy"
        ) from exc

    with fits.open(path, memmap=False) as hdul:
        meta = {"source_file": os.path.abspath(path)}
        primary = hdul[0].header
        for key in ("TELESCOP", "INSTRUME", "DETECTOR", "OPT_ELEM",
                    "GRATING", "TARGNAME", "ROOTNAME", "APERTURE"):
            if key in primary:
                meta[key.lower()] = primary[key]

        candidates = ([hdul[ext]] if ext is not None else list(hdul))

        # First pass: bintable HDUs.
        for hdu in candidates:
            if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue
            data = hdu.data
            if data is None or len(data) == 0:
                continue
            try:
                return _from_table(data.columns.names,
                                   lambda name: data[name], meta)
            except ValueError:
                continue  # table without recognizable columns; keep looking

        # Second pass: image HDU with linear WCS.
        for hdu in candidates:
            if hdu.data is None or not isinstance(hdu.data, np.ndarray):
                continue
            if hdu.data.ndim != 1:
                continue
            hdr = hdu.header
            crval = hdr.get("CRVAL1")
            cdelt = hdr.get("CDELT1", hdr.get("CD1_1"))
            if crval is None or cdelt is None:
                continue
            crpix = hdr.get("CRPIX1", 1.0)
            n = hdu.data.size
            pix = np.arange(n, dtype=float)
            wave = crval + (pix - (crpix - 1.0)) * cdelt
            if str(hdr.get("CTYPE1", "")).upper().startswith("LOG"):
                wave = 10.0 ** wave
            meta["columns"] = {"wavelength": "WCS", "flux": "image data",
                               "error": None, "dq": None}
            return Spectrum(wave, hdu.data.astype(float), meta=meta)

    raise ValueError(
        f"No readable spectrum found in {path}. "
        "Try specifying ext=, or check the file with astropy."
    )


# --------------------------------------------------------------------------
# ASCII
# --------------------------------------------------------------------------

def _read_ascii(path: str) -> Spectrum:
    # Try to detect a header line with column names.
    colnames = None
    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                tokens = stripped.lstrip("#").replace(",", " ").split()
                if tokens and not _is_number(tokens[0]):
                    colnames = tokens
                continue
            tokens = stripped.replace(",", " ").split()
            if tokens and not _is_number(tokens[0]):
                colnames = tokens
            break

    delimiter = "," if path.lower().endswith(".csv") else None
    data = np.genfromtxt(path, delimiter=delimiter, comments="#",
                         skip_header=1 if (colnames and not _starts_numeric(path)) else 0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"ASCII file {path} needs at least 2 columns (wave, flux)")

    meta = {"source_file": os.path.abspath(path)}
    if colnames and len(colnames) >= data.shape[1]:
        cols = {name: data[:, i] for i, name in enumerate(colnames[: data.shape[1]])}
        try:
            return _from_table(list(cols), lambda n: cols[n], meta)
        except ValueError:
            pass  # fall through to positional interpretation

    error = data[:, 2] if data.shape[1] >= 3 else None
    meta["columns"] = {"wavelength": "col 1", "flux": "col 2",
                       "error": "col 3" if error is not None else None, "dq": None}
    return _make_spectrum(data[:, 0], data[:, 1], error, None, meta)


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _starts_numeric(path: str) -> bool:
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            return _is_number(s.replace(",", " ").split()[0])
    return False


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def read_spectrum(path: str, ext: Optional[int | str] = None) -> Spectrum:
    """Read a spectrum from FITS or ASCII, auto-detecting the layout.

    Parameters
    ----------
    path : str
        Input file. ``.fits``/``.fit``/``.fts`` are treated as FITS;
        anything else is read as whitespace- or comma-separated text.
    ext : int or str, optional
        Restrict FITS reading to a single extension.

    Returns
    -------
    Spectrum
    """
    lower = path.lower()
    if lower.endswith((".fits", ".fit", ".fts", ".fits.gz")):
        return _read_fits(path, ext=ext)
    return _read_ascii(path)
