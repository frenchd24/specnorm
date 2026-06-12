"""Write normalization results.

Output columns: WAVELENGTH, FLUX, NORM_FLUX, ERROR (plus NORM_ERROR).
FITS (bintable) or ASCII depending on the output filename extension.
"""

from __future__ import annotations

import json

import numpy as np

from .spectrum import Spectrum, NormalizedSpectrum


def write_masked(spec: Spectrum, path: str) -> str:
    """Write an intermediate masked spectrum (before/independent of fitting).

    Columns: WAVELENGTH, FLUX, ERROR, MASK — unified convention:
    **0 = masked/excluded, 1 = good**.  The masked wavelength regions
    are also recorded in the header / file comments.
    """
    mask = (~spec.mask.astype(bool)).astype(np.int16)
    regions = spec.meta.get("mask_regions", [])

    if path.lower().endswith((".fits", ".fit", ".fts")):
        from astropy.io import fits
        cols = [
            fits.Column(name="WAVELENGTH", format="D", array=spec.wavelength),
            fits.Column(name="FLUX", format="D", array=spec.flux),
            fits.Column(name="ERROR", format="D", array=spec.error),
            fits.Column(name="MASK", format="I", array=mask),
        ]
        table = fits.BinTableHDU.from_columns(cols, name="MASKEDSPEC")
        primary = fits.PrimaryHDU()
        primary.header["ORIGIN"] = "specnorm"
        if "source_file" in spec.meta:
            primary.header["HISTORY"] = f"masked from {spec.meta['source_file']}"
        if "binning" in spec.meta:
            primary.header["BINNING"] = (spec.meta["binning"], "pixel binning")
        for (m0, m1) in regions:
            primary.header.add_comment(f"masked region {m0:.3f}-{m1:.3f}")
        fits.HDUList([primary, table]).writeto(path, overwrite=True)
    else:
        data = np.column_stack([spec.wavelength, spec.flux, spec.error, mask])
        header = ("specnorm masked spectrum\n"
                  + "\n".join(f"masked region {m0:.3f}-{m1:.3f}"
                              for (m0, m1) in regions)
                  + ("\n" if regions else "")
                  + "mask convention: 0 = masked/excluded, 1 = good\n"
                  + "    WAVELENGTH             FLUX            ERROR   MASK")
        np.savetxt(path, data, fmt="%14.6f %16.8e %16.8e %6d", header=header)
    return path


def write_spectrum(result: NormalizedSpectrum, path: str,
                   include_continuum: bool = True,
                   style: str = "voigt",
                   plain: bool = False) -> str:
    """Write a NormalizedSpectrum to FITS or ASCII.

    Parameters
    ----------
    result : NormalizedSpectrum
    path : str
        Output filename. ``.fits`` -> FITS bintable; anything else ->
        whitespace-separated text with a commented header.
    include_continuum : bool
        Also write the fitted continuum and normalized error columns
        (``style='full'`` only).
    style : {'voigt', 'minimal', 'full'}
        'voigt' (default): four columns following VoigtFit's input
        conventions — wavelength, normalized flux, normalized error,
        mask — where the MASK convention is *inverted* relative to
        'minimal': **1 = good pixel (include in fit), 0 = exclude**,
        matching VoigtFit / fitsutil.  Pixels are marked 0 if they are
        user-masked, have no fitted continuum (NaN), or have a
        non-positive error.  NaN and negative fluxes/errors are written
        as 0.  FITS columns are named WAVE / FLUX / ERR / MASK in the
        first table extension, as expected by VoigtFit's loader.
        'minimal': WAVELENGTH, NORM_FLUX, ERROR, MASK.

        All styles share one mask convention: **0 = masked/excluded,
        1 = good** (matching VoigtFit).
        'full': WAVELENGTH, FLUX, NORM_FLUX, ERROR
        [, NORM_ERROR, CONTINUUM].

    ASCII output uses fixed-point notation with 6 decimals
    (e.g. 1215.123456) for wavelength and for all normalized
    quantities; raw flux-unit columns in 'full' style keep exponent
    notation (STIS/COS fluxes ~1e-13 would be unreadable otherwise).
    FITS output is binary doubles, so formatting does not apply.

    plain : bool
        ASCII only: write values separated by a single space, with no
        header lines and no column padding, e.g.
        ``1200.000700 0.806969 0.023594 0``.  NaN/inf values and
        negative fluxes/errors are replaced with 0 so the file can be
        fed directly into line-fitting tools.  Ignored for FITS.

    Returns
    -------
    str : the path written.
    """
    if style not in ("voigt", "full", "minimal"):
        raise ValueError(
            f"style must be 'voigt', 'minimal' or 'full', got {style!r}")
    if path.lower().endswith((".fits", ".fit", ".fts")):
        _write_fits(result, path, include_continuum, style)
    else:
        _write_ascii(result, path, include_continuum, style, plain)
    return path


def _result_mask(result: NormalizedSpectrum) -> np.ndarray:
    """Mask column, unified convention: 1 = good pixel, 0 = masked."""
    if result.mask is not None:
        return (~result.mask.astype(bool)).astype(np.int16)
    return np.ones(result.wavelength.size, dtype=np.int16)


def _voigt_arrays(result: NormalizedSpectrum):
    """Arrays for VoigtFit-style output.

    Returns (wave, norm_flux, norm_error, mask) where mask follows the
    VoigtFit convention: 1 = good pixel (included in fit), 0 = exclude.
    Pixels are excluded if user-masked, lacking a fitted continuum, or
    carrying a non-positive error (VoigtFit weights by 1/error, so such
    pixels must be masked out rather than passed with sigma = 0).
    NaN/negative fluxes and errors are written as 0.
    """
    nf_raw = result.norm_flux
    ne_raw = result.norm_error
    good = np.isfinite(nf_raw) & np.isfinite(ne_raw) & (ne_raw > 0)
    if result.mask is not None:
        good &= ~result.mask.astype(bool)
    nf = _clean(nf_raw, no_negative=True)
    ne = _clean(ne_raw, no_negative=True)
    return result.wavelength, nf, ne, good.astype(np.int16)


def _clean(values: np.ndarray, no_negative: bool = False) -> np.ndarray:
    """Replace non-finite values (and, optionally, negatives) with 0.

    Used for ``plain`` ASCII output, which is meant to be fed directly
    into line-fitting tools that choke on NaNs and negative fluxes.
    """
    values = np.asarray(values, dtype=float)
    bad = ~np.isfinite(values)
    if no_negative:
        bad |= values < 0
    if bad.any():
        values = np.where(bad, 0.0, values)
    return values


def _write_fits(result: NormalizedSpectrum, path: str,
                include_continuum: bool, style: str = "full"):
    from astropy.io import fits

    if style == "voigt":
        # Column names VoigtFit's FITS loader recognizes, in the first
        # (and only) table extension; MASK: 1 = include, 0 = exclude.
        vw, vf, ve, vm = _voigt_arrays(result)
        cols = [
            fits.Column(name="WAVE", format="D", array=vw),
            fits.Column(name="FLUX", format="D", array=vf),
            fits.Column(name="ERR", format="D", array=ve),
            fits.Column(name="MASK", format="I", array=vm),
        ]
    elif style == "minimal":
        cols = [
            fits.Column(name="WAVELENGTH", format="D", array=result.wavelength),
            fits.Column(name="NORM_FLUX", format="D", array=result.norm_flux),
            fits.Column(name="ERROR", format="D", array=result.norm_error),
            fits.Column(name="MASK", format="I", array=_result_mask(result)),
        ]
    else:
        cols = [
            fits.Column(name="WAVELENGTH", format="D", array=result.wavelength),
            fits.Column(name="FLUX", format="D", array=result.flux),
            fits.Column(name="NORM_FLUX", format="D", array=result.norm_flux),
            fits.Column(name="ERROR", format="D", array=result.error),
        ]
        if include_continuum:
            cols.append(fits.Column(name="NORM_ERROR", format="D",
                                    array=result.norm_error))
            cols.append(fits.Column(name="CONTINUUM", format="D",
                                    array=result.continuum))

    table = fits.BinTableHDU.from_columns(
        cols, name="SPECTRUM" if style == "voigt" else "NORMSPEC")
    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "specnorm"
    if style == "voigt":
        primary.header["NORMFLUX"] = (True, "flux is continuum-normalized")
        primary.header.add_comment(
            "VoigtFit-ready: MASK 1 = include in fit, 0 = exclude")
        primary.header.add_comment(
            "use the 'norm' keyword in the VoigtFit data statement")
    for key in ("telescop", "instrume", "detector", "opt_elem", "targname",
                "rootname"):
        if key in result.meta:
            primary.header[key.upper()] = result.meta[key]
    if "source_file" in result.meta:
        primary.header["HISTORY"] = f"normalized from {result.meta['source_file']}"
    if "specnorm" in result.meta:
        sn = result.meta["specnorm"]
        if sn.get("binning", 1) > 1:
            primary.header["BINNING"] = (sn["binning"], "pixel binning")
        for (m0, m1) in sn.get("mask_regions", []):
            primary.header.add_comment(f"masked region {m0:.3f}-{m1:.3f}")
        for line in json.dumps(sn["windows"]).split(","):
            primary.header.add_comment(line[:70])

    fits.HDUList([primary, table]).writeto(path, overwrite=True)


def _write_ascii(result: NormalizedSpectrum, path: str,
                 include_continuum: bool, style: str = "full",
                 plain: bool = False):
    if style == "voigt":
        vw, vf, ve, vm = _voigt_arrays(result)
        data = np.column_stack([vw, vf, ve, vm.astype(float)])
        if plain:
            np.savetxt(path, data, fmt="%.6f %.6f %.6f %d")
            return
        # '#' comment lines are skipped by np.loadtxt, which is what
        # VoigtFit uses to read ASCII tables, so a brief header is safe.
        header = ("specnorm output, VoigtFit-ready (use the 'norm' keyword)\n"
                  "columns: wavelength  norm_flux  norm_error  "
                  "mask (1=include in fit, 0=exclude)")
        np.savetxt(path, data, fmt="%14.6f %12.6f %12.6f %12d", header=header)
        return
    if style == "minimal":
        names = ["WAVELENGTH", "NORM_FLUX", "ERROR", "MASK"]
        data = np.column_stack([result.wavelength, result.norm_flux,
                                result.norm_error,
                                _result_mask(result).astype(float)])
        if plain:
            nf = _clean(data[:, 1], no_negative=True)
            ne = _clean(data[:, 2], no_negative=True)
            out = np.column_stack([data[:, 0], nf, ne, data[:, 3]])
            np.savetxt(path, out, fmt="%.6f %.6f %.6f %d")
            return
        header = ("specnorm output (minimal; ERROR is normalized, "
                  "MASK 0 = masked, 1 = good)\n"
                  + f"{names[0]:>14s} " + " ".join(f"{n:>12s}" for n in names[1:]))
        np.savetxt(path, data, fmt="%14.6f %12.6f %12.6f %12d", header=header)
        return

    names = ["WAVELENGTH", "FLUX", "NORM_FLUX", "ERROR"]
    arrays = [result.wavelength, result.flux, result.norm_flux, result.error]
    if include_continuum:
        names += ["NORM_ERROR", "CONTINUUM"]
        arrays += [result.norm_error, result.continuum]

    data = np.column_stack(arrays)
    if plain:
        # Sanitize: NaN/inf -> 0 everywhere; negatives -> 0 in flux and
        # error columns (FLUX, NORM_FLUX, ERROR, NORM_ERROR).
        for i, name in enumerate(names):
            if i == 0:
                continue  # wavelength untouched
            no_neg = name in ("FLUX", "NORM_FLUX", "ERROR", "NORM_ERROR")
            data[:, i] = _clean(data[:, i], no_negative=no_neg)
        np.savetxt(path, data, fmt="%.6f " + " ".join("%.8e" for _ in names[1:]))
        return
    header = ("specnorm output\n"
              + f"{names[0]:>14s} " + " ".join(f"{n:>16s}" for n in names[1:]))
    # Fixed-point wavelength; flux-unit columns stay in exponent form.
    fmt = "%14.6f " + " ".join("%16.8e" for _ in names[1:])
    np.savetxt(path, data, fmt=fmt, header=header)
