"""
specnorm
========

Interactive continuum normalization for astronomical spectra, tailored to
HST/STIS and HST/COS data but flexible enough for generic FITS / ASCII spectra.

Typical usage
-------------
Command line::

    specnorm my_spectrum_x1d.fits -o my_spectrum_norm.fits

Python::

    from specnorm import read_spectrum, normalize_interactive, write_spectrum

    spec = read_spectrum("oc8c11020_x1d.fits")
    result = normalize_interactive(spec, window=50.0, fitter="spline")
    write_spectrum(result, "normalized.fits")
"""

from .spectrum import Spectrum, NormalizedSpectrum, bin_spectrum
from .io import read_spectrum
from .writer import write_spectrum, write_masked
from .fitters import SplineFitter, PolynomialFitter, ChebyshevFitter, make_fitter
from .gui import ContinuumGUI, normalize_interactive, AIRGLOW_REGIONS
from .plotting import plot_overview

__version__ = "0.9.0"

__all__ = [
    "Spectrum",
    "NormalizedSpectrum",
    "read_spectrum",
    "write_spectrum",
    "write_masked",
    "bin_spectrum",
    "AIRGLOW_REGIONS",
    "plot_overview",
    "SplineFitter",
    "PolynomialFitter",
    "ChebyshevFitter",
    "make_fitter",
    "ContinuumGUI",
    "normalize_interactive",
]
