"""Command-line interface: ``specnorm input.fits -o output.fits``."""

from __future__ import annotations

import argparse
import os
import sys

from .io import read_spectrum
from .spectrum import bin_spectrum
from .gui import normalize_interactive, AIRGLOW_REGIONS, load_session
from .writer import write_spectrum
from .plotting import plot_overview


def _parse_region(text: str):
    try:
        w0, w1 = (float(v) for v in text.replace(",", ":").split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"mask region must look like 1213:1218, got {text!r}")
    return (w0, w1)


def out_root_for_session(args) -> str:
    """Base path for the session file (mirrors the output naming)."""
    if args.output:
        return args.output
    return os.path.splitext(args.input)[0] + "_norm.fits"


def _write_outputs(args, result, out) -> int:
    style = args.style
    if args.minimal:
        style = "minimal"
    write_spectrum(result, out, style=style, plain=args.plain)
    if args.full:
        root_o, ext_o = os.path.splitext(out)
        write_spectrum(result, root_o + "_full" + (ext_o or ".fits"),
                       style="full", plain=args.plain)
        print(f"Full-info output: {root_o}_full{ext_o or '.fits'}")
    if style == "voigt":
        print("VoigtFit-ready output: MASK 1 = include in fit, 0 = exclude.")
        print("In your VoigtFit parameter file, remember the 'norm' keyword:")
        print(f"  data  '{out}'  <resolution>  norm")
    if not args.no_overview:
        ov_path = os.path.splitext(out)[0] + "_overview.pdf"
        plot_overview(result, ov_path, zoom=args.overview_zoom,
                      window=args.window if args.window > 0 else None,
                      overlap=args.overview_overlap)
        print(f"Overview plot: {ov_path}")
    return 0


def _run_resumed(args) -> int:
    """Continue or edit a saved session."""
    if not os.path.exists(args.resume):
        print(f"Session file not found: {args.resume}", file=sys.stderr)
        return 1
    try:
        gui, native = load_session(args.resume)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Could not read session {args.resume}: {exc}", file=sys.stderr)
        return 1
    n_acc = sum(s.accepted for s in gui.states)
    print(f"Resumed {args.resume}: {len(gui.states)} windows, "
          f"{n_acc} with accepted fits")
    gaps = gui._coverage_gaps()
    print(f"{len(gaps)} region(s) still unfitted"
          if gaps else "All fittable regions are covered")
    result = gui.run()
    if native is not None:
        result = gui.assemble_on(native)
    if not any(w["accepted"] for w in result.meta["specnorm"]["windows"]):
        print("No windows accepted — nothing written.", file=sys.stderr)
        return 1
    out = args.output
    if out is None:
        src = gui.source.get("input", "spectrum")
        out = os.path.splitext(src)[0] + "_norm.fits"
    _write_outputs(args, result, out)
    print(f"Wrote {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="specnorm",
        description="Interactive continuum normalization for STIS/COS and "
                    "generic spectra.",
    )
    parser.add_argument("input", nargs="?",
                        help="Input spectrum (FITS or ASCII); omit when "
                             "using --resume")
    parser.add_argument("--resume", metavar="SESSION.json",
                        help="Reopen a saved session to continue or edit "
                             "the fits (the spectrum and all settings are "
                             "read back from it)")
    parser.add_argument("--session", metavar="PATH",
                        help="Where to save the session (default: "
                             "<output>_session.json; 'none' to skip)")
    parser.add_argument("-o", "--output",
                        help="Output file (default: <input>_norm.fits)")
    parser.add_argument("-w", "--window", type=float, default=20.0,
                        help="Window width in wavelength units "
                             "(0 = whole spectrum at once; default 20)")
    parser.add_argument("--overlap", type=float, default=0.15,
                        help="Fractional overlap between windows (default 0.15)")
    parser.add_argument("-m", "--model", default="spline",
                        choices=["spline", "poly", "cheb"],
                        help="Initial continuum model (default spline)")
    parser.add_argument("-d", "--degree", type=int, default=3,
                        help="Initial polynomial/Chebyshev degree, 1-5 "
                             "(default 3)")
    parser.add_argument("-b", "--bin", type=int, default=2, dest="nbin",
                        help="Bin the spectrum by N pixels for FITTING "
                             "(default 2; use 1 for no binning). The output "
                             "is written at native resolution regardless, "
                             "unless --write-binned is given.")
    parser.add_argument("--write-binned", action="store_true",
                        help="Write the binned spectrum instead of "
                             "evaluating the continuum on the native-"
                             "resolution pixels")
    parser.add_argument("--mask", type=_parse_region, action="append",
                        default=[], metavar="W0:W1",
                        help="Hide a wavelength region from the continuum "
                             "FIT only; the data stay unflagged in the "
                             "output (repeatable), e.g. --mask 1213:1218.5")
    parser.add_argument("--exclude", type=_parse_region, action="append",
                        default=[], metavar="W0:W1",
                        help="Exclude a region entirely: ignored by the fit "
                             "AND flagged as masked in the output "
                             "(repeatable), for genuinely bad data")
    parser.add_argument("--airglow", action="store_true",
                        help="Exclude common geocoronal airglow (contaminated "
                             "data, so masked in the output too): Ly-alpha "
                             "1213-1218.5, OI 1301-1307, OI] 1354.5-1356.5")
    parser.add_argument("--masked-output", default=None, metavar="PATH",
                        help="Where to write the intermediate masked "
                             "spectrum (default: <input>_masked.fits; "
                             "pass 'none' to skip)")
    parser.add_argument("--low-rej", type=float, default=1.5,
                        help="Chebyshev sigma-clip threshold below the fit "
                             "(absorption; default 1.5, <=0 disables)")
    parser.add_argument("--high-rej", type=float, default=3.5,
                        help="Chebyshev sigma-clip threshold above the fit "
                             "(emission/spikes; default 3.5, <=0 disables)")
    parser.add_argument("--niterate", type=int, default=20,
                        help="Max Chebyshev reject-refit iterations (default 20)")
    parser.add_argument("--grow", type=int, default=6,
                        help="Grow each rejected pixel by N neighbours "
                             "(default 6)")
    parser.add_argument("--min-pix", type=int, default=3,
                        help="Minimum consecutive pixels below low-rej for "
                             "rejection, so isolated noise dips are kept "
                             "(default 3; 1 disables)")
    parser.add_argument("--style", default="voigt",
                        choices=["voigt", "minimal", "full"],
                        help="Output format (default: voigt, ready for "
                             "VoigtFit: wavelength, norm flux, norm error, "
                             "mask with 1=include/0=exclude)")
    parser.add_argument("--full", action="store_true",
                        help="Write BOTH the VoigtFit-style output and a "
                             "full 6-column file (<output>_full.<ext>: "
                             "WAVELENGTH, FLUX, NORM_FLUX, ERROR, "
                             "NORM_ERROR, CONTINUUM)")
    parser.add_argument("--minimal", action="store_true",
                        help="Shortcut for --style minimal (4 columns: "
                             "WAVELENGTH, NORM_FLUX, ERROR, MASK with "
                             "0=masked, 1=good)")
    parser.add_argument("--plain", action="store_true",
                        help="ASCII output: single-space separated values, "
                             "no header, NaNs and negative fluxes written "
                             "as 0 (e.g. '1200.000700 0.806969 0.023594 0')")
    parser.add_argument("--no-overview", action="store_true",
                        help="Skip the overview plot (default: an overview "
                             "PDF of data + continuum is saved alongside "
                             "the output)")
    parser.add_argument("--overview-zoom", type=float, default=3.0,
                        help="Overview panel width as a multiple of the "
                             "fitting window (default 3)")
    parser.add_argument("--overview-overlap", type=float, default=0.15,
                        help="Fractional overlap between overview panels, "
                             "so rows join up visibly (default 0.15)")
    parser.add_argument("--ext", default=None,
                        help="FITS extension to read (number or name)")
    parser.add_argument("--no-dq-mask", action="store_true",
                        help="Do not mask non-zero DQ pixels")
    args = parser.parse_args(argv)

    if args.resume:
        return _run_resumed(args)

    if not args.input:
        parser.error("an input spectrum is required unless --resume is used")

    ext = args.ext
    if ext is not None and ext.isdigit():
        ext = int(ext)

    spec = read_spectrum(args.input, ext=ext)
    info = spec.meta.get("instrume", "unknown instrument")
    print(f"Read {len(spec)} points [{spec.wmin:.1f}-{spec.wmax:.1f}] "
          f"from {args.input} ({info})")
    print("Columns used:", spec.meta.get("columns"))
    n_drop = spec.meta.get("n_unphysical_dropped", 0)
    if n_drop:
        print(f"Dropped {n_drop} fill/padding pixels with wavelength <= 0")

    spec_native = spec
    if args.nbin > 1:
        spec = bin_spectrum(spec, args.nbin)
        print(f"Binned x{args.nbin} for fitting -> {len(spec)} points")

    mask_regions = list(args.mask)
    exclude_regions = list(args.exclude)
    if args.airglow:
        exclude_regions += AIRGLOW_REGIONS

    root, _ = os.path.splitext(args.input)
    masked_out = args.masked_output
    if masked_out is None:
        masked_out = root + "_masked.fits"
    elif masked_out.lower() == "none":
        masked_out = None

    session_out = args.session
    if session_out is None:
        session_out = os.path.splitext(out_root_for_session(args))[0] + \
            "_session.json"
    elif session_out.lower() == "none":
        session_out = None

    result = normalize_interactive(
        spec, window=args.window, overlap=args.overlap,
        fitter=args.model, degree=args.degree,
        mask_dq=not args.no_dq_mask,
        mask_regions=mask_regions,
        exclude_regions=exclude_regions,
        masked_path=masked_out,
        low_rej=args.low_rej, high_rej=args.high_rej,
        niterate=args.niterate, grow=args.grow, min_pix=args.min_pix,
        output_on=(spec_native if args.nbin > 1 and not args.write_binned
                   else None),
        session_path=session_out,
        source={"input": os.path.abspath(args.input), "ext": ext,
                "bin": args.nbin},
    )
    if session_out:
        print(f"Session saved: {session_out}")
        print(f"  resume with: specnorm --resume {session_out}")
    if args.nbin > 1 and not args.write_binned:
        print(f"Output evaluated on the native grid "
              f"({len(spec_native)} px; fit used x{args.nbin} binning)")
    elif args.nbin > 1:
        print(f"Output on the x{args.nbin}-binned grid ({len(spec)} px)")
    if masked_out:
        print(f"Intermediate masked spectrum: {masked_out}")

    n_acc = sum(w["accepted"] for w in result.meta["specnorm"]["windows"])
    if n_acc == 0:
        print("No windows accepted — no normalized file written.",
              file=sys.stderr)
        return 1

    out = args.output
    if out is None:
        out = root + "_norm.fits"
    style = args.style
    if args.minimal:
        style = "minimal"
    write_spectrum(result, out, style=style, plain=args.plain)
    if args.full:
        root_o, ext_o = os.path.splitext(out)
        full_out = root_o + "_full" + (ext_o or ".fits")
        write_spectrum(result, full_out, style="full", plain=args.plain)
        print(f"Full-info output: {full_out}")
    if not args.no_overview:
        ov_path = os.path.splitext(out)[0] + "_overview.pdf"
        plot_overview(result, ov_path, zoom=args.overview_zoom,
                      window=args.window if args.window > 0 else None,
                      overlap=args.overview_overlap)
        print(f"Overview plot: {ov_path}")
    if style == "voigt":
        print("VoigtFit-ready output: MASK 1 = include in fit, 0 = exclude.")
        print("In your VoigtFit parameter file, remember the 'norm' keyword:")
        print(f"  data  '{out}'  <resolution>  norm")
    print(f"Wrote {out} ({n_acc} window(s) accepted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
