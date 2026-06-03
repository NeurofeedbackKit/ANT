"""Command-line interface for MNE-RT.

Usage
-----
::

    mne-rt --help
    mne-rt --version
    mne-rt info
    mne-rt demo     [options]
    mne-rt baseline [options]
    mne-rt run      [options]

Install note
------------
After ``pip install -e .`` the entry-point in ``pyproject.toml`` exposes
the ``mne-rt`` shell command::

    [project.scripts]
    mne-rt = "mne_rt.cli:main"
"""
from __future__ import annotations

import argparse
import sys
import textwrap


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mne-rt",
        description=textwrap.dedent("""\
            MNE-RT
            ─────────────────────────────────────
            Real-time M/EEG signal processing and analysis.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use 'mne-rt <command> --help' for per-command options.",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print the installed MNE-RT version and exit.",
    )
    parser.add_argument(
        "--verbose", "-v",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging verbosity level (default: WARNING).",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    _add_info_parser(subparsers)
    _add_demo_parser(subparsers)
    _add_baseline_parser(subparsers)
    _add_run_parser(subparsers)

    return parser


# ---------------------------------------------------------------------------
# Sub-command parsers
# ---------------------------------------------------------------------------

def _add_info_parser(sub):
    p = sub.add_parser(
        "info",
        help="Display system and dependency information.",
        description="Print MNE-RT version, Python version, and key dependency versions.",
    )
    return p


def _add_demo_parser(sub):
    p = sub.add_parser(
        "demo",
        help="Launch a demo real-time session from simulated EEG data.",
        description=textwrap.dedent("""\
            Run a full demo real-time session using simulated EEG.
            No amplifier or recording file is required.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--duration", type=float, default=120.0,
        help="Session duration in seconds (default: 120).",
    )
    p.add_argument(
        "--modality", nargs="+",
        default=["sensor_power", "band_ratio", "entropy", "hjorth"],
        metavar="MODALITY",
        help=(
            "Feature modality(ies) to demonstrate.  "
            "Available: sensor_power, band_ratio, entropy, hjorth, "
            "sensor_connectivity, erd_ers, laterality, spectral_centroid, "
            "cfc_sensor, scp, peak_alpha_freq, connectivity_ratio.  "
            "(default: sensor_power band_ratio entropy hjorth)"
        ),
    )
    p.add_argument(
        "--winsize", type=float, default=1.0,
        help="Analysis window length in seconds (default: 1.0).",
    )
    p.add_argument(
        "--no-signal", action="store_true",
        help="Disable the scrolling real-time signal plot.",
    )
    p.add_argument(
        "--no-raw", action="store_true",
        help="Disable the raw stream viewer.",
    )
    p.add_argument(
        "--no-topo", action="store_true",
        help="Disable the real-time scalp topomap display.",
    )
    p.add_argument(
        "--no-brain", action="store_true",
        help="Disable the 3-D brain activation display even if FreeSurfer is found.",
    )
    # ERP / epoch plot flags
    p.add_argument(
        "--erp", action="store_true",
        help="Enable the scalp-layout ERP plot (requires a stimulus channel).",
    )
    p.add_argument(
        "--butterfly", action="store_true",
        help="Enable the butterfly overlay plot (all channels, region-coloured).",
    )
    p.add_argument(
        "--compare-evoked", action="store_true",
        help="Enable the per-channel comparison plot with SEM ribbons and peak markers.",
    )
    p.add_argument(
        "--tfr", action="store_true",
        help="Enable the Morlet wavelet TFR heatmap plot.",
    )
    p.add_argument(
        "--subjects-fs-dir", metavar="DIR",
        help=(
            "FreeSurfer subjects directory (must contain fsaverage5).  "
            "Auto-detected from FREESURFER_HOME/subjects if not given."
        ),
    )
    p.add_argument(
        "--surf",
        choices=["inflated", "pial", "white", "sphere"],
        default="inflated",
        help="Cortical surface geometry for brain display (default: inflated).",
    )
    p.add_argument(
        "--smoothing", type=float, default=0.25, metavar="ALPHA",
        help=(
            "EMA smoothing factor for feature values (default: 0.25). "
            "1.0 = no smoothing; 0.1 = heavy smoothing."
        ),
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Skip saving session data and report at the end of the demo.",
    )
    return p


def _add_baseline_parser(sub):
    p = sub.add_parser(
        "baseline",
        help="Record a resting-state baseline session.",
        description=textwrap.dedent("""\
            Connect to a live LSL stream (or simulate one) and record a
            baseline segment.  The inverse operator is computed and saved.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_session_args(p)
    p.add_argument(
        "--duration", type=float, default=120.0,
        help="Baseline duration in seconds (default: 120).",
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Use simulated data instead of a live LSL stream.",
    )
    p.add_argument(
        "--fname", metavar="FILE",
        help="Any MNE-readable file to simulate (.fif, .vhdr, .edf, .bdf, .set, …) — requires --mock.",
    )
    return p


def _add_run_parser(sub):
    p = sub.add_parser(
        "run",
        help="Run a real-time M/EEG main session.",
        description=textwrap.dedent("""\
            Connect to an LSL stream, extract real-time features,
            and drive all configured visualisation windows.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_session_args(p)
    p.add_argument(
        "--duration", type=float, required=True,
        help="Session duration in seconds.",
    )
    p.add_argument(
        "--modality", nargs="+",
        default=["sensor_power"],
        metavar="MODALITY",
        help="Feature modality(ies) to extract (default: sensor_power).",
    )
    p.add_argument(
        "--winsize", type=float, default=1.0,
        help="Analysis window length in seconds (default: 1.0).",
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Use simulated data instead of a live LSL stream.",
    )
    p.add_argument(
        "--fname", metavar="FILE",
        help="Any MNE-readable file to simulate (.fif, .vhdr, .edf, .bdf, .set, …) — requires --mock.",
    )
    p.add_argument(
        "--artifact-correction",
        choices=["lms", "orica", "gedai", "asr", "maxwell"],
        default=None,
        help=(
            "Real-time artifact correction method (default: none). "
            "lms/orica/gedai/asr: EEG/MEG. maxwell: MEG only (SSS/tSSS)."
        ),
    )
    p.add_argument(
        "--no-signal", action="store_true",
        help="Disable the scrolling real-time signal plot.",
    )
    p.add_argument(
        "--no-raw", action="store_true",
        help="Disable the raw stream viewer.",
    )
    p.add_argument(
        "--topo", action="store_true",
        help="Show the real-time scalp topomap display.",
    )
    p.add_argument(
        "--brain", action="store_true",
        help="Show the 3-D brain activation display.",
    )
    # ERP / epoch plot flags
    p.add_argument(
        "--erp", action="store_true",
        help="Enable the scalp-layout ERP plot (requires a stimulus channel).",
    )
    p.add_argument(
        "--butterfly", action="store_true",
        help="Enable the butterfly overlay plot (all channels, region-coloured).",
    )
    p.add_argument(
        "--compare-evoked", action="store_true",
        help="Enable the per-channel comparison plot with SEM ribbons and peak markers.",
    )
    p.add_argument(
        "--tfr", action="store_true",
        help="Enable the Morlet wavelet TFR heatmap plot.",
    )
    p.add_argument(
        "--surf",
        choices=["inflated", "pial", "white", "sphere"],
        default="inflated",
        help="Brain surface geometry (default: inflated).",
    )
    p.add_argument(
        "--osc-host", metavar="HOST", default=None,
        help="Enable OSC output and send to this host (e.g. 127.0.0.1).",
    )
    p.add_argument(
        "--osc-port", type=int, default=9000, metavar="PORT",
        help="OSC destination port (default: 9000).",
    )
    p.add_argument(
        "--osc-prefix", metavar="PREFIX", default="/mne_rt",
        help="OSC address prefix (default: /mne_rt).",
    )
    p.add_argument(
        "--lsl-output", action="store_true",
        help=(
            "Broadcast feature values as an LSL stream outlet named 'MNE_RT'.  "
            "Any LSL-aware application (PsychoPy, Psychtoolbox, OpenViBE, …) "
            "can subscribe to this stream.  Faster and more reliable than OSC "
            "for same-machine integration."
        ),
    )
    p.add_argument(
        "--lsl-stream-name", metavar="NAME", default="MNE_RT",
        help="LSL outlet stream name (default: MNE_RT).  Only used with --lsl-output.",
    )
    p.add_argument(
        "--smoothing", type=float, default=0.25, metavar="ALPHA",
        help=(
            "EMA smoothing factor for feature values (default: 0.25). "
            "1.0 = no smoothing; 0.1 = heavy smoothing."
        ),
    )
    return p


def _add_common_session_args(p: argparse.ArgumentParser) -> None:
    """Add subject/session args shared by baseline and run sub-commands."""
    p.add_argument(
        "--subject", required=True, metavar="ID",
        help="Subject identifier (BIDS subject label, e.g. 'sub01').",
    )
    p.add_argument(
        "--session", default="01", metavar="LABEL",
        help="BIDS session label (e.g. '01', 'pre', 'week1'; default: '01').",
    )
    p.add_argument(
        "--subjects-dir", required=True, metavar="DIR",
        help="Root directory containing one folder per subject.",
    )
    p.add_argument(
        "--montage", default="easycap-M1", metavar="NAME",
        help="EEG montage name or .bvct file path (default: easycap-M1).",
    )
    p.add_argument(
        "--data-type", choices=["eeg", "meg"], default="eeg",
        help="Recording modality (default: eeg).",
    )
    p.add_argument(
        "--subjects-fs-dir", metavar="DIR",
        help="FreeSurfer subjects directory (required for source modalities).",
    )


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_info(args) -> None:
    """Print version and dependency information."""
    import platform

    lines = [
        "MNE-RT — Real-Time M/EEG Analysis",
        "─" * 40,
    ]
    try:
        from mne_rt import __version__
        lines.append(f"  mne-rt version : {__version__}")
    except Exception:
        lines.append("  mne-rt version : unknown")

    lines.append(f"  Python         : {platform.python_version()}")

    for pkg in ["mne", "numpy", "scipy", "pyvista", "PyQt6", "pyqtgraph",
                "mne_lsl", "mne_connectivity", "mne_features"]:
        try:
            from importlib.metadata import version
            lines.append(f"  {pkg:<22}: {version(pkg)}")
        except Exception:
            lines.append(f"  {pkg:<22}: not installed")

    print("\n".join(lines))


def _cmd_demo(args) -> None:
    """Run a demo real-time session from simulated EEG."""
    from mne_rt import RTStream, set_log_level

    set_log_level(args.verbose)

    print("MNE-RT Demo — simulating EEG …")
    import os
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="mne_rt_demo_"))
    subjects_dir = str(tmp)

    # Use the bundled pericalcarine simulation (loops automatically via n_repeat=inf)
    _pkg_root = Path(__file__).parent.parent.parent  # repo root
    fname_sim = _pkg_root / "data" / "simulated" / "pericalcarine-lh_10Hz_1-raw.fif"
    if not fname_sim.is_file():
        raise FileNotFoundError(
            f"Demo simulation file not found: {fname_sim}\n"
            "Re-run from the MNE-RT repository root or reinstall the package."
        )

    # Resolve FreeSurfer subjects directory: explicit arg → env vars → known paths
    subjects_fs_dir = getattr(args, "subjects_fs_dir", None)
    if subjects_fs_dir is None:
        _candidates = []
        if os.environ.get("FREESURFER_HOME"):
            _candidates.append(Path(os.environ["FREESURFER_HOME"]) / "subjects")
        if os.environ.get("SUBJECTS_DIR"):
            _candidates.append(Path(os.environ["SUBJECTS_DIR"]))
        _candidates += [
            Path("/Applications/freesurfer/dev/subjects"),
            Path("/usr/local/freesurfer/subjects"),
        ]
        for _d in _candidates:
            if _d.is_dir() and (_d / "fsaverage5").is_dir():
                subjects_fs_dir = str(_d)
                break

    show_brain = (subjects_fs_dir is not None) and not getattr(args, "no_brain", True)
    show_topo = not getattr(args, "no_topo", False)
    do_save = not getattr(args, "no_save", True)
    if show_brain:
        print(f"Brain activation: using {subjects_fs_dir}")

    nf = RTStream(
        subject_id="demo",
        session="01",
        subjects_dir=subjects_dir,
        montage="easycap-M1",
        data_type="eeg",
        subjects_fs_dir=subjects_fs_dir if show_brain else None,
        verbose=args.verbose,
    )
    nf.connect_to_lsl(mock_lsl=True, fname=str(fname_sim), verbose=args.verbose)

    # Always record a brief baseline so the report and ERD/ERS work
    print("Recording brief baseline (10 s) …")
    nf.record_baseline(baseline_duration=10, verbose=args.verbose)

    nf.record_main(
        duration=args.duration,
        modality=args.modality,
        winsize=args.winsize,
        signal_smoothing=args.smoothing,
        show_nf_signal=not args.no_signal,
        show_raw_signal=not args.no_raw,
        show_topo=show_topo,
        show_brain_activation=show_brain,
        brain_surf=getattr(args, "surf", "pial"),
        save_raw=do_save,
        verbose=args.verbose,
    )

    if do_save:
        saved = nf.save()
        for kind, path in saved.items():
            print(f"  [{kind}] → {path}")
        try:
            report_path = nf.create_report()
            print(f"  [report] → {report_path}")
        except Exception as exc:
            print(f"  [report] skipped ({exc})")


def _cmd_baseline(args) -> None:
    """Record a baseline session."""
    from mne_rt import RTStream, set_log_level
    set_log_level(args.verbose)

    nf = RTStream(
        subject_id=args.subject,
        session=args.session,
        subjects_dir=args.subjects_dir,
        montage=args.montage,
        data_type=args.data_type,
        subjects_fs_dir=args.subjects_fs_dir,
        verbose=args.verbose,
    )
    nf.connect_to_lsl(
        mock_lsl=args.mock,
        fname=getattr(args, "fname", None),
        verbose=args.verbose,
    )
    nf.record_baseline(baseline_duration=args.duration, verbose=args.verbose)
    print(f"Baseline complete.  Data saved to: {nf.subject_dir}")


def _cmd_run(args) -> None:
    """Run a real-time M/EEG session."""
    from mne_rt import RTStream, set_log_level
    set_log_level(args.verbose)

    artifact_correction = args.artifact_correction or False

    nf = RTStream(
        subject_id=args.subject,
        session=args.session,
        subjects_dir=args.subjects_dir,
        montage=args.montage,
        data_type=args.data_type,
        subjects_fs_dir=getattr(args, "subjects_fs_dir", None),
        artifact_correction=artifact_correction,
        verbose=args.verbose,
    )
    nf.connect_to_lsl(
        mock_lsl=args.mock,
        fname=getattr(args, "fname", None),
        verbose=args.verbose,
    )

    if artifact_correction == "gedai":
        print("Fitting GEDAI denoiser from baseline …")
        nf.record_baseline(baseline_duration=60, verbose=args.verbose)
        nf.fit_gedai()
    elif artifact_correction == "asr":
        print("Fitting ASR denoiser from baseline …")
        nf.record_baseline(baseline_duration=60, verbose=args.verbose)
        nf.fit_asr()
    elif artifact_correction == "maxwell":
        print("Computing SSS/tSSS Maxwell filter from sensor geometry …")
        nf.fit_maxwell()

    osc_sender = None
    if getattr(args, "osc_host", None):
        from mne_rt.osc import OSCSender
        osc_sender = OSCSender(
            host=args.osc_host,
            port=args.osc_port,
            prefix=args.osc_prefix,
        )
        print(f"OSC output → {osc_sender.target}  prefix={osc_sender.prefix}")

    lsl_sender = None
    if getattr(args, "lsl_output", False):
        from mne_rt.lsl_output import LSLSender
        lsl_sender = LSLSender(stream_name=getattr(args, "lsl_stream_name", "MNE_RT"))
        print(f"LSL output → stream '{lsl_sender.stream_name}'")

    try:
        nf.record_main(
            duration=args.duration,
            modality=args.modality,
            winsize=args.winsize,
            signal_smoothing=args.smoothing,
            show_nf_signal=not args.no_signal,
            show_raw_signal=not args.no_raw,
            show_topo=args.topo,
            show_brain_activation=args.brain,
            osc_sender=osc_sender,
            lsl_sender=lsl_sender,
            verbose=args.verbose,
        )
    finally:
        if osc_sender is not None:
            osc_sender.close()
        if lsl_sender is not None:
            lsl_sender.close()

    print(f"Session complete.  Data saved to: {nf.subject_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str | None
        Command-line arguments.  ``None`` uses ``sys.argv[1:]``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        try:
            from mne_rt import __version__
            print(f"mne-rt {__version__}")
        except Exception:
            print("mne-rt (version unknown)")
        return

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "info":     _cmd_info,
        "demo":     _cmd_demo,
        "baseline": _cmd_baseline,
        "run":      _cmd_run,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
