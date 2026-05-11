"""Simulated M/EEG data generation for offline testing and demos.

This module provides :func:`simulate_raw` for generating realistic EEG or MEG
recordings using the MNE-Python simulation pipeline (fsaverage source space,
dipole activation, noise model).

Functions
---------
simulate_raw
    Generate a synthetic EEG or MEG Raw object with a dipolar source.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

import mne
from mne.datasets import fetch_fsaverage
from mne.label import select_sources
from mne.simulation import (
    SourceSimulator,
    simulate_raw as _mne_simulate_raw,
    add_noise,
    add_eog,
)

_MEG_AMPLITUDE_SCALE = 1e-12   # 1 pT — typical MEG dipole projection amplitude
_EEG_AMPLITUDE_SCALE = 10e-9   # 10 nV scaling factor (matches original code)


def simulate_raw(
    brain_label: str,
    frequency: float,
    amplitude: float,
    duration: float,
    gap_duration: float,
    n_repetition: int,
    start: float,
    data_type: str = "eeg",
    sfreq: float = 256.0,
    n_eeg_channels: int = 64,
    iir_filter: list = [0.2, -0.2, 0.04],
    add_eog_artifacts: bool = True,
    fname_save: Optional[Union[str, Path]] = None,
    verbose: Union[bool, str, None] = None,
) -> mne.io.RawArray:
    """Generate a synthetic EEG or MEG recording with a sinusoidal source.

    A forward solution is created for the ``fsaverage`` template brain and a
    sinusoidal dipole is injected into the specified cortical label.  The
    signal is projected to sensor space, optionally repeated with inter-epoch
    gaps, sensor noise is added, and the result is returned (and optionally
    saved) as an :class:`mne.io.RawArray`.

    Parameters
    ----------
    brain_label : str
        Regexp matching a cortical label in the fsaverage parcellation (e.g.
        ``"bankssts-lh"`` for alpha, ``"precentral-lh"`` for motor).
    frequency : float
        Frequency of the simulated sine wave (Hz).
    amplitude : float
        Amplitude scaling factor.  Multiplied by ``10e-9`` for EEG or
        ``1e-12`` for MEG; pass ``1.0`` for a standard physiological signal.
    duration : float
        Duration of each signal epoch (seconds).
    gap_duration : float
        Silence gap between consecutive epochs (seconds).
    n_repetition : int
        Number of epochs to simulate.
    start : float
        Start time of the first epoch (seconds from recording start).
    data_type : {"eeg", "meg"}, default "eeg"
        Sensor modality to simulate.  ``"meg"`` creates a magnetometer
        (gradiometer-free) sensor layout using the ``Vectorview-all`` template.
    sfreq : float, default 256.0
        Sampling frequency (Hz).  Used only when ``data_type="eeg"`` to build
        the synthetic sensor layout.
    n_eeg_channels : int, default 64
        Number of EEG channels.  Must be one of the standard MNE montage
        channel counts (32, 64, 128, or 256 channels of ``biosemi*``/``easycap*``
        montages).  Ignored when ``data_type="meg"``.
    iir_filter : array_like, default [0.2, -0.2, 0.04]
        IIR denominator coefficients passed to :func:`mne.simulation.add_noise`.
    add_eog_artifacts : bool, default True
        If ``True``, add simulated EOG blink artefacts.
    fname_save : str | Path | None, default None
        Path to write the output ``.fif`` file.  If ``None``, the file is
        saved to ``data/simulated/<label>_<freq>Hz_<data_type>-raw.fif``
        relative to the repository root (or current working directory).
    verbose : bool | str | int | None, default None
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        The simulated raw recording.

    Raises
    ------
    ValueError
        If ``data_type`` is not ``"eeg"`` or ``"meg"``.

    Notes
    -----
    The function requires the MNE ``fsaverage`` dataset which is downloaded
    automatically on first call via :func:`mne.datasets.fetch_fsaverage`.

    For MEG, a Vectorview-all info template is used (magnetometers + planar
    gradiometers).  For EEG, a ``biosemi64`` (or ``biosemi32``/``biosemi128``
    for other channel counts) standard layout is created programmatically.

    Examples
    --------
    Simulate alpha-band EEG in the left parieto-occipital region::

        from ant.tools.simulation import simulate_raw
        raw = simulate_raw(
            brain_label="bankssts-lh",
            frequency=10.0,
            amplitude=1.0,
            duration=2.0,
            gap_duration=1.0,
            n_repetition=5,
            start=0.0,
            data_type="eeg",
        )

    Simulate beta-band MEG over left motor cortex::

        raw = simulate_raw(
            brain_label="precentral-lh",
            frequency=20.0,
            amplitude=1.0,
            duration=2.0,
            gap_duration=1.0,
            n_repetition=5,
            start=0.0,
            data_type="meg",
        )
    """
    if data_type not in ("eeg", "meg"):
        raise ValueError(f"data_type must be 'eeg' or 'meg', got {data_type!r}")

    mne.set_log_level(verbose=verbose)

    # ------------------------------------------------------------------
    # Build sensor info
    # ------------------------------------------------------------------
    raw_info = _make_sensor_info(data_type, sfreq, n_eeg_channels)

    # ------------------------------------------------------------------
    # fsaverage source space + forward solution
    # ------------------------------------------------------------------
    fs_dir = fetch_fsaverage(verbose=verbose)
    subjects_dir = os.path.dirname(fs_dir)
    subject = "fsaverage"
    trans = "fsaverage"
    src_fif = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem_fif = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    fwd = mne.make_forward_solution(
        raw_info,
        trans=trans,
        src=src_fif,
        bem=bem_fif,
        meg=(data_type == "meg"),
        eeg=(data_type == "eeg"),
        verbose=verbose,
    )
    src = fwd["src"]

    # ------------------------------------------------------------------
    # Source time series
    # ------------------------------------------------------------------
    tstep = 1.0 / raw_info["sfreq"]
    n_samples = int(duration * raw_info["sfreq"])
    t = np.arange(n_samples) * tstep

    amp_scale = _MEG_AMPLITUDE_SCALE if data_type == "meg" else _EEG_AMPLITUDE_SCALE
    source_time_series = np.sin(2.0 * np.pi * frequency * t) * amp_scale * amplitude

    # Pick a single central vertex in the target label
    selected_label = mne.read_labels_from_annot(
        subject, regexp=brain_label, subjects_dir=subjects_dir, verbose=verbose
    )[0]
    label = select_sources(
        subject,
        selected_label,
        location="center",
        extent=1,
        grow_outside=True,
        subjects_dir=subjects_dir,
    )

    # Build events: one onset per epoch, separated by gap_duration samples
    gap_samples = int(gap_duration * raw_info["sfreq"])
    start_samples = int(start * raw_info["sfreq"])
    events = np.zeros((n_repetition, 3), dtype=int)
    events[:, 0] = start_samples + gap_samples * np.arange(n_repetition)
    events[:, 2] = 1

    source_sim = SourceSimulator(src, tstep=tstep)
    source_sim.add_data(label, source_time_series, events)

    # ------------------------------------------------------------------
    # Project to sensors and add noise
    # ------------------------------------------------------------------
    raw = _mne_simulate_raw(raw_info, source_sim, forward=fwd, verbose=verbose)
    cov = mne.make_ad_hoc_cov(raw.info, verbose=verbose)
    add_noise(raw, cov, iir_filter=iir_filter, verbose=verbose)
    if add_eog_artifacts:
        add_eog(raw, verbose=verbose)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if fname_save is None:
        sim_dir = Path.cwd() / "data" / "simulated"
        sim_dir.mkdir(parents=True, exist_ok=True)
        fname_save = sim_dir / f"{brain_label}_{frequency}Hz_{data_type}-raw.fif"
    raw.save(fname=Path(fname_save), overwrite=True)

    return raw


# ---------------------------------------------------------------------------
# Backwards-compatibility alias
# ---------------------------------------------------------------------------

def simulate_eeg_raw(
    brain_label: str,
    frequency: float,
    amplitude: float,
    duration: float,
    gap_duration: float,
    n_repetition: int,
    start: float,
    iir_filter: list = [0.2, -0.2, 0.04],
    fname_save=None,
    verbose=None,
) -> mne.io.RawArray:
    """Backwards-compatible wrapper — use :func:`simulate_raw` instead.

    Parameters
    ----------
    brain_label, frequency, amplitude, duration, gap_duration, n_repetition,
    start, iir_filter, fname_save, verbose
        See :func:`simulate_raw`.

    Returns
    -------
    raw : mne.io.Raw
    """
    return simulate_raw(
        brain_label=brain_label,
        frequency=frequency,
        amplitude=amplitude,
        duration=duration,
        gap_duration=gap_duration,
        n_repetition=n_repetition,
        start=start,
        data_type="eeg",
        iir_filter=iir_filter,
        fname_save=fname_save,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_sensor_info(data_type: str, sfreq: float, n_eeg_channels: int) -> mne.Info:
    """Build an :class:`mne.Info` object for the requested modality.

    For EEG a standard montage is used; for MEG the MNE sample dataset
    Vectorview info is used as template (magnetometers + gradiometers).
    """
    if data_type == "meg":
        try:
            sample_dir = mne.datasets.sample.data_path(verbose=False)
        except Exception as exc:
            raise RuntimeError(
                "MEG simulation requires the MNE sample dataset. "
                "Install it with: mne.datasets.sample.data_path()"
            ) from exc
        raw_fname = os.path.join(sample_dir, "MEG", "sample", "sample_audvis_raw.fif")
        info = mne.io.read_info(raw_fname, verbose=False)
        # Keep only MEG channels; resample info sfreq if different
        meg_picks = mne.pick_types(info, meg=True, eeg=False, stim=False, exclude="bads")
        info = mne.pick_info(info, sel=meg_picks)
        info["sfreq"] = float(sfreq)
        return info

    # EEG — use a standard biosemi/easycap montage
    montage_map = {
        32:  "easycap-M10",
        64:  "biosemi64",
        128: "biosemi128",
        256: "biosemi256",
    }
    montage_name = montage_map.get(n_eeg_channels, "biosemi64")
    montage = mne.channels.make_standard_montage(montage_name)
    info = mne.create_info(
        ch_names=montage.ch_names,
        sfreq=float(sfreq),
        ch_types="eeg",
    )
    info.set_montage(montage)
    return info
