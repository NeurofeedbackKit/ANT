"""
Complete closed-loop NF session
================================

Full end-to-end pipeline with ANT:

1. Simulate a 64-channel EEG recording with alpha rhythm modulation.
2. Stream it through a mock LSL player.
3. Record a resting-state baseline and compute the noise covariance.
4. Run a closed-loop NF session extracting multiple modalities in parallel.
5. Save the NF data and create an HTML report.
6. Inspect the feature time-series.

The three interactive windows — :class:`~ant.viz.NFSignalPlot` (NF signal),
:class:`~ant.viz.TopoPlot` (scalp topomap), and
:class:`~ant.viz.BrainPlot` (3D brain) — open automatically during
``record_main`` when ``show_nf_signal=True``, ``show_topo=True``, and
``show_brain_activation=True`` respectively.  This example runs headlessly
for documentation purposes.

.. admonition:: Interactive session

    To open all three live windows, change the ``record_main`` call to::

        nf.record_main(
            duration=300,
            modality=["sensor_power", "erd_ers", "hjorth", "laterality"],
            show_nf_signal=True,
            show_topo=True,
            show_brain_activation=True,     # requires subjects_fs_dir
        )

    Videos of the three windows recorded during a real session are shown
    below.

.. raw:: html

    <div style="display:flex; gap:12px; justify-content:center; margin:20px 0;">
      <div style="text-align:center;">
        <video width="320" autoplay muted loop
               style="border-radius:8px; border:1px solid rgba(200,200,200,0.3);">
          <source src="../_static/nf_demo.mov" type="video/quicktime">
        </video>
        <p style="font-size:11px; color:#888; margin-top:4px;">NFSignalPlot</p>
      </div>
      <div style="text-align:center;">
        <video width="320" autoplay muted loop
               style="border-radius:8px; border:1px solid rgba(200,200,200,0.3);">
          <source src="../_static/brain.mov" type="video/quicktime">
        </video>
        <p style="font-size:11px; color:#888; margin-top:4px;">BrainPlot</p>
      </div>
    </div>
"""

# %%
# Simulate a recording
# --------------------
# :func:`~ant.tools.simulate_raw` generates a synthetic 64-channel biosemi64
# EEG recording with a configurable alpha burst pattern.  The simulated signal
# has a 10 Hz alpha component whose amplitude is modulated every few seconds,
# mimicking the type of EEG a participant might produce during an alpha
# neurofeedback session.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ant import NFRealtime
from ant.tools import simulate_raw

# Results land in ~/ANT_session_results — inspect subject dir and HTML report there
tmp = Path.home() / "ANT_session_results"
tmp.mkdir(parents=True, exist_ok=True)

# 5-minute simulation: 10 Hz alpha bursts, biosemi64 layout
fname_sim = tmp / "simulated_eeg.fif"
simulate_raw(
    brain_label="lateraloccipital-lh",
    frequency=10.0,
    amplitude=1e-6,
    duration=5.0,
    gap_duration=3.0,
    n_repetition=4,
    start=2.0,
    data_type="eeg",
    sfreq=256.0,
    fname_save=str(fname_sim),
    verbose=False,
)
print(f"Simulated EEG saved to: {fname_sim}")

# %%
# Set up the NF session
# ---------------------
# :class:`~ant.NFRealtime` holds all session state: subject metadata, LSL
# stream handle, inverse operator, and recorded NF data.

subjects_dir = tmp / "subjects"
subjects_dir.mkdir(exist_ok=True)

nf = NFRealtime(
    subject_id="sub01",
    visit=1,
    session="main",
    subjects_dir=str(subjects_dir),
    montage="biosemi64",
    data_type="eeg",
    verbose=False,
)

# %%
# Connect to the mock LSL stream
# --------------------------------
# ``mock_lsl=True`` starts an :class:`mne_lsl.lsl.PlayerLSL` that replays the
# simulated FIF file as a real-time LSL stream at its original sampling rate.
# Replace ``mock_lsl=False`` and remove ``fname`` to connect to a live amplifier.

nf.connect_to_lsl(mock_lsl=True, fname=str(fname_sim), verbose=False)

# %%
# Record a resting-state baseline
# ---------------------------------
# The 10-second baseline is used to:
#
# * Estimate channel-wise power for ERD/ERS normalisation.
# * Fit the forward model and inverse operator (required for source modalities).
# * Compute a blink template for artefact detection.
#
# For this headless example we use only sensor-space modalities so the inverse
# operator step is skipped (``compute_inv=False``).

nf.record_baseline(baseline_duration=10, verbose=False)

# %%
# Run the closed-loop NF loop
# ----------------------------
# Four modalities are computed in parallel on each 1-second window.
# All display windows are disabled here; see the note at the top of the
# example for how to enable them interactively.

nf.record_main(
    duration=30,
    modality=["sensor_power", "erd_ers", "hjorth", "spectral_centroid"],
    winsize=1.0,
    show_nf_signal=False,
    show_raw_signal=False,
    show_topo=False,
    save_raw=True,
    verbose=False,
)

# %%
# Save data and generate the HTML report
# ----------------------------------------
# :meth:`~ant.NFRealtime.save` writes the NF feature time-series as a JSON
# file and disconnects the stream.  :meth:`~ant.NFRealtime.create_report`
# builds a self-contained MNE HTML report with baseline recording info,
# PSD, and the NF feature plots.

saved = nf.save()
for kind, path in saved.items():
    print(f"  [{kind}] → {path}")

report_path = nf.create_report(open_browser=False)
print(f"  [report] → {report_path}")

# %%
# Inspect the NF feature time-series
# -------------------------------------
# After the session, ``nf.nf_data`` is a dict mapping modality name to a list
# of scalar values — one per analysis window.  We plot all four modalities.

fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
fig.suptitle(
    "ANT — Recorded NF features  (simulated 10 Hz alpha session)",
    fontweight="bold", fontsize=13,
)

palette = ["#5DA5A4", "#FF6B6B", "#FFD93D", "#6BCB77"]
labels = {
    "sensor_power":      "Sensor power  (V²/Hz)",
    "erd_ers":           "ERD/ERS  (%)",
    "hjorth":            "Hjorth complexity  (a.u.)",
    "spectral_centroid": "Spectral centroid  (Hz)",
}

for ax, (mod, vals), color in zip(axes.flat, nf.nf_data.items(), palette):
    t = np.arange(len(vals))
    ax.plot(t, vals, color=color, lw=1.8)
    ax.fill_between(t, vals, alpha=0.12, color=color)
    ax.axhline(np.mean(vals), ls="--", lw=0.9, color="#888", label="mean")
    ax.set_title(labels.get(mod, mod), fontsize=10, fontweight="bold")
    ax.set_xlabel("Window index (1 s each)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

plt.show()
