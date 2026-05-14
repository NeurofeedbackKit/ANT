"""
Alpha laterality neurofeedback with adaptive protocol
======================================================

Frontal alpha asymmetry is a well-established biomarker in clinical
neuroscience, particularly in depression and attention research.
This example demonstrates:

1. Simulate EEG with right-hemisphere alpha enhancement.
2. Extract the **laterality** modality (log-ratio of hemispheric power).
3. Couple the NF signal to a :class:`~ant.protocols.ZScoreProtocol` that
   normalises against a rolling baseline and rewards right-lateralised alpha.
4. Plot the laterality index alongside the adaptive reward magnitude and the
   z-score threshold evolution.

The ``laterality`` modality computes:

.. math::

   L = \\log\\!\\left(\\frac{P_\\mathrm{right}}{P_\\mathrm{left}}\\right)

where :math:`P` is mean alpha power per hemisphere.  Positive values indicate
right-dominant alpha; the protocol rewards the participant when the z-scored
laterality exceeds 0.5 standard deviations.
"""

# %%
# Simulate EEG with right-lateralised alpha
# ------------------------------------------
# We use :func:`~ant.tools.simulate_raw` to synthesise a recording where the
# alpha source is placed in the right occipital region, producing an asymmetric
# spatial distribution across the scalp.

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ant import NFRealtime
from ant.protocols import ZScoreProtocol
from ant.tools import simulate_raw

tmp = Path(tempfile.mkdtemp(prefix="ant_laterality_"))
fname_sim = tmp / "right_alpha.fif"

simulate_raw(
    brain_label="lateraloccipital-rh",   # right-lateralised source
    frequency=10.0,
    amplitude=1.5e-6,
    duration=4.0,
    gap_duration=2.0,
    n_repetition=6,
    start=1.0,
    data_type="eeg",
    sfreq=256.0,
    fname_save=str(fname_sim),
    verbose=False,
)

# %%
# Session setup and baseline
# --------------------------
# We record a 15-second baseline to initialise the ERD/ERS normaliser and
# seed the :class:`~ant.protocols.ZScoreProtocol`'s running statistics.

subjects_dir = tmp / "subjects"
subjects_dir.mkdir()

nf = NFRealtime(
    subject_id="sub01",
    visit=1,
    session="main",
    subjects_dir=str(subjects_dir),
    montage="biosemi64",
    data_type="eeg",
    verbose=False,
)
nf.connect_to_lsl(mock_lsl=True, fname=str(fname_sim), verbose=False)
nf.record_baseline(baseline_duration=15, verbose=False)

# %%
# Run the laterality NF session
# ------------------------------
# Extract the ``laterality`` modality for 45 seconds.

nf.record_main(
    duration=45,
    modality=["laterality"],
    winsize=1.0,
    show_nf_signal=False,
    show_raw_signal=False,
    show_topo=False,
    verbose=False,
)
nf.save()

# %%
# Apply the ZScore protocol offline
# -----------------------------------
# :class:`~ant.protocols.ZScoreProtocol` normalises the stream of laterality
# values against the first ``warmup_windows`` samples and then rewards windows
# whose z-score exceeds ``zscore_threshold``.

lat_vals = np.asarray(nf.nf_data["laterality"])

proto = ZScoreProtocol(
    direction="up",
    warmup_windows=10,
    zscore_threshold=0.5,
    smoothing=0.1,
)

rewards: list[float] = []
thresholds: list[float] = []
zscores: list[float] = []

for v in lat_vals:
    crossed, magnitude = proto.evaluate(v)
    rewards.append(magnitude if crossed else 0.0)
    thresholds.append(proto.zscore_threshold)
    zscores.append(proto.zscore if proto.zscore is not None else 0.0)

print(
    f"n_evaluated: {proto.n_evaluated}  |  "
    f"μ={proto.mean_:.4f}  σ={proto.std_:.4f}  |  "
    f"current z={proto.zscore:.2f}"
)

# %%
# Visualise the laterality NF signal and reward
# -----------------------------------------------
# The top panel shows the raw laterality index; the bottom panel shows the
# z-scored version with reward events highlighted as vertical bars.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
fig.suptitle(
    "Frontal Alpha Laterality NF  ·  ZScoreProtocol (warmup=10, threshold=0.5 SD)",
    fontsize=12, fontweight="bold",
)

t = np.arange(len(lat_vals))

# ── Raw laterality ───────────────────────────────────────────────────────────
ax1.plot(t, lat_vals, color="#5DA5A4", lw=1.8, label="Laterality index")
ax1.axhline(0, ls=":", lw=0.8, color="#888")
ax1.fill_between(t, lat_vals, 0, where=lat_vals > 0, alpha=0.18, color="#5DA5A4",
                  label="Right-dominant")
ax1.fill_between(t, lat_vals, 0, where=lat_vals < 0, alpha=0.12, color="#FF6B6B",
                  label="Left-dominant")
ax1.set_ylabel("Laterality  (log R/L)", fontsize=9)
ax1.legend(fontsize=8, frameon=False, loc="upper right")
ax1.spines[["top", "right"]].set_visible(False)

# ── Z-scored laterality + rewards ────────────────────────────────────────────
ax2.plot(t, zscores, color="#4D96FF", lw=1.8, label="z-score")
ax2.axhline(proto.zscore_threshold, ls="--", lw=1.2, color="#FFD93D",
             label=f"Threshold (+{proto.zscore_threshold} SD)")
ax2.axhline(0, ls=":", lw=0.8, color="#888")

reward_arr = np.asarray(rewards)
ax2.vlines(t[reward_arr > 0], ymin=-0.5, ymax=reward_arr[reward_arr > 0],
           color="#6BCB77", alpha=0.6, lw=2.5, label="Reward")

ax2.set_ylabel("Z-score  (a.u.)", fontsize=9)
ax2.set_xlabel("Analysis window (1 s each)", fontsize=9)
ax2.legend(fontsize=8, frameon=False, loc="upper right")
ax2.spines[["top", "right"]].set_visible(False)

plt.show()
