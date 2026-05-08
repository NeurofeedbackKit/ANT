"""
Real-time neurofeedback session
================================

This example demonstrates a complete closed-loop EEG neurofeedback session
using ANT.  We use a mock LSL stream backed by the bundled sample recording
so that no amplifier is required.

The session follows the standard ANT workflow:

1. Create an :class:`~ant.NFRealtime` session object.
2. Connect to a simulated LSL stream (``mock_lsl=True``).
3. Run the main neurofeedback loop with two sensor-space modalities.
4. Inspect the recorded NF feature time-series.
"""

# %%
# Setup
# -----
# We create a temporary directory for subject data and instantiate the session
# object.  ``session="main"`` allows us to skip a formal baseline when using
# sensor-space modalities only.

import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ant import NFRealtime

subjects_dir = tempfile.mkdtemp(prefix="ant_example_")

nf = NFRealtime(
    subject_id="demo",
    visit=1,
    session="main",
    subjects_dir=subjects_dir,
    montage="easycap-M1",
    data_type="eeg",
    verbose=False,
)

# %%
# Connect to mock stream
# ----------------------
# ``mock_lsl=True`` starts an :class:`mne_lsl.lsl.StreamPlayer` replaying the
# bundled sample recording.  ``fname=None`` picks the default sample file
# automatically.

nf.connect_to_lsl(mock_lsl=True, timeout=30.0, verbose=False)

# %%
# Run the NF session
# ------------------
# We extract two modalities in parallel:
#
# * ``sensor_power`` — mean band power in the alpha range (8–12 Hz)
# * ``band_ratio`` — theta / beta ratio (4–8 Hz vs 12–30 Hz)
#
# All display windows are disabled so the session runs headlessly.  The
# ``winsize=1.0`` parameter sets a 1-second analysis window.

nf.record_main(
    duration=15,
    modality=["sensor_power", "band_ratio"],
    winsize=1.0,
    show_raw_signal=False,
    show_nf_signal=False,
    show_brain_activation=False,
    verbose=False,
)

# %%
# Inspect the NF feature time-series
# ------------------------------------
# After the session, ``nf.nf_data`` holds a dict mapping each modality name
# to a list of scalar values — one per analysis window.

print("Modalities recorded:", list(nf.nf_data.keys()))
for mod, vals in nf.nf_data.items():
    arr = np.asarray(vals)
    print(f"  {mod}: {len(arr)} samples, mean={arr.mean():.4f}, std={arr.std():.4f}")

# %%
# Plot NF signals
# ---------------
# The feature values are plotted as a time-series.  Vertical dashed lines
# mark each 5-second epoch boundary.

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
fig.suptitle("ANT — recorded NF features (mock session)", fontweight="bold")

colors = ["#2E86AB", "#E84855"]
labels = {"sensor_power": "Alpha power (a.u.)", "band_ratio": "θ/β ratio (a.u.)"}

for ax, (mod, vals), color in zip(axes, nf.nf_data.items(), colors):
    t = np.arange(len(vals))            # one point per 1-s window
    ax.plot(t, vals, color=color, lw=1.5, label=mod)
    ax.fill_between(t, vals, alpha=0.15, color=color)
    ax.axhline(np.mean(vals), ls="--", lw=0.8, color="grey", label="mean")
    for boundary in range(0, len(vals), 5):
        ax.axvline(boundary, ls=":", lw=0.6, color="grey", alpha=0.4)
    ax.set_ylabel(labels.get(mod, mod), fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

axes[-1].set_xlabel("Analysis window index (1 s each)")
fig.tight_layout()
plt.show()
