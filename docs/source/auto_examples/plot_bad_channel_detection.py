"""
Real-time bad channel detection
=================================

EEG and MEG recordings often contain disconnected, noisy, or poorly coupled
channels that contaminate downstream artifact correction and NF feature
extraction.  :class:`~ant.tools.BadChannelDetector` evaluates independent
criteria per incoming window and uses a rolling majority-vote to flag
persistently bad channels — no baseline recording is needed.

This example uses the MNE sample dataset (60 EEG channels, 60 s), which
already contains one known bad channel (``EEG 053``).  We also inject a
second artificially degraded channel so we can validate both detection paths.

The data is broadcast over a local LSL stream with
:class:`~mne_lsl.player.PlayerLSL` and processed chunk-by-chunk as it arrives
through :class:`~mne_lsl.stream.StreamLSL` — the same pipeline used in a live
recording session.

**Criteria used here:**

* ``"flat"`` — RMS below threshold (dead/disconnected electrode).
* ``"hf_noise"`` — abnormally high high-frequency power fraction (EMG, cable
  noise).  ``EEG 053`` has an HF ratio ~4.5× above the median — the
  signature of this dataset's known bad channel.

.. note::

   The ``"variance"`` and ``"correlation"`` criteria are omitted because the
   MNE sample EEG was recorded *inside an MEG scanner*.  In that environment,
   channel-to-channel amplitude spread and spatial neighbour correlation are
   atypical, causing false positives on every channel.  For standalone EEG
   setups, enable all four criteria.
"""

# %%
# Load MNE sample EEG data
# --------------------------
# We pick only EEG channels, un-mark the known bad so the detector must find
# it, and inject one additional artificial flat channel for a controlled
# validation.

import os
import tempfile
import time

import matplotlib.pyplot as plt
import mne
import numpy as np

from ant.tools import BadChannelDetector

mne.set_log_level("WARNING")
plt.style.use("default")

sample_path = mne.datasets.sample.data_path()
raw_file = os.path.join(sample_path, "MEG", "sample", "sample_audvis_raw.fif")

raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
raw.pick_types(meg=False, eeg=True, stim=False, exclude=[])
raw.crop(tmin=0.0, tmax=60.0)
raw.filter(l_freq=1.0, h_freq=None, verbose=False)

# The file marks EEG 053 as bad — clear it so the detector must find it
KNOWN_BAD = "EEG 053"
raw.info["bads"] = []
print(f"EEG channels: {len(raw.ch_names)}  |  known bad: {KNOWN_BAD}")

# %%
# Inject an additional flat (dead) channel
# -----------------------------------------

INJECTED_BAD = raw.ch_names[10]   # EEG 011
flat_start = 15.0                  # flat from t = 15 s onwards

data = raw.get_data()
flat_idx = raw.ch_names.index(INJECTED_BAD)
data[flat_idx, int(flat_start * raw.info["sfreq"]):] = 0.0
raw._data = data
print(f"Injected flat channel : {INJECTED_BAD}  (flat from t = {flat_start:.0f} s)")

# %%
# Configure the detector
# -----------------------
# We use only ``"flat"`` and ``"hf_noise"`` because the MNE sample EEG was
# recorded inside an MEG scanner — correlation and variance criteria
# produce false positives on every channel in that environment.
# ``hf_threshold=6.0`` (conservative robust z-score) flags only the genuinely
# anomalous HF content of ``EEG 053`` while leaving all clean channels alone.

sfreq    = round(raw.info["sfreq"])
info     = raw.info

detector = BadChannelDetector(
    info,
    method=["flat", "hf_noise"],
    flat_threshold=1e-8,   # below 10 nV RMS → flat
    hf_threshold=6.0,      # robust z-score; EEG 053 scores ~10 here
    history_windows=20,
    min_bad_frac=0.6,      # bad in ≥ 60 % of recent windows
)

# %%
# Stream via LSL and detect bad channels
# ----------------------------------------
# We broadcast the injected raw over a local LSL stream and consume it with
# :class:`~mne_lsl.stream.StreamLSL`.  The detector receives the same 1-second
# chunks it would in a live session.  Falls back to chunk-by-chunk if
# ``mne_lsl`` is unavailable (e.g. during documentation builds).

score_history  = {ch: [] for ch in raw.ch_names}
bad_per_window = []
window_times   = []
chunk_size     = sfreq   # 1-second window

_streaming_used = False

try:
    from mne_lsl.player import PlayerLSL
    from mne_lsl.stream import StreamLSL
    if not (hasattr(PlayerLSL, "__mro__") and "MagicMock" not in str(type(PlayerLSL))):
        raise ImportError("mne_lsl is mocked")

    STREAM_NAME = "ANT_BadCh_demo"
    with tempfile.NamedTemporaryFile(suffix="_raw.fif", delete=False) as _f:
        _tmp_path = _f.name
    raw.save(_tmp_path, overwrite=True, verbose=False)

    _player = PlayerLSL(_tmp_path, chunk_size=chunk_size, name=STREAM_NAME, n_repeat=1)
    _player.start()
    time.sleep(0.5)

    _stream = StreamLSL(bufsize=4.0, name=STREAM_NAME)
    _stream.connect(acquisition_delay=0.005, timeout=10.0)
    _sfreq_lsl = float(_stream.info["sfreq"])
    print(f"Streaming: {STREAM_NAME}  |  sfreq={_sfreq_lsl:.0f} Hz  |  "
          f"n_ch={int(_stream.info['nchan'])}")

    _n_chunks   = int(raw.times[-1])   # 60 one-second windows
    _t_deadline = time.perf_counter() + raw.times[-1] + 15.0
    _k          = 0

    while _k < _n_chunks and time.perf_counter() < _t_deadline:
        if _stream.n_new_samples < chunk_size:
            time.sleep(0.005)
            continue
        chunk, _ = _stream.get_data(winsize=1.0)      # (n_ch, ~chunk_size)
        bad       = detector.update(chunk)
        bad_per_window.append(list(bad))
        window_times.append((_k + 0.5))
        for ch, sc in detector.scores_.items():
            score_history[ch].append(sc)
        _k += 1

    _stream.disconnect()
    try:
        _player.stop()
    except RuntimeError:
        pass
    os.unlink(_tmp_path)
    _streaming_used = True
    print(f"Processed {_k} windows via LSL streaming")

except Exception as _exc:
    print(f"LSL streaming unavailable ({_exc}), using chunk-by-chunk fallback.")
    n_chunks = len(raw.times) // chunk_size
    for k in range(n_chunks):
        sl    = slice(k * chunk_size, (k + 1) * chunk_size)
        chunk = raw.get_data()[:, sl]
        bad   = detector.update(chunk)
        bad_per_window.append(list(bad))
        window_times.append((k + 0.5) * chunk_size / sfreq)
        for ch, sc in detector.scores_.items():
            score_history[ch].append(sc)

# Final bad channels
final_bad = detector.get_bad_channels()
print(f"\nDetected bad channels : {final_bad}")
print(f"Known bad   : {KNOWN_BAD}  → {'✓ found' if KNOWN_BAD in final_bad else '✗ missed'}")
print(f"Injected bad: {INJECTED_BAD} → {'✓ found' if INJECTED_BAD in final_bad else '✗ missed'}")
mode = "LSL stream" if _streaming_used else "chunk-by-chunk"
print(f"Processing mode: {mode}")

# %%
# Figure 1 — Badness score over time
# ------------------------------------
# Each line is one EEG channel's rolling badness score (0 = never bad, 1 =
# always bad in recent windows).  The two bad channels climb above the 0.5
# decision threshold while all others stay near zero.

fig1, (ax_scores, ax_flag) = plt.subplots(
    2, 1, figsize=(14, 9), sharex=True,
    gridspec_kw={"hspace": 0.12, "height_ratios": [3, 1]},
)
fig1.suptitle(
    "Real-time Bad Channel Detection — BadChannelDetector\n"
    f"MNE sample EEG  |  known bad: {KNOWN_BAD}  |  injected flat: {INJECTED_BAD}",
    fontsize=12, fontweight="bold", y=0.99,
)

score_arr = np.array([score_history[ch] for ch in raw.ch_names])

for i, ch in enumerate(raw.ch_names):
    if ch in (KNOWN_BAD, INJECTED_BAD):
        continue
    ax_scores.plot(window_times, score_arr[i], color="#b0b0b8", lw=0.6, alpha=0.5)

for ch, color, ls in [(KNOWN_BAD, "#D32F2F", "-"), (INJECTED_BAD, "#1565C0", "--")]:
    if ch in raw.ch_names:
        i = raw.ch_names.index(ch)
        ax_scores.plot(window_times, score_arr[i], color=color, lw=2.2,
                       label=f"{ch} (bad)", zorder=5)

ax_scores.axhline(0.5, color="#E65100", lw=1.4, ls=":", label="threshold = 0.5")
ax_scores.axvline(flat_start, color="#1565C0", lw=1.0, ls="--", alpha=0.6,
                  label=f"{INJECTED_BAD} goes flat at {flat_start:.0f} s")
ax_scores.set_ylabel("Badness score (0–1)", fontsize=11)
ax_scores.set_ylim(-0.02, 1.05)
ax_scores.legend(fontsize=10, frameon=False, loc="upper left")
ax_scores.spines[["top", "right"]].set_visible(False)

n_bad_ts = [len(b) for b in bad_per_window]
ax_flag.bar(window_times, n_bad_ts, width=0.8, color="#7B1FA2", alpha=0.8)
ax_flag.set_ylabel("# bad channels", fontsize=11)
ax_flag.set_xlabel("Time (s)", fontsize=11)
ax_flag.spines[["top", "right"]].set_visible(False)

fig1.tight_layout()

# %%
# Figure 2 — Final per-channel scores and EEG signal
# ----------------------------------------------------

fig2, (ax_bar, ax_eeg) = plt.subplots(1, 2, figsize=(15, 6),
                                        gridspec_kw={"wspace": 0.35})
fig2.suptitle(
    "Final Badness Scores and Raw EEG Signal\n"
    "Top-15 channels by badness score highlighted",
    fontsize=12, fontweight="bold",
)

final_scores = detector.get_scores()
sorted_chs   = sorted(final_scores, key=final_scores.get, reverse=True)
top_n        = 15
top_chs      = sorted_chs[:top_n]
top_scores   = [final_scores[ch] for ch in top_chs]
bar_colors   = ["#D32F2F" if ch in final_bad else "#90A4AE" for ch in top_chs]

ax_bar.barh(range(top_n), top_scores, color=bar_colors, edgecolor="white")
ax_bar.set_yticks(range(top_n))
ax_bar.set_yticklabels(top_chs, fontsize=9)
ax_bar.invert_yaxis()
ax_bar.axvline(0.5, color="#E65100", lw=1.4, ls=":", label="threshold")
ax_bar.set_xlabel("Badness score", fontsize=11)
ax_bar.set_title("Top 15 channels by badness score\n(red = declared bad)", fontsize=11)
ax_bar.legend(fontsize=10, frameon=False)
ax_bar.spines[["top", "right"]].set_visible(False)

t_plot = raw.times[:int(30 * sfreq)]
scale  = 1e6
for ch, color in [(KNOWN_BAD, "#D32F2F"), (INJECTED_BAD, "#1565C0")]:
    if ch in raw.ch_names:
        idx = raw.ch_names.index(ch)
        ax_eeg.plot(t_plot, raw.get_data()[idx, :len(t_plot)] * scale,
                    color=color, lw=1.2, label=f"{ch}")
ax_eeg.axvline(flat_start, color="#1565C0", lw=1.0, ls="--", alpha=0.6)
ax_eeg.set_xlabel("Time (s)", fontsize=11)
ax_eeg.set_ylabel("µV", fontsize=11)
ax_eeg.set_title("Raw signal of bad channels (first 30 s)", fontsize=11)
ax_eeg.legend(fontsize=10, frameon=False)
ax_eeg.spines[["top", "right"]].set_visible(False)

fig2.tight_layout()
