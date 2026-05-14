"""
Real-time Maxwell filtering (SSS) with LSL streaming
=====================================================

Signal Space Separation (SSS) is the gold-standard preprocessing step for
MEG data, suppressing external interference while preserving brain signals.
ANT's :class:`~ant.tools.RTMaxwellFilter` pre-computes the SSS projection
operator once from sensor geometry and applies it as a single matrix multiply
per incoming chunk — zero added latency, numerically equivalent to offline
MNE processing.

This example covers two complementary demonstrations:

**Numerical equivalence (chunk-by-chunk)**
   Apply RT-SSS on manually sliced 1-second windows and compare with offline
   :func:`mne.preprocessing.maxwell_filter`.  Because the same linear projector
   is applied, the residual is at floating-point precision and Pearson r ≈ 1.0
   for every channel.

**Live streaming latency (LSL)**
   Broadcast the same recording over a local LSL stream using
   :class:`mne_lsl.player.PlayerLSL`, receive it with
   :class:`mne_lsl.stream.StreamLSL`, and apply RT-SSS on every arriving chunk.
   Per-chunk wall-clock latency (``get_data`` + ``transform``) is recorded and
   plotted — this is the metric that matters for online BCI/NF latency budgets.

.. note::

   RT-SSS and offline SSS produce **numerically identical** output for basic
   SSS (no tSSS, no movement compensation).  Both apply the same pre-computed
   projector :math:`\\mathbf{P}_{\\mathrm{SSS}} =
   \\mathbf{S}_{\\mathrm{in}} \\mathbf{S}_{\\mathrm{in}}^\\dagger`.
   Because matrix projection is linear, applying it chunk-by-chunk gives the
   same result as batch offline processing.
"""

# %%
# Load MNE sample data and apply offline SSS
# -------------------------------------------

import os
import tempfile
import time

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from mne.preprocessing import maxwell_filter
from scipy.stats import pearsonr

from ant.tools import RTMaxwellFilter

mne.set_log_level("WARNING")
plt.style.use("default")

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
)

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
raw.pick_types(meg=True, eeg=False, stim=False, exclude=[])
raw.crop(tmin=30.0, tmax=90.0)
raw.filter(l_freq=1.0, h_freq=100.0, verbose=False)

print(f"Raw MEG: {len(raw.ch_names)} channels, "
      f"{raw.times[-1]:.1f} s, sfreq={raw.info['sfreq']:.3f} Hz")

raw_offline = maxwell_filter(raw, origin="auto", int_order=8, ext_order=3, verbose=False)
data_offline = raw_offline.get_data()
print(f"Offline SSS done: data shape {data_offline.shape}")

# %%
# Fit RTMaxwellFilter
# --------------------
# The SSS operator depends only on sensor geometry — no baseline recording
# is required.

rt_mf = RTMaxwellFilter(int_order=8, ext_order=3)
rt_mf.fit(raw.info)

print(rt_mf)
print(f"Internal moments retained: {rt_mf.n_use_in}")

# %%
# Apply RT-SSS chunk-by-chunk (numerical equivalence check)
# ----------------------------------------------------------
# We simulate a real-time stream by slicing 1-second windows and calling
# :meth:`~ant.tools.RTMaxwellFilter.transform` on each.  This confirms that
# chunked online processing is numerically identical to offline SSS.

sfreq       = raw.info["sfreq"]           # 600.614 Hz (float)
chunk_samps = round(sfreq)                # samples per 1-second window
data_raw    = raw.get_data()              # (n_ch, n_times)
n_ch, n_times = data_raw.shape

n_chunks = n_times // chunk_samps
data_rt  = np.zeros_like(data_raw)

for i in range(n_chunks):
    sl = slice(i * chunk_samps, (i + 1) * chunk_samps)
    data_rt[:, sl] = rt_mf.transform(data_raw[:, sl])

if n_times % chunk_samps:
    sl = slice(n_chunks * chunk_samps, n_times)
    data_rt[:, sl] = rt_mf.transform(data_raw[:, sl])

print(f"RT-SSS applied over {n_chunks} chunks of {chunk_samps} samples each")

# %%
# Residual analysis — confirming numerical equivalence
# -----------------------------------------------------

mag_picks     = mne.pick_types(raw.info, meg="mag",  exclude=[])
grad_picks    = mne.pick_types(raw.info, meg="grad", exclude=[])
all_meg_picks = mne.pick_types(raw.info, meg=True,   exclude=[])

residuals           = data_offline - data_rt
residual_rms_fT     = np.sqrt(np.mean(residuals[mag_picks] ** 2)) * 1e15
signal_rms_fT       = np.sqrt(np.mean(data_offline[mag_picks] ** 2)) * 1e15

print(f"Signal RMS  (mag): {signal_rms_fT:.1f} fT")
print(f"Residual RMS(mag): {residual_rms_fT:.6f} fT  "
      f"({residual_rms_fT / signal_rms_fT * 100:.2e} % of signal)")

# %%
# Per-channel Pearson correlation (offline vs. RT-SSS)
# ------------------------------------------------------

corr_all = np.array(
    [pearsonr(data_offline[ch], data_rt[ch])[0] for ch in all_meg_picks]
)
ch_types = np.array(
    ["mag" if ch in mag_picks else "grad" for ch in all_meg_picks]
)

print(f"Mean Pearson r (offline vs RT-SSS): {corr_all.mean():.6f}")
print(f"  Magnetometers : {corr_all[ch_types == 'mag'].mean():.6f}")
print(f"  Gradiometers  : {corr_all[ch_types == 'grad'].mean():.6f}")

# %%
# Live streaming latency via LSL
# --------------------------------
# We broadcast the same recording over a local LSL stream with
# :class:`~mne_lsl.player.PlayerLSL` and receive it with
# :class:`~mne_lsl.stream.StreamLSL`.  For each arriving chunk we measure
# **wall-clock latency** = time for one ``get_data`` call + one
# ``RTMaxwellFilter.transform`` call.  This is the latency budget that
# matters for online brain–computer interface and neurofeedback applications.
#
# The streaming section is skipped gracefully when ``mne_lsl`` is not
# available (e.g. during documentation builds).

chunk_latencies = np.array([])   # populated if mne_lsl is available
n_lsl_chunks    = 0

try:
    from mne_lsl.player import PlayerLSL
    from mne_lsl.stream import StreamLSL

    # Verify these are real classes, not documentation mocks
    if not hasattr(PlayerLSL, "__mro__") or "MagicMock" in str(type(PlayerLSL)):
        raise ImportError("mne_lsl is mocked")

    STREAM_NAME = "ANT_RT_SSS_demo"
    _latencies  = []

    with tempfile.NamedTemporaryFile(suffix="_raw.fif", delete=False) as tmp:
        _tmp_path = tmp.name
    raw.save(_tmp_path, overwrite=True, verbose=False)

    _player = PlayerLSL(_tmp_path, chunk_size=chunk_samps, name=STREAM_NAME, n_repeat=1)
    _player.start()
    time.sleep(0.5)

    _stream = StreamLSL(bufsize=4.0, name=STREAM_NAME)
    _stream.connect(acquisition_delay=0.005, timeout=10.0)
    _sfreq_lsl = float(_stream.info["sfreq"])
    _n_ch_lsl  = int(_stream.info["nchan"])
    print(f"Connected to LSL stream: {STREAM_NAME}  |  "
          f"sfreq={_sfreq_lsl:.3f} Hz  |  n_ch={_n_ch_lsl}")

    _t_deadline = time.perf_counter() + n_times / sfreq + 30.0
    while n_lsl_chunks < n_chunks and time.perf_counter() < _t_deadline:
        if _stream.n_new_samples < chunk_samps:
            time.sleep(0.005)
            continue
        _t0 = time.perf_counter()
        _chunk, _ = _stream.get_data(winsize=1.0)
        rt_mf.transform(_chunk)
        _latencies.append((time.perf_counter() - _t0) * 1000.0)
        n_lsl_chunks += 1

    _stream.disconnect()
    try:
        _player.stop()
    except RuntimeError:
        pass
    os.unlink(_tmp_path)

    chunk_latencies = np.array(_latencies)
    print(f"Streamed {n_lsl_chunks} chunks via LSL")
    if len(chunk_latencies):
        print(f"Latency — mean: {chunk_latencies.mean():.2f} ms  "
              f"median: {np.median(chunk_latencies):.2f} ms  "
              f"p95: {np.percentile(chunk_latencies, 95):.2f} ms")

except Exception as _exc:
    print(f"LSL streaming section skipped: {_exc}")

# %%
# Figure 1 — Time-series comparison and residual
# ------------------------------------------------

sns.set_theme(style="ticks", font_scale=1.0)
plt.rcParams["font.sans-serif"] = ["Helvetica"]

target_names = ["MEG 0111", "MEG 0121", "MEG 0131"]
plot_chs = [raw.ch_names.index(n) for n in target_names if n in raw.ch_names]
if len(plot_chs) < 3:
    plot_chs = list(mag_picks[:3])

t        = raw.times
t_mask   = t <= 10.0
scale_fT = 1e15

fig1, axes = plt.subplots(
    4, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1, 0.8], "hspace": 0.25},
)
fig1.suptitle(
    "RT-SSS vs. Offline SSS — Time Series\nMNE Sample MEG (chunk-by-chunk comparison)",
    fontsize=12, fontweight="bold", y=0.99,
)

colors = {
    "raw":      "#D1D1D1",
    "offline":  "#005EB8",
    "rt":       "#FF4F00",
    "residual": "#6A0DAD",
}

for i, (ax, ch_idx) in enumerate(zip(axes[:3], plot_chs)):
    ch_name = raw.ch_names[ch_idx]
    ax.plot(t[t_mask], data_raw[ch_idx, t_mask] * scale_fT,
            color=colors["raw"], lw=1.0, label="Raw", zorder=1)
    ax.plot(t[t_mask], data_offline[ch_idx, t_mask] * scale_fT,
            color=colors["offline"], lw=2.0, label="Offline SSS", zorder=3)
    ax.plot(t[t_mask], data_rt[ch_idx, t_mask] * scale_fT,
            color=colors["rt"], lw=1.2, ls=(0, (3, 1.5)), label="RT-SSS", zorder=4)
    ax.set_ylabel("fT", fontweight="bold", fontsize=10)
    ax.set_title(f"Channel: {ch_name}", fontsize=11, loc="left", fontweight="semibold")
    sns.despine(ax=ax, trim=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if i == 0:
        ax.legend(bbox_to_anchor=(1.0, 1.15), loc="upper right", ncol=3,
                  frameon=False, fontsize=10)

res_ch        = plot_chs[0]
residual_plot = residuals[res_ch, t_mask] * 1e18   # T → aT
ax_res        = axes[3]
ax_res.fill_between(t[t_mask], residual_plot, color=colors["residual"], alpha=0.2)
ax_res.plot(t[t_mask], residual_plot, color=colors["residual"], lw=1.2)
ax_res.axhline(0, color="black", lw=0.8, alpha=0.6)
ax_res.set_ylabel("aT", fontweight="bold", fontsize=10)
ax_res.set_xlabel("Time (seconds)", fontsize=11)
rms_text = f"RMS Error = {np.sqrt(np.mean(residuals[res_ch]**2))*1e18:.3f} aT"
ax_res.set_title(f"Residual (Offline − RT-SSS) | {rms_text}",
                 fontsize=10, loc="left", color="#444444")
sns.despine(ax=ax_res, trim=True)
fig1.tight_layout()

# %%
# Figure 2 — Power spectral density
# -----------------------------------

from mne.time_frequency import psd_array_welch

psd_kwargs = dict(
    sfreq=sfreq,
    n_fft=int(4 * sfreq),
    n_overlap=int(2 * sfreq),
    verbose=False,
)

def _mean_psd(data, picks, **kw):
    psds, freqs = psd_array_welch(data[picks], **kw)
    return freqs, psds.mean(axis=0)

f_raw, pxx_raw     = _mean_psd(data_raw,     mag_picks, **psd_kwargs)
f_off, pxx_offline = _mean_psd(data_offline, mag_picks, **psd_kwargs)
f_rt,  pxx_rt      = _mean_psd(data_rt,      mag_picks, **psd_kwargs)

fig2, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("PSD Comparison — Raw / Offline SSS / RT-SSS",
              fontsize=12, fontweight="bold")

for ax, flim, title in [
    (ax_full, (1, 100), "Full range 1–100 Hz"),
    (ax_zoom, (1,  40), "Close-up 1–40 Hz"),
]:
    mask = (f_raw >= flim[0]) & (f_raw <= flim[1])
    ax.semilogy(f_raw[mask], pxx_raw[mask],
                color="#CCCCCC", lw=1.0, label="Raw", zorder=1)
    ax.semilogy(f_off[mask], pxx_offline[mask],
                color="#1565C0", lw=2.5, label="Offline SSS", zorder=2)
    ax.semilogy(f_rt[mask], pxx_rt[mask],
                color="#E65100", lw=1.5, ls=(0, (3, 1)), label="RT-SSS", zorder=3)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("PSD (T²/Hz)", fontsize=11)
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold")
    sns.despine(ax=ax)
    if ax == ax_full:
        ax.legend(fontsize=10, frameon=False, loc="upper right")

fig2.tight_layout()

# %%
# Figure 3 — Per-channel correlation and LSL streaming latency
# -------------------------------------------------------------
# Left: Pearson r for every MEG channel (offline vs. RT-SSS) — all channels
# cluster at r ≈ 1.
# Right: per-chunk processing latency measured during the live LSL stream
# (``get_data`` + ``RTMaxwellFilter.transform`` per 1-second window).

fig3, (ax_corr, ax_lat) = plt.subplots(1, 2, figsize=(14, 5),
                                        gridspec_kw={"wspace": 0.35})
fig3.suptitle(
    "Per-channel Correlation (chunk-by-chunk) and LSL Streaming Latency",
    fontsize=12, fontweight="bold",
)

# Correlation scatter
mag_mask  = ch_types == "mag"
grad_mask = ch_types == "grad"
ax_corr.scatter(np.where(mag_mask)[0],  corr_all[mag_mask],
                s=18, color="#1565C0", alpha=0.8, label="Magnetometers")
ax_corr.scatter(np.where(grad_mask)[0], corr_all[grad_mask],
                s=18, color="#E65100", alpha=0.8, label="Gradiometers")
ax_corr.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
ax_corr.set_ylim(corr_all.min() - 0.001, 1.001)
ax_corr.set_xlabel("Channel index", fontsize=11)
ax_corr.set_ylabel("Pearson r", fontsize=11)
ax_corr.set_title(
    f"Offline vs. RT-SSS\nmean r = {corr_all.mean():.6f}", fontsize=11
)
ax_corr.legend(fontsize=10, frameon=False)
ax_corr.spines[["top", "right"]].set_visible(False)

# Latency bar chart
if len(chunk_latencies):
    chunk_indices = np.arange(len(chunk_latencies))
    ax_lat.bar(chunk_indices, chunk_latencies, color="#607D8B", alpha=0.8, width=0.85)
    ax_lat.axhline(chunk_latencies.mean(), color="#D32F2F", lw=1.5, ls="--",
                   label=f"Mean = {chunk_latencies.mean():.1f} ms")
    ax_lat.axhline(np.percentile(chunk_latencies, 95), color="#FF6F00", lw=1.2, ls=":",
                   label=f"p95 = {np.percentile(chunk_latencies, 95):.1f} ms")
    ax_lat.legend(fontsize=10, frameon=False)

ax_lat.set_xlabel("Chunk index (1 s windows)", fontsize=11)
ax_lat.set_ylabel("Latency (ms)", fontsize=11)
ax_lat.set_title(
    f"Per-chunk LSL latency\n"
    f"(get_data + RTMaxwellFilter.transform)  ·  {len(chunk_latencies)} chunks",
    fontsize=11,
)
ax_lat.spines[["top", "right"]].set_visible(False)

fig3.tight_layout()
