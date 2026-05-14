"""
Real-time artifact correction comparison
=========================================

This example builds a 32-channel EEG recording, injects realistic artifacts,
and compares five artifact handling approaches side-by-side:

* :class:`~ant.tools.AdaptiveLMSFilter` — reference-based adaptive filter.
* :class:`~ant.tools.ASRDenoiser` — baseline-calibrated subspace rejection.
* :class:`~ant.tools.GEDAIDenoiser` — band-selective eigendecomposition.
* :class:`~ant.tools.ORICA` — online independent component analysis.
* :class:`~ant.tools.RiemannianPotatoDetector` — Riemannian covariance-based
  **artifact detection** (flags windows; does not reconstruct the signal).
"""

# %%
# Synthetic EEG with 10-20 montage
# ---------------------------------

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch, butter, sosfiltfilt

from ant.tools import AdaptiveLMSFilter, ASRDenoiser, GEDAIDenoiser, ORICA
from ant.tools import RiemannianPotatoDetector

rng = np.random.default_rng(42)
mne.set_log_level("WARNING")
plt.style.use("default")

SFREQ = 256.0
DURATION = 30.0
N_TIMES = int(SFREQ * DURATION)
t = np.arange(N_TIMES) / SFREQ
N_CAL = int(5.0 * SFREQ)   # first 5 s kept clean for ASR/GEDAI calibration

EEG_NAMES = [
    "Fp1", "Fp2", "F7",  "F3",  "Fz",  "F4",  "F8",
    "T7",  "C3",  "Cz",  "C4",  "T8",
    "P7",  "P3",  "Pz",  "P4",  "P8",
    "O1",  "Oz",  "O2",
    "AF3", "AF4", "FC5", "FC1", "FC2", "FC6",
    "CP5", "CP1", "CP2", "CP6",
    "TP9", "TP10",
]
N_EEG = len(EEG_NAMES)   # 32


def _pink_noise(n_ch, n_times, rng_, amplitude=5e-6):
    freqs = np.fft.rfftfreq(n_times)
    freqs[0] = 1.0
    wn = rng_.standard_normal((n_ch, n_times))
    spectrum = np.fft.rfft(wn, axis=1) / np.sqrt(freqs)
    pink = np.real(np.fft.irfft(spectrum, n=n_times, axis=1))
    pink /= pink.std(axis=1, keepdims=True) + 1e-30
    return pink * amplitude


data_clean = _pink_noise(N_EEG, N_TIMES, rng, amplitude=5e-6)
posterior = [EEG_NAMES.index(c) for c in ["O1", "Oz", "O2", "P3", "Pz", "P4"]]
data_clean[posterior] += 3e-6 * np.sin(2 * np.pi * 10.0 * t)

info = mne.create_info(ch_names=EEG_NAMES, sfreq=SFREQ, ch_types="eeg", verbose=False)
raw_clean = mne.io.RawArray(data_clean, info, verbose=False)
raw_clean.set_montage(
    mne.channels.make_standard_montage("standard_1020"),
    match_case=False, on_missing="ignore", verbose=False,
)

# %%
# Artifact injection
# ------------------

raw_noisy = raw_clean.copy()
mne.simulation.add_eog(raw_noisy, random_state=42, verbose=False)
raw_noisy._data[:, :N_CAL] = data_clean[:, :N_CAL]
data_noisy_eeg = raw_noisy.get_data()

heart_rate = 70.0 / 60.0
beat_times = np.arange(5.0, DURATION, 1.0 / heart_rate)


def _qrs(t_rel):
    p  = 0.15 * np.exp(-((t_rel + 0.15) / 0.025) ** 2)
    q  = -0.10 * np.exp(-((t_rel - 0.005) / 0.008) ** 2)
    r  = 1.00  * np.exp(-((t_rel) / 0.012) ** 2)
    s  = -0.20 * np.exp(-((t_rel + 0.020) / 0.012) ** 2)
    tw = 0.25  * np.exp(-((t_rel - 0.22) / 0.06) ** 2)
    return p + q + r + s + tw


ecg_pattern = np.zeros(N_EEG)
for ch in ["T7", "T8", "TP9", "TP10", "P7", "P8", "CP5", "CP6"]:
    ecg_pattern[EEG_NAMES.index(ch)] = 1.0
for ch in ["C3", "Cz", "C4", "P3", "Pz", "P4"]:
    ecg_pattern[EEG_NAMES.index(ch)] = 0.5
ecg_pattern /= ecg_pattern.max()

for tb in beat_times:
    t_rel = t - tb
    beat_mask = (t_rel > -0.3) & (t_rel < 0.5)
    qrs_wave = np.zeros(N_TIMES)
    qrs_wave[beat_mask] = _qrs(t_rel[beat_mask])
    data_noisy_eeg += np.outer(ecg_pattern * 2e-6, qrs_wave)

sos_emg = butter(4, [40.0, 120.0], btype="bandpass", fs=SFREQ, output="sos")
for t0 in [8.0, 18.0]:
    i0, i1 = int(t0 * SFREQ), int((t0 + 1.0) * SFREQ)
    for ch in ["T7", "T8", "TP9", "TP10"]:
        ci = EEG_NAMES.index(ch)
        burst = sosfiltfilt(sos_emg, rng.standard_normal(i1 - i0) * 30e-6)
        data_noisy_eeg[ci, i0:i1] += burst

fp1_idx = EEG_NAMES.index("Fp1")
fp2_idx = EEG_NAMES.index("Fp2")
eog_ref = 0.5 * (data_noisy_eeg[fp1_idx] + data_noisy_eeg[fp2_idx])
eog_ref += rng.standard_normal(N_TIMES) * 0.05e-6
data_noisy = np.vstack([data_noisy_eeg, eog_ref[np.newaxis]])  # (33, N_TIMES)

print(f"Clean EEG RMS : {data_clean.std(axis=1).mean()*1e6:.2f} µV")
print(f"Noisy EEG RMS : {data_noisy[:N_EEG].std(axis=1).mean()*1e6:.2f} µV")

# %%
# LMS adaptive filter
# --------------------

lms = AdaptiveLMSFilter(ref_ch_idx=N_EEG, n_taps=8, mu=0.005)
chunk_size = int(SFREQ)
data_lms = np.zeros_like(data_noisy)
for k in range(N_TIMES // chunk_size):
    sl = slice(k * chunk_size, (k + 1) * chunk_size)
    data_lms[:, sl] = lms.transform(data_noisy[:, sl])
print(f"LMS  frontal RMS = {data_lms[:7].std()*1e6:.1f} µV")

# %%
# ASR (Artifact Subspace Reconstruction)
# ----------------------------------------

asr = ASRDenoiser(cutoff=5.0)
asr.fit(data_noisy[:N_EEG, :N_CAL], sfreq=SFREQ, window_len=1.0)
data_asr = data_noisy.copy()
data_asr[:N_EEG, N_CAL:] = asr.transform(data_noisy[:N_EEG, N_CAL:])
print(f"ASR  frontal RMS = {data_asr[:7].std()*1e6:.1f} µV")

# %%
# GEDAI
# ------

gedai = GEDAIDenoiser(n_channels=N_EEG)
gedai.fit_from_raw(data_noisy[:N_EEG, :N_CAL], sfreq=SFREQ, band=(8.0, 30.0))
n_noise = max(2, int(0.20 * N_EEG))
artifact_idx = gedai.find_noise_components(n_noise=n_noise)
data_gedai = data_noisy.copy()
data_gedai[:N_EEG] = gedai.denoise(data_noisy[:N_EEG], artifact_idx)
print(f"GEDAI frontal RMS = {data_gedai[:7].std()*1e6:.1f} µV  "
      f"(removed {len(artifact_idx)} components)")

# %%
# ORICA — Online ICA
# -------------------
# :class:`~ant.tools.ORICA` updates its ICA unmixing matrix online each chunk.
# After processing the full recording we identify artifact components by their
# Pearson correlation with the frontal EOG reference, then use
# :meth:`~ant.tools.ORICA.denoise` to back-project with those components
# zeroed out.

orica = ORICA(n_channels=N_EEG, learning_rate=0.005, block_size=chunk_size)
orica_sources_list = []

for k in range(N_TIMES // chunk_size):
    sl = slice(k * chunk_size, (k + 1) * chunk_size)
    src = orica.transform(data_noisy[:N_EEG, sl])
    orica_sources_list.append(src)

# Full source matrix: (n_components, n_times)
sources_full = np.concatenate(orica_sources_list, axis=1)

# Identify artifact components by correlation with EOG reference
corr_eog = np.array([
    np.corrcoef(sources_full[i], eog_ref)[0, 1]
    for i in range(N_EEG)
])
n_art = max(1, int(0.10 * N_EEG))  # remove top 10 % most EOG-correlated
art_comps = list(np.argsort(np.abs(corr_eog))[::-1][:n_art])

# Back-project via built-in denoise (zeros artifact ICs, reconstructs sensor data)
data_orica = orica.denoise(data_noisy[:N_EEG], art_comps)
print(f"ORICA frontal RMS = {data_orica[:7].std()*1e6:.1f} µV  "
      f"(zeroed {n_art} component(s), max |r_eog|={np.abs(corr_eog).max():.3f})")

# %%
# Riemannian Potato — artifact detection
# ----------------------------------------
# The :class:`~ant.tools.RiemannianPotatoDetector` flags windows whose
# covariance matrix is too far from the clean calibration set in the
# Riemannian metric.  It does **not** reconstruct the signal; instead
# it marks windows as clean or artifactual so the NF engine can skip them.

potato = RiemannianPotatoDetector(threshold=3.0)
# Calibrate on the first 5 s (N_CAL samples) split into 1-s windows
cal_windows = np.array([
    data_noisy[:N_EEG, k * chunk_size:(k + 1) * chunk_size]
    for k in range(N_CAL // chunk_size)
])
potato.fit(cal_windows)

potato_clean = []   # True = clean window
potato_z     = []   # z-score per window
for k in range(N_TIMES // chunk_size):
    sl = slice(k * chunk_size, (k + 1) * chunk_size)
    is_clean, z = potato.detect(data_noisy[:N_EEG, sl])
    potato_clean.append(is_clean)
    potato_z.append(z)

potato_clean = np.array(potato_clean)
potato_z     = np.array(potato_z)
n_rejected   = int((~potato_clean).sum())
pct_rejected = 100.0 * n_rejected / len(potato_clean)
print(f"Riemannian Potato: {n_rejected}/{len(potato_clean)} windows flagged "
      f"({pct_rejected:.0f} %)")

# %%
# Evaluation metrics
# -------------------

eval_sl = slice(N_CAL, None)
clean_ref = data_clean[:, eval_sl]
noisy_ref = data_noisy[:N_EEG, eval_sl]


def _pearson_mean(a, b):
    az = a - a.mean(axis=1, keepdims=True)
    bz = b - b.mean(axis=1, keepdims=True)
    num = (az * bz).sum(axis=1)
    den = np.sqrt((az ** 2).sum(axis=1) * (bz ** 2).sum(axis=1)) + 1e-30
    return float((num / den).mean())


def _snr_gain_db(clean, noisy, corrected):
    pre  = np.var(clean - noisy)
    post = np.var(clean - corrected) + 1e-60
    return float(10.0 * np.log10(pre / post))


methods = {
    "LMS"  : data_lms[:N_EEG, eval_sl],
    "ASR"  : data_asr[:N_EEG, eval_sl],
    "GEDAI": data_gedai[:N_EEG, eval_sl],
    "ORICA": data_orica[:, eval_sl],
}

metrics = {}
for name, corr in methods.items():
    r   = _pearson_mean(clean_ref, corr)
    snr = _snr_gain_db(clean_ref, noisy_ref, corr)
    metrics[name] = dict(r=r, snr=snr)
    print(f"{name:5s}  Pearson r = {r:.4f}  |  SNR gain = {snr:+.2f} dB")

# %%
# Figure 1 — Time-series comparison
# ------------------------------------

COLORS = {
    "Clean": "#555555", "Noisy": "#D32F2F",
    "LMS": "#1565C0", "ASR": "#2E7D32",
    "GEDAI": "#E65100", "ORICA": "#6A1B9A",
}

t20 = t[t <= 20.0]
s20 = slice(0, len(t20))

fig1, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True,
                           gridspec_kw={"hspace": 0.18})
fig1.suptitle(
    "Real-time Artifact Correction — Time-series (0–20 s)\n"
    "LMS · ASR · GEDAI · ORICA  vs. clean reference",
    fontsize=13, fontweight="bold", y=0.99,
)

ch_rows = [
    (EEG_NAMES.index("Fp1"), "Frontal — Fp1 (blink + cardiac)"),
    (EEG_NAMES.index("T7"),  "Temporal — T7 (muscle burst at 8 s and 18 s)"),
]

for ax, (ch, title) in zip(axes[:2], ch_rows):
    ax.plot(t20, data_clean[ch, s20] * 1e6,
            color=COLORS["Clean"], lw=2.0, label="Clean", zorder=5)
    ax.plot(t20, data_noisy[ch, s20] * 1e6,
            color=COLORS["Noisy"], lw=0.8, alpha=0.50, label="Noisy", zorder=2)
    for mname, mdata in [("LMS", data_lms), ("ASR", data_asr),
                         ("GEDAI", data_gedai), ("ORICA", data_orica)]:
        ax.plot(t20, mdata[ch, s20] * 1e6, color=COLORS[mname],
                lw=1.2, alpha=0.9, label=mname, zorder=4)
    ax.set_ylabel("µV", fontsize=10)
    ax.set_title(title, fontsize=10, loc="left", pad=3)
    ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=6)
    ax.spines[["top", "right"]].set_visible(False)

# Riemannian Potato rejection mask
ax3 = axes[2]
window_centers = (np.arange(len(potato_clean)) + 0.5) * (chunk_size / SFREQ)
window_centers_20 = window_centers[window_centers <= 20.0]
is_art = ~potato_clean[:len(window_centers_20)]

for i, (wc, art) in enumerate(zip(window_centers_20, is_art)):
    color = "#D32F2F" if art else "#2E7D32"
    ax3.axvspan(wc - 0.5, wc + 0.5, alpha=0.35, color=color, linewidth=0)

ax3.plot(window_centers_20, potato_z[:len(window_centers_20)],
         color="#6A1B9A", lw=1.5, zorder=3, label="Riemannian z-score")
ax3.axhline(3.0, color="#D32F2F", lw=1.2, ls="--", label="threshold = 3.0")
ax3.set_ylabel("z-score", fontsize=10)
ax3.set_xlabel("Time (s)", fontsize=10)
ax3.set_title(
    "Riemannian Potato detection — green = clean window · red = artifact window",
    fontsize=10, loc="left", pad=3,
)
ax3.legend(fontsize=9, frameon=False)
ax3.spines[["top", "right"]].set_visible(False)

fig1.tight_layout()

# %%
# Figure 2 — PSD and quantitative metrics
# ------------------------------------------

fig2, (ax_psd, ax_bar) = plt.subplots(1, 2, figsize=(15, 7),
                                       gridspec_kw={"wspace": 0.32})
fig2.suptitle(
    "Artifact Correction — PSD and Quantitative Metrics\n"
    "LMS · ASR · GEDAI · ORICA  |  Pearson r with clean signal · SNR gain",
    fontsize=12, fontweight="bold",
)

psd_kw = dict(fs=SFREQ, nperseg=int(4 * SFREQ), noverlap=int(2 * SFREQ))
psd_sets = [
    ("Clean",  data_clean,          COLORS["Clean"],  dict(lw=2.4, ls="-")),
    ("Noisy",  data_noisy[:N_EEG],  COLORS["Noisy"],  dict(lw=1.2, ls="-", alpha=0.6)),
    ("LMS",    data_lms[:N_EEG],    COLORS["LMS"],    dict(lw=1.5, ls="--")),
    ("ASR",    data_asr[:N_EEG],    COLORS["ASR"],    dict(lw=1.5, ls="-.")),
    ("GEDAI",  data_gedai[:N_EEG],  COLORS["GEDAI"],  dict(lw=1.5, ls=":")),
    ("ORICA",  data_orica,          COLORS["ORICA"],  dict(lw=1.5, ls=(0, (3, 1, 1, 1)))),
]

for label, dat, col, kw in psd_sets:
    psds = [welch(dat[i], **psd_kw)[1] for i in range(dat.shape[0])]
    f_arr = welch(dat[0], **psd_kw)[0]
    mask = (f_arr >= 1.0) & (f_arr <= 120.0)
    ax_psd.semilogy(f_arr[mask], np.mean(psds, axis=0)[mask], color=col, label=label, **kw)

ax_psd.set_xlabel("Frequency (Hz)", fontsize=11)
ax_psd.set_ylabel("PSD  (V²/Hz)", fontsize=11)
ax_psd.set_title("Power spectral density (avg, EEG channels)", fontsize=11)
ax_psd.legend(fontsize=10, frameon=False)
ax_psd.spines[["top", "right"]].set_visible(False)

names    = list(metrics)
r_vals   = [metrics[n]["r"]   for n in names]
snr_vals = [metrics[n]["snr"] for n in names]
bar_cols = [COLORS[n] for n in names]

x = np.arange(len(names))
w = 0.33
b1 = ax_bar.bar(x - w / 2, r_vals,   w, color=bar_cols, alpha=0.85, label="Pearson r")
b2 = ax_bar.bar(x + w / 2, snr_vals, w, color=bar_cols, alpha=0.40,
                edgecolor=bar_cols, linewidth=1.8, label="SNR gain (dB)")
for bar, val in zip(b1, r_vals):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(b2, snr_vals):
    offset = 0.12 if val >= 0 else -0.7
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{val:+.1f}", ha="center", va="bottom", fontsize=9)

ax_bar.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(names, fontsize=12)
ax_bar.set_ylabel("Metric value", fontsize=11)
ax_bar.set_title("Pearson r with clean signal  |  SNR gain (dB)", fontsize=11)
ax_bar.legend(fontsize=10, frameon=False)
ax_bar.spines[["top", "right"]].set_visible(False)

fig2.tight_layout()
