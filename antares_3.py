from pathlib import Path
import logging
import yaml
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mne.io import read_raw_fif
from mne import (
    make_fixed_length_epochs,
    set_log_level,
    Report,
    read_labels_from_annot,
    extract_label_time_course,
)
from mne.channels import make_standard_montage
from mne.time_frequency import psd_array_multitaper
from mne.minimum_norm import (
    read_inverse_operator,
    apply_inverse_epochs
)
from mne_connectivity import spectral_connectivity_time

from autoreject import AutoReject
from fooof import FOOOF
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit import NormData

set_log_level("Error")

def preprocess_baseline(subject_id, visit, subjects_dir):

    ## read the file
    subject_dir = subjects_dir / subject_id
    fname = subject_dir / "baseline" / f"visit_{visit}-raw.fif"
    raw = read_raw_fif(fname, preload=True)
    raw.filter(0.1, 40)
    raw.set_eeg_reference('average', projection=True)
    montage_name = "easycap-M1"
    montage = make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="warn")

    ## preprocess
    epochs = make_fixed_length_epochs(raw, duration=10, preload=True)
    ar = AutoReject(
                    n_interpolate=np.array([1, 4, 8]),
                    consensus=np.linspace(0, 1.0, 11),
                    cv=5,
                    n_jobs=1,
                    random_state=11,
                    verbose=True
                    )
    ar.fit(epochs)
    epochs, reject_log = ar.transform(epochs, return_log=True)
    epochs.save(subject_dir / "baseline" / f"preproc_v{visit}-epo.fif", overwrite=True)

    ## create report
    report = Report(title=f"{subject_id}_visit_{visit}")
    fig_reject = reject_log.plot(show=False)
    report.add_figure(fig=fig_reject, title="autoreject log", image_format="PNG")
    report.save(fname=subject_dir / "baseline" / f"report_v{visit}.html", overwrite=True, open_browser=False)

    return epochs


def extract_features(epochs, subject_id, visit, subjects_dir):

    ## read the file
    subject_dir = subjects_dir / subject_id
    features_dir = subject_dir / "features"
    features_dir.mkdir(exist_ok=True)

    # Create reliability metrics directory
    reliability_dir = subject_dir / "reliability_metrics"
    reliability_dir.mkdir(exist_ok=True)

    freq_bands = {
                "delta": [1, 6],
                "theta": [6.5, 8.5],
                "alpha_0": [8.5, 12.5],
                "alpha_1": [8.5, 10.5],
                "alpha_2": [10.5, 12.5],
                "beta_0": [12.5, 30],
                "beta_1": [12.5, 18.5],
                "beta_2": [18.5, 21],
                "beta_3": [21, 30],
                "gamma": [30, 40]
                }

    ## compute power in sensor
    epochs.pick_types(eeg=True)
    epochs_ts = epochs.get_data(picks="eeg")
    ch_names = epochs.info["ch_names"]

    print("Computing band powers in sensor space ...")
    psd_chs, freqs = epochs.compute_psd(
                                            fmin=freq_bands["delta"][0],
                                            fmax=freq_bands["gamma"][1]
                                            ).get_data(return_freqs=True)

    ## mask to get each
    columns = []
    all_band_powers = []
    for band_name, (min_freq, max_freq) in freq_bands.items():
        band_mask = (freqs >= min_freq) & (freqs <= max_freq)
        band_powers = np.trapezoid(
            psd_chs[:, :, band_mask],
            freqs[band_mask],
            axis=-1
        )
        all_band_powers.append(band_powers)
        columns.extend([f"{ch}_{band_name}" for ch in ch_names])

    chs_power = np.concatenate(all_band_powers, axis=1)
    df_power = pd.DataFrame(chs_power, columns=columns)
    df_power.to_csv(features_dir / f"power_sensor_v{visit}.csv",index=False)

    print("Computing reliability metrics for sensor power ...")
    power_reliability = compute_reliability_metrics(
        chs_power,
        columns,
        feature_type="power_sensor"
    )
    power_reliability.to_csv(
        reliability_dir / f"power_sensor_v{visit}.csv", 
        index=False
    )
    # ============================================================================

    ## compute power in source
    inv_fname = subject_dir / "inv" / f"visit_{visit}-inv.fif"
    inverse_operator = read_inverse_operator(inv_fname)
    labels = read_labels_from_annot(subject="fsaverage", subjects_dir=None, parc="aparc")[:-1]
    stcs = apply_inverse_epochs(
                                epochs,
                                inverse_operator,
                                lambda2=1.0 / (1.0 ** 2),
                                method="dSPM",
                                label=None,
                                pick_ori="normal",
                                return_generator=False
                                )
    label_ts = extract_label_time_course(
                                        stcs,
                                        labels,
                                        inverse_operator["src"],
                                        mode="mean_flip",
                                        return_generator=False,
                                        )
    label_ts = np.array(label_ts)
    lb_names = [lb.name for lb in labels]

    n_epochs, n_labels, n_times = label_ts.shape
    reshaped_data = label_ts.reshape(-1, n_times)

    psd, freqs = psd_array_multitaper(
                                    reshaped_data,
                                    sfreq=epochs.info["sfreq"],
                                    fmin=freq_bands["delta"][0],
                                    fmax=freq_bands["gamma"][1]
                                    )
    columns = []
    labels_power = []
    for band_name, (min_freq, max_freq) in freq_bands.items():
        band_mask = (freqs >= min_freq) & (freqs <= max_freq)

        # integrate PSD over band
        band_powers = np.trapezoid(
                                psd[:, band_mask],
                                freqs[band_mask],
                                axis=-1
                            )
        labels_power.append(band_powers.reshape(n_epochs, n_labels))
        columns.extend([f"{lb.name}_{band_name}" for lb in labels])

    labels_power = np.concatenate(labels_power, axis=1)
    df_power = pd.DataFrame(labels_power, columns=columns)
    df_power.to_csv(features_dir / f"power_source_v{visit}.csv",index=False)

    # ============================================================================
    print("Computing reliability metrics for source power ...")
    source_power_reliability = compute_reliability_metrics(
        labels_power,
        columns,
        feature_type="power_source"
    )
    source_power_reliability.to_csv(
        reliability_dir / f"power_source_v{visit}.csv", 
        index=False
    )
    # ============================================================================

    ## compute connectivity in both sensor and source
    for mode in ["sensor", "source"]:
        if mode == "sensor":
            data_ts = epochs_ts
            names = ch_names
        elif mode == "source":
            data_ts = label_ts
            names = lb_names
            
        n_nodes = data_ts.shape[1]
        i_lower, j_lower = np.tril_indices(n_nodes, k=-1)

        columns = []
        freq_cons = []
        for key, value in freq_bands.items():
            n_cycles = value[1] / 6 if key == "delta" else 7

            for con_method in ["coh"]:
                print(f"Computing {con_method} connectivity values for {key} frange...")

                con = spectral_connectivity_time(
                    data_ts,
                    freqs=np.arange(value[0], value[1], 5),
                    method=con_method,
                    average=False,
                    sfreq=epochs.info["sfreq"],
                    mode="cwt_morlet",
                    fmin=value[0],
                    fmax=value[1],
                    faverage=True,
                    n_cycles=n_cycles,
                )

                con_matrix = np.squeeze(con.get_data(output="dense"))  # n_epochs × n_nodes × n_nodes
                cons = []
                for ep_con in con_matrix:
                    cons.append(ep_con[i_lower, j_lower])

                freq_cons.append(np.array(cons))

                columns += [
                    f"{names[i]}_vs_{names[j]}_{key}_{con_method}"
                    for i, j in zip(i_lower, j_lower)
                ]

        freq_cons = np.concatenate(freq_cons, axis=-1)
        df_conn = pd.DataFrame(freq_cons, columns=columns).T

        print("Saving connectivity values ...")
        df_conn.to_csv(features_dir / f"conn_{mode}_v{visit}.csv", index=False)

        print(f"Computing reliability metrics for {mode} connectivity ...")
        conn_reliability = compute_reliability_metrics(
            freq_cons,
            columns,
            feature_type=f"conn_{mode}"
        )
        conn_reliability.to_csv(
            reliability_dir / f"conn_{mode}_v{visit}.csv", 
            index=False
        )
        del df_conn

    # ============================================================================

    ## compute aperiodic features in both sensor and source
    for mode in ["sensor", "source"]:
        if mode == "sensor":
            data_ts = epochs_ts
            names = ch_names
        elif mode == "source":
            data_ts = label_ts
            names = lb_names

        print("Computing aperiodic values ...")
        fmin, fmax = 1, 40
        ep_psds, freqs = psd_array_multitaper(
            data_ts, epochs.info["sfreq"], fmin, fmax
        )
        avg_psd = ep_psds.mean(axis=0)
        fm = FOOOF()
        row_dict = {}

        for idx, name in enumerate(names):
            print(f"Processing {mode} {idx + 1} / {len(names)} ...")
            fm.fit(
                freqs=freqs,
                power_spectrum=avg_psd[idx],
                freq_range=[fmin, fmax],
            )
            offset, exponent = fm.aperiodic_params_
            row_dict[f"{name}_offset"] = offset
            row_dict[f"{name}_exponent"] = exponent

        df_aperiodic = pd.DataFrame([row_dict])

        print("Saving aperiodic values ...")
        df_aperiodic.to_csv(features_dir / f"aperiodic_{mode}_v{visit}.csv", index=False)
        del df_aperiodic


def compute_reliability_metrics(epoch_features, feature_names, feature_type=""):
    """
    Compute ICC, SNR, dynamic range, and stationarity for each feature.
    
    Parameters
    ----------
    epoch_features : ndarray, shape (n_epochs, n_features)
        Feature values across epochs
    feature_names : list of str
        Names of each feature
    feature_type : str
        Description of feature type (for logging)
    
    Returns
    -------
    df_reliability : pd.DataFrame
        Reliability metrics for each feature
    """
    
    n_epochs, n_features = epoch_features.shape
    
    # Split epochs in half for ICC computation
    half_point = n_epochs // 2
    first_half = epoch_features[:half_point, :]
    second_half = epoch_features[half_point:2*half_point, :]
    
    reliability_metrics = []
    
    for feat_idx, feat_name in enumerate(feature_names):
        feat_values = epoch_features[:, feat_idx]
        
        # 1. ICC (Intraclass Correlation) - split-half reliability
        # Using correlation between halves as proxy for ICC
        if len(first_half) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                icc_corr, _ = spearmanr(
                    first_half[:, feat_idx], 
                    second_half[:, feat_idx]
                )
            # Spearman-Brown correction for full-length reliability
            icc = (2 * icc_corr) / (1 + icc_corr) if not np.isnan(icc_corr) else 0
        else:
            icc = 0
        
        # 2. SNR (Signal-to-Noise Ratio)
        mean_val = np.mean(feat_values)
        std_val = np.std(feat_values)
        snr = np.abs(mean_val) / std_val if std_val > 0 else 0
        
        # 3. Dynamic Range (normalized)
        p10 = np.percentile(feat_values, 10)
        p90 = np.percentile(feat_values, 90)
        median_val = np.median(feat_values)
        dynamic_range = (p90 - p10) / np.abs(median_val) if median_val != 0 else 0
        
        # 4. Stationarity - using variance of rolling window variances
        # More stable variance across time = more stationary
        window_size = max(2, n_epochs // 5)  # ~5 windows
        rolling_vars = []
        for i in range(0, n_epochs - window_size + 1, max(1, window_size // 2)):
            window_data = feat_values[i:i + window_size]
            rolling_vars.append(np.var(window_data))
        
        if len(rolling_vars) > 1:
            # Coefficient of variation of variance
            cv_variance = np.std(rolling_vars) / np.mean(rolling_vars)
            stationarity = 1 / (1 + cv_variance)  # Higher = more stationary
        else:
            stationarity = 1.0
        
        reliability_metrics.append({
            'feature_name': feat_name,
            'feature_type': feature_type,
            'icc': icc,
            'snr': snr,
            'dynamic_range': dynamic_range,
            'stationarity': stationarity,
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'p10': p10,
            'p90': p90,
            'n_epochs': n_epochs
        })
    
    return pd.DataFrame(reliability_metrics)


def compute_dev_scores(subjects_dir, subject_id, age, sex, visit, models_dir):

    if sex == "m":
        sex_code = 1
    if sex == "f":
        sex_code = 0

    fname_audio = subjects_dir / subject_id / "audiometry" / f"audio.csv"
    df_audio = pd.read_csv(fname_audio)
    pta = df_audio["PTA4_mean"].values[0]
    freq_bands = {
                    "delta": [1, 6],
                    "theta": [6.5, 8.5],
                    "alpha_0": [8.5, 12.5],
                    "alpha_1": [8.5, 10.5],
                    "alpha_2": [10.5, 12.5],
                    "beta_0": [12.5, 30],
                    "beta_1": [12.5, 18.5],
                    "beta_2": [18.5, 21],
                    "beta_3": [21, 30],
                    "gamma": [30, 40]
                    }

    for mode in ["power", "conn"]:
        for space in ["sensor", "source"]:

            if mode == "conn":
                model_path = models_dir / space / "conn_coh" / "full_model"
            else:
                model_path = models_dir / space / mode / "full_model"
            
            if not model_path.exists():
                continue

            fname_feature = subjects_dir / subject_id / "features" / f"{mode}_{space}_v{visit}.csv"
            df_f = pd.read_csv(fname_feature)
            if mode == "conn":
                labels = read_labels_from_annot(subject="fsaverage", subjects_dir=None, parc="aparc")[:-1]
                lb_names = [lb.name for lb in labels]
                n_nodes = len(lb_names)
                i_lower, j_lower = np.tril_indices(n_nodes, k=-1)
                columns = []
                for key in freq_bands:
                    columns += [
                            f"{lb_names[i]}_vs_{lb_names[j]}_{key}_coh"
                            for i, j in zip(i_lower, j_lower)
                        ]
                
                df_f = df_f.T.copy()
                df_f.columns = columns
            else:
                df_f = df_f.mean(axis=0).to_frame().T

            df_f["SITE"] = "zuerich"
            df_f["subject_id"] = subject_id
            df_f["age"] = age
            df_f["sex"] = sex_code
            df_f["PTA4_mean"] = pta

            feature_cols = df_f.columns.to_list()[:-5]
            covar_cols = df_f.columns.to_list()[-3:]

            kwargs = {
                        "covariates": covar_cols,
                        "batch_effects": ["SITE"],
                        "response_vars": feature_cols, 
                        "subject_ids": "subject_id"
                        }
            norm_data = NormData.from_dataframe(
                                                    name="train",
                                                    dataframe=df_f,
                                                    **kwargs
                                                    )
            
            model = NormativeModel.load(str(model_path))
            data = model.compute_zscores(norm_data)
            zscores = data.Z.data
            my_dict = {"feature_name": feature_cols, "deviation": zscores[0]}
            df_z = pd.DataFrame(my_dict)

            fanme_reliability = subjects_dir / subject_id / "reliability_metrics" / f"{mode}_{space}_v{visit}.csv"
            df_r = pd.read_csv(fanme_reliability)
            df_r = df_r.merge(df_z, on="feature_name")
            df_r.to_csv(fanme_reliability, index=False)


if __name__ == "__main__":

    with open("config_master.yml", "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--age", required=True)
    parser.add_argument("--sex", required=True)
    parser.add_argument("--visit", required=True)
    args = parser.parse_args()

    subject_id = args.subject_id
    age = int(args.age)
    sex = args.sex
    visit = int(args.visit)
    subjects_dir = config.get("subjects_dir", "./")
    subjects_dir = Path(subjects_dir)
    models_dir = config.get("models_dir", "./")
    models_dir = Path(models_dir)

    epochs = preprocess_baseline(subject_id, visit, subjects_dir)
    extract_features(epochs, subject_id, visit, subjects_dir)
    compute_dev_scores(subjects_dir, subject_id, age, sex, visit, models_dir)