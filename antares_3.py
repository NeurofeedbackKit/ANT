from pathlib import Path
import logging
import yaml
import argparse
import warnings
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

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
    log_dir = Path(subjects_dir) / subject_id / "logs"
    log_file = log_dir / f"v_{visit}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.info(f"Preprocessing of baseline recording initiated ...")


    fname = subject_dir / "baseline" / f"visit_{visit}-raw.fif"
    raw = read_raw_fif(fname, preload=True)
    raw.filter(0.1, 40)
    logging.info(f"Recording bandpass filtered between 0.1 and 40 Hz.")

    raw.set_eeg_reference('average', projection=True)
    montage_name = "easycap-M1"
    montage = make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="warn")
    logging.info(f"Recording average re-referenced and easycap-M1 montage is set on channels.")

    ## preprocess
    epochs = make_fixed_length_epochs(raw, duration=10, preload=True)
    logging.info(f"Recording is splitted into 10 second epochs.")
    logging.info(f"Autoreject initiated ...")
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
    logging.info(f"Autoreject finished and epochs are saved ...")

    ## create report
    report = Report(title=f"{subject_id}_visit_{visit}")
    fig_reject = reject_log.plot(show=False)
    report.add_figure(fig=fig_reject, title="autoreject log", image_format="PNG")
    report.save(fname=subject_dir / "baseline" / f"report_v{visit}.html", overwrite=True, open_browser=False)
    logging.info(f"Report of autoreject process is created.")

    return epochs


def extract_features(epochs, subject_id, visit, subjects_dir):

    ## path adjusting
    subject_dir = subjects_dir / subject_id
    features_dir = subject_dir / "features"
    features_dir.mkdir(exist_ok=True)
    log_dir = Path(subjects_dir) / subject_id / "logs"
    log_file = log_dir / f"v_{visit}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.info(f"Feature extraction initiated ...")

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

    logging.info(f"Computing band powers in sensor space ...")
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
    logging.info(f"Computing band powers in sensor space finished.")
    
    logging.info(f"Computing reliability metrics for sensor power ...")
    power_reliability = compute_reliability_metrics(
        chs_power,
        columns,
        feature_type="power_sensor"
    )
    power_reliability.to_csv(
        reliability_dir / f"power_sensor_v{visit}.csv", 
        index=False
    )
    logging.info(f"Reliability metrics for sensor power are computed.")

    ## compute power in source
    logging.info(f"Computing band powers in source space ...")
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
    logging.info(f"Computing band powers in source space finished.")

    logging.info(f"Computing reliability metrics for sensor power ...")
    source_power_reliability = compute_reliability_metrics(
        labels_power,
        columns,
        feature_type="power_source"
    )
    source_power_reliability.to_csv(
        reliability_dir / f"power_source_v{visit}.csv", 
        index=False
    )
    logging.info(f"Reliability metrics for source power are computed.")


    ## compute connectivity in both sensor and source
    
    for mode in ["sensor", "source"]:
        logging.info(f"Computing connectivity features in {mode} space ...")
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
        df_conn.to_csv(features_dir / f"conn_{mode}_v{visit}.csv", index=False)
        logging.info(f"Connectivity features in {mode} space are computed.")

        logging.info(f"Computing reliability metrics for {mode} connectivity ...")
        conn_reliability = compute_reliability_metrics(
            freq_cons,
            columns,
            feature_type=f"conn_{mode}"
        )
        conn_reliability.to_csv(
            reliability_dir / f"conn_{mode}_v{visit}.csv", 
            index=False
        )
        logging.info(f"Reliability metrics for {mode} connectivity are computed.")
        del df_conn

    # ============================================================================

    ## compute aperiodic features in both sensor and source
    for mode in ["sensor", "source"]:
        logging.info(f"Computing aperiodic features in {mode} space ...")
        if mode == "sensor":
            data_ts = epochs_ts
            names = ch_names
        elif mode == "source":
            data_ts = label_ts
            names = lb_names

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
        df_aperiodic.to_csv(features_dir / f"aperiodic_{mode}_v{visit}.csv", index=False)
        logging.info(f"aperiodic features in {mode} space are computed.")
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

    log_dir = Path(subjects_dir) / subject_id / "logs"
    log_file = log_dir / f"v_{visit}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

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
            logging.info(f"Computing deviation scores for {mode} modality in {space} space ...")
            
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


def rank_features(subject_id, subjects_dir, visit):

    subject_dir = subjects_dir / subject_id
    log_dir = Path(subjects_dir) / subject_id / "logs"
    log_file = log_dir / f"v_{visit}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    
    reliability_dir = subject_dir / "reliability_metrics"
    reliability_files = list(reliability_dir.glob(f"*_v{visit}.csv"))

    all_rankings = []
    for rel_file in reliability_files:
        df_r = pd.read_csv(rel_file)
        if not "deviation" in df_r.columns.tolist():
            continue

        for idx, row in df_r.iterrows():
            feat_name = row['feature_name']
            z_score = row['deviation']
            icc = row['icc']
            snr = row['snr']
            dynamic_range = row['dynamic_range']
            stationarity = row['stationarity']
            
            # Apply thresholds
            meets_criteria = (
                z_score >= 1.5 and  
                icc >= 0.6 and      
                snr >= 2.0 and      
                dynamic_range >= 0.3
            )
            
            # Compute composite score (weighted)
            # Normalize each metric to 0-1 range for fair weighting
            norm_z = min(z_score / 3.0, 1.0)  
            norm_icc = min(max(icc, 0), 1.0)
            norm_snr = min(snr / 10.0, 1.0)  
            norm_dr = min(dynamic_range / 2.0, 1.0)
            norm_stat = min(stationarity, 1.0)
            
            composite_score = (
                0.30 * norm_z +           # Clinical relevance
                0.25 * norm_icc +         # Reliability
                0.20 * norm_snr +         # Real-time detectability
                0.15 * norm_dr +          # Trainability
                0.10 * norm_stat          # Stability
            )
            
            all_rankings.append({
                'feature_name': feat_name,
                'feature_type': row['feature_type'],
                'z_score': z_score,
                'icc': icc,
                'snr': snr,
                'dynamic_range': dynamic_range,
                'stationarity': stationarity,
                'composite_score': composite_score,
                'meets_criteria': meets_criteria
            })

    df_ranked = pd.DataFrame(all_rankings)
    df_ranked = df_ranked.sort_values('composite_score', ascending=False)
    df_ranked['rank'] = range(1, len(df_ranked) + 1)

    # Select top feature
    viable_features = df_ranked[df_ranked['meets_criteria'] == True]

    if len(viable_features) > 0:
        selected_feature = viable_features.iloc[0].to_dict()
        backup_features = viable_features.iloc[1:4].to_dict('records')
        selection_status = "success"
        selection_message = f"Selected {selected_feature['feature_name']} (composite score: {selected_feature['composite_score']:.3f})"
    else:
        # Relaxed selection: just pick highest composite score
        selected_feature = df_ranked.iloc[0].to_dict()
        backup_features = df_ranked.iloc[1:4].to_dict('records')
        selection_status = "relaxed_criteria"
        selection_message = f"No features met all criteria. Selected best available: {selected_feature['feature_name']}"

    # Save ranking table
    output_dir = subject_dir / "nf_selection"
    output_dir.mkdir(exist_ok=True)

    df_ranked.to_csv(
        output_dir / f"feature_ranking_v{visit}.csv",
        index=False
    )

    # Create report dictionary
    report_dict = {
        'subject_id': subject_id,
        'visit': visit,
        'selection_status': selection_status,
        'selection_message': selection_message,
        'selected_feature': selected_feature,
        'backup_features': backup_features,
        'n_features_evaluated': len(df_ranked),
        'n_features_meeting_criteria': len(viable_features),
        'ranking_table': df_ranked
    }

    # Save JSON report
    json_report = {k: v for k, v in report_dict.items() if k != 'ranking_table'}
    json_report['selected_feature'] = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in selected_feature.items()
    }

    with open(output_dir / f"selection_report_v{visit}.json", 'w') as f:
        json.dump(json_report, f, indent=2)

    logging.info(f"Composite score is computed per feature, sorted and saved to find the best feature(s).")
    return report_dict


def generate_selection_report_html(report_dict, subjects_dir):
    """
    Generate an HTML report for feature selection.
    """
    
    subject_id = report_dict['subject_id']
    visit = report_dict['visit']
    subject_dir = subjects_dir / subject_id

    log_dir = Path(subjects_dir) / subject_id / "logs"
    log_file = log_dir / f"v_{visit}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    output_dir = subject_dir / "nf_selection"
    
    selected = report_dict['selected_feature']
    backups = report_dict['backup_features']
    df_ranked = report_dict['ranking_table']
    
    # Create visualizations
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Feature Selection Report: {subject_id} - Visit {visit}", 
                fontsize=16, fontweight='bold')
    
    # 1. Top 20 features by composite score
    ax = axes[0, 0]
    top_20 = df_ranked.head(20)
    colors = ['#2ecc71' if meets else '#e74c3c' 
                for meets in top_20['meets_criteria']]
    ax.barh(range(len(top_20)), top_20['composite_score'], color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature_name'], fontsize=8)
    ax.set_xlabel('Composite Score')
    ax.set_title('Top 20 Features by Composite Score')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.invert_yaxis()
    
    # Add legend

    legend_elements = [
        Patch(facecolor='#2ecc71', label='Meets all criteria'),
        Patch(facecolor='#e74c3c', label='Below threshold')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # 2. Metric breakdown for selected feature
    ax = axes[0, 1]
    metrics = ['z_score', 'icc', 'snr', 'dynamic_range', 'stationarity']
    values = [selected[m] for m in metrics]
    thresholds = [1.5, 0.6, 2.0, 0.3, 0.7]
    
    x = np.arange(len(metrics))
    bars = ax.bar(x, values, color='#3498db', alpha=0.7)
    ax.scatter(x, thresholds, color='red', s=100, marker='_', linewidths=3, 
                label='Threshold', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(['Z-score', 'ICC', 'SNR', 'Dyn. Range', 'Stationarity'], 
                        rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(f'Selected Feature Metrics\n{selected["feature_name"]}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Distribution of metrics across all features
    ax = axes[1, 0]
    metric_to_plot = 'composite_score'
    ax.hist(df_ranked[metric_to_plot], bins=30, color='#95a5a6', alpha=0.7, 
            edgecolor='black')
    ax.axvline(selected[metric_to_plot], color='#2ecc71', linewidth=3, 
            label='Selected feature', linestyle='--')
    if len(backups) > 0:
        for backup in backups:
            ax.axvline(backup[metric_to_plot], color='#f39c12', linewidth=2, 
                    alpha=0.6, linestyle=':')
    ax.set_xlabel('Composite Score')
    ax.set_ylabel('Number of Features')
    ax.set_title('Distribution of Composite Scores')
    ax.legend()
    
    # 4. Scatter: ICC vs SNR (colored by composite score)
    ax = axes[1, 1]
    scatter = ax.scatter(df_ranked['icc'], df_ranked['snr'], 
                        c=df_ranked['composite_score'], 
                        cmap='viridis', s=50, alpha=0.6)
    ax.scatter(selected['icc'], selected['snr'], 
            color='#2ecc71', s=300, marker='*', 
            edgecolor='black', linewidth=2, label='Selected', zorder=5)
    
    # Mark thresholds
    ax.axhline(2.0, color='red', linestyle='--', alpha=0.3, label='SNR threshold')
    ax.axvline(0.6, color='red', linestyle='--', alpha=0.3, label='ICC threshold')
    
    ax.set_xlabel('ICC (Reliability)')
    ax.set_ylabel('SNR')
    ax.set_title('Feature Space: ICC vs SNR')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Composite Score')
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f"selection_report_v{visit}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Report of the computed scores is created and saved.")
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neurofeedback Feature Selection - {subject_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 40px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .status-success {{
                background-color: #2ecc71;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .status-relaxed {{
                background-color: #f39c12;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .feature-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px 10px 0;
            }}
            .metric-label {{
                font-weight: bold;
                color: #555;
            }}
            .metric-value {{
                font-size: 1.3em;
                color: #2c3e50;
            }}
            .good {{ color: #2ecc71; }}
            .warning {{ color: #f39c12; }}
            .bad {{ color: #e74c3c; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                margin-top: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #667eea;
                color: white;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            img {{
                width: 100%;
                border-radius: 10px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Neurofeedback Feature Selection Report</h1>
            <h2>Subject: {subject_id} | Visit: {visit}</h2>
        </div>
        
        <div class="{'status-success' if report_dict['selection_status'] == 'success' else 'status-relaxed'}">
            <strong>Status:</strong> {report_dict['selection_message']}
        </div>
        
        <div class="feature-card">
            <h2> Selected Feature</h2>
            <h3>{selected['feature_name']}</h3>
            <p><em>Type: {selected['feature_type']}</em></p>
            
            <div class="metric">
                <span class="metric-label">Composite Score:</span>
                <span class="metric-value {'good' if selected['composite_score'] > 0.6 else 'warning' if selected['composite_score'] > 0.4 else 'bad'}">
                    {selected['composite_score']:.3f}
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Z-Score (deviation):</span>
                <span class="metric-value {'good' if selected['z_score'] >= 1.5 else 'bad'}">
                    {selected['z_score']:.2f}
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">ICC (reliability):</span>
                <span class="metric-value {'good' if selected['icc'] >= 0.6 else 'bad'}">
                    {selected['icc']:.3f}
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">SNR:</span>
                <span class="metric-value {'good' if selected['snr'] >= 2.0 else 'bad'}">
                    {selected['snr']:.2f}
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Dynamic Range:</span>
                <span class="metric-value {'good' if selected['dynamic_range'] >= 0.3 else 'bad'}">
                    {selected['dynamic_range']:.2f}
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Stationarity:</span>
                <span class="metric-value">
                    {selected['stationarity']:.3f}
                </span>
            </div>
            
            <hr>
            
        </div>
    """
    
    # Add backup features
    if len(backups) > 0:
        html_content += """
        <div class="feature-card">
            <h2> Backup Features</h2>
            <p>Alternative targets in case primary feature is not trainable:</p>
        """
        
        for i, backup in enumerate(backups, 1):
            html_content += f"""
            <div style="margin: 15px 0; padding: 10px; background: #ecf0f1; border-radius: 5px;">
                <strong>{i}. {backup['feature_name']}</strong> 
                (Composite: {backup['composite_score']:.3f}, 
                Z-score: {backup['z_score']:.2f}, 
                ICC: {backup['icc']:.3f}, 
                SNR: {backup['snr']:.2f})
            </div>
            """
        
        html_content += "</div>"
    
    # Add visualization
    html_content += f"""
        <div class="feature-card">
            <h2> Visual Analysis</h2>
            <img src="selection_report_v{visit}.png" alt="Feature selection visualizations">
        </div>
    """
    
    # Add top 10 table
    html_content += """
        <div class="feature-card">
            <h2> Top 10 Ranked Features</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature Name</th>
                        <th>Type</th>
                        <th>Composite</th>
                        <th>Z-Score</th>
                        <th>ICC</th>
                        <th>SNR</th>
                        <th>Dyn. Range</th>
                        <th>Meets Criteria</th>
                    </tr>
                </thead>
                <tbody>
        """
    
    for _, row in df_ranked.head(10).iterrows():
        meets_icon = "True" if row['meets_criteria'] else "False"
        html_content += f"""
                    <tr>
                        <td>{int(row['rank'])}</td>
                        <td><strong>{row['feature_name']}</strong></td>
                        <td>{row['feature_type']}</td>
                        <td>{row['composite_score']:.3f}</td>
                        <td>{row['z_score']:.2f}</td>
                        <td>{row['icc']:.3f}</td>
                        <td>{row['snr']:.2f}</td>
                        <td>{row['dynamic_range']:.2f}</td>
                        <td>{meets_icon}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="feature-card">
            <h2> Interpretation Guide</h2>
            <ul>
                <li><strong>Composite Score:</strong> Weighted combination of all metrics (higher is better)</li>
                <li><strong>Z-Score:</strong> Deviation from normative data (> 1.5 recommended)</li>
                <li><strong>ICC:</strong> Split-half reliability (> 0.6 recommended)</li>
                <li><strong>SNR:</strong> Signal-to-noise ratio (> 2.0 recommended)</li>
                <li><strong>Dynamic Range:</strong> Modulation potential (> 0.3 recommended)</li>
                <li><strong>Stationarity:</strong> Temporal stability (higher is more stable)</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(output_dir / f"selection_report_v{visit}.html", 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to: {output_dir / f'selection_report_v{visit}.html'}")
    return output_dir / f"selection_report_v{visit}.html"


def track_feature_across_sessions(subject_id, subjects_dir, current_visit):
    """
    Compare selected feature stability across sessions.
    Decide whether to continue with same feature or switch.
    """
    subject_dir = subjects_dir / subject_id
    selection_dir = subject_dir / "nf_selection"
    
    # Load all previous session reports
    session_reports = []
    for v in range(1, current_visit + 1):
        json_path = selection_dir / f"selection_report_v{v}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                session_reports.append(json.load(f))
    
    if len(session_reports) < 2:
        return None  # Need at least 2 sessions to compare
    
    # Get primary feature from session 1
    primary_feature_name = session_reports[0]['selected_feature']['feature_name']
    
    # Track this feature's metrics across sessions
    tracking_df = []
    for report in session_reports:
        visit_num = report['visit']
        
        # Load full ranking for this visit
        ranking_path = selection_dir / f"feature_ranking_v{visit_num}.csv"
        df_rank = pd.read_csv(ranking_path)
        
        # Find our primary feature
        primary_row = df_rank[df_rank['feature_name'] == primary_feature_name]
        
        if len(primary_row) > 0:
            tracking_df.append({
                'visit': visit_num,
                'feature_name': primary_feature_name,
                **primary_row.iloc[0].to_dict()
            })
    
    df_tracking = pd.DataFrame(tracking_df)
    
    # Compute stability
    icc_stability = df_tracking['icc'].std() < 0.15  # Low variation
    snr_stability = df_tracking['snr'].std() < 1.0
    
    # Decision logic
    current_metrics = df_tracking.iloc[-1]
    
    if current_metrics['meets_criteria'] and (icc_stability or snr_stability):
        decision = "MAINTAIN"
        message = f"Continue training {primary_feature_name} - stable and reliable"
    elif current_metrics['snr'] < 1.5:
        decision = "SWITCH"
        message = f"SNR dropped too low ({current_metrics['snr']:.2f}) - consider switching"
    else:
        decision = "MAINTAIN"
        message = f"Continue with {primary_feature_name} despite some variability"
    
    # Save tracking report
    df_tracking.to_csv(
        selection_dir / f"feature_tracking_through_v{current_visit}.csv",
        index=False
    )
    
    # Create tracking visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Feature Tracking: {primary_feature_name} - Visits 1-{current_visit}", 
                fontsize=14, fontweight='bold')
    
    visits = df_tracking['visit'].values
    
    axes[0, 0].plot(visits, df_tracking['icc'], marker='o', linewidth=2, markersize=8)
    axes[0, 0].axhline(0.6, color='red', linestyle='--', label='Threshold')
    axes[0, 0].set_ylabel('ICC')
    axes[0, 0].set_title('Reliability Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(visits, df_tracking['snr'], marker='o', linewidth=2, markersize=8, color='orange')
    axes[0, 1].axhline(2.0, color='red', linestyle='--', label='Threshold')
    axes[0, 1].set_ylabel('SNR')
    axes[0, 1].set_title('Signal-to-Noise Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(visits, df_tracking['z_score'], marker='o', linewidth=2, markersize=8, color='green')
    axes[1, 0].axhline(1.5, color='red', linestyle='--', label='Threshold')
    axes[1, 0].set_ylabel('Z-Score')
    axes[1, 0].set_xlabel('Visit')
    axes[1, 0].set_title('Deviation from Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(visits, df_tracking['composite_score'], marker='o', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_ylabel('Composite Score')
    axes[1, 1].set_xlabel('Visit')
    axes[1, 1].set_title('Overall Quality Score')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(selection_dir / f"feature_tracking_v{current_visit}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'decision': decision,
        'message': message,
        'tracking_data': df_tracking,
        'primary_feature': primary_feature_name
    }


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
    report_dict = rank_features(subject_id, subjects_dir, visit)
    generate_selection_report_html(report_dict, subjects_dir)
    
    if int(visit) > 1:
        track_feature_across_sessions(subject_id, subjects_dir, visit)