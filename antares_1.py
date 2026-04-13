from pathlib import Path
import logging
import yaml
import sys

from scipy.io import loadmat
import numpy as np
import pandas as pd

def clean_input(prompt):
    sys.stderr.write(prompt)
    sys.stderr.flush()
    return sys.stdin.readline().strip()

def get_user_info(config):

    while True:
        subject_id = clean_input("Enter subject ID: ").strip().lower()
        if subject_id.isalpha() and len(subject_id) == 4:
            break
        print("Invalid ID. Try again.\n")

    while True:
        age_input = clean_input("Enter age (integer, e.g., 25): ").strip()
        try:
            age = int(age_input)
            break
        except ValueError:
            print("Invalid age.\n")

    while True:
        sex = clean_input("Enter sex ('f' or 'm'): ").strip().lower()
        if sex in ["f", "m"]:
            break
        print("Invalid sex input.\n")
    
    while True:
        visit_input = clean_input("Enter visit number: ").strip()
        try:
            visit = int(visit_input)
            break
        except ValueError:
            print("Invalid visit number.\n")

    # configure logging
    subjects_dir = config.get("subjects_dir", "./")
    log_dir = Path(subjects_dir) / subject_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"v_{visit}.log"
    if log_file.exists():
        raise ValueError(f"The log file for visit {visit} already exists.")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.info(f"Collected subject data: subject_id={subject_id}, age={age}, sex={sex}, visit={visit}")

    return subject_id, age, sex, visit


def compute_pta(subject_id, config):

    subjects_dir = config.get("subjects_dir", "./")
    log_dir = Path(subjects_dir) / subject_id / "logs"
    log_file = log_dir / "antares.log"
    audiometry_dir = config.get("audiometry_dir", "./")

    # configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logging.info("Checking and Computing PTA")
    audiometry_dir = Path(audiometry_dir)
    subject_audio_dir = audiometry_dir / subject_id

    # The frequencies you want as columns
    target_freqs = [125, 250, 500, 1000, 2000, 4000, 6000, 8000, 12000]
    freq_cols = []
    for hemi in ["L", "R"]:
        for f in target_freqs:
            freq_cols.append(f"A{hemi}E_{f}")

    row = {"subject": subject_id}
    for col in freq_cols:
        row[col] = np.nan

    # check if subject has audiometry data
    if subject_audio_dir.exists():
        for fname in subject_audio_dir.iterdir():
            for hemi in ["L", "R"]:
                if fname.name.lower().startswith(f"{subject_id.lower()} {hemi.lower()}"):
                    logging.info(f"Subject audiometry data is found for {hemi} ear.")

                    data = loadmat(fname)
                    freqs = data["betweenRuns"]["var1Sequence"][0][0][0]
                    thrs  = data["betweenRuns"]["thresholds"][0][0][0]

                    # Sort
                    order = np.argsort(freqs)
                    freqs_sorted = freqs[order]
                    thrs_sorted  = thrs[order]

                    for f, thr in zip(freqs_sorted, thrs_sorted):
                        f_int = int(f)
                        col_name = f"A{hemi}E_{f_int}"
                        if col_name in row:
                            row[col_name] = thr
    else:
        raise ValueError(f"subject {subject_id} does not have audiometry data in this path: {subject_audio_dir}")

    df = pd.DataFrame([row])

    ## fix calibration if needed
    calibration = False
    if calibration:
        hemis = ["L", "R"]
        for hemi in hemis:
            df[f"A{hemi}E_250"] = df[f"A{hemi}E_250"] - 8
            df[f"A{hemi}E_2000"] = df[f"A{hemi}E_2000"] + 3

    ## compute PTA
    freqs = [500, 1000, 2000, 4000]
    cols_l = [f"ALE_{f}" for f in freqs]
    cols_r = [f"ARE_{f}" for f in freqs]
    df["PTA4_L"] = df[cols_l].mean(axis=1)
    df["PTA4_R"] = df[cols_r].mean(axis=1)
    df["PTA4_mean"] = df[["PTA4_L", "PTA4_R"]].mean(axis=1)
    logging.info(f"PTA is computed and the value is: {df['PTA4_mean'].values[0]} dB")

    audio_dir = Path(subjects_dir) / subject_id / "audiometry"
    audio_dir.mkdir(exist_ok=True)
    df.to_csv(audio_dir / "audio.csv", index=False)
    logging.info(f"Audiometry information is saved.")
    return df["PTA4_mean"].values


if __name__ == "__main__":

    # Load config
    with open("config_master.yml", "r") as f:
        config = yaml.safe_load(f)

    subject_id, age, sex, visit = get_user_info(config)
    print(f"{subject_id},{age},{sex},{visit}")
    pta = compute_pta(subject_id, config)