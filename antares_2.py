import yaml
import logging
from pathlib import Path
import time
import argparse
from ant import NFRealtime

if __name__ == "__main__":

    with open("config_master.yml", "r") as f:
        config = yaml.safe_load(f)

    ## now connect to stream and record RS EEG
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--visit", required=True)
    args = parser.parse_args()

    subject_id = args.subject_id
    visit = int(args.visit)
    subjects_dir = config.get("subjects_dir", "./")
    log_dir = Path(subjects_dir) / subject_id / "logs"
    baseline_duration = config.get("baseline_duration")

    log_file = log_dir / f"v_{visit}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.info(f"Baseline recording initiated ...")
    kwargs = {
            "subject_id": subject_id,
            "visit": visit,
            "subjects_dir": Path(subjects_dir),
            "montage": "easycap-M1",
            "mri": False,
            "artifact_correction": False,
            "verbose": False
            }
    nf = NFRealtime(session="baseline", **kwargs)
    
    # Connect to a mock LSL stream (we are using our simulated data)
    fname = "/Users/payamsadeghishabestari/ANT/data/simulated/pericalcarine-lh_10_2-raw.fif"
    nf.connect_to_lsl(mock_lsl=True, fname=fname)
    logging.info(f"Connected to LSL stream ...")
    time.sleep(4)
    nf.record_baseline(baseline_duration=baseline_duration)
    logging.info(f"Baseline recording finished ...")