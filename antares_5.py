from pathlib import Path
import logging
import argparse
import yaml
import time
import shutil
import json
import re
from ant import NFRealtime

def parse_feature_name(feature_name, feature_type):
    """
    Parse feature name to extract relevant parameters for YAML.
    
    Parameters
    ----------
    feature_name : str
        Feature name like "C3_alpha_0" or "transversetemporal-lh_vs_transversetemporal-rh_alpha_0_coh"
    feature_type : str
        Feature type like "power_sensor", "conn_source", "aperiodic_sensor", etc.
    
    Returns
    -------
    dict with parsed parameters
    """
    params = {}
    
    # Extract frequency band if present
    band_map = {
        'delta': [1, 6],
        'theta': [6.5, 8.5],
        'alpha_0': [8.5, 12.5],
        'alpha_1': [8.5, 10.5],
        'alpha_2': [10.5, 12.5],
        'beta_0': [12.5, 30],
        'beta_1': [12.5, 18.5],
        'beta_2': [18.5, 21],
        'beta_3': [21, 30],
        'gamma': [30, 40]
    }
    

    for band_name, frange in band_map.items():
        if band_name in feature_name:
            params['band'] = band_name
            params['frange'] = frange
            break
    
    if feature_type == "power_sensor":
        parts = feature_name.split('_')
        params['channel'] = parts[0]
        params['modality'] = 'sensor_power'
        
    elif feature_type == "power_source":
        parts = feature_name.split('_')
        params['brain_label'] = parts[0]
        params['modality'] = 'source_power'
        
    elif feature_type == "conn_sensor":
        match = re.match(r'(.+?)_vs_(.+?)_(.+?)_(.+)', feature_name)
        if match:
            params['channel_1'] = match.group(1)
            params['channel_2'] = match.group(2)
            params['method'] = match.group(4)
            params['modality'] = 'sensor_connectivity'
            
    elif feature_type == "conn_source":
        match = re.match(r'(.+?)_vs_(.+?)_(.+?)_(.+)', feature_name)
        if match:
            params['brain_label_1'] = match.group(1)
            params['brain_label_2'] = match.group(2)
            params['method'] = match.group(4)
            params['modality'] = 'source_connectivity'
            
    elif feature_type in ["aperiodic_sensor", "aperiodic_source"]:
        parts = feature_name.split('_')
        aperiodic_param = parts[-1]  # 'exponent' or 'offset'
        node_name = '_'.join(parts[:-1])  # everything before last underscore
        
        params['aperiodic_param'] = aperiodic_param
        if feature_type == "aperiodic_sensor":
            params['channel'] = node_name
            params['modality'] = 'sensor_power'  # Closest match in YAML
        else:
            params['brain_label'] = node_name
            params['modality'] = 'source_power'
    
    return params


def create_nf_yaml_from_selection(
                                subject_id,
                                visit,
                                subjects_dir, 
                                template_yaml_path,
                                output_yaml_path=None
                                ):
    """
    Create a personalized NF YAML file based on selected feature.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    visit : int
        Visit number
    subjects_dir : Path or str
        Root directory for subjects
    template_yaml_path : Path or str
        Path to the template YAML file
    output_yaml_path : Path or str, optional
        Output path for the new YAML. If None, creates temp file in subject dir
    
    Returns
    -------
    Path to created YAML file
    """
    subjects_dir = Path(subjects_dir)
    log_dir = Path(subjects_dir) / subject_id / "logs"
    log_file = log_dir / f"v_{visit}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    template_yaml_path = Path(template_yaml_path)
    json_path = subjects_dir / subject_id / "nf_selection" / f"selection_report_v{visit}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Selection report not found: {json_path}")
    
    with open(json_path, 'r') as f:
        report = json.load(f)
    
    feature_name = report['selected_feature']['feature_name']
    feature_type = report['selected_feature']['feature_type']
    
    # Parse feature to get parameters
    params = parse_feature_name(feature_name, feature_type)
    with open(template_yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    nf_config = {'NF_modality': {}}
    modality = params.get('modality')
    if modality == 'sensor_power':
        nf_config['NF_modality']['sensor_power'] = {
            'frange': params.get('frange', [8, 13]),
            'method': 'welch',
            'relative': False,
            'selected_channel': params.get('channel', None),
            'selected_feature': feature_name
        }
        
    elif modality == 'source_power':
        nf_config['NF_modality']['source_power'] = {
            'frange': params.get('frange', [8, 13]),
            'brain_label': params.get('brain_label', 'pericalcarine-lh'),
            'atlas': 'aparc',
            'method': 'dSPM',
            'selected_feature': feature_name
        }
        
    elif modality == 'sensor_connectivity':
        ch1 = params.get('channel_1')
        ch2 = params.get('channel_2')
        nf_config['NF_modality']['sensor_connectivity'] = {
            'frange': params.get('frange', [8, 13]),
            'channels': [[ch1, ch2]],
            'method': params.get('method', 'coh'),
            'mode': 'cwt_morlet',
            'selected_feature': feature_name
        }
        
    elif modality == 'source_connectivity':
        nf_config['NF_modality']['source_connectivity'] = {
            'frange': params.get('frange', [8, 13]),
            'brain_label_1': params.get('brain_label_1', 'transversetemporal-lh'),
            'brain_label_2': params.get('brain_label_2', 'transversetemporal-rh'),
            'atlas': 'aparc',
            'method': params.get('method', 'coh'),
            'mode': 'cwt_morlet',
            'selected_feature': feature_name
        }
    
    else:
        print(f"Unknown modality for {feature_type}, using template defaults")
        nf_config = yaml_config
    

    if output_yaml_path is None:
        output_yaml_path = subjects_dir / subject_id / "nf_selection" / f"nf_config_v{visit}.yml"
    else:
        output_yaml_path = Path(output_yaml_path)
    
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    class FlowStyleList(list):
        pass
    
    def represent_flow_style_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    
    yaml.add_representer(FlowStyleList, represent_flow_style_list)
    
    # Convert all lists to FlowStyleList
    def convert_lists_to_flow_style(obj):
        if isinstance(obj, dict):
            return {k: convert_lists_to_flow_style(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return FlowStyleList(convert_lists_to_flow_style(item) for item in obj)
        else:
            return obj
    
    nf_config = convert_lists_to_flow_style(nf_config)
    
    # Save YAML
    with open(output_yaml_path, 'w') as f:
        yaml.dump(nf_config, f, default_flow_style=False, sort_keys=False)
    
    logging.info(f"Created personalized NF config: {output_yaml_path}")
    return output_yaml_path



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
main_duration = config.get("main_duration")
template_yaml = "config_methods.yml"
yaml_path = create_nf_yaml_from_selection(
                                        subject_id,
                                        visit,
                                        subjects_dir,
                                        template_yaml
                                        )
with open(yaml_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

modality = list(yaml_data["NF_modality"].keys())[0]
if modality == "sensor_power":
    picks = yaml_data["NF_modality"][modality] # should be fixed !!!


subjects_dir = Path(subjects_dir)
subject_dir = subjects_dir / subject_id

kwargs = {
        "subject_id": subject_id,
        "visit": visit,
        "subjects_dir": subjects_dir,
        "montage": "easycap-M1",
        "mri": False,
        "artifact_correction": False,
        "config_file": str(yaml_path),
        "verbose": False
        }
nf = NFRealtime(session="main", **kwargs)
logging.info(f"Neurofeedback module fro the main recording initiated.")

fname = "/Users/payamsadeghishabestari/ANT/data/simulated/pericalcarine-lh_10_2-raw.fif"
nf.connect_to_lsl(mock_lsl=True, fname=fname)
logging.info(f"Connected to the LSL stream ...")
time.sleep(1)

nf.record_main(
    duration=main_duration,
    modality=modality,
    picks=None,
    winsize=1,
    estimate_delays=True,
    modality_params=None,
    show_raw_signal=False,
    show_nf_signal=True,
    time_window=20,
    show_design_viz=False,
    design_viz="VisualRorschach",
    show_brain_activation=False
)
nf.save(nf_data=True, acq_delay=True, raw_data=False)
logging.info(f"Main recording finished and data are saved.")

## clean and rename folders at the end
report_dir = subject_dir / "reports"
nf_data_dir = subject_dir / "neurofeedback"
new_nf_data_path = nf_data_dir.with_name("nf_data")

if report_dir.exists() and report_dir.is_dir():
    shutil.rmtree(report_dir)

if new_nf_data_path.exists() and new_nf_data_path.is_dir():
    shutil.rmtree(new_nf_data_path)

if nf_data_dir.exists():
    nf_data_dir.rename(new_nf_data_path)

logging.info(f"Subject directory cleaned and restructured.")

'''
from smbprotocol.connection import Connection
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect
from smbprotocol.open import Open

server = "IDNAS32"
username = "your_username"
password = "your_password"

conn = Connection(uuid="random-id", server=server, port=445)
conn.connect()

session = Session(conn, username=username, password=password)
session.connect()

tree = TreeConnect(session, r"\\IDNAS32\G_USZ_ORL$")
tree.connect()

# Example: open root directory
dir_open = Open(tree, "")
dir_open.create()

for info in dir_open.query_directory("*"):
    print(info.file_name)
'''