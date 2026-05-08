<p align="center">
  <img src="docs/source/_static/ANT_Logo_Horizontal.svg" alt="ANT Logo" width="480"/>
</p>

<p align="center">
  <strong>Advanced Neurofeedback Toolbox</strong><br>
  Real-time M/EEG neurofeedback for research and clinical use
</p>

<p align="center">
  <a href="https://pypi.org/project/ANT/"><img alt="PyPI" src="https://img.shields.io/pypi/v/ANT?color=blue&logo=pypi&logoColor=white"></a>
  <a href="https://pypi.org/project/ANT/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/ANT?logo=python&logoColor=white"></a>
  <a href="https://github.com/payamsash/ANT/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/payamsash/ANT?color=green"></a>
  <a href="https://payamsash.github.io/ANT/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-brightgreen?logo=readthedocs&logoColor=white"></a>
</p>

---

**ANT** is an open-source Python package for **real-time closed-loop M/EEG neurofeedback**.
Built on [MNE-Python](https://mne.tools) and the [Lab Streaming Layer (LSL)](https://labstreaminglayer.org), it covers the full pipeline — from amplifier to 3D brain display — in a single, researcher-friendly API.

## Highlights

| Feature | Details |
|---------|---------|
| **14+ NF modalities** | Alpha power, ERD/ERS, laterality, Hjorth, spectral centroid, CFC, graph metrics, source power … |
| **Sensor & source space** | Full MNE inverse-operator pipeline for source-level NF |
| **Real-time artifact correction** | ORICA (online ICA), adaptive LMS, GEDAI (GED-based spatial filters) |
| **Three parallel windows** | Raw stream viewer · NF signal monitor · 3D brain activation |
| **OSC output** | Send feedback values to Max/MSP, SuperCollider, Pure Data |
| **CLI** | `ANT demo`, `ANT baseline`, `ANT run` — no Python required |
| **Mock mode** | Works without hardware using bundled sample EEG data |

## Installation

```bash
pip install ANT                 # core package (OSC output included)
pip install "ANT[full]"         # all extras (viz, dev, docs)
```

<details>
<summary>Other installation methods</summary>

**uv (fast Rust-based installer):**
```bash
uv pip install ANT
```

**conda / mamba:**
```bash
conda env create -f environment.yml
conda activate ant
```

**Development install from source:**
```bash
git clone https://github.com/payamsash/ANT.git
cd ANT
pip install -e ".[dev]"
```

</details>

Verify the installation:
```bash
ANT info     # print versions of ANT and all key dependencies
ANT demo     # run a 60-second mock neurofeedback session
```

## Quick start

```python
from ant import NFRealtime

# 1 — Create a session object
nf = NFRealtime(
    subject_id="sub01",
    visit=1,
    session="main",
    subjects_dir="/data/subjects",
    montage="easycap-M1",
)

# 2 — Connect to a live LSL stream (or a mock replay)
nf.connect_to_lsl(mock_lsl=True)

# 3 — Record a resting-state baseline (computes inverse operator)
nf.record_baseline(baseline_duration=120)

# 4 — Run the closed-loop NF session
nf.record_main(
    duration=300,
    modality=["sensor_power", "erd_ers", "laterality"],
    show_nf_signal=True,
    show_brain_activation=True,
)
```

## Architecture

<p align="center">
  <a href="https://payamsash.github.io/ANT/_static/ant_workflow.html">
    <img src="docs/source/_static/ant_workflow.svg"
         alt="ANT Processing Pipeline"
         width="100%"
         style="border-radius:10px; border:1px solid rgba(200,200,200,0.15);
                box-shadow:0 6px 18px rgba(0,0,0,0.35);">
  </a>
</p>
<p align="center"><sub>Click to open the interactive diagram</sub></p>

The acquisition loop runs in a **background daemon thread**; all three visualisation windows share a **Qt event loop** on the main thread, updated at ~30 fps via a pump timer.

## Available NF modalities

| Key | Description |
|-----|-------------|
| `sensor_power` | Mean band power across channels |
| `band_ratio` | Power ratio between two bands (e.g. θ/β) |
| `erd_ers` | Event-related de/synchronisation (baseline-normalised) |
| `laterality` | Log power asymmetry right vs. left hemisphere |
| `hjorth` | Hjorth activity, mobility, complexity |
| `spectral_centroid` | Frequency-weighted spectral centroid |
| `entropy` | Spectral / approximate / sample entropy |
| `argmax_freq` | Dominant frequency peak |
| `individual_peak_power` | Power at the individual spectral peak |
| `cfc_sensor` | Cross-frequency coupling (sensor space) |
| `sensor_connectivity` | Functional connectivity (PLI, correlation) |
| `sensor_graph` | Graph Laplacian from sensor connectivity |
| `source_power` | Source-space band power |
| `source_connectivity` | Source-space functional connectivity |
| `source_graph` | Graph Laplacian from source connectivity |

## CLI

```bash
# Quick demo — no amplifier needed
ANT demo --duration 60 --modality sensor_power band_ratio

# Record a resting-state baseline
ANT baseline --subject sub01 --subjects-dir /data --duration 120 --mock

# Run a full NF session with OSC output
ANT run --subject sub01 --subjects-dir /data --duration 600 \
        --modality sensor_power erd_ers \
        --osc-host 127.0.0.1 --osc-port 9000
```

## Documentation

Full documentation, API reference, and gallery examples:  
**[payamsash.github.io/ANT](https://payamsash.github.io/ANT/)**

## Cite

If you use ANT, please cite:

```bibtex
@inproceedings{shabestari2025advances,
  title     = {Advances on Real Time M/EEG Neural Feature Extraction},
  author    = {Shabestari, Payam S and Ribes, Delphine and D{\'e}fayes, Lara
               and Cai, Danpeng and Groves, Emily and Behjat, Harry H
               and Van de Ville, Dimitri and Kleinjung, Tobias
               and Naas, Adrian and Henchoz, Nicolas and others},
  booktitle = {2025 IEEE 38th International Symposium on Computer-Based
               Medical Systems (CBMS)},
  pages     = {337--338},
  year      = {2025},
  organization = {IEEE}
}
```

## Acknowledgements

Development was supported by the [Swiss National Science Foundation](https://www.snf.ch/en)
(grant number - 208164 — *Advancing Neurofeedback in Tinnitus*).

## License

[MIT License](LICENSE) — © 2025 Payam S. Shabestari
