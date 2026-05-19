<p align="center">
  <img src="docs/source/_static/ANT_Logo_Horizontal.svg" alt="ANT Logo" width="480"/>
</p>

<p align="center">
  <strong>Advanced Neurofeedback Toolbox</strong><br>
  Real-time M/EEG neurofeedback for research and clinical use
</p>

<p align="center">
  <a href="https://pypi.org/project/ant-nf/"><img alt="PyPI" src="https://img.shields.io/pypi/v/ant-nf?color=blue&logo=pypi&logoColor=white"></a>
  <a href="https://pypi.org/project/ant-nf/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/ant-nf?logo=python&logoColor=white"></a>
  <a href="https://github.com/payamsash/ANT/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/payamsash/ANT?color=green"></a>
  <a href="https://payamsash.github.io/ANT/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-brightgreen?logo=readthedocs&logoColor=white"></a>
</p>

---

**ANT** is an open-source Python package for **real-time closed-loop M/EEG neurofeedback**. Built on [MNE-Python](https://mne.tools) and the [Lab Streaming Layer (LSL)](https://labstreaminglayer.org), it covers the full pipeline — from amplifier to 3D brain display — in a single, researcher-friendly API.

## Highlights

<table>
  <thead>
    <tr>
      <th width="30%">Feature</th>
      <th>Details</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>20+ NF modalities</strong></td>
      <td>Band power, ERD/ERS, laterality, Hjorth parameters, spectral centroid, slow cortical potentials, CFC, functional connectivity, graph Laplacian — in sensor and source space</td>
    </tr>
    <tr bgcolor="#f6f8fa">
      <td><strong>Adaptive protocols</strong></td>
      <td>Z-score, threshold, percentile, staircase, operant, reinforcement learning, sham, and transfer — all evaluated <em>inside</em> the acquisition loop on every analysis window</td>
    </tr>
    <tr>
      <td><strong>Real-time artifact correction</strong></td>
      <td>ASR, adaptive LMS, GEDAI (GED-based spatial filters), ORICA (online ICA), Riemannian covariance detection</td>
    </tr>
    <tr bgcolor="#f6f8fa">
      <td><strong>Real-time Maxwell filtering</strong></td>
      <td>Pre-computed SSS/tSSS projector for zero-latency MEG denoising; numerically equivalent to offline MNE</td>
    </tr>
    <tr>
      <td><strong>Three live displays</strong></td>
      <td>Raw stream viewer · NF signal monitor · 3D cortical activation map — all updating at ~30 fps via a shared Qt event loop</td>
    </tr>
    <tr bgcolor="#f6f8fa">
      <td><strong>External output</strong></td>
      <td>OSC (Max/MSP, SuperCollider, Pure Data) and LSL outlet (PsychoPy, OpenViBE, Psychtoolbox) for reward delivery to any application</td>
    </tr>
    <tr>
      <td><strong>BIDS-compatible saving</strong></td>
      <td>Session data saved as JSON + optional TSV with full metadata, artifact rate, and SNR</td>
    </tr>
    <tr bgcolor="#f6f8fa">
      <td><strong>CLI</strong></td>
      <td><code>ANT info</code> · <code>ANT demo</code> · <code>ANT baseline</code> · <code>ANT run</code> — no Python required</td>
    </tr>
    <tr>
      <td><strong>Mock mode</strong></td>
      <td>Full pipeline without hardware via built-in LSL replay from any MNE-readable file</td>
    </tr>
  </tbody>
</table>

## Installation

```bash
pip install ant-nf                 # core package
pip install "ant-nf[full]"         # all extras: viz, dev, docs
```

<details>
<summary>Other installation methods</summary>

**uv (fast Rust-based installer):**
```bash
uv pip install ant-nf
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
ANT info     # print ANT version and all key dependency versions
ANT demo     # run a 60-second mock neurofeedback session
```

## Quick start

```python
from ant import NFRealtime
from ant.protocols import ZScoreProtocol

# 1 — Create a session object
nf = NFRealtime(
    subject_id="sub01",
    session="01",
    subjects_dir="/data/subjects",
    montage="easycap-M1",
)

# 2 — Connect to a live LSL stream (or replay a file without hardware)
nf.connect_to_lsl(mock_lsl=True)

# 3 — Record a resting-state baseline
nf.record_baseline(baseline_duration=120)

# 4 — Run a closed-loop NF session with an adaptive reward protocol
nf.record_main(
    duration=300,
    modality=["sensor_power", "erd_ers", "laterality"],
    protocol=ZScoreProtocol(direction="up", zscore_threshold=0.5),
    show_nf_signal=True,
    show_topo=True,
)

# 5 — Save results (JSON + companion TSV)
nf.save(bids_tsv=True)
```

## CLI

```bash
# Print ANT version and all dependency versions
ANT info

# Quick demo — no amplifier or files needed (default: 120 s)
ANT demo --duration 60 --modality sensor_power erd_ers laterality

# Record a resting-state baseline from a live LSL stream
ANT baseline --subject sub01 --subjects-dir /data --session 01 --duration 120

# Record a baseline from a file (mock mode)
ANT baseline --subject sub01 --subjects-dir /data --mock --fname recording.fif

# Run a full session with real-time artifact correction and displays
ANT run --subject sub01 --subjects-dir /data --duration 600 \
        --modality sensor_power erd_ers laterality \
        --artifact-correction asr \
        --topo --brain

# Stream reward values to Max/MSP or SuperCollider via OSC
ANT run --subject sub01 --subjects-dir /data --duration 600 \
        --modality sensor_power laterality \
        --osc-host 127.0.0.1 --osc-port 9000

# Broadcast NF values as an LSL outlet (PsychoPy, OpenViBE, Psychtoolbox …)
ANT run --subject sub01 --subjects-dir /data --duration 600 \
        --modality sensor_power --lsl-output
```

## Cite

If you use ANT in your research, please cite:

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
(grant number 208164 — *Advancing Neurofeedback in Tinnitus*).

## License

[MIT License](LICENSE) — © 2025 Payam S. Shabestari
