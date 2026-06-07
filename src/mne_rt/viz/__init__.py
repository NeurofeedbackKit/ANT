"""Real-time visualisation components for MNE-RT.

Classes
-------
NFPlot
    Scrolling dark-themed multi-channel NF signal monitor (PyQt6 + pyqtgraph).
RawPlot
    Scrolling dark-themed raw M/EEG channel viewer (PyQt6 + pyqtgraph).
BrainPlot
    Interactive 3D cortical surface with activity overlay (PyVista).
TopomapPlot
    Real-time scalp topomap showing per-band power distribution (matplotlib).
TopoPlot
    Live-updating scalp-layout ERP display — one mini-plot per electrode.
TFRPlot
    Real-time Morlet wavelet TFR heatmaps per channel and condition.
ButterflyPlot
    Real-time butterfly overlay: all channels per condition coloured by
    scalp region.
CompareEvoked
    Real-time per-channel condition comparison with SEM shading, peak
    markers, and a clickable scalp-topomap for interactive channel selection.
"""
from .nf_plot import NFPlot
from .raw_plot import RawPlot
from .brain_plot import BrainPlot
from .topomap_plot import TopomapPlot
from .topo_plot import TopoPlot
from .tfr_plot import TFRPlot
from .butterfly_plot import ButterflyPlot
from .compare_evoked import CompareEvoked

__all__ = [
    "NFPlot",
    "RawPlot",
    "BrainPlot",
    "TopomapPlot",
    "TopoPlot",
    "TFRPlot",
    "ButterflyPlot",
    "CompareEvoked",
]
