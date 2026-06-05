"""Real-time visualisation components for MNE-RT.

Classes
-------
SignalPlot
    Scrolling dark-themed multi-channel signal monitor (PyQt6 + pyqtgraph).
BrainPlot
    Interactive 3D cortical surface with activity overlay (PyVista).
TopoPlot
    Real-time scalp topomap showing per-band power distribution (matplotlib).
ERPPlot
    Live-updating evoked-potential display with scalp-layout channel grid.
TFRPlot
    Real-time Morlet wavelet TFR heatmaps per channel and condition.
ButterflyPlot
    Real-time butterfly overlay: all channels per condition coloured by
    scalp region.
CompareEvoked
    Real-time per-channel condition comparison with SEM shading, peak
    markers, and a clickable scalp-topomap for interactive channel selection.
"""
from .signal_plot import SignalPlot
from .brain_plot import BrainPlot
from .topo_plot import TopoPlot
from .erp_plot import ERPPlot
from .tfr_plot import TFRPlot
from .butterfly_plot import ButterflyPlot
from .compare_evoked import CompareEvoked

__all__ = [
    "SignalPlot",
    "BrainPlot",
    "TopoPlot",
    "ERPPlot",
    "TFRPlot",
    "ButterflyPlot",
    "CompareEvoked",
]
