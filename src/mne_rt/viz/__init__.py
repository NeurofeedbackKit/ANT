"""Real-time visualisation components for the Advanced Neurofeedback Toolbox.

Classes
-------
NFSignalPlot
    Scrolling dark-themed multi-channel NF signal monitor (PyQt6 + pyqtgraph).
BrainPlot
    Interactive 3D cortical surface with activity overlay (PyVista).
TopoPlot
    Real-time scalp topomap showing per-band power distribution (matplotlib).
"""
from .nf_plot import NFSignalPlot
from .brain_plot import BrainPlot
from .topo_plot import TopoPlot

__all__ = ["NFSignalPlot", "BrainPlot", "TopoPlot"]
