"""Advanced Neurofeedback Toolbox (ANT).

ANT is an open-source Python package for **real-time processing and
visualisation of M/EEG neurofeedback experiments**.

Main entry points
-----------------
:class:`~ant.NFRealtime`
    Complete NF session controller — LSL streaming, artifact correction,
    feature extraction, and real-time visualisation.
:class:`~ant.viz.NFSignalPlot`
    Scrolling multi-channel NF signal monitor (PyQt6 + pyqtgraph).
:class:`~ant.viz.BrainPlot`
    Interactive 3D cortical surface with activity overlay (PyVista).
:class:`~ant.tools.ORICA`
    Online Recursive ICA for real-time artifact removal.
:class:`~ant.tools.GEDAIDenoiser`
    GED-based spatial filter for artifact identification and removal.

Verbosity
---------
All public methods accept a ``verbose`` keyword argument following
MNE's convention.  You can also set the package-wide level::

    import ant
    ant.set_log_level("INFO")   # or True / False / "DEBUG" / "WARNING"
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ANT")
except PackageNotFoundError:
    __version__ = "0.0.0"

from ant._logging import logger, set_log_level  # noqa: F401 — public API
from ant.realtime_nf import NFRealtime
from ant.viz import BrainPlot, NFSignalPlot, TopoPlot
from ant.tools import ORICA, GEDAIDenoiser
from ant.tools.lms import AdaptiveLMSFilter
from ant.tools.asr import ASRDenoiser
from ant.tools.maxwell import RTMaxwellFilter
from ant.tools.bad_channel_detector import BadChannelDetector
from ant.osc import OSCSender
from ant.protocols import ThresholdProtocol, ZScoreProtocol, PercentileProtocol, LinearTrendProtocol
from ant.lsl_output import LSLSender
from ant.combiners import (  # noqa: F401 — public API
    FeatureCombiner,
    WeightedSumCombiner,
    GeometricMeanCombiner,
    ZScoredNormCombiner,
    LearnedCombiner,
)

__all__ = [
    "NFRealtime",
    "BrainPlot",
    "NFSignalPlot",
    "TopoPlot",
    "ORICA",
    "GEDAIDenoiser",
    "AdaptiveLMSFilter",
    "ASRDenoiser",
    "RTMaxwellFilter",
    "ThresholdProtocol",
    "ZScoreProtocol",
    "PercentileProtocol",
    "LinearTrendProtocol",
    "BadChannelDetector",
    "OSCSender",
    "LSLSender",
    "FeatureCombiner",
    "WeightedSumCombiner",
    "GeometricMeanCombiner",
    "ZScoredNormCombiner",
    "LearnedCombiner",
    "set_log_level",
    "logger",
    "__version__",
]
