from .threshold import ThresholdProtocol
from .zscores import ZScoreProtocol
from .percentile import PercentileProtocol
from .linear_trend import LinearTrendProtocol
from .sham import ShamProtocol
from .staircase import UpDownStaircaseProtocol
from .multiband import MultiBandProtocol
from .rl_protocol import RLProtocol
from .operant import OperantProtocol
from .transfer import TransferProtocol

__all__ = [
    "ThresholdProtocol",
    "ZScoreProtocol",
    "PercentileProtocol",
    "LinearTrendProtocol",
    "ShamProtocol",
    "UpDownStaircaseProtocol",
    "MultiBandProtocol",
    "RLProtocol",
    "OperantProtocol",
    "TransferProtocol",
]

# Sphinx autodoc resolves cross-references against the objects inventory using
# cls.__module__.  Patching to the package namespace ensures that
# :class:`~ant.protocols.ZScoreProtocol` (and siblings) are registered as
# ``ant.protocols.*`` py:class entries, not ``ant.protocols.zscores.*``.
for _cls in [
    ThresholdProtocol,
    ZScoreProtocol,
    PercentileProtocol,
    LinearTrendProtocol,
    ShamProtocol,
    UpDownStaircaseProtocol,
    MultiBandProtocol,
    RLProtocol,
    OperantProtocol,
    TransferProtocol,
]:
    _cls.__module__ = "ant.protocols"
del _cls
