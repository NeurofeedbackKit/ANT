.. _api:

API Reference
=============

This page provides the complete API reference for the
**Advanced Neurofeedback Toolbox (ANT)**.

Core
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.NFRealtime

Visualisation
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.viz.NFSignalPlot
   ant.viz.BrainPlot
   ant.viz.TopoPlot

Artifact correction
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.tools.AdaptiveLMSFilter
   ant.tools.ORICA
   ant.tools.GEDAIDenoiser
   ant.tools.ASRDenoiser
   ant.tools.RTMaxwellFilter

Quality control
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.tools.BadChannelDetector
   ant.tools.RiemannianPotatoDetector

NF Protocols
------------

See :doc:`protocols` for the full protocol guide with formulas and examples.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.protocols.ThresholdProtocol
   ant.protocols.ZScoreProtocol
   ant.protocols.PercentileProtocol
   ant.protocols.LinearTrendProtocol
   ant.protocols.ShamProtocol
   ant.protocols.UpDownStaircaseProtocol
   ant.protocols.MultiBandProtocol

Feedback output
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.osc.OSCSender
   ant.lsl_output.LSLSender

Tools & utilities
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.tools.simulate_raw
   ant.tools.simulate_nf_session
   ant.tools.compute_instantaneous_phase

BIDS I/O
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.tools.save_as_bids

Logging
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.set_log_level
