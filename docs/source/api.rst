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

NF Protocols
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.protocols.ThresholdProtocol
   ant.protocols.ZScoreProtocol
   ant.protocols.PercentileProtocol

OSC output
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.osc.OSCSender

Tools & utilities
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.tools.simulate_raw

Logging
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ant.set_log_level
