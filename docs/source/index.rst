.. raw:: html

    <div style="text-align:center; margin-bottom:20px;">
        <img src="_static/ANT_Logo_Horizontal.svg" alt="ANT Logo" width="500"/>
    </div>

**Advanced Neurofeedback Toolbox (ANT)** is an open-source Python library for
**real-time M/EEG neurofeedback**, built on `MNE-Python <https://mne.tools>`_
and the `Lab Streaming Layer <https://labstreaminglayer.org>`_ (LSL).
It covers the full closed-loop pipeline — from amplifier to 3-D brain display —
in a single, researcher-friendly API designed for both clinical and basic-science
applications.

.. raw:: html

    <div style="height:16px;"></div>

Key capabilities
----------------

.. list-table::
   :widths: 5 45
   :header-rows: 0

   * - 🧠
     - **17 real-time NF modalities** — sensor power, ERD/ERS, Hjorth parameters,
       spectral centroid and peak, band ratio, cross-frequency coupling (CFC),
       weighted Phase Lag Index (wPLI), graph-theory metrics, and more.
       See :doc:`modalities` for the full list.

   * - 📡
     - **Sensor-space and source-space** processing using
       `MNE <https://mne.tools>`_ inverse operators (eLORETA, MNE, dSPM).
       Compute and stream cortical-source activity with a single parameter.

   * - 🔧
     - **Live artifact correction** —
       :class:`~ant.tools.ORICA` (online ICA),
       :class:`~ant.tools.AdaptiveLMS` (adaptive least-mean-squares),
       :class:`~ant.tools.GEDAIDenoiser` (generalised eigendecomposition),
       :class:`~ant.tools.ASRDenoiser` (artifact subspace reconstruction), and
       :class:`~ant.tools.RTMaxwellFilter` (real-time Maxwell/SSS filtering for MEG).
       See :doc:`denoising` for algorithm details and benchmarks.

   * - 🔍
     - **Real-time quality control** — :class:`~ant.tools.BadChannelDetector`
       flags flat, noisy, or de-correlated channels on every incoming window
       using a robust rolling-vote mechanism — no baseline recording required.

   * - 🎯
     - **Adaptive NF protocols** — :class:`~ant.protocols.ThresholdProtocol`,
       :class:`~ant.protocols.ZScoreProtocol`,
       :class:`~ant.protocols.PercentileProtocol`, and
       :class:`~ant.protocols.LinearTrendProtocol`
       give fine-grained control over when to issue a reward.

   * - 🖥️
     - **Three parallel visualisation windows** — a scrolling
       :class:`~ant.viz.NFSignalPlot`, a live
       `MNE-style <https://mne.tools/stable/visualization.html>`_ topographic map,
       and an interactive :class:`~ant.viz.BrainPlot` (3-D cortical surface with
       colour-mapped activity, hemisphere toggles, and surface switching).

   * - 📤
     - **Dual feedback output** — broadcast values via OSC (Max/MSP,
       SuperCollider, TouchDesigner) with :class:`~ant.osc.OSCSender`, or over
       `LSL <https://labstreaminglayer.org>`_ with :class:`~ant.lsl_output.LSLSender`
       for low-latency same-machine integration with PsychoPy, OpenViBE, BCI2000,
       and other LSL-aware apps.

   * - ⌨️
     - **CLI** — launch full NF sessions with a single ``ANT run`` command,
       driven by a YAML config file. See :doc:`cli`.

.. raw:: html

    <div style="height:16px;"></div>

.. tabs::

   .. tab:: NF Signal

      .. raw:: html

         <div style="text-align:center; margin: 20px 0;">
             <video width="100%" style="max-width: 850px; border-radius: 15px; display: block; margin: 0 auto;" autoplay muted loop>
                 <source src="_static/NFSignal.mov" type="video/quicktime">
             </video>
         </div>

   .. tab:: Topo Plot

      .. raw:: html

         <div style="text-align:center; margin: 20px 0;">
             <video width="100%" style="max-width: 850px; border-radius: 15px; display: block; margin: 0 auto;" autoplay muted loop>
                 <source src="_static/TopoPlot.mov" type="video/quicktime">
             </video>
         </div>

   .. tab:: Brain Plot

      .. raw:: html

         <div style="text-align:center; margin: 20px 0;">
             <video width="100%" style="max-width: 850px; border-radius: 15px; display: block; margin: 0 auto;" autoplay muted loop>
                 <source src="_static/BrainPlot.mov" type="video/quicktime">
             </video>
         </div>

.. toctree::
   :hidden:
   :caption: Getting started

   install

.. toctree::
   :hidden:
   :caption: Reference

   api
   cli
   denoising
   modalities

.. toctree::
   :hidden:
   :caption: Examples

   auto_examples/index

.. raw:: html

    <div style="height:24px;"></div>

Quick install
-------------

.. tabs::

   .. tab:: pip

      .. code-block:: bash

          pip install ANT                 # core  (MNE, LSL, OSC included)
          pip install "ANT[full]"         # + 3-D viz, dev tools, docs

   .. tab:: uv

      .. code-block:: bash

          # Install uv once
          curl -LsSf https://astral.sh/uv/install.sh | sh

          uv pip install ANT
          uv pip install "ANT[full]"      # + 3-D viz, dev tools, docs

          # Editable install from source
          git clone https://github.com/payamsash/ANT.git
          cd ANT && uv pip install -e ".[dev]"

   .. tab:: conda / mamba

      .. code-block:: bash

          mamba create -n ant python=3.11
          mamba activate ant
          pip install "ANT[full]"

See :doc:`install` for full instructions.

Quick start
-----------

.. code-block:: python

    from ant import NFRealtime

    nf = NFRealtime(
        "sub01",
        visit=1,
        session="main",
        subjects_dir="/data/subjects",
        montage="easycap-M1",
    )
    nf.connect_to_lsl(mock_lsl=True)          # or connect to a real amplifier
    nf.record_main(
        duration=300,
        modality=["sensor_power", "erd_ers"],
        show_nf_signal=True,
    )

Pipeline overview
-----------------

.. code-block:: none

    Amplifier / mock LSL stream
          ↓  (mne-lsl StreamInlet)
    BadChannelDetector       ← flags flat / noisy channels every window
          ↓
    Artifact correction      ← ORICA / LMS / GEDAI / ASR / RT-SSS
          ↓
    Feature extraction       ← 17 NF modalities  (sensor or source space)
          ↓
    NF protocol              ← Threshold / Z-score / Percentile / LinearTrend
          ↓  ↘
    NFSignalPlot    BrainPlot        ← live visualisation
          ↓
    OSCSender / LSLSender    ← feedback to stimulus software

Cite
----

If you use ANT, please cite :footcite:`shabestari2025advances`.

.. footbibliography::

.. tabs::

    .. tab:: APA

        .. code-block:: none

            Shabestari, P. S., Ribes, D., Défayes, L., Cai, D., Groves, E.,
            Behjat, H. H., … & Neff, P. (2025). Advances on Real Time M/EEG
            Neural Feature Extraction. IEEE CBMS 2025.

    .. tab:: BibTeX

        .. code-block:: bibtex

            @inproceedings{shabestari2025advances,
                title   = {Advances on Real Time M/EEG Neural Feature Extraction},
                author  = {Shabestari, Payam S and others},
                booktitle = {2025 IEEE 38th CBMS},
                pages   = {337--338},
                year    = {2025},
                organization = {IEEE}
            }

.. raw:: html

    <div style="height:20px;"></div>

.. image:: _static/SNF.png
    :align: right
    :alt: SNSF
    :width: 320

Development was supported by the
`Swiss National Science Foundation <https://www.snf.ch/en>`_ (grant number — 208164).
