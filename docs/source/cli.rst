.. _cli:

Command-Line Interface
======================

ANT provides an ``ANT`` shell command with four sub-commands.

.. code-block:: text

    ANT --help
    ANT --version
    ANT info
    ANT demo [options]
    ANT baseline --subject ID --subjects-dir DIR [options]
    ANT run     --subject ID --subjects-dir DIR --duration N [options]

``ANT info``
------------

Print the installed ANT version and all key dependency versions.

.. code-block:: console

    $ ANT info
    Advanced Neurofeedback Toolbox (ANT)
    ────────────────────────────────────
      ANT version  : 1.0.0
      Python       : 3.11.9
      mne          : 1.8.0
      ...

``ANT demo``
------------

Launch a full demo NF session from simulated EEG (no amplifier needed).
The real-time scalp topomap (δ/θ/α/β/γ) is shown by default; the 3D brain
display is opt-in (requires FreeSurfer ``fsaverage5``).

.. code-block:: console

    $ ANT demo
    $ ANT demo --duration 120 --modality sensor_power erd_ers
    $ ANT demo --no-topo
    $ ANT demo --brain --subjects-fs-dir /path/to/fsaverage

Options:

.. list-table::
   :header-rows: 1
   :widths: 25 15 45

   * - Flag
     - Default
     - Description
   * - ``--duration``
     - 120
     - Session duration in seconds
   * - ``--modality``
     - sensor_power band_ratio entropy hjorth
     - NF modality(ies) to demonstrate
   * - ``--winsize``
     - 1.0
     - Analysis window length (s)
   * - ``--no-signal``
     - —
     - Disable NF signal plot
   * - ``--no-raw``
     - —
     - Disable raw stream viewer
   * - ``--no-topo``
     - —
     - Disable the real-time scalp topomap (on by default)
   * - ``--brain``
     - —
     - Enable 3D brain display
   * - ``--subjects-fs-dir``
     - —
     - FreeSurfer subjects dir (for ``--brain``)
   * - ``--surf``
     - inflated
     - Brain surface geometry
   * - ``--no-save``
     - —
     - Skip saving NF data and session report

``ANT baseline``
----------------

Record a resting-state baseline and compute the inverse operator.

.. code-block:: console

    $ ANT baseline --subject sub01 --subjects-dir /data/subjects
    $ ANT baseline --subject sub01 --subjects-dir /data --duration 180 --mock

Required:

- ``--subject ID`` — subject identifier string
- ``--subjects-dir DIR`` — root directory with one folder per subject

Options:

.. list-table::
   :header-rows: 1
   :widths: 25 15 45

   * - Flag
     - Default
     - Description
   * - ``--duration``
     - 120
     - Baseline duration in seconds
   * - ``--mock``
     - —
     - Use simulated data instead of live LSL
   * - ``--fname``
     - —
     - Any MNE-readable file (.fif, .vhdr, .edf, .bdf, …) for ``--mock``
   * - ``--montage``
     - easycap-M1
     - EEG montage name or ``.bvct`` path
   * - ``--data-type``
     - eeg
     - ``eeg`` or ``meg``

``ANT run``
-----------

Run a closed-loop NF main session.

.. code-block:: console

    $ ANT run --subject sub01 --subjects-dir /data --duration 600 \
              --modality sensor_power erd_ers

    # With topomap display
    $ ANT run --subject sub01 --subjects-dir /data --duration 600 \
              --modality sensor_power --topo

    # With OSC output to Max/MSP
    $ ANT run --subject sub01 --subjects-dir /data --duration 600 \
              --modality sensor_power --osc-host 127.0.0.1 --osc-port 9000

Options (in addition to baseline flags):

.. list-table::
   :header-rows: 1
   :widths: 25 15 45

   * - Flag
     - Default
     - Description
   * - ``--duration``
     - *required*
     - Session duration in seconds
   * - ``--modality``
     - sensor_power
     - NF modality(ies): ``sensor_power``, ``band_ratio``, ``entropy``,
       ``hjorth``, ``spectral_centroid``, ``erd_ers``, ``laterality``,
       ``laterality_erd_ers``, ``wpli_sensor``, ``sensor_connectivity``,
       ``source_power``, ``source_connectivity``, ``sensor_graph``,
       ``source_graph``, ``cfc_sensor``, ``argmax_freq``,
       ``individual_peak_power``
   * - ``--winsize``
     - 1.0
     - Analysis window length (s)
   * - ``--artifact-correction``
     - —
     - ``lms``, ``orica``, ``gedai``, ``asr``, or ``maxwell`` (MEG only)
   * - ``--ring-buffer``
     - —
     - Use sliding ring-buffer (50 % overlap)
   * - ``--no-signal``
     - —
     - Disable NF signal plot
   * - ``--no-raw``
     - —
     - Disable raw stream viewer
   * - ``--topo``
     - —
     - Enable real-time scalp topomap (δ/θ/α/β/γ bands)
   * - ``--brain``
     - —
     - Enable 3D brain display
   * - ``--osc-host``
     - —
     - Enable OSC output; target hostname
   * - ``--osc-port``
     - 9000
     - OSC destination port
   * - ``--osc-prefix``
     - /ant
     - OSC address prefix

Available NF modalities
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - Modality key
     - Description
   * - ``sensor_power``
     - Mean band power across channels
   * - ``band_ratio``
     - Power ratio between two frequency bands (e.g. θ/β)
   * - ``erd_ers``
     - Event-related de/synchronisation (baseline normalised)
   * - ``laterality``
     - Log power asymmetry between right and left hemispheres
   * - ``hjorth``
     - Hjorth activity, mobility, complexity (time-domain)
   * - ``spectral_centroid``
     - Frequency-weighted spectral centroid
   * - ``entropy``
     - Spectral, approximate, or sample entropy
   * - ``argmax_freq``
     - Dominant frequency peak
   * - ``individual_peak_power``
     - Power at the individualised spectral peak
   * - ``cfc_sensor``
     - Cross-frequency coupling (sensor space)
   * - ``sensor_connectivity``
     - Functional connectivity (PLI, correlation)
   * - ``sensor_graph``
     - Graph-Laplacian learning from sensor connectivity
   * - ``source_power``
     - Source-space band power (requires inverse operator)
   * - ``source_connectivity``
     - Source-space functional connectivity
   * - ``source_graph``
     - Graph-Laplacian learning from source connectivity

For the mathematical background of every modality, see :doc:`modalities`.
