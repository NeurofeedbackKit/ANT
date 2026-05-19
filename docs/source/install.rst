.. _install:

Installation
============

Requirements
------------

- Python ≥ 3.11
- A running `Lab Streaming Layer (LSL) <https://labstreaminglayer.org>`_ stream
  **or** any MNE-readable file for mock mode
  (``.fif``, ``.vhdr``, ``.edf``, ``.bdf``, ``.set``, …)
- FreeSurfer subjects directory *(only for source-localisation and brain-plot features)*

pip (recommended)
-----------------

.. code-block:: bash

    # Latest release (OSC output included)
    pip install ant-nf

    # All optional extras (viz, dev, docs)
    pip install "ant-nf[full]"

Editable / development install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/payamsash/ANT.git
    cd ANT
    pip install -e ".[dev]"

uv — recommended fast installer
---------------------------------

`uv <https://docs.astral.sh/uv/>`_ resolves and installs packages in Rust —
typically **10–20× faster** than plain ``pip``.  It reads ``pyproject.toml``
directly and handles all extras.

.. code-block:: bash

    # Install uv once
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install ANT into an active environment
    uv pip install ant-nf
    uv pip install "ant-nf[full]"   # all extras (viz, dev, docs)

    # Editable install from source (recommended for development)
    git clone https://github.com/payamsash/ANT.git
    cd ANT
    uv pip install -e ".[dev]"

.. note::

   ``uv add ANT`` is for adding ANT as a dependency *of another project*.
   Inside the ANT source tree use ``uv pip install -e .`` instead.

conda / mamba
-------------

The provided ``environment.yml`` creates a complete conda environment.
**mamba** (or micromamba) is strongly recommended over plain ``conda``
because it uses a faster C++ dependency solver — environment creation
typically completes in under 2 minutes instead of 10–20 minutes.

.. code-block:: bash

    # Install mamba into base (once)
    conda install -n base -c conda-forge mamba

    # Create the ANT environment
    mamba env create -f environment.yml   # ~2 min

    # Or with plain conda (slower)
    conda env create -f environment.yml

    # Activate
    conda activate ant

    # Update after pulling new changes
    mamba env update -f environment.yml --prune

Verifying
---------

.. code-block:: bash

    ANT info     # prints ANT and dependency versions
    ANT demo     # runs a 120-second mock NF session

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 10 40 30

   * - Extra
     - What it adds
     - Install command
   * - ``viz``
     - 3D brain visualisation (pyvista, pyvistaqt) — needed for BrainPlot
     - ``pip install "ant-nf[viz]"``
   * - ``dev``
     - Testing only (pytest, pytest-cov)
     - ``pip install "ant-nf[dev]"``
   * - ``lint``
     - Linting and formatting (ruff, mypy, pre-commit)
     - ``pip install "ant-nf[lint]"``
   * - ``docs``
     - Documentation build tools (Sphinx, sphinx-gallery, …)
     - ``pip install "ant-nf[docs]"``
   * - ``full``
     - All of the above
     - ``pip install "ant-nf[full]"``
