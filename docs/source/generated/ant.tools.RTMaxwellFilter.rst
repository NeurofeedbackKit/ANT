ant.tools.RTMaxwellFilter
=========================

.. currentmodule:: ant.tools

.. warning::

   ``RTMaxwellFilter`` has not been comprehensively tested on live MEG data.
   The pre-computed SSS matrix approach works well in benchmarks but may
   produce unexpected results with non-standard sensor configurations or
   heavily contaminated data.  Verify outputs against a reference tool
   (e.g. :func:`mne.preprocessing.maxwell_filter`) before use in a study.

.. autoclass:: RTMaxwellFilter


   .. rubric:: Methods

   .. autosummary::

      ~RTMaxwellFilter.fit
      ~RTMaxwellFilter.transform


