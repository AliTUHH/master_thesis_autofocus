"""Simulation-based inference (SBI) for the autofocus problem: amortised posterior ``q(z01 | hologram)``.

Sub-modules:

* :mod:`src.sbi.npe` -- prior from ``meta.json``, embedding networks, neural posterior estimation with
  ``sbi``, saving/loading of the posterior and the hybrid interface to HoloWizard's ``find_focus``;
* :mod:`src.sbi.calibration` -- coverage, SBC ranks, uncertainty/error diagnostics (pure NumPy/SciPy);
* :mod:`src.sbi.simulator` -- sketch of an online HoloForge simulator ``theta -> x`` for ``sbi``;
* :mod:`src.sbi.train_npe`, :mod:`src.sbi.evaluate_npe` -- command line interfaces.
"""
