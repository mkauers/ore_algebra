r"""
Index of examples

The following submodules provide useful data from various applications for use
with ore_algebra (e.g., annihilating operators for generating series of
combinatorial objects) along with examples of computations that one can do
starting from this data.

.. autosummary::

    cbt
    fcc
    iint
    periods
    polya
    ssw
    stdfun

::

    sage: from ore_algebra.examples import fcc, ssw
    sage: ssw.dop[1,0,0]
    (16*t^4 - t^2)*Dt^3 + (144*t^3 - 9*t)*Dt^2 + (288*t^2 - 15)*Dt + 96*t
    sage: fcc.dop4.order()
    4
"""
