# -*- coding: utf-8 - vim: tw=80
"""
Some convenience functions for features not yet easily accessible from methods
of differential operators.
"""

from ..ore_algebra import DifferentialOperators
from . import analytic_continuation as ancont
from . import polynomial_approximation as polapprox

def transition_matrices(dop, path, eps=1e-16):
    r"""
    Compute several transition matrices at once.

    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, x, Dx = DifferentialOperators()

        sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
        sage: tms = transition_matrices(dop, [k/5 for k in range(5)], 1e-10)
        sage: tms[2]
        (
            [ 1.0... [0.3805063771...]]
        2/5, [      0 [0.8620689655...]]
        )

        sage: transition_matrices(Dx - 1, [i/5 for i in range(6)], 1e-10)
        [(0,   [1.000000000...]),
         (1/5, [[1.221402758...]]),
         (2/5, [[1.491824697...]]),
         (3/5, [[1.822118800...]]),
         (4/5, [[2.225540928...]]),
         (1,   [[2.718281828...]])]
    """
    from .differential_operator import DifferentialOperator
    dop = DifferentialOperator(dop)
    ctx = ancont.Context(dop, path, eps, keep="all")
    pairs = ancont.analytic_continuation(ctx)
    return pairs

def _value_from_mat(mat):
    if mat.nrows():
        return mat[0][0]
    else:
        return mat.base_ring().zero()

def multi_eval_diffeq(dop, ini, path, eps=1e-16):
    """
    Evaluate a solution at several points along a path.

    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, x, Dx = DifferentialOperators()
        sage: QQi.<i> = QuadraticField(-1, 'I')

        sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
        sage: multi_eval_diffeq(dop, [0, 1], [k/5 for k in range(5)], 1e-10)
        [(0, 0),
        (1/5, [0.197395559...]),
        (2/5, [0.380506377...]),
        (3/5, [0.540419500...]),
        (4/5, [0.674740942...])]

    The logarithm::

        sage: multi_eval_diffeq(Dx*x*Dx, ini=[0, 1], path=[1, i, -1])
        [(1,  0),
         (i,  [...] + [1.57079632679489...]*I),
         (-1, [...] + [3.14159265358979...]*I)]

    XXX: make similar examples work with points in RLF/CLF (bug with binsplit?)

    TESTS::

        sage: multi_eval_diffeq(Dx - 1, ini=[42], path=[1])
        [(1, 42.000...)]
    """
    from .differential_operator import DifferentialOperator
    dop = DifferentialOperator(dop)
    ctx = ancont.Context(dop, path, eps, keep="all")
    pairs = ancont.analytic_continuation(ctx, ini=ini)
    return [(point, _value_from_mat(mat)) for point, mat in pairs]

polynomial_approximation_on_disk = polapprox.on_disk
polynomial_approximation_on_interval = polapprox.on_interval
