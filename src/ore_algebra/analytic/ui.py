# -*- coding: utf-8 - vim: tw=80
"""
Some convenience functions for direct use of the features of this package

Ultimately, the typical way to use it should be through methods of objects such
as differential operators and D-finite functions, not through this module!


FIXME: silence deprecation warnings::

    sage: def ignore(*args): pass
    sage: sage.misc.superseded.warning=ignore

EXAMPLES::

    sage: from ore_algebra import *
    sage: from ore_algebra.analytic.ui import *
    sage: QQi.<i> = QuadraticField(-1)
    sage: Pol.<x> = QQ[]
    sage: Dop.<Dx> = OreAlgebra(Pol)

    sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

    sage: eval_diffeq(dop, [0, 1], [0, 1+i], 1e-30)
    1.01722196789785136772278896... + 0.40235947810852509365018983...*I

    sage: transition_matrix(dop, [0, 1+i], 1e-10)
    [                              1 1.017221967... + 0.402359478...*I]
    [                              0 0.200000000... - 0.400000000...*I]

Return the values (resp. transition matrices) corresponding to several points
along the path at once::

    sage: multi_eval_diffeq(dop, [0, 1], [k/5 for k in range(5)], 1e-10)
    [(0, 0),
    (1/5, 0.19739555985...),
    (2/5, 0.3805063771...),
    (3/5, 0.5404195002...),
    (4/5, 0.674740942...)]

    sage: tms = transition_matrices(dop, [k/5 for k in range(5)], 1e-10)
    sage: tms[2]
    (
        [             1 0.3805063771...]
    2/5, [             0 0.8620689655...]
    )


"""

# NOTE: to run the tests, use something like
#
#     SAGE_PATH="$PWD" sage -t ore_algebra/analytic/ui.py

# from . import analytic_continuation as ancont
from ore_algebra.analytic import analytic_continuation as ancont

def transition_matrix(dop, path, eps):
    """
    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, i, x, Dx = Diffops()
        sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

        sage: transition_matrix(dop, [0, 1], 1e-20)
        [                      1 0.7853981633974483096...]
        [                      0 0.5000000000000000000...]

        sage: transition_matrix(dop, [0, 1+i], 1e-10)
        [                              1 1.017221967... + 0.402359478...*I]
        [                              0 0.200000000... - 0.400000000...*I]


    """
    ctx = ancont.Context(dop, path, eps)
    pairs = ancont.analytic_continuation(ctx)
    assert len(pairs) == 1
    return pairs[0][1]

def transition_matrices(dop, path, eps):
    ctx = ancont.Context(dop, path, eps, keep="all")
    pairs = ancont.analytic_continuation(ctx)
    return pairs

def eval_diffeq(dop, ini, path, eps):
    ctx = ancont.Context(dop, path, eps)
    pairs = ancont.analytic_continuation(ctx, ini=ini)
    assert len(pairs) == 1
    _, mat = pairs[0]
    return mat[0][0]

def multi_eval_diffeq(dop, ini, path, eps):
    ctx = ancont.Context(dop, path, eps, keep="all")
    pairs = ancont.analytic_continuation(ctx, ini=ini)
    return [(point, mat[0][0]) for point, mat in pairs]

def polynomial_approximation_on_disk(dop, ini, path, rad, eps):
    raise NotImplementedError

def polynomial_approximation_on_interval(dop, ini, path, rad, eps):
    raise NotImplementedError

def make_proc(xxx): # ??? - ou object DFiniteFunction ?
    pass

def Diffops(sx='x'):
    """
    Return the Ore algebra of differential operators with polynomial
    coefficients over ℚ(i), along with objects representing i, x and d/dx.

    EXAMPLE::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, i, x, Dx = Diffops()
        sage: Dops
        Univariate Ore algebra in Dx over Univariate Polynomial Ring in x over
        Number Field in i with defining polynomial x^2 + 1
        sage: (x - i)*Dx + 1
        (x - i)*Dx + 1
    """
    from sage.rings.number_field.number_field import QuadraticField
    from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
    from ore_algebra.ore_algebra import OreAlgebra
    QQi, i = QuadraticField(-1, 'i').objgen()
    Pol, x = PolynomialRing(QQi, sx).objgen()
    Dop, Dx = OreAlgebra(Pol, 'D' + sx).objgen()
    return Dop, i, x, Dx
