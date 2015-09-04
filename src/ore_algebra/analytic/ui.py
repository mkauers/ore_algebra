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
    sage: Pol.<x> = QQ[]
    sage: Dop.<Dx> = OreAlgebra(Pol)
    sage: QQi.<i> = QuadraticField(-1)

    sage: transition_matrix(Dx - 1, [0, 1], 1e-10)
    [2.718281828...]

    sage: transition_matrix(Dx - 1, [0, i], 1e-10)
    [0.5403023058... + 0.8414709848...*I]

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

TESTS::

    sage: transition_matrix(dop, [], 1e-10)
    Traceback (most recent call last):
    ...
    ValueError: empty path

    sage: transition_matrix(Dop.zero(), [0,1], 1e-10)
    Traceback (most recent call last):
    ...
    ValueError: operator must be nonzero

    sage: transition_matrix(Dx, [0, 1], 1e-10)[0,0].parent()
    Real Interval Field with 36 bits of precision

    sage: Dx_C = OreAlgebra(QQi['x'], 'Dx').gen()
    sage: transition_matrix(Dx_C, [0, 1], 1e-10)[0,0].parent()
    Complex Interval Field with 36 bits of precision
"""

# NOTE: to run the tests, use something like
#
#     SAGE_PATH="$PWD" sage -t ore_algebra/analytic/ui.py

# from . import analytic_continuation as ancont
from ore_algebra.analytic import analytic_continuation as ancont

def transition_matrix(dop, path, eps=1e-16):
    r"""
    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, x, Dx = Diffops()
        sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

        sage: transition_matrix(dop, [0, 1], 1e-20)
        [                      1 0.7853981633974483096...]
        [                      0 0.5000000000000000000...]

        sage: transition_matrix(dop, [0, 1+i], 1e-10)
        [                              1 1.017221967... + 0.402359478...*I]
        [                              0 0.200000000... - 0.400000000...*I]

        sage: transition_matrix(dop, [0, i], 1e-10)
        Traceback (most recent call last):
        ...
        ValueError: Step 0 --> I passes through or too close to
        singular point 1*I

    TESTS::

        sage: transition_matrix(dop, [0, 0])
        [1 0]
        [0 1]
        sage: transition_matrix(Dops(1), [0, 1])
        []
        sage: transition_matrix(Dx, [0, 1])
        [1]

    """
    ctx = ancont.Context(dop, path, eps)
    pairs = ancont.analytic_continuation(ctx)
    assert len(pairs) == 1
    return pairs[0][1]

def transition_matrices(dop, path, eps=1e-16):
    r"""
    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, x, Dx = Diffops()
        sage: transition_matrices(Dx - 1, [i/5 for i in range(6)], 1e-10)
        [(0, [1]),
         (1/5, [1.2214027581...]),
         (2/5, [1.491824697...]),
         (3/5, [1.822118800...]),
         (4/5, [2.225540928...]),
         (1, [2.71828182...])]
    """
    ctx = ancont.Context(dop, path, eps, keep="all")
    pairs = ancont.analytic_continuation(ctx)
    return pairs

def _value_from_mat(mat):
    if mat.nrows():
        return mat[0][0]
    else:
        return mat.base_ring().zero()

def eval_diffeq(dop, ini, path, eps=1e-16):
    """
    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, x, Dx = Diffops()

        sage: eval_diffeq(Dx - 1, ini=[1], path=[0,1], eps=1e-50)
        2.7182818284590452353602874713526624977572470936999...

        sage: eval_diffeq((x^2 + 1)*Dx^2 + 2*x*Dx, ini=[0, 1], path=[0, 1+i])
        1.017221967897851... + 0.402359478108525...*I

    TESTS:

    Trivial cases::

        sage: eval_diffeq(Dx, ini=[0], path=[0, 1+i])
        0
        sage: eval_diffeq(Dx, ini=[42], path=[0, 1+i])
        42
        sage: eval_diffeq(Dops(1), ini=[], path=[0, 1+i])
        0
        sage: eval_diffeq(Dops(x+1), ini=[], path=[0, 1+i])
        0

    A recurrence with constant coefficients::

        sage: eval_diffeq(Dx - (x - 1), ini=[1], path=[0, i/30])
        0.998889403147415... - 0.03330865088952795...*I

    """
    ctx = ancont.Context(dop, path, eps)
    pairs = ancont.analytic_continuation(ctx, ini=ini)
    assert len(pairs) == 1
    _, mat = pairs[0]
    return _value_from_mat(mat)

def multi_eval_diffeq(dop, ini, path, eps=1e-16):
    ctx = ancont.Context(dop, path, eps, keep="all")
    pairs = ancont.analytic_continuation(ctx, ini=ini)
    return [(point, _value_from_mat(mat)) for point, mat in pairs]

def polynomial_approximation_on_disk(dop, ini, path, rad, eps):
    raise NotImplementedError

def polynomial_approximation_on_interval(dop, ini, path, rad, eps):
    raise NotImplementedError

def make_proc(xxx): # ??? - ou object DFiniteFunction ?
    pass

def Diffops(sx='x'):
    """
    Return the Ore algebra of differential operators with polynomial
    coefficients over ℚ, along with objects representing, x and d/dx

    EXAMPLE::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, x, Dx = Diffops()
        sage: Dops
        Univariate Ore algebra in Dx over Univariate Polynomial Ring in x over Rational Field
        sage: x*Dx + 1
        x*Dx + 1
    """
    from sage.rings.rational_field import QQ
    from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
    from ore_algebra.ore_algebra import OreAlgebra
    Pol, x = PolynomialRing(QQ, sx).objgen()
    Dop, Dx = OreAlgebra(Pol, 'D' + sx).objgen()
    return Dop, x, Dx
