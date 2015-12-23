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
    [[2.718281828...]]

    sage: transition_matrix(Dx - 1, [0, i], 1e-10)
    [[0.540302305...] + [0.8414709848...]*I]


    sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

    sage: eval_diffeq(dop, [0, 1], [0, 1+i], 1e-30)
    [1.01722196789785136772278896...] + [0.40235947810852509365018983...]*I

    sage: transition_matrix(dop, [0, 1+i], 1e-10)
    [ 1.00...  [1.017221967...] + [0.402359478...]*I]
    [       0 [0.200000000...] + [-0.400000000...]*I]

Return the values (resp. transition matrices) corresponding to several points
along the path at once::

    sage: multi_eval_diffeq(dop, [0, 1], [k/5 for k in range(5)], 1e-10)
    [(0, 0),
    (1/5, [0.197395559...]),
    (2/5, [0.380506377...]),
    (3/5, [0.540419500...]),
    (4/5, [0.674740942...])]

    sage: tms = transition_matrices(dop, [k/5 for k in range(5)], 1e-10)
    sage: tms[2]
    (
         [ 1.0... [0.3805063771...]]
    2/5, [      0 [0.8620689655...]]
    )

Display some information on what is going on::

    sage: import logging
    sage: logging.basicConfig()
    sage: logging.getLogger('ore_algebra.analytic').setLevel(logging.INFO)
    sage: transition_matrix(dop, [0, 1], 1e-20)
    INFO:ore_algebra.analytic.analytic_continuation:0 --> 1/2: ordinary case
    INFO:ore_algebra.analytic.naive_sum:target error = ...
    INFO:ore_algebra.analytic.naive_sum:summed ... terms, ...
    ...
    [  1.00...  [0.7853981633974483096...]]
    [         0 [0.5000000000000000000...]]
    sage: logging.getLogger('ore_algebra.analytic').setLevel(logging.WARNING)

Connection to a singular point::

    sage: NF.<sqrt2> = QuadraticField(2)
    sage: transition_matrix((x^2 - 2)*Dx^2 + x + 1, [0, 1, sqrt2], 1e-10)
    [ [2.49388...] + [...]*I  [2.40894...] + [...]*I]
    [[-0.20354...] + [...]*I  [0.20437...] + [6.45961...]*I]

Empty paths are not allowed::

    sage: transition_matrix(Dx - 1, path=[])
    Traceback (most recent call last):
    ...
    ValueError: empty path

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
    Real ball field with 3... bits precision

    sage: Dx_C = OreAlgebra(QQi['x'], 'Dx').gen()
    sage: transition_matrix(Dx_C, [0, 1], 1e-10)[0,0].parent()
    Complex ball field with 3... bits precision
"""

# NOTE: to run the tests, use something like
#
#     SAGE_PATH="$PWD" sage -t ore_algebra/analytic/ui.py

from ore_algebra.analytic import analytic_continuation as ancont
from ore_algebra.analytic import polynomial_approximation as polapprox

def transition_matrix(dop, path, eps=1e-16):
    r"""
    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, x, Dx = Diffops()
        sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

        sage: transition_matrix(dop, [0, 1], 1e-20)
        [  1.0... [0.7853981633974483096...]]
        [       0 [0.5000000000000000000...]]

        sage: transition_matrix(dop, [0, 1+i], 1e-10)
        [  1.0... [1.017221967...] + [0.402359478...]*I]
        [       0 [0.200000000...] + [-0.40000000...]*I]

    An operator annihilating `\exp + \arctan`::

        sage: transition_matrix(
        ....:       (x+1)*(x^2+1)*Dx^3-(x-1)*(x^2-3)*Dx^2-2*(x^2+2*x-1)*Dx,
        ....:       [0, 1+i], 1e-10)
        [ 1.0... [1.017221967...] + [0.402359478...]*I  [-1.0970560...] + [3.76999161...]*I]
        [      0 [0.200000000...] + [-0.40000000...]*I  [2.53738788...] + [5.37471057...]*I]
        [      0 [-0.04000000...] + [0.280000000...]*I  [1.54869394...] + [1.72735528...]*I]

    TESTS::

        sage: transition_matrix(dop, [0, 0])
        [1.00...        0]
        [      0  1.00...]
        sage: transition_matrix(Dops(1), [0, 1])
        []
        sage: transition_matrix(Dx, [0, 1])
        [1.00...]
        sage: transition_matrix(Dx - 1, path=[1/3])
        [1.00...]
        sage: transition_matrix(Dx - 1, path=[1, 0])
        [[0.3678794411714423...]]
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
        [(0,   [1.000000000...]),
         (1/5, [[1.221402758...]]),
         (2/5, [[1.491824697...]]),
         (3/5, [[1.822118800...]]),
         (4/5, [[2.225540928...]]),
         (1,   [[2.718281828...]])]
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
        [2.7182818284590452353602874713526624977572470936999...]

        sage: eval_diffeq((x^2 + 1)*Dx^2 + 2*x*Dx, ini=[0, 1], path=[0, 1+i])
        [1.017221967897851...] + [0.402359478108525...]*I

        sage: eval_diffeq(Dx - 1, ini=[], path=[0, 1])
        Traceback (most recent call last):
        ...
        ValueError: incorrect initial values: []

    TESTS:

    Trivial cases::

        sage: eval_diffeq(Dx, ini=[0], path=[0, 1+i])
        0
        sage: eval_diffeq(Dx, ini=[42], path=[0, 1+i])
        42.00...
        sage: eval_diffeq(Dops(1), ini=[], path=[0, 1+i])
        0
        sage: eval_diffeq(Dops(x+1), ini=[], path=[0, 1+i])
        0
        sage: eval_diffeq(Dx - 1, ini=[42], path=[1])
        42.00...

    A recurrence with constant coefficients::

        sage: eval_diffeq(Dx - (x - 1), ini=[1], path=[0, i/30])
        [0.99888940314741...] + [-0.03330865088952795...]*I

    Some harder examples::

        sage: Dops, z, Dz = Diffops('z')
        sage: dop = (z+1)*(3*z^2-z+2)*Dz^3 + (5*z^3+4*z^2+2*z+4)*Dz^2 \
        ....:       + (z+1)*Dz + (4*z^3+2*z^2+5)
        sage: QQ.<i> = QuadraticField(-1, 'i')
        sage: path = [0,-2/5+3/5*i,-2/5+i,-1/5+7/5*i]
        sage: eval_diffeq(dop, [0,i,0], path, 1e-100) # long time
        [-1.55984814406032211873265079934059338934133466448795950045370633754599013023595723610120655516690697...] +
        [-0.71077649435126718436732868786933143977590474796181040457770769545915514069493451433687429553335665...]*I

    Errors::

        sage: eval_diffeq(Dx - 1, ini=[1, 2], path=[0, 1])
        Traceback (most recent call last):
        ...
        ValueError: incorrect initial values: [1, 2]
        sage: eval_diffeq(Dx - 1, ini=["a"], path=[0, 1])
        Traceback (most recent call last):
        ...
        ValueError: incorrect initial values: ['a']

    """
    ctx = ancont.Context(dop, path, eps)
    pairs = ancont.analytic_continuation(ctx, ini=ini)
    assert len(pairs) == 1
    _, mat = pairs[0]
    return _value_from_mat(mat)

def multi_eval_diffeq(dop, ini, path, eps=1e-16):
    """
    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, x, Dx = Diffops()
        sage: QQi.<i> = QuadraticField(-1, 'I')

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
    ctx = ancont.Context(dop, path, eps, keep="all")
    pairs = ancont.analytic_continuation(ctx, ini=ini)
    return [(point, _value_from_mat(mat)) for point, mat in pairs]

polynomial_approximation_on_disk = polapprox.on_disk
polynomial_approximation_on_interval = polapprox.on_interval

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
