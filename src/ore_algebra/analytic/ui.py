# -*- coding: utf-8 - vim: tw=80
"""
Some convenience functions for direct use of the features of this package

Ultimately, the typical way to use it should be through methods of objects such
as differential operators and D-finite functions, not through this module!

EXAMPLES:

As a basic example, we compute exp(1) as the only entry of the transition matrix
from 0 to 1 for the differential equation y' = y::

    sage: from ore_algebra import *
    sage: from ore_algebra.analytic.ui import *

    sage: Pol.<x> = QQ[]
    sage: Dop.<Dx> = OreAlgebra(Pol)
    sage: QQi.<i> = QuadraticField(-1)

An operator of order 2 annihilating arctan(x) and the constants::

    sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

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
    INFO:ore_algebra.analytic.analytic_continuation:path: 0 --> 1/2 --> 1
    INFO:ore_algebra.analytic.analytic_continuation:0 --> 1/2: ordinary case
    INFO:ore_algebra.analytic.bounds:bounding local operator...
    ...
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

This kind of connection matrices linking ordinary points to regular singular
points can be used to compute classical special functions, like Bessel
functions::

    sage: alg = QQbar(-20)^(1/3)
    sage: transition_matrix(x*Dx^2 + Dx + x, [0, alg], 1e-8)
    [ [3.7849872...] +  [1.7263190...]*I  [1.3140884...] + [-2.3112610...]*I]
    [ [1.0831414...] + [-3.3595150...]*I  [-2.0854436...] + [-0.7923237...]*I]

    sage: t = SR.var('t')
    sage: f1 = (ln(2) - euler_gamma - I*pi/2)*bessel_J(0, t) - bessel_K(0, I*t)
    sage: f2 = bessel_J(0, t)
    sage: matrix([[f1, f2], [diff(f1, t), diff(f2, t)]]).subs(t=alg.n()).n()
    [ 3.7849872... + 1.7263190...*I    1.3140884... - 2.3112610...*I]
    [ 1.0831414... - 3.3595150...*I   -2.0854436... - 0.7923237...*I]

or the cosine integral::

    sage: dop = x*Dx^3 + 2*Dx^2 + x*Dx
    sage: ini = [1, CBF(euler_gamma), 0]

    sage: eval_diffeq(dop, ini, path=[0, sqrt(2)])
    [0.46365280236686...]
    sage: CBF(sqrt(2)).ci()
    [0.46365280236686...]

    sage: eval_diffeq(dop, ini, path=[0, 456/123*i+1])
    [6.1267878728616...] + [-3.39197789100074...]*I
    sage: CBF(456/123*I + 1).ci()
    [6.126787872861...] + [-3.391977891000...]*I

The slightly less classical Whittaker functions are an interesting test case as
they involve irrational exponents. The following example was checked against
NumGfun::

    sage: transition_matrix(4*x^2*Dx^2 + (-x^2+8*x-11), [0, 10])
    [[-3.829367993175840...]  [7.857756823216673...]]
    [[-1.135875563239369...]  [1.426170676718429...]]

This one (checked against NumGfun too) has both algebraic exponents and an
algebraic evaluation point::

    sage: alg = NumberField(x^6+86*x^5+71*x^4-80*x^3+2*x^2+7*x+24, 'alg', embedding=CC(0.6515637 + 0.3731162*I)).gen()
    sage: transition_matrix(4*x^2*Dx^2 + (-x^2+8*x-11), [0, alg])
    [[2.503339393562986...]  + [-0.714903133441901...]*I [0.2144377477885843...] + [0.3310657638490197...]*I]
    [[-0.4755983564143503...] + [2.154602091528463...]*I [0.9461935691709922...] + [0.3918807160953653...]*I]

Another use of “singular” transition matrices is in combinatorics, in relation
with singularity analysis. Here is the constant factor in the asymptotic
expansion of Apéry numbers (compare M. D. Hirschhorn, Estimating the Apéry
numbers, *Fibonacci Quart.* 50, 2012, 129--131), computed as a connection
constant::

    sage: Dops, z, Dz = Diffops("z")
    sage: dop = (z^2*(z^2-34*z+1)*Dz^3 + 3*z*(2*z^2-51*z+1)*Dz^2
    ....:       + (7*z^2-112*z+1)*Dz + (z-5))
    sage: roots = dop.leading_coefficient().roots(AA)
    sage: roots
    [(0, 2), (0.02943725152285942?, 1), (33.97056274847714?, 1)]
    sage: mat = transition_matrix(dop, [0, roots[1][0]], 1e-10); mat
    [   [4.84605561...]   [-3.77845406...]    [1.47302427...]]
    [[-14.9569783...]*I        [+/- ...]*I  [4.54637624...]*I]
    [  [-59.9006990...]    [28.7075916...]   [-18.2076291...]]
    sage: cst = -((1/4)*I)*(1+2^(1/2))^2*2^(3/4)/(pi*(2*2^(1/2)-3))
    sage: mat[1][2].overlaps(CBF(cst))
    True

An example kindly provided by Christoph Koutschan::

    sage: Dops, a, Da = Diffops("a")
    sage: dop = ((1315013644371957611900*a^2+263002728874391522380*a+13150136443719576119)*Da^3
    ....: + (2630027288743915223800*a^2+16306169190212274387560*a+1604316646133788286518)*Da^2
    ....: + (1315013644371957611900*a^2-39881765316802329075320*a+35449082663034775873349)*Da
    ....: + (-278967152068515080896550+6575068221859788059500*a))
    sage: ini = [5494216492395559/3051757812500000000000000000000,
    ....:        6932746783438351/610351562500000000000000000000,
    ....:        1/2 * 1142339612827789/19073486328125000000000000000]
    sage: eval_diffeq(dop, list(ini), [0, 84])
    [0.011501537469552017...]


TESTS::

    sage: transition_matrix(x*Dx + 1, [0, 1], 1e-10)
    [1.00...]

    sage: transition_matrix(x*Dx + 1, [0, 0], 1e-10)
    [1.00...]

    sage: mat = (transition_matrix(x*Dx^3 + 2*Dx^2 + x*Dx, [-1, 0, i, -1])
    ....:       - identity_matrix(3))
    sage: all(y.rad() < 1e-13 for row in mat for y in row)
    True
"""

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
        [ 1.0... [1.017221967...] + [0.402359478...]*I  [-1.097056...] + [3.76999161...]*I]
        [      0 [0.200000000...] + [-0.40000000...]*I  [2.5373878...] + [5.37471057...]*I]
        [      0 [-0.04000000...] + [0.280000000...]*I  [1.5486939...] + [1.72735528...]*I]

    TESTS::

        sage: Dops, z, Dz = Diffops()
        sage: dop = ((-1/8*z^2 + 5/21*z - 1/4)*Dz^10 + (5/4*z + 5)*Dz^9
        ....:       + (-4*z^2 + 1/17*z)*Dz^8 + (-2/7*z^2 - 2*z)*Dz^7
        ....:       + (z + 2)*Dz^6 + (z^2 - 5/2*z)*Dz^5 + (-2*z + 2)*Dz^4
        ....:       + (1/2*z^2 + 1/2)*Dz^2 + (-3*z^2 - z + 17)*Dz - 1/9*z^2 + 1)
        sage: transition_matrix(dop, [0, 1/2], 1e-8) # TODO: double-check
        [ [1.000000...] [0.5000001...] [0.2500000...] [0.1250000...] [0.06250034...] [0.03124997...] [0.01563582...] [0.007812569...] [0.003902495...] [0.0154870...]]
        [ [2.0684...e-7...] [1.000003...] [1.000000...] [0.7500000...] [0.5000095...] [0.3124993...] [0.1878035...] [0.1093771...] [0.06238342...] [0.4144342...]]
        [ [2.870591...e-6...] [4.878849...e-5...] [1.000007...] [1.500000...] [1.500131...] [1.249991...] [0.9417146...] [0.6562816...] [0.4357331...] [5.546944...]]
        [ [2.669703...e-5...] [0.0004537350...] [6.757877...e-5...] [1.000008...] [2.001225...] [2.499915...] [2.539217...] [2.187812...] [1.732392...] [50.29996...]]
        [[0.0001897787...] [0.003225386...] [0.0004861918...] [6.478386...e-5...] [1.008705...] [2.499387...] [4.028872...] [4.377308...] [4.243648...] [352.2441...]]
        [ [0.001111382...] [0.01888838...] [0.002866610...] [0.0003872062...] [0.05095480...] [0.9963641...] [4.633435...] [5.263837...] [6.208066...] [2047.872...]]
        [ [0.005615658...] [0.09543994...] [0.01452990...] [0.001976353...] [0.2574113...] [-0.01848697...] [9.254196...] [3.570713...] [2.939124...] [10318.86...]]
        [ [0.02520549...] [0.4283749...] [0.06529065...] [0.008906705...] [1.155284...] [-0.08317410...] [37.04945...] [1.318793...] [-14.33800...] [46278.27...]]
        [ [0.1024054...] [1.740410...] [0.2653474...] [0.03623250...] [4.693620...] [-0.3381535...] [150.5263...] [1.296916...] [-73.65196...] [187989.1...]]
        [ [0.3815220...] [6.484077...] [0.9886403...] [0.1350288...] [17.48650...] [-1.260009...] [560.8023...] [4.833181...] [-278.2608...] [700357.9...]]
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

    TESTS:

    A recurrence with constant coefficients::

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, x, Dx = Diffops()
        sage: eval_diffeq(Dx - (x - 1), ini=[1], path=[0, i/30])
        [0.99888940314741...] + [-0.03330865088952795...]*I

    Some harder examples::

        sage: Dops, z, Dz = Diffops('z')
        sage: dop = (z+1)*(3*z^2-z+2)*Dz^3 + (5*z^3+4*z^2+2*z+4)*Dz^2 \
        ....:       + (z+1)*Dz + (4*z^3+2*z^2+5)
        sage: QQ.<i> = QuadraticField(-1, 'i')
        sage: path = [0,-2/5+3/5*i,-2/5+i,-1/5+7/5*i]
        sage: eval_diffeq(dop, [0,i,0], path, 1e-150) # long time (4.2 s)
        [-1.5598481440603221187326507993405933893413346644879595004537063375459901302359572361012065551669069...] +
        [-0.7107764943512671843673286878693314397759047479618104045777076954591551406949345143368742955333566...]*I

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

def make_proc(xxx): # ???
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
