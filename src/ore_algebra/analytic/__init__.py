# -*- coding: utf-8 - vim: tw=80
r"""
D-finite analytic functions

TODO: add general information about the subpackage here

Basic Usage
===========

EXAMPLES::

    sage: from ore_algebra import OreAlgebra, DifferentialOperators
    sage: Pol.<x> = QQ[]
    sage: Dop.<Dx> = OreAlgebra(Pol)
    sage: QQi.<i> = QuadraticField(-1)

An operator of order 2 annihilating arctan(x) and the constants::

    sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

    sage: dop.numerical_solution([0, 1], [0, 1], 1e-30)
    [0.7853981633974483096156608458198...]

    sage: dop.numerical_transition_matrix([0, 1], 1e-20)
    [  1.0... [0.7853981633974483096...]]
    [       0 [0.5000000000000000000...]]

Display some information on what is going on::

    sage: import logging
    sage: logging.basicConfig()
    sage: logger = logging.getLogger('ore_algebra.analytic')
    sage: logger.setLevel(logging.INFO)
    sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
    sage: dop.numerical_transition_matrix([0, 1], 1e-20)
    INFO:ore_algebra.analytic.analytic_continuation:path: 0 --> 1/2 --> 1
    INFO:ore_algebra.analytic.analytic_continuation:0 --> 1/2: ordinary case
    INFO:ore_algebra.analytic.bounds:bounding local operator...
    ...
    INFO:ore_algebra.analytic.naive_sum:summed ... terms, ...
    ...
    [  1.00...  [0.7853981633974483096...]]
    [         0 [0.5000000000000000000...]]
    sage: logger.setLevel(logging.WARNING)

An operator annihilating `\exp + \arctan`::

    sage: dop = (x+1)*(x^2+1)*Dx^3-(x-1)*(x^2-3)*Dx^2-2*(x^2+2*x-1)*Dx
    sage: dop.numerical_transition_matrix( [0, 1+i], 1e-10)
    [ 1.0... [1.017221967...] + [0.402359478...]*I  [-1.097056...] + [3.76999161...]*I]
    [      0 [0.200000000...] + [-0.40000000...]*I  [2.5373878...] + [5.37471057...]*I]
    [      0 [-0.04000000...] + [0.280000000...]*I  [1.5486939...] + [1.72735528...]*I]

Regular Singular Connection Problems
====================================

Connection to a singular point::

    sage: NF.<sqrt2> = QuadraticField(2)
    sage: dop = (x^2 - 2)*Dx^2 + x + 1
    sage: dop.numerical_transition_matrix([0, 1, sqrt2], 1e-10)
    [ [2.49388...] + [...]*I  [2.40894...] + [...]*I]
    [[-0.20354...] + [...]*I  [0.20437...] + [6.45961...]*I]

This kind of connection matrices linking ordinary points to regular singular
points can be used to compute classical special functions, like Bessel
functions::

    sage: alg = QQbar(-20)^(1/3)
    sage: (x*Dx^2 + Dx + x).numerical_transition_matrix([0, alg], 1e-8)
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

    sage: dop.numerical_solution(ini, path=[0, sqrt(2)])
    [0.46365280236686...]
    sage: CBF(sqrt(2)).ci()
    [0.46365280236686...]

    sage: dop.numerical_solution(ini, path=[0, 456/123*i+1])
    [6.1267878728616...] + [-3.39197789100074...]*I
    sage: CBF(456/123*I + 1).ci()
    [6.126787872861...] + [-3.391977891000...]*I

The slightly less classical Whittaker functions are an interesting test case as
they involve irrational exponents::

    sage: dop = 4*x^2*Dx^2 + (-x^2+8*x-11)
    sage: dop.numerical_transition_matrix([0, 10])
    [[-3.829367993175840...]  [7.857756823216673...]]
    [[-1.135875563239369...]  [1.426170676718429...]]

This one has both algebraic exponents and an algebraic evaluation point::

    sage: alg = NumberField(x^6+86*x^5+71*x^4-80*x^3+2*x^2+7*x+24, 'alg',
    ....:                   embedding=CC(0.6515637 + 0.3731162*I)).gen()
    sage: dop = 4*x^2*Dx^2 + (-x^2+8*x-11)
    sage: dop.numerical_transition_matrix([0, alg])
    [[2.503339393562986...]  + [-0.714903133441901...]*I [0.2144377477885843...] + [0.3310657638490197...]*I]
    [[-0.4755983564143503...] + [2.154602091528463...]*I [0.9461935691709922...] + [0.3918807160953653...]*I]

Another use of “singular” transition matrices is in combinatorics, in relation
with singularity analysis. Here is the constant factor in the asymptotic
expansion of Apéry numbers (compare M. D. Hirschhorn, Estimating the Apéry
numbers, *Fibonacci Quart.* 50, 2012, 129--131), computed as a connection
constant::

    sage: Dops, z, Dz = DifferentialOperators(QQ, 'z')
    sage: dop = (z^2*(z^2-34*z+1)*Dz^3 + 3*z*(2*z^2-51*z+1)*Dz^2
    ....:       + (7*z^2-112*z+1)*Dz + (z-5))
    sage: roots = dop.leading_coefficient().roots(AA)
    sage: roots
    [(0, 2), (0.02943725152285942?, 1), (33.97056274847714?, 1)]
    sage: mat = dop.numerical_transition_matrix([0, roots[1][0]], 1e-10)
    sage: mat.list()
    [[4.846055616...] +        [+/- ...]*I,
     [-3.77845406...] +        [+/- ...]*I,
     [1.473024273...] +        [+/- ...]*I,
            [+/- ...] + [-14.9569783...]*I,
            [+/- ...] +        [+/- ...]*I,
            [+/- ...] + [4.546376247...]*I,
     [-59.9006990...] +        [+/- ...]*I,
     [28.70759161...] +        [+/- ...]*I,
     [-18.2076291...] +        [+/- ...]*I]
    sage: cst = -((1/4)*I)*(1+2^(1/2))^2*2^(3/4)/(pi*(2*2^(1/2)-3))
    sage: mat[1][2].overlaps(CBF(cst))
    True

TESTS:

    sage: import ore_algebra.analytic.polynomial_approximation as pa

Some corner cases::

    sage: (x*Dx + 1).numerical_transition_matrix([0, 1], 1e-10)
    [1.00...]

    sage: (x*Dx + 1).numerical_transition_matrix([0, 0], 1e-10)
    [1.00...]

    sage: dop = x*Dx^3 + 2*Dx^2 + x*Dx
    sage: mat = dop.numerical_transition_matrix([-1, 0, i, -1])
    sage: id = identity_matrix(3)
    sage: all(y.rad() < 1e-13 for row in (mat - id) for y in row)
    True

A recurrence with constant coefficients::

    sage: (Dx - (x - 1)).numerical_solution(ini=[1], path=[0, i/30])
    [0.99888940314741...] + [-0.03330865088952795...]*I

A few larger or harder examples::

    sage: _, z, Dz = DifferentialOperators()

    sage: dop = ((-1/8*z^2 + 5/21*z - 1/4)*Dz^10 + (5/4*z + 5)*Dz^9
    ....:       + (-4*z^2 + 1/17*z)*Dz^8 + (-2/7*z^2 - 2*z)*Dz^7
    ....:       + (z + 2)*Dz^6 + (z^2 - 5/2*z)*Dz^5 + (-2*z + 2)*Dz^4
    ....:       + (1/2*z^2 + 1/2)*Dz^2 + (-3*z^2 - z + 17)*Dz - 1/9*z^2 + 1)
    sage: dop.numerical_transition_matrix([0, 1/2], 1e-8) # TODO: double-check
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

    sage: dop = (z+1)*(3*z^2-z+2)*Dz^3 + (5*z^3+4*z^2+2*z+4)*Dz^2 \
    ....:       + (z+1)*Dz + (4*z^3+2*z^2+5)
    sage: path = [0,-2/5+3/5*i,-2/5+i,-1/5+7/5*i]
    sage: dop.numerical_solution([0,i,0], path, 1e-150) # long time (4.2 s)
    [-1.5598481440603221187326507993405933893413346644879595004537063375459901302359572361012065551669069...] +
    [-0.7107764943512671843673286878693314397759047479618104045777076954591551406949345143368742955333566...]*I

Operators with rational function coefficients::

    sage: dop = (x/x)*Dx - 1
    sage: dop.parent()
    Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field
    sage: dop.numerical_solution([1], [0, 1])
    [2.71828182845904...]
    sage: dop.numerical_transition_matrix([0, 1])
    [[2.71828182845904...]]
    sage: dop.local_basis_monomials(0)
    [1]
    sage: dop.numerical_solution([1], [0,1], 1e-30, algorithm='binsplit')
    [2.7182818284590452353602874713...]
    sage: _ = pa.on_disk(dop, [1], [0], 1, 1e-3)

    sage: ((x/1)*Dx^2 - 1).local_basis_monomials(0)
    [1, x]
    sage: ((x/1)*Dx^2 - 1).numerical_transition_matrix([0, 1])
    [[0.0340875989376363...]   [1.59063685463732...]]
    [[-0.579827135138349...]   [2.27958530233606...]]
    sage: ((x/1)*Dx^2 - 1).numerical_transition_matrix([0, 1], algorithm='binsplit')
    [[0.0340875989376363...]   [1.59063685463732...]]
    [[-0.579827135138349...]   [2.27958530233606...]]
"""

# NOTE: to run the tests, use something like
#
#     SAGE_PATH="$PWD" sage -t --force-lib ore_algebra/analytic/

