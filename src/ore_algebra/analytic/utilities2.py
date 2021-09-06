# -*- coding: utf-8 - vim: tw=80
"""
Some useful functions (relating to ComplexOptimisticField, guessing, ...)
"""

# Copyright 2021 Alexandre Goyer, Inria Saclay Ile-de-France
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

from sage.rings.real_mpfr import RealField
from sage.rings.rational_field import QQ
from sage.rings.qqbar import QQbar
from sage.matrix.matrix_dense import Matrix_dense
from sage.modules.free_module_element import FreeModuleElement_generic_dense
from sage.rings.polynomial.polynomial_element import Polynomial

from sage.rings.qqbar import number_field_elements_from_algebraics

from sage.matrix.constructor import matrix
from sage.functions.other import floor
from sage.functions.log import log
from sage.arith.misc import algdep

from .accuracy import PrecisionError
from .complex_optimistic_field import ComplexOptimisticField


def overlaps(a, b):
    """
    Function overlaps designed for vectors and matrices.
    """
    l = len(a.list())
    return all(a.list()[i].overlaps(b.list()[i]) for i in range(l))

def customized_accuracy(x):

    """
    Return either the absolute accuracy of x if x contains 0 or the relative
    accuracy of x if x does not contains 0.

    Note that works also if x is a list or a vector or a matrix (minimal
    accuracy of the coefficients).

    INPUT:
     - 'x' - a complex ball

    OUTPUT:
     - 'acc' - a nonnegative integer
    """

    if isinstance(x, FreeModuleElement_generic_dense) or \
    isinstance(x, Matrix_dense) or isinstance(x, Polynomial):
        x = x.list()
        acc = min(customized_accuracy(c) for c in x)
        return acc

    if isinstance(x, list):
        acc = min(customized_accuracy(c) for c in x)
        return acc

    if x.contains_zero() :
        acc = -log(x.rad(), 2)
        if x.rad()!=0: acc = int(acc)
    else:
        acc = x.accuracy()

    return acc


def power_series_coerce(x, S):

    if isinstance(x, list):
        return [power_series_coerce(y, S) for y in x]

    result = S.zero()
    for c, mon in x:
        if c!=0:
            result += c*S.gen()**mon.n

    return result

def derivatives(f, m):

    result = [f]
    for k in range(m):
        f = f.derivative()
        result.append(f)

    return result



#######################
##### Polynomials #####
#######################


def _clean(pol):

    l = list(pol)
    while len(l)>0 and l[-1].contains_zero(): l.pop()
    cpol = pol.parent()(l)

    return cpol



def GCD(a, b):

    r"""
    Return a *non-rigorous* gcd of the polynomials "a" and "b".

    Note: this function is designed for BallField as base ring.

    Some words about the correction of this function:
    Let a· and b· be fixed. If a and b are precise enough, GCD(a, b) contains
    the gcd of a and b.

    INPUT:

     -- "a" -- polynomial
     -- "b" -- polynomial

    OUTPUT:

     -- "a" -- polynomial


    EXAMPLE::

        sage: from diffop_factorization.polynomials import GCD
        sage: P.<x> = CBF[]; a = CBF(pi)
        sage: p, q = (x-1)*(x-2)**2, (x-2)*(x-3)**2
        sage: p, q = p(x*a), q(x*a)
        sage: d = GCD(p, q); d(x/a).monic()
        ([1.0000000000 +/- 1.34e-12])*x + [-2.00000000000 +/- 1.94e-12]

    """

    a, b = _clean(a), _clean(b)
    if a==0: return b
    if b==0: return a
    if a.degree() < b.degree(): return GCD(b, a)

    while b != 0:
        a, b = b, a.quo_rem(b)[1]
        b = _clean(b)

    return a



def XGCD(a, b):

    r"""
    Return the *non-rigorous* monic gcd of the polynomials "a" and "b" and the
    coefficients in the Bezout identity.

    Note: this function is designed for BallField as base ring.

    Some words about the correction of this function:
    Let a· and b· be fixed.

    INPUT:

     -- "a" -- polynomial
     -- "b" -- polynomial

    OUTPUT:

     -- "d" -- polynomial
     -- "u" -- polynomial
     -- "v" -- polynomial


    EXAMPLE::

        sage: from diffop_factorization.polynomials import XGCD, _clean
        sage: P.<x> = CBF[]; a = CBF(pi)
        sage: p, q = (x-1)*(x-2)**2, (x-2)*(x-3)**2
        sage: p, q = p(x*a), q(x*a)
        sage: d, u, v = XGCD(p, q); d, 2/a
        (([1.0000000000 +/- 1.33e-12])*x + [-0.63661977237 +/- 3.04e-12],
         [0.636619772367581 +/- 4.28e-16])
        sage: _clean(u*p + v*q)
        ([1.000000000 +/- 3.39e-11])*x + [-0.63661977237 +/- 6.95e-12]

    """

    P = a.parent()

    a, b = _clean(a), _clean(b)
    if a==0: return b, P.zero(), P.one()
    if b==0: return a, P.one(), P.zero()
    if a.degree() < b.degree():
        d, v, u = XGCD(b, a)
        return d, u, v

    r0, u0, v0, r1, u1, v1 = a, P.one(), P.zero(), b, P.zero(), P.one()
    while r1!=0:
        r0, (q, r1) = r1, r0.quo_rem(r1)
        u0, v0, u1, v1 = u1, v1, u0 - q*u1, v0 - q*v1
        r1 = _clean(r1)

    lc = r0.leading_coefficient()
    d, u, v = r0.monic(), _clean(u0/lc), _clean(v0/lc)

    return d, u, v



def radical(pol):

    r"""
    Return the *non-rigorous* radical of the polynomial "pol".

    Note: this function is designed for BallField as base ring.

    Some words about the correction of this function:
    Let pol· be fixed. If pol is precise enough, radical(pol)) contains the
    radical of pol·.


    INPUT:

     -- "pol" -- polynomial


    OUTPUT:

     -- "rad" -- polynomial


    EXAMPLE::

        sage: from diffop_factorization.polynomials import radical
        sage: P.<x> = CBF[]; a = CBF(pi)
        sage: p = (x-1)*(x-2)**2
        sage: p = p(x*a).monic(); p
        ([1.0000000000000 +/- 1.57e-15])*x^3 + ([-1.59154943091895 +/- 5.18e-15])*x^2 + ([0.81056946913870 +/- 2.96e-15])*x + [-0.129006137732798 +/- 1.12e-16]
        sage: rad = radical(p); rad
        ([-44.4132198049 +/- 2.39e-11])*x^2 + ([42.4115008235 +/- 7.69e-11])*x + [-9.0000000000 +/- 3.38e-11]
        sage: rad.roots(multiplicities=False)
        [[0.3183098862 +/- 2.31e-11] + [+/- 6.86e-12]*I,
         [0.6366197724 +/- 4.20e-11] + [+/- 9.55e-12]*I]
        sage: [1/a, 2/a]
        [[0.318309886183791 +/- 4.43e-16], [0.636619772367581 +/- 4.28e-16]]

    """

    d = GCD(pol, pol.derivative())
    rad = _clean(pol.quo_rem(d)[0])

    if rad==0:
        raise PrecisionError("Cannot compute the radical of this polynomial.")

    return rad



def roots(pol, *, multiplicities=False):

    r"""
    Return the roots of the polynomial "pol".

    Note: this function is designed for CBF or COF as base ring.

    Some words about the correction of this algorithm:


    INPUT:

     -- "mat"            -- n×n matrix
     -- "multiplicities" -- boolean


    OUTPUT:

     -- "s" -- list of complex numbers

    If 'multiplicities=True' is specified, "s" is a list of couples (r, m) with
    r a complex number and m a positive integer.


    EXAMPLE::

    """

    K = pol.base_ring()
    if isinstance(K, ComplexOptimisticField):
        pol = pol.change_ring(K._ball_field)

    try:
        res = radical(pol).roots(multiplicities=False)
        res = [K(r) for r in res]
    except ValueError:
        raise PrecisionError("Cannot compute the roots of this polynomial.") from None

    if not multiplicities: return res

    n = pol.degree()
    derivatives = [pol]
    for i in range(n):
        pol = pol.derivative()
        derivatives.append(pol)

    for j, ev in enumerate(res):
        m = 1
        evaluations = [p(ev) for p in derivatives]
        while evaluations[m].contains_zero():
            m = m + 1
        res[j] = (ev, m)

    if sum(m for _, m in res)<n:
        raise PrecisionError("Cannot compute multiplicities.")

    return res



####################
##### Guessing #####
####################



def hp_approximants (F, d):

    r"""
    Return the Hermite-Padé approximants of F at order d.

    Let F = [f1, ..., fm]. This function returns a list of polynomials P =
    [p1, ..., pm] such that:
    - max(deg(p1), ..., deg(pm)) is minimal,
    - p1*f1 + ... + pm*fm = O(x^d).

    Note that this function calls some methods of the Library of Polynomial
    Matrices, see https://github.com/vneiger/pml to install it (if necessary).

    INPUT:
     - "F" - a list of polynomials or series

    OUTPUT:
     - "P" - a list of polynomials

    EXAMPLES::

        sage: from diffop_factorization.guessing_tools import hp_approx
        sage: f = taylor(log(1+x), x, 0, 8).series(x).truncate().polynomial(QQ); f
        -1/8*x^8 + 1/7*x^7 - 1/6*x^6 + 1/5*x^5 - 1/4*x^4 + 1/3*x^3 - 1/2*x^2 + x
        sage: F = [f, f.derivative(), f.derivative().derivative()]
        sage: P = hp_approximants(F, 5); P
        (0, 1, x + 1)
        sage: from ore_algebra import OreAlgebra
        sage: Pol.<x> = QQ[]; OA.<Dx> = OreAlgebra(Pol)
        sage: diffop = OA(list(P)); diffop
        (x + 1)*Dx^2 + Dx
        sage: diffop(log(1+x))
        0

    """

    try:
        F = [f.truncate() for f in F]
    except: pass

    mat = matrix(len(F), 1, F)
    basis = mat.minimal_approximant_basis(d)
    rdeg = basis.row_degrees()
    i = min(range(len(rdeg)), key = lambda i: rdeg[i])

    return list(basis[i])



def guess_rational_numbers(x, p=None):

    r"""
    Guess rational coefficients for a vector or a matrix or a polynomial or a
    list or just a complex number.

    Note: this function is designed for ComplexOptimisticField as base ring.

    INPUT:
     - 'x' - object with approximate coefficients

    OUTPUT:
     - 'r' - object with rational coefficients

    EXAMPLES::

        sage: from diffop_factorization.complex_optimistic_field import ComplexOptimisticField
        sage: from diffop_factorization.guessing import guess_rational
        sage: C = ComplexOptimisticField(30, 2^-10)
        sage: a = 1/3 - C(1+I)*C(2^-20)
        sage: P.<x> = C[]; pol = (1/a)*x + a; pol
        ([3.0000086 +/- 2.86e-8] + [8.5831180e-6 +/- 7.79e-14]*I)*x + [0.333332379 +/- 8.15e-10] - [9.53674316e-7 +/- 4.07e-16]*I
        sage: guess_rational(pol)
        3*x + 1/3

    """

    if isinstance(x, list) :
        return [guess_rational_numbers(c, p=p) for c in x]

    if isinstance(x, FreeModuleElement_generic_dense) or isinstance(x, Matrix_dense) or isinstance(x, Polynomial):
        return x.parent().change_ring(QQ)(guess_rational_numbers(x.list(), p=p))

    if p is None:
        eps = x.parent().eps
        p = floor(-log(eps, 2))
    else:
        eps = RealField(30).one() >> p
    if not x.imag().above_abs().mid()<eps:
        raise PrecisionError('This number does not seem a rational number.')
    x = x.real().mid()

    return x.nearby_rational(max_error=x.parent()(eps))



def guess_algebraic_numbers(x, d=2, p=None):

    r"""
    Guess algebraic coefficients for a vector or a matrix or a polynomial or a
    list or just a complex number.

    INPUT:
     - 'x' - an object with approximate coefficients
     - 'p' - a positive integer
     - 'd' - a positive integer

    OUTPUT:
     - 'a' - an object with algebraic coefficients

    EXAMPLES::

        sage: from diffop_factorization.guessing_tools import guess_algebraic
        sage: a = CC(sqrt(2))
        sage: guess_algebraic(a)
        1.414213562373095?
        sage: _.minpoly()
        x^2 - 2

    """

    if isinstance(x, list) :
        return [guess_algebraic_numbers(c, d=d, p=p) for c in x]

    if isinstance(x, FreeModuleElement_generic_dense) or \
    isinstance(x, Matrix_dense) or isinstance(x, Polynomial):
        return x.parent().change_ring(QQbar)(guess_algebraic_numbers(x.list(), p=p, d=d))

    if p is None: p = floor(-log(x.parent().eps, 2))

    pol = algdep(x.mid(), degree=d, known_bits=p)
    roots = pol.roots(QQbar, multiplicities=False)
    i = min(range(len(roots)), key = lambda i: abs(roots[i] - x.mid()))

    return roots[i]


def guess_exact_numbers(x, d=1):

    if d==1: return guess_rational_numbers(x)
    return guess_algebraic_numbers(x, d=d)
