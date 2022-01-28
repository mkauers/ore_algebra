# -*- coding: utf-8 - vim: tw=80
"""
Miscellaneous utilities
"""

# Copyright 2015, 2016, 2017, 2018 Marc Mezzarobba
# Copyright 2015, 2016, 2017, 2018 Centre national de la recherche scientifique
# Copyright 2015, 2016, 2017, 2018 Université Pierre et Marie Curie
#
# Copyright 2021 Alexandre Goyer, Inria Saclay Ile-de-France
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

import itertools
from builtins import zip

import sage.rings.complex_arb
import sage.rings.real_arb

from sage.categories.pushout import pushout
from sage.misc.cachefunc import cached_function
from sage.misc.misc import cputime
from sage.rings.all import ZZ, QQ, QQbar, CIF, CBF, RationalField
from sage.rings.real_mpfr import RealField
from sage.rings.number_field.number_field import (NumberField,
        NumberField_quadratic, is_NumberField)
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.matrix.matrix_dense import Matrix_dense
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector, FreeModuleElement_generic_dense
from sage.rings.polynomial.polynomial_element import Polynomial
from sage.rings.qqbar import number_field_elements_from_algebraics
from sage.functions.all import log, floor
from sage.arith.misc import algdep, gcd
from sage.arith.functions import lcm
from sage.functions.other import binomial

from .accuracy import PrecisionError
from .complex_optimistic_field import ComplexOptimisticField

from ore_algebra.ideal import uncouple, solve_triangular_system
from ore_algebra.tools import clear_denominators

######################################################################
# Timing
######################################################################

class Clock(object):
    def __init__(self, name="time"):
        self.name = name
        self._sum = 0.
        self._tic = None
    def __repr__(self):
        return "{} = {}".format(self.name, self.total())
    def since_tic(self):
        return 0. if self._tic is None else cputime(self._tic)
    def total(self):
        return self._sum + self.since_tic()
    def tic(self, t=None):
        assert self._tic is None
        self._tic = cputime() if t is None else t
    def toc(self):
        self._sum += cputime(self._tic)
        self._tic = None

class Stats(object):
    def __repr__(self):
        return ", ".join(str(clock) for clock in self.__dict__.values()
                                    if isinstance(clock, Clock))

######################################################################
# Numeric fields
######################################################################

_RBFmin = sage.rings.real_arb.RealBallField(2)
_CBFmin = sage.rings.complex_arb.ComplexBallField(2)

def is_numeric_parent(parent):
    return _CBFmin.has_coerce_map_from(parent)

def is_real_parent(parent):
    return _RBFmin.has_coerce_map_from(parent)

def is_QQi(parent):
    return (isinstance(parent, NumberField_quadratic)
                and list(parent.polynomial()) == [1,0,1]
                and CBF(parent.gen()).imag().is_one())

def ball_field(eps, real):
    prec = prec_from_eps(eps)
    if real:
        return sage.rings.real_arb.RealBallField(prec)
    else:
        return sage.rings.complex_arb.ComplexBallField(prec)

################################################################################
# Number fields and orders
################################################################################

def good_number_field(nf):
    if isinstance(nf, NumberField_quadratic):
        # avoid denominators in the representation of elements
        disc = nf.discriminant()
        if disc % 4 == 1:
            nf, _, hom = nf.change_generator(nf(nf.discriminant()).sqrt())
            return nf, hom
    return nf, nf.hom(nf)

def as_embedded_number_field_elements(algs):
    try:
        nf, elts, _ = number_field_elements_from_algebraics(algs, embedded=True,
                                                            minimal=True)
    except NotImplementedError: # compatibility with Sage <= 9.3
        nf, elts, emb = number_field_elements_from_algebraics(algs)
        if nf is not QQ:
            nf = NumberField(nf.polynomial(), nf.variable_name(),
                        embedding=emb(nf.gen()))
            elts = [elt.polynomial()(nf.gen()) for elt in elts]
            nf, hom = good_number_field(nf)
            elts = [hom(elt) for elt in elts]
        assert [QQbar.coerce(elt) == alg for alg, elt in zip(algs, elts)]
    return nf, elts

def as_embedded_number_field_element(alg):
    return as_embedded_number_field_elements([alg])[1][0]

def number_field_with_integer_gen(K):
    r"""
    TESTS::

        sage: from ore_algebra.analytic.utilities import number_field_with_integer_gen
        sage: K = NumberField(6*x^2 + (2/3)*x - 9/17, 'a')
        sage: number_field_with_integer_gen(K)[0]
        Number Field in x306a with defining polynomial x^2 + 34*x - 8262 ...
    """
    if K is QQ:
        return QQ, ZZ
    den = K.polynomial().monic().denominator()
    if den.is_one():
        # Ensure that we return the same number field object (coercions can be
        # slow!)
        intNF = K
    else:
        intgen = K.gen() * den
        ### Attempt to work around various problems with embeddings
        emb = K.coerce_embedding()
        embgen = emb(intgen) if emb else intgen
        # Write K.gen() = α = β/q where q = den, and
        # K.polynomial() = q + p[d-1]·X^(d-1) + ··· + p[0].
        # By clearing denominators in P(β/q) = 0, one gets
        # β^d + q·p[d-1]·β^(d-1) + ··· + p[0]·q^(d-1) = 0.
        intNF = NumberField(intgen.minpoly(), "x" + str(den) + str(K.gen()),
                            embedding=embgen)
        assert intNF != K
    intNF, _ = good_number_field(intNF)
    # Work around weaknesses in coercions involving order elements,
    # including #14982 (fixed). Used to trigger #14989 (fixed).
    #return intNF, intNF.order(intNF.gen())
    return intNF, intNF

def invert_order_element(alg):
    if alg in ZZ:
        return 1, alg
    else:
        Order = alg.parent()
        pol = alg.polynomial().change_ring(ZZ)
        modulus = Order.gen(1).minpoly()
        den, num, _ = pol.xgcd(modulus)  # hopefully fraction-free!
        return Order(num), ZZ(den)

def mypushout(X, Y):
    if X.has_coerce_map_from(Y):
        return X
    elif Y.has_coerce_map_from(X):
        return Y
    else:
        Z = pushout(X, Y)
        assert (is_NumberField(Z) if is_NumberField(X) and is_NumberField(Y)
                                  else True)
        return Z

######################################################################
# Sage features
######################################################################

@cached_function
def has_new_ComplexBall_constructor():
    from sage.rings.complex_arb import ComplexBall, CBF
    try:
        ComplexBall(CBF, QQ(1), QQ(1))
    except TypeError:
        return False
    else:
        return True

######################################################################
# Miscellaneous stuff
######################################################################

def prec_from_eps(eps):
    return -eps.lower().log2().floor() + 4

def split(cond, objs):
    matching, not_matching = [], []
    for x in objs:
        (matching if cond(x) else not_matching).append(x)
    return matching, not_matching

def short_str(obj, n=60):
    s = str(obj)
    if len(s) < n:
        return s
    else:
        return s[:n/2-2] + "..." + s[-n/2 + 2:]

# Adapted from itertools manual
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

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

######################################################################
# Relating to ComplexOptimisticField
######################################################################

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

    Note that works also if x is a list or a vector or a matrix (minimum of the
    accuracies of the coefficients).

    INPUT:
     - 'x' - a complex ball

    OUTPUT:
     - 'acc' - a nonnegative integer or +infinity
    """

    if isinstance(x, FreeModuleElement_generic_dense) or \
    isinstance(x, Matrix_dense) or isinstance(x, Polynomial):
        xl = x.list()
        if len(xl)==0: xl = [x[0]] # for zero polynomial
        acc = min(customized_accuracy(c) for c in xl)
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

######################################################################
# Guessing
######################################################################

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



######################################################################
# Eigenring
######################################################################

def RRem(A, B):

    r""" Adapted version of right reminder computation, where A is a list of
    operators [A0, A1, ..., Am] representing the operator A0(a) + A1(a)*D + ...
    + Am(a)*D^m for a generic function a.
    """

    D, m, n = B.parent().gen(), len(A) - 1, B.order()
    if m<n: return A
    qn, Am = B.leading_coefficient(), A[-1]
    A = [ A[i] - (1/qn)*c*Am for i, c in enumerate(D**(m - n)*B) if i<m ]
    while len(A)>1 and A[-1]==0: A.pop()

    return RRem(A, B)

def cleaned_parent(L1, L2):

    r""" Try to find a commun parent, and in the case where the coefficients
    live in QQ, embed it in ZZ.
    """

    OA1, OA2 = L1.parent(), L2.parent()
    if OA1!=OA2:
        commun_parent = False
        try:
            OA1.coerce(L2)
            commun_parent = True
            OA = OA1
        except TypeError: pass
        try:
            OA2.coerce(L1)
            commun_parent = True
            OA = OA2
        except TypeError: pass
        if not commun_parent:
            raise TypeError("L1 and L2 must have the same parent.")
    else: OA = OA1

    R = OA.base_ring().base_ring()
    if isinstance(R, RationalField):
        I = R.ring_of_integers()
        L1, L2 = clear_denominators(L1)[0], clear_denominators(L2)[0]
        L1 = lcm([R(cc).denominator() for c in L1 for cc in c])*L1
        L2 = lcm([R(cc).denominator() for c in L2 for cc in c])*L2
        OA = OA.change_ring(PolynomialRing(I, L1.base_ring().gen().variable_name()))

    L1, L2 = OA(L1), OA(L2)
    return L1, L2, OA


def eigenring(L1, L2=None, infolevel=0):

    r"""
    Compute a basis of the space of the A + DL2 such that L2 divides L1A from
    the right.
    """

    if L2==None: L2 = L1
    m, n = L1.order(), L2.order()
    L1, L2, OA = cleaned_parent(L1, L2)
    D, zero = OA.gen(), OA.zero()

    # computation of the associated matrix N
    L1a = [ sum( binomial(k, i)*L1[k]*D**(k - i) for k in range(i, m + 1) )\
           for i in range(m + 1) ]
    N = [ RRem([zero]*j + L1a, L2) for j in range(n) ]
    N = [ Nj + [zero]*(n - len(Nj)) for Nj in N ]
    N = [ [ N[i][j] for i in range(n) ] for j in range(n) ]

    # solving Na=0
    basis = solve_triangular_system(uncouple(N, infolevel=infolevel), [[0]*n])

    # re-building the associated elements in the eigenring
    output = []
    for u, l in basis:
        op = D.parent().change_ring(D.parent().base_ring().fraction_field())(u)
        if not op.is_zero():
            d = gcd([c for pol in op for c in pol.numerator()])
            output.append((1/d)*op)

    return output


def euler_representation(dop):

    r"""
    Return the list of the coefficients of dop with respect to the powers of
    z*Dz.
    """

    z, n = dop.base_ring().gen(), dop.order()
    output = [ dop[0] ] + [0]*n
    l = [0] # coefficients of T(T-1)...(T-k+1) (initial: k=0)

    for k in range(1, n+1):

        newl = [0]
        for i in range(1, len(l)):
            newl.append((-k+1)*l[i]+l[i-1])
        l = newl + [1]

        ck = dop[k]
        for j in range(1, k+1):
            output[j] += ck*z**(-k)*l[j]

    return output



def dop_valuation(dop):

    r"""
    Return the smallest i such that a_{i,j} != 0 for some j where
    dop = sum a_{i,j} z^i (z*Dz)^j.
    """

    v = min(a.valuation() for a in euler_representation(dop))

    return v


def local_bound_problem(R, L, c):

    r"""
    Compute a lower bound for the valuation of any operator r such that
    RightRemainder(Rr, L) = c, see [van Hoeij, Rational solutions of the mixed
    differential equation..., Lemma 1, 1996].

    INPUT:

     -- "R, L" -- linear differential operators
     -- "c"    -- 0 or 1

    OUTPUT:

     -- "v0r" -- integer or None if it is certified that no such r exists

    """

    z = R.base_ring().gen()

    list_of_possible_v0r = []
    exponentsofL = L.indicial_polynomial(z).roots(QQbar, multiplicities=False)
    exponentsofR = R.indicial_polynomial(z).roots(QQbar, multiplicities=False)
    for eL in exponentsofL:
        for eR in exponentsofR:
            if eL - eR in ZZ:
                list_of_possible_v0r.append(eR - eL)

    if c!=0: list_of_possible_v0r.append(-dop_valuation(R))
    if list_of_possible_v0r==[]: return None
    return min(list_of_possible_v0r)


def common_denominator_for_mixed_equation(R, L, c):

    r"""
    Compute a polynomial D such that for any operator r = \sum r_i * Dz^i
    satisfying RightRemainder(Rr, L) = c, D*r_i is polynomial for all i.

    !!! ne peut pas utiliser PlainDifferentialOperator
    ou LinearDifferentialOperator !!! (circular import)
    """

    z = R.base_ring().gen()

    singL = PlainDifferentialOperator(L)._singularities(QQbar, multiplicities=False)
    singR = PlainDifferentialOperator(R)._singularities(QQbar, multiplicities=False)
    sing = list(set(singL) | set(singR))
    D = 1
    for s in sing:
        if not s in QQ:
            Rs, _ = PlainDifferentialOperator(R).extend_scalars(s)
            Ls, s = PlainDifferentialOperator(L).extend_scalars(s)
            Rs = R.annihilator_of_composition(z + s)
            Ls = L.annihilator_of_composition(z + s)
        else:
            Rs = R.annihilator_of_composition(z + s)
            Ls = L.annihilator_of_composition(z + s)
        v0r = local_bound_problem(Rs, Ls, c)
        if v0r==None: return None
        D = D*(z-s)^(-v0r)

    return D
