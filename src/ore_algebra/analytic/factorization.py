# -*- coding: utf-8 - vim: tw=80
"""
Symbolic-numeric algorithm for the factorization of linear differential
operators.
"""

# Copyright 2021 Alexandre Goyer, Inria Saclay Ile-de-France
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

import collections

from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.real_mpfr import RealField
from sage.rings.qqbar import QQbar
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.polynomial.polynomial_element import Polynomial
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.matrix.special import block_matrix, identity_matrix, diagonal_matrix
from sage.misc.misc_c import prod
from sage.arith.functions import lcm
from sage.functions.other import binomial, factorial
from sage.arith.misc import valuation, gcd
from sage.misc.misc import cputime
from sage.plot.line import line2d

import cProfile
import pstats
from sage.misc.functional import numerical_approx

from ore_algebra.guessing import guess

from .monodromy import _monodromy_matrices
from .differential_operator import PlainDifferentialOperator
from .accuracy import PrecisionError
from .complex_optimistic_field import ComplexOptimisticField
from .utilities import (customized_accuracy, power_series_coerce, derivatives,
                        hp_approximants, guess_exact_numbers,
                        guess_rational_numbers, guess_algebraic_numbers,
                        euler_representation)
from .linear_algebra import (invariant_subspace, row_echelon_form,
                             gen_eigenspaces, orbit)


Radii = RealField(30)

MonoData = collections.namedtuple("MonoData", ["precision", "matrices", "points", "loss"])
NewtonEdge = collections.namedtuple("NewtonEdge", ["slope", "startpoint", "length", "polynomial"])

class LinearDifferentialOperator(PlainDifferentialOperator):

    r"""
    A subclass of linear differential operators for internal use.
    Assumptions: polynomial coefficients and 0 is an ordinary point.
    """

    def __init__(self, dop):

        if not dop:
            raise ValueError("operator must be nonzero")
        if not dop.parent().is_D():
            raise ValueError("expected an operator in K(x)[D]")
        _, _, _, dop = dop.numerator()._normalize_base_ring()
        den = lcm(c.denominator() for c in dop)
        dop *= den
        super(LinearDifferentialOperator, self).__init__(dop)

        self.z = self.base_ring().gen()
        self.Dz = self.parent().gen()
        self.n = self.order()

        self.order_of_truncation = max(100, 2*self.degree() + self.n)
        self.algebraicity_degree = self.base_ring().base_ring().degree()
        self.precision = 100 #100*self.algebraicity_degree

        self.monodromy_data = MonoData(0, [], None, 0)

        self.fuchsian_info = None


    def is_fuchsian(self):

        r"""
        Return True if "self" is fuchian, False otherwise.

        Fuch's criterion: p is a regular point of a_n*Dz^n + ... + a_0 (with a_i
        polynomial) iff no (z-p)^{n-k}*a_k/a_n admits p as pole.
        """

        coeffs = self.coefficients()
        fac = coeffs.pop().factor()
        for (f, m) in fac:
            for k, ak in enumerate(coeffs):
                mk = valuation(ak, f)
                if mk - m < k - self.n: return False

        dop = self.annihilator_of_composition(1/self.z)
        for k, frac in enumerate(dop.monic().coefficients()[:-1]):
            d = (self.z**(self.n - k)*frac).denominator()
            if d(0)==0: return False

        return True


    def monodromy(self, precision, verbose=False):

        r"""
        Compute a generating set of matrices for the monodromy group of "self"
        at 0, such that the (customized) precision of each coefficient is at
        least equal to "precision".
        """

        if verbose: print("Monodromy computation with wanted precision = " + str(precision) + ".")
        if self.monodromy_data.precision<precision:
            success, increment, loss = False, 50, self.monodromy_data.loss
            if self.monodromy_data.points==None:
                useful_singularities = LinearDifferentialOperator(self.desingularize())._singularities(QQbar)
            else:
                useful_singularities = self.monodromy_data.points
            while not success:
                try:
                    p = precision + loss + increment
                    if verbose: print("Try with precision = " + str(p) + ".")
                    it = _monodromy_matrices(self, 0, eps=Radii.one()>>p, sing=useful_singularities)
                    points, matrices = [], []
                    for pt, mat, is_scalar in it:
                        if not is_scalar: matrices.append(mat); points.append(pt)
                    output_precision = min(min([customized_accuracy(mat.list()) for mat in matrices], default=p), p)
                    if output_precision<precision:
                        if verbose: print("Insufficient precision, loss = " + str(p - output_precision) + ".")
                        increment = 50 if loss==0 else increment<<1
                    else: success=True
                    loss = max(loss, p - output_precision)
                except (ZeroDivisionError, PrecisionError):
                    if verbose: print("Insufficient precision for computing monodromy.")
                    increment = increment<<1
            self.monodromy_data =  MonoData(output_precision, matrices, points, loss)


    def _symbolic_guessing(self, v=None):

        r"""
        Return a non-trivial right factor thanks to a *rational* oracle that
        indicates a good linear combination of the solutions of "self" at 0,
        that is, a solution annihilated by an operator of smaller order.
        """

        if v==None: v = vector([1] + [0]*(self.n - 1))
        T = self.order_of_truncation
        R = self.base_ring().base_ring()

        while True:

            #S = PowerSeriesRing(R, default_prec=T + 1)
            #basis = self.local_basis_expansions(QQ.zero(), T + 1)
            #f = power_series_coerce(v*vector(basis), S)
            basis = self.power_series_solutions(T + 4); basis.reverse()
            f = v*vector(basis)
            pols = hp_approximants([f, f.derivative()], T)
            dop = self.parent()(pols)
            if self%dop==0: return dop
            T = T<<1


    def _guessing(self, v, m):

        r"""
        Return a non-trivial right factor thanks to an oracle that indicates a
        good linear combination of the solutions of "self" at 0, that is, a
        solution annihilated by an operator of smaller order (=m).
        """

        T0, T, d0 = 25, self.order_of_truncation, 0
        p = customized_accuracy(v)
        if p<50: raise PrecisionError("Loosing too much precision to attempt the guessing part.")
        C = ComplexOptimisticField(p, eps=Radii.one()>>p//3)
        v = v.change_ring(C)

        while T0<=2*T:

            S = PowerSeriesRing(C, default_prec=T0 + m)
            basis = self.local_basis_expansions(QQ.zero(), T0 + m) # avoiding the re-computation of the first terms?
            basis = power_series_coerce(basis, S)
            f = v*vector(basis)
            pols = hp_approximants(derivatives(f, m), T0)
            p, d1 = customized_accuracy(pols), max(pol.degree() for pol in pols)
            if d1==d0:
                alg_deg = self.algebraicity_degree
                while 50*alg_deg<p:
                    try:
                        exact_pols = guess_exact_numbers(pols, alg_deg)
                        coeffs = [c for pol in exact_pols for c in pol.coefficients()]
                        selftilde = self.extend_scalars(*coeffs)[0] # not recursive, need an embedding
                        dop = selftilde.parent()(exact_pols)
                        if selftilde%dop==0:
                            self, self.algebraicity_degree = selftilde, alg_deg
                            self.z, self.Dz = self.base_ring().gen(), self.parent().gen()
                            return dop
                    except PrecisionError: pass
                    alg_deg = alg_deg + 1
            d0, T0 = d1, T0<<1

        self.order_of_truncation = self.order_of_truncation<<1

        raise PrecisionError("Insufficient precision for the guessing part.")


    def right_factor(self, verbose=False):

        r"""
        Return either a non-trivial right factor of "self" or the string
        'irreducible' if "self" is irreducible.
        """

        if self.n<2: return 'irreducible'
        if verbose: print("Try to factorize an operator of order " + str(self.n) + ".")
        if self.fuchsian_info==None:
            self.fuchsian_info = self.is_fuchsian()
            if not self.fuchsian_info: print("WARNING: The operator is not fuchsian: termination is not guaranteed.")

        self.monodromy(self.precision, verbose=verbose)
        self.precision = self.monodromy_data.precision
        matrices = self.monodromy_data.matrices
        if verbose: print("Monodromy computed with precision = " + str(self.precision) + ".")

        if matrices==[]:
            if verbose: print("Any subspace is invariant --> symbolic guessing.")
            dop = self._symbolic_guessing()
        else:
            try:
                V = invariant_subspace(matrices, verbose=verbose)
                if V is None: return 'irreducible'
                if verbose: print("Find an invariant subspace of dimension " + str(len(V)) + " --> guessing.")
                dop = self._guessing(V[0], len(V))
            except PrecisionError:
                if verbose: print("Insufficient precision.")
                self.precision = self.precision<<1
                return self.right_factor(verbose=verbose)

        return dop

def right_factor(dop, verbose=False, hybrid=False):

    r"""
    Return either a non-trivial right factor of "dop" or the string
    'irreducible' if "dop" is irreducible.
    """

    if dop.order()<2: return 'irreducible'
    rfactor = try_rational(dop)
    if not rfactor is None: return rfactor
    if hybrid:
        rfactor = try_vanHoeij(dop)
        if not rfactor is None: return rfactor

    coeffs, z0, z = dop.monic().coefficients(), QQ.zero(), dop.base_ring().gen()
    while min(c.valuation(z - z0) for c in coeffs)<0: z0 = z0 + QQ.one()
    shifted_dop = LinearDifferentialOperator(dop.annihilator_of_composition(z + z0))

    output = shifted_dop.right_factor(verbose=verbose)
    if output=='irreducible': return 'irreducible'
    output = output.annihilator_of_composition(z - z0)
    return output


def _factor(dop, verbose=False, splitting_only=False):

    R = rfactor(dop, verbose=verbose, splitting_only=splitting_only)
    if R==None: return [dop]
    OA = R.parent(); OA = OA.change_ring(OA.base_ring().fraction_field())
    Q = OA(dop)//R
    return _factor(Q, verbose=verbose, splitting_only=splitting_only) + _factor(R, verbose=verbose, splitting_only=splitting_only)


def factor(dop, verbose=False, splitting_only=False):

    r"""
    Return a list of irreductible operators [L1, L2, ..., Lr] such that L is
    equal to the composition L1.L2...Lr.
    """

    output = _factor(dop, verbose=verbose, splitting_only=splitting_only)
    K0, K1 = output[0].base_ring().base_ring(), output[-1].base_ring().base_ring()
    if K0 != K1:
        A = output[0].parent()
        output = [A(f) for f in output]
    return output

def _is_irreducible(dop, verbose=False):

    coeffs, z0, z = dop.monic().coefficients(), QQ.zero(), dop.base_ring().gen()
    while min(c.valuation(z - z0) for c in coeffs)<0: z0 = z0 + QQ.one()
    dop = LinearDifferentialOperator(dop.annihilator_of_composition(z + z0))

    return dop._is_irreducible(verbose=verbose)

def my_newton_polygon(dop):

    r"""
    Computes the Newton polygon of ``self`` at 0.

    INPUT:

      - ``dop`` -- a linear differential operator which polynomial coefficients

    OUTPUT:

    EXAMPLES::

    """

    n = dop.order(); z = dop.base_ring().gen()
    Pols, X = PolynomialRing(QQ, 'X').objgen()

    points = [ ((QQ(i), QQ(c.valuation(z))), c.coefficients()[0]) \
               for i, c in enumerate(dop.to_T('Tz').list()) if c!=0 ]

    (i1, j1), c1 = points[0]
    for (i, j), c in points:
        if j<=j1: (i1, j1), c1 = (i, j), c

    Edges = []
    if i1>0:
        poly = dop.indicial_polynomial(z, var = 'X')

        #pol = c1*X**i1
        #for (i, j), c in points:
        #    if i<i1 and j==j1: pol += c*X**i
        ## it is the same think (pol = poly)

        Edges.append( NewtonEdge(QQ(0), (0, j1), i1, poly) )

    while i1<n:
        poly = c1; (i2, j2), c2 = points[-1]; s = (j2 - j1)/(i2 - i1)
        for (i, j), c in points:
            if i>i1:
                t = (j - j1)/(i - i1)
                if t<s:
                    poly = c1; s = t
                    if t<=s:
                        poly += c*X**((i - i1)//s.denominator()) # REDUCED characteristic polynomial
                        (i2, j2), c2 = (i, j), c
        Edges.append( NewtonEdge(s, (i1, j1), i2 - i1, poly) )
        (i1, j1), c1 = (i2, j2), c2

    return Edges


def display_newton_polygon(dop):

    Edges = my_newton_polygon(dop)

    (i, j) = Edges[0].startpoint
    L1 = line2d([(i - 3, j), (i, j)], thickness=3)
    e = Edges[-1]; s = e.slope; (i,j) = e.startpoint; l = e.length
    L2 = line2d([(i + l, j + l*s), (i + l, j + l*s + 3)], thickness=3)

    L = sum(line2d([e.startpoint, (e.startpoint[0] + e.length, e.startpoint[1] \
            + e.length*e.slope)], marker='o', thickness=3) for e in Edges)

    return L1 + L + L2


def exponents(dop, multiplicities=True):

    if dop.base_ring().base_ring()==QQ:
        FLS = LaurentSeriesRing(QQ, dop.base_ring().variable_name())
    else:
        FLS = LaurentSeriesRing(QQbar, dop.base_ring().variable_name())
    l = euler_representation(LinearDifferentialOperator(dop))
    if FLS.base_ring()==QQ:
        l = [FLS(c) for c in l]
    else:
        Pol, _ = PolynomialRing(QQbar, dop.base_ring().variable_name()).objgen()
        l = [FLS(Pol.fraction_field()(c)) for c in l]
    vmin = min(c.valuation() for c in l)
    Pols, X = PolynomialRing(QQ, 'X').objgen()
    pol = sum(FLS.base_ring()(c.coefficients()[0])*X**i for i, c in enumerate(l) if c.valuation()==vmin)
    r = pol.roots(QQbar, multiplicities=multiplicities)

    return r


def S(dop, e):
    """ map: Tz --> Tz + e """

    l = euler_representation(LinearDifferentialOperator(dop))
    for i, c in enumerate(l):
        for k in range(i):
            l[k] += binomial(i, k)*e**(i - k)*c
    T = dop.base_ring().gen()*dop.parent().gen()
    output = sum(c*T**i for i, c in enumerate(l))

    return output

def search_exp_part_with_mult1(dop):

    dop = LinearDifferentialOperator(dop)
    lc = dop.leading_coefficient()//gcd(dop.list())
    for f, _ in list(lc.factor()) + [ (1/dop.base_ring().gen(), None) ]:
        pol = dop.indicial_polynomial(f)
        roots = pol.roots(QQbar)
        for r, m in roots:
            if m==1:
                success = True
                for s, l in roots:
                    if s!=r and r-s in ZZ: success = False
                if success: return (f, r)

    return (None, None)

def guessing_via_series(L, einZZ):
    """ assumption: 0 is an exponential part of multiplicity 1 (at 0) """
    if not einZZ: # if e in ZZ, this test has already been done
        R = try_rational(L)
        if not R is None: return R
    r = L.order(); A = L.parent()
    t = len(L.desingularize().leading_coefficient().roots(QQbar))
    b = min(1000, max(50, (r - 1)**2*(r - 2)*(t - 1)))
    try:
        R = guess(L.power_series_solutions(b)[0].list(), A, order=r - 1) # ne marche pas avec une extension algébrique (-> TypeError)
        if 0<R.order()<r and L%R==0: return R
    except (ValueError, TypeError): pass
    La = L.adjoint()
    Ra = try_rational(La)
    if not Ra is None: return (La//Ra).adjoint()
    ea = ZZ([e for e in exponents(La, False) if e in ZZ][0]); La = S(La, ea)
    try:
        Ra = guess(La.power_series_solutions(b)[0].list(), A, order=r - 1)
        if 0<Ra.order()<r and La%Ra==0: return S(La//Ra, -ea).adjoint()
    except (ValueError, TypeError): return None



def try_vanHoeij(L):
    """ try to find a factor thank to an exponential part of multiplicity 1 """
    z, (p, e) = L.base_ring().gen(), search_exp_part_with_mult1(L)
    if not e in QQ: return None # not efficient enough for now (implem to be improved)
    if p==None: return None
    if (p*z).is_one():
        L = L.annihilator_of_composition(p)
        e = search_exp_part_with_mult1(L)[1]
        L, e = LinearDifferentialOperator(L).extend_scalars(e)
        L = S(L, e)
    elif p.degree()==1:
        s = -p[0]/p[1]
        L, e = LinearDifferentialOperator(L).extend_scalars(e)
        L = S(L.annihilator_of_composition(z + s), e)
    else: return None # to be implemented?
    R = guessing_via_series(L, e in ZZ)
    if R==None: return None
    if (p*z).is_one(): return S(R, -e).annihilator_of_composition(p)
    elif p.degree()==1: return S(R, -e).annihilator_of_composition(z - s)

########################
### Hybrid algorithm ###
########################

def reduced_row_echelon_form(mat):
    R, p = row_echelon_form(mat, pivots=True)
    rows = list(R)
    for j in p.keys():
        for i in range(p[j]):
            rows[i] = rows[i] - rows[i][j]*rows[p[j]]
    return matrix(rows)

def minimal_multiplicity(dop, pol):

    """
    Return (e, m) where e is an exponent of dop at a root of pol with a minimal
    multiplicity modulo ZZ (m).

    -> Representative of smallest real part (pas OK)
    -> minimal algebraic degree (OK)
    """

    N = dop.indicial_polynomial(pol)
    exponents = N.roots(QQbar)
    exponents.sort(key = lambda x: x[0].degree())

    good_exponent, min_mult = exponents[0][0], dop.order()
    done_indices = []
    for i, (e, m) in enumerate(exponents):
        if not i in done_indices:
            multiplicitymodZZ = m
            for j, (f, n) in enumerate(exponents[(i+1):]):
                if e - f in ZZ:
                    multiplicitymodZZ += n
                    done_indices.append(i + 1 + j)
            if multiplicitymodZZ<min_mult:
                min_mult = multiplicitymodZZ
                good_exponent = e
            #done_indices.append(i) --> useless
    return good_exponent, min_mult

def mydegree(pol): # for handling the case 1/z (point at infinity)
    if isinstance(pol, Polynomial):
        return pol.degree()
    return 1

def good_singular_point(dop):

    r"""
    Return (s, e, m) where ``s`` is a singular point (possibly ``infinity``) of
    ``dop`` admitting an exponent ``e`` of minnimal mutliplicity ``m`` mod ZZ.

    INPUT:

      - ``dop`` -- differential operator

    OUTPUT:

      - ``s`` -- element of QQbar
      - ``e`` -- element of QQbar
      - ``m`` -- positive integer

    """

    z = dop.base_ring().gen()
    dop = LinearDifferentialOperator(dop)
    lc = dop.leading_coefficient()//gcd(dop.list())

    all_min_mult = []
    for pol, _ in list(lc.factor()) + [ (1/z, None) ]:
        e, m = minimal_multiplicity(dop, pol)
        all_min_mult.append((pol, e, m))

    min_mult = min(all_min_mult, key = lambda x: x[2])[2]
    good_sings = [ x for x in all_min_mult if x[2]==min_mult ]

    min_deg = mydegree(min(good_sings, key = lambda x: mydegree(x[0]))[0])
    good_sings = [ x for x in all_min_mult if mydegree(x[0])==min_deg ]

    pol, e, m = good_sings[0]
    if isinstance(pol, Polynomial):
        s = pol.roots(QQbar, multiplicities=False)[0]
    else:
        s = 'infinity'

    return s, e, m

def good_base_point(dop):

    s, e, m = good_singular_point(dop)
    if s=='infinity': return 0

    z0 = s.real().ceil()
    sings = LinearDifferentialOperator(dop)._singularities(ZZ)
    while z0 in sings: z0 = z0 + QQ.one()

    return z0

def largest_modulus_of_exponents(dop):

    z = dop.base_ring().gen()
    dop = LinearDifferentialOperator(dop)
    lc = dop.leading_coefficient()//gcd(dop.list())

    out = 0
    for pol, _ in list(lc.factor()) + [ (1/z, None) ]:
        local_exponents = dop.indicial_polynomial(pol).roots(QQbar, multiplicities=False)
        local_largest_modulus = max([x.abs().ceil() for x in local_exponents])
        out = max(local_largest_modulus, out)

    return out

def degree_bound_for_right_factor(dop):

    r = dop.order() - 1
    S = len(dop.desingularize().leading_coefficient().roots(QQbar))
    E = largest_modulus_of_exponents(dop)
    bound = r**2*(S + 1)*E + r*S + r**2*(r - 1)*(S - 1)/2

    return bound

def try_rational(dop):

    D = dop.parent().gen()
    for (f,) in dop.rational_solutions():
        d = f.gcd(f.derivative())
        rfactor = (1/d)*(f*D - f.derivative())
        return rfactor

    return None

def combination(mono):
    prec, C = customized_accuracy(mono), mono[0].base_ring()
    ran = lambda : C(QQ.random_element(prec), QQ.random_element(prec))
    return sum(ran()*mat for mat in mono)

adjoint = lambda dop: sum((-dop.parent().gen())**i*pi for i, pi in enumerate(dop.list()))
der_list = lambda l: [f.derivative() for f in l]
der_mat = lambda mat: matrix([der_list(row) for row in mat])

def diffop_companion_matrix(dop, r):
    A =  block_matrix([[matrix(r-1,1, [0]*(r-1)), identity_matrix(r-1)], \
                       [matrix([[-dop[0]]]), \
                        -matrix(1,r-1, dop.list()[1:-1])]], subdivide=False)
    return A

def transitionYtoV(A):
    r = A.nrows()
    B = [identity_matrix(A.base_ring(), r)]
    for k in range(1, r):
        Bk = der_mat(B[k-1]) - B[k-1]*(A.transpose())
        B.append(Bk)
    P = matrix([B[k][-1] for k in range(r)])
    return P(0)

def try_simple_eigenvalue(dop, mono, tmp, order, bound, alg_degree, verbose=False):

    r = dop.order()
    mat = combination(mono)
    GenEigSpaces = gen_eigenspaces(mat)
    for space in GenEigSpaces:
        if space['multiplicity']==1:
            tmp[0] = True
            if verbose: print("Find a simple eigenvalue")
            ic = space['basis'][0]
            b, R = minimal_annihilator(dop, ic, order, bound, alg_degree, mono=mono, verbose=verbose)
            if b and R!=dop: return True, R
            r, adj_dop = dop.order(), adjoint(dop)
            adj_mono = [mat.transpose() for mat in mono]
            A = diffop_companion_matrix(dop, r)
            P = transitionYtoV(A)
            T = diagonal_matrix([factorial(i) for i in range(r)])
            adj_ic = T*(~P).transpose()*T*ic
            adj_b, adj_Q = minimal_annihilator(adj_dop, adj_ic, order, bound, alg_degree, mono=adj_mono, verbose=verbose)
            if adj_b and adj_Q!=adj_dop:
                print('Yes!')
                return True, adjoint(adj_dop//adj_Q)
            if b and adj_b and R.order()==adj_Q.order()==r:
                return True, None
            break

    return False, dop

def try_one_dim_eigenspaces(dop, mono, tmp, order, bound, alg_degree, verbose=False):

    r = dop.order()
    mat = combination(mono)
    GenEigSpaces = gen_eigenspaces(mat)
    if all(space['multiplicity']==1 for space in GenEigSpaces):
        tmp[0] = True
        if verbose: print("Find a matrix with one-dimensional eigenspaces")
        success = True
        for space in GenEigSpaces:
            ic = space['basis'][0]
            b, R = minimal_annihilator(dop, ic, order, bound, alg_degree, mono=mono, verbose=verbose)
            if b:
                if R!=dop: return True, R
            else: success = False
        if success: return True, None

    return False, None

def try_splitting(dop, mono, order, bound, alg_degree, verbose=False):

    if verbose: print("Start Splitting Method")

    invspace = invariant_subspace(mono)
    if invspace==None: return True, None

    if verbose: print("Find an invariant subspace of dimension", len(invspace))

    ic = reduced_row_echelon_form(matrix(invspace))[0]
    b, R = minimal_annihilator(dop, ic, order, bound, alg_degree, verbose=verbose)
    if b:
        if R!=dop: return True, R
        return True, None

    return False, None


def minimal_annihilator_exact(dop, ic, order, bound, basis=None, verbose=False):

    """
    Return either (True, R) where R is a factor of dop annhilating (dop, ic)
    (strict if any, dop ortherwise) or (False, dop) if cannot conclude
    (can appear only if order!=bound).
    """

    r = dop.order()

    if basis==None:
        basis = dop.power_series_solutions(order + r + 10)
        basis.reverse()

    if all(x in QQ for x in ic):
        f = vector([QQ(x) for x in ic])*vector(basis)
        try:
            if verbose: print("Rational Hermite-Padé computation at order", order)
            R = guess(f.list(), dop.parent(), order=r - 1)
            if R==1: breakpoint()
            if dop%R==0: return True, R
            else: return False, dop
        except ValueError:
            return order==bound, dop

    f = vector(ic)*vector(basis)
    if verbose: print("Algebraic Hermite-Padé computation at order", order)
    R = hp_approximants(derivatives(f, r - 1), order)
    dop = LinearDifferentialOperator(dop).extend_scalars(*ic)[0]
    R = dop.parent()(R)
    if dop%R==0: return True, R

    return False, dop # on aimerait pouvoir mettre (order==bound, dop) ici

def minimal_annihilator(dop, ic, order, bound, alg_degree, basis=None, mono=None, verbose=False):

    r"""
    Return either (True, R) where R is a factor of dop annhilating (dop, ic)
    (strict if any, dop ortherwise) or (False, dop) if cannot conclude
    (can appear only if order!=bound).

    Note for me: for now, (True, dop) can be returned only thanks to mono.
    """

    r = dop.order()

    if mono!=None:
        orb = orbit(mono, ic)
        if len(orb)==r: return True, dop

    if basis==None:
        basis = dop.power_series_solutions(order + r + 10)
        basis.reverse()

    prec = customized_accuracy(ic)
    if prec>50:
        ic1, ic2, d = 0, 1, 1
        while d<=alg_degree and ic1!=ic2:
            if d==1:
                try:
                    ic1 = guess_rational_numbers(ic, p=prec-20)
                    ic2 = guess_rational_numbers(ic, p=prec-30)
                except PrecisionError: pass
            else:
                ic1 = guess_algebraic_numbers(ic, d=d, p=prec - 20)
                ic2 = guess_algebraic_numbers(ic, d=d, p=prec - 30)
            d = d + 1

        if ic1==ic2:
            if verbose: print("Exact coefficients OK")
            b, R = minimal_annihilator_exact(dop, ic1, order, bound, basis)
            if b and R!=dop: return True, R

    # improvement: no ball HP approximants --> True, dop

    return False, None

def rfactor(dop, order=None, bound=None, alg_degree=1, precision=None, loss=None, splitting_only=False, verbose=False):

    z = dop.base_ring().gen()
    if dop.order()<2: return None

    if verbose: print('Factorization of an operator of order', dop.order())

    R = try_rational(dop)
    if R!=None: return R

    # try eigenring

    z0 = good_base_point(dop)
    dop = dop.annihilator_of_composition(z + z0)

    if bound==None:
        bound = degree_bound_for_right_factor(dop)
        if verbose: print("Degree bound for right factor", bound)
    if order==None:
        order = min(bound, max(20, min(dop.order()*dop.degree(), 100)))
    if precision==None: precision = 200
    if loss==None: loss=0
    if verbose: print("Current order of truncation", order)
    if verbose: print("Current algebraic degree", alg_degree)

    precision_error_occured=True
    while precision_error_occured:
        try:
            it = _monodromy_matrices(dop, 0, eps=Radii.one()>>precision)
            mono = []
            if verbose: print("Start monodromy computation with precision", precision)
            for pt, mat, scal in it:
                if not scal:
                    local_loss = max(0, precision - customized_accuracy(mat))
                    if local_loss>loss:
                        loss = local_loss
                        if verbose: print("loss =", loss)
                    if verbose: print("New monodromy matrix computed")
                    mono.append(mat)
                    tmp = [False]
                    if splitting_only:
                        b, R = try_splitting(dop, mono, order, bound, alg_degree, verbose=verbose)
                        if b:
                            if verbose: print("Conclude with Splitting Method")
                            if R==None: return None
                            else: return R.annihilator_of_composition(z - z0)
                    else:
                        b, R = try_simple_eigenvalue(dop, mono, tmp, order, bound, alg_degree, verbose=verbose)
                        if b:
                            if verbose: print("Conclude with Simple Eigenvalue Method")
                            if R==None: return None
                            else: return R.annihilator_of_composition(z - z0)
                        if not tmp[0]:
                            b, R = try_one_dim_eigenspaces(dop, mono, tmp, order, bound, alg_degree, verbose=verbose)
                            if b:
                                if verbose: print("Conclude with One-D Eignespaces Method")
                                if R==None: return None
                                else: return R.annihilator_of_composition(z - z0)
                            if not tmp[0] and len(mono)>1:
                                b, R = try_splitting(dop, mono, order, bound, alg_degree, verbose=verbose)
                                if b:
                                    if verbose: print("Conclude with Splitting Method")
                                    if R==None: return None
                                    else: return R.annihilator_of_composition(z - z0)
            precision_error_occured = False
        except (ZeroDivisionError, PrecisionError):
            pass

        precision = max(precision + loss, (precision<<1) - loss)

    return rfactor(dop, min(bound, order<<1), bound, alg_degree + 1, precision, loss, verbose=verbose)

def profil_factor(dop, verbose=False, splitting_only=False):
    fac = [None]
    def fun():
        fac[0] = dop.factor(verbose=verbose, splitting_only=splitting_only)
        return
    cProfile.runctx('fun()', None, {'fun': fun}, 'tmp_stats')
    s = pstats.Stats('tmp_stats')
    key_tot = ('~', 0, '<built-in method builtins.exec>')
    time_tot = numerical_approx(s.stats[key_tot][3], digits=3)
    time_mono, time_hprat, time_hpalg, time_grat, time_galg = [0]*5
    for key in s.stats.keys():
        if key[2] == '_monodromy_matrices':
            time_mono = numerical_approx(s.stats[key][3], digits=3)
        if key[2] == 'guess':
            time_hprat = numerical_approx(s.stats[key][3], digits=3)
        if key[2] == 'hp_approximants':
            time_hpalg = numerical_approx(s.stats[key][3], digits=3)
        if key[2]=='guess_rational_numbers':
            time_grat = numerical_approx(s.stats[key][3], digits=3)
        if key[2]=='guess_algebraic_numbers':
            time_galg = numerical_approx(s.stats[key][3], digits=3)
    profil = {'total' : time_tot, 'monodromy': time_mono, \
    'hermitepade': time_hprat + time_hpalg, \
    'guesscoefficients': time_grat + time_galg}
    return fac[0], profil
