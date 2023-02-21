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
import tempfile
import cProfile
import pstats

from ore_algebra import guess
from ore_algebra.analytic.accuracy import PrecisionError
from ore_algebra.analytic.complex_optimistic_field import ComplexOptimisticField
from ore_algebra.analytic.differential_operator import PlainDifferentialOperator
from ore_algebra.analytic.linear_algebra import (invariant_subspace,
                                                 row_echelon_form, ker,
                                                 gen_eigenspaces, orbit,
                                                 customized_accuracy)
from ore_algebra.analytic.monodromy import _monodromy_matrices
from ore_algebra.analytic.utilities import as_embedded_number_field_elements
from ore_algebra.guessing import guess
from sage.arith.functions import lcm
from sage.arith.misc import valuation, algdep, gcd
from sage.functions.all import log, floor
from sage.functions.other import binomial, factorial
from sage.matrix.constructor import matrix
from sage.matrix.matrix_dense import Matrix_dense
from sage.matrix.special import block_matrix, identity_matrix, diagonal_matrix
from sage.misc.functional import numerical_approx
from sage.misc.misc import cputime
from sage.misc.misc_c import prod
from sage.modules.free_module_element import vector, FreeModuleElement_generic_dense
from sage.rings.integer_ring import ZZ
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.qqbar import QQbar
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RealField
from sympy.core.numbers import oo

Radii = RealField(30)

MonoData = collections.namedtuple("MonoData", ["precision", "matrices", "points", "loss"])

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

        self.order_of_truncation = max(100, 2*self.degree() + self.order())
        self.algebraicity_degree = self.base_ring().base_ring().degree()
        self.precision = 100 #100*self.algebraicity_degree

        self.monodromy_data = MonoData(0, [], None, 0)

        self.fuchsian_info = None


    def is_fuchsian(self):

        r"""
        Return True if ``self`` is fuchsian, False otherwise.

        Fuch's criterion: p is a regular point of a_n*Dz^n + ... + a_0 (with a_i
        polynomial) iff no (z-p)^{n-k}*a_k/a_n admits p as pole.
        """

        coeffs = self.coefficients()
        fac = coeffs.pop().factor()
        for (f, m) in fac:
            for k, ak in enumerate(coeffs):
                mk = valuation(ak, f)
                if mk - m < k - self.order(): return False

        dop = self.annihilator_of_composition(1/self.base_ring().gen())
        for k, frac in enumerate(dop.monic().coefficients()[:-1]):
            d = (self.base_ring().gen()**(self.order() - k)*frac).denominator()
            if d(0)==0: return False

        return True


    def monodromy(self, precision, verbose=False):

        r"""
        Compute a generating set of matrices for the monodromy group of ``self``
        at 0, such that the (customized) precision of each coefficient is at
        least equal to ``precision``.
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


########################
### Hybrid algorithm ###
########################

def try_rational(dop):

    D = dop.parent().gen()
    for (f,) in dop.rational_solutions():
        d = f.gcd(f.derivative())
        R = (1/d)*(f*D - f.derivative())
        return R

    return None


def annihilator(dop, ic, order, bound, alg_degree, mono=None, verbose=False):

    r, OA = dop.order(), dop.parent()
    d = r - 1
    base_field = OA.base_ring().base_ring()

    if mono!=None:
        orb = orbit(mono, ic)
        d = len(orb)
        if d==r: return dop
        ic = reduced_row_echelon_form(matrix(orb))[0]

    symb_ic, K = guess_symbolic_coefficients(ic, alg_degree, verbose=verbose)
    if symb_ic!="NothingFound":
        if base_field!=QQ and K!=QQ:
            K = K.composite_fields(base_field)[0]
            symb_ic = [K(x) for x in symb_ic]
        S = PowerSeriesRing(K, default_prec=order + d)
        sol_basis = dop.local_basis_expansions(QQ.zero(), order + d)
        sol_basis = [ _formal_finite_sum_to_power_series(sol, S) for sol in sol_basis ]
        f = vector(symb_ic) * vector(sol_basis)
        if K==QQ and base_field==QQ:
            v = f.valuation()
            try:
                R = _Se(guess(f.list()[v:], OA, order=d), -v)
                if 0<R.order()<r and dop%R==0: return R
            except ValueError: pass
        else:
            der = [ f.truncate() ]
            for k in range(d): der.append( der[-1].derivative() )
            mat = matrix(d + 1, 1, der)
            if verbose: print("Try guessing annihilator with HP approximants")
            min_basis = mat.minimal_approximant_basis(order)
            rdeg = min_basis.row_degrees()
            if max(rdeg) > 1 + min(rdeg): # to avoid useless (possibly large) computation
                i0 = min(range(len(rdeg)), key=lambda i: rdeg[i])
                R, g = LinearDifferentialOperator(dop).extend_scalars(K.gen())
                R = R.parent()(list(min_basis[i0]))
                if dop%R==0: return R

    if order>r*(bound + 1) and verbose:
        print("Ball Hermite--Padé approximants not implemented yet")

    return "Inconclusive"


def one_dimensional_eigenspaces(dop, mono, order, bound, alg_degree, verbose=False):

    """
    output: a nontrivial right factor R of dop, or None, or ``NotGoodConditions``,
    or ``Inconclusive``
    """

    mat = random_combination(mono)
    id = mat.parent().one()
    Spaces = gen_eigenspaces(mat)
    conclusive = True
    goodconditions = True
    for space in Spaces:
        eigvalue = space["eigenvalue"]
        eigspace = ker(mat - eigvalue*id)
        if len(eigspace)>1:
            goodconditions = False
            break
        R = annihilator(dop, eigspace[0], order, bound, alg_degree, mono, verbose)
        if R=="Inconclusive": conclusive = False
        if R!=dop: return R
    if not goodconditions: return "NotGoodConditions"
    if conclusive: return None
    return "Inconclusive"


def simple_eigenvalue(dop, mono, order, bound, alg_degree, verbose=False):

    """
    output: a nontrivial right factor R of dop, or None, or ``NotGoodConditions``,
    or ``Inconclusive``

    Assumption: dop is monic.
    """

    mat = random_combination(mono)
    id = mat.parent().one()
    Spaces = gen_eigenspaces(mat)
    goodconditions = False
    for space in Spaces:
        if space['multiplicity']==1:
            goodconditions = True
            ic = space['basis'][0]
            R = annihilator(dop, ic, order, bound, alg_degree, mono, verbose)
            if R!="Inconclusive" and R!=dop: return R
            adj_dop = myadjoint(dop)
            Q = transition_matrix_for_adjoint(dop)
            adj_mat = Q * mat.transpose() * (~Q)
            adj_mono = [ Q * m.transpose() * (~Q) for m in mono ]
            eigspace = ker(adj_mat - space['eigenvalue']*id)
            if eigspace==[]: return "Inconclusive"
            if len(eigspace)>1: break # raise PrecisionError ?
            adj_ic = eigspace[0]
            adj_Q = annihilator(adj_dop, adj_ic, order, bound, alg_degree, adj_mono, verbose) # bound différent?
            if adj_Q!="Inconclusive" and adj_Q!=adj_dop:
                return myadjoint(adj_dop//adj_Q)
            if R==dop and adj_Q==adj_dop: return None
            break
    if not goodconditions: return "NotGoodConditions"
    return "Inconclusive"


def multiple_eigenvalue(dop, mono, order, bound, alg_degree, verbose=False):
    """
    output: a nontrivial right factor R of dop, or None, or ``Inconclusive``
    """

    r = dop.order()
    invspace = invariant_subspace(mono)
    if invspace==None: return None # devrait jamais arriver quand utilisé à l'intérieur de rfactor
    R = annihilator(dop, invspace[0], order, bound, alg_degree, mono, verbose)
    if R!="Inconclusive" and R.order()<r: return R
    return "Inconclusive"


def _factor(dop, data, verbose=False):

    R, data = rfactor(dop, data, verbose)
    if R==None: return [dop], data
    OA = R.parent(); OA = OA.change_ring(OA.base_ring().fraction_field())
    Q = OA(dop)//R
    fac1, data = _factor(Q, data, verbose)
    fac2, data = _factor(R, data, verbose)
    return fac1 + fac2, data


def factor(dop, return_data=False, verbose=False):

    r"""
    Return a list of irreductible operators [L1, L2, ..., Lr] such that L is
    equal to the composition L1.L2...Lr.
    """

    data = ProfileData(0,0,1,0,0)
    output, data = _factor(dop, data, verbose)
    K0, K1 = output[0].base_ring().base_ring(), output[-1].base_ring().base_ring()
    if K0 != K1:
        A = output[0].parent()
        output = [A(f) for f in output]
    if return_data: return output, data
    return output

def rfactor(dop, data, verbose=False):

    r = dop.order()
    if r<2: return None, data
    if verbose: print("### Try factoring an operator of order", r)
    z = dop.base_ring().gen()
    R = try_rational(dop)
    if R!=None: return R, data

    s0, sings = QQ.zero(), LinearDifferentialOperator(dop)._singularities(QQbar)
    while s0 in sings: s0 = s0 + QQ.one()
    dop = dop.annihilator_of_composition(z + s0).monic()
    R, data = _rfactor(dop, data=data, verbose=verbose)
    if R==None: return None, data
    return R.annihilator_of_composition(z - s0), data


def rfactor_when_galois_algebra_is_trivial(dop, data, order, verbose=False):

    if verbose: print("Galois algebra is trivial: symbolic HP approximants method at order", order)
    K, r = dop.base_ring().base_ring(), dop.order()
    S = PowerSeriesRing(K, default_prec=order + r)
    f = dop.local_basis_expansions(QQ.zero(), order + r)[0]
    f = _formal_finite_sum_to_power_series(f, S)

    der = [ f.truncate() ]
    for k in range(r - 1): der.append( der[-1].derivative() )
    mat = matrix(r, 1, der)
    min_basis = mat.minimal_approximant_basis(max(order//r, 1))
    rdeg = min_basis.row_degrees()
    i0 = min(range(len(rdeg)), key = lambda i: rdeg[i])
    #R, g = LinearDifferentialOperator(dop).extend_scalars(K.gen())
    R = dop.parent()(list(min_basis[i0]))
    if dop%R==0: return R, data

    order = order<<1
    data = maj(data, [None, order, None, None, None])
    return rfactor_when_galois_algebra_is_trivial(dop, data, order, verbose)

def _rfactor(dop, data, order=None, bound=None, alg_degree=None, precision=None, loss=0, verbose=False):
    """
    Assumption: dop is monic and 0 is not singular.
    """

    r = dop.order()
    if bound==None:
        bound = degree_bound_for_right_factor(dop)
        if verbose: print("Degree bound for right factor", bound)
    if order==None:
        deg_of_dop = LinearDifferentialOperator(dop).degree()
        order = max(min( r*deg_of_dop, 100, bound*(r + 1) + 1 ), 1)
    if alg_degree==None:
        alg_degree = dop.base_ring().base_ring().degree()
    if precision==None:
        precision = 50*(r + 1)

    data = maj(data, [precision-loss, order, alg_degree, None, None])
    if verbose:
        print("Current order of truncation", order)
        print("Current working precision", precision, "(before monodromy computation)")
        print("Current algebraic degree", alg_degree)
        print("Start computing monodromy matrices")

    try:
        mono, it = [], _monodromy_matrices(dop, 0, eps=Radii.one()>>precision)
        for pt, mat, scal in it:
            if not scal:
                local_loss = max(0, precision - customized_accuracy(mat))
                if local_loss>loss:
                    loss = local_loss
                    if verbose: print("loss =", loss)
                mono.append(mat)
                if verbose: print(len(mono), "matrices computed")
                conclusive_method = "One_Dimensional"
                R = one_dimensional_eigenspaces(dop, mono, order, bound, alg_degree, verbose)
                if R=="NotGoodConditions":
                    conclusive_method = "Simple_Eigenvalue"
                    R = simple_eigenvalue(dop, mono, order, bound, alg_degree, verbose)
                    if R=="NotGoodConditions":
                        conclusive_method = "Multiple_Eigenvalue"
                        R = multiple_eigenvalue(dop, mono, order, bound, alg_degree, verbose)
                if R!="Inconclusive":
                    if verbose: print("Conclude with " + conclusive_method + " method")
                    try:
                        cond_nb = max([_frobenius_norm(m)*_frobenius_norm(~m) for m in mono])
                        cond_nb = cond_nb.log(10).ceil()
                    except (ZeroDivisionError, PrecisionError):
                        cond_nb = oo
                    return R, maj(data, [None, None, None, len(mono), cond_nb])
        if mono==[]:
            return rfactor_when_galois_algebra_is_trivial(dop, data, order, verbose)


    except (ZeroDivisionError, PrecisionError):
        precision += max(150, precision - loss)
        #precision = max( precision + loss, (precision<<1) - loss )
        return _rfactor(dop, data, order, bound, alg_degree, precision, loss, verbose)

    precision += max(150, precision - loss)
    #precision = max( precision + loss, (precision<<1) - loss ) # trop violent?
    order = min( bound*(r + 1) + 1, order<<1 )
    return _rfactor(dop, data, order, bound, alg_degree + 1, precision, loss, verbose)


################################################################################
### Tools ######################################################################
################################################################################


def reduced_row_echelon_form(mat):
    R, p = row_echelon_form(mat, pivots=True)
    rows = list(R)
    for j in p.keys():
        for i in range(p[j]):
            rows[i] = rows[i] - rows[i][j]*rows[p[j]]
    return matrix(rows)

def _local_exponents(dop, multiplicities=True):
    ind_pol = dop.indicial_polynomial(dop.base_ring().gen())
    return ind_pol.roots(QQbar, multiplicities=multiplicities)

def largest_modulus_of_exponents(dop):

    z = dop.base_ring().gen()
    dop = LinearDifferentialOperator(dop)
    lc = dop.leading_coefficient()//gcd(dop.list())

    out = 0
    for pol, _ in list(lc.factor()) + [ (1/z, None) ]:
        local_exponents = dop.indicial_polynomial(pol).roots(QQbar, multiplicities=False)
        local_largest_modulus = max([x.abs().ceil() for x in local_exponents], default=QQbar.zero())
        out = max(local_largest_modulus, out)

    return out

def degree_bound_for_right_factor(dop):

    r = dop.order() - 1
    #S = len(dop.desingularize().leading_coefficient().roots(QQbar)) # trop lent (exemple QPP)
    S = len(LinearDifferentialOperator(dop).leading_coefficient().roots(QQbar))
    E = largest_modulus_of_exponents(dop)
    bound = r**2*(S + 1)*E + r*S + r**2*(r - 1)*(S - 1)/2

    return ZZ(bound)

def random_combination(mono):
    prec, C = customized_accuracy(mono), mono[0].base_ring()
    if prec<10: raise PrecisionError
    ran = lambda : C(QQ.random_element(prec), QQ.random_element(prec))
    return sum(ran()*mat for mat in mono)


myadjoint = lambda dop: sum((-dop.parent().gen())**i*pi for i, pi in enumerate(dop.list()))

def diffop_companion_matrix(dop):
    r = dop.order()
    A = block_matrix([[matrix(r - 1 , 1, [0]*(r - 1)), identity_matrix(r - 1)],\
                      [ -matrix([[-dop.list()[0]]]) ,\
                        -matrix(1, r - 1, dop.list()[1:-1] )]], subdivide=False)
    return A

def transition_matrix_for_adjoint(dop):

    """
    Return an invertible constant matrix Q such that: if M is the monodromy of
    dop along a loop gamma, then the monodromy of the adjoint of dop along
    gamma^{-1} is equal to Q*M.transpose()*(~Q), where the monodromies are
    computed in the basis given by .local_basis_expansions method.

    Assumptions: dop is monic, 0 is the base point, and 0 is not singular.
    """

    AT = diffop_companion_matrix(dop).transpose()
    r = dop.order()
    B = [identity_matrix(dop.base_ring(), r)]
    for k in range(1, r):
        Bk = B[k - 1].derivative() - B[k - 1] * AT
        B.append(Bk)
    P = matrix([ B[k][-1] for k in range(r) ])
    Delta = diagonal_matrix(QQ, [1/factorial(i) for i in range(r)])
    Q = Delta * P(0) * Delta
    return Q


def guess_symbolic_coefficients(vec, alg_degree, verbose=False):

    """
    Return a reasonable symbolic vector contained in the ball vector ``vec``
    and its field of coefficients if something reasonable is found, or
    ``NothingFound`` otherwise.

    INPUT:
     -- ``vec`` -- ball vector
     -- ``alg_degree``   -- positive integer

    OUTPUT:
     -- ``symb_vec`` -- vector with exact coefficients, or ``NothingFound``
     -- ``K``        -- QQ, or a number field, or None (if ``symb_vec``=``NothingFound``)

    EXAMPLES::
        sage: C = ComplexBallField()
        sage: err = C(0).add_error(RR.one()>>40)
        sage: vec = vector(C, [C(sqrt(2)) + err, 3 + err])
        sage: guess_symbolic_coefficients(vec, 1)
        ('NothingFound', None)
        sage: guess_symbolic_coefficients(vec, 2)
        ([a, 3],
         Number Field in a with defining polynomial y^2 - 2 with a = 1.414213562373095?)
    """

    if verbose: print("Try guessing symbolic coefficients")

    # first fast attempt working well if rational
    v1, v2 = [], []
    for x in vec:
        if not x.imag().contains_zero(): break
        x, err = x.real().mid(), x.rad()
        err1, err2 = err, 2*err/3
        v1.append(x.nearby_rational(max_error=x.parent()(err1)))
        v2.append(x.nearby_rational(max_error=x.parent()(err2)))
    if len(v1)==len(vec) and v1==v2:
        if verbose: print("Find rational coefficients")
        return v1, QQ

    p = customized_accuracy(vec)
    if p<30: return "NothingFound", None
    for d in range(2, alg_degree + 1):
        v1, v2 = [], []
        for x in vec:
            v1.append(algdep(x.mid(), degree=d, known_bits=p-10))
            v2.append(algdep(x.mid(), degree=d, known_bits=p-20))
        if v1==v2:
            symb_vec = []
            for i, x in enumerate(vec):
                roots = v1[i].roots(QQbar, multiplicities=False)
                k = len(roots)
                i = min(range(k), key = lambda i: abs(roots[i] - x.mid()))
                symb_vec.append(roots[i])
            K, symb_vec = as_embedded_number_field_elements(symb_vec)
            if not all(symb_vec[i] in x for i, x in enumerate(vec)): return "NothingFound", None
            if verbose: print("Find algebraic coefficients in a number field of degree", K.degree())
            return symb_vec, K

    return "NothingFound", None

_frobenius_norm = lambda m: sum([x.abs().mid()**2 for x in m.list()]).sqrt()

def _formal_finite_sum_to_power_series(f, PSR):

    """ Assumtion: x is extended at 0 (shift otherwise). """

    if isinstance(f, list):
        return [ _formal_finite_sum_to_power_series(g, PSR) for g in f ]

    out = PSR.zero()
    for constant, monomial in f:
        if constant!=0:
            out += constant*PSR.gen()**monomial.n

    return out

def _euler_representation(dop):

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

def _Se(dop, e):

    """ map: Tz --> Tz + e """

    l = _euler_representation(LinearDifferentialOperator(dop))
    for i, c in enumerate(l):
        for k in range(i):
            l[k] += binomial(i, k)*e**(i - k)*c
    T = dop.base_ring().gen()*dop.parent().gen()
    output = sum(c*T**i for i, c in enumerate(l))

    return output

################################################################################
### Guessing tools #############################################################
################################################################################
#
#def hp_approximants(F, sigma):
#
#    r"""
#    Return an Hermite--Padé approximant of ``F`` at order ``sigma``.
#
#    Let ``F = [f1, ..., fm]``. This function returns a list of polynomials ``P =
#    [p1, ..., pm]`` such that:
#    - ``max(deg(p1), ..., deg(pm))`` is minimal,
#    - ``p1*f1 + ... + pm*fm = O(x^sigma)``.
#
#    INPUT:
#     - ``F`` - a list of polynomials or power series
#
#    OUTPUT:
#     - ``P`` - a list of polynomials
#
#    EXAMPLES::
#
#        sage: from ore_algebra.analytic.factorization import hp_approximants
#        sage: f = taylor(log(1+x), x, 0, 8).series(x).truncate().polynomial(QQ)
#        sage: F = [f, f.derivative(), f.derivative().derivative()]
#        sage: P = hp_approximants(F, 5); P
#        [0, 1, x + 1]
#        sage: from ore_algebra import OreAlgebra
#        sage: Pols.<x> = QQ[]; Dops.<Dx> = OreAlgebra(Pols); dop = Dops(P)
#        sage: dop, dop(log(1+x))
#        ((x + 1)*Dx^2 + Dx, 0)
#
#    """
#
#    try:
#        F = [f.truncate() for f in F]
#    except: pass
#
#    mat = matrix(len(F), 1, F)
#    basis = mat.minimal_approximant_basis(sigma)
#    rdeg = basis.row_degrees()
#    i = min(range(len(rdeg)), key = lambda i: rdeg[i])
#
#    return list(basis[i])
#
#
#
#def guess_rational_numbers(x, p=None):
#
#    r"""
#    Guess rational coefficients for a vector or a matrix or a polynomial or a
#    list or just a complex number.
#
#    Note: this function is designed for ComplexOptimisticField as base ring.
#
#    INPUT:
#     - 'x' - object with approximate coefficients
#
#    OUTPUT:
#     - 'r' - object with rational coefficients
#
#    EXAMPLES::
#
#        sage: from ore_algebra.analytic.complex_optimistic_field import ComplexOptimisticField
#        sage: from ore_algebra.analytic.factorization import guess_rational_numbers
#        sage: C = ComplexOptimisticField(30, 2^-10)
#        sage: a = 1/3 - C(1+I)*C(2^-20)
#        sage: Pols.<x> = C[]; pol = (1/a)*x + a; pol
#        ([3.0000086 +/- 2.86e-8] + [8.5831180e-6 +/- 7.79e-14]*I)*x + [0.333332379 +/- 8.15e-10] - [9.53674316e-7 +/- 4.07e-16]*I
#        sage: guess_rational_numbers(pol)
#        3*x + 1/3
#
#    """
#
#    if isinstance(x, list) :
#        return [guess_rational_numbers(c, p=p) for c in x]
#
#    if isinstance(x, FreeModuleElement_generic_dense) or isinstance(x, Matrix_dense) or isinstance(x, Polynomial):
#        return x.parent().change_ring(QQ)(guess_rational_numbers(x.list(), p=p))
#
#    if p is None:
#        eps = x.parent().eps
#        p = floor(-log(eps, 2)) # does eps.log2().floor() work? try Marc's function prec_from_eps?
#    else:
#        eps = RealField(30).one() >> p
#    if not x.imag().above_abs().mid()<eps:
#        raise PrecisionError('This number does not seem a rational number.')
#    x = x.real().mid()
#
#    return x.nearby_rational(max_error=x.parent()(eps))
#
#
#
#def guess_algebraic_numbers(x, d=2, p=None):
#
#    r"""
#    Guess algebraic coefficients for a vector or a matrix or a polynomial or a
#    list or just a complex number.
#
#    INPUT:
#     - 'x' - an object with approximate coefficients
#     - 'd' - a positive integer (bound for algebraicity degree)
#     - 'p' - a positive integer (number of known bits)
#
#    OUTPUT:
#     - 'a' - an object with algebraic coefficients
#
#    EXAMPLES::
#
#        sage: from ore_algebra.analytic.complex_optimistic_field import ComplexOptimisticField
#        sage: from ore_algebra.analytic.factorization import guess_algebraic_numbers
#        sage: a = ComplexOptimisticField()(sqrt(2))
#        sage: guess_algebraic_numbers(a)
#        1.414213562373095?
#        sage: _.minpoly()
#        x^2 - 2
#
#    """
#
#    if isinstance(x, list) :
#        return [guess_algebraic_numbers(c, d=d, p=p) for c in x]
#
#    if isinstance(x, FreeModuleElement_generic_dense) or \
#    isinstance(x, Matrix_dense) or isinstance(x, Polynomial):
#        return x.parent().change_ring(QQbar)(guess_algebraic_numbers(x.list(), p=p, d=d))
#
#    if p is None: p = floor(-log(x.parent().eps, 2))
#
#    pol = algdep(x.mid(), degree=d, known_bits=p)
#    roots = pol.roots(QQbar, multiplicities=False)
#    i = min(range(len(roots)), key = lambda i: abs(roots[i] - x.mid()))
#
#    return roots[i]




################################################################################
### Profiling #############################################################
################################################################################

ProfileData = collections.namedtuple("ProfileData", ["bit_precision", "truncation_order", \
                                     "algebraicity_degree", "number_of_monodromy_matrices", "log_condition_number"])

def maj(data, l):
    for i, x in enumerate(l):
        y = data[i]
        if x==None:
            l[i] = y
        else:
            l[i] = max(x, y)
    return ProfileData(l[0], l[1], l[2], l[3], l[4])

def profile_factor(dop, verbose=False):
    fac, data = [None], [None]
    def fun():
        fac[0], data[0] = rfactor(dop, data=ProfileData(0,0,1,0,0), verbose=verbose)
        return
    with tempfile.NamedTemporaryFile() as tmp_stats:
        cProfile.runctx('fun()', None, {'fun': fun}, tmp_stats.name)
        s = pstats.Stats(tmp_stats.name)
    if fac[0]==None:
        fac[0] = [dop]
    else:
        fac[0] = [dop//fac[0], fac[0]]
    key_tot = ('~', 0, '<built-in method builtins.exec>')
    time_tot = numerical_approx(s.stats[key_tot][3], digits=3)
    time_mono, time_hprat, time_hpalg, time_guess_coeff = [0]*4
    for key in s.stats.keys():
        if key[2] == '_monodromy_matrices':
            time_mono = numerical_approx(s.stats[key][3], digits=3)
        if key[2] == 'guess':
            time_hprat = numerical_approx(s.stats[key][3], digits=3)
        if key[2] == "<method 'minimal_approximant_basis' of 'sage.matrix.matrix_polynomial_dense.Matrix_polynomial_dense' objects>":
            time_hpalg = numerical_approx(s.stats[key][3], digits=3)
        if key[2]=='guess_symbolic_coefficients':
            time_guess_coeff = numerical_approx(s.stats[key][3], digits=3)
    profile = {'time_total' : time_tot, 'time_monodromy': time_mono, \
    'time_hermitepade': time_hprat + time_hpalg, \
    'time_guesscoefficients': time_guess_coeff, \
    'bit_precision': data[0].bit_precision,\
    'truncation_order': data[0].truncation_order,\
    'algebraicity_degree': data[0].algebraicity_degree,\
    'number_of_monodromy_matrices': data[0].number_of_monodromy_matrices,\
    'log_condition_number': data[0].log_condition_number
    }
    return fac[0], profile
