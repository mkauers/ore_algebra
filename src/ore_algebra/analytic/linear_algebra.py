# -*- coding: utf-8 - vim: tw=80
"""
Linear algebra over ComplexOptimisticField. Gauss reduction, generalized
eigenspaces, invariant subspace, ...
"""

# Copyright 2021 Alexandre Goyer, Inria Saclay Ile-de-France
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/


from ore_algebra.analytic.accuracy import PrecisionError
from ore_algebra.analytic.complex_optimistic_field import ComplexOptimisticField

from sage.arith.functions import lcm
from sage.arith.misc import algdep, gcd
from sage.functions.all import log, floor
from sage.functions.other import binomial
from sage.matrix.constructor import matrix
from sage.matrix.matrix_dense import Matrix_dense
from sage.matrix.special import identity_matrix, block_diagonal_matrix
from sage.misc.misc_c import prod
from sage.modules.free_module import VectorSpace
from sage.modules.free_module_element import vector, FreeModuleElement_generic_dense
try:
    from sage.rings.complex_mpfr import ComplexField
except ModuleNotFoundError: # versions of sage older than 9.3
    from sage.rings.complex_field import ComplexField
from sage.rings.polynomial.polynomial_element import Polynomial
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import number_field_elements_from_algebraics
from sage.rings.real_mpfr import RealField


################################################################################
### Some useful functions ######################################################
################################################################################

def customized_accuracy(x):

    """
    Return either the absolute accuracy of x if x contains 0 or the relative
    accuracy of x if x does not contains 0.

    Note that this function works also if x is a vector, a matrix, a polynomial
    or a list (minimum of the accuracies of the coefficients).

    INPUT:
     - 'x' - a complex ball objet

    OUTPUT:
     - 'acc' - a nonnegative integer (max = 9223372036854775807)

    EXAMPLES::

        sage: from ore_algebra.analytic.linear_algebra import customized_accuracy
        sage: a = ComplexBallField().one()
        sage: a.accuracy(), customized_accuracy(a)
        (9223372036854775807, 9223372036854775807)
        sage: a = a/3
        sage: a.accuracy(), customized_accuracy(a)
        (51, 51)
        sage: a = a - 1/3
        sage: a.accuracy(), customized_accuracy(a)
        (-9223372036854775807, 52)

    """

    if x==[]: return 9223372036854775807
    if isinstance(x, FreeModuleElement_generic_dense) or \
    isinstance(x, Matrix_dense) or isinstance(x, Polynomial):
        return customized_accuracy(x.list())
    if isinstance(x, list):
        return min(customized_accuracy(c) for c in x)

    if x.contains_zero():
        if x.rad().is_zero(): return 9223372036854775807
        return max(0, (-log(x.rad(), 2)).floor())

    return x.accuracy()




################################################################################
### Polynomials avec ComplexOptimisticField ####################################
################################################################################

def _clean(pol):

    l = list(pol)
    while len(l)>0 and l[-1].contains_zero(): l.pop()
    cpol = pol.parent()(l)

    return cpol

def GCD(a, b):

    r"""
    Return a *non-rigorous* gcd of the polynomials ``a`` and ``b``.

    Note: this function is designed for BallField as base ring.

    Some words about the correction of this function:
    Let a· and b· be fixed. If a and b are precise enough, GCD(a, b) contains
    the gcd of a· and b·.

    INPUT:

     -- ``a`` -- polynomial
     -- ``b`` -- polynomial

    OUTPUT:

     -- ``a`` -- polynomial


    EXAMPLES::

        sage: from ore_algebra.analytic.linear_algebra import GCD
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
    Return a *non-rigorous* monic gcd of the polynomials ``a`` and ``b`` and the
    coefficients in the Bezout identity.

    Note: this function is designed for BallField as base ring.

    INPUT:

     -- ``a`` -- polynomial
     -- ``b`` -- polynomial

    OUTPUT:

     -- ``d`` -- polynomial
     -- ``u`` -- polynomial
     -- ``v`` -- polynomial


    EXAMPLES::

        sage: from ore_algebra.analytic.linear_algebra import XGCD, _clean
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



def squarefree_part(pol):

    r"""
    Return a *non-rigorous* squarefree part of the polynomial ``pol``.

    Note: this function is designed for BallField as base ring.

    Some words about the correction of this function:
    Let pol· be fixed. If pol is precise enough, squarefree_part(pol) contains
    the squarefree part of pol·.


    INPUT:

     -- ``pol`` -- polynomial


    OUTPUT:

     -- ``sfp`` -- polynomial


    EXAMPLES::

        sage: from ore_algebra.analytic.linear_algebra import squarefree_part
        sage: P.<x> = CBF[]; a = CBF(pi)
        sage: p = (x-1)*(x-2)**2
        sage: p = p(x*a).monic(); p
        ([1.0000000000000 +/- 1.57e-15])*x^3 + ([-1.59154943091895 +/- 5.18e-15])*x^2 + ([0.81056946913870 +/- 2.96e-15])*x + [-0.129006137732798 +/- 1.12e-16]
        sage: sfp = squarefree_part(p); sfp
        ([-44.4132198049 +/- 2.39e-11])*x^2 + ([42.4115008235 +/- 7.69e-11])*x + [-9.0000000000 +/- 3.38e-11]
        sage: sfp.roots(multiplicities=False)
        [[0.3183098862 +/- 2.31e-11] + [+/- 6.86e-12]*I,
         [0.6366197724 +/- 4.20e-11] + [+/- 9.55e-12]*I]
        sage: [1/a, 2/a]
        [[0.318309886183791 +/- 4.43e-16], [0.636619772367581 +/- 4.28e-16]]

    """

    d = GCD(pol, pol.derivative())
    sfp = _clean(pol.quo_rem(d)[0])

    if sfp==0:
        raise PrecisionError("Cannot compute the squarefree part of this polynomial.")

    return sfp


def _derivatives(f, m):

    result = [f]
    for k in range(m):
        f = f.derivative()
        result.append(f)

    return result

def roots(pol, *, multiplicities=False):

    r"""
    Return the roots of the polynomial ``pol``.

    Note: this function is designed for CBF or COF as base ring.

    Some words about the correction of this algorithm:


    INPUT:

     -- ``mat``            -- n×n matrix
     -- ``multiplicities`` -- boolean


    OUTPUT:

     -- ``s`` -- list of complex numbers

    If 'multiplicities=True' is specified, ``s`` is a list of couples (r, m) with
    r a complex number and m a positive integer.


    EXAMPLES::

        sage: from ore_algebra.analytic.linear_algebra import roots
        sage: P.<x> = CBF[]; a = CBF(pi)
        sage: p = (x-1)*(x-2)**2; p = p(x*a).monic()
        sage: roots(p, multiplicities=True)
        [([0.3183098862 +/- 2.31e-11] + [+/- 6.86e-12]*I, 1),
         ([0.6366197724 +/- 4.20e-11] + [+/- 9.55e-12]*I, 2)]
        sage: p.roots(multiplicities=False,proof=False)
        <ipython-input-7-b3a160b9a5ea>:1: UserWarning: roots may have been lost
          p.roots(multiplicities=False,proof=False)
        [[0.3183098861838 +/- 2.69e-14] + [+/- 1.75e-14]*I,
         [0.636620 +/- 8.62e-7] + [+/- 6.48e-7]*I,
         [0.636620 +/- 8.62e-7] + [+/- 6.48e-7]*I]

    """

    K, n = pol.base_ring(), pol.degree()
    if isinstance(K, ComplexOptimisticField):
        pol = pol.change_ring(K._ball_field)

    try:
        res = squarefree_part(pol).roots(multiplicities=False)
        res = [K(r) for r in res]
    except ValueError:
        raise PrecisionError("Cannot compute the roots of this polynomial.") from None

    if not multiplicities: return res

    for j, ev in enumerate(res):
        m = 1
        evaluations = [p(ev) for p in _derivatives(pol, n)]
        while evaluations[m].contains_zero():
            m = m + 1
        res[j] = (ev, m)

    if sum(m for _, m in res)<n:
        raise PrecisionError("Cannot compute multiplicities.")

    return res




################################################################################
### Gaussian reduction and applications ########################################
################################################################################


def row_echelon_form(mat, *, transformation=False, pivots=False, prec_pivots={}):

    r"""
    Return a row echelon form of ``mat``.

    Note: this function is designed for BallField as base ring.

    The computed row echelon form is semi-reduced: the pivots are set to 1 and
    the coefficients below the pivots are set to 0 but the coefficients above
    the pivots may be nonzero.

    Some words about the correction of this function:
    Let (R, T, p) be the output for row_echelon_form(mat, transformation=True, pivots=True).
    For any mat· in mat, there are R· in R and T· in T such that:
    i) for each j of p, R[p[j],j] = 1 and R[i,j] = 0 for i > p[j] (exactly),
    ii) R· = T· * mat· and T· is invertible,
    iii) the length of p cannot exceed the rank of mat·,
    iv) only the len(p) first rows of R do not contain the null row.
    Reversely, let mat· be fixed. If mat is precise enough, no PrecisionError
    is raised and the length of p is equal to the rank of mat·.

    Assumption: for p=prec_pivots, the len(p) first rows of mat satisfy i).


    INPUT:

     -- ``mat``            -- m×n matrix
     -- ``transformation`` -- boolean (optional, default: False)
     -- ``pivots``         -- boolean (optional, default: False)
     -- ``prec_pivots``    -- dictionary (optional, default: {})


    OUTPUT:

     -- ``R`` -- m×n matrix
     -- ``T`` -- m×m matrix (if 'transformation=True' is specified)
     -- ``p`` -- dictionary (if 'pivots=True' is specified)

    The keys of the dictionary ``p`` are the indices of the columns which contain
    a pivot and p[j] is the corresponding row index.


    EXAMPLES:

    An example with a generic 3×3 matrix. ::

        sage: from diffop_factorization.reduction import row_echelon_form
        sage: K = RealBallField(20)
        sage: mat = matrix(K, 3, [RR.random_element() for cpt in range(9)]); mat
        [ [-0.655408 +/- 6.67e-8]  [-0.134570 +/- 4.86e-7]   [0.150069 +/- 2.40e-7]]
        [ [-0.222183 +/- 3.23e-7]  [-0.833284 +/- 2.13e-7]   [0.885925 +/- 5.44e-8]]
        [ [-0.790008 +/- 1.47e-7] [-0.0786588 +/- 3.42e-8]   [0.146726 +/- 9.49e-8]]
        sage: R, T, p = row_echelon_form(mat, transformation=True, pivots=True); R
        [                1.00000  [0.099567 +/- 4.54e-7] [-0.185727 +/- 8.63e-7]]
        [                      0                 1.00000   [-1.0413 +/- 1.48e-5]]
        [                      0                       0                 1.00000]
        sage: T*mat
        [  [1.00000 +/- 4.63e-6]  [0.099567 +/- 4.54e-7] [-0.185727 +/- 8.63e-7]]
        [          [+/- 4.65e-6]    [1.0000 +/- 8.31e-6]   [-1.0413 +/- 1.48e-5]]
        [          [+/- 1.95e-3]           [+/- 4.11e-4]     [1.000 +/- 5.21e-4]]
        sage: p
        {0: 0, 1: 1, 2: 2}

    An example with a singular 3×3 matrix. ::

        sage: from diffop_factorization.reduction import row_echelon_form
        sage: K = RealBallField(20)
        sage: mat = random_matrix(QQ, 3, 3, algorithm='echelonizable', rank=2)
        sage: ran = matrix(K, 3, [RR.random_element() for i in range(9)])
        sage: mat = ~ran * mat * ran; mat
        [ [3.089 +/- 8.73e-4]  [-2.05 +/- 2.67e-3] [-1.466 +/- 7.17e-4]]
        [ [1.024 +/- 7.86e-4] [-10.81 +/- 3.26e-3] [-5.653 +/- 4.33e-4]]
        [[-1.204 +/- 1.78e-4]  [8.927 +/- 4.31e-4]  [4.718 +/- 3.11e-4]]
        sage: row_echelon_form(mat)
        [             1.00000 [-0.664 +/- 5.83e-4] [-0.475 +/- 6.38e-4]]
        [                   0              1.00000  [0.510 +/- 3.73e-4]]
        [                   0                    0        [+/- 2.18e-3]]

    An example with a generic 3×4 matrix. ::

        sage: from diffop_factorization.reduction import row_echelon_form
        sage: K = RealBallField(10)
        sage: mat =  matrix(K, 3, 4, [RR.random_element() for i in range(12)]); mat
        [[-0.794 +/- 2.56e-4]  [0.649 +/- 4.95e-4] [-0.109 +/- 1.70e-4]  [0.798 +/- 8.63e-5]]
        [[-0.645 +/- 3.14e-4] [-0.882 +/- 3.60e-4] [-0.555 +/- 4.51e-4] [-0.986 +/- 4.24e-4]]
        [ [0.298 +/- 5.29e-5]  [0.217 +/- 4.81e-4] [-0.521 +/- 1.37e-4]  [0.626 +/- 3.28e-4]]
        sage: row_echelon_form(mat)
        [               1.00 [-0.82 +/- 6.85e-3] [0.137 +/- 8.34e-4] [-1.00 +/- 6.71e-3]]
        [                  0                1.00  [0.33 +/- 4.73e-3]    [1.2 +/- 0.0535]]
        [                  0                   0                1.00   [-0.5 +/- 0.0772]]

    An example with a generic 4×3 matrix. ::

        sage: from diffop_factorization.reduction import row_echelon_form
        sage: K = RealBallField(20)
        sage: mat =  matrix(K, 4, 3, [RR.random_element() for i in range(12)]); mat
        [  [0.314239 +/- 3.15e-7]   [0.572433 +/- 2.10e-7]   [0.794824 +/- 9.85e-8]]
        [  [0.237214 +/- 1.11e-7]  [-0.579073 +/- 4.26e-7]   [0.444660 +/- 1.36e-7]]
        [  [0.565036 +/- 3.48e-7]   [0.938426 +/- 3.17e-7]  [-0.674036 +/- 1.86e-7]]
        [[-0.0756391 +/- 2.13e-8]  [-0.868542 +/- 2.49e-8]  [-0.827502 +/- 2.90e-7]]
        sage: R, T = row_echelon_form(mat, transformation=True); R
        [               1.00000   [1.6608 +/- 3.04e-5] [-1.19291 +/- 7.88e-6]]
        [                     0                1.00000 [-0.74779 +/- 7.43e-6]]
        [                     0                      0                1.00000]
        [                     0                      0                      0]
        sage: T*mat
        [ [1.00000 +/- 4.68e-6]   [1.6608 +/- 3.04e-5] [-1.19291 +/- 7.88e-6]]
        [         [+/- 4.09e-6]   [1.0000 +/- 9.39e-6] [-0.74779 +/- 7.43e-6]]
        [         [+/- 7.08e-6]          [+/- 2.07e-5]    [1.000 +/- 1.65e-5]]
        [         [+/- 1.86e-5]          [+/- 4.95e-5]          [+/- 4.17e-5]]
        sage: T.det()
        [1.23 +/- 5.18e-3]

    """

    m, n, C = mat.nrows(), mat.ncols(), mat.base_ring()
    T = identity_matrix(C, m)
    p = prec_pivots.copy()
    r = len(p)

    for j in p:
        col = T*vector(mat[:,j])
        for i in range(r, m):
            T[i] = [T[i,k] - col[i]*T[p[j],k] for k in range(m)]

    for j in range(n):
        if not j in p:
            r, col = len(p), T*vector(mat[:,j])
            i = max((l for l in range(r, m) if col[l].is_nonzero()), \
            key=lambda l: col[l].below_abs(), default=None)
            if i is not None:
                p[j] = r
                T[i], T[r], col[i], col[r] = T[r], T[i], col[r], col[i]
                T[r] = [T[r,k]/col[r] for k in range(m)]
                for l in range(r+1, m):
                    T[l] = [T[l,k] - col[l]*T[r,k] for k in range(m)]

    R = T*mat
    for j in p:
        R[p[j],j] = 1
        for i in range(p[j]+1, m): R[i,j] = 0

    if transformation:
        if T.det().contains_zero():
            raise PrecisionError("Cannot compute an invertible matrix.")
        if pivots: return R, T, p
        else: return R, T
    if pivots: return R, p
    else: return R


def orbit(Mats, vec, *, transition=False, pivots=False):

    r"""
    Return a basis of the smallest subspace containing ``vec`` and invariant under
    (the action of) the matrices of ``Mats``.

    Note: this function is designed for BallField as base ring.

    Some words about the correction of this function:
    Let (b, T) be the output for orbit(Mats, vec, transition=True). For any
    selection [mat1·, ..., matk·] in Mats = [mat1, ..., matk] and any vec· in
    vec, there are a selection [b1·, ..., bs·] in b = [b1, ..., bs] and a
    selection [T1·, ..., Ts·] in T = [T1, ..., Ts] such that:
    i) the subspace spanned by b1·, ..., bs· is contained in the smallest
    subspace containing vec· and invariant under mat1·, ..., matk·,
    ii) the vectors b1·, ..., bs· are linearly independent,
    iii) for each i, bi· = Ti·*vec· and Ti· is polynomial in mat1·, ..., matk·.
    Reversely, let [mat1·, ..., matk·] and vec· be fixed. If Mats and vec are
    precise enough, no PrecisionError is raised and there is selection
    [b1·, ..., bs·] which is a basis of the smallest subspace containing vec·
    and invariant under mat1·, ..., matk·.

    In particular, if b has length the dimension, this proves that whatever the
    selection [mat1·, ..., matk·] in Mats and whatever vec· in vec, the smallest
    subspace containing vec· and invariant under mat1·, ..., matk· is the entire
    space.


    INPUT:

     -- ``Mats``       -- list of n×n matrices
     -- ``vec``        -- vector of size n
     -- ``transition`` -- boolean (optional, default: False)
     -- ``pivots``     -- boolean (optional, default: False)

    OUTPUT:

     -- ``b`` -- list of vectors of size n
     -- ``T`` -- list of n×n matrix (if 'transition=True' is specified)
     -- ``p`` -- dictionary (if 'pivots=True' is specified)


    EXAMPLES:

    An example with one matrix. ::

        sage: from diffop_factorization.reduction import orbit
        sage: mat = matrix(RBF, [[1, 1, 0], [0, 1, 1], [0, 0, 1]])
        sage: u, v, w = list(identity_matrix(RBF, 3))
        sage: ran = MatrixSpace(RR, 3).random_element().change_ring(RBF)
        sage: u, v, w, mat = ~ran * u, ~ran * v, ~ran * w, ~ran * mat * ran
        sage: len(orbit([mat], u)), len(orbit([mat], v)), len(orbit([mat], w))
        (1, 2, 3)
        sage: b, T = orbit([mat], v, transition=True)
        sage: b[0], T[0]*v
        ((1.000000000000000, [1.630667151553 +/- 2.48e-13], [1.710645064117 +/- 4.32e-13]),
         ([1.00000000000 +/- 2.41e-13], [1.630667151553 +/- 2.48e-13], [1.710645064117 +/- 4.32e-13]))
        sage: b[1], T[1]*v
        ((0, 1.000000000000000, [5.2453796816 +/- 5.24e-11]),
         ([+/- 5.79e-12], [1.0000000000 +/- 1.53e-11], [5.2453796816 +/- 5.25e-11]))

    An example with two matrices. ::

        sage: from diffop_factorization.reduction import orbit
        sage: mat1 = matrix(CBF, [[0, 0, 1], [0, 0, 0], [0, 0, 0]])
        sage: mat2 = matrix(CBF, [[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        sage: vec = vector(CBF, [0, 0, 1])
        sage: ran = MatrixSpace(CC, 3).random_element().change_ring(CBF)
        sage: vec, mat1, mat2 =  ~ran * vec, ~ran * mat1 * ran, ~ran * mat2 * ran
        sage: len(orbit([mat1], u)), len(orbit([mat1, mat2], u))
        (2, 3)

    """

    n, C = len(vec), vec.base_ring()

    b, S, p = row_echelon_form(matrix(vec), transformation=True, pivots=True)
    if transition: T = [S[0,0]*identity_matrix(C, n)]

    if len(p)==0: # case where vec contains the null vector
        if transition: return [], []
        else: return []

    r, new = 1, range(0, 1)
    while len(new) > 0 and r < n:

        b = b.stack(matrix([mat*vector(b[i]) for mat in Mats for i in new]))
        if transition:
            T.extend([mat*T[i] for mat in Mats for i in new])

        if transition:
            b, S, p = row_echelon_form(b, transformation=True, pivots=True, prec_pivots = p)
        else:
            b, p = row_echelon_form(b, pivots=True, prec_pivots = p)

        new = range(r, len(p))
        if transition:
            T[r:] = [sum(S[i,j]*Tj for j, Tj in enumerate(T)) for i in new]
        r = len(p)
        b = b[:r]

    b = list(b)
    if transition:
        if pivots: return b, T, p
        else : return b, T
    if pivots: return b, p
    else: return b



def generated_algebra(Mats, transformation=False):

    r"""
    Return a basis of the unitary algebra generated by the matrices of ``Mats``.

    Note: this function is designed for BallField as base ring.
    Let b be the output for generated_algebra(Mats). For any selection
    [mat1·, ..., matk·] in Mats = [mat1, ..., matk], there is a selection
    [b1·, ..., bs·] in b = [b1, ..., bs] such that:
    i) the subspace spanned by b1·, ..., bs· is contained in the algebra
    generated by mat1·, ..., matk·,
    ii) the matrices b1·, ..., bs· are linearly independent,
    iii) for each i, bi· is polynomial in mat1·, ..., matk·.
    Reversely, let [mat1·, ..., matk·] be fixed. If Mats is precise enough, no
    PrecisionError is raised and there is selection [b1·, ..., bs·] in the
    output which is a basis of the algebra generated by mat1·, ..., matk·.

    In particular, if b has length n×n where n is the dimension, this proves
    that whatever the selection [mat1·, ..., matk·] in Mats, the matrices
    mat1·, ..., matk· generate the entire algebra of matrices.

    INPUT:

     -- ``Mats`` -- list of n×n matrices

    OUTPUT:

     -- ``b`` -- list of n×n matrices


    EXAMPLES::

        sage: from diffop_factorization.reduction import generated_algebra
        sage: n, C = 4, ComplexBallField(100)
        sage: mat1 = MatrixSpace(CC, n).random_element().change_ring(C)
        sage: mat2 = MatrixSpace(CC, n).random_element().change_ring(C)
        sage: len(generated_algebra([mat1, mat2]))
        16
        sage: len(generated_algebra([mat1**2, mat1**3]))
        4

    """


    mat = Mats[0]
    n, C = mat.nrows(), mat.base_ring()

    if transformation:
        b, T, p = row_echelon_form(matrix([mat.list() for mat in Mats]), transformation=True, pivots=True)
        l = [sum(T[i,j]*Mats[j] for j in range(len(Mats))) for i in range(len(p))]
    else: b, p = row_echelon_form(matrix([mat.list() for mat in Mats]), pivots=True)

    r, b, new = len(p), b[:len(p)], range(0, len(p))

    while len(new)>0 and r<n**2:

        b = b.stack(matrix([(matrix(n,b[i])*matrix(n,b[j])).list() for i in range(r) for j in new]))
        if transformation:
            l.extend(l[i]*l[j] for i in range(r) for j in new)
            b, T, p = row_echelon_form(b, transformation=True, pivots=True, prec_pivots = p)
            l = [sum(T[i,j]*l[j] for j in range(len(l))) for i in range(len(p))]
        else: b, p = row_echelon_form(b, pivots=True, prec_pivots = p)

        r, b, new = len(p), b[:len(p)], range(r, len(p))

    b = [matrix(n, x) for x in list(b)]

    if transformation: return b, l
    return b



def ker(mat):

    r"""
    Return a basis of the right kernel of ``mat``.

    Note: this function is designed for BallField as base ring.

    Some words about the correction of this function:
    Let b be the output for ker(mat). For any mat· in mat, there are b· in b
    such that
    i) Ker(mat·) is include in Span(b·),
    ii) the vectors of b· are linearly independent.
    Reversely, let mat· be fixed. If mat is precise enough, no PrecisionError
    is raised and the inclusion is an equality.


    INPUT:

     -- ``mat`` -- m×n matrix


    OUTPUT:

     -- ``b`` -- list of vectors of size n


    EXAMPLE::

        sage: from diffop_factorization.reduction import ker
        sage: mat = random_matrix(QQ, 3, algorithm='diagonalizable', eigenvalues=[0,1], dimensions=[2,1])
        sage: ran = MatrixSpace(CC, 3).random_element().change_ring(CBF)
        sage: mat = ~ran * mat * ran; mat
        [[-3.93916258348 +/- 3.39e-12] + [-29.98301156589 +/- 5.56e-12]*I [-32.31388378951 +/- 4.75e-12] + [-7.27390030643 +/- 3.95e-12]*I  [4.31307439346 +/- 4.62e-12] + [-20.93143334627 +/- 7.06e-12]*I]
        [  [-9.32991924526 +/- 4.57e-12] + [8.03479991844 +/- 1.94e-12]*I   [4.73706780030 +/- 2.23e-12] + [12.62682833737 +/- 1.31e-12]*I   [-8.09008374791 +/- 4.77e-12] + [3.20418406755 +/- 4.82e-12]*I]
        [  [8.31773777000 +/- 1.05e-12] + [23.10971869354 +/- 1.23e-12]*I [26.900986486615 +/- 9.18e-13] + [0.176324436596 +/- 6.03e-13]*I  [0.202094783181 +/- 7.11e-13] + [17.35618322852 +/- 3.05e-12]*I]
        sage: ker(mat)
        [(1.000000000000000, [-0.314815526682 +/- 6.76e-13] + [-0.857002364379 +/- 5.99e-13]*I, 0),
         (0, [-0.011740965563 +/- 3.70e-13] + [-0.645110654883 +/- 3.88e-13]*I, 1.000000000000000)]
        sage: v1, v2 = ker(mat)
        sage: mat*v1, mat*v2
        (([+/- 1.59e-11] + [+/- 1.59e-11]*I, [+/- 6.85e-12] + [+/- 6.86e-12]*I, [+/- 7.76e-12] + [+/- 7.79e-12]*I),
         ([+/- 1.21e-11] + [+/- 1.21e-11]*I, [+/- 5.20e-12] + [+/- 5.20e-12]*I, [+/- 6.05e-12] + [+/- 6.08e-12]*I))

    """

    R, T, p = row_echelon_form(mat.transpose(), transformation=True, pivots=True)
    b = list(T[len(p):])

    return b



def intersection(K1, K2):

    r"""
    Compute the intersection of two subspaces.
    This function is designed for ComplexBall elements (optimistic arithmetic).

    INPUT:

     -- ``K1`` -- list of vectors
     -- ``K2`` -- list of vectors

    OUTPUT:
     -- ``K`` -- list of vectors

    """

    K = ker(matrix(K1 + K2).transpose())
    K = [sum(v[i]*K1[i] for i in range(len(K1))) for v in K]
    ref, p = row_echelon_form(matrix(K), pivots=True)
    K = list(ref[:len(p)])

    return K


################################################################################
### Eigenvalues, eigenvectors ##################################################
################################################################################


def eigenvalues(mat, multiplicities=False):

    r"""
    Return the eigenvalues of \\mat``.

    Note: this function is designed for ComplexBallField as base ring.

    See function 'roots' of polynomials module for more details.
    """

    eigvals = roots(mat.charpoly(algorithm="df"), multiplicities=multiplicities)

    return eigvals



def gen_eigenspaces(mat, *, projections=False):

    r"""
    Return the generalized eigenspaces of ``mat``.

    Note: this function is designed for ComplexBallField as base ring.

    Some words about the correction of this algorithm:
    Let GenEigSpaces be the output for gen_eigenspaces(mat, projections=True).
    For any mat· in mat, there is a selection [space1·, ..., spacek·] in
    GenEigSpaces=[space1, ..., spacek] where each spacei· is a selection
    {'eigenvalue' : li·, 'multiplicity' : mi, 'eigenvectors' : bi·,
    'projection' : pi·} in spacei, such that:
    i) exactly mi eigenvalue(s) belong(s) to li,
    ii) the li· are pairwise disjoints,
    iii) the sum of the mi is equal to the dimension,
    iv) for each i, bi· is a list of mi linearly independent vectors.
    Reversely, let mat· be fixed. If mat is precise enough, no PrecisionError
    is raised and the spaces in gen_eigspaces(mat) correspond to the generalized
    eigenspaces of mat one-to-one.

    INPUT:

     -- ``mat``         -- n×n matrix
     -- ``projections`` -- boolean


    OUTPUT:

     -- ``GenEigSpaces`` -- list of dictionary

    Each dictionary of ``GenEigSpaces`` represents a generalized eigenspace of
    ``mat``, whose keys are the following strings:
     - 'eigenvalue'   : complex number
     - 'multiplicity' : integer
     - 'basis'        : list of vectors
     - 'projection'   : polynomial (if 'projections=True' is specified).


    EXAMPLES:

    A generic example ::

        sage: from diffop_factorization.eigenspaces import gen_eigenspaces
        sage: mat = MatrixSpace(CC, 3).random_element().change_ring(CBF)
        sage: gen_eigenspaces(mat)
        [{'eigenvalue': [0.008731294334 +/- 5.79e-13] + [-0.457554487127 +/- 3.74e-13]*I,
          'multiplicity': 1,
          'basis': [([-0.081853906993 +/- 6.52e-13] + [-0.087179008676 +/- 7.81e-13]*I, 1.000000000000000, [-0.342339920094 +/- 3.62e-13] + [0.098986705031 +/- 3.30e-13]*I)]},
         {'eigenvalue': [0.339079560015 +/- 4.32e-13] + [0.888549954437 +/- 3.85e-13]*I,
          'multiplicity': 1,
          'basis': [([-0.409699282668 +/- 6.41e-13] + [0.247473920868 +/- 5.86e-13]*I, [-0.066871473678 +/- 7.28e-13] + [0.685673697712 +/- 7.82e-13]*I, 1.000000000000000)]},
         {'eigenvalue': [0.606834685948 +/- 7.61e-13] + [-0.992982372445 +/- 8.72e-13]*I,
          'multiplicity': 1,
          'basis': [([-0.63246660126 +/- 2.18e-12] + [-0.05887072347 +/- 3.44e-12]*I, 1.000000000000000, [-0.59541382749 +/- 5.20e-12] + [0.10852249946 +/- 5.37e-12]*I)]}]

    An example with a multiple eigenvalue ::

        sage: from diffop_factorization.eigenspaces import gen_eigenspaces
        sage: r2, r3 = CBF(sqrt(2)), CBF(sqrt(3))
        sage: mat = jordan_block(r2, 2).block_sum(matrix([r3]))
        sage: ran = MatrixSpace(CC, 3).random_element().change_ring(CBF)
        sage: mat = ~ran * mat * ran
        sage: GenEigSpaces = gen_eigenspaces(mat, projections=True)
        sage: [(space['eigenvalue'], space['multiplicity']) for space in GenEigSpaces]
        [([1.41421356 +/- 7.84e-9] + [+/- 5.47e-9]*I, 2),
         ([1.73205081 +/- 8.48e-9] + [+/- 6.05e-9]*I, 1)]
        sage: ev, vec = GenEigSpaces[1]['eigenvalue'], GenEigSpaces[1]['basis'][0]
        sage: (mat - ev*identity_matrix(CBF, 3))*vec
        ([+/- 1.37e-6] + [+/- 1.37e-6]*I, [+/- 4.33e-7] + [+/- 4.33e-7]*I, [+/- 1.17e-6] + [+/- 1.17e-6]*I)
        sage: T = matrix(GenEigSpaces[0]['basis'] + GenEigSpaces[1]['basis']).transpose()
        sage: pol = GenEigSpaces[0]['projection']
        sage: P = ~T * pol(mat) * T; P
        [  [1.00 +/- 1.19e-4] + [+/- 1.19e-4]*I        [+/- 8.62e-5] + [+/- 8.62e-5]*I        [+/- 1.53e-4] + [+/- 1.53e-4]*I]
        [       [+/- 3.97e-5] + [+/- 3.97e-5]*I [1.0000 +/- 4.89e-5] + [+/- 4.89e-5]*I        [+/- 5.52e-5] + [+/- 5.52e-5]*I]
        [       [+/- 9.55e-5] + [+/- 9.55e-5]*I        [+/- 7.19e-5] + [+/- 7.19e-5]*I        [+/- 1.25e-4] + [+/- 1.25e-4]*I]

    """

    n, C = mat.nrows(), mat.base_ring()
    I = identity_matrix(C, n)
    Pol, x = PolynomialRing(C, 'x').objgen()

    s = eigenvalues(mat, multiplicities=True)

    if projections:
        k = len(s)
        P = [(x - ev)**m for ev, m in s]
        Q = [Pol(prod(P[j] for j in range(k) if j != i)) for i in range(k)]
        d, u1, u2 = XGCD(Pol(Q[0]), Pol(sum(Q[1:])))
        proj = [u*q for u, q in zip([u1] + [u2]*(k - 1), Q)]

    GenEigSpaces = []
    for i, (ev, m) in enumerate(s):
        b = ker((mat - ev*I)**m)
        if len(b)!=m:
            raise PrecisionError("Cannot compute a basis of this generalized eigenspace. ")
        space = {'eigenvalue' : ev, 'multiplicity' : m, 'basis' : b}
        if projections:
            space['projection'] = proj[i]
        GenEigSpaces.append(space)

    return GenEigSpaces

################################################################################
### The class Splitting as in [van der Hoeven, 2007] ###########################
################################################################################


class Splitting():

    r"""
    """

    def __init__(self, Mats):

        self.n = Mats[0].nrows()
        self.C = Mats[0].base_ring()
        self.I = identity_matrix(self.C, self.n)

        self.matrices = Mats.copy()
        self.partition = [self.n]
        self.basis = self.I          # column-wise
        self.projections = [self.I]



    def refine(self, mat):

        new_dec, s = [], 0
        for j, nj in enumerate(self.partition):
            new_dec.append(gen_eigenspaces(mat.submatrix(s, s, nj, nj), projections=True))
            s = s + nj

        self.partition = [space['multiplicity'] for bloc in new_dec for space in bloc]
        self.projections = [p*space['projection'](mat)*p for j, p in enumerate(self.projections) for space in new_dec[j]]

        T = matrix()
        for bloc in new_dec:
            basis = []
            for s in bloc:
                basis.extend(s['basis'])
            T = T.block_sum(matrix(basis).transpose())
        self.basis = T*self.basis
        try:
            invT = ~T
        except ZeroDivisionError:
            raise PrecisionError("Cannot compute the transition to the old basis from the new one.")
        self.matrices = [invT*M*T for M in self.matrices]
        self.projections = [invT*p*T for p in self.projections]

        return



    def check_lines(self):

        s = 0
        for j, nj in enumerate(self.partition):
            if nj==1:
                p = self.projections[j]
                p[s,s] = p[s,s] - self.C.one()
                err = max(sum(p[i,j].above_abs() for j in range(self.n)) for i in range(self.n))
                err = self.C.zero().add_error(err)
                vec = self.I[s] + vector([err]*self.n)
                V = orbit(self.matrices, vec)
                if len(V)==0:
                    raise PrecisionError('Projections are not precise enough.')
                if len(V)<self.n:
                    T = self.basis
                    V = [T*v for v in V]
                    return (True, V)
            s = s + nj

        return (False, None)



    def COF_version(self):

        prec1 = self.C.precision()
        prec2 = min(customized_accuracy(M) for M in self.matrices)
        prec2 = min(prec2, customized_accuracy(self.basis))

        if 2*prec2<prec1: raise PrecisionError("Losing too much precision to continue.")

        COF = ComplexOptimisticField(prec1, eps = RealField(30).one()>>(3*prec1//8))

        Mats = [M.change_ring(COF) for M in self.matrices]
        b = self.basis.change_ring(COF)

        return COF, Mats, b



    def check_nolines(self, verbose=False, returnK=False):

        (COF, Mats, b), p = self.COF_version(), self.partition

        s=0
        for j, nj in enumerate(p):
            if nj>1:
                if verbose: print('Check in a subspace of dimension', nj)
                ind = range(s, s + nj)
                basis =  identity_matrix(COF, self.n)[s:s+nj]
                K = VectorSpace(COF, nj)
                for M in Mats:
                    mat = M.matrix_from_rows_and_columns(ind, ind)
                    K = intersect_eigenvectors(K, mat)
                if returnK: return K
                while K.dimension()>0:
                    if verbose: print('dim K =', K.dimension())
                    vec0 = vector(COF, [0]*s + list(K.basis()[0]) + [0]*(self.n - s - nj))
                    V, T, p = orbit(Mats, vec0, transition=True, pivots=True)
                    if len(V)<self.n:
                        V = [(b*v).change_ring(self.C) for v in V]
                        return (True, V)
                    vec1 = basis[0]
                    if len(row_echelon_form(matrix([vec0, vec1]), pivots=True)[1])==1: vec1 = basis[1]
                    lc = linear_combination(vec1, V, p)
                    M = sum(cj*T[j] for j, cj in enumerate(lc))
                    mat = M.matrix_from_rows_and_columns(ind, ind)
                    if len(eigenvalues(mat))>1: return ('new_matrix', M)
                    K = intersect_eigenvectors(K, mat)

            s = s + nj

        return (False, None)



def linear_combination(vec, Vecs, p):

    n = len(Vecs)
    p = {value:key for key, value in p.items()}
    lc = [0]*n
    for i in range(n-1, -1, -1):
        x = vec[p[i]]
        vec = vec - x*Vecs[i]
        lc[p[i]] = x

    return lc



def intersect_eigenvectors(K, mat):

    eigvals = eigenvalues(mat)
    if len(eigvals)>1:
        raise PrecisionError('This matrix seems have several eigenvalues.')
    K = K.intersection((mat-eigvals[0]*(mat.parent().one())).right_kernel())
    return K



def invariant_subspace(Mats, *, verbose=False):

    r"""
    Return either a nontrivial subspace invariant under the action of the
    matrices of ``Mats`` or None if there is none.

    Note: this function is designed for BallField as base ring.

    Note: only the output None is rigorous, in the following sense. If
    invariant_subspace(Mats) is None than for any [M1·, ..., Mr·] in
    Mats=[M1, ..., Mr], there is no nontrivial subpace invariant under the
    action of M1·, ..., Mr·.

    INPUT:

     -- ``Mats`` -- list of n×n matrices


    OUTPUT:

     -- ``V`` -- list of vectors of size n or None


    EXAMPLE::

        sage: from diffop_factorization import invariant_subspace
        sage: C = ComplexBallField(30)
        sage: mat1 = matrix(C, [[1, 1], [1, 1]])
        sage: mat2 = matrix(C, [[3, -1], [0 , 2]])
        sage: invariant_subspace([mat1, mat2])
        [([1.00 +/- 1.17e-3] + [+/- 1.17e-3]*I, [1.00 +/- 9.96e-4] + [+/- 9.96e-4]*I)]

    """

    if Mats==[]: raise TypeError("This function requires at least one matrix.")

    n, C = Mats[0].nrows(), Mats[0].base_ring()
    if len(Mats)==1:
        mat = Mats[0]
        Spaces = gen_eigenspaces(mat)
        v = ker(mat - Spaces[0]['eigenvalue']*(mat.parent().one()))[0]
        orb = orbit(Mats, v)
        if len(orb)==n: raise Exception("problem with invariant_subspace computation: case 'one matrix'")
        return orb

    split = Splitting(Mats)
    mat = sum(C(ComplexField().random_element())*M for M in split.matrices)
    split.refine(mat)

    hope = True
    while hope:
        if verbose: print("The partition is currently " + str(split.partition) + ".")
        b, V = split.check_lines()
        if b: return V
        if verbose: print("Lines checked.")

        if len(split.partition)==n: return None
        if verbose: print("Need to check nolines.")

        b, x = split.check_nolines(verbose=verbose)
        if b=='new_matrix': split.refine(x)
        elif b: return x
        else: hope=False

    if verbose: print("Need to compute a basis of the algebra.")
    Mats = generated_algebra(Mats)
    if len(Mats)==n**2:
        return None
    else:
        if verbose: print("Restart with the basis of the algebra.")
        return invariant_subspace(Mats)


def _commutation_problem_as_linear_system(mat):

    r"""
    Compute S such that (x_{i,j})*mat = mat*(x_{i,j}) iff the vector
    (x_{1,1}, x_{1,2}, ..., x_{n,n}) belongs to the right kernel of S.
    """

    n, matT = mat.nrows(), mat.transpose()
    S = block_diagonal_matrix([matT]*n)
    for i in range(n):
        li, l = list(mat[i]), []
        for k in range(n): l.extend([li[k]] + [0]*(n-1))
        for j in range(n):
            S[n*i + j] = [m - l[k] for k, m in enumerate(S[n*i + j])]
            l = [0] + l[:-1]

    return S



def centralizer(alg):

    r"""
    Compute the centralizer of an algebra.
    This function is designed for ComplexBall entries (optimistic arithmetic).

    INPUT:

     -- ``alg`` -- list of matrices

    OUTPUT:
     -- ``C`` -- list of matrices

    """

    mat = alg[0]; n = mat.nrows()

    S = _commutation_problem_as_linear_system(mat)
    K = list(row_echelon_form(matrix(ker(S))))

    for mat in alg[1:]:
        S = _commutation_problem_as_linear_system(mat)
        K = intersection(K, ker(S))

    C = [matrix(n, n, v) for v in K]

    return C
