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
        sage: from ore_algebra.analytic.linear_algebra import row_echelon_form
        sage: mat = matrix(RealBallField(20), 3, 3, [8.16,56.6,-17,5.11,-4.5,-7,89.2,66,-44.4]); mat
        [ [8.16000 +/- 1e-9]  [56.6000 +/- 1e-9]            -17.0000]
        [[5.11000 +/- 1e-10]            -4.50000            -7.00000]
        [ [89.2000 +/- 1e-8]             66.0000 [-44.4000 +/- 3e-9]]
        sage: R, T, p = row_echelon_form(mat, transformation=True, pivots=True); R
        [               1.00000  [0.73991 +/- 3.79e-6] [-0.49776 +/- 4.39e-6]]
        [                     0                1.00000 [-0.25589 +/- 5.04e-6]]
        [                     0                      0                1.00000]
        sage: T*mat
        [  [1.0000 +/- 4.61e-6]  [0.73991 +/- 3.79e-6] [-0.49776 +/- 4.39e-6]]
        [         [+/- 3.37e-6]    [1.000 +/- 1.05e-5] [-0.25589 +/- 5.04e-6]]
        [         [+/- 4.07e-5]          [+/- 5.77e-5]    [1.000 +/- 3.84e-5]]
        sage: p
        {0: 0, 1: 1, 2: 2}

    An example with a singular 3×3 matrix. ::
        sage: from ore_algebra.analytic.linear_algebra import row_echelon_form
        sage: mat = matrix(RealBallField(20), 3, 3, [i/pi for i in [1,2..9]]); mat
        [[0.31831 +/- 1.65e-6] [0.63662 +/- 4.25e-6] [0.95493 +/- 7.33e-6]]
        [[1.27324 +/- 8.50e-6]  [1.5915 +/- 5.28e-5]  [1.9099 +/- 5.47e-5]]
        [ [2.2282 +/- 4.68e-5]  [2.5465 +/- 3.70e-5]  [2.8648 +/- 3.49e-5]]
        sage: R, T, p = row_echelon_form(mat, transformation=True, pivots=True); R
        [             1.00000 [1.1429 +/- 5.76e-5] [1.2857 +/- 2.81e-5]]
        [                   0              1.00000  [2.000 +/- 2.26e-4]]
        [                   0                    0        [+/- 2.42e-4]]
        sage: T*mat
        [ [1.000 +/- 1.33e-5] [1.1429 +/- 5.76e-5] [1.2857 +/- 2.81e-5]]
        [       [+/- 1.08e-4]   [1.00 +/- 1.64e-4]  [2.000 +/- 2.26e-4]]
        [       [+/- 1.25e-4]        [+/- 1.84e-4]        [+/- 2.42e-4]]
        sage: p
        {0: 0, 1: 1}

    An example with a generic 2×3 matrix. ::
        sage: from ore_algebra.analytic.linear_algebra import row_echelon_form
        sage: mat = matrix(RealBallField(20), 2, 3, [8.16,56.6,-17,5.11,-4.5,-7]); mat
        [ [8.16000 +/- 1e-9]  [56.6000 +/- 1e-9]            -17.0000]
        [[5.11000 +/- 1e-10]            -4.50000            -7.00000]
        sage: R, T, p = row_echelon_form(mat, transformation=True, pivots=True); R
        [               1.00000   [6.9363 +/- 4.81e-5]  [-2.0833 +/- 4.57e-5]]
        [                     0                1.00000 [-0.09127 +/- 8.09e-6]]
        sage: T*mat
        [ [1.00000 +/- 4.76e-6]   [6.9363 +/- 4.81e-5]  [-2.0833 +/- 4.57e-5]]
        [         [+/- 3.59e-6]   [1.0000 +/- 1.68e-5] [-0.09127 +/- 8.09e-6]]

    An example with a generic 3×2 matrix. ::
        sage: from ore_algebra.analytic.linear_algebra import row_echelon_form
        sage: mat = matrix(RealBallField(20), 3, 2, [8.16,56.6,-17,5.11,-4.5,-7]); mat
        [ [8.16000 +/- 1e-9]  [56.6000 +/- 1e-9]]
        [           -17.0000 [5.11000 +/- 1e-10]]
        [           -4.50000            -7.00000]
        sage: R, T, p = row_echelon_form(mat, transformation=True, pivots=True); R
        [                1.00000 [-0.300588 +/- 9.13e-7]]
        [                      0                 1.00000]
        [                      0                       0]
        sage: T*mat
        [   [1.0000 +/- 2.93e-6] [-0.300588 +/- 9.13e-7]]
        [          [+/- 2.63e-6]   [1.00000 +/- 7.14e-6]]
        [          [+/- 5.32e-5]           [+/- 9.13e-5]]
        sage: T.det()
        [0.0009961 +/- 2.35e-8]
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
        sage: from ore_algebra.analytic.linear_algebra import orbit
        sage: mat = matrix(RBF, [[1, 1, 0], [0, 1, 1], [0, 0, 1]])
        sage: u, v, w = list(identity_matrix(RBF, 3))
        sage: ran = matrix(RBF, 3, 3, [8.16,56.6,-17,5.11,-4.5,-7,89.2,66,-44.4])
        sage: u, v, w, mat = ~ran*u, ~ran*v, ~ran*w, ~ran*mat*ran
        sage: len(orbit([mat], u)), len(orbit([mat], v)), len(orbit([mat], w))
        (1, 2, 3)
        sage: b, T = orbit([mat], v, transition=True)
        sage: b[0], T[0]*v
        ((1.000000000000000, [0.8296641361859 +/- 5.41e-14], [3.242293535772 +/- 3.24e-13]),
         ([1.000000000000 +/- 6.51e-14], [0.8296641361859 +/- 5.41e-14], [3.242293535772 +/- 3.24e-13]))
        sage: b[1], T[1]*v
        ((0, 1.000000000000000, [1.4864864865 +/- 4.16e-11]),
         ([+/- 3.89e-11], [1.000000000 +/- 2.97e-11], [1.486486486 +/- 6.24e-10]))

    An example with two matrices. ::
        sage: from ore_algebra.analytic.linear_algebra import orbit
        sage: mat1 = matrix(CBF, [[0, 0, 1], [0, 0, 0], [0, 0, 0]])
        sage: mat2 = matrix(CBF, [[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        sage: vec = vector(CBF, [0, 0, 1])
        sage: ran = MatrixSpace(CC, 3).random_element().change_ring(CBF)
        sage: vec, mat1, mat2 =  ~ran*vec, ~ran*mat1*ran, ~ran*mat2*ran
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

        sage: from ore_algebra.analytic.linear_algebra import generated_algebra
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


    EXAMPLES::
        sage: from ore_algebra.analytic.linear_algebra import ker
        sage: mat = matrix(QQ, 3, 3, [-6, -12, -6, 3, 6, 3, 1, 2, 1]) # such that dim(ker)=2
        sage: ran = matrix(CBF, 3, 3, [8.16,56.6,-17,5.11,-4.5,-7,89.2,66,-44.4])
        sage: mat = ~ran*mat*ran; mat
        [   [0.980818359715 +/- 5.23e-13]    [1.035703343221 +/- 4.40e-13]  [-0.6874298598489 +/- 5.30e-14]]
        [[-21.1037881400433 +/- 4.23e-14] [-22.2847214418007 +/- 5.36e-14]  [14.7910915203501 +/- 2.76e-14]]
        [[-31.8229959359513 +/- 9.12e-14]  [-33.603758489720 +/- 1.58e-13]  [22.3039030820852 +/- 8.40e-14]]
        sage: ker(mat)
        [(1.000000000000000, [-0.947007042254 +/- 6.10e-13], 0),
         (0, [0.663732394366 +/- 2.89e-13], 1.000000000000000)]
        sage: v1, v2 = ker(mat); mat*v1, mat*v2
        (([+/- 2.72e-13], [+/- 3.00e-12], [+/- 4.56e-12]),
         ([+/- 1.91e-13], [+/- 2.11e-12], [+/- 3.20e-12]))
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
### Polynomials ################################################################
################################################################################

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


def roots(pol, *, multiplicities=False):
    r"""
    Return the roots of the polynomial ``pol``.

    Note: this function is designed for CBF or COF as base ring.

    Some words about the correction of this algorithm:


    INPUT:
     -- ``mat``            -- n×n matrix
     -- ``multiplicities`` -- boolean


    OUTPUT: a list of complex numbers

    If `multiplicities=True` is specified, ``s`` is a list of couples (r, m) with
    r a complex number and m a positive integer.


    EXAMPLES::
        sage: from ore_algebra.analytic.linear_algebra import roots
        sage: P.<x> = CBF[]; a = CBF(pi)
        sage: p = (x-1)*(x-2)**2; p = p(x*a).monic()
        sage: roots(p, multiplicities=True)
        [([0.3183098862 +/- 2.31e-11] + [+/- 6.86e-12]*I, 1),
         ([0.6366197724 +/- 4.20e-11] + [+/- 9.55e-12]*I, 2)]
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
### Eigenvalues, eigenvectors ##################################################
################################################################################

def eigenvalues(mat, multiplicities=False):
    r"""
    Return the eigenvalues of \\mat``.

    Note: this function is designed for ComplexBallField as base ring.

    See function `roots` of polynomials module for more details.
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
        sage: from ore_algebra.analytic.linear_algebra import gen_eigenspaces
        sage: mat = matrix(CBF, 3, 3, [8.16,56.6,-17,5.11,-4.5,-7,89.2,66,-44.4])
        sage: gen_eigenspaces(mat)
            [{'basis': [([-0.042000493002 +/- 5.15e-13] + [+/- 2.83e-13]*I, [0.326161958389 +/- 3.36e-13] + [+/- 1.79e-13]*I, 1.000000000000000)],
              'eigenvalue': [-26.6197547221 +/- 1.30e-11] + [+/- 6.99e-12]*I,
              'multiplicity': 1},
             {'basis': [([0.380400612236 +/- 8.71e-13] + [0.254042964594 +/- 8.64e-13]*I, [0.051638526510 +/- 6.77e-13] + [0.150937316687 +/- 4.22e-13]*I, 1.000000000000000)],
              'eigenvalue': [-7.06012263895 +/- 8.17e-12] + [32.62249534308 +/- 5.22e-12]*I,
              'multiplicity': 1},
             {'basis': [([0.380400612236 +/- 8.71e-13] + [-0.254042964594 +/- 8.64e-13]*I, [0.051638526510 +/- 6.77e-13] + [-0.150937316687 +/- 4.22e-13]*I, 1.000000000000000)],
              'eigenvalue': [-7.06012263895 +/- 8.17e-12] + [-32.62249534308 +/- 5.22e-12]*I,
              'multiplicity': 1}]

    An example with a multiple eigenvalue ::
        sage: from ore_algebra.analytic.linear_algebra import gen_eigenspaces
        sage: r2, r3 = CBF(sqrt(2)), CBF(sqrt(3))
        sage: mat = jordan_block(r2, 2).block_sum(matrix([r3]))
        sage: ran = ran = matrix(CBF, 3, 3, [8.16,56.6,-17,5.11,-4.5,-7,89.2,66,-44.4])
        sage: mat = ~ran*mat*ran
        sage: GenEigSpaces = gen_eigenspaces(mat, projections=True)
        sage: [(space['eigenvalue'], space['multiplicity']) for space in GenEigSpaces]
        [([1.41421356 +/- 4.10e-9] + [+/- 1.72e-9]*I, 2),
         ([1.73205081 +/- 4.34e-9] + [+/- 1.90e-9]*I, 1)]
        sage: ev, vec = GenEigSpaces[1]['eigenvalue'], GenEigSpaces[1]['basis'][0]
        sage: (mat - ev*identity_matrix(CBF, 3))*vec
        ([+/- 5.35e-9] + [+/- 5.35e-9]*I, [+/- 3.54e-9] + [+/- 3.54e-9]*I, [+/- 6.79e-9] + [+/- 6.79e-9]*I)
        sage: T = matrix(GenEigSpaces[0]['basis'] + GenEigSpaces[1]['basis']).transpose()
        sage: pol = GenEigSpaces[0]['projection']
        sage: P = ~T * pol(mat) * T; P
        [[1.000 +/- 1.41e-5] + [+/- 1.41e-5]*I       [+/- 2.79e-6] + [+/- 2.79e-6]*I       [+/- 4.29e-6] + [+/- 4.29e-6]*I]
        [      [+/- 3.23e-5] + [+/- 3.23e-5]*I [1.000 +/- 3.04e-5] + [+/- 3.04e-5]*I       [+/- 3.48e-5] + [+/- 3.48e-5]*I]
        [      [+/- 4.01e-5] + [+/- 4.01e-5]*I       [+/- 2.39e-5] + [+/- 2.39e-5]*I       [+/- 3.44e-5] + [+/- 3.44e-5]*I]
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


    EXAMPLES::
        sage: from ore_algebra.analytic.linear_algebra import invariant_subspace
        sage: mat1 = matrix(CBF, [[1, 1], [1, 1]])
        sage: mat2 = matrix(CBF, [[3, -1], [0 , 2]])
        sage: vec = invariant_subspace([mat1, mat2])[0]
        sage: 1 in vec[0] and 1 in vec[1]
        True
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


################################################################################
### Useful functions ###########################################################
################################################################################

def _clean(pol):

    l = list(pol)
    while len(l)>0 and l[-1].contains_zero(): l.pop()
    cpol = pol.parent()(l)

    return cpol

def _derivatives(f, m):

    result = [f]
    for k in range(m):
        f = f.derivative()
        result.append(f)

    return result

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
    Return the centralizer of the algebra generated by ``alg``.
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
