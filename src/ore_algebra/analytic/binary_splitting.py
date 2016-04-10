# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of convergent D-finite series by binary splitting
"""

import copy
import logging
import pprint

import sage.categories.pushout
import sage.structure.coerce_exceptions

import ore_algebra

from sage.matrix.constructor import matrix
from sage.matrix.matrix_space import MatrixSpace
from sage.arith.all import lcm
from sage.rings.complex_arb import ComplexBallField
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.number_field.number_field import NumberField, is_NumberField
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.real_arb import RealBallField

from ore_algebra.analytic import bounds, utilities

logger = logging.getLogger(__name__)

class StepMatrix(object):
    """
    A structured matrix that maps a vector of s coefficients and a partial sum
    (both around some truncation index n) of a D-finite series to a similar
    vector corresponding to the partial sum truncated at order n + p for some p.
    The partial sum (but not the coefficients) typically depend on a
    perturbation parameter δ, making it possible to compute several derivatives
    of the series at once.
    """

    # NOTES:
    # - store the indices the StepMatrix covers?

    def __init__(self, rec_mat, rec_den, pow_num, pow_den, sums_row, ord):
        self.rec_mat = rec_mat
        self.rec_den = rec_den
        self.pow_num = pow_num
        self.pow_den = ZZ(pow_den)
        self.sums_row = sums_row
        self.ord = ord
        self.BigScalars = sums_row.base_ring()
        self.Mat_big_scalars = rec_mat.parent().change_ring(self.BigScalars)

    def imulleft(low, high): # pylint: disable=no-self-argument
        # TODO: Still very slow.
        # - use polynomials (mul_trunc and friends) rather than power series?
        # - rewrite everything using lower-level operations...
        # - consider special-casing ℚ[i]
        mat = high.Mat_big_scalars.matrix(
                [high.BigScalars(x, check=False)
                 for x in low.rec_mat.list()],
                coerce=False)
        low.rec_mat = high.rec_mat*low.rec_mat       # Mat(rec_Ints)
        tmp = high.sums_row*mat                      # Vec(sums_Ints[[δ]]/<δ^k>)
        for i in xrange(mat.nrows()):
            tmp[0,i] = tmp[0,i]._mul_trunc_(low.pow_num, low.ord)
        assert tmp[0][0].degree() < low.ord
        low.sums_row = tmp + high.rec_den * high.pow_den * low.sums_row
        assert low.sums_row[0][0].degree() < low.ord
        # TODO: try caching the powers of (pow_num/pow_den)? this will probably
        # not change anything for algebraic evaluation points, but it might
        # make a difference when the evaluation point is more complicated
        low.pow_num = low.pow_num._mul_trunc_(high.pow_num, low.ord)
                                                           # pow_Ints[[δ]]/<δ^k>
        low.pow_den *= high.pow_den                  # ZZ
        low.rec_den *= high.rec_den                  # rec_Ints
        return low

    def copy(self):
        new = copy.copy(self)
        new.rec_mat = copy.copy(self.rec_mat)
        new.sums_row = copy.copy(self.sums_row)
        return new

    def __mul__(high, low): # pylint: disable=no-self-argument
        return low.copy().imulleft(high)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    # The following methods don't really make sense for steps that don't start
    # at index 0, and should perhaps become module-level functions. On the other
    # hand, they are useful for external code that doesn't want to know about
    # the details of the StepMatrix data structure. Let's see how things evolve
    # with the implementation of logarithmic series.

    def term(self, parent, orddeq, j):
        r"""
        Given a StepMatrix representing a product B(n-1)···B(0) where B is the
        recurrence matrix associated to some differential operator P, return the
        term of index n of the fundamental solution of P of the form
        y[j](z) = z^j + O(z^r), 0 <= j < r = order(P).

        XXX: move this to a class that knows about orddeq?
        """
        orddiff = self.rec_mat.nrows() - orddeq
        # XXX: orddiff, orddiff+j ?
        num = parent(self.rec_mat[orddiff + j, orddiff])*parent(self.pow_num[0])
        den = parent(self.rec_den)*parent(self.pow_den)
        return num/den

    def partial_sums_num_den(self, rows, cols):
        """
        Return a matrix of partial sums of the series and its derivatives, as an
        unevaluated quotient of a matrix of algebraic integers and an integer.

        Not tested with the current code.
        """
        numer_as_series = self.sums_row[0, -cols:]
        C = numer_as_series.base_ring().base_ring()
        numer = matrix(C, rows, cols, lambda i, j: numer_as_series[0,j][i])
        rec_den_inv_num, rec_den_inv_den = _invert_order_element(self.rec_den)
        numer *= rec_den_inv_num
        denom = rec_den_inv_den * self.pow_den
        return (numer, denom)

    def partial_sums(self, ring, rows, cols):
        """
        Return a matrix of partial sums of the series and its derivatives.
        """
        numer_as_series = self.sums_row[0, -cols:]
        numer = matrix(ring, rows, cols, lambda i, j: numer_as_series[0,j][i])
        denom = ring(self.rec_den) * ring(self.pow_den)
        return numer/denom

class MatrixRec(object):
    """
    A matrix recurrence simultaneously generating the coefficients and partial
    sums of solutions of an ODE, and possibly derivatives of this solution.

    Note: Mathematically, the recurrence matrix has the structure of a
    StepMatrix (depending on parameters). However, this class does not
    derive from StepMatrix as the data structure is different.
    """

    def __init__(self, diffop, dz, derivatives):

        # Matrix corresponding to the recurrence on the series coefficients

        deq_Scalars = diffop.base_ring().base_ring()
        if deq_Scalars.characteristic() != 0:
            raise ValueError("only makes sense for scalar rings of "
                             "characteristic 0")
        ### Work around number field uniqueness problems (→ slow coercions)
        if dz.parent() == deq_Scalars:
            dz = deq_Scalars(dz)
        ###
        assert deq_Scalars is dz.parent() or deq_Scalars != dz.parent()

        # Reduce to the case of a number field generated by an algebraic
        # integer. This is mainly intended to avoid computing gcds (due to
        # denominators in the representation of number field elements) in the
        # product tree, but could also be useful to extend the field using Pari
        # in the future.
        NF_rec, AlgInts_rec = _number_field_with_integer_gen(deq_Scalars)
        Rops = ore_algebra.OreAlgebra(PolynomialRing(deq_Scalars, 'n'), 'Sn')
        recop = diffop.to_S(Rops).primitive_part().numerator()
        # ore_algebra currently does not support orders as scalar rings
        recop = recop.change_ring(recop.base_ring().change_ring(NF_rec))
        recop *= lcm([p.denominator() for p in recop.coefficients()])
        self.recop = recop

        self.orddeq = diffop.order()
        self.ordrec = recop.order()
        self.orddiff = self.ordrec - self.orddeq
        self.derivatives = derivatives

        self.zvar = diffop.base_ring().variable_name()

        self.rec_matrix_ring = MatrixSpace(AlgInts_rec, self.ordrec, self.ordrec)
        Pols_rec, n = PolynomialRing(AlgInts_rec, 'n').objgen()
        self.rec_coeffs = [-Pols_rec(recop[i])(n - self.orddiff)
                                                   for i in xrange(self.ordrec)]
        self.rec_den = Pols_rec(recop.leading_coefficient())(n - self.orddiff)
        # Guard against various problems related to number field embeddings and
        # uniqueness
        assert Pols_rec.base_ring() is AlgInts_rec
        assert self.rec_den.base_ring() is AlgInts_rec
        assert self.rec_den(self.rec_den.base_ring().zero()).parent() is AlgInts_rec

        # Also store a version of the recurrence operator of the form
        # b[0](n) + b[1](n) S^(-1) + ··· + b[s](n) S^(-s).
        # This is convenient to share code with other implementations, or at
        # least make the implementations easier to compare.
        # XXX: understand what to do about variable names!
        self.bwrec = [recop[self.ordrec-k](diffop.base_ring().gen()-self.ordrec)
                      for k in xrange(self.ordrec+1)]

        # Power of dz. Note that this part does not depend on n.

        E = dz.parent()
        if is_NumberField(E):
            # In fact we should probably do something similar for dz in any
            # finite-dimensional Q-algebra. (But how?)
            NF_pow, AlgInts_pow = _number_field_with_integer_gen(E)
            pow_den = NF_pow(dz).denominator()
        else:
            # This includes the case E = ZZ, but dz could live in pretty much
            # any algebra over deq_Scalars (including matrices, intervals...).
            # Then the computation of sums_row may take time, but we still hope
            # to gain something on the computation of the coefficients thanks
            # to the use of binary splitting.
            AlgInts_pow = E
            pow_den = 1
        assert AlgInts_pow is AlgInts_rec or AlgInts_pow != AlgInts_rec

        # If we extend the removal of denominators above to algebras other than
        # number fields, it would probably make more sense to move this into
        # the caller. --> support dz in non-com ring (mat)? power series work
        # only over com rings
        Series_pow = PolynomialRing(AlgInts_pow, 'delta')
        pow_num = Series_pow([pow_den*dz, pow_den])

        # Partial sums

        # We need a parent containing both the coefficients of the operator and
        # the evaluation point.
        # XXX: Is this the correct way to get one? Should we use
        # canonical_coercion()? Something else?
        # XXX: This is not powerful enough to find a number field containing
        # two given number fields (both given with embeddings into CC)

        # Work around #14982 (fixed) + weaknesses of the coercion framework for orders
        #Series_sums = sage.categories.pushout.pushout(AlgInts_rec, Series_pow)
        try:
            AlgInts_sums = sage.categories.pushout.pushout(AlgInts_rec, AlgInts_pow)
        except sage.structure.coerce_exceptions.CoercionException:
            AlgInts_sums = sage.categories.pushout.pushout(NF_rec, AlgInts_pow)
        assert AlgInts_sums is AlgInts_rec or AlgInts_sums != AlgInts_rec
        assert AlgInts_sums is AlgInts_pow or AlgInts_sums != AlgInts_pow

        Series_sums = PolynomialRing(AlgInts_sums, 'delta')
        assert Series_sums.base_ring() is AlgInts_sums
        # for speed
        self.Series_sums = Series_sums
        self.series_class_sums = type(Series_sums.gen())

        Mat_rec = MatrixSpace(AlgInts_rec, self.ordrec, self.ordrec)
        self._eval_template = StepMatrix(
                rec_mat=Mat_rec.zero_matrix(),
                rec_den=None, # 1
                pow_num=pow_num,
                pow_den=ZZ(pow_den),
                sums_row=MatrixSpace(Series_sums, 1, self.ordrec).zero_matrix(),
                ord=derivatives)
        assert self._eval_template.rec_mat.base_ring() is AlgInts_rec

    def __call__(self, n):
        stepmat = self._eval_template.copy()
        stepmat.rec_den = self.rec_den(n)
        for i in xrange(self.ordrec-1):
            stepmat.rec_mat[i, i+1] = stepmat.rec_den
        for i in xrange(self.ordrec):
            stepmat.rec_mat[self.ordrec-1, i] = self.rec_coeffs[i](n)
        # TODO: fix redundancy--the rec_den*pow_den probabably doesn't belong
        # here
        # XXX: should we give a truncation order?
        den = stepmat.rec_den * stepmat.pow_den
        den = self.series_class_sums(self.Series_sums, den)
        stepmat.sums_row[0, self.orddiff] = den
        #R = stepmat.sums_row.base_ring().base_ring()
        #assert den.parent() is R or den.parent() != R
        return stepmat

    def one(self):
        id = self._eval_template.rec_mat.parent().identity_matrix()
        zero_row = self._eval_template.sums_row.parent().zero_matrix()
        one_num = self._eval_template.pow_num.parent()(1)
        return StepMatrix(id, ZZ(1), one_num, ZZ(1), zero_row, self.derivatives)

    def binsplit(self, low, high, threshold=64):
        if high - low <= threshold:
            mat = self.one()
            for n in xrange(low, high):
                mat.imulleft(self(n))
        else:
            mid = (low + high) // 2
            mat = self.binsplit(low, mid)
            mat.imulleft(self.binsplit(mid, high))
        return mat

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def residual(self, prod, n, j):
        r"""
        Compute the residual associated with the fundamental solution of index j.
        """
        r, s = self.orddeq, self.ordrec
        IC = bounds.IC
        # Compute the "missing" coefficients u(n-s), ..., u(n-s'-1) s'=s-r):
        # indeed, it is convenient to compute the residuals starting from
        # u(n-s), ..., u(n-1), while our recurrence matrices produce the partial
        # sum of index n along with the vector [u(n-s'), ..., u(n+r-1)].
        last = [IC.zero()]*r  # u(n-s), ..., u(n-s'-1)
        last.extend([IC(c)/IC(prod.rec_den)             # u(n-s'), ..., u(n+r-1)
                     for c in prod.rec_mat.column(s-r+j)])  # XXX: check column index
        rop = self.recop
        for i in xrange(r-1, -1, -1): # compute u(n-s+i)
            last[i] = ~(rop[0](n-s+i))*sum(rop[k](n-s+i)*last[i+k]    # u(n-s+i)
                                           for k in xrange(1, s+1))
        # Now compute the residual. WARNING: this residual must correspond to
        # the operator stored in maj.dop, which typically isn't self.diffop (but
        # an operator in θx equal to x^k·self.diffop for some k).
        return bounds.residual(self.bwrec, n, list(reversed(last[:s])),
                self.zvar)

    def residuals(self, prod, n):
        return [self.residual(prod, n, j) for j in xrange(self.orddeq)]

def binsplit_step_seq(first_step=64):
    "first_step >= 1"
    low, high = 0, first_step
    while True:
        yield (low, high)
        low, high = high, 2*high

def fundamental_matrix_ordinary(dop, pt, eps, rows, maj):
    r"""
    INPUT:

    - ``eps`` -- a bound on the tail (does not take into account roundoff errors
      such as that committed when converting the result to intervals)
    """
    logger.info("target error = %s", eps)
    rec = MatrixRec(dop, pt, rows)
    prod = rec.one()
    tail_bound = n = None
    done = False
    for last, n in binsplit_step_seq():
        prod = rec.binsplit(last, n) * prod
        logger.debug("n = %d, tail bound = %s", n, tail_bound)
        est = prod.term(bounds.IC, dop.order(), 0).abs()
        if est < eps: # use bounds.AbsoluteError???
            majeqrhs = maj.maj_eq_rhs(rec.residuals(prod, n))
            for i in xrange(5):
                tail_bound = maj.matrix_sol_tail_bound(n, bounds.IC(pt).abs(),
                                                             majeqrhs, ord=rows)
                if tail_bound < eps: # XXX: clarify stopping criterion
                    done = True
                    break
                # note that we may get a majorant that has already been refined
                if n > 64*2**maj._effort:
                    maj.refine()
            if done: break
    is_real = RealBallField(2).has_coerce_map_from(pt.parent())
    Intervals = utilities.ball_field(eps, is_real)
    mat = prod.partial_sums(Intervals, rows, dop.order())
    # Account for the dropped high-order terms in the intervals we return.
    err = tail_bound.abs()
    mat = mat.apply_map(lambda x: x.add_error(err)) # XXX - overest
    err = bounds.IR(max(x.rad() for row in mat for x in row))
    logger.info("summed %d terms, tail <= %s, coeffwise error <= %s", n,
            tail_bound, err)
    return mat


################################################################################
# Number fields and orders
################################################################################

def _number_field_with_integer_gen(K):
    if K is QQ:
        return QQ, ZZ
    den = K.defining_polynomial().denominator()
    if den.is_one():
        # Ensure that we return the same number field object (coercions can be
        # slow!)
        intNF = K
    else:
        intgen = K.gen() * den
        ### Attempt to work around various problems with embeddings
        emb = K.coerce_embedding()
        embgen = emb(intgen) if emb else intgen
        intNF = NumberField(intgen.minpoly(), str(K.gen) + str(den),
                            embedding=embgen)
        assert intNF != K
    # Work around weaknesses in coercions involving order elements,
    # including #14982 (fixed). Used to trigger #14989 (fixed).
    #return intNF, intNF.order(intNF.gen())
    return intNF, intNF

def _invert_order_element(alg):
    if alg in ZZ:
        return 1, alg
    else:
        Order = alg.parent()
        pol = alg.polynomial().change_ring(ZZ)
        modulus = Order.gen(1).minpoly()
        den, num, _ = pol.xgcd(modulus)  # hopefully fraction-free!
        return Order(num), ZZ(den)

