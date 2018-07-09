# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of convergent D-finite series by binary splitting

TESTS::

    sage: from ore_algebra import DifferentialOperators
    sage: Dops, x, Dx = DifferentialOperators(QQ, 'x')

    sage: import logging
    sage: logging.basicConfig()
    sage: logger = logging.getLogger('ore_algebra.analytic.binary_splitting')
    sage: logger.setLevel(logging.INFO)

    sage: ((x^2 + 1)*Dx^2 + 2*x*Dx).numerical_solution([0, 1],
    ....:         [0, i+1, 2*i, i-1, 0], algorithm="binsplit")
    INFO:ore_algebra.analytic.binary_splitting:...
    [3.14159265358979...] + [+/- ...]*I

    sage: from ore_algebra.analytic.examples.misc import koutschan1
    sage: koutschan1.dop.numerical_solution(koutschan1.ini, [0, 84], algorithm="binsplit")
    INFO:ore_algebra.analytic.binary_splitting:...
    [0.011501537469552017...]

    sage: ((x + 1)*Dx^2 + Dx).numerical_transition_matrix([0,1/2], algorithm='binsplit')
    INFO:ore_algebra.analytic.binary_splitting:...
    [ [1.00000000000000...] [0.4054651081081643...]]
    [             [+/- ...] [0.6666666666666666...]]

    sage: ((x + 1)*Dx^3 + Dx).numerical_transition_matrix([0,1/2], algorithm='binsplit')
    INFO:ore_algebra.analytic.binary_splitting:...
    [  [1.000000000000000...]  [0.4815453970799961...]  [0.2456596136789682...]]
    [               [+/- ...]  [0.8936357901691244...]  [0.9667328760004665...]]
    [               [+/- ...] [-0.1959698689702905...]  [0.9070244207738327...]]

    sage: ((x + 1)*Dx^3 + Dx^2).numerical_transition_matrix([0,1/2], algorithm='binsplit')
    INFO:ore_algebra.analytic.binary_splitting:...
    [ [1.000000000000000...] [0.5000000000000000...] [0.2163953243244931...]]
    [                      0  [1.000000000000000...] [0.8109302162163287...]]
    [                      0                       0 [0.6666666666666666...]]

    sage: (Dx - 1).numerical_solution([1], [0, i + pi], algorithm="binsplit")
    INFO:ore_algebra.analytic.binary_splitting:...
    [12.5029695888765...] + [19.4722214188416...]*I

    sage: from ore_algebra.examples import fcc
    sage: fcc.dop5.numerical_solution( # long time (6.7 s)
    ....:          [0, 0, 0, 0, 1, 0], [0, 1/5+i/2, 1],
    ....:          1e-60, algorithm='binsplit')
    INFO:ore_algebra.analytic.binary_splitting:...
    [1.04885235135491485162956376369999275945402550465206640313845...] + [+/- ...]*I

    sage: logger.setLevel(logging.WARNING)

    sage: from ore_algebra.analytic.binary_splitting import MatrixRec

    sage: rec = MatrixRec((x-2)*Dx^3 - 1, RBF(1/3), 3, 10)
    sage: rec.partial_sums(rec.binsplit(0, 10), RBF, 3)
    [ [0.996...]  [0.333...] [0.111...]]
    [[-0.029...]  [0.996...] [0.666...]]
    [[-0.091...] [-0.015...] [0.996...]]

    sage: QQi.<i> = QuadraticField(-1)
    sage: rec = MatrixRec((x-i)*Dx^3 - 1, RBF(1/3), 3, 10)
    sage: rec.partial_sums(rec.binsplit(0, 10), CBF, 3)
    [ [1.0005...] + [0.0061...]*I [0.3333...] + [0.0005...]*I [0.1111...] + [6.6513...]*I]
    [[0.0059...]  + [0.0545...]*I [1.0009...] + [0.0059...]*I [0.6668...] + [0.0009...]*I]
    [ [0.0261...] + [0.1609...]*I [0.0057...] + [0.0263...]*I [1.0014...] + [0.0057...]*I]
"""

import copy
import logging
import pprint

import sage.categories.pushout
import sage.structure.coerce_exceptions
import sage.rings.polynomial.polynomial_element as polyelt
import sage.rings.polynomial.polynomial_ring as polyring
import sage.rings.polynomial.polynomial_ring_constructor as polyringconstr

from sage.matrix.constructor import matrix
from sage.matrix.matrix_space import MatrixSpace
from sage.arith.all import lcm
from sage.rings.all import ZZ, QQ, RLF, CLF, RealBallField, ComplexBallField
from sage.rings.number_field.number_field import is_NumberField
from sage.rings.power_series_ring import PowerSeriesRing

from .. import ore_algebra
from . import accuracy, bounds, utilities

from .local_solutions import bw_shift_rec
from .safe_cmp import *

logger = logging.getLogger(__name__)

def PolynomialRing(base, var):
    if is_NumberField(base) and base is not QQ:
        return polyring.PolynomialRing_field(base, var,
                element_class=polyelt.Polynomial_generic_dense)
    else:
        return polyringconstr.PolynomialRing(base, var)

class StepMatrix(object):
    r"""
    A structured matrix that maps a vector of s coefficients and a partial sum
    (both around some truncation index n) of a D-finite series to a similar
    vector corresponding to the partial sum truncated at order n + p for some p.
    The partial sum (but not the coefficients) typically depends on a
    perturbation parameter δ, making it possible to compute several derivatives
    of the series at once.
    """

    # No __init__ for speed reasons. See MatrixRec. Fields:
    #     self.rec_mat
    #     self.rec_den
    #     self.pow_num
    #     self.pow_den
    #     self.sums_row
    #     self.ord
    #     self.BigScalars, typically should be sums_row.base_ring()
    #     self.Mat_big_scalars, typically should be
    #                         self.rec_mat.parent().change_ring(self.BigScalars)

    def copy(self):
        new = copy.copy(self)
        new.rec_mat = copy.copy(self.rec_mat)
        new.sums_row = copy.copy(self.sums_row)
        return new

    def __mul__(high, low): # pylint: disable=no-self-argument
        return low.copy().imulleft(high)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

class StepMatrix_generic(StepMatrix):

    binsplit_threshold = 64

    def imulleft(low, high): # pylint: disable=no-self-argument
        # TODO: Still very slow.
        # - rewrite everything using lower-level operations...
        # - consider special-casing ℚ[i]
        assert high.idx_start == low.idx_end
        mat = low.rec_mat.list()
        mat = [low.BigScalars.Element(low.BigScalars, [x], check=False) for x in mat]
        mat = type(high.sums_row)(low.Mat_big_scalars, mat, copy=False, coerce=False)
        # A more general but complicated solution would be to create a
        # MatrixMatrixAction once and for all.
        low.rec_mat = high.rec_mat._multiply_classical(low.rec_mat) # Mat(rec_Ints)
        tmp = high.sums_row._multiply_classical(mat) # Vec(sums_Ints[[δ]]/<δ^k>)
        for i in xrange(mat.nrows()):
            tmp[0,i] = tmp[0,i]._mul_trunc_(low.BigScalars(low.pow_num), low.ord)
        #assert tmp[0][0].degree() < low.ord
        tmp2 = high.rec_den * high.pow_den
        tmp3 = low.sums_row._lmul_(low.BigScalars(tmp2))
        low.sums_row = tmp._add_(tmp3)
        #assert low.sums_row[0][0].degree() < low.ord
        # TODO: try caching the powers of (pow_num/pow_den)? this will probably
        # not change anything for algebraic evaluation points, but it might
        # make a difference when the evaluation point is more complicated
        low.pow_num = low.pow_num._mul_trunc_(high.pow_num, low.ord)
                                                           # pow_Ints[[δ]]/<δ^k>
        low.pow_den *= high.pow_den                  # ZZ
        low.rec_den *= high.rec_den                  # rec_Ints
        low.idx_end = high.idx_end
        return low

class StepMatrix_arb(StepMatrix):

    binsplit_threshold = 8

    def imulleft(low, high):
        assert high.idx_start == low.idx_end
        ordrec = low.rec_mat.nrows()
        # prod.sums_row = high.sums_row * low.rec_mat * low.pow_num + O(δ^ord)
        #               + high.rec_den * high.pow_den * low.sums_row
        tmp = [high.sums_row[0,j]._mul_trunc_(low.pow_num, low.ord)
               for j in range(high.sums_row.ncols())]
        low.sums_row = low.sums_row._lmul_(high.rec_den * high.pow_den)
        # XXX try converting tmp to an ord×ordrec matrix instead?
        for j in xrange(ordrec):
            for i in xrange(ordrec):
                low.sums_row[0,j] += low.rec_mat[i,j]*tmp[i]
        # With moderately large matrices, all the time goes in the following
        # line, even with native arb matrices.
        low.rec_mat = high.rec_mat*low.rec_mat # Mat(XBF)
        low.pow_num = low.pow_num._mul_trunc_(high.pow_num, low.ord) # XBF[δ]
        low.pow_den = low.pow_den._mul_(high.pow_den) # XBF (ZZ)
        low.rec_den = low.rec_den._mul_(high.rec_den) # XBF
        low.idx_end = high.idx_end
        return low

class MatrixRec(object):
    r"""
    A matrix recurrence simultaneously generating the coefficients and partial
    sums of solutions of an ODE, and possibly derivatives of this solution.

    Note: Mathematically, the recurrence matrix has the structure of a
    StepMatrix (depending on parameters). However, this class does not
    derive from StepMatrix as the data structure is different.
    """

    def __init__(self, diffop, dz, derivatives, nterms_est):

        #### Recurrence operator & matrix

        bwrec = bw_shift_rec(diffop, clear_denominators=True)

        # It can happen that the recop (after normalization to Sn-valuation 0)
        # has order < orddeq, and our strategy of using vectors of coefficients
        # of the form [u(n-s'), ..., u(n+r-1)] with s'=s-r does not work in this
        # case. In this case, we artificially extend the recurrence matrix to
        # something of the form
        #
        #    [ 0 1         ]
        #    [     1       ]
        #    [       …     ]  ∈ K[n]^(ordmat×ordmat)
        #    [         1   ]    (K = Frac(AlgInts_rec))
        #    [           1 ]
        #    [ 0 … 0 * … * ]  <-- ordrec nonzeros
        #
        # (This may change with the extension to regular singular points.)

        self.ordrec = bwrec.order
        self.orddeq = diffop.order()
        ordmat = max(self.ordrec, self.orddeq)
        self.orddelta = ordmat - self.orddeq

        #### Choose computation domains

        deq_Scalars = diffop.base_ring().base_ring()
        E = dz.parent()
        assert deq_Scalars is E or deq_Scalars != E
        # Set self.AlgInts_{rec,pow,sums} and self.pow_den
        if _can_use_CBF(E) and _can_use_CBF(deq_Scalars):
            # Work with arb balls and matrices, when possible with entries in ZZ
            # or ZZ[i]. Round the entries that are larger than the target
            # precision (+ some guard digits) in the upper levels of the tree.
            # Choice of working precision TBI.
            prec = 8 + nterms_est*(1 + ZZ(ZZ(ordmat).nbits()).nbits())
            self._init_CBF(deq_Scalars, E, dz, prec)
        else:
            self._init_generic(deq_Scalars, E, dz)

        self.bwrec = bwrec.change_base(self.AlgInts_rec)
        self.Mat_rec = MatrixSpace(self.AlgInts_rec, ordmat, ordmat)

        assert self.bwrec[0].base_ring() is self.AlgInts_rec # uniqueness
        assert self.bwrec[0](0).parent() is self.AlgInts_rec #   issues...

        #### Power of dz. Note that this part does not depend on n.

        Series_pow = PolynomialRing(self.AlgInts_pow, 'delta')
        self.pow_num = Series_pow([self.pow_den*dz, self.pow_den])
        self.derivatives = derivatives

        #### Partial sums

        self.Series_sums = PolynomialRing(self.AlgInts_sums, 'delta')
        self.series_class_sums = type(self.Series_sums.gen())
        assert self.Series_sums.base_ring() is self.AlgInts_sums

        self.Mat_sums_row = MatrixSpace(self.Series_sums, 1, ordmat)
        self.Mat_series_sums = self.Mat_rec.change_ring(self.Series_sums)

    def _init_CBF(self, deq_Scalars, E, dz, prec):
        self.StepMatrix_class = StepMatrix_arb
        if _can_use_RBF(E) and _can_use_RBF(deq_Scalars):
            dom = RealBallField(prec)
        else:
            dom = ComplexBallField(prec)
        if is_NumberField(E):
            self.pow_den = dom(dz.denominator())
        else:
            self.pow_den = dom.one()
        self.AlgInts_rec = self.AlgInts_pow = self.AlgInts_sums = dom

    def _init_generic(self, deq_Scalars, E, dz):
        self.StepMatrix_class = StepMatrix_generic
        # Reduce to the case of a number field generated by an algebraic
        # integer. This is mainly intended to avoid computing gcds (due to
        # denominators in the representation of number field elements) in
        # the product tree, but could also be useful to extend the field
        # using Pari in the future.
        NF_rec, AlgInts_rec = utilities.number_field_with_integer_gen(deq_Scalars)
        if is_NumberField(E):
            # In fact we should probably do something similar for dz in any
            # finite-dimensional Q-algebra. (But how?)
            NF_pow, AlgInts_pow = utilities.number_field_with_integer_gen(E)
            pow_den = NF_pow(dz).denominator()
        else:
            # This includes the case E = ZZ, but dz could live in pretty
            # much any algebra over deq_Scalars (including matrices,
            # intervals...). Then the computation of sums_row may take time,
            # but we still hope to gain something on the computation of the
            # coefficients and/or limit interval blow-up thanks to the use
            # of binary splitting.
            AlgInts_pow = E
            pow_den = ZZ.one()
        assert pow_den.parent() is ZZ

        # We need a parent containing both the coefficients of the operator and
        # the evaluation point.

        # Work around #14982 (fixed) + weaknesses of the coercion framework for orders
        #Series_sums = sage.categories.pushout.pushout(AlgInts_rec, Series_pow)
        try:
            AlgInts_sums = sage.categories.pushout.pushout(AlgInts_rec, AlgInts_pow)
        except sage.structure.coerce_exceptions.CoercionException:
            AlgInts_sums = sage.categories.pushout.pushout(NF_rec, AlgInts_pow)

        # Guard against various problems related to number field embeddings and
        # uniqueness
        assert AlgInts_pow is AlgInts_rec or AlgInts_pow != AlgInts_rec
        assert AlgInts_sums is AlgInts_rec or AlgInts_sums != AlgInts_rec
        assert AlgInts_sums is AlgInts_pow or AlgInts_sums != AlgInts_pow

        self.pow_den = pow_den
        self.AlgInts_rec = AlgInts_rec
        self.AlgInts_pow = AlgInts_pow
        self.AlgInts_sums = AlgInts_sums

    def __call__(self, n):
        stepmat = self.StepMatrix_class()
        stepmat.idx_start = n
        stepmat.idx_end = n + 1
        stepmat.rec_den = self.bwrec[0](n + self.orddeq)
        stepmat.rec_mat = self.Mat_rec.matrix()
        for i in xrange(self.Mat_rec.nrows() - 1):
            stepmat.rec_mat[i, i+1] = stepmat.rec_den
        for i in xrange(self.ordrec):
            stepmat.rec_mat[-1, -1-i] = -self.bwrec[i+1](n + self.orddeq)
        stepmat.pow_num = self.pow_num
        stepmat.pow_den = self.pow_den
        # TODO: fix redundancy--the rec_den*pow_den probabably doesn't belong
        # here
        # XXX: should we give a truncation order?
        den = stepmat.rec_den * stepmat.pow_den
        den = self.series_class_sums(self.Series_sums, [den])
        stepmat.sums_row = self.Mat_sums_row.matrix()
        stepmat.sums_row[0, self.orddelta] = den
        stepmat.ord = self.derivatives

        stepmat.BigScalars = self.Series_sums # XXX unused in arb case
        stepmat.Mat_big_scalars = self.Mat_series_sums

        return stepmat

    def one(self, n):
        stepmat = self.StepMatrix_class()
        stepmat.idx_start = stepmat.idx_end = n
        stepmat.rec_mat = self.Mat_rec.identity_matrix()
        stepmat.rec_den = self.bwrec[0].base_ring().one()
        stepmat.pow_num = self.pow_num.parent().one()
        stepmat.pow_den = self.pow_den.parent().one()
        stepmat.sums_row = self.Mat_sums_row.matrix()
        stepmat.ord = self.derivatives

        stepmat.BigScalars = self.Series_sums # XXX unused in arb case
        stepmat.Mat_big_scalars = self.Mat_series_sums

        return stepmat

    def binsplit(self, low, high):
        if high - low <= self.StepMatrix_class.binsplit_threshold:
            mat = self.one(low)
            for n in xrange(low, high):
                mat.imulleft(self(n))
        else:
            mid = (low + high) // 2
            mat = self.binsplit(low, mid)
            mat.imulleft(self.binsplit(mid, high))
        return mat

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    # XXX: needs testing
    def normalized_residual(self, maj, prod, n, j):
        r"""
        Compute the normalized residual associated with the fundamental
        solution of index j.

        TESTS::

            sage: from ore_algebra import *
            sage: DOP, t, D = DifferentialOperators()
            sage: ode = D + 1/4/(t - 1/2)
            sage: ode.numerical_transition_matrix([0,1+I,1], 1e-100, algorithm='binsplit')
            [[0.707...2078...] + [0.707...]*I]
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
        for i in xrange(r-1, -1, -1): # compute u(n-s+i)
            last[i] = ~(self.bwrec[s](n+i))*sum(self.bwrec[s-k](n+i)*last[i+k]
                                                for k in xrange(1, s+1))
        # Now compute the residual. WARNING: this residual must correspond to
        # the operator stored in maj.dop, which typically isn't self.diffop (but
        # an operator in θx equal to x^k·self.diffop for some k).
        # XXX: do not recompute this every time!
        bwrnp = [[[pol(n + i)] for pol in self.bwrec] for i in range(s)]
        altlast = [[c] for c in reversed(last[:s])]
        return maj.normalized_residual(n, altlast, bwrnp)

    def normalized_residuals(self, maj, prod, n):
        return [self.normalized_residual(maj, prod, n, j)
                for j in xrange(self.orddeq)]

    def term(self, prod, parent, j):
        r"""
        Given a matrix representing a product B(n-1)···B(0) where B is the
        recurrence matrix associated to some differential operator P, return the
        term of index n of the fundamental solution of P of the form
        y[j](z) = z^j + O(z^r), 0 <= j < r = order(P).
        """
        orddelta = self.orddelta
        num = parent(prod.rec_mat[orddelta + j, orddelta])*parent(prod.pow_num[0])
        den = parent(prod.rec_den)*parent(prod.pow_den)
        return num/den

    def partial_sums(self, prod, ring, rows):
        r"""
        Return a matrix of partial sums of the series and its derivatives.
        """
        numer = matrix(ring, rows, self.orddeq,
                       lambda i, j: prod.sums_row[0, self.orddelta+j][i])
        denom = ring(prod.rec_den)*ring(prod.pow_den)
        return numer/denom

    def error_estimate(self, prod):
        orddelta = self.orddelta
        num1 = sum(abs(prod.rec_mat[orddelta + j, orddelta])
                   for j in range(self.orddeq))
        num2 = sum(abs(a) for a in prod.pow_num)
        den = abs(prod.rec_den)*abs(prod.pow_den)
        return num1*num2/den

def binsplit_step_seq(start):
    low, high = start, start + 64
    while True:
        yield (low, high)
        low, high = high, 2*high

def fundamental_matrix_ordinary(dop, pt, eps, rows, maj, fail_fast):
    r"""
    INPUT:

    - ``eps`` -- a bound on the tail (does not take into account roundoff errors
      such as that committed when converting the result to intervals)
    """
    logger.log(logging.INFO - 1, "target error = %s", eps)
    rec = MatrixRec(dop, pt, rows, utilities.prec_from_eps(eps))
    rad = bounds.IC(pt).abs()
    prod, n, tail_bound = rec.one(0), None, bounds.IR('inf')
    # XXX clarify exact criterion
    stop = accuracy.StoppingCriterion(maj=maj, eps=eps, fast_fail=False)
    def get_residuals():
        return rec.normalized_residuals(maj, prod, n)
    def get_bound(maj):
        return maj.bound(rad, rows=rows, cols=rows)
    for last, n in binsplit_step_seq(0):
        prod = rec.binsplit(last, n) * prod
        done, tail_bound = stop.check(get_bound, get_residuals, None,
                False, n, tail_bound, rec.error_estimate(prod), next_stride=n)
        if done:
            break
    is_real = utilities.is_real_parent(pt.parent())
    Intervals = utilities.ball_field(eps, is_real)
    mat = rec.partial_sums(prod, Intervals, rows)
    # Account for the dropped high-order terms in the intervals we return.
    err = tail_bound.abs()
    mat = mat.apply_map(lambda x: x.add_error(err)) # XXX - overest
    err = bounds.IR(max(x.rad() for row in mat for x in row))
    logger.info("summed %d terms, tail <= %s, coeffwise error <= %s", n,
            tail_bound, err)
    return mat

def _can_use_CBF(E):
    return (isinstance(E, (RealBallField, ComplexBallField))
            or E is QQ or utilities.is_QQi(E)
            or E is RLF or E is CLF)

def _can_use_RBF(E):
    return isinstance(E, RealBallField) or E is QQ or E is RLF
