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
    [              [+/- ...]  [1.000000000000000...] [0.8109302162163287...]]
    [              [+/- ...]               [+/- ...] [0.6666666666666666...]]

    sage: (Dx - 1).numerical_solution([1], [0, i + pi], algorithm="binsplit")
    INFO:ore_algebra.analytic.binary_splitting:...
    [12.5029695888765...] + [19.4722214188416...]*I

    sage: from ore_algebra.examples import fcc
    sage: fcc.dop5.numerical_solution( # long time (7.2 s)
    ....:          [0, 0, 0, 0, 1, 0], [0, 1/5+i/2, 1],
    ....:          1e-60, algorithm='binsplit')
    INFO:ore_algebra.analytic.binary_splitting:...
    [1.04885235135491485162956376369999275945402550465206640313845...] + [+/- ...]*I

    sage: logger.setLevel(logging.WARNING)

    sage: from ore_algebra.analytic.binary_splitting import MatrixRec

    sage: rec = MatrixRec((x-2)*Dx^3 - 1, 0, RBF(1/3), 3, 10)
    sage: rec.partial_sums([rec.binsplit(2, 7)*rec.ordinary_ini(2,3)], RBF, 3)
    [[0.1110739597...]]
    [ [0.666100823...]]
    [ [0.996527777...]]

    sage: QQi.<i> = QuadraticField(-1)
    sage: rec = MatrixRec((x-i)*Dx^3 - 1, 0, RBF(1/3), 3, 10)
    sage: rec.partial_sums([rec.binsplit(2, 7)*rec.ordinary_ini(0,3)], CBF, 3)
    [[1.0005...] + [0.0061...]*I]
    [[0.0059...] + [0.0545...]*I]
    [  [0.02...] +  [0.160...]*I]
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
        for i in xrange(mat.ncols()):
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
               for j in xrange(ordrec)]
        low.sums_row = low.sums_row._lmul_(high.rec_den * high.pow_den)
        # XXX try converting tmp to an ord×ordrec matrix instead?
        for j in xrange(low.rec_mat.ncols()):
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

    Conventions:

        [ u(n-s+1)·z^n ]   [ 0 1     |   ]  [ u(n-s)·z^n ]
        [      ⋮       ]   [     ⋱   |   ]  [     ⋮      ]
        [              ]   [       1 | 0 ]  [            ]
        [ u(n)    ·z^n ] = [ * * ⋯ * |   ]  [ u(n-1)·z^n ]
        [ ------------ ]   [ --------+-- ]  [ ---------- ]
        [     σ(n)     ]   [ 0 ⋯ 0 1 | 1 ]  [   σ(n-1)   ]

              U(n)       =      B(n)            U(n-1)

        U(n) = B(n)···B(1)·U(0)
    """

    def __init__(self, diffop, shift, dz, derivatives, nterms_est):

        # Recurrence operator & matrix
        bwrec = bw_shift_rec(diffop, clear_denominators=True)
        self.ordrec = bwrec.order

        # Choose computation domains
        deq_Scalars = diffop.base_ring().base_ring()
        E = dz.parent()
        assert deq_Scalars is E or deq_Scalars != E
        # Set self.AlgInts_{rec,pow,sums} and self.pow_den
        if _can_use_CBF(E) and _can_use_CBF(deq_Scalars):
            # Work with arb balls and matrices, when possible with entries in ZZ
            # or ZZ[i]. Round the entries that are larger than the target
            # precision (+ some guard digits) in the upper levels of the tree.
            # Choice of working precision TBI.
            prec = 8 + nterms_est*(1 + ZZ(ZZ(self.ordrec).nbits()).nbits())
            self._init_CBF(deq_Scalars, E, dz, prec)
        else:
            self._init_generic(deq_Scalars, E, dz)

        self.bwrec = bwrec.change_base(self.AlgInts_rec)
        self.Mat_rec = MatrixSpace(self.AlgInts_rec, self.ordrec, self.ordrec)

        assert self.bwrec[0].base_ring() is self.AlgInts_rec # uniqueness
        assert self.bwrec[0](0).parent() is self.AlgInts_rec #   issues...

        # Power of dz. Note that this part does not depend on n.
        Series_pow = PolynomialRing(self.AlgInts_pow, 'delta')
        self.pow_num = Series_pow([self.pow_den*dz, self.pow_den])
        self.derivatives = derivatives

        # Partial sums
        self.Series_sums = PolynomialRing(self.AlgInts_sums, 'delta')
        self.series_class_sums = type(self.Series_sums.gen())
        assert self.Series_sums.base_ring() is self.AlgInts_sums

        self.Mat_sums_row = MatrixSpace(self.Series_sums, 1, self.ordrec)
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
        stepmat.idx_start = n - 1
        stepmat.idx_end = n
        stepmat.rec_den = self.bwrec[0](n)
        stepmat.rec_mat = self.Mat_rec.matrix()
        for i in xrange(self.Mat_rec.nrows() - 1):
            stepmat.rec_mat[i, i+1] = stepmat.rec_den
        for i in xrange(self.ordrec):
            stepmat.rec_mat[-1, -1-i] = -self.bwrec[i+1](n)
        stepmat.pow_num = self.pow_num
        stepmat.pow_den = self.pow_den
        # TODO: fix redundancy--the rec_den*pow_den probabably doesn't belong
        # here
        # XXX: should we give a truncation order?
        den = stepmat.rec_den * stepmat.pow_den
        den = self.series_class_sums(self.Series_sums, [den])
        stepmat.sums_row = self.Mat_sums_row.matrix()
        stepmat.sums_row[0, -1] = den
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

    def ordinary_ini(self, i, orddeq): # temporary
        stepmat = self.StepMatrix_class()
        stepmat.idx_start = i
        stepmat.idx_end = orddeq - 1
        stepmat.rec_mat = matrix(self.AlgInts_rec, self.ordrec, 1)
        if orddeq - i <= self.ordrec:
            stepmat.rec_mat[-orddeq+i] = 1
        stepmat.rec_den = self.bwrec[0].base_ring().one()
        stepmat.pow_num = self.pow_num.power_trunc(orddeq - 1, self.derivatives)
        stepmat.pow_den = self.pow_den**(orddeq - 1)
        stepmat.sums_row = matrix(self.Series_sums, 1, 1,
                self.pow_num.power_trunc(i, self.derivatives)
                        * self.pow_den**(orddeq-1-i) if i < orddeq - 1 else 0)
        stepmat.ord = self.derivatives

        stepmat.BigScalars = self.Series_sums # XXX unused in arb case
        stepmat.Mat_big_scalars = self.Mat_series_sums

        return stepmat

    def binsplit(self, low, high):
        r"""
        Compute R(high)·R(high-1)···R(low+1) by binary splitting.
        """
        if high - low <= self.StepMatrix_class.binsplit_threshold:
            mat = self.one(low)
            for n in xrange(low + 1, high + 1):
                mat.imulleft(self(n))
        else:
            mid = (low + high) // 2
            mat = self.binsplit(low, mid)
            mat.imulleft(self.binsplit(mid, high))
        assert mat.idx_start == low and mat.idx_end == high
        return mat

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def normalized_residual(self, maj, prod, n):
        r"""
        Compute the normalized residual associated with the fundamental
        solution.

        TESTS::

            sage: from ore_algebra import *
            sage: DOP, t, D = DifferentialOperators()
            sage: ode = D + 1/4/(t - 1/2)
            sage: ode.numerical_transition_matrix([0,1+I,1], 1e-100, algorithm='binsplit')
            [[0.707...2078...] + [0.707...]*I]
        """
        # WARNING: this residual must correspond to the operator stored in
        # maj.dop, which typically isn't self.diffop (but an operator in θx
        # equal to x^k·self.diffop for some k).
        IC = bounds.IC
        last = [[IC(c)/IC(prod.rec_den)]                 # [u(n-1), ..., u(n-s)]
                for c in reversed(prod.rec_mat.column(-1))]
        # XXX: do not recompute this every time!
        bwrnp = [[[pol(n + i)] for pol in self.bwrec]
                 for i in range(self.ordrec)]
        return maj.normalized_residual(n, last, bwrnp)

    def term(self, prod, ring, j):
        r"""
        Given a matrix representing a product B(n)···B(1) where B is the
        backward recurrence matrix (U(n) = B(n)·U(n-1)) associated to some
        differential operator P, return the term of index n of the fundamental
        solution of P of the form
        y[j](z) = z^j + O(z^r), 0 <= j < r = order(P).
        """
        num = ring(prod.rec_mat[-1, -1])*ring(prod.pow_num[0])
        den = ring(prod.rec_den)*ring(prod.pow_den)
        return num/den

    def partial_sums(self, prods, ring, rows):
        r"""
        Return a matrix of partial sums of solutions and their derivatives.

        The result has size rows × len(prods).
        """
        coeffs = [prod.sums_row[0,-1].padded_list(rows) for prod in prods]
        numer = matrix(ring, coeffs).transpose()
        assert numer.dimensions() == (rows, len(prods))
        denom = ring(prod.rec_den)*ring(prod.pow_den)
        return numer/denom

    def error_estimate(self, prod):
        num1 = abs(prod.rec_mat[-1, -1])
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
    rec = MatrixRec(dop, 0, pt, rows, utilities.prec_from_eps(eps))
    rad = bounds.IC(pt).abs()
    # Each solution is represented as a column StepMatrix to be applied to the
    # column vector [ 1 ] to get the coefficients and partial sum of the series.
    # In the general regular case, we will also need to record a Log shift to be
    # used in place of the 1 in the vector, and to update that shift when
    # crossing a singular index.
    prods, n, tail_bound = [], None, bounds.IR('inf')
    # XXX clarify exact criterion
    stop = accuracy.StoppingCriterion(maj=maj, eps=eps, fast_fail=False)
    class BoundCallbacks(accuracy.BoundCallbacks):
        def get_residuals(self):
            return [rec.normalized_residual(maj, mat, n) for mat in prods]
        def get_bound(self, residuals):
            maj = self.get_maj(stop, n, residuals)
            return maj.bound(rad, rows=rows, cols=rows)
    cb = BoundCallbacks()
    prods = [rec.ordinary_ini(i, dop.order()) for i in range(dop.order())]
    for last, n in binsplit_step_seq(dop.order() - 1):
        fwd = rec.binsplit(last, n)
        for prod in prods:
            prod.imulleft(fwd)
        done, tail_bound = stop.check(cb, False, n, tail_bound,
                                      rec.error_estimate(prod), next_stride=n)
        if done:
            break
    is_real = utilities.is_real_parent(pt.parent())
    Intervals = utilities.ball_field(eps, is_real)
    mat = rec.partial_sums(prods, Intervals, rows)
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
