# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of convergent D-finite series by direct summation
"""

# Copyright 2015, 2016, 2017, 2018, 2019 Marc Mezzarobba
# Copyright 2015, 2016, 2017, 2018, 2019 Centre national de la recherche scientifique
# Copyright 2015, 2016, 2017, 2018 Université Pierre et Marie Curie
# Copyright 2019 Sorbonne Université
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

# TODO:
# - support summing a given number of terms rather than until a target accuracy
# is reached?
# - cythonize critical parts?

from __future__ import division, print_function
from six.moves import range

import collections, logging, sys, warnings

from itertools import count, chain, repeat

from sage.matrix.constructor import identity_matrix, matrix
from sage.modules.free_module_element import vector
from sage.rings.all import ZZ, QQ, RR, QQbar, infinity
from sage.rings.complex_arb import ComplexBallField, CBF, ComplexBall
from sage.rings.integer import Integer
from sage.rings.number_field.number_field_base import is_NumberField
from sage.rings.polynomial import polynomial_element
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.real_arb import RealBallField, RBF, RealBall

from .. import ore_algebra
from . import accuracy, bounds, utilities
from .context import Context, dctx
from .differential_operator import DifferentialOperator
from .local_solutions import (bw_shift_rec, FundamentalSolution,
        LogSeriesInitialValues, LocalBasisMapper, log_series_values)
from .path import EvaluationPoint
from .safe_cmp import *
from .utilities import short_str

logger = logging.getLogger(__name__)

def cyPartialSum():
    try:
        from . import naive_sum_c
        return naive_sum_c.PartialSum
    except ImportError:
        warnings.warn("Cython implementation unavailable, "
                      "falling back to slower Python implementation")
        return PartialSum

##########################
# Argument processing etc.
##########################

def series_sum(dop, ini, pt, tgt_error, maj=None, bwrec=None, stop=None,
               fail_fast=False, effort=2, stride=None, **kwds):
    r"""
    Sum a (generalized) series solution of dop.

    This is a somewhat more user-friendly wrapper to the series summation
    routines, mainly for testing purposes. The analytic continuation code
    typically calls lower level pieces directly.

    EXAMPLES::

        sage: from sage.rings.real_arb import RealBallField, RBF
        sage: from sage.rings.complex_arb import ComplexBallField, CBF
        sage: QQi.<i> = QuadraticField(-1)

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.naive_sum import series_sum, EvaluationPoint
        sage: Dops, x, Dx = DifferentialOperators()

        sage: dop = ((4*x^2 + 3/58*x - 8)*Dx^10 + (2*x^2 - 2*x)*Dx^9 +
        ....:       (x^2 - 1)*Dx^8 + (6*x^2 - 1/2*x + 4)*Dx^7 +
        ....:       (3/2*x^2 + 2/5*x + 1)*Dx^6 + (-1/6*x^2 + x)*Dx^5 +
        ....:       (-1/5*x^2 + 2*x - 1)*Dx^4 + (8*x^2 + x)*Dx^3 +
        ....:       (-1/5*x^2 + 9/5*x + 5/2)*Dx^2 + (7/30*x - 12)*Dx +
        ....:       8/7*x^2 - x - 2)
        sage: ini = [CBF(-1/16, -2), CBF(-17/2, -1/2), CBF(-1, 1), CBF(5/2, 0),
        ....:       CBF(1, 3/29), CBF(-1/2, -2), CBF(0, 0), CBF(80, -30),
        ....:       CBF(1, -5), CBF(-1/2, 11)]

    Funny: on the following example, both the evaluation point and most of the
    initial values are exact, so that we end up with a significantly better
    approximation than requested::

        sage: series_sum(dop, ini, 1/2, RBF(1e-16))
        ([-3.575140703474456...] + [-2.2884877202396862...]*I)

        sage: import logging; logging.basicConfig()
        sage: series_sum(dop, ini, 1/2, RBF(1e-30))
        WARNING:ore_algebra.analytic.naive_sum:input intervals may be too wide
        ...
        ([-3.5751407034...] + [-2.2884877202...]*I)

    In normal usage ``pt`` should be an object coercible to a complex ball or an
    :class:`EvaluationPoint` that wraps such an object. In addition, there is
    some support for ``EvaluationPoints`` wrapping identity polynomials. Other
    cases might work by accident. ::

        sage: from ore_algebra.analytic.accuracy import AbsoluteError
        sage: series_sum(Dx - 1, [RBF(1)],
        ....:         EvaluationPoint(x, jet_order=2, rad=RBF(1)),
        ....:         AbsoluteError(1e-3), stride=1)
        (... + [0.0083...]*x^5 + [0.0416...]*x^4 + [0.1666...]*x^3
        + 0.5000...*x^2 + x + [1.000...],
        ... + [0.0083...]*x^5 + [0.0416...]*x^4 + [0.1666...]*x^3
        + [0.5000...]*x^2 + x + [1.000...])

    TESTS::

        sage: b = series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [RBF(0), RBF(1)],
        ....:                         7/10, RBF(1e-30))
        sage: b.parent()
        Vector space of dimension 1 over Real ball field with ... precision
        sage: b[0].rad().exact_rational() < 10^(-30)
        True
        sage: b[0].overlaps(RealBallField(130)(7/10).arctan())
        True

        sage: b = series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [CBF(0), CBF(1)],
        ....:                         (i+1)/2, RBF(1e-30))
        sage: b.parent()
        Vector space of dimension 1 over Complex ball field with ... precision
        sage: b[0].overlaps(ComplexBallField(130)((1+i)/2).arctan())
        True

        sage: series_sum(x*Dx^2 + Dx + x, [0], 1/2, 1e-10)
        Traceback (most recent call last):
        ...
        ValueError: invalid initial data for x*Dx^2 + Dx + x at 0

        sage: iv = RBF(RIF(-10^(-6), 10^(-6)))
        sage: series_sum(((6+x)^2 + 1)*Dx^2+2*(6+x)*Dx, [iv, iv], 4, RBF(1e-10))
        WARNING:...
        ([+/- ...])

        sage: series_sum(Dx-1, [0], 2, 1e-50, stride=1)
        (0)

    Test that automatic precision increases do something reasonable::

        sage: logger = logging.getLogger('ore_algebra.analytic.naive_sum')
        sage: logger.setLevel(logging.INFO)

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, 1/3], 5/7, 1e-16, effort=100)
        INFO:...
        ([0.20674982866094049...])

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], 5/7, 1e-16, effort=100)
        WARNING:ore_algebra.analytic.naive_sum:input intervals may be too wide compared to requested accuracy
        ...
        ([0.206749828660940...])

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], RBF(5/7), 1e-12, effort=100)
        INFO:...
        ([0.2067498286609...])

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], RBF(5/7), 1e-20, effort=100)
        WARNING:ore_algebra.analytic.naive_sum:input intervals may be too wide compared to requested accuracy
        ...
        INFO:ore_algebra.analytic.naive_sum:lost too much precision, giving up
        ([0.20674982866094...])

        sage: xx = EvaluationPoint(x, jet_order=2, rad=RBF(1/4))
        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, 1/3], xx, 1e-30)[0](1/6)
        INFO:...
        [0.05504955913820894609304276321...]

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], xx, 1e-16)[0](1/6)
        WARNING:ore_algebra.analytic.naive_sum:input intervals may be too wide compared to requested accuracy
        ...
        [0.055049559138208...]

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], xx, 1e-30)[0](1/6)
        WARNING:ore_algebra.analytic.naive_sum:input intervals may be too wide compared to requested accuracy
        ...
        INFO:ore_algebra.analytic.naive_sum:lost too much precision, giving up
        [0.055049559138208...]

        sage: logger.setLevel(logging.WARNING)
    """

    dop = DifferentialOperator(dop)
    if not isinstance(ini, LogSeriesInitialValues):
        ini = LogSeriesInitialValues(ZZ.zero(), ini, dop)
    if not isinstance(pt, EvaluationPoint):
        pt = EvaluationPoint(pt)
    if isinstance(tgt_error, accuracy.RelativeError) and pt.jet_order > 1:
        raise TypeError("relative error not supported when computing derivatives")
    if not isinstance(tgt_error, accuracy.AccuracyTest):
        tgt_error = accuracy.AbsoluteError(tgt_error)
        input_accuracy = min(pt.accuracy(), ini.accuracy())
        if input_accuracy < -tgt_error.eps.upper().log2().floor():
            logger.warning("input intervals may be too wide "
                           "compared to requested accuracy")

    if maj is None:
        special_shifts = [(s, len(v)) for s, v in ini.shift.items()]
        maj = bounds.DiffOpBound(dop, ini.expo, special_shifts)
    if bwrec is None:
        bwrec = bw_shift_rec(dop, shift=ini.expo)
    if stop is None:
        stop = accuracy.StoppingCriterion(maj, tgt_error.eps)

    ctx = Context(**kwds)

    sols = interval_series_sum_wrapper(dop, [ini], pt, tgt_error, bwrec, stop,
                                       fail_fast, effort, stride, ctx)
    assert len(sols) == 1
    sols[0].update_downshifts(pt, [0])
    return sols[0].downshifts[0]

def guard_bits(dop, maj, pt, ordrec, nterms):
    r"""
    Helper for choosing a working precision.

    This is done under the assumption that the first terms of the coefficient
    sequence are computed in interval arithmetic, and then, starting from some
    cutoff index, we switch to something like floating-point arithmetic with an
    rounding error bound computed on the side. This function returns a suggested
    cutoff index and a corresponding number of guard bits to add to the
    precision of the output.

    The computation done by this function is heuristic, but the output does not
    affect the correctness of the final result (only its sharpness and/or the
    computation time).

    The algorithm is based on what we can expect to happen at an ordinary point
    and may or may not work in the regular singular case.
    """

    new_cost = cur_cost = sys.maxsize
    new_bits = cur_bits = None
    new_n0 = cur_n0 = orddeq = dop.order()
    refine = False

    cst = abs(bounds.IC(maj.dop.leading_coefficient()[0])) # ???

    while True:

        # Roughly speaking, the computation of a new coefficient of the series
        # *multiplies* the diameter by the order of the recurrence (minus two).
        # Thus, it is not unreasonable that the loss of precision is of the
        # order of log2(ordrec^nterms). This observation is far from explaining
        # everything, though; in particular, it completely ignores the size of
        # the coefficients. Anyhow, this formula seems to work reasonaly well in
        # practice. It is perhaps a bit pessimistic for simple equations.
        guard_bits_intervals = new_n0*max(1, ZZ(ordrec - 2).nbits())

        # est_rnd_err = rough estimate of global round-off error
        # ≈ (local error for a single term) × (propagation factor)
        # ≈ (ordrec × working prec epsilon) × (value of majorant series)
        rnd_maj = maj(new_n0)
        rnd_maj >>= new_n0
        est_lg_rnd_fac = (cst*rnd_maj.bound(pt.rad, rows=orddeq)).log(2)
        est_lg_rnd_err = 2*bounds.IR(ordrec + 1).log(2)
        if not est_lg_rnd_fac < bounds.IR.zero():
            est_lg_rnd_err += est_lg_rnd_fac
        if est_lg_rnd_fac.is_finite():
            guard_bits_squashed = int(est_lg_rnd_err.ceil().upper()) + 2
        else:
            guard_bits_squashed = sys.maxsize

        # We expect the effective working precision to decrease linearly in the
        # first phase due to interval blow-up, and then stabilize around (target
        # prec + guard_bits_squashed).
        new_cost = (new_n0//2)*guard_bits_intervals + nterms*guard_bits_squashed
        new_bits = guard_bits_intervals + guard_bits_squashed

        logger.debug(
                "n0 = %s, terms = %s, guard bits = %s+%s = %s, cost = %s",
                new_n0, nterms, guard_bits_intervals, guard_bits_squashed,
                new_bits, new_cost)

        if cur_cost <= new_cost < sys.maxsize:
            return cur_n0, cur_bits

        if (refine and maj.can_refine() and
             guard_bits_squashed > guard_bits_intervals + 50):
            maj.refine()
        else:
            new_n0, cur_n0 = new_n0*2, new_n0
            cur_cost = new_cost
            cur_bits = new_bits
        refine = not refine

        if new_n0 > nterms:
            return nterms, guard_bits_intervals

def _use_inexact_recurrence(bwrec, prec):
    Scalars = bwrec.Scalars
    if not is_NumberField(Scalars):
        return False
    if ((Scalars is QQ or utilities.is_QQi(Scalars))
            and bwrec[-1][0][0].numerator().nbits() < 10*prec):
        return False
    if prec <= 4000:
        return True
    h = max(a.numerator().nbits() for p in bwrec.coeff[::3]
                                  for i in range(0, p.degree(), 10)
                                  for a in p[i])
    prefer_inexact = ( 4*(h + 16)*Scalars.degree()**2 + 4000 >= prec )
    logger.debug("using %sexact version of recurrence with algebraic coeffs "
            "of degree %s", "in" if prefer_inexact else "", Scalars.degree())
    return prefer_inexact

def interval_series_sum_wrapper(dop, inis, pt, tgt_error, bwrec, stop,
                                fail_fast, effort, stride, ctx=dctx):

    real = pt.is_real_or_symbolic() and all(ini.is_real(dop) for ini in inis)
    if pt.is_numeric and cyPartialSum() is not PartialSum:
        ivs = ComplexBallField
    elif real:
        ivs = RealBallField
    else:
        ivs = ComplexBallField
    input_accuracy = max(0, min(chain([pt.accuracy()],
                                      (ini.accuracy() for ini in inis))))
    logger.log(logging.INFO - 1, "target error = %s", tgt_error)
    if stride is None:
        stride = min(max(50, 2*bwrec.order), max(2, input_accuracy))

    ordinary = dop.leading_coefficient()[0] != 0
    bit_prec0 = utilities.prec_from_eps(tgt_error.eps)
    old_bit_prec = 8 + bit_prec0*(1 + ZZ(bwrec.order - 2).nbits())
    if ctx.squash_intervals and ordinary:
        nterms, lg_mag = dop.est_terms(pt, bit_prec0)
        nterms = (bwrec.order*dop.order() + nterms)*1.2 # let's be pragmatic
        nterms = ZZ((nterms//stride + 1)*stride)
        bit_prec0 += ZZ(dop._naive_height()).nbits() + lg_mag + nterms.nbits()
        n0_squash, g = guard_bits(dop, stop.maj, pt, bwrec.order, nterms)
        # adding twice the computed number of guard bits seems to work better
        # in practice, but I don't really understand why
        bit_prec = bit_prec0 + 2*g
        logger.info("initial working precision = %s + %s = %s (naive = %s), "
                    "squashing intervals for n >= %s",
                    bit_prec0, 2*g, bit_prec, old_bit_prec, n0_squash)
        if fail_fast and bit_prec > 4*bit_prec0 and effort <= 1:
            raise accuracy.PrecisionError
    else:
        bit_prec = old_bit_prec
        n0_squash = sys.maxsize
        logger.info("initial working precision = %s bits", bit_prec)
    max_prec = bit_prec + 2*input_accuracy

    err=None
    for attempt in count(1):
        Intervals = ivs(bit_prec)
        ini_are_accurate = 2*input_accuracy > bit_prec
        # Strictly decrease eps every time to avoid situations where doit
        # would be happy with the result and stop at the same point despite
        # the higher bit_prec. Since attempt starts at 1, we have a bit of
        # room for round-off errors.
        stop.reset(tgt_error.eps >> (4*attempt),
                   stop.fast_fail and ini_are_accurate)

        if _use_inexact_recurrence(bwrec, bit_prec):
            bwrec1 = bwrec.change_base(Intervals)
        else:
            bwrec1 = bwrec

        try:
            sols = series_sum_regular(Intervals, dop, bwrec1, inis, pt, stop,
                                      stride, n0_squash, real)
        except accuracy.PrecisionError:
            if attempt > effort:
                raise
        else:
            logger.debug("bit_prec = %s, err = %s (tgt = %s)", bit_prec,
                        max(sol.total_error for sol in sols), tgt_error)
            if all(tgt_error.reached(
                            sol.total_error,
                            abs(sol.value[0]) if pt.is_numeric else None)
                    for sol in sols):
                return sols

        # if interval squashing didn't give accurate result, switch back to the
        # classical method
        n0_squash = sys.maxsize

        bit_prec *= 2
        if attempt <= effort and bit_prec < max_prec:
            logger.info("lost too much precision, restarting with %d bits",
                        bit_prec)
            continue
        if fail_fast:
            raise accuracy.PrecisionError
        else:
            logger.info("lost too much precision, giving up")
            return sols

################################################################################
# Regular singular points
################################################################################

class HighestSolMapper(LocalBasisMapper):

    def __init__(self, dop, pt, eps, fail_fast, effort, ctx=dctx):
        super(self.__class__, self).__init__(dop)
        self.pt = pt
        self.eps = eps
        self.fail_fast = fail_fast
        self.effort = effort
        self.ctx = ctx
        self.ordinary = (dop.leading_coefficient()[0] != 0)

    def process_modZ_class(self):
        logger.info(r"solutions z^(%s+n)·log(z)^k/k! + ···, n = %s",
                    self.leftmost, ", ".join(str(s) for s, _ in self.shifts))
        maj = bounds.DiffOpBound(self.edop, self.leftmost,
                        special_shifts=(None if self.ordinary else self.shifts),
                        bound_inverse="solve",
                        pol_part_len=(4 if self.ordinary else None))
        stop = accuracy.StoppingCriterion(maj, self.eps.eps)
        # Compute the "highest" (in terms powers of log) solution of each
        # valuation
        inis = [LogSeriesInitialValues(
                    expo=self.leftmost,
                    mults=self.shifts,
                    values={(s, m-1): ZZ.one()})
                for s, m in self.shifts]
        highest_sols = interval_series_sum_wrapper(self.dop, inis, self.pt,
                self.eps, self.shifted_bwrec, stop, self.fail_fast, self.effort,
                None, self.ctx)
        self.highest_sols = {}
        for (s, m), sol in zip(self.shifts, highest_sols):
            sol.update_downshifts(self.pt, range(m))
            self.highest_sols[s] = sol
        self.sols = {}
        super(self.__class__, self).process_modZ_class()

    def fun(self, ini):
        # Non-highest solutions of a given valuation can be deduced from the
        # highest one up to correcting factors that only involve solutions
        # further to the right. We are relying on the iteration order, which
        # ensures that all other solutions involved already have been
        # computed.
        highest = self.highest_sols[self.shift]
        delta = self.mult - 1 - self.log_power
        value = highest.downshifts[delta]
        for s, m in self.shifts:
            if s > self.shift:
                for k in range(max(m - delta, 0), m):
                    cc = highest.critical_coeffs[s][k+delta]
                    value -= cc*self.sols[s,k]
        self.sols[self.shift, self.log_power] = value
        return vector(value)

def fundamental_matrix_regular(dop, pt, eps, fail_fast, effort, ctx=dctx):
    r"""
    Fundamental matrix at a possibly regular singular point

    TESTS::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.naive_sum import *
        sage: from ore_algebra.analytic.differential_operator import DifferentialOperator
        sage: from ore_algebra.analytic.path import EvaluationPoint as EP
        sage: Dops, x, Dx = DifferentialOperators()

        sage: fundamental_matrix_regular(
        ....:         DifferentialOperator(x*Dx^2 + (1-x)*Dx),
        ....:         EP(1, 2), RBF(1e-10), False, 2)
        [[1.317902...] [1.000000...]]
        [[2.718281...]     [+/- ...]]

        sage: dop = DifferentialOperator(
        ....:         (x+1)*(x^2+1)*Dx^3-(x-1)*(x^2-3)*Dx^2-2*(x^2+2*x-1)*Dx)
        sage: fundamental_matrix_regular(dop, EP(1/3, 3), RBF(1e-10), False, 2)
        [ [1.000000...]  [0.321750554...]  [0.147723741...]]
        [     [+/- ...]  [0.900000000...]  [0.991224850...]]
        [     [+/- ...]  [-0.27000000...]  [1.935612425...]]

        sage: dop = DifferentialOperator(
        ....:     (2*x^6 - x^5 - 3*x^4 - x^3 + x^2)*Dx^4
        ....:     + (-2*x^6 + 5*x^5 - 11*x^3 - 6*x^2 + 6*x)*Dx^3
        ....:     + (2*x^6 - 3*x^5 - 6*x^4 + 7*x^3 + 8*x^2 - 6*x + 6)*Dx^2
        ....:     + (-2*x^6 + 3*x^5 + 5*x^4 - 2*x^3 - 9*x^2 + 9*x)*Dx)
        sage: fundamental_matrix_regular(dop, EP(RBF(1/3), 4), RBF(1e-10), False, 2)
        [ [3.1788470...] [-1.064032...]  [1.000...] [0.3287250...]]
        [ [-8.981931...] [3.2281834...]    [+/-...] [0.9586537...]]
        [  [26.18828...] [-4.063756...]    [+/-...] [-0.123080...]]
        [ [-80.24671...]  [9.190740...]    [+/-...] [-0.119259...]]

        sage: dop = x*Dx^3 + 2*Dx^2 + x*Dx
        sage: ini = [1, CBF(euler_gamma), 0]
        sage: dop.numerical_solution(ini, [0, RBF(1/3)], 1e-14)
        [-0.549046117782...]
    """
    eps_col = bounds.IR(eps)/bounds.IR(dop.order()).sqrt()
    eps_col = accuracy.AbsoluteError(eps_col)
    unr = HighestSolMapper(dop, pt, eps_col, fail_fast, effort, ctx=dctx)
    cols = unr.run()
    return matrix([sol.value for sol in cols]).transpose()

class PartialSum(object):

    def __init__(self, Intervals, Jets, ini, ordrec, real):

        self.Intervals = Intervals
        self.Jets = Jets
        self._use_sum_of_products = hasattr(Intervals, '_sum_of_products')
        self.ini = ini
        self.ordrec = ordrec

        self.force_real = real and isinstance(Intervals, ComplexBallField)

        self.log_prec = 0
        self.trunc = 0 # first term _not_ in the sum
        # Start with vectors of length 1 instead of 0 (but still with log_prec
        # == 0) to avoid having to resize them, especially in the ordinary case
        last = [[Intervals.zero()] for _ in range(ordrec + 1)]
        self.last = collections.deque(last) # u[trunc-1], u[trunc-2], ...
        self.critical_coeffs = {}
        # ...but starting with partial sums of length 0 is better in some
        # corner cases
        self.psum = []
        self.tail_bound = bounds.IR(infinity)
        self.total_error = bounds.IR(infinity)

        self.value = None
        self.downshifts = []

    def coeff_estimate(self):
        return sum(abs(a) for log_jet in self.last for a in log_jet)

    def next_term_ordinary_initial_part(self, n, jetpow):
        r"""
        Similar to next_term(), but limited to n < orddeq at ordinary points,
        does not support squasing, and does not require evaluating the
        recurrence.
        """
        self.last.rotate(1)
        self.trunc += 1
        self.last[0][0] = self.Intervals(self.ini.shift[n][0])
        if not self.ini.shift[n][0].is_zero():
            self.log_prec = 1
            if not self.psum:
                self.psum.append(self.Jets.zero())
            self.psum[0] += jetpow._lmul_(self.last[0][0])

    def handle_singular_index(self, n, mult):

        self.critical_coeffs[n] = list(c.real() if self.force_real else c
                                       for c in self.last[0])

        nz = mult - _ctz(self.last[0], mult)
        self.log_prec += nz
        for l in self.last:
            _resize_list(l, self.log_prec, self.Intervals.zero())
        _resize_list(self.psum, self.log_prec, self.Jets.zero())

    def next_term(self, n, mult, bwrec_n, cst, jetpow, squash):

        assert n == self.trunc
        self.last.rotate(1)
        self.trunc += 1

        zero = self.Intervals.zero()

        if mult > 0:
            self.last[0] = [zero]*(self.log_prec + mult)

        for p in range(self.log_prec - 1, -1, -1):
            terms = chain(
                    ((bwrec_n[i][j], self.last[i][p+j])
                        for j in range(self.log_prec - p)
                        for i in range(self.ordrec, 0, -1)),
                    ((bwrec_n[0][j], self.last[0][p+j])
                        for j in range(mult + 1, mult + self.log_prec - p)))
            if self._use_sum_of_products:
                combin = self.Intervals._sum_of_products(terms)
            else:
                combin = sum((a*b for a, b in terms), zero)
            self.last[0][mult + p] = cst * combin

        err = None
        if mult == 0 and squash:
            err = accuracy.IR(self.last[0][0].rad())
            self.last[0][0] = self.last[0][0].squash()

        for p in range(mult - 1, -1, -1):
            self.last[0][p] = self.Intervals(self.ini.shift[n][p])

        if mult > 0:
            self.handle_singular_index(n, mult)

        if self.log_prec == mult == 0:
            return accuracy.IR.zero()

        for k in range(self.log_prec):
            self.psum[k] += jetpow._lmul_(self.last[0][k])

        return err

    def update_enclosure(self, Jets, pt, tb):
        self.series = vector(Jets, self.log_prec)
        for i, t in enumerate(self.psum):
            self.series[i] = Jets([_add_error(t[k], tb)
                                   for k in range(pt.jet_order)])
        # log_series_values() may decide to introduce complex numbers if there
        # are logs, and hence the parent of the partial sum may switch from real
        # to complex during the computation...
        [self.value] = log_series_values(Jets, self.ini.expo, self.series, pt)
        self.total_error = max(chain(iter([bounds.IR.zero()]),
                                     (_get_error(c) for c in self.value)))

    def update_downshifts(self, pt, downshift):
        r"""
        Compute the values of the partial sums of this solution and its "down
        shifts".

        The down shifts are obtained by decreasing k by one in each occurrence
        of log(z)^k/k!, and removing the terms where k < 0.

        Unlike the other variants, this function forgets the imaginary part of
        the computed partial sums if self.force_real is set.
        """
        Jets = self.series.base_ring()
        if self.force_real:
            Jets = Jets.change_ring(Jets.base().base())
            assert all(c.imag().contains_zero()
                       for jet in self.series for c in jet)
            jets = [Jets([c.real() for c in jet]) for jet in self.series]
            series = vector(Jets, self.log_prec, jets)
        else:
            series = self.series
        self.downshifts = log_series_values(Jets, self.ini.expo, series,
                pt, downshift=downshift)

    def bare_value(self, Jets, pt):
        r"""
        Value taking into account logs etc. but ignoring the truncation error.
        """
        psum = vector(Jets, self.psum)
        [v] = log_series_values(Jets, self.ini.expo, psum, pt)
        return v

    def interval_width(self):
        try:
            return max(c.rad() for c in self.value)
        except RuntimeError:
            return RealField(30)('inf')

def series_sum_regular(Intervals, dop, bwrec, inis, pt, stop, stride,
                       n0_squash, real):
    r"""
    Compute partial sums of one or several logarithmic series solution of an
    operator that may have a regular singular point at the origin.

    The solutions must be logarithmic series with exponents in a same single
    coset of ℂ/ℤ. They will be evaluated at the same point and using the same
    evaluation parameters. In other words, the only thing in which they can
    differ is the initial conditions.

    TESTS::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.naive_sum import *
        sage: Dops, x, Dx = DifferentialOperators()

    Test that we correctly compute solutions of large valuations, and that when
    there are several solutions with very different valuations, we can stop
    before reaching the largest one if the initial values there are zero.
    (Unfortunately, the bounds in this kind of situation are currently so
    pessimistic that this ability rarely helps in practice!) ::

        sage: #dop = (Dx-1).lclm(x*Dx-1000)
        sage: dop = (x^2-1000*x)*Dx^2 + (-x^2+999000)*Dx + 1000*x - 999000
        sage: logger = logging.getLogger('ore_algebra.analytic.naive_sum')
        sage: logger.setLevel(logging.INFO) # TBI
        sage: series_sum(dop, {0:(1,), 1000:(0,)}, 1/10000000, 1e-16)
        INFO:ore_algebra.analytic.naive_sum:...
        INFO:ore_algebra.analytic.naive_sum:summed 50 terms, ...
        ([1.000000100000005...])
        sage: logger.setLevel(logging.WARNING)
        sage: series_sum(dop, {0: (1,), 1000: (1/1000,)}, 1, 1e-10)
        ([2.719281828...])

    Test that we correctly take into account the errors on terms of polynomials
    that are not represented because they are zero::

        sage: dop = x*Dx^2 + Dx + x
        sage: ini = LogSeriesInitialValues(0, {0: (1, 0)})
        sage: maj = bounds.DiffOpBound(dop, special_shifts=[(0, 1)], max_effort=0)
        sage: series_sum(dop, ini, QQ(2), 1e-8, stride=1, maj=maj)
        ([0.2238907...])
    """

    if not inis:
        return []
    assert inis[0].compatible(inis)
    mult_dict = inis[0].mult_dict()

    jet = pt.jet(Intervals)
    ord = pt.jet_order
    Jets = jet.parent() # != Intervals[x] in general (symbolic points...)
    jetpow = Jets.one()
    radpow = bounds.IR.one() # bound on abs(pt)^n in the series part (=> starts
                             # at 1 regardless of ini.expo)
    tail_bound = bounds.IR(infinity)

    ordinary = (dop.leading_coefficient()[0] != 0)

    if n0_squash < sys.maxsize:
        assert ordinary
        rnd_maj = stop.maj(n0_squash)
        rnd_maj >>= n0_squash # XXX (a) useful? (b) check correctness
        rnd_den = rnd_maj.exp_part_coeffs_lbounds()
        rnd_loc = bounds.IR.zero()

    last_index_with_ini = max(chain(iter([dop.order()]),
                                    (ini.last_index() for ini in inis)))

    PS = cyPartialSum() if pt.is_numeric else PartialSum
    sols = [PS(Intervals, Jets, ini, bwrec.order, real) for ini in inis]

    class BoundCallbacks(accuracy.BoundCallbacks):
        def get_residuals(self):
            # Since this is called _before_ computing the new term, the relevant
            # coefficients are given by last[:-1], not last[1:]
            assert all(sol.trunc == n for sol in sols)
            return [stop.maj.normalized_residual(n, list(sol.last)[:-1],
                                                 bwrec_nplus)
                    for sol in sols]
        def get_bound(self, residuals):
            # XXX consider maintaining separate tail bounds, and stopping the
            # summation of some series before the others
            maj = self.get_maj(stop, n, residuals)
            tb = maj.bound(pt.rad, rows=ord)
            for sol in sols:
                sol.update_enclosure(Jets, pt, tb)
            return max(sol.total_error for sol in sols)
        def get_value(self):
            assert len(sols) == 1
            return sols[0].bare_value(Jets, pt)
    cb = BoundCallbacks()

    log_prec = 1
    precomp_len = max(1, bwrec.order) # hack for recurrences of order zero
    start = int(dop.order()) if ordinary else 0
    assert start <= n0_squash # the special path doesn't squash its result
    # The next terms of the sum may need a higher log-prec than the current one.
    rec_add_log_prec = sum(len(v) for s, v in inis[0].shift.items()
                                   if start <= s < start + precomp_len)
    assert rec_add_log_prec == 0 or not ordinary
    bwrec_nplus = collections.deque(
            (bwrec.eval_series(Intervals, start + i,
                               log_prec + rec_add_log_prec)
                for i in range(precomp_len)),
            maxlen=precomp_len)

    for n in count():

        if n%stride == 0 and n > 0:
            assert log_prec == 1 or not ordinary
            radpowest = (abs(jetpow[0]) if pt.is_numeric
                         else Intervals(pt.rad**n))
            est = sum(sol.coeff_estimate() for sol in sols)*radpowest
            sing = (n <= last_index_with_ini) or (mult > 0) # ?
            done, tail_bound = stop.check(cb, sing, n, tail_bound, est, stride)
            if done:
                break

        if n < start:
            assert ordinary
            for sol in sols:
                sol.next_term_ordinary_initial_part(n, jetpow)

        else:
            # seems faster than relying on __missing__()
            mult = mult_dict[n] if n in mult_dict else 0
            cst = - ~bwrec_nplus[0][0][mult]
            squash = (n >= n0_squash)
            if squash:
                rnd_shift, hom_maj_coeff_lb = next(rnd_den)
                assert n0_squash + rnd_shift == n
            for sol in sols:
                err = sol.next_term(n, mult, bwrec_nplus[0], cst, jetpow, squash)
                if squash:
                    rnd_loc = rnd_loc.max(n*err/hom_maj_coeff_lb)
                    if not rnd_loc.is_finite(): # normalize NaNs and infinities
                        rnd_loc = rnd_loc.parent()('inf')
            if mult > 0:
                log_prec = max(1, max(sol.log_prec for sol in sols))

            rec_add_log_prec += mult_dict[n + precomp_len] - mult
            bwrec_nplus.append(bwrec.eval_series(Intervals, n + precomp_len,
                                                 log_prec + rec_add_log_prec))

        jetpow = jetpow._mul_trunc_(jet, ord)
        radpow *= pt.rad

    # Accumulated round-off errors
    # XXX: maybe move this to PartialSum, and/or do it at every convergence
    # check
    if n0_squash < sys.maxsize:
        # |ind(n)| = cst·|monic_ind(n)|
        cst = abs(bounds.IC(stop.maj.dop.leading_coefficient()[0]))
        rnd_fac = cst*rnd_maj.bound(pt.rad, rows=ord)/n0_squash
        rnd_err = rnd_loc*rnd_fac
        for sol in sols:
            sol.update_enclosure(Jets, pt, tail_bound + rnd_err)
    else:
        rnd_err = bounds.IR.zero()

    logger.info("summed %d terms, tails = %s (est = %s), rnd_err <= %s, "
                "interval width <= %s",
            n, tail_bound, bounds.IR(est), rnd_err,
            max(sol.interval_width() for sol in sols) if pt.is_numeric
                                                      else None)

    return sols

################################################################################
# Miscellaneous utilities
################################################################################

# Temporary: later on, polynomials with ball coefficients could implement
# add_error themselves.
def _add_error(approx, error):
    if isinstance(approx, polynomial_element.Polynomial):
        return approx[0].add_error(error) + ((approx >> 1) << 1)
    else:
        return approx.add_error(error)

def _get_error(approx):
    if isinstance(approx, polynomial_element.Polynomial):
        return approx[0].abs().rad_as_ball()
    else:
        return approx.abs().rad_as_ball()

def _ctz(vec, maxlen):
    z = 0
    for m in range(maxlen):
        if vec[-1 - m].is_zero():
            z += 1
        else:
            break
    return z

def _resize_list(l, n, z):
    n0 = len(l)
    if n > n0:
        l.extend(repeat(z, n - n0))
    elif n < n0:
        l[n:] = []
