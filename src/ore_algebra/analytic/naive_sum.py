# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of convergent D-finite series by direct summation
"""

# TODO:
# - support summing a given number of terms rather than until a target accuracy
# is reached?
# - cythonize critical parts?

from __future__ import division, print_function

import collections, logging, sys

from itertools import count, chain

from sage.matrix.constructor import identity_matrix, matrix
from sage.modules.free_module_element import vector
from sage.rings.all import ZZ, QQ, RR, QQbar, infinity
from sage.rings.complex_arb import ComplexBallField, CBF, ComplexBall
from sage.rings.integer import Integer
from sage.rings.polynomial import polynomial_element
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.real_arb import RealBallField, RBF, RealBall

from .. import ore_algebra
from . import accuracy, bounds, utilities
from .differential_operator import DifferentialOperator
from .local_solutions import (bw_shift_rec, FundamentalSolution,
        LogSeriesInitialValues, LocalBasisMapper, log_series_values)
from .path import EvaluationPoint
from .safe_cmp import *
from .utilities import short_str

logger = logging.getLogger(__name__)

################################################################################
# Argument processing etc. (common to the ordinary and the regular case)
################################################################################

def series_sum(dop, ini, pt, tgt_error, maj=None, bwrec=None, stop=None,
               fail_fast=False, effort=2, **kwds):
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
    :class:`EvaluationPoint` that wraps such an object. Polynomials (wrapped in
    EvaluationPoints) are also supported to some extent (essentially, this is
    intended for use with polynomial indeterminates, and anything else that
    works does so by accident). ::

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
            logger.warn("input intervals may be too wide "
                        "compared to requested accuracy")

    if maj is None:
        special_shifts = [(s, len(v)) for s, v in ini.shift.iteritems()]
        maj = bounds.DiffOpBound(dop, ini.expo, special_shifts)
    if bwrec is None:
        bwrec = bw_shift_rec(dop, shift=ini.expo)
    if stop is None:
        stop = accuracy.StoppingCriterion(maj, tgt_error.eps)

    sols = interval_series_sum_wrapper(False, dop, [ini], pt, tgt_error,
                                        bwrec, stop, fail_fast, effort, **kwds)
    assert len(sols) == 1
    return sols[0].value

def guard_bits(dop, maj, pt, ordrec, nterms):

    new_cost = cur_cost = sys.maxint
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
            guard_bits_squashed = sys.maxint

        # We expect the effective working precision to decrease linearly in the
        # first phase due to interval blow-up, and then stabilize around (target
        # prec + guard_bits_squashed).
        new_cost = (new_n0//2)*guard_bits_intervals + nterms*guard_bits_squashed
        new_bits = guard_bits_intervals + guard_bits_squashed

        logger.debug(
                "n0 = %s, terms = %s, guard bits = %s+%s = %s, cost = %s",
                new_n0, nterms, guard_bits_intervals, guard_bits_squashed,
                new_bits, new_cost)

        if cur_cost <= new_cost < sys.maxint:
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

def interval_series_sum_wrapper(ordinary, dop, inis, pt, tgt_error, bwrec, stop,
                                fail_fast, effort, stride=None,
                                squash_intervals=False):

    ordinary = False

    if stride is None:
        stride = max(50, 2*bwrec.order)
    if pt.is_real_or_symbolic() and all(ini.is_real(dop) for ini in inis):
        ivs = RealBallField
    else:
        ivs = ComplexBallField
    input_accuracy = min(chain([pt.accuracy()],
                               (ini.accuracy() for ini in inis)))
    logger.log(logging.INFO - 1, "target error = %s", tgt_error)

    bit_prec0 = utilities.prec_from_eps(tgt_error.eps)
    old_bit_prec = 8 + bit_prec0*(1 + ZZ(bwrec.order - 2).nbits())
    if ordinary and squash_intervals:
        nterms, lg_mag = dop.est_terms(pt, bit_prec0)
        nterms = (bwrec.order*dop.order() + nterms)*1.2 # let's be pragmatic
        nterms = ZZ((nterms//stride + 1)*stride)
        bit_prec0 += ZZ(dop._naive_height()).nbits() + lg_mag + nterms.nbits()
        n0_squash, g = guard_bits(dop, stop.maj, pt, bwrec.order, nterms)
        # adding twice the computed number of guard bits seems to work better
        # in practice, but I don't really understand why
        bit_prec = bit_prec0 + 2*g
        logger.info("initial working precision = %s + %s = %s (old = %s), "
                    "squashing intervals for n >= %s",
                    bit_prec0, 2*g, bit_prec, old_bit_prec, n0_squash)
        if fail_fast and bit_prec > 4*bit_prec0 and effort <= 1:
            raise accuracy.PrecisionError
    else:
        bit_prec = old_bit_prec
        n0_squash = sys.maxint
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

        if ordinary: # temporarily need two variants
            try:
                assert len(inis) == 1
                psum = series_sum_ordinary(Intervals, dop, bwrec, inis[0], pt,
                                           stop, stride, n0_squash)
            except accuracy.PrecisionError:
                if attempt > effort:
                    raise
            else:
                err = max(_get_error(c) for c in psum)
                logger.debug("bit_prec=%s, err=%s (tgt=%s)", bit_prec, err,
                            tgt_error)
                abs_sum = abs(psum[0]) if pt.is_numeric else None
                if tgt_error.reached(err, abs_sum):
                    return psum
        else:
            try:
                sols = series_sum_regular(Intervals, dop, bwrec, inis, pt, stop,
                                          stride, n0_squash)
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

        bit_prec *= 2
        if attempt <= effort and bit_prec < max_prec:
            logger.info("lost too much precision, restarting with %d bits",
                        bit_prec)
            continue
        if fail_fast:
            raise accuracy.PrecisionError
        else:
            logger.info("lost too much precision, giving up")
            return psum if ordinary else sols

################################################################################
# Ordinary points
################################################################################

def series_sum_ordinary(Intervals, dop, bwrec, ini, pt, stop, stride,
                        n0_squash):

    jet = pt.jet(Intervals)
    Jets = jet.parent() # polynomial ring!
    ord = pt.jet_order
    jetpow = Jets.one()

    ordrec = bwrec.order
    assert ini.expo.is_zero()
    last = collections.deque([Intervals.zero()]*(ordrec - dop.order() + 1))
    last.extend(Intervals(ini.shift[n][0])
                for n in xrange(dop.order() - 1, -1, -1))
    assert len(last) == ordrec + 1 # not ordrec!
    psum = Jets.zero()

    tail_bound = bounds.IR(infinity)

    start = dop.order()
    # Evaluate the coefficients a bit in advance as we are going to need them to
    # compute the residuals. This is not ideal at high working precision, but
    # already saves a lot of time compared to doing the evaluations twice.
    bwrec_ev = bwrec.eval_method(Intervals)
    bwrec_nplus = collections.deque(
            (bwrec_ev(start+i) for i in xrange(ordrec)),
            maxlen=ordrec)

    class BoundCallbacks(accuracy.BoundCallbacks):
        def get_residuals(self):
            return [stop.maj.normalized_residual(n, [[c] for c in last][1:],
                        [[[c] for c in l] for l in bwrec_nplus])]
        def get_bound(self, residuals):
            return self.get_maj(stop, n, residuals).bound(pt.rad, rows=ord)
        def get_value(self):
            return psum
    cb = BoundCallbacks()

    if n0_squash < sys.maxint:
        rnd_maj = stop.maj(n0_squash)
        rnd_maj >>= n0_squash # XXX (a) useful? (b) check correctness
        rnd_den = rnd_maj.exp_part_coeffs_lbounds()
        rnd_loc = bounds.IR.zero()

    for n in count():
        last.rotate(1)
        #last[0] = None
        # At this point last[0] should be considered undefined (it will hold
        # the coefficient of z^n later in the loop body) and last[1], ...
        # last[ordrec] are the coefficients of z^(n-1), ..., z^(n-ordrec)
        if n%stride == 0:
            radpowest = abs(jetpow[0] if pt.is_numeric
                            else Intervals(pt.rad**n))
            est = sum(abs(a) for a in last)*radpowest
            sing = (n <= start)
            done, tail_bound = stop.check(cb, sing, n, tail_bound, est, stride)
            if done:
                break
        if n >= start:
            bwrec_n = (bwrec_nplus[0] if bwrec_nplus else bwrec_ev(n))
            comb = sum(bwrec_n[k]*last[k] for k in xrange(1, ordrec+1))
            last[0] = -~bwrec_n[0]*comb
            if n >= n0_squash:
                rnd_shift, hom_maj_coeff_lb = next(rnd_den)
                assert n0_squash + rnd_shift == n
                rnd_loc = rnd_loc.max(
                        n*bounds.IR(last[0].rad())/hom_maj_coeff_lb)
                last[0] = last[0].squash()
            bwrec_nplus.append(bwrec_ev(n+bwrec.order))
            # logger.debug("n = %s, [c(n), c(n-1), ...] = %s", n, list(last))
        term = Jets(last[0])._mul_trunc_(jetpow, ord)
        psum += term
        jetpow = jetpow._mul_trunc_(jet, ord)

    # Accumulated round-off errors (|ind(n)| = cst·|monic_ind(n)|)
    if n0_squash < sys.maxint:
        cst = abs(bounds.IC(stop.maj.dop.leading_coefficient()[0]))
        rnd_fac = cst*rnd_maj.bound(pt.rad, rows=ord)/n0_squash
        rnd_err = rnd_loc*rnd_fac
    else:
        rnd_err = bounds.IR.zero()

    logger.info("summed %d terms, tail <= %s (est = %s), rnd err <= %s, "
                "interval width <= %s",
            n, tail_bound, bounds.IR(est), rnd_err,
            max(psum[i].rad() for i in range(ord)) if pt.is_numeric else "n/a")

    # Account for the truncation and round-off errors in the intervals we return
    # (tail_bound is actually a bound on the Frobenius norm of the error matrix,
    # so there is some overestimation).
    #
    # WARNING: For symbolic x, the resulting polynomials have to be interpreted
    # with some care: in particular, it would be incorrect to evaluate a
    # polynomial result with real coefficients at a complex point. Our current
    # mechanism to choose whether to add a real or complex error bound in this
    # case is pretty fragile.
    err = tail_bound + rnd_err
    res = vector(_add_error(psum[i], err) for i in xrange(ord))

    return res

# XXX: pass ctx (→ real/complex?)?
def fundamental_matrix_ordinary(dop, pt, eps, fail_fast, effort):
    if pt.branch != (0,):
        logger.warn("nontrivial branch choice at ordinary point")
    eps_col = bounds.IR(eps)/bounds.IR(dop.order()).sqrt()
    eps_col = accuracy.AbsoluteError(eps_col)
    bwrec = bw_shift_rec(dop)
    inis = [
        LogSeriesInitialValues(ZZ.zero(), ini, dop, check=False)
        for ini in identity_matrix(dop.order())]
    maj = bounds.DiffOpBound(dop, pol_part_len=4, bound_inverse="solve")
    assert len(maj.special_shifts) == 1 and maj.special_shifts[0] == 1
    stop = accuracy.StoppingCriterion(maj, eps_col.eps)
    cols = [
        interval_series_sum_wrapper(True, dop, [ini], pt,
                                    eps_col, bwrec, stop, fail_fast, effort)
        for ini in inis]
    return matrix(cols).transpose()

################################################################################
# Regular singular points
################################################################################

# TODO: Avoid redundant computations at multiple roots of the indicial equation
# (easy in principle after cleaning up series_sum() for a single root of high
# multiplicity, needs partial sums from one root to the next or something
# similar in the general case).

def fundamental_matrix_regular(dop, pt, eps, fail_fast, effort):
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
    class Mapper(LocalBasisMapper):
        def process_modZ_class(self):
            logger.info(r"solutions z^(%s+n)·log(z)^k/k! + ···, n = %s",
                        self.leftmost, ", ".join(str(s) for s, _ in self.shifts))
            maj = bounds.DiffOpBound(dop, self.leftmost, self.shifts,
                                     bound_inverse="solve")
            stop = accuracy.StoppingCriterion(maj, eps_col.eps)
            # Compute the "highest" (in terms powers of log) solution of each
            # valuation
            inis = [LogSeriesInitialValues(
                        expo=self.leftmost,
                        mults=self.shifts,
                        values={(s, m-1): ZZ.one()})
                    for s, m in self.shifts]
            highest_sols = interval_series_sum_wrapper(False, dop, inis, pt,
                         eps_col, self.shifted_bwrec, stop, fail_fast, effort)
            self.highest_sols = {}
            for (s, m), sol in zip(self.shifts, highest_sols):
                sol.update_downshifts(pt, range(m))
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

    cols = Mapper(dop).run()
    return matrix([sol.value for sol in cols]).transpose()

class PartialSum(object):

    def __init__(self, Intervals, Jets, ini, ordrec):

        self.Intervals = Intervals
        self.ini = ini
        self.ordrec = ordrec

        self.log_prec = 0
        self.trunc = 0 # first term _not_ in the sum
        last = [vector(Intervals, 0) for _ in xrange(ordrec + 1)]
        self.last = collections.deque(last) # u[trunc-1], u[trunc-2], ...
        self.critical_coeffs = {}
        self.psum = vector(Jets, 0)
        self.tail_bound = bounds.IR(infinity)
        self.total_error = bounds.IR(infinity)

    def coeff_estimate(self):
        return sum(abs(a) for log_jet in self.last for a in log_jet)

    def next_term(self, n, mult, bwrec_nplus, cst, jetpow, squash):

        assert n == self.trunc
        self.last.rotate(1)
        self.trunc += 1

        if mult > 0:
            self.last[0] = vector(self.Intervals, self.log_prec + mult)

        zero = self.Intervals.zero()

        for p in xrange(self.log_prec - 1, -1, -1):
            combin  = sum((bwrec_nplus[0][i][j]*self.last[i][p+j]
                            for j in xrange(self.log_prec - p)
                            for i in xrange(self.ordrec, 0, -1)),
                          zero)
            combin += sum((bwrec_nplus[0][0][j]*self.last[0][p+j]
                           for j in xrange(mult + 1, mult + self.log_prec - p)),
                          zero)
            self.last[0][mult + p] = cst * combin
        for p in xrange(mult - 1, -1, -1):
            self.last[0][p] = self.ini.shift[n][p]

        if mult > 0:

            self.critical_coeffs[n] = self.last[0]

            nz = mult - _ctz(self.last[0], mult)
            self.log_prec += nz
            for i in xrange(len(self.last)):
                self.last[i] = _resize_vector(self.last[i], self.log_prec)
            self.psum = _resize_vector(self.psum, self.log_prec)

        if squash:
            err = last[0][0].rad()
            last[0][0] = last[0][0].squash()
        else:
            err = None

        self.psum += self.last[0]*jetpow

        return err

    def update_enclosure(self, Jets, pt, tb):
        self.series = vector(Jets, self.log_prec)
        for i, t in enumerate(self.psum):
            self.series[i] = Jets([_add_error(t[k], tb)
                                   for k in xrange(pt.jet_order)])
        # log_series_values() may decide to introduce complex numbers if there
        # are logs, and hence the parent of the partial sum may switch from real
        # to complex during the computation...
        [self.value] = log_series_values(Jets, self.ini.expo, self.series, pt)
        self.total_error = max(chain(iter([bounds.IR.zero()]),
                                     (_get_error(c) for c in self.value)))

    def update_downshifts(self, pt, downshift):
        self.downshifts = log_series_values(self.series.base_ring(),
                self.ini.expo, self.series, pt, downshift=downshift)

    def bare_value(self, Jets, pt):
        r"""
        Value taking into account logs etc. but ignoring the truncation error.
        """
        [v] = log_series_values(Jets, self.ini.expo, self.psum, pt)
        return v

    def interval_width(self):
        return max(c.rad() for c in self.value)

def series_sum_regular(Intervals, dop, bwrec, inis, pt, stop, stride,
                        n0_squash):
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

    if n0_squash < sys.maxint:
        assert ordinary
        self.rnd_den = rnd_maj.exp_part_coeffs_lbounds()
        self.rnd_loc = bounds.IR.zero()
        rnd_maj = stop.maj(n0_squash)
        rnd_maj >>= n0_squash # XXX (a) useful? (b) check correctness

    last_index_with_ini = max(chain(iter([dop.order()]),
                                    (ini.last_index() for ini in inis)))
    sols = [PartialSum(Intervals, Jets, ini, bwrec.order) for ini in inis]

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
    if ordinary: # TBI?
        rec_add_log_prec = 1
    else:
        rec_add_log_prec = sum(len(v) for s, v in ini.shift.iteritems()
                                      if s < precomp_len)
    bwrec_nplus = collections.deque(
            (bwrec.eval_series(Intervals, i, log_prec + rec_add_log_prec)
                for i in xrange(precomp_len)),
            maxlen=precomp_len)

    for n in count():

        if n%stride == 0 and n > 0:
            if ordinary:
                assert log_prec == 1
            radpowest = (abs(jetpow[0]) if pt.is_numeric
                         else Intervals(pt.rad**n))
            est = sum(sol.coeff_estimate() for sol in sols)*radpowest
            sing = (n <= last_index_with_ini) or (mult > 0) # ?
            done, tail_bound = stop.check(cb, sing, n, tail_bound, est, stride)
            if done:
                break

        mult = mult_dict[n]
        cst = - ~bwrec_nplus[0][0][mult]
        if n >= n0_squash:
            rnd_shift, hom_maj_coeff_lb = next(rnd_den)
            assert n0_squash + rnd_shift == n
        for sol in sols:
            err = sol.next_term(n, mult, bwrec_nplus, cst, jetpow, n >= n0_squash)
            if n >= n0_squash:
                rnd_loc = rnd_loc.max(n*err/hom_maj_coeff_lb)
        if mult > 0:
            log_prec = max(1, max(sol.log_prec for sol in sols))
        jetpow = jetpow._mul_trunc_(jet, ord)
        radpow *= pt.rad

        if ordinary: # TBI?
            rec_add_log_prec = 1 if mult_dict[n + precomp_len] else 0
        else:
            rec_add_log_prec += mult_dict[n + precomp_len] - mult
        bwrec_nplus.append(bwrec.eval_series(Intervals, n+precomp_len,
                                             log_prec + rec_add_log_prec))

    # Accumulated round-off errors
    # XXX: maybe move this to PartialSum, and/or do it at every convergence
    # check
    if n0_squash < sys.maxint:
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
    for m in xrange(maxlen):
        if vec[-1 - m].is_zero():
            z += 1
        else:
            break
    return z

def _resize_vector(vec, length):
    new = vector(vec.base_ring(), length)
    for i in xrange(min(length, len(vec))):
        new[i] = vec[i]
    return new
