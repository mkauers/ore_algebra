# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of convergent D-finite series by direct summation
"""

# TODO:
# - support summing a given number of terms rather than until a target accuracy
# is reached?
# - cythonize critical parts?

from __future__ import division, print_function

import collections, itertools, logging

from sage.categories.pushout import pushout
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
        LogSeriesInitialValues, LocalBasisMapper, log_series_value)
from .safe_cmp import *
from .utilities import short_str

logger = logging.getLogger(__name__)

################################################################################
# Argument processing etc. (common to the ordinary and the regular case)
################################################################################

class EvaluationPoint(object):
    r"""
    Series evaluation point/jet.

    A ring element (a complex number, a polynomial indeterminate, perhaps
    someday a matrix) where to evaluate the partial sum of a series, along with
    a “jet order” used to compute derivatives and a bound on the norm of the
    mathematical quantity it represents that can be used to bound the truncation
    error.
    """

    # XXX: choose a single place to set the default value for jet_order
    def __init__(self, pt, rad=None, jet_order=1, branch=(0,)):
        self.pt = pt
        self.rad = (bounds.IR.coerce(rad) if rad is not None
                    else bounds.IC(pt).above_abs())
        self.jet_order = jet_order
        self.branch=branch

        self.is_numeric = utilities.is_numeric_parent(pt.parent())

    def __repr__(self):
        fmt = "{} + η + O(η^{}) (with |.| ≤ {})"
        return fmt.format(self.pt, self.jet_order + 1, self.rad)

    def jet(self, Intervals):
        base_ring = (Intervals if self.is_numeric
                     else pushout(self.pt.parent(), Intervals))
        Pol = PolynomialRing(base_ring, 'delta')
        return Pol([self.pt, 1]).truncate(self.jet_order)

    def is_real(self):
        return utilities.is_real_parent(self.pt.parent())

    def accuracy(self):
        if self.pt.parent().is_exact():
            return bounds.IR.maximal_accuracy()
        elif isinstance(self.pt.parent(), (RealBallField, ComplexBallField)):
            return self.pt.accuracy()
        else:
            raise ValueError

def series_sum(dop, ini, pt, tgt_error, maj=None, bwrec=None, stop=None,
        max_prec=100000, fail_fast=False, **kwds):
    r"""
    Sum a (generalized) series solution of dop.

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
        ....:         EvaluationPoint(x, rad=RBF(1), jet_order=2),
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

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, 1/3], 5/7, 1e-16, max_prec=10**1000)
        INFO:...
        ([0.20674982866094049...])

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], 5/7, 1e-16, max_prec=10**1000)
        WARNING:ore_algebra.analytic.naive_sum:input intervals may be too wide compared to requested accuracy
        ...
        ([0.206749828660940...])

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], RBF(5/7), 1e-12, max_prec=10**1000)
        INFO:...
        ([0.2067498286609...])

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], RBF(5/7), 1e-20, max_prec=10**1000)
        WARNING:ore_algebra.analytic.naive_sum:input intervals may be too wide compared to requested accuracy
        ...
        INFO:ore_algebra.analytic.naive_sum:lost too much precision, giving up
        ([0.20674982866094...])

        sage: xx = EvaluationPoint(x, rad=RBF(1/4), jet_order=2)
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
    if not isinstance(tgt_error, accuracy.OldStoppingCriterion):
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

    doit = (series_sum_ordinary if dop.leading_coefficient().valuation() == 0
            else series_sum_regular)

    return interval_series_sum_wrapper(doit, dop, ini, pt, tgt_error, bwrec,
                                       stop, fail_fast, max_prec, **kwds)

def guard_bits(dop, maj, pt, ordrec, nterms):

    inf = 1 << 60
    new_cost = cur_cost = inf
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
            guard_bits_squashed = inf

        # We expect the effective working precision to decrease linearly in the
        # first phase due to interval blow-up, and then stabilize around (target
        # prec + guard_bits_squashed).
        new_cost = (new_n0//2)*guard_bits_intervals + nterms*guard_bits_squashed
        new_bits = guard_bits_intervals + guard_bits_squashed

        logger.debug(
                "n0 = %s, terms = %s, guard bits = %s+%s = %s, cost = %s",
                new_n0, nterms, guard_bits_intervals, guard_bits_squashed,
                new_bits, new_cost)

        if cur_cost <= new_cost < inf:
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

def interval_series_sum_wrapper(doit, dop, ini, pt, tgt_error, bwrec, stop,
                                fail_fast, max_prec, stride=None,
                                squash_intervals=False):

    if stride is None:
        stride = max(50, 2*bwrec.order)
    ivs = (RealBallField
           if ini.is_real(dop) and (pt.is_real() or not pt.is_numeric)
           else ComplexBallField)
    input_accuracy = min(pt.accuracy(), ini.accuracy())

    bit_prec0 = utilities.prec_from_eps(tgt_error.eps)
    old_bit_prec = 8 + bit_prec0*(1 + ZZ(bwrec.order - 2).nbits())
    if doit is series_sum_ordinary and squash_intervals:
        nterms, lg_mag = dop.est_terms(pt, bit_prec0)
        nterms = (bwrec.order*dop.order() + nterms)*1.2 # let's be pragmatic
        nterms = ZZ((nterms//stride + 1)*stride)
        bit_prec0 += ZZ(dop._naive_height()).nbits() + lg_mag + nterms.nbits()
        n0_squash, g = guard_bits(dop, stop.maj, pt, bwrec.order, nterms)
        # adding twice the computed number of guard bits seems to work better
        # in practice, but I don't really understand why
        bit_prec = bit_prec0 + 2*g
        logger.info("working precision = %s + %s = %s (old = %s), "
                    "squashing intervals for n >= %s",
                    bit_prec0, g, bit_prec, old_bit_prec, n0_squash)
        if bit_prec > 4*bit_prec0 and fail_fast:
            raise accuracy.PrecisionError
    else:
        bit_prec = old_bit_prec
        n0_squash = 1 << 60

    if max_prec is None:
        max_prec = bit_prec*3
    max_prec = min(max_prec, bit_prec + 2*input_accuracy)
    logger.log(logging.INFO - 1, "target error = %s", tgt_error)
    logger.info("initial precision = %s bits", bit_prec)

    err=None
    for attempt in itertools.count(1):
        try:
            Intervals = ivs(bit_prec)
            ini_are_accurate = 2*min(pt.accuracy(), ini.accuracy()) > bit_prec
            # Strictly decrease eps each time to avoid situations where doit
            # would be happy with the result and stop at the same point despite
            # the higher bit_prec. Since attempt starts at 1, we have a bit of
            # room for round-off errors.
            stop.reset(tgt_error.eps >> (4*attempt),
                       stop.fast_fail and ini_are_accurate)
            psum = doit(Intervals, dop, bwrec, ini, pt, stop, stride, n0_squash)
            err = max(_get_error(c) for c in psum)
            logger.debug("bit_prec=%s, err=%s (tgt=%s)", bit_prec, err,
                    tgt_error)
            abs_sum = abs(psum[0]) if pt.is_numeric else None
            if tgt_error.reached(err, abs_sum):
                return psum
        except accuracy.PrecisionError:
            if 2*bit_prec > max_prec:
                logger.info("lost too much precision, giving up")
                raise
        if fail_fast:
            raise accuracy.PrecisionError
        bit_prec *= 2
        if bit_prec > max_prec:
            logger.info("lost too much precision, giving up")
            return psum
        logger.info("lost too much precision, restarting with %d bits",
                    bit_prec)

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

    rnd_maj = stop.maj(n0_squash)
    rnd_maj >>= n0_squash # XXX (a) useful? (b) check correctness
    rnd_den = rnd_maj.exp_part_coeffs_lbounds()
    rnd_loc = bounds.IR.zero()

    for n in itertools.count():
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
    cst = abs(bounds.IC(stop.maj.dop.leading_coefficient()[0]))
    rnd_fac = cst*rnd_maj.bound(pt.rad, rows=ord)/n0_squash
    rnd_err = rnd_loc*rnd_fac

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
def fundamental_matrix_ordinary(dop, pt, eps, rows, branch, fail_fast):
    if branch != (0,):
        logger.warn("nontrivial branch choice at ordinary point")
    eps_col = bounds.IR(eps)/bounds.IR(dop.order()).sqrt()
    eps_col = accuracy.AbsoluteError(eps_col)
    evpt = EvaluationPoint(pt, jet_order=rows)
    bwrec = bw_shift_rec(dop)
    inis = [
        LogSeriesInitialValues(ZZ.zero(), ini, dop, check=False)
        for ini in identity_matrix(dop.order())]
    maj = bounds.DiffOpBound(dop, pol_part_len=4, bound_inverse="solve")
    assert len(maj.special_shifts) == 1 and maj.special_shifts[0] == 1
    stop = accuracy.StoppingCriterion(maj, eps_col.eps)
    cols = [
        interval_series_sum_wrapper(series_sum_ordinary, dop, ini, evpt,
                                    eps_col, bwrec, stop, fail_fast,
                                    max_prec=None)
        for ini in inis]
    return matrix(cols).transpose()

################################################################################
# Regular singular points
################################################################################

# TODO: Avoid redundant computations at multiple roots of the initial equation
# (easy in principle after cleaning up series_sum() for a single root of high
# multiplicity, needs partial sums from one root to the next or something
# similar in the general case).

def fundamental_matrix_regular(dop, pt, eps, rows, branch, fail_fast):
    r"""
    Fundamental matrix at a possibly regular singular point

    TESTS::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.naive_sum import *
        sage: from ore_algebra.analytic.differential_operator import DifferentialOperator
        sage: Dops, x, Dx = DifferentialOperators()

        sage: fundamental_matrix_regular(
        ....:         DifferentialOperator(x*Dx^2 + (1-x)*Dx),
        ....:         1, RBF(1e-10), 2, (0,), 100)
        [[1.317902...] 1.000000...]
        [[2.718281...]           0]

        sage: dop = DifferentialOperator(
        ....:         (x+1)*(x^2+1)*Dx^3-(x-1)*(x^2-3)*Dx^2-2*(x^2+2*x-1)*Dx)
        sage: fundamental_matrix_regular(dop, 1/3, RBF(1e-10), 3, (0,), 100)
        [1.0000000...  [0.321750554...]  [0.147723741...]]
        [           0  [0.900000000...]  [0.991224850...]]
        [           0  [-0.27000000...]  [1.935612425...]]

        sage: dop = DifferentialOperator(
        ....:     (2*x^6 - x^5 - 3*x^4 - x^3 + x^2)*Dx^4
        ....:     + (-2*x^6 + 5*x^5 - 11*x^3 - 6*x^2 + 6*x)*Dx^3
        ....:     + (2*x^6 - 3*x^5 - 6*x^4 + 7*x^3 + 8*x^2 - 6*x + 6)*Dx^2
        ....:     + (-2*x^6 + 3*x^5 + 5*x^4 - 2*x^3 - 9*x^2 + 9*x)*Dx)
        sage: fundamental_matrix_regular(dop, RBF(1/3), RBF(1e-10), 4, (0,), 100)
        [ [3.1788470...] [-1.064032...]  [1.000...] [0.3287250...]]
        [ [-8.981931...] [3.2281834...]    [+/-...] [0.9586537...]]
        [  [26.18828...] [-4.063756...]    [+/-...] [-0.123080...]]
        [ [-80.24671...]  [9.190740...]    [+/-...] [-0.119259...]]

        sage: dop = x*Dx^3 + 2*Dx^2 + x*Dx
        sage: ini = [1, CBF(euler_gamma), 0]
        sage: dop.numerical_solution(ini, [0, RBF(1/3)], 1e-14)
        [-0.549046117782...]
    """
    evpt = EvaluationPoint(pt, jet_order=rows, branch=branch)
    eps_col = bounds.IR(eps)/bounds.IR(dop.order()).sqrt()
    eps_col = accuracy.AbsoluteError(eps_col)
    class Mapper(LocalBasisMapper):
        def process_modZ_class(self):
            maj = bounds.DiffOpBound(dop, self.leftmost, self.shifts,
                                     bound_inverse="solve")
            self.stop = accuracy.StoppingCriterion(maj, eps_col.eps)
            super(self.__class__, self).process_modZ_class()
        def fun(self, ini):
            return interval_series_sum_wrapper(series_sum_regular, dop, ini,
                    evpt, eps_col, self.shifted_bwrec, self.stop, fail_fast,
                    max_prec=None)
    cols = Mapper(dop).run()
    return matrix([sol.value for sol in cols]).transpose()

# This function only handles the case of a “single” series, i.e. a series where
# all indices differ from each other by integers. But since we need logic to go
# past singular indices anyway, we can allow for general initial conditions (at
# roots of the indicial equation belonging to the same shift-equivalence class),
# not just initial conditions associated to canonical solutions.
def series_sum_regular(Intervals, dop, bwrec, ini, pt, stop, stride, n0_squash):
    r"""
    Sum a (logarithmic) series solution of an operator that may have a regular
    singular point at the origin.

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
        sage: dop.numerical_transition_matrix([0,1/10000000])
        INFO:ore_algebra.analytic.naive_sum:...
        INFO:ore_algebra.analytic.naive_sum:summed 50 terms, ...
        [ [1.000000100000005...] [1.0000000000000000e-7000...]]
        [ [1.000000100000005...] [1.0000000000000000e-6990...]]
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

    jet = pt.jet(Intervals)
    Jets = jet.parent()
    ord = pt.jet_order
    jetpow = Jets.one()
    radpow = bounds.IR.one() # bound on abs(pt)^n in the series part (=> starts
                             # at 1 regardless of ini.expo)

    log_prec = sum(len(v) for v in ini.shift.itervalues())
    last_index_with_ini = max([dop.order()]
            + [s for s, vals in ini.shift.iteritems()
                 if not all(v.is_zero() for v in vals)])
    last = collections.deque([vector(Intervals, log_prec)
                              for _ in xrange(bwrec.order + 1)])
    psum = vector(Jets, log_prec)

    # Every few iterations, heuristically check if we have converged and if
    # we still have enough precision. If it looks like the target error may
    # be reached, perform a rigorous check. Our stopping criterion currently
    # (1) only works at “generic” indices, and (2) assumes that the initial
    # values at exceptional indices larger than n are zero, so we also
    # ensure that we are in this case. (Both assumptions could be lifted,
    # (1) by using a slightly more complicated formula for the tail bound,
    # and (2) if we had code to compute lower bounds on coefficients of
    # series expansions of majorants.)
    tail_bound = bounds.IR(infinity)
    bit_prec = Intervals.precision()
    ini_are_accurate = 2*min(pt.accuracy(), ini.accuracy()) > bit_prec

    # TODO: improve the automatic increase of precision for large x^λ:
    # we currently check the series part only (which would sort of make
    # sense in a relative error setting)
    class BoundCallbacks(accuracy.BoundCallbacks):
        def __init__(self):
            self.val = None
        def get_residuals(self):
            return [stop.maj.normalized_residual(n, list(last)[1:], bwrec_nplus)]
        def get_bound(self, residuals):
            maj = self.get_maj(stop, n, residuals)
            tb = maj.bound(pt.rad, rows=pt.jet_order)
            my_psum = vector(Jets, [[t[i].add_error(tb.abs())
                                    for i in range(ord)] for t in psum])
            # XXX decouple this from the summation => less redundant computation
            # of local monodromy matrices
            self.val = log_series_value(Jets, ord, ini.expo, my_psum, jet[0],
                                        branch=pt.branch)
            return max([RBF.zero()] + [_get_error(c) for c in self.val])
        def get_value(self):
            my_psum = vector(Jets, [[t[i] for i in range(ord)] for t in psum])
            my_val = log_series_value(Jets, ord, ini.expo, my_psum, jet[0],
                                    branch=pt.branch)
            return my_val
    cb = BoundCallbacks()

    precomp_len = max(1, bwrec.order) # hack for recurrences of order zero
    bwrec_nplus = collections.deque(
            (bwrec.eval_series(Intervals, i, log_prec)
                for i in xrange(precomp_len)),
            maxlen=precomp_len)
    for n in itertools.count():
        last.rotate(1)
        logger.log(logging.DEBUG - 2, "n = %s, [c(n), c(n-1), ...] = %s", n, list(last))
        logger.log(logging.DEBUG - 1, "n = %s, sum = %s", n, psum)
        mult = len(ini.shift.get(n, ()))

        if n%stride == 0:
            radpowest = abs(jetpow[0])
            est = sum(abs(a) for log_jet in last for a in log_jet) * radpowest
            sing = (n <= last_index_with_ini) or (mult > 0)
            done, tail_bound = stop.check(cb, sing, n, tail_bound, est, stride)
            if done:
                break

        for p in xrange(log_prec - mult - 1, -1, -1):
            combin  = sum(bwrec_nplus[0][i][j]*last[i][p+j]
                          for j in xrange(log_prec - p)
                          for i in xrange(bwrec.order, 0, -1))
            combin += sum(bwrec_nplus[0][0][j]*last[0][p+j]
                          for j in xrange(mult + 1, log_prec - p))
            last[0][mult + p] = - ~bwrec_nplus[0][0][mult] * combin
        for p in xrange(mult - 1, -1, -1):
            last[0][p] = ini.shift[n][p]
        psum += last[0]*jetpow
        jetpow = jetpow._mul_trunc_(jet, ord)
        radpow *= pt.rad
        bwrec_nplus.append(bwrec.eval_series(Intervals, n+precomp_len, log_prec))
    logger.info("summed %d terms, global tail bound = %s (est = %s)",
            n, tail_bound, bounds.IR(est))
    result = vector(cb.val[i] for i in xrange(ord))
    return result

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
