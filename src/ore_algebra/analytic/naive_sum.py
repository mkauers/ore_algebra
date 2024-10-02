# vim: tw=80
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

import collections
import logging
import sys

from itertools import count, chain, repeat

from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector
from sage.rings.all import ZZ, QQ, infinity
from sage.rings.complex_arb import ComplexBallField
from sage.rings.number_field import number_field_base
from sage.rings.polynomial import polynomial_element
from sage.rings.real_arb import RealBallField, RBF
from sage.structure.sequence import Sequence

from . import accuracy, bounds, utilities
from .context import Context, dctx
from .differential_operator import DifferentialOperator
from .local_solutions import (bw_shift_rec, LogSeriesInitialValues,
                              HighestSolMapper, log_series_values)
from .path import EvaluationPoint_base, EvaluationPoint

logger = logging.getLogger(__name__)

################################################################################
# Series summation
################################################################################

def cy_classes():
    try:
        from . import naive_sum_c
        return naive_sum_c.CoefficientSequence, naive_sum_c.PartialSum
    except ImportError:
        utilities.warn_no_cython_extensions(logger, fallback=True)
        return CoefficientSequence, PartialSum

class CoefficientSequence:

    def __init__(self, Intervals, ini, ordrec, real):
        r"""
        Coefficient sequence of a D-finite logarithmic series
        (that is, for a specific equation and specific initial values)
        """

        self.Intervals = Intervals
        self._use_sum_of_products = hasattr(Intervals, '_sum_of_products')
        self.ini = ini
        self.ordrec = ordrec

        # XXX get rid of this?
        self.force_real = real and isinstance(Intervals, ComplexBallField)

        self.log_prec = 0
        self.nterms = 0 # self.last[i] == u[n-1-i]

        self.critical_coeffs = {}

        # Start with vectors of length 1 instead of 0 (but still with log_prec
        # == 0) to avoid having to resize them, especially in the ordinary case
        last = [[Intervals.zero()] for _ in range(ordrec + 1)]
        self.last = collections.deque(last) # u[trunc-1], u[trunc-2], ...

    def coeff_estimate(self):
        return sum(abs(a) for log_jet in self.last for a in log_jet)

    def next_term_ordinary_initial_part(self, n):
        r"""
        Similar to next_term(), but limited to n < orddeq at ordinary points,
        does not support squasing, and does not require evaluating the
        recurrence.
        """
        self.last.rotate(1)
        self.last[0][0] = self.Intervals(self.ini.shift[n][0])
        if not self.ini.shift[n][0].is_zero():
            self.log_prec = 1
        self.nterms += 1

    def handle_singular_index(self, n, mult):

        self.critical_coeffs[n] = [
                c.real() if self.force_real else c
                for c in self.last[0]]

        nz = mult - utilities.ctz(self.last[0], mult)
        self.log_prec += nz
        for l in self.last:
            _resize_list(l, self.log_prec, self.Intervals.zero())

    def next_term(self, n, mult, bwrec_n, cst, squash):

        self.last.rotate(1)

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
            err = RBF(self.last[0][0].rad())
            self.last[0][0] = self.last[0][0].squash()

        for p in range(mult - 1, -1, -1):
            self.last[0][p] = self.Intervals(self.ini.shift[n][p])

        if mult > 0:
            self.handle_singular_index(n, mult)

        self.nterms += 1

        return err

class PartialSum:

    def __init__(self, cseq, Jets, ord, pt, pt_opts, IR):

        # Final data

        # Sequence coefficients
        self.cseq = cseq
        # Jets and orders (for computing derivatives)
        self.Jets = Jets
        self.ord = ord
        # Evaluation point and options used to evaluate singular terms
        self.pt = pt
        self.pt_opts = pt_opts
        # Parent used for error bounds
        self._IR = IR

        # Dynamic data

        self.trunc = 0 # first term _not_ in the sum
        # Though CoefficientSequences start with vector of length 1, here,
        # starting with partial sums of length 0 is better in some corner cases
        self.psum = []
        self.tail_bound = self._IR(infinity)
        self.total_error = self._IR(infinity)

        self.series = None
        self.value = None
        self.downshifts = []

    def next_term_ordinary_initial_part(self, jetpow):
        self.trunc += 1
        if not self.cseq.ini.shift[self.trunc-1][0].is_zero():
            if not self.psum:
                self.psum.append(self.Jets.zero())
            self.psum[0] += jetpow._lmul_(self.cseq.last[0][0])

    def next_term(self, jetpow, mult):
        self.trunc += 1
        if mult > 0:
            _resize_list(self.psum, self.cseq.log_prec, self.Jets.zero())
        for k in range(self.cseq.log_prec):
            self.psum[k] += jetpow._lmul_(self.cseq.last[0][k])

    def update_enclosure(self, tb):
        self.series = vector(self.Jets, self.cseq.log_prec)
        for i, t in enumerate(self.psum):
            self.series[i] = self.Jets([_add_error(t[k], tb)
                                   for k in range(self.ord)])
        # log_series_values() may decide to introduce complex numbers if there
        # are logs, and hence the parent of the partial sum may switch from real
        # to complex during the computation...
        [self.value] = log_series_values(self.Jets, self.cseq.ini.expo,
                                         self.series, self.pt, self.ord,
                                         *self.pt_opts)
        self.total_error = max(chain(iter([self._IR.zero()]),
                                     (_get_error(c) for c in self.value)))

    def update_downshifts(self, downshift):
        r"""
        Compute the values of the partial sums of this solution and its "down
        shifts".

        The down shifts are obtained by decreasing k by one in each occurrence
        of log(z)^k/k!, and removing the terms where k < 0.

        Unlike the other variants, this function forgets the imaginary part of
        the computed partial sums if self.force_real is set.
        """
        if self.cseq.force_real:
            Jets = self.Jets.change_ring(self.Jets.base().base())
            assert all(c.imag().contains_zero()
                       for jet in self.series for c in jet)
            jets = [Jets([c.real() for c in jet]) for jet in self.series]
            series = vector(Jets, self.cseq.log_prec, jets)
        else:
            Jets = self.Jets
            series = self.series
        self.downshifts = log_series_values(Jets, self.cseq.ini.expo, series,
                                            self.pt, self.ord, *self.pt_opts,
                                            downshift=downshift)

    def bare_value(self):
        r"""
        Value taking into account logs etc. but ignoring the truncation error.
        """
        psum = vector(self.Jets, self.psum)
        [v] = log_series_values(self.Jets, self.cseq.ini.expo, psum,
                                self.pt, self.ord, *self.pt_opts)
        return v

MPartialSums = collections.namedtuple("MPartialSums", ["cseq", "psums"])

class RecUnroller:
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
        sage: import logging; logging.basicConfig()
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

    def __init__(self, dop, inis, evpts, bwrec, *, ctx=dctx):

        dop = DifferentialOperator(dop)
        self.dop = dop

        # if not isinstance(inis, (list, tuple)):
        #     inis = [inis]
        # inis = [ini if isinstance(ini, LogSeriesInitialValues)
        #         else LogSeriesInitialValues(ZZ.zero(), ini, self.dop)
        #         for ini in inis]
        if inis:
            assert inis[0].compatible(inis)
        self.inis = inis

        if not isinstance(evpts, EvaluationPoint_base):
            if isinstance(evpts, (list, tuple)):
                evpts = tuple(Sequence(evpts))
            evpts = EvaluationPoint(evpts)
        self.evpts = evpts

        if bwrec is None:
            bwrec = bw_shift_rec(dop)
        self.orig_bwrec = bwrec

        self.ctx = ctx

        self._init_final()

        # Dynamic data

        self.Intervals = None
        self.bwrec_nplus = None
        self.est = None
        self.jetpows = None
        self.jets = None
        self.n = None
        self._last_enclosure_update = -1
        self.mult = None
        self.log_prec = None
        self.radpow = None
        self.rec_add_log_prec = None
        self.shifted_bwrec = None
        self.sols = None
        self.tail_bound = None              # bound on the series parts
        self.current_error = None           # ~radius of output if we stop now

        # Experimental rounding error analysis stuff
        self.n0_squash = sys.maxsize
        self.rnd_den = None
        self.rnd_err = None
        self.rnd_loc = None
        self.rnd_maj = None

    def _init_final(self):
        r"""
        Initialize computed parameters that need no update.
        """
        self._IR = self.ctx.IR
        self.mult_dict = self.inis[0].mult_dict() if self.inis else None
        Scalars = self.dop.base_ring().base_ring()
        self.real = (self.evpts.is_real_or_symbolic
                     and all(ini.is_real(Scalars) for ini in self.inis))
        self.ordinary = (self.dop.leading_coefficient()[0] != 0)
        self.last_index_with_ini = max(
            chain(iter([self.dop.order()]),
                  (ini.last_index() for ini in self.inis)))
        # hack for recurrences of order zero
        self.precomp_len = max(1, self.orig_bwrec.order)
        self.start = int(self.dop.order()) if self.ordinary else 0

    def _intervals(self, bit_prec):
        if self.evpts.is_numeric and cy_classes()[0] is not CoefficientSequence:
            return ComplexBallField(bit_prec)
        elif self.real:
            return RealBallField(bit_prec)
        else:
            return ComplexBallField(bit_prec)

    # Main summation loop

    def sum(self, bit_prec, stride):
        if not self.inis:
            return # []
        self._init_sums(bit_prec)
        self._init_error_analysis()
        for self.n in count():
            self.mult = (self.mult_dict[self.n]
                         if self.n in self.mult_dict else 0)
            if self.n % stride == 0:
                if self._check_convergence(stride):
                    break
            self._next_term()
        rnd_err = self._error_analysis()
        self._report_stats(rnd_err)
        return # self.sols

    def _init_sums(self, bit_prec):

        self.Intervals = self._intervals(bit_prec)

        # XXX make it possible to force the use of a given rec?
        leftmost = self.inis[0].expo # XXX fragile
        if _use_inexact_recurrence(self.orig_bwrec, leftmost, bit_prec):
            self.shifted_bwrec = self.orig_bwrec.shift(
                                               leftmost.as_ball(self.Intervals))
        else:
            self.shifted_bwrec = self.orig_bwrec.shift_by_PolynomialRoot(
                                                                       leftmost)


        Jets, self.jets = self.evpts.jets(self.Intervals)
        self.jetpows = [Jets.one()]*len(self.jets)

        self.radpow = self._IR.one()
        self.tail_bound = self.current_error = self._IR(infinity)

        if self.evpts.is_numeric:
            CS, PS = cy_classes()
        else:
            CS, PS = CoefficientSequence, PartialSum

        self.sols = []
        for ini in self.inis:
            cseq = CS(self.Intervals, ini, self.shifted_bwrec.order, self.real)
            # FIXME the branch should be computed separately for each component
            # of the evaluation point, taking into account the orientation of
            # the step
            psums = [PS(cseq, Jets, self.evpts.jet_order,
                        self.evpts.approx(self.Intervals, i),
                        (self.evpts.is_numeric,), self._IR)
                    for i in range(len(self.evpts))]
            self.sols.append(MPartialSums(cseq, psums))

        self.log_prec = 1
        # The next terms of the sum may need a higher log-prec than the current
        # one.
        self.rec_add_log_prec = sum(
            len(v) for s, v in self.inis[0].shift.items()
                   if self.start <= s < self.start + self.precomp_len)
        assert self.rec_add_log_prec == 0 or not self.ordinary
        self.bwrec_nplus = collections.deque(
                (self.shifted_bwrec.eval_series(self.Intervals, self.start + i,
                                self.log_prec + self.rec_add_log_prec)
                    for i in range(self.precomp_len)),
                maxlen=self.precomp_len)

    def _next_term(self):

        if self.n < self.start:
            assert self.ordinary
            for (cseq, psums) in self.sols:
                cseq.next_term_ordinary_initial_part(self.n)
                for (jetpow, psum) in zip(self.jetpows, psums):
                    psum.next_term_ordinary_initial_part(jetpow)
        else:
            # seems faster than relying on __missing__()
            cst = - ~self.bwrec_nplus[0][0][self.mult]
            squash = (self.n >= self.n0_squash)
            if squash:
                rnd_shift, hom_maj_coeff_lb = next(self.rnd_den)
                assert self.n0_squash + rnd_shift == self.n
            for (cseq, psums) in self.sols:
                err = cseq.next_term(self.n, self.mult, self.bwrec_nplus[0],
                                     cst, squash)
                if squash:
                    # XXX lookup of IR and/or conversion is slow
                    self.rnd_loc = self.rnd_loc.max(self._IR(self.n*err)
                                                    /hom_maj_coeff_lb)
                    # normalize NaNs and infinities
                    if not self.rnd_loc.is_finite():
                        self.rnd_loc = self.rnd_loc.parent()('inf')
                for (jetpow, psum) in zip(self.jetpows, psums):
                    psum.next_term(jetpow, self.mult)
            if self.mult > 0:
                self.log_prec = max(1, max(cseq.log_prec
                                           for cseq, _ in self.sols))

            self.rec_add_log_prec += self.mult_dict[self.n + self.precomp_len]
            self.rec_add_log_prec -= self.mult
            self.bwrec_nplus.append(
                self.shifted_bwrec.eval_series(
                    self.Intervals,
                    self.n + self.precomp_len,
                    self.log_prec + self.rec_add_log_prec))

        for i in range(len(self.jetpows)):
            self.jetpows[i] = self.jetpows[i]._mul_trunc_(self.jets[i],
                                                          self.evpts.jet_order)
        self.radpow *= self.evpts.rad

    def _check_convergence(self, stride):
        raise NotImplementedError

    def _report_stats(self, rnd_err):
        width = None
        if self.evpts.is_numeric:
            width = max(psum.total_error
                        for _, psums in self.sols
                        for psum in psums)
        logger.info("summed %d terms, tails = %s (est = %s), rnd_err <= %s, "
                    "interval width <= %s",
                    self.n, self.tail_bound, self._IR(self.est), rnd_err, width)

    # Data extraction

    def update_enclosures(self, err):
        r"""
        Update the “full” (= including singular factors and the error bound on
        the series part passed on input) solutions as well as a bound on the
        current maximum error on the “full” solution.
        """
        self.current_error = self._IR.zero()
        for _, psums in self.sols:
            for psum in psums:
                psum.update_enclosure(err)
                self.current_error = self.current_error.max(psum.total_error)
        self._last_enclosure_update = self.n

    def values_single_seq(self):
        assert len(self.inis) == 1
        assert self._last_enclosure_update == self.n
        [(_, psums)] = self.sols
        for psum in psums:
            psum.update_downshifts((0,))
        values = tuple(psum.downshifts[0] for psum in psums)
        return values

    def value(self):
        assert len(self.evpts) == 1
        assert self._last_enclosure_update == self.n
        return self.values_single_seq()[0]

    # Experimental rounding error analysis stuff
    # (Requires self.rnd_maj to be set. Only RecUnroller_tail_bound does set it at the
    # moment, but in principle, it could be decoupled from tail bounds.)

    def _init_error_analysis(self):
        if self.n0_squash == sys.maxsize:
            return
        assert self.ordinary
        # the special path does not squash its result
        assert self.start <= self.n0_squash
        self.rnd_den = self.rnd_maj.exp_part_coeffs_lbounds()
        self.rnd_loc = self._IR.zero()

    def _error_analysis(self):
        if self.n0_squash == sys.maxsize:
            return self._IR.zero()
        rnd_fac = self.rnd_maj.bound(self.evpts.rad, rows=self.evpts.jet_order)
        rnd_fac /= self.n0_squash
        rnd_err = self.rnd_loc*rnd_fac
        self.update_enclosures(self.tail_bound + rnd_err)
        return rnd_err

class RecUnroller_partial_sum(RecUnroller):
    r"""
    Compute a partial sum of a series by naive unrolling of a recurrence.

    Halt after a given number of terms.
    """

    def __init__(self, dop, inis, evpts, bwrec, terms, *, ctx=dctx):
        super().__init__(dop, inis, evpts, bwrec, ctx=ctx)
        self.__terms = terms

    def _check_convergence(self, stride):
        if self.n >= self.__terms :
            assert self.n == self.__terms or self.__terms < 0, "overshot"
            return True
        return False

class RecUnroller_tail_bound(RecUnroller):
    r"""
    Compute the sum of a series by naive unrolling of a recurrence.

    The result is an enclosure of the value of the sum, i.e., it includes a
    bound on the tail.
    """

    # XXX clean up / make interface more consistent with RecUnroller_partial_sum

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._stop = None

    def sum_auto(self, eps, maj, effort, fail_fast, stride=None):

        self._stop = accuracy.StopOnRigorousBound(maj, eps)

        input_accuracy = utilities.input_accuracy(self.dop, self.evpts,
                                                  self.inis)

        if stride is None:
            stride = min(max(50, 2*self.orig_bwrec.order),
                         max(2, input_accuracy))

        bit_prec, self.n0_squash = self._choose_working_precision(eps, maj,
                                                      effort, fail_fast, stride)
        max_prec = bit_prec + 2*input_accuracy

        if self.n0_squash < sys.maxsize:
            self.rnd_maj = maj(self.n0_squash)
            self.rnd_maj >>= self.n0_squash
            # |ind(n)| = cst·|monic_ind(n)|
            self.rnd_maj *= abs(self.ctx.IC(maj.dop.leading_coefficient()[0]))

        for attempt in count(1):
            logger.debug("attempt #%d (of max %d)", attempt, effort + 1)

            ini_are_accurate = 2*input_accuracy > bit_prec
            # Strictly decrease eps every time to avoid situations where doit
            # would be happy with the result and stop at the same point despite
            # the higher bit_prec. Since attempt starts at 1, we have a bit of
            # room for round-off errors.
            self._stop.reset(eps >> (4*attempt),
                              self._stop.fast_fail and ini_are_accurate)

            try:
                self.sum(bit_prec, stride)
            except accuracy.PrecisionError:
                if attempt > effort:
                    raise
            else:
                logger.debug("bit_prec = %s, err = %s (tgt = %s)", bit_prec,
                            self.current_error, eps)
                if all(psum.total_error < eps
                       for _, psums in self.sols for psum in psums):
                    return # self.sols

            # if interval squashing didn't give accurate results, switch back to
            # the classical method
            self.n0_squash = sys.maxsize

            bit_prec *= 2
            if attempt <= effort and bit_prec < max_prec:
                logger.info("lost too much precision, restarting with %d bits",
                            bit_prec)
                continue
            if fail_fast:
                raise accuracy.PrecisionError
            else:
                logger.info("lost too much precision, giving up")
                return # self.sols

    def _check_convergence(self, stride):
        if self.n <= self.last_index_with_ini or self.mult > 0:
            # currently not implemented by error bounds (but could be supported
            # in principle)
            return False
        assert self.log_prec == 1 or not self.ordinary
        self.est = sum(cseq.coeff_estimate() for cseq, _ in self.sols)
        self.est *= self.Intervals(self.radpow).squash()
        done, self.tail_bound = self._stop.check(self, self.n, self.tail_bound,
                                                 self.est, stride)
        return done

    # BoundCallbacks interface

    def get_residuals(self, stop, n):
        # Since this is called _before_ computing the new term, the relevant
        # coefficients are given by last[:-1], not last[1:]

        assert all(cseq.nterms == self.n == n for cseq, _ in self.sols)
        return [stop.maj.normalized_residual(n, list(cseq.last)[:-1],
                                             self.bwrec_nplus)
                for cseq, _ in self.sols]

    def get_bound(self, stop, n, resid):
        if self.n <= self.last_index_with_ini or self.mult > 0:
            raise NotImplementedError
        # XXX consider maintaining separate tail bounds, and stopping the
        # summation of some series before the others
        maj = stop.maj.tail_majorant(n, resid)
        tb = maj.bound(self.evpts.rad, rows=self.evpts.jet_order)
        self.update_enclosures(tb)
        return self.current_error

    def _choose_working_precision(self, eps, maj, effort, fail_fast, stride):
        ordrec = self.orig_bwrec.order
        bit_prec0 = utilities.prec_from_eps(eps)
        old_bit_prec = 8 + bit_prec0*(1 + ZZ(ordrec - 2).nbits())
        if not(self.ctx.squash_intervals and self.ordinary):
            logger.info("initial working precision = %s bits", old_bit_prec)
            return old_bit_prec, sys.maxsize
        nterms, lg_mag = self.dop.est_terms(self.evpts, bit_prec0)
        nterms = (ordrec*self.dop.order() + nterms)*1.2 # be pragmatic
        nterms = ZZ((nterms//stride + 1)*stride)
        bit_prec0 += ZZ(self.dop._naive_height()).nbits()
        bit_prec0 += lg_mag + nterms.nbits()
        n0_squash, g = guard_bits(self.dop, maj, self.evpts,
                                  ordrec, nterms)
        # adding twice the computed number of guard bits seems to work better
        # in practice, but I don't really understand why
        bit_prec = bit_prec0 + 2*g
        logger.info("initial working precision = %s + %s = %s (naive = %s), "
                    "squashing intervals for n >= %s",
                    bit_prec0, 2*g, bit_prec, old_bit_prec, n0_squash)
        if fail_fast and bit_prec > 4*bit_prec0 and effort <= 1:
            raise accuracy.PrecisionError
        return bit_prec, n0_squash

def guard_bits(dop, maj, evpts, ordrec, nterms):
    r"""
    Helper for choosing a working precision.

    This is done under the assumption that the first terms of the coefficient
    sequence are computed in interval arithmetic, and then, starting from some
    cutoff index, we switch to something like floating-point arithmetic with a
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

    cst = abs(maj.IC(maj.dop.leading_coefficient()[0])) # ???

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
        rnd_maj *= cst
        est_lg_rnd_fac = rnd_maj.bound(evpts.rad, rows=orddeq).log(2)
        est_lg_rnd_err = 2*maj.IR(ordrec + 1).log(2)
        if not est_lg_rnd_fac < maj.IR.zero():
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

def _use_inexact_recurrence(bwrec, leftmost, prec):
    Scalars = bwrec.Scalars
    if not isinstance(Scalars, number_field_base.NumberField):
        return False
    if ((Scalars is QQ or utilities.is_QQi(Scalars))
            and leftmost.is_rational()
            and bwrec[-1][0][0].numerator().nbits() < 10*prec):
        return False
    if prec <= 4000:
        return True
    h = max(a.numerator().nbits() for p in bwrec.coeff[::3]
                                  for i in range(0, p.degree(), 10)
                                  for a in p[i])
    deg = Scalars.degree()*leftmost.pol.degree()
    prefer_inexact = ( 4*(h + 16)*deg**2 + 4000 >= prec )
    logger.debug("using %sexact version of recurrence with algebraic coeffs "
            "of degree %s", "in" if prefer_inexact else "", Scalars.degree())
    return prefer_inexact

def series_sum(dop, ini, evpts, tgt_error, *, maj=None, bwrec=None,
               fail_fast=False, effort=2, stride=None, **kwds):
    r"""
    Sum a (generalized) series solution of dop.

    This is a semi-deprecated, somewhat more user-friendly wrapper to the series
    summation routines, mainly for testing purposes. The analytic continuation
    code typically calls lower level pieces directly.

    Note that this functions returns a tuple of values when given multiple
    evaluation points, but a bare value (instead of a tuple of length one) for a
    single point, regardless how the points were specified.

    EXAMPLES::

        sage: from sage.rings.real_arb import RealBallField, RBF
        sage: from sage.rings.complex_arb import ComplexBallField, CBF
        sage: QQi.<i> = QuadraticField(-1)

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.naive_sum import series_sum
        sage: from ore_algebra.analytic.path import EvaluationPoint
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

        sage: series_sum(dop, ini, 1/2, RBF(1e-30))
        ([-3.5751407034...] + [-2.2884877202...]*I)

    In normal usage ``evpts`` should be an object coercible to a complex ball, a
    tuple of such objects, or an :class:`EvaluationPoint` that wraps such a
    tuple. In addition, there is some support for ``EvaluationPoints`` wrapping
    identity polynomials. Other cases might work by accident. ::

        sage: series_sum(Dx - 1, [RBF(1)],
        ....:         EvaluationPoint(x, jet_order=2, rad=RBF(1)),
        ....:         1e-3, stride=1)
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
        ([+/- ...])

        sage: series_sum(Dx-1, [0], 2, 1e-50, stride=1)
        (0)

        sage: series_sum(Dx-1, [1], [1, CBF(i*pi)], 1e-15)
        (([2.7182818284590...] + [+/- ...]*I),
         ([-1.000000000000...] + [+/- ...]*I))

    Test that automatic precision increases do something reasonable::

        sage: import logging; logging.basicConfig()
        sage: logger = logging.getLogger('ore_algebra.analytic.naive_sum')
        sage: logger.setLevel(logging.INFO)

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, 1/3], 5/7, 1e-16, effort=100)
        INFO:...
        ([0.20674982866094049...])

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], 5/7, 1e-16, effort=100)
        INFO:...
        ([0.206749828660940...])

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], RBF(5/7), 1e-12, effort=100)
        INFO:...
        ([0.2067498286609...])

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], RBF(5/7), 1e-20, effort=100)
        INFO:...
        INFO:ore_algebra.analytic.naive_sum:lost too much precision, giving up
        ([0.20674982866094...])

        sage: xx = EvaluationPoint(x, jet_order=2, rad=RBF(1/4))
        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, 1/3], xx, 1e-30)[0](1/6)
        INFO:...
        [0.05504955913820894609304276321...]

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], xx, 1e-16)[0](1/6)
        INFO:...
        [0.055049559138208...]

        sage: series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [0, RBF(1/3)], xx, 1e-30)[0](1/6)
        INFO:...
        INFO:ore_algebra.analytic.naive_sum:lost too much precision, giving up
        [0.055049559138208...]

        sage: logger.setLevel(logging.WARNING)
    """

    ctx = Context(**kwds)

    dop = DifferentialOperator(dop)
    if not isinstance(ini, LogSeriesInitialValues):
        # single set of initial values, given as a list
        ini = LogSeriesInitialValues(ZZ.zero(), ini, dop)
    if maj is None:
        special_shifts = [(s, len(v)) for s, v in ini.shift.items()]
        maj = bounds.DiffOpBound(dop, ini.expo, special_shifts, ctx=ctx)
    tgt_error = ctx.IR(tgt_error)

    unr = RecUnroller_tail_bound(dop, [ini], evpts, bwrec, ctx=ctx)
    unr.sum_auto(tgt_error, maj, effort, fail_fast, stride)
    values = unr.values_single_seq()
    if len(unr.evpts) == 1:
        return values[0]
    else:
        return values

################################################################################
# Transition matrices
################################################################################

class HighestSolMapper_tail_bound(HighestSolMapper):

    def __init__(self, dop, evpts, eps, fail_fast, effort, *, ctx):
        super().__init__(dop, evpts, ctx=ctx)
        self.eps = eps
        self.fail_fast = fail_fast
        self.effort = effort

    def do_sum(self, inis):
        maj = bounds.DiffOpBound(self.dop, self.leftmost,
                        special_shifts=(None if self.ordinary else self.shifts),
                        bound_inverse="solve",
                        pol_part_len=(4 if self.ordinary else None),
                        ind_roots=self.all_roots,
                        ctx=self.ctx)
        unr = RecUnroller_tail_bound(self.dop, inis, self.evpts, self.bwrec,
                                     ctx=self.ctx)
        unr.sum_auto(self.eps, maj, self.effort, self.fail_fast)
        assert unr._last_enclosure_update == unr.n
        return unr.sols

def fundamental_matrix_regular(dop, evpts, eps, fail_fast, effort, ctx=dctx):
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
        [
        [[1.317902...] [1.000000...]]
        [[2.718281...]     [+/- ...]]
        ]

        sage: dop = DifferentialOperator(
        ....:         (x+1)*(x^2+1)*Dx^3-(x-1)*(x^2-3)*Dx^2-2*(x^2+2*x-1)*Dx)
        sage: fundamental_matrix_regular(dop, EP(1/3, 3), RBF(1e-10), False, 2)
        [
        [ [1.000000...]  [0.321750554...]  [0.147723741...]]
        [     [+/- ...]  [0.900000000...]  [0.991224850...]]
        [     [+/- ...]  [-0.27000000...]  [1.935612425...]]
        ]

        sage: dop = DifferentialOperator(
        ....:     (2*x^6 - x^5 - 3*x^4 - x^3 + x^2)*Dx^4
        ....:     + (-2*x^6 + 5*x^5 - 11*x^3 - 6*x^2 + 6*x)*Dx^3
        ....:     + (2*x^6 - 3*x^5 - 6*x^4 + 7*x^3 + 8*x^2 - 6*x + 6)*Dx^2
        ....:     + (-2*x^6 + 3*x^5 + 5*x^4 - 2*x^3 - 9*x^2 + 9*x)*Dx)
        sage: fundamental_matrix_regular(dop, EP(RBF(1/3), 4), RBF(1e-10), False, 2)
        [
        [ [3.1788470...] [-1.064032...]  [1.000...] [0.3287250...]]
        [ [-8.981931...] [3.2281834...]    [+/-...] [0.9586537...]]
        [  [26.18828...] [-4.063756...]    [+/-...] [-0.123080...]]
        [ [-80.24671...]  [9.190740...]    [+/-...] [-0.119259...]]
        ]

        sage: dop = x*Dx^3 + 2*Dx^2 + x*Dx
        sage: ini = [1, CBF(euler_gamma), 0]
        sage: dop.numerical_solution(ini, [0, RBF(1/3)], 1e-14)
        [-0.549046117782...]
    """
    eps_col = ctx.IR(eps)/ctx.IR(dop.order()).sqrt()
    hsm = HighestSolMapper_tail_bound(dop, evpts, eps_col, fail_fast, effort,
                                      ctx=ctx)
    cols = hsm.run()
    mats = [matrix([sol.value[i] for sol in cols]).transpose()
            for i in range(len(evpts))]
    return mats

class HighestSolMapper_partial_sums(HighestSolMapper):

    def __init__(self, dop, evpts, trunc_index, bit_prec, *,
                 inclusive, ctx):
        super().__init__(dop, evpts, ctx=ctx)
        if ctx.squash_intervals:
            raise NotImplementedError
        self.trunc_index = trunc_index
        self.inclusive = inclusive
        self.bit_prec = bit_prec

    def do_sum(self, inis):
        terms = self.trunc_index - self.leftmost.as_algebraic().real()
        if self.inclusive:
            terms = terms.floor() + 1
        else:
            terms = terms.ceil()
        unr = RecUnroller_partial_sum(self.dop, inis, self.evpts, self.bwrec,
                                      terms=int(terms), ctx=self.ctx)
        unr.sum(self.bit_prec, stride=1)
        unr.update_enclosures(self.ctx.IR.zero())
        return unr.sols

def fundamental_matrix_regular_truncated(dop, evpts, trunc_index, bit_prec,
                                         inclusive=False, ctx=dctx):
    r"""
    Compute the values at the points given in `evpts` of the canonical basis of
    solutions of ``dop``, all truncated at the same absolute order
    ``trunc_index``, along with derivatives of the truncated series.

    The output is organized in a matrix with derivatives renormalized in the
    usual way.

    TESTS::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.naive_sum import *
        sage: from ore_algebra.analytic.differential_operator import DifferentialOperator
        sage: from ore_algebra.analytic.path import EvaluationPoint as EP
        sage: Dops, x, Dx = DifferentialOperators()

        sage: [fundamental_matrix_regular_truncated(Dx-1, EP(1), k, 30)[0] for k in range(4)]
        [[0], [1.00000000], [2.00000000], [2.50000000]]

        sage: [fundamental_matrix_regular_truncated(Dx-1, EP(1), k, 30, inclusive=True)[0] for k in range(4)]
        [[1.00000000], [2.00000000], [2.50000000], [[2.66666666 +/- 8.15e-9]]]

        sage: fundamental_matrix_regular_truncated((x*Dx-3)^2, EP(1/2, 2), 3, 30)[0]
        [0 0]
        [0 0]
        sage: a = RBF(1/2)
        sage: mat = fundamental_matrix_regular_truncated((x*Dx-3)^2, EP(a, 2), 4, 30)[0]
        sage: [ref in res for res, ref in zip(mat.list(), [a^3*log(a), a^3, 3*a^2*log(a)+a^2, 3*a^2])]
        [True, True, True, True]

        sage: fundamental_matrix_regular_truncated(((x*Dx)^2+1).lclm(Dx-1), EP(a, 3), 0, 30)[0]
        [0 0 0]
        [0 0 0]
        [0 0 0]
        sage: fundamental_matrix_regular_truncated(((x*Dx)^2+1).lclm(Dx-1), EP(a, 1), 1, 30)[0]
        [ [0.7692...] + [0.6389...]*I [0.7692...] + [-0.6389...]*I  1.000...]

        sage: [fundamental_matrix_regular_truncated(
        ....:         ((x*Dx)^2-2).lclm(Dx-1), EP(a, 1), k, 30)[0]
        ....:  for k in range(-2, 3)]
        [[0                   0            0],
        [[2.665...]           0            0],
        [[2.665...]           0            0],
        [[2.665...]    1.000...            0],
        [ [2.665...]   1.500...   [0.375...]]]

        sage: [fundamental_matrix_regular_truncated(
        ....:         ((x*Dx)^2-2).lclm(Dx-1), EP(a, 1), k, 30, inclusive=True)[0]
        ....:  for k in range(-2, 3)]
        [[0                   0            0],
        [[2.665...]           0            0],
        [[2.665...]    1.000...            0],
        [[2.665...]    1.500...            0],
        [ [2.665...]   1.625...   [0.375...]]]
    """
    ctx = Context(ctx)
    ctx.squash_intervals = False
    dop = DifferentialOperator(dop)
    hsm = HighestSolMapper_partial_sums(dop, evpts, trunc_index, bit_prec,
                                        inclusive=inclusive, ctx=ctx)
    cols = hsm.run()
    mats = [matrix([sol.value[i] for sol in cols]).transpose()
            for i in range(len(evpts))]
    return mats

################################################################################
# Bound recording
################################################################################

BoundRecord = collections.namedtuple("BoundRecord", ["n", "psum", "maj", "b"])

class BoundRecorder(RecUnroller_tail_bound):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.recorded_bounds = None

    def _init_sums(self, *args, **kwds):
        super()._init_sums(*args, **kwds)
        self.recorded_bounds = []

    def _check_convergence(self, stride):
        assert len(self.sols) == 1
        if self.n <= self.last_index_with_ini or self.mult > 0:
            bound = self._stop.maj.IR('inf')
            maj = None
        else:
            resid = self.get_residuals(self._stop, self.n)
            bound = self.get_bound(self._stop, self.n, resid)
            maj = self._stop.maj.tail_majorant(self.n, resid)
        val = self.sols[0][1][0].bare_value()
        self.recorded_bounds.append(BoundRecord(self.n, val, maj, bound))
        # Call the standard _check_convergence(), but ignore its verdict, and
        # only self._stop if we really have obtained a bound < eps.
        super()._check_convergence(stride)
        return (self.tail_bound < self._stop.eps)

################################################################################
# Utilities
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

def _resize_list(l, n, z):
    n0 = len(l)
    if n > n0:
        l.extend(repeat(z, n - n0))
    elif n < n0:
        l[n:] = []
