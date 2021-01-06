# -*- coding: utf-8 - vim: tw=80
r"""
Accuracy management
"""

# Copyright 2015, 2016, 2017, 2018 Marc Mezzarobba
# Copyright 2015, 2016, 2017, 2018 Centre national de la recherche scientifique
# Copyright 2015, 2016, 2017, 2018 Université Pierre et Marie Curie
# Copyright 2019 Sorbonne Université
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

import collections, logging

from sage.rings.all import  ZZ, QQ, RR
from sage.rings.all import RealBallField, ComplexBallField

from .safe_cmp import *

logger = logging.getLogger(__name__)

IR = RealBallField()
IC = IR.complex_field()

######################################################################
# Convergence check
######################################################################

class PrecisionError(Exception):
    pass

class BoundCallbacks(object):
    r"""
    Used by StoppingCriterion.
    """

    def get_residuals(self):
        r"""
        Return the normalized residuals.

        The format is up to the client: the residuals will be passed as is to
        get_bound and get_maj.
        """
        raise NotImplementedError

    def get_bound(self, residuals):
        r"""
        Bound on the total error on the current partial sum (should return
        whatever quantity the client considers the “tail bound” and wants to
        make < ε, possibly accounting for logs etc.)
        """
        raise NotImplementedError

    def get_maj(self, stop, n, resid):
        r"""
        Return the majorant of the tail computed from the residuals.

        Optional: only required from clients that support bound recording.
        """
        return stop.maj.tail_majorant(n, resid)

    def get_value(self):
        r"""
        Return the current partial sum.

        The result should include logs etc. just like get_bound(), but *not*
        take into account the tail bound in the intervals.

        Optional: only required from clients that support bound recording.
        """
        raise NotImplementedError

class StoppingCriterion(object):
    r"""
    Condition for dynamically deciding where to truncate a series.

    Based on rigorous (absolute) error bounds, observed interval blow-up, and
    various heuristics to choose a course of action when, e.g., adding more
    terms will reduce the method error but possibly increase the round-off error
    or interval width.
    """

    def __init__(self, maj, eps, fast_fail=True):
        self.maj = maj
        self.eps = eps
        self.prec = ZZ(eps.log(2).lower().floor()) - 2
        self.fast_fail = fast_fail
        self.force = False

    def check(self, cb, sing, n, ini_tb, est, next_stride):
        r"""
        Test if it is time to halt the computation of the sum of a series.

        INPUT:

        - cb: BoundCallbacks object, see the documentation of that class;
        - sing: boolean, True signals a singular index where we should return
          infinity without even trying to compute a better bound (this is useful
          to avoid special-casing singular indices elsewhere);
        - n: current index;
        - ini_tb: previous tail bound (can be infinite);
        - est: real interval, heuristic estimate of the absolute value of the
          tail of the series, **whose lower bound must tend to zero or
          eventually become negative**, and whose width can be used to provide
          an indication of interval blow-up in the computation; typically
          something like abs(first nonzero neglected term);
        - next_stride: indication of how many additional terms the caller
          intends to sum before calling us again.

        OUTPUT:

        (done, bound) where done is a boolean indicating if it is time to stop
        the computation, and bound is a rigorous (possibly infinite) bound on
        the tail of the series.

        TESTS::

            sage: from ore_algebra import DifferentialOperators
            sage: from ore_algebra.analytic.bounds import DiffOpBound
            sage: from ore_algebra.analytic.naive_sum import series_sum
            sage: Dops, x, Dx = DifferentialOperators()
            sage: maj = DiffOpBound(Dx-1, max_effort=0)
            sage: series_sum(Dx-1, [1], 2, 1e-50, stride=1)
            ([7.3890560989306502272304274605750078131803155705...])
        """
        if sing:
            return False, IR('inf')

        eps = self.eps

        # XXX We shouldn't force the caller to give a full-precision estimate...
        accuracy = max(est.accuracy(), -int((est.rad() or RR(1)).log(2)))
        width = IR(est.rad())
        est = IR(est)
        intervals_blowing_up = (accuracy < self.prec or
                                safe_le(self.eps >> 2, width))
        if intervals_blowing_up:
            if self.fast_fail:
                logger.debug("n=%d, est=%s, width=%s", n, est, width)
                raise PrecisionError
            if safe_le(self.eps, width):
                # Aim for tail_bound < width instead of tail_bound < self.eps
                eps = width

        if safe_lt(eps, est) and accuracy >= self.prec and not self.force:
            # It is important to test the inequality with the *interval* est, to
            # avoid getting caught in an infinite loop when est increases
            # indefinitely due to interval blow-up. When however the lower bound
            # of est increases too, we are probably climbing a term hump (think
            # exp(1000)): it is better not to halt the computation yet, in the
            # hope of getting at least a reasonable relative error.
            logger.debug("n=%d, est=%s, width=%s", n, est, width)
            assert width.is_finite()
            return False, IR('inf')

        resid = cb.get_residuals()

        tb = IR('inf')
        while True:
            prev_tb = tb
            tb = cb.get_bound(resid)
            logger.debug("n=%d, est=%s, width=%s, tail_bound=%s",
                         n, est, width, tb)
            bound_getting_worse = ini_tb.is_finite() and not safe_lt(tb, ini_tb)
            if safe_lt(tb, eps):
                logger.debug("--> ok")
                return True, tb
            elif self.force and not self.maj.can_refine():
                break
            elif (prev_tb.is_finite() and not safe_le(tb, prev_tb >> 8)
                    or not self.maj.can_refine()):
                if bound_getting_worse:
                    # The bounds are out of control, stop asap.
                    # Subtle point: We could also end up here because of a hump.
                    # But then, typically, est > ε, so that we shouldn't even
                    # have entered the rigorous phase unless the intervals are
                    # blowing up badly.
                    # Note: it seems best *not* to do this when prev_tb is
                    # non-finite, because we may be waiting to get past a
                    # singularity of the recurrence. (XXX: Unfortunately, we
                    # have no way of deciding if the bound is infinite because
                    # of a genuine mathematical reason or an evaluation issue.)
                    logger.debug("--> bounds out of control ({} became {})"
                                .format(ini_tb, tb))
                    return True, tb
                else:
                    # Refining no longer seems to help: sum more terms
                    logger.debug("--> refining doesn't help")
                    break
            elif intervals_blowing_up or bound_getting_worse:
                # Adding more terms is likely to make the result worse and
                # worse, but we can try refining the majorant as much as
                # possible before giving up. (Note that unlike in the previous
                # branch, we do not stop asap if the bound is getting worse in
                # the present case.)
                logger.debug("--> intervals blowing up or bound getting worse")
                self.maj.refine()
            else:
                thr = tb*est**(QQ(next_stride*(self.maj.effort()**2 + 2))/(n+1))
                if safe_le(thr, eps):
                    # Try summing a few more terms before refining
                    logger.debug("--> above refinement threshold ({} <= {})"
                                 .format(thr, eps))
                    break
                else:
                    logger.debug("--> bad bound but refining may help")
                    self.maj.refine()
        logger.debug("--> ko")
        return False, tb

    def reset(self, eps, fast_fail):
        self.eps = eps
        self.fast_fail = fast_fail

######################################################################
# Bound recording
######################################################################

BoundRecord = collections.namedtuple("BoundRecord", ["n", "psum", "maj", "b"])

class BoundRecorder(StoppingCriterion):

    def __init__(self, maj, eps, fast_fail=False):
        super(self.__class__, self).__init__(maj, eps, fast_fail=True)
        self.force = True
        self.recd = []

    def check(self, cb, sing, n, *args):
        if sing:
            bound = IR('inf')
            maj = None
        else:
            resid = cb.get_residuals()
            bound = cb.get_bound(resid)
            maj = cb.get_maj(self, n, resid)
        self.recd.append(BoundRecord(n, cb.get_value(), maj, bound))
        return super(self.__class__, self).check(cb, sing, n, *args)

    def reset(self, *args):
        super(self.__class__, self).reset(*args)
        self.recd = []

######################################################################
# Absolute and relative errors
######################################################################

class AccuracyTest(object):
    r"""
    Accuracy test.

    Compared to StoppingCriterion (which is specifically for series summation
    loops), these classes are intended for higher-level algorithm that just want
    to check whether a certain result is satisfactory according to some accuracy
    measure.
    """
    pass

class AbsoluteError(AccuracyTest):

    def __init__(self, eps):
        self.eps = IR(eps)

    def reached(self, err, abs_val=None):
        return safe_lt(err.abs(), self.eps)

    def __repr__(self):
        return str(self.eps.lower()) + " (absolute)"

class RelativeError(AccuracyTest):

    def __init__(self, eps, cutoff=None):
        self.eps = IR(eps)
        self.cutoff = eps if cutoff is None else IR(cutoff)

    def reached(self, err, val):
        # NOTE: we could provide a slightly faster test when err is a
        # non-rigorous estimate (not a true tail bound)
        return (safe_le(abs(err), self.eps*(abs(val) - err))
                or safe_le(abs_val + err, self.cutoff))

    def __repr__(self):
        return str(self.eps.lower()) + " (relative)"
