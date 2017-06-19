# -*- coding: utf-8 - vim: tw=80
r"""
Accuracy management
"""

import logging

from sage.rings.all import QQ, ZZ, RR

from .bounds import IR
from .safe_cmp import *

logger = logging.getLogger(__name__)

class PrecisionError(Exception):
    pass

class StoppingCriterion(object):

    def __init__(self, maj, eps, get_residuals, get_bound,
                 fast_fail=False, force=False):
        self.maj = maj
        self.eps = eps
        self.prec = ZZ(eps.log(2).lower().floor()) - 2
        self.get_residuals = get_residuals
        self.get_bound = get_bound
        self.fast_fail = fast_fail
        self.force = force

    def check(self, n, ini_tb, est, next_stride):
        r"""
        Test if it is time to halt the computation of the sum of a series.

        INPUT:

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
        """

        eps = self.eps

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

        resid = self.get_residuals()

        tb = IR('inf')
        while True:
            prev_tb = tb
            tb = self.get_bound(resid)
            logger.debug("n=%d, est=%s, width=%s, tail_bound=%s",
                         n, est, width, tb)
            if safe_lt(tb, eps):
                return True, tb
            elif ini_tb.is_finite() and not safe_lt(tb, ini_tb):
                # The bounds are out of control, stop asap.
                # Subtle point: We could also end up here because of a hump. But
                # then, typically, est > ε, so that we shouldn't even have
                # entered the rigorous phase unless the intervals are blowing up
                # badly.
                return True, tb
            elif prev_tb.is_finite() and not safe_le(tb, prev_tb >> 8):
                # Refining no longer seems to help: sum more terms
                break
            elif intervals_blowing_up:
                self.maj.refine()
            else:
                thr = tb*est**(QQ(next_stride*(self.maj._effort**2 + 2))/n)
                if safe_le(thr, eps):
                    # Try summing a few more terms before refining
                    break
                else:
                    self.maj.refine()
        return False, tb


######################################################################
# Absolute and relative errors
######################################################################

class OldStoppingCriterion(object):
    pass

class AbsoluteError(OldStoppingCriterion):

    def __init__(self, eps):
        self.eps = IR(eps)

    def reached(self, err, abs_val=None):
        return safe_lt(err.abs(), self.eps)

    def __repr__(self):
        return str(self.eps.lower()) + " (absolute)"

    def __rshift__(self, n):
        return AbsoluteError(self.eps >> n)

class RelativeError(OldStoppingCriterion):

    def __init__(self, eps, cutoff=None):
        self.eps = IR(eps)
        self.cutoff = eps if cutoff is None else IR(cutoff)

    def reached(self, err, abs_val):
        # NOTE: we could provide a slightly faster test when err is a
        # non-rigorous estimate (not a true tail bound)
        return (safe_le(err.abs(), self.eps*(abs_val - err))
                or safe_le(abs_val + err, self.cutoff))

    def __repr__(self):
        return str(self.eps.lower()) + " (relative)"

    def __rshift__(self, n):
        return RelativeError(self.eps >> n, self.cutoff >> n)
