# -*- coding: utf-8 - vim: tw=80
r"""
Accuracy management
"""

import logging

from sage.rings.all import QQ

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
        self.get_residuals = get_residuals
        self.get_bound = get_bound
        self.fast_fail = fast_fail
        self.force = force
        self.prev_est = IR('inf')

    def check(self, n, ini_tb, est, width, next_stride):

        eps = self.eps
        prev_est = self.prev_est
        self.prev_est = est

        intervals_blowing_up = safe_le(self.eps >> 2, width)
        if intervals_blowing_up:
            if self.fast_fail:
                logger.debug("n=%d, est=%s, width=%s", n, est, width)
                raise PrecisionError
            if safe_le(self.eps, width):
                # Aim for tail_bound < width instead of tail_bound < self.eps
                eps = width

        if safe_lt(eps, est) and safe_lt(eps, prev_est) and not self.force:
            # Only do this for est < prev_est to avoid getting into an infinite
            # loop when est increases indefinitely (typically due to interval
            # blow-up)
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
            elif ini_tb.is_finite() and not safe_le(tb, ini_tb.above_abs()):
                # The bounds are out of control, stop asap
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
