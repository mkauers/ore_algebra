# -*- coding: utf-8 - vim: tw=80
r"""
Accuracy management
"""

import logging

from ore_algebra.analytic.bounds import IR
from ore_algebra.analytic.safe_cmp import *

logger = logging.getLogger(__name__)

######################################################################
# Absolute and relative errors
######################################################################

class PrecisionError(Exception):
    pass

class StoppingCriterion(object):
    pass

class AbsoluteError(StoppingCriterion):

    def __init__(self, eps, precise=False):
        self.eps = IR(eps)
        self.precise = precise

    def reached(self, err, abs_val=None):
        if abs_val is not None and safe_gt(abs_val.rad_as_ball(), self.eps):
            if self.precise:
                raise PrecisionError
            else:
                # XXX: take logger from creator?
                logger.warn("interval too wide wrt target accuracy "
                            "(lost too much precision?)")
                return True
        return safe_lt(err.abs(), self.eps)

    def __repr__(self):
        return str(self.eps) + " (absolute)"

class RelativeError(StoppingCriterion):
    def __init__(self, eps, cutoff=None):
        self.eps = IR(eps)
        self.cutoff = eps if cutoff is None else IR(cutoff)
    def reached(self, err, abs_val):
        # NOTE: we could provide a slightly faster test when err is a
        # non-rigorous estimate (not a true tail bound)
        # XXX: raise PrecisionError if we can not conclude
        return (safe_le(err.abs(), self.eps*(abs_val - err))
                or safe_le(abs_val + err, self.cutoff))
    def __repr__(self):
        return str(self.eps) + " (relative)"

