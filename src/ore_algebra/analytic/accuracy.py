# -*- coding: utf-8 - vim: tw=80
r"""
Accuracy management
"""

import logging

from .bounds import IR
from .safe_cmp import *

logger = logging.getLogger(__name__)

######################################################################
# Absolute and relative errors
######################################################################

class StoppingCriterion(object):
    pass

class AbsoluteError(StoppingCriterion):

    def __init__(self, eps):
        self.eps = IR(eps)

    def reached(self, err, abs_val=None):
        return safe_lt(err.abs(), self.eps)

    def __repr__(self):
        return str(self.eps.lower()) + " (absolute)"

    def __rshift__(self, n):
        return AbsoluteError(self.eps >> n)

class RelativeError(StoppingCriterion):

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
