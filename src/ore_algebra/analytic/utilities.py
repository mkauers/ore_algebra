# -*- coding: utf-8 - vim: tw=80
"""
Utilities

(some of which could perhaps be upstreamed at some point)
"""

import logging

from sage.misc.cachefunc import cached_function
from sage.misc.misc import cputime
from sage.rings.all import QQbar, CIF
from sage.structure.element import parent

logger = logging.getLogger(__name__)

######################################################################
# Timing
######################################################################

class Clock(object):
    def __init__(self, name="time"):
        self.name = name
        self._sum = 0.
        self._tic = None
    def __repr__(self):
        return "{} = {}".format(self.name, self.total())
    def since_tic(self):
        return 0. if self._tic is None else cputime(self._tic)
    def total(self):
        return self._sum + self.since_tic()
    def tic(self, t=None):
        assert self._tic is None
        self._tic = cputime() if t is None else t
    def toc(self):
        self._sum += cputime(self._tic)
        self._tic = None

class Stats(object):
    def __init__(self):
        self.time_total = Clock("total")
        self.time_total.tic()
    def __repr__(self):
        return ", ".join(str(clock) for clock in self.__dict__.values()
                                    if isinstance(clock, Clock))

######################################################################
# Safe comparisons
######################################################################

def _check_parents(a, b):
    # comparison between different parent may be okay if the elements are
    # instances of the same class (e.g., balls with different precisions)
    if parent(a) is not parent(b) and type(a) is not type(b):
        raise TypeError("unsafe comparison", parent(a), parent(b))

def safe_lt(a, b):
    _check_parents(a, b)
    return a < b

def safe_le(a, b):
    _check_parents(a, b)
    return a <= b

def safe_gt(a, b):
    _check_parents(a, b)
    return a > b

def safe_ge(a, b):
    _check_parents(a, b)
    return a >= b

def safe_eq(a, b):
    if parent(a) is not parent(b):
        logger.debug("comparing elements of %s and %s",
                     parent(a), parent(b))
        return False
    else:
        return a == b

def safe_ne(a, b):
    if parent(a) is not parent(b):
        logger.debug("comparing elements of %s and %s",
                     parent(a), parent(b))
        return True
    else:
        return a != b

######################################################################
# Differential operators
######################################################################

# These functions should probably become methods of suitable subclasses of
# OreOperator, or of a custom wrapper.

@cached_function
def dop_singularities(dop, dom=QQbar):
    return [descr[0] for descr in dop.leading_coefficient().roots(dom)]

def sing_as_alg(dop, iv):
    pol = dop.leading_coefficient().radical()
    return QQbar.polynomial_root(pol, CIF(iv))

######################################################################
# Miscellaneous stuff
######################################################################

def is_interval_field(parent):
    from sage.rings.real_mpfi import is_RealIntervalField
    from sage.rings.complex_interval_field import is_ComplexIntervalField
    # TODO: arb
    return is_RealIntervalField(parent) or is_ComplexIntervalField(parent)

def prec_from_eps(eps):
    return -eps.lower().log2().floor()
