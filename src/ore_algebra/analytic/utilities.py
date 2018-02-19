# -*- coding: utf-8 - vim: tw=80
"""
Miscellaneous utilities
"""

import sage.rings.complex_arb
import sage.rings.real_arb

from sage.misc.cachefunc import cached_function
from sage.misc.misc import cputime
from sage.rings.all import QQ, QQbar, CIF
from sage.rings.number_field.number_field import NumberField_quadratic
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

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
    def __repr__(self):
        return ", ".join(str(clock) for clock in self.__dict__.values()
                                    if isinstance(clock, Clock))

######################################################################
# Numeric fields
######################################################################

_RBFmin = sage.rings.real_arb.RealBallField(2)
_CBFmin = sage.rings.complex_arb.ComplexBallField(2)

def is_numeric_parent(parent):
    return _CBFmin.has_coerce_map_from(parent)

def is_real_parent(parent):
    return _RBFmin.has_coerce_map_from(parent)

def is_QQi(parent):
    return (isinstance(parent, NumberField_quadratic)
                and list(parent.polynomial()) == [1,0,1])

######################################################################
# Miscellaneous stuff
######################################################################

def prec_from_eps(eps):
    return -eps.lower().log2().floor() + 4

def ball_field(eps, real):
    prec = prec_from_eps(eps)
    if real:
        return sage.rings.real_arb.RealBallField(prec)
    else:
        return sage.rings.complex_arb.ComplexBallField(prec)

def split(cond, objs):
    matching, not_matching = [], []
    for x in objs:
        (matching if cond(x) else not_matching).append(x)
    return matching, not_matching

def as_embedded_number_field_element(alg):
    from sage.rings.number_field.number_field import NumberField
    nf, elt, emb = alg.as_number_field_element()
    if nf is QQ:
        res = elt
    else:
        embnf = NumberField(nf.polynomial(), nf.variable_name(),
                    embedding=emb(nf.gen()))
        res = elt.polynomial()(embnf.gen())
    # assert QQbar.coerce(res) == alg
    return res

def short_str(obj, n=60):
    s = str(obj)
    if len(s) < n:
        return s
    else:
        return s[:n/2-2] + "..." + s[-n/2 + 2:]
