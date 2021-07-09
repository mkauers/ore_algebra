# -*- coding: utf-8 - vim: tw=80
"""
Miscellaneous utilities
"""

# Copyright 2015, 2016, 2017, 2018 Marc Mezzarobba
# Copyright 2015, 2016, 2017, 2018 Centre national de la recherche scientifique
# Copyright 2015, 2016, 2017, 2018 Université Pierre et Marie Curie
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

import itertools
from builtins import zip

import sage.rings.complex_arb
import sage.rings.real_arb

from sage.categories.pushout import pushout
from sage.misc.cachefunc import cached_function
from sage.misc.misc import cputime
from sage.rings.all import ZZ, QQ, QQbar, CIF, CBF
from sage.rings.number_field.number_field import (NumberField,
        NumberField_quadratic, is_NumberField)
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import number_field_elements_from_algebraics

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
                and list(parent.polynomial()) == [1,0,1]
                and CBF(parent.gen()).imag().is_one())

def ball_field(eps, real):
    prec = prec_from_eps(eps)
    if real:
        return sage.rings.real_arb.RealBallField(prec)
    else:
        return sage.rings.complex_arb.ComplexBallField(prec)

################################################################################
# Number fields and orders
################################################################################

def good_number_field(nf):
    if isinstance(nf, NumberField_quadratic):
        # avoid denominators in the representation of elements
        disc = nf.discriminant()
        if disc % 4 == 1:
            nf, _, hom = nf.change_generator(nf(nf.discriminant()).sqrt())
            return nf, hom
    return nf, nf.hom(nf)

def as_embedded_number_field_elements(algs):
    try:
        nf, elts, _ = number_field_elements_from_algebraics(algs, embedded=True)
    except NotImplementedError: # compatibility with Sage <= 9.3
        nf, elts, emb = number_field_elements_from_algebraics(algs)
        if nf is not QQ:
            nf = NumberField(nf.polynomial(), nf.variable_name(),
                        embedding=emb(nf.gen()))
            elts = [elt.polynomial()(nf.gen()) for elt in elts]
            nf, hom = good_number_field(nf)
            elts = [hom(elt) for elt in elts]
        assert [QQbar.coerce(elt) == alg for alg, elt in zip(algs, elts)]
    return nf, elts

def as_embedded_number_field_element(alg):
    return as_embedded_number_field_elements([alg])[1][0]

def number_field_with_integer_gen(K):
    r"""
    TESTS::

        sage: from ore_algebra.analytic.utilities import number_field_with_integer_gen
        sage: K = NumberField(6*x^2 + (2/3)*x - 9/17, 'a')
        sage: number_field_with_integer_gen(K)[0]
        Number Field in x306a with defining polynomial x^2 + 34*x - 8262 ...
    """
    if K is QQ:
        return QQ, ZZ
    den = K.polynomial().monic().denominator()
    if den.is_one():
        # Ensure that we return the same number field object (coercions can be
        # slow!)
        intNF = K
    else:
        intgen = K.gen() * den
        ### Attempt to work around various problems with embeddings
        emb = K.coerce_embedding()
        embgen = emb(intgen) if emb else intgen
        # Write K.gen() = α = β/q where q = den, and
        # K.polynomial() = q + p[d-1]·X^(d-1) + ··· + p[0].
        # By clearing denominators in P(β/q) = 0, one gets
        # β^d + q·p[d-1]·β^(d-1) + ··· + p[0]·q^(d-1) = 0.
        intNF = NumberField(intgen.minpoly(), "x" + str(den) + str(K.gen()),
                            embedding=embgen)
        assert intNF != K
    intNF, _ = good_number_field(intNF)
    # Work around weaknesses in coercions involving order elements,
    # including #14982 (fixed). Used to trigger #14989 (fixed).
    #return intNF, intNF.order(intNF.gen())
    return intNF, intNF

def invert_order_element(alg):
    if alg in ZZ:
        return 1, alg
    else:
        Order = alg.parent()
        pol = alg.polynomial().change_ring(ZZ)
        modulus = Order.gen(1).minpoly()
        den, num, _ = pol.xgcd(modulus)  # hopefully fraction-free!
        return Order(num), ZZ(den)

def mypushout(X, Y):
    if X.has_coerce_map_from(Y):
        return X
    elif Y.has_coerce_map_from(X):
        return Y
    else:
        Z = pushout(X, Y)
        Z, _ = good_number_field(Z)
        assert (is_NumberField(Z) if is_NumberField(X) and is_NumberField(Y)
                                  else True)
        return Z

######################################################################
# Sage features
######################################################################

@cached_function
def has_new_ComplexBall_constructor():
    from sage.rings.complex_arb import ComplexBall, CBF
    try:
        ComplexBall(CBF, QQ(1), QQ(1))
    except TypeError:
        return False
    else:
        return True

######################################################################
# Miscellaneous stuff
######################################################################

def prec_from_eps(eps):
    return -eps.lower().log2().floor() + 4

def split(cond, objs):
    matching, not_matching = [], []
    for x in objs:
        (matching if cond(x) else not_matching).append(x)
    return matching, not_matching

def short_str(obj, n=60):
    s = str(obj)
    if len(s) < n:
        return s
    else:
        return s[:n/2-2] + "..." + s[-n/2 + 2:]

# Adapted from itertools manual
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
