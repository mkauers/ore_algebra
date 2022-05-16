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
from sage.misc.cachefunc import cached_function, cached_method
from sage.misc.misc import cputime
from sage.rings.all import ZZ, QQ, QQbar, CIF, CBF
from sage.rings.complex_interval_field import ComplexIntervalField
from sage.rings.number_field.number_field import (NumberField,
        NumberField_quadratic, is_NumberField)
from sage.rings.number_field.number_field_element import NumberFieldElement
from sage.rings.number_field.number_field_element_quadratic import NumberFieldElement_quadratic
from sage.rings.polynomial.complex_roots import complex_roots
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import (qq_generator, AlgebraicGenerator, AlgebraicNumber,
                              ANExtensionElement, ANRoot)
from sage.rings.rational import Rational
from sage.structure.coerce_exceptions import CoercionException
from sage.structure.element import coercion_model
from sage.structure.sequence import Sequence

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


def qqbar_to_cbf(tgt, elt):
    return tgt(elt.interval_fast(ComplexIntervalField(tgt.precision())))

################################################################################
# Number fields and orders
################################################################################


def internal_denominator(a):
    r"""
    TESTS::

        sage: from ore_algebra import OreAlgebra
        sage: Dx = OreAlgebra(PolynomialRing(QQ, 'x'), 'Dx').gen()
        sage: from ore_algebra.analytic.utilities import internal_denominator
        sage: K.<a> = QuadraticField(1/27)
        sage: internal_denominator(a)
        9
        sage: (Dx - a).local_basis_expansions(0)
        [1 + a*x + 1/54*x^2 + 1/162*a*x^3]
    """
    if isinstance(a, NumberFieldElement_quadratic):
        # return the denominator in the internal representation based on √disc
        return a.__reduce__()[-1][-1]
    else:
        return a.denominator()


def as_embedded_number_field_elements(algs):
    # Adapted (in part) from sage's number_field_elements_from algebraics(),
    # because the latter loses too much time trying to detect if the numbers are
    # real.
    gen = qq_generator
    algs = [QQbar.coerce(a) for a in algs]
    for a in algs:
        a.simplify()
        gen = gen.union(a._exact_field())
    nf = gen._field
    if nf is not QQ:
        gen_emb = AlgebraicNumber(ANExtensionElement(gen, nf.gen()))
        nf = NumberField(nf.polynomial(), nf.variable_name(),
                         embedding=gen_emb)
        algs = [gen(a._exact_value()).polynomial()(nf.gen()) for a in algs]
    return nf, algs


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
        # Attempt to work around various problems with embeddings
        emb = K.coerce_embedding()
        embgen = emb(intgen) if emb else intgen
        # Write K.gen() = α = β/q where q = den, and
        # K.polynomial() = q + p[d-1]·X^(d-1) + ··· + p[0].
        # By clearing denominators in P(β/q) = 0, one gets
        # β^d + q·p[d-1]·β^(d-1) + ··· + p[0]·q^(d-1) = 0.
        intNF = NumberField(intgen.minpoly(), "x" + str(den) + str(K.gen()),
                            embedding=embgen)
        assert intNF != K
    # Work around weaknesses in coercions involving order elements,
    # including #14982 (fixed). Used to trigger #14989 (fixed).
    # return intNF, intNF.order(intNF.gen())
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
        if is_NumberField(X) and is_NumberField(Y) and not is_NumberField(Z):
            # we likely obtained a parent where both number fields have a
            # canonical embedding, typically QQbar...
            raise CoercionException
        return Z


def extend_scalars(Scalars, *pts):
    gen = Scalars.gen()
    try:
        # Largely redundant with the other branch, but may do a better job
        # in some cases, e.g. pushout(QQ, QQ(α)), where as_enf_elts() would
        # invent new generator names.
        NF = coercion_model.common_parent(Scalars, *pts)
        if not is_NumberField(NF):
            raise CoercionException
        gen1 = NF.coerce(gen)
        pts1 = tuple(NF.coerce(pt) for pt in pts)
    except (CoercionException, TypeError):
        NF, val1 = as_embedded_number_field_elements((gen,)+pts)
        gen1, pts1 = val1[0], tuple(val1[1:])
    hom = Scalars.hom([gen1], codomain=NF)
    return (hom,) + pts1


def my_sequence(points):
    try:
        universe = coercion_model.common_parent(*points)
    except TypeError:
        universe, points = as_embedded_number_field_elements(
                                            [QQbar.coerce(pt) for pt in points])
    return Sequence(points, universe=universe)

######################################################################
# Algebraic numbers
######################################################################


class PolynomialRoot:
    r"""
    Root of an irreducible polynomial over a number field

    This class provides an ad hoc representation of algebraic numbers that
    allows us to perform some simple operations more efficiently than by using
    Sage's algebraic numbers or number field elements.

    It is mainly intended for cases where one manipulates all the roots of a
    given irreducible polynomial--typically singularities and local exponents.
    """

    def __init__(self, pol, all_roots, index):
        assert pol.is_monic()
        self.pol = pol  # may have coefficients in a number field
        self.all_roots = all_roots
        self.index = index

    def __repr__(self):
        return repr(self.as_algebraic())

    def __eq__(self, other):
        if self.pol is other.pol:
            return self.index == other.index
        elif self.pol.parent() is other.pol.parent():
            return False
        elif other.is_zero():
            return self.is_zero()
        else:
            # We could compare self.as_algebraic() with other.as_algebraic(),
            # but that would break hashing.
            raise NotImplementedError

    def __hash__(self):
        return hash((self.pol, self.index))

    def as_ball(self, tgt):
        alg = self.as_algebraic()
        prec = tgt.precision()
        if alg._value.prec() < prec:
            # avoid the loop in AlgebraicNumber_base._more_precision()...
            alg._value = alg._descr._interval_fast(prec)
        return tgt(alg._value)

    _acb_ = _complex_mpfr_field_ = _complex_mpfi_ = as_ball

    @cached_method
    def as_algebraic(self):
        if self.pol.base_ring() is QQ:
            # bypass ANRoot.exactify()
            nf = NumberField(self.pol, 'a', check=False)
            rt = ANRoot(self.pol, self.all_roots[self.index])
            gen = AlgebraicGenerator(nf, rt)
            return AlgebraicNumber(ANExtensionElement(gen, nf.gen()))
        else:
            return QQbar.polynomial_root(self.pol, self.all_roots[self.index])

    def _algebraic_(self, field):
        return field(self.as_algebraic())

    @cached_method
    def as_number_field_element(self):
        if self.pol.degree() == 1:
            val = -self.pol[0]
            if val.parent() is QQ or val in QQ:
                val = QQ(val)
            return val
        return as_embedded_number_field_element(self.as_algebraic())

    @cached_method
    def as_exact(self):
        if self.pol.degree() == 1:
            a = self.as_number_field_element()
            parent = a.parent()
            if parent is QQ or isinstance(parent, NumberField_quadratic):
                return a
        return self.as_algebraic()

    def conjugate(self):
        if self.pol.base_ring() is not QQ:
            raise NotImplementedError
        conj = self.all_roots[self.index].conjugate()
        index = next(i for i, rt in enumerate(self.all_roots)
                       if rt.overlaps(conj))
        return PolynomialRoot(self.pol, self.all_roots, index)

    def try_eq_conjugate(self, other):
        return (self.pol.base_ring() is QQ
                and self.all_roots is other.all_roots
                and self.all_roots[self.index].conjugate().overlaps(
                                                   self.all_roots[other.index]))

    def is_rational(self):
        return self.pol.degree() == 1 and (self.pol.base_ring() is QQ
                                           or self.pol[0] in QQ)

    def try_integer(self):
        if self.pol.degree() > 1:
            return None
        try:
            return -ZZ(self.pol[0])
        except (TypeError, ValueError):
            return None

    def is_zero(self):
        return self.pol == self.pol.parent().gen()

    def sign_imag(self):
        im = self.all_roots[self.index].imag()
        if im.is_zero():
            return 0
        elif im.lower() > 0:
            return +1
        elif im.upper() < 0:
            return -1
        # Should no longer be necessary with #33150
        return int(self.as_algebraic().imag().sign())

    @classmethod
    def make(cls, value):
        r"""
        Convenience method to create simple PolynomialRoot objects.

        Warning: Comparison of the resulting objects with each other or to
        PolynomialRoot objects created using root_of_irred is not supported.
        """
        if isinstance(value, PolynomialRoot):
            return value
        elif isinstance(value, (Rational, NumberFieldElement)):
            Pol = PolynomialRing(value.parent(), 'a')
            pol = Pol([-value, value.parent().one()])
            return cls(pol, [CIF(value)], 0)
        value = QQbar.coerce(value)
        pol = value.minpoly()
        roots, _ = zip(*complex_roots(pol, skip_squarefree=True))
        indices = [i for i, iv in enumerate(roots) if value in iv]
        assert len(indices) == 1
        return cls(pol, roots, indices[0])


def roots_of_irred(pol):
    if pol.degree() == 1:
        pol = pol.monic()
        return [PolynomialRoot(pol, [-CIF(pol[0])], 0)]
    roots, _ = zip(*complex_roots(pol, skip_squarefree=True))
    assert not any(a.overlaps(b) for a in roots for b in roots
                                 if a is not b)
    return [PolynomialRoot(pol, roots, i) for i in range(len(roots))]

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
