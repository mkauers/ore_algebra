# vim: tw=80
r"""
Ad-hoc algebraic numbers
"""

from sage.misc.cachefunc import cached_method
from sage.rings.integer_ring import Z as ZZ
from sage.rings.rational_field import Q as QQ
from sage.rings.qqbar import QQbar
from sage.rings.cif import CIF
from sage.rings.complex_arb import CBF
from sage.rings.number_field.number_field import NumberField, NumberField_quadratic
from sage.rings.number_field.number_field_element import NumberFieldElement
from sage.rings.polynomial.complex_roots import complex_roots
from sage.rings.polynomial.real_roots import real_roots
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational import Rational
from sage.rings.qqbar import (AlgebraicGenerator, AlgebraicNumber,
                              ANExtensionElement, ANRoot)
from sage.structure.sage_object import SageObject

from . import geometry

from .utilities import as_embedded_number_field_element, is_real_parent

################################################################################
# Roots of a common minpoly
################################################################################

class PolynomialRoot(SageObject):
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
        self.pol = pol # may have coefficients in a number field
        # all_roots is assumed to contain isolating intervals
        # it is shared between the roots and may get modified
        assert isinstance(all_roots, list)
        self.all_roots = all_roots
        self.index = index

    def __repr__(self):
        return repr(self.as_algebraic())

    def __eq__(self, other):
        if self.pol is other.pol:
            return self.index == other.index
        elif self.pol.parent() is other.pol.parent() and self.pol != other.pol:
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
        prec = tgt.precision()
        if self.all_roots[self.index].prec() >= prec:
            return tgt(self.all_roots[self.index])
        alg = self.as_algebraic()
        if alg._value.prec() < prec:
            # avoid the loop in AlgebraicNumber_base._more_precision()...
            alg._value = alg._descr._interval_fast(prec)
        return tgt(alg._value)

    _acb_ = _complex_mpfr_field_ = _complex_mpfi_ = as_ball

    def __complex__(self):
        return complex(self.all_roots[self.index])

    def interval(self):
        return self.all_roots[self.index]

    @cached_method
    def as_algebraic(self):
        r"""
        TESTS:

        Check that we return well-formed algebraic numbers when pol ∈ QQ[x] has
        a denominator::

            sage: from ore_algebra.analytic.polynomial_root import roots_of_irred
            sage: Pol.<z> = QQ[]
            sage: rts = roots_of_irred(z^3 + 59/23*z^2 - 59/23*z - 279/23)
            sage: [a.as_algebraic().imag().sign() for a in rts]
            [0, -1, 1]
        """
        if self.pol.base_ring() is QQ:
            # bypass ANRoot.exactify()
            assert self.pol.is_monic()
            den = self.pol.denominator()
            if den.is_one():
                pol = self.pol
            else:
                deg = self.pol.degree()
                denpow = ZZ.one()
                newc = [None]*(deg + 1)
                newc[deg] = QQ.one()
                for i in range(deg - 1, -1, -1):
                    denpow *= den
                    newc[i] = denpow*self.pol[i]
                pol = self.pol.parent()(newc)
                assert pol.denominator().is_one()
            nf = NumberField(pol, 'a', check=False)
            rt = ANRoot(pol, self.all_roots[self.index]*den)
            gen = AlgebraicGenerator(nf, rt)
            return AlgebraicNumber(ANExtensionElement(gen, (1/den)*nf.gen()))
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

    def algdeg(self):
        if self.pol.base_ring() is QQ:
            return self.pol.degree()
        else:
            return self.as_number_field_element().parent().absolute_degree()

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
        candidates = [i for i, rt in enumerate(self.all_roots)
                        if rt.overlaps(conj)]
        if len(candidates) != 1:
            raise NotImplementedError("isolating intervals of conjugate roots "
                                      "are not conjugate intervals")
        return PolynomialRoot(self.pol, self.all_roots, candidates[0])

    def try_eq_conjugate(self, other):
        return (self.pol.base_ring() is QQ
                and self.all_roots is other.all_roots
                and self.conjugate().index == other.index)

    def try_eq_opp_conjugate(self, other):
        if not (self.pol.base_ring() is QQ
                and self.all_roots is other.all_roots # same minpoly
                and all(c.is_zero() for c in list(self.pol)[1::2])): # even poly
            return False
        # We know that the opposite conjugate must be among the roots of
        # self.poly. If we can prove using the isolating intervals that it is
        # equal to other, self and other have the same imaginary part.
        oppconj = -self.all_roots[self.index].conjugate()
        return all(oppconj.overlaps(rt) == (i == other.index)
                   for i, rt in enumerate(self.all_roots))

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

    def detect_real_roots(self):
        r"""
        Try to make the imaginary parts of real roots exactly zero
        """
        if not is_real_parent(self.pol.base_ring()):
            return
        for rt in self.all_roots:
            im = rt.imag()
            if im.contains_zero() and not im.is_zero():
                break
        else:
            return
        myCIF = self.all_roots[0].parent()
        rrts = real_roots(self.pol, retval='interval', skip_squarefree=True,
                          max_diameter=2.**-myCIF.prec())
        for rrt, _ in rrts:
            rrt = myCIF(rrt)
            compat = [i for i, rt in enumerate(self.all_roots)
                        if rt.overlaps(rrt)]
            if len(compat) == 1:
                self.all_roots[compat[0]] = rrt

    def is_real(self):
        r"""
        Try to decide if this root is real. May return false negatives.
        """
        if self.all_roots[self.index].imag().is_zero():
            return True
        self.detect_real_roots()
        return self.all_roots[self.index].imag().is_zero()

    def sign_imag(self):
        r"""
        TESTS::

            sage: from ore_algebra.analytic.polynomial_root import roots_of_irred
            sage: Pol.<z> = QQ[]
            sage: pol = (z^4 + 6552580/3600863*z^3 + 6913064/32407767*z^2 -
            ....:        1009036400/875009709*z - 470919776/875009709)
            sage: rts = roots_of_irred(pol)
            sage: rts[-1].sign_imag()
            0
        """
        im = self.all_roots[self.index].imag()
        if im.is_zero():
            return 0
        elif im.lower() > 0:
            return +1
        elif im.upper() < 0:
            return -1
        self.detect_real_roots()
        if self.all_roots[self.index].imag().is_zero():
            return 0
        return int(self.as_algebraic().imag().sign())

    def cmp_real(self, other):
        if self is other:
            return 0
        re_self = self.all_roots[self.index].real()
        re_other = other.all_roots[other.index].real()
        if re_self < re_other:
            return -1
        elif re_self > re_other:
            return +1
        elif re_self == re_other:
            return 0
        try:
            if self == other:
                return 0
        except NotImplementedError:
            pass
        if self.try_eq_conjugate(other):
            return 0
        delta = self.as_algebraic() - other.as_algebraic()
        return int(delta.real().sign())

    def cmp_imag(self, other):
        if self is other:
            return 0
        im_self = self.all_roots[self.index].imag()
        im_other = other.all_roots[other.index].imag()
        if im_self < im_other:
            return -1
        elif im_self > im_other:
            return +1
        elif im_self == im_other:
            return 0
        try:
            if self == other:
                return 0
        except NotImplementedError:
            pass
        # TODO: case g(x) = f(x + a) for real a, using [x^(n-1)]f?
        if im_self.contains_zero() or im_other.contains_zero():
            self.detect_real_roots()
            if im_self == im_other:
                return 0
        if self.try_eq_opp_conjugate(other):
            return 0
        delta = self.as_algebraic() - other.as_algebraic()
        return int(delta.imag().sign())

    @classmethod
    def make(cls, value):
        r"""
        Convenience method to create simple PolynomialRoot objects.

        Warning: Comparison (with ==) of the resulting objects with each other
        or to PolynomialRoot objects created using root_of_irred is not
        supported.
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
        return cls(pol, list(roots), indices[0])

def roots_of_irred(pol):
    if pol.degree() == 1:
        pol = pol.monic()
        return [PolynomialRoot(pol, [-CIF(pol[0])], 0)]
    roots, _ = zip(*complex_roots(pol, skip_squarefree=True))
    assert not any(a.overlaps(b) for a in roots for b in roots
                                 if a is not b)
    roots = list(roots)
    return [PolynomialRoot(pol, roots, i) for i in range(len(roots))]

################################################################################
# Utilities for sorting PolynomialRoot objects
################################################################################

class _sort_key:

    def __init__(self, a):
        assert isinstance(a, PolynomialRoot)
        self.value = a

    def __eq__(self, other):
        return self.value == other.value

class sort_key_left_to_right_real_last(_sort_key):
    r"""
    Sort by increasing real part, then by absolute value of the imaginary part,
    then by increasing imaginary part.

    Thus purely real values come last for each real part. This is intended for
    sorting the exponential parts of the local solutions of a differential
    operator.

    TESTS::

        sage: from ore_algebra.analytic.polynomial_root import (roots_of_irred,
        ....:         sort_key_left_to_right_real_last)
        sage: Pol.<x> = QQ[]
        sage: v = [*roots_of_irred(x^4+1), *roots_of_irred(x^2-1/2),
        ....:      *roots_of_irred(x^2+1/2), *roots_of_irred(x^4-2)]
        sage: v.sort(key=sort_key_left_to_right_real_last)
        sage: v
        [-1.189207115002722?,
        -0.7071067811865475? - 0.7071067811865475?*I,
        -0.7071067811865475? + 0.7071067811865475?*I,
        -0.7071067811865475?,
        -1.189207115002722?*I,
        1.189207115002722?*I,
        -0.7071067811865475?*I,
        0.7071067811865475?*I,
        0.7071067811865475? - 0.7071067811865475?*I,
        0.7071067811865475? + 0.7071067811865475?*I,
        0.7071067811865475?,
        1.189207115002722?]
    """

    def __lt__(self, other):
        s = self.value.cmp_real(other.value)
        if s < 0:
            return True
        elif s > 0:
            return False
        # We know that the real parts are equal.
        # Compare the imaginary parts in such a way that purely real values come
        # last (for consistency with sort_key_by_asympt).
        im_self = self.value.interval().imag()
        im_other = other.value.interval().imag()
        if abs(im_self) > abs(im_other):
            return True
        elif abs(im_self) < abs(im_other):
            return False
        if self.value.try_eq_conjugate(other.value):
            # then the conjugate with negative imaginary part must come first
            return self.value.as_algebraic().imag() < 0
        other.value.detect_real_roots()
        if im_other.is_zero():
            # then self == other or |im(self)| > 0
            return False
        if self.value.as_algebraic() == other.value.as_algebraic():
            return False
        # Since the real parts are equal, the largest imaginary part corresponds
        # to the largest absolute value, but comparing the absolute values might
        # be faster
        abs0 = abs(self.value.as_algebraic())
        abs1 = abs(other.value.as_algebraic())
        if abs0 != abs1:
            return abs0 > abs1
        else:
            # We know that self != other, so abs(self) == abs(other) here means
            # that they are conjugates
            return self.value.as_algebraic().imag() < 0

class sort_key_bottom_to_top_with_cuts(_sort_key):
    r"""
    Sort objects by increasing imaginary part, then by decreasing real part.

    This is intended for sorting singular points in a way compatible with the
    “stickiness” rules for branch cuts.

    TESTS::

        sage: from ore_algebra.analytic.polynomial_root import (roots_of_irred,
        ....:         sort_key_bottom_to_top_with_cuts)
        sage: Pol.<x> = QQ[]
        sage: v = [*roots_of_irred(x^4+1), *roots_of_irred(x^2-1/2),
        ....:      *roots_of_irred(x^2+1/2), *roots_of_irred(x^4-2)]
        sage: v.sort(key=sort_key_bottom_to_top_with_cuts)
        sage: v
        [-1.189207115002722?*I,
        0.7071067811865475? - 0.7071067811865475?*I,
        -0.7071067811865475?*I,
        -0.7071067811865475? - 0.7071067811865475?*I,
        1.189207115002722?,
        0.7071067811865475?,
        -0.7071067811865475?,
        -1.189207115002722?,
        0.7071067811865475? + 0.7071067811865475?*I,
        0.7071067811865475?*I,
        -0.7071067811865475? + 0.7071067811865475?*I,
        1.189207115002722?*I]
    """

    def __lt__(self, other):
        s = self.value.cmp_imag(other.value)
        if s < 0:
            return True
        elif s > 0:
            return False
        return self.value.cmp_real(other.value) > 0

################################################################################
# Geometric predicates
################################################################################

def orient2d(p, q, r):
    r"""
    A version of the standard 2D orientation predicate for ``PolynomialRoot``
    objects.

    Positive when p, q, r are ordered counterclockwise around a point inside
    their convex hull, i.e., when r lies to the left of pq.
    """
    bp, bq, br = CBF(p), CBF(q), CBF(r)
    try:
        return geometry.orient2d_interval(bp, bq, br)
    except ValueError:
        pass
    ap, aq, ar = QQbar(p), QQbar(q), QQbar(r)
    # XXX are there some special cases that we need to handle efficiently?
    ratio = (ar - ap)/(aq - ap)
    return ratio.imag().sign()
