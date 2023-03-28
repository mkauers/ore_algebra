r"""
Ad-hoc algebraic numbers
"""

from sage.misc.cachefunc import cached_method
from sage.rings.all import ZZ, QQ, QQbar, CIF
from sage.rings.number_field.number_field import NumberField, NumberField_quadratic
from sage.rings.number_field.number_field_element import NumberFieldElement
from sage.rings.polynomial.complex_roots import complex_roots
from sage.rings.polynomial.real_roots import real_roots
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational import Rational
from sage.rings.qqbar import (AlgebraicGenerator, AlgebraicNumber,
                              ANExtensionElement, ANRoot)

from .utilities import as_embedded_number_field_element, is_real_parent

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
            x = self.pol.parent().gen()
            den = self.pol.monic().denominator()
            pol = self.pol(x/den).monic()
            assert pol.denominator().is_one()
            nf = NumberField(pol, 'a', check=False)
            rt = ANRoot(pol, self.all_roots[self.index]*den)
            gen = AlgebraicGenerator(nf, rt)
            return AlgebraicNumber(ANExtensionElement(gen, nf.gen()/den))
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
