# -*- coding: utf-8 - vim: tw=80
r"""
Custom differential operators
"""

import sage.categories.pushout as pushout

from sage.arith.all import lcm
from sage.misc.cachefunc import cached_method
from sage.rings.all import CIF, QQbar
from sage.rings.complex_arb import ComplexBallField
from sage.rings.complex_interval_field import ComplexIntervalField

from ..ore_operator_1_1 import UnivariateDifferentialOperatorOverUnivariateRing

def DifferentialOperator(dop):
    if isinstance(dop, PlainDifferentialOperator):
        return dop
    else:
        return PlainDifferentialOperator(dop)

class PlainDifferentialOperator(UnivariateDifferentialOperatorOverUnivariateRing):
    r"""
    A subclass of differential operators for internal use by the numerical
    evaluation code.
    """

    def __init__(self, dop):
        if not dop:
            raise ValueError("operator must be nonzero")
        if not dop.parent().is_D():
            raise ValueError("expected an operator in K(x)[D]")
        _, _, _, dop = dop.numerator()._normalize_base_ring()
        den = lcm(c.denominator() for c in dop)
        dop *= den
        super(PlainDifferentialOperator, self).__init__(
                dop.parent(), dop)

    def singularities(self, *args):
        raise NotImplementedError("use _singularities()")

    @cached_method
    def _singularities(self, dom, include_apparent=True, multiplicities=False):
        dop = self if include_apparent else self.desingularize() # TBI
        if isinstance(dom, ComplexBallField): # TBI
            dom1 = ComplexIntervalField(dom.precision())
        else:
            dom1 = dom
        lc = dop.leading_coefficient()
        sing = lc.roots(dom1, multiplicities=multiplicities)
        if dom1 is not dom:
            sing = [dom(s) for s in sing]
        return sing

    def _sing_as_alg(dop, iv):
        pol = dop.leading_coefficient().radical()
        return QQbar.polynomial_root(pol, CIF(iv))

    def shift(self, delta):
        r"""
        TESTS::

            sage: from ore_algebra import DifferentialOperators
            sage: from ore_algebra.analytic.path import Point
            sage: from ore_algebra.analytic.differential_operator import DifferentialOperator
            sage: Dops, x, Dx = DifferentialOperators()
            sage: DifferentialOperator(x*Dx - 1).shift(Point(1))
            (x + 1)*Dx - 1
            sage: dop = DifferentialOperator(x*Dx - 1)
            sage: dop.shift(Point(RBF(1/2), dop))
            (2*x + 1)*Dx - 2
        """
        Pols_dop = self.base_ring()
        # NOTE: pushout(QQ[x], K) doesn't handle embeddings well, and creates
        # an L equal but not identical to K. But then other constructors like
        # PolynomialRing(L, x) sometimes return objects over K found in cache,
        # leading to endless headaches with slow coercions. But the version here
        # may be closer to what I really want in any case.
        # XXX: This seems to work in the usual trivial case where we are looking
        # for a scalar domain containing QQ and QQ[i], but probably won't be
        # enough if we really have two different number fields with embeddings
        ex = delta.exact()
        Scalars = pushout.pushout(Pols_dop.base_ring(), ex.value.parent())
        Pols = Pols_dop.change_ring(Scalars)
        A, B = self.base_ring().base_ring(), ex.value.parent()
        C = Pols.base_ring()
        assert C is A or C != A
        assert C is B or C != B
        dop_P = self.change_ring(Pols)
        # Gcd-avoiding shift by an algebraic delta
        deg = dop_P.degree()
        den = ex.value.denominator()
        num = den*ex.value
        lin = Pols([num, den])
        x = Pols.gen()
        def shift_poly(pol):
            pol = (pol.reverse(deg)(den*x)).reverse(deg)
            return pol(lin)
        shifted = dop_P.map_coefficients(shift_poly)
        return ShiftedDifferentialOperator(shifted, self, delta)

class ShiftedDifferentialOperator(PlainDifferentialOperator):

    def __init__(self, dop, orig, delta):
        super(ShiftedDifferentialOperator, self).__init__(dop)
        self._orig = orig
        self._delta = delta

    def _singularities(self, dom, include_apparent=True, multiplicities=False):
        sing = self._orig._singularities(dom, include_apparent, multiplicities)
        delta = dom(self._delta.value)
        if multiplicities:
            return [(s - delta, m) for s, m in sing]
        else:
            return [s - delta for s in sing]
