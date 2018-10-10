# -*- coding: utf-8 - vim: tw=80
r"""
Custom differential operators
"""

from sage.arith.all import lcm
from sage.categories.pushout import pushout
from sage.misc.cachefunc import cached_method
from sage.rings.all import CIF, QQbar, QQ
from sage.rings.complex_arb import ComplexBallField
from sage.rings.complex_interval_field import ComplexIntervalField
from sage.structure.coerce_exceptions import CoercionException

from ..ore_algebra import OreAlgebra
from ..ore_operator_1_1 import UnivariateDifferentialOperatorOverUnivariateRing

from .utilities import as_embedded_number_field_elements

from . import utilities

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

    @cached_method
    def growth_parameters(self):
        from .bounds import growth_parameters
        return growth_parameters(self)

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

    def extend_scalars(self, pt):
        r"""
        Extend the ground field so that the new field contains pt.
        """
        Dops = self.parent()
        Pols = Dops.base_ring()
        Scalars = Pols.base_ring()
        if Scalars.has_coerce_map_from(pt.parent()):
            return self, pt
        gen = Scalars.gen()
        try:
            # Largely redundant with the other branch, but may do a better job
            # in some cases, e.g. pushout(QQ, QQ(α)), where as_enf_elts() would
            # invent new generator names.
            NF = pushout(Scalars, pt.parent())
            gen1 = NF.coerce(gen)
            pt1 = NF.coerce(pt)
        except CoercionException:
            NF, (gen1, pt1) = as_embedded_number_field_elements([gen,pt])
        hom = Scalars.hom([gen1], codomain=NF)
        Dops1 = OreAlgebra(Pols.change_ring(NF),
                (Dops.variable_name(), {}, {Pols.gen(): Pols.one()}))
        dop1 = Dops1([pol.map_coefficients(hom) for pol in self])
        dop1 = PlainDifferentialOperator(dop1)
        assert dop1.base_ring().base_ring() is NF
        return dop1, pt1

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
        # an L equal but not identical to K. And then other constructors like
        # PolynomialRing(L, x) sometimes return objects over K found in cache,
        # leading to endless headaches with slow coercions.
        dop_P, ex = self.extend_scalars(delta.exact().value)
        Pols = dop_P.base_ring()
        # Gcd-avoiding shift by an algebraic delta
        deg = dop_P.degree()
        den = ex.denominator()
        num = den*ex
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
