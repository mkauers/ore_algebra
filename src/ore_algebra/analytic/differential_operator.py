# -*- coding: utf-8 - vim: tw=80
r"""
Custom differential operators
"""

# Copyright 2018 Marc Mezzarobba
# Copyright 2018 Centre national de la recherche scientifique
# Copyright 2018 Université Pierre et Marie Curie
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

from sage.arith.all import lcm
from sage.misc.cachefunc import cached_method
from sage.rings.all import CIF, QQbar, QQ, ZZ
from sage.rings.complex_arb import ComplexBallField
from sage.rings.complex_interval_field import ComplexIntervalField
from sage.rings.infinity import infinity
from sage.rings.number_field.number_field import is_NumberField
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.coerce_exceptions import CoercionException
from sage.structure.element import coercion_model

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
    def _indicial_polynomial_at_zero(self):
        # Adapted from the version in ore_operator_1_1

        op = self.numerator()
        R = op.base_ring()
        y = R.gen()

        b = min(c.valuation() - j for j, c in enumerate(op))

        s = R.zero()
        y_ff_i = R.one()
        for i, c in enumerate(op):
            s += c[b + i]*y_ff_i
            y_ff_i *= y - i

        return s

    @cached_method
    def _naive_height(self):
        def h(c):
            den = c.denominator()
            num = den*c
            l = list(num)
            l.append(den)
            return max(ZZ(a).nbits() for a in l)
        return max(h(c) for pol in self for c in pol)

    @cached_method
    def _my_to_S(self):
        # Using the primitive part here would break the computation of residuals!
        # TODO: add test (arctan); better fix?
        # Other interesting cases: operators of the form P(Θ) (with constant
        # coefficients)
        #rop = self.to_S(Rops).primitive_part().numerator()
        return self.to_S(self._shift_alg())

    @cached_method
    def growth_parameters(self):
        from .bounds import growth_parameters
        return growth_parameters(self)

    def singularities(self, *args):
        raise NotImplementedError("use _singularities()")

    @cached_method
    def _singularities(self, dom, include_apparent=True, multiplicities=False):
        if not multiplicities:
            rts = self._singularities(dom, include_apparent, multiplicities=True)
            return [s for s, _ in rts]
        if isinstance(dom, ComplexBallField): # TBI
            dom1 = ComplexIntervalField(dom.precision())
            rts = self._singularities(dom1, include_apparent, multiplicities)
            return [(dom(s), m) for s, m in rts]
        dop = self if include_apparent else self.desingularize() # TBI
        lc = dop.leading_coefficient()
        try:
            return lc.roots(dom)
        except NotImplementedError:
            return lc.change_ring(QQbar).roots(dom)

    def _sing_as_alg(dop, iv):
        pol = dop.leading_coefficient().radical()
        return QQbar.polynomial_root(pol, CIF(iv))

    @cached_method
    def est_cvrad(self):
        # not rigorous! (because of the contains_zero())
        from .bounds import IR, IC
        sing = [a for a in self._singularities(IC) if not a.contains_zero()]
        if not sing:
            return IR('inf')
        else:
            return min(a.below_abs() for a in sing)

    @cached_method
    def _est_growth(self):
        # Originally intended for the case of ordinary points only; may need
        # improvements for singular points.
        from .bounds import growth_parameters, IR, IC
        kappa, alpha0 = growth_parameters(self)
        if kappa is infinity:
            return kappa, IR.zero()
        # The asymptotic exponential growth may not be such a great estimate
        # for the actual growth during the pre-convergence stage, so we
        # complement it by another one.
        rop = self._my_to_S()
        lc = rop.leading_coefficient()
        n0 = 1
        while lc(n0).is_zero():
            n0 += 1
        alpha1 = max(abs(IC(pol(n0))) for pol in list(rop)[:-1])/abs(IC(lc(n0)))
        alpha = IR(alpha0).max(alpha1)
        return kappa, alpha

    def est_terms(self, pt, prec):
        r"""
        Estimate the number of terms of series expansion at 0 of solutions of
        this operator necessary to reach prec bits of accuracy at pt, and the
        maximum log-magnitude of these terms.
        """
        from .bounds import IR
        # pt should be an EvaluationPoint
        prec = IR(prec)
        cvrad = self.est_cvrad()
        if cvrad.is_infinity():
            kappa, alpha = self._est_growth()
            if kappa is infinity:
                return 0, 0
            ratio = IR(alpha*pt.rad)
            hump = IR.one().exp() * ratio**(~kappa)
            klgp = kappa*prec.log(2)
            est = hump + prec/(klgp.max(-ratio.log(2)))
            mag = hump.log(2).max(IR.zero())
        else:
            est = prec/(cvrad/pt.rad).log(2)
            mag = IR.zero()
        return int(est.ceil().upper()), int(mag.ceil().upper())

    def extend_scalars(self, *pts):
        r"""
        Extend the ground field so that the new field contains pts.
        """
        Dops = self.parent()
        Pols = Dops.base_ring()
        Scalars = Pols.base_ring()
        if all(Scalars.has_coerce_map_from(pt.parent()) for pt in pts):
            return (self,) + pts
        gen = Scalars.gen()
        try:
            # Largely redundant with the other branch, but may do a better job
            # in some cases, e.g. pushout(QQ, QQ(α)), where as_enf_elts() would
            # invent new generator names.
            NF0 = coercion_model.common_parent(Scalars, *pts)
            if not is_NumberField(NF0):
                raise CoercionException
            NF, hom = utilities.good_number_field(NF0)
            gen1 = hom(NF0.coerce(gen))
            pts1 = tuple(hom(NF0.coerce(pt)) for pt in pts)
        except (CoercionException, TypeError):
            NF, val1 = as_embedded_number_field_elements((gen,)+pts)
            gen1, pts1 = val1[0], tuple(val1[1:])
        hom = Scalars.hom([gen1], codomain=NF)
        Dops1 = OreAlgebra(Pols.change_ring(NF),
                (Dops.variable_name(), {}, {Pols.gen(): Pols.one()}))
        dop1 = Dops1([pol.map_coefficients(hom) for pol in self])
        dop1 = PlainDifferentialOperator(dop1)
        assert dop1.base_ring().base_ring() is NF
        return (dop1,) + pts1

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

    @cached_method
    def _theta_alg_with_base(self, Scalars):
        Dop = self.parent()
        Pol = Dop.base_ring().change_ring(Scalars)
        x = Pol.gen()
        return OreAlgebra(Pol, 'T'+str(x))

    def _theta_alg(self):
        return self._theta_alg_with_base(self.parent().base_ring().base_ring())

    @cached_method
    def _shift_alg_with_base(self, Scalars):
        Pols_n, n = PolynomialRing(Scalars, 'n').objgen()
        return OreAlgebra(Pols_n, 'Sn')

    def _shift_alg(self):
        return self._shift_alg_with_base(self.base_ring().base_ring())

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

    def _theta_alg(self):
        return self._orig._theta_alg_with_base(self.base_ring().base_ring())

    def _shift_alg(self):
        return self._orig._shift_alg_with_base(self.base_ring().base_ring())
