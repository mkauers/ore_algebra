# vim: tw=80
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

from ..ore_algebra import OreAlgebra
from ..differential_operator_1_1 import UnivariateDifferentialOperatorOverUnivariateRing

from .context import dctx
from .polynomial_root import roots_of_irred
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
        den = lcm(utilities.internal_denominator(c) for pol in dop for c in pol)
        dop *= den
        super(PlainDifferentialOperator, self).__init__(
                dop.parent(), dop)

    @cached_method
    def _indicial_polynomial_at_zero(self):
        # Adapted from the version in differential_operator_1_1

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
            den = utilities.internal_denominator(c)
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
    def growth_parameters(self, *, bit_prec=53):
        r"""
        Find κ, α such that the solutions of dop grow at most like
        sum(α^n*x^n/n!^κ) ≈ exp(κ*(α·x)^(1/κ)).

        EXAMPLES::

            sage: from ore_algebra import *
            sage: DiffOps, x, Dx = DifferentialOperators()
            sage: from ore_algebra.analytic.differential_operator import DifferentialOperator
            sage: DifferentialOperator(Dx^2 + 2*x*Dx).growth_parameters() # erf(x)
            (1/2, [1.4...])
            sage: DifferentialOperator(Dx^2 + 8*x*Dx).growth_parameters() # erf(2*x)
            (1/2, [2.8...])
            sage: DifferentialOperator(Dx^2 - x).growth_parameters() # Airy
            (2/3, [1.0...])
            sage: DifferentialOperator(x*Dx^2 + (1-x)*Dx).growth_parameters() # Ei(1, -x)
            (1, [1.0...])
            sage: DifferentialOperator((Dx-1).lclm(Dx-2)).growth_parameters()
            (1, [2.0...])
            sage: DifferentialOperator((Dx - x).lclm(Dx^2 - 1)).growth_parameters()
            (1/2, [1.0...])
            sage: DifferentialOperator(x^2*Dx^2 + x*Dx + 1).growth_parameters()
            (+Infinity, 0)
        """
        from .bounds import abs_min_nonzero_root
        assert self.leading_coefficient().is_term()
        # Newton polygon. In terms of the coefficient sequence,
        # (S^(-j)·((n+1)S)^i)(α^n/n!^κ) ≈ α^(i-j)·n^(i+κ(j-i)).
        # In terms of asymptotics at infinity,
        # (x^j·D^i)(exp(κ·(α·x)^(1/κ))) ≈ α^(i/κ)·x^((i+κ(j-i))/κ)·exp(...).
        # Thus, we want the largest (negative) κ s.t. i+κ(j-i) is max and reached
        # twice, and then the largest |α| with sum[edge](a[i,j]·α^(i/κ))=0.
        # (Note that the equation on α resulting from the first formulation
        # simplifies thanks to i+κ(j-i)=cst on the edge.)
        # For a differential operator of order r, there may be more than r + 1
        # different values of i (<-> solutions of the associated recurrence), but
        # at most r + 1 values of h = j-i and hence at most r *negative* slopes.
        # Or maybe a better way to look at this is to say that we are considering
        # the classical Newton polygon at infinity (as in Loday-Richaud 2016,
        # Def. 3.3.10) but we are interested in the inverses of the slopes.
        points = [(ZZ(j-i), ZZ(i), c) for (i, pol) in enumerate(self)
                                    for (j, c) in enumerate(pol)
                                    if not c.is_zero()]
        h0, i0, _ = max(points, key=lambda p: (p[1], p[0]))
        hull = [(h, i, c) for (h, i, c) in points if h > h0 and i < i0]
        if not hull: # generalized polynomial
            return infinity, ZZ.zero()
        slope = max((i-i0)/(h-h0) for h, i, c in hull)
        Pol = self.base_ring()
        eqn = Pol({i0 - i: c for (h, i, c) in points if i == i0 + slope*(h-h0)})
        expo_growth = abs_min_nonzero_root(eqn, prec=bit_prec)**slope
        return -slope, expo_growth

    def singularities(self, *args):
        raise NotImplementedError("use _singularities()")

    @cached_method
    def split_leading_coefficient(self):
        lc = self.leading_coefficient()
        if lc.base_ring() is not QQ: # not worth the effort
            return lc, lc.parent().one()
        dlc = self.desingularize(m=1).leading_coefficient()
        alc, rem = lc.quo_rem(dlc)
        assert rem.is_zero()
        # "Partly apparent" factors go in the non-apparent one
        while True:
            g = dlc.gcd(alc)
            if g.is_one():
                return dlc, alc
            alc //= g
            dlc *= g

    @cached_method
    def _singularities(self, dom=None, multiplicities=False, apparent=None):
        r"""
        Complex singularities of self, as elements of dom.

        INPUT:

        - ``dom`` - parent; pass ``dom=None`` to get the singularities as
          ``PolynomialRoot`` objects.
        - ``multiplicities`` - boolean.
        - ``apparent`` - ``None`` to compute all singularities; the results with
          ``apparent=True`` and ``apparent=False`` form a disjoint union of the
          singularities, with all non-apparent singularities (and, possibly,
          some apparent ones) contained in the subset corresponding to
          ``apparent=False``.
        """
        if dom is not None or not multiplicities:
            # Memoize the version with all information
            sing = self._singularities(None, multiplicities=True,
                                       apparent=apparent)
            if dom is not None:
                sing = [(dom(rt), mult) for rt, mult in sing]
            if not multiplicities:
                sing = [s for s, _ in sing]
            return sing
        if apparent is None:
            # We might already have computed part of the singularities. If that
            # is the case, compute only the remaining ones. Otherwise, though,
            # we do not want to pay the price of trying to desingularize.
            for b in [False, True]:
                try:
                    sing = self._singularities.cached(dom, multiplicities, b)
                except KeyError:
                    continue
                sing += self._singularities(dom, multiplicities, not b)
                return sing
            pol = self.leading_coefficient()
        else:
            dlc, alc = self.split_leading_coefficient()
            pol = alc if apparent else dlc
        sing = []
        for fac, mult in pol.factor():
            roots = roots_of_irred(fac)
            sing.extend((rt, mult) for rt in roots)
        return sing

    def _sing_as_alg(self, iv):
        pol = self.leading_coefficient().radical()
        return QQbar.polynomial_root(pol, CIF(iv))

    @cached_method
    def est_cvrad(self, IR):
        # not rigorous! (because of the contains_zero())
        IC = IR.complex_field()
        sing = [a for a in self._singularities(IC) if not a.contains_zero()]
        if not sing:
            return IR('inf')
        else:
            return min(a.below_abs() for a in sing)

    @cached_method
    def _est_growth(self, IR):
        # Originally intended for the case of ordinary points only; may need
        # improvements for singular points.
        IC = IR.complex_field()
        kappa, alpha0 = self.growth_parameters(bit_prec=IR.precision())
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

    def est_terms(self, pt, prec, ctx=dctx):
        r"""
        Estimate the number of terms of series expansion at 0 of solutions of
        this operator necessary to reach prec bits of accuracy at pt, and the
        maximum log-magnitude of these terms.
        """
        # pt should be an EvaluationPoint
        prec = ctx.IR(prec)
        cvrad = self.est_cvrad(ctx.IR)
        if cvrad.is_infinity():
            kappa, alpha = self._est_growth(ctx.IR)
            if kappa is infinity:
                return 0, 0
            ratio = ctx.IR(alpha*pt.rad)
            hump = ctx.IR.one().exp() * ratio**(~kappa)
            klgp = kappa*prec.log(2)
            est = hump + prec/(klgp.max(-ratio.log(2)))
            mag = hump.log(2).max(ctx.IR.zero())
        else:
            est = prec/(cvrad/pt.rad).log(2)
            mag = ctx.IR.zero()
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
        hom, *pts1 = utilities.extend_scalars(Scalars, *pts)
        Dops1 = OreAlgebra(Pols.change_ring(hom.codomain()),
                (Dops.variable_name(), {}, {Pols.gen(): Pols.one()}))
        dop1 = Dops1([pol.map_coefficients(hom) for pol in self])
        dop1 = PlainDifferentialOperator(dop1)
        assert dop1.base_ring().base_ring() is hom.codomain()
        return (dop1,) + tuple(pts1)

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
        den = utilities.internal_denominator(ex)
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

    def _singularities(self, dom, multiplicities=False, apparent=None):
        sing = self._orig._singularities(dom, multiplicities, apparent)
        delta = dom(self._delta.value)
        if multiplicities:
            return [(s - delta, m) for s, m in sing]
        else:
            return [s - delta for s in sing]

    def _theta_alg(self):
        return self._orig._theta_alg_with_base(self.base_ring().base_ring())

    def _shift_alg(self):
        return self._orig._shift_alg_with_base(self.base_ring().base_ring())
