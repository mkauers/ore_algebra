# -*- coding: utf-8 - vim: tw=79
r"""
Error bounds

This module provides tools for computing rigorous bounds on the tails of
power series and generalized series expansions of differentially finite
functions at ordinary points and regular singular points of their defining
equations.

The main use of these bounds is to decide at which order to truncate the series
expansions used at each integration step of the numerical solution algorithm.
In normal usage of the package, this is done automatically, and the resulting
bounds are taken into account in the returned intervals, so that there is no
need for users to call this module directly.

We refer to [M19] for a description of the bound computation algorithm. The
documentation of this module assumes that the reader is familiar with the
contents of that paper.

EXAMPLE:

This example shows how the features of this module fit together in a simple
case, but is probably hard to adapt to other situations without some
familiarity with [M19].

Consider the following operator::

    sage: from ore_algebra import OreAlgebra
    sage: Pol.<t> = QQ[]
    sage: Dop.<Dt> = OreAlgebra(Pol)
    sage: dop = Dt^2 + 3*t^2/(t^3 + 27)*Dt + t/(t^3 + 27)

In order to work with bounds on the tails of its solutions, we first create a
DiffOpBound object. The DiffOpBound object only depends on the operator itself,
not on a specific solution or truncation order. ::

    sage: from ore_algebra.analytic.bounds import *
    sage: maj = DiffOpBound(dop)
    sage: maj
    1.000000000000000/((-t + [2.9959...])^3)*exp(int(POL+1.0000...*NUM/(-t + [2.9959...])^3))
    where
    POL=~0*n^-2*t^0 + ~0*n^-2*t^1 + ~[0.111111 +/- 1.12e-7]*t^2,
    NUM=~0*n^-2*t^3 + ~0*n^-2*t^4 + ~[0.111111 +/- 1.12e-7]*t^5

Let us now focus on a particular solution, say ::

    sage: sol = dop.power_series_solutions(20)[0]; sol
    t - 1/81*t^4 + 25/91854*t^7 - 80/11160261*t^10 + 2420/11751754833*t^13 -
    847/135984591639*t^16 + 244783/1255681719194526*t^19 + O(t^20)

and compute a bound on the tail hidden in the O() term. We are in the
ideal situation where we have at our disposal the terms of the series coming
just before the truncation order in which we are interested. Using the last few
of these coefficients::

    sage: coeffs = list(sol)[-maj.dop.degree():]; coeffs
    [0, 0, 244783/1255681719194526]

we compute a “normalized residual” depending on the particular solution and
truncation point::

    sage: res = maj.normalized_residual(20, [[c] for c in reversed(coeffs)])
    sage: res
    [([1.68779500484630e-10 +/- 3.67e-25])*t^2]

We are now in a position to compute a majorant series for the tail::

    sage: tmaj = maj.tail_majorant(20, [res]); tmaj
    (t^22*([1.687795004846299e-10 +/- 6.55e-26]) * (-t + [2.995941329747438 +/-
    4.31e-16])^-3)*exp(int([0.1169590648741211 +/- 1.04e-17]*t^2 +
    [0.1169590648741211 +/- 1.04e-17]*t^5/((-t + [2.995941329747438 +/-
    4.31e-16])^3)))

Finally, to deduce a numeric bound, we fix a disk of validity |t| ≤ ρ and
bound the value at ρ of the majorant series::

    sage: tmaj.bound(RBF(1))
    [2.212445766787241e-11 +/- 5.44e-27]
    sage: tmaj.bound(RBF(2))
    [0.00346092101224178 +/- 9.91e-19]

If we are not happy with these bounds, we can spend more computational effort
trying to make them tighter::

    sage: maj.refine()
    sage: maj.refine()
    sage: maj.tail_majorant(20, [res]).bound(RBF(2))
    [0.001295756152020835 +/- 4.55e-19]

Another object occasionally of interest is ::

    sage: maj(20)
    (1.00... * (-t + [2.99...])^-2 * (-t + 3.00...)^-1)*exp(int([0.11...]*t^2 +
    [0.0043...]*t^5 + [0.0043...]*t^8/((-t + 3.00...) * (-t + [2.99...]) * (-t
    + [2.99...]))))

This is an auxiliary series used internally by ``tail_majorant()`` and in which
the truncation order is fixed, but not the specific solution. (Actually, what
is fixed is a _minimal_ truncation order: the auxiliary series can also be used
to obtain bounds on tails of higher order. However, choosing the order
parameter of the auxiliary series close to the actual order of truncation
yields tighter bounds.)

REFERENCE:

    [M19] Marc Mezzarobba, Truncation Bounds for Differentially Finite Series.
    *Annales Henri Lebesgue* 2:99–148, 2019.
    <https://hal.archives-ouvertes.fr/hal-01817568>
    <https://doi.org/10.5802/ahl.17>
"""

# Copyright 2015, 2016, 2017, 2018, 2019 Marc Mezzarobba
# Copyright 2015, 2016, 2017, 2018, 2019 Centre national de la recherche scientifique
# Copyright 2015, 2016, 2017, 2018 Université Pierre et Marie Curie
# Copyright 2019 Sorbonne Université
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

from __future__ import division, print_function
from six.moves import range

import collections, itertools, logging, sys, warnings

from sage.arith.srange import srange
from sage.misc.cachefunc import cached_function, cached_method
from sage.misc.lazy_string import lazy_string
from sage.misc.misc_c import prod
from sage.misc.random_testing import random_testing
from sage.rings.all import ComplexIntervalField
from sage.rings.complex_arb import CBF, ComplexBallField, ComplexBall
from sage.rings.infinity import infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import NumberField_quadratic
from sage.rings.polynomial.complex_roots import complex_roots
from sage.rings.polynomial.polynomial_element import Polynomial
from sage.rings.polynomial.polynomial_ring import polygen
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.qqbar import QQbar
from sage.rings.rational_field import QQ
from sage.rings.real_arb import RBF, RealBall
from sage.rings.real_mpfi import RIF
from sage.rings.real_mpfr import RealField, RR
from sage.structure.factorization import Factorization

from .. import ore_algebra
from . import accuracy, local_solutions, utilities

from .differential_operator import DifferentialOperator
from .safe_cmp import *
from .accuracy import IR, IC

myCIF = ComplexIntervalField(IC.precision())

logger = logging.getLogger(__name__)

class BoundPrecisionError(Exception):
    pass

######################################################################
# Majorant series
######################################################################

class MajorantSeries(object):
    r"""
    A formal power series with nonnegative coefficients
    """

    def __init__(self, variable_name, cvrad=IR.zero()):
        self.variable_name = variable_name
        self.cvrad = IR(cvrad)
        assert self.cvrad >= IR.zero()

    def bound_series(self, rad, ord):
        r"""
        Compute a termwise bound on the series expansion of self at rad to
        order O(x^ord).

        More precisely, the upper bound of each interval coefficient is a bound
        on the corresponding coefficient of self (which itself is a bound on
        the absolute value of the corresponding coefficient of the series this
        object is intended to bound).
        """
        return self.series(rad, ord)

    def series(self, rad, ord):
        r"""
        Compute the series expansion of self at rad to order O(x^ord).

        With rad = 0, this returns the majorant series itself. More generally,
        this can be used to obtain bounds on the derivatives of the series this
        majorant bounds on disks contained within its disk of convergence.
        """
        raise NotImplementedError

    def __call__(self, rad):
        r"""
        Bound the value of this series at rad ≥ 0.
        """
        return self.series(rad, 1)[0]

    def bound(self, rad, rows=1, cols=1, tail=None):
        """
        Bound the Frobenius norm of the matrix of the given dimensions whose
        columns are all equal to

            [g(rad), g'(rad), g''(rad)/2, ..., 1/(r-1)!·g^(r-1)(rad)]

        and g is this majorant series. Typically, g(z) is a common majorant
        series of the elements of a basis of solutions of some differential
        equation, and the result is then a bound on the corresponding
        fundamental matrix Y(ζ) for all for all ζ with `|ζ|` ≤ rad.
        """
        if not safe_le(rad, self.cvrad): # intervals!
            return IR(infinity)
        elif tail is None:
            ser = self.bound_series(rad, rows)
        else:
            ser = self.bound_tail_series(rad, rows, tail)
        sqnorm = IR(cols)*sum((c.above_abs()**2 for c in ser), IR.zero())
        return sqnorm.sqrtpos()

    def _test(self, fun=0, prec=50, return_difference=False):
        r"""
        Check that ``self`` is *plausibly* a majorant of ``fun``.

        This function in intended for debugging purposes. It does *not* perform
        a rigorous check that ``self`` is a majorant series of ``fun``, and may
        yield false positives (but no false negatives).

        The reference function ``fun`` should be convertible to a series with
        complex ball coefficients. If ``fun`` is omitted, check that ``self``
        has nonnegative coefficients.

        TESTS::

            sage: from ore_algebra.analytic.bounds import *
            sage: Pol.<z> = RBF[]
            sage: maj = RationalMajorant([(Pol(1), Factorization([(1-z,1)]))])
            sage: maj._test(11/10*z^30)
            Traceback (most recent call last):
            ...
            AssertionError: (30, [-0.10000000000000 +/- 8.00e-16], '< 0')
        """
        Series = PowerSeriesRing(IR, self.variable_name, prec)
        # CIF to work around problem with sage power series, should be IC
        ComplexSeries = PowerSeriesRing(myCIF, self.variable_name, prec)
        maj = Series(self.bound_series(0, prec))
        ref = Series([iv.abs() for iv in ComplexSeries(fun)], prec=prec)
        delta = (maj - ref).padded_list()
        if len(delta) < prec:
            warnings.warn("checking {} term(s) instead of {} (cancellation"
                    " during series expansion?)".format(len(delta), prec))
        for i, c in enumerate(delta):
            # the lower endpoint of a coefficient of maj is not a bound in
            # general, and the series expansion can overestimate the
            # coefficients of ref
            if c < IR.zero():
                raise AssertionError(i, c, '< 0')
        if return_difference:
            return delta

def _zero_free_rad(pols):
    r"""
    Return the radius of a disk around the origin without zeros of any of the
    polynomials in pols.
    """
    if all(pol.degree() == 0 for pol in pols):
        return IR(infinity)
    if all(pol.degree() == 1 and (pol.leading_coefficient().is_one()
                                  or pol.leading_coefficient().abs().is_one())
           for pol in pols):
        rad = IR(infinity).min(*(IC(pol[0]).abs() for pol in pols))
        rad = IR(rad.lower())
        assert rad >= IR.zero()
        return rad
    raise NotImplementedError

class RationalMajorant(MajorantSeries):
    r"""
    A rational power series with nonnegative coefficients, represented as an
    unevaluated sum of rational fractions with factored denominators.

    The terms must be ordered by decreasing radii of convergence as estimated
    from the denominator (some numerators may be zero, or, more generally,
    the numerator and denominator may have common factors).

    TESTS::

        sage: from ore_algebra.analytic.bounds import *
        sage: Pol.<z> = RBF[]
        sage: den = Factorization([(1-z, 2), (2-z, 1)])
        sage: one = Pol.one().factor()
        sage: maj = RationalMajorant([(1 + z, one), (z^2, den), (Pol(0), den)])
        sage: maj
        1.000... + 1.000...*z + z^2/((-z + 2.000...) * (-z + 1.000...)^2)
        sage: maj(1/2)
        [2.166...]
        sage: maj*(z^10)
        1.000...*z^10 + 1.000...*z^11 + z^12/((-z + 2.000...) * (-z + 1.000...)^2)
        sage: maj.cvrad
        1.000000000000000
        sage: maj.series(0, 4)
        1.250000000000000*z^3 + 0.5000000000000000*z^2 + z + 1.000000000000000
        sage: maj._test()
        sage: maj._test(1 + z + z^2/((1-z)^2*(2-z)), return_difference=True)
        [0, 0, 0, ...]
        sage: maj._test(1 + z + z^2/((1-z)*(2-z)), return_difference=True)
        [0, 0, 0, 0.5000000000000000, 1.250000000000000, ...]

        sage: RationalMajorant([(Pol(0), den), (Pol(0), den)]).cvrad
        [+/- inf]
    """

    def __init__(self, fracs):
        self.Poly = Poly = fracs[0][0].parent().change_ring(IR)
        self._Poly_IC = fracs[0][0].parent().change_ring(IC)
        fracs = [(num, den) for num, den in fracs if num]
        if fracs:
            cvrad = _zero_free_rad([-fac for fac, _ in fracs[-1][1]
                                         if fac.degree() > 0])
        else:
            cvrad = IR(infinity)
        # assert cvrad.identical(
        #         _zero_free_rad([-fac for num, den in fracs if num
        #                              for fac, _ in den if fac.degree() > 0]))
        super(self.__class__, self).__init__(Poly.variable_name(), cvrad=cvrad)
        self.fracs = []
        for num, den in fracs:
            if isinstance(num, Polynomial) and isinstance(den, Factorization):
                if not den.unit().is_one():
                    raise ValueError("expected a denominator with unit part 1")
                assert den.universe() is Poly or list(den) == []
                if not num.is_zero():
                    self.fracs.append((num, den))
            else:
                raise TypeError

    def __repr__(self):
        res = ""
        Series = self.Poly.completion(self.Poly.gen())
        def term(num, den):
            if den.value() == 1:
                return repr(Series(num))
            elif num.is_term():
                return "{}/({})".format(num, den)
            else:
                return "({})/({})".format(num._coeff_repr(), den)
        res = " + ".join(term(num, den) for num, den in self.fracs if num)
        return res if res != "" else "0"

    def series(self, rad, ord):
        Pol = self._Poly_IC # XXX: switch to self.Poly once arb_polys are interfaced
        pert_rad = Pol([rad, 1])
        res = Pol.zero()
        for num, den in self.fracs:
            den_ser = self.fracs[0][0].parent().one()
            for lin, mult in den:
                fac_ser = lin(pert_rad).power_trunc(mult, ord)
                den_ser = den_ser._mul_trunc_(fac_ser, ord)
            den_ser = Pol(den_ser)
            # slow; hopefully the fast Taylor shift will help...
            num_ser = Pol(num).compose_trunc(pert_rad, ord)
            res += num_ser._mul_trunc_(den_ser.inverse_series_trunc(ord), ord)
        return res

    def bound_integral(self, rad, ord):
        r"""
        Compute a termwise bound on the series expansion of int(self, 0..z) at
        z = rad, to order O(z^ord).
        """
        # For each summand f = num/den of self, we bound the series int(f,0..z)
        # by int(num,0..z)/den(z), using the fact that num and 1/den have
        # nonnegative coefficients and the bound int(fg) << int(f)·g (which
        # can be proved by integrating by parts). We then compose with rad+ε to
        # get the desired series expansion.
        # (Alternative algorithm: only bound the constant term this way,
        # use self.series().integral() for the remaining terms. Probably
        # slightly tighter and costlier.)
        Pol = self._Poly_IC # XXX: switch to self.Poly
        pert_rad = Pol([rad, 1])
        res = Pol.zero()
        for num, den in self.fracs:
            den_ser = Pol.one()
            for lin, mult in den:
                # composing with pert_rad is slow
                fac_ser = lin(pert_rad).power_trunc(mult, ord)
                den_ser = den_ser._mul_trunc_(fac_ser, ord)
            num_ser = Pol(num.integral()).compose_trunc(pert_rad, ord)
            res += num_ser._mul_trunc_(den_ser.inverse_series_trunc(ord), ord)
            # logger.debug("num=%s, den=%s", num, den)
        logger.debug("integral bound=%s", res)
        return res

    def series0(self, ord):
        Pol = self._Poly_IC # XXX should be IR eventually
        res = Pol.zero()
        for num, den_facto in self.fracs:
            den = prod((lin**mult for lin, mult in den_facto), Pol.one()) #slow
            res += num._mul_trunc_(den.inverse_series_trunc(ord), ord)
        return res

    def __mul__(self, pol):
        """
        Multiplication by a polynomial.

        Note that this does not change the radius of convergence.
        """
        assert isinstance(pol, Polynomial)
        return RationalMajorant([(pol*num, den) for num, den in self.fracs])

class HyperexpMajorant(MajorantSeries):
    r"""
    A formal power series of the form rat1(z)·exp(int(rat2(ζ), ζ=0..z)), with
    nonnegative coefficients.

    The fraction rat1 is represented in the form z^shift*num(z)/den(z).

    TESTS::

        sage: from ore_algebra.analytic.bounds import *
        sage: Pol.<z> = RBF[]
        sage: one = Pol.one().factor()
        sage: den0 = Factorization([(1-z,1)])
        sage: integrand = RationalMajorant([(4+4*z, one), (z^2, den0)])
        sage: den = Factorization([(1/3-z, 1)])
        sage: maj = HyperexpMajorant(integrand, Pol.one(), den); maj
        (1.00... * (-z + [0.333...])^-1)*exp(int(4.0...
                                                + 4.0...*z + z^2/(-z + 1.0...)))
        sage: maj.cvrad
        [0.333...]
        sage: maj.bound_series(0, 4)
        ([336.000...])*z^3 + ([93.000...])*z^2 + ([21.000...])*z + [3.000...]
        sage: maj._test()
        sage: maj*=z^20
        sage: maj
        (z^20*1.00... * (-z + [0.333...])^-1)*exp(int(4.000...
                                            + 4.000...*z + z^2/(-z + 1.000...)))
        sage: maj._test()
    """

    # The choice of having the integral start at zero (i.e., choosing the
    # exponential part that is equal to one at 0, instead of a constant
    # multiple) is arbitrary, in the sense that the exponential part appearing
    # in the “homogeneous” part of the majorant will be compensated by the one
    # in the denominator of the integrand in the variation-of-constants
    # formula. Of course, the choice needs to be consistent.

    def __init__(self, integrand, num, den, shift=0):
        assert isinstance(integrand, RationalMajorant)
        assert isinstance(num, Polynomial)
        assert isinstance(den, Factorization)
        assert isinstance(shift, int) and shift >= 0
        cvrad = integrand.cvrad.min(_zero_free_rad([pol for (pol, m) in den]))
        super(self.__class__, self).__init__(integrand.variable_name, cvrad)
        self.integrand = integrand
        self.num = num
        self.den = den
        self.shift = shift

    def __repr__(self):
        if self.shift > 0:
            shift_part = "{}^{}*".format(self.num.variable_name(), self.shift)
        else:
            shift_part = ""
        return "({}{})*exp(int({}))".format(shift_part, (~self.den)*self.num,
                                                                self.integrand)

    @cached_method
    def _den_expanded(self):
        return prod(pol**m for (pol, m) in self.den)

    def inv_exp_part_series0(self, ord):
        # This uses the fact that the integral in the definition of self starts
        # at zero!
        return (- self.integrand.series0(ord-1)).integral()._exp_series(ord)

    # XXX: needs testing!
    def exp_part_coeffs_lbounds(self):
        r"""
        Return a lower bound on the coefficients of the series expansion at the
        origin *of the exponential part*.

        The output is a generator yielding (index, coeff lower bound) pairs.

        It works by finding a “minorant series” for the asymptotically dominant
        term of the integrand. In the simplest case, the minorant series is
        generated by a two-term recurrence. (We could get tighter bounds at a
        higher cost by repeating the process for other terms and multiplying
        the resulting bounds, and/or using higher-order recurrences in more
        cases while still avoiding subtractions.)

        Like with inv_exp_part_series0, the rational part of the majorant is
        ignored, and the returned bound makes use of the fact that the
        integration bounds are chosen so that the exponential part is 1 at 0.
        """
        if self.integrand.cvrad.is_infinity():
            if not self.integrand.fracs:
                return(((n, IR.zero()) for n in itertools.count()))
            else:
                return self._exp_part_coeff_lbound_entire()
        else:
            return self._exp_part_coeff_lbound_finite_radius()

    def _exp_part_coeff_lbound_entire(self):

        # lb[0] = 1,
        # n·lb[n] = num[0]·lb[n-1] + ··· + num[s-1]·lb[n-s]
        num, den = self.integrand.fracs[-1]
        length = num.degree() + 1
        assert den.value().is_one()

        last = collections.deque([IR.zero()]*(length + 1))
        last[0] = IR.one()
        yield 0, last[0]

        for n in itertools.count(1):
            last.rotate(1)
            last[0] = sum(b*last[k+1] for k, b in enumerate(num))/IR(n)
            yield n, last[0]

        # After a while, the “dominant” term should be enough. This is
        # difficult to get right, tough, because of “hump” effects when the
        # non-dominant terms have comparatively large coefficients. (In any
        # case, we should at least wait until n >= length, so that the lower
        # bound is nonzero for every nonzero term.)

        # # Find the term of highest degree d with a
        # # non-negligible coefficient, for a bound ~1/n!^(1/d).
        # thr = IR('-inf').max(*iter(num)) >> (IR.precision()//2)
        # for d in range(length - 1, -1, -1):
        #     if safe_gt(num[d], thr):
        #         shift = d
        #         break
        #
        # for n in itertools.count(length):
        #     last.rotate(1)
        #     last[0] = (num[shift]*last[shift+1])/IR(n)
        #     yield n, last[0]

        return

    def _exp_part_coeff_lbound_finite_radius(self):

        # lb[0] = 1,
        # n*lb[n] = α·(n-1)·lb[n-1] + num[0]·lb[n-1]
        #                     + ··· + num[s-1]·lb[n-s]
        num, _ = self.integrand.fracs[-1]
        alpha = 1/self.integrand.cvrad

        # The term involving α vanishes for n = 1. Some terms lb[1], lb[2],
        # may be zero (typically *will* be zero, due to the use of polynomial
        # parts). We want strictly positive lower bounds, so we compute
        # these first few terms separately. (It may also be possible to avoid
        # the issue by changing the algorithm to use a slightly different
        # minorant. This remains to be investigated.)
        start = num.valuation()
        ser = self.integrand.series0(start - 1)
        initial_terms = ser.integral()._exp_series(start)
        for n in range(start):
            val = initial_terms[n].below_abs()
            yield n, val

        # Then we have lb[start] ≥ (num[start]/(start+1))lb[0] > 0.
        cur = num[start]/(start + 1)
        yield start, cur

        # From there on, lb[n] is nonzero, and the inequality
        # n·lb[n] ≥ α·(n-1)·lb[n-1] yields a nontrivial lower bound.
        for n in itertools.count(start + 1):
            cur *= alpha*(1-1/IR(n+1))
            yield n, cur

    def bound_series(self, rad, ord):
        r"""
        Numeric bounds on the values at rad of this majorant series and of its
        first few derivatives (up to factorials).

        ALGORITHM:

        [M19, Algorithm 8.1]

        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.bounds import DiffOpBound
            sage: Dops, x, Dx = DifferentialOperators()
            sage: maj = DiffOpBound(Dx-1)(10)
            sage: maj.bound(RBF(1000))
            [1.97007111401...e+434 +/- ...]
        """
        # Compute the derivatives “by automatic differentiation”. This is
        # crucial for performance with operators of large order.
        Pol = PolynomialRing(IC, self.variable_name) # XXX: should be IR
        pert_rad = Pol([rad, 1])
        shx_ser = pert_rad.power_trunc(self.shift, ord)
        num_ser = Pol(self.num).compose_trunc(pert_rad, ord) # XXX: remove Pol()
        den_ser = Pol(self._den_expanded()).compose_trunc(pert_rad, ord)
        if den_ser[0].contains_zero():
            # we will never obtain a finite bound using this result
            raise BoundPrecisionError
        assert num_ser.parent() is den_ser.parent()
        rat_ser = (shx_ser._mul_trunc_(num_ser, ord)
                          ._mul_trunc_(den_ser.inverse_series_trunc(ord), ord))
        # Majorant series for the integral. Note that we need the constant term
        # here, since we assume in inv_exp_part_series0 and elsewhere that the
        # exponential part is one at rad=0.
        int_ser = self.integrand.bound_integral(rad, ord)
        exp_ser = int_ser._exp_series(ord)
        ser = rat_ser._mul_trunc_(exp_ser, ord)
        return ser

    def _saddle_point_bound(self, rad, ord, n, aux_rad):
        r"""
        Compute a termwise bound on the first ord derivatives of the *remainder
        of order start_index* of this majorant series, using evaluations at
        aux_rad.
        """
        assert safe_le(rad, aux_rad)
        assert safe_le(aux_rad, self.cvrad)
        ser = self.bound_series(aux_rad, ord)
        ratio = rad/aux_rad
        eps = ser.parent().gen()
        return ratio**n * ser(eps/ratio)

    def bound_tail_series(self, rad, ord, n):
        r"""
        Compute a termwise bound on the first ord derivatives of the *remainder
        of order start_index* of this majorant series.

        ALGORITHM:

        [M19, Section 8.2]
        """
        import numpy
        from sage.numerical.optimize import find_local_minimum
        b0 = self.bound_series(rad, ord)
        if n < self.shift:
            return b0
        def bound(r):
            r = IR(numpy.real(r))
            return self._saddle_point_bound(rad, 1, n, r)[0].log().mid()
        right = (float(rad)*float(n) if self.cvrad.is_infinity()
                 else self.cvrad.lower())
        _, aux_rad = find_local_minimum(bound, rad.upper(), right, tol=.125)
        aux_rad = IR(numpy.real(aux_rad))
        b1 = self._saddle_point_bound(rad, ord, n, aux_rad)
        Ser = b1.parent().change_ring(IR)
        ser = Ser([a0.above_abs().min(a1.above_abs())
                  for a0, a1 in zip(b0, b1)])
        return ser

    def __imul__(self, pol):
        r"""
        IN-PLACE multiplication by a polynomial. Use with care!

        Note that this does not change the radius of convergence.
        """
        valuation = pol.valuation() if pol else 0
        self.shift += valuation
        self.num *= (pol >> valuation)
        return self

    def __irshift__(self, n):
        r"""
        IN-PLACE multiplication by x^n. Use with care!
        """
        self.shift += n
        return self

######################################################################
# Majorants for reciprocals of polynomials (“denominators”)
######################################################################

def graeffe(pol):
    r"""
    Compute the Graeffe iterate of a polynomial.

    EXAMPLES::

        sage: from ore_algebra.analytic.bounds import graeffe
        sage: Pol.<x> = QQ[]

        sage: pol = 6*x^5 - 2*x^4 - 2*x^3 + 2*x^2 + 1/12*x^2^2
        sage: sorted(graeffe(pol).roots(CC))
        [(0.000000000000000, 2), (0.110618733062304 - 0.436710223946931*I, 1),
        (0.110618733062304 + 0.436710223946931*I, 1), (0.547473953628478, 1)]
        sage: sorted([(z^2, m) for z, m in pol.roots(CC)])
        [(0.000000000000000, 2), (0.110618733062304 - 0.436710223946931*I, 1),
        (0.110618733062304 + 0.436710223946931*I, 1), (0.547473953628478, 1)]

    TESTS::

        sage: graeffe(CIF['x'].zero())
        0
        sage: graeffe(RIF['x'](-1/3))
        0.1111111111111111?
    """
    deg = pol.degree()
    Parent = pol.parent()
    pol_even = Parent([pol[2*i] for i in range(deg//2+1)])
    pol_odd = Parent([pol[2*i+1] for i in range(deg//2+1)])
    pm = -1 if deg % 2 else 1
    graeffe_iterate = pm * (pol_even**2 - (pol_odd**2).shift(1))
    return graeffe_iterate

def abs_min_nonzero_root(pol, tol=RR(1e-2), min_log=RR('-inf'), prec=None):
    r"""
    Compute an enclosure of the absolute value of the nonzero complex root of
    ``pol`` closest to the origin.

    INPUT:

    - ``pol`` -- Nonzero polynomial.

    - ``tol`` -- An indication of the required relative accuracy (interval
      width over exact value). It is currently *not* guaranteed that the
      relative accuracy will be smaller than ``tol``.

    - ``min_log`` -- Return a bound larger than ``2^min_log``. The function
      may loop if there is a nonzero root of modulus bounded by that value.

    - ``prec`` -- working precision.

    ALGORITHM:

    Essentially the method of Davenport & Mignotte (1990).

    EXAMPLES::

        sage: from ore_algebra.analytic.bounds import abs_min_nonzero_root
        sage: Pol.<z> = QQ[]
        sage: pol = 1/10*z^3 + z^2 + 1/7
        sage: sorted(z[0].abs() for z in pol.roots(CC))
        [0.377695553183559, 0.377695553183559, 10.0142451007998]

        sage: abs_min_nonzero_root(pol)
        [0.38 +/- 3.31e-3]

        sage: abs_min_nonzero_root(pol, tol=1e-10)
        [0.3776955532 +/- 2.41e-11]

        sage: abs_min_nonzero_root(pol, min_log=-1.4047042967)
        [0.3776955532 +/- 2.41e-11]

        sage: abs_min_nonzero_root(pol, min_log=-1.4047042966)
        Traceback (most recent call last):
        ...
        ValueError: there is a root smaller than 2^(-1.40470429660000)

        sage: abs_min_nonzero_root(pol, tol=1e-50)
        [0.3776955531835593496507263902642801708344727099333...]

        sage: abs_min_nonzero_root(Pol.zero())
        Traceback (most recent call last):
        ...
        ValueError: expected a nonzero polynomial

    TESTS::

        sage: abs_min_nonzero_root(CBF['x'].one())
        +Infinity
        sage: abs_min_nonzero_root(CBF['x'].gen())
        +Infinity
        sage: abs_min_nonzero_root(CBF['x'].gen() - 1/3)
        [0.33 +/- 3.34e-3]

    An example where the ability to increase the precision is used::

        sage: import logging; logging.basicConfig()
        sage: logger = logging.getLogger('ore_algebra.analytic.bounds')
        sage: logger.setLevel(logging.DEBUG)

        sage: abs_min_nonzero_root(z^7 - 2*(1000*z^2-1), tol=1e-20)
        DEBUG:ore_algebra.analytic.bounds:failed to bound the roots...
        [0.03162277660143379332...]

    And one where that used to be the case::

        sage: from ore_algebra import DifferentialOperators
        sage: from ore_algebra.analytic.bounds import DiffOpBound
        sage: Dops, x, Dx = DifferentialOperators()
        sage: dop = (x^2 + 10*x + 50)*Dx^2 + Dx + 1
        sage: maj = DiffOpBound(dop, bound_inverse="simple")
        INFO:ore_algebra.analytic.bounds:bounding local operator (simple, pol_part_len=None, max_effort=2)...

        sage: logger.setLevel(logging.WARNING)
    """
    if prec is None:
        prec = IR.precision()
    tol = RealField(prec)(tol)
    myIR = type(IR)(prec)
    myRIF = type(RIF)(prec) # XXX: could use balls with recent arb (> intersect)
    if pol.is_zero():
        raise ValueError("expected a nonzero polynomial")
    pol >>= pol.valuation()
    deg = pol.degree()
    if deg == 0:
        return infinity
    pol = pol/pol[0]
    mypol = pol.change_ring(myIR.complex_field())
    i = 0
    lg_rad = myRIF(-infinity, infinity)        # left-right intervals because we
    encl = myRIF(1, 2*deg).log(2)              # compute intersections
    neg_infty = myRIF('-inf')
    while (safe_le(lg_rad.lower(rnd='RNDN'), min_log)
              # *relative* error on 2^lg_rad
           or safe_gt(lg_rad.absolute_diameter(), tol)):
        prev_lg_rad = lg_rad
        # The smallest root of the current mypol is between 2^(-1-m) and
        # (2·deg)·2^(-1-m), cf. Davenport & Mignotte (1990), Grégoire (2012).
        m = myRIF(-infinity).max(*(myRIF(_log2abs(mypol[k])/k)
                                for k in range(1, deg+1)))
        lg_rad = (-(1 + m) + encl) >> i
        lg_rad = prev_lg_rad.intersection(lg_rad)
        stalled = (lg_rad.endpoints() == prev_lg_rad.endpoints())
        if (neg_infty in lg_rad or lg_rad.is_NaN() or stalled):
            prec *= 2
            logger.debug("failed to bound the roots of %s, "
                    "retrying with prec=%s bits", pol, prec)
            return abs_min_nonzero_root(pol, tol, min_log, prec)
        logger.log(logging.DEBUG - 1, "i = %s\trad ∈ %s\tdiam=%s",
                i, lg_rad.exp2().str(style='brackets'),
                lg_rad.absolute_diameter())
        # detect gross input errors (this does not prevent all infinite loops)
        if safe_le(lg_rad.upper(rnd='RNDN'), min_log):
            raise ValueError("there is a root smaller than 2^({})"
                             .format(min_log))
        mypol = graeffe(mypol)
        i += 1
    res = myIR(2)**myIR(lg_rad)
    if not safe_le(2*res.rad_as_ball()/res, myIR(tol)):
        logger.debug("required tolerance may not be met")
    return res

def growth_parameters(dop):
    r"""
    Find κ, α such that the solutions of dop grow at most like
    sum(α^n*x^n/n!^κ) ≈ exp(κ*(α·x)^(1/κ)).

    EXAMPLES::

        sage: from ore_algebra import *
        sage: DiffOps, x, Dx = DifferentialOperators()
        sage: from ore_algebra.analytic.bounds import growth_parameters
        sage: growth_parameters(Dx^2 + 2*x*Dx) # erf(x)
        (1/2, [1.4...])
        sage: growth_parameters(Dx^2 + 8*x*Dx) # erf(2*x)
        (1/2, [2.8...])
        sage: growth_parameters(Dx^2 - x) # Airy
        (2/3, [1.0...])
        sage: growth_parameters(x*Dx^2 + (1-x)*Dx) # Ei(1, -x)
        (1, [1.0...])
        sage: growth_parameters((Dx-1).lclm(Dx-2))
        (1, [2.0...])
        sage: growth_parameters((Dx - x).lclm(Dx^2 - 1))
        (1/2, [1.0...])
        sage: growth_parameters(x^2*Dx^2 + x*Dx + 1)
        (+Infinity, 0)
    """
    assert dop.leading_coefficient().is_term()
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
    points = [(ZZ(j-i), ZZ(i), c) for (i, pol) in enumerate(dop)
                                  for (j, c) in enumerate(pol)
                                  if not c.is_zero()]
    h0, i0, _ = max(points, key=lambda p: (p[1], p[0]))
    hull = [(h, i, c) for (h, i, c) in points if h > h0 and i < i0]
    if not hull: # generalized polynomial
        return infinity, ZZ.zero()
    slope = max((i-i0)/(h-h0) for h, i, c in hull)
    Pol = dop.base_ring()
    eqn = Pol({i0 - i: c for (h, i, c) in points if i == i0 + slope*(h-h0)})
    expo_growth = abs_min_nonzero_root(eqn)**slope
    return -slope, expo_growth

######################################################################
# Bounds on rational functions of n
######################################################################

# key=... to avoid comparing number fields
# XXX: tie life to a suitable object
@cached_function(key=lambda p: (id(p.parent()), p))
def _complex_roots(pol):
    if not pol.parent() is QQ: # QQ typical (ordinary points)
        pol = pol.change_ring(QQbar)
    return [(IC(rt), mult) for rt, mult in pol.roots(myCIF)]

# Possible improvements:
# - better take into account the range of derivatives needed at each step,
# - allow ord to vary?

class RatSeqBound(object):
    r"""
    Bounds on the tails of a.e. rational sequences and their derivatives.

    Consider a vector of rational sequences sharing the same denominator, ::

        f(n) = nums(n)/den(n) = [num[i](n)/den(n)]_i.

    We assume that den is monic and deg(nums) < deg(den). Let ::

                  ⎧ f^(t)(n),                       n ∉ exceptional_indices,
        F[t](n) = ⎨ (d/dX)^t(num(n+X)/(X^{-m}·den(n+X)))(X=0)),
                  ⎩                                 exceptional_indices[n] = m

                    (the first formula is the specialization to m = 0 of the
                    second one),

        ref(n, ord) = sum[t=0..ord-1](|n*F[t](n)/t!|),

        τ(n) = sum[k=-∞..n](exceptional_indices[k]).

    An instance of this class represents a vector of bounds b(n) such that ::

        ∀ k ≥ n,   |ref(k, τ(k))| ≤ b(n)  (componentwise).

    The bounds are *not* guaranteed to be nonincreasing.

    Such bounds appear as coefficients in the parametrized majorant series
    associated to differential operators, see :class:`DiffOpBound`. The
    ability to bound a sum of derivatives rather than a single rational
    function is useful to support logarithmic solutions at regular singular
    points. Vectors of bounds are supported purely for performance reasons,
    to avoid redundant computations on the indices and denominators.

    ALGORITHM:

    This version essentially bounds the numerators (from above) and the
    denominator (from below) separately. This simple strategy works well in the
    typical case where the indicial equation has only small roots, and makes it
    easy to share part of the computation over a vector of bounds. In the
    presence of, e.g., large real roots, however, it is not much better than
    waiting to get past the largest root.

    See the git history for a tighter but more expensive alternative.

    EXAMPLES::

        sage: Pols.<n> = QQ[]
        sage: from ore_algebra.analytic.bounds import RatSeqBound

        sage: bnd = RatSeqBound([Pols(1)], n*(n-1)); bnd
        bound(1/(n^2 - n), ord≤1)
            = +infinity, +infinity, 1.0000, 0.50000, 0.33333, 0.25000, 0.20000,
            0.16667, ..., ~1.00000*n^-1
        sage: [bnd(k)[0] for k in range(5)]
        [[+/- inf], [+/- inf], [1.000...], [0.500...], [0.333...]]
        sage: bnd._test()
        sage: bnd.plot()
        Graphics object...

        sage: bnd = RatSeqBound([-n], n*(n-3), {-1: 1, 0:1, 3:1}); bnd
        bound(-1/(n - 3), ord≤3)
            = 74.767, 74.767, 22.439, 12.000, 12.000, 4.3750, 2.8889, 2.2969,
            ..., ~1.00000
            [74.7...]    for  n <= 0,
            [12.0...]    for  n <= 3
        sage: [(bnd.ref(k, bnd.ord(k))[0], bnd(k)[0]) for k in range(5)]
        [(0,          [74.767...]),
         (0.750...,   [74.767...]),
         (4.000...,   [22.439...]),
         ([3.000...], [12.000...]),
         (12.000...,  [12.000...])]
        sage: bnd._test()

        sage: RatSeqBound([n], n, {})
        Traceback (most recent call last):
        ...
        ValueError: expected deg(num) < deg(den)

        sage: bnd = RatSeqBound([n^5-100*n^4+2], n^3*(n-1/2)*(n-2)^2,
        ....:                   {0:3, 2:2})
        sage: bnd._test(200)
        sage: bnd.plot()
        Graphics object...

        sage: bndvec = RatSeqBound([n, n^2, n^3], (n+1)^4, {-1: 1})
        sage: for bnd in bndvec:
        ....:     bnd._test()

    TESTS::

        sage: RatSeqBound([Pols(3)], n)(10)
        [3.000...]
        sage: QQi.<i> = QuadraticField(-1, 'i')
        sage: RatSeqBound([Pols(1)], n+i)._test()
        sage: RatSeqBound([-n], n*(n-3), {3:1})._test()
        sage: RatSeqBound([-n], n*(n-3), {0:1})._test()
        sage: RatSeqBound([-n], n*(n-3), {0:1,3:1})._test()
        sage: RatSeqBound([CBF(i)*n], n*(n-QQbar(i)), {0:1})._test()
        sage: RatSeqBound([QQi['n'](3*i+1)], n + (i-1)/3, {-1: 1})._test()

        sage: from ore_algebra.analytic.bounds import _test_RatSeqBound
        sage: _test_RatSeqBound() # long time
        sage: _test_RatSeqBound(base=QQi, number=3, deg=3) # long time
    """

    def __init__(self, nums, den, exceptional_indices={-1: 1}):
        r"""
        INPUT:

        - den - polynomial with complex coefficients,
        - nums - list of polynomials with complex coefficients, each
          with deg(num) < deg(den);
        - exceptional_indices - dictionary {index: multiplicity},  typically
            - either {-1: 1}, corresponding to bounds on |nums(n)/den(n)| (for
              n≥ 0) that only become finite after the last integer zero of den
              (accepting negative indices here is a bit of a hack, but is
              convenient to test the bounds in simple cases...),
            - or the integer zeros of den, the context of evaluations at
              regular singular points.

        In the main application, den is the indicial equation of a differential
        operator and the nums are coefficients of related recurrence operators,
        both shifted so that some root of interest of the indicial equation is
        mapped to zero.
        """
        deg = den.degree()
        if any(num.degree() >= deg for num in nums):
            raise ValueError("expected deg(num) < deg(den)")
        self.nums = []
        self._ivnums = []
        self._rcpq_nums = []
        assert den.is_monic()
        self.den = den
        self._ivden = den.change_ring(IC)
        self._rcpq_den = den.change_ring(IC).reverse()
        self.exn = dict((int(n), int(m))
                        for n, m in exceptional_indices.items())
        # temporary(?), for compatibility with the previous version
        if not self.exn:
            self.exn = {0: 1}
        self._Pol = self._rcpq_den.parent()
        self._pol_class = self._Pol.Element
        self.extend(nums)

    def extend(self, nums):
        r"""
        Add new sequences to this bound, without changing the rest of the data.

        Use with care!
        """
        self.nums.extend(nums)
        ivnums = [num.change_ring(IC) for num in nums]
        self._ivnums.extend(ivnums)
        deg = self.den.degree()
        # rcpq_num/rcpq_den = (1/n)*rat(1/n)
        self._rcpq_nums.extend([num.reverse(deg-1) for num in ivnums])
        self._stairs.clear_cache()

    def __len__(self):
        return len(self.nums)

    def entries_repr(self, type):
        if type == "asympt":
            fmt = "{asympt}"
        elif type == "short":
            fmt = "bound({rat}, ord≤{ord})"
        elif type == "full":
            fmt  = "bound({rat}, ord≤{ord})\n"
            fmt += "    = {list},\n"
            fmt += "      ..., {asympt}"
            fmt += "{stairs}"
        n = self.den.variable_name()
        bnds = list(zip(*(self(k) for k in range(8))))
        stairs = self._stairs(len(self))
        dscs = []
        assert len(self.nums) == len(bnds) == len(stairs)
        for (num, bnd, seq_stairs) in zip(self.nums, bnds, stairs):
            lim = abs(ComplexBallField(20)(num.leading_coefficient()))
            deg = num.degree() - self.den.degree() + 1
            asymptfmt = "~{lim}" if deg == 0 else "~{lim}*n^{deg}"
            stairsstr = ',\n'.join(
                    ["    {}\tfor  {} <= {}".format(val, n, edge)
                     for edge, val in seq_stairs])
            dscs.append(
                fmt.format(
                    rat=num/self.den,
                    ord=sum(m for (n, m) in self.exn.items()),
                    list=", ".join(str(b.n(20)) for b in bnd),
                    asympt=asymptfmt.format(lim=lim, deg=deg),
                    stairs=stairsstr if seq_stairs else ""))
        return dscs

    def __repr__(self):
        return "\n".join(self.entries_repr("full"))

    def __getitem__(self, i):
        return RatSeqBound([self.nums[i]], self.den, self.exn)

    @cached_method
    def _den_data(self):
        r"""
        Shared part of the computation of _lbound_den(n) for varying n.

        OUTPUT:

        A lower bound on self.den/n^r (where r = deg(self.den)) in the format
        that _lbound_den expects. That is, a list of tuples (root, mult, n_min,
        global_lbound) where
        - root ranges over a subset of the roots of den;
        - mult is the multiplicity of root in den;
        - n_min is an integer s.t. |1-root/n| is nondecreasing for n ≥ nmin;
        - global_lbound is a real (ball) s.t. |1-root/n|**mult ≥ global_lbound
          for all n ∈ ⟦1,∞) ∖ exn (in particular, for n < n_min).

        Often (but not always), all integer roots of den will belong to the
        exceptional set, and in this case the returned global_lbound will be
        strictly positive.

        TESTS::

            sage: from ore_algebra.analytic.bounds import RatSeqBound
            sage: Pols.<n> = QQ[]
            sage: num = -2*n^9 - n^8 + 2*n^6 - 16*n^5 - 1/8*n^3 - n^2 - 11*n - 3/4
            sage: den = (n^13 - 12727/82*n^12 + 14847/656*n^11 + 1865161/1312
            ....: *n^10 + 1678935/2624*n^9 - 14402125/5248*n^8 - 7555021/5248*
            ....: n^7 + 2333839/1312*n^6 + 612159/656*n^5 - 1590097/5248*n^4 -
            ....: 830505/5248*n^3 + 5375/2624*n^2 + 795/656*n)
            sage: exns = {-1: 1}
            sage: bnd = RatSeqBound([num], den, exns)
            sage: bnd._test()
        """
        den_data = []
        for root, mult in _complex_roots(self.den):
            re = root.real()
            # When Re(α) ≤ 0, the sequence |1-α/n| decreases to 1.
            if safe_le(re, IR.zero()):
                continue
            # Otherwise, it first decreases to its minimum (which may be 0 if α
            # is an integer), then increases to 1. We compute the minimum and a
            # value of n after which the sequence is nondecreasing.
            crit_n = root.abs()**2/re
            if crit_n.diameter() > 16:
                # This includes the case where re contains zero.
                n_min = sys.maxsize # infinity
                # Lower bound valid for all real n > 0, i.e., not taking
                # account the discrete effects.
                global_lbound = (root.imag()/root.abs()).below_abs()
            else:
                ns = list(range(crit_n.lower().floor(), crit_n.upper().ceil() + 1))
                n_min = ns[-1]
                # We skip exceptional indices among the candidates in the
                # computation of the global lower bound, and consider the
                # adjacent integers above and below instead. In particular,
                # when the current root is equal to an exceptional index, the
                # global minimum over ℕ is zero, but we want a nonzero lower
                # bound over ℕ ∖ exn. There can be several consecutive
                # exceptional indices (this is even quite typical).
                while ns[-1] in self.exn:
                    ns.append(ns[-1] + 1) # append to avoid overwriting ns[0]
                while ns[0] in self.exn:
                    ns[0] -= 1
                global_lbound = IR.one().min(*(
                        (IC.one() - root/n).abs()
                        for n in ns if n >= 1 and not n in self.exn))
            global_lbound = global_lbound.below_abs()**mult # point ball
            den_data.append((root, mult, n_min, global_lbound))
        return den_data

    def _lbound_den(self, n):
        r"""
        A lower bound on prod[den(α) = 0](|1-α/k|) valid for all k ≥ n with
        n, k ∈ ℕ ∖ exn.

        Reference: [M19, Lemma 7.2]
        """
        assert n not in self.exn
        if n == 0:
            return IR.zero() # _den_data() assumes n ≥ 1
        res = IR.one()
        for root, mult, n_min, global_lbound in self._den_data():
            if n < n_min:
                # note that global_lbound already takes mult into account
                res *= global_lbound
            else:
                res *= abs((IC.one() - root/n))**mult
        return res

    def _bound_rat(self, n, ord, tight=None):
        r"""
        A componentwise bound on the vector ref[ord](k), valid for all k ≥ n
        with n, k ∉ exn.

        When ord = 1, this method simply evaluates the reciprocal polynomials
        of nums and den, rescaled by a suitable power of n, on an interval of
        the form [0,1/n]. (It works for exceptional indices, but doesn't do
        anything clever to take advantage of them.) More generally, a similar
        evaluation on an interval jet of the form [0,1/n] + ε + O(ε^ord)
        yields bounds for the derivatives as well.

        ALGORITHM:

        Essentially [M19, Algorithm 7.1].
        """
        assert n not in self.exn
        iv = IR.zero().union(~IR(n))
        # jet = 1/(n+ε) = n⁻¹/(1+n⁻¹ε)
        jet0 = self._pol_class(self._Pol, [IR.one(), iv])
        jet1 = jet0.inverse_series_trunc(ord)
        jet = iv*jet1
        # Most expensive part. Perhaps consider simplifying rcpq_num, rcpq_den
        # by bounding the high-degree terms for large n???
        nums = [num.compose_trunc(jet, ord) for num in self._rcpq_nums]
        den = self._rcpq_den.compose_trunc(jet, ord)
        invabscst = IR.one()
        if tight or tight is None and den[0].accuracy() < 0:
            # Replace the constant coefficient by a tighter bound (in
            # particular, one that should be finite even in the presence of
            # poles at exceptional or non-integer indices). More precisely,
            # since den has complex coefficients, we use the lower bound on the
            # absolute value of den(0) to compute a complex ball enclosing
            # 1/den(0), and multiply the computed den by this ball. We will
            # later multiply the complete bound by the same value.
            lb = self._lbound_den(n)
            invabscst = IR.zero().union(~lb)
            # invabscst = IR(~RIF(self._lbound_den(n).lower(), lb.upper()))
            invcst = IC.zero().add_error(invabscst)
            den = 1 + (invcst*(den >> 1) << 1)
            logger.debug("lb=%s, refined den=%s", lb, den)
        # num/den = invcst⁻¹·(n+ε)·f(1/(n+ε))
        # ser0 = (1+ε/n)⁻¹/den
        # ser = ser0·num = invcst⁻¹·n·f(n+ε)
        invden = den.inverse_series_trunc(ord)
        ser0 = jet1._mul_trunc_(invden, ord)
        bounds = []
        for num in nums:
            ser = ser0._mul_trunc_(num, ord)
            bound = (invabscst*sum(c.above_abs() for c in ser)).above_abs()
            # logger.debug(lazy_string(lambda: "bound[%s](%s) = %s = %s" % (
            # num, n, "+".join([str(invabscst*c.above_abs()) for c in ser]),
            # bound)))
            if not bound.is_finite():
                bound = IR(infinity) # replace NaN by +∞ (as max(NaN, 42) = 42)
            bounds.append(bound)
        return bounds

    @cached_method
    def _stairs(self, count):
        r"""
        Shared part of the computation of _bound_exn(n) for varying n.

        OUTPUT:

        A list whose element of index i is a list of pairs (edge, val), ordered
        by increasing edge, and such that |ref(n)[i]| ≤ val for all n ≥ edge.

        ALGORITHM:

        Part of [M19, Algorithm 7.4]
        """
        # consistency check, we need to recompute or at least extend the stairs
        # each time the sequence of numerators is extended
        assert count == len(self.nums)
        stairs = [[(infinity, IR.zero())] for _ in self.nums]
        ord = sum(m for n, m in self.exn.items())
        exn = sorted([n for n in self.exn if n >= 0], reverse=True)
        for n in exn:
            # We want the bound to hold for ordinary k ≥ n too, so we take the
            # max of the exceptional value at n and the value at n + 1, when
            # n + 1 is an ordinary index. (When n + 1 is an exceptional index,
            # it has already been done during the previous iteration.)
            refs = self.ref(n, ord)
            if n + 1 not in exn:
                rats = self._bound_rat(n + 1, ord)
            else:
                rats = [IR.zero()]*len(refs)
            assert len(refs) == len(rats) == len(stairs) == len(self.nums)
            for (ref, rat, seq_stairs) in zip(refs, rats, stairs):
                val = ref.max(rat)
                if val.upper() > seq_stairs[-1][1].upper():
                    seq_stairs.append((n, val))
            ord -= self.exn[n]
        for seq_stairs in stairs:
            seq_stairs.reverse()
            # remove (∞,0) (faster and permits testing "stairs == []")
            seq_stairs.pop()
        return stairs

    def _bound_exn(self, n):
        r"""
        A list of *non-increasing* staircase functions defined on the whole
        of ℕ such that, whenever *n* (sic) is an exceptional index, the
        inequality ref(k) ≤ _bound_exn(n) holds (componentwise) for all k ≥ n
        (both ordinary and exceptional).

        (The pairs returned by _stairs() correspond to the *upper right* corner
        of each stair: the index associated to a given value is the last time
        this value will be reached by the staircase function _bound_exn().
        One may well have |f[i](n)| > _bound_exn(n)[i] when n is ordinary.)

        ALGORITHM:

        Part of [M19, Algorithm 7.4]
        """
        # Return the value associated to the smallest step larger than n. (This
        # might be counter-intuitive!)
        def doit(seq_stairs):
            for (edge, val) in seq_stairs:
                if n <= edge:
                    return val
            return IR.zero()
        stairs = self._stairs(len(self.nums))
        return [doit(seq_stairs) for seq_stairs in stairs]

    def ord(self, n):
        return sum(m for (k, m) in self.exn.items() if k <= n)

    def __call__(self, n, tight=None):
        r"""
        The bounds.

        ALGORITHM:

        [M19, Algorithm 7.4]
        """
        bound_exn = self._bound_exn(n)
        if n in self.exn:
            return bound_exn
        else:
            # Note: we could accept an optional parameter ord ≤ self.ord(n) and
            # use it in the call to _bound_rat() to provide tighter bounds
            # (once n is larger than the exceptional indices) if we detect that
            # we are computing a solution of log-degree smaller than the
            # generic bound (like at ordinary points, except that in the case
            # of ordinary points we can tell in advance that this will happen).
            # In fact, we could even do it before the last exceptional index,
            # but that would complicate (and perhaps slow down) _stairs() and
            # _bound_exn().
            ord = self.ord(n)
            bound_rat = self._bound_rat(n, ord, tight)
            return [b1.max(b2) for b1, b2 in zip(bound_rat, bound_exn)]

    def ref(self, n, ord):
        r"""
        Reference value for a single n.
        """
        jet = self._pol_class(self._Pol, [n, 1])
        nums = [num.compose_trunc(jet, ord) for num in self._ivnums]
        mult = self.exn.get(n, 0)
        # den has a root of order mult at n, so den(pert) = O(X^mult), but the
        # computed value might include terms of degree < mult with interval
        # coefficients containing zero
        den = self._ivden.compose_trunc(jet, ord + mult) >> mult
        invden = den.inverse_series_trunc(ord)
        sers = [num._mul_trunc_(invden, ord) for num in nums]
        my_n = IR(n)
        return [my_n*sum((c.abs() for c in ser), IR.zero()) for ser in sers]

    def plot(self, rng=range(40), tight=None):
        r"""
        Plot this bound and its reference function.

        The vector of nums/bounds must have length one.

        EXAMPLES::

            sage: from ore_algebra.analytic.bounds import RatSeqBound
            sage: Pols.<n> = QQ[]
            sage: i = QuadraticField(-1).gen()
            sage: bnd = RatSeqBound(
            ....:     [CBF(i)*n+42], n*(n-3)*(n-i-20), {0:1,3:1})
            sage: bnd.plot()
            Graphics object consisting of ... graphics primitives
            sage: bnd.plot(range(30))
            Graphics object consisting of ... graphics primitives
        """
        if len(self.nums) != 1:
            raise NotImplementedError("expected a single sequence")
        from sage.plot.plot import list_plot
        inf = RR('inf')
        # avoid empty plots and matplotlib warnings
        def pltfilter(it):
            return [(x, RR(y)) for (x, y) in it if 0 < RR(y) < inf]
        ref = []
        for k in rng:
            iv = self.ref(k, self.ord(k))[0]
            if iv.is_finite():
                ref.append((k, iv.upper()))
        p  = list_plot(pltfilter(ref), plotjoined=True, color='black',
                       linestyle="--", scale='semilogy',
                       legend_label=r"ref.\ value")
        p += list_plot(
                pltfilter((k, self(k, tight=tight)[0].upper()) for k in rng),
                plotjoined=True, color='blue', scale='semilogy',
                legend_label="bound")
        p += list_plot(
                pltfilter((k,
                    self._bound_rat(k, self.ord(k), tight=False)[0].upper())
                    for k in rng if k not in self.exn),
                size=20, color="blue", marker='^', scale='semilogy',
                legend_label=r"$M_{\mathrm{gen}}$ (coarse)")
        p += list_plot(
                pltfilter((k,
                    self._bound_rat(k, self.ord(k), tight=True)[0].upper())
                    for k in rng if k not in self.exn),
                size=20, color='blue', marker="v", scale='semilogy',
                legend_label=r"$M_{\mathrm{gen}}$ (tight)")
        p += list_plot(
                pltfilter((k, self._bound_exn(k)[0].upper())
                          for k in rng),
                size=15, color='blue', marker="s", scale='semilogy',
                legend_label=r"$M_{\mathrm{exn}}$")
        m = max(rng)
        p += list_plot(
                pltfilter((e, v.upper())
                          for (e, v) in self._stairs(1)[0]
                          if e <= m),
                size=20, marker='x', color='blue', scale='semilogy',
                legend_label=r"$\{S(n)\}$")
        p.set_legend_options(handlelength=2, numpoints=3, shadow=False)
        return p

    # TODO: add a way to _test() all bounds generated during a given
    # computation
    def _test(self, nmax=100, kmax=10, ordmax=5):
        r"""
        Test that this bound is well-formed and plausibly does bound ref.

        The vector of nums/bounds must have length one.
        """
        if len(self.nums) != 1:
            raise NotImplementedError("expected a single sequence")
        deg = self.den.degree()
        # Well-formedness
        for n, mult in self.exn.items():
            if n >= 0:
                pol = self.den
                for i in range(mult):
                    assert pol(n).is_zero()
                    pol = pol.derivative()
        # Test _lbound_den()
        for n in range(nmax):
            if n not in self.exn:
                lb = self._lbound_den(n)
                assert not (lb*IR(n)**deg > IC(self.den(n)).abs())
                if n + 1 not in self.exn:
                    assert not (self._lbound_den(n+1) < lb)
        testrange = list(range(nmax)) + [nmax + (1 << k) for k in range(kmax)]
        testrange.reverse()
        # Test _bound_rat()
        ref = [IR(0) for _ in range(ordmax + 1)]
        for n in testrange:
            if n not in self.exn:
                rat = self.nums[0]/self.den
                ref_n = IR(0)
                for ord in range(ordmax + 1):
                    ref_n += rat(IC(n)).abs()/ZZ(ord).factorial()
                    ref[ord] = ref[ord].max(ref_n)
                    bound = self._bound_rat(n, ord+1)[0]
                    assert not (bound < ref[ord])
                    rat = rat.derivative()
        # Test the complete bound
        ref = IR(0)
        for n in testrange:
            n = ref.max(self.ref(n, self.ord(n))[0])
            assert not (self(n)[0] < ref)

@random_testing
def _test_RatSeqBound(number=10, base=QQ, deg=20, verbose=False):
    r"""
    Randomized testing helper.

    EXAMPLES::

        sage: from ore_algebra.analytic.bounds import _test_RatSeqBound
        sage: _test_RatSeqBound(number=1, deg=4, verbose=True, seed=0)
        num = 1/6
        den = n^4 - 7/8*n^3 - 5/16*n^2 + 3/16*n
        exns = {-1: 1}

    """
    from sage.combinat.subset import Subsets
    Pols, n = PolynomialRing(base, 'n').objgen()
    PolsZ = PolynomialRing(ZZ, 'n')
    assert deg >= 1
    for _ in range(number):
        dlin = ZZ.random_element(deg) # < deg
        drnd = ZZ.random_element(1, deg - dlin + 1)
        dnum = ZZ.random_element(dlin + drnd)
        num = Pols.random_element(degree=dnum)
        den0 = prod((PolsZ.random_element(degree=1) for _ in range(dlin)),
                    PolsZ.one())
        den = (den0 * Pols.random_element(degree=drnd)).monic()
        try:
            roots = den.numerator().roots(ZZ)
        except (TypeError, NotImplementedError):
            # If sage is unable to find the roots over this base ring, test
            # with the part that is guaranteed to factor completely over ℤ.
            roots = den0.roots(ZZ)
        roots = [(r, m) for (r, m) in roots if r >= 0]
        exns = dict(Subsets(roots).random_element())
        if not exns:
            exns = {-1: 1}
        if verbose:
            print("num = {}\nden = {}\nexns = {}".format(num, den, exns))
        bnd = RatSeqBound([num], den, exns)
        bnd._test()

################################################################################
# Bounds for differential equations
################################################################################

def bound_polynomials(pols, Poly=None):
    r"""
    Compute a common majorant polynomial for the polynomials in ``pol``.

    Note that this returns a *majorant*, not some kind of enclosure.

    TESTS::

        sage: from ore_algebra.analytic.bounds import bound_polynomials
        sage: Pol.<z> = PolynomialRing(QuadraticField(-1, 'i'), sparse=True)
        sage: bound_polynomials([(-1/3+z) << (10^10), (-2*z) << (10^10)])
        2.000...*z^10000000001 + [0.333...]*z^10000000000
        sage: bound_polynomials([Pol(0)])
        0
        sage: bound_polynomials([])
        Traceback (most recent call last):
        ...
        IndexError: list index out of range
    """
    assert isinstance(pols, list)
    if Poly is None:
        Poly = pols[0].parent()
    PolyIR = Poly.change_ring(IR)
    if not pols:
        return PolyIR.zero()
    deg = max(pol.degree() for pol in pols)
    val = min(deg, min(pol.valuation() for pol in pols))
    order = len(pols)
    def coeff_bound(n):
        return IR.zero().max(*(
            IC(pols[k][n]).above_abs()
            for k in range(order)))
    maj = PolyIR([coeff_bound(n) for n in range(val, deg + 1)])
    maj <<= val
    return maj

class DiffOpBound(object):
    r"""
    A “bound” on the “inverse” of a differential operator at a regular point.

    A DiffOpBound may be thought of as a sequence of formal power series

        v[n](z) = 1/den(z) · exp ∫ (pol[n](z) + cst·z^ℓ·num[n](z)/den(z))

    where

    * cst is a real number,
    * den(z) is a polynomial with constant coefficients,
    * pol[n](z) and num[n](z) are polynomials with coefficients depending on n
      (given by RatSeqBound objects), and ℓ >= deg(pol[n]).

    These series can be used to bound the tails of logarithmic power series
    solutions y(z) of dop(y) = 0 belonging to a certain subspace (see the
    documentation of __init__() for details). More precisely, write

        y(z) - ỹ(z) = z^λ·(u[0](z)/0! + u[1](z)·log(z)/1! + ···)

    where y(z) is a solution of self.dop (in the correct subspace, with
    λ = self.leftmost) and ỹ(z) is its truncation to O(z^n1). Then, for
    suitable n0 ∈ ℕ and p(z) ∈ ℝ_+[z], the series ŷ(z) = v[n0](z)·p(z) is a
    common majorant of u[0], u[1], ...

    In the typical situation where n0 ≤ n1 and y(z) does not depend on initial
    conditions “past” n1, a polynomial p(z) of valuation at least n1 with this
    property can be computed using the methods :meth:`normalized_residual()`
    and :meth:`rhs()`.
    Variants with different p hold in more general settings. See the
    documentation of normalized_residual() and rhs() for more information.

    Note that multiplying dop by a rational function changes p(z).

    DiffOpBounds are refinable: calling the method :meth:`refine()` will try to
    replace the parametrized series v[n](z) by one giving tighter bounds. The
    main effect of refinement is to increase the degree of the polynomial part.
    This can be done several times, but repeated calls to refine() quickly
    become expensive.

    EXAMPLES::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.bounds import *
        sage: Dops, x, Dx = DifferentialOperators()

    A majorant sequence::

        sage: maj = DiffOpBound((x^2 + 1)*Dx^2 + 2*x*Dx, pol_part_len=0)
        sage: print(maj.__repr__(asympt=False))
        1.000.../((-x + [0.994...])^2)*exp(int(POL+1.000...*NUM/(-x + [0.994...])^2))
        where
        POL=0,
        NUM=bound(0, ord≤1)*x^0 +
        bound((-2.000...*n + 2.000...)/(n^2 - n), ord≤1)*x^1

    A majorant series extracted from that sequence::

        sage: maj(3)
        (1.00... * (-x + [0.994...])^-2)*exp(int([3.000...]...)^2)))

    An example with a nontrivial polynomial part::

        sage: dop = (x+1)*(x^2+1)*Dx^3-(x-1)*(x^2-3)*Dx^2-2*(x^2+2*x-1)*Dx
        sage: DiffOpBound(dop, pol_part_len=3)
        1.000.../((-x + [0.99...])^3)*exp(int(POL+1.000...*NUM/(-x + [0.99...])^3)) where
        POL=~6.000...*x^0 + ~3.000...*x^1 + ~5.000...*x^2,
        NUM=~7.000...*x^3 + ~2.000...*x^4 + ~5.000...*x^5

    Refining::

        sage: from ore_algebra.examples import fcc
        sage: maj = DiffOpBound(fcc.dop5, special_shifts=[(0, 1)])
        sage: maj.maj_den
        (-z + [0.2047...])^13
        sage: maj.maj_den.value()(1/10)
        [1.825...]
        sage: maj.refine()
        sage: maj.maj_den.value()(1/10)
        [436565.0...]
        sage: maj.majseq_pol_part(10)
        [[41.256...], [188.43...]]
        sage: maj.refine()
        sage: maj.majseq_pol_part(10)
        [[41.256...], [188.43...], [920.6...], [4518.9...]]

    TESTS::

        sage: print(DiffOpBound(Dx - 1, pol_part_len=0).__repr__(asympt=False))
        1.000.../(1.000...)*exp(int(POL+1.000...*NUM/1.000...)) where
        POL=0,
        NUM=bound(-1.000.../n, ord≤1)*x^0

        sage: QQi.<i> = QuadraticField(-1)
        sage: for dop in [
        ....:     # orders <= 1 are not supported
        ....:     Dx, Dx - 1, 1/1000*Dx - 1, i*Dx, Dx + i, Dx^2,
        ....:     (x^2 + 1)*Dx^2 + 2*x*Dx,
        ....:     Dx^2 - x*Dx
        ....: ]:
        ....:     DiffOpBound(dop)._test()

        sage: for l in range(10):
        ....:     DiffOpBound(Dx - 5*x^4, pol_part_len=l)._test()
        ....:     DiffOpBound((1-x^5)*Dx - 5*x^4, pol_part_len=l)._test()

        sage: from ore_algebra.analytic.bounds import _test_diffop_bound
        sage: _test_diffop_bound() # long time
    """

    def __init__(self, dop, leftmost=ZZ.zero(), special_shifts=None,
            max_effort=2, pol_part_len=None, bound_inverse="simple"):
        r"""
        Construct a DiffOpBound for a subset of the solutions of dop.

        INPUT:

        * dop: element of K(z)[Dz] (K a number field), with 0 as a regular
          (i.e., ordinary or regular singular) point;
        * leftmost (default 0): algebraic number;
        * special_shifts (optional): list of (shift, mult) pairs, where shift
          is a nonnegative integer and (leftmost + shift) is a root of
          multiplicity mult of the indicial polynomial of dop.

        OUTPUT:

        If special_shifts is left to its default value or contains all the
        (n, m) such that λ + n (λ = leftmost) is a root of multiplicity m of
        the indicial equation, the resulting bound applies to the generalized
        series solutions of dop that belong to z^λ·ℂ[[z]][log(z)].

        Some of the roots may be omitted. In this case, the bound only applies
        to solutions in which, additionally, the coefficient of z^n has degree
        w.r.t. log(z) strictly less than

            sum { m : (s,m) ∈ special_shifts, s ≤ n }.

        Note that the special case of ordinary points (where the generic
        log-degree bound given by the multiplicity of the roots of the indicial
        equation is pessimistic) is automatically taken into account.

        The remaining parameters are used to set properties of the DiffOpBound
        object related to the effort/tightness trade-off of the algorithm. They
        have no influence on the semantics of the bound.

        ALGORITHM:

        Essentially corresponds to [M19, Algorithm 6.1].
        """

        logger.info("bounding local operator "
                "(%s, pol_part_len=%s, max_effort=%s)...",
                bound_inverse, pol_part_len, max_effort)

        self._dop_D = dop = DifferentialOperator(dop)
        Pols_z = dop.base_ring()
        self.dop = dop_T = dop.to_T(dop._theta_alg())

        lc = dop_T.leading_coefficient()
        if lc.is_term() and not lc.is_constant():
            raise ValueError("irregular singular operator", dop)

        self._rcoeffs = _dop_rcoeffs_of_T(dop_T, IC)

        self.leftmost = leftmost
        if self._dop_D.leading_coefficient()[0] != 0:
            # Ordinary point: even though the indicial equation usually has
            # several integer roots, the solutions are plain power series.
            self.special_shifts = {0: 1}
        else:
            if special_shifts is None:
                special_shifts = local_solutions.exponent_shifts(
                                                         self._dop_D, leftmost)
            self.special_shifts = dict(special_shifts)

        # XXX Consider switching to an interface where the user simply chooses
        # the initial effort (and refine() accepts an effort value)
        self.bound_inverse = bound_inverse
        self.max_effort = max_effort
        self._effort = 0
        if bound_inverse == "solve":
            self._effort += 1
        self._dop_deg = self.dop.degree()
        default_pol_part_len = self._dop_deg//2 + 2
        if pol_part_len is None:
            pol_part_len = default_pol_part_len
        else:
            self._effort += (ZZ(pol_part_len)//default_pol_part_len).nbits()

        self.Poly = Pols_z.change_ring(IR) # TBI
        one = self.Poly.one()
        self.__facto_one = Factorization([(one, 1)], unit=one, sort=False,
                                                     simplify=False)

        self.CPol_z = Pols_z.change_ring(IC)
        self.CPol_zn = PolynomialRing(self.CPol_z, 'n')
        CPol_n = PolynomialRing(IC, 'n')
        self.CPol_nz = PolynomialRing(CPol_n, Pols_z.variable_name())

        self._update_den_bound()
        first_nz, rem_num_nz = self._split_dop(pol_part_len)
        self.alg_idx = self.leftmost + polygen(Pols_z.base_ring(), 'n')
        # indicial polynomial, shifted so that integer roots correspond to
        # series in z^λ·ℂ[[z]][log(z)]
        # (mathematically equal to first_nz[0](self.alg_idx), but the latter
        # has interval coefficients, and we need an exact version to compute
        # the roots)
        z = Pols_z.gen()
        self.ind = self._dop_D._indicial_polynomial_at_zero().monic()(self.alg_idx)
        assert self.ind.is_monic()
        assert self.ind.base_ring().is_exact()
        self.majseq_pol_part = RatSeqBound([], self.ind, self.special_shifts)
        self._update_num_bound(pol_part_len, first_nz, rem_num_nz)

    def __repr__(self, asympt=True):
        fmt = ("{cst}/({den})*exp(int(POL+{cst}*NUM/{den})) where\n"
               "POL={pol},\n"
               "NUM={num}\n")
        def pol_repr(ratseqbounds, shift):
            if len(ratseqbounds) == 0:
                return 0
            coeff = ratseqbounds.entries_repr("asympt" if asympt else "short")
            var = self.Poly.variable_name()
            return " + ".join("{}*{}^{}".format(c, var, n + shift)
                              for n, c in enumerate(coeff))
        return fmt.format(
                cst=self.cst, den=self.maj_den,
                num=pol_repr(self.majseq_num, shift=len(self.majseq_pol_part)),
                pol=pol_repr(self.majseq_pol_part, shift=0))

    @cached_method
    def _poles(self):
        sing = self._dop_D._singularities(myCIF, multiplicities=True)
        nz = [(s, m) for s, m in sing if not s.contains_zero()]
        if sum(m for s, m in nz) == self.dop.leading_coefficient().degree():
            return nz
        else:
            raise NotImplementedError

    def _update_den_bound(self):
        r"""
        Set self.cst, self.maj_den so that cst/maj_den is a majorant series
        of the leading coefficient of dop.
        """
        den = self.dop.leading_coefficient()
        if den.degree() <= 0:
            facs = []
        # below_abs()/lower() to get thin intervals
        elif self.bound_inverse == "simple":
            rad = abs_min_nonzero_root(den).below_abs(test_zero=True)
            facs = [(self.Poly([rad, -1]), den.degree())]
        elif self.bound_inverse == "solve":
            facs = [(self.Poly([IR(iv.abs().lower()), -1]), mult)
                    for iv, mult in self._poles()]
        else:
            raise ValueError("algorithm")
        self.cst = ~abs(IC(den.leading_coefficient()))
        self.maj_den = Factorization(facs, unit=self.Poly.one(),
                                     sort=False, simplify=False)

    @cached_method
    def _dop_ball_lc(self):
        lc = self.dop.leading_coefficient()
        lcdeg = lc.degree()
        some_coeffs = [lc[i] for i in range(0, lcdeg+1, 1+lcdeg//3)]
        if isinstance(lc.base_ring(), NumberField_quadratic):
            some_coeffs = [c.numerator() for c in some_coeffs]
        prec = max(a.numerator().nbits() for c in some_coeffs for a in c)
        prec += 50
        CBFp = ComplexBallField(prec)
        Pol = PolynomialRing(CBFp, self.Poly.variable_name())
        return Pol([IC(c) for c in lc], check=False)

    def _split_dop(self, pol_part_len):
        r"""
        Split self.dop.monic() into a truncated series in z and a remainder.

        Let lc denote the leading coefficient of dop. This function computes
        two operators first, rem ∈ K[θ][z] such that

            dop·lc⁻¹ = first + rem_num·z^ℓ·lc⁻¹,    deg[z](first) < ℓ

        where ℓ = pol_part_len + 1. Thus, first is the Taylor expansion in z to
        order O(z^ℓ) of dop·lc⁻¹ written with θ on the left.

        In the output, first and rem_num are encoded as elements of a
        commutative polynomial ring K[n][z]. More precisely, θ is replaced by a
        commutative variable n, with the convention that n^i·z^j should be
        mapped to θ^i·z^j with θ on the left when translating back.
        """
        # XXX: This function recomputes the series expansion from scratch every
        # time. Use Newton's method to update it instead?
        Pol_zn = self.CPol_zn
        orddeq = self.dop.order()

        # Compute the initial part of the series expansion.
        lc = self._dop_ball_lc()
        inv = lc.inverse_series_trunc(pol_part_len + 1)
        if inv[0].contains_zero():
            logging.warn("probable interval blow-up in bound computation")
        # Including rcoeffs[-1] here actually is redundant: by construction,
        # the only term involving n^ordeq  in first will be 1·n^ordeq·z^0.
        first_zn = Pol_zn([pol._mul_trunc_(inv, pol_part_len + 1)
                           for pol in self._rcoeffs])
        # Force the leading coefficient to one after interval computations
        assert all(pol.contains_zero() for pol in first_zn[orddeq] >> 1)
        n = Pol_zn.gen()
        first_zn = n**orddeq + first_zn[:orddeq]
        first_nz = _switch_vars(first_zn, self.CPol_nz)
        # Would hold in exact arithmetic
        # assert first_nz[0] == self._dop_D.indicial_polynomial(z, n).monic()
        assert all(pol.degree() < self.dop.order() for pol in first_nz >> 1)

        # Now compute rem_num as (dop - first·lc)·z^(-pol_part_len-1)
        dop_zn = Pol_zn(self._rcoeffs)
        # By construction (since lc is the leading coefficient of dop and
        # first_nz = 1·n^orddeq + ···), rem_num_0_zn has degree < orddeq in n.
        # Truncate as the interval subtraction may leave inexact zeros.
        rem_num_0_zn = (dop_zn - first_zn*lc)[:orddeq]
        rem_num_0_nz = _switch_vars(rem_num_0_zn, self.CPol_nz)
        # Would hold in exact arithmetic
        # assert rem_num_0_nz.valuation() >= pol_part_len + 1
        rem_num_nz = rem_num_0_nz >> (pol_part_len + 1)

        return first_nz, rem_num_nz

    def _update_num_bound(self, pol_part_len, first_nz, rem_num_nz):
        old_pol_part_len = len(self.majseq_pol_part)
        # We ignore the coefficient first_nz[0], which amounts to multiplying
        # the integrand by z⁻¹, as prescribed by the theory. Since, by
        # definition, majseq_num starts at the degree following that of
        # majseq_pol_part, it gets shifted as well.
        self.majseq_pol_part.extend([first_nz[i](self.alg_idx)
                for i in range(old_pol_part_len + 1, pol_part_len + 1)])
        assert len(self.majseq_pol_part) == pol_part_len
        self.majseq_num = RatSeqBound(
                [pol(self.alg_idx) for pol in rem_num_nz],
                self.ind, self.special_shifts)

    def effort(self):
        return self._effort

    def can_refine(self):
        return self._effort < self.max_effort

    def refine(self):
        # XXX: make it possible to increase the precision of IR, IC
        if not self.can_refine():
            logger.debug("majorant no longer refinable")
            return
        self._effort += 1
        logger.info("refining majorant (effort = %s)...", self._effort)
        if self.bound_inverse == 'simple':
            self.bound_inverse = 'solve'
            self._update_den_bound()
        else:
            new_pol_part_len = max(2, 2*self.pol_part_len())
            split = self._split_dop(new_pol_part_len)
            self._update_num_bound(new_pol_part_len, *split)

    def pol_part_len(self):
        return len(self.majseq_pol_part)

    def __call__(self, n):
        r"""
        Return a term v[n] of the majorant sequence.
        """
        maj_pol_part = self.Poly(self.majseq_pol_part(n))
        # XXX: perhaps use sparse polys or add explicit support for a shift
        # in RationalMajorant
        maj_num_pre_shift = self.Poly(self.majseq_num(n))
        maj_num = (self.cst*maj_num_pre_shift) << self.pol_part_len()
        terms = [(maj_pol_part, self.__facto_one), (maj_num, self.maj_den)]
        rat_maj = RationalMajorant(terms)
        # The rational part “compensates” the change of unknown function
        # involving the leading coefficient of the operator.
        maj = HyperexpMajorant(integrand=rat_maj, num=self.Poly(self.cst),
                den=self.maj_den)
        return maj

    @cached_method
    def bwrec(self):
        return local_solutions.bw_shift_rec(self.dop, shift=self.leftmost)

    def normalized_residual(self, n, last, bwrec_nplus=None, Ring=None):
        r"""
        Compute the “normalized residual” associated to a truncated solution
        of dop(y) = 0.

        Consider a solution

            y(z) = z^λ·sum[i,k](y[i,k]·z^i·log(z)^k/k!)

        of self.dop(y) = 0, and its truncated series expansion

            ỹ(z) = z^λ·sum[i<n,k](y[i,k]·z^i·log(z)^k/k!).

        Denote s = deg[z](dop(z,θ)). The equation

            monic(dop(z=0,θ))(f(z)) = dop(ỹ)

        has at least one solution of the form

            f(z) = z^(λ+n)·sum[k](f[k](z)·log(z)^k/k!)

        for a finite list [f[0], f[1], ...] of polynomials of degree ≤ s-1
        (exactly one when none of λ+n, λ+n+1, ..., λ+n+s-1 is a root of the
        indicial polynomial dop(z=0,n)).

        This method takes as input the truncation order n and the
        coefficients ::

            last = [[y[n-1,0], y[n-1,1], ...],
                    [y[n-2,0], y[n-2,1], ...],
                    ...,
                    [y[n-s,0], y[n-s,1], ...]],

        and returns a list [f[0], f[1], ...] as above.

        In order to avoid redundant computations, is possible to pass as
        additional input the series expansions around λ+n+j (0≤j≤s) of the
        coefficients of the recurrence operator dop(S⁻¹,ν) =
        sum[0≤i≤s](b[i](ν)·S⁻¹) associated to dop.

        The optional Ring parameter makes it possible to choose the coefficient
        domain. It is there for debugging purposes.

        .. WARNING::

            The bound holds for the normalized residual computed using the
            operator ``self.dop``, not the one given as input to ``__init__``.
            These operators differ by a power-of-x factor, which may change the
            normalized residual.

        ALGORITHM:

        [M19, Algorithm 6.9]

        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.bounds import *
            sage: Dops, t, Dt = DifferentialOperators(QQ, 't')

        Compute the normalized residual associated to a truncation of the
        exponential series::

            sage: trunc = t._exp_series(5); trunc
            1/24*t^4 + 1/6*t^3 + 1/2*t^2 + t + 1
            sage: maj = DiffOpBound(Dt - 1)
            sage: nres = maj.normalized_residual(5, [[trunc[4]]]); nres
            [[-0.00833333333333333 +/- 5.77e-18]]

        Check that it has the expected properties::

            sage: dopT = (Dt - 1).to_T('Tt'); dopT
            Tt - t
            sage: dopT.map_coefficients(lambda pol: pol[0])(nres[0]*t^5)
            ([-0.0416666666666667 +/- 6.40e-17])*t^5
            sage: (Dt - 1).to_T('Tt')(trunc).change_ring(CBF)
            ([-0.0416666666666667 +/- 4.26e-17])*t^5

        Note that using Dt - 1 instead of θt - t makes a difference in the
        result, since it amounts to a division by t::

            sage: (Dt - 1)(trunc).change_ring(CBF)
            ([-0.0416666666666667 +/- 4.26e-17])*t^4

        TESTS::

            sage: maj = DiffOpBound(Dt^2 + 1)
            sage: trunc = t._sin_series(5) + t._cos_series(5)
            sage: maj._check_normalized_residual(5, [trunc], ZZ.zero(), QQ)
            0

            sage: Pol.<n> = CBF[]
            sage: Jets.<eta> = CBF[]
            sage: bwrec = [n*(n-1), Pol(0), Pol(1)]
            sage: bwrec_nplus = [[Jets(pol(5+i)) for pol in bwrec]
            ....:                for i in [0,1]]
            sage: last = [[trunc[4]], [trunc[3]]]
            sage: res1 = maj.normalized_residual(5, last, bwrec_nplus)
            sage: res2 = maj.normalized_residual(5, last)
            sage: len(res1) == len(res2) == 1
            True
            sage: res1[0] - res2[0]
            ([+/- ...e-18])*t + [+/- ...e-18]

        This operator annihilates t^(1/3)*[1/(1-t)+log(t)^2*exp(t)]+exp(t)::

            sage: dop = ((81*(-1+t))*t^4*(3*t^6-19*t^5+61*t^4-85*t^3+106*t^2
            ....: -22*t+28)*Dt^5-27*t^3*(36*t^8-315*t^7+1346*t^6-3250*t^5
            ....: +4990*t^4-5545*t^3+2788*t^2-1690*t+560)*Dt^4+27*t^2*(54*t^9
            ....: -555*t^8+2678*t^7-7656*t^6+13370*t^5-17723*t^4+13070*t^3
            ....: -6254*t^2+4740*t-644)*Dt^3-3*t*(324*t^10-3915*t^9+20871*t^8
            ....: -67614*t^7+130952*t^6-190111*t^5+180307*t^4-71632*t^3
            ....: +73414*t^2-26368*t-868)*Dt^2+(243*t^11-3645*t^10+21276*t^9
            ....: -77346*t^8+163611*t^7-249067*t^6+297146*t^5-83366*t^4
            ....: +109352*t^3-97772*t^2-4648*t+896)*Dt+162*t^10-1107*t^9
            ....: +5292*t^8-12486*t^7+17908*t^6-37889*t^5-6034*t^4-1970*t^3
            ....: +36056*t^2+2044*t-896)

        We check that the residuals corresponding to various truncated
        solutions (both without and with logs, with lefmost=1/3 and leftmost=0)
        are correctly computed::

            sage: n = 20
            sage: zero = t.parent().zero()

            sage: maj = DiffOpBound(dop, leftmost=0, special_shifts=[(0, 1)])
            sage: trunc = [t._exp_series(n), zero, zero]
            sage: maj._check_normalized_residual(n, trunc, 0, QQ)
            0

            sage: maj = DiffOpBound(dop, leftmost=1/3, special_shifts=[(0, 1)])
            sage: trunc = [(1-t).inverse_series_trunc(n), zero, zero]
            sage: maj._check_normalized_residual(n, trunc, 1/3, QQ)
            0
            sage: trunc = [(1-t).inverse_series_trunc(n), zero, 2*t._exp_series(n)]
            sage: maj._check_normalized_residual(n, trunc, 1/3, QQ)
            0
        """
        deg = self._dop_deg
        logs = max(len(logpol) for logpol in last) if last else 1
        if Ring is None:
            use_sum_of_products, Ring = _use_sum_of_products(last, bwrec_nplus)
        else:
            use_sum_of_products = False
        if bwrec_nplus is None:
            bwrec = self.bwrec()
            # Suboptimal: For a given i, we are only going to need the
            # b[i](λ+n+i+j+ε) for j < s - i.
            bwrec_nplus = [bwrec.eval_series(Ring, n+i, logs)
                           for i in range(deg)]
        # Check that we have been given/computed enough shifts of the
        # recurrence, and that the orders are consistent. We only have
        # len(bwrec_nplus[0]) - 1 == ordrec >= deg, not ordrec == deg,
        # because bwrec might be of the form ...+(..)*S^(-s)+0*S^(-s-1)+...
        assert (bwrec_nplus == [] and deg == 0
                or len(bwrec_nplus) >= len(bwrec_nplus[0]) - 1 >= deg)

        # res(z) = z^(λ + n)·sum[k,d]( res[k][d]·z^d·log^k(z)/k!)
        #   f(z) = z^(λ + n)·sum[k,d](nres[k][d]·z^d·log^k(z)/k!)
        res = [[None]*deg for _ in range(logs)]
        nres = [[None]*deg for _ in range(logs)]
        # Since our indicial polynomial is monic,
        # b₀(n) = bwrec_nplus[0][0][0] = lc(dop)(0)·ind(n) = cst·ind(n)
        cst = self.dop.leading_coefficient()[0]
        #assert deg == 0 or IC(cst*self.ind(n) - bwrec_nplus[0][0][0]).contains_zero()

        # For each d, compute the coefficients of z^(λ+n+d)·log(z)^k/k! in the
        # normalized residual. This is done by solving a triangular system with
        # (cst ×) the coefficients of the residual corresponding to the same d
        # on the rhs. The coefficients of the residual are computed on the fly.
        for d in range(deg):
            lc = bwrec_nplus[d][0][0]
            assert not (lc.parent() is IC and lc.contains_zero())
            inv = ~lc
            for k in reversed(range(logs)):
                # Coefficient of z^(λ+n+d)·log(z)^k/k! in dop(ỹ)
                if use_sum_of_products:
                    bwrec_nplusd = bwrec_nplus[d]
                    res[k][d] = Ring._sum_of_products(
                            (bwrec_nplusd[d+i+1][j], last[i][k+j])
                            for i in range(deg - d)
                            for j in range(logs - k))
                else:
                    res[k][d] = sum(
                            [Ring(bwrec_nplus[d][d+i+1][j])*Ring(last[i][k+j])
                            for i in range(deg - d)
                            for j in range(logs - k)])
                # Deduce the corresponding coefficient of nres
                # XXX For simplicity, we limit ourselves to the “generic” case
                # where none of the n+d is a root of the indicial polynomial.
                cor = sum(bwrec_nplus[d][0][u]*nres[k+u][d]
                          for u in range(1, logs-k))
                # It is expected that both cst and res are large (and inv only
                # balances one of them) when the operator “dilates” its
                # argument. This is because the normalized residual is defined
                # using a monic indicial polynomial.
                nres[k][d] = inv*(cst*res[k][d] - cor)
        Poly = self.CPol_z if Ring is IC else self.Poly.change_ring(Ring)
        return [Poly(coeff) for coeff in nres]

    def _check_normalized_residual(self, n, trunc, expo, Ring):
        r"""
        Test the output of normalized_residual().

        This is done by comparing

            monic(dop(z=0,θ))(f(z))      and       dop(ỹ(z)),

        where f(z) is the output of normalized_residual() and ỹ(z) is a
        solution of dop truncated at order O(z^n).

        The parameter trunc must be a list of polynomials such that

            ỹ(z) = z^expo·sum[k](trunc[k](z)·log(z)^k/k!).

        Ideally, Ring should be IR or IC in most cases; unfortunately, this
        often doesn't work due to various weaknesses of Sage.
        """
        ordrec = self.dop.degree()
        last = reversed(list(zip(*(pol.padded_list(n)[n-ordrec:n]
                                   for pol in trunc))))
        coeff = self.normalized_residual(n, list(last), Ring=Ring)
        from sage.all import log, SR
        z = SR.var(self.Poly.variable_name())
        nres = z**(self.leftmost + n)*sum(pol*log(z)**k/ZZ(k).factorial()
                                          for k, pol in enumerate(coeff))
        trunc_full = z**expo*sum(pol*log(z)**k/ZZ(k).factorial()
                                 for k, pol in enumerate(trunc))
        lc = self.dop.leading_coefficient()
        dop0 = self.dop.map_coefficients(lambda pol: pol[0]/lc[0])
        Poly = self.Poly.change_ring(Ring)
        out = (dop0(nres)/z**self.leftmost).expand()
        ref = (self.dop(trunc_full)/z**self.leftmost).expand()
        return (out-ref).expand()

    def rhs(self, n1, normalized_residuals, maj=None):
        r"""
        Compute the right-hand side of a majorant equation valid for each of
        the given normalized residuals.

        INPUT:

        A list of normalized residuals q (as computed by normalized_residual()
        i.e., in particular, with an implicit z^n factor) corresponding to
        solutions y of self.dop truncated to a same order n1. Optionally, a
        HyperexpMajorant maj = self(n0) for some n0 ≤ n1.

        OUTPUT:

        A polynomial (q#)(z) such that, with (q^)(z) = z^n1·(q#)(z), ::

            z·ŷ'(z) - ŷ(z) = (q^)(z)·v[n0](z)·den(z)                     (*)

        is a majorant equation of self.dop(ỹ) = Q₀(θ)·q(z) (where Q₀ = monic
        indicial polynomial) for all q ∈ normalized_residuals. More precisely,
        if y(z) is a solution of dop(y) = 0 associated to one of the q's, if
        ŷ(z) is a solution of (*), and if ::

            |y[λ+n,k]| ≤ ŷ[n]   for   n ≥ n1,   0 ≤ k < mult(n, Q₀),     (**)

        then `|y[λ+n,k]| ≤ ŷ[n]` for *all* n ≥ n1, k ≥ 0. If maj is omitted, the
        bound will hold for any choice of n0 ≤ n1 in (*), but may be coarser
        than that corresponding to a particular n0.

        The typical application is with n0 = n1 larger than the n's
        corresponding to roots λ+n of Q₀ where the y have nonzero initial
        values. In this case, one can take ::

            ŷ(z) = v[n0](z)·∫(w⁻¹·(q^)(w)·dw, w=0..z)

        and the conditions (**) trivially hold true. (In general, one would
        need to adjust the integration constant so that they do.)

        Note: Some of the above actually makes sense for n1 < n0 as well,
        provided that (**) also hold for n1 ≤ n < n0 and k ≥ 0 and that q^ be
        suitably modified.
        """
        # Let res(z) denote a normalized residual. In general, for any
        # polynomial (res^)(z) s.t. (res^)[n] ≥ |λ+n|*|res[n,k]| for all n, k,
        # the series v[n0](z)*∫(w⁻¹*(res^)(w)/h[n0](w)) where
        # h[n0](z) = v[n0](z)*den(z) is a majorant for the tail of the
        # solution. To make the integral easy to compute, we choose
        # (res^) = (q^)(z)*h[n0](z), i.e., as a polynomial multiple of h.
        nres_bound = bound_polynomials([pol for nres in normalized_residuals
                                            for pol in nres],
                                        self.Poly)
        Pols = nres_bound.parent()
        aux = Pols([(n1 + j)*c for j, c in enumerate(nres_bound)],
                   check=False)
        if maj is None:
            # As h[n0](z) has nonnegative coefficients and h[n0](0) = 1, it is
            # always enough to take (q^)[n] ≥ |λ+n|*max[k](|res[n,k]|), that
            # is, (q#)(z) = aux(z).
            return aux
        else:
            # Tighter choice: compute a truncated series expansion f(z) of
            # aux(z)/h(z) s.t. aux(z) = f(z)*h(z) + O(z^(1+deg(aux))). Then,
            # any f^ s.t. f^[n] ≥ max(0,f[n]) is a valid q^.
            ord = aux.degree() + 1
            inv = maj.inv_exp_part_series0(ord) # XXX slow
            # assert all(c.imag().contains_zero() for c in inv)
            inv = Pols([c.real() for c in inv], check=False)
            f = aux._mul_trunc_(inv, ord)
            z = IR.zero()
            return Pols([z.max(c) for c in f], check=False)

    def tail_majorant(self, n, normalized_residuals):
        r"""
        Bound the tails of order ``n`` of solutions of ``self.dop(y) == 0``.

        INPUT:

        A list of normalized residuals q (as computed by normalized_residual(),
        i.e., in particular, with an implicit z^n factor) corresponding to
        solutions y of self.dop truncated to a same order n.

        The truncation order n is required to be larger than all n' such that
        self.leftmost + n' is a root of the indicial polynomial of self.dop,
        and the solution of interest has nonzero initial values there.

        OUTPUT:

        A HyperexpMajorant representing a common majorant series for the
        tails y[n:](z) of the corresponding solutions.

        ALGORITHM:

        Essentially [M19, Algorithm 6.11]
        """
        maj = self(n)
        # XXX Better without maj? (speed/tightness trade-off)
        rhs = self.rhs(n, normalized_residuals, maj)
        # logger.debug("n=%s, maj(n)=%s, rhs=%s", n, maj, rhs)
        # Shift by n to account for the implicit z^n, then by -1 because of the
        # formula ∫(w⁻¹·(q^)(w)·dw.
        pol = (rhs << (n - 1)).integral() # XXX potential perf issue with <<
        maj *= pol
        return maj

    def _test(self, ini=None, prec=100):
        r"""
        Check that the majorants produced by this DiffOpBound bound the tails
        of the solutions of the associated operator.

        This is a heuristic check for testing purposes, nothing rigorous!

        This method currently does not support regular singular points.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.bounds import *
            sage: Dops, x, Dx = DifferentialOperators()
            sage: maj = DiffOpBound(Dx - 1)
            sage: maj._test()
            sage: maj._test([3], 200)
        """
        if (self._dop_D.leading_coefficient()[0].is_zero()
                or not self.leftmost.is_zero()):
            raise NotImplementedError
        ord = self.dop.order()
        if ini is None:
            from sage.rings.number_field.number_field import QuadraticField
            QQi = QuadraticField(-1)
            ini = [QQi.random_element() for _ in range(ord)]
        sol = self.dop.power_series_solutions(prec)
        Series = PowerSeriesRing(CBF, self.dop.base_ring().variable_name())
        ref = sum((ini[k]*sol[k] for k in range(ord)), Series(0)).polynomial()
        # XXX This won't work at regular singular points (even for power series
        # solutions), because tail_majorant(), by basing on rhs(), assumes that
        # we are past all nonzero initial conditions.
        for n in [ord, ord + 1, ord + 2, ord + 50]:
            logger.info("truncation order = %d", n)
            if n + 30 >= prec:
                warnings.warn("insufficient precision")
            last = [[ref[n-i]] for i in range(1, self.dop.degree() + 1)]
            resid = self.normalized_residual(n, last)
            maj = self.tail_majorant(n, [resid])
            tail = (ref >> n) << n
            maj_ser = maj.bound_series(0, n + 30)
            logger.info(["|{}| <= {}".format(tail[i], maj_ser[i])
                         for i in range(n + 30)])
            maj._test(tail)

    @cached_method
    def _random_ini(self):
        return local_solutions.random_ini(self._dop_D)

    def _test_point(self):
        rad = abs_min_nonzero_root(self._dop_D.leading_coefficient())
        pt = QQ(2) if rad == infinity else RIF(rad/2).simplest_rational()
        return pt

    def plot(self, ini=None, pt=None, eps=RBF(1e-50), tails=None,
             color="blue", title=True, intervals=True, **opts):
        r"""
        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.differential_operator import DifferentialOperator
            sage: from ore_algebra.analytic.bounds import DiffOpBound
            sage: from ore_algebra.analytic.local_solutions import LogSeriesInitialValues
            sage: Dops, x, Dx = DifferentialOperators()

            sage: DiffOpBound(Dx - 1).plot([CBF(1)], CBF(i)/2, RBF(1e-20))
            Graphics object consisting of 4 graphics primitives

            sage: DiffOpBound(x*Dx^3 + 2*Dx^2 + x*Dx).plot(eps=1e-8)
            Graphics object consisting of 4 graphics primitives

            sage: dop = DifferentialOperator(x*Dx^2 + Dx + x)
            sage: DiffOpBound(dop, 0, [(0,2)]).plot(eps=1e-8,
            ....:       ini=LogSeriesInitialValues(0, {0: (1, 0)}, dop))
            Graphics object consisting of 4 graphics primitives

            sage: dop = ((x^2 + 10*x + 50)*Dx^10 + (5/9*x^2 + 50/9*x + 155/9)*Dx^9
            ....: + (-10/3*x^2 - 100/3*x - 190/3)*Dx^8 + (30*x^2 + 300*x + 815)*Dx^7
            ....: + (145*x^2 + 1445*x + 3605)*Dx^6 + (5/2*x^2 + 25*x + 115/2)*Dx^5
            ....: + (20*x^2 + 395/2*x + 1975/4)*Dx^4 + (-5*x^2 - 50*x - 130)*Dx^3
            ....: + (5/4*x^2 + 25/2*x + 105/4)*Dx^2 + (-20*x^2 - 195*x - 480)*Dx
            ....: + 5*x - 10)
            sage: DiffOpBound(dop, 0, [], pol_part_len=4, # not tested
            ....:         bound_inverse="solve").plot(eps=1e-10)
            Graphics object consisting of 4 graphics primitives

        TESTS::

            sage: DiffOpBound(Dx - 1).plot()
            Graphics object consisting of 4 graphics primitives
        """
        import sage.plot.all as plot
        from . import naive_sum

        logger = logging.getLogger("ore_algebra.analytic.bounds.plot")

        if ini is None:
            ini = self._random_ini()
        if pt is None:
            pt = self._test_point()
        eps = RBF(eps)
        logger.info("operator: %s", str(self.dop)[:60])
        logger.info("point: %s", pt)
        logger.info("initial values: %s", ini)

        saved_max_effort = self.max_effort
        self.max_effort = 0
        recorder = accuracy.BoundRecorder(maj=self, eps=eps>>2)
        ref_sum = naive_sum.series_sum(self._dop_D, ini, pt, eps>>2, maj=self,
                                       stride=1, stop=recorder)
        P = ref_sum[0].parent()
        recd = recorder.recd[:-1]
        assert all(ref_sum[0].overlaps(P(rec.psum[0]).add_error(rec.b))
                   for rec in recd)
        self.max_effort = saved_max_effort
        # Note: this won't work well when the errors get close to the double
        # precision underflow threshold.
        err = [(rec.n, (rec.psum[0]-ref_sum[0]).abs()) for rec in recd]

        # avoid empty plots, matplotlib warnings, ...
        def pltfilter(it, eps=float(eps), large=float(1e200)):
            return [(x, float(y)) for (x, y) in it if eps < float(y) < large]
        myplot = plot.plot([])
        if intervals:
            myplot += plot.line( # reference value - upper
                    pltfilter((n, v.upper()) for (n, v) in err),
                    color="black", scale="semilogy")
        myplot += plot.line( # reference value - main
                pltfilter((n, v.lower()) for (n, v) in err),
                color="black", scale="semilogy")
        if intervals:
            myplot += plot.line( # bound - lower
                    pltfilter((rec.n, rec.b.lower()) for rec in recd),
                    color=color, scale="semilogy", **opts)
        myplot += plot.line( # bound - main
                pltfilter((rec.n, rec.b.upper()) for rec in recd),
                color=color, scale="semilogy", **opts)
        if tails is not None:
            nmax = recd[-1].n + 1
            for rec in recd: # bounds for n1 > n based on the residual at n
                if (rec.n - recd[0].n) % tails == 0:
                    r = CBF(pt).abs()
                    data = [(n1, rec.maj.bound(r, tail=n1).upper())
                            for n1 in range(rec.n, nmax)
                            if rec.maj is not None]
                    data = pltfilter(data)
                    myplot += plot.line(data, color=color, scale="semilogy")
        ymax = myplot.ymax()
        if title:
            if title is True:
                title = repr(self._dop_D)
                title = title if len(title) < 50 else title[:47]+"..."
                title += " @ x=" + repr(pt)
            myplot += plot.plot([], title=title)
        return myplot

    def plot_refinements(self, n=4, legend_fmt="{inv}, pplen={pplen}", **kwds):
        import sage.plot.all as plot
        p = plot.plot([])
        styles = [':', '-.', '--', '-']
        d = len(styles) - n
        if d > 0:
            styles = styles[d:]
        for i in range(n):
            lab = legend_fmt.format(
                    inv=self.bound_inverse,
                    pplen=self.pol_part_len())
            p += self.plot(intervals=False,
                           legend_label=lab,
                           linestyle=styles[i%len(styles)],
                           **kwds)
            self.refine()
        p.set_legend_options(handlelength=4, shadow=False)
        return p

class MultiDiffOpBound(object):
    r"""
    Ad hoc wrapper for passing several DiffOpBounds to StoppingCriterion.

    (Useful for handling several valuation groups at once.)
    """

    def __init__(self, majs):
        self.majs = majs

    def can_refine(self):
        return any(m.can_refine() for m in self.majs)

    def refine(self):
        for m in self.majs:
            m.refine()

    def effort(self):
        return min(m.effort() for m in self.majs)

# Perhaps better: work with a "true" Ore algebra K[θ][z]. Use Euclidean
# division to compute the truncation in DiffOpBound._update_num_bound.
# Extracting the Qj(θ) would then be easy, and I may no longer need the
# coefficients of θ "on the right".
def _dop_rcoeffs_of_T(dop, base_ring):
    r"""
    Compute the coefficients of dop as an operator in θ but with θ on the left.

    EXAMPLES::

        sage: from ore_algebra import OreAlgebra
        sage: from ore_algebra.analytic.bounds import _dop_rcoeffs_of_T
        sage: Pols.<x> = QQ[]; Dops.<Tx> = OreAlgebra(Pols)
        sage: dop = (1/250*x^4 + 21/50*x^3)*Tx - 6/125*x^4 + 6/25*x^3
        sage: coeff = _dop_rcoeffs_of_T(dop, QQ); coeff
        [-8/125*x^4 - 51/50*x^3, 1/250*x^4 + 21/50*x^3]
        sage: sum(Tx^i*c for i, c in enumerate(coeff)) == dop
        True

    TESTS::

        sage: _dop_rcoeffs_of_T(Dops.zero(), QQ)
        []
        sage: _dop_rcoeffs_of_T(Dops.one(), QQ)
        [1]
        sage: _dop_rcoeffs_of_T(Dops.gen(), QQ)
        [0, 1]
    """
    assert dop.parent().is_T()
    Pols = dop.base_ring().change_ring(base_ring)
    ordlen, deglen = dop.order() + 1, dop.degree() + 1
    binomial = [[0]*(ordlen) for _ in range(ordlen)]
    for n in range(ordlen):
        binomial[n][0] = 1
        for k in range(1, n + 1):
            binomial[n][k] = binomial[n-1][k-1] + binomial[n-1][k]
    res = [None]*(ordlen)
    for k in range(ordlen):
        pol = [base_ring.zero()]*(deglen)
        for j in range(deglen):
            pow = 1
            for i in range(ordlen - k):
                pol[j] += pow*binomial[k+i][i]*base_ring(dop[k+i][j])
                pow *= (-j)
        res[k] = Pols(pol)
    return res

@random_testing
def _test_diffop_bound(
        ords=range(1, 5),
        degs=range(5),
        pplens=[1, 2, 5],
        prec=100,
        verbose=False
    ):
    r"""
    Randomized testing of :func:`DiffOpBound`.

    EXAMPLES::

    Just an example of how to use this function; the real tests are run from
    the docstring of DiffOpBound. ::

        sage: from ore_algebra.analytic.bounds import _test_diffop_bound
        sage: _test_diffop_bound(ords=[2], degs=[2], pplens=[1], prec=100,
        ....:         seed=0, verbose=True)
        testing operator: 5/927*Dx^2 + ((-2/463*i - 1/463)*x + 1/463*i)*Dx - 95/396*i + 1/396
    """
    from sage.rings.number_field.number_field import QuadraticField

    QQi = QuadraticField(-1, 'i')
    Pols, x = PolynomialRing(QQi, 'x').objgen()
    Dops, Dx = ore_algebra.OreAlgebra(Pols, 'Dx').objgen()

    for ord in ords:
        for deg in degs:
            dop = Dops(0)
            while dop.leading_coefficient()(0).is_zero():
                dop = Dops([Pols.random_element(degree=(0, deg))
                                /ZZ.random_element(1,1000)
                            for _ in range(ord + 1)])
            if verbose:
                print("testing operator:", dop)
            for pplen in pplens:
                maj = DiffOpBound(dop, pol_part_len=pplen)
                maj._test(prec=prec)

def _switch_vars(pol, Ayx):
    Ay = Ayx.base_ring()
    if pol.is_zero():
        return Ayx.zero()
    dy = pol.degree()
    dx = max(c.degree() for c in pol)
    return Ayx([Ay([pol[j][i] for j in range(dy+1)]) for i in range(dx+1)])

def _use_sum_of_products(last, bwrec_nplus):
    if not (last and last[0] and bwrec_nplus and bwrec_nplus[0] and
            bwrec_nplus[0][0]):
        return False, IC
    b0 = last[0][0]
    b1 = bwrec_nplus[0][0][0]
    if (isinstance(b0, RealBall) and isinstance(b1, RealBall)
            and hasattr(IR, '_sum_of_products')):
        return True, IR
    elif (isinstance(b0, ComplexBall) and isinstance(b1, ComplexBall)
            and hasattr(IC, '_sum_of_products')):
        return True, IC
    else:
        return False, IC

def _log2abs(x):
    upper = x.above_abs().log(2)
    below = x.below_abs()
    if below.contains_zero():
        lower = below.parent()(-infinity)
    else:
        lower = below.log(2)
    return lower.union(upper)
