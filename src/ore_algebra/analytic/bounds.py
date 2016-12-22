# -*- coding: utf-8 - vim: tw=80
"""
Error bounds
"""

# TODO:
# - this module uses at least three different object types for things that are
# essentially rational fractions (QuotientRingElements, Factorizations, and
# Rational Majorants) --> simplify?

import itertools, logging, textwrap, warnings

import sage.rings.polynomial.real_roots as real_roots

from sage.arith.srange import srange
from sage.misc.cachefunc import cached_function, cached_method
from sage.misc.misc_c import prod
from sage.rings.all import CIF
from sage.rings.complex_arb import CBF
from sage.rings.infinity import infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_element import Polynomial
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.qqbar import QQbar, AA
from sage.rings.rational_field import QQ
from sage.rings.real_arb import RBF
from sage.rings.real_mpfi import RIF
from sage.rings.real_mpfr import RealField, RR
from sage.structure.factorization import Factorization

from .. import ore_algebra
from . import utilities

from .safe_cmp import *
from .shiftless import squarefree_part

logger = logging.getLogger(__name__)

IR, IC = RBF, CBF # TBI

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

    def series(self, rad, ord):
        r"""
        Compute the series expansion of self at rad to order O(x^ord).

        With rad = 0, this returns a truncation of the majorant series itself.
        More generally, this can be used to obtain bounds on the derivatives of
        the series majorized on disks contained within its disk of convergence.
        """
        raise NotImplementedError

    def __call__(self, rad):
        return self.series(rad, 1)

    def bound(self, rad, derivatives=1):
        """
        Bound the Frobenius norm of the vector

            [g(rad), g'(rad), g''(rad)/2, ..., 1/(d-1)!·g^(d-1)(rad)]

        where d = ``derivatives`` and g is this majorant series. The result is
        a bound for

            [f(z), f'(z), f''(z)/2, ..., 1/(d-1)!·f^(d-1)(z)]

        for all z with |z| ≤ rad.
        """
        if not safe_le(rad, self.cvrad): # intervals!
            return IR(infinity)
        else:
            ser = self.series(rad, derivatives)
            sqnorm = sum((c.abs()**2 for c in ser), IR.zero())
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
        ComplexSeries = PowerSeriesRing(CIF, self.variable_name, prec)
        maj = Series(self.series(0, prec))
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
    if all(pol.degree() == 1 and pol.leading_coefficient().abs().is_one()
            for pol in pols):
        rad = IR(infinity).min(*(IR(pol[0].abs()) for pol in pols))
        rad = IR(rad.lower())
        assert rad >= IR.zero()
        return rad
    raise NotImplementedError

class RationalMajorant(MajorantSeries):
    """
    A rational power series with nonnegative coefficients, represented as an
    unevaluated sum of rational fractions with factored denominators.

    TESTS::

        sage: from ore_algebra.analytic.bounds import *
        sage: Pol.<z> = RBF[]
        sage: den = Factorization([(1-z, 2), (2-z, 1)])
        sage: one = Pol.one().factor()
        sage: maj = RationalMajorant([(1 + z, one), (z^2, den)]) ; maj
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
    """

    def __init__(self, fracs):
        self.Poly = Poly = fracs[0][0].parent().change_ring(IR)
        self._Poly_IC = fracs[0][0].parent().change_ring(IC)
        cvrad = _zero_free_rad([-fac for _, den in fracs for fac, _ in den if fac.degree() > 0])
        super(self.__class__, self).__init__(Poly.variable_name(), cvrad=cvrad)
        self.fracs = []
        for num, den in fracs:
            if isinstance(num, Polynomial) and isinstance(den, Factorization):
                if not den.unit().is_one():
                    raise ValueError("expected a denominator with unit part 1")
                assert den.universe() is Poly or list(den) == []
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
            den_ser = Pol.one()
            for lin, mult in den:
                fac_ser = lin(pert_rad).power_trunc(mult, ord)
                den_ser = den_ser._mul_trunc_(fac_ser, ord)
            # slow; hopefully the fast Taylor shift will help...
            num_ser = Pol(num).compose_trunc(pert_rad, ord)
            res += num_ser._mul_trunc_(den_ser.inverse_series_trunc(ord), ord)
        return res

    def __mul__(self, pol):
        """
        Multiplication by a polynomial.

        Note that this does not change the radius of convergence.
        """
        assert isinstance(pol, Polynomial)
        return RationalMajorant([(pol*num, den) for num, den in self.fracs])

class HyperexpMajorant(MajorantSeries):
    """
    A formal power series of the form rat1(z) + exp(int(rat2(ζ), ζ=0..z)), with
    nonnegative coefficients.

    The fraction rat1 is represented in the form z^shift*num(z)/den(z).

    TESTS::

        sage: from ore_algebra.analytic.bounds import *
        sage: Pol.<z> = RBF[]
        sage: one = Pol.one().factor()
        sage: integrand = RationalMajorant([(4+4*z, one), (z^2, Factorization([(1-z,1)]))])
        sage: den = Factorization([(1/3-z, 1)])
        sage: maj = HyperexpMajorant(integrand, Pol.one(), den); maj
        (1.00... * (-z + [0.333...])^-1)*exp(int(4.0... + 4.0...*z + z^2/(-z + 1.0...)))
        sage: maj.cvrad
        [0.333...]
        sage: maj.series(0, 4)
        ([336.000...])*z^3 + ([93.000...])*z^2 + ([21.000...])*z + [3.000...]
        sage: maj._test()
        sage: maj*=z^20
        sage: maj
        (z^20*1.00... * (-z + [0.333...])^-1)*exp(int(4.000... + 4.000...*z + z^2/(-z + 1.000...)))
        sage: maj._test()

    """

    def __init__(self, integrand, num, den, shift=0):
        assert isinstance(integrand, RationalMajorant)
        assert isinstance(den, Factorization)
        assert isinstance(num, Polynomial)
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

    def series(self, rad, ord):
        # Compute the derivatives “by automatic differentiation”. This is
        # crucial for performance with operators of large order.
        Pol = PolynomialRing(IC, self.variable_name)
        pert_rad = Pol([rad, 1]) # XXX: should be IR
        shx_ser = pert_rad.power_trunc(self.shift, ord)
        num_ser = Pol(self.num).compose_trunc(pert_rad, ord) # XXX: remove Pol()
        den_ser = Pol(self._den_expanded()).compose_trunc(pert_rad, ord)
        assert num_ser.parent() is den_ser.parent()
        rat_ser = (shx_ser._mul_trunc_(num_ser, ord)
                          ._mul_trunc_(den_ser.inverse_series_trunc(ord), ord))
        # XXX: double-check (integral...)
        exp_ser = self.integrand.series(rad, ord).integral()._exp_series(ord)
        ser = rat_ser._mul_trunc_(exp_ser, ord)
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

######################################################################
# Majorants for reciprocals of polynomials ("denominators")
######################################################################

def graeffe(pol):
    r"""
    Compute the Graeffe iterate of this polynomial.

    EXAMPLES:

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
    pol_even = Parent([pol[2*i]   for i in xrange(deg/2+1)])
    pol_odd = Parent([pol[2*i+1] for i in xrange(deg/2+1)])
    graeffe_iterate = (-1)**deg * (pol_even**2 - (pol_odd**2).shift(1))
    return graeffe_iterate

def abs_min_nonzero_root(pol, tol=RR(1e-2), lg_larger_than=RR('-inf'),
        prec=IR.precision()):
    r"""
    Compute an enclosure of the absolute value of the nonzero complex root of
    ``pol`` closest to the origin.

    INPUT:

    - ``pol`` -- Nonzero polynomial.

    - ``tol`` -- An indication of the required relative accuracy (interval
      width over exact value). It is currently *not* guaranteed that the
      relative accuracy will be smaller than ``tol``.

    - ``lg_larger_than`` -- A lower bound on the binary logarithm of acceptable
      results. The function may loop if ``exact result <= 2^lg_larger_than``.

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

        sage: abs_min_nonzero_root(pol, lg_larger_than=-1.4047042967)
        [0.3776955532 +/- 2.41e-11]

        sage: abs_min_nonzero_root(pol, lg_larger_than=-1.4047042966)
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

        sage: from ore_algebra.analytic.ui import *
        sage: Dops, x, Dx = Diffops()
        sage: eval_diffeq((x^2 + 10*x + 50)*Dx^2 + Dx + 1, [-1,1], [0, 1/10])
        [-0.90000329853426...]
    """
    myIR = type(IR)(prec)
    myRIF = type(RIF)(prec)
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
    while (safe_le(lg_rad.lower(rnd='RNDN'), lg_larger_than)
              # *relative* error on 2^lg_rad
           or safe_gt(lg_rad.absolute_diameter(), tol)):
        prev_lg_rad = lg_rad
        # The smallest root of the current mypol is between 2^(-1-m) and
        # (2·deg)·2^(-1-m), cf. Davenport & Mignotte (1990), Grégoire (2012).
        m = myIR(-infinity).max(*(mypol[k].abs().log(2)/k
                                for k in xrange(1, deg+1)))
        lg_rad = (-(1 + myRIF(m)) + encl) >> i
        lg_rad = prev_lg_rad.intersection(lg_rad)
        if lg_rad.lower() == -infinity or cmp(lg_rad, prev_lg_rad) == 0:
            prec *= 2
            logger.debug("failed to bound the roots of %s, "
                    "retrying with prec=%s bits", mypol, prec)
            return abs_min_nonzero_root(pol, RealField(prec)(tol),
                                        lg_larger_than, prec)
        logger.log(logging.DEBUG - 1, "i = %s\trad ∈ %s\tdiam=%s",
                i, lg_rad.exp2().str(style='brackets'),
                lg_rad.absolute_diameter())
        # detect gross input errors (this does not prevent all infinite loops)
        if safe_le(lg_rad.upper(rnd='RNDN'), lg_larger_than):
            raise ValueError("there is a root smaller than 2^({})"
                             .format(lg_larger_than))
        mypol = graeffe(mypol)
        i += 1
    res = myIR(2)**myIR(lg_rad)
    if not safe_le(2*res.rad_as_ball()/res, myIR(tol)):
        logger.debug("required tolerance may not be met")
    return res

######################################################################
# Bounds on rational functions of n
######################################################################

@cached_function # XXX: tie life to a suitable object
def _complex_roots(pol):
    if not pol.parent() is QQ: # QQ typical (ordinary points)
        pol = pol.change_ring(QQbar)
    return [(IC(rt), mult) for rt, mult in pol.roots(CIF)]

class RatSeqBound(object):
    r"""
    A *nonincreasing* bound on a sequence of the form

        ⎰ sum[i](|nums[i](n)/den(n)^(i+1)|),   n ∉ exceptions,
        ⎱ exceptions[n],                       n ∈ exceptions,

    where den and the nums[i] are polynomials such that the nums[i]/den^(i+1)
    have nonpositive degree.

    This version bounds the numerators (from above) and the denominators (from
    below) separately. This simple strategy works well in the typical case where
    the indicial equation has only small roots. In the presence of, e.g., large
    real roots, however, it is not much better than waiting to get past the
    largest root.

    EXAMPLES::

        sage: from ore_algebra.analytic.bounds import RatSeqBound
        sage: Pols.<n> = QQ[]
        sage: bnd = RatSeqBound([Pols(3), n^2], n*(n-3), {3: RBF(10)})
        sage: bnd
        max(
          |(3)/(n^2 - 3*n)|
        + |(n^2)/((...)^2)|,
          10.00000000000000     for  n <= 3
        )
        sage: bnd.asympt_repr()
        '~(3.000000000000000*n^-2 + 1.000000000000000*n^-2)'
        sage: [(bnd.ref(k), bnd(k)) for k in range(5)]
        [([+/- inf], [+/- inf]),
         (1.750000000000000, [28.00000014901161 +/- 1.94e-15]),
         (2.500000000000000, 10.00000000000000),
         (10.00000000000000, 10.00000000000000),
         (1.750000000000000, [1.750000009313226 +/- 2.54e-16])]
        sage: bnd.plot()
        Graphics object consisting of 2 graphics primitives
    """

    def __init__(self, nums, den, exceptions):
        self.almost_one = IR(15)/16
        # Reference values (exceptions is also used to compute the bounds)
        self.nums = nums
        self.den = den
        self.exn = exceptions
        # Dynamically computed data on when various parts of the bound become
        # close enough to their limit that it is not worth recomputing them.
        self._num_converged = [2**62 for _ in nums] # faster than symbolic infty
        self._den_converged = 2**62
        # Precomputed bound data
        self.num_data = self._precompute_num_data()
        self.den_data = self._precompute_den_data()
        self.stairs = self._precompute_stairs()
        # TODO: add a way to _test() all bounds generated during a given
        # computation
        #if self.ctx is not None and self.ctx.check_bounds:
        #    self._test()

    def __repr__(self):
        n = self.den.variable_name()
        fmt = ("max(\n  {rat},\n{stairs}\n)" if self.stairs else "({rat})")
        ratstr = "\n+ ".join(
                "|({num})/({den})|".format(
                    num=num,
                    den=(self.den if t == 0 else "(...)^" + str(t+1)))
                for t, num in enumerate(self.nums))
        stairsstr = ',\n'.join("  {}\tfor  {} <= {}".format(val, n, edge)
                                for edge, val in self.stairs)
        return fmt.format(rat=ratstr, stairs=stairsstr)

    def asympt_repr(self):
        r"""
        Simplified repr() that gives an asymptotic equivalent of the bound
        instead of a longer description valid for all n.
        """
        terms = []
        for t, num in enumerate(self.nums):
            deg = num.degree() - (t + 1)*self.den.degree()
            lim = abs(IC(
                num.leading_coefficient()/self.den.leading_coefficient()))
            if deg == 0:
                terms.append(str(lim))
            else:
                terms.append("{}*n^{}".format(lim, deg))
        # note to self: in this version of the code, stairs is irrelevant to the
        # asymptotics (its limit is always zero)
        if len(terms) > 1:
            return "~(" + " + ".join(terms) + ")"
        else:
            return "~" + terms[0]

    # Possible improvement: extract and cache???
    def _precompute_den_data(self):
        r"""
        Return a lower bound on self.den/n^r (where r = deg(self.den)) in the
        format that _lbound_den expects in self.den_data.

        OUTPUT:

        A list of tuples (root, mult, n_min, global_lbound) where
        - root ranges over a subset of the roots of den;
        - mult is the multiplicity of root in den;
        - n_min is n integer s.t. |1-root/n| is nondecreasing for n ≥ nmin;
        - global_lbound is a real (ball) s.t. |1-root/n| ≥ global_lbound for all
          n ∈ ℕ (in particular, for n < n_min).
        """
        den_data = []
        for root, mult in _complex_roots(self.den):
            re = root.real()
            # When Re(α) ≤ 0, the sequence |1-α/n| decreases to 1.
            if safe_le(re, IR.zero()):
                continue
            # Otherwise, it first decreases to its minimum (which may be 0 if
            # α is an integer), then increases to 1. We precompute the minimum
            # and a value of n after which the sequence is nondecreasing.
            crit_n = root.abs()**2/re
            ns = srange(ZZ(crit_n.floor()), ZZ(crit_n.ceil()) + 1)
            n_min = ns[-1]
            # When the minimum over ℕ is reached at an exceptional index, we
            # want to "skip" it in the computation of the global bound. So we
            # replace each candidate argmin that is an exceptional index by the
            # two adjacent integers--using the fact that the candidates form a
            # range. (Consecutive exceptional indices are fairly common!)
            while ns[-1] in self.exn:
                ns.append(ns[-1] + 1) # append to avoid overwriting ns[0]
            while ns[0] in self.exn:
                ns[0] -= 1
            global_lbound = IR.one().min(*(
                    (IC.one() - root/n).abs()
                    for n in ns if not n in self.exn))
            global_lbound = global_lbound.below_abs()**mult # point ball
            den_data.append((root, mult, n_min, global_lbound))
        return den_data

    def _lbound_den(self, n):
        r"""
        A *nondecreasing* lower bound on prod[den(α) = 0](|1-α/n|) valid
        for n ∈ ℕ \ {exceptions}.

        self.den_data must be set (using self._precompute_den_data) before
        calling this method.
        """
        assert n not in self.exn
        if n > self._den_converged:
            return self.almost_one
        res = IR.one()
        for root, mult, n_min, global_lbound in self.den_data:
            if n < n_min:
                # note that global_lbound already takes mult into account
                res *= global_lbound
            else:
                res *= abs((IC.one() - root/n))**mult
        if safe_ge(res, self.almost_one):
            self._den_converged = n
            return self.almost_one # just so that the sequence is nondecreasing
        return res

    def _precompute_num_data(self):
        r"""
        Return the list of self.num[t](n)/n^((t+1)·r), r = deg(self.den),
        *as polynomials in 1/n*, in the format _bound_num expects in
        self.num_data.
        """
        deg = 0
        num_data = []
        for num in self.nums:
            deg += self.den.degree()
            assert num.degree() <= deg
            #rev = num.reverse(deg)
            rev = Polynomial.reverse(num, deg) # work around sage bug #21194
            num_data.append(rev.change_ring(IC))
        return num_data

    def _bound_num(self, ord, n):
        r"""
        A very simple upper bound on |num(n)/n^((t+1)·r)|, nonincreasing with n.

        This method simply evaluates the reciprocal polynomial of num, rescaled
        by a suitable power of n, on a interval of the form [0,1/n]. (It works
        for exceptional indices, but doesn't do anything clever to take
        advantage of them.)

        self.num_data must be set (using self._precompute_num_data) before
        calling this function.
        """
        rcpq_num = self.num_data[ord]
        almost_lim = abs(rcpq_num[0])/self.almost_one
        if n > self._num_converged[ord]:
            return almost_lim
        iv = IR.zero().union(~IR(n))
        bound = rcpq_num(iv).above_abs()
        if bound < almost_lim: #safe_le(bound, almost_lim):
            self._num_converged[ord] = n
            return almost_lim # so that the sequence of bounds is nonincreasing
        return bound

    def _precompute_stairs(self):
        r"""
        Return the data structure expected by _bound_exn in self.stairs.

        self.num_data and self.den_data need to be set.

        OUTPUT:

        A list of pairs (edge, val), ordered by increasing edge, such that
        |self.ref(n)| ≤ val for all n ≥ edge.
        """
        if not self.exn:
            return []
        stairs = [(infinity, IR.zero())]
        for n in sorted(self.exn, reverse=True):
            # We need the global bound to be nonincreasing, so we take the max
            # of the exceptional value and the next ordinary index.
            for next in itertools.count(n):
                if next not in self.exn:
                    break
            val = self.exn[n].max(self._bound_rat(next))
            if val.upper() > stairs[-1][1].upper():
                stairs.append((n, val))
        stairs.reverse()
        stairs.pop() # remove (∞,0) (so that stairs == [] makes sense, + faster)
        return stairs

    def _bound_exn(self, n):
        r"""
        A *nonincreasing* staircase function defined on the whole of ℕ that
        bounds the values of ref(k) for all k ≥ n whenever n is an exceptional
        index (so that max(_bound_exn(n), _bound_rat(n)) is nonincreasing).

        self.stairs must be set (using _precompute_stairs) before calling this
        method.
        """
        # Return the value associated to the smallest step larger than n. (This
        # might be counter-intuitive!)
        for (edge, val) in self.stairs:
            if n <= edge:
                return val
        return IR.zero()

    def _bound_rat(self, n):
        lden = self._lbound_den(n)
        bound = sum(
                self._bound_num(t, n)/lden**(t+1)
                for t in xrange(len(self.num_data)))
        if not bound.is_finite():
            return IR(infinity) # replace NaN by +∞ (as max(NaN, 42) = 42)
        return bound

    def __call__(self, n):
        bound_rat = IR.zero() if n in self.exn else self._bound_rat(n)
        bound_exn = self._bound_exn(n)
        return bound_rat.max(bound_exn)

    def ref(self, n):
        if n in self.exn:
            return abs(self.exn[n])
        else:
            return sum(
                    (IC(num(n))/IC(self.den(n))**(i+1)).abs()
                    for i, num in enumerate(self.nums))

    def plot(self, rng=xrange(100)):
        r"""
        Plot this bound and its reference value.

        EXAMPLES::

            sage: from ore_algebra.analytic.bounds import bound_ratio_derivatives
            sage: Pols.<n> = QQ[]
            sage: i = QuadraticField(-1).gen()
            sage: bnd = bound_ratio_derivatives(
            ....:     CBF(i)*n+42, n*(n-3)*(n-i-20), [(0,1),(3,1)])
            sage: bnd.plot()
            Graphics object consisting of 2 graphics primitives
            sage: bnd.plot(xrange(30))
            Graphics object consisting of 2 graphics primitives
        """
        from sage.plot.plot import list_plot
        p1 = list_plot(
                [(k, RR(self.ref(k).upper()))
                    for k in rng if self.ref(k).is_finite()],
                plotjoined=True, color='black', scale="semilogy")
        # Plots come up empty when one of the y-coordinates is +∞, so we may as
        # well start with the first finite value.
        rng2 = itertools.dropwhile(lambda k: self(k).is_infinity(), rng)
        p2 = list_plot(
                [(k, RR(self(k).upper())) for k in rng2],
                plotjoined=True, color='blue', scale="semilogy")
        return p1 + p2

    def _test(self, nmax=100):
        deg = self.den.degree()
        for n in range(nmax):
            if n not in self.exn:
                lb = self._lbound_den(n)
                assert not (lb*IR(n)**deg > IC(self.den(n)).abs())
                if n + 1 not in self.exn:
                    assert not (self._lbound_den(n+1) < lb)
                for i, num in enumerate(self.nums):
                    bound = self._bound_num(i, n)
                    assert not (bound*IR(n)**((i+1)*deg) < IC(num(n)).abs())
                    assert not (bound < self._bound_num(i, n+1))
            bound = self(n)
            next = self(n+1)
            assert not (bound < self.ref(n))
            assert not (bound < next)

# Possible improvement: better take into account the range of derivatives needed
# at each step.
def bound_ratio_derivatives(num, den, nat_poles):
    r"""
    Compute a nonincreasing sequence that bounds the sequence |n·num(n)/den(n)|
    where n ranges over the natural numbers (when nat_poles=[]) or (more
    generally) a related sequence that comes up in bounds on rational operators.

    INPUT:

    - num, den - polynomials with complex coefficients, with deg(num) < deg(n);
    - nat_poles - list of pairs, a subset of the natural zeros of den and their
      multiplicities, typically
      - either the full list of integer zeros (or a “right segment”), in the
        context of evaluations at regular singular points,
      - or empty if one is not interested in derivatives and willing to do with
        an infinite bound up to the rightmost integer zero of den.

    OUTPUT:

    A :class:`RatSeqBound` representing a nonincreasing b: ℕ → [0, ∞] such that

        b(n) ≥ sum(|n·f^(t)(n)/t!|, t=0..M),                (n, _) ∉ nat_poles,
        b(n) ≥ (some similar sum with t shifted),           (n, m) ∈ nat_poles,

    where f = num/den and (currently) M ≥ sum[(_, m) ∈ nat_poles](m).

    NOTES:

    - In the main application this is intended for, den is the indicial equation
      of a differential operator and num is another coefficient of some related
      recurrence operator, both shifted so that some root of interest of the
      indicial equation goes to zero.
    - In the application, we probably don't really need b to be nonincreasing,
      only that b(n) ≥ min(|num(k)/den(k)|, ...) when 0 ≤ n ≤ k.

    EXAMPLES::

        sage: Pols.<n> = QQ[]
        sage: from ore_algebra.analytic.bounds import bound_ratio_derivatives

        sage: bnd = bound_ratio_derivatives(Pols(1), n*(n-1), {}); bnd
        (|(n)/(n^2 - n)|)
        sage: [bnd(k) for k in range(5)]
        [[+/- inf], [+/- inf], [1.000...], [0.500...], [0.333...]]

        sage: bnd = bound_ratio_derivatives(-n, n*(n-3), [(0,1), (3,1)]); bnd
        max(
          |(-n^2)/(n^2 - 3*n)|
        + |(n^3)/((...)^2)|
        + |(-n^4)/((...)^3)|,
          [84.2666...]      for  n <= 0,
          [12.2666...]      for  n <= 3
        )
        sage: [(bnd.ref(k), bnd(k)) for k in range(5)]
        [(0,          [84.266...]),
         (0.875...,   [84.266...]),
         (6.000...,   [28.266...]),
         ([3.000...], [12.266...]),
         (12.000...,  [12.266...])]

        sage: bound_ratio_derivatives(n, n, {})
        Traceback (most recent call last):
        ...
        ValueError: expected deg(num) < deg(den)

    TESTS::

        sage: bnd = bound_ratio_derivatives(Pols(3), n, {}); bnd
        (|(3*n)/(n)|)
        sage: bnd._test()
        sage: i = QuadraticField(-1).gen()
        sage: bound_ratio_derivatives(Pols(1), n+i, {})._test()
        sage: bound_ratio_derivatives(-n, n*(n-3), [(3,1)])._test()
        sage: bound_ratio_derivatives(-n, n*(n-3), [(0,1)])._test()
        sage: bound_ratio_derivatives(-n, n*(n-3), [(0,1),(3,1)])._test()
        sage: bound_ratio_derivatives(CBF(i)*n, n*(n-QQbar(i)), [(0,1),(3,1)])._test()
        sage: bound_ratio_derivatives(n^5-100*n^4+2, n^3*(n-1/2)*(n-2)^2, [(0,3), (2,2)])._test()
    """
    if num.degree() >= den.degree():
        raise ValueError("expected deg(num) < deg(den)")

    Pol = num.parent()

    max_mult = max(mult for _, mult in nat_poles) if nat_poles else 0
    derivatives = 1 + sum(mult for _, mult in nat_poles)
    Jets = utilities.jets(IC, 'X', derivatives + max_mult)

    # Compute exceptions first, since we are going to overwrite num.
    # Sage won't return Laurent series here.
    ex_series = {}
    for n, mult in nat_poles:
        pert = Jets([n, 1]) # n + X
        # den has a root of order mult at n, so den(pert) = O(X^mult), but the
        # computed value might include terms of degree < mult with interval
        # coefficients containing zero
        ex_series[n] = n*num(pert)/Jets(den(pert).lift() >> mult)
    exns = { n: sum(ser[t].abs() for t in xrange(derivatives))
             for n, ser in ex_series.iteritems() }

    denpow = den
    dendiff = den.derivative()
    nums = []
    # Possible improvement: find a way to evaluate the derivatives on an
    # interval directly, without explicitly computing their numerators???
    for t in range(derivatives):
        # At this point denpow = den^(t+1) and num/denpow = D^t(orig num/den)/t!
        nums.append(num << 1) # the shift accounts for the "n·" in the spec
        num = num.derivative()*den/(t + 1) - num*dendiff
        denpow *= den

    seqbound = RatSeqBound(nums, den, exns)
    return seqbound

################################################################################
# Bounds for differential equations
################################################################################

def bound_polynomials(pols):
    r"""
    Compute a common majorant polynomial for the polynomials in ``pol``.

    Note that this returns a _majorant_, not some kind of enclosure.

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
    PolyIC = pols[0].parent().change_ring(IC)
    deg = max(pol.degree() for pol in pols)
    val = min(deg, min(pol.valuation() for pol in pols))
    pols = [PolyIC(pol) for pol in pols] # TBI
    order = Integer(len(pols))
    PolyIR = PolyIC.change_ring(IR)
    def coeff_bound(n):
        return IR.zero().max(*(
            pols[k][n].above_abs()
            for k in xrange(order)))
    maj = PolyIR([coeff_bound(n)
                  for n in xrange(val, deg + 1)])
    maj <<= val
    return maj

class DiffOpBound(object):
    r"""
    A "bound on the inverse" of a differential operator at a regular point.

    This is an object that can be used to bound the tails of logarithmic power
    series solutions with terms supported by leftmost + ℕ of a differential
    operator. Given a residual q = dop·ỹ where ỹ(z) = y[:N](z) is the truncation
    at order N of some (logarithmic) solution y of dop·y = 0, it can be used
    to compute a majorant series of the coefficients u[0], u[1], ... of the tail

        y(z) - ỹ(z) = u[0](z) + u[1](z)·log(z) + u[2](z)·log(z)² + ···.

    That majorant series of the tail is represented by a HyperexpMajorant.

    Note that multiplying dop by a rational function changes the residual.

    More precisely, a DiffOpBound represents a *parametrized* formal power
    series v[n](z) with the property that, if N and ỹ are as above with
    N - n ∈ ℕ, then v[n](z)·B(q)(z), for some logarithmic polynomial B(q)
    derived from q, is a majorant of y(z) - ỹ(z).
    XXX: Here B(q) can be taken to be any majorant polynomial of q, but tighter
    choices are possible (see below for details).

    The sequence v[n](z) is of the form

        1/den(z) * exp(int(cst*num[n](z)/den(z) + pol[n](z)))

    where

    * num[n](z) and pol[n](z) are polynomials with coefficients depending on n
      (given by RatSeqBound objects), with val(num[n]) >= deg(pol[n]),

    * den(z) is a polynomial (with constant coefficients),

    * cst is a constant.

    XXX: DiffOpBounds are refinable.

    DATA:

    - ``dop`` - the operator to which the bound applies (and which should be
        used to compute the residuals),

    - ``cst`` - constant (real ball),

    - ``majseq_pol_part`` - *list* of coefficients of ``pol_part``,

    - ``majseq_num`` - *list* of coefficients [c[d], c[d+1], ...] of
      ``num``, starting at degree d = deg(pol_part) + 1,

    - ``maj_den`` - ``Factorization``,

    - ``ind`` - polynomial to be used in the computation of tail bounds
        from residuals, typically the indicial polynomial of ``dop``.

    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: from ore_algebra.analytic.bounds import *
        sage: Dops, x, Dx = Diffops()

    A majorant sequence::

        sage: maj = DiffOpBound((x^2 + 1)*Dx^2 + 2*x*Dx, pol_part_len=0)
        sage: print(maj.__repr__(asympt=False))
        1/((-x + [0.994...])^2)*exp(int(POL+1.000...*NUM/(-x + [0.994...])^2))
        where
        POL=0,
        NUM=(|(0)/(n^2 - n)|)*z^0 + (|(-2*n^2 - 2*n)/(n^2 - n)|)*z^1

    A majorant series extracted from that sequence::

        sage: maj(3)
        (1.00... * (-x + [0.994...])^-2)*exp(int([4.000...]...)^2)))

    An example with a nontrivial polynomial part::

        sage: dop = (x+1)*(x^2+1)*Dx^3-(x-1)*(x^2-3)*Dx^2-2*(x^2+2*x-1)*Dx
        sage: DiffOpBound(dop, pol_part_len=3)
        1/((-x + [0.9965035284306323 +/- 2.07e-17])^3)*exp(int(POL+1.000...*NUM/(-x + [0.9965035284306323 +/- 2.07e-17])^3)) where
        POL=~6.0000000000...*z^0 + ~3.0000000000...*z^1 + ~5.0000000000...*z^2,
        NUM=~7.0000000000...*z^3 + ~2.0000000000...*z^4 + ~5.0000000000...*z^5

    TESTS::

        sage: print(DiffOpBound(Dx - 1, pol_part_len=0).__repr__(asympt=False))
        1/(1.000...)*exp(int(POL+1.000...*NUM/1.000...))
        where
        POL=0,
        NUM=(|(-n)/(n)|)*z^0

        sage: QQi.<i> = QuadraticField(-1)
        sage: for dop in [
        ....:     # orders <= 1 are not supported
        ....:     Dx, Dx - 1, i*Dx, Dx + i, Dx^2,
        ....:     (x^2 + 1)*Dx^2 + 2*x*Dx,
        ....:     Dx^2 - x*Dx
        ....: ]:
        ....:     DiffOpBound(dop)._test()

        sage: for l in xrange(10):
        ....:     DiffOpBound(Dx - 5*x^4, pol_part_len=l)._test()
        ....:     DiffOpBound((1-x^5)*Dx - 5*x^4, pol_part_len=l)._test()

        sage: from ore_algebra.analytic.bounds import _test_diffop_bound
        sage: _test_diffop_bound() # long time
    """

    def __init__(self, dop, leftmost=ZZ.zero(), special_shifts=[],
            refinable=True, pol_part_len=2, bound_inverse="simple"):
        r"""
        INPUT:

        * special_shifts: list of nonneg integers n s.t. leftmost+n is a root of
          the indicial equation where we are interested in "new" powers of log
          that may appear, with associated multiplicities

        .. WARNING::

            The bounds depend on residuals computed using the “normalized”
            operator ``self.dop``, not the operator ``dop`` given as input to
            ``__init__``. The normalized operator is the product of ``dop`` by a
            power of x.

        """

        logger.info("bounding local operator...")

        self.stats = BoundDiffopStats()
        self.stats.time_total.tic()

        if not dop.parent().is_D():
            raise ValueError("expected an operator in K(x)[D]")
        _, Pols_z, _, dop = dop._normalize_base_ring()
        self._dop_D = dop
        self.dop = dop_T = dop.to_T('T' + Pols_z.variable_name()) # slow
        self._rcoeffs = _dop_rcoeffs_of_T(dop_T)

        self.Poly = Pols_z.change_ring(IR) # TBI
        one = self.Poly.one()
        self.__facto_one = Factorization([(one, 1)], unit=one, sort=False, simplify=False)

        lc = dop_T.leading_coefficient()
        if lc.is_term() and not lc.is_constant():
            raise ValueError("irregular singular operator", dop)

        self.leftmost = leftmost
        self.special_shifts = special_shifts

        self.bound_inverse = bound_inverse
        self.majseq_pol_part = []
        self.refinable = refinable
        self._effort = 0

        self._update_den_bound()
        self._update_num_bound(pol_part_len)

        self.stats.time_total.toc()
        logger.info("...done, time: %s", self.stats)

    def __repr__(self, asympt=True):
        fmt = ("1/({den})*exp(int(POL+{cst}*NUM/{den})) where\n"
               "POL={pol},\n"
               "NUM={num}\n")
        def pol_repr(ratseqbounds, shift=0):
            if len(ratseqbounds):
                return " + ".join(
                        "{}*z^{}".format(
                            c.asympt_repr() if asympt else c,
                            n + shift)
                        for n, c in enumerate(ratseqbounds))
            else:
                return 0
        return fmt.format(
                cst=self.cst, den=self.maj_den,
                num=pol_repr(self.majseq_num, shift=len(self.majseq_pol_part)),
                pol=pol_repr(self.majseq_pol_part))

    @cached_method
    def _poles(self):
        lc = self.dop.leading_coefficient()
        try:
            return lc.roots(CIF)
        except NotImplementedError:
            return lc.change_ring(QQbar).roots(CIF)

    def _update_den_bound(self):
        den = self.dop.leading_coefficient()

        Poly = den.parent().change_ring(IR)
        if den.degree() <= 0:
            facs = []
        # below_abs()/lower() to get thin intervals
        elif self.bound_inverse == "simple":
            rad = abs_min_nonzero_root(den).below_abs(test_zero=True)
            facs = [(Poly([rad, -1]), den.degree())]
        elif self.bound_inverse == "solve":
            facs = [(Poly([IR(iv.abs().lower()), -1]), mult)
                    for iv, mult in self._poles()]
        else:
            raise ValueError("algorithm")
        self.cst = ~abs(IC(den.leading_coefficient()))
        self.maj_den = Factorization(facs, unit=Poly.one(), sort=False, simplify=False)

    def _update_num_bound(self, pol_part_len):

        self.stats.time_decomp_op.tic()
        lc = self.dop.leading_coefficient()
        inv = lc.inverse_series_trunc(pol_part_len + 1) # XXX: incremental Newton ?
        MPol, (z, n) = self.dop.base_ring().extend_variables('n').objgens()
        # Including rcoeffs[-1] here actually is redundant, as, by construction,
        # the only term in first to involve n^ordeq will be 1·n^ordeq·z^0.
        first = sum(n**j*pol._mul_trunc_(inv, pol_part_len + 1)
                    for j, pol in enumerate(self._rcoeffs))
        first_nz = first.polynomial(z)
        first_zn = first.polynomial(n)
        logger.log(logging.DEBUG - 1, "first: %s", first_nz)
        assert first_nz[0] == self._dop_D.indicial_polynomial(z, n).monic()
        assert all(pol.degree() < self.dop.order() for pol in first_nz >> 1)

        T = self.dop.parent().gen()
        pol_part = sum(T**j*pol for j, pol in enumerate(first_zn)) # slow
        # logger.debug("pol_part: %s", pol_part)
        rem_num = self.dop - pol_part*lc # in theory, slow for large pol_part_len
        logger.log(logging.DEBUG - 1, "rem_num: %s", rem_num)
        it = enumerate(_dop_rcoeffs_of_T(rem_num)) # slow
        rem_num_nz = MPol(sum(n**j*pol for j, pol in it)).polynomial(z)
        assert rem_num_nz.valuation() >= pol_part_len + 1
        rem_num_nz >>= (pol_part_len + 1)
        logger.log(logging.DEBUG - 1, "rem_num_nz: %s", rem_num_nz)
        self.stats.time_decomp_op.toc()

        # XXX: make this independent of pol_part_len?
        alg_idx = self.leftmost + first_nz.base_ring().gen()
        # XXX: check if ind needs to be shifted (ind(n ± leftmost))
        self.ind = first_nz[0](alg_idx)

        # We ignore the coefficient first_nz[0], which amounts to multiplying
        # the integrand by z⁻¹, as prescribed by the theory. Since, by
        # definition, majseq_num starts at the degree following that of
        # majseq_pol_part, it gets shifted as well.
        old_pol_part_len = len(self.majseq_pol_part)
        self.stats.time_bound_ratio.tic()
        self.majseq_pol_part.extend([
                bound_ratio_derivatives(first_nz[i](alg_idx), self.ind,
                                        self.special_shifts)
                for i in xrange(old_pol_part_len + 1, pol_part_len + 1)])
        assert len(self.majseq_pol_part) == pol_part_len
        self.majseq_num = [
                bound_ratio_derivatives(pol(alg_idx), self.ind,
                                        self.special_shifts)
                for pol in rem_num_nz]
        self.stats.time_bound_ratio.toc()

    def refine(self):
        # XXX: make it possible to increase the precision of IR, IC
        if not self.refinable:
            logger.debug("refining disabled")
            return
        self._effort += 1
        logger.info("refining majorant (effort = %s)...", self._effort)
        self.stats.time_total.tic()
        if self.bound_inverse == 'simple':
            self.bound_inverse = 'solve'
            self._update_den_bound()
        else:
            self._update_num_bound(max(2, 2*self.pol_part_len()))
        self.stats.time_total.toc()
        logger.info("...done, cumulative time: %s", self.stats)

    def pol_part_len(self):
        return len(self.majseq_pol_part)

    def __call__(self, n):
        r"""
        Return a term of the majorant sequence.
        """
        maj_pol_part = self.Poly([fun(n) for fun in self.majseq_pol_part])
        # XXX: perhaps use sparse polys or add explicit support for a shift
        # in RationalMajorant
        maj_num_pre_shift = self.Poly([fun(n) for fun in self.majseq_num])
        maj_num = (self.cst*maj_num_pre_shift) << self.pol_part_len()
        terms = [(maj_pol_part, self.__facto_one), (maj_num, self.maj_den)]
        rat_maj = RationalMajorant(terms)
        # The rational part “compensates” the change of unknown function
        # involving the leading coefficient of the operator. We compute ~den
        # by hand because Factorization.__invert__() can be very slow.
        maj = HyperexpMajorant(integrand=rat_maj, num=self.Poly.one(),
                den=self.maj_den)
        return maj

    # Extracted from tail_majorant for (partial) compatibility with the regular
    # singular case.
    def maj_eq_rhs(self, residuals):
        abs_residual = bound_polynomials(residuals)
        logger.debug("lc(abs_res) = %s", abs_residual.leading_coefficient())
        # In general, a majorant series for the tail of order n is given by
        # self(n)(z)*int(t⁻¹*aux(t)/self(n)(t)) where aux(t) is a polynomial
        # s.t. |aux[k]| >= (k/indicial_eq(k))*abs_residual[k]. This bound is not
        # very convenient to compute. But since self(n) has nonnegative
        # coefficients and self(n)(0) = 1, we can replace aux by aux*self(n) in
        # the formula. (XXX: How much do we lose?) Since k/indicial_eq(k) <= 1
        # (ordinary point!), we could in fact take aux = abs_residual*self(n),
        # yielding a very simple bound. (We would lose an additional factor of
        # about n^(ordeq-1).)
        Pols = abs_residual.parent()
        aux = Pols(dict(
            (k, (c*k/self.ind(k)).above_abs())
            for k, c in abs_residual.dict().iteritems()))
        return aux

    # XXX: make interval evaluation more precise? (not crucial as we only need
    # an upper bound, but...)
    def tail_majorant(self, n, majeqrhs):
        r"""
        Bound the tails of order ``N`` of solutions of ``self.dop(y) == 0``.

        INPUT:

        - ``n`` - integer, ``n <= N``, typically ``n == N``. (Technically, this
          function should work for ``n < N `` too, but this is unused, untested,
          and not very useful with the current code structure.)

        - ``residuals`` - list of polynomials of the form ``self.dop(y[:N])``
          where y satisfies ``self.dop(y) == 0``.

        OUTPUT:

        A (common) majorant series of the tails ``y[N:](z)`` of the solutions
        corresponding to the elements of ``residuals``.
        """
        assert majeqrhs.valuation() >= n >= self.dop.order() >= 1
        maj = self(n)
        logger.debug("maj(%s) = %s", n, maj)
        pol = (majeqrhs >> 1).integral()
        assert pol.parent().is_sparse()
        maj *= pol
        logger.debug("maj = %s", maj)
        return maj

    # XXX: rename ord to rows?
    def matrix_sol_tail_bound(self, n, rad, majeqrhs, ord=None):
        r"""
        Bound the Frobenius norm of the tail starting of order ``n`` of the
        series expansion of the matrix ``(y_j^(i)(z)/i!)_{i,j}`` where the
        ``y_j`` are the solutions associated to the elements of ``residuals``,
        and ``0 ≤ j < ord``. The bound is valid for ``|z| < rad``.
        """
        if ord is None: ord=self.dop.order()
        maj = self.tail_majorant(n, majeqrhs)
        # Since (y[n:])' << maj => (y')[n:] << maj, this bound is valid for the
        # tails of a column of the form [y, y', y''/2, y'''/6, ...] or
        # [y, θy, θ²y/2, θ³y/6, ...].
        col_bound = maj.bound(rad, derivatives=ord)
        logger.debug("maj(%s).bound() = %s", n, self(n).bound(rad))
        logger.debug("n = %s, col_bound = %s", n, col_bound)
        return IR(ord).sqrt()*col_bound

    def _test(self, ini=None, prec=100):
        r"""
        Check that the majorants produced by this DiffOpBound bound the tails of
        the solutions of the associated operator.

        This is a heuristic check for testing purposes, nothing rigorous!

        EXAMPLES::

            sage: from ore_algebra.analytic.ui import *
            sage: from ore_algebra.analytic.bounds import *
            sage: Dops, x, Dx = Diffops()
            sage: maj = DiffOpBound(Dx - 1)
            sage: maj._test()
            sage: maj._test([3], 200)
        """
        ord = self.dop.order()
        if ini is None:
            from sage.rings.number_field.number_field import QuadraticField
            QQi = QuadraticField(-1)
            ini = [QQi.random_element() for _ in xrange(ord)]
        sol = self.dop.power_series_solutions(prec)
        Series = PowerSeriesRing(CBF, self.dop.base_ring().variable_name())
        ref = sum((ini[k]*sol[k] for k in xrange(ord)), Series(0)).polynomial()
        for n in [ord, ord + 1, ord + 2, ord + 50]:
            logger.info("truncation order = %d", n)
            if n + 30 >= prec:
                warnings.warn("insufficient precision")
            resid = self.dop(ref[:n])
            resid = PolynomialRing(resid.base_ring(), resid.variable_name(),
                                                             sparse=True)(resid)
            # we know a priori that val(resid) >= n mathematically, but interval
            # computations may give inexact zeros for some of the coefficients
            assert all(c.contains_zero() for c in resid[:n])
            resid = (resid >> n) << n
            maj = self.tail_majorant(n, self.maj_eq_rhs([resid]))
            tail = (ref >> n) << n
            logger.info("%s << %s", Series(tail, n+30), maj.series(0, n+30))
            maj._test(tail)

def _dop_rcoeffs_of_T(dop):
    """
    Compute the coefficients of dop as an operator in θ but with θ on the left.
    """
    # Perhaps better: work with a "true" Ore algebra K[θ][z]. Use Euclidean
    # division to compute the truncation in DiffOpBound._update_num_bound.
    # Extracting the Qj(θ) would then be easy, and I may no longer need the
    # coefficients of θ "on the right".
    Pols_z = dop.base_ring()
    Pols_n, n = Pols_z.change_var('n').objgen()
    Rops = ore_algebra.OreAlgebra(Pols_n, 'Sn')
    rop = dop.to_S(Rops) if dop else Rops(0)
    bwd_rop_as_pol = (rop.polynomial().reverse().change_variable_name('Bn')
                         .map_coefficients(lambda pol: pol(n-rop.order())))
    MPol = Pols_n.extend_variables('Bn')
    bwd_rop_rcoeffof_n = MPol(bwd_rop_as_pol).polynomial(MPol.gen(0)).list()
    val = min(pol.valuation() for pol in dop.coefficients()
              + [Pols_z.zero()]) # TBI; 0 to handle dop=0
    res = [Pols_z(c) << val for c in bwd_rop_rcoeffof_n]
    return res

class BoundDiffopStats(utilities.Stats):
    """
    Store timings for various parts of the bound computation algorithm.
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self.time_total = utilities.Clock("total")
        self.time_bound_ratio = utilities.Clock("doing RatSeqBound precomp.")
        self.time_decomp_op = utilities.Clock("decomposing op")

def _test_diffop_bound(
        ords=xrange(1, 5),
        degs=xrange(5),
        pplens=[1, 2, 5],
        prec=100
    ):
    r"""
    Randomized testing of :func:`DiffOpBound`.

    EXAMPLES::

        sage: import logging; logging.basicConfig(level=logging.INFO)
        sage: from ore_algebra.analytic.bounds import _test_diffop_bound
        sage: _test_diffop_bound() # not tested - done in DiffOpBound
        INFO:ore_algebra.analytic.bounds:testing operator: (-i + 2)*Dx + i - 1
        ...
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
                            for _ in xrange(ord + 1)])
            logger.info("testing operator: %s", dop)
            for pplen in pplens:
                maj = DiffOpBound(dop, pol_part_len=pplen)
                maj._test(prec=prec)

def residual(n, bwrec_nplus, last, z):
    r"""
    Compute the polynomial residual (up to sign) obtained by a applying a diff
    op P to a partial sum of a power series solution y of P·y=0.

    INPUT:

    - ``n`` -- truncation order
    - ``bwrec_nplus`` -- nested list/iterable ::

        [[b[0](n),     ..., b[s](n)],
         [b[0](n+1),   ..., b[s](n+1)],
         ...
         [b[0](n+s-1), ..., b[s](n+s-1)]]

      of coefficients evaluated at n of the recurrence operator b[0] + b[1] S⁻¹
      + ··· + b[s] S^(-s) associated to P (by the direct substitution x ⟼ S⁻¹,
      θ ⟼ n; no additional multiplication by x^k is allowed!)
    - ``last`` -- the last s coefficients u[n-1], u[n-2], ... u[n-s] of the
      truncated series, in that order
    - ``z`` -- variable name for the result

    EXAMPLES::

        sage: from ore_algebra import DifferentialOperators
        sage: from ore_algebra.analytic.bounds import *
        sage: dop, t, Dt = DifferentialOperators(QQ, 't')
        sage: Pol.<n> = QQ[]

        sage: trunc = t._exp_series(5); trunc
        1/24*t^4 + 1/6*t^3 + 1/2*t^2 + t + 1
        sage: bwrec = [n, Pol(1)]
        sage: bwrec_nplus = [[pol(5) for pol in bwrec]]
        sage: residual(5, bwrec_nplus, [trunc[4]], t)
        ([0.0416666666666667 +/- 4.26e-17])*t^5
        sage: (Dt - 1).to_T('Tt')(trunc).change_ring(CBF)
        ([-0.0416666666666667 +/- 4.26e-17])*t^5

    Note that using Dt - 1 instead of θt - t makes a difference in the result,
    since it amounts to a division by t::

        sage: (Dt - 1)(trunc).change_ring(CBF)
        ([-0.0416666666666667 +/- 4.26e-17])*t^4

    ::

        sage: trunc = t._sin_series(5) + t._cos_series(5)
        sage: bwrec = [n*(n-1), Pol(0), Pol(1)]
        sage: bwrec_nplus = [[pol(5+i) for pol in bwrec] for i in [0,1]]
        sage: residual(5, bwrec_nplus, [trunc[4], trunc[3]], t)
        ([0.041666...])*t^6 + ([-0.16666...])*t^5
        sage: (Dt^2 + 1).to_T('Tt')(trunc).change_ring(CBF)
        ([0.041666...])*t^6 + ([-0.16666...])*t^5
    """
    ordrec = len(bwrec_nplus)
    assert ordrec == 0 or ordrec == len(bwrec_nplus[0]) - 1
    rescoef = [
        sum((bwrec_nplus[i][i+k+1])*IC(last[k])
            for k in xrange(ordrec-i))
        for i in xrange(ordrec)]
    IvPols = PolynomialRing(IC, z, sparse=True)
    return IvPols(rescoef) << n

def maj_eq_rhs_with_logs(n, bwrec, bwrec_nplus, last, z, logs):
    r"""
    Compute the rhs of a majorant equation for the tail from the last terms of a
    truncated series.

    This roughly corresponds to residual() followed by maj.maj_eq_rhs() in the
    ordinary case. TODO: more consistent interface.

    INPUT:

    - ``bwrec``: a recurrence *“in n”*, i.e. already shifted by λ
    - ``n``: int index where to evaluate the rec to compute the residual, must
      be an *generic* index (i.e. QO(λ+n)≠0, where Q0 = indicial poly)
    - ``last``: coefficients of λ+n-1, λ+n-2, ... of the solution
    - ...

    OUTPUT:

    A *polynomial* q̂ of valuation at least ``n`` (i.e., *not* shifted by λ) s.t.

        (n+i) · |∑[t=0..logs-1] ((d/dX)^t(Q0⁻¹))(λ+n+i)/t!|
              · max[k](|q[λ+n+i, k]|)                                   (*)
        ≤ q̂[n+i]

    for all i ≥ n and all k, where q[λ+j,t] is the coefficient of
    z^(λ+j)·log(z)^t/t! in the residual. Note that (*) implies

        |∑[t=0..logs-1] (n+i)/t!·((d/dX)^t(Q0⁻¹))(λ+n+i)·q[λ+n+i,k+t]| ≤ q̂[n+i].
    """
    ordrec = bwrec.order
    assert len(bwrec_nplus) >= len(bwrec_nplus[0]) - 1 == ordrec
    # Compute the coefficients of the residual:
    # residual = z^(λ + n)·(sum(rescoef[i][j]·z^i·log^j(z)/j!)
    rescoef = [[None]*logs for _ in xrange(ordrec)]
    for i in xrange(ordrec):
        for j in xrange(logs):
            # significant overestimation here (apparently not too problematic)
            rescoef[i][j] = sum(
                    IC(bwrec_nplus[i][i+k+1][p])*IC(last[k][j+p])
                    for k in xrange(ordrec - i)
                    for p in xrange(logs - j))
    # For lack of a convenient data structure to return these coefficients,
    # compute a “majorant” polynomial right away. For simplicity, we only handle
    # the generic case.
    def invlcmaj(i):
        # sum(1/t!·(1/Q0)^(t)(λ + n + i)·Sk^t)
        invlc = bwrec.eval_inverse_lcoeff_series(IC, n+i, logs)
        return sum(t.abs() for t in invlc)
    polcoef = [(n + i)*invlcmaj(i)*max(t.abs() for t in rescoef[i])
               for i in xrange(ordrec)]
    IvPols = PolynomialRing(IR, z, sparse=True)
    return IvPols(polcoef) << n





