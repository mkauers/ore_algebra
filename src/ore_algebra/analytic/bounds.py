# -*- coding: utf-8 - vim: tw=80
"""
Error bounds
"""

# TODO:
# - this module uses at least three different object types for things that are
# essentially rational fractions (QuotientRingElements, Factorizations, and
# Rational Majorants) --> simplify?

import logging, warnings

import sage.rings.polynomial.real_roots as real_roots

from sage.misc.misc_c import prod
from sage.rings.all import CIF
from sage.rings.complex_arb import CBF
from sage.rings.infinity import infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import NumberField_quadratic
from sage.rings.polynomial.polynomial_element import Polynomial
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.qqbar import QQbar, AA
from sage.rings.rational_field import QQ
from sage.rings.real_arb import RBF
from sage.rings.real_mpfi import RIF
from sage.rings.real_mpfr import RealField, RR
from sage.structure.factorization import Factorization

from ore_algebra.ore_algebra import OreAlgebra

from ore_algebra.analytic import utilities
from ore_algebra.analytic.safe_cmp import *

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

    def eval(self, ev):
        r"""
        Evaluate this majorant using the evaluator ``ev``.

        Typically the evaluator is a converter to a parent that supports all the
        basic operations (+*/, integral...) appearing in the expression of the
        majorant.
        """
        raise NotImplementedError

    def __call__(self, z):
        return self.eval(lambda obj: obj(z))

    def series(self, prec=10):
        Series = PowerSeriesRing(IR, self.variable_name, default_prec=prec)
        return self.eval(Series).truncate_powerseries(prec)

    def bound(self, rad, **kwds):
        if not safe_le(rad, self.cvrad): # intervals!
            return IR(infinity)
        else:
            return self._bound(rad, **kwds)

    def _bound(self, rad):
        return self(rad)

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
            sage: maj = RationalMajorant(Pol(1), Factorization([(1-z,1)]), Pol(0))
            sage: maj._test(11/10*z^30)
            Traceback (most recent call last):
            ...
            AssertionError: (30, [-0.10000000000000 +/- 8.00e-16], '< 0')
        """
        Series = PowerSeriesRing(IR, self.variable_name, prec)
        # CIF to work around problem with sage power series, should be IC
        ComplexSeries = PowerSeriesRing(CIF, self.variable_name, prec)
        maj = self.series(prec)
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

def _pole_free_rad(fac):
    if isinstance(fac, Factorization):
        den = [pol for (pol, mult) in fac if mult < 0]
        if all(pol.degree() == 1 and pol.leading_coefficient().abs().is_one()
               for pol in den):
            rad = IR(infinity).min(*(IR(pol[0].abs()) for pol in den))
            rad = IR(rad.lower())
            assert rad >= IR.zero()
            return rad
    raise NotImplementedError  # pb dérivées???

class RationalMajorant(MajorantSeries):
    """
    A rational power series with nonnegative coefficients, represented in the
    form pol + num/den.

    TESTS::

        sage: from ore_algebra.analytic.bounds import *
        sage: Pol.<z> = RBF[]
        sage: den = Factorization([(1-z, 2), (2-z, 1)])
        sage: maj = RationalMajorant(z^2, den, 1 + z); maj
        1.000... + 1.000...*z + z^2/((-z + 2.000...) * (-z + 1.000...)^2)
        sage: maj(z).parent()
        Fraction Field of Univariate Polynomial Ring in z over Real ball field
        with 53 bits precision
        sage: maj(1/2)
        [2.166...]
        sage: maj*(z^10)
        1.000...*z^10 + 1.000...*z^11 + z^12/((-z + 2.000...) * (-z + 1.000...)^2)
        sage: maj.bound_antiderivative()
        1.00...*z + 0.50...*z^2 + [0.33...]*z^3/((-z + 2.00...) * (-z + 1.00...)^2)
        sage: maj.cvrad
        1.000000000000000
        sage: maj.series(4)
        1.000... + 1.000...*z + 0.500...*z^2 + 1.250...*z^3 + O(z^4)
        sage: maj._test()
        sage: maj._test(1 + z + z^2/((1-z)^2*(2-z)), return_difference=True)
        [0, 0, 0, ...]
        sage: maj._test(1 + z + z^2/((1-z)*(2-z)), return_difference=True)
        [0, 0, 0, 0.5000000000000000, 1.250000000000000, ...]
    """

    def __init__(self, num, den, pol):
        if isinstance(num, Polynomial) and isinstance(den, Factorization):
            Poly = num.parent().change_ring(IR)
            if not den.unit().is_one():
                raise ValueError("expected a denominator with unit part 1")
            assert num.valuation() > pol.degree()
            assert den.universe() is Poly or den.value() == 1
            super(self.__class__, self).__init__(Poly.variable_name(),
                    cvrad=_pole_free_rad(~den))
            self.num = Poly(num)
            self.pol = Poly(pol)
            self.den = den
            self.var = Poly.gen()
        else:
            raise TypeError

    def __repr__(self):
        res = ""
        if self.pol:
            Poly = self.pol.parent()
            pol_as_series = Poly.completion(Poly.gen())(self.pol)
            res += repr(pol_as_series) + " + "
        res += self.num._coeff_repr()
        if self.den:
            res += "/(" + repr(self.den) + ")"
        return res

    def eval(self, ev):
        # may by better than den.value()(z) in some cases
        den = prod(ev(lin)**mult for (lin, mult) in self.den)
        return ev(self.pol) + ev(self.num)/den

    def bound_antiderivative(self):
        # When u, v have nonneg coeffs, int(u·v) is majorized by int(u)·v.
        # This is a little bit pessimistic but yields a rational bound,
        # avoiding antiderivatives of rational functions.
        return RationalMajorant(self.num.integral(),
                                self.den,
                                self.pol.integral())

    def __mul__(self, pol):
        """"
        Multiplication by a polynomial.

        Note that this does not change the radius of convergence.
        """
        if pol.parent() is self.num.parent():
            return RationalMajorant(self.num*pol, self.den, self.pol*pol)
        else:
            raise TypeError

class HyperexpMajorant(MajorantSeries):
    """
    A formal power series of the form rat1(z) + exp(int(rat2(ζ), ζ=0..z)), with
    nonnegative coefficients.

    TESTS::

        sage: from ore_algebra.analytic.bounds import *
        sage: Pol.<z> = RBF[]
        sage: integrand = RationalMajorant(z^2, Factorization([(1-z,1)]), 4+4*z)
        sage: rat = Factorization([(1/3-z, -1)])
        sage: maj = HyperexpMajorant(integrand, rat); maj
        ((-z + [0.333...])^-1)*exp(int(4.0... + 4.0...*z + z^2/(-z + 1.0...)))
        sage: maj.cvrad
        [0.333...]
        sage: maj.series(4)
        [3.000...] + [21.000...]*z + [93.000...]*z^2 + [336.000...]*z^3 + O(z^4)
        sage: maj._test()
    """

    def __init__(self, integrand, rat):
        if isinstance(integrand, RationalMajorant) and isinstance(rat,
                Factorization):
            cvrad = integrand.cvrad.min(_pole_free_rad(rat))
            super(self.__class__, self).__init__(integrand.variable_name, cvrad)
            self.integrand = integrand
            self.rat = rat
        else:
            raise TypeError

    def __repr__(self):
        return "({})*exp(int({}))".format(self.rat, self.integrand)

    def eval(self, ev):
        integrand = self.integrand.eval(ev)
        return ev(self.rat.value()) * integrand.integral().exp()

    def _bound(self, rad, derivatives=1):
        """
        Bound the Frobenius norm of the vector

            [g(rad), g'(rad), g''(rad)/2, ..., 1/(d-1)!·g^(d-1)(rad)]

        where d = ``derivatives`` and g is this majorant series. The result is
        a bound for

            [f(z), f'(z), f''(z)/2, ..., 1/(d-1)!·f^(d-1)(z)]

        for all z with |z| ≤ rad.
        """
        rat = self.rat.value()
        # Compute the derivatives by “automatic differentiation”. This is
        # crucial for performance with operators of large order.
        Series = PowerSeriesRing(IR, 'eps', default_prec=derivatives)
        pert_rad = Series([rad, 1], derivatives)
        ser = rat(pert_rad)*self.integrand(pert_rad).integral().exp()
        rat_part = sum(coeff**2 for coeff in ser.truncate(derivatives))
        exp_part = (2*self.integrand.bound_antiderivative()(rad)).exp()
        return (rat_part*exp_part).sqrtpos()

    def __mul__(self, pol):
        """"
        Multiplication by a polynomial.

        Note that this does not change the radius of convergence.
        """
        return HyperexpMajorant(self.integrand, self.rat*pol)

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
            logger.info("failed to bound the roots of %s, "
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

def bound_inverse_poly(den, algorithm="simple"):
    """
    Return a majorant series ``cst/fac`` for ``1/den``, as a pair ``(cst, fac)``
    where ``fac`` is a ``Factorization`` object with linear factors.

    EXAMPLES::

        sage: from ore_algebra.analytic.bounds import *
        sage: Pol.<x> = QQ[]
        sage: pol = 2*x + 1
        sage: cst, den = bound_inverse_poly(pol)
        sage: maj = RationalMajorant(Pol(cst), den, Pol(0)); maj
        0.5000000000000000/(-x + [0.4972960558102933 +/- 4.71e-17])
        sage: maj._test(1/pol)

    TESTS::

        sage: for pol in [Pol(1), Pol(-42), 2*x+1, x^3 + x^2 + x + 1, 5*x^2-7]:
        ....:     for algo in ['simple', 'solve']:
        ....:         cst, den = bound_inverse_poly(pol, algorithm=algo)
        ....:         maj = RationalMajorant(Pol(0)+cst, den, Pol(0))
        ....:         maj._test(1/pol)
    """
    Poly = den.parent().change_ring(IR)
    if den.degree() <= 0:
        factors = []
    else:
        # below_abs()/lower() to get thin intervals
        if algorithm == "simple":
            rad = abs_min_nonzero_root(den).below_abs(test_zero=True)
            factors = [(Poly([rad, -1]), den.degree())]
        elif algorithm == "solve":
            try:
                poles = den.roots(CIF)
            except NotImplementedError:
                poles = den.change_ring(QQbar).roots(CIF)
            factors = [(Poly([IR(iv.abs().lower()), -1]), mult)
                        for iv, mult in poles]
        else:
            raise ValueError("algorithm")
    num = ~abs(IC(den.leading_coefficient()))
    return num, Factorization(factors, unit=Poly(1))

######################################################################
# Bounds on rational functions of n
######################################################################

class SeqBound(object):
    # XXX: try to simplify *SeqBound...
    pass

class RatSeqBound(SeqBound):
    r"""
    A piecewise-constant-piecewise-rational nonincreasing sequence.

    This is intended to represent a sequence b(n) such that |f(k)| <= b(n) for
    all k >= n, for a certain (rational) sequence f(n) = num(n)/den(n). The
    bound is defined by

    - the two polynomials num, den, with deg(num) <= deg(den),

    - and a list of pairs (n[i], v[i]) with n[i-1] <= n[i], n[-1] = ∞,
      v[i-1] >= v[i], and such that

          |f(k)| <= max(|f(n)|, v[i]) for n[i-1] < n <= k <= n[i].
    """

    def __init__(self, num, den, stairs):
        self.num = num
        self.den = den
        self.stairs = stairs

    def __repr__(self):
        fmt = "max(\n  |({num})/({den})|,\n{stairs}\n)"
        n = self.num.variable_name()
        stairsstr = ',\n'.join("  {}\tfor  {} <= {}".format(val, n, edge)
                                for edge, val in self.stairs)
        r = fmt.format(num=self.num, den=self.den, stairs=stairsstr)
        return r

    def asympt_repr(self):
        deg = self.num.degree() - self.den.degree()
        steplim = self.stairs[-1][1]
        ratlim = IC(self.num().leading_coefficient()
                    /self.den.leading_coefficient())
        if deg == 0:
            return "~{}".format(max(abs(steplim), abs(ratlim)).mid())
        else:
            return "~max({}, {}*n^{})".format(steplim.mid(), ratlim.mid(), deg)

    def lim(self):
        deg = self.num.degree() - self.den.degree()
        steplim = abs(self.stairs[-1][1])
        if deg < 0:
            return steplim
        elif deg == 0:
            ratlim = IC(self.num().leading_coefficient()
                        /self.den.leading_coefficient())
            return max(abs(ratlim), steplim)
        else:
            assert False

    def stairs_step(self, n):
        for (edge, val) in self.stairs:
            if n <= edge:
                return val
        assert False

    def __call__(self, n):
        step = self.stairs_step(n)
        if step.upper() == infinity: # TODO: arb is_finite?
            return step
        else:
            # TODO: avoid recomputing cst every time once it becomes <= next + ε?
            val = (IC(self.num(n))/IC(self.den(n))).above_abs()
            return step.max(val)

    def plot(self, n=30):
        from sage.plot.plot import list_plot
        rat = self.num/self.den
        p1 = list_plot([RR(abs(rat(k))) if self.den(k) else RR('inf')
                        for k in range(n)],
                marker='o', plotjoined=True)
        p2 = list_plot([self.stairs_step(k).upper() for k in range(n)],
                plotjoined=True, linestyle=':', color='black')
        p3 = list_plot([self(k).upper() for k in range(n)],
                marker='o', plotjoined=True, color='blue')
        return p1 + p2 + p3

    def _test(self, n=100):
        for k in range(n):
            if self(k) < IR(self.num(k)/self.den(k)).abs():
                raise AssertionError

class SumSeqBound(SeqBound):
    r"""
    A sum of :class:`SeqBound`s.
    """

    def __init__(self, terms):
        self.terms = terms

    def __repr__(self):
        return '(' + ' + '.join(repr(term) for term in self.terms) + ')'

    def asympt_repr(self):
        deg = max(t.num.degree() - t.den.degree() for t in self.terms)
        if deg == 0:
            return "~{}".format(sum(t.lim() for t in self.terms).mid())
        else:
            return "~max({}, {}·n^{})".format(
                sum(t.lim() for t in self.terms).mid(),
                sum(t.num().leading_coefficient()/t.den.leading_coefficient()
                    for t in self.terms
                    if t.num.degree() - t.den.degree() == deg).mid(),
                deg)

    def __call__(self, n):
        return sum(term(n) for term in self.terms)

def bound_real_roots(pol):
    if pol.is_zero(): # XXX: may not play well with intervals
        return -infinity
    bound = real_roots.cl_maximum_root(pol.change_ring(RIF).list())
    bound = RIF._upper_field()(bound) # work around weakness of cl_maximum_root
    bound = bound.nextabove().ceil()
    return bound

def nonneg_roots(pol):
    r"""
    Return a list of intervals with rational endpoints (represented by pairs)
    containing all nonnegative roots of pol.
    """
    bounds = (QQ.zero(), bound_real_roots(pol))
    if pol.base_ring() is AA:
        if bounds[1] < QQ(20):
            # don't bother with computing the roots (too slow)
            return [bounds]
        bounds = None
    roots = real_roots.real_roots(pol, bounds=bounds)
    if roots and roots[-1][0][1]:
        diam = ~roots[-1][0][1]
        while any(rt >= QQ.zero() and rt - lt > QQ(10)
                  for ((lt, rt), _) in roots):
            # max_diameter is a relative diameter --> pb for large roots
            logger.debug("refining (diam=%s)...", diam)
            roots = real_roots.real_roots(pol, bounds=bounds, max_diameter=diam)
            diam >>= 1
    return [root for (root, mult) in roots if root[1] >= QQ.zero()]

upper_inf = RIF(infinity).upper()

# TODO: computation of roots should be shared between calls corresponding to
# different shift equivalence classes...
def bound_ratio_large_n(num, den, exceptions={}, min_drop=IR(1.1), stats=None):
    """
    Given two polynomials num and den with complex coefficients, return a
    function b: ℕ → [0, ∞] such that

        b(n) >= min(|num(k)/den(k)|, exceptions[k])

    for all n, k ∈ ℕ with 0 <= n <= k. (The idea is that exceptions[k] will
    typically be specified only when den(k) = 0, but may be omitted even in
    this case if one is willing to accept that b(n) = ∞ up to the largest
    integer root of den.)

    EXAMPLES::

        sage: from ore_algebra.analytic.bounds import bound_ratio_large_n
        sage: Pols.<n> = QQ[]

        sage: num = (n^3-2/3*n^2-10*n+2)*(n^3-30*n+8)*(n^3-10/9*n+1/54)
        sage: den = (n^3-5/2*n^2+n+2/5)*(n^3-1/2*n^2+3*n+2)*(n^3-81/5*n-14/15)
        sage: bnd1 = bound_ratio_large_n(num, den); bnd1
        max(
          |(n^9 + ([-0.66...])*n^8 + ([-41.1...])*n^7 + ...)/(n^9 - ...)|,
          [22.77116...]     for  n <= 2,
          [12.72438...]     for  n <= 4,
          [1.052785...]     for  n <= +Infinity
        )
        sage: bnd1.plot(12)
        Graphics object consisting of 3 graphics primitives

        sage: num = (n^2-3/2*n-6/7)*(n^2+1/8*n+1/12)*(n^3-1/44*n^2+1/11*n+9/22)
        sage: den = (n^3-1/2*n^2+1/13)*(n^3-28*n+35)*(n^3-31/5)
        sage: bnd2 = bound_ratio_large_n(num, den); bnd2
        max(
          ...
          [0.231763...]   for  n <= 4,
          [0.200420...]   for  n <= 5,
          0               for  n <= +Infinity
        )
        sage: bnd2.plot()
        Graphics object consisting of 3 graphics primitives

    TESTS::

        sage: bnd1._test()
        sage: bnd2._test()

        sage: bound_ratio_large_n(n, Pols(1))
        Traceback (most recent call last):
        ...
        ValueError: expected deg(num) <= deg(den)

        sage: bound_ratio_large_n(Pols(1), Pols(3))
        max(
          |([0.333...])/(1.000...)|,
          [0.333...]     for  n <= +Infinity
        )

        sage: i = QuadraticField(-1).gen()
        sage: bound_ratio_large_n(n, n + i)
        max(
          |(n)/(n + I)|,
          1.000000000000000     for  n <= +Infinity
        )
    """
    rat = num/den
    num, den = rat.numerator(), rat.denominator()
    logger.debug("bounding rational function ~%s/%s",
            num.change_ring(CBF), den.change_ring(CBF))

    if num.is_zero():
        return RatSeqBound(num, den, [(infinity, IR.zero())])
    if num.degree() > den.degree():
        raise ValueError("expected deg(num) <= deg(den)")

    Scalars = num.base_ring()
    if (Scalars is QQ or isinstance(Scalars, NumberField_quadratic)
                         and Scalars.gen()**2 == -1):
        RealScalars = QQ
    else:
        num, den = num.change_ring(QQbar), den.change_ring(QQbar)
        RealScalars = AA
    def sqn(pol):
        re, im = (pol.map_coefficients(which, new_base_ring=RealScalars)
                  for which in (lambda coef: coef.real(),
                                lambda coef: coef.imag()))
        return re**2 + im**2
    sqn_num, sqn_den = sqn(num), sqn(den)
    crit = sqn_num.diff()*sqn_den - sqn_den.diff()*sqn_num

    logger.debug("computing roots, degrees=(%s, %s)...",
            sqn_den.degree(), crit.degree())
    if stats: stats.time_roots.tic()
    roots = nonneg_roots(sqn_den) # we want real coefficients
    roots.extend(nonneg_roots(crit))
    if stats: stats.time_roots.toc()

    logger.debug("found %s roots, now building staircase...", len(roots))
    if stats: stats.time_staircases.tic()
    num, den = num.change_ring(IC), den.change_ring(IC)
    thrs = set(n for iv in roots for n in xrange(iv[0].floor(), iv[1].ceil()))
    thrs = list(thrs)
    thrs.sort(reverse=True)
    thr_vals = [(n, (exceptions[n] if n in exceptions
                     else num(n).abs()/den(n).abs()))
                for n in thrs]
    lim = (num[den.degree()]/den.leading_coefficient()).abs()
    stairs = [(infinity, lim)]
    for (n, val) in thr_vals:
        if val.upper() > (min_drop*stairs[-1][1]).upper():
            stairs.append((n, val))
        elif val.upper() > stairs[-1][1].upper():
            # avoid unnecessarily large staircases
            stairs[-1] = (stairs[-1][0], val)
        if val.upper() == upper_inf:
            break
    stairs.reverse()
    logger.log(logging.INFO-2, "done building staircase, size=%s", len(stairs))
    if stats: stats.time_staircases.toc()

    return RatSeqBound(num, den, stairs)

def bound_ratio_derivatives(num, den, nat_poles, stats=None):
    r"""
    Compute a bound on sum(n·|f^(t)(n)/t!|, t=0..derivatives-1) similar to the
    one returned by bound_ratio_large_n. XXX: update descr

    The variable of num, den represents an integer index shift.
    """
    Pol = num.parent()

    max_mult = max(mult for _, mult in nat_poles) if nat_poles else 0
    derivatives = 1 + sum(mult for _, mult in nat_poles)
    Jets = utilities.jets(IC, 'X', derivatives + max_mult)
    # Sage won't return Laurent series here. Even this version currently doesn't
    # work with the mainline (coercions involving quotients polynomial rings for
    # which factor() is not implemented are broken).
    ex_series = {}
    for n, mult in nat_poles:
        pert = Jets([n, 1]) # n + X
        # den has a root of order mult at n, so den(pert) = O(X^mult), but the
        # computed value might include terms of degree < mult with interval
        # coefficients containing zero
        ex_series[n] = n*num(pert)/Jets(den(pert).lift() >> mult)

    rat = num/den
    denpow = den
    terms = []
    for t in range(derivatives):
        ex = {n: ser[t].abs() for n, ser in ex_series.iteritems()}
        bnd = bound_ratio_large_n(num << 1, denpow, exceptions=ex, stats=stats)
        terms.append(bnd)
        rat = rat.derivative()/(t + 1)
        denpow *= den
        num = Pol(rat*denpow)
    if len(terms) == 1:
        return terms[0]
    else:
        return SumSeqBound(terms)

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
      (given by SeqBound objects), with val(num[n]) >= deg(pol[n]),

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

        sage: maj = DiffOpBound((x^2 + 1)*Dx^2 + 2*x*Dx)
        sage: print(maj.__repr__(asympt=False))
        1/((-x + [0.994...])^2)*exp(int(POL+1.000...*NUM/(-x + [0.994...])^2))
        where
        POL=0,
        NUM=max(
          |(0)/(1)|,
          0     for  n <= +Infinity
        )*z^0 + max(
          |(-2.000...*n - 2.000...)/(n - 1.000...)|,
          [+/- inf]     for  n <= 1,
          2.000...     for  n <= +Infinity
        )*z^1

    A majorant series extracted from that sequence::

        sage: maj(3)
        ((-x + [0.994...])^-2)*exp(int(4.000...])^2)))
        sage: print(DiffOpBound(Dx - 1).__repr__(asympt=False))
        1/(1.000...)*exp(int(POL+1.000...*NUM/1.000...))
        where
        POL=0,
        NUM=max(
        |(-1.000...)/(1.000...)|,
        1.000...     for  n <= +Infinity
        )*z^0

    An example with an effort parameter::

        sage: dop = (x+1)*(x^2+1)*Dx^3-(x-1)*(x^2-3)*Dx^2-2*(x^2+2*x-1)*Dx
        sage: DiffOpBound(dop, pol_part_len=3)
        1/((-x + [0.9965035284306323 +/- 2.07e-17])^3)*exp(int(POL+1.000...*NUM/(-x + [0.9965035284306323 +/- 2.07e-17])^3)) where
        POL=~6.00000000000000*z^0 + ~3.00000000000000*z^1 + ~5.00000000000000*z^2,
        NUM=~7.00000000000000*z^3 + ~2.00000000000000*z^4 + ~5.00000000000000*z^5

    TESTS::

        sage: QQi.<i> = QuadraticField(-1)
        sage: for dop in [
        ....:     # orders <= 1 are not supported
        ....:     Dx, Dx - 1, i*Dx, Dx + i, Dx^2,
        ....:     (x^2 + 1)*Dx^2 + 2*x*Dx,
        ....:     Dx^2 - x*Dx
        ....: ]:
        ....:     DiffOpBound(dop)._test()

        sage: from ore_algebra.analytic.bounds import _test_diffop_bound
        sage: _test_diffop_bound()
    """

    def __init__(self, dop, leftmost=ZZ.zero(), special_shifts=[],
            pol_part_len=0, bound_inverse="simple"):
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

        lc = dop_T.leading_coefficient()
        if lc.is_term() and not lc.is_constant():
            raise ValueError("irregular singular operator", dop)

        self.leftmost = leftmost
        self.special_shifts = special_shifts

        self.bound_inverse = bound_inverse
        self.pol_part_len = pol_part_len
        self._effort = 0
        self._refine_interval = 2
        self._maybe_refine_called = 0

        self._update_den_bound()
        self._update_num_bound()

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

    def _update_den_bound(self):
        lc = self.dop.leading_coefficient()
        self.cst, self.maj_den = bound_inverse_poly(lc,
                algorithm=self.bound_inverse)

    def _update_num_bound(self):

        pol_part_len = self.pol_part_len

        Pols_z, z = self.dop.base_ring().objgen()
        Trunc = Pols_z.quo(z**(pol_part_len+1))
        lc = self.dop.leading_coefficient()
        inv = ~Trunc(lc)
        MPol, (z, n) = Pols_z.extend_variables('n').objgens()
        # Including rcoeffs[-1] here actually is redundant, as, by construction,
        # the only term in first to involve n^ordeq will be 1·n^ordeq·z^0.
        first = sum(n**j*(Trunc(pol)*inv).lift()
                    for j, pol in enumerate(self._rcoeffs))
        first_nz = first.polynomial(z)
        first_zn = first.polynomial(n)
        logger.log(logging.DEBUG - 1, "first: %s", first_nz)
        assert first_nz[0] == self._dop_D.indicial_polynomial(z, n).monic()
        assert all(pol.degree() < self.dop.order() for pol in first_nz[1:])

        self.stats.time_decomp_op.tic()
        T = self.dop.parent().gen()
        pol_part = sum(T**j*pol for j, pol in enumerate(first_zn)) # slow
        logger.debug("pol_part: %s", pol_part)
        rem_num = self.dop - pol_part*lc # in theory, slow for large pol_part_len
        logger.log(logging.DEBUG - 1, "rem_num: %s", rem_num)
        it = enumerate(_dop_rcoeffs_of_T(rem_num))
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
        # majseq_pol_part, it gets shifted as well. The "<< 1" in the next few
        # lines have nothing to do with that, they are multiplications by *n*.
        self.majseq_pol_part = [
                bound_ratio_derivatives(first_nz[i](alg_idx), self.ind,
                                        self.special_shifts, stats=self.stats)
                for i in xrange(1, pol_part_len + 1)]
        self.majseq_num = [
                bound_ratio_derivatives(pol(alg_idx), self.ind,
                                        self.special_shifts, stats=self.stats)
                for pol in rem_num_nz]
        assert len(self.majseq_pol_part) == pol_part_len

    def refine(self):
        # XXX: make it possible to increase the precision of IR, IC
        self._effort += 1
        logger.info("refining majorant (effort = %s)...", self._effort)
        self.stats.time_total.tic()
        if self.bound_inverse == 'simple':
            self.bound_inverse = 'solve'
            self._update_den_bound()
        else:
            self.pol_part_len = max(2, 2*self.pol_part_len)
            self._update_num_bound()
        self.stats.time_total.toc()
        logger.info("...done, cumulative time: %s", self.stats)

    def maybe_refine(self):
        self._maybe_refine_called += 1
        if self._maybe_refine_called >= self._refine_interval:
            self.refine()
            self._maybe_refine_called = 0
            self._refine_interval *= 2

    def reset_refinment_counter(self):
        self._maybe_refine_called = 0

    def __call__(self, n):
        r"""
        Return a term of the majorant sequence.
        """
        maj_pol_part = self.Poly([fun(n) for fun in self.majseq_pol_part])
        maj_num = (self.Poly([fun(n) for fun in self.majseq_num])
                << len(self.majseq_pol_part))
        rat_maj = RationalMajorant(self.cst*maj_num, self.maj_den, maj_pol_part)
        maj = HyperexpMajorant(integrand=rat_maj, rat=~self.maj_den)
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
        maj = self(n)*(majeqrhs >> 1).integral()
        logger.debug("maj(%s) = %s", n, self(n))
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
        logger.debug("col_bound = %s", col_bound)
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
            # we know a priori that val(resid) >= n mathematically, but interval
            # computations may give inexact zeros for some of the coefficients
            assert all(c.contains_zero() for c in resid[:n])
            resid = resid[n:]
            maj = self.tail_majorant(n, self.maj_eq_rhs([resid]))
            logger.info("%s << %s", Series(ref[n:], n+30), maj.series(n+30))
            maj._test(ref[n:])

# Perhaps better: work with a "true" Ore algebra K[θ][z]. Use Euclidean
# division to compute the truncation. Extracting the Qj(θ) would then be easy,
# and I may no longer need the coefficients of θ "on the right".

def _dop_rcoeffs_of_T(dop):
    """
    Compute the coefficients of dop as an operator in θ but with θ on the left.
    """
    Pols_z = dop.base_ring()
    Pols_n, n = Pols_z.change_var('n').objgen()
    Rops = OreAlgebra(Pols_n, 'Sn')
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
        self.time_roots = utilities.Clock("computing roots")
        self.time_staircases = utilities.Clock("building staircases")
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
        sage: _test_diffop_bound() # not tested
        INFO:ore_algebra.analytic.bounds:testing operator: (-i + 2)*Dx + i - 1
        ...
    """
    from sage.rings.number_field.number_field import QuadraticField
    from ore_algebra import OreAlgebra

    QQi = QuadraticField(-1, 'i')
    Pols, x = PolynomialRing(QQi, 'x').objgen()
    Dops, Dx = OreAlgebra(Pols, 'Dx').objgen()

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

def residual(bwrec, n, last, z):
    r"""
    Compute the polynomial residual, up to sign, obtained by a applying a diff
    op P to a partial sum of a power series solution y of P·y=0.

    INPUT:

    - ``bwrec`` -- list [b[0], ..., b[s]] of coefficients of the recurrence
      operator associated to P (by the direct substitution x |--> S⁻¹, θ |--> n;
      no additional multiplication by x^k is allowed!), written in the form
      b[0](n) + b[1](n) S⁻¹ + ···

    - ``n`` -- truncation order

    - ``last`` -- the last s+1 coefficients u[n-1], u[n-2], ... of the
      truncated series, in that order

    - ``z`` -- variable name for the result

    EXAMPLES::

        sage: from ore_algebra import OreAlgebra
        sage: from ore_algebra.analytic.bounds import *
        sage: Pol_t.<t> = QQ[]; Pol_n.<n> = QQ[]
        sage: Dop.<Dt> = OreAlgebra(Pol_t)

        sage: trunc = t._exp_series(5); trunc
        1/24*t^4 + 1/6*t^3 + 1/2*t^2 + t + 1
        sage: residual([n, Pol_n(1)], 5, [trunc[4]], t)
        ([0.0416666666666667 +/- 4.26e-17])*t^5
        sage: (Dt - 1).to_T('Tt')(trunc).change_ring(CBF)
        ([-0.0416666666666667 +/- 4.26e-17])*t^5

    Note that using Dt - 1 instead of θt - t makes a difference in the result,
    since it amounts to a division by t::

        sage: (Dt - 1)(trunc).change_ring(CBF)
        ([-0.0416666666666667 +/- 4.26e-17])*t^4

    ::

        sage: trunc = t._sin_series(5) + t._cos_series(5)
        sage: residual([n*(n-1), Pol_n(0), Pol_n(1)], 5, [trunc[4], trunc[3]], t)
        ([0.041666...])*t^6 + ([-0.16666...])*t^5
        sage: (Dt^2 + 1).to_T('Tt')(trunc).change_ring(CBF)
        ([0.041666...])*t^6 + ([-0.16666...])*t^5
    """
    # NOTE: later on I may want to compute the residuals directly in each
    # implementation of summation, to avoid recomputing known quantities (as
    # this function currently does)
    ordrec = len(bwrec) - 1
    rescoef = [
        sum(IC(bwrec[i+k+1](n+i))*IC(last[k])
            for k in xrange(ordrec-i))
        for i in xrange(ordrec)]
    IvPols = PolynomialRing(IC, z, sparse=True)
    return IvPols(rescoef) << n

# This roughly corresponds to residual() followed by maj.maj_eq_rhs() in the
# ordinary case. TODO: more consistent interface.
def maj_eq_rhs_with_logs(bwrec, n, last, z, logs, RecJets):
    ordrec = len(bwrec) - 1
    # Compute the coefficients of the residual:
    # residual = z^(lambda + n)·(sum(rescoef[i][j]·z^i·log^j(z)/j!)
    rescoef = [[None]*logs for _ in xrange(ordrec)]
    for i in xrange(ordrec):
        for j in xrange(logs):
            idx_pert = RecJets([n + i, 1])
            bwrec_i = [b(idx_pert) for b in bwrec]
            # significant overestimation here (apparently not too problematic)
            rescoef[i][j] = sum(
                    IC(bwrec_i[i+k+1][p])*IC(last[k][j+p])
                    for k in xrange(ordrec - i)
                    for p in xrange(logs - j))
    # For lack of a convenient data structure to return these coefficients,
    # compute a “majorant” polynomial right away. Here for simplicity we only
    # handle the generic case.
    idx_pert = RecJets([n, 1])
    invlc = ~(bwrec[0](idx_pert)) # sum(1/t!·(1/Q0)^(t)(λ + n)·Sk^t)
    invlcmaj = sum(IC(t).abs() for t in invlc)
    polcoef = [(n + i)*invlcmaj*max(t.abs() for t in rescoef[i])
               for i in xrange(ordrec)]
    IvPols = PolynomialRing(IR, z, sparse=True)
    return IvPols(polcoef) << n





