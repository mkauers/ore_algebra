# -*- coding: utf-8 - vim: tw=80
"""
Error bounds
"""

# TODO:
# - this module uses at least three different object types for things that are
# essentially rational fractions (QuotientRingElements, Factorizations, and
# Rational Majorants) --> simplify?

import logging

import sage.rings.complex_interval_field
import sage.rings.polynomial.real_roots as real_roots

from sage.misc.misc import cputime, verbose
from sage.misc.misc_c import prod
from sage.rings.infinity import infinity
from sage.rings.integer import Integer
from sage.rings.polynomial.polynomial_element import Polynomial
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.real_mpfi import RIF
from sage.rings.real_mpfr import RR
from sage.structure.factorization import Factorization
from sage.structure.sage_object import SageObject
from sage.symbolic.ring import SR

from ore_algebra.ore_algebra import OreAlgebra

from . import utilities

logger = logging.getLogger(__name__)

IR = RIF # TBI
IC = sage.rings.complex_interval_field.ComplexIntervalField()

class BoundPrecisionError(Exception):
    pass

######################################################################
# Majorant series
######################################################################

class MajorantSeries(object):

    def __init__(self, cvrad=IR.zero()):
        "A formal power series with nonnegative coefficients"
        self.cvrad = IR(cvrad)
    def __call__(self, z):
        raise NotImplementedError
    def _repr_(self):
        return repr(self(SR.var('z'))) # TBI: var name
    def bound(self, rad, **kwds):
        if not rad <= self.cvrad: # intervals!
            return infinity
        else:
            return self._bound_doit(rad, **kwds)
    def _bound_doit(self, rad):
        # pourrait en général renvoyer quelque chose de plus simple à calculer
        # que self(rad), ou être implémenté alors que __call__ ne l'est pas
        return self(rad)
    def bound_tail(self, order, rad):
        # TODO: implém générique (quid des dérivées ?)
        raise NotImplementedError

def _pole_free_rad(fac):
    if isinstance(fac, Factorization):
        den = (pol for (pol, mult) in fac if mult < 0)
        if all(pol.is_monic() and pol.degree() == 1 for pol in den):
            rad = IR(infinity).min(*(pol[0].abs() for pol in den))
            return IR(rad.lower())
    raise NotImplementedError  # pb dérivées???

class RationalMajorant(MajorantSeries):

    def __init__(self, num, den, pol=0):
        """
        A rational power series with nonnegative coefficients, represented in
        the form pol + num/den
        """
        if isinstance(num, Polynomial) and isinstance(den, Factorization):
            Poly = num.parent().change_ring(IR)
            if not den.unit().is_one():
                raise ValueError("expected a monic denominator")
            assert den.universe() is Poly or den.value() == 1
            super(self.__class__, self).__init__(cvrad=_pole_free_rad(~den))
            self.num = Poly(num)
            self.pol = Poly(pol)
            self.den = den
            self.var = Poly.gen()
        else:
            raise TypeError

    def __call__(self, z=None):
        if z is None:
            z = self.var
        # better than den.value()(z) when z is inexact
        den = prod(lin(z)**mult for (lin, mult) in self.den)
        return self.pol(z) + self.num(z)/den

    def _repr_(self):
        res = ""
        if self.pol:
            Poly = self.pol.parent()
            pol_as_series = Poly.completion(Poly.gen())(self.pol)
            res += repr(pol_as_series) + " + "
        res += self.num._coeff_repr()
        if self.den:
            res += "/(" + repr(self.den) + ")"
        return res

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
        if pol.parent() == self.num.parent():
            return RationalMajorant(self.num*pol, self.den, self.pol*pol)
        else:
            raise TypeError

class HyperexpMajorant(MajorantSeries):

    def __init__(self, integrand, rat):
        """
        A formal power series of the form rat1(z) + exp(int(rat2(ζ), ζ=0..z)),
        with nonnegative coefficients.
        """
        if isinstance(integrand, RationalMajorant):
            cvrad = integrand.cvrad.max(_pole_free_rad(rat))
            super(self.__class__, self).__init__(cvrad)
            self.integrand = integrand
            self.rat = rat
        else:
            raise TypeError

    def __call__(self, z):
        # TBI: variable names
        # XXX: l'évaluation sur une variable de SR a pour effet
        # indésirable de mettre le numérateur sous forme de Horner
        dummy = SR.var(z)
        integrand = self.integrand(dummy)
        return self.rat.value() * integrand.integral(dummy, hold=True).exp()

    def _bound_doit(self, rad, derivatives=1):
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
        return (rat_part*exp_part).sqrt() # XXX: sqrtpos?

    # derivatives? fusionner avec bound ?
    def bound_tail(self, order, rad):
        raise NotImplementedError

    def __mul__(self, pol):
        """"
        Multiplication by a polynomial.

        Note that this does not change the radius of convergence.
        """
        return HyperexpMajorant(self.integrand, self.rat*pol)

def _check_maj(fun, maj, prec=50):
    Series = PowerSeriesRing(IR, 'z', default_prec=prec)
    z = Series.gen()
    delta = list(maj(z) - Series([iv.abs() for iv in list(fun(z))]))
    # iv.upper as majcoef.lower() is not a bound in general, and
    # funcoef.upper() can be overestimated during the series expansion
    delta = [iv.upper() for iv in delta]
    if all(c >= 0. for c in delta):
        return delta
    else:
        raise ValueError(delta)

######################################################################
# Majorants for reciprocals of polynomials ("denominators")
######################################################################

def graeffe(pol):
    deg = pol.degree()
    Parent = pol.parent()
    pol_even = Parent([pol[2*i]   for i in xrange(deg/2+1)])
    pol_odd = Parent([pol[2*i+1] for i in xrange(deg/2+1)])
    graeffe_iterate = (-1)**deg * (pol_even**2 - (pol_odd**2).shift(1))
    return graeffe_iterate

def abs_min_nonzero_root(pol, tol=RR(1e-2), ensure_larger_than=0.):
    """
    abs_min_nonzero_root(1/10*z^3 + z^2 + 1/7)
    """
    if pol.is_zero():
        raise ValueError("expected a nonzero polynomial")
    pol >>= pol.valuation()
    deg = pol.degree()
    if deg == 0:
        return infinity
    lg_target_rad = IR(ensure_larger_than).log2()
    pol = pol.change_ring(IR.complex_field())
    pol = pol/pol[0]
    i = 0
    encl = IR(1, 2*deg).log2()
    lg_rad = IR(-infinity, infinity)
    while (lg_rad.lower() <= lg_target_rad
           or lg_rad.absolute_diameter() > tol/2): # log2(1+x) ≤ 2*x
        prev_lg_rad = lg_rad
        # The smallest root of the current pol is within 2^(-1-m) and
        # (2·deg)·2^(-1-m), cf. Davenport & Mignotte (1990), Grégoire (2012).
        m = IR(-infinity).max(*(pol[k].abs().log2()/k
                                for k in xrange(1, deg+1)))
        lg_rad = (-(1+m) + encl) >> i
        lg_rad = prev_lg_rad.intersection(lg_rad)
        if lg_rad.lower() == -infinity or cmp(lg_rad, prev_lg_rad) == 0:
            raise ArithmeticError
        verbose("i = {}\trad ∈ {}\tdiam={}"
                .format(i, lg_rad.exp2().str(style='brackets'),
                        lg_rad.absolute_diameter()),
            level=10)
        # detect gross input errors (this does not prevent all infinite loops)
        if lg_rad < lg_target_rad:
            raise ValueError("there is a root smaller than " + str(ensure_larger_than))
        pol = graeffe(pol)
        i += 1
    res = lg_rad.exp2()
    assert(res.absolute_diameter().exact_rational()
           /res.lower().exact_rational() < tol.exact_rational())
    return res

def bound_inverse_poly_simple(den):
    Poly = den.parent().change_ring(IR)
    if den.degree() <= 0:
        fac = Factorization([], unit=Poly(1))
    else:
        # thin interval containing the lower bound
        rad = IR(abs_min_nonzero_root(den).lower())
        fac = Factorization([(Poly([-rad, 1]), den.degree())])
    return ~abs(den.leading_coefficient()), fac

def bound_inverse_poly_solve(den):
    Poly = den.parent().change_ring(IR)
    if den.degree() <= 0:
        fac = Factorization([], unit=Poly(1))
    else:
        poles = den.roots(IC)
        # thin interval containing the lower bounds
        fac = Factorization([
            (Poly(-[IR(iv.abs().lower()), 1]), mult)
            for iv, mult in poles])
    return ~abs(den.leading_coefficient()), fac

bound_inverse_poly = bound_inverse_poly_simple

######################################################################
# Bounds on rational functions of n
######################################################################

class RatSeqBound(object):
    def __init__(self, num, den, stairs):
        self.num = num
        self.den = den
        self.stairs = stairs
    def __repr__(self):
        fmt = "Bound on |({num})/({den})|: {stairs}"
        r = fmt.format(num=self.num, den=self.den, stairs=self.stairs)
        return r
    def stairs_step(self, n):
        for (pt, val) in self.stairs:
            if n <= pt:
                return val
        assert False
    def __call__(self, n):
        step = self.stairs_step(n)
        if step.upper() == infinity: # TODO: arb is_finite?
            return step
        else:
            # TODO: avoid recomputing cst every time once it becomes <= next + ε?
            val = self.num(n).abs()/self.den(n).abs()
            return step.max(val)

def bound_real_roots(pol):
    if pol.is_zero(): # XXX: may not play well with intervals
        return -infinity
    bound = real_roots.cl_maximum_root(pol.change_ring(RIF).list())
    bound = RIF._upper_field()(bound) # work around weakness of cl_maximum_root
    bound = bound.nextabove().ceil()
    return bound

# TODO: share code with the main implementation (if I keep both versions)
def bound_ratio_large_n_nosolve(num, den):
    """
    Given two polynomials num and den, return a function a(n) such that

        0 < |num(k)| < a(n)·|den(k)|

    for all k >= n >= 0. Note that a may take infinite values.

    This version should work with polynomials with interval coefficients.
    """
    if num.degree() > den.degree():
        raise ValueError("expected deg(num) <= deg(den)")

    def sqn(pol):
        RealScalars = num.base_ring().base_ring()
        re, im = (pol.map_coefficients(which, new_base_ring=RealScalars)
                  for which in (lambda coef: coef.real(),
                                lambda coef: coef.imag()))
        return re**2 + im**2
    sqn_num, sqn_den = sqn(num), sqn(den)
    crit = sqn_num.diff()*sqn_den - sqn_den.diff()*sqn_num

    finite_from = max(2, bound_real_roots(sqn_den))
    monotonic_from = max(finite_from, bound_real_roots(crit))

    orig_den = den
    num = num.change_ring(IC)
    den = den.change_ring(IC)
    def bound_term(n): return num(n).abs()/den(n).abs()
    lim = (num[den.degree()]/den.leading_coefficient()).abs()

    # We would compute these later anyway (unless we are more clever here)
    last = 0
    nonincr_or_le_lim_from = None
    finite_from = 0
    verbose("monotonic from {}, starting extended search".format(monotonic_from), level=9)
    tic = cputime()
    for n in xrange(monotonic_from, 0, -1):
        val = bound_term(n)
        if (nonincr_or_le_lim_from is None
                and not (val <= lim)
                and not (val >= last)): # interval comparisons
            nonincr_or_le_lim_from = n + 1
        if not orig_den(n):
            finite_from = n + 1
            break
        last = val
    if nonincr_or_le_lim_from is None:
        nonincr_or_le_lim_from = finite_from # TBI?

    ini_range = xrange(finite_from, nonincr_or_le_lim_from+1) # +1 for clarity when empty
    ini_bound = lim.max(*(bound_term(n) for n in ini_range))

    verbose("finite from {}, ini_bound={}, ↘/≤lim from {}, lim={}"
            .format(finite_from, ini_bound, nonincr_or_le_lim_from, lim),
            level=9, t=tic)

    stairs = [(finite_from, infinity), (nonincr_or_le_lim_from, ini_bound),
              (infinity, lim)]
    return RatSeqBound(num, den, stairs)

def nonneg_roots(pol):
    bound = bound_real_roots(pol)
    roots = real_roots.real_roots(pol, bounds=(QQ(0), bound))
    if roots and roots[-1][0][1]:
        diam = ~roots[-1][0][1]
        while any(rt - lt > QQ(10) for ((lt, rt), _) in roots):
            # max_diameter is a relative diameter --> pb for large roots
            verbose("largest root diameter = {}, refining"
                        .format(roots[-1][0][1] - roots[-1][0][0].n(10)),
                    level=10)
            roots = real_roots.real_roots(pol, bounds=(QQ(0), bound),
                                          max_diameter=diam)
            diam >>= 1
    return roots

upper_inf = IR(infinity).upper()

def bound_ratio_large_n_solve(num, den, min_drop=IR(1.1), stats=None):
    """
    Given two polynomials num and den, return a function a(n) such that

        0 < |num(k)| < a(n)·|den(k)|

    for all k >= n >= 0. Note that a may take infinite values.
    """
    if num.is_zero():
        return RatSeqBound(num, den, [(infinity, IR.zero())])
    if num.degree() > den.degree():
        raise ValueError("expected deg(num) <= deg(den)")

    def sqn(pol):
        RealScalars = num.base_ring().base_ring()
        re, im = (pol.map_coefficients(which, new_base_ring=RealScalars)
                  for which in (lambda coef: coef.real(),
                                lambda coef: coef.imag()))
        return re**2 + im**2
    sqn_num, sqn_den = sqn(num), sqn(den)
    crit = sqn_num.diff()*sqn_den - sqn_den.diff()*sqn_num

    if stats: stats.time_roots.tic()
    roots = nonneg_roots(sqn_den) # we want real coefficients
    roots.extend(nonneg_roots(crit))
    roots = [descr[0] for descr in roots] # throw away mults
    if stats: stats.time_roots.toc()

    if stats: stats.time_staircases.tic()
    num, den = num.change_ring(IC), den.change_ring(IC)
    thrs = set(n for iv in roots for n in xrange(iv[0].floor(), iv[1].ceil()))
    thrs = list(thrs)
    thrs.sort(reverse=True)
    thr_vals = [(n, num(n).abs()/den(n).abs()) for n in thrs]
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
    verbose("done building staircase, size = {}".format(len(stairs)), level=10)
    if stats: stats.time_staircases.toc()

    return RatSeqBound(num, den, stairs)

bound_ratio_large_n = bound_ratio_large_n_solve

################################################################################
# Bounds for differential equations
################################################################################

def bound_polynomials(pols):
    # Returns a majorant, _not_ a polynomial with interval coefficients
    # enclosing the absolute values or anything like that!
    PolyIC = pols[0].parent().change_ring(IC)
    length = 1 + max(pol.degree() for pol in pols)
    pols = [PolyIC(pol) for pol in pols] # TBI
    order = Integer(len(pols))
    PolyIR = PolyIC.change_ring(IR)
    maj = PolyIR([
        IR.zero().max(*(
            abs(pols[k][n])/(order-1).binomial(k)
            for k in xrange(order)))
        for n in xrange(length) ])
    maj = maj.map_coefficients(lambda iv: IR(iv.magnitude()))
    return maj

class DiffOpBound(object):
    def __init__(self, dop, cst, majseq_pol_part, majseq_num, maj_den):
        self.dop = dop
        self.Poly = dop.base_ring().change_ring(IR)
        self.cst = cst
        self.majseq_pol_part = majseq_pol_part
        self.majseq_num = majseq_num
        self.maj_den = maj_den
    def __repr__(self):
        return "Cst: {}\nPolPart: {}\nNum: {}\nDen: {}".format(
                self.cst, self.majseq_pol_part, self.majseq_num,
                self.maj_den)
    def __call__(self, n):
        maj_pol_part = self.Poly([fun(n) for fun in self.majseq_pol_part])
        maj_num = (self.Poly([fun(n) for fun in self.majseq_num])
                   >> len(self.majseq_pol_part))
        rat_maj = RationalMajorant(self.cst*maj_num, self.maj_den, maj_pol_part)
        maj = HyperexpMajorant(rat_maj, ~self.maj_den)
        return maj
    def matrix_sol_tail_bound(self, n, rad, residuals, ord=None):
        if ord is None: ord=self.dop.order()
        abs_residual = bound_polynomials(residuals)
        maj = self(n)*abs_residual.integral()
        # Since (y[n:])' << maj => (y')[n:] << maj, this bound is valid for the
        # tails of a column of the form [y, y', y''/2, y'''/6, ...] or
        # [y, θy, θ²y/2, θ³y/6, ...].
        col_bound = maj.bound(rad, derivatives=ord)
        # logger.debug("lc(abs_res) = %s, maj(n).bound() = %s, col_bound = %s",
        #         abs_residual.leading_coefficient(), self(n).bound(rad), col_bound)
        return IR(ord).sqrt()*col_bound

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
    assert dop.is_zero() or dop.leading_coefficient() == res[-1]
    return res

class BoundDiffopStats(utilities.Stats):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.time_roots = utilities.Clock("computing roots")
        self.time_staircases = utilities.Clock("building staircases")
        self.time_decomp_op = utilities.Clock("decomposing op")

def bound_diffop(dop, pol_part_len=0):
    stats = BoundDiffopStats()
    verbose("bounding diff op...", level=5)
    _, Pols_z, _, dop = dop._normalize_base_ring()
    z = Pols_z.gen()
    lc = dop.leading_coefficient()
    if lc.is_term() and not lc.is_constant():
        raise ValueError("irregular singular operator")
    rcoeffs = _dop_rcoeffs_of_T(dop)
    Trunc = Pols_z.quo(z**(pol_part_len+1))
    inv = ~Trunc(lc)
    MPol, (z, n) = Pols_z.extend_variables('n').objgens()
    # Including rcoeffs[-1] here is actually redundant, as by construction the
    # only term in first to involve n^ordeq will be 1·n^ordeq·z^0. But I find
    # the code easier to understand this way.
    first = sum(n**j*(Trunc(pol)*inv).lift()
                for j, pol in enumerate(rcoeffs))
    first_nz = first.polynomial(z)
    first_zn = first.polynomial(n)
    verbose("first: {}".format(first_nz), level=8)
    assert first_nz[0] == dop.indicial_polynomial(z, n).monic()
    assert all(pol.degree() < dop.order() for pol in first_nz[1:])

    stats.time_decomp_op.tic()
    dop_T = dop.to_T('T' + str(z)) # slow
    T = dop_T.parent().gen()
    pol_part = sum(T**j*pol for j, pol in enumerate(first_zn)) # slow
    verbose("pol_part: {}".format(pol_part), level=8)
    rem_num = dop_T - pol_part*lc # inefficient in theory for large pol_part_len
    verbose("rem_num: {}".format(rem_num), level=8)
    it = enumerate(_dop_rcoeffs_of_T(rem_num))
    rem_num_nz = MPol(sum(n**j*pol for j, pol in it)).polynomial(z)
    assert rem_num_nz.valuation() >= pol_part_len + 1
    rem_num_nz >>= pol_part_len + 1
    stats.time_decomp_op.toc()
    verbose("rem_num_nz: {}".format(rem_num_nz), level=8)

    ind = first_nz[0]
    cst, maj_den = bound_inverse_poly(lc)
    majseq_pol_part = [bound_ratio_large_n(pol << 1, ind, stats=stats)
                       for pol in first_nz[1:]]
    majseq_num = [bound_ratio_large_n(pol << 1, ind, stats=stats)
                  for pol in rem_num_nz]
    maj = DiffOpBound(dop, cst, majseq_pol_part, majseq_num, maj_den)
    verbose("...done, time: {}".format(stats), level=5)
    return maj

def residual(bwrec, n, last):
    r"""
    Compute the polynomial residual (up to sign?) obtained by a applying a diff
    op P to a partial sum of a power series solution y of P·y=0.

    INPUT:

    - ``bwrec`` -- list [b[0], ..., b[s]] of coefficients of the recurrence
      operator associated to P, written in the form b[0](n) + b[1](n) S⁻¹ + ···

    - ``n`` -- truncation order

    - ``last`` -- the last s+1 coefficients u[n-1], u[n-2], ... of the
      truncated series (in that order).
    """
    # NOTE: later on I may want to compute the residuals directly in each
    # implementation of summation, to avoid recomputing known quantities (as
    # this function currently does)
    ordrec = len(bwrec) - 1
    rescoef = [
        sum(IC(bwrec[i+k+1](n+i))*IC(last[k])
            for k in xrange(ordrec-i))
        for i in xrange(ordrec)]
    IvPols = PolynomialRing(IC, bwrec[0].numerator().variable_name(), sparse=True)
    return IvPols(rescoef) << n

######################################################################
# Absolute and relative errors
######################################################################

class PrecisionError(Exception):
    pass

class ErrorCriterion(object):
    pass

class AbsoluteError(ErrorCriterion):
    def __init__(self, eps):
        self.eps = IR(eps)
    def reached(self, err, abs_val=IR('nan')):
        if utilities.rad(abs_val) > self.eps:
            # no need to continue! (TBI...)
            #raise PrecisionError
            return True
        return err.abs() < self.eps
    def __repr__(self):
        return str(self.eps) + " (absolute)"

class RelativeError(ErrorCriterion):
    def __init__(self, eps, cutoff=None):
        self.eps = IR(eps)
        self.cutoff = eps if cutoff is None else IR(cutoff)
    def reached(self, err, abs_val):
        # NOTE: we could provide a slightly faster test when err is a
        # non-rigorous estimate (not a true tail bound)
        # XXX: raise PrecisionError if we can not conclude
        return (err.abs() <= self.eps*(abs_val - err)
                or abs_val + err < self.cutoff)
    def __repr__(self):
        return str(self.eps) + " (relative)"
