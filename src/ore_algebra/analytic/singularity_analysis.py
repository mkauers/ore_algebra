# coding: utf-8 - vim: tw=80
r"""
Bounds on sequences by singularity analysis

Main author: Ruiwen Dong <ruiwen.dong@polytechique.edu>

This module currently requires Sage development branch u/mmezzarobba/tmp/bterms
(= Sage 9.5.beta0 + #32229 + #32451 + a patch for combining BTerms with the same
growth); ``check_seq_bound`` and the tests depending on it may need additional
patches.

EXAMPLES::

    sage: from ore_algebra import (OreAlgebra, DFiniteFunctionRing,
    ....:         UnivariateDFiniteFunction)
    sage: from ore_algebra.analytic.singularity_analysis import (
    ....:         bound_coefficients, check_seq_bound)

    sage: Pols_z.<z> = PolynomialRing(QQ)
    sage: Diff.<Dz> = OreAlgebra(Pols_z)

Membrane example::

    sage: seqini = [72, 1932, 31248, 790101/2, 17208645/4, 338898609/8, 1551478257/4]
    sage: deq = (8388593*z^2*(3*z^4 - 164*z^3 + 370*z^2 - 164*z + 3)*(z + 1)^2*(z^2 - 6*z + 1)^2*(z - 1)^3*Dz^3
    ....: + 8388593*z*(z + 1)*(z^2 - 6*z + 1)*(66*z^8 - 3943*z^7 + 18981*z^6 - 16759*z^5 - 30383*z^4 + 47123*z^3 - 17577*z^2 + 971*z - 15)*(z - 1)^2*Dz^2
    ....: + 16777186*(z - 1)*(210*z^12 - 13761*z^11 + 101088*z^10 - 178437*z^9 - 248334*z^8 + 930590*z^7 - 446064*z^6 - 694834*z^5 + 794998*z^4 - 267421*z^3 + 24144*z^2 - 649*z + 6)*Dz
    ....: + 6341776308*z^12 - 427012938072*z^11 + 2435594423178*z^10 - 2400915979716*z^9 - 10724094731502*z^8 + 26272536406048*z^7 - 8496738740956*z^6 - 30570113263064*z^5 + 39394376229112*z^4 - 19173572139496*z^3 + 3825886272626*z^2 - 170758199108*z + 2701126946)

    sage: asy = bound_coefficients(deq, seqini, order=5, prec=200) # long time (5 s)
    doctest:...: FutureWarning: This class/method/function is marked as
    experimental. ...
    sage: asy # long time
    1.000...*5.828427124746190?^n*(([8.0719562915...] + [+/- ...]*I)*n^3*log(n)
    + ([1.3714048996...82527...] + [+/- ...]*I)*n^3
    + ([50.509130873...07157...] + [+/- ...]*I)*n^2*log(n)
    + ([29.698551451...84781...] + [+/- ...]*I)*n^2
    + ...
    + ([-0.283779713...91869...] + [+/- ...]*I)*n^(-1)*log(n)
    + ([35.493938347...65227...] + [+/- ...]*I)*n^(-1)
    + B([115882.7...]*n^(-2)*log(n)^2, n >= 50))

    sage: DFR = DFiniteFunctionRing(deq.parent())
    sage: ref = UnivariateDFiniteFunction(DFR, deq, seqini)
    sage: check_seq_bound(asy.expand(), ref, list(range(100)) + list(range(200, 230)) + [1000]) # long time

Algebraic example::

    sage: deq = (4*z^4 - 4*z^3 + z^2 - 2*z + 1)*Dz + (-4*z^3 + 4*z^2 - z - 1)
    sage: bound_coefficients(deq, [1], order=5) # long time (13 s)
    1.000...*2^n*([0.564189583547...]*n^(-1/2) + [-0.105785546915...]*n^(-3/2)
    + [-0.117906807499...]*n^(-5/2) + [-0.375001499318...]*n^(-7/2)
    + [-1.255580304110...]*n^(-9/2) + B([1304.15...]*n^(-11/2), n >= 50))

Diagonal example (Sage is not yet able to correctly combine and order the error
terms here)::

    sage: seqini = [1, -3, 9, -3, -279, 2997]
    sage: deq = (z^2*(81*z^2 + 14*z + 1)*Dz^3 + 3*z*(162*z^2 + 21*z + 1)*Dz^2
    ....:        + (21*z + 1)*(27*z + 1)*Dz + 3*(27*z + 1))

    sage: bound_coefficients(deq, seqini, order=2) # long time (3.5 s)
    1.000000000000000*9.00000000000000?^n*(([0.30660...] + [0.14643...]*I)*(e^(I*arg(-0.77777...? + 0.62853...?*I)))^n*n^(-3/2)
    + ([-0.26554...] + [-0.03529...]*I)*(e^(I*arg(-0.77777...? + 0.62853...?*I)))^n*n^(-5/2)
    + B([16.04...]*(e^(I*arg(-0.77777...? + 0.62853...?*I)))^n*n^(-7/2), n >= 50)
    + ([0.30660...] + [-0.14643...]*I)*(e^(I*arg(-0.77777...? - 0.62853...?*I)))^n*n^(-3/2)
    + ([-0.26554...] + [0.03529...]*I)*(e^(I*arg(-0.77777...? - 0.62853...?*I)))^n*n^(-5/2)
    + B([16.04...]*(e^(I*arg(-0.77777...? - 0.62853...?*I)))^n*n^(-7/2), n >= 50)
    + B([2.06...]*n^(-7/2), n >= 50))

Complex exponents example::

    sage: deq = (z-2)^2*Dz^2 + z*(z-2)*Dz + 1
    sage: seqini = [1, 2, -1/8]
    sage: asy = bound_coefficients(deq, seqini, order=3) # long time (2 s)
    sage: asy # long time
    1.000000000000000*(1/2)^n*(([1.124337...] + [0.462219...]*I)*n^(-0.500000...? + 0.866025...?*I)
    + ([1.124337...] + [-0.462219...]*I)*n^(-0.500000...? - 0.866025...?*I)
    + ([-0.400293...] + [0.973704...]*I)*n^(-1.500000...? + 0.866025...?*I)
    + ([-0.400293...] + [-0.973704...]*I)*n^(-1.500000...? - 0.866025...?*I)
    + ([0.451623...] + [-0.356367...]*I)*n^(-2.500000...? + 0.866025...?*I)
    + ([0.451623...] + [0.356367...]*I)*n^(-2.500000...? - 0.866025...?*I)
    + B([2761.73...]*n^(-7/2), n >= 50))

    sage: ref = UnivariateDFiniteFunction(DFR, deq, seqini)
    sage: #check_seq_bound(asy.expand(), ref, range(1000)) # buggy
    sage: # Temporary workaround
    sage: from ore_algebra.analytic.singularity_analysis import contribution_all_singularity, eval_bound
    sage: b = contribution_all_singularity(seqini, deq, order=3) # long time (1.9 s)
    sage: all(eval_bound(b[1], j).contains_exact(ref[j]) for j in range(b[0], 150)) # long time
    True


TESTS:

Incorrect output::

    sage: bound_coefficients(((z-1)*Dz+1)^2, [0])
    Traceback (most recent call last):
    ...
    ValueError: not enough initial values

    sage: from ore_algebra.analytic.singularity_analysis import *
    sage: bound_coefficients(Dz-1, [1])
    Traceback (most recent call last):
    ...
    NotImplementedError: no nonzero finite singularities
"""

import collections
import logging

from sage.all import *


from ..ore_algebra import OreAlgebra
from . import utilities
from .bounds import DiffOpBound
from .differential_operator import DifferentialOperator
from .local_solutions import (
        critical_monomials,
        FundamentalSolution,
        LocalBasisMapper,
        log_series,
        LogSeriesInitialValues,
)
from .path import Point

logger = logging.getLogger(__name__)

def trim_univariate(pol, order, varbound):
    r"""
    Replace each x^{order+j} with B(0, varbound)^j*x^order.
    """
    err = pol.base_ring().zero()
    for j, c in enumerate(list(pol)[order:]):
        err = err.add_error(abs(c)*varbound**j)
    coeff = pol.padded_list(order)
    coeff.append(err)
    return pol.parent()(coeff)

def trim_expr(f, order, n0):
    """
    Truncate and bound an expression f(1/n) to a given degree

    1/n^(order+t) (t > 0) is replaced by 1/n^order * CB(0).add_error(1/n0^t)
    """
    Expr = f.parent()
    invn = Expr.gen(0)
    CB = Expr.base_ring()
    g = Expr.zero()
    # TODO: simplify...
    for c, mon in f:
        deg = mon.degree(invn)
        if deg > order:
            tuple_mon_g = tuple(map(lambda x, y: x - y, mon.exponents()[0],
                                    (invn**(deg - order)).exponents()[0]))
            mon_g = prod(Expr.gens()[j]**(tuple_mon_g[j])
                         for j in range(len(tuple_mon_g)))
            c_g = ((c if c.mid() == 0 else CB(0).add_error(abs(c)))
                   / CB(n0**(deg - order)))
            g = g + c_g * mon_g
        else:
            g = g + c*mon
    return g

def trim_expr_series(f, order, n0):
    trimmed = f.parent()([trim_expr(c, order, n0) for c in f])
    return trimmed.add_bigoh(f.parent().default_prec())

################################################################################
# Path choice
#################################################################################

def _sing_in_disk(elts, rad, large_value):
    for j, x in enumerate(elts):
        mag = abs(x)
        if mag > rad:
            return elts[:j], mag
    return elts, large_value

def _choose_big_radius(all_exn_pts, dominant_sing, next_sing_rad):
    # [DMM, (11)]
    max_smallrad = min(abs(ex - ds) for ds in dominant_sing
                                    for ex in all_exn_pts
                                    if ex != ds)
    dom_rad = abs(dominant_sing[-1])
    rad = min(next_sing_rad*RBF(0.9) + dom_rad/10,
              dom_rad + max_smallrad*RBF(0.8))
    logger.info("radius of outer circle = %s", rad)
    return rad

def _check_big_radius(rad, dominant_sing):
    if not dominant_sing:
        raise ValueError("No singularity in the given disk")
    if not abs(dominant_sing[-1]) < rad:
        raise ValueError(f"Singularity {dominant_sing[-1]} is too close to "
                         "the border of the disk")
    rad0 = abs(CBF(dominant_sing[0]))
    bad_sing = [s for s in dominant_sing[1:] if abs(CBF(s)) > rad0]
    if bad_sing:
        warnings.warn("The given disk contains singular points of non-minimal "
                      f"modulus: {bad_sing}")

def _classify_sing(deq, known_analytic, rad):

    # Interesting points = all sing of the equation, plus the origin

    all_exn_pts = deq._singularities(QQbar, multiplicities=False)
    if not any(s.is_zero() for s in all_exn_pts):
        all_exn_pts.append(QQbar.zero())

    # Potential singularities of the function, sorted by magnitude

    singularities = deq._singularities(QQbar, include_apparent=False,
                                       multiplicities=False)
    singularities = [s for s in singularities if not s in known_analytic]
    singularities.sort(key=lambda s: abs(s))
    logger.debug(f"potential singularities: {singularities}")

    if not singularities:
        raise NotImplementedError("no nonzero finite singularities")

    # Dominant singularities

    # dominant_sing is the list of potential dominant
    # singularities of the function, not of "dominant singular points". It does
    # not include singular points of the equation lying in the disk where the
    # function is known to be analytic.
    if rad is None:
        dominant_sing, next_sing_rad = _sing_in_disk(singularities,
                abs(singularities[0]), abs(singularities[-1])*3)
        rad = _choose_big_radius(all_exn_pts, dominant_sing, next_sing_rad)
    else:
        rad = RBF(rad)
        dominant_sing, _ = _sing_in_disk(singularities, rad,
                abs(singularities[-1])*2 + rad*2)
        _check_big_radius(rad, dominant_sing)
    logger.info("dominant singularities: %s", dominant_sing)

    return all_exn_pts, dominant_sing, rad

################################################################################
# Contribution of a logarithmic monomial
# (variant with error bounds of Sage's SingularityAnalysis)
#################################################################################

def truncated_psi(n, m, invz):
    """
    Compute psi^(m)(z) (or, for m = 0, psi(z) - log(z)) truncated at z^(-m-2n-1)
    with error bound of order z^(-m-2n)

    INPUT:

    - n: integer, non-negative
    - m: integer, non-negative
    - invz: element of polynomial ring, representing 1/z

    TESTS::

        sage: from ore_algebra.analytic.singularity_analysis import truncated_psi
        sage: Pol.<invz> = CBF[]
        sage: truncated_psi(3, 0, invz)
        ([+/- ...] + [+/- ...]*I)*invz^6
        + ([0.008333...])*invz^4 + ([-0.08333...])*invz^2 - 0.5000...*invz
        sage: truncated_psi(3, 1, invz)
        ([+/- ...] + [+/- ...]*I)*invz^7
        + ([-0.0333...])*invz^5 + ([0.1666...])*invz^3 + 0.5000...*invz^2 + invz
        sage: truncated_psi(3, 2, invz)
        ([+/- ...] + [+/- ...]*I)*invz^8
        + ([0.1666...])*invz^6 + ([-0.5000...])*invz^4 - invz^3 - invz^2
        sage: truncated_psi(2, 2, invz)
        ([+/- 0.16...] + [+/- ...]*I)*invz^6
        + ([-0.5000...])*invz^4 - invz^3 - invz^2
    """
    assert n >= 1
    CB = invz.parent().base_ring()
    err = abs(rising_factorial(2*n + 1, m - 1)*bernoulli(2*n))
    sign = (-1)**(m+1)
    ser = sign * (
        + gamma(m+1) * invz**(m+1) / 2
        + sum(bernoulli(2*k)*invz**(2*k+m)*rising_factorial(2*k+1, m-1)
              for k in range(1, n)))
    if m != 0:
        ser += sign * gamma(m) * invz**m
    res = ser + CB(0).add_error(err)*invz**(2*n+m)
    return res

def _generalized_bernoulli(Ring, sigma, count):
    t = polygen(Ring, 't')
    ser = t._exp_series(count + 1) >> 1       # (e^t - 1)/t
    ser = -2*sigma*ser._log_series(count)     # -2σ·log((e^t - 1)/t)
    ser = (ser >> 2) << 2                     # -2σ·log((e^t - 1)/t) + σt
    ser = ser._exp_series(count)              # ((e^t - 1)/t)^(-2σ) * e^(σt)
    bern = [ser[n]*ZZ(n).factorial() for n in range(count)]
    return bern

def truncated_gamma_ratio(alpha, order, u, s):
    """
    Find a truncated expression with error bound for Γ(n+α)/Γ(n+1)/(n+α/2)^(α-1)

    INPUT:

    - alpha: complex number α, !!cannot be negative integer or zero!!
    - order: order of truncation
    - u: element of polynomial ring, representing 1/(n+α/2)
    - s: positive number where n >= s*|alpha| is guaranteed, s > 2

    OUTPUT:

    - a polynomial in CB[u] such that Γ(n+α)/Γ(n+1)/(n+α/2)^(α-1)
      is in its range when n >= s*|alpha|
    """

    Pol = u.parent()
    CB = Pol.base_ring()
    n_gam = ceil((1+order)/2)
    gen_bern = _generalized_bernoulli(CB, alpha/2, 2*n_gam)
    gen_bern_abs = _generalized_bernoulli(CB, abs(alpha/2), 2*n_gam + 1)

    ratio_gamma_asy = Pol([CB(1 - alpha).rising_factorial(j) / factorial(j) * b
                           if j % 2 == 0 else 0
                           for j, b in enumerate(gen_bern)])
    half = RBF.one()/2
    _alpha = CBF(alpha)
    Rnw_bound = (abs((1 - _alpha.real()).rising_factorial(2*n_gam))
                 / factorial(2*n_gam)
                 * abs(gen_bern_abs[2*n_gam])
                 * (abs(_alpha.imag())*(half/s).arcsin()).exp()
                 * ((s+half)/(s-half))**(max(0, -alpha.real()+1+2*n_gam)))
    assert not Rnw_bound < 0
    ratio_gamma = ratio_gamma_asy + CB(0).add_error(Rnw_bound) * u**(2*n_gam)
    return ratio_gamma

def truncated_inverse(alpha, order, invn, s):
    r"""
    (n + α)^(-1) as a polynomial in invn = 1/n plus an error term O(invn^order)
    """
    CB = invn.parent().base_ring()
    err = abs(alpha)**order / (1 - 1/s)
    ser = sum((-alpha)**(j-1) * invn**j for j in range(1, order))
    return ser + CB(0).add_error(err) * invn**(order)

def truncated_log(alpha, order, invn, s):
    r"""
    log(1 + α/n) as a polynomial in invn = 1/n plus an error term O(invn^order)
    """
    CB = invn.parent().base_ring()
    err = (1 + 1/s).log()*abs(CBF(alpha))**order
    ser = - sum((-alpha)**j * invn**j / j for j in range(1, order))
    return ser + CB(0).add_error(err) * invn**(order)

def truncated_power(alpha, order, invn, s):
    """
    Compute a bound for a truncated (1 + α/2n)^(α-1)

    INPUT:

    - alpha: complex number α, !!cannot be negative integer or zero!!
    - order: order of truncation
    - invn: element of polynomial ring, representing 1/n
    - s: positive number where n >= s*|alpha| is guaranteed, s > 2

    OUTPUT:

    - trunc_power: a polynomial in CB[invn] such that (1 + α/2n)^(α-1) is in its
      range when n >= s*|alpha|
    """
    CB = invn.parent().base_ring()
    _alpha = CBF(alpha)
    a = alpha.real() - 1
    h = RBF(1)/2
    err = (abs(_alpha)**(order) / (1 - 1/s)
           * (abs(alpha.imag())/2).exp()
           * max((3*h)**a, h**a))
    assert not err < 0
    t = polygen(CB, 't')
    ser = (alpha - 1) * (1 + alpha/2 * t)._log_series(order)
    ser = ser._exp_series(order)
    return ser(invn) + CB(0).add_error(err) * invn**(order)

def bound_coeff_mono(Expr, exact_alpha, log_order, order, n0, s):
    """
    Bound [z^n] (1-z)^(-α) * log(1/(1-z))^k by an expression
    of the form n^(α-1) * P(1/n, log(n)) for all k < log_order

    INPUT:

    - Expr: output ring, of the form C[invn, logn]
    - alpha: complex number, representing α
    - log_order: non-negative integer
    - order: expansion order wrt n
    - n0: integer > -alpha, lower bound of validity range
    - s: real number > 2 s.t. n >= s*|alpha| for all n >= n0

    OUTPUT:

    - a list of length log_order of polynomials in invn, logn,
      corresponding to k = 0, ..., log_order - 1
    """
    # All entries can probably be deduced from the last one using Frobenius'
    # method, but I don't think it helps much computationally(?)

    if exact_alpha.is_integer() and exact_alpha <= 0:
        reflect = True
        exact_alpha = ZZ(exact_alpha)
        # Our choice of n0 implies that the coefficients corresponding to k = 0
        # will be zero. In particular, we handle (1-z)^-α purely by returning a
        # validity range that starts after the last nonzero term.
        assert n0 >= -exact_alpha
    else:
        reflect = False

    CB = Expr.base_ring()
    alpha = CB(exact_alpha)
    invn, logn = Expr.gens()
    order = max(0, order)

    # Bound for (n+α/2)^(1-α) * Γ(n+α)/Γ(n+1)
    u = polygen(CB, 'u') # u stands for 1/(n+α/2)
    f = truncated_gamma_ratio(alpha, order, u, s)
    truncated_u = truncated_inverse(alpha/2, order, invn, s)
    f = trim_expr(f(truncated_u), order, n0)

    # Bound for (1 + α/2n)^(α-1) = (n+α/2)^(α-1) * n^(1-α)
    g = truncated_power(alpha, order, invn, s)

    # Bound for 1/Γ(n+α) (d/dα)^k [Γ(n+α)/Γ(α)]
    # Use PowerSeriesRing because polynomials do not implement all the
    # "truncated" operations we need
    Pol_invz, invz = PolynomialRing(CB, 'invz').objgen() # z = n + α
    Series_z, eps = PowerSeriesRing(Pol_invz, 'eps', log_order).objgen()
    order_psi = max(1, ceil(order/2))
    if not reflect:
        pols = [(truncated_psi(order_psi, m, invz) - alpha.psi(m))
                / (m + 1).factorial() for m in srange(log_order)]
        p = Series_z([0] + pols)
        hh1 = (1/alpha.gamma())*p.exp()
    else:
        pols = [(truncated_psi(order_psi, m, invz)
                 + (-1)**(m+1)*(1-alpha).psi(m)) / (m + 1).factorial()
                for m in srange(log_order - 1)]
        p = Series_z([0] + pols)
        _pi = CB(pi)
        sine = (-1 if exact_alpha%2 else 1)*(_pi*eps).sin()
        hh1 = ((1-alpha).gamma()/_pi)*(p.exp()*sine)
    invz_bound = ~(n0 - abs(alpha))
    hh1 = hh1.parent()([trim_univariate(c, order, invz_bound)
                        for k, c in enumerate(hh1)])

    Series, eps = PowerSeriesRing(Expr, 'eps', log_order).objgen()
    truncated_invz = truncated_inverse(alpha, order, invn, s)
    h1 = hh1.map_coefficients(Hom(Pol_invz, Expr)(truncated_invz)) # ∈ Expr[[z]]
    assert h1.parent() is Series
    h1 = trim_expr_series(h1, order, n0)

    # (n + α)^ε as a series in ε with coeffs in ℂ[logn][[invn]] (trunc+bounded)
    h2 = (truncated_log(alpha, order, invn, s)*eps).exp()
    h2 = trim_expr_series(h2, order, n0)
    h3 = (logn*eps).exp()

    h = h1*h2*h3
    h = trim_expr_series(h, order, n0)

    logger.debug("    f = %s", f)
    logger.debug("    g = %s", g)
    logger.debug("    h1 = %s", h1)
    logger.debug("    h2 = %s", h2)
    logger.debug("    h3 = %s", h3)

    full_prod = f * g * h
    full_prod = trim_expr_series(full_prod, order, n0)
    res = [ZZ(k).factorial()*c
           for k, c in enumerate(full_prod.padded_list(log_order))]
    for k, pol in enumerate(res):
        logger.debug("    1/(1-z)^%s*log(1/(1-z))^%s --> %s",
                     exact_alpha, k, pol)
    return res

#################################################################################
# Contribution of a single regular singularity
#################################################################################

def _modZ_class_ini(dop, inivec, leftmost, mults, struct):
    r"""
    Compute a LogSeriesInitialValues object corresponding to the part with a
    given local exponent mod 1 of a local solution specified by a vector of
    initial conditions.
    """
    values = { (sol.shift, sol.log_power): c
                for sol, c in zip(struct, inivec)
                if sol.leftmost == leftmost }
    ini = LogSeriesInitialValues(dop=dop, expo=leftmost, values=values,
            mults=mults, check=False)
    return ini

def _my_log_series(dop, bwrec, inivec, leftmost, mults, struct, order):
    r"""
    Similar to _modZ_class_ini() followed by log_series(), but attempts to
    minimize interval swell by unrolling the recurrence in exact arithmetic
    (once per relevant initial value) and taking a linear combination.

    The output is a list of lists, not vectors.
    """
    log_len = sum(m for _, m in mults)
    res = [[inivec.base_ring().zero()]*log_len for _ in range(order)]
    for sol, c in zip(struct, inivec):
        if c.is_zero() or sol.leftmost != leftmost:
            continue
        values = { (sol1.shift, sol1.log_power): QQ.zero()
                   for sol1 in struct if sol1.leftmost == leftmost }
        values[sol.shift, sol.log_power] = QQ.one()
        ini = LogSeriesInitialValues(dop=dop, expo=leftmost, values=values,
                mults=mults, check=False)
        ser = log_series(ini, bwrec, order)
        for i in range(order):
            for j, a in enumerate(ser[i]):
                res[i][j] += c*a
    return res

ExponentGroupData = collections.namedtuple('ExponentGroupData', [
    'val',  # exponent group (lefmost element mod ℤ)
    'bound',  # coefficient bound (explicit part + local error term)
    'initial_terms'  # explicit part of the local solution
])

SingularityData = collections.namedtuple('SingularityData', [
    'rho',  # the singularity
    'expo_group_data',  # as above, for each exponent group
    'min_val_rho',  # Re(valuation) of singular part (ignoring analytic terms)
])


class SingularityAnalyzer(LocalBasisMapper):

    def __init__(self, dop, inivec, *, rho, rad, Expr, abs_order, n0,
                 struct):

        super().__init__(dop)

        self.inivec = inivec
        self.rho = rho
        self.rad = rad
        self.Expr = Expr
        self.abs_order = abs_order
        self.n0 = n0
        self._local_basis_structure = struct

    def process_modZ_class(self):

        logger.info("sing=%s, valuation=%s", self.rho, self.leftmost)

        order = (self.abs_order - self.leftmost.real()).ceil()
        # XXX don't hardocode this; ensure order1 ≥ bwrec.order
        order1 = order + 49
        # XXX Works, and should be faster, but leads to worse bounds due to
        # using the recurrence in interval arithmetic
        # ini = _modZ_class_ini(self.edop, self.inivec, self.leftmost, self.shifts,
        #                       self._local_basis_structure) # TBI?
        # ser = log_series(ini, self.shifted_bwrec, order1)
        ser = _my_log_series(self.edop, self.shifted_bwrec, self.inivec,
                self.leftmost, self.shifts, self._local_basis_structure, order1)

        CB = CBF # XXX
        smallrad = self.rad - CB(self.rho).below_abs()
        # XXX do we really a bound on the tail *of order `order`*? why not
        # compute a bound on the tail of order `order1` and put everything else
        # in the "explicit terms" below?
        vb = _bound_tail(self.edop, self.leftmost, smallrad, order, ser)

        s = RBF(self.n0) / (abs(self.leftmost) + abs(order))
        assert s > 2

        # (Bound on) Maximum power of log that may occur the sol defined by ini.
        # (We could use the complete family of critical monomials for a tighter
        # bound when order1 < max shift, since for now we currently are
        # computing it anyway...)
        assert self.shifts[0][0] == 0 and order1 > 0
        kappa = max(k for shift, mult in self.shifts if shift < order1
                      for k, c in enumerate(ser[shift]) if not c.is_zero())
        kappa += sum(mult for shift, mult in self.shifts if shift >= order1)
        # XXX ici aussi il faut traiter le cas particulier des entiers

        bound_lead_terms, initial_terms = _bound_local_integral_explicit_terms(
                self.rho, self.leftmost, order, self.Expr, s, self.n0, ser[:order])
        bound_int_SnLn = _bound_local_integral_of_tail(self.rho,
                self.leftmost, order, self.Expr, s, self.n0, vb, kappa)

        logger.info("  explicit part = %s", bound_lead_terms)
        logger.info("  local error term = %s", bound_int_SnLn)

        data = ExponentGroupData(
            val = self.leftmost,
            bound = bound_lead_terms + bound_int_SnLn,
            initial_terms = initial_terms)

        # XXX Abusing FundamentalSolution somewhat; consider creating another
        # type of record compatible with FundamentalSolution if this stays.
        # The log_power field is not meaningful, but we need _some_ integer
        # value to please code that will try to sort the solutions.
        sol = FundamentalSolution(leftmost=self.leftmost, shift=ZZ.zero(),
                                  log_power=kappa, value=data)
        self.irred_factor_cols.append(sol)

def _bound_tail(dop, leftmost, smallrad, order, series):
    r"""
    Upper-bound the tail of order ``order`` of a logarithmic series solution of
    ``dop`` with exponents in ``leftmost`` + ℤ, on a disk of radius
    ``smallrad``, using ``order1`` ≥ ``order`` explicitly computed terms given
    as input in ``series`` and a bound based on the method of majorants for the
    terms of index ≥ ``order1``.
    """
    assert order <= len(series)
    maj = DiffOpBound(dop, leftmost=leftmost, pol_part_len=30, # XXX
                                                    bound_inverse="solve")
    ordrec = maj.dop.degree()
    last = list(reversed(series[-ordrec:]))
    order1 = len(series)
    # Coefficients of the normalized residual in the sense of [Mez19, Sec.
    # 6.3], with the indexing conventions of [Mez19, Prop. 6.10]
    CB = CBF # TBI
    res = maj.normalized_residual(order1, last, Ring=CB)
    # Majorant series of [the components of] the tail of the local expansion
    # of f at ρ. See [Mez19, Sec. 4.3] and [Mez19, Algo. 6.11].
    tmaj = maj.tail_majorant(order1, [res])
    # Make a second copy of the bound before we modify it in place.
    tmaj1 = maj.tail_majorant(order1, [res])
    # Shift it (= factor out z^order) ==> majorant series of the tails
    # of the coefficients of log(z)^k/k!
    tmaj1 >>= -order
    # Bound on the *values* for |z| <= smallrad of the analytic functions
    # appearing as coefficients of log(z)^k/k! in the tail of order 'order1' of
    # the local expansion
    tb = tmaj1.bound(smallrad)
    # Bound on the intermediate terms
    ib = sum(smallrad**n1 * max(c.above_abs() for c in vec)
            for n1, vec in enumerate(series[order:]))
    # Same as tb, but for the tail of order 'order'
    return tb + ib

def _bound_local_integral_of_tail(rho, val_rho, order, Expr, s, n0, vb, kappa):

    invn, logn = Expr.gens()

    CB = CBF # These are error terms, no need for high prec. Still, TBI.
    RB = RBF

    # Change representation from log(z-ρ) to log(1/(1 - z/ρ))
    # The h_i are cofactors of powers of log(z-ρ), not log(1/(1-z/ρ)).
    # Define the B polynomial in a way that accounts for that.
    ll = abs(CB(-rho).log())
    B = vb*RB['z']([
            sum([ll**(m - j) * binomial(m, j) / factorial(m)
                    for m in range(j, kappa + 1)])
            for j in range(kappa + 1)])

    # Sub polynomial factor for bound on S(n)
    cst_S = CB(0).add_error(CB(abs(rho)).pow(val_rho.real()+order)
            * ((abs(CB(rho).arg()) + 2*RB(pi))*abs(val_rho.imag())).exp()
            * CB(1 - 1/n0).pow(CB(-n0-1)))
    bound_S = cst_S*B(CB(pi)+logn)
    # Sub polynomial factor for bound on L(n)
    if val_rho + order <= 0:
        C_nur = 1
    else:
        C_nur = 2 * (CB(e) / (CB(val_rho.real()) + order)
                            * (s - 2)/(2*s)).pow((RB(val_rho.real()) + order))
    cst_L = (CB(0).add_error(C_nur * CB(1/pi)
                                * CB(abs(rho)).pow(RB(val_rho.real())+order))
        * ((abs(CB(rho).arg()) + 2*RB(pi))*abs(val_rho.imag())).exp())
    bound_L = cst_L*B(CB(pi)+logn)

    return Expr(bound_S + bound_L) * invn**order

def _bound_local_integral_explicit_terms(rho, val_rho, order, Expr, s, n0, ser):

    invn, logn = Expr.gens()
    CB = Expr.base_ring()

    # Rewrite the local expansion in terms of new variables Z = z - ρ,
    # L = log(1/(1-z/rho))

    PolL, L = PolynomialRing(CB, 'L').objgen()
    PolL, Z = PolynomialRing(PolL, 'Z').objgen()
    mylog = CB.coerce(-rho).log() - L # = log(z - ρ) for Im(z) ≥ 0
    locf_ini_terms = sum(c/ZZ(k).factorial() * mylog**k * Z**shift
                         for shift, vec in enumerate(ser)
                         for k, c in enumerate(vec))

    bound_lead_terms = Expr.zero()
    for degZ, slice in enumerate(locf_ini_terms):
        logger.debug("  Z^(%s - %s)*(...)...", -val_rho, degZ)
        # XXX could be shared between singularities with common exponents...
        # (=> tie to an object and add @cached_method decorator?)
        coeff_bounds = bound_coeff_mono(Expr, -val_rho-degZ, slice.degree() + 1,
                                        order - degZ, n0, s)
        new_term = (CB(-rho).pow(CB(val_rho+degZ)) * invn**(degZ)
                    * sum(c*coeff_bounds[degL] for degL, c in enumerate(slice)))
        bound_lead_terms += new_term
        logger.debug("  Z^%s*(%s) --> %s", -val_rho-degZ, slice, new_term)

    return bound_lead_terms, locf_ini_terms

def contribution_single_singularity(deq, ini, rho, rad, Expr,
        rel_order, n0):

    eps = RBF.one() >>  Expr.base_ring().precision() + 13
    tmat = deq.numerical_transition_matrix([0, rho], eps, assume_analytic=True)
    coord_all = tmat*ini

    ldop = deq.shift(Point(rho, deq))

    # Redundant work; TBI
    # (Cases where we really need this to detect non-analyticity are rare...)
    crit = critical_monomials(ldop)

    # XXX could move to SingularityAnalyzer if we no longer return min_val_rho
    nonanalytic = [sol for sol in crit if not (
        sol.leftmost.is_integer()
        and sol.leftmost + sol.shift >= 0
        and all(c.is_zero() for term in sol.value.values() for c in term[1:]))]
    if not nonanalytic:
        return None
    min_val_rho = (nonanalytic[0].leftmost + nonanalytic[0].shift).real()
    abs_order = rel_order + min_val_rho

    # Split the local expansion of f according to the local exponents mod ℤ. For
    # each group (ℤ-coset) of exponents, compute coefficient asymptotics (and
    # some auxiliary data). Again: each element of the output corresponds to a
    # whole ℤ-coset of exponents, already incorporating initial values.
    analyzer = SingularityAnalyzer(dop=ldop, inivec=coord_all, rho=rho,
            rad=rad, Expr=Expr, abs_order=abs_order, n0=n0,
            struct=crit)
    data = analyzer.run()

    data1 = SingularityData(
        rho = rho,
        expo_group_data = [sol.value for sol in data],
        min_val_rho = min_val_rho
    )

    return data1

################################################################################
# Exponentially small error term
################################################################################

def numerical_sol_big_circle(deq, ini, dominant_sing, rad, halfside):
    """
    Bound the values of a solution `f` on the large circle

    INPUT:

    - ``deq``: a linear ODE that the generating function satisfies
    - ``ini``: coefficients of `f` in the basis at zero
    - ``dominant_sing``: list of dominant singularities
    - ``rad``: radius of big circle
    - ``halfside``: half of side length of covering squares

    OUTPUT:

    A list of (square, value) pairs where the squares cover the large circle
    and, for each square, the corresponding value is a complex interval
    containing the image of the square by the analytic continuation of `f` to a
    multi-slit disk.
    """
    logger.info("starting to compute values on outer circle...")
    clock = utilities.Clock()
    clock.tic()

    I = CBF.gen(0)
    # TODO: eps could be chosen as a function of halfside or something...
    eps = RBF.one() >> 100
    halfside = RBF(halfside)

    sings = [CBF(s) for s in dominant_sing]
    sings.sort(key=lambda s: s.arg())
    num_sings = len(sings)
    pairs = []
    num_sq = 0
    for j0 in range(num_sings):
        j1 = (j0 + 1) % num_sings
        arg0 = sings[j0].arg()
        arg1 = sings[j1].arg()
        if j1 == 0:
            # last arc is a bit special: we need to add 2*pi to ending
            arg1 += 2*RBF.pi()

        # Compute initial values at a point on the large circle, halfway between
        # two adjacent dominant singularities
        hub = rad * ((arg0 + arg1)/2 * I).exp()
        tmat_hub = deq.numerical_transition_matrix([0, hub], eps,
                                                   assume_analytic=True)
        ini_hub = tmat_hub*vector(ini)

        # From there, walk along the large circle in both directions
        halfarc = (arg1 - arg0)/2
        np = ZZ(((halfarc*rad / (2*halfside)).above_abs()).ceil()) + 2
        num_sq += np
        # TODO: optimize case of real coefficients
        # TODO: check path correctness (plot?) in complex cases
        for side in [1, -1]:
            squares = [[(hub*(side*halfarc*k/np*I).exp()).add_error(halfside)]
                       for k in range(np+1)]
            path = [hub] + squares
            pairs += deq.numerical_solution(ini_hub, path, eps)

    clock.toc()
    logger.info("...done, %d squares of half-side %s, %s",
                num_sq, halfside, clock)
    return pairs

def max_big_circle(deq, ini, dominant_sing, sing_data, rad, halfside):

    pairs = numerical_sol_big_circle(deq, ini, dominant_sing, rad, halfside)
    covering, f_big_circle = zip(*pairs)

    sum_g = [
        sum((_z-_rho).pow(CBF(edata.val))
                # some of the _z may lead to arguments of log that cross the
                # branch cut, but that's okay
                * edata.initial_terms(Z=_z-_rho, L=(~(1-_z/_rho)).log())
            for sdata in sing_data for _rho in (CBF(sdata.rho),)
            for edata in sdata.expo_group_data)
        for j, _z in enumerate(covering)]
    res = RBF.zero().max(*((s - vv).above_abs()
                           for s, vv in zip(sum_g, f_big_circle)))
    return res

def absorb_exponentially_small_term(CB, cst, ratio, beta, final_kappa, n0, n):
    if beta <= n0 * ratio.log():
        _beta = RBF(beta)
        cst1 = _beta.exp()*(_beta/ratio.log()).pow(_beta)
    else:
        cst1 = ratio**n0 * CBF(n0)**(-beta)
    rad_err = cst*cst1 / CBF(n0).log()**final_kappa
    return (CB(0).add_error(rad_err) * n**QQbar(beta) * log(n)**final_kappa)

def add_error_term(bound, rho, term, n):
    for i, (rho1, local_bound) in enumerate(bound):
        if rho1 == rho:
            # We know that the last term is an error term with the same
            # power of n and log(n) as error_term_big_circle
            local_bound[-1] = (local_bound[-1] + term).collect(n)
        return
    else:
        bound.append([rho, term])

################################################################################
# Conversion to an asymptotic expansion
################################################################################

class FormalProduct:

    def __init__(self, exponential_factor, series_factor):
        self._exponential_factor = exponential_factor
        self._series_factor = series_factor

    def __repr__(self):
        return f"{self._exponential_factor}*({self._series_factor})"

    def exponential_factor(self):
        return self._exponential_factor

    def series_factor(self):
        return self._series_factor

    def expand(self):
        return self._exponential_factor*self._series_factor

def to_asymptotic_expansion(Coeff, name, term_data, n0, beta, kappa,
                            big_oh_rad=None):

    from sage.categories.cartesian_product import cartesian_product
    from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing
    from sage.rings.asymptotic.growth_group import (
            ExponentialGrowthGroup,
            GrowthGroup,
            MonomialGrowthGroup,
    )
    from sage.rings.asymptotic.term_monoid import (
            BTerm,
            BTermMonoid,
            ExactTermMonoid,
    )
    from sage.rings.asymptotic.term_monoid import DefaultTermMonoidFactory
    from sage.symbolic.operators import add_vararg

    n = SR.var(name)

    # XXX detect cases where we can use 1 or ±1 or U as Arg
    Exp, Arg = ExponentialGrowthGroup.factory(QQbar, name, return_factors=True)
    # AsymptoticRing does not split MonomialGrowthGroups with non-real
    # exponent groups in growth*non-growth parts, presumably because this has no
    # impact on term ordering. Let us do the same.
    Pow = MonomialGrowthGroup(QQbar, name)
    Log = MonomialGrowthGroup(ZZ, f"log({name})")
    Growth = cartesian_product([Arg, Exp, Pow, Log])
    Asy = AsymptoticRing(Growth, coefficient_ring=Coeff)
    ET = Asy.term_monoid('exact')
    BT = Asy.term_monoid('B').change_parameter(
            coefficient_ring=Coeff._real_field())
    OT = Asy.term_monoid('O')

    def make_arg_factor(dir):
        if dir.imag().is_zero() and dir.real() >= 0:
            return ET.one()
        else:
            return ET(Arg(raw_element=dir))

    rho0 = term_data[0][0]
    mag0 = abs(rho0)
    exp_factor = ET(Exp(raw_element=~mag0))
    if all(rho == rho0 for rho, _ in term_data[1:]):
        exp_factor *= make_arg_factor(~rho0)
    else:
        rho0 = mag0

    error_term_growth = ET(n**beta*log(n)**kappa)

    terms = []
    for rho, symterms in term_data:
        dir = rho0/rho
        assert abs(dir).is_one() # need an additional growth factor otherwise
        arg_factor = make_arg_factor(dir)
        for symterm in symterms:
            term = arg_factor*ET(symterm.subs(n=n))
            if term.growth == error_term_growth:
                assert term.coefficient.contains_zero()
                term = BT(term.growth, coefficient=term.coefficient.above_abs(),
                        valid_from={name: n0})
            terms.append(term)

    if big_oh_rad is not None:
        terms.append(OT((rho0/big_oh_rad)**n))

    return FormalProduct(Asy(exp_factor), Asy(terms))

################################################################################
# Final bound
################################################################################

def _coeff_zero(seqini, deq):
    """
    Find coefficients of generating function in the basis with these expansions
    at the origin

    INPUT:

    - seqini : list, initial terms of the sequence
    - deq : a linear ODE that the generating function satisfies

    OUTPUT:

    - coeff : vector, coefficients of generating function in the basis with
      these expansions at the origin
    """

    list_basis = deq.local_basis_expansions(0)
    list_coeff = []
    for basis in list_basis:
        mon = next(m for c, m in basis if not c == 0)
        if mon.k == 0 and mon.n >= 0:
            if mon.n >= len(seqini):
                raise ValueError(f"not enough initial values")
            list_coeff.append(seqini[mon.n])
        else:
            list_coeff.append(0)
    return vector(list_coeff)

def _bound_validity_range(n0, dominant_sing, order):

    # Make sure the disks B(ρ, |ρ|/n) contain no other singular point

    # FIXME: currently DOES NOT match [DMM, (10)]
    if len(dominant_sing) > 1:
        min_dist = min(s0.dist_to_sing() for s0 in dominant_sing)
        n1 = ceil(2*abs(dominant_sing[-1])/min_dist)
    else:
        n1 = 0

    # Make sure that n0 > 2*|α| for all exponents α we encounter

    max_abs_val = max(abs(sol.leftmost) # TODO: avoid redundant computation...
                      for s0 in dominant_sing
                      for sol in s0.local_basis_structure())
    n2 = max_abs_val + order + 1
    # FIXME: slightly different from [DMM, (46)]
    n0 = max(n0, ceil(2.1*n2), n1)

    logger.debug(f"{n1=}, {n2=}, {n0=}")
    return n0

def truncate_tail_SR(val, f, beta, kappa, n0, n):
    """
    Truncate an expression n^val*f(1/n) to a given order

    1/n^(beta+t), t>=0 is replaced by cst*logn^kappa/n^beta

    INPUT:

    - val: algebraic number
    - f: polynomial in invn (representing 1/n) and logn (representing log(n))
    - beta, kappa: parameters of the new error term
    - n0: integer, result valid for n >= n0
    - n : symbolic variable

    OUTPUT:

    a list of symbolic expressions whose sum forms the new bound
    """
    Expr = f.parent()
    CB = Expr.base_ring()
    invn, logn = Expr.gens()
    g = []
    error_term = SR.zero()
    for deg_invn in range(f.degree(invn) + 1):
        for c, mon in list(f.coefficient({invn: deg_invn})):
            deg_logn = mon.degree(logn)
            if val.real() - deg_invn > beta:
                g.append(c * n**(val - deg_invn) * log(n)**deg_logn)
            else:
                c_g = (((c if c.mid() == 0 else CB(0).add_error(abs(c)))
                        / CB(n0).pow(beta + deg_invn - val.real()))
                       * CB(n0).log().pow(deg_logn - kappa))
                error_term += c_g * n**beta * log(n)**kappa
    g.append(error_term)
    return g

def bound_coefficients(deq, seqini, name='n', order=3, prec=53, n0=0, *,
                       known_analytic=[0], rad=None, halfside=None,
                       output='asymptotic_expansion',
                       ignore_exponentially_small_term=False):
    """
    Compute a bound for the n-th element of a holonomic sequence

    INPUT:

    - ``deq``: a linear ODE that the generating function satisfies
    - ``seqini``: list, initial elements of sequence, long enough to determine
      the sequence
    - `name` (optional): variable name to be used in the formula
    - ``order`` (optional): expansion order, counting only powers of the
      variable (as opposed to log factors)
    - ``prec`` (optional): numeric working precision (in bits)
    - ``n0`` (optional): restrict to n >= n0 (note that the final validity range
      may be smaller)
    - ``known_analytic`` (optional, default `[0]`): list of points where the
      generating function is known to be analytic, default is [0]
    - ``rad`` (optional, default: automatic choice): radius of the outer part of
      the integration contour (corresponding to an exponentially small error
      term)
    - ``halfside`` (optional, default: automatic choice): resolution parameter
      used in the computation of the exponentially small error term
    - ``output`` (optional, default: ``asymptotic_expansion``): set to ``list``
      to get the results as a list of terms instead of an
      ``AsymptoticRingElement``

    OUTPUT:

    - when ``output='asymptotic_expansion'``: an ``AsymptoticRingElement`` with
      a B-term encoding the error bound

    - when ``output='list'``: a pair ``(N0, bound)`` where
        - ``N0`` is an integer such that the bound is valid when ``n >= N0``
        - ``bound`` is list of lists of the form ``[rho, ser]`` where ``rho`` is
          in ``QQbar`` and ``ser`` is a symbolic expression
      such that the sum of ``rho**(-n) * ser`` is a bound for the `n`-th element
      of the input sequence
    """
    if output not in ['asymptotic_expansion', 'list']:
        raise ValueError(f"unknown output format: {output}")

    CB = ComplexBallField(prec)

    deq = DifferentialOperator(deq)

    # Identify dominant singularities, choose big radius
    all_exn_pts, dominant_sing, rad = _classify_sing(deq, known_analytic, rad)

    # Compute validity range
    _dominant_sing = [Point(s, deq) for s in dominant_sing] # tmp, should use Points everywhere
    n0 = _bound_validity_range(n0, _dominant_sing, order)

    # Convert initial sequence terms to solution coordinates in the basis at 0
    ini = _coeff_zero(seqini, deq)

    # Contribution of each singular point

    Expr = PolynomialRing(CB, ['invn', 'logn'], order='lex')
    invn, logn = Expr.gens()

    sing_data = [contribution_single_singularity(deq, ini, rho, rad, Expr,
                                                 order, n0)
                 for rho in dominant_sing]
    sing_data = [sdata for sdata in sing_data if sdata is not None]

    # All error terms will be reduced to the form cst*n^β*log(n)^final_kappa
    final_kappa = max(edata.bound.degree(logn)
                      for sdata in sing_data
                      for edata in sdata.expo_group_data)
    beta = - min(sdata.min_val_rho for sdata in sing_data) - 1 - order

    n = SR.var(name)
    bound = [(sdata.rho,
              [term for edata in sdata.expo_group_data
               for term in truncate_tail_SR(-edata.val-1, edata.bound, beta,
                                            final_kappa, n0, n)])
             for sdata in sing_data]

    # Exponentially small error term

    if ignore_exponentially_small_term:
        big_oh_rad = QQ(rad.below_abs())
    else:
        big_oh_rad = None
        if halfside is None:
            halfside = min(abs(abs(ex) - rad) for ex in all_exn_pts)/10
        cst = max_big_circle(deq, ini, dominant_sing, sing_data, rad, halfside)
        mag_dom = abs(dominant_sing[0])
        error_term_big_circle = absorb_exponentially_small_term(CB, cst,
            mag_dom/rad, beta, final_kappa, n0, n)
        logger.info("global error term = %s*%s^(-%s) ∈ %s*%s^(-%s)", cst, rad,
                    name, error_term_big_circle, mag_dom, name)
        add_error_term(bound, mag_dom, error_term_big_circle, n)

    if output == 'list':
        return n0, bound
    else:
        try:
            asy = to_asymptotic_expansion(CB, name, bound, n0,
                                          beta, final_kappa, big_oh_rad)
        except (ImportError, ValueError):
            raise RuntimeError(f"conversion of bound {bound} to an asymptotic "
                               "expansion failed, try with output='list' or a "
                               "newer Sage version")
        return asy

def eval_bound(bound, n_num, prec = 53):
    """
    Evaluation of a bound produced in contribution_all_singularity()
    """
    CBFp = ComplexBallField(prec)
    list_eval = [rho**(-n_num) * ser.subs(n = n_num) for rho, ser in bound]
    return CBFp(sum(list_eval))


# XXX test the tester!
def check_seq_bound(asy, ref, indices=None, *, verbose=False, force=False):
    r"""
    """
    Coeff = asy.parent().coefficient_ring
    myCBF = Coeff.complex_field()
    myRBF = myCBF.base()
    # An asymptotic ring with exponents etc. in CBF instead of QQbar, to make it
    # possible to evaluate a^n, n^b
    BGG = cartesian_product([
        ExponentialGrowthGroup.factory(myCBF, 'n',
                                       extend_by_non_growth_group=True),
        MonomialGrowthGroup.factory(myRBF, 'n',
                                    extend_by_non_growth_group=True),
        GrowthGroup('log(n)^ZZ')])
    BAsy = AsymptoticRing(BGG, myCBF)
    # XXX wrong results in the presence of complex exponents (#32500)
    basy = BAsy(asy, simplify=False)
    exact_part = basy.exact_part()
    error_part = basy.error_part()
    assert len(error_part.summands) == 1
    bterm = error_part.summands.pop()
    assert isinstance(bterm, BTerm)
    error_ball = myCBF.zero().add_error(bterm.coefficient)
    (name,) = asy.variable_names()
    validity = bterm.valid_from[name]
    if indices is None:
        try:
            rb = int(len(ref))
        except TypeError:
            rb = validity + 10
        indices = range(validity, rb)
    for n in indices:
        if n < validity and not force:
            continue
        bn = myRBF(n)
        one = myRBF.one()
        refval = ref[n]
        asyval = exact_part.substitute({name: bn})
        err0 = bterm.growth._substitute_({name: bn, '_one_': one})
        relbound = (asyval - refval)/err0
        if relbound not in error_ball:
            absbound = asyval + error_ball*err0
            if absbound.contains_exact(refval):
                # this may be a false warning due to overestimation at
                # evaluation time
                print(f"{name} = {n}, magnitude of interval rescaled "
                        f"error {relbound} is larger than {bterm.coefficient}")
            else:
                # definitely wrong
                msg = (f"{name} = {n}, computed enclosure "
                        f"{absbound} does not contain reference value {refval}"
                        f" ≈ {myCBF(refval)}")
                if force:
                    print(msg)
                else:
                    raise AssertionError(msg)
        else:
            if verbose:
                print(f"{name} = {n}, {relbound} contained in {error_ball}")
