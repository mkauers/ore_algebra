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


"""

import collections
import logging
import time

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
from .ui import multi_eval_diffeq

logger = logging.getLogger(__name__)

def truncate_tail(f, deg, min_n, invn, kappa = None, logn = None):
    """
    Truncate and bound an expression f(1/n) to a given degree

    If kappa is None, 1/n^(deg+t) (t > 0) will be truncated to
        1/n^deg * CB(0).add_error(1/min_n^t)
    If kappa is not None, then 1/n^(deg+t) will be truncated to
        logn^kappa/n^deg * CB(0).add_error(**)

    INPUT:

    - f : polynomial in invn = 1/n to be truncated
    - deg : desired degree (in invn) of polynomial after truncation
    - min_n : positive number where n >= min_n is guaranteed
    - invn : element of polynomial ring, representing 1/n
    - kappa : integer, desired degree (in logn) of polynomial after truncation
    - logn : element of polynomial ring, representing log(n)

    OUTPUT:

    - g : a polynomial in CB[invn] such that f is in its range when n >= min_n
    """
    R = f.parent()
    CB = R.base_ring()
    g = R(0)
    if kappa is None:
        for c, mon in f:
            deg_w = mon.degree(invn)
            if deg_w > deg:
                tuple_mon_g = tuple(map(lambda x, y: x - y, mon.exponents()[0],
                                        (invn**(deg_w - deg)).exponents()[0]))
                mon_g = prod(R.gens()[j]**(tuple_mon_g[j])
                             for j in range(len(tuple_mon_g)))
                c_g = ((c if c.mid() == 0 else CB(0).add_error(c.above_abs()))
                        / CB(min_n**(deg_w - deg)))
                g = g + c_g * mon_g
            else:
                g = g + c*mon
    else:
        for c, mon in f:
            deg_w = mon.degree(invn)
            deg_logn = mon.degree(logn)
            if deg_w >= deg:
                tuple_mon_g = tuple(map(lambda x, y, z: x - y + z, mon.exponents()[0],
                                        (invn**(deg_w - deg)).exponents()[0],
                                        (logn**(kappa - deg_logn)).exponents()[0]))
                mon_g = prod(R.gens()[j]**(tuple_mon_g[j])
                             for j in range(len(tuple_mon_g)))
                c_g = ((c if c.mid() == 0 else CB(0).add_error(c.above_abs()))
                        / CB(min_n**(deg_w - deg))) * CB(min_n).log().pow(deg_logn - kappa)
                g = g + c_g * mon_g
            else:
                g = g + c*mon
    return g

def truncate_tail_SR(val, f, deg, min_n, invn, kappa, logn, n):
    """
    Truncate and bound an expression n^val*f(1/n) to a given degree
    1/n^(deg+t), t>=0 will be truncated to
        logn^kappa/n^deg * CB(0).add_error(**)

    INPUT:

    - f : polynomial in invn = 1/n to be truncated
    - deg : desired degree (in n) of expression after truncation
    - min_n : positive number where n >= min_n is guaranteed
    - invn : element of polynomial ring, representing 1/n
    - kappa : integer, desired degree (in logn) of polynomial after truncation
    - logn : element of polynomial ring, representing log(n)
    - n : symbolic ring variable

    OUTPUT:

    - g : an Symbolic Ring expression in variable n, such that f is in its range when n >= min_n
    """
    R = f.parent()
    CB = R.base_ring()
    g = SR(0)
    for c, mon in f:
        deg_w = mon.degree(invn)
        deg_logn = mon.degree(logn)
        if val.real() - deg_w <= deg:
            c_g = ((c if c.mid() == 0 else CB(0).add_error(c.above_abs()))
                    / CB(min_n).pow(deg + deg_w - val.real())) * CB(min_n).log().pow(deg_logn - kappa)
            g = g + c_g * n**deg * log(n)**kappa
        else:
            g = g + c * n**(val - deg_w) * log(n)**deg_logn
    return g

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
    logger.info("Radius of large circle: R₀ = %s", rad)
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
    logger.debug("dominant singularities: %s", dominant_sing)

    return all_exn_pts, dominant_sing, rad

################################################################################
# Contribution of a logarithmic monomial
# (variant with error bounds of Sage's SingularityAnalysis)
#################################################################################

def Lie_der(f, tup_x_k):
    """
    find f'

    INPUT:

    - f : polynomial in variables x_0, x_1, ..., x_(n-1)
    - tup_x_k : tuple of variables x_0, x_1, ..., x_n

    OUTPUT:

    - f_prime : f', where (x_i)' = x_(i+1)
    """
    n = len(tup_x_k) - 1
    f_prime = sum([derivative(f, tup_x_k[k]) * tup_x_k[k+1] for k in range(n)])
    return f_prime

def _der_expf_R(k, R):
    """
    Auxiliary function used in recursion for der_expf

    Find an expression for [(d/dx)^k exp(f(x))]/exp(f(x)) given in a polynomial
    ring R
    """
    tup_f_k = R.gens()
    if k == 0:
        return R(1)
    else:
        der_k_minus_one = _der_expf_R(k-1, R)
        der_k = tup_f_k[0] * der_k_minus_one + Lie_der(der_k_minus_one, tup_f_k)
        return der_k

def der_expf(k):
    """
    Find an expression for [(d/dx)^k exp(f(x))]/exp(f(x))
    """
    R = PolynomialRing(ZZ, 'f', k)
    der_k = _der_expf_R(k, R)
    return der_k

def truncated_psi(n, m, v, logz):
    """
    Compute psi^(m)(z) truncated at z^(-m-2n-1) with error bound of order
    z^(-m-2n-2)

    INPUT:

    - n : integer, non-negative
    - m : integer, non-negative
    - v : element of polynomial ring, representing 1/z
    - logz : element of polynomial ring, representing log(z)
    """
    R = v.parent()
    CB = R.base_ring()
    Enz_coeff = (2*abs(bernoulli(2*n+2)) * (m + 2*n + 2)**(m + 2*n + 2)
            / (2*n + 1)**(2*n + 1) / (m + 1)**(m + 1)
            * gamma(m+2) / (2*n + 1) / (2*n + 2))
    if m == 0:
        return R(logz - v / 2
                - sum(bernoulli(2*k)*v**(2*k)/(2*k) for k in range(1,n+1))
                + CB(0).add_error(Enz_coeff)*v**(2*n+2))
    else:
        return R((-1)**(m+1) * (gamma(m) * v**m + gamma(m+1) * v**(m+1) / 2
            + sum(bernoulli(2*k)*v**(2*k+m)*rising_factorial(2*k+1, m-1)
                  for k in range(1,n+1)))
            + CB(0).add_error(Enz_coeff)*v**(2*n+m+2))

def truncated_logder(alpha, l, order, v, logz, min_n=None):
    """
    Find a truncated expression with error bound for
    [(d/dα)^l (Γ(n+α)/Γ(α))] / (Γ(n+α)/Γ(α))

    INPUT:

    - alpha : complex number α, !!cannot be negative integer or zero!!
    - l : integer, non-negative
    - order : order of truncation
    - v : element of polynomial ring, representing 1/(n+α)
    - logz : element of polynomial ring, representing log(n+α)
    - min_n : positive number where n >= min_n is guaranteed, min_n > -alpha
      needed

    OUTPUT:

    a polynomial in CB[v] such that [(d/dα)^l (Γ(n+α)/Γ(α))] / (Γ(n+α)/Γ(α)) is
    in its range when n >= max(s*|alpha|, min_n)
    """
    list_f = []
    R = v.parent()
    CB = R.base_ring()
    for m in range(l):
        n = max(0, ceil((order - m - 1)/2))
        Enz_coeff = (abs(bernoulli(2*n+2)) * (m + 2*n + 2)**(m + 2*n + 2)
                / (2*n + 1)**(2*n + 1) / (m + 1)**(m + 1)
                * gamma(m+2) / (2*n + 1) / (2*n + 2))
        list_f.append(truncated_psi(n, m, v, logz) - CB(alpha).psi(m))
        if not min_n is None:
            list_f[-1] = truncate_tail(list_f[-1], order+1, min_n + alpha, v)
    p = der_expf(l)
    if not min_n is None:
        return R(1) if l == 0 else truncate_tail(p(list_f), order+1, min_n + alpha, v)
    else:
        return p(list_f)

def _generalized_bernoulli(Ring, sigma, count):
    t = polygen(Ring, 't')
    ser = t._exp_series(count + 1) >> 1       # (e^t - 1)/t
    ser = -2*sigma*ser._log_series(count)     # -2σ·log((e^t - 1)/t)
    ser = (ser >> 2) << 2                     # -2σ·log((e^t - 1)/t) + σt
    ser = ser._exp_series(count)              # ((e^t - 1)/t)^(-2σ) * e^(σt)
    bern = [c*ZZ(n).factorial() for n, c in enumerate(ser)]
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

    - ratio_gamma : a polynomial in CB[u] such that Γ(n+α)/Γ(n+1)/(n+α/2)^(α-1)
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
    Rnw_bound = ((1 - _alpha.real()).rising_factorial(2*n_gam)
                 / factorial(2*n_gam)
                 * abs(gen_bern_abs[2*n_gam])
                 * (abs(_alpha.imag())*(half/s).arcsin()).exp()
                 * ((s+half)/(s-half))**(max(0, -alpha.real()+1+2*n_gam)))
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
    t = polygen(CB, 't')
    ser = (alpha - 1) * (1 + alpha/2 * t)._log_series(order)
    ser = ser._exp_series(order)
    return ser(invn) + CB(0).add_error(err) * invn**(order)

def bound_coeff_mono(Expr, alpha, log_order, order, s, n0):
    """
    Compute a bound for [z^n] (1-z)^(-α) * log(1/(1-z))^log_order,
    of the form n^(α-1) * P(1/n, log(n))

    INPUT:

    - alpha: complex number, representing α
    - log_order: non-negative integer
    - order: degree of P wrt. the variable 1/n
    - s: positive number where n >= s*|alpha| is guaranteed, s > 2
    - n0: positive number where n >= n0 is guaranteed, n0 > -alpha needed

    OUTPUT:

    - P: polynomial in invn, logn
    """
    CB = Expr.base_ring()
    exact_alpha = QQbar(alpha)
    alpha = CB(alpha)
    v, logz, invn, logn = Expr.gens()
    order = max(0, order)
    if not (exact_alpha.is_integer() and exact_alpha <= 0):
        # Value of 1/Γ(α)
        c = 1/gamma(alpha)
        # Bound for (n+α/2)^(1-α) * Γ(n+α)/Γ(n+1)
        u = polygen(CB, 'u') # u stands for 1/(n+α/2)
        f = truncated_gamma_ratio(alpha, order, u, s)
        truncated_u = truncated_inverse(alpha/2, order, invn, s)
        f_z = truncate_tail(f.subs({u: truncated_u}), order, n0, invn)
        # Bound for [(d/dα)^log_order (Γ(n+α)/Γ(α)Γ(n+1))] / (Γ(n+α)/Γ(α)Γ(n+1))
        # v stands for 1/(n+α)
        g = truncated_logder(alpha, log_order, order, v, logz, n0)
        truncated_v = truncated_inverse(alpha, order, invn, s)
        truncated_logz = logn + truncated_log(alpha, order, invn, s)
        g_z = truncate_tail(
            g.subs({v: truncated_v, logz: truncated_logz}),
            order, n0, invn)
        # Bound for (1 + α/2n)^(α-1) = (n+α/2)^(α-1) * n^(1-α)
        h_z = truncated_power(alpha, order, invn, s)
        product_all = c * f_z * g_z * h_z
        return truncate_tail(product_all, order, n0, invn)
    elif log_order == 0:
        # Terminating expansion of the form (1-z)^N, N = -α ∈ ℕ
        # The only nontrivial case n0 <= α should not happen with our n0,
        # but might be worth supporting in the future.
        assert not n0 <= -alpha
        return Expr(0)
    else:
        # |alpha| decreases, so n >= s*|alpha| still holds
        poly_rec_1 = bound_coeff_mono(Expr, alpha + 1, log_order, order, s, n0 - 1)
        poly_rec_2 = bound_coeff_mono(Expr, alpha + 1, log_order - 1, order, s, n0 - 1)
        #u = 1/(n-1)
        bound_error_u = CB(1 / (1 - 1/(n0 - 1)))
        truncated_u = (sum(CB(1) * invn**j for j in range(1, order+1))
                + CB(0).add_error(bound_error_u) * invn**(order+1))
        bound_error_logz = CB(abs(log(2) * 2**(order+1)) / (1 - 2/(n0 - 1)))
        truncated_logz = (logn
                - sum(CB(1) * invn**j / j
                      for j in range(1, order+1))
                + CB(0).add_error(bound_error_logz) * invn**(order+1))
        ss = (CB(alpha) * poly_rec_1.subs({invn : truncated_u, logz : truncated_logz})
            + CB(log_order) * poly_rec_2.subs({invn : truncated_u, logz : truncated_logz}))
        return truncate_tail(ss, order, n0, invn)

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
    'kappa',  # max power of log
    'bound',  # coefficient bound (explicit part + local error term)
    'initial_terms'  # explicit part of the local solution
])

SingularityData = collections.namedtuple('SingularityData', [
    'rho',  # the singularity
    'expo_group_data',  # as above, for each exponent group
    'min_val_rho',  # Re(valuation) of singular part (ignoring analytic terms)
])


class SingularityAnalyzer(LocalBasisMapper):

    def __init__(self, dop, inivec, *, rho, rad, Expr, abs_order, min_n,
                 struct):

        super().__init__(dop)

        self.inivec = inivec
        self.rho = rho
        self.rad = rad
        self.Expr = Expr
        self.abs_order = abs_order
        self.min_n = min_n
        self._local_basis_structure = struct

    def process_modZ_class(self):

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

        s = RBF(self.min_n) / (abs(self.leftmost) + abs(order))
        assert s > 2

        # (Bound on) Maximum power of log that may occur the sol defined by ini.
        # (We could use the complete family of critical monomials for a tighter
        # bound when order1 < max shift, since for now we currently are
        # computing it anyway...)
        assert self.shifts[0][0] == 0 and order1 > 0
        kappa = max(k for shift, mult in self.shifts if shift < order1
                      for k, c in enumerate(ser[shift]) if not c.is_zero())
        kappa += sum(mult for shift, mult in self.shifts if shift >= order1)

        bound_lead_terms, initial_terms = _bound_local_integral_explicit_terms(
                self.rho, self.leftmost, order, self.Expr, s, self.min_n, ser[:order])
        bound_int_SnLn = _bound_local_integral_of_tail(self.rho,
                self.leftmost, order, self.Expr, s, self.min_n, vb, kappa)

        logger.debug("sing=%s, valuation=%s", self.rho, self.leftmost)
        logger.debug("  leading terms = %s", bound_lead_terms)
        logger.debug("  tail bound = %s", bound_int_SnLn)

        data = ExponentGroupData(
            val = self.leftmost,
            kappa = kappa,
            bound = bound_lead_terms + bound_int_SnLn,
            initial_terms = initial_terms)

        # XXX abusing FundamentalSolution somewhat; not sure if log_power=kappa
        # is really appropriate; consider creating another type of record
        # compatible with FundamentalSolution if this stays
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

def _bound_local_integral_of_tail(rho, val_rho, order, Expr, s, min_n, vb, kappa):

    _, _, invn, logn = Expr.gens()

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
            * CB(1 - 1/min_n).pow(CB(-min_n-1)))
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

    return (bound_S + bound_L) * invn**order

def _bound_local_integral_explicit_terms(rho, val_rho, order, Expr, s, min_n, ser):

    _, _, invn, logn = Expr.gens()
    CB = Expr.base_ring()

    # Rewrite the local expansion in terms of new variables Z = z - ρ,
    # L = log(1/(1-z/rho))

    Z, L = PolynomialRing(CB, ['Z', 'L']).gens()
    mylog = CB.coerce(-rho).log() - L # = log(z - ρ) for Im(z) ≥ 0
    locf_ini_terms = sum(c/ZZ(k).factorial() * mylog**k * Z**shift
                         for shift, vec in enumerate(ser)
                         for k, c in enumerate(vec))

    bound_lead_terms = sum(
            c * CB(- rho).pow(CB(val_rho+degZ))
              * invn**(degZ)
              * bound_coeff_mono(Expr, -val_rho-degZ, degL, order - degZ,
                                  s, min_n)
            for ((degZ, degL), c) in locf_ini_terms.iterator_exp_coeff())

    return bound_lead_terms, locf_ini_terms

def contribution_single_singularity(deq, ini, rho, rad, Expr,
        rel_order, min_n):

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
            rad=rad, Expr=Expr, abs_order=abs_order, min_n=min_n,
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
    logger.info("Bounding on large circle...")
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
    logger.info("Covered circle with %d squares, %s", num_sq, clock)
    return pairs

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

def to_asymptotic_expansion(Coeff, name, term_data, n0):

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

    n_as_sym = SR.var(name)

    # XXX detect cases where we can use 1 or ±1 or U as Arg
    Exp, Arg = ExponentialGrowthGroup.factory(QQbar, name, return_factors=True)
    # AsymptoticRing does not split MonomialGrowthGroups with non-real
    # exponent groups in growth*non-growth parts, presumably because this has no
    # impact on term ordering. Let us do the same.
    Pow = MonomialGrowthGroup(QQbar, name)
    Log = MonomialGrowthGroup(ZZ, f"log({name})")
    Growth = cartesian_product([Arg, Exp, Pow, Log])
    # (n,) = Growth.gens_monomial()
    Asy = AsymptoticRing(Growth, coefficient_ring=Coeff)
    ET = Asy.term_monoid('exact')
    BT = Asy.term_monoid('B').change_parameter(
            coefficient_ring=Coeff._real_field())

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

    terms = []
    for rho, expr in term_data:
        dir = rho0/rho
        assert abs(dir).is_one() # need an additional growth factor otherwise
        arg_factor = make_arg_factor(dir)
        if expr.operator() == add_vararg:
            symterms = expr.operands()
        else:
            symterms = [expr]
        for symterm in symterms:
            term = arg_factor*ET(symterm.subs(n=n_as_sym))
            if term.coefficient.contains_zero():
                term = BT(term.growth, coefficient=term.coefficient.above_abs(),
                        valid_from={name: n0})
            terms.append(term)

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
            list_coeff.append(seqini[mon.n])
        else:
            list_coeff.append(0)
    return vector(list_coeff)


def _bound_validity_range(min_n, dominant_sing, order):

    # Make sure the disks B(ρ, |ρ|/n) contain no other singular point

    # FIXME: currently DOES NOT match [DMM, (10)]
    if len(dominant_sing) > 1:
        min_dist = min(s0.dist_to_sing() for s0 in dominant_sing)
        n1 = ceil(2*abs(dominant_sing[-1])/min_dist)
    else:
        n1 = 0

    # Make sure that min_n > 2*|α| for all exponents α we encounter

    max_abs_val = max(abs(sol.leftmost) # TODO: avoid redundant computation...
                      for s0 in dominant_sing
                      for sol in s0.local_basis_structure())
    n2 = max_abs_val + order + 1
    # FIXME: slightly different from [DMM, (46)]
    min_n = max(min_n, ceil(2.1*n2), n1)

    logger.debug(f"{n1=}, {n2=}, {min_n=}")
    return min_n


def bound_coefficients(deq, seqini, name='n', order=3, prec=53, n0=0, *,
                       known_analytic=[0], rad=None, halfside=None,
                       output='asymptotic_expansion'):
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

    # XXX split in v, logz and invn, logn?
    Expr = PolynomialRing(CB, ['v', 'logz', 'invn', 'logn'], order='lex')

    sing_data = [contribution_single_singularity(deq, ini, rho, rad, Expr,
                                                 order, n0)
                 for rho in dominant_sing]
    sing_data = [sdata for sdata in sing_data if sdata is not None]

    final_kappa = max(edata.kappa for sdata in sing_data
                                  for edata in sdata.expo_group_data)
    beta = - min(sdata.min_val_rho for sdata in sing_data) - 1 - order

    # Exponentially small error term

    if halfside is None:
        halfside = min(abs(abs(ex) - rad) for ex in all_exn_pts)/10
    logger.info("half-side of small squares: %s", halfside)

    pairs = numerical_sol_big_circle(deq, ini, dominant_sing, rad, halfside)
    covering, f_big_circle = zip(*pairs)

    sum_g = [
        sum((_z-_rho).pow(CB(edata.val))
                # some of the _z may lead to arguments of log that cross the
                # branch cut, but that's okay
                * edata.initial_terms(_z-_rho, (~(1-_z/_rho)).log())
            for sdata in sing_data for _rho in (CB(sdata.rho),)
            for edata in sdata.expo_group_data)
        for j, _z in enumerate(covering)]
    max_big_circle = RBF.zero().max(*((s - vv).above_abs()
                                     for s, vv in zip(sum_g, f_big_circle)))

    # Assemble the bounds

    _, _, invn, logn = Expr.gens()
    n = SR.var(name)
    bound = [
        [sdata.rho,
         sum(truncate_tail_SR(-edata.val-1, edata.bound, beta, n0,
                              invn, final_kappa, logn, n)
             for edata in sdata.expo_group_data)]
        for sdata in sing_data
    ]

    # Absorb exponentially small term in previous error term

    M = RBF(abs(dominant_sing[0]))
    if beta <= n0 * (M/rad).log():
        _beta = RBF(beta)
        cst = _beta.exp()*(_beta/(M/rad).log()).pow(_beta)
    else:
        cst = (M/CB(rad))**n0 * CB(n0)**(-beta)
    rad_err = cst*max_big_circle / CB(n0).log()**final_kappa
    error_term_big_circle = (CB(0).add_error(rad_err) *
                             SR(n**QQbar(beta)) *
                             SR(log(n))**final_kappa)
    mag_dom = abs(dominant_sing[0])

    for i, (rho, local_bound) in enumerate(bound):
        if rho == mag_dom:
            bound[i][1] += error_term_big_circle
            break
    else:
        bound.append[mag_dom, error_term_big_circle]

    if output == 'list':
        return n0, bound
    else:
        try:
            asy = to_asymptotic_expansion(CB, name, bound, n0)
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
