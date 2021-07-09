# coding: utf-8
r"""
Bounds on sequences by singularity analysis

Author: Ruiwen Dong <ruiwen.dong@polytechique.edu>

EXAMPLES::

    sage: from ore_algebra import OreAlgebra
    sage: from ore_algebra.analytic.singularity_analysis import contribution_all_singularity

    sage: Pols.<z> = PolynomialRing(QQ)
    sage: Diff.<Dz> = OreAlgebra(Pols)

Membrane example ::

    sage: seqini = [72, 1932, 31248, 790101/2, 17208645/4, 338898609/8, 1551478257/4]
    sage: deq = (8388593*z^2*(3*z^4 - 164*z^3 + 370*z^2 - 164*z + 3)*(z + 1)^2*(z^2 - 6*z + 1)^2*(z - 1)^3*Dz^3
    ....: + 8388593*z*(z + 1)*(z^2 - 6*z + 1)*(66*z^8 - 3943*z^7 + 18981*z^6 - 16759*z^5 - 30383*z^4 + 47123*z^3 - 17577*z^2 + 971*z - 15)*(z - 1)^2*Dz^2
    ....: + 16777186*(z - 1)*(210*z^12 - 13761*z^11 + 101088*z^10 - 178437*z^9 - 248334*z^8 + 930590*z^7 - 446064*z^6 - 694834*z^5 + 794998*z^4 - 267421*z^3 + 24144*z^2 - 649*z + 6)*Dz
    ....: + 6341776308*z^12 - 427012938072*z^11 + 2435594423178*z^10 - 2400915979716*z^9 - 10724094731502*z^8 + 26272536406048*z^7 - 8496738740956*z^6 - 30570113263064*z^5 + 39394376229112*z^4 - 19173572139496*z^3 + 3825886272626*z^2 - 170758199108*z + 2701126946)

    sage: desing_deq = deq.desingularize()
    sage: desing_deq.leading_coefficient().factor()
    (z - 1)^2 * z^2 * (z^2 - 6*z + 1)^2
    sage: b = contribution_all_singularity(seqini, deq, total_order = 2)
    Radius of large circle: 0.29354351163520964
    halfside of small squares: 0.012197063638139974
    Bounding on large circle:
    Covered circle with 40 squares
    ...

Algebraic example::

    sage: seqini = [1]
    sage: deq = (4*z^4 - 4*z^3 + z^2 - 2*z + 1)*Dz + (-4*z^3 + 4*z^2 - z - 1)

    sage: desing_deq = deq.desingularize()
    sage: desing_deq.leading_coefficient().factor()
    (4) * (z - 1) * (z - 1/2) * (z^2 + 1/2*z + 1/2)

    sage: b = contribution_all_singularity(seqini, deq, total_order = 5)
    ...

Diagonal example::

    sage: seqini = [1, -3, 9, -3, -279, 2997]
    sage: deq = z^2*(81*z^2 + 14*z + 1)*Dz^3 + 3*z*(162*z^2 + 21*z + 1)*Dz^2 + (21*z + 1)*(27*z + 1)*Dz + 3*(27*z + 1)

    sage: desing_deq = deq.desingularize()
    sage: desing_deq.leading_coefficient().factor()
    (81) * z^2 * (z^2 + 14/81*z + 1/81)
    sage: b = contribution_all_singularity(seqini, deq, total_order = 2)
    ...

Complex exponents example::

    sage: seqini = [11, 12]
    sage: deq = (z-2)^2 * Dz^2 + z*(z-2) * Dz + 1
    sage: #deq.local_basis_expansions(2)
    sage: b = contribution_all_singularity(seqini, deq, total_order = 4)
    ...
"""

from sage.all import *

from ..ore_algebra import OreAlgebra
from .differential_operator import DifferentialOperator
from .path import Point
from .bounds import DiffOpBound
from .ui import multi_eval_diffeq
from .utilities import as_embedded_number_field_elements

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
    Enz_coeff = (2*abs(bernoulli(2*n+2)) * (m + 2*n + 2)**(m + 2*n + 2)
            / (2*n + 1)**(2*n + 1) / (m + 1)**(m + 1)
            * gamma(m+2) / (2*n + 1) / (2*n + 2))
    if m == 0:
        return R(logz - v / 2
                - sum(bernoulli(2*k)*v**(2*k)/(2*k) for k in range(1,n+1))
                + CBF(0).add_error(Enz_coeff)*v**(2*n+2))
    else:
        return R((-1)**(m+1) * (gamma(m) * v**m + gamma(m+1) * v**(m+1) / 2
            + sum(bernoulli(2*k)*v**(2*k+m)*rising_factorial(2*k+1, m-1)
                  for k in range(1,n+1)))
            + CBF(0).add_error(Enz_coeff)*v**(2*n+m+2))

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

    a polynomial in CBF[v] such that [(d/dα)^l (Γ(n+α)/Γ(α))] / (Γ(n+α)/Γ(α)) is
    in its range when n >= max(s*|alpha|, min_n)
    """
    list_f = []
    R = v.parent()
    for m in range(l):
        n = max(0, ceil((order - m - 1)/2))
        Enz_coeff = (abs(bernoulli(2*n+2)) * (m + 2*n + 2)**(m + 2*n + 2)
                / (2*n + 1)**(2*n + 1) / (m + 1)**(m + 1)
                * gamma(m+2) / (2*n + 1) / (2*n + 2))
        list_f.append(truncated_psi(n, m, v, logz) - CBF(alpha).psi(m))
        if not min_n is None:
            list_f[-1] = truncate_tail(list_f[-1], order+1, min_n + alpha, v)
    p = der_expf(l)
    if not min_n is None:
        return R(1) if l == 0 else truncate_tail(p(list_f), order+1, min_n + alpha, v)
    else:
        return p(list_f)

def truncated_ratio_gamma(alpha, order, u, s):
    """
    Find a truncated expression with error bound for Γ(n+α)/Γ(n+1)/(n+α/2)^(α-1)

    INPUT:

    - alpha : complex number α, !!cannot be negative integer or zero!!
    - order : order of truncation
    - u : element of polynomial ring, representing 1/(n+α/2)
    - s : positive number where n >= s*|alpha| is guaranteed, s > 2

    OUTPUT:

    - ratio_gamma : a polynomial in CBF[u] such that Γ(n+α)/Γ(n+1)/(n+α/2)^(α-1)
      is in its range when n >= s*|alpha|
    """
    rho = CBF(alpha/2)
    n_gam = ceil((1+order)/2)
    #Compute B_2l^2r(r) where B_n^m(r) is the generalized Bernoulli polynomial
    t = polygen(CBF, 't')
    foo = (t._exp_series(2*n_gam + 1) >> 1)
    foo = -2*rho*foo._log_series(2*n_gam)
    foo = (foo >> 2) << 2
    foo = foo._exp_series(2*n_gam)
    series_gen_bern = [c*ZZ(n).factorial() for n, c in enumerate(foo)]

    #Compute B_2l^2r(|r|) where B_n^m(|r|) is the generalized Bernoulli polynomial
    foo = (t._exp_series(2*n_gam + 2) >> 1)
    foo = -2*abs(rho)*foo._log_series(2*n_gam+1)
    foo = (foo >> 2) << 2
    foo = foo._exp_series(2*n_gam+1)
    series_gen_bern_abs = [c*ZZ(n).factorial() for n, c in enumerate(foo)]

    vd = var('vd') #We should change this
    ratio_gamma_asy = sum(
            CBF(binomial(2*j-2*vd, 2*j).subs({vd : rho})
                * series_gen_bern[2*j]) * u**(2*j)
            for j in range(n_gam))
    Rnw_bound = (CBF(binomial(2*n_gam-2*vd, 2*n_gam).subs({vd : rho.real()}))
            * CBF(abs(series_gen_bern_abs[2*n_gam]))
            * CBF(abs(alpha.imag())/2/s).exp()
            * CBF((s+1/2)/(s-1/2)).pow(max(0, -2*rho.real()+1+2*n_gam)))
    ratio_gamma = ratio_gamma_asy + CBF(0).add_error(Rnw_bound) * u**(2*n_gam)
    return ratio_gamma

def truncated_power(alpha, order, w, s):
    """
    Compute a bound for a truncated (1 + α/2n)^(α-1)

    INPUT:

    - alpha : complex number α, !!cannot be negative integer or zero!!
    - order : order of truncation
    - w : element of polynomial ring, representing 1/n
    - s : positive number where n >= s*|alpha| is guaranteed, s > 2

    OUTPUT:

    - trunc_power : a polynomial in CBF[w] such that (1 + α/2n)^(α-1) is in its
      range when n >= s*|alpha|
    """
    alpha_CBF = CBF(alpha)
    Mr = (CBF(abs(alpha)).pow(order+1) / (1 - 1/s)
            * CBF(abs(alpha.imag())/2).exp()
            * (CBF(3/2).pow(alpha.real() - 1) if alpha.real() - 1 > 0
               else CBF(1/2).pow(alpha.real() - 1)))
    t = polygen(CBF, 't')
    foo = (alpha_CBF - 1) * (1 + alpha_CBF/2 * t)._log_series(order + 1)
    foo = foo._exp_series(order + 1)
    trunc_power = foo(w) + CBF(0).add_error(Mr) * w**(order+1)
    return trunc_power

def truncate_tail(f, deg, min_n, w, kappa = None, logn = None):
    """
    Truncate and bound an expression f(1/n) to a given degree
    1/n^(deg+t) will be truncated to 1/n^deg * CBF(0).add_error(1/min_n^t)

    INPUT:

    - f : polynomial in w = 1/n to be truncated
    - deg : desired degree of polynomial after truncation
    - min_n : positive number where n >= min_n is guaranteed
    - w : element of polynomial ring, representing 1/n

    OUTPUT:

    - g : a polynomial in CBF[w] such that f is in its range when n >= min_n
    """
    R = f.parent()
    g = R(0)
    if kappa is None:
        for c, mon in f:
            deg_w = mon.degree(w)
            if deg_w > deg:
                tuple_mon_g = tuple(map(lambda x, y: x - y, mon.exponents()[0],
                                        (w**(deg_w - deg)).exponents()[0]))
                mon_g = prod(R.gens()[j]**(tuple_mon_g[j])
                             for j in range(len(tuple_mon_g)))
                c_g = ((c if c.mid() == 0 else CBF(0).add_error(c.above_abs()))
                        / CBF(min_n**(deg_w - deg)))
                g = g + c_g * mon_g
            else:
                g = g + c*mon
    else:
        for c, mon in f:
            deg_w = mon.degree(w)
            deg_logn = mon.degree(logn)
            if deg_w >= deg:
                tuple_mon_g = tuple(map(
                    lambda x, y, z: x - y + z, mon.exponents()[0],
                    (w**(deg_w - deg)).exponents()[0],
                    (logn**(kappa - deg_logn)).exponents()[0]))
                mon_g = prod(R.gens()[j]**(tuple_mon_g[j])
                             for j in range(len(tuple_mon_g)))
                c_g = ((c if c.mid() == 0 else CBF(0).add_error(c.above_abs()))
                        / CBF(min_n**(deg_w - deg))
                        * CBF(min_n).log().pow(deg_logn - kappa))
                g = g + c_g * mon_g
            else:
                g = g + c*mon
    return g

def bound_coeff_mono(alpha, l, deg, w, logn, s=5, min_n=50):
    """
    Compute a bound for [z^n] (1-z)^(-α) * log(1/(1-z))^l,
    of the form n^(α-1) * P(1/n, log(n))

    INPUT:

    - alpha : complex number, representing α
    - l : non-negative integer
    - deg : degree of P wrt. the variable 1/n
    - w : variable representing 1/n
    - logn : element of the same polynomial ring as w, variable representing
      log(n)
    - s : positive number where n >= s*|alpha| is guaranteed, s > 2
    - min_n : positive number where n >= min_n is guaranteed, min_n > -alpha
      needed

    OUTPUT:

    - P : polynomial in w, logn
    """
    R = w.parent()
    v, logz, u, _, _ = R.gens()
    order = max(0, deg - 1)
    if not (QQbar(alpha).is_integer() and QQbar(alpha) <= 0):
        # Value of 1/Γ(α)
        c = CBF(1/gamma(alpha))
        # Bound for (n+α/2)^(1-α) * Γ(n+α)/Γ(n+1)
        f = truncated_ratio_gamma(alpha, order, u, s)
        bound_error_u = CBF(abs(alpha/2)**order / (1 - 1/(2*s)))
        truncated_u = (sum(CBF(-alpha/2)**(j-1) * w**j for j in range(1, order+1))
                + CBF(0).add_error(bound_error_u) * w**(order+1))
        f_z = truncate_tail(f.subs({u : truncated_u}), deg, min_n, w)
        # Bound for [(d/dα)^l (Γ(n+α)/Γ(α)Γ(n+1))] / (Γ(n+α)/Γ(α)Γ(n+1))
        g = truncated_logder(alpha, l, order, v, logz, min_n)
        bound_error_v = CBF(abs(alpha)**order / (1 - 1/s))
        truncated_v = (sum(CBF(-alpha)**(j-1) * w**j for j in range(1, order+1))
                + CBF(0).add_error(bound_error_v) * w**(order+1))
        bound_error_logz = (CBF(log(2)+1/2)
                * CBF(2*abs(alpha)).pow(order+1)
                / CBF(1 - 2/s))
        truncated_logz = (logn
                - sum(CBF(-alpha)**j * w**j / j
                      for j in range(1, order+1))
                + CBF(0).add_error(bound_error_logz) * w**(order+1))
        g_z = truncate_tail(
                g.subs({v : truncated_v, logz : truncated_logz}),
                deg, min_n, w)
        # Bound for (1 + α/2n)^(α-1) = (n+α/2)^(α-1) * n^(1-α)
        h_z = truncated_power(alpha, order, w, s)
        product_all = c * f_z * g_z * h_z
        return truncate_tail(product_all, deg, min_n, w)
    elif l == 0:
        assert min_n > -int(alpha), "min_n too small!"
        return R(0)
    else:
        # |alpha| decreases, so n >= s*|alpha| still holds
        poly_rec_1 = bound_coeff_mono(alpha + 1, l, deg, u, logz, s, min_n - 1)
        poly_rec_2 = bound_coeff_mono(alpha + 1, l - 1, deg, u, logz, s, min_n - 1)
        #u = 1/(n-1)
        bound_error_u = CBF(1 / (1 - 1/(min_n - 1)))
        truncated_u = (sum(CBF(1) * w**j for j in range(1, order+1))
                + CBF(0).add_error(bound_error_u) * w**(order+1))
        bound_error_logz = CBF(abs(log(2) * 2**(order+1)) / (1 - 2/(min_n - 1)))
        truncated_logz = (logn
                - sum(CBF(1) * w**j / j
                      for j in range(1, order+1))
                + CBF(0).add_error(bound_error_logz) * w**(order+1))
        ss = (CBF(alpha) * poly_rec_1.subs({u : truncated_u, logz : truncated_logz})
            + CBF(l) * poly_rec_2.subs({u : truncated_u, logz : truncated_logz}))
        return truncate_tail(ss, deg, min_n, w)

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

def valuation_FS(fs):
    """
    Valuation of a formal series of logmonomials
    """
    v = min([mon.shift for c, mon in fs if not c==0]) + list(fs)[0][1].expo
    return v

def minshift_FS(fs):
    """
    Valuation of a formal series of logmonomials
    """
    v = min([mon.shift for c, mon in fs if not c==0])
    return v

def extract_local(fs, rho, order=1):
    """
    Extract the local terms of a formal series of logmonomials, return a
    polynomial in L=logz, Z=z

    INPUT:

    - fs : formal series
    - rho : algebraic number
    - order : integer

    OUTPUT:

    - L : element of polynomial ring, denoting log(z)
    - Z : element of same polynomial ring, denoting z
    - loc : element of same polynomial ring, local truncation (after variable
      change) of fs
    """
    L, Z = PolynomialRing(CBF, ['L', 'Z']).gens()
    mylog = CBF(-rho).log() - L
    ms = minshift_FS(fs)
    loc = sum(c * mylog**mon.k * Z**mon.shift
              for c, mon in fs
              if mon.shift < ms + order)
    return L, Z, loc

def numerical_sol_big_circle(coeff_zero, deq, list_dom_sing, rad, halfside):
    """
    Compute numerical solutions of f on big circle of radius rad

    INPUT:

    - coeff_zero : vector, coefficients corresponding to the basis at zero
    - deq : a linear ODE that the generating function satisfies
    - list_dom_sing : list of algebraic numbers, list of dominant singularities
    - rad : radius of big circle
    - halfside : half of side length of covering squares
    """
    import time
    print("Bounding on large circle: ")
    begin_time = time.time()

    sings = list_dom_sing.copy()
    sings.sort(key=lambda s: arg(s))
    num_sings = len(sings)
    C = ComplexBallField(200)
    R = RealBallField(200)
    pairs = []
    num_sq = 0
    for j in range(num_sings - 1):
        hub = C(rad) * ((C(sings[j]).arg() + C(sings[j+1]).arg())/2 * C(I)).exp()
        halfarc = (C(sings[j+1]).arg() - C(sings[j]).arg()) / 2
        np = ceil(float((halfarc*C(rad) / R(2*halfside)).above_abs())) + 2
        num_sq = num_sq + np
        circle_upper = [(hub*C(halfarc*k/np*I).exp()).add_error(halfside)
                        for k in range(np+1)]
        circle_lower = [(hub*C(-halfarc*k/np*I).exp()).add_error(halfside)
                        for k in range(np+1)]
        path_upper = [0] + [[_z] for _z in circle_upper]
        path_lower = [0] + [[_z] for _z in circle_lower]
        pairs += deq.numerical_solution(coeff_zero, path_upper, 1e-30,
                                        assume_analytic=True)
        pairs += deq.numerical_solution(coeff_zero, path_lower, 1e-30,
                                        assume_analytic=True)
    # last arc is a bit special: we need to add 2*pi to ending
    j = num_sings - 1
    hub = C(rad) * ((C(sings[j]).arg() + C(sings[0]).arg() + C(2*pi))/2 * C(I)).exp()
    halfarc = (C(sings[0]).arg() + R(2*pi) - C(sings[j]).arg()) / 2
    np = ceil(float((halfarc*C(rad) / R(2*halfside)).above_abs())) + 2
    num_sq = num_sq + np
    circle_upper = [(hub*C(halfarc*k/np*I).exp()).add_error(halfside)
                    for k in range(np+1)]
    circle_lower = [(hub*C(-halfarc*k/np*I).exp()).add_error(halfside)
                    for k in range(np+1)]
    path_upper = [0] + [[_z] for _z in circle_upper]
    path_lower = [0] + [[_z] for _z in circle_lower]
    pairs += deq.numerical_solution(coeff_zero, path_upper, 1e-30,
            assume_analytic=True)
    pairs += deq.numerical_solution(coeff_zero, path_lower, 1e-30,
            assume_analytic=True)

    end_time = time.time()
    print("Covered circle with %d squares\n" % num_sq)
    print("%03.3f seconds \n" % (end_time - begin_time))
    return pairs

def contribution_single_singularity(coeff_zero, deq, rho, rad_input,
        coord_big_circle, total_order=1, min_n=50):
    """
    Compute a lower bound of a dominant singularity's contribution to f_n

    INPUT:

    - coeff_zero : vector, coefficients corresponding to the basis at zero
    - deq : a linear ODE that the generating function satisfies
    - rho : singularity
    - rad_input : real number, such that rho is the only singularity in B(0, R)
    - coord_big_circle : coordinates of the "big circle" of radius rad_input
    - total_order : positive integer
    - min_n : positive integer, n > min_n
    OUTPUT:
    - list_val : list of valuation of solutions at rho
    - list_bound : list of expressions of bound
    - val_big_circle : values on big circle
    - max_kappa
    """

    z = deq.parent().base_ring().gens()[0]

    rad = RBF(rad_input)

    loc = deq.local_basis_expansions(rho)

    coord_all = deq.numerical_transition_matrix([0, rho], 1e-20,
                                              assume_analytic=True) * coeff_zero

    # Regroup elements of the loc basis according to valuation modulo ZZ
    list_expo = [list(f)[0][1].expo for f in loc]
    list_expo_unique = list(set(list_expo))
    list_ind = [[ind for ind, x in enumerate(list_expo) if x == expo]
                for expo in list_expo_unique]

    # Initialize results and bounds for different basis
    list_val_rho = []
    list_val = []
    list_coord = []
    list_locf_long = []
    list_bound = []
    list_kappa = []
    val_big_circle = [CBF(0)] * len(coord_big_circle)

    v, logz, u, w, logn = PolynomialRing(CBF,
            ["v", "logz", "u", "w", "logn"], order='lex').gens()

    import numpy
    import time

    for ind_basis in list_ind:
        # We consider only elements of the basis with the same expo
        coord = [0]*len(coord_all)
        for j in ind_basis:
            coord[j] = coord_all[j]
        coord = vector(coord)

        # Local expansion of f at z=rho, in terms of variables Z and L
        locf_long = sum([c*ser for c, ser in zip(coord, loc)])
        list_val_rho.append(valuation_FS(locf_long))
        list_coord.append(coord)
        list_locf_long.append(locf_long)

    list_val_real = [v.real() for v in list_val_rho]
    min_val_rho = min(list_val_real)
    num_bas = 0

    for ind_basis in list_ind:

        val_rho = list_val_rho[num_bas]
        coord = list_coord[num_bas]
        locf_long = list_locf_long[num_bas]

        order = max(0, ceil(total_order - (val_rho.real() - min_val_rho)))

        num_bas = num_bas + 1
        print("Computing basis %d \n" % num_bas)
        cycle_begin_time = time.time()

        # Local expansion of f at z=rho, in terms of variables Z and L
        L, Z, locf_ini_terms = extract_local(locf_long, rho, order)

        #> before going further, you could kill any local solution known to be
        # analytic at rho

        # Maximum power of log(z - rho) that can appear in a solution
        K, rho1, _ = rho.as_number_field_element()
        K_ext = K.extension(QQbar(val_rho).minpoly(), 'val_rho1')
        val_rho1 = K_ext.gen()
        rho1 = K_ext(rho1)
        ind_poly = deq.change_ring(K_ext[z]).indicial_polynomial(z - rho1)
        alpha = ind_poly.variables()[0]
        expo = ind_poly.subs(alpha = alpha + val_rho1).roots()
        kappa = ZZ(sum(mult for r, mult in expo if r.is_integer()) - 1)

        list_kappa.append(kappa)

        list_val.append(val_rho)

        #small radius
        smallrad = rad - CBF(rho).below_abs()

        # Convert the equation to an enriched data structure necessary for
        # calling some of the bound computation routines, and shift to the
        # origin.

        # Make a copy of deq with a base field including val_rho
        K1, list_val_rho1 = as_embedded_number_field_elements([val_rho])
        val_rho1 = list_val_rho1[0]
        deq1 = deq.change_ring(K1[z])
        ldop = DifferentialOperator(deq1).shift(Point(rho, deq1))
        maj = DiffOpBound(ldop, leftmost=val_rho1, pol_part_len=30,
                                                          bound_inverse="solve")

        # We want to bound the series h0, h1, ..., h_kappa on small disk
        # |z - ρ| < smallrad.
        # The bounds that ore_algebra computes are too coarse, so we bound the
        # tails corresponding to a truncation at index order1 > order, and
        # handle the intermediate terms separately.
        # Due to limitations of the implementation of remainder bounds, order_0
        # must be >= maj.dop.degree() for what follows to work.

        order1 = 49 + order #TODO: Maybe optimize this
        loc1 = deq1.local_basis_expansions(rho, order1)
        locf1 = sum(c*ser for c, ser in zip(coord, loc1))

        # The last few coefficients of the local expansion of f will be used
        # to compute a residual associated to that particular solution.
        coeff = { (mon.shift, mon.k): c*ZZ(mon.k).factorial()
                  for c, mon in locf1 } #(z^shift*log^k)
        last = list(reversed([[coeff.get((shift, k), 0) for k in range(kappa+1)]
                          for shift in range(order1)]))

        # Coefficients of the normalized residual in the sense of [Mez19, Sec. 6.3],
        # with the indexing conventions of [Mez19, Prop. 6.10]

        res = maj.normalized_residual(order1, last, Ring=coord[0].parent())

        # Majorant series of [the components of] the tail of the local expansion
        # of f at ρ. See [Mez19, Sec. 4.3] and [Mez19, Algo. 6.11].
        #
        # Roughly speaking, the tail satisfies the inhomogeneous equation
        # ldop(tail) = -rhs with rhs = ldop(locf1), and is therefore bounded by
        # solution of MAJ(Y) = Q for a suitably chosen Q. We solve the latter
        # equation by variation of constants to obtain an explicity bound Y.
        tmaj = maj.tail_majorant(order1, [res])
        # Make a second copy of the bound before we modify it in place.
        tmaj1 = maj.tail_majorant(order1, [res])
        # Shift it (= factor out (z-ρ)^order) ==> majorant series of the tails
        # of the coefficients of log(z)^k/k!, i.e., of h0, h1, and 2*h2
        tmaj1 >>= -order

        # Bound on the tails, valid for all |z| ≤ smallrad
        tb = tmaj1.bound(smallrad)

        # Bound on the intermediate terms
        ib = sum(smallrad**(n-order) *
                 max(coeff.get((n,k), RBF(0)).above_abs()
                     for k in range(kappa + 1))
                 for n in range(order, order1))
        # Bound on the *values* for |z-ρ| <= smallrad of the functions
        # h0, h1, 2*h2.
        vb = tb + ib

        # Change representation from log(z-ρ) to log(1/(1 - z/ρ))
        # The h_i are cofactors of powers of log(z-ρ), not log(1/(1-z/ρ)).
        # Define the B polynomial in a way that accounts for that.
        ll = abs(CBF(-rho).log())
        B = vb*RBF['z']([
                sum([ll**(m - j) * binomial(m, j) / factorial(m)
                     for m in range(j, kappa + 1)])
                for j in range(kappa + 1)])

        s = floor(min_n / (abs(val_rho) + abs(order)))
        assert s>2, "min_n too small! Cannot guarantee s>2"

        # Sub polynomial factor for bound on S(n)
        cst_S = (CBF(0).add_error(CBF(abs(rho)).pow(val_rho.real()+order)
            * ((abs(CBF(rho).arg()) + 2*RBF(pi))*abs(val_rho.imag())).exp()
            * CBF(1 - 1/min_n).pow(CBF(-min_n-1))))
        bound_S = cst_S*B(CBF(pi)+logn)
        # Sub polynomial factor for bound on L(n)
        if val_rho + order <= 0:
            C_nur = 1
        else:
            C_nur = 2 * (CBF(e) / (CBF(val_rho.real()) + order)
                             * (s - 2)/(2*s)).pow((RBF(val_rho.real()) + order))
        cst_L = (CBF(0).add_error(C_nur * CBF(1/pi)
                                 * CBF(abs(rho)).pow(RBF(val_rho.real())+order))
            * ((abs(CBF(rho).arg()) + 2*RBF(pi))*abs(val_rho.imag())).exp())
        bound_L = cst_L*B(CBF(pi)+logn)

        # Values of the tail of the local expansion.
        # With the first square (and possibly some others), the argument of the
        # log that we substitute for L crosses the branch cut. This is okay
        # because the enclosure returned by Arb takes both branches into
        # account.

        _zeta = CBF(rho)
        dom_big_circle = [
                (_z-_zeta).pow(CBF(val_rho))
                    * locf_ini_terms((~(1-_z/_zeta)).log(), _z-_zeta)
                for _z in coord_big_circle]
        val_big_circle = [val_big_circle[j] + dom_big_circle[j]
                          for j in range(len(dom_big_circle))]

        list_coef_deg = [(c, mon.degree(L), mon.degree(Z))
                         for c, mon in list(locf_ini_terms)]

        bound_lead_terms = sum(
                tup[0]
                    * CBF(- rho).pow(CBF(val_rho+tup[2]))
                    * w**(tup[2])
                    * bound_coeff_mono(-val_rho-tup[2], tup[1], order - tup[2],
                                       w, logn, s, min_n)
                for tup in list_coef_deg)
        bound_int_SnLn = (bound_S + bound_L) * w**order

        list_bound.append(bound_lead_terms + bound_int_SnLn)

        cycle_end_time = time.time()
        print("Computing of basis %d finished, time: %9.2f" %
                (num_bas, cycle_end_time - cycle_begin_time))

    max_kappa = max(list_kappa)
    return list_val, list_bound, val_big_circle, max_kappa

def contribution_all_singularity(seqini, deq, singularities=None,
        known_analytic=[0], rad=None, total_order=1, min_n=50, halfside=None):
    """
    Compute a bound for the n-th element of a holonomic sequence

    INPUT:

    - seqini : list, initial elements of sequence, long enough to determine the
      sequence
    - deq : a linear ODE that the generating function satisfies
    - singularities : list of algebraic numbers, dominant singularities. If
      None, compute automatically
    - known_analytic : list of points where the generating function is known to
      be analytic, default is [0]
    - rad : radius of the big circle R_0. If None, compute automatically
    - total_order : integer, order to which the bound is computed
    - min_n : integer, bound is valid when n > min_n

    OUTPUT:

    - bound : symbolic expression in variable 'n', a bound for the n-th element
      of a holonomic sequence
    """
    coeff_zero = _coeff_zero(seqini, deq)

    if singularities is None:
        desing_deq = deq.desingularize()
        list_sing = desing_deq.leading_coefficient().roots(QQbar,
                                                           multiplicities=False)
    else:
        list_sing = singularities.copy()

    list_exception = deq.leading_coefficient().roots(QQbar,
                                                     multiplicities=False) + [0]

    #Exclude points known to be analytic
    list_sing = [s for s in list_sing if not s in known_analytic]

    list_sing.sort(key=lambda s: abs(s))

    if rad is None:
        #Find all dominant singularities
        sing_inf = abs(list_sing[-1])*3
        list_sing.append(sing_inf)
        k = next(j for j,v in enumerate(list_sing) if abs(v) > abs(list_sing[0]))
        list_dom_sing = list_sing[:k]
        max_smallrad = min(
                min(abs(ex - ds)
                    for ex in list_exception
                    if not (ex - ds) == 0)
                for ds in list_dom_sing)
        # This should probably change to avoid smaller fake singularities
        rad_input = min(
                abs(list_sing[k])*0.9 + abs(list_sing[0])*0.1,
                abs(list_sing[0]) + max_smallrad*0.8)
        print("Radius of large circle:", rad_input)
    else:
        sing_inf = abs(list_sing[-1])*2 + rad * 2
        list_sing.append(sing_inf)
        k = next(j for j,v in enumerate(list_sing) if abs(v) > rad)
        assert not k == 0, "No singularity contained in given radius"
        assert not abs(list_sing[k-1]) == rad, "A singularity is on the given radius"
        list_dom_sing = list_sing[:k]
        rad_input = rad

    #Make sure the disks B(ρ, |ρ|/n) do not touch each other
    if len(list_dom_sing) > 1:
        min_dist = min(
                min(abs(ex - ds)
                    for ex in list_dom_sing
                    if not (ex - ds) == 0)
                for ds in list_dom_sing)
        assert min_n > ceil(2*abs(list_dom_sing[-1])/min_dist), (
                "Please increase min_n to at least "
                +str(ceil(2*abs(list_dom_sing[-1])/min_dist)))

    #Automatically choose halfside
    if halfside is None:
        halfside = min(abs(abs(ex) - rad_input) for ex in list_exception)/10
        print("halfside of small squares:", halfside)

    pairs = numerical_sol_big_circle(coeff_zero, deq, list_dom_sing, rad_input,
                                                                       halfside)
    coord_big_circle = [z for z, _ in pairs]
    f_big_circle = [f for _, f in pairs]
    max_big_circle = RBF(0)

    #A.<n> = AsymptoticRing(growth_group='n^QQbar * log(n)^ZZ * QQbar^n', coefficient_ring=CBF)
    #bound = A(0)
    n = SR.var('n')
    bound = 0
    list_val_bigcirc = []
    list_data = []
    list_max_kappa = []

    for rho in list_dom_sing:
        list_val, list_bound, val_big_circle, max_kappa = contribution_single_singularity(
                coeff_zero, deq, rho, rad_input, coord_big_circle, total_order,
                min_n)
        list_data.append((rho, list_val, list_bound))
        list_max_kappa.append(max_kappa)
        list_val_bigcirc.append(val_big_circle)

    final_kappa = max(list_max_kappa)
    v, logz, u, w, logn = list_data[0][2][0].parent().gens()

    for rho, list_val, list_bound in list_data:
        bound += sum([
            SR(QQbar(1/rho)**n) * SR(n**QQbar(-val-1))
                * truncate_tail(poly_bound, poly_bound.degree(w), min_n, w,
                                          final_kappa, logn)(0,0,0, 1/n, log(n))
            for val, poly_bound in zip(list_val, list_bound)])

    sum_g = [sum(v) for v in zip(*list_val_bigcirc)]
    for v in list_val_bigcirc:
        max_big_circle = max_big_circle.max(*(
            (s - vv).above_abs()
            for s, vv in zip(sum_g, v)))
    max_big_circle = max_big_circle.max(*(
        (s - vv).above_abs()
        for s, vv in zip(sum_g, f_big_circle)))
    #Simplify bound contributed by big circle
    M = RBF(abs(list_dom_sing[0]))
    re_gam = max([-val.real()-1 for val in list_val])
    rad_err = max_big_circle * (((CBF(e) * (total_order - re_gam)
                    / (M/CBF(rad_input)).log()).pow(RBF(re_gam - total_order))
                    / CBF(min_n).log().pow(final_kappa))
                if re_gam <= total_order + min_n * (M/RBF(rad_input)).log()
                else ((M/CBF(rad_input)).pow(min_n)
                    * CBF(min_n).pow(total_order - re_gam)
                    / CBF(min_n).log().pow(final_kappa)))
    bound += CBF(0).add_error(rad_err) * (SR(QQbar(1/abs(list_dom_sing[0]))**n)
                                          * SR(n**QQbar(re_gam))
                                          * SR(1/n)**total_order
                                          * SR(log(n))**final_kappa)
    return bound
