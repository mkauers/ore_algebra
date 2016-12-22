# -*- coding: utf-8 - vim: tw=80
"""
Rigorous approximation of D-finite functions by polynomials
"""

# TODO:
# - currently we can compute the first few derivatives all at once, but there is
#   no support for computing a remote derivative using Dx^k mod dop (nor an
#   arbitrary combination of derivatives)
# - support returning polynomials with point interval or non-interval
#   coefficients

import logging
logger = logging.getLogger(__name__)

import sage.rings.real_arb
import sage.rings.complex_arb

from . import accuracy, analytic_continuation as ancont, bounds, utilities

from .naive_sum import series_sum, EvaluationPoint
from .safe_cmp import *

def combine_radii(pol):
    coeff = list(pol)
    for i in xrange(1, len(coeff)):
        coeff[0] = coeff[0].add_error(coeff[i].rad())
        coeff[i] = coeff[i].squash()
    return pol.parent()(coeff)

def taylor_economization(pol, eps):
    r"""
    Remove terms from the polynomial ``pol``, starting with the high-order
    terms, in such a way that its value on the disk |z| < 1 changes at most
    by ``eps``.

    A bound on the difference between the result and the input polynomial is
    added to the constant term, so that, for any complex number z with |z| < 1,
    the value of the result at z contains that of ``pol``.

    EXAMPLES::

        sage: from ore_algebra.analytic.polynomial_approximation import taylor_economization
        sage: pol = polygen(QQ, 'x')._exp_series(10).change_ring(RBF); pol
        [2.755...e-6 +/- 5.96e-22]*x^9 + [2.480...e-5 +/- 4.96e-21]*x^8
        + [0.0001... +/- 3.97e-20]*x^7 + [0.0013... +/- 4.92e-19]*x^6 + ... +
        x + 1.000000000000000
        sage: newpol = taylor_economization(pol, RBF(1e-3)); newpol
        ([0.0013... +/- 4.92e-19])*x^6 + ... + x +
        [1.000 +/- 2.26e-4] + [+/- 2.26e-4]*I

    TESTS::

        sage: from ore_algebra.analytic.polynomial_approximation import _test_fun_approx
        sage: _test_fun_approx(newpol, pol, disk_rad=1)

        sage: Pols.<x> = RBF[]
        sage: taylor_economization(x^5 + 10*x^4 + x^3 + 2*x + 10, RBF(3))
        10.0000...*x^4 + 2.0000...*x + [1e+1 +/- 2.01] + [+/- 2.01]*I
    """
    Coefs = pol.base_ring()
    coef = list(pol)
    delta_bound = eps.parent().zero()
    zero = Coefs.zero()
    for i in xrange(pol.degree(), -1, -1):
        tmp_bound = delta_bound + abs(pol[i])
        if safe_lt(tmp_bound, eps):
            delta_bound = tmp_bound
            coef[i] = zero
    coef[0] = Coefs.complex_field()(coef[0]).add_error(delta_bound)
    CPol = pol.parent().change_ring(Coefs.complex_field())
    return CPol(coef)

def chebyshev_polynomials(ring, n):
    r"""
    Return the list of the Chebyshev polynomials T[0], ..., T[``n``-1] as
    elements of ``ring``.

    EXAMPLES::

        sage: from ore_algebra.analytic.polynomial_approximation import chebyshev_polynomials
        sage: chebyshev_polynomials(ZZ['x'], 3)
        [1, x, 2*x^2 - 1]

    TESTS::

        sage: chebyshev_polynomials(QQ['x'], 0)
        []
    """
    x = ring.gen()
    cheb_T = [None]*n
    if n >= 1:
        cheb_T[0] = ring.one()
    if n >= 2:
        cheb_T[1] = x
    for k in xrange(1, n - 1):
        cheb_T[k+1] = 2*x*cheb_T[k] - cheb_T[k-1]
    return cheb_T

def general_economization(economization_polynomials, pol, eps):
    r"""
    Decrease the degree of ``pol`` by subtracting a linear combination of the
    polynomials produced by ``economization_polynomials``.

    INPUT:

    - ``economization_polynomials(ring, n)`` - function returning a list
      [E[0], ..., E[n-1]] of ``n`` elements of ``ring``; it is assumed that E[k]
      is a polynomial of degree exactly ``k`` such that |E[k](x)| ≤ 1 when x
      lies in some domain of interest;

    - ``pol`` - polynomial with real or complex ball coefficients;

    - ``eps`` - real ball, maximum error that may be added to the polynomial
      (see below).

    OUTPUT:

    A polynomial with the same parent as ``pol``. A bound on the linear
    combination of economization polynomials is added to the constant term.
    For *real* x in the domain where they are all bounded by 1, the value of the
    result at x contains that of ``pol``. This also holds true for complex x
    when ``pol`` has complex coefficients, but *not* in general for the
    evaluation at complex points of polynomials with real coefficients.

    TESTS::

        sage: def monomials(ring, n):
        ....:     x = ring.gen()
        ....:     return [x**k for k in range(n)]
        sage: Pols.<x> = RBF[]
        sage: general_economization(monomials,
        ....:         x^5 + 10*x^4 + x^3 + 2*x + 10, RBF(3))
        10.00000000000000*x^4 + 2.000000000000000*x + [1e+1 +/- 2.01]
    """
    ecopol = economization_polynomials(pol.parent(), pol.degree() + 1)
    delta_bound = eps.parent().zero()
    newpol = pol
    for k in xrange(pol.degree(), -1, -1):
        c = newpol[k]/ecopol[k].leading_coefficient()
        tmp_bound = delta_bound + abs(c)
        if safe_lt(tmp_bound, eps):
            delta_bound = tmp_bound
            # lc → exact zero
            high = (newpol >> (k + 1)) << (k + 1)
            low = newpol[:k]
            newpol = high + low - c*ecopol[k][:k]
    newpol = newpol[0].add_error(delta_bound) + ((newpol >> 1) << 1)
    return newpol

def chebyshev_economization(pol, eps):
    r"""
    Decrease the degree of ``pol`` in such a way that its value on the real
    segment [-1, 1] changes at most by ``eps``.

    EXAMPLES::

        sage: from ore_algebra.analytic.polynomial_approximation import taylor_economization
        sage: pol = polygen(QQ, 'x')._exp_series(10).change_ring(RBF)
        sage: newpol = chebyshev_economization(pol, RBF(1e-3)); newpol
        [0.04379...]*x^4 + [0.17734...]*x^3 + [0.49919...]*x^2 + [0.9973...]*x +
        [1.000 +/- 6...e-4]

    TESTS::

        sage: from ore_algebra.analytic.polynomial_approximation import _test_fun_approx
        sage: _test_fun_approx(newpol, pol, interval_rad=1)
    """
    pol1 = combine_radii(pol)
    return general_economization(chebyshev_polynomials, pol1, eps)

def mixed_economization(): # ?
    pass

def doit(dop, ini, path, rad, eps, derivatives, economization, x_is_real):

    # Merge with analytic_continuation.analytic_continuation()???

    eps1 = bounds.IR(eps)/2
    rad = bounds.IR(rad) # TBI
    ctx = ancont.Context(dop, path, eps/2)

    pairs = ancont.analytic_continuation(ctx, ini=ini)
    local_ini = pairs[0][1]

    Scalars = utilities.ball_field(eps1, x_is_real and ctx.real())
    x = dop.base_ring().change_ring(Scalars).gen()

    local_dop = ctx.path.vert[-1].local_diffop()
    evpt = EvaluationPoint(x, rad=rad, jet_order=derivatives)
    polys = series_sum(local_dop, local_ini.column(0), evpt,
                                accuracy.AbsoluteError(eps1),
                                stride=5)

    def postprocess(pol):
        return economization(pol(rad*x), eps1)(x/rad)
    new_polys = polys.apply_map(postprocess)

    return new_polys

def on_disk(dop, ini, path, rad, eps):
    r"""
    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: from ore_algebra.analytic import polynomial_approximation as polapprox
        sage: QQi.<i> = QuadraticField(-1, 'I')
        sage: Dops, x, Dx = Diffops()

        sage: polapprox.on_disk(Dx - 1, [1], [0], 1, 1e-3)
        ([0.001...])*x^6 + ([0.008...])*x^5 + ([0.041...])*x^4
        + ([0.16...])*x^3 + 0.50...*x^2 + x + [1.00...] + [+/- ...]*I

    TESTS::

        sage: from ore_algebra.analytic.polynomial_approximation import _test_fun_approx
        sage: pol = polapprox.on_disk(Dx - 1, [1], [0, i], 1, 1e-20)
        sage: _test_fun_approx(pol, lambda b: (i + b).exp(), disk_rad=1, prec=200)

    ::

        sage: pol = polapprox.on_disk(Dx^2 + 2*x*Dx, [0, 2/sqrt(RBF(pi))], [0], 2, 1e-10)
        sage: _test_fun_approx(pol, lambda x: x.erf(), disk_rad=1)
        sage: pol[0].abs() < RBF(10)^-10
        True

    Ensure that the polynomial we computed correctly takes into account that x
    may be complex, even though the Taylor series it is based on has real
    coefficients::

        sage: pol.parent()
        Univariate Polynomial Ring in x over Complex ball field with ... bits precision
        sage: pol[0].imag().is_zero()
        False
    """
    # Always use a *complex* ball field (otherwise the error term will be
    # real even though it represents a bound on a series tail involving a
    # complex x). (TBI!)
    return doit(dop, ini, path, rad, eps, 1, taylor_economization, False)[0]

def on_interval(dop, ini, path, eps, rad=None):
    r"""
    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: from ore_algebra.analytic import polynomial_approximation as polapprox
        sage: Dops, x, Dx = Diffops()

        sage: pol1 = polapprox.on_interval(Dx - 1, [1], [0], 1e-3, rad=1); pol1
        [0.008...]*x^5 + [0.04...]*x^4 + [0.166...]*x^3 + [0.499...]*x^2
        + [1.0...]*x + [1.0...]

        sage: pol2 = polapprox.on_interval(Dx - 1, [1], [0, [1, 2]], 1e-3); pol2
        [0.189...]*x^4 + [0.7...]*x^3 + [2.24...]*x^2 + [4.4...]*x + [4.4...]

    TESTS::

        sage: from ore_algebra.analytic.polynomial_approximation import _test_fun_approx
        sage: _test_fun_approx(pol1, lambda b: b.exp(), interval_rad=1)
        sage: _test_fun_approx(pol2, lambda b: (3/2 + b).exp(), interval_rad=.5)
        sage: polapprox.on_interval(Dx - 1, [1], [0, [1, 2]], 1e-3, rad=1)
        Traceback (most recent call last):
        ...
        TypeError: unexpected value for point: [1, 2]
        sage: polapprox.on_interval(Dx - 1, [1], [0], 1e-3)
        Traceback (most recent call last):
        ...
        TypeError: missing radius

        sage: pol = polapprox.on_interval(Dx^2 - x,
        ....:         ini=[1/(gamma(2/3)*3^(2/3)), -1/(gamma(1/3)*3^(1/3))],
        ....:         path=[0,[-1,1]], eps=1e-8)
        sage: _test_fun_approx(pol, lambda x: RBF(CBF(x).airy_ai()), interval_rad=1)
    """
    if rad is None:
        try:
            left, right = path[-1]
        except TypeError:
            raise TypeError("missing radius")
        mid = (left + right)/2
        rad = (right - left)/2
        mypath = path[:-1] + [mid]
    else:
        mypath = path
    return doit(dop, ini, mypath, rad, eps, 1, chebyshev_economization, True)[0]

def _test_fun_approx(pol, ref, disk_rad=None, interval_rad=None,
        prec=53, test_count=100):
    r"""
    EXAMPLES::

        sage: from ore_algebra.analytic.polynomial_approximation import _test_fun_approx
        sage: _test_fun_approx(lambda x: x.exp(), lambda x: x.exp() + x/1000,
        ....:                  interval_rad=1)
        Traceback (most recent call last):
        ...
        AssertionError: z = ..., ref(z) = ... not in pol(z) = ...
    """
    from sage.rings.real_mpfr import RealField
    from sage.rings.real_arb import RealBallField
    from sage.rings.complex_arb import ComplexBallField
    my_RR = RealField(prec)
    my_RBF = RealBallField(prec)
    my_CBF = ComplexBallField(prec)
    if bool(disk_rad) == bool(interval_rad):
        raise ValueError
    rad = disk_rad or interval_rad
    for _ in xrange(test_count):
        rho = my_RBF(my_RR.random_element(-rad, rad))
        if disk_rad:
            exp_i_theta = my_CBF(my_RR.random_element(0, 1)).exppii()
            z = rho*exp_i_theta
        elif interval_rad:
            z = rho
        ref_z = ref(z)
        pol_z = pol(z)
        if not ref_z.overlaps(pol_z):
            fmt = "z = {}, ref(z) = {} not in pol(z) = {}"
            raise AssertionError(fmt.format(z, ref_z, pol_z))
