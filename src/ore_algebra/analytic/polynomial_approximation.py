# -*- coding: utf-8 - vim: tw=80
"""
Rigorous approximation of D-finite functions by polynomials

FIXME: silence deprecation warnings::

    sage: def ignore(*args): pass
    sage: sage.misc.superseded.warning=ignore
"""

# TODO:
# - currently we can compute the first few derivatives all at once, but there is
#   no support for computing a remote derivative using Dx^k mod dop (nor an
#   arbitrary combination of derivatives)

import logging
logger = logging.getLogger(__name__)

import sage.rings.real_arb
import sage.rings.complex_ball_acb

import ore_algebra.analytic.accuracy as accuracy
import ore_algebra.analytic.analytic_continuation as ancont
import ore_algebra.analytic.bounds as bounds
import ore_algebra.analytic.utilities as utilities

from ore_algebra.analytic.naive_sum import series_sum_ordinary

def taylor_economization():
    pass

def chebyshev_economization():
    pass

def mixed_economization(): # ?
    pass

def doit(dop, ini, path, rad, eps, derivatives):

    # Merge with analytic_continuation.analytic_continuation()???

    eps = bounds.IR(eps)
    rad = bounds.IR(rad) # TBI
    ctx = ancont.Context(dop, path, eps/2)

    pairs = ancont.analytic_continuation(ctx, ini=ini)
    local_ini = pairs[0][1]

    prec = utilities.prec_from_eps(eps/2) # TBI
    Scalars = (sage.rings.real_arb.RealBallField(prec) if ctx.real()
               else sage.rings.complex_ball_acb.ComplexBallField(prec))
    x = dop.base_ring().change_ring(Scalars).gen()

    local_dop = ctx.path.vert[-1].local_diffop()
    polys = series_sum_ordinary(local_dop, local_ini.column(0), x,
                                accuracy.AbsoluteError(eps/2),
                                stride=5, rad=rad, derivatives=derivatives)

    # TODO: postprocess

    return polys

def on_disk(dop, ini, path, rad, eps):
    r"""
    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: from ore_algebra.analytic import polynomial_approximation as polapprox
        sage: QQi.<i> = QuadraticField(-1, 'I')
        sage: Dops, x, Dx = Diffops()

    TESTS::

        sage: from sage.rings.real_arb import RBF
        sage: from ore_algebra.analytic.polynomial_approximation import _test_fun_approx
        sage: pol = polapprox.on_disk(Dx - 1, [1], [0, i], 1, 1e-20)
        sage: _test_fun_approx(pol, lambda b: (i + b).exp(), disk_rad=1, prec=200)
        sage: pol = polapprox.on_disk(Dx^2 + 2*x*Dx, [0, 2/sqrt(RBF(pi))], [0], 2, 1e-10)
        sage: _test_fun_approx(pol, lambda x: x.erf(), disk_rad=1)
    """
    return doit(dop, ini, path, rad, eps, 1)[0]

def on_interval():
    pass

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
    from sage.rings.complex_ball_acb import ComplexBallField
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
