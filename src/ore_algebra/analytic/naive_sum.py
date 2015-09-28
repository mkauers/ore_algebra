# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of convergent D-finite series by direct summation

NOTES:

- cythonize?

FIXME: silence deprecation warnings::

    sage: def ignore(*args): pass
    sage: sage.misc.superseded.warning=ignore
"""

import collections, itertools, logging

from sage.matrix.constructor import identity_matrix, matrix
from sage.modules.free_module_element import vector
from sage.rings.infinity import infinity
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.sequence import Sequence

from ore_algebra.ore_algebra import OreAlgebra

from ore_algebra.analytic import accuracy, bounds, utilities
from ore_algebra.analytic.safe_cmp import safe_lt

logger = logging.getLogger(__name__)

def series_sum_ordinary(dop, ini, pt, tgt_error,
        maj=None, derivatives=1, stride=50,
        record_bounds_in=None):
    r"""
    EXAMPLES::

        sage: from sage.rings.real_arb import RealBallField, RBF
        sage: from sage.rings.complex_ball_acb import ComplexBallField, CBF
        sage: QQi.<i> = QuadraticField(-1)

        sage: from ore_algebra.analytic.ui import *
        sage: from ore_algebra.analytic.naive_sum import series_sum_ordinary
        sage: Dops, x, Dx = Diffops()

        sage: dop = ((4*x^2 + 3/58*x - 8)*Dx^10 + (2*x^2 - 2*x)*Dx^9 +
        ....:       (x^2 - 1)*Dx^8 + (6*x^2 - 1/2*x + 4)*Dx^7 +
        ....:       (3/2*x^2 + 2/5*x + 1)*Dx^6 + (-1/6*x^2 + x)*Dx^5 +
        ....:       (-1/5*x^2 + 2*x - 1)*Dx^4 + (8*x^2 + x)*Dx^3 +
        ....:       (-1/5*x^2 + 9/5*x + 5/2)*Dx^2 + (7/30*x - 12)*Dx +
        ....:       8/7*x^2 - x - 2)
        sage: ini = [CBF(-1/16, -2), CBF(-17/2, -1/2), CBF(-1, 1), CBF(5/2, 0),
        ....:       CBF(1, 3/29), CBF(-1/2, -2), CBF(0, 0), CBF(80, -30),
        ....:       CBF(1, -5), CBF(-1/2, 11)]

    Funny: on the following example, both the evaluation point and most of the
    initial values are exact, so that we end up with a significantly better
    approximation than requested::

        sage: series_sum_ordinary(dop, ini, 1/2, RBF(1e-16))
        ([-3.575140703474456...] + [-2.2884877202396862...]*I)

        sage: import logging; logging.basicConfig()
        sage: series_sum_ordinary(dop, ini, 1/2, RBF(1e-30))
        WARNING:ore_algebra.analytic.naive_sum:input intervals too wide wrt
        requested accuracy
        ...
        ([-3.5751407034...] + [-2.2884877202...]*I)

    TESTS::

        sage: b = series_sum_ordinary((x^2 + 1)*Dx^2 + 2*x*Dx, [RBF(0), RBF(1)],
        ....:                         7/10, RBF(1e-30))
        sage: b.parent()
        Vector space of dimension 1 over Real ball field with ... precision
        sage: b[0].rad().exact_rational() < 10^(-30)
        True
        sage: b[0].overlaps(RealBallField(130)(7/10).arctan())
        True

        sage: b = series_sum_ordinary((x^2 + 1)*Dx^2 + 2*x*Dx, [RBF(0), RBF(1)],
        ....:                         (i+1)/2, RBF(1e-30))
        Traceback (most recent call last):
            ...
        TypeError: unable to convert 1/2*i + 1/2 to a RealBall

        sage: b = series_sum_ordinary((x^2 + 1)*Dx^2 + 2*x*Dx, [CBF(0), CBF(1)],
        ....:                         (i+1)/2, RBF(1e-30))
        sage: b.parent()
        Vector space of dimension 1 over Complex ball field with ... precision
        sage: b[0].overlaps(ComplexBallField(130)((1+i)/2).arctan())
        True
    """

    logger.info("target error = %s", tgt_error.lower())
    _, Pols, Scalars, dop = dop._normalize_base_ring()
    Rops = OreAlgebra(PolynomialRing(Scalars, 'n'), 'Sn')
    rop = dop.to_S(Rops).primitive_part().numerator()
    ordrec = rop.order()
    bwrec = [rop[ordrec-k](Pols.gen()-ordrec) for k in xrange(ordrec+1)]

    if len(ini) != dop.order():
        raise ValueError('need {} initial values'.format(dop.order()))
    ini = Sequence(ini)

    if not isinstance(tgt_error, accuracy.ErrorCriterion):
        input_is_precise = (
                (pt.parent().is_exact()
                    or safe_lt(bounds.IR(pt.rad()), tgt_error))
                and all(safe_lt(bounds.IR(x.rad()), tgt_error) for x in ini))
        tgt_error = accuracy.AbsoluteError(tgt_error, input_is_precise)
        if not input_is_precise:
            logger.warn("input intervals too wide wrt requested accuracy")

    rad = abs(bounds.IC(pt))
    if maj is None: # let's support calling this function directly...
        maj = bounds.bound_diffop(dop)
    logger.log(logging.DEBUG-1, "Majorant:\n%s", maj)

    if isinstance(tgt_error, accuracy.RelativeError) and derivatives > 1:
        raise TypeError("relative error only supported for derivatives == 1")

    # XXX: probably should support exact ini here...
    Intervals = ini.universe()
    while True:
        try:
            psum = series_sum_ordinary_doit(Intervals, bwrec, ini, pt, rad,
                tgt_error, maj, derivatives, stride, record_bounds_in)
            break
        except accuracy.PrecisionError:
            new_prec = Intervals.precision()*2
            logger.info("lost too much precision, restarting with %d bits",
                        new_prec)
            Intervals = type(Intervals)(new_prec)
    return psum

# TODO:
# - add support for pt=x (polynomial indet)
# - make it possible to specify a number of terms to sum rather than a target
#   accuracy

def series_sum_ordinary_doit(Intervals, bwrec, ini, pt, rad, tgt_error, maj,
        derivatives, stride, record_bounds_in):

    ini = Sequence(ini, universe = Intervals) # TBI ?
    pt = Intervals(pt) # we might support parent(pt) != Intervals later on
    PertPols = PolynomialRing(pt.parent(), 'eta')
    Jets = PertPols.quo(PertPols.one() << derivatives) # Faster than genuine series
    pt = Jets([pt, 1])

    ordrec = len(bwrec) - 1
    last = collections.deque([Intervals.zero()]*(ordrec - len(ini) + 1))
    last.extend(reversed(ini))
    assert len(last) == ordrec + 1 # not ordrec!

    # Singular part. Not the most natural thing to do here, but hopefully
    # generalizes well to the regular singular case.
    ptpow = pt.parent().one()
    psum = Intervals.zero()
    for n in range(len(ini)):
        last.rotate(1)
        term = Jets(last[0])*ptpow
        psum += term
        ptpow *= pt

    tail_bound = bounds.IR(infinity)
    for n in itertools.count(len(ini)):
        last.rotate(1)
        #last[0] = None
        # At this point last[0] should be considered undefined (it will hold
        # the coefficient of z^n later in the loop body) and last[1], ...
        # last[ordrec] are the coefficients of z^(n-1), ..., z^(n-ordrec)
        abs_sum = abs(psum[0])
        if n%stride == 0:
            logger.debug("n=%s, sum=%s, last tail_bound=%s",
                         n, psum[0], tail_bound.upper())
        if n%stride == 0 and (tgt_error.reached(term[0], abs_sum)
                              or record_bounds_in is not None):
            # Warning: this residual must correspond to the operator stored in
            # maj.dop, which typically isn't the operator series_sum_ordinary
            # was called on (but the result of its conversion to an operator
            # in θx, i.e. its product by a power of x).
            residual = bounds.residual(bwrec, n, list(last)[1:],
                    maj.Poly.variable_name())
            tail_bound = maj.matrix_sol_tail_bound(n, rad, [residual], ord=derivatives)
            if record_bounds_in is not None:
                record_bounds_in.append((n, psum, tail_bound))
            if tgt_error.reached(tail_bound, abs_sum):
                break
        comb = sum(Intervals(bwrec[k](n))*last[k] for k in xrange(1, ordrec+1))
        last[0] = -Intervals(~bwrec[0](n))*comb
        # logger.debug("n = %s, [c(n), c(n-1), ...] = %s", n, list(last))
        term = Jets(last[0])*ptpow
        psum += term
        ptpow *= pt
    # Account for the dropped high-order terms in the intervals we return.
    # - Is this the right place do that?
    # - Overestimation: tail_bound is actually a bound on the Frobenius norm of
    #   the error! (TBI?)
    err = tail_bound.abs()
    res = vector(x.shake(err) for x in psum)
    logger.info("summed %d terms, tail <= %s, coeffwise error <= %s", n,
            tail_bound, max(x.rad() for x in res))
    return res

# XXX: drop the 'ring' parameter? pass ctx (→ real/complex?)?
def fundamental_matrix_ordinary(dop, pt, ring, eps, rows, maj):
    eps_col = bounds.IR(eps)/bounds.IR(dop.order()).sqrt()
    prec = utilities.prec_from_eps(eps)
    my_ring = type(ring)(prec + 3*prec.nbits() + 5) # TBI!!!
    cols = [
        series_sum_ordinary(dop, ini, pt, eps_col, maj=maj, derivatives=rows)
        for ini in identity_matrix(my_ring, dop.order())]
    return matrix(cols).transpose().change_ring(ring)

def plot_bounds(dop, ini, pt, eps, pplen=0):
    r"""
    EXAMPLES::

        sage: from sage.rings.real_arb import RBF
        sage: from sage.rings.complex_ball_acb import CBF
        sage: from ore_algebra.analytic.ui import Diffops
        sage: from ore_algebra.analytic import naive_sum
        sage: Dops, x, Dx = Diffops()

        sage: naive_sum.plot_bounds(Dx - 1, [CBF(1)], CBF(i)/2, RBF(1e-20))
        Graphics object consisting of 2 graphics primitives
    """
    import sage.plot.all as plot
    recd = []
    maj = bounds.bound_diffop(dop, pol_part_len=pplen)  # cache in ctx?
    ref_sum = series_sum_ordinary(dop, ini, pt, eps, stride=1, derivatives=1,
                                  record_bounds_in=recd, maj=maj)
    # Note: this won't work well when the errors get close to the double
    # precision underflow threshold.
    error_plot = plot.line(
            [(n, (psum[0]-ref_sum[0]).abs().upper())
             for n, psum, _ in recd],
            color="black", scale="semilogy")
    error_plot = plot.line(
            [(n, (psum[0]-ref_sum[0]).abs().lower())
             for n, psum, _ in recd],
            color="black", scale="semilogy")
    bound_plot = plot.line([(n, bound.upper()) for n, _, bound in recd],
                           color="blue", scale="semilogy")
    return error_plot + bound_plot

