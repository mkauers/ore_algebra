# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of convergent D-finite series by direct summation

NOTES:

- cythonize?
"""

import collections, itertools, logging

from sage.matrix.constructor import identity_matrix, matrix
from sage.modules.free_module_element import vector
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.sequence import Sequence

from ore_algebra.ore_algebra import OreAlgebra

from . import bounds, utilities

from .bounds import bound_diffop

logger = logging.getLogger(__name__)

def series_sum_ordinary(dop, ini, pt, tgt_error,
        maj=None, derivatives=None, stride=50,
        record_bounds_in=None):

    logger.info("target error = %s", tgt_error)
    _, Pols, Scalars, dop = dop._normalize_base_ring()
    Rops = OreAlgebra(PolynomialRing(Scalars, 'n'), 'Sn')
    rop = dop.to_S(Rops).primitive_part().numerator()
    ordrec = rop.order()
    bwrec = [rop[ordrec-k](Pols.gen()-ordrec) for k in xrange(ordrec+1)]

    if not isinstance(tgt_error, bounds.ErrorCriterion):
        tgt_error = bounds.AbsoluteError(tgt_error)
    rad = abs(bounds.IC(pt))
    if maj is None: # let's support calling this function directly...
        maj = bound_diffop(dop)

    ini = Sequence(ini)
    Intervals = ini.universe()
    last = collections.deque(itertools.repeat(Intervals.zero(), ordrec - len(ini) + 1))
    last.extend(reversed(ini))
    assert len(last) == ordrec + 1 # not ordrec!

    if derivatives is None:
        derivatives = dop.order()
    if isinstance(tgt_error, bounds.RelativeError) and derivatives > 1:
        raise TypeError("relative error only supported for derivatives == 0")
    pt = Intervals(pt) # ?????
    PertPols = PolynomialRing(pt.parent(), 'eta')
    Jets = PertPols.quo(PertPols.one() << derivatives) # Faster than genuine series
    pt = Jets([pt, 1])

    # Singular part. Not the most natural thing to do here, but hopefully
    # generalizes well to the regular singular case.
    ptpow = pt.parent().one()
    psum = Intervals.zero()
    for n in range(len(ini)):
        last.rotate(1)
        term = Jets(last[0])*ptpow
        psum += term
        ptpow *= pt

    tail_bound = None
    if len(ini) != dop.order():
        raise ValueError('need {} initial values'.format(dop.order()))
    for n in itertools.count(len(ini)):
        last.rotate(1)
        #last[0] = None
        # At this point last[0] should be considered undefined (it will hold
        # the coefficient of z^n later in the loop body) and last[1], ...
        # last[ordrec] are the coefficients of z^(n-1), ..., z^(n-ordrec)
        abs_sum = abs(psum[0])
        if n%stride == 0:
            logger.debug("n=%s, sum=%s, last tail_bound=%s",
                         n, psum[0], tail_bound)
        if n%stride == 0 and (tgt_error.reached(term[0], abs_sum)
                              or record_bounds_in is not None):
            residual = bounds.residual(bwrec, n, list(last)[1:])
            tail_bound = maj.matrix_sol_tail_bound(n, rad, [residual], ord=derivatives)
            if record_bounds_in is not None:
                record_bounds_in.append((n, psum, tail_bound))
            if tgt_error.reached(tail_bound, abs_sum):
                break
        last[0] = - Intervals(~bwrec[0](n))*sum(Intervals(bwrec[k](n))*last[k]
                                     for k in xrange(1, ordrec+1))
        # logger.debug("n = %s, [c(n), c(n-1), ...] = %s", n, list(last))
        term = Jets(last[0])*ptpow
        psum += term
        ptpow *= pt
    # Account for the dropped high-order terms in the intervals we return.
    # - Is this the right place do that?
    # - Overestimation: tail_bound is actually a bound on the Frobenius norm of
    #   the error! (TBI?)
    mag = tail_bound.magnitude()
    # XXX: works for both real and complex interval fields but just by chance
    tail_enclosure = Intervals((-mag, mag), (-mag, mag))
    res = vector(x + tail_enclosure for x in psum)
    logger.info("summed %d terms, tail <= %s, coeffwise error <= %s", n,
            tail_bound, bounds.IR(max(utilities.rad(x) for x in res)))
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
    import sage.plot.all as plot
    recd = []
    maj = bound_diffop(dop, pol_part_len=pplen)  # cache in ctx?
    ref_sum = series_sum_ordinary(dop, ini, pt, eps, stride=1, derivatives=1,
                                  record_bounds_in=recd, maj=maj)
    error_plot = plot.line(
            [(n, (psum[0]-ref_sum[0]).abs().lower())
             for n, psum, _ in recd],
            color="black", scale="semilogy")
    bound_plot = plot.line([(n, bound.upper()) for n, _, bound in recd],
                           color="blue", scale="semilogy")
    return error_plot + bound_plot

