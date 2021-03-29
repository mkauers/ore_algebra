# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of univariate D-finite functions by numerical analytic continuation
"""

# Copyright 2015, 2016, 2017, 2018, 2019 Marc Mezzarobba
# Copyright 2015, 2016, 2017, 2018, 2019 Centre national de la recherche scientifique
# Copyright 2015, 2016, 2017, 2018 Université Pierre et Marie Curie
# Copyright 2019 Sorbonne Université
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

import logging

import sage.rings.all as rings
import sage.rings.real_arb
import sage.rings.complex_arb

from . import accuracy, bounds, utilities
from . import naive_sum, binary_splitting

from sage.matrix.constructor import identity_matrix, matrix
from sage.rings.complex_arb import ComplexBallField
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field_element import NumberFieldElement
from sage.rings.real_arb import RealBallField
from sage.structure.element import Matrix, canonical_coercion
from sage.structure.sequence import Sequence

from .context import Context, dctx # re-export Context
from .differential_operator import DifferentialOperator
from .path import Path, Step

logger = logging.getLogger(__name__)

def step_transition_matrix(dop, step, eps, rows=None, split=0, ctx=dctx):
    r"""
    TESTS::

        sage: from ore_algebra.examples import fcc
        sage: fcc.dop4.numerical_solution([0, 0, 0, 1], [0, 1], 1e-3)
        [1...] + [+/- ...]*I
    """

    order = dop.order()
    if rows is None:
        rows = order
    z0, z1 = step
    if order == 0:
        logger.debug("%s: trivial case", step)
        return matrix(ZZ) # 0 by 0
    elif z0.value == z1.value:
        logger.debug("%s: trivial case", step)
        return identity_matrix(ZZ, order)[:rows]
    elif z0.is_ordinary() and z1.is_ordinary():
        logger.info("%s: ordinary case", step)
        if z0.is_exact():
            inverse = False
        # XXX maybe also invert the step when z1 is much simpler than z0
        else: # can happen with the very first step
            step = Step(z1, z0, max_split=0)
            inverse = True
    elif z0.is_regular() and z1.is_ordinary():
        logger.info("%s: regular singular case (going out)", step)
        inverse = False
    elif z0.is_ordinary() and z1.is_regular():
        logger.info("%s: regular singular case (going in)", step)
        step = Step(z1, z0)
        inverse = True
        eps /= 2
    else:
        raise ValueError(z0, z1)
    try:
        mat = regular_step_transition_matrix(dop, step, eps, rows,
                fail_fast=(step.max_split > 0), effort=split, ctx=ctx)
    except (accuracy.PrecisionError, bounds.BoundPrecisionError):
        if step.max_split == 0:
            raise # XXX: can we return something?
        logger.info("splitting step...")
        s0, s1 = step.split()
        m0 = step_transition_matrix(dop, s0, eps/4, None, split+1, ctx)
        m1 = step_transition_matrix(dop, s1, eps/4, rows, split+1, ctx)
        mat = m1*m0
    if inverse:
        try:
            mat = ~mat
        except ZeroDivisionError:
            raise accuracy.PrecisionError(
                    f"failed to invert the fundamental matrix at {z1}")
    return mat

def _use_binsplit(dop, step, tgt_prec, base_point_size, ctx):
    if ctx.prefer_binsplit():
        return True
    elif ctx.prefer_naive():
        return False
    else:
        # (Cost of a bit burst step via a truncation at prec base_point_size)
        #   ≈ ordrec³·nterms·(op_height + base_point_size + ordrec + δ)
        # where δ is related to the algebraic degree of the point
        #
        # (Cost of direct summation) ≈ ordrec·nterms·prec,
        # assuming that the costs related to polynomial evaluation, basis size
        # and structure, etc., are the same in both cases, and neglecting
        # interval issues that may increase the cost of direct summation to
        # something like ordrec·nterms·(prec + nterms·log(ordrec))
        ordrec = dop._my_to_S().order()
        # XXX pulling the algdeg term out of my hat
        sz = dop._naive_height() + base_point_size + 16*ordrec
        sz += 16*step.algdeg()**2
        # The exponent of ordrec should become 2 according to the above
        # heuristic once interval squashing works properly
        est = 256 + sz*ordrec**2
        use_binsplit = (tgt_prec >= est)
        logger.debug("tgt_prec = %s %s %s", tgt_prec,
                ">=" if use_binsplit else "<", est)
        return use_binsplit

def regular_step_transition_matrix(dop, step, eps, rows, fail_fast, effort,
                                   ctx=dctx):

    def args():
        ldop = dop.shift(step.start)
        return (ldop, step.evpt(rows), eps, fail_fast, effort, ctx)

    tgt_prec = utilities.prec_from_eps(eps)
    bit_burst_prec = max(2*step.prec(tgt_prec), ctx.bit_burst_thr)
    # binsplit_prec only matters when bit_burst_thr is large, could be replaced
    # by bit_burst_prec otherwise
    binsplit_prec = min(bit_burst_prec,
                        max(step.start.bit_burst_bits(tgt_prec),
                            step.end.bit_burst_bits(tgt_prec)))
    use_binsplit = _use_binsplit(dop, step, tgt_prec, binsplit_prec, ctx)
    use_fallback = not ctx.force_algorithm and (use_binsplit or not fail_fast)

    while True:

        if use_binsplit:
            sub = step.bit_burst_split(tgt_prec, bit_burst_prec)
            if sub:
                # Assuming bitsize(step.start) << bitsize(step.end):
                # * this step should fall into the bb/bs branch again (and in
                #   the pure bb sub-branch if bitsize(step.start) is small):
                mat0 = regular_step_transition_matrix(dop, sub[0], eps>>1,
                        dop.order(), fail_fast, effort, ctx)
                # * this one will typically come back to the bit-burst
                #   sub-branch in the first few iterations, and then to one of
                #   the other ones (typically direct summation, unless
                #   target prec >> prec of endpoints >> 0):
                mat1 = regular_step_transition_matrix(dop, sub[1], eps>>1,
                        rows, fail_fast, effort, ctx)
                return mat1*mat0

        if step.type == "bit-burst":
            logger.info("%s", step)

        if use_binsplit:
            try:
                return binary_splitting.fundamental_matrix_regular(*args())
            except NotImplementedError:
                if not use_fallback:
                    raise
                logger.info("falling back to direct summation")
        else:
            try:
                return naive_sum.fundamental_matrix_regular(*args())
            except accuracy.PrecisionError:
                if not use_fallback:
                    raise
                logger.info("not enough precision, trying binary splitting "
                            "as a fallback")

        use_binsplit = not use_binsplit
        use_fallback = False

def _process_path(dop, path, ctx):

    if not isinstance(path, Path):
        path = Path(path, dop)
        if not any(v.keep_value() for v in path.vert):
            path.vert[-1].options['keep_value'] = True

    if not ctx.assume_analytic:
        path.check_singularity()
    if not all(x.is_regular() for x in path.vert):
        raise NotImplementedError("analytic continuation through irregular "
                                  "singular points is not supported")

    if ctx.assume_analytic:
        path = path.bypass_singularities()
        path.check_singularity()

    if ctx.deform:
        path = path.deform_or_subdivide()
    else:
        path = path.subdivide()
    path.check_singularity()
    path.check_convergence()

    if ctx.recorder is not None:
        ctx.recorder.path = path

    return path

def analytic_continuation(dop, path, eps, ctx=dctx, ini=None, post=None,
                          return_local_bases=False):
    """
    INPUT:

    - ``ini`` (constant matrix, optional) - initial values, one column per
      solution
    - ``post`` (matrix of polynomial/rational functions, optional) - linear
      combinations of the first Taylor coefficients to take, as a function of
      the evaluation point
    - ``return_local_bases`` (boolean) - if True, also compute and return the
      structure of local bases at all points where we are computing values of
      the solution

    OUTPUT:

    A list of dictionaries with information on the computed solution(s) at each
    evaluation point.

    TESTS::

        sage: from ore_algebra import DifferentialOperators
        sage: _, x, Dx = DifferentialOperators()
        sage: (Dx^2 + 2*x*Dx).numerical_solution([0, 2/sqrt(pi)], [0,i])
        [+/- ...] + [1.65042575879754...]*I
    """

    if dop.is_zero():
        raise ValueError("operator must be nonzero")
    _, _, _, dop = dop._normalize_base_ring()

    path = _process_path(dop, path, ctx)
    logger.info("path: %s", path)

    eps = bounds.IR(eps)
    eps1 = (eps/(1 + len(path))) >> 2
    prec = utilities.prec_from_eps(eps1)

    if ini is not None:
        if not isinstance(ini, Matrix): # should this be here?
            try:
                ini = matrix(dop.order(), 1, list(ini))
            except (TypeError, ValueError):
                raise ValueError("incorrect initial values: {}".format(ini))
        try:
            ini = ini.change_ring(RealBallField(prec))
        except (TypeError, ValueError):
            ini = ini.change_ring(ComplexBallField(prec))

    def point_dict(point, value):
        if ini is not None:
            value = value*ini
        if post is not None and not post.is_one():
            value = post(point.value)*value
        rec = {"point": point.value, "value": value}
        if return_local_bases:
            rec["structure"] = point.local_basis_structure()
        return rec

    res = []
    z0 = path.vert[0]
    # XXX still imperfect in the case of a high-precision starting point with
    # relatively large radius... (do we care?)
    main = Step(z0, z0.simple_approx(ctx=ctx))
    path_mat = step_transition_matrix(dop, main, eps1, ctx=ctx)
    if z0.keep_value():
        res.append(point_dict(z0, identity_matrix(ZZ, dop.order())))
    for step in path:
        main, dev = step.chain_simple(main.end, ctx=ctx)
        main_mat = step_transition_matrix(dop, main, eps1, ctx=ctx)
        path_mat = main_mat*path_mat
        if dev is not None:
            dev_mat = path_mat
            for sub in dev:
                sub_mat = step_transition_matrix(dop, sub, eps1, ctx=ctx)
                dev_mat = sub_mat*dev_mat
            res.append(point_dict(step.end, dev_mat))

    cm = sage.structure.element.get_coercion_model()
    real = (rings.RIF.has_coerce_map_from(dop.base_ring().base_ring())
            and all(v.is_real() for v in path.vert))
    OutputIntervals = cm.common_parent(
            utilities.ball_field(eps, real),
            *[rec["value"].base_ring() for rec in res])
    for rec in res:
        rec["value"] = rec["value"].change_ring(OutputIntervals)
    return res

def normalize_post_transform(dop, post_transform):
    if post_transform is None:
        post_transform = dop.parent().one()
    else:
        _, post_transform = canonical_coercion(dop, post_transform)
    return post_transform % dop
