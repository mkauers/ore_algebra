# vim: tw=80
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
import sys

import sage.rings.real_arb
import sage.rings.complex_arb

from . import accuracy, bounds, utilities

from sage.matrix.constructor import identity_matrix, matrix
from sage.rings.complex_arb import ComplexBallField
from sage.rings.integer_ring import ZZ
from sage.rings.real_arb import RealBallField
from sage.rings.real_mpfi import RIF
from sage.structure.element import Matrix, canonical_coercion

from .context import Context as Context  # re-export
from .context import dctx
from .monodromy import formal_monodromy
from .path import EvaluationPoint_step, Path, Step
from .utilities import invmat

logger = logging.getLogger(__name__)

def step_transition_matrix(dop, steps, eps, rows=None, split=0, ctx=dctx):
    r"""
    Transition matrices for one or more steps with the same starting point.

    Compute the transition matrices from one point to one or more points within
    the local disk of convergence, introducing substeps if convergence is too
    slow.

    TESTS::

        sage: from ore_algebra.examples import fcc
        sage: fcc.dop4.numerical_solution([0, 0, 0, 1], [0, 1], 1e-3)
        [1...] + [+/- ...]*I
    """

    assert all(step.start is steps[0].start for step in steps)
    z0 = steps[0].start
    order = dop.order()
    if rows is None:
        rows = order

    if order == 0:
        logger.debug("%s: trivial case", steps)
        return [matrix(ZZ)]*len(steps) # 0 by 0
    elif len(steps) == 1 and steps[0].is_trivial():
        logger.debug("%s: trivial case", steps)
        return [identity_matrix(ZZ, order)[:rows]]
    elif z0.is_ordinary():
        logger.info("%s: ordinary case", steps)
    elif z0.is_regular():
        logger.info("%s: regular singular case", steps)
    else:
        raise ValueError(steps)

    try:
        fail_fast = all(step.max_split > 0 for step in steps)
        mat = step_transition_matrix_bit_burst(dop, steps, eps, rows,
                fail_fast=fail_fast, effort=split, ctx=ctx)

        for i, step in enumerate(steps):
            if step.reversed:
                try:
                    inv = invmat(mat[i])
                    rad, invrad = mat[i].trace().rad(), inv.trace().rad()
                    if invrad**2 > rad:
                        logger.info("precision loss in inverse: rad=%s, inv.rad=%s",
                                    rad, invrad)
                    mat[i] = inv
                except ZeroDivisionError:
                    # split step *and* increase precision
                    eps = eps*eps
                    raise accuracy.PrecisionError(
                            f"failed to invert transition matrix {step}")

    except (accuracy.PrecisionError, bounds.BoundPrecisionError,
            bounds.BadBound):
        if any(step.max_split <= 0 for step in steps):
            raise
        logger.info("splitting step...")
        split0, split1 = zip(*(step.split() for step in steps))
        mat0 = step_transition_matrix(dop, tuple(split0),
                                      eps/4, None, split+1, ctx)
        mat1 = [step_transition_matrix(dop, (s,), eps/4, rows, split+1, ctx)[0]
                for s in split1]
        mat = [m0*m1 if step.reversed else m1*m0
               for step, m0, m1 in zip(steps, mat0, mat1)]

    return mat

def _use_binsplit(dop, steps, tgt_prec, base_point_size, ctx):
    if ctx.algorithms[0] == "auto":
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
        sz += 16*max(step.algdeg() for step in steps)**2
        # The exponent of ordrec should become 2 according to the above
        # heuristic once interval squashing works properly
        est = 256 + sz*ordrec**2
        use_binsplit = (tgt_prec >= est)
        logger.debug("tgt_prec = %s %s %s", tgt_prec,
                ">=" if use_binsplit else "<", est)
        return use_binsplit
    else:
        return ctx.algorithms[0] == "binsplit"

def step_transition_matrix_bit_burst(dop, steps, eps, rows, fail_fast, effort,
                                     ctx=dctx):
    r"""
    Transition matrices for one or more steps with the same starting point.

    Automatic algorithm choice, using the bit-burst method at high precision.
    """

    # assert all(step.start is steps[0].start for step in steps)
    z0 = steps[0].start
    ldop = dop.shift(steps[0].start)
    points = EvaluationPoint_step(steps, jet_order=rows)

    tgt_prec = utilities.prec_from_eps(eps)
    # Precision of the intermediate approximation that the bit-burst method
    # would use for the first substep of the current step. We only consider
    # using the bit-burst method for simple steps since points of large bit size
    # should appear only at the ends of the path or at the end of a detour.
    if len(steps) == 1:
        bit_burst_prec = max(2*steps[0].prec(tgt_prec), ctx.bit_burst_thr)
    else:
        bit_burst_prec = sys.maxsize
    # bit size of base point of the step; only matters when bit_burst_thr is
    # large, could be replaced by bit_burst_prec otherwise
    binsplit_prec = min(bit_burst_prec,
                        max(z0.bit_burst_bits(tgt_prec),
                            *(step.end.bit_burst_bits(tgt_prec)
                              for step in steps)))
    use_binsplit = _use_binsplit(dop, steps, tgt_prec, binsplit_prec, ctx)
    use_fallback = ((use_binsplit or not fail_fast) and
                    (len(ctx.algorithms) > 1 or "auto" in ctx.algorithms))

    while True:

        # Try using the bit-burst method. This only makes sense if we are
        # considering using binary splitting. The substeps thus introduced are
        # simple steps as well.
        if use_binsplit and len(steps) == 1:
            sub = steps[0].bit_burst_split(tgt_prec, bit_burst_prec)
            if sub:
                # Assuming bitsize(step.start) << bitsize(step.end):
                # * this step should fall into the bb/bs branch again (and in
                #   the pure bb sub-branch if bitsize(step.start) is small):
                (mat0,) = step_transition_matrix_bit_burst(dop, sub[0:1], eps>>1,
                        dop.order(), fail_fast, effort, ctx)
                # * this one will typically come back to the bit-burst
                #   sub-branch in the first few iterations, and then to one of
                #   the other ones (typically direct summation, unless
                #   target prec >> prec of endpoints >> 0):
                (mat1,) = step_transition_matrix_bit_burst(dop, sub[1:], eps>>1,
                        rows, fail_fast, effort, ctx)
                return (mat1*mat0,)

        if steps[0].type == "bit-burst":
            logger.info("%s", steps[0])

        if use_binsplit:
            try:
                from . import binary_splitting
                return binary_splitting.fundamental_matrix_regular(
                    ldop, points, eps, fail_fast, effort, ctx)
            except NotImplementedError:
                if not use_fallback:
                    raise
                logger.info("falling back to direct summation")
        else:
            if ctx.prefer_algorithm("naive", "dac"):
                from . import naive_sum as mod
            else:
                try:
                    from . import dac_sum as mod
                except ModuleNotFoundError:
                    if "naive" in ctx.algorithms or "auto" in ctx.algorithms:
                        from . import naive_sum as mod
                    else:
                        raise
            try:
                return mod.fundamental_matrix_regular(
                    ldop, points, eps, fail_fast, effort, ctx)
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
        if not any(v.store_value() for v in path.vert):
            path.vert[-1].options['store_value'] = True

    if not ctx.assume_analytic:
        path.check_singularity()
    if not all(x.is_regular() for x in path.vert):
        raise NotImplementedError("analytic continuation through irregular "
                                  "singular points is not supported")

    if ctx.assume_analytic:
        path = path.bypass_singularities()
        path.check_singularity()

    mode = 2 if ctx.two_point_mode else 1
    if ctx.deform:
        path = path.deform_or_subdivide(mode)
    else:
        path = path.subdivide(mode)
    path = path.simplify_points_add_detours(ctx)
    path.check_singularity()
    path.check_convergence()

    if ctx.recorder is not None:
        ctx.recorder.path = path

    return path

def _normalize_ini(ini, dop, eps):
    if ini is None:
        return None
    if not isinstance(ini, Matrix): # should this be here?
        try:
            ini = matrix(dop.order(), 1, list(ini))
        except (TypeError, ValueError):
            raise ValueError("incorrect initial values: {}".format(ini))
    prec = utilities.prec_from_eps(eps)
    try:
        ini = ini.change_ring(RealBallField(prec))
    except (TypeError, ValueError):
        ini = ini.change_ring(ComplexBallField(prec))
    return ini

def _process_detour(dop, point, val_mat, eps, ctx=dctx):
    if point.detour_to is None:
        return val_mat
    ex = point.detour_to.exact_approx()
    detour0 = Step(point, ex, type="detour", max_split=0)
    [sub_mat] = step_transition_matrix(dop, [detour0], eps, ctx=ctx)
    val_mat = sub_mat*val_mat
    if ex is not point.detour_to:
        # TODO: use multi-point evaluation + Cauchy bounds to handle
        # evaluations on large intervals
        detour1 = Step(ex, point.detour_to, type="detour", max_split=0)
        [sub_mat] = step_transition_matrix(dop, [detour1], eps, ctx=ctx)
        val_mat = sub_mat*val_mat
    return val_mat

def _branch_change_matrix(dop, point, branch, eps):
    assert isinstance(branch, tuple)
    prec = utilities.prec_from_eps(eps)
    ring = ComplexBallField(prec)
    mon = formal_monodromy(dop, point, ring)
    return sum(mon**b for b in branch)/len(branch)

def analytic_continuation(dop, path, eps, ctx=dctx, ini=None, post=None,
                          return_local_bases=False):
    """
    Analytic continuation along a path.

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

    eps = ctx.IR(eps)
    eps1 = (eps/(1 + len(path))) >> 4

    ini = _normalize_ini(ini, dop, eps1)

    def maybe_push_point_dict(lst, point, value):
        if point.detour_to is not None:
            assert not point.store_value()
            point = point.detour_to
        if not point.store_value():
            return
        if ini is not None:
            value = value*ini
        if post is not None and not post.is_one():
            value = post(point.value)*value
        rec = {"point": point.value, "value": value}
        if return_local_bases:
            rec["structure"] = point.local_basis_structure(critical_monomials=False)
        lst.append(rec)

    res = []

    z0 = path.vert[0]
    path_mat = identity_matrix(ZZ, dop.order())
    maybe_push_point_dict(res, z0, path_mat) # value at z0 = identity
    path_mat = invmat(_process_detour(dop, z0, path_mat, eps1, ctx=ctx))

    steps = list(path.steps())
    i = 0
    while i < len(steps):
        if (ctx.two_point_mode
                and steps[i].reversed and i + 1 < len(steps)
                and not steps[i+1].reversed):
            np = 2
        else:
            np = 1
        main_mats = step_transition_matrix(dop, steps[i:i+np], eps1, ctx=ctx)
        for step, main_mat in zip(steps[i:i+np], main_mats):
            path_mat = main_mat*path_mat
            point = step.start if step.reversed else step.end
            branch = point.options.get("outgoing_branch")
            if branch is not None:
                branch_mat = _branch_change_matrix(dop, point, branch, eps1)
                path_mat = branch_mat*path_mat
            val_mat = _process_detour(dop, point, path_mat, eps1, ctx=ctx)
            maybe_push_point_dict(res, point, val_mat)

        i += np

    cm = sage.structure.element.get_coercion_model()
    real = (RIF.has_coerce_map_from(dop.base_ring().base_ring())
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
