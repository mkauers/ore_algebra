# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of univariate D-finite functions by numerical analytic continuation
"""

import logging

import sage.rings.all as rings
import sage.rings.real_arb
import sage.rings.complex_arb

from . import accuracy, bounds, utilities

from sage.matrix.constructor import identity_matrix, matrix
from sage.rings.complex_arb import ComplexBallField
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field_element import NumberFieldElement
from sage.rings.real_arb import RealBallField
from sage.structure.element import Matrix, canonical_coercion
from sage.structure.sequence import Sequence

from .path import Path, Step

logger = logging.getLogger(__name__)

# TODO: clean up and reorganize
class Context(object):

    def __init__(self, dop, path, eps, keep="last", algorithm=None):
        if not dop:
            raise ValueError("operator must be nonzero")
        _, _, _, self.dop = dop._normalize_base_ring()
        # XXX: allow the user to specify their own Path
        self.path = self.initial_path = Path(path, self.dop)
        self.initial_path.check_singularity()
        if not all(x.is_regular() for x in self.path.vert):
            raise NotImplementedError("analytic continuation through irregular "
                                             "singular points is not supported")
        if keep == "all":
            for v in self.path.vert:
                v.keep_value = True
        elif keep == "last":
            self.path.vert[-1].keep_value = True
        else:
            raise ValueError("keep", keep)

        # XXX: decide what to do about all this

        if isinstance(algorithm, str):
            if algorithm == "naive":
                from . import naive_sum as mod
            elif algorithm == "binsplit":
                from . import binary_splitting as mod
            else:
                raise ValueError("algorithm", algorithm)
            self.fundamental_matrix_ordinary = mod.fundamental_matrix_ordinary
        else:
            self.fundamental_matrix_ordinary = None

        self.subdivide = True
        self.optimize_path = self.use_bit_burst = False
        if self.subdivide:
            if self.optimize_path:
                self.path = self.path.optimize_by_homotopy()
            self.path = self.path.subdivide()
            if self.use_bit_burst:
                self.path = self.path.bit_burst()

        self.path.check_singularity()
        self.path.check_convergence()

        # XXX: self.ring
        self.eps = bounds.IR(eps)

    def _repr_(self):
        # TODO: display useful info/stats...
        return "Analytic continuation problem " + str(self.initial_path)

    def real(self):
        return (rings.RIF.has_coerce_map_from(self.dop.base_ring().base_ring())
                and all(v.is_real() for v in self.path.vert))

def ordinary_step_transition_matrix(step, eps, rows, ctx=None):
    from . import naive_sum, binary_splitting
    ldop = step.start.local_diffop()
    deg = ldop.degree()
    # cache in ctx?
    maj = bounds.DiffOpBound(ldop, pol_part_len=4, bound_inverse="solve")
    assert len(maj.special_shifts) == 1 and maj.special_shifts[0] == 1
    if ctx is not None and ctx.fundamental_matrix_ordinary is not None:
        return ctx.fundamental_matrix_ordinary(
                ldop, step.delta(), eps, rows, maj)
    elif step.is_exact():
        thr = 256 + 32*deg
        a = step.cvg_ratio()
        if eps > a.max(a.parent().one() >> 100)**thr: # TBI
            try:
                return naive_sum.fundamental_matrix_ordinary(
                        ldop, step.delta(), eps, rows, maj, max_prec=4*thr)
            except accuracy.PrecisionError:
                pass
        return binary_splitting.fundamental_matrix_ordinary(
                ldop, step.delta(), eps, rows, maj)
    else:
        return naive_sum.fundamental_matrix_ordinary(
                ldop, step.delta(), eps, rows, maj, max_prec=(1<<30))

def singular_step_transition_matrix(step, eps, rows, ctx=None, determination=0):
    from .naive_sum import fundamental_matrix_regular
    ldop = step.start.local_diffop()
    mat = fundamental_matrix_regular(ldop, step.delta(), eps, rows,
                                     determination)
    return mat

def inverse_singular_step_transition_matrix(step, eps, rows, ctx=None):
    rev_step = Step(step.end, step.start)
    mat = singular_step_transition_matrix(rev_step, eps/2, rows)
    return ~mat

def step_transition_matrix(step, eps, rows=None, ctx=None):
    order = step.start.dop.order()
    if rows is None:
        rows = order
    z0, z1 = step
    if order == 0:
        logger.info("%s: trivial case", step)
        return matrix(ZZ) # 0 by 0
    elif z0.value == z1.value:
        logger.info("%s: trivial case", step)
        return identity_matrix(ZZ, order)[:rows]
    elif z0.is_ordinary() and z1.is_ordinary():
        logger.info("%s: ordinary case", step)
        logger.debug("fraction of cvrad: %s/%s", step.length(), z0.dist_to_sing())
        fun = ordinary_step_transition_matrix
    elif z0.is_regular() and z1.is_ordinary():
        logger.info("%s: regular singular case (going out)", step)
        logger.debug("fraction of cvrad: %s/%s", step.length(), z0.dist_to_sing())
        fun = singular_step_transition_matrix
    elif z0.is_ordinary() and z1.is_regular():
        logger.info("%s: regular singular case (going in)", step)
        logger.debug("fraction of cvrad: %s/%s", step.length(), z1.dist_to_sing())
        fun = inverse_singular_step_transition_matrix
    else:
        raise TypeError(type(z0), type(z1))
    return fun(step, eps, rows, ctx=ctx)

def analytic_continuation(ctx, ini=None, post=None):
    """
    INPUT:

    - ``ini`` (constant matrix, optional) - initial values, one column per
      solution
    - ``post`` (matrix of polynomial/rational functions, optional) - linear
      combinations of the first Taylor coefficients to take, as a function of
      the evaluation point

    TESTS::

        sage: from ore_algebra import DifferentialOperators
        sage: _, x, Dx = DifferentialOperators()
        sage: (Dx^2 + 2*x*Dx).numerical_solution([0, 2/sqrt(pi)], [0,i])
        [+/- ...] + [1.65042575879754...]*I
    """
    logger.info("path: %s", ctx.path)
    eps1 = (ctx.eps/(1 + len(ctx.path))) >> 2 # TBI, +: move to ctx?
    prec = utilities.prec_from_eps(eps1)
    if ini is not None:
        if not isinstance(ini, Matrix): # should this be here?
            try:
                ini = matrix(ctx.dop.order(), 1, list(ini))
            except (TypeError, ValueError):
                raise ValueError("incorrect initial values: {}".format(ini))
        try:
            ini = ini.change_ring(RealBallField(prec))
        except ValueError:
            ini = ini.change_ring(ComplexBallField(prec))
    res = []
    path_mat = identity_matrix(ZZ, ctx.dop.order())
    def store_value_if_wanted(point):
        if point.keep_value:
            value = path_mat
            if ini is not None:  value = value*ini
            if post is not None: value = post(point.value)*value
            res.append((point.value, value))
    store_value_if_wanted(ctx.path.vert[0])
    for step in ctx.path:
        step_mat = step_transition_matrix(step, eps1, ctx=ctx)
        path_mat = step_mat*path_mat
        store_value_if_wanted(step.end)
    cm = sage.structure.element.get_coercion_model()
    OutputIntervals = cm.common_parent(
            utilities.ball_field(ctx.eps, ctx.real()),
            *[mat.base_ring() for pt, mat in res])
    return [(pt, mat.change_ring(OutputIntervals)) for pt, mat in res]

def normalize_post_transform(dop, post_transform):
    if post_transform is None:
        post_transform = dop.parent().one()
    else:
        _, post_transform = canonical_coercion(dop, post_transform)
    return post_transform % dop
