# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of univariate D-finite functions by numerical analytic continuation
"""

import logging

import sage.rings.all as rings
import sage.rings.real_arb
import sage.rings.complex_arb

from sage.matrix.constructor import identity_matrix, matrix
from sage.rings.complex_arb import ComplexBallField
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field_element import NumberFieldElement
from sage.rings.real_arb import RealBallField
from sage.structure.sequence import Sequence

from ore_algebra.analytic import bounds

from ore_algebra.analytic.path import Path, OrdinaryPoint, RegularPoint, IrregularSingularPoint, Step
from ore_algebra.analytic.utilities import *

logger = logging.getLogger(__name__)

class Context(object):

    def __init__(self, dop, path, eps, keep="last",
            summation_algorithm=None):
        if not dop:
            raise ValueError("operator must be nonzero")
        self.dop = dop
        # XXX: allow the user to specify their own Path
        self.path = self.initial_path = Path(path, self.dop, classify=True)
        self.initial_path.check_singularity()
        if any(isinstance(x, IrregularSingularPoint) for x in self.path.vert):
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

        if isinstance(summation_algorithm, str):
            if summation_algorithm == "naive":
                from . import naive_sum as mod
            elif summation_algorithm == "binsplit":
                from . import binary_splitting as mod
            else:
                raise ValueError("summation_algorithm", summation_algorithm)
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

        self.binary_splitting_eps_threshold = bounds.IR(1e-100)

        # XXX: self.ring
        self.eps = bounds.IR(eps)

    def _repr_(self):
        # TODO: display useful info/stats...
        return "Analytic continuation problem " + str(self.initial_path)

    def real(self):
        return (rings.RIF.has_coerce_map_from(self.dop.base_ring().base_ring())
                and all(v.is_real() for v in self.path.vert))

def ordinary_step_transition_matrix(ctx, step, eps, rows, pplen=2):
    # TODO: adjust pplen automatically?
    ldop = step.start.local_diffop()
    maj = bounds.DiffOpBound(ldop, pol_part_len=pplen)  # cache in ctx?
    if ctx.fundamental_matrix_ordinary is not None:
        fundamental_matrix_ordinary = ctx.fundamental_matrix_ordinary
    elif step.is_exact() and eps < ctx.binary_splitting_eps_threshold:
        from .binary_splitting import fundamental_matrix_ordinary
    else:
        from .naive_sum import fundamental_matrix_ordinary
    mat = fundamental_matrix_ordinary(ldop, step.delta(), eps, rows, maj)
    return mat

def singular_step_transition_matrix(ctx, step, eps, rows, pplen=2):
    from .naive_sum import fundamental_matrix_regular
    ldop = step.start.local_diffop()
    mat = fundamental_matrix_regular(ldop, step.delta(), eps, rows, pplen)
    return mat

def inverse_singular_step_transition_matrix(ctx, step, eps, rows):
    rev_step = Step(step.end, step.start)
    mat = singular_step_transition_matrix(ctx, rev_step, eps/2, rows)
    return ~mat

def step_transition_matrix(ctx, step, eps, rows=None):
    if rows is None:
        rows = ctx.dop.order()
    z0, z1 = step
    if ctx.dop.order() == 0:
        logger.info("%s: trivial case", step)
        return matrix(ZZ) # 0 by 0
    elif z0.value == z1.value:
        logger.info("%s: trivial case", step)
        return identity_matrix(ZZ, ctx.dop.order())[:rows]
    elif isinstance(z0, OrdinaryPoint) and isinstance(z1, OrdinaryPoint):
        logger.info("%s: ordinary case", step)
        fun = ordinary_step_transition_matrix
    elif isinstance(z0, RegularPoint) and isinstance(z1, OrdinaryPoint):
        logger.info("%s: regular singular case (going out)", step)
        fun = singular_step_transition_matrix
    elif isinstance(z0, OrdinaryPoint) and isinstance(z1, RegularPoint):
        logger.info("%s: regular singular case (going in)", step)
        fun = inverse_singular_step_transition_matrix
    else:
        raise TypeError(type(z0), type(z1))
    return fun(ctx, step, eps, rows)

def analytic_continuation(ctx, ini=None, post=None):
    """
    Here ini and post both are matrices.

    XXX: coerce ini and post into the appropriate ring, adjust eps...
    """
    if isinstance(ini, list): # should this be here?
        try:
            ini = matrix(ctx.dop.order(), 1, ini)
        except (TypeError, ValueError):
            raise ValueError("incorrect initial values: {}".format(ini))
    eps1 = (ctx.eps/(1 + len(ctx.path))) >> 2 # TBI, +: move to ctx?
    res = []
    path_mat = identity_matrix(ZZ, ctx.dop.order())
    def store_value_if_wanted(point):
        if point.keep_value:
            value = path_mat
            if ini is not None:  value = value*ini
            if post is not None: value = post*value
            res.append((point.value, value))
    store_value_if_wanted(ctx.path.vert[0])
    for step in ctx.path:
        step_mat = step_transition_matrix(ctx, step, eps1)
        path_mat = step_mat*path_mat
        store_value_if_wanted(step.end)
    cm = sage.structure.element.get_coercion_model()
    OutputIntervals = cm.common_parent(
            ball_field(ctx.eps, ctx.real()),
            *[mat.base_ring() for pt, mat in res])
    return [(pt, mat.change_ring(OutputIntervals)) for pt, mat in res]
