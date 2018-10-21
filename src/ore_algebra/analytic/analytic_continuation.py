# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of univariate D-finite functions by numerical analytic continuation
"""

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

from .differential_operator import DifferentialOperator
from .path import Path, Step

logger = logging.getLogger(__name__)

class Context(object):

    def __init__(self, dop=None, path=None, eps=None,
            keep="last",
            algorithm=None,
            force_algorithm=False,
            assume_analytic=False):

        # TODO: dop, path, eps...

        if not keep in ["all", "last"]:
            raise ValueError("keep", keep)
        self.keep = keep

        if not algorithm in [None, "naive", "binsplit"]:
            raise ValueError("algorithm", algorithm)
        self.algorithm = algorithm

        self.force_algorithm = force_algorithm

        self.assume_analytic = assume_analytic

        self.max_split = 3

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def prefer_binsplit(self):
        return self.algorithm == "binsplit"

    def force_binsplit(self):
        return self.prefer_binsplit() and self.force_algorithm

    def prefer_naive(self):
        return self.algorithm == "naive"

    def force_naive(self):
        return self.prefer_naive() and self.force_algorithm

default_ctx = Context()

def step_transition_matrix(dop, step, eps, rows=None, split=0, ctx=default_ctx):
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
        logger.info("%s: trivial case", step)
        return matrix(ZZ) # 0 by 0
    elif z0.value == z1.value:
        logger.info("%s: trivial case", step)
        return identity_matrix(ZZ, order)[:rows]
    elif z0.is_ordinary() and z1.is_ordinary():
        logger.info("%s: ordinary case", step)
        ordinary = True
        inverse = False
    elif z0.is_regular() and z1.is_ordinary():
        logger.info("%s: regular singular case (going out)", step)
        ordinary = False
        inverse = False
    elif z0.is_ordinary() and z1.is_regular():
        logger.info("%s: regular singular case (going in)", step)
        step = Step(z1, z0)
        ordinary = False
        inverse = True
        eps /= 2
    else:
        raise ValueError(z0, z1)
    try:
        mat = regular_step_transition_matrix(dop, step, eps, rows,
                  fail_fast=(split < ctx.max_split), effort=split,
                  ordinary=ordinary, ctx=ctx)
    except (accuracy.PrecisionError, bounds.BoundPrecisionError):
        # XXX it would be nicer to return something in this case...
        if split >= ctx.max_split:
            raise
        logger.info("splitting step...")
        s0, s1 = step.split()
        m0 = step_transition_matrix(dop, s0, eps/4, None, split+1, ctx)
        m1 = step_transition_matrix(dop, s1, eps/4, rows, split+1, ctx)
        mat = m1*m0
    if inverse:
        mat = ~mat
    return mat

def _use_binsplit(dop, step, eps):
    if step.is_exact() and step.branch == (0,):
        # very very crude
        logprec = -eps.log()
        logratio = -step.cvg_ratio().log() # may be nan (entire functions)
        # don't discourage binary splitting too much for very small steps /
        # entire functions
        terms_est = logprec/logratio.min(logprec.log())
        return (terms_est >= 256 + 32*dop.degree()**2)
    else:
        return False

def regular_step_transition_matrix(dop, step, eps, rows, fail_fast, effort,
                                   ordinary, ctx=default_ctx):
    ldop = dop.shift(step.start)
    args = (ldop, step.delta(), eps, rows, step.branch, fail_fast, effort)
    if ctx.force_binsplit():
        return binary_splitting.fundamental_matrix_regular(*args)
    elif ctx.prefer_binsplit() or _use_binsplit(ldop, step, eps):
        try:
            return binary_splitting.fundamental_matrix_regular(*args)
        except NotImplementedError:
            logger.info("not implemented: falling back on direct summation")
            return _naive_fundamental_matrix_regular(ordinary, *args)
    elif ctx.force_naive() or fail_fast:
        return _naive_fundamental_matrix_regular(ordinary, *args)
    else:
        try:
            return _naive_fundamental_matrix_regular(ordinary, *args)
        except accuracy.PrecisionError as exn:
            try:
                logger.info("not enough precision, trying binary splitting "
                            "as a fallback")
                return binary_splitting.fundamental_matrix_regular(*args)
            except NotImplementedError:
                logger.info("unable to use binary splitting")
                raise exn

def _naive_fundamental_matrix_regular(ordinary, *args):
    if ordinary:
        return naive_sum.fundamental_matrix_ordinary(*args)
    else:
        return naive_sum.fundamental_matrix_regular(*args)

def _process_path(dop, path, ctx):

    if not isinstance(path, Path):
        path = Path(path, dop)

    if not ctx.assume_analytic:
        path.check_singularity()
    if not all(x.is_regular() for x in path.vert):
        raise NotImplementedError("analytic continuation through irregular "
                                  "singular points is not supported")

    # FIXME: prevents the reuse of points...
    if ctx.keep == "all":
        for v in path.vert:
            v.options['keep_value'] = True
    elif ctx.keep == "last":
        for v in path.vert:
            v.options['keep_value'] = False
        path.vert[-1].options['keep_value'] = True

    if ctx.assume_analytic:
        path = path.bypass_singularities()
        path.check_singularity()

    path = path.subdivide()
    path.check_singularity()
    path.check_convergence()

    ctx.path = path # TBI

    return path

def analytic_continuation(dop, path, eps, ctx=default_ctx, ini=None, post=None):
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

    res = []
    path_mat = identity_matrix(ZZ, dop.order())
    def store_value_if_wanted(point):
        if point.options.get('keep_value'):
            value = path_mat
            if ini is not None:  value = value*ini
            if post is not None: value = post(point.value)*value
            res.append((point.value, value))
    store_value_if_wanted(path.vert[0])
    for step in path:
        step_mat = step_transition_matrix(dop, step, eps1, ctx=ctx)
        path_mat = step_mat*path_mat
        store_value_if_wanted(step.end)
    cm = sage.structure.element.get_coercion_model()
    real = (rings.RIF.has_coerce_map_from(dop.base_ring().base_ring())
            and all(v.is_real() for v in path.vert))
    OutputIntervals = cm.common_parent(
            utilities.ball_field(eps, real),
            *[mat.base_ring() for pt, mat in res])
    return [(pt, mat.change_ring(OutputIntervals)) for pt, mat in res]

def normalize_post_transform(dop, post_transform):
    if post_transform is None:
        post_transform = dop.parent().one()
    else:
        _, post_transform = canonical_coercion(dop, post_transform)
    return post_transform % dop
