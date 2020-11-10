# -*- coding: utf-8 - vim: tw=80
r"""
Monodromy matrices
"""

# Copyright 2017, 2018, 2019 Marc Mezzarobba
# Copyright 2017, 2018, 2019 Centre national de la recherche scientifique
# Copyright 2017, 2018 Université Pierre et Marie Curie
# Copyright 2019 Sorbonne Université
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

import logging

import sage.matrix.special as matrix

from sage.graphs.graph import Graph
from sage.matrix.matrix_complex_ball_dense import Matrix_complex_ball_dense
from sage.misc.misc_c import prod
from sage.rings.all import (CC, CBF, ComplexBallField, QQ, QQbar,
        QuadraticField, RBF)
from sage.symbolic.all import I, pi

from . import analytic_continuation as ancont, local_solutions, path, utilities

from .differential_operator import DifferentialOperator

logger = logging.getLogger(__name__)

QQi = QuadraticField(-1, 'i')

def _identity_matrix(point, eps):
    Scalars = ComplexBallField(utilities.prec_from_eps(eps))
    return matrix.identity_matrix(Scalars, point.dop.order())

def _local_monodromy_loop(dop, x, eps):
    polygon = path.polygon_around(x)
    n = len(polygon)
    mats = []
    for i in range(n):
        step = [polygon[i], polygon[(i+1)%n]]
        logger.debug("center = %s, step = %s", x, step)
        step[0].options['keep_value'] = False # XXX bugware
        step[1].options['keep_value'] = True
        mat = x.dop.numerical_transition_matrix(step, eps, assume_analytic=True)
        prec = utilities.prec_from_eps(eps)
        assert all(c.accuracy() >= prec//2 or c.above_abs()**2 <= eps
                   for c in mat.list())
        mats.append(mat)
    return polygon, mats

def _local_monodromy_formal(dop, x, eps):
    assert x.is_regular()
    base = path.polygon_around(x, size=1)[0] # TBI?
    rows = x.dop.order()
    step_in = path.Step(base, x)
    mat_in = ancont.step_transition_matrix(dop, step_in, eps)
    step_out = path.Step(x, base, branch=(1,))
    mat_out = ancont.step_transition_matrix(dop, step_out, eps)
    return [base, x], [mat_in, mat_out]

def _local_monodromy(dop, x, eps, algorithm):
    if x.is_ordinary():
        return [x], [_identity_matrix(x, eps)]
    elif x.is_regular() and algorithm == "connect": # and alg deg not too large?
        return _local_monodromy_formal(dop, x, eps)
    else:
        return _local_monodromy_loop(dop, x, eps)

def _closest_unsafe(lst, x):
    x = CC(x.value)
    return min(enumerate(lst), key=lambda y: abs(CC(y[1].value) - x))

def _sing_tree(dop, base):
    sing = dop._singularities(QQbar, include_apparent=False)
    sing = [path.Point(x, dop) for x in sing]
    verts = [base] + sing
    graph = Graph([verts, lambda x, y: x is not y])
    def length(edge):
        x, y, _ = edge
        return abs(CC(x.value) - CC(y.value))
    tree = graph.min_spanning_tree(length)
    return Graph(tree)

def monodromy_matrices(dop, base, eps=1e-16, algorithm="connect"):
    r"""
    Compute generators of the monodromy group of ``dop`` with base point
    ``base``.

    OUTPUT:

    A list of matrices, each encoding the analytic continuation of solutions
    along a simple positive loop based in ``base`` around a singular point
    of ``dop`` (with no other singular point inside the loop). Identity matrices
    may be omitted. The precise choice of elements of the fundamental group
    corresponding to each matrix (position with respect to the other
    singular points, order) are unspecified.

    EXAMPLES::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.monodromy import monodromy_matrices
        sage: Dops, x, Dx = DifferentialOperators()

        sage: monodromy_matrices(Dx*x*Dx, 1)
        [
        [  1.0000...  [6.2831853071795...]*I]
        [          0               1.0000...]
        ]

        sage: monodromy_matrices(Dx*x*Dx, 1, algorithm="loop")
        [
        [[1.0000...] + [+/- ...]*I  [+/- ...] + [6.283185307179...]*I]
        [  [+/- ...] + [+/- ...]*I           [1.0000...] + [+/- ...]*I]
        ]

        sage: monodromy_matrices(Dx*x*Dx, 1/2)
        [
        [ [1.0000...] + [+/- ...]*I  [+/- ...] + [3.1415926535897...]*I]
        [   [+/- ...] + [+/- ...]*I           [1.0000...] + [+/- ...]*I]
        ]
    """

    dop = DifferentialOperator(dop)
    base = path.Point(base, dop)
    if not base.is_regular():
        raise ValueError("base point must be regular")
    eps = RBF(eps)
    if not (algorithm == "connect" or algorithm == "loop"):
        raise ValueError("unknown algorithm")

    id_mat = _identity_matrix(base, eps)
    def matprod(elts):
        return prod(reversed(elts), id_mat)

    # TODO: filter out the factors of the leading coefficient that correspond to
    # apparent singularities (may require improvements to the analytic
    # continuation code)

    tree = _sing_tree(dop, base)
    polygon_base, local_monodromy_base = _local_monodromy(dop, base, eps, algorithm)
    result = [] if base.is_ordinary() else local_monodromy_base

    def dfs(x, path, path_mat, polygon_x, local_monodromy_x):
        x.seen = True
        for y in [z for z in tree.neighbors(x) if not z.seen]:

            logger.info("Computing local monodromy around %s via %s", y, path)

            polygon_y, local_monodromy_y = _local_monodromy(dop, y, eps, algorithm)

            anchor_index_x, anchor_x = _closest_unsafe(polygon_x, y)
            anchor_index_y, anchor_y = _closest_unsafe(polygon_y, x)
            bypass_mat_x = matprod(local_monodromy_x[:anchor_index_x])
            if anchor_index_y > 0:
                bypass_mat_y = matprod(local_monodromy_y[anchor_index_y:])
            else:
                bypass_mat_y = id_mat
            anchor_x.options["keep_value"] = False # XXX bugware
            anchor_y.options["keep_value"] = True
            edge_mat = dop.numerical_transition_matrix([anchor_x, anchor_y], eps, assume_analytic=True)
            new_path_mat = bypass_mat_y*edge_mat*bypass_mat_x*path_mat
            assert isinstance(new_path_mat, Matrix_complex_ball_dense)

            local_mat = matprod(local_monodromy_y)
            based_mat = (~new_path_mat)*local_mat*new_path_mat
            result.append(based_mat)

            dfs(y, path + [y], new_path_mat, polygon_y, local_monodromy_y)

    for x in tree:
        x.seen = False
    dfs(base, [base], id_mat, polygon_base, local_monodromy_base)
    return result

def _tests():
    r"""
    TESTS::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.monodromy import monodromy_matrices
        sage: Dops, x, Dx = DifferentialOperators()

        sage: def norm(m):
        ....:     return sum(c.abs()**2 for c in m.list()).sqrtpos()

        sage: dop = x*(x-3)*(x-4)*(x^2 - 6*x + 10)*Dx^2 - 1
        sage: mon = monodromy_matrices(dop, -1)

        sage: mon[0]
        [[1.00000000000...] + [-0.0519404348206...]*I          [+/- ...] + [-0.052049785176...]*I]
        [         [+/- ...] + [0.0518313141967...]*I  [1.00000000000...] + [0.0519404348206...]*I]
        sage: m0 = dop.numerical_transition_matrix([-1, -i, 1, i, -1])
        sage: norm(m0 - mon[0]) < RBF(1e-10)
        True

        sage: m1 = dop.numerical_transition_matrix([-1, -i, 3-i, 3+1/2, 3+i, 2-i, -1])
        sage: norm(m1 - mon[1]) < RBF(1e-10)
        True

        sage: m2 = dop.numerical_transition_matrix([-1, -3-2*i, 4-i, 3-i/2, -1])
        sage: m3 = dop.numerical_transition_matrix([-1, -i, 5, 4+i, 3-i, -i, -1])
        sage: m4 = dop.numerical_transition_matrix([-1, -3-i/2, 3+1/2, 3+1/2+i, 3+2*i, 3-1/2+i, 3+1/2, 3-i/2, -1])

        sage: any(norm(m2 - mon[i]) < RBF(1e-10) and norm(m3 - mon[j]) < RBF(1e-10)
        ....:                                    and norm(m4 - mon[k]) < RBF(1e-10)
        ....:     for (i, j, k) in Permutations([2, 3, 4]))
        True
    """
    pass
