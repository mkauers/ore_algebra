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
from sage.functions.all import exp
from sage.symbolic.all import pi, SR

from . import analytic_continuation as ancont, local_solutions, path, utilities

from .differential_operator import DifferentialOperator
from .local_solutions import LocalBasisMapper, log_series

logger = logging.getLogger(__name__)

QQi = QuadraticField(-1, 'i')

##############################################################################
# Formal monodromy
##############################################################################

def formal_monodromy(dop, sing, ring=SR):
    r"""
    Compute the formal monodromy matrix of a system of fundamental solutions at
    a given singular point.

    INPUT:

    - ``dop`` - a differential operator,
    - ``sing`` - one of its singular points (typically given as an element of
      ``QQbar`` or of a parent with a coercion to ``QQbar``).

    OUTPUT:

    The formal monodromy matrix around ``sing`` in the basis given by
    ``dop.local_basis_expansions(sing)``.

    EXAMPLES::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.monodromy import formal_monodromy
        sage: Dops, x, Dx = DifferentialOperators()

        sage: dop = Dx*x*Dx
        sage: dop.local_basis_expansions(0)
        [log(x), 1]
        sage: formal_monodromy(dop, 0)
        [       1      0]
        [(2*I*pi)      1]

        sage: mon = formal_monodromy(dop, 0, CBF)
        sage: tm = dop.numerical_transition_matrix([0,1])
        sage: tm*mon*~tm
        [ 1.0000... [6.2831853071795...]*I]
        [         0      1.000000000000000]
        sage: dop.numerical_transition_matrix([1, i, -1, -i, 1])
        [ [1.0000...] + [+/- ...]*I [+/- ...] + [6.2831853071795...]*I]
        [   [+/- ...] + [+/- ...]*I          [1.0000...] + [+/- ...]*I]

        sage: dop = ((x*Dx)^2 - 1/4)^2*((x*Dx)-3/2)
        sage: dop.local_basis_expansions(0)
        [x^(-1/2)*log(x), x^(-1/2), x^(1/2)*log(x), x^(1/2), x^(3/2)]
        sage: formal_monodromy(dop, 0)
        [       -1         0         0         0         0]
        [-(2*I*pi)        -1         0         0         0]
        [        0         0        -1         0         0]
        [        0         0 -(2*I*pi)        -1         0]
        [        0         0         0         0        -1]

        sage: dop = (x*Dx-1/2+x^2)^2*(x*Dx-5/2)
        sage: dop.local_basis_expansions(0)
        [x^(1/2)*log(x) + 1/2*x^(5/2)*log(x)^2 - 1/2*x^(5/2)*log(x) - 1/8*x^(9/2)*log(x) + 1/8*x^(9/2),
        x^(1/2) + x^(5/2)*log(x) - 1/8*x^(9/2),
        x^(5/2)]
        sage: formal_monodromy(dop, 0)
        [               -1                 0                 0]
        [        -(2*I*pi)                -1                 0]
        [-(-I*pi - 2*pi^2)         -(2*I*pi)                -1]

    TESTS::

        sage: formal_monodromy(Dx*x*Dx, 1)
        [1 0]
        [0 1]
        sage: formal_monodromy((x*Dx)^2 - 2, 0)
        [e^(-1.414213562373095?*(2*I*pi))                                0]
        [                               0  e^(1.414213562373095?*(2*I*pi))]

        sage: from ore_algebra.analytic.monodromy import _test_formal_monodromy
        sage: _test_formal_monodromy(Dx*x*Dx)
        sage: _test_formal_monodromy((x*Dx-1/2+x^2)^2*(x*Dx-5/2))
        sage: _test_formal_monodromy((x*Dx)^2 - 2)
    """
    dop = DifferentialOperator(dop)
    sing = path.Point(sing, dop)
    ldop = dop.shift(sing)
    mon, _ = _formal_monodromy_naive(ldop, ring)
    return mon

def _formal_monodromy_naive(dop, ring):

    class Mapper(LocalBasisMapper):
        def fun(self, ini):
            order = max(s for s, _ in self.shifts) + 1
            ser = log_series(ini, self.shifted_bwrec, order)
            return {s: ser[s] for s, _ in self.shifts}

    crit = Mapper(dop).run()
    return _formal_monodromy_from_critical_monomials(crit, ring)

def _formal_monodromy_from_critical_monomials(critical_monomials, ring):
    r"""
    Compute the formal monodromy matrix of the canonical system of fundamental
    solutions at the origin.

    INPUT:

    - ``critical_monomials``: list of ``FundamentalSolution`` objects ``sol``
      such that, if ``sol = z^(λ+n)·(1 + Õ(z)`` where ``λ`` is the leftmost
      valuation of a group of solutions and ``s`` is another shift of ``λ``
      appearing in the basis, then ``sol.value[s]`` contains the list of
      coefficients of ``z^(λ+s)·log(z)^k/k!``, ``k = 0, 1, ...`` in ``sol``

    - ``ring``

    OUTPUT:

    - the formal monodromy matrix, with coefficients in ``ring``

    - a boolean flag indicating whether the local monodromy is scalar (useful
      when ``ring`` is an inexact ring!)
    """

    mat = matrix.matrix(ring, len(critical_monomials))
    twopii = 2*pi*QQbar(QQi.gen())
    expo0 = critical_monomials[0].leftmost
    scalar = True

    for j, jsol in enumerate(critical_monomials):

        for i, isol in enumerate(critical_monomials):
            if isol.leftmost != jsol.leftmost:
                continue
            for k, c in enumerate(jsol.value[isol.shift]):
                delta = k - isol.log_power
                if c.is_zero():
                    continue
                if delta >= 0:
                    # explicit conversion sometimes necessary (Sage bug #31551)
                    mat[i,j] += ring(c)*twopii**delta/delta.factorial()
                if delta >= 1:
                    scalar = False

        expo = jsol.leftmost
        if expo != expo0:
            scalar = False
        if expo.parent() is QQ:
            eigv = ring(QQbar.zeta(expo.denominator())**expo.numerator())
        else:
            # conversion via QQbar seems necessary with some number fields
            eigv = twopii.mul(QQbar(expo), hold=True).exp(hold=True)
        eigv = ring(eigv)
        if ring is SR:
            _rescale_col_hold_nontrivial(mat, j, eigv)
        else:
            mat.rescale_col(j, eigv)

    return mat, scalar

def _rescale_col_hold_nontrivial(mat, j, c):
    for i in range(mat.nrows()):
        if mat[i,j].is_zero():
            pass
        elif mat[i,j].is_one():
            mat[i,j] = c
        else:
            mat[i,j] = mat[i,j].mul(c, hold=True)

def _test_formal_monodromy(dop):
    i = QQi.gen()
    ref = dop.numerical_transition_matrix([1, i, -1, -i, 1])
    tmat = dop.numerical_transition_matrix([0,1])
    fmon = formal_monodromy(dop, 0, CBF)
    mon = tmat*fmon*~tmat
    assert all(c.contains_zero() and c.rad() < 1e-10 for c in (ref - mon).list())

##############################################################################
# Local monodromy based close to the singular point
##############################################################################

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
    # TODO: get the critical coefficients as a byproduct of the computation of
    # the transition matrix instead
    ldop = dop.shift(x)
    Scalars = ComplexBallField(utilities.prec_from_eps(eps))
    formal, _ = _formal_monodromy_naive(ldop, Scalars)
    return [x], [formal]

def _local_monodromy(dop, x, eps):
    if x.is_ordinary():
        return [x], [_identity_matrix(x, eps)]
    elif x.is_regular():
        # and alg deg not too large? naive summation with inexact recurrences
        # works decently even at algebraic points...
        return _local_monodromy_formal(dop, x, eps)
    else:
        return _local_monodromy_loop(dop, x, eps)

def _closest_unsafe(lst, x):
    x = CC(x.value)
    return min(enumerate(lst), key=lambda y: abs(CC(y[1].value) - x))

##############################################################################
# Generators of the monodromy group
##############################################################################

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

def monodromy_matrices(dop, base, eps=1e-16):
    r"""
    Compute generators of the monodromy group of ``dop`` with base point
    ``base``.

    OUTPUT:

    A list of matrices, each encoding the analytic continuation of solutions
    along a simple positive loop based in ``base`` around a singular point
    of ``dop`` (with no other singular point inside the loop). Identity matrices
    may be omitted. The precise choice of elements of the fundamental group
    corresponding to each matrix (position with respect to the other
    singular points, order) is unspecified.

    EXAMPLES::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.monodromy import monodromy_matrices
        sage: Dops, x, Dx = DifferentialOperators()

        sage: monodromy_matrices(Dx*x*Dx, 1)
        [
        [  1.0000...  [6.2831853071795...]*I]
        [          0               1.0000...]
        ]

        sage: monodromy_matrices(Dx*x*Dx, 1/2)
        [
        [ 1.0000...  [+/- ...] + [3.1415926535897...]*I]
        [         0                           1.0000...]
        ]
    """

    dop = DifferentialOperator(dop)
    base = path.Point(base, dop)
    if not base.is_regular():
        raise ValueError("base point must be regular")
    eps = RBF(eps)

    id_mat = _identity_matrix(base, eps)
    def matprod(elts):
        return prod(reversed(elts), id_mat)

    tree = _sing_tree(dop, base)
    polygon_base, local_monodromy_base = _local_monodromy(dop, base, eps)
    result = [] if base.is_ordinary() else local_monodromy_base

    def dfs(x, path, path_mat, polygon_x, local_monodromy_x):

        x.seen = True

        for y in tree.neighbors(x):

            if y.seen:
                continue

            logger.info("Computing local monodromy around %s via %s", y, path)

            polygon_y, local_monodromy_y = _local_monodromy(dop, y, eps)

            anchor_index_x, anchor_x = _closest_unsafe(polygon_x, y)
            anchor_index_y, anchor_y = _closest_unsafe(polygon_y, x)
            bypass_mat_x = matprod(local_monodromy_x[:anchor_index_x])
            if anchor_index_y > 0:
                bypass_mat_y = matprod(local_monodromy_y[anchor_index_y:])
            else:
                bypass_mat_y = id_mat
            anchor_x.options["keep_value"] = False # XXX bugware
            anchor_y.options["keep_value"] = True
            edge_mat = dop.numerical_transition_matrix([anchor_x, anchor_y],
                                                      eps, assume_analytic=True)
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

        sage: m1 = dop.numerical_transition_matrix([-1, i/2, 3-i/2, 3+1/2, 3+i/2, i/2, -1])
        sage: norm(m1 - mon[1]) < RBF(1e-10)
        True

        sage: m2 = dop.numerical_transition_matrix([-1, i/2, 3+i/2, 4-i/2, 4+1/2, 4+i/2, i/2, -1])
        sage: norm(m2 - mon[2]) < RBF(1e-10)
        True

        sage: m3 = dop.numerical_transition_matrix([-1, i/2, 3+i/2, 3+1/2, 3-i-1/2, 3-i-i/2, 3+1/2, 3+i/2, i/2, -1])
        sage: norm(m3 - mon[3]) < RBF(1e-10)
        True

        sage: m4 = dop.numerical_transition_matrix([-1, 3+i+1/2, 3+2*i, -1])
        sage: norm(m4 - mon[4]) < RBF(1e-10)
        True
    """
    pass
