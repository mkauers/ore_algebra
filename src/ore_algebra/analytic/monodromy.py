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
    # XXX should use binary splitting when the indicial polynomial has large
    # dispersion
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
# Generators of the monodromy group
##############################################################################

# XXX we use assume_analytic to bypass singularities in cases where the
# transition matrix may not actually be analytic but we do not care which path
# is taken

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

class PointWithMonodromyData(path.Point):

    def __init__(self, point, dop, *, want_self=False, want_conj=False):
        super().__init__(point, dop)
        self.want_self: bool = want_self
        self.want_conj: bool = want_conj
        self.local_monodromy = None
        self.polygon = None
        self.done: bool = False

def _try_merge_conjugate_singularities(dop, sing, base, todo):
    if any(c not in QQ for pol in dop for c in pol):
        return False
    need_conjugates = False
    sgn = 1 if QQbar.coerce(base.value).imag() >= 0 else -1
    for x in sing:
        if sgn*x.imag() < 0:
            need_conjugates = True
            del todo[x]
            xconj = x.conjugate()
            if xconj not in todo:
                todo[xconj] = LocalMonodromyData()
            todo[xconj].want_conj = True
    return need_conjugates

def _spanning_tree(base, verts):
    graph = Graph([list(verts), lambda x, y: x is not y])
    def length(edge):
        x, y, _ = edge
        return abs(CC(x.value) - CC(y.value))
    tree = graph.min_spanning_tree(length)
    tree = Graph(tree)
    tree.add_vertex(base)
    return tree

def _closest_unsafe(lst, x):
    x = CC(x.value)
    return min(enumerate(lst), key=lambda y: abs(CC(y[1].value) - x))

def _extend_path_mat(dop, path_mat, x, y, eps, matprod):
    anchor_index_x, anchor_x = _closest_unsafe(x.polygon, y)
    anchor_index_y, anchor_y = _closest_unsafe(y.polygon, x)
    bypass_mat_x = matprod(x.local_monodromy[:anchor_index_x])
    bypass_mat_y = matprod(y.local_monodromy[anchor_index_y:]
                           if anchor_index_y > 0
                           else [])
    anchor_x.options["keep_value"] = False # XXX bugware
    anchor_y.options["keep_value"] = True
    edge_mat = dop.numerical_transition_matrix([anchor_x, anchor_y],
                                                eps, assume_analytic=True)
    new_path_mat = bypass_mat_y*edge_mat*bypass_mat_x*path_mat
    assert isinstance(new_path_mat, Matrix_complex_ball_dense)
    return new_path_mat

def _monodromy_matrices(dop, base, eps=1e-16, sing=None):
    dop = DifferentialOperator(dop)
    base = QQbar.coerce(base)
    eps = RBF(eps)
    if sing is None:
        sing = dop._singularities(QQbar)
    else:
        sing = [QQbar.coerce(s) for s in sing]

    todo = {x: PointWithMonodromyData(x, dop, want_self=True, want_conj=False)
            for x in sing}
    base = todo.setdefault(base, PointWithMonodromyData(base, dop))
    if not base.is_regular():
        raise ValueError("irregular singular base point")
    # If the coefficients are rational, reduce to handling singularities in the
    # same half-plane as the base point.
    need_conjugates = _try_merge_conjugate_singularities(dop, sing, base, todo)

    Scalars = ComplexBallField(utilities.prec_from_eps(eps))
    id_mat = matrix.identity_matrix(Scalars, dop.order())
    def matprod(elts):
        return prod(reversed(elts), id_mat)

    for key, point in list(todo.items()):
        # We could call _local_monodromy_loop() if point is irregular, but
        # delaying it may allow us to start returning results earlier.
        if point.is_regular():
            mon, scalar = _formal_monodromy_naive(dop.shift(point), Scalars)
            if scalar:
                # No need to compute the connection matrices then!
                # XXX When we do need them, though, it would be better to get
                # the formal monodromy as a byproduct of their computation.
                if point.want_self:
                    yield mon
                if point.want_conj:
                    logger.info("Computing local monodromy around %s by "
                                "complex conjugation", point.conjugate())
                    yield ~mon.conjugate()
                if point is not base:
                    del todo[key]
                    continue
            point.local_monodromy = [mon]
            point.polygon = [point]

    if need_conjugates:
        base_conj_mat = dop.numerical_transition_matrix(
                            [base, base.conjugate()], eps, assume_analytic=True)
        def conjugate_monodromy(point, mat):
            return ~base_conj_mat*~mat.conjugate()*base_conj_mat

    tree = _spanning_tree(base, todo.values())

    def dfs(x, path, path_mat):

        logger.info("Computing local monodromy around %s via %s", x, path)

        local_mat = matprod(x.local_monodromy)
        based_mat = (~path_mat)*local_mat*path_mat

        if x.want_self:
            yield based_mat
        if x.want_conj:
            logger.info("Computing local monodromy around %s by complex "
                        "conjugation", point.conjugate())
            yield conjugate_monodromy(point, based_mat)

        x.done = True

        for y in tree.neighbors(x):
            if y.done:
                continue
            if y.local_monodromy is None:
                y.polygon, y.local_monodromy = _local_monodromy_loop(dop, y, eps)
            new_path_mat = _extend_path_mat(dop, path_mat, x, y, eps, matprod)
            yield from dfs(y, path + [y], new_path_mat)

    yield from dfs(base, [base], id_mat)

def monodromy_matrices(dop, base, eps=1e-16, sing=None):
    r"""
    Compute generators of the monodromy group of ``dop`` with base point
    ``base``.

    OUTPUT:

    A list of matrices, each encoding the analytic continuation of solutions
    along a simple positive loop based in ``base`` around a singular point
    of ``dop`` (with no other singular point inside the loop). The precise
    choice of elements of the fundamental group corresponding to each matrix
    (position with respect to the other singular points, order) is unspecified.

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

        sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
        sage: mon = monodromy_matrices(dop, 0)
        sage: mon[0]
        [ [1.000000000000...] + [+/- ...]*I [3.141592653589...] + [+/- ...]*I]
        [           [+/- ...] + [+/- ...]*I [1.000000000000...] + [+/- ...]*I]
        sage: mon[1]
        [ [1.000000000000...] + [+/- ...]*I [-3.14159265358...] + [+/- ...]*I]
        [           [+/- ...] + [+/- ...]*I [1.000000000000...] + [+/- ...]*I]

    The base point can also be a singular point::

        sage: monodromy_matrices(Dx*x*Dx, 0)
        [
        [     1.0000000000000000                    0]
        [[6.28318530717958...]*I   1.0000000000000000]
        ]
    """
    return list(_monodromy_matrices(dop, base, eps, sing))

def _test_monodromy_matrices():
    r"""
    TESTS::

        sage: from ore_algebra.analytic.monodromy import _test_monodromy_matrices
        sage: _test_monodromy_matrices()
    """
    from sage.all import matrix
    from ore_algebra import DifferentialOperators
    Dops, x, Dx = DifferentialOperators()

    h = QQ(1)/2
    i = QQi.gen()

    def norm(m):
        return sum(c.abs()**2 for c in m.list()).sqrtpos()

    mon = monodromy_matrices((x**2+1)*Dx-1, QQ(1000000))
    assert norm(mon[0] - CBF(pi).exp()) < RBF(1e-10)
    assert norm(mon[1] - CBF(-pi).exp()) < RBF(1e-10)

    mon = monodromy_matrices((x**2-1)*Dx-1, QQ(0))
    assert all(m == -1 for m in mon)

    dop = (x**2 + 1)*Dx**2 + 2*x*Dx
    mon = monodromy_matrices(dop, QQbar(i+1)) # mon[0] <--> i
    assert norm(mon[0] - matrix(CBF, [[1,pi*(1+2*i)], [0,1]])) < RBF(1e-10)
    assert norm(mon[1] - matrix(CBF, [[1,-pi*(1+2*i)], [0,1]])) < RBF(1e-10)
    mon = monodromy_matrices(dop, QQbar(-i+1)) # mon[0] <--> -i
    assert norm(mon[0] - matrix(CBF, [[1,pi*(-1+2*i)], [0,1]])) < RBF(1e-10)
    assert norm(mon[1] - matrix(CBF, [[1,pi*(1-2*i)], [0,1]])) < RBF(1e-10)
    mon = monodromy_matrices(dop, QQbar(i)) # mon[0] <--> i
    assert norm(mon[0] - matrix(CBF, [[1,0], [2*pi*i,1]])) < RBF(1e-10)
    assert norm(mon[1] - matrix(CBF, [[1,0], [-2*pi*i,1]])) < RBF(1e-10)
    mon = monodromy_matrices(dop, QQbar(i), sing=[QQbar(i)])
    assert len(mon) == 1
    assert norm(mon[0] - matrix(CBF, [[1,0], [2*pi*i,1]])) < RBF(1e-10)
    mon = monodromy_matrices(dop, QQbar(i), sing=[QQbar(-i)])
    assert len(mon) == 1
    assert norm(mon[0] - matrix(CBF, [[1,0], [-2*pi*i,1]])) < RBF(1e-10)
    mon = monodromy_matrices(dop, QQbar(-i), sing=[QQbar(i)])
    assert len(mon) == 1
    assert norm(mon[0] - matrix(CBF, [[1,0], [-2*pi*i,1]])) < RBF(1e-10)
    mon = monodromy_matrices(dop, QQbar(i), sing=[])
    assert mon == []

    dop = (x**2+1)*(x**2-1)*Dx**2 + 1
    mon = monodromy_matrices(dop, QQ(0), sing=[QQ(1),QQbar(i)])
    m0 = dop.numerical_transition_matrix([0,i+1,2*i,i-1,0])
    assert norm(m0 - mon[0]) < RBF(1e-10)
    m1 = dop.numerical_transition_matrix([0,1-i,2,1+i,0])
    assert norm(m1 - mon[1]) < RBF(1e-10)

    dop = x*(x-3)*(x-4)*(x**2 - 6*x + 10)*Dx**2 - 1
    mon = monodromy_matrices(dop, QQ(-1))
    m0 = dop.numerical_transition_matrix([-1,-i,1,i,-1])
    assert norm(m0 - mon[0]) < RBF(1e-10)
    m1 = dop.numerical_transition_matrix([-1,i/2,3-i/2,3+h,3+i/2,i/2,-1])
    assert norm(m1 - mon[1]) < RBF(1e-10)
    m2 = dop.numerical_transition_matrix([-1,i/2,3+i/2,4-i/2,4+h,4+i/2,i/2,-1])
    assert norm(m2 - mon[2]) < RBF(1e-10)
    m3 = dop.numerical_transition_matrix([-1,3+i+h,3+2*i,-1])
    assert norm(m3 - mon[3]) < RBF(1e-10)
    m4 = dop.numerical_transition_matrix([-1,3-2*i,3-i+h,3-i/2,-1])
    assert norm(m4 - mon[4]) < RBF(1e-10)

    dop = (x-i)**2*(x+i)*Dx - 1
    mon = monodromy_matrices(dop, 0)
    assert norm(mon[0] + i) < RBF(1e-10)
    assert norm(mon[1] - i) < RBF(1e-10)

    dop = (x-i)**2*(x+i)*Dx**2 - 1
    mon = monodromy_matrices(dop, 0)
    m0 = dop.numerical_transition_matrix([0,i+1,2*i,i-1,0])
    assert norm(m0 - mon[0]) < RBF(1e-10)
    m1 = dop.numerical_transition_matrix([0,-i-1,-2*i,-i+1,0])
    assert norm(m1 - mon[1]) < RBF(1e-10)
