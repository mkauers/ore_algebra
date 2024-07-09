# vim: tw=80
r"""
Monodromy matrices

TESTS::

    sage: from ore_algebra import *
    sage: from ore_algebra.analytic.monodromy import monodromy_matrices
    sage: Dops, z, Dz = DifferentialOperators()

Thanks to Alexandre Goyer for this example, which used to crash because of a bug
in the handling of algebraic numbers in the binary splitting code::

    sage: dop = z^2*(14*z-3)*(23*z^3+128*z^2+128*z-256)*Dz^3+2*z*(1449*z^4+5255*z^3+1744*z^2-2208*z+672)*Dz^2+2*(2898*z^4+5513*z^3-1081*z^2-414*z+48)*Dz+1932*z^3+344*z^2-690*z-12
    sage: dop = dop.annihilator_of_composition(z-1)
    sage: monodromy_matrices(dop, 0, algorithm="binsplit", eps=1e-20)[-1].trace()
    [2.00000000000...] + [1.00000000000...]*I
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

import collections
import logging

import sage.matrix.special as matrix

from sage.graphs.graph import Graph
from sage.matrix.matrix_complex_ball_dense import Matrix_complex_ball_dense
from sage.misc.cachefunc import cached_method
from sage.misc.misc_c import prod
from sage.rings.all import (CC, CBF, ComplexBallField, QQ, QQbar,
        QuadraticField, RBF)
from sage.symbolic.all import pi, SR

from . import path, utilities

from .context import Context
from .differential_operator import DifferentialOperator
from .polynomial_root import PolynomialRoot

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

        sage: from ore_algebra.analytic.monodromy import formal_monodromy
        sage: dop = (x^9 - 46656*x^3)*Dx^4 + (34*x^8 + 93312*x^2)*Dx^3 + (385*x^7 - 139968*x)*Dx^2 + (1695*x^6 + 139968)*Dx + 2401*x^5
        sage: a = QQbar.polynomial_root(dop.leading_coefficient(), CIF(3, RIF(5.1, 5.2)))
        sage: formal_monodromy(dop, a)
        [                                          1  0  0  0]
        [                            (67/17496*I*pi)  1  0  0]
        [   (-9347/5038848*I*pi*(3*I*sqrt(1/3) - 1))  0  1  0]
        [(-56135/45349632*I*pi*(-3*I*sqrt(1/3) - 1))  0  0  1]
    """
    dop = DifferentialOperator(dop)
    sing = path.Point(sing, dop)
    # XXX should use binary splitting when the indicial polynomial has large
    # dispersion
    crit = sing.local_basis_structure()
    mor = ring.hom(ring) # we need a Morphism but don't known the source ring
    mon, _ = _formal_monodromy_from_critical_monomials(crit, mor)
    return mon

def _formal_monodromy_from_critical_monomials(critical_monomials, mor):
    r"""
    Compute the formal monodromy matrix of the canonical system of fundamental
    solutions at the origin.

    INPUT:

    - ``critical_monomials``: critical monomials in the format output by
      :meth:`ore_algebra.analytic.path.Point.local_basis_structure`

    - ``mor``: a morphism from the parent of critical monomials to a ring
      suitable for representing the entries of the formal monodromy matrix
      (typically ``CBF`` or ``SR``)

    OUTPUT:

    - the formal monodromy matrix, with coefficients in the codomain of ``mor``

    - a boolean flag indicating whether the local monodromy is scalar (useful
      when the target is an inexact ring!)
    """

    ring = mor.codomain()
    mat = matrix.matrix(ring, len(critical_monomials))
    twopii = 2*pi*QQbar(QQi.gen())
    expo0 = critical_monomials[0].leftmost
    scalar = True

    for j, jsol in enumerate(critical_monomials):

        for i, isol in enumerate(critical_monomials):
            if isol.leftmost is not jsol.leftmost:
                continue
            for k, c in enumerate(jsol.value[isol.shift]):
                delta = k - isol.log_power
                if c.is_zero():
                    continue
                if delta >= 0:
                    # explicit conversion sometimes necessary (Sage bug #31551)
                    mat[i,j] += mor(c)*twopii**delta/delta.factorial()
                if delta >= 1:
                    scalar = False

        expo = jsol.leftmost
        if expo is not expo0:
            scalar = False
        if isinstance(ring, ComplexBallField): # optimization
            eigv = (ring(expo)*2).exppii()
        elif expo.is_rational():
            rat = expo.as_number_field_element()
            eigv = ring(QQbar.zeta(rat.denominator())**rat.numerator())
        else:
            # conversion via QQbar seems necessary with some number fields
            # XXX We should actually follow expo along mor, but this is not easy
            # to do with the current code structure, see comment in
            # _monodromy_matrices().
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

def _local_monodromy_loop(x, eps, ctx, effort=3):
    r"""
    TESTS::

        sage: from ore_algebra.analytic.examples.misc import large_irregular_dop
        sage: from ore_algebra.analytic.monodromy import monodromy_matrices
        sage: monodromy_matrices(large_irregular_dop, 53/10, 1e-1000, # not tested
        ....:                    sing=[21/4], bounds_prec=128)[0][-1,-1]
        [3.7975339214908450838528472219121833930467139062244128500652173291400519760900043930710428392780611358292087744635087753829152392259557126284093959961212186601...e+1158 +/- ...] +
        [-1.0907799857818368151501817928994979524542825249884040984277371301954332475924993671608411858764307809823591665306474900630885106316087831863249971763581367813...e+1158 +/- ...]*I
    """
    polygon = path.polygon_around(x)
    n = len(polygon)
    mats = []
    for i in range(n):
        step = [polygon[i], polygon[(i+1)%n]]
        logger.debug("center = %s, step = %s", x, step)
        step[0].options['store_value'] = False # XXX bugware
        step[1].options['store_value'] = True
        for attempt in range(effort):
            mat = x.dop.numerical_transition_matrix(step, eps, ctx=ctx)
            prec = utilities.prec_from_eps(eps)
            accuracy = min((c.accuracy() for c in mat.list()
                                         if c.above_abs()**2 > eps),
                           default=prec)
            if accuracy >= prec//2:
                break
            eps /= (1 << (prec - accuracy))
            logger.debug("decreasing eps to %s...", eps)
        mats.append(mat)
    return polygon, mats

class TodoItem:

    def __init__(self, alg, dop, *, want_self=False, want_conj=False):
        self.alg = alg
        self._dop = dop
        self.want_self: bool = want_self
        self.want_conj: bool = want_conj
        self.local_monodromy = None
        self.polygon = None
        self.done: bool = False

    def __repr__(self):
        return repr(self.alg)

    @cached_method
    def point(self):
        return path.Point(self.alg, self._dop)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __lt__(self, other):
        # Sage graphs require vertices to be comparable
        return id(self) < id(other)

def _merge_conjugate_singularities(dop, sing, base, todo):
    need_conjugates = False
    sgn = 1 if base.alg.sign_imag() >= 0 else -1
    for x in sing:
        if sgn*x.sign_imag() < 0:
            need_conjugates = True
            del todo[x]
            xconj = x.conjugate()
            item = todo.get(xconj)
            if item is None:
                todo[xconj] = item = TodoItem(xconj, dop)
            item.want_conj = True
    return need_conjugates

def _spanning_tree(base, verts):
    graph = Graph([list(verts), lambda x, y: x is not y])
    def length(edge):
        x, y, _ = edge
        return abs(CC(x.alg) - CC(y.alg))
    tree = graph.min_spanning_tree(weight_function=length)
    tree = Graph(tree)
    tree.add_vertex(base)
    return tree

def _closest_unsafe(lst, x):
    x = CC(x.alg)
    return min(enumerate(lst), key=lambda y: abs(CC(y[1].value) - x))

def _extend_path_mat(dop, path_mat, inv_path_mat, x, y, eps, matprod, ctx):
    anchor_index_x, anchor_x = _closest_unsafe(x.polygon, y)
    anchor_index_y, anchor_y = _closest_unsafe(y.polygon, x)
    bypass_mat_x = matprod(x.local_monodromy[:anchor_index_x])
    bypass_mat_y = matprod(y.local_monodromy[anchor_index_y:]
                           if anchor_index_y > 0
                           else [])
    if anchor_y.is_singular():
        # Avoid computing inverses of inverses
        path = [anchor_y, anchor_x]
        invert = True
    else:
        path = [anchor_x, anchor_y]
        invert = False
    path[0].options["store_value"] = False # XXX bugware
    path[1].options["store_value"] = True
    edge_mat = dop.numerical_transition_matrix(path, eps, ctx=ctx)
    inv_edge_mat = ~edge_mat
    if invert:
        edge_mat, inv_edge_mat = inv_edge_mat, edge_mat
    ext_mat = bypass_mat_y*edge_mat*bypass_mat_x
    inv_ext_mat = ~bypass_mat_x*inv_edge_mat*~bypass_mat_y
    new_path_mat = ext_mat*path_mat
    new_inv_path_mat = inv_path_mat*inv_ext_mat
    assert isinstance(new_path_mat, Matrix_complex_ball_dense)
    return new_path_mat, new_inv_path_mat

LocalMonodromyData = collections.namedtuple("LocalMonodromyData",
        ["point", "monodromy", "is_scalar"])

def _monodromy_matrices(dop, base, eps=1e-16, sing=None, **kwds):
    r"""
    Return an iterator over local monodromy matrices of ``dop`` with base point
    ``base``.

    INPUT:

    See :func:`monodromy_matrices`

    OUTPUT:

    A list of ``LocalMonodromyData`` named tuples, with fields:

    - ``point`` -- a singular point of ``dop``, represented as an element of
      ``QQbar``,
    - ``monodromy`` -- a local monodromy matrix attached to this point,
      represented as a matrix with entries in a complex ball field,
    - ``is_scalar`` -- boolean, ``True`` iff the code could certify that the
      ``monodromy`` is exactly a scalar matrix.

    EXAMPLES::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.monodromy import _monodromy_matrices
        sage: Dops, x, Dx = DifferentialOperators()
        sage: rat = 1/(x^2-1)
        sage: dop = (rat*Dx - rat.derivative()).lclm(Dx*x*Dx)
        sage: [rec.point for rec in _monodromy_matrices(dop, 0) if not rec.is_scalar]
        [0]

    TESTS::

        sage: from ore_algebra.examples import fcc
        sage: mon = list(_monodromy_matrices(fcc.dop5, -1, 2**(-2**7))) # long time (1.9 s)
        sage: [rec.monodromy[0][0] for rec in mon if rec.point == -5/3] # long time
        [[1.01088578589319884254557667137848...]]

    Thanks to Alexandre Goyer for these examples::

        sage: L1 = ((x^5 - x^4 + x^3)*Dx^3 + (27/8*x^4 - 25/9*x^3 + 8*x^2)*Dx^2
        ....:      + (37/24*x^3 - 25/9*x^2 + 14*x)*Dx - 2*x^2 - 3/4*x + 4)
        sage: L2 = ((x^5 - 9/4*x^4 + x^3)*Dx^3 + (11/6*x^4 - 31/4*x^3 + 7*x^2)*Dx^2
        ....:      + (7/30*x^3 - 101/20*x^2 + 10*x)*Dx + 4/5*x^2 + 5/6*x + 2)
        sage: L = L1*L2
        sage: L = L.parent()(L.annihilator_of_composition(x+1))
        sage: mon = list(_monodromy_matrices(L, 0, eps=1e-30)) # long time (1.4 s)
        sage: mon[-1][0], mon[-1][1][0][0] # long time
        (0.6403882032022075?,
        [1.15462187280628880820271...] + [-0.018967673022432256251718...]*I)

        sage: list(_monodromy_matrices(Dx*x, 0))
        [LocalMonodromyData(point=0, monodromy=[1.0000000000000000], is_scalar=True)]
        sage: list(_monodromy_matrices(Dx*(x-i)*(x+i), i))
        [LocalMonodromyData(point=1*I, monodromy=[1.0000000000000000], is_scalar=True),
        LocalMonodromyData(point=-1*I, monodromy=[1.0000000000000000], is_scalar=True)]
    """
    ctx = Context(**kwds)
    ctx.assume_analytic = True
    dop = dop.numerator()
    if all(c in QQ for pol in dop for c in pol):
        dop = dop.change_ring(dop.base_ring().change_ring(QQ))
    dop = DifferentialOperator(dop)
    eps = RBF(eps)
    if sing is None:
        sing = dop._singularities(apparent=False)
    else:
        sing = [x for x in dop._singularities() if x.as_algebraic() in sing]

    # Normalize base point. If it is one of the singularities, make sure we
    # represent them by the same object (and thus by a PolynomialRoot compatible
    # with the remaining singularities).
    base = QQbar.coerce(base)
    base_iv = CBF(base)
    for s in dop._singularities():
        if base_iv in s.as_ball(CBF) and base == s.as_algebraic():
            base = s
            break
    else:
        base = PolynomialRoot.make(base)

    todo = {x: TodoItem(x, dop, want_self=True, want_conj=False)
            for x in sing}
    base = todo.setdefault(base, TodoItem(base, dop))
    if not base.point().is_regular():
        raise ValueError("irregular singular base point")
    # If the coefficients are rational, reduce to handling singularities in the
    # same half-plane as the base point, and share some computations between
    # Galois conjugates.
    need_conjugates = False
    crit_cache = None
    if dop.base_ring().base_ring() is QQ:
        need_conjugates = _merge_conjugate_singularities(dop, sing, base, todo)
        # TODO: do something like that even over number fields?
        # XXX this is actually a bit costly: do it only after checking that the
        # monodromy is not scalar?
        # XXX keep the cache from one run to the next when increasing prec?
        crit_cache = {}

    Scalars = ComplexBallField(utilities.prec_from_eps(eps))
    id_mat = matrix.identity_matrix(Scalars, dop.order())
    def matprod(elts):
        return prod(reversed(elts), id_mat)

    for key, todoitem in list(todo.items()):
        point = todoitem.point()
        # We could call _local_monodromy_loop() if point is irregular, but
        # delaying it may allow us to start returning results earlier.
        if point.is_regular():
            point_value = point.as_sage_value()
            if crit_cache is None or point.algdeg() == 1:
                crit = point.local_basis_structure()
                emb = point_value.parent().hom(Scalars)
            else:
                mpol = point_value.minpoly()
                try:
                    NF, crit = crit_cache[mpol]
                except KeyError:
                    NF = point_value.parent()
                    crit = point.local_basis_structure()
                    # Only store the critical monomials for reusing when all
                    # local exponents are rational. We need to restrict to this
                    # case because we do not have the technology in place to
                    # follow algebraic exponents along the embedding of NF in ℂ.
                    # (They are represented as elements of "new" number fields
                    # given by as_embedded_number_field_element(), even when
                    # they actually lie in NF itself as opposed to a further
                    # algebraic extension. XXX: Ideally, LocalBasisMapper should
                    # give us access to the tower of extensions in which the
                    # exponents "naturally" live.)
                    if all(sol.leftmost.is_rational() for sol in crit):
                        crit_cache[mpol] = NF, crit
                emb = NF.hom([Scalars(point_value.parent().gen())], check=False)
            mon, scalar = _formal_monodromy_from_critical_monomials(crit, emb)
            if scalar:
                # No need to compute the connection matrices then!
                # XXX When we do need them, though, it would be better to get
                # the formal monodromy as a byproduct of their computation.
                if todoitem.want_self:
                    yield LocalMonodromyData(key.as_algebraic(), mon, True)
                if todoitem.want_conj:
                    conj = key.conjugate()
                    logger.info("Computing local monodromy around %s by "
                                "complex conjugation", conj)
                    conj_mat = ~mon.conjugate()
                    yield LocalMonodromyData(conj.as_algebraic(), conj_mat, True)
                if todoitem is not base:
                    del todo[key]
                    continue
                else:
                    todoitem.want_self = todoitem.want_conj = False
            todoitem.local_monodromy = [mon]
            todoitem.polygon = [point]

    if need_conjugates:
        base_conj_mat = dop.numerical_transition_matrix(
            [base.alg.as_exact(), base.alg.conjugate().as_exact()],
            eps, ctx=ctx)
        def conjugate_monodromy(mat):
            return ~base_conj_mat*~mat.conjugate()*base_conj_mat

    tree = _spanning_tree(base, todo.values())

    def dfs(x, path, path_mat, inv_path_mat):

        logger.info("Computing local monodromy around %s via %s", x, path)

        local_mat = matprod(x.local_monodromy)
        based_mat = inv_path_mat*local_mat*path_mat

        if x.want_self:
            yield LocalMonodromyData(x.alg.as_algebraic(), based_mat, False)
        if x.want_conj:
            conj = x.alg.conjugate()
            logger.info("Computing local monodromy around %s by complex "
                        "conjugation", conj)
            conj_mat = conjugate_monodromy(based_mat)
            yield LocalMonodromyData(conj.as_algebraic(), conj_mat, False)

        x.done = True

        for y in tree.neighbors(x):
            if y.done:
                continue
            if y.local_monodromy is None:
                y.polygon, y.local_monodromy = _local_monodromy_loop(y.point(),
                                                                     eps, ctx)
            new_path_mat, new_inv_path_mat = _extend_path_mat(dop, path_mat,
                                          inv_path_mat, x, y, eps, matprod, ctx)
            yield from dfs(y, path + [y], new_path_mat, new_inv_path_mat)

    yield from dfs(base, [base], id_mat, id_mat)

def monodromy_matrices(dop, base, eps=1e-16, sing=None, **kwds):
    r"""
    Compute generators of the monodromy group of ``dop`` with base point
    ``base``.

    INPUT:

    - ``dop`` - differential operator
    - ``base`` - base point, must be coercible to ``QQbar``
    - ``eps`` - absolute tolerance (indicative)
    - ``sing`` (optional) - list of singularities to consider ; each entry must
      coerce into ``QQbar``. By default, all except maybe some apparent ones,
      i.e., compute generators of the monodromy group.

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

        sage: monodromy_matrices(Dx*x*Dx, 1, 1e-30)
        [
        [  1.0000...  [6.2831853071795864769252867665...]*I]
        [          0     1.00000000000000000000000000000...]
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

        sage: dop = (x**2*Dx + 3)*((x-3)*Dx + 4*x**5) # B. Salvy
        sage: dop.leading_coefficient().factor()
        (x - 3) * x^2
        sage: mon = monodromy_matrices(dop, 1)
        sage: mon[0].is_one() # around 3
        True
        sage: mon[1] # around 0
        [[1.000000000...] + [-2.8675932949...]*I          [+/- ...] + [1.4337966474...]*I]
        [       [+/- ...] + [-5.7351865899...]*I  [1.0000000000...] + [2.8675932949...]*I]

    The base point can be a singular point::

        sage: monodromy_matrices(Dx*x*Dx, 0)
        [
        [     1.0000000000000000                    0]
        [[6.28318530717958...]*I   1.0000000000000000]
        ]

    ...but currently not an irregular singular one::

        sage: monodromy_matrices(x^2*Dx - 1, 0)
        Traceback (most recent call last):
        ...
        ValueError: irregular singular base point

    One can limit the computation to a subset of the monodromy matrices::

        sage: from ore_algebra.examples import fcc
        sage: mon = monodromy_matrices(fcc.dop5, 0, sing=[0, 1])
        sage: mon[0]
        [                  1.00...                         0            0           0         0        0]
        [              [6.28...]*I                   1.00...            0           0         0        0]
        [               [-19.7...]               [6.28...]*I      1.00...           0         0        0]
        [             [-41.3...]*I                [-19.7...]  [6.28...]*I      1.00...        0        0]
        [                [64.9...]              [-41.3...]*I   [-19.7...]  [6.28...]*I  1.00...        0]
        [[-2.96...] + [-7.14...]*I  [-2.96...] + [0.94...]*I  [0.94...]*I            0        0  1.00...]
        sage: mon[1][2,5]
        [-9.310353...] + [+/- ...]*I
        sage: mon[1].trace()
        [4.000000...] + [+/- ...]*I
    """
    it = _monodromy_matrices(dop, base, eps, sing, **kwds)
    return [mat for _, mat, _ in it]

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
