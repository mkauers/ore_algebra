# -*- coding: utf-8 - vim: tw=80
r"""
Shiftless decomposition
"""

# Copyright 2016 Marc Mezzarobba
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

# Most of the functions in this file should be upstreamed, as methods of the
# indicated classes.

import logging

from six.moves import range

from sage.misc.cachefunc import cached_function
from sage.misc.misc_c import prod
from sage.structure.sequence import Sequence
from sage.rings.infinity import minus_infinity
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

logger = logging.getLogger(__name__)

# -> EuclideanDomains.ParentMethods

def gcd_free_basis(ring, elts):
    r"""
    Compute a set of coprime elements that can be used to express the
    elements of ``elts``.

    INPUT:

    - ``elts`` - A sequence of elements of ``ring``.

    OUTPUT:

    A GCD-free basis (also called a coprime base) of ``elts``; that is,
    a set of pairwise relatively prime elements of ``ring`` such that
    any element of ``elts`` can be written as a product of elements of
    the set.

    ALGORITHM:

    Naive implementation of the algorithm described in Section 4.8 of
    Bach & Shallit [BachShallit1996]_.

    .. [BachShallit1996] Eric Bach, Jeffrey Shallit.
        *Algorithmic Number Theory, Vol. 1: Efficient Algorithms*.
        MIT Press, 1996. ISBN 978-0262024051.

    EXAMPLES::

        sage: from ore_algebra.analytic.shiftless import *
        sage: gcd_free_basis(ZZ, [1])
        set()
        sage: gcd_free_basis(ZZ, [4, 30, 14, 49])
        {2, 7, 15}

        sage: Pol.<x> = QQ[]
        sage: gcd_free_basis(Pol,[
        ....:     (x+1)^3*(x+2)^3*(x+3), (x+1)*(x+2)*(x+3),
        ....:     (x+1)*(x+2)*(x+4)])
        {x + 3, x + 4, x^2 + 3*x + 2}
    """
    def refine(a, b):
        g = a.gcd(b)
        if g.is_unit():
            return (a, set(), b)
        l1, s1, r1 = refine(a//g, g)
        l2, s2, r2 = refine(r1, b//g)
        s1.update(s2)
        s1.add(l2)
        return (l1, s1, r2)
    elts = Sequence(elts, universe=ring)
    res = set()
    if len(elts) == 1:
        res.update(elts)
    else:
        r = elts[-1]
        for t in gcd_free_basis(ring, elts[:-1]):
            l, s, r = refine(t, r)
            res.update(s)
            res.add(l)
        res.add(r)
    units = [x for x in res if x.is_unit()]
    res.difference_update(units)
    return res

# -> UniqueFactorizationDomains.ElementMethods

# -> Polynomial
# note: parent() can become _parent in .pyx

def dispersion_set(self, other=None):
    r"""
    Compute the dispersion set of two polynomials.

    The dispersion set of `p` and `q` is the set of nonnegative integers
    `n` such that `f(x + n)` and `g(x)` have a nonconstant common factor.

    When ``other`` is ``None``, compute the auto-dispersion set of
    ``self``, i.e., its dispersion set with itself.

    ALGORITHM:

    See Section 4 of Man & Wright [ManWright1994]_.

    .. [ManWright1994] Yiu-Kwong Man and Francis J. Wright.
       *Fast Polynomial Dispersion Computation and its Application to
       Indefinite Summation*. ISSAC 1994.

    .. SEEALSO:: :meth:`dispersion`

    EXAMPLES::

        sage: from ore_algebra.analytic.shiftless import *
        sage: Pol.<x> = QQ[]
        sage: dispersion_set(x, x + 1)
        [1]
        sage: dispersion_set(x + 1, x)
        []

        sage: pol = x^3 + x - 7
        sage: dispersion_set(pol*pol(x+3)^2)
        [0, 3]
    """
    other = self if other is None else self.parent().coerce(other)
    x = self.parent().gen()
    shifts = set()
    for p, _ in self.factor():
        # need both due to the semantics of is_primitive() over fields
        assert p.is_monic() or p.is_primitive()
        for q, _ in other.factor():
            m, n = p.degree(), q.degree()
            assert q.is_monic() or q.is_primitive()
            if m != n or p[n] != q[n]:
                continue
            alpha = (q[n-1] - p[n-1])/(n*p[n])
            if alpha.is_integer(): # ZZ() might work for non-integers...
                alpha = ZZ(alpha)
            else:
                continue
            if alpha < 0 or alpha in shifts:
                continue
            if n >= 1 and p(x + alpha) != q:
                continue
            shifts.add(alpha)
    return list(shifts)

def dispersion(self, other=None):
    r"""
    Compute the dispersion of a pair of polynomials.

    The dispersion of `p` and `q` is the largest nonnegative integer `n`
    such that `f(x + n)` and `g(x)` have a nonconstant common factor.

    When ``other`` is ``None``, compute the auto-dispersion of ``self``,
    i.e., its dispersion with itself.

    .. SEEALSO:: :meth:`dispersion_set`

    EXAMPLES::

        sage: from ore_algebra.analytic.shiftless import *
        sage: Pol.<x> = QQ[]
        sage: dispersion(x, x + 1)
        1
        sage: dispersion(x + 1, x)
        -Infinity

        sage: Pol.<x> = QQbar[]
        sage: pol = Pol([sqrt(5), 1, 3/2])
        sage: dispersion(pol)
        0
        sage: dispersion(pol*pol(x+3))
        3
    """
    shifts = dispersion_set(self, other)
    return max(shifts) if len(shifts) > 0 else minus_infinity

# TODO: debug, document
def shiftless_decomposition(self):
    r"""
    Compute a shiftless decomposition of this polynomial.

    OUTPUT:

    A decomposition of ``self`` of the form

        .. math:: c \prod_i \prod_j g_i(x + h_{i,j})^{e_{i,j}}

    where

    * the `g_i` are monic squarefree polynomials of degree at least one,
    * `g_i(x)` and `g_j(x+h)` (with `i \neq j`) are coprime for all
      `h \in \ZZ`,
    * `g_i(x)` and `g_i(x+h)` are coprime for all nonzero `h \in \ZZ`,
    * `e_{i,j}` and `h_{i,j}` are integers with `e_{i,j} \geq 1`
      and `0 = h_{i,1} < h_{i,2} < \cdots`.

    ALGORITHM:

    Naïve implementation of the algorithm given in Section 3 of
    [GerhardGiesbrechtStorjohannZima2003]_.

    .. [GerhardGiesbrechtStorjohannZima2003] J. Gerhard, M. Giesbrecht,
       A. Storjohann and E. V. Zima.
       *Shiftless Decomposition and Polynomial-time Rational Summation*.
       ISSAC 2003.

    EXAMPLES::

        sage: from ore_algebra.analytic.shiftless import *
        sage: Pol.<y> = QQ[]
        sage: shiftless_decomposition((y-1)*(y-1/2)*y) # random order
        (1, [(y - 1/2, [(0, 1)]), (y - 1, [(0, 1), (1, 1)])])
    """
    quo = self.monic()
    Pol, x = quo.parent().objgen()
    unit = Pol.base_ring()(self.leading_coefficient())
    by_mult = [Pol.one()]
    while not quo.is_one():
        sqf = quo.radical()
        by_mult.append(sqf)
        quo //= sqf
    parts = set()
    shifts = dispersion_set(by_mult[1])
    for shift in shifts:
        parts.update(by_mult[1].gcd(f(x + eps*shift))
                     for f in by_mult[1:]
                     for eps in [1, -1])
        parts = gcd_free_basis(Pol, parts)
    def mult(part):
        for m in range(len(by_mult) - 1, -1, -1):
            if part.divides(by_mult[m]): # the paper says part^m?!
                return m
        assert False
    shifts_of = dict() # factor -> nonnegative integer shifts
    remaining_parts = parts.copy()
    while remaining_parts:
        cur = remaining_parts.pop()
        shifts_of_cur = []
        for shift in shifts:
            shifted = cur(x + shift)
            if shifted in parts:
                remaining_parts.discard(shifted)
                shifts_of.pop(shifted, None)
                shifts_of_cur.append((shift, mult(shifted)))
        shifts_of[cur] = shifts_of_cur
    # TODO: perhaps return some kind of Factorization object
    assert prod(part(x + s)**mult
                for part, shifts in shifts_of.items()
                for (s, mult) in shifts)*unit == self
    return (unit,
        [(part, shifts)
          for part, shifts in shifts_of.items()])

# key=... to avoid comparing number fields
# XXX: tie life to a suitable object
# @cached_function(key=lambda p: (id(p.parent()), p))
def my_shiftless_decomposition(pol):
    IniRing = pol.parent().base()
    try:
        pol = pol.monic().change_ring(QQ)
    except TypeError:
        logger.debug("failed to reduce to rational coefficients")
    x = pol.parent().gen()
    _, fac = shiftless_decomposition(pol(-x))
    return [(polynomial(-x).change_ring(IniRing), shifts)
            for polynomial, shifts in fac]
