# vim: tw=80
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

from sage.misc.misc_c import prod
from sage.rings.rational_field import QQ

logger = logging.getLogger(__name__)

# -> Polynomial
# note: parent() can become _parent in .pyx

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
        sage: shiftless_decomposition(2*y^2*(y+1))
        (2, [(y, [(0, 2), (1, 1)])])
        sage: shiftless_decomposition((y-1)*(y-1/2)*y) # random order
        (1, [(y - 1/2, [(0, 1)]), (y - 1, [(0, 1), (1, 1)])])
        sage: shiftless_decomposition(Pol.gen())
        (1, [(y, [(0, 1)])])
        sage: shiftless_decomposition(2*Pol.one())
        (2, [])
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
    shifts = by_mult[1].dispersion_set() if len(by_mult) > 1 else []
    for shift in shifts:
        parts.update(by_mult[1].gcd(f(x + eps*shift))
                     for f in by_mult[1:]
                     for eps in [1, -1])
        parts = set(Pol.gcd_free_basis(parts))
    def mult(part):
        for m in range(len(by_mult) - 1, -1, -1):
            if part.divides(by_mult[m]): # the paper says part^m?!
                return m
        assert False
    shifts_of = {} # factor -> nonnegative integer shifts
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
    return (unit, list(shifts_of.items()))

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
