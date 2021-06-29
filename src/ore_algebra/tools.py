"""
Auxiliary functions
"""

#############################################################################
#  Copyright (C) 2014 Manuel Kauers (mkauers@gmail.com),                    #
#                     Maximilian Jaroschek (mjarosch@risc.jku.at),          #
#                     Fredrik Johansson (fjohanss@risc.jku.at).             #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  https://www.gnu.org/licenses/                                             #
#############################################################################

from __future__ import absolute_import

from sage.structure.element import Element, canonical_coercion
from sage.arith.all import gcd
from sage.functions.other import real_part
from sage.matrix.constructor import Matrix
from sage.rings.qqbar import QQbar
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
try:
    from sage.rings.complex_mpfr import ComplexField
except ImportError:
    from sage.rings.complex_field import ComplexField
from sage.rings.number_field.number_field_base import is_NumberField
from sage.rings.fraction_field import FractionField_generic
from sage.rings.fraction_field_element import FractionFieldElement


def q_log(q, u):
    """
    Determines, if possible, an integer n such that q^n = u.

    Requires that both q and u belong to either QQ or some rational function field over QQ.

    q must not be zero or a root of unity.

    A ValueError is thrown if no n exists.
    """
    if q in QQ and u in QQ:
        qq, uu = q, u
    else:
        q, u = canonical_coercion(q, u)
        ev = dict((y, hash(y)) for y in u.parent().gens_dict_recursive())
        qq, uu = q(**ev), u(**ev)

    n = ComplexField(53)(uu.n().log()/qq.n().log()).real_part().round()
    if q**n == u:
        return n
    else:
        raise ValueError


def make_factor_iterator(ring, multiplicities=True):
    """
    Creates an iterator for factoring polynomials in the given ring.

    The ring must be a univariate polynomial ring over some base ring R, and the
    method will attempt to construct a factorizer for elements as if they were
    elements of Frac(R)[x]. Only factors with positive x-degree will be returned.
    The factors will not be casted back to elements of R[x]. If multiplicities is set
    to True (default), the iterator will return pairs (p, e), otherwise just the
    irreducible factors p.

    EXAMPLES::

        sage: from ore_algebra.tools import make_factor_iterator
        sage: R0.<a,b> = ZZ['a','b']; R.<x> = R0['x']
        sage: f = make_factor_iterator(R)
        sage: [(p, e) for p, e in f(((a+b)*x - 2)^3*(2*x+a)*(2*x+b))]
        [(2*x + b, 1), (2*x + a, 1), ((a + b)*x - 2, 3)]
        sage: f = make_factor_iterator(ZZ[x])
        sage: [(p, e) for p, e in f((2*x-3)*(4*x^3-5)*(3*x^5-4))]
        [(2*x - 3, 1), (4*x^3 - 5, 1), (3*x^5 - 4, 1)]
        sage: f = make_factor_iterator(QQ.extension(QQ[x](x^2+1), "ii")[x])
        sage: [(p, e) for p, e in f((x^2+1)^2*(4*x^3-5)*(3*x^5-4))]
        [(x - ii, 2), (x + ii, 2), (x^3 - 5/4, 1), (x^5 - 4/3, 1)]
    """
    R = ring.ring() if ring.is_field() else ring
    x = R.gen()
    C = R.base_ring().fraction_field()
    if C in (QQ, QQbar):
        # R = QQ[x] or QQbar[x]
        flush = (lambda p: R(p.numerator())
                 ) if R.base_ring() is ZZ else (lambda p: p)
        if multiplicities:
            def factors(p):
                for f, e in C[x](p).factor():
                    if f.degree() > 0:
                        yield flush(f), e
        else:
            def factors(p):
                for f, e in C[x](p).factor():
                    if f.degree() > 0:
                        yield flush(f)
    elif is_NumberField(R.base_ring()):
        # R = QQ(alpha)[x]
        if multiplicities:
            def factors(p):
                for u, e in R(p).factor():
                    if u.degree() > 0:
                        yield u, e
        else:
            def factors(p):
                for u, e in R(p).factor():
                    if u.degree() > 0:
                        yield u
    elif C.base_ring() in (ZZ, QQ) and C == C.base_ring()[R.base_ring().gens()].fraction_field():
        # R = QQ(t1,...)[x]
        gens = C.gens() + (x,)
        R_ext = QQ[gens]
        x_ext = R_ext(x)
        R = QQ[C.gens()][x]
        if multiplicities:
            def factors(p):
                for u, e in R_ext(p.numerator()).factor():
                    if u.degree(x_ext) > 0:
                        yield R(u), e
        else:
            def factors(p):
                for u, e in R_ext(p.numerator()).factor():
                    if u.degree(x_ext) > 0:
                        yield R(u)
    else:
        raise NotImplementedError(ring)

    return factors


def shift_factor(p, ram=ZZ.one(), q=1):
    """
    Returns the roots of p in an appropriate extension of the base ring, sorted according to
    shift equivalence classes.

    INPUT:

    - ``p`` -- a univariate polynomial over QQ or a number field
    - ``ram`` (optional) -- positive integer
    - ``q`` (optional) -- if set to a quantity different from 1 or 0, the factorization will be
      made according to the q-shift instead of the ordinary shift. The value must not be a root
      of unity.

    OUTPUT:

    A list of pairs (q, e) where

    - q is an irreducible factor of p
    - e is a tuple of pairs (a, b) of nonnegative integers
    - p = c*prod( sigma^(a/ram)(q)^b for (q, e) in output list for (a, b) in e ) for some nonzero constant c
      (in the q-case, a possible power of x is also omitted)
    - e[0][0] == 0, and e[i][0] < e[i+1][0] for all i
    - any two distinct q have no roots at integer distance.

    The constant domain must have characteristic zero.

    In the q-case, ramification greater than 1 requires that q^(1/ram) exists in the constant domain.

    Note that rootof(q) is the largest root of every class. The other roots are given by rootof(q) - e[i][0]/ram.

    EXAMPLES::

        sage: from ore_algebra.tools import shift_factor
        sage: x = ZZ['x'].gen()
        sage: shift_factor((x-2)*(x-4)*(x-8)*(2*x+3)*(2*x+15))
        [[x - 8, [(0, 1), (4, 1), (6, 1)]], [2*x + 3, [(0, 1), (6, 1)]]]
        sage: shift_factor((x-2)*(x-4)*(x-8)*(2*x+3)*(2*x+15), q=2)
        [[-1/8*x + 1, [(0, 1), (1, 1), (2, 1)]], [2/3*x + 1, [(0, 1)]], [2/15*x + 1, [(0, 1)]]]
    """

    classes = []
    x = p.parent().gen()

    qq = q
    assert(x.parent().characteristic() == 0)
    if qq == 1:
        def sigma(u, n=1):
            return u(x + n)

        def candidate(u, v):
            d = u.degree()
            return ram*(u[d]*v[d-1] - u[d-1]*v[d])/(u[d]*v[d]*d)
    else:
        def sigma(u, n=1):
            return u(x*qq**n)

        def candidate(u, v):
            d = u.degree()
            try:
                return -q_log(qq, (u[d]/v[d])**ram)/d
            except:
                return None

    for (q, b) in make_factor_iterator(p.parent())(p):

        if q.degree() < 1:
            continue
        if qq != 1:
            if q[0].is_zero():
                continue
            else:
                q /= q[0]

        # have we already seen a member of the shift equivalence class of q?
        new = True
        for i in range(len(classes)):
            u = classes[i][0]
            if u.degree() != q.degree():
                continue
            a = candidate(q, u)
            if a not in ZZ or sigma(q, a/ram) != u:
                continue
            # yes, we have: q(x+a) == u(x); u(x-a) == q(x)
            # register it and stop searching
            a = ZZ(a)
            new = False
            if a < 0:
                classes[i][1].append((-a, b))
            elif a > 0:
                classes[i][0] = q
                classes[i][1] = [(n+a, m) for (n, m) in classes[i][1]]
                classes[i][1].append((0, b))
            break

        # no, we haven't. this is the first.
        if new:
            classes.append([q, [(0, b)]])

    for c in classes:
        c[1].sort(key=lambda e: e[0])

    return classes


def _my_lcm(elts):  # non monic
    l = ZZ.one()
    for p in elts:
        l *= p//p.gcd(l)
    return l


def _clear_denominators_1(elt):
    dom = elt.parent().ring()
    base = dom.base_ring()
    num, den = elt.numerator(), elt.denominator()
    if isinstance(base, FractionField_generic):
        numnum, numden = clear_denominators(num, base.ring())
        dennum, denden = clear_denominators(den, base.ring())
        newdom = dom.change_ring(base.ring())
        numnum = newdom(numnum)
        dennum = newdom(dennum)
        gnum = numnum.gcd(dennum)
        numnum, dennum = numnum//gnum, dennum//gnum
        gden = numden.gcd(denden)
        numden, denden = numden//gden, denden//gden
        return (numnum, numden, dennum, denden)
    else:
        return (num, dom.one(), den, dom.one())


def _has_coefficients(elt):
    if isinstance(elt, (list, tuple)):
        return True
    elif isinstance(elt, Element):
        parent_ = elt.parent()
        return parent_.base_ring() is not parent_ and hasattr(parent_, 'change_ring')


def clear_denominators(elts, dom=None):
    r"""
    Recursively clear denominators in a list (or other iterable) of elements.

    Typically intended for elements of fields like QQ(x)(y).
    """
    if not elts:
        return elts, dom.one()
    if all(isinstance(elt, FractionFieldElement) for elt in elts):
        split = [_clear_denominators_1(elt) for elt in elts]
        lcmnum = _my_lcm((dennum for _, _, dennum, _ in split))
        lcmden = _my_lcm((numden for _, numden, _, _ in split))
        num = [(denden*(lcmden//numden))*(numnum*(lcmnum//dennum))
               for (numnum, numden, dennum, denden) in split]
        den = lcmden*lcmnum
        g = gcd(num + [den])  # XXX: can we be more specific here?
        num = [p//g for p in num]
        den = den//g
    elif all(_has_coefficients(elt) for elt in elts):
        nums, den = clear_denominators([elt for l in elts for elt in l])
        s = 0
        num = []
        for elt in elts:
            if isinstance(elt, Element):
                r = len(list(elt))
                newdom = elt.parent().change_ring(nums[0].parent())
            else:
                r = len(elt)
                newdom = type(elt)
            num.append(newdom(nums[s:s+r]))
            s += r
    else:
        num, den = elts, (dom or elts[0].parent()).one()
    if isinstance(elts, Element):
        num = elts.parent().change_ring(num[0].parent())(num)
    else:
        num = type(elts)(num)
    # assert all(b/den == a for a, b in zip(elts, num))
    # assert gcd(num + [den]).is_one()
    return num, den


def _residue(obj, place=None):
    r"""
    Return the residue of an element of a valued ring or field.

    INPUT:

    - ``obj`` -- an element of a valued ring or field. So far power series (without a place, see below) and rational fractions are supported.

    - ``place`` (default: 0) -- the place at which to evaluate the
      valuation

    OUTPUT:

    The valuation 0 coefficient of a series expansion of ``obj`` at
    ``place``. If the object has negative valuation, an error is
    raised.

    EXAMPLES::

        sage: from ore_algebra.tools import _residue
        sage: P.<q> = PolynomialRing(QQ)
        sage: K = P.fraction_field()
        sage: _residue(K(1))
        1
        sage: _residue((1+q)/q)
        Traceback (most recent call last):
        ...
        TypeError: The element has negative valuation.
        sage: _residue((1+q)/q, place=1+q)
        0

        sage: L.<t> = LaurentSeriesRing(QQ)
        sage: _residue(L(1))
        1
        sage: _residue(1/(1+t^2)-1)
        0
        sage: _residue((1+t)/t)
        Traceback (most recent call last):
        ...
        TypeError: The element has negative valuation.
        sage: _residue((1+t)/t, place=1+t)
        Traceback (most recent call last):
        ...
        NotImplementedError: Valuation at specific place not implemented for Laurent series

    """

    try:
        num = obj.numerator()
        polring = num.parent()
        den = polring(obj.denominator())
        if place:
            num = num % place
            den = den % place
        if den[0] == 0:
            raise TypeError("The element has negative valuation.")
        res = num[0]/den[0]
        # If we want lowest valuation instead, we can use the first coefficient...
    except AttributeError:  # Dirty...
        if place:
            raise NotImplementedError(
                "Valuation at specific place not implemented for Laurent series")
        if obj.valuation() < 0:
            raise TypeError("The element has negative valuation.")
        res = obj[0]
    return res


def _vect_val_fct(v, place=None):
    r"""
    Compute the valuation of a vector.

    INPUT:

    - ``v`` -- a vector over a valued field, or anything iterable with values in a valued field.

    - ``place`` (default: None) -- the point at which to evaluate the valuation; not supported over all fields

    OUTPUT:

    The minimum valuation of the coordinates of ``v``

    EXAMPLES::

        sage: from ore_algebra.tools import _vect_val_fct
        sage: P.<q> = PolynomialRing(QQ)
        sage: K = P.fraction_field()
        sage: _vect_val_fct([K(1),K(1/q)])
        -1
        sage: _vect_val_fct([K(q),K(q^2)])
        1
        sage: _vect_val_fct([K(1+q),K(q+q^2)],place=1+q)
        1

    # TODO: Add examples with power series?
    """
    if place:
        return min(vv.valuation(place) for vv in v)
    else:
        return min(vv.valuation() for vv in v)


def _vect_elim_fct(basis, place=None, dim=None, infolevel=0, residue_fct=None):
    r"""
    Find a relation between vectors raising the valuation.

    INPUT:

    - ``basis`` -- a list of length ``d`` of vectors over a valued
      fields, or of iterables with values in a valued field. The
      elements should be iterable at least up to index ``dim-1``. All
      elements should have valuation 0, and the valuation 0 part of
      the first ``(d-1)`` elements of the basis should be linearly
      independent.

    - ``dim`` (default: length of the basis) -- the dimension of the
      vector space to consider; it should be at least ``d``.

    - ``place`` (default: None) -- the point at which to evaluate the valuation; not supported over all fields

    OUTPUT:

    A list ``alpha`` of ``d`` elements of the base field such that
    ``alpha[0]*basis[0] + ... + alpha[d-1]*basis[d-1]`` has positive
    valuation, or `None` if no such relation exists.

    Under the assumptions, it is guaranteed that ``alpha[d-1] == 1``.

    EXAMPLES::

    # TODO: Add examples with plain vectors?

        sage: from ore_algebra.tools import _vect_elim_fct
        sage: k.<q> = LaurentSeriesRing(QQ)
        sage: V.<t> = PolynomialRing(k)
        sage: b = [V(1),V(1+q*t)]
        sage: v = _vect_elim_fct(b); print(v)
        (-1, 1)
        sage: v[0]*b[0] + v[1]*b[1]
        q*t
        sage: v = _vect_elim_fct([V(1),V(t)]); print(v)
        None

    Beware of unexpected results if ``dim`` is not properly set or
    obviously guessable. ::

        sage: V.<t> = PolynomialRing(k)
        sage: v = _vect_elim_fct([t+q]); print(v)
        (1)
        sage: v = _vect_elim_fct([t+q],dim=2); print(v)
        None

    Over the rationals, we can work over different places::

        sage: P.<q> = PolynomialRing(QQ)
        sage: k = P.fraction_field()
        sage: V.<t> = PolynomialRing(k)
        sage: b = [V(1),V(1+q*t)]
        sage: v = _vect_elim_fct(b)
        sage: print(v)
        (-1, 1)
        sage: v[0]*b[0] + v[1]*b[1]
        q*t
        sage: v = _vect_elim_fct(b, place=1+q)
        sage: print(v)
        None

    Changing the valuation place is not supported for Laurent series::

        sage: k.<q> = LaurentSeriesRing(QQ)
        sage: V.<t> = PolynomialRing(k)
        sage: v = _vect_elim_fct([t+q],place=2); print(v)
        Traceback (most recent call last):
        ...
        NotImplementedError: Valuation at specific place not implemented for Laurent series
    
    """

    # Helpers
    print2 = print if infolevel >= 2 else lambda *a, **k: None

    d = len(basis)
    if dim is None:
        dim = d

    if residue_fct is None:
        residue_fct = _residue

    M = Matrix([[residue_fct(basis[i][j], place) for j in range(dim)]
                for i in range(d)])
    print2(" [elim_fct] Matrix:")
    print2(M)
    K = M.left_kernel().basis()
    if K:
        return (1/K[0][-1])*K[0]
    else:
        return None


def roots_at_integer_distance(f1, f2):
    r"""
    Return integer distances between two roots of the polynomials.

    INPUT:

    - ``f1,f2`` -- two polynomials in one variable

    OUTPUT:

    A list of all integers ``i`` such that there is a root ``x`` of ``f1`` such that ``x+i`` is a root of ``f2``.

    EXAMPLES::

        sage: from ore_algebra.tools import roots_at_integer_distance
        sage: P.<x> = PolynomialRing(QQ)
        sage: roots_at_integer_distance(x*(x+1),x-2)
        [3, 2]
        sage: roots_at_integer_distance(x*(x+1),1)
        []
        sage: roots_at_integer_distance(x,2*x)
        [0]

    With rational numbers::

        sage: roots_at_integer_distance(x,2*x-1)
        []
        sage: roots_at_integer_distance((3*x-1)*(3*x-4),(3*x+2))
        [-1, -2]
        sage: roots_at_integer_distance((3*x-1),(3*x+1))
        []

    With algebraic numbers::

        sage: roots_at_integer_distance((x^2-2)*((x+1)^2-2),((x-1)^2-2))
        [2, 1]
        sage: roots_at_integer_distance(x^2-2,x^2-3)
        []

    """
    Pol = f1.parent().extend_variables("i")
    xx, ii = Pol.gens()
    resultant = f1(xx-ii).resultant(Pol(f2)).univariate_polynomial()
    roots = [a for (a, m) in resultant.roots() if a.is_integer()]
    return roots


def generalized_series_default_iota(z, j):
    if j == 0:
        return z-real_part(z).floor()
    else:
        return z-real_part(z).ceil()+1


def generalized_series_term_valuation(z, i, j, iota=None):
    r"""
    Given z, i, j, return the valuation of the term x^(z+i) log(x)^j
    """
    if iota is None:
        iota = generalized_series_default_iota
    return int(ZZ(z+i-iota(z, j)))
