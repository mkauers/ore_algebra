
"""
tools
=====

collection of auxiliary functions.

"""

#############################################################################
#  Copyright (C) 2014 Manuel Kauers (mkauers@gmail.com),                    #
#                     Maximilian Jaroschek (mjarosch@risc.jku.at),          #
#                     Fredrik Johansson (fjohanss@risc.jku.at).             #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

from sage.structure.element import RingElement, canonical_coercion
from sage.rings.arith import gcd, lcm, previous_prime as pp
from sage.rings.qqbar import QQbar
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.complex_field import ComplexField

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
        ev = dict( (y, hash(y)) for y in u.parent().gens_dict_recursive() )
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

      sage: R0.<a,b> = ZZ['a','b']; R.<x> = R0['x']
      sage: f = make_factor_iterator(R)
      sage: [(p, e) for p, e in f(((a+b)*x - 2)^3*(2*x+a)*(2*x+b))]
      [(2*x + b, 1), (2*x + a, 1), ((a + b)*x - 2, 3)]
      sage: f = make_factor_iterator(ZZ[x])
      sage: [(p, e) for p, e in f((2*x-3)*(4*x^3-5)*(3*x^5-4))]
      [(2*x - 3, 1), (4*x^3 - 5, 1), (3*x^5 - 4, 1)]

    """
    R = ring.ring() if ring.is_field() else ring 
    x = R.gen(); C = R.base_ring().fraction_field()
    if C in (QQ, QQbar):
        flush = (lambda p: R(p.numerator())) if R.base_ring() is ZZ else (lambda p: p)
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
    elif C.base_ring() in (ZZ, QQ) and C == C.base_ring()[R.base_ring().gens()].fraction_field():
        # R = QQ(...)[x]
        gens = C.gens() + (x,)
        R_ext = QQ[gens]; x_ext = R_ext(x)
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
        raise NotImplementedError, ring 

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
                q/=q[0]

        # have we already seen a member of the shift equivalence class of q? 
        new = True; 
        for i in xrange(len(classes)):
            u = classes[i][0]
            if u.degree() != q.degree():
                continue
            a = candidate(q, u)
            if a not in ZZ or sigma(q, a/ram) != u:
                continue
            # yes, we have: q(x+a) == u(x); u(x-a) == q(x)
            # register it and stop searching
            a = ZZ(a); new = False
            if a < 0:
                classes[i][1].append((-a, b))
            elif a > 0:
                classes[i][0] = q
                classes[i][1] = [(n+a,m) for (n,m) in classes[i][1]]
                classes[i][1].append((0, b))
            break

        # no, we haven't. this is the first.
        if new:
            classes.append( [q, [(0, b)]] )

    for c in classes:
        c[1].sort(key=lambda e: e[0])

    return classes


def uncouple(mat, algebra=None):
    """
    Triangularizes an operator matrix. 

    The matrix is to be specified as lists of lists. The inner lists represent the rows of the matrix.
    The output matrix has the same number of columns, but perhaps a smaller number of rows. It will be
    in staircase form. Row operations applied during the transformation act on the matrix from the left.
    No column swaps are preformed. 

    not yet tested. code looks reasonable, but output looks suspicious...

    """

    A = mat[0][0].parent() if algebra is None else algebra
    Arat = A.change_ring(A.base_ring().fraction_field())
    Apol = Arat.change_ring(Arat.base_ring().ring())
    Pol = Apol.base_ring()
    mat = [map(Arat, list(m)) for m in mat] # private copy

    # clear denominators and content
    def clean_row(i):
        d = Pol(lcm([Pol(Arat(c).denominator()) for c in mat[i]]))
        mat[i] = [d*m for m in mat[i]]
        g = gcd([Apol(c).content() for c in mat[i]])
        mat[i] = [Apol(c).map_coefficients(lambda p: p//g) for c in mat[i]]

    for i in xrange(len(mat)):
        clean_row(i)

    r = 0 # all rows before this one have been handled. 
    for c in xrange(len(mat[0])):
        # idea: if there are three or more nonzero elements, compute the gcrd of two random 
        # linear combinations to produce two new rows, then combine them to a new row whose lc
        # is the gcd of the lc's of the two rows. Use this row for elimination.
        # if there are only two nonzero elements, just take their gcrd and use it for elimination.
        # if there is only one nonzero element, use it for elimination.
        # if there is no nonzero element, there is nothing to do.

        nonzero = [i for i in xrange(r, len(mat)) if not mat[i][c].is_zero()]
        
        # select or construct pivot
        if len(nonzero) == 0:
            continue
        elif len(nonzero) == 1:
            piv_row = nonzero[0]
        elif len(nonzero) == 2:
            i, j = nonzero
            G, S, T = mat[i][c].xgcrd(mat[j][c])
            d = lcm(S.denominator(), T.denominator())
            S, T = d*S, d*T
            mat.append([S*mat[i][l] + T*mat[j][l] for l in xrange(len(mat[0]))])
            piv_row = len(mat) - 1
        else:
            m = len(nonzero)
            row1 = [sum((19+2**i)*mat[nonzero[i]][l] for i in xrange(m)) for l in xrange(len(mat[0]))]
            row2 = [sum((17+3**i)*mat[nonzero[i]][l] for i in xrange(m)) for l in xrange(len(mat[0]))]
            row3 = [sum((13+5**i)*mat[nonzero[i]][l] for i in xrange(m)) for l in xrange(len(mat[0]))]
            G1, S1, T1 = row1[c].xgcrd(row2[c])
            d = lcm(S1.denominator(), T1.denominator()); G1, S1, T1 = d*G1, d*S1, d*T1
            G2, S2, T2 = row2[c].xgcrd(row3[c])
            d = lcm(S2.denominator(), T2.denominator()); G2, S2, T2 = d*G2, d*S2, d*T2
            assert(G1.order() == G2.order())
            g, s, t = Pol(G1.leading_coefficient()).xgcd(Pol(G2.leading_coefficient()))
            mat.append([ s*S1*row1[l] + (s*T1 + t*S2)*row2[l] + t*T2*row3[l] for l in xrange(len(mat[0]))])
            piv_row = len(mat) - 1

        # move pivot to front
        mat[r][c], mat[piv_row][c] = mat[piv_row][c], mat[r][c]
        piv = mat[r][c]

        # perform elimination 
        for i in xrange(r + 1, len(mat)):
            Q, R = mat[i][c].quo_rem(piv); assert(R.is_zero()); 
            d = Arat(Q).denominator(); Q = Apol(d*Q)
            for j in xrange(c, len(mat[0])):
                mat[i][j] = d*mat[i][j] - Q*mat[r][j]
            clean_row(i)

        r += 1
                
    return [row for row in mat if not all(p.is_zero() for p in row)]
