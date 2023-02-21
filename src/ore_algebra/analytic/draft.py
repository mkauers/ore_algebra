# du code pour

from ore_algebra import guess
from ore_algebra.analytic.factorization import try_rational, _local_exponents
from ore_algebra.ideal import uncouple, solve_triangular_system
from ore_algebra.analytic.factorization import LinearDifferentialOperator
from ore_algebra.tools import clear_denominators
from sage.arith.functions import lcm
from sage.arith.misc import gcd
from sage.functions.other import binomial
from sage.plot.line import line2d
from sage.rings.polynomial.polynomial_element import Polynomial
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import QQbar

NewtonEdge = collections.namedtuple("NewtonEdge", ["slope", "startpoint", "length", "polynomial"])

def RRem(A, B):

    r""" Adapted version of right reminder computation, where A is a list of
    operators [A0, A1, ..., Am] representing the operator A0(a) + A1(a)*D + ...
    + Am(a)*D^m for a generic function a.
    """

    D, m, n = B.parent().gen(), len(A) - 1, B.order()
    if m<n: return A
    qn, Am = B.leading_coefficient(), A[-1]
    A = [ A[i] - (1/qn)*c*Am for i, c in enumerate(D**(m - n)*B) if i<m ]
    while len(A)>1 and A[-1]==0: A.pop()

    return RRem(A, B)

def cleaned_parent(L1, L2):

    r""" Try to find a commun parent, and in the case where the coefficients
    live in QQ, embed it in ZZ.
    """

    OA1, OA2 = L1.parent(), L2.parent()
    if OA1!=OA2:
        commun_parent = False
        try:
            OA1.coerce(L2)
            commun_parent = True
            OA = OA1
        except TypeError: pass
        try:
            OA2.coerce(L1)
            commun_parent = True
            OA = OA2
        except TypeError: pass
        if not commun_parent:
            raise TypeError("L1 and L2 must have the same parent.")
    else: OA = OA1

    R = OA.base_ring().base_ring()
    if isinstance(R, RationalField):
        I = R.ring_of_integers()
        L1, L2 = clear_denominators(L1)[0], clear_denominators(L2)[0]
        L1 = lcm([R(cc).denominator() for c in L1 for cc in c])*L1
        L2 = lcm([R(cc).denominator() for c in L2 for cc in c])*L2
        OA = OA.change_ring(PolynomialRing(I, L1.base_ring().gen().variable_name()))

    L1, L2 = OA(L1), OA(L2)
    return L1, L2, OA


def eigenring(L1, L2=None, infolevel=0):

    r"""
    Compute a basis of the space of the A + DL2 such that L2 divides L1A from
    the right.
    """

    if L2==None: L2 = L1
    m, n = L1.order(), L2.order()
    L1, L2, OA = cleaned_parent(L1, L2)
    D, zero = OA.gen(), OA.zero()

    # computation of the associated matrix N
    L1a = [ sum( binomial(k, i)*L1[k]*D**(k - i) for k in range(i, m + 1) )\
           for i in range(m + 1) ]
    N = [ RRem([zero]*j + L1a, L2) for j in range(n) ]
    N = [ Nj + [zero]*(n - len(Nj)) for Nj in N ]
    N = [ [ N[i][j] for i in range(n) ] for j in range(n) ]

    # solving Na=0
    basis = solve_triangular_system(uncouple(N, infolevel=infolevel), [[0]*n])

    # re-building the associated elements in the eigenring
    output = []
    for u, l in basis:
        op = D.parent().change_ring(D.parent().base_ring().fraction_field())(u)
        if not op.is_zero():
            d = gcd([c for pol in op for c in pol.numerator()])
            output.append((1/d)*op)

    return output


def dop_valuation(dop):

    r"""
    Return the smallest i such that a_{i,j} != 0 for some j where
    dop = sum a_{i,j} z^i (z*Dz)^j.
    """

    v = min(a.valuation() for a in _euler_representation(dop))

    return v


def local_bound_problem(R, L, c):

    r"""
    Compute a lower bound for the valuation of any operator r such that
    RightRemainder(Rr, L) = c, see [van Hoeij, Rational solutions of the mixed
    differential equation..., Lemma 1, 1996].

    INPUT:

     -- ``R, L`` -- linear differential operators
     -- ``c``    -- 0 or 1

    OUTPUT:

     -- ``v0r`` -- integer or None if it is certified that no such r exists

    """

    z = R.base_ring().gen()

    list_of_possible_v0r = []
    exponentsofL = L.indicial_polynomial(z).roots(QQbar, multiplicities=False)
    exponentsofR = R.indicial_polynomial(z).roots(QQbar, multiplicities=False)
    for eL in exponentsofL:
        for eR in exponentsofR:
            if eL - eR in ZZ:
                list_of_possible_v0r.append(eR - eL)

    if c!=0: list_of_possible_v0r.append(-dop_valuation(R))
    if list_of_possible_v0r==[]: return None
    return min(list_of_possible_v0r)


def common_denominator_for_mixed_equation(R, L, c):

    r"""
    Compute a polynomial D such that for any operator r = \sum r_i * Dz^i
    satisfying RightRemainder(Rr, L) = c, D*r_i is polynomial for all i.

    !!! ne peut pas utiliser PlainDifferentialOperator
    ou LinearDifferentialOperator !!! (circular import)
    """

    z = R.base_ring().gen()

    singL = PlainDifferentialOperator(L)._singularities(QQbar, multiplicities=False)
    singR = PlainDifferentialOperator(R)._singularities(QQbar, multiplicities=False)
    sing = list(set(singL) | set(singR))
    D = 1
    for s in sing:
        if not s in QQ:
            Rs, _ = PlainDifferentialOperator(R).extend_scalars(s)
            Ls, s = PlainDifferentialOperator(L).extend_scalars(s)
            Rs = R.annihilator_of_composition(z + s)
            Ls = L.annihilator_of_composition(z + s)
        else:
            Rs = R.annihilator_of_composition(z + s)
            Ls = L.annihilator_of_composition(z + s)
        v0r = local_bound_problem(Rs, Ls, c)
        if v0r==None: return None
        D = D*(z-s)^(-v0r)

    return D

def my_newton_polygon(dop):

    r"""
    Computes the Newton polygon of ``self`` at ``0``.

    INPUT:

      - ``dop`` -- a linear differential operator which polynomial coefficients

    OUTPUT:

    EXAMPLES::

    """

    n = dop.order(); z = dop.base_ring().gen()
    Pols, X = PolynomialRing(QQ, 'X').objgen()

    points = [ ((QQ(i), QQ(c.valuation(z))), c.coefficients()[0]) \
               for i, c in enumerate(dop.to_T('Tz').list()) if c!=0 ]

    (i1, j1), c1 = points[0]
    for (i, j), c in points:
        if j<=j1: (i1, j1), c1 = (i, j), c

    Edges = []
    if i1>0:
        poly = dop.indicial_polynomial(z, var = 'X')

        #pol = c1*X**i1
        #for (i, j), c in points:
        #    if i<i1 and j==j1: pol += c*X**i
        ## it is the same think (pol = poly)

        Edges.append( NewtonEdge(QQ(0), (0, j1), i1, poly) )

    while i1<n:
        poly = c1; (i2, j2), c2 = points[-1]; s = (j2 - j1)/(i2 - i1)
        for (i, j), c in points:
            if i>i1:
                t = (j - j1)/(i - i1)
                if t<s:
                    poly = c1; s = t
                    if t<=s:
                        poly += c*X**((i - i1)//s.denominator()) # REDUCED characteristic polynomial
                        (i2, j2), c2 = (i, j), c
        Edges.append( NewtonEdge(s, (i1, j1), i2 - i1, poly) )
        (i1, j1), c1 = (i2, j2), c2

    return Edges


def display_newton_polygon(dop):

    Edges = my_newton_polygon(dop)

    (i, j) = Edges[0].startpoint
    L1 = line2d([(i - 3, j), (i, j)], thickness=3)
    e = Edges[-1]; s = e.slope; (i,j) = e.startpoint; l = e.length
    L2 = line2d([(i + l, j + l*s), (i + l, j + l*s + 3)], thickness=3)

    L = sum(line2d([e.startpoint, (e.startpoint[0] + e.length, e.startpoint[1] \
            + e.length*e.slope)], marker='o', thickness=3) for e in Edges)

    return L1 + L + L2


def search_exp_part_with_mult1(dop):

    dop = LinearDifferentialOperator(dop)
    lc = dop.leading_coefficient()//gcd(dop.list())
    for f, _ in list(lc.factor()) + [ (1/dop.base_ring().gen(), None) ]:
        pol = dop.indicial_polynomial(f)
        roots = pol.roots(QQbar)
        for r, m in roots:
            if m==1:
                success = True
                for s, l in roots:
                    if s!=r and r-s in ZZ: success = False
                if success: return (f, r)

    return (None, None)

def min_diff_exp(dop):

    """
    returns ``(m, m_int)`` where m is the maximal difference, resp. the maximal
    integer difference, between local exponents at a singularity.

    The point at infinity is always considered as a singularity.
    """

    dop, z = LinearDifferentialOperator(dop), dop.base_ring().gen()
    lc = dop.leading_coefficient()//gcd(dop.list())
    sings = lc.roots(multiplicities=False)

    roots = dop.indicial_polynomial(1/z).roots(QQbar, multiplicities=False)
    l = [ (r-s).abs() for r in roots for s in roots ]
    m, m_int = max(l, default=0), max([x for x in l if x in ZZ], default=0)

    for f, _ in lc.factor():
        roots = dop.indicial_polynomial(f).roots(QQbar, multiplicities=False)
        l = [ (r-s).abs() for r in roots for s in roots ]
        m, m_int = max([m] + l), max([m_int] + [x for x in l if x in ZZ])

    return m, m_int

def guessing_via_series(L, einZZ):
    """ assumption: 0 is an exponential part of multiplicity 1 (at 0) """
    if not einZZ: # if e in ZZ, this test has already been done
        R = try_rational(L)
        if not R is None: return R
    r = L.order(); A = L.parent()
    t = len(L.desingularize().leading_coefficient().roots(QQbar))
    b = min(1000, max(50, (r - 1)**2*(r - 2)*(t - 1)))
    try:
        R = guess(L.power_series_solutions(b)[0].list(), A, order=r - 1) # don't work with algebraic extension (-> TypeError)
        if 0<R.order()<r and L%R==0: return R
    except (ValueError, TypeError): pass
    La = L.adjoint()
    Ra = try_rational(La)
    if not Ra is None: return (La//Ra).adjoint()
    ea = ZZ([e for e in _local_exponents(La, False) if e in ZZ][0]); La = Se(La, ea)
    try:
        Ra = guess(La.power_series_solutions(b)[0].list(), A, order=r - 1)
        if 0<Ra.order()<r and La%Ra==0: return Se(La//Ra, -ea).adjoint()
    except (ValueError, TypeError): return None



def try_vanHoeij(L):
    """ try to find a factor thank to an exponential part of multiplicity 1 """
    z, (p, e) = L.base_ring().gen(), search_exp_part_with_mult1(L)
    if not e in QQ: return None # not efficient enough for now (implem to be improved)
    if p==None: return None
    if (p*z).is_one():
        L = L.annihilator_of_composition(p)
        e = search_exp_part_with_mult1(L)[1]
        L, e = LinearDifferentialOperator(L).extend_scalars(e)
        L = Se(L, e)
    elif p.degree()==1:
        s = -p[0]/p[1]
        L, e = LinearDifferentialOperator(L).extend_scalars(e)
        L = Se(L.annihilator_of_composition(z + s), e)
    else: return None # to be implemented?
    R = guessing_via_series(L, e in ZZ)
    if R==None: return None
    if (p*z).is_one(): return Se(R, -e).annihilator_of_composition(p)
    elif p.degree()==1: return Se(R, -e).annihilator_of_composition(z - s)

def mydegree(pol): # for handling the case 1/z (point at infinity)
    if isinstance(pol, Polynomial):
        return pol.degree()
    return 1


def good_singular_point(dop):

    r"""
    Return (s, e, m) where ``s`` is a singular point (possibly ``infinity``) of
    ``dop`` admitting an exponent ``e`` of minnimal mutliplicity ``m`` mod ZZ.

    INPUT:

      - ``dop`` -- differential operator

    OUTPUT:

      - ``s`` -- element of QQbar
      - ``e`` -- element of QQbar
      - ``m`` -- positive integer

    """

    z = dop.base_ring().gen()
    dop = LinearDifferentialOperator(dop)
    lc = dop.leading_coefficient()//gcd(dop.list())

    all_min_mult = []
    for pol, _ in list(lc.factor()) + [ (1/z, None) ]:
        e, m = minimal_multiplicity(dop, pol)
        all_min_mult.append((pol, e, m))

    min_mult = min(all_min_mult, key = lambda x: x[2])[2]
    good_sings = [ x for x in all_min_mult if x[2]==min_mult ]

    min_deg = mydegree(min(good_sings, key = lambda x: mydegree(x[0]))[0])
    good_sings = [ x for x in all_min_mult if mydegree(x[0])==min_deg ]

    pol, e, m = good_sings[0]
    if isinstance(pol, Polynomial):
        s = pol.roots(QQbar, multiplicities=False)[0]
    else:
        s = 'infinity'

    return s, e, m


def good_base_point(dop):

    s, e, m = good_singular_point(dop)
    if s=='infinity': return 0

    z0 = s.real().ceil()
    sings = LinearDifferentialOperator(dop)._singularities(QQbar)
    while z0 in sings: z0 = z0 + QQ.one()

    return z0

def minimal_multiplicity(dop, pol):

    """
    Return (e, m) where e is an exponent of dop at a root of pol with a minimal
    multiplicity modulo ZZ (m).

    -> Representative of smallest real part (pas OK)
    -> minimal algebraic degree (OK)
    """

    z, r = dop.base_ring().gen(), dop.order()

    if (pol*z).is_one() or pol.degree()==1:
        N = dop.indicial_polynomial(pol)
    else:
        s = pol.roots(QQbar, multiplicities=False)[0]
        newdop, s = LinearDifferentialOperator(dop).extend_scalars(s)
        z, r = newdop.base_ring().gen(), dop.order()
        N = newdop.indicial_polynomial(z - s)
    exponents = N.roots(QQbar)
    exponents.sort(key = lambda x: x[0].degree())

    good_exponent, min_mult = exponents[0][0], r
    done_indices = []
    for i, (e, m) in enumerate(exponents):
        if not i in done_indices:
            multiplicitymodZZ = m
            for j, (f, n) in enumerate(exponents[(i+1):]):
                if e - f in ZZ:
                    multiplicitymodZZ += n
                    done_indices.append(i + 1 + j)
            if multiplicitymodZZ<min_mult:
                min_mult = multiplicitymodZZ
                good_exponent = e
            #done_indices.append(i) --> useless
    return good_exponent, min_mult
