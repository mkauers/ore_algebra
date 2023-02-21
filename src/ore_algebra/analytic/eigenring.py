# du code sale pour calculer l'eigenring de manière symbolique avec les moyens de sage

from ore_algebra.ideal import uncouple, solve_triangular_system
from ore_algebra.tools import clear_denominators
from sage.arith.functions import lcm
from sage.arith.misc import gcd
from sage.functions.other import binomial


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


def euler_representation(dop):

    r"""
    Return the list of the coefficients of dop with respect to the powers of
    z*Dz.
    """

    z, n = dop.base_ring().gen(), dop.order()
    output = [ dop[0] ] + [0]*n
    l = [0] # coefficients of T(T-1)...(T-k+1) (initial: k=0)

    for k in range(1, n+1):

        newl = [0]
        for i in range(1, len(l)):
            newl.append((-k+1)*l[i]+l[i-1])
        l = newl + [1]

        ck = dop[k]
        for j in range(1, k+1):
            output[j] += ck*z**(-k)*l[j]

    return output



def dop_valuation(dop):

    r"""
    Return the smallest i such that a_{i,j} != 0 for some j where
    dop = sum a_{i,j} z^i (z*Dz)^j.
    """

    v = min(a.valuation() for a in euler_representation(dop))

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
