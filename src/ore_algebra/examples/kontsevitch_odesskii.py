# coding: utf-8
r"""
p-determinants and monodromy, after Kontsevich and Odesskii

This module implements a heuristic analytic method for computing the
series h(t) defined in: Maxim Kontsevich and Alexander Odesskii,
*p-Determinants and monodromy of differential operators*, arXiv:2009.12159
[math.AG].

The method implemented here is a slower(?) alternative to the algorithm
suggested in the article, and provides an original illustration of
extreme-precision computations with generalized series expansions at regular
singularities.

Thanks to Alin Bostan (and, indirectly, Duco van Straten) for suggesting this
example.

EXAMPLE::

    sage: from ore_algebra.examples.kontsevitch_odesskii import h, h_series
    sage: h_series(10)
    [0,
     0,
     1/4,
     1/24,
     101/576,
     239/17280,
     19153/115200,
     -1516283/72576000,
     23167560743/121927680000,
     -5350452180523/76814438400000]
"""

import logging, sys
from sage.all import ( ceil, ComplexBallField, ComplexField, i, matrix, pari,
        pi, PolynomialRing, QQ, sqrt, ZZ, )
from ore_algebra import DifferentialOperators

logging.basicConfig()

logger = logging.getLogger("ore_algebra.examples.kontsevitch_odesskii")

def h(t, prec):

    Dop, x, Dx = DifferentialOperators(QQ)
    L = Dx * (x*(x-1)*(x-t)) * Dx + x

    hprec = prec + 100
    C = ComplexField(hprec)
    CBF = ComplexBallField(hprec)

    # Formal monodromy + connection matrices
    base = t/2
    m1 = L.numerical_transition_matrix([0,base], ZZ(2)**(-prec))
    m2 = L.numerical_transition_matrix([t, base], ZZ(2)**(-prec))
    delta = matrix(CBF, [[1, 0], [2*pi*i, 1]])
    mat = m1*delta*~m1*m2*delta*~m2

    # log(eigenvalue)
    tr = mat.trace().real().mid()
    Pol, la = PolynomialRing(C, 'la').objgen()
    char = (la+1/la-tr).numerator()
    rt = char.roots(multiplicities=False)[0]
    val = (rt.log()/C(2*i*pi))**2

    return val

def reconstruct(val, terms, t):

    coeffs = []
    cur = val
    for _ in range(terms):
        # simplest_rational is too slow
        rat = QQ(pari.bestappr(cur, (1/t).sqrt()))
        cur = (cur - rat)/t
        coeffs.append(rat)
    return coeffs

def h_series(terms):

    # Expected size, corresponding working precision
    sz = ceil((terms + 1)**2 * sqrt(ZZ(terms + 1).nbits())/2)
    prec = terms*(sz + 4)
    logger.info("terms=%s, size=%s, prec=%s", terms, sz, prec)
    t = ZZ(2)**(-sz)
    val = h(t, prec)
    logger.info("h(t) = %s", val)
    ser = reconstruct(val, terms, t)
    return ser


# print([(coeffs[k+1].denom()/coeffs[k].denom()) for k in range(2, len(coeffs)-2)])
