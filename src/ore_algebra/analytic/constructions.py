# -*- coding: utf-8 - vim: tw=80
"""
Some functions to work with constructions (direct sum, adjoint, tensor product,
symmetric product, etc...).
"""

# Copyright 2022 Alexandre Goyer, Inria de Saclay
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/


from sage.matrix.constructor import matrix
from sage.rings.all import ZZ, QQ, QQbar, CBF, RationalField
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing


def indices(n, r):
    """
    Return the list [ [0,...,0,0,0], [0,...,0,0,1], ..., [0,...,0,0,r-1],
    [0,...,0,1,1], [0,...,0,1,2], ..., [0,...,0,1,r-1], [0,...,0,2,2], ...,
    [r-2,r-1,...,r-1,r-2], [r-2,r-1,...,r-1,r-1], [r-1,...,r-1] ].

    It corresponds to an enumeration of the ascending sequences i1, i2, ..., in
    with each ij in {0, ..., r-1}.
    """

    def f(I, ind):
        if len(I)<n:
            for i in range(max(I, default=0), r): f(I+[i], ind)
        else:
            ind.append(I)
        return

    ind = []
    f([], ind)

    return ind

def symmetric_power(sys, n):
    """
    Compute the ``n``-th symmetric power of the system ``sys``.
    """

    r = sys.nrows()
    ind = indices(n, r)
    s = len(ind)

    def g(I, out):
        if len(I)<n:
            for i in range(max(I, default=0), r): g(I+[i], out)
        else:
            for j in range(r):
                for k in range(n):
                    J, l = I[:k] + I[(k+1):], 0
                    while l<n-1 and j>J[l]: l += 1
                    J = J[:l] + [j] + J[l:]
                    out[s*ind.index(I) + ind.index(J)] += sys[I[k], j]
        return

    out = [0]*(s*s)
    g([], out)

    return matrix(s, s, out)
