
"""
Ideals
"""

#############################################################################
#  Copyright (C) 2015, 2016, 2017                                           #
#                Manuel Kauers (mkauers@gmail.com),                         #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

from __future__ import absolute_import, division, print_function

from datetime import datetime
from functools import cmp_to_key, reduce
from sage.rings.noncommutative_ideals import Ideal_nc
from sage.arith.all import gcd, lcm
from sage.misc.all import prod, add
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.modules.free_module_element import vector
from sage.matrix.constructor import Matrix, matrix
from sage.misc.cachefunc import cached_function
from sage.misc.lazy_string import lazy_string
from sage.structure.all import coercion_model
from sage.structure.element import RingElement, canonical_coercion
from sage.rings.fraction_field import is_FractionField

from . import nullspace

from .tools import clear_denominators

class OreLeftIdeal(Ideal_nc):

    def __init__(self, ring, gens, coerce=True, is_known_to_be_a_groebner_basis=False):
        if not ring.base_ring().is_field():
            ring = ring.change_ring(ring.base_ring().fraction_field())
        gens = tuple([ring(g).numerator() for g in gens])
        if is_known_to_be_a_groebner_basis:
            self.__gb = gens        
        Ideal_nc.__init__(self, ring, gens, coerce, "left")

    def _lm_poly_ideal(self):
        R = self.ring().associated_commutative_algebra().change_ring(QQ)
        return R.ideal([g.lm().polynomial().change_ring(QQ) for g in self.groebner_basis()])

    def __eq__(self, other):
        """
        Checks two ideals for being equal. Ideals not belonging to the same ring will raise an error.
        """

        if self.ring() is not other.ring(): 
            raise ValueError
        
        # if rings are the same, then in particular the orders are the same
        A = self.groebner_basis()
        B = other.groebner_basis()

        if len(A) != len(B):
            return False
        
        return all(A[i].lc()*B[i] == B[i].lc()*A[i] for i in range(len(A)))

    def __le__(self, other):
        """
        Checks whether self is contained in other. If the ideals belong to different rings, a ValueError is thrown.
        """
        if self.ring() is not other.ring():
            raise ValueError

        G = other.groebner_basis()
        return all(b.reduce(G) == 0 for b in self.gens())

    def __ge__(self, other):
        return other <= self

    def _contains_(self, other):
        return other.reduce(self.groebner_basis()).is_zero()
    
    def dimension(self):
        """
        Returns the Hilbert dimension of self.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: A.ideal([Dx - 5]).dimension()
           1
           sage: A.ideal([Dx - 3, Dy - 4]).dimension()
           0

        """
        return self._lm_poly_ideal().dimension()

    def vector_space_dimension(self):
        """
        Returns the vector space dimension of the A-module A/self 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: A.ideal([Dx - 5]).vector_space_dimension()
           +Infinity
           sage: A.ideal([Dx - 3, Dy - 4]).vector_space_dimension()
           1
           sage: A.ideal([Dx^3 - 3, Dy^9 - 4]).vector_space_dimension()
           27

        """
        return self._lm_poly_ideal().vector_space_dimension()

    def vector_space_basis(self):
        """
        Returns a vector space basis of the A-module A/self if self is zero dimensional. 

        Raises an assertion error if self is not zero dimensional. 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: A.ideal([Dx - 3, Dy - 4]).vector_space_basis()
           [1]
           sage: A.ideal([Dx^2 - 3, Dy^4 - 4]).vector_space_basis()
           [1, Dy, Dx, Dy^2, Dx*Dy, Dy^3, Dx*Dy^2, Dx*Dy^3]

        """
        if self.dimension() == -1:
            return []
        
        assert self.dimension() == 0
        
        lms = self._lm_poly_ideal()
        basis = [lms.ring().one()]

        for g in lms.ring().gens():
            newbasis = []
            for i in range(len(basis)):
                newbasis.append(basis[i])
                while (g*newbasis[-1]).reduce(lms) == g*newbasis[-1]:
                    newbasis.append(g*newbasis[-1])
            basis = newbasis

        basis.sort(key=smallest_lt_first)
        return list(map(self.ring(), basis))

    def multiplication_matrix(self, idx):
        """
        Returns the multiplication matrix associated to the given generator of the ambient ring.

        Raises an assertion error if self is not zero dimensional. 

        INPUT:

           idx -- a generator of the anbient algebra. May be either an integer or an Ore algebra element
                  or a string representation of the generator

        OUTPUT:

           A matrix M such that when v is the coefficient vector of some element L of A/self with respect
           to the basis given by self.vector_space_basis(), then M*sigma(v) + delta(v) is the coefficient
           vector of D*L, where D is the idx'th generator of self.ring().

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: id = A.ideal([Dx - 2, Dy - 3])
           sage: id.multiplication_matrix(Dx)
           [2]
           sage: id.multiplication_matrix(Dy)
           [3]
           sage: id = A.ideal([Dx + (2*x + 2*y - 1)*Dy - 1, (2*x + 2*y)*Dy^2 + Dy])
           sage: id.multiplication_matrix(Dx)
           [               1                0]
           [  -2*x - 2*y + 1 (-1)/(2*x + 2*y)]
           sage: id.multiplication_matrix(Dy)
           [               0                0]
           [               1 (-1)/(2*x + 2*y)]

           sage: R.<n,k> = ZZ[]
           sage: A.<Sn,Sk> = OreAlgebra(R)
           sage: id = A.ideal([1 + 6*k + 6*n + (10 - 8*k - 8*n)*Sk + (-3 + 2*k + 2*n)*Sk^2 - 4*Sn, 1 - Sk - Sn + Sk*Sn,  1 - 4*k - 4*n + Sk + (-7 + 6*k + 6*n)*Sn + (3 - 2*k - 2*n)*Sn^2])
           sage: id.multiplication_matrix(Sn)
           [                               0                               -1 (-4*n - 4*k + 1)/(2*n + 2*k - 3)]
           [                               0                                1                1/(2*n + 2*k - 3)]
           [                               1                                1  (6*n + 6*k - 7)/(2*n + 2*k - 3)]
           sage: id.multiplication_matrix(Sk)
           [                               0 (-6*n - 6*k - 1)/(2*n + 2*k - 3)                               -1]
           [                               1 (8*n + 8*k - 10)/(2*n + 2*k - 3)                                1]
           [                               0                4/(2*n + 2*k - 3)                                1]

        """
        D = self.ring().gen(self.ring()._gen_to_idx(idx))
        G = self.groebner_basis()
        B = self.vector_space_basis()

        mat = [(D*b).reduce(G) for b in B]
        mat = [[m[b.exp()] for b in B] for m in mat]
        return matrix(self.ring().base_ring(), mat).transpose()
        
    def groebner_basis(self, infolevel=0, update_hook=None):
        """
        Returns the Groebner basis of this ideal. 

        INPUT:

        - infolevel -- integer indicating the verbosity of progress reports
        - update_hook -- a function which is envoked on (G, C, h) right before
          a new nonzero polynomial h is integrated into the basis G (a list of
          operators). The list C contains the critical pairs, each pair is
          encoded as a tuple (lcm(lm(a),lm(b)), a, b, s), where a and b are
          operators and s is the sugar associated to the S-polynomial of a and
          b. The hook facility gives a possibility to interfere with the
          computation. Fiddling with the lists G and C may destroy correctness
          or termination.

        OUTPUT:

        The Groebner basis of self. The output is cached.

        The monomial order is taken from the ambient ring (the common parent of
        the generators of self). Conceptually, the base ring of the ambient
        ring is treated as a field, even if it is in fact only a polynomial
        ring.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: sorted(A.ideal([Dy^5*(Dx + (-1 + 2*x - 2*y)*Dy - 1), (-2 + 2*x - 2*y)*Dy^2 - 3*Dy]).groebner_basis())
           [(2*x - 2*y - 2)*Dy^2 + (-3)*Dy, (2*x - 2*y - 2)*Dx*Dy + 3*Dy]
           sage: sorted(A.ideal([Dy^3*(Dx + (-1 + 2*x - 2*y)*Dy - 1), Dx^2*((-2 + 2*x - 2*y)*Dy^2 - 3*Dy)]).groebner_basis())
           [3*Dx^2*Dy + (-4)*Dx*Dy^2 + (-7)*Dy^3,
            (2*x - 2*y - 2)*Dy^4 + (-7)*Dy^3,
            (2*x - 2*y - 2)*Dx*Dy^3 + 7*Dy^3]

           sage: R.<n,k> = ZZ[]
           sage: A.<Sn,Sk> = OreAlgebra(R)
           sage: A.ideal([-5+2*k-n + Sk + (3-2*k+n)*Sn, -2+4*k-2*n + (7-6*k+3*n)*Sk + (-3+2*k-n)*Sk^2]).groebner_basis()
           [(n - 2*k + 3)*Sn + Sk - n + 2*k - 5,
            (n - 2*k + 3)*Sk^2 + (-3*n + 6*k - 7)*Sk + 2*n - 4*k + 2]

        """
        try:
            return list(self.__gb)
        except:
            pass

        gens = self.gens()
        if all(g.is_zero() for g in gens):
            self.__gb = ()
            return []

        A = gens[0].parent()
        A = A.change_ring(A.base_ring().ring())
        gens = list(map(A, gens))
        
        # ~~~ relatively naive code ~~~

        # tools
        def info(i, msg):
            if infolevel >= i:
                print(msg)

        X = list(map(A, self.ring().gens()))
        def maketerm(e): # exponent vector to monomial
            return prod( x**i for x, i in zip(X, e) )

        def lcm(u, v): # exponent vector of lcm(lt(u),lt(v))
            return u.exp().emax(v.exp())
        
        def makepair(u, v): # first entry is lcm(lt(u),lt(v))
            ht = maketerm(lcm(u, v))
            s = max(ht.tdeg() - u.lm().tdeg() + u.sugar, ht.tdeg() - v.lm().tdeg() + v.sugar)
            return (ht, u, v, s)

        def update(G, B, h): # inspired by alg UPDATE in BWK p 230, but without using the first criterion 
            if h.is_zero():
                info(2, "new polynomial is zero")
                return G, B
            info(2, "new polynomial has lm=" + str(h.lm()) + ", determine new pairs...")
            if update_hook is not None:
                info(1, "invoking update_hook...")
                update_hook(G, B, h)
            C = [g for g in G]
            C.sort(key=smallest_lt_first, reverse=True) # smallest leading term last
            # 1. discard the pairs (C[i],h) for which there is another pair (C[j], h) with lt(C[j],h)|lt(C[i],h)
            for i in range(len(C)): 
                for j in range(len(C)):
                    if C[j] is not None and i != j and min(lcm(C[i],h).esub(lcm(C[j],h))) >= 0:
                        C[i] = None
                        break
            C = [makepair(g, h) for g in C if g is not None]
            # 2. discard an old pair (u,v) if (u,h,v) is a bb-pair but (h,u,v) and (u,v,h) are not
            for b in B:
                if min(b[0].exp().esub(h.exp())) < 0 or b[0].exp() in [lcm(h, b[1]), lcm(h, b[2])]:
                    C.append(b)
            C.sort(key=lambda c: -c[-1]) # smallest pair last
            # 3. update basis
            G = [g for g in G if min(g.exp().esub(h.exp())) < 0] + [h]
            return G, C

        # initialization
        info(1, "initialization...")
        G = [] # current basis, sorted such that smallest leading term comes first
        C = [] # current list of critical pairs, sorted such that smallest lcm comes last
        for g in gens:
            g.sugar = g.tdeg()
            G, C = update(G, C, g.reduce(G, normalize=True, coerce=False))

        # buchberger loop
        info(1, "main loop...")
        while len(C) > 0:
            t, u, v, s = C.pop()
            info(2, datetime.today().ctime() + ": " + str(len(C) + 1) + " pairs left; taking pair with lcm(lm,lm)=" + str(t))            
            uterm = v.lc()*maketerm(t.exp().esub(u.exp()))
            vterm = u.lc()*maketerm(t.exp().esub(v.exp()))
            spol = uterm*u - vterm*v; spol.sugar = s
            G, C = update(G, C, spol.reduce(G, normalize=True, infolevel=infolevel-2, coerce=False))

        # autoreduction
        info(2, "autoreduction...")
        for i in range(len(G)): 
            G[i] = G[i].reduce(G[:i] + G[i+1:], normalize=True, infolevel=infolevel-3, coerce=False)
        G = [g for g in G if not g.is_zero()]
        G.sort(key=smallest_lt_first)

        # todo: normalize coefficients of coefficients to ensure uniqueness
        
        # done
        info(1, "completion completed, Groebner basis has " + str(len(G)) + " elements.")
        self.__gb = tuple(G)
        return G

    def operator_to_vector(self, L):
        """
        Returns the vector in R^n corresponding to the equivalence class of L in A/self.
        This only works if self is a zero dimensional ideal.

        INPUT:

           L -- an element of the ambient algebra A of 'self'

        OUTPUT:

           a vector of base ring elements 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: I = A.ideal([(-4*y^3 + 2*y)*Dy^2 + (20*y^4 + 4*y^2 - 3)*Dy - 50*y^3 + 5*y, (-10*y^2 + 5)*Dx^2 + (-6*x*y + 10*y)*Dy + 30*x*y^2 - 25])
           sage: I.operator_to_vector(3*Dx^2 + (5*x+y)*Dy - 3*x+8*y)
           ((12*x*y^2 + 16*y^3 + 3*x - 8*y - 15)/(2*y^2 - 1), (50*x*y^2 + 10*y^3 - 18*x*y - 25*x + 25*y)/(10*y^2 - 5), 0, 0)
           sage: I.vector_to_operator(_)
           ((50*x*y^2 + 10*y^3 - 18*x*y - 25*x + 25*y)/(10*y^2 - 5))*Dy + (12*x*y^2 + 16*y^3 + 3*x - 8*y - 15)/(2*y^2 - 1)
           sage: (3*Dx^2 + (5*x+y)*Dy - 3*x+8*y - _).reduce(I)
           0 
        
        """
        L = self.ring()(L).reduce(self.groebner_basis())
        return vector(self.ring().base_ring(), [L.monomial_coefficient(b) for b in self.vector_space_basis()])

    def vector_to_operator(self, vec):
        """
        Returns a representative of the equivalence class of A/self corresponding to the given coefficient vector.

        INPUT:
        
           vec -- a vector with coefficients in self.ring().base_ring() and dimension self.vector_space_dimension()

        OUTPUT:

           an operator L in A such that self.operator_to_vector(L) == vec

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: I = A.ideal([(-4*y^3 + 2*y)*Dy^2 + (20*y^4 + 4*y^2 - 3)*Dy - 50*y^3 + 5*y, (-10*y^2 + 5)*Dx^2 + (-6*x*y + 10*y)*Dy + 30*x*y^2 - 25])
           sage: I.operator_to_vector(3*Dx^2 + (5*x+y)*Dy - 3*x+8*y)
           ((12*x*y^2 + 16*y^3 + 3*x - 8*y - 15)/(2*y^2 - 1), (50*x*y^2 + 10*y^3 - 18*x*y - 25*x + 25*y)/(10*y^2 - 5), 0, 0)
           sage: I.vector_to_operator(_)
           ((50*x*y^2 + 10*y^3 - 18*x*y - 25*x + 25*y)/(10*y^2 - 5))*Dy + (12*x*y^2 + 16*y^3 + 3*x - 8*y - 15)/(2*y^2 - 1)
           sage: (3*Dx^2 + (5*x+y)*Dy - 3*x+8*y - _).reduce(I)
           0 

        """
        return self.ring()(dict((b.exp(), vec[i]) for i, b in enumerate(self.vector_space_basis())))
    
    def intersection(self, other, infolevel=0, solver=None):
        """
        Computes the intersection of self with the other ideal.

        INPUT:

           other -- a left ideal with the same ambient ring as self
           infolevel (optional) -- verbosity of progress reports
           solver (optional) -- callable to be used for finding right kernels of matrices over the ambient algebra's base ring

        OUTPUT:

           The intersection of self with other. 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<n,k> = ZZ[]
           sage: A.<Sn,Sk> = OreAlgebra(R)
           sage: id1 = A.ideal([Sk-2*k,Sn-3])
           sage: id2 = A.ideal([Sk-3,Sn-n])
           sage: id3 = id1.intersection(id2)
           sage: id3
           Left Ideal ((-2*k + 3)*Sn + (-n + 3)*Sk + 2*n*k - 9, (-2*k + 3)*Sk^2 + (4*k^2 + 4*k - 9)*Sk - 12*k^2 + 6*k) of Multivariate Ore algebra in Sn, Sk over Fraction Field of Multivariate Polynomial Ring in n, k over Integer Ring
           sage: id3 == id3.intersection(id3)
           True
           sage: id3 <= id1
           True
           sage: id3 <= id2
           True

           sage: R.<x, y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: id1 = A.ideal([2*Dx-Dy, (x+2*y)*Dy^2 + Dy])
           sage: id2 = A.ideal([Dy - x, Dx - y])
           sage: id3 = id1.intersection(id2)
           sage: id3
           Left Ideal ((-x^2 + 4*y^2)*Dy^2 + (-2*x^3 - 4*x^2*y - 2*x)*Dx + (x^3 + 2*x^2*y + 2*y)*Dy, (-x^2 + 4*y^2)*Dx*Dy + (-2*x^2*y - 4*x*y^2 - 3*x - 4*y)*Dx + (x^2*y + 2*x*y^2 + x + 3*y)*Dy, (-2*x^2 + 8*y^2)*Dx^2 + (-4*x*y^2 - 8*y^3 - x)*Dx + (2*x*y^2 + 4*y^3 + y)*Dy) of Multivariate Ore algebra in Dx, Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring
           sage: id3 == id3.intersection(id3)
           True
           sage: id3 <= id1
           True
           sage: id3 <= id2
           True

        """
        if max(self.dimension(), other.dimension()) > 0:
            raise NotImplementedError

        n = self.vector_space_dimension()
        m = other.vector_space_dimension()

        if n == 0:
            return other
        elif m == 0:
            return self

        A = self.ring()
        gen_matrices = {}
        self_zero = matrix(A.base_ring(), n, m); other_zero = matrix(A.base_ring(), m, n)
        for X in A.gens():
            # [ M_self     0    ] 
            # [  0      M_other ]
            gen_matrices[X] = other_zero.augment(other.multiplication_matrix(X).transpose()).transpose()
            gen_matrices[X] = self.multiplication_matrix(X).transpose().augment(self_zero).transpose().augment(gen_matrices[X])
            
        return A.ideal(fglm(A, vector(A.base_ring(), [1] + [0]*(n-1) + [1] + [0]*(m-1)), \
                            gen_matrices, infolevel=infolevel, solver=solver), is_known_to_be_a_groebner_basis=True)

    def symmetric_product(self, other, infolevel=0, solver=None):
        """
        Computes the symmetric product of self with the other ideal.

        INPUT:

           other -- a left ideal with the same ambient ring as self
           infolevel (optional) -- verbosity of progress reports
           solver (optional) -- callable to be used for finding right kernels of matrices over the ambient algebra's base ring

        OUTPUT:

           The symmetric product of self with other. 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<n,k> = ZZ[]
           sage: A.<Sn,Sk> = OreAlgebra(R)
           sage: id1 = A.ideal([Sk-2*k,Sn-3])
           sage: id2 = A.ideal([Sk-3,Sn-n])
           sage: id3 = A.ideal([Sk^2-3*k*Sk+1,Sn-3])
           sage: id1.intersection(id2).symmetric_product(id3)
           Left Ideal ((-6*k^2 + 3*k)*Sn*Sk + (-3*n + 9)*Sk^2 + (4*k^2 + 4*k - 9)*Sn + (18*n*k^2 + 18*n*k - 81*k)*Sk - 12*n*k^2 - 12*n*k + 81, (-n + 3)*Sn^2 + (3*n^2 + 3*n - 27)*Sn - 27*n^2 + 54*n, (18*n*k^2 - 9*n*k - 54*k^2 + 27*k)*Sk^3 + (-108*n*k^4 - 432*n*k^3 + 324*k^4 - 285*n*k^2 + 1296*k^3 + 63*n*k + 855*k^2 - 3*n - 189*k + 9)*Sk^2 + (216*k^5 + 200*k^4 - 442*k^3 - 548*k^2 - 50*k - 9)*Sn + (972*n*k^5 + 2430*n*k^4 - 2916*k^5 + 1836*n*k^3 - 7290*k^4 + 324*n*k^2 - 5508*k^3 - 54*n*k - 972*k^2 + 162*k)*Sk - 648*n*k^5 - 1572*n*k^4 - 1104*n*k^3 + 2916*k^4 - 192*n*k^2 + 7290*k^3 - 12*n*k + 5508*k^2 + 486*k + 81) of Multivariate Ore algebra in Sn, Sk over Fraction Field of Multivariate Polynomial Ring in n, k over Integer Ring
           sage: (id1.symmetric_product(id3)).intersection(id2.symmetric_product(id3)) == _
           True

           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: id1 = A.ideal([(x+y)*Dx + 1, (x+y)*Dy + 1])
           sage: id2 = A.ideal([Dx - 1, y*Dy + 3])
           sage: id3 = A.ideal([Dx^2 + 3*Dx - x, y^2*Dy + 2])
           sage: id1.intersection(id2).symmetric_product(id3)
           Left Ideal ((3*x^2*y^4 + 5*x*y^5 + 2*y^6)*Dy^2 + (12*x^2*y^3 + 24*x*y^4 + 10*y^5 + 12*x^2*y^2 + 20*x*y^3 + 8*y^4)*Dy + 12*x*y^3 + 6*y^4 + 12*x^2*y + 28*x*y^2 + 12*y^3 + 12*x^2 + 20*x*y + 8*y^2, (9*x^3*y + 21*x^2*y^2 + 16*x*y^3 + 4*y^4)*Dx^2 + (6*x^3*y^2 + 16*x^2*y^3 + 14*x*y^4 + 4*y^5 + 6*x^2*y^2 + 10*x*y^3 + 4*y^4)*Dx*Dy + (27*x^3*y + 69*x^2*y^2 + 58*x*y^3 + 16*y^4 + 12*x^3 + 50*x^2*y + 58*x*y^2 + 20*y^3 + 12*x^2 + 20*x*y + 8*y^2)*Dx + (6*x^3*y^2 + 16*x^2*y^3 + 14*x*y^4 + 4*y^5 + 9*x^2*y^2 + 13*x*y^3 + 4*y^4 - 2*y^3)*Dy - 9*x^4*y - 21*x^3*y^2 - 16*x^2*y^3 - 4*x*y^4 + 6*x^2*y^2 + 10*x*y^3 + 4*y^4 + 12*x^3 + 59*x^2*y + 67*x*y^2 + 20*y^3 + 18*x^2 + 26*x*y + 2*y^2 - 4*y) of Multivariate Ore algebra in Dx, Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring
           sage: (id1.symmetric_product(id3)).intersection(id2.symmetric_product(id3)) == _
           True

        """
        if max(self.dimension(), other.dimension()) > 0:
            raise NotImplementedError

        n = self.vector_space_dimension()
        m = other.vector_space_dimension()

        if n > m:
            return other.symmetric_product(self)
        # now n <= m
        elif n == 0:
            return self
        elif n == 1:
            pass ## todo: special code for D-finite times hyper. 

        A = self.ring()
        gen_matrices = {}
        from sage.matrix.matrix_space import MatrixSpace
        I_nm = MatrixSpace(A.base_ring(), n*m, n*m).one()
        I_n = MatrixSpace(A.base_ring(), n, n).one()
        I_m = MatrixSpace(A.base_ring(), m, m).one()
        for X in A.gens():
            (w0, w1, w2) = A._product_rule(X) # raises an exception if no product rule is available
            # w0*I_(nm) + w1*(self.mm(X) tensor I_m + I_n tensor other.mm(X)) + w2*(self.mm(X) tensor other.mm(X))
            self_MM = self.multiplication_matrix(X)
            other_MM = other.multiplication_matrix(X)
            M = w0*I_nm
            if w1 != 0:
                M += w1*(self_MM.tensor_product(I_m) + I_n.tensor_product(other_MM))
            if w2 != 0:
                M += w2*(self_MM.tensor_product(other_MM))
            gen_matrices[X] = M
            
        return A.ideal(fglm(A, vector(A.base_ring(), [1] + [0]*(n*m-1)), \
                            gen_matrices, infolevel=infolevel, solver=solver), is_known_to_be_a_groebner_basis=True)

    def eliminate(self, vars, infolevel=0, solver=None, early_termination=False):
        """
        Returns the elimination ideal of this with the algebra in which the generators in the list 'vars' are eliminated.

        INPUT:
        
           vars -- list of generators of the ambient algebra to be eliminated
           infolevel (optional) -- verbosity of progress reports
           solver (optional) -- callable to be used for finding right kernels of matrices over the ambient algebra's base ring
           early_termination (optional) -- if set to True, the computation is stopped as soon as the first nonzero element of the elimination ideal is found

        OUTPUT:

           The elimination ideal of 'self' with respect to the specified variables

        The current implementation is limited to zero dimensional ideals.           

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: id = A.ideal([(4*y^3 - 2*y)*Dy^2 + (-20*y^4 - 4*y^2 + 3)*Dy + 50*y^3 - 5*y, Dx*Dy + (-5*y)*Dx + (-3*x)*Dy + 15*x*y, (10*y^2 - 5)*Dx^2 + (18*x^2*y - 4*y)*Dy - 90*x^2*y^2 - 30*y^2 + 25])
           sage: id.eliminate([Dx])
           Left Ideal ((4*y^3 - 2*y)*Dy^2 + (-20*y^4 - 4*y^2 + 3)*Dy + 50*y^3 - 5*y) of Univariate Ore algebra in Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring
           sage: id.eliminate([Dy])
           Left Ideal ((-9*x^2 + 2)*Dx^3 + (27*x^3 + 12*x)*Dx^2 + (45*x^2 - 10)*Dx - 135*x^3 - 60*x) of Univariate Ore algebra in Dx over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring
           sage: A(_.gens()[0]).reduce(id.groebner_basis())
           0
        
        """

        if len(vars) == 0:
            return self        
        
        R = self.ring()
        vars = list(map(R, vars))
        vars = [g for g in R.gens() if g not in vars]
        target_algebra = R.subalgebra(vars)
        
        if self.dimension() == -1:
            return target_algebra.ideal([1])
        elif self.dimension() > 0:
            raise NotImplementedError

        mats = dict((g, self.multiplication_matrix(g)) for g in vars)
        return target_algebra.ideal(fglm(target_algebra, self.operator_to_vector(1), mats, infolevel=infolevel, solver=solver, early_termination=early_termination), is_known_to_be_a_groebner_basis=True)


    def annihilator_of_associate(self, L, infolevel=0, solver=None, early_termination=False):
        """
        Given an operator L of the ambient algebra of 'self', this computes an ideal of annihilating operators for L(f), where f
        is any solution of 'self'. 

        The method is only implemented for zero dimensional ideals. 

        INPUT:

           L -- an element of the ambient algebra of 'self'
           infolevel (optional) -- verbosity of progress reports
           solver (optional) -- callable to be used for finding right kernels of matrices over the ambient algebra's base ring
           early_termination (optional) -- if set to True, the computation is stopped as soon as the first nonzero element of the output ideal is found

        OUTPUT:

           an ideal of annihilating operators for L(f), where f is any solution of 'self'.
           If early_termination is set to True, the output ideal will have only one generator. 
           Otherwise, the output ideal will be zero dimensional. 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: I = A.ideal([(-4*y^3 + 2*y)*Dy^2 + (20*y^4 + 4*y^2 - 3)*Dy - 50*y^3 + 5*y, (-10*y^2 + 5)*Dx^2 + (-6*x*y + 10*y)*Dy + 30*x*y^2 - 25])
           sage: I.annihilator_of_associate(3*y*Dx-5*x*Dy)
           Left Ideal ((2400*x*y^7 - 480*x*y^5 + 120*x*y^3)*Dx*Dy + (-864*x*y^10 + 864*x*y^8 + 5000*x^2*y^6 + 1200*y^8 - 216*x*y^6 - 1000*x^2*y^4 - 720*y^6 + 50*x^2*y^2 + 60*y^4)*Dy^2 + (-12000*x*y^8 - 120*x*y^4 - 120*x*y^2)*Dx + (4320*x*y^11 + 432*x*y^9 - 25000*x^2*y^7 - 6000*y^9 - 2808*x*y^7 - 12500*x^2*y^5 - 3000*y^7 + 756*x*y^5 + 2250*x^2*y^3 + 2460*y^5 - 75*x^2*y - 210*y^3)*Dy - 15120*x*y^10 + 6048*x*y^8 + 37500*x^2*y^6 + 21000*y^8 + 2268*x*y^6 + 15000*x^2*y^4 - 756*x*y^4 - 2625*x^2*y^2 - 2310*y^4 + 75*x^2 + 210*y^2, (62500*x^3*y^6 - 12500*x^3*y^4 + 3000*x*y^6 + 3125*x^3*y^2 - 600*x*y^4 + 150*x*y^2)*Dx^2 + (-4500*x^2*y^8 + 7500*x^4*y^4 + 2250*x^2*y^6 + 216*y^8 - 12500*x^3*y^4 + 12500*x*y^6 + 360*x^2*y^4 - 108*y^6 - 7350*x*y^4 + 250*x*y^2)*Dy^2 + (-125000*x^2*y^6 + 25000*x^2*y^4 - 6250*x^2*y^2)*Dx + (22500*x^2*y^9 + 13500*x^2*y^7 - 1080*y^9 - 62500*x*y^7 - 30000*x^4*y^3 - 7875*x^2*y^5 - 648*y^7 + 50000*x^3*y^3 - 25000*x*y^5 - 1440*x^2*y^3 + 378*y^5 + 23775*x*y^3 - 375*x*y)*Dy - 187500*x^4*y^6 - 78750*x^2*y^8 + 37500*x^4*y^4 - 16875*x^2*y^6 + 3780*y^8 + 218750*x*y^6 + 30000*x^4*y^2 + 9675*x^2*y^4 + 378*y^6 - 65625*x^3*y^2 + 3125*x*y^4 + 1440*x^2*y^2 - 378*y^4 - 23900*x*y^2 + 375*x, (80*y^7 - 16*y^5 + 4*y^3)*Dy^3 + (-400*y^8 - 480*y^6 + 60*y^4 - 12*y^2)*Dy^2 + (1600*y^7 + 1380*y^5 - 132*y^3 + 21*y)*Dy - 2100*y^6 - 1680*y^4 + 147*y^2 - 21) of Multivariate Ore algebra in Dx, Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring

        """
        if self.dimension() == -1:
            return self
        elif self.dimension() > 0:
            raise NotImplementedError
        
        A = self.ring()
        mats = dict((g, self.multiplication_matrix(g)) for g in A.gens())
        return A.ideal(fglm(A, self.operator_to_vector(L), mats, infolevel=infolevel, solver=solver, early_termination=early_termination), is_known_to_be_a_groebner_basis=True)


    def annihilator_of_composition(self, **kwargs):
        """
        Computes an ideal of annihilating operators for all functions f(g1,...,gn) where f(x1,...,xn) is 
        any function annihilated by 'self' and g1,...,gn are elements of an algebraic function field. 

        INPUT:

           A list of optional arguments that state which variables (lhs) are to be mapped to which algebraic functions (rhs).
           All right hand sides must be coercible to a common parent, which must be either a rational function field or an
           algebraic extension of a rational function field. Variables for which no substitution rule is specified are mapped 
           to themselves.

           infolevel (optional) -- nonnegative integer indicating verbosity of progress reports
           early_termination (optional) -- if set to true, this stops as soon as the first nonzero ideal element has been found

        OUTPUT:

           An annihilating ideal for the composition of a solution of self with the specified algebraic functions.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: K.<t> = R.fraction_field()['t']
           sage: K.<t> = R.fraction_field().extension(t^3 - x^2*(y+1))
           sage: S.<u,v> = ZZ[]
           sage: A.<Du,Dv> = OreAlgebra(S)
           sage: id = A.ideal([u^2*Du + (u*v-2)*Dv - u, (-2+2*u*v)*Dv^2 + u*Dv])
           sage: id.annihilator_of_composition(u=x,v=y)
           Left Ideal (x^2*Dx + (x*y - 2)*Dy - x, (2*x*y - 2)*Dy^2 + x*Dy) of Multivariate Ore algebra in Dx, Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring
           sage: id.annihilator_of_composition(u=u^2,v=1/(u-v))
           Left Ideal ((-u^3)*Du + (-3*u^3 + 2*u^2*v + 4*u^2 - 8*u*v + 4*v^2)*Dv + 2*u^2, (-2*u^3 + 2*u^2*v + 2*u^2 - 4*u*v + 2*v^2)*Dv^2 + (3*u^2 - 4*u + 4*v)*Dv) of Multivariate Ore algebra in Du, Dv over Fraction Field of Multivariate Polynomial Ring in u, v over Integer Ring
           sage: id.annihilator_of_composition(v=0)
           Left Ideal (Dv, Du^2) of Multivariate Ore algebra in Du, Dv over Fraction Field of Multivariate Polynomial Ring in u, v over Integer Ring
           sage: id.annihilator_of_composition(u=t,v=1-t)
           Left Ideal ((-x)*Dx + (2*y + 2)*Dy, (216*x^4*y^5 + 1080*x^4*y^4 + 2160*x^4*y^3 + 2160*x^4*y^2 + 432*x^2*y^4 + 1080*x^4*y + 1728*x^2*y^3 + 216*x^4 + 2592*x^2*y^2 + 1728*x^2*y + 216*y^3 + 432*x^2 + 648*y^2 + 648*y + 216)*Dy^4 + (1440*x^4*y^4 + 5760*x^4*y^3 + 8640*x^4*y^2 + 5760*x^4*y + 2232*x^2*y^3 + 1440*x^4 + 6696*x^2*y^2 + 6696*x^2*y + 2232*x^2 + 792*y^2 + 1584*y + 792)*Dy^3 + (1920*x^4*y^3 + 5760*x^4*y^2 + 5760*x^4*y + 1920*x^4 + 2094*x^2*y^2 + 4188*x^2*y + 2094*x^2 + 336*y + 336)*Dy^2 + (320*x^4*y^2 + 640*x^4*y + 320*x^4 + 187*x^2*y + 187*x^2 - 16)*Dy + 3*x^2) of Multivariate Ore algebra in Dx, Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring

        """

        if len(kwargs) == 0: # no substitutions specified
            return self

        infolevel = kwargs.setdefault('infolevel', 0)
        del kwargs['infolevel']
        early_termination = kwargs.setdefault('early_termination', False)
        del kwargs['early_termination']

        def info(i, msg):
            if infolevel >= i:
                print(msg)
        
        # domains for outer function
        A = self.ring()                                          ## eg ZZ[x,y][Dx,Dy]
        K = A.base_ring().fraction_field()
        if not all(A.is_D(g) for g in A.gens()):
            raise NotImplementedError

        # domains for inner functions
        for x in K.gens(): kwargs.setdefault(str(x), x)          ## add missing arguments
        R = reduce(canonical_coercion, kwargs.values())[0].parent() ## eg QQ(u,v)(t)
        R = R.fraction_field()                                   
        R0 = R if is_FractionField(R) else R.base_ring()         ## eg QQ(u,v)
        from .ore_algebra import OreAlgebra
        B = OreAlgebra(R0, *['D' + str(x) for x in R0.gens()])   ## eg QQ(u,v)[Du,Du]

        info(1, "Outer functions recognized as " + ("rational" if is_FractionField(R) else "irrational algebraic"))
        info(1, "Output is going to be an ideal of " + repr(B))

        if self.dimension() == -1:
            return B.ideal([B.one()])

        # create derivatives
        der = {}
        if is_FractionField(R):
            mulmat = lambda m: matrix(R, [m])
            dim = 1
            for D in B.gens():
                der[D] = mulmat(0)
                for x in K.gens():
                    der[x, D] = mulmat(B.delta(D)(kwargs[str(x)]))
        else: # algebraic extension field
            mulmat = lambda m: R(m).matrix().transpose()
            minpoly = R.modulus().monic()
            dim = minpoly.degree()
            dermat = matrix(R0, dim, lambda i, j: j if i==j-1 else 0)
            q = minpoly.xgcd(minpoly.derivative())[2]
            for D in B.gens():
                d = B.delta(D)
                p = -R(q*minpoly.map_coefficients(d)) ## D of generator t of extension
                der[D] = mulmat(p)*dermat
                for x in K.gens():
                    u = R(kwargs[str(x)])
                    # apply D to image of x=u(y,t) : u_y + u_t*D(t)
                    der[x, D] = mulmat(R(list(map(d, u.list()))) + R(minpoly.parent()(u.list()).derivative())*p)
                
        info(1, "Construction of derivatives completed.")
        
        # create multiplication matrices
        n = self.vector_space_dimension()
        old_mats = {}
        for Da in A.gens():
            M = self.multiplication_matrix(Da)
            for i in range(n):
                for j in range(n):
                    # in the multiplication matrix M for Da wrt self, perform the substitution on the entries and record
                    # for each entry the multiplication matrix wrt the standard basis of the extension field.
                    old_mats[Da, i, j] = mulmat(R(M[i, j].numerator().subs(**kwargs)/M[i, j].denominator().subs(**kwargs)))

        info(1, "Substitution for multiplication matrices of A completed.")

        mats = {}
        for D in B.gens():
            blocks = {}
            for i in range(n):
                for j in range(n):
                    blocks[i, j] = sum(der[K(A.is_D(Da)), D]*old_mats[Da, i, j] for Da in A.gens())
                    if i == j:
                        blocks[i, j] += der[D] 
            mats[D] = matrix(R0, n*dim, lambda i, j: blocks[i//dim, j//dim][i%dim, j%dim])

        info(1, "Inflation of multiplication matrices completed.")
        # create one-vector
        one = [0]*(n*dim); one[0] = 1; one = vector(R0, one)

        info(1, "Entering FGLM...")
        return B.ideal(fglm(B, one, mats, infolevel=infolevel-1, solver=B._solver(R0), early_termination=early_termination), is_known_to_be_a_groebner_basis=True)

    
    def ct(self, D, algebra=None, certificates=True, early_termination=False, infolevel=0, iteration_limit=0):
        """
        Computes an ideal of telescopers for self.

        INPUT:

        - delta -- a telescoper-certificate pair is a pair (P, Q) such that P -
          D*Q is an element of self, P is an element of the specified algebra,
          and Q is an element of the ambient algebra of self. Typical choices
          are Dx or Sk - 1.
        - algebra -- a subalgebra of the ambient algebra of self. It is
          required that all elements of this subalgebra commute with D. If no
          subalgebra is specified, the method takes the subalgebra generated by
          all the generators of the ambient algebra of self which commute with D
        - certificates -- if True (default), the method returns a pair (T, C)
          where T is the telescoper ideal and C is a list whose ith element is
          a certificate for the ith generator of T
        - early_termination -- if True, the computation terminates as soon as
          the first nonzero telescoper has been found.
        - infolevel -- verbosity of progress report
        - iteration_limit -- if set to a positive integer, the computation is
          terminated as soon as the support of the telescopers in the ansatz
          exceeds the specified number.

        OUTPUT:

        The ideal of all telescopers of self with respect to D, and (if
        requested) a list of corresponding certificates

        EXAMPLES::

           sage: from ore_algebra import *
           sage: # binomial theorem
           sage: R.<n,k> = ZZ[]
           sage: A.<Sn,Sk> = OreAlgebra(R)
           sage: A.ideal([(k+1)*Sk+(k-n),(1-k+n)*Sn+(-1-n)]).ct(Sk-1) # random
           ([Sn - 2], [k/(k - n - 1)])

           sage: # gfun of legendre polynomials 
           sage: R.<t,n,x> = ZZ[]
           sage: A.<Dt,Dx,Sn> = OreAlgebra(R)
           sage: A.ideal([(-t+t*x^2)*Dx + (-1-n)*Sn+(t*x+n*t*x),(2+n)*Sn^2+(-3*t*x-2*n*t*x)*Sn+(t^2+n*t^2), t*Dt-n]).ct(Sn-1) # random
           ([(-t^2 + 2*t*x - 1)*Dx + t, (-t^2 + 2*t*x - 1)*Dt - t + x],
            [(((-t*x + 1)/(t*x^2 - t))*n + (-t*x + 1)/(t*x^2 - t))*Sn + ((2*t*x^2 - t - x)/(x^2 - 1))*n + (t*x^2 - x)/(x^2 - 1),
             (1/t*n + 1/t)*Sn + ((-2*t*x + 1)/t)*n - x])

           sage: # integral of a bivariate algebraic function
           sage: R.<x,y> = ZZ[]
           sage: K.<t> = R.fraction_field()['t']
           sage: K.<t> = R.fraction_field().extension((x-y)*t^3 + t - x^2*y - x^2)
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: A.ideal([x*Dx-1,y*Dy-1]).annihilator_of_composition(x=t).ct(Dy, certificates=False)
           [(252*x^5 - 108*x^4 - 81*x^3 + 36*x^2)*Dx^2 + (-504*x^4 + 108*x^3 + 36*x)*Dx + 224*x^3 + 360*x^2 + 66*x - 16]
        """

        def info(i, msg):
            if i <= infolevel:
                print(msg)
        
        # D must involve exactly one generator Dgen of the ambient algebra, and this generator must move exactly one
        # generator var of the ground ring. 
        A = self.ring()
        K = A.base_ring().fraction_field()
        Dgen = [Dgen for Dgen in A.gens() if D.degree(Dgen) != 0] # the unique generator of A appearing in D
        var = [k for k in A.base_ring().gens() if D*k != k*D] # D must commute with all but exactly one 
        if len(Dgen) != 1 or len(var) != 1:
            raise ValueError("bad choice of delta")
        Dgen, var = Dgen[0], var[0]

        from .ore_algebra import OreAlgebra
        
        if algebra is None:
            G = A.base_ring().base_ring()
            Ggens = tuple(x for x in A.base_ring().gens() if x != var)
            G = G[Ggens].fraction_field()
            algebra = OreAlgebra(G, *tuple((str(g), A.sigma(g), A.delta(g)) for g in A.gens() if g != Dgen))
            
        gen_matrices = dict( (d, self.multiplication_matrix(A(d))) for d in algebra.gens() )
        sigma = dict( (d, algebra.sigma(d)) for d in algebra.gens() )
        delta = dict( (d, algebra.delta(d)) for d in algebra.gens() )
        
        info(1, "Output ideal will belong to " + repr(algebra))
        info(2, "Summation/Integration variable recognized as " + str(var))
        
        # translate delta to matrix equation over univariate operator algebra
        info(1, "Constructing coupled system...")
        GG = algebra.base_ring().fraction_field()[var].fraction_field()
        AA = OreAlgebra(GG, (str(Dgen), {str(var):GG(A.sigma(Dgen)(var))}, {str(var):GG(A.delta(Dgen)(var))}))
        info(2, "Coupled system will have entries in " + repr(AA))

        def wrap_sigma(s):
            return lambda p: p.numerator().map_coefficients(s)/p.denominator().map_coefficients(s)
        def wrap_delta(d, s):
            if s is None:
                def fun(p):
                    a = p.numerator(); b = p.denominator()
                    return a.map_coefficients(d)/b - a*b.map_coefficients(d)/(b**2)
            else:
                def fun(p):
                    a = p.numerator(); b = p.denominator()
                    return a.map_coefficients(d)/b - a.map_coefficients(s)*b.map_coefficients(d)/(b*b.map_coefficients(s))
            return fun
        
        for d in algebra.gens():
            sigma[d] = None if sigma[d].is_identity() else wrap_sigma(sigma[d])
            delta[d] = None if delta[d].is_zero() else wrap_delta(delta[d], sigma[d])
        
        M = self.multiplication_matrix(Dgen).change_ring(GG)
        
        if A.is_D(Dgen) and D == Dgen:
            sys = [[AA(m) for m in row] for row in M]
            for i in range(len(sys)):
                sys[i][i] += AA(Dgen)
        elif A.is_S(Dgen) and D == Dgen - 1:
            sys = [[AA(m)*AA(Dgen) for m in row] for row in M]
            for i in range(len(sys)):
                sys[i][i] -= 1
        else:
            raise NotImplementedError

        # uncouple
        info(1, lazy_string(lambda: datetime.today().ctime() + ": Uncoupling coupled system..."))
        T, U = uncouple(sys, extended=True, infolevel=infolevel+2)
        info(2, lazy_string(lambda: datetime.today().ctime() + ": Uncoupling completed."))
        if algebra.is_D():
            @cached_function
            def action(p):
                return p.derivative()
            action_kw = {"action": action}
        else:
            action_kw = {}
        Ufy = lambda v: [sum(U[i][j](v[j], **action_kw) for j in range(len(v))) for i in range(len(U))]

        # start fglm-like procedure
        info(1, "Initiating FGLM-like iteration...")

        telescopers = []
        iterator = MonomialIterator(algebra)
        terms = {next(iterator)[0] : vector(GG, [1] + [0]*(len(T)-1))}
        coresolver = nullspace.kronecker(nullspace.gauss())
        zerocount = [0] # so many zero telescopers have been found so far
        solver = [nullspace.quick_check(coresolver, cutoffdim=0)]

        def findrelation():
            zero = 0; sol = []
            for (g, c) in solve_triangular_system(T, rhs, solver=solver[0]):
                if all(q.is_zero() for q in c):
                    zero += 1
                else:
                    sol.append((g, c))
            if zero > zerocount[0]:
                zerocount[0] = zero
                solver[0] = nullspace.quick_check(coresolver, cutoffdim=zero)
            if len(sol) == 0:
                return None
            P, Q = add([sol[0][1][i]*B[i] for i in range(len(B))]), None
            if certificates:
                Q = self.vector_to_operator(sol[0][0])
            return P, Q

        B = [algebra.one()] ## terms under the stairs
        rhs = [Ufy(terms[B[0]])] ## vectors corresponding to monomials in B, but in T-coordinates
        
        info(1, "next monomial: 1")
        info(2, lazy_string(lambda: datetime.today().ctime() + ": calling solver..."))
        sol = findrelation()

        if sol is not None:
            telescopers.append(sol)
        else:
            try:
                while True:
                    tau, d = next(iterator)
                    info(1, "next monomial: " + str(tau*d))
                    
                    # construct corresponding vector
                    v = terms[tau] if sigma[d] is None else terms[tau].apply_map(sigma[d])
                    v = gen_matrices[d]*v
                    if delta[d] is not None:
                        v += terms[tau].apply_map(delta[d])
                    terms[tau*d] = v
                    B.append(tau*d); rhs.append(Ufy(terms[tau*d]))
                
                    # solve
                    info(2, lazy_string(lambda: datetime.today().ctime() + ": calling solver..."))
                    sol = findrelation()
                    info(3, lazy_string(lambda: datetime.today().ctime() + ": solving completed."))
                    if sol is not None:
                        info(2, "telescoper detected.")
                        telescopers.append(sol)
                        B.pop(); rhs.pop(); del terms[tau*d]; iterator.declare_step()
                        if early_termination is True:
                            info(3, "early termination")
                            break

                    if iteration_limit > 0 and len(B) >= iteration_limit:
                        info(1, "iteration limit exceeded")
                        break
            except StopIteration:
                pass

        info(1, "Search completed. Normalizing...")

        for i in range(len(telescopers)):
            content = gcd([p.numerator() for p in telescopers[i][0].coefficients()])
            T = telescopers[i][0].map_coefficients(lambda p : p/content)
            if certificates:
                content = telescopers[i][1].base_ring()(content)
                C = telescopers[i][1].map_coefficients(lambda p: p/content)
            else:
                C = None
            telescopers[i] = (T, C)

        info(1, "done.")
        
        if certificates:
            return [T for T, _ in telescopers], [C for _, C in telescopers]
        else:
            return [T for T, _ in telescopers]
        
            
    creative_telescoping = ct
    
class MonomialIterator(object):
    """
    Iterate in increasing order over the monomials that are below some staircase that is determined in parallel to the iteration.
    Tool for FGLM.
    """

    def __init__(self, algebra):
        self.__gens = algebra.gens()
        self.__pool = [(algebra.one(), algebra.one())]
        self.__prev = None
        self.__stairs = [] 

    def __next__(self):
        """
        Returns (tau, D) such that tau*D is the next term in the iteration, D is an algebra generator, and tau
        is a term that was output earlier. The first output does not follow this rule but is (1, 1). 

        tau*D is the smallest term (in the term order) which has not been listed before and which is not a multiple
        of some term that has been declared as a step in the staircase
        """
        try:
            u, v = self.__pool.pop()
        except:
            raise StopIteration
        self.__prev = next = u*v
        for g in self.__gens:
            self.__pool.append((next, g))
        self.__clear_pool()
        return (u, v)

    def declare_step(self):
        """
        Informs this iterator that among the terms to be outputted in the future there should not be any multiples
        of the previously otuputted term. 
        """
        self.__stairs.append(self.__prev)
        self.__clear_pool()

    def __clear_pool(self):
        self.__pool = [u for u in self.__pool
                       if not (u[0]*u[1]).reduce(self.__stairs).is_zero()]
        # discard stuff above the stairs
        self.__pool.sort(key=lambda u: smallest_lt_first(u[0]*u[1]), reverse=True) # smallest last
        prev = None
        for i in range(len(self.__pool)): # discard double entries
            tau = self.__pool[i]
            tau = tau[0]*tau[1]
            if tau == prev:
                self.__pool[i - 1] = None
            prev = tau
        self.__pool = [tau for tau in self.__pool if tau is not None]

def fglm(algebra, one_vector, gen_matrices, infolevel=0, solver=None, early_termination=False):
    """
    Constructs a Groebner basis using linear algebra.

    INPUT:

      algebra -- target algebra A = K(x,...)[X,...]
      one_vector -- a vector in K(x,...)^n corresponding to the term 1
      gen_matrices -- a dictionary mapping the generators of A to nxn multiplication matrices
      infolevel -- verbosity of progress reports
      solver -- callable to be used to determine bases of right kernels of matrices over K(x,...)
      early_termination -- if set to True, this returns only the first nonzero Groebner basis element encountered

    OUTPUT:

      The one_vector and the gen_matrices together with the sigma's and delta's of A
      turn K(x,...)^n into a left-A-module. The output of this function is the Groebner
      basis of the kernel of the natural homomorphism A --> K(x,...)^n.

    EXAMPLES::

      sage: from ore_algebra import *
      sage: R.<n,k> = ZZ[]
      sage: A.<Sn,Sk> = OreAlgebra(R)
      sage: id1 = A.ideal([Sk-2*k,Sn-3])
      sage: id2 = A.ideal([Sk-3,Sn-n])
      sage: id1.intersection(id2)
      Left Ideal ((-2*k + 3)*Sn + (-n + 3)*Sk + 2*n*k - 9, (-2*k + 3)*Sk^2 + (4*k^2 + 4*k - 9)*Sk - 12*k^2 + 6*k) of Multivariate Ore algebra in Sn, Sk over Fraction Field of Multivariate Polynomial Ring in n, k over Integer Ring
      sage: _.intersection(_)
      Left Ideal ((-2*k + 3)*Sn + (-n + 3)*Sk + 2*n*k - 9, (-2*k + 3)*Sk^2 + (4*k^2 + 4*k - 9)*Sk - 12*k^2 + 6*k) of Multivariate Ore algebra in Sn, Sk over Fraction Field of Multivariate Polynomial Ring in n, k over Integer Ring

      sage: R.<x, y> = ZZ[]
      sage: A.<Dx,Dy> = OreAlgebra(R)
      sage: id1 = A.ideal([2*Dx-Dy, (x+2*y)*Dy^2 + Dy])
      sage: id2 = A.ideal([Dy - x, Dx - y])
      sage: id1.intersection(id2)
      Left Ideal ((-x^2 + 4*y^2)*Dy^2 + (-2*x^3 - 4*x^2*y - 2*x)*Dx + (x^3 + 2*x^2*y + 2*y)*Dy, (-x^2 + 4*y^2)*Dx*Dy + (-2*x^2*y - 4*x*y^2 - 3*x - 4*y)*Dx + (x^2*y + 2*x*y^2 + x + 3*y)*Dy, (-2*x^2 + 8*y^2)*Dx^2 + (-4*x*y^2 - 8*y^3 - x)*Dx + (2*x*y^2 + 4*y^3 + y)*Dy) of Multivariate Ore algebra in Dx, Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring
      sage: _.intersection(_)
      Left Ideal ((-x^2 + 4*y^2)*Dy^2 + (-2*x^3 - 4*x^2*y - 2*x)*Dx + (x^3 + 2*x^2*y + 2*y)*Dy, (-x^2 + 4*y^2)*Dx*Dy + (-2*x^2*y - 4*x*y^2 - 3*x - 4*y)*Dx + (x^2*y + 2*x*y^2 + x + 3*y)*Dy, (-2*x^2 + 8*y^2)*Dx^2 + (-4*x*y^2 - 8*y^3 - x)*Dx + (2*x*y^2 + 4*y^3 + y)*Dy) of Multivariate Ore algebra in Dx, Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring

    """

    if one_vector.is_zero():
        return [algebra.one()]

    def info(i, msg):
        if infolevel >= i:
            print(msg)
    
    basis = []; terms = {}
    sigma = dict( (d, algebra.sigma(d)) for d in algebra.gens() )
    delta = dict( (d, algebra.delta(d)) for d in algebra.gens() )

    iterator = MonomialIterator(algebra)
    terms[next(iterator)[0]] = one_vector ## map terms->vectors

    B = [algebra.one()] ## terms under the stairs
    M = matrix(algebra.base_ring(), len(one_vector), 1, one_vector) ## matrix whose columns are the vectors corresponding to B

    if solver is None:
        solver = algebra._solver()
    
    try:
        while True:
            tau, d = next(iterator) # current term
            info(1, "next monomial: " + str(tau*d))

            # corresponding vector
            v = terms[tau] if sigma[d].is_identity() else terms[tau].apply_map(sigma[d])
            v = gen_matrices[d]*v
            if not delta[d].is_zero():
                v += terms[tau].apply_map(delta[d])
            terms[tau*d] = v
            
            B.append(tau*d); Mold = M; M = M.augment(terms[tau*d])
            ker = solver(M, infolevel=infolevel-3)
            if len(ker) > 0:
                info(2, "relation found.")
                basis.append(add([ker[0][i]*B[i] for i in range(len(B))])) ## new basis element
                B.pop(); M = Mold; iterator.declare_step() ## current term is not under the stairs
                del terms[tau*d] # no longer needed -- hint to garbage collector
                if early_termination:
                    break
    except StopIteration:
        pass

    return basis


def uncouple(mat, algebra=None, extended=False, column_swaps=False, infolevel=0):
    """
    Triangularizes an operator matrix. 

    The matrix is to be specified as lists of lists. The inner lists represent the rows of the matrix.
    The output matrix will be in staircase form. Row operations applied during the transformation act 
    on the matrix from the left.

    EXAMPLES::

      sage: from ore_algebra import *
      sage: from ore_algebra.ideal import uncouple
      sage: R.<x> = ZZ[]
      sage: A.<Dx> = OreAlgebra(R)
      sage: uncouple([[Dx, 3*Dx - 1], [x-Dx, x]])
      [[x, 3*Dx + x - 1], [0, -3*x*Dx^2 + (2*x^2 + x + 3)*Dx - x^2 - 1]]
      sage: uncouple([[Dx-x, 2,x], [3, x, Dx-4], [x*Dx-4, 4-x, 4]])
      [[3, x, Dx - 4],
       [0, x^3 + 5*x - 12, (x^2 - 4)*Dx - x^2 + 4],
       [0,
        0,
        (-3*x^4 + 4*x^3 - 15*x^2 + 56*x - 48)*Dx^2 + (x^6 + x^5 + 9*x^4 - 9*x^3 - 4*x^2 - 88*x + 112)*Dx - x^5 + 4*x^4 - 19*x^3 + 44*x^2 - 88*x + 80]]

    """

    if column_swaps:
        raise NotImplementedError

    if algebra is None:
        A = coercion_model.common_parent(*[elt for row in mat for elt in row])
    else:
        A = algebra
    A_ff = None
    n = len(mat); m = len(mat[0])
    U = [None]*n
    V = [None]*n

    for i, row in enumerate(mat):
        row[:], d = clear_denominators([A.coerce(elt) for elt in row])
        if A_ff is None:
            A_ff = row[0].parent()
            zero = A_ff.zero()
            one = A_ff.one()
        d = A_ff([d])
        if extended:
            U[i] = [d if i == j else zero for j in range(n)]
            V[i] = [(one if i == j else zero) for j in range(m)]

    r = 0 # all rows before this one have been handled. 
    for c in range(m):

        nonzero = [i for i in range(r, n) if not mat[i][c].is_zero()]
        if len(nonzero) == 0:
            continue
        
        while len(nonzero) > 1:

            piv_row = min(nonzero, key=lambda i: (mat[i][c].order(), mat[i][c].degree()))
                
            # move pivot to front
            mat[r], mat[piv_row] = mat[piv_row], mat[r]
            piv = mat[r][c]
            if extended:
                U[r], U[piv_row] = U[piv_row], U[r]    
            
            # perform elimination 
            for i in nonzero:
                if i > r:
                    d, Q, mat[i][c] = mat[i][c].pseudo_quo_rem(mat[r][c])
                    for j in range(c + 1, m):
                        mat[i][j] = d*mat[i][j] - Q*mat[r][j]
                    if extended:
                        for j in range(n):
                            U[i][j] = d*U[i][j] - Q*U[r][j]

            nonzero = [i for i in range(r, n) if not mat[i][c].is_zero()]

        # move pivot to front
        piv_row = nonzero[0]
        mat[r], mat[piv_row] = mat[piv_row], mat[r]
        piv = mat[r][c]
        if extended:
            U[r], U[piv_row] = U[piv_row], U[r]    

        g = gcd([a for op in mat[r] + (U[r] if extended else []) for a in op])
        for j in range(n):
            mat[r][j] = A_ff([a//g for a in mat[r][j]])
            if extended:
                U[r][j] = A_ff([a//g for a in U[r][j]])

        r += 1

    mat = [[A(a) for a in row] for row in mat]

    if extended:
        U = [[A(a) for a in row] for row in U]
        V = [[A(a) for a in row] for row in V]
        if column_swaps:
            return mat, U, V
        else:
            return mat, U
    else:
        return mat

    
def solve_triangular_system(mat, rhs, solver=None):
    """
    Constructs a vector space basis for the uncoupled system mat*f = rhs

    INPUT:

      mat -- an upper triangular matrix of univariate operators, given as a list of list. 
      rhs -- a vector of right hand sides.

    OUTPUT:

      a list of pairs (u, c) where u is a list of rational functions and c is a list of constants such that mat*u == c*rhs
      and every other pair (u, c) with mat*u == c*rhs is a linear combination (with constant coefficients) of the pairs in the
      output

    EXAMPLES::

      sage: from ore_algebra import *
      sage: from ore_algebra.ideal import solve_triangular_system
      sage: R.<x> = ZZ[]; A.<Dx> = OreAlgebra(R); 
      sage: solve_triangular_system([list(map(A, [Dx-x, 2,x])), list(map(A, [0,x,Dx-4])), list(map(A, [0,0,Dx]))], [[1,0,0],[0,1,0],[0,0,1]])
      [([1, 0, 1], [0, -4, 0]), ([x, 4, x], [9, 1, 1])]

    """

    n = len(mat)
    m = len(mat[0])

    for i in range(n):
        for j in range(i):
            assert mat[i][j].is_zero()

    sol = tuple(([0]*len(mat), [1 if i==j else 0 for j in range(len(rhs))]) for i in range(len(rhs))) # init
    for i in range(m - 1, -1, -1):
        xrhs = tuple(sum(s[1][j]*rhs[j][i] for j in range(len(rhs)))-sum(mat[i][j](s[0][j]) for j in range(i+1, m)) for s in sol)
        xsol = list(map(list, mat[i][i].rational_solutions(rhs=xrhs,solver=solver)))
        for k in range(len(xsol)):
            xsol[k][0] = [xsol[k][0]]*(i+1) + [sum(xsol[k][l+1]*sol[l][0][j] for l in range(len(xrhs))) for j in range(i+1, m)]
            xsol[k] = (xsol[k][0], [sum(xsol[k][l+1]*sol[l][1][j] for l in range(len(xrhs))) for j in range(len(rhs))])
        sol = xsol

    return sol

smallest_lt_first = cmp_to_key(
        lambda u,v: 1 if (u.lm()+v.lm()).lm() == u.lm() else -1)

