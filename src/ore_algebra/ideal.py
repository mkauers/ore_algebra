
"""
ideal
=====

"""

#############################################################################
#  Copyright (C) 2015, 2016                                                 #
#                Manuel Kauers (mkauers@gmail.com),                         #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

from datetime import datetime
from sage.rings.noncommutative_ideals import Ideal_nc
from sage.misc.all import prod, add
from sage.rings.rational_field import QQ
from sage.modules.free_module_element import vector
from sage.matrix.constructor import Matrix, matrix

from . import nullspace

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
        assert self.dimension() <= 0
        
        lms = self._lm_poly_ideal()
        basis = [lms.ring().one()]

        for g in lms.ring().gens():
            newbasis = []
            for i in range(len(basis)):
                newbasis.append(basis[i])
                while (g*newbasis[-1]).reduce(lms) == g*newbasis[-1]:
                    newbasis.append(g*newbasis[-1])
            basis = newbasis

        basis.sort(cmp=lambda u,v: 1 if (u + v).lm() == u else -1)
        return map(self.ring(), basis)

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
           [                                0                                -1  (4*n + 4*k - 1)/(-2*n - 2*k + 3)]
           [                                0                                 1             (-1)/(-2*n - 2*k + 3)]
           [                                1                                 1 (-6*n - 6*k + 7)/(-2*n - 2*k + 3)]
           sage: id.multiplication_matrix(Sk)
           [                               0 (-6*n - 6*k - 1)/(2*n + 2*k - 3)                               -1]
           [                               1 (8*n + 8*k - 10)/(2*n + 2*k - 3)                                1]
           [                               0                4/(2*n + 2*k - 3)                                1]

        """
        D = self.ring().gen(self.ring()._gen_to_idx(idx))
        B = self.vector_space_basis()
        G = self.groebner_basis()

        mat = [(D*b).reduce(G) for b in B]
        mat = [[m[b.exp()] for b in B] for m in mat]
        return matrix(self.ring().base_ring(), mat).transpose()
        
    def groebner_basis(self, infolevel=0, update_hook=None):
        """
        Returns the Groebner basis of this ideal. 

        INPUT:
        
           infolevel -- integer indicating the verbosity of progress reports
           update_hook -- a function which is envoked on (G, C, h) right before a new nonzero polynomial h is integrated 
                        into the basis G (a list of operators). The list C contains the critical pairs, each pair
                        is encoded as a tuple (lcm(lm(a),lm(b)), a, b, s), where a and b are operators and 
                        s is the sugar associated to the S-polynomial of a and b. The hook facility gives a 
                        possibility to interfere with the computation. Fiddling with the lists G and C may 
                        destroy correctness or termination.

        OUTPUT:

           The Groebner basis of self. The output is cached.
           The monomial order is taken from the ambient ring (the common parent of the generators of self).
           Conceptually, the base ring of the ambient ring is treated as a field, even if it is in fact only
           a polynomial ring. 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: A.ideal([Dy^5*(Dx + (-1 + 2*x - 2*y)*Dy - 1), (-2 + 2*x - 2*y)*Dy^2 - 3*Dy]).groebner_basis()
           [(2*x - 2*y - 2)*Dy^2 + (-3)*Dy, (2*x - 2*y - 2)*Dx*Dy + 3*Dy]
           sage: A.ideal([Dy^3*(Dx + (-1 + 2*x - 2*y)*Dy - 1), Dx^2*((-2 + 2*x - 2*y)*Dy^2 - 3*Dy)]).groebner_basis()
           [(2*x - 2*y - 2)*Dx*Dy^3 + 7*Dy^3,
            (2*x - 2*y - 2)*Dy^4 + (-7)*Dy^3,
            (-3)*Dx^2*Dy + 4*Dx*Dy^2 + 7*Dy^3]

           sage: R.<n,k> = ZZ[]
           sage: A.<Sn,Sk> = OreAlgebra(R)
           sage: A.ideal([-5+2*k-n + Sk + (3-2*k+n)*Sn, -2+4*k-2*n + (7-6*k+3*n)*Sk + (-3+2*k-n)*Sk^2]).groebner_basis()
           [(n - 2*k + 3)*Sn + Sk - n + 2*k - 5,
            (-n + 2*k - 3)*Sk^2 + (3*n - 6*k + 7)*Sk - 2*n + 4*k - 2]

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
        gens = map(A, gens)
        
        # ~~~ relatively naive code ~~~

        # tools
        def info(i, msg):
            if infolevel >= i:
                print msg

        X = map(A, self.ring().gens())
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
            C = [g for g in G]; C.sort(cmp=lambda u,v: -1 if (u.lm() + v.lm()).lm() == u.lm() else 1) # smallest leading term last
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
        G.sort(cmp=lambda u,v: 1 if (u.lm()+v.lm()).lm() == u.lm() else -1) # smallest leading terms first
            
        # done
        info(1, "completion completed, Groebner basis has " + str(len(G)) + " elements.")
        self.__gb = tuple(G)
        return G

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
           sage:

        """
        if self.dimension() + other.dimension() != 0:
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
            

        return A.ideal(fglm(A, vector(A.base_ring(), [1] + [0]*(n-1) + [1] + [0]*(m - 1)), \
                            gen_matrices, infolevel=infolevel, solver=solver), is_known_to_be_a_groebner_basis=True)
            
    
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

    def next(self):
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
        self.__pool = filter(lambda u: not (u[0]*u[1]).reduce(self.__stairs).is_zero(), self.__pool) # discard stuff above the stairs
        self.__pool.sort(cmp=lambda u, v: -1 if (u[0]*u[1] + v[0]*v[1]).lm() == u[0]*u[1] else 1) # smallest last
        prev = None
        for i in range(len(self.__pool)): # discard double entries
            tau = self.__pool[i]
            tau = tau[0]*tau[1]
            if tau == prev:
                self.__pool[i - 1] = None
            prev = tau
        self.__pool = [tau for tau in self.__pool if tau is not None]

def fglm(algebra, one_vector, gen_matrices, infolevel=0, solver=None):
    """
    Constructs a Groebner basis using linear algebra.

    INPUT:

      algebra -- target algebra A = K(x,...)[X,...]
      one_vector -- a vector in K(x,...)^n corresponding to the term 1
      gen_matrices -- a dictionary mapping the generators of A to multiplication matrices
      solver -- callable to be used to determine bases of right kernels of matrices over K(x,...)

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
      Left Ideal ((x^2 - 4*y^2)*Dy^2 + (2*x^3 + 4*x^2*y + 2*x)*Dx + (-x^3 - 2*x^2*y - 2*y)*Dy, (x^2 - 4*y^2)*Dx*Dy + (2*x^2*y + 4*x*y^2 + 3*x + 4*y)*Dx + (-x^2*y - 2*x*y^2 - x - 3*y)*Dy, (2*x^2 - 8*y^2)*Dx^2 + (4*x*y^2 + 8*y^3 + x)*Dx + (-2*x*y^2 - 4*y^3 - y)*Dy) of Multivariate Ore algebra in Dx, Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring
      sage: _.intersection(_)
      Left Ideal ((x^2 - 4*y^2)*Dy^2 + (2*x^3 + 4*x^2*y + 2*x)*Dx + (-x^3 - 2*x^2*y - 2*y)*Dy, (x^2 - 4*y^2)*Dx*Dy + (2*x^2*y + 4*x*y^2 + 3*x + 4*y)*Dx + (-x^2*y - 2*x*y^2 - x - 3*y)*Dy, (2*x^2 - 8*y^2)*Dx^2 + (4*x*y^2 + 8*y^3 + x)*Dx + (-2*x*y^2 - 4*y^3 - y)*Dy) of Multivariate Ore algebra in Dx, Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring

    """

    if one_vector.is_zero():
        return [algebra.one()]

    def info(i, msg):
        if infolevel >= i:
            print msg
    
    basis = []; terms = {}
    sigma = dict( (d, algebra.sigma(d)) for d in algebra.gens() )
    delta = dict( (d, algebra.delta(d)) for d in algebra.gens() )

    iterator = MonomialIterator(algebra)
    terms[iterator.next()[0]] = one_vector ## map terms->vectors

    B = [algebra.one()] ## terms under the stairs
    M = matrix(algebra.base_ring(), len(one_vector), 1, one_vector) ## matrix whose columns are the vectors corresponding to B

    if solver is None:
        solver = algebra._solver()
    
    try:
        while True:
            tau, d = iterator.next() # current term
            info(1, "next monomial: " + str(tau*d))
            terms[tau*d] = gen_matrices[d]*terms[tau].apply_map(sigma[d]) + terms[tau].apply_map(delta[d]) # corresponding vector
            B.append(tau*d); Mold = M; M = M.augment(terms[tau*d])
            ker = solver(M) 
            if len(ker) > 0:
                info(2, "relation found.")
                basis.append(add([ker[0][i]*B[i] for i in range(len(B))])) ## new basis element
                B.pop(); M = Mold; iterator.declare_step() ## current term is not under the stairs
                del terms[tau*d] # no longer needed -- hint to garbage collector
    except StopIteration:
        pass

    return basis
