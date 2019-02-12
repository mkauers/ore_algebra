
"""
Multivariate operators
"""


#############################################################################
#  Copyright (C) 2013, 2014, 2017                                           #
#                Manuel Kauers (mkauers@gmail.com),                         #
#                Maximilian Jaroschek (mjarosch@risc.jku.at),               #
#                Fredrik Johansson (fjohanss@risc.jku.at).                  #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

from __future__ import absolute_import, division, print_function

from datetime import datetime
from functools import reduce

from sage.structure.element import RingElement, canonical_coercion
from sage.structure.richcmp import richcmp
from sage.arith.all import gcd, lcm
from sage.matrix.constructor import Matrix, matrix
from sage.misc.all import prod, add
from sage.misc.lazy_string import lazy_string
from sage.modules.free_module_element import vector
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.infinity import infinity

from .ore_operator import OreOperator

class MultivariateOreOperator(OreOperator):
    """
    An Ore operator. Instances of this class represent elements of Ore algebras with more than
    one generator.
    """

    # constructor

    def __init__(self, parent, data): 
        OreOperator.__init__(self, parent)
        if isinstance(data, OreOperator):
            data = data.polynomial()
        self.__poly = parent.associated_commutative_algebra()(data)

    # action

    def __call__(self, f, **kwds):
        
        A = self.parent()
        gens = A.gens()
        make_der = lambda x, e=1: (lambda u: 0 if u in QQ else u.derivative(x, e))
        for d in gens:
            if str(d) not in kwds:
                if A.is_D(d):
                    kwds[str(d)] = make_der(A.is_D(d))
                else:
                    raise NotImplementedError

        terms = self.__poly.dict()
        out = 0
        for e in terms:
            u = f
            for i in range(len(gens)):
                d = kwds[str(gens[i])]
                if e[i] > 0:
                    try:
                        u = d(u, e[i])
                    except:
                        for j in range(e[i]):
                            u = d(u)
            out += terms[e]*u
            
        return out

    # tests

    def __nonzero__(self):
        return not self.__poly.is_zero()

    def _richcmp_(self, other, op):
        return richcmp(self.__poly, other.__poly, op)

    def _is_atomic(self):
        return self.__poly._is_atomic()
       
    def is_gen(self):
        return self.__poly.is_gen()

    # conversion

    def polynomial(self):
        return self.__poly            

    def change_ring(self, R):
        """
        Return a copy of this operator but with coefficients in R, if at
        all possible.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: op = Dx^2 + 5*x*Dx + 1
          sage: op.parent()
          Univariate Ore algebra in Dx over Univariate Polynomial Ring in x over Rational Field
          sage: op = op.change_ring(R.fraction_field())
          sage: op.parent()
          Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field
        
        """
        if R == self.base_ring():
            return self
        else:
            return self.parent().change_ring(R)(self)

    def _repr(self):
        return self.__poly._repr_()

    def _repr_(self):
        return self._repr()

    def _latex_(self, name=None):
        return self.__poly._latex_()

    def dict(self):
        return self.__poly.dict()

    def list(self):
        return self.__poly.list()

    # arithmetic

    def _mul_(self, other):

        A = self.parent(); n = A.ngens()
        sigma = [ A.sigma(i) for i in range(n) ] 
        delta = [ A.delta(i) for i in range(n) ]
        D = [ A.gen(i).polynomial() for i in range(n) ]
        
        monomial_times_other = {tuple(0 for i in range(n)): other.polynomial()}

        def multiple(exp):
            exp = tuple(int(e) for e in exp)
            if exp not in monomial_times_other:
                i = n - 1
                while exp[i] == 0:
                    i -= 1
                sub = list(exp); sub[i] -= 1; prev = multiple(sub)
                new = prev.map_coefficients(sigma[i])*D[i] + prev.map_coefficients(delta[i])
                monomial_times_other[exp] = new
            return monomial_times_other[exp]

        out = A.zero(); poly = self.__poly
        for exp in poly.dict():
            out += poly[exp]*multiple(exp)

        monomial_times_other.clear() # support garbage collector
        return A(out)

    def _add_(self, other):
        return self.parent()(self.__poly + other.__poly)

    def _neg_(self):
        return self.parent()(self.polynomial()._neg_())

    def quo_rem(self, other):
        raise NotImplementedError

    # coefficient-related functions

    def __getitem__(self, n):
        return self.__poly[n]

    def __setitem__(self, n, value):
        raise IndexError("Operators are immutable")

    def __hash__(self):
        return hash(self.__poly)

    def leading_coefficient(self):
        return self.__poly.leading_coefficient()

    def lc(self):
        """
        Returns the leading coefficient of self

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = QQ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: p = (3*x+y-3)*Dx^3*Dy^2 + Dx - Dy + 1
           sage: p.lc()
           3*x + y - 3
           sage: (0*p).lc()
           0

        """
        return self.__poly.lc()

    def lm(self):
        """
        Returns the leading monomial of self

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = QQ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: p = (3*x+y-3)*Dx^3*Dy^2 + Dx - Dy + 1
           sage: p.lm()
           Dx^3*Dy^2
           sage: (0*p).lm()
           0

        """
        return self.parent()(self.__poly.lm())

    def lt(self):
        """
        Returns the leading term of self

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = QQ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: p = (3*x+y-3)*Dx^3*Dy^2 + Dx - Dy + 1
           sage: p.lt()
           (3*x + y - 3)*Dx^3*Dy^2
           sage: (0*p).lt()
           0

        """
        return self.parent()(self.__poly.lt())

    def exp(self):
        """
        Returns the exponent vector of the leading monomial of self

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x,y> = QQ[]
           sage: A.<Dx,Dy> = OreAlgebra(R)
           sage: p = (3*x+y-3)*Dx^3*Dy^2 + Dx - Dy + 1
           sage: p.exp()
           (3, 2)
           sage: (0*p).exp()
           (-1, -1)

        """
        try:
            return self.__poly.exponents()[0]
        except:
            # zero operator
            return tuple( [-1]*self.parent().ngens() )
    
    def constant_coefficient(self):
        return self.__poly.constant_coefficient()

    def monomial_coefficient(self, term):
        return self.__poly.monomial_coefficient(self.parent()(term).__poly)
    
    def map_coefficients(self, f, new_base_ring = None):
        if new_base_ring is None:
            return self.parent()(self.__poly.map_coefficients(f))
        else:
            return self.parent()(self.__poly.map_coefficients(f, new_base_ring=new_base_ring))

    def coefficients(self):
        return self.__poly.coefficients()

    def exponents(self):
        return self.__poly.exponents()

    def tdeg(self):
        return max(sum(e) for e in self.exponents())

    def degree(self, D):
        return self.__poly.degree(self.parent().associated_commutative_algebra()(str(D)))

    # ==== reduction ====

    def reduce(self, basis, normalize=False, cofactors=False, infolevel=0, coerce=True):
        """
        Compute the remainder of self with respect to the given basis.

        INPUT:

        - basis -- a list of elements of elements of self's parent (or objects
          that can be coerced to such elements), or a left ideal
        - normalize -- 'True' to allow the output to be some K-multiple of the
          actual result, where K is the parent's basering.
        - cofactors -- 'True' to return also the cofactors infolevel --
          nonnegative integer indicating the desired verbosity
        - coerce -- set to 'False' to save some speed if normalize is 'True'
          and you know that 'self' as well as the elements of 'basis' belong to
          the same algebra and this algebra has a multivariate polynomial ring
          as base ring.

        OUTPUT:

           if self is p and basis=[b1,..,bn], this returns an operator r such that
           p - r is in the left ideal generated by [b1,..,bn] and lt(r) is as small as possible
           in the order of self's parent. 

           if normalize is set to True, the output r0 is such that there exists some c in the parent's
           base ring such that r = (1/c)*r0 is as above. 

           if cofactors is set to True, then instead of r the method returns (r0, [p1,...,pn], c) such
           that c*p - r0 = p1*b1 + ... + pn*bn and c is a nonzero element of the parent's base ring 
           (and c=1 if normalize=False) and the leading term of r0 is as small as possible. 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: P.<x,y> = ZZ[]
           sage: A.<Dx,Dy> = OreAlgebra(P)
           sage: p = Dx^2*Dy^1-1; basis = [(x-y)*Dx+y,(x+y)*Dy-2]
           sage: p.reduce(basis)
           (x^4 - 2*x^3*y + 2*x*y^3 - y^4 - 2*x^2*y - 4*x*y^2 + 2*y^3 - x^2 - 4*x*y + y^2)/(-x^4 + 2*x^3*y - 2*x*y^3 + y^4)
           sage: p.reduce(basis, normalize=True)
           1
           sage: u = p.reduce(basis, cofactors=True)
           sage: u[2]*p - u[0] - (u[1][0]*basis[0] + u[1][1]*basis[1])
           0
           sage: u = p.reduce(basis, cofactors=True, normalize=True)
           sage: u[2]*p - u[0] - (u[1][0]*basis[0] + u[1][1]*basis[1])
           0
           sage: A.<Sx,Sy> = OreAlgebra(ZZ[x,y])
           sage: p = Sx^2*Sy^1-1; basis = [(x-y)*Sx+y,(x+y)*Sy-2]
           sage: p.reduce(basis)
           (-x^3 + x^2*y + x*y^2 - y^3 + x^2 + y^2 + 4*y + 2)/(x^3 - x^2*y - x*y^2 + y^3 - x^2 + y^2)
           sage: p.reduce(basis, normalize=True)
           1
           sage: u = p.reduce(basis, cofactors=True)
           sage: u[2]*p - u[0] - (u[1][0]*basis[0] + u[1][1]*basis[1])
           0
           sage: u = p.reduce(basis, cofactors=True, normalize=True)
           sage: u[2]*p - u[0] - (u[1][0]*basis[0] + u[1][1]*basis[1])
           0
        
        """

        # ~~~ naive code ~~~
        
        def info(i, msg):
            if infolevel >= i:
                print(msg)

        try:
            # handle case where input is an ideal 
            return self.reduce(basis.groebner_basis(), normalize=normalize, coerce=coerce, cofactors=cofactors, infolevel=infolevel)
        except AttributeError:
            pass

        # assuming basis is a list of operators
                
        if normalize and coerce:

            if self.base_ring().is_field():
                info(1, "switch to polynomial base ring")
                A = self.parent() ## K(x,y)[Dx,Dy]
                R = A.base_ring() ## K(x,y)
                B = R.ring() ## K[x,y]
                A = A.change_ring(B) ## K[x,y][Dx, Dy]
                c = self.denominator()
                out = list(A(c*self).reduce(basis, normalize=True, cofactors=cofactors, infolevel=infolevel))
                out[2] *= c
                return tuple(out)

            d = reduce(lambda u, v: u.lcm(v), [b.denominator() for b in basis], self.denominator())
            if not d.is_one():
                info(1, "clearing denominator of basis elements")
                out = list(self.reduce([self.parent()(d*b) for b in basis], normalize=True, cofactors=cofactors, infolevel=infolevel))
                out[1] = [o*d for o in out[1]]
                return tuple(out)
            
            basis = list(map(self.parent(), basis))

        exp = [vector(ZZ, b.exp()) for b in basis]
        gens = self.parent().gens(); p = self; r0 = self.parent().zero(); c = self.base_ring().one()
        cofs = [self.parent().zero()]*len(basis)
        range_basis = [k for k in range(len(basis)) if not basis[k].is_zero()]

        # keep track of sugar if sugar is associated to self and to all the basis elements
        basis_sugar = [0]*len(basis)
        try:
            for i in range(len(basis)):
                basis_sugar[i] = basis[i].sugar
            sugar = self.sugar
        except AttributeError:
            sugar = None

        while not p.is_zero():

            info(1, lazy_string(lambda: datetime.today().ctime() + ": " + str(len(p.coefficients())) + " terms left; continuing with " + str(p.lm())))
            
            e = vector(ZZ, p.exp())
            candidates = list(filter(lambda i: min(e - exp[i]) >= 0, range_basis))

            if len(candidates) == 0:
                info(2, "term goes to remainder")
                r0 += p.lt()
                p -= p.lt()
            else:
                k = candidates[0] ## care for a more clever choice?
                b = basis[k]; tau = prod(x**i for x, i in zip(gens, e - exp[k]))
                info(2, str(len(candidates)) + " basis elements apply, taking no " + str(k) + " with leading monomial " + str(b.lm()))
                b0 = tau*b; b0lc = b0.lc();
                if sugar is not None:
                    sugar = max(sugar, tau.tdeg() + basis_sugar[k])
                if normalize:
                    c *= b0lc; r0 = b0lc*r0
                    if cofactors:
                        for i in range(len(cofs)):
                            cofs[i] = b0lc*cofs[i]
                        cofs[k] += p.lc()*tau
                    p = b0lc*p - p.lc()*b0
                    ## clear content
                    gcd = p.base_ring().zero()
                    for u in [p, r0] + cofs:
                        for uu in u.coefficients():
                            gcd = gcd.gcd(uu)
                    if not gcd.is_one() and not gcd.is_zero():
                        c /= gcd
                        p = p.map_coefficients(lambda u: u//gcd)
                        r0 = r0.map_coefficients(lambda u: u//gcd)
                        for i in range(len(cofs)):
                            cofs[i] = cofs[i].map_coefficients(lambda u: u//gcd)
                else:
                    if cofactors:
                        cofs[k] += (p.lc()/b0lc)*tau
                    p -= (p.lc()/b0lc)*b0

        if normalize and not r0.is_zero() and r0.lc().parent().base_ring() is ZZ:
            ## make leading term of leading coefficient of r0 positive
            try:
                sgn = r0.lc().lc().sign()
            except:
                try:
                    sgn = r0.lc().coefficients(sparse=True)[-1].sign()
                except:
                    sgn = ZZ.one()
            r0 *= sgn
            c *= sgn
            if cofactors:
                for i in range(len(cofs)):
                    cofs[i] *= sgn

        if sugar is not None:
            r0.sugar = sugar
            
        info(1, "reduction completed, remainder has " + str(len(r0.coefficients())) + " terms.")
        return (r0, cofs, c) if cofactors else r0
