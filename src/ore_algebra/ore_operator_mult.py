
"""
ore_operator_mult
=================

"""


#############################################################################
#  Copyright (C) 2013, 2014                                                 #
#                Manuel Kauers (mkauers@gmail.com),                         #
#                Maximilian Jaroschek (mjarosch@risc.jku.at),               #
#                Fredrik Johansson (fjohanss@risc.jku.at).                  #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

from sage.structure.element import RingElement, canonical_coercion
from sage.rings.arith import gcd, lcm
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.infinity import infinity

from ore_operator import *

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
        raise NotImplementedError

    # tests

    def __nonzero__(self):
        return not self.__poly.is_zero()

    def __cmp__(self, other):
        return cmp(self.__poly, other.__poly)

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
        sigma = [ A.sigma(i) for i in xrange(n) ] 
        delta = [ A.delta(i) for i in xrange(n) ]
        D = [ A.gen(i).polynomial() for i in xrange(n) ]
        
        monomial_times_other = {tuple(0 for i in xrange(n)): other.polynomial()}

        def multiple(exp):
            exp = tuple(int(e) for e in exp)
            if not monomial_times_other.has_key(exp):
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
        raise IndexError, "Operators are immutable"

    def leading_coefficient(self):
        return self.__poly.leading_coefficient()

    def lc(self):
        return self.__poly.lc()

    def constant_coefficient(self):
        return self.__poly.constant_coefficient()

    def map_coefficients(self, f, new_base_ring = None):
        if new_base_ring is None:
            return self.parent()(self.__poly.map_coefficients(f))
        else:
            return self.parent()(self.__poly.map_coefficients(f, new_base_ring=new_base_ring))

    def coefficients(self):
        return self.__poly.coefficients()

    def exponents(self):
        return self.__poly.exponents()
