
"""
Operators
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

from __future__ import absolute_import

from functools import reduce

from sage.structure.element import RingElement, canonical_coercion
from sage.structure.richcmp import richcmp
from sage.arith.all import gcd, lcm
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.infinity import infinity
from sage.functions.generalized import sign

class OreOperator(RingElement):
    """
    An Ore operator. This is an abstract class whose instances represent elements of ``OreAlgebra``.

    In addition to usual ``RingElement`` features, Ore operators provide coefficient extraction
    functionality and the possibility of letting an operator act on another object. The latter
    is provided through ``__call__``.

    """

    # constructor

    def __init__(self, parent, is_gen = False, construct=False):
        RingElement.__init__(self, parent)
        self._is_gen = is_gen

    def __copy__(self):
        """
        Return a "copy" of self. This is just self, since in Sage
        operators are immutable this just returns self again.
        """
        return self

    # action

    def __call__(self, f, **kwds):
        """
        Lets ``self`` act on ``f`` and returns the result.
        The meaning of the action corresponding to the generator
        of the Ore algebra can be specified with a keyword arguments
        whose left hand sides are the names of the generator and the
        right hand side some callable object. If no such information
        is provided for some generator, a default function is used.
        The choice of the default depends on the subclass. 

        The parent of ``f`` must be a ring supporting conversion
        from the base ring of ``self``. (There is room for generalization.)

        EXAMPLES::

           # In differential operator algebras, generators acts as derivations
           sage: from ore_algebra import *
           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R.fraction_field(), "Dx")
           sage: (Dx^5)(x^5) # acting on base ring elements
           120
           sage: (x*Dx - 1)(x)
           0
           sage: RR = PowerSeriesRing(QQ, "x", 5)
           sage: 1/(1-RR.gen())
           1 + x + x^2 + x^3 + x^4 + O(x^5)
           sage: (Dx^2 - (5*x-3)*Dx - 1)(_) # acting on something else
           4 + 6*x + 10*x^2 + O(x^3)

           # In shift operator algebras, generators act per default as shifts
           sage: R.<x> = QQ['x']
           sage: A.<Sx> = OreAlgebra(R.fraction_field(), "Sx")
           sage: (Sx - 1)(x)
           1
           sage: (Sx - 1)(1/4*x*(x-1)*(x-2)*(x-3))
           x^3 - 3*x^2 + 2*x
           sage: factor(_)
           (x - 2) * (x - 1) * x
           sage: (Sx - 1)(1/4*x*(x-1)*(x-2)*(x-3), action=lambda p:p(2*x)) # let Sx act as q-shift
           15/4*x^4 - 21/2*x^3 + 33/4*x^2 - 3/2*x

        """
        raise NotImplementedError

    # tests

    def __nonzero__(self):
        raise NotImplementedError

    def _is_atomic(self):
        raise NotImplementedError

    def is_monic(self):
        """
        Returns True if this polynomial is monic. The zero operator is by definition not monic.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: (Dx^3 + (5*x+3)*Dx + (71*x+1)).is_monic()
          True
          sage: ((5*x+3)*Dx^2 + (71*x+1)).is_monic()
          False 
        
        """
        if self.is_zero():
            return False
        else:
            return self.leading_coefficient().is_one()

    def is_unit(self):
        """
        Return True if this operator is a unit.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: A(x).is_unit()
          False
          sage: A.<Dx> = OreAlgebra(R.fraction_field(), 'Dx')
          sage: A(x).is_unit()
          True
          
        """
        if len(self.exponents()) > 1:
            return False
        else:
            return self.constant_coefficient().is_unit()
       
    def is_gen(self):
        """
        Return True if this operator is one of the generators of the parent Ore algebra. 
                
        Important - this function doesn't return True if self equals the
        generator; it returns True if self *is* the generator.
        """
        raise NotImplementedError

    def prec(self):
        """
        Return the precision of this operator. This is always infinity,
        since operators are of infinite precision by definition (there is
        no big-oh).
        """
        return infinity

    # conversion
        
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

    def __iter__(self):
        return iter(self.list())

    def _repr(self, name=None):
        raise NotImplementedError

    def _repr_(self):
        return self._repr()

    def _latex_(self, name=None):
        raise NotImplementedError
        
    def _sage_input_(self, sib, coerced):
        raise NotImplementedError

    def dict(self):
        """
        Return a sparse dictionary representation of this operator.
        """
        raise NotImplementedError

    def list(self):
        """
        Return a new copy of the list of the underlying elements of self.
        """
        raise NotImplementedError

    # arithmetic

    def __floordiv__(self,right):
        """
        Quotient of quotient with remainder.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: U = (15*x^2 + 29*x + 5)*Dx^2 + (5*x^2 - 50*x - 41)*Dx - 2*x + 64
           sage: V = (3*x+5)*Dx + (x-9)
           sage: U//V
           ((5*x^2 + 29/3*x + 5/3)/(x + 5/3))*Dx + (-64/9*x^2 - 68/3*x - 175/9)/(x^2 + 10/3*x + 25/9)
        
        """
        Q, _ = self.quo_rem(right)
        return Q
        
    def __mod__(self, other):
        """
        Remainder of quotient with remainder.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: U = (15*x^2 + 29*x + 5)*Dx^2 + (5*x^2 - 50*x - 41)*Dx - 2*x + 64
           sage: V = (3*x+5)*Dx + (x-9)
           sage: U % V
           (1/9*x^3 - 2*x^2 + 49/9*x)/(x^2 + 10/3*x + 25/9)
        
        """
        _, R = self.quo_rem(other)
        return R

    def quo_rem(self, other):
        """
        Quotient and remainder.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R.fraction_field(), 'Dx')
          sage: U = (15*x^2 + 29*x + 5)*Dx^2 + (5*x^2 - 50*x - 41)*Dx - 2*x + 64
          sage: V = (3*x+5)*Dx + (x-9)
          sage: Q, R = U.quo_rem(V)
          sage: Q*V + R == U
          True 
        
        """
        raise NotImplementedError

    # base ring related functions
        
    def base_ring(self):
        """
        Return the base ring of the parent of self.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: OreAlgebra(QQ['x'], 'Dx').random_element().base_ring()
           Univariate Polynomial Ring in x over Rational Field
        
        """
        return self.parent().base_ring()

    def base_extend(self, R):
        """
        Return a copy of this operator but with coefficients in R, if
        there is a natural map from coefficient ring of self to R.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: L = OreAlgebra(QQ['x'], 'Dx').random_element()
           sage: L = L.base_extend(QQ['x'].fraction_field())
           sage: L.parent()
           Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field

        """
        return self.parent().base_extend(R)(self)

    # coefficient-related functions

    def __getitem__(self, n):
        raise NotImplementedError

    def __setitem__(self, n, value):
        raise IndexError("Operators are immutable")

    def is_primitive(self, n=None, n_prime_divs=None):
        """
        Returns ``True`` if this operator's content is a unit of the base ring. 
        """
        return self.content().is_unit()

    def is_monomial(self):
        """
        Returns True if self is a monomial, i.e., a power product of the generators. 
        """
        return len(self.exponents()) == 1 and self.leading_coefficient() == self.parent().base_ring().one()

    def leading_coefficient(self):
        """
        Return the leading coefficient of this operator. 
        """
        raise NotImplementedError

    def constant_coefficient(self):
        r"""
        Return the coefficient of `\partial^0` of this operator. 
        """
        raise NotImplementedError

    def monic(self):
        """
        Return this operator divided from the left by its leading coefficient.
        Does not change this operator. If the leading coefficient does not have
        a multiplicative inverse in the base ring of ``self``'s parent, the
        the method returns an element of a suitably extended algebra.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: (x*Dx + 1).monic()
          Dx + 1/x
          sage: _.parent()
          Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field
        
        """
        if self.is_zero():
            raise ZeroDivisionError
        elif self.is_monic():
            return self
        R = self.base_ring().fraction_field()
        a = ~R(self.leading_coefficient())
        A = self.parent()
        if R != A.base_ring():
            S = A.base_extend(R)
            return a*S(self)
        else:
            return a*self

    def content(self, proof=True):
        """
        Returns the content of ``self``.

        If the base ring of ``self``'s parent is a field, the method returns the leading coefficient.

        If the base ring is not a field, then it is a polynomial ring. In this case,
        the method returns the greatest common divisor of the nonzero coefficients of
        ``self``. If the base ring does not know how to compute gcds, the method returns `1`.

        If ``proof`` is set to ``False``, the gcd of two random linear combinations of
        the coefficients is taken instead of the gcd of all the coefficients. 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (5*x^2*Dx + 10*x).content()
           5*x
           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (5*x^2*Dx + 10*x).content()
           x
           sage: (5*x^2*Dx + 10*x).content(proof=False)
           x
           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R.fraction_field(), 'Dx')
           sage: (5*x^2*Dx + 10*x).content()
           5*x^2
        
        """
        R = self.base_ring()
        if self.is_zero():
            return R.one()
        elif R.is_field():
            return self.leading_coefficient()
        else:

            coeffs = self.coefficients() # nonzero coefficients only
            if len(coeffs) == 1:
                return coeffs[0]
            
            try:
                a = sum(R(29*i+13)*coeffs[i] for i in range(len(coeffs)))
                b = sum(R(31*i+17)*coeffs[i] for i in range(len(coeffs)))
                try:
                    c = a.gcd(b)
                except:
                    c = R.zero()
                if not proof and not c.is_zero() and \
                   sum(len(p.coefficients()) for p in coeffs) > 1000: # no shortcut for small operators
                    return c

                coeffs.append(c)
                if R.ngens() == 1:
                    # move polynomials of lower degree to front
                    coeffs.sort(key=lambda p: p.degree())
                else:
                    # move polynomials with fewer terms to front
                    coeffs.sort(key=lambda p: len(p.exponents()))

                return gcd(coeffs)
            except:
                return R.one()

    def primitive_part(self, proof=True):
        """
        Returns the primitive part of ``self``.

        It is obtained by dividing ``self`` from the left by ``self.content()``.

        The ``proof`` option is passed on to the content computation. 

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: (5*x^2*Dx + 10*x).primitive_part()
          x*Dx + 2
          sage: A.<Dx> = OreAlgebra(R.fraction_field(), 'Dx')
          sage: (5*x^2*Dx + 10*x).primitive_part()
          Dx + 2/x
        
        """
        c = self.content(proof=proof)
        if c.is_one():
            return self
        elif self.base_ring().is_field():
            return self.map_coefficients(lambda p: p/c)
        else:
            return self.map_coefficients(lambda p: p//c)

    def normalize(self, proof=True):
        """
        Returns a normal form of ``self``.

        Call two operators `A,B` equivalent iff there exist nonzero elements `p,q` of the base ring
        such that `p*A=q*B`. Then `A` and `B` are equivalent iff their normal forms as computed by
        this method agree.

        The normal form is a left multiple of ``self`` by an element of (the fraction field of) the
        base ring. An attempt is made in choosing a "simple" representative of the equivalence class.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: (10*(x+1)*Dx - 5*x).normalize()
          (x + 1)*Dx - 1/2*x
        
        """
        if self.is_zero():
            return self
        num = self.numerator().primitive_part(proof=proof)
        c = num.leading_coefficient()
        while not c.is_unit() and c.parent() is not c.parent().base_ring():
            try:
                c = c.leading_coefficient()
            except:
                try:
                    c = c.lc()
                except:
                    break
        while c.parent() is not c.parent().base_ring():
            try:
                c = c.parent().base_ring()(c)
            except:
                break
        if not c.is_unit(): 
            try:
                c = sign(c)
            except:
                c = c.parent().one()
        return self.parent()((~c)*num)

    def map_coefficients(self, f, new_base_ring = None):
        """
        Returns the operator obtained by applying ``f`` to the non-zero
        coefficients of self.
        """
        raise NotImplementedError

    def coefficients(self):
        """
        Return the coefficients of the monomials appearing in self.
        """
        raise NotImplementedError

    def exponents(self):
        """
        Return the exponents of the monomials appearing in self.
        """
        raise NotImplementedError
             
    # numerator and denominator

    def numerator(self):
        r"""
        Return a numerator of ``self``.

        If the base ring of ``self``'s parent is not a field, this returns
        ``self``.

        If the base ring is a field, then it is the fraction field of a
        polynomial ring. In this case, the method returns
        ``self.denominator()*self`` and tries to cast the result into the Ore
        algebra whose base ring is just the polynomial ring. If this fails (for
        example, because some `\sigma` maps a polynomial to a rational
        function), the result will be returned as element of the original
        algebra.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']
          sage: A.<Dx> = OreAlgebra(R.fraction_field(), 'Dx')
          sage: op = (5*x+3)/(3*x+5)*Dx + (7*x+1)/(2*x+5)
          sage: op.numerator()
          (10*x^2 + 31*x + 15)*Dx + 21*x^2 + 38*x + 5
          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R.fraction_field(), 'Dx')
          sage: op = (5*x+3)/(3*x+5)*Dx + (7*x+1)/(2*x+5)
          sage: op.numerator()
          (5/3*x^2 + 31/6*x + 5/2)*Dx + 7/2*x^2 + 19/3*x + 5/6          

        """
        A = self.parent(); R = A.base_ring()

        if not R.is_field():
            return self

        op = self.denominator()*self;

        try:
            op = A.change_ring(R.ring())(op)
        except:
            pass

        return op

    def denominator(self):
        """
        Return a denominator of self.

        If the base ring of the algebra of ``self`` is not a field, this returns the one element
        of the base ring.

        If the base ring is a field, then it is the fraction field of a
        polynomial ring. In this case, the method returns the least common multiple
        of the denominators of all the coefficients of ``self``.
        It is an element of the polynomial ring. 

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']
          sage: A.<Dx> = OreAlgebra(R.fraction_field(), 'Dx')
          sage: op = (5*x+3)/(3*x+5)*Dx + (7*x+1)/(2*x+5)
          sage: op.denominator()
          6*x^2 + 25*x + 25
          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R.fraction_field(), 'Dx')
          sage: op = (5*x+3)/(3*x+5)*Dx + (7*x+1)/(2*x+5)
          sage: op.denominator()
          x^2 + 25/6*x + 25/6
          
        """
        A = self.parent(); R = A.base_ring()

        if not R.is_field():
            return R.one()
        else:
            return R(lcm([R(c.denominator()) for c in self.coefficients()]))


#############################################################################################################
    
class UnivariateOreOperator(OreOperator):
    """
    Element of an Ore algebra with a single generator and a commutative field as base ring.     
    """

    def __init__(self, parent, *data, **kwargs):
        super(OreOperator, self).__init__(parent)
        if len(data) == 1 and isinstance(data[0], OreOperator):
            # CASE 1:  *data is an OreOperator, possibly from a different algebra
            self._poly = parent.associated_commutative_algebra()(data[0].polynomial(), **kwargs)
        else:
            # CASE 2:  *data can be coerced to a commutative polynomial         
            self._poly = parent.associated_commutative_algebra()(*data, **kwargs)

    # action

    def __call__(self, f, **kwds):

        if "action" in kwds:
            D = kwds["action"]
        else:
            D = lambda p:p

        if self.is_zero():
            return self.base_ring().zero()*f

        try:
            f, _ = canonical_coercion(f, self.base_ring().zero())
        except:
            pass
        R = f.parent()
        coeffs = self.coefficients(sparse=False)
        Dif = f; result = R(coeffs[0])*f; 
        for i in range(1, self.order() + 1):
            Dif = D(Dif)
            result += R(coeffs[i])*Dif
        
        return result

    # tests

    def __nonzero__(self):
        return bool(self._poly).__nonzero__()

    def _richcmp_(self, other, op):
        return richcmp(self.polynomial(), other.polynomial(), op)

    def _is_atomic(self):
        return self._poly._is_atomic()

    def is_monic(self):
        return self._poly.is_monic()

    def is_unit(self):
        return self._poly.is_unit()
       
    def is_gen(self):
        return self._poly.is_gen()

    def __hash__(self):
        return hash(self._poly)
    
    is_monic.__doc__ = OreOperator.is_monic.__doc__
    is_unit.__doc__ = OreOperator.is_unit.__doc__
    is_gen.__doc__ = OreOperator.is_gen.__doc__
    
    # conversion
    
    def __iter__(self):
        return iter(self.list())

    def __float__(self):
        return self._poly.__float__()

    def __int__(self):
        return self._poly.__int__()

    def _integer_(self, ZZ):
        return self._poly._integer_(ZZ)

    def _rational_(self):
        return self._poly._rational_()

    def _symbolic_(self, R):
        return self._poly._symbolic_(R)

    def __long__(self):
        return self._poly.__long__()

    def _repr(self, name=None):
        return self._poly._repr(name=name)

    def _latex_(self, name=None):
        return self._poly._latex_(name=name)
        
    def _sage_input_(self, sib, coerced):
        
        if self.order() > 0:
            gen = sib.gen(self.parent())
            coeffs = self.list()
            terms = []
            for i in range(len(coeffs)-1, -1, -1):
                if i > 0:
                    if i > 1:
                        gen_pow = gen**sib.int(i)
                    else:
                        gen_pow = gen
                    terms.append(sib.prod((sib(coeffs[i], True), gen_pow), simplify=True))
                else:
                    terms.append(sib(coeffs[i], True))
            return sib.sum(terms, simplify=True)
        elif coerced:
            return sib(self[0], True)
        else:
            return sib(self.parent())(sib(self[0], True))

    def dict(self):
        return self._poly.dict()

    def list(self):
        return self._poly.list()

    def polynomial(self):
        return self._poly

    # arithmetic

    def _add_(self, right):
        return self.parent()(self.polynomial() + right.polynomial())
    
    def _neg_(self):
        return self.parent()(self.polynomial()._neg_())

    def _mul_(self, right):

        if self.is_zero(): return self
        if right.is_zero(): return right

        R = self.parent() # Ore algebra
        sigma = R.sigma(); delta = R.delta()
        D = R.associated_commutative_algebra().gen()

        if sigma.is_identity():
            def times_D(b):
                return b*D + b.map_coefficients(delta)
        elif delta.is_zero():
            def times_D(b):
                return b.map_coefficients(sigma)*D
        else:
            def times_D(b):
                return b.map_coefficients(sigma)*D + b.map_coefficients(delta)

        DiB = right.polynomial() # D^i * B, for i=0,1,2,...
        res = self[0]*DiB
        for i in range(1, self.order() + 1):
            DiB = times_D(DiB)
            res += self[i]*DiB

        return R(res)

    def _rmul_(self, left):
        return self.parent()([left*c for c in self])

    def reduce(self, basis, normalize=False, cofactors=False, infolevel=0, coerce=True):
        ## compatibility method for multivariate case

        if cofactors:
            raise NotImplementedError

        try:
            # handle case where input is an ideal 
            return self.reduce(basis.groebner_basis(), normalize=normalize, coerce=coerce, cofactors=cofactors, infolevel=infolevel)
        except AttributeError:
            pass
        
        p = self
        for b in basis:
            p = p % b

        return p.normalize() if normalize else p
    
    def quo_rem(self, other, fractionFree=False):

        if other.is_zero(): 
            raise ZeroDivisionError("other must be nonzero")

        if (self.order() < other.order()):
            return (self.parent().zero(),self)

        p=self
        q=other
        R = self.parent()
        if fractionFree==False and not R.base_ring().is_field():
            R = R.change_ring(R.base_ring().fraction_field())
            p=R(p)
            q=R(q)
        sigma = R.sigma()
        D = R.gen()
        orddiff = p.order() - q.order()
        cfquo = R.one()
        quo = R.zero()

        qlcs = [q.leading_coefficient()]
        for i in range(orddiff): qlcs.append(sigma(qlcs[-1]))

        if fractionFree: op = lambda x,y:x//y
        else: op = lambda x,y:x/y
        while(orddiff >= 0):
            currentOrder=p.order()
            cfquo = op(p.leading_coefficient(),qlcs[orddiff]) * D**(orddiff)
            quo = quo+cfquo
            p = p - cfquo*q
            if p.order()==currentOrder:
                p = self
                q = other
                op = lambda x,y:x/y
            orddiff = p.order() - q.order()
        return (quo,p)

    quo_rem.__doc__ = OreOperator.quo_rem.__doc__

    def pseudo_quo_rem(self, other):

        if other.is_zero():
            raise ZeroDivisionError("other must be nonzero")

        assert other.parent() is self.parent()

        ord = other.order()
        if self.order() < ord:
            return self.base_ring().one(), self.parent().zero(), self

        p, q = self, other

        # XXX: remove the (right?) content of self and other???

        D = self.parent().gen()
        sigma = self.parent().sigma()
        sigma_lc = [q.leading_coefficient()]
        for i in range(p.order() - q.order()):
            sigma_lc.append(sigma(sigma_lc[-1]))

        den, quo, rem = p.base_ring().one(), p.parent().zero(), p

        while rem.order() >= ord:

            a = sigma_lc[rem.order() - ord]
            b = rem.leading_coefficient()

            g = a.gcd(b)
            try:
                a //= g
                b //= g
            except TypeError:
                a = sigma_lc[rem.order() - ord]
                b = rem.leading_coefficient()

            cfquo = b*D**(rem.order() - ord)
            den = a*den
            quo = a*quo + cfquo
            rem = a*rem - cfquo*other

        # assert den*self == quo*other + rem
        # assert quo.parent() is self.parent()
        # assert rem.parent() is self.parent()

        return den, quo, rem

    def gcrd(self, *other, **kwargs):
        """
        Returns the greatest common right divisor of ``self`` and one or more ``other`` operators.

        INPUT:

        - ``other`` -- one or more operators which together with ``self`` can be coerced to a common parent.
        - ``prs`` (default: "essential") -- pseudo remainder sequence to be used. Possible values are
          "essential", "primitive", "classic", "subresultant", "monic".
        
        OUTPUT:

        An operator of maximum possible order which right divides ``self`` and all the ``other`` operators.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: A = OreAlgebra(ZZ['n'], 'Sn')
           sage: G = A.random_element(2)
           sage: L1, L2 = A.random_element(7), A.random_element(5)
           sage: while L1.gcrd(L2) != 1: L2 = A.random_element(5)                       
           sage: L1, L2 = L1*G, L2*G                                                        
           sage: L1.gcrd(L2) == G.normalize()
           True
           sage: L3, S, T = L1.xgcrd(L2)                             
           sage: S*L1 + T*L2 == L3
           True

        """

        if len(other) > 1:
            return reduce(lambda p, q: p.gcrd(q), other, self)
        elif len(other) == 0:
            return self

        other = other[0]
        if self.is_zero():
            return other
        elif other.is_zero():
            return self
        elif self in self.base_ring() or other in self.base_ring():
            return self.parent().one()
        elif self.parent() is not other.parent():
            A, B = canonical_coercion(self, other)
            return A.gcrd(B, **kwargs)

        prs = kwargs["prs"] if "prs" in kwargs else None
        infolevel = kwargs["infolevel"] if "infolevel" in kwargs else 0

        r = (self,other)
        if (r[0].order()<r[1].order()):
            r=(other,self)

        R = self.parent()

        prslist = {"essential" : __essentialPRS__,
                   "primitive" : __primitivePRS__,
                   "classic" : __classicPRS__,
                   "subresultant" : __subresultantPRS__,
                   "monic" : __monicPRS__,
                   }

        try:
            prs = prslist[prs]
        except:
            if self.base_ring().is_field():
                prs = __classicPRS__
            else:
                prs = __essentialPRS__

        additional = []
        while not r[1].is_zero():
            (r2,q,alpha,beta,correct)=prs(r,additional)
            if not correct:
                if infolevel>0: print("switching to primitive PRS")
                prs = __primitivePRS__
            else:
                r=r2
                if infolevel>1: print(r[0].order())
        r=r[0]

        return r.normalize()

    def xgcrd(self, other, **kwargs):
        """
        Returns the greatest common right divisor of ``self`` and ``other`` together with the cofactors. 

        INPUT:

        - ``other`` -- one operator which together with ``self`` can be coerced to a common parent.
        - ``prs`` (default: "essential") -- pseudo remainder sequence to be used. Possible values are
          "essential", "primitive", "classic", "subresultant", "monic".
        
        OUTPUT:

        A triple `(g, s, t)` of operators such that `g` is the greatest common right divisor of ``self`` and
        ``other`` and `g = s*p+t*q` where `p` is ``self`` and `q` is ``other``.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: A = OreAlgebra(ZZ['n'], 'Sn')
           sage: G = A.random_element(2)
           sage: L1, L2 = A.random_element(7), A.random_element(5)
           sage: while L1.gcrd(L2) != 1: L2 = A.random_element(5)                       
           sage: L1, L2 = L1*G, L2*G                                                        
           sage: L1.gcrd(L2) == G.normalize()
           True
           sage: L3, S, T = L1.xgcrd(L2)                             
           sage: S*L1 + T*L2 == L3
           True

        """
        prs = kwargs["prs"] if "prs" in kwargs else None
        infolevel = kwargs["infolevel"] if "infolevel" in kwargs else 0
        return self._xeuclid(other, prs, "bezout", infolevel)

    def _xeuclid(self, other, prs=None, retval="bezout", infolevel=0):
        # retval == "bezout" ===> returns (g, u, v) st gcrd(self, other) == g == u*self + v*other
        # retval == "syzygy" ===> returns the smallest degree syzygy (u, v) of self and other

        if self.parent() is not other.parent():
            A, B = canonical_coercion(self, other)
            return A._xeuclid(B, prs, retval, infolevel)
        elif retval == "bezout":
            if self.is_zero() or other in other.base_ring():
                return other, self.parent().zero(), self.parent().one()
            elif other.is_zero() or self in self.base_ring():
                return self, self.parent().one(), self.parent().zero()
        elif retval == "syzygy":
            if other.is_zero():
                return self.parent().zero(), self.parent().one()
            elif self.is_zero():
                return self.parent().one(), self.parent().zero()

        prslist = {"essential" : __essentialPRS__,
                   "primitive" : __primitivePRS__,
                   "classic" : __classicPRS__,
                   "subresultant" : __subresultantPRS__,
                   "monic" : __monicPRS__,
        }

        if retval == "syzygy": 
            prs = __primitivePRS__ # overrule any given options
        else:
            try:
                prs = prslist[prs]
            except:
                if self.base_ring().is_field():
                    prs = __classicPRS__
                else:
                    prs = __essentialPRS__

        r = (self, other)
        if r[0].order() < r[1].order():
            r = (other, self)
        
        R = r[0].parent()
        RF = R.change_ring(R.base_ring().fraction_field())

        a11, a12, a21, a22 = RF.one(), RF.zero(), RF.zero(), RF.one()

        if prs is None:
            prs = __classicPRS__ if R.base_ring().is_field() else (__essentialPRS__ if retval=="bezout" else __primitivePRS__)

        additional = []

        while not r[1].is_zero():  
            (r2, q, alpha, beta, correct) = prs(r, additional)
            if not correct:
                if infolevel>0: print("switching to primitive PRS")
                prs = __primitivePRS__
            else:
                r = r2; bInv = ~beta
                a11, a12, a21, a22 = a21, a22, bInv*(alpha*a11 - q*a21), bInv*(alpha*a12 - q*a22)
                if infolevel>1: print(r[0].order())
        if retval == "syzygy":
            c = a21.denominator().lcm(a22.denominator())
            return (c*a21, c*a22)

        r = r[0]
        c = RF.base_ring().one() if prs is __classicPRS__ else ~r.content()
        return (self.parent()(c*r), c*a11, c*a12) if self.order()>=other.order() else (self.parent()(c*r), c*a12, c*a11)

    def lclm(self, *other, **kwargs):
        """
        Computes the least common left multiple of ``self`` and ``other``.

        That is, it returns an operator `L` of minimal order such that there
        exist `U` and `V` with `L=U*self=V*other`. The base ring of the
        parent of `U` and `V` is the fraction field of the base ring of the
        parent of ``self`` and ``other``. The parent of `L` is the same as
        the parent of the input operators.

        If more than one operator is given, the function computes the lclm
        of all the operators.

        The optional argument ``algorithm`` allows to select between the following
        methods.

        * ``linalg`` (default) -- makes an ansatz for cofactors and solves a linear
          system over the base ring. 
          Through the optional argument ``solver``, a callable object can be
          provided which the function should use for computing the kernel of
          matrices with entries in the Ore algebra's base ring. 

        * ``euclid`` -- uses the extended Euclidean algorithm to compute a minimal
          syzygy between the operators in the input. Further optional arguments
          can be passed as explained in the docstring of ``xgcrd``.

        * ``guess`` -- computes the first terms of a solution of ``self`` and ``other``
          and guesses from these a minimal operator annihilating a generic linear
          combination. Unless the input are recurrence operators, an keyword argument
          ``to_list`` has to be present which specifies a function for computing the
          terms (input: an operator, a list of initial values, and the desired number
          of terms). This method is heuristic. It may be much faster than the others,
          but with low probability its output is incorrect or it aborts with an error. 

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = ZZ['x']
            sage: Alg.<Dx> = OreAlgebra(R, 'Dx')
            sage: A = 5*(x+1)*Dx + (x - 7); B = (3*x+5)*Dx - (8*x+1)
            sage: L = A.lclm(B)
            sage: L
            (645*x^4 + 2155*x^3 + 1785*x^2 - 475*x - 750)*Dx^2 + (-1591*x^4 - 3696*x^3 - 3664*x^2 - 2380*x - 725)*Dx - 344*x^4 + 2133*x^3 + 2911*x^2 + 1383*x + 1285
            sage: A*B
            (15*x^2 + 40*x + 25)*Dx^2 + (-37*x^2 - 46*x - 25)*Dx - 8*x^2 + 15*x - 33
            sage: B.lclm(A*B)
            (15*x^2 + 40*x + 25)*Dx^2 + (-37*x^2 - 46*x - 25)*Dx - 8*x^2 + 15*x - 33
            sage: B.lclm(L, A*B) 
            (3225*x^5 + 18275*x^4 + 42050*x^3 + 49550*x^2 + 29925*x + 7375)*Dx^3 + (-7310*x^5 - 32035*x^4 - 64640*x^3 - 70730*x^2 - 40090*x - 9275)*Dx^2 + (-3311*x^5 - 3913*x^4 - 6134*x^3 - 20306*x^2 - 25147*x - 9605)*Dx - 344*x^5 + 645*x^4 - 7180*x^3 + 2054*x^2 + 30044*x + 22509

        
        """

        if len(other) != 1:
            # possible improvement: rewrite algorithms to allow multiple arguments where possible
            other = list(other); other.append(self); other.sort(key=lambda p: p.order())
            return reduce(lambda p, q: p.lclm(q, **kwargs), other)
        elif len(other) == 0:
            return self

        other = other[0]
        if self.is_zero() or other.is_zero():
            return self.parent().zero()
        elif self.order() == 0:
            return other
        elif other in self.base_ring():
            return self
        elif self.parent() is not other.parent():
            A, B = canonical_coercion(self, other)
            return A.lclm(B, **kwargs)
        elif not isinstance(other, UnivariateOreOperator):
            raise TypeError("unexpected argument in lclm")

        if not "algorithm" in kwargs or kwargs['algorithm'] == 'linalg':
            return self._lclm_linalg(other, **kwargs)
        elif kwargs['algorithm'] == 'euclid':
            del kwargs['algorithm']; kwargs['retval'] = 'syzygy'
            u, v = self._xeuclid(other, **kwargs)
            return (u*other).normalize()
        elif kwargs['algorithm'] == 'guess':
            del kwargs['algorithm']
            return self._lclm_guess(other, **kwargs)
        else:
            raise ValueError("unknown algorithm: " + str(kwargs['algorithm']))

    def _lclm_linalg(self, other, **kwargs):
        """
        lclm algorithm based on ansatz and linear algebra over the base ring. 

        see docstring of lclm for further information. 
        """

        solver = kwargs["solver"] if "solver" in kwargs else None

        A = self.numerator(); r = A.order()
        B = other.numerator(); s = B.order()
        D = A.parent().gen()

        t = max(r, s) # current hypothesis for the order of the lclm

        rowsA = [A]
        for i in range(t - r):
            rowsA.append(D*rowsA[-1])
        rowsB = [B]
        for i in range(t - s):
            rowsB.append(D*rowsB[-1])

        from sage.matrix.constructor import Matrix
        if solver == None:
            solver = A.parent()._solver()

        sys = Matrix(list(map(lambda p: p.coefficients(sparse=False,padd=t), rowsA + rowsB))).transpose()
        sol = solver(sys)

        while len(sol) == 0:
            t += 1
            rowsA.append(D*rowsA[-1]); rowsB.append(D*rowsB[-1])
            sys = Matrix(list(map(lambda p: p.coefficients(sparse=False,padd=t), rowsA + rowsB))).transpose()
            sol = solver(sys)

        U = A.parent()(list(sol[0])[:t+1-r])
        return self.parent()((U*A).normalize())

    def _lclm_guess(self, other, **kwargs):
        """
        lclm algorithm based on guessing.

        see docstring of lclm for further information.
        """

        # lclm based on guessing an operator for a generic linear combination of two solutions. 
        
        A = self.parent(); R = A.base_ring(); K = R.base_ring().fraction_field()
        if 'to_list' in kwargs:
            terms = kwargs['to_list']
        elif A.is_S():
            terms = lambda L, n : L.to_list([K.random_element() for i in range(L.order())], n)
        else:
            raise TypeError("don't know how to expand a generic solution for operators in " + str(A))

        U = self.normalize().numerator(); V = other.normalize().numerator()

        # bound on the order of the output
        r_lcm = U.order() + V.order() 

        # expected degree of the non-removable part of the leading coefficient
        # heuristic: assume a factor of lc is removable if its multiplicity is 1 and its degree is >20
        d_ess = sum([ p.degree() for L in (U, V)
                                 for p, e in L.leading_coefficient().factor()
                                 if e==1 and p.degree() > 20 ])
        d_ess = U.leading_coefficient().degree() + V.leading_coefficient().degree() - d_ess

        # expected degree of the removable part of the leading coefficient
        d_res = U.degree()*V.order() + V.degree()*U.order()

        # assuming existence of left multiples of size (r,d) where  d >= d_ess + d_res/(r - r_lcm + 1)
        # optimal (r,d) as follows:
        from math import sqrt
        r = r_lcm - 1 + d_res*r_lcm/sqrt((1+d_ess)*d_res*r_lcm)
        d = d_ess + sqrt((1+d_ess)*d_res*r_lcm)/r_lcm

        n = int(1.20 * (r + 2) * (d + 2) + 10) # number of terms needed + some buffer

        data = list(map(lambda p, q: 1234*p + 4321*q, terms(U, n), terms(V, n)))

        from guessing import guess
        return guess(data, self.parent(), min_order=r_lcm)

    def xlclm(self, other):
        """
        Computes the least common left multiple of ``self`` and ``other`` along
        with the appropriate cofactors. 

        That is, it returns a triple `(L,U,V)` such that `L=U*self=V*other` and
        `L` has minimal possible order.
        The base ring of the parent of `U` and `V` is the fraction field of the
        base ring of the parent of ``self`` and ``other``.
        The parent of `L` is the same as the parent of the input operators.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = QQ['x']
            sage: Alg.<Dx> = OreAlgebra(R, 'Dx')
            sage: A = 5*(x+1)*Dx + (x - 7); B = (3*x+5)*Dx - (8*x+1)
            sage: L, U, V = A.xlclm(B)
            sage: L == U*A
            True
            sage: L == V*B
            True
            sage: L.parent()
            Univariate Ore algebra in Dx over Univariate Polynomial Ring in x over Rational Field
            sage: U.parent()
            Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field
        
        """
        A = self; B = other; L = self.lclm(other)
        K = L.parent().base_ring()

        if K.is_field():
            L0 = L
        else:
            K = K.fraction_field()
            A = A.change_ring(K)
            B = B.change_ring(K)
            L0 = L.change_ring(K)
        
        return (L, L0 // A, L0 // B)

    def resultant(self, other):
        """
        Returns the resultant between this operator and ``other``. 

        INPUT:
        
        - ``other`` -- some operator that lives in the same algebra as ``self``.
        
        OUTPUT:

        The resultant between ``self`` and ``other``, which is defined as the determinant of the
        `(n+m) x (n+m)` matrix `[ A, D*A, ..., D^{m-1}*A, B, D*B, ..., D^{n-1}*B ]` constructed
        from the coefficient vectors of the operators obtained from ``self`` and ``other`` by
        multiplying them by powers of the parent's generator. 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: L1 = (5*x+3)*Dx^3 + (7*x+4)*Dx^2 + (3*x+2)*Dx + (4*x-1)
           sage: L2 = (8*x-7)*Dx^2 + (x+9)*Dx + (2*x-3)
           sage: L1.resultant(L2)
           2934*x^5 - 8200*x^4 + 32161*x^3 - 83588*x^2 - 67505*x + 42514
           sage: L2.resultant(L1)
           -2934*x^5 + 8200*x^4 - 32161*x^3 + 83588*x^2 + 67505*x - 42514

           sage: R.<x> = ZZ['x']
           sage: A.<Sx> = OreAlgebra(R, 'Sx')
           sage: L1 = (5*x+3)*Sx^3 + (7*x+4)*Sx^2 + (3*x+2)*Sx + (4*x-1)
           sage: L2 = (8*x-7)*Sx^2 + (x+9)*Sx + (2*x-3)
           sage: L1.resultant(L2)
           2934*x^5 + 3536*x^4 + 11142*x^3 + 16298*x^2 - 1257*x - 2260
           sage: L2.resultant(L1)
           -2934*x^5 - 3536*x^4 - 11142*x^3 - 16298*x^2 + 1257*x + 2260

        """
        if self.parent() is not other.parent():
            A, B = canonical_coercion(self, other)
            return A.resultant(B)

        n = self.order(); m = other.order()

        if n < m:
            return other.resultant(self) * (-1)**(n+m)

        Alg = self.parent(); s = Alg.sigma()
        mat = []; A = None; D = Alg.gen()

        # for better performance, we don't use the sylvester matrix 
        for i in range(m):
            A = self if A == None else D*A
            mat.append((A % other).coefficients(sparse=False,padd=m-1))

        from sage.matrix.constructor import matrix      
        return s.factorial(other.leading_coefficient(), n) * matrix(Alg.base_ring().fraction_field(), mat).det()

    def companion_matrix(self):
        r"""
        Returns the companion matrix of ``self``.

        If `r` is the order of ``self`` and `y` is annihilated by ``self``, then the companion matrix
        as computed by this method is an `r\times r` matrix `M` such that 
        `[\partial y,\partial^2 y,\dots,\partial^r y] = M [y,\partial y,\dots,\partial^{r-1}y]^T`.

        In the shift case, if `c_i` is a sequence annihilated by ``self``, then also
        `[c_{i+1}, c_{i+2}, \ldots, c_{i+r}]^T = M(i) [c_i, c_{i+1}, \ldots, c_{i+r-1}]^T`

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<n> = QQ['n']
            sage: A.<Sn> = OreAlgebra(R, 'Sn')
            sage: M = ((-n-4)*Sn**2 + (5+2*n)*Sn + (3+3*n)).companion_matrix()
            sage: M
            [                0                 1]
            [(3*n + 3)/(n + 4) (2*n + 5)/(n + 4)]
            sage: initial = Matrix([[1],[1]])
            sage: [prod(M(k) for k in range(n, -1, -1)) * initial for n in range(10)]
            [
            [1]  [2]  [4]  [ 9]  [21]  [ 51]  [127]  [323]  [ 835]  [2188]
            [2], [4], [9], [21], [51], [127], [323], [835], [2188], [5798]
            ]

        """
        from sage.matrix.constructor import Matrix
        ring = self.base_ring().fraction_field()
        r = self.order()
        M = Matrix(ring, r, r)
        for i in range(r-1):
            M[i, i+1] = 1
        for j in range(r):
            M[r - 1, j] = self[j] / (-self[r])
        return M

    def symmetric_product(self, other, solver=None):
        """
        Returns the symmetric product of ``self`` and ``other``.

        The symmetric product of two operators `A` and `B` is a minimal order
        operator `C` such that for all \"functions\" `f` and `g` with `A.f=B.g=0`
        we have `C.(fg)=0`.

        The method requires that a product rule is associated to the Ore algebra
        where ``self`` and ``other`` live. (See docstring of OreAlgebra for information
        about product rules.)

        If no ``solver`` is specified, the the Ore algebra's solver is used.         

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (Dx - 1).symmetric_product(x*Dx - 1)
           x*Dx - x - 1
           sage: (x*Dx - 1).symmetric_product(Dx - 1)
           x*Dx - x - 1
           sage: ((x+1)*Dx^2 + (x-1)*Dx + 8).symmetric_product((x-1)*Dx^2 + (2*x+3)*Dx + (8*x+5))
           (29*x^8 - 4*x^7 - 55*x^6 - 34*x^5 - 23*x^4 + 80*x^3 + 95*x^2 - 42*x - 46)*Dx^4 + (174*x^8 + 150*x^7 + 48*x^6 - 294*x^5 - 864*x^4 - 646*x^3 + 232*x^2 + 790*x + 410)*Dx^3 + (783*x^8 + 1661*x^7 - 181*x^6 - 1783*x^5 - 3161*x^4 - 3713*x^3 + 213*x^2 + 107*x - 1126)*Dx^2 + (1566*x^8 + 5091*x^7 + 2394*x^6 + 2911*x^5 - 10586*x^4 - 23587*x^3 - 18334*x^2 - 2047*x + 5152)*Dx + 2552*x^8 + 3795*x^7 + 8341*x^6 + 295*x^5 - 6394*x^4 - 24831*x^3 - 35327*x^2 - 23667*x - 13708
           
           sage: A.<Sx> = OreAlgebra(R, 'Sx')
           sage: (Sx - 2).symmetric_product(x*Sx - (x+1))
           x*Sx - 2*x - 2
           sage: (x*Sx - (x+1)).symmetric_product(Sx - 2)
           x*Sx - 2*x - 2
           sage: ((x+1)*Sx^2 + (x-1)*Sx + 8).symmetric_product((x-1)*Sx^2 + (2*x+3)*Sx + (8*x+5))
           (-8*x^8 - 13*x^7 + 300*x^6 + 1640*x^5 + 3698*x^4 + 4373*x^3 + 2730*x^2 + 720*x)*Sx^4 + (16*x^8 + 34*x^7 - 483*x^6 - 1947*x^5 - 2299*x^4 - 2055*x^3 - 4994*x^2 - 4592*x)*Sx^3 + (-64*x^8 + 816*x^7 + 1855*x^6 - 21135*x^5 - 76919*x^4 - 35377*x^3 + 179208*x^2 + 283136*x + 125440)*Sx^2 + (1024*x^7 + 1792*x^6 - 39792*x^5 - 250472*x^4 - 578320*x^3 - 446424*x^2 + 206528*x + 326144)*Sx - 32768*x^6 - 61440*x^5 + 956928*x^4 + 4897984*x^3 + 9390784*x^2 + 7923200*x + 2329600
        
        """
        if not isinstance(other, UnivariateOreOperator):
            raise TypeError("unexpected argument in symmetric_product")

        if self.parent() != other.parent():
            A, B = canonical_coercion(self, other)
            return A.symmetric_product(B, solver=solver)

        R = self.base_ring().fraction_field(); zero = R.zero(); one = R.one()
        
        A = self.change_ring(R);  a = A.order()
        B = other.change_ring(R); b = B.order()

        Alg = A.parent(); sigma = Alg.sigma(); delta = Alg.delta();

        if A.is_zero() or B.is_zero():
            return A
        elif min(a, b) < 1:
            return A.parent().one()
        elif a == 1 and b > 1:
            A, B, a, b = B, A, b, a

        pr = Alg._product_rule()
        if pr is None:
            raise ValueError("no product rule found")

        if b == 1:
            
            D = A.parent().gen(); D1 = D(R.one())
            h = -B[0]/B[1] # B = D - h
            if h == D1:
                return A            

            # define g such that (D - h)(u) == 0 iff (D - g)(1/u) == 0.
            g = (D1 - pr[0] - pr[1]*h)/(pr[1] + pr[2]*h)
            
            # define p, q such that "D*1/u == p*1/u*D + q*1/u" 
            #p = (g - D1)/(D1 - h); q = g - p*D1
            p = pr[1] + pr[2]*g; q = pr[0] + pr[1]*g

            # calculate L with L(u*v)=0 iff A(v)=0 and B(u)=0 using A(1/u * u*v) = 0
            coeffs = A.coefficients(sparse=False); L = coeffs[0]; Dk = A.parent().one()
            for i in range(1, A.order() + 1):
                #Dk = Dk.map_coefficients(sigma_u)*D + Dk.map_coefficients(delta_u) [[buggy??]]
                Dk = (p*D + q)*Dk
                c = coeffs[i]
                if not c.is_zero():
                    L += c*Dk
            
            return A.parent()(L).normalize()

        # general case via linear algebra

        Ared = tuple(-A[i]/A[a] for i in range(a)); Bred = tuple(-B[j]/B[b] for j in range(b))

        if solver is None:
            solver = Alg._solver()

        # Dkuv[i][j] is the coefficient of D^i(u)*D^j(v) in the normal form of D^k(u*v) 
        Dkuv = [[zero for i in range(b + 1)] for j in range(a + 1)]; Dkuv[0][0] = one
        
        mat = [[Dkuv[i][j] for i in range(a) for j in range(b)]]

        from sage.matrix.constructor import Matrix
        sol = solver(Matrix(mat).transpose())

        while len(sol) == 0:

            # push
            for i in range(a - 1, -1, -1):
                for j in range(b - 1, -1, -1):
                    s = sigma(Dkuv[i][j])
                    Dkuv[i + 1][j + 1] += s*pr[2]
                    Dkuv[i][j + 1] += s*pr[1]
                    Dkuv[i + 1][j] += s*pr[1]
                    Dkuv[i][j] = delta(Dkuv[i][j]) + s*pr[0]

            # reduce
            for i in range(a + 1):
                if not Dkuv[i][b] == zero:
                    for j in range(b):
                        Dkuv[i][j] += Bred[j]*Dkuv[i][b]
                    Dkuv[i][b] = zero

            for j in range(b): # not b + 1
                if not Dkuv[a][j] == zero:
                    for i in range(a):
                        Dkuv[i][j] += Ared[i]*Dkuv[a][j]
                    Dkuv[a][j] = zero

            # solve
            mat.append([Dkuv[i][j] for i in range(a) for j in range(b)])
            sol = solver(Matrix(mat).transpose())

        L = A.parent()(list(sol[0]))
        return L

    def symmetric_power(self, exp, solver=None):
        """
        Returns a symmetric power of this operator.

        The `n` th symmetric power of an operator `L` is a minimal order operator `Q`
        such that for all \"functions\" `f` annihilated by `L` the operator `Q` annihilates
        the function `f^n`.

        For further information, see the docstring of ``symmetric_product``.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (Dx^2 + x*Dx - 2).symmetric_power(3)
           Dx^4 + 6*x*Dx^3 + (11*x^2 - 16)*Dx^2 + (6*x^3 - 53*x)*Dx - 36*x^2 + 24
           sage: A.<Sx> = OreAlgebra(R, 'Sx')
           sage: (Sx^2 + x*Sx - 2).symmetric_power(2)
           -x*Sx^3 + (x^3 + 2*x^2 + 3*x + 2)*Sx^2 + (2*x^3 + 2*x^2 + 4*x)*Sx - 8*x - 8
           sage: A.random_element().symmetric_power(0)
           Sx - 1
        
        """
        if exp < 0:
            raise TypeError("unexpected exponent received in symmetric_power")
        elif exp == 0:
            D = self.parent().gen(); R = D.base_ring()
            return D - R(D(R.one())) # annihilator of 1
        elif exp == 1:
            return self
        elif exp % 2 == 1:
            L = self.symmetric_power(exp - 1, solver=solver)
            return L.symmetric_product(self, solver=solver)
        elif exp % 2 == 0:
            L = self.symmetric_power(exp/2, solver=solver)
            return L.symmetric_product(L, solver=solver)
        else:
            raise TypeError("unexpected exponent received in symmetric_power")

    def annihilator_of_associate(self, other, solver=None):
        """
        Computes an operator `L` with `L(other(f))=0` for all `f` with `self(f)=0`.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (Dx^2 + x*Dx + 5).annihilator_of_associate(Dx + 7*x+3)
           (42*x^2 + 39*x + 7)*Dx^2 + (42*x^3 + 39*x^2 - 77*x - 39)*Dx + 168*x^2 + 174*x + 61
           sage: A.<Sx> = OreAlgebra(R, 'Sx')
           sage: (Sx^2 + x*Sx + 5).annihilator_of_associate(Sx + 7*x+3)
           (42*x^2 + 88*x + 35)*Sx^2 + (42*x^3 + 130*x^2 + 53*x - 65)*Sx + 210*x^2 + 860*x + 825

        """
        if not isinstance(other, UnivariateOreOperator):
            raise TypeError("unexpected argument in symmetric_product")

        if self.parent() != other.parent():
            A, B = canonical_coercion(self, other)
            return A.annihilator_of_associate(B, solver=solver)

        if self.is_zero():
            return self
        elif other.is_zero():
            return self.parent().one()

        R = self.base_ring().fraction_field()
        A = self.change_ring(R); a = A.order()
        B = other.change_ring(R) % A
        D = A.parent().gen()

        if solver == None:
            solver = A.parent()._solver()

        mat = [B.coefficients(sparse=False,padd=a-1)]

        from sage.matrix.constructor import Matrix
        sol = solver(Matrix(mat).transpose())

        while len(sol) == 0:
            B = (D*B) % A
            mat.append(B.coefficients(sparse=False,padd=a-1))
            sol = solver(Matrix(mat).transpose())

        L = A.parent()(list(sol[0]))
        return L

    def annihilator_of_polynomial(self, poly, solver=None, blocks=1):
        """
        Returns an annihilating operator of a polynomial expression evaluated at solutions of ``self``.

        INPUT:

        - ``poly`` -- a multivariate polynomial, say in the variables `x0,x1,x2,...`, with coefficients
          in the base ring of the parent of ``self``.
          The number of variables in the parent of ``poly`` must be at least the order of ``self``.
        - ``solver`` -- if specified, this function will be used for computing the nullspace of 
          polynomial matrices
        - ``blocks`` -- if set to an integer greater than 1, the variables of the polynomial ring 
          represent the shifts of several solutions of this operator. In this case, the polynomial
          ring must have ``blocks*n`` many variables, for some `n` which is at least the order of ``self``.
          Then the first ``n`` variables represent the shifts of one solution, the second ``n`` variables
          represent the shifts of a second solution, and so on.
        
        OUTPUT:

          An operator `L` with the following property. 
          Let `A` be the parent of ``self``.
          For a function `f` write `Df,D^2f,...` for the functions obtained from `f` by letting the generator
          of `A` act on them. 
          Then the output `L` is such that for all `f` annihilated by ``self`` we have
          `L(p(f,Df,D^2f,...))=0`, where `p` is the input polynomial.

        The method requires that a product rule is associated to `A`. 
        (See docstring of OreAlgebra for information about product rules.)

        NOTE:

          Even for small input, the output operator may be very large, and its computation may need a lot of time.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: K.<x> = ZZ['x']; K = K.fraction_field(); R.<n> = K['n']
          sage: A.<Sn> = OreAlgebra(R, 'Sn'); R.<y0,y1,y2> = K['n'][]
          sage: L = (n+2)*Sn^2 - (2*n+3)*x*Sn + (n+1)
          sage: L.annihilator_of_polynomial(y1^2-y2*y0) # random; Turan's determinant
          (2*n^4 + 31*n^3 + 177*n^2 + 440*n + 400)*Sn^3 + ((-8*x^2 + 2)*n^4 + (-100*x^2 + 21)*n^3 + (-462*x^2 + 75)*n^2 + (-935*x^2 + 99)*n - 700*x^2 + 28)*Sn^2 + ((8*x^2 - 2)*n^4 + (92*x^2 - 27)*n^3 + (390*x^2 - 129)*n^2 + (721*x^2 - 261)*n + 490*x^2 - 190)*Sn - 2*n^4 - 17*n^3 - 51*n^2 - 64*n - 28
          sage: M = L.annihilator_of_associate(Sn).symmetric_power(2).lclm(L.annihilator_of_associate(Sn^2).symmetric_product(L)) # same by lower level methods. 
          sage: M.order() # overshoots
          7
          sage: M % L.annihilator_of_polynomial(y1^2-y2*y0) 
          0

          sage: K = ZZ; R.<x> = K['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx'); R.<y0,y1> = K['x'][]
          sage: L = (2*x+3)*Dx^2 + (4*x+5)*Dx + (6*x+7)
          sage: L.annihilator_of_polynomial((2*x+1)*y0^2-y1^2)
          (16*x^7 + 112*x^6 + 312*x^5 + 388*x^4 + 85*x^3 - 300*x^2 - 303*x - 90)*Dx^3 + (96*x^7 + 600*x^6 + 1420*x^5 + 1218*x^4 - 747*x^3 - 2285*x^2 - 1623*x - 387)*Dx^2 + (320*x^7 + 1920*x^6 + 4288*x^5 + 3288*x^4 - 2556*x^3 - 6470*x^2 - 4322*x - 1014)*Dx + 384*x^7 + 2144*x^6 + 4080*x^5 + 1064*x^4 - 7076*x^3 - 10872*x^2 - 6592*x - 1552

        """

        if self.is_zero():
            return self
        elif self.order() == 0:
            return self.one()
        
        A = self.parent(); pr = A._product_rule(); R = poly.parent(); r = self.order(); vars = R.gens()
        if len(vars) % blocks != 0 or len(vars) < r*blocks:
            raise TypeError("illegal number of variables")
        elif R.base_ring().fraction_field() is not self.base_ring().fraction_field():
            raise TypeError("poly must live in a ring with coefficient field " + str(self.base_ring()) + ".")
        elif pr is None:
            raise ValueError("no product rule found")

        K = R.base_ring().fraction_field()
        A = A.change_ring(K); R = R.change_ring(K); 
        L = A(self); poly = R(poly)
        sigma = A.sigma(); delta = A.delta()

        shift_cache = { R.one().exponents()[0] : R.one() }
        for j in range(blocks):
            J = j*len(vars)//blocks
            for i in range(r - 1):
                shift_cache[vars[J + i].exponents()[0]] = vars[J + i + 1]
            shift_cache[vars[J + r - 1].exponents()[0]] = \
                (-1/L.leading_coefficient())*sum(L[i]*vars[J + i] for i in range(r))

        def shift(p): # computes D( p ), as element of R
            out = R.zero()
            for m, c in zip(p.monomials(), p.coefficients()):
                exp = m.exponents()[0]
                if exp not in shift_cache:
                    x = vars[min([i for i in range(len(vars)) if exp[i] > 0])]
                    m0 = m//x; A = shift_cache[x.exponents()[0]]; B = shift(m0)
                    shift_cache[exp] = pr[0]*m + pr[1]*(A*m0 + x*B) + pr[2]*A*B
                out += sigma(c)*shift_cache[exp]
            return p.map_coefficients(delta) + out
        
        if len(vars) > blocks*r:
            subs = {}
            for j in range(blocks):
                J = j*len(vars)//blocks; p = vars[J]; subs[str(p)] = p
                for i in range(len(vars)//blocks - 1):
                    p = shift(p); subs[str(vars[J + i + 1])] = p
            poly = poly(**subs)

        if solver is None:
            solver = A._solver()

        shifts = [poly]; basis = set(poly.monomials()) # set of all monomials appearing in any of the shifts
        from sage.matrix.constructor import Matrix
        sol = []

        while len(sol) == 0:

            shifts.append(shift(shifts[-1]))
            basis = basis.union(shifts[-1].monomials())
            sol = solver(Matrix(K, [[shifts[i].monomial_coefficient(m) for i in range(len(shifts))] for m in basis ]))

        return self.parent()(list(sol[0]))

    def exterior_power(self, k, skip=[]):
        """
        Returns the `k`-th exterior power of this operator.

        This is an operator which annihilates the Wronskian of any `k` solutions of this operator. 
        The exterior power is unique up to left-multiplication by base ring elements. This method
        returns a normalized operator. 

        If the optional argument ``skip`` is supplied, we take a `k` times `k` Wronskian in which 
        the rows corresponding to the `i`-th derivative is skipped for all `i` in the list. 

        When `k` exceeds the order of ``self``, we raise an error rather than returning the operator 1.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: t = ZZ['t'].gen(); A.<Dt> = OreAlgebra(ZZ[t])
           sage: L = (6*t^2 - 10*t - 2)*Dt^3 + (-3*t^2 + 2*t + 7)*Dt^2 + (t + 3)*Dt + 7*t^2 - t + 1
           sage: L.exterior_power(1)
           (6*t^2 - 10*t - 2)*Dt^3 + (-3*t^2 + 2*t + 7)*Dt^2 + (t + 3)*Dt + 7*t^2 - t + 1
           sage: L.exterior_power(2)
           (36*t^4 - 120*t^3 + 76*t^2 + 40*t + 4)*Dt^3 + (-36*t^4 + 84*t^3 + 56*t^2 - 148*t - 28)*Dt^2 + (9*t^4 - 6*t^3 - 12*t^2 - 76*t + 109)*Dt - 42*t^4 + 73*t^3 - 15*t^2 - 15*t + 51
           sage: L.exterior_power(3)
           (6*t^2 - 10*t - 2)*Dt - 3*t^2 + 2*t + 7
        
        """
        from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
        from sage.matrix.constructor import matrix

        r = self.order(); assert(1 <= k <= r); B = self.base_ring()
        R = PolynomialRing(B, ['f' + str(i) + '_' + str(j) for i in range(k) for j in range(r + len(skip)) ])
        poly = matrix(R, k, r + len(skip), R.gens()).delete_columns(skip).submatrix(0, 0, k, k).det()
        return self.annihilator_of_polynomial(poly, blocks=k).normalize()

    def adjoint(self):
        """
        Returns the adjoint of this operator. 

        The adjoint is a map `a` from the Ore algebra to itself with the property that 
        `a(A*B)==a(B)*a(A)` and `a(a(A))==A` for all operators `A` and `B`. 

        This method may not be defined for every Ore algebra. A necessary (but not 
        sufficient) requirement is that sigma be invertible.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ[]; A.<Dx> = OreAlgebra(R)
          sage: L = (x+5)*Dx^2 + (x-3)*Dx + (5*x+7)
          sage: L.adjoint().adjoint() == L
          True
          sage: M = (8*x-7)*Dx^3 + (4*x+5)*Dx + (9*x-1)
          sage: (M*L).adjoint() == L.adjoint()*M.adjoint()
          True
          sage: A.<Sx> = OreAlgebra(R)
          sage: L = (x+5)*Sx^2 + (x-3)*Sx + (5*x+7)
          sage: L.adjoint().adjoint() == L
          True
          sage: M = (8*x-7)*Sx^3 + (4*x+5)*Sx + (9*x-1)
          sage: (M*L).adjoint() == L.adjoint()*M.adjoint()
          True
          sage: R.<x> = QQ[] # ensures that sigma of A below is invertible
          sage: A.<Qx> = OreAlgebra(R, q=2)
          sage: L = (x+5)*Qx^2 + (x-3)*Qx + (5*x+7)
          sage: L.adjoint().adjoint() == L
          True
          sage: M = (8*x-7)*Qx^3 + (4*x+5)*Qx + (9*x-1)
          sage: (M*L).adjoint() == L.adjoint()*M.adjoint()
          True

        """
        A = self.parent(); sinv = A.sigma().inverse(); delta = A.delta(); out = A.zero(); r = self.order(); D = A.gen()

        for c in reversed(self.coefficients(sparse=False)):
            out = c + out.map_coefficients(sinv)*D - out.map_coefficients(sinv).map_coefficients(delta)

        # at this point, out is the desired operator as element of k(x)[D, sinv, -(delta o sinv)]. 
        # mapping this back to the original algebra requires a case distinction.

        x = A.base_ring().gen()
        if A.is_D() or A.is_S():
            return out.map_coefficients(lambda p: p(-x))
        elif A.is_Q():
            return A.change_ring(A.base_ring().fraction_field())(out).map_coefficients(lambda p: p(~x))
        else:
            raise NotImplementedError

    # coefficient-related functions
    
    def order(self):
        """
        Returns the order of this operator, which is defined as the maximal power `i` of the
        generator which has a nonzero coefficient. The zero operator has order `-1`.
        """
        return self.polynomial().degree()

    def valuation(self):
        r"""
        Returns the valuation of this operator, which is defined as the minimal power `i` of the
        generator which has a nonzero coefficient. The zero operator has order `\infty`.
        """
        if self == self.parent().zero():
            return infinity
        else:
            return min(self.exponents())

    def __getitem__(self, n):
        return self.polynomial()[n]

    def __setitem__(self, n, value):
        raise IndexError("Operators are immutable")

    def leading_coefficient(self):
        return self.polynomial().leading_coefficient()

    def constant_coefficient(self):
        return self.polynomial()[0]

    leading_coefficient.__doc__ = OreOperator.leading_coefficient.__doc__
    constant_coefficient.__doc__ = OreOperator.constant_coefficient.__doc__

    def map_coefficients(self, f, new_base_ring = None):
        """
        Returns the polynomial obtained by applying ``f`` to the non-zero
        coefficients of self.
        """
        poly = self.polynomial().map_coefficients(f, new_base_ring = new_base_ring)
        if new_base_ring == None:
            return self.parent()(poly)
        else:
            return self.parent().base_extend(new_base_ring)(poly)

    def coefficients(self, **args):
        """
        Return the coefficient vector of this operator.

        If the degree is less than the number given in the optional
        argument ``padd``, the list is padded with zeros so as to ensure 
        that the output has length ``padd`` + 1. Any further 

        EXAMPLES::

           sage: from ore_algebra import *
           sage: A.<Sx> = OreAlgebra(ZZ['x'], 'Sx')
           sage: (5*Sx^3-4).coefficients(sparse=False)
           [-4, 0, 0, 5]
           sage: (5*Sx^3-4).coefficients(sparse=False,padd=5)
           [-4, 0, 0, 5, 0, 0]
           sage: (5*Sx^3-4).coefficients(sparse=False,padd=1)
           [-4, 0, 0, 5]
        
        """
        padd = args.setdefault("padd", -1)
        args['padd'] = 0; del args['padd']
        c = self.polynomial().coefficients(**args)
        if len(c) <= padd:
            z = self.base_ring().zero()
            c = c + [z for i in range(padd + 1 - len(c))]
        return c

    def exponents(self):
        return self.polynomial().exponents()

    coefficients.__doc__ = OreOperator.coefficients.__doc__
    exponents.__doc__ = OreOperator.exponents.__doc__

#############################################################################################################

def __primitivePRS__(r,additional):
    """
    Computes one division step in the primitive polynomial remainder sequence.
    """

    orddiff = r[0].order()-r[1].order()

    R = r[0].parent()
    alpha = R.sigma().factorial(r[1].leading_coefficient(),orddiff+1)
    newRem = (alpha*r[0]).quo_rem(r[1],fractionFree=True)
    beta = newRem[1].content()
    r2 = newRem[1].map_coefficients(lambda p: p//beta)
    
    return ((r[1],r2),newRem[0],alpha,beta,True)

def __classicPRS__(r,additional):
    """
    Computes one division step in the classic polynomial remainder sequence.
    """

    newRem = r[0].quo_rem(r[1])
    return ((r[1],newRem[1]),newRem[0],r[0].parent().base_ring().one(),r[0].parent().base_ring().one(),True)

def __monicPRS__(r,additional):
    """
    Computes one division step in the monic polynomial remainder sequence.
    """

    newRem = r[0].quo_rem(r[1])
    beta = newRem[1].leading_coefficient() if not newRem[1].is_zero() else r[0].parent().base_ring().one()
    return ((r[1],newRem[1].primitive_part()),newRem[0],r[0].parent().base_ring().one(),beta,True)

#def __essentialPRS__(r,additional):
#    """
#    Computes one division step in the improved polynomial remainder sequence.
#    """

#    d1 = r[0].order()
#    d0 = r[1].order()
#    orddiff = d1-d0

#    R = r[0].parent()
#    Rbase = R.base_ring()
#    sigma = R.sigma()

#    if (len(additional)==0):
#        essentialPart = gcd(sigma(r[0].leading_coefficient(),-orddiff),r[1].leading_coefficient())
#        phi = Rbase.one()
#        beta = (-Rbase.one())**(orddiff+1)*sigma.factorial(sigma(phi,1),orddiff)
#    else:
#        (d2,oldalpha,k,essentialPart,phi) = (additional.pop(),additional.pop(),additional.pop(),additional.pop(),additional.pop())
#        phi = oldalpha / sigma.factorial(sigma(phi,1),d2-d0-1)
#        beta = oldalpha.parent()(((-Rbase.one())**(orddiff+1)*sigma.factorial(sigma(phi,1),orddiff)*k))
#        essentialPart = sigma(essentialPart,-orddiff)

#    k = r[1].leading_coefficient()//essentialPart
#    if k==0:
#        return ((0,0),0,0,0,False)
#    alpha = sigma.factorial(k,orddiff)
#    alpha2=alpha*sigma(k,orddiff)
#    newRem = (alpha2*r[0]).quo_rem(r[1],fractionFree=True)
#    r2 = newRem[1].map_coefficients(lambda p: p//beta)
#    if r2.parent() is not r[1].parent():
#        return ((0,0),0,0,0,False)
#    additional.extend([phi,essentialPart,k,alpha,d1])

#    return ((r[1],r2),newRem[0],alpha2,beta,True)

def __essentialPRS__(r,additional):
    """
    Computes one division step in the essential polynomial remainder sequence.
    """

    d1 = r[0].order()
    d0 = r[1].order()
    orddiff = d1-d0

    R = r[0].parent()
    Rbase = R.base_ring()
    sigma = R.sigma()

    if (len(additional)==0):
        phi = -Rbase.one()
        initD=d0+d1
        essentialPart = sigma(gcd(sigma(r[0].leading_coefficient(),-orddiff),r[1].leading_coefficient()),-d0)
        gamma1 = 1
        gamma2 = sigma.factorial(sigma(essentialPart,d0),orddiff+1)
        beta = (-Rbase.one())*sigma.factorial(sigma(phi,1),orddiff)*gamma2
    else:
        (initD,essentialPart,gamma0,gamma1,d2,phi) = (additional.pop(),additional.pop(),additional.pop(),additional.pop(),additional.pop(),additional.pop())
        orddiff2 = d2-d1
        gamma2 = sigma.factorial(sigma(essentialPart,d1),orddiff2)*gamma1*sigma.factorial(sigma(essentialPart,initD-d0+1),orddiff2)
        phi = sigma.factorial(-gamma0*r[0].leading_coefficient(),orddiff2) / sigma.factorial(sigma(phi,1),orddiff2-1)
        beta = (-Rbase.one())*sigma.factorial(sigma(phi,1),orddiff)*r[0].leading_coefficient()*gamma2/sigma.factorial(gamma1,orddiff+1)

    alpha = sigma.factorial(r[1].leading_coefficient(),orddiff+1)
    newRem = (alpha*r[0]).quo_rem(r[1],fractionFree=True)
    try:
        r2 = newRem[1].map_coefficients(lambda p: p/beta)
    except:
        return ((0,0),0,0,0,False)
    additional.extend([phi,d1,gamma2,gamma1,essentialPart,initD])

    return ((r[1],r2),newRem[0],alpha,beta,True)

def __subresultantPRS__(r,additional):
    """
    Computes one division step in the subresultant polynomial remainder sequence.
    """

    d1 = r[0].order()
    d0 = r[1].order()
    orddiff = d1-d0

    R = r[0].parent()
    Rbase = R.base_ring()
    sigma = R.sigma()

    if (len(additional)==0):
        phi = -Rbase.one()
        beta = (-Rbase.one())*sigma.factorial(sigma(phi,1),orddiff)
    else:
        (d2,phi) = (additional.pop(),additional.pop())
        orddiff2 = d2-d1
        phi = sigma.factorial(-r[0].leading_coefficient(),orddiff2) / sigma.factorial(sigma(phi,1),orddiff2-1)
        beta = (-Rbase.one())*sigma.factorial(sigma(phi,1),orddiff)*r[0].leading_coefficient()

    alpha = sigma.factorial(r[1].leading_coefficient(),orddiff+1)
    newRem = (alpha*r[0]).quo_rem(r[1],fractionFree=True)
    r2 = newRem[1].map_coefficients(lambda p: p//beta)
    additional.extend([phi,d1])

    return ((r[1],r2),newRem[0],alpha,beta,True)
