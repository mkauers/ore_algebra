 
"""
ore_operator
============

"""

from sage.structure.element import RingElement, canonical_coercion
from sage.rings.ring import Algebra
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing
from sage.rings.number_field.number_field import is_NumberField
from sage.rings.fraction_field import is_FractionField
from sage.rings.arith import gcd, lcm
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ

class OreOperator(RingElement):
    """
    An Ore operator. This is an abstract class whose instances represent elements of ``OreAlgebra``.

    In addition to usual ``RingElement`` features, Ore operators provide coefficient extraction
    functionality and the possibility of letting an operator act on another object. The latter
    is provided through ``call``.

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
           sage: (Sx - 1)(1/4*x*(x-1)*(x-2)*(x-3), Sx=lambda p:p(2*x)) # let Sx act as q-shift
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

          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: (Dx^3 + (5*x+3)*Dx + (71*x+1)).is_monic()
          True
          sage: ((5*x+3)*Dx^2 + (71*x+1)).is_monic()
          False 
        
        """
        raise NotImplementedError

    def is_unit(self):
        """
        Return True if this operator is a unit.

        EXAMPLES::

          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: A(x).is_unit()
          False
          sage: A.<Dx> = OreAlgebra(R.fraction_field(), 'Dx')
          sage: A(x).is_unit()
          True
          
        """
        raise NotImplementedError
       
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
        return infinity.infinity
    
    # conversion
        
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

    def __iter__(self):
        return iter(self.list())

    def __float__(self):
        return NotImplementedError

    def __int__(self):
        return NotImplementedError

    def _integer_(self, ZZ):
        return NotImplementedError

    def _rational_(self):
        return NotImplementedError

    def _symbolic_(self, R):
        raise NotImplementedError

    def __long__(self):
        raise NotImplementedError

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

    def __invert__(self):
        """
        This returns ``1/self``, an object which is meaningful only if ``self`` can be coerced
        to the base ring of its parent, and admits a multiplicative inverse, possibly in a
        suitably extended ring.

        EXAMPLES::

           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: A
           Univariate Ore algebra in Dx over Univariate Polynomial Ring in x over Rational Field
           sage: ~A(x)
           1/x
           sage: _.parent()
           Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field
        
        """
        return self.parent().one()/self

    def __div__(self, right):
        """
        Exact right division. Uses division with remainder, and returns the quotient if the
        remainder is zero. Otherwise a ``ValueError`` is raised.

        EXAMPLES::

           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: U = (15*x^2 + 28*x + 5)*Dx^2 + (5*x^2 - 50*x - 41)*Dx - 2*x + 64
           sage: V = (3*x+5)*Dx + (x-9)
           sage: U/V
           (5*x + 1)*Dx - 7
           sage: _*V == U
           True
        
        """
        Q, R = self.quo_rem(right)
        if R == R.parent().zero():
            return Q
        else:
            raise ValueError, "Cannot divide the given OreOperators"
                   
    def __floordiv__(self,right):
        """
        Quotient of quotient with remainder.

        EXAMPLES::

           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: U = (15*x^2 + 29*x + 5)*Dx^2 + (5*x^2 - 50*x - 41)*Dx - 2*x + 64
           sage: V = (3*x+5)*Dx + (x-9)
           sage: U//V
           ((15*x^2 + 29*x + 5)/(3*x + 5))*Dx + (-64*x^2 - 204*x - 175)/(9*x^2 + 30*x + 25)
        
        """
        Q, _ = self.quo_rem(right)
        return Q
        
    def __mod__(self, other):
        """
        Remainder of quotient with remainder.

        EXAMPLES::

           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: U = (15*x^2 + 29*x + 5)*Dx^2 + (5*x^2 - 50*x - 41)*Dx - 2*x + 64
           sage: V = (3*x+5)*Dx + (x-9)
           sage: U % V
           (3*x^3 - 54*x^2 + 147*x)/(27*x^2 + 90*x + 75)
        
        """
        _, R = self.quo_rem(other)
        return R

    def quo_rem(self, other):
        """
        Quotient and remainder.

        EXAMPLES::

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

           sage: OreAlgebra(QQ['x'], 'Dx').random_element().base_ring()
           Univariate Polynomial Ring in x over Rational Field
        
        """
        return self.parent().base_ring()

    def base_extend(self, R):
        """
        Return a copy of this operator but with coefficients in R, if
        there is a natural map from coefficient ring of self to R.

        EXAMPLES::

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
        raise IndexError, "Operators are immutable"

    def is_primitive(self, n=None, n_prime_divs=None):
        """
        Returns ``True`` if this operator's content is a unit of the base ring. 
        """
        return self.content().is_unit()

    def is_monomial(self):
        """
        Returns True if self is a monomial, i.e., a power of the generator.
        """
        return len(self.exponents()) == 1 and self.leading_coefficient() == self.parent().base_ring().one()

    def leading_coefficient(self):
        """
        Return the leading coefficient of this operator. 
        """
        raise NotImplementedError

    def constant_coefficient(self):
        """
        Return the leading coefficient of this operator. 
        """
        raise NotImplementedError

    def monic(self):
        """
        Return this operator divided from the left by its leading coefficient.
        Does not change this operator. If the leading coefficient does not have
        a multiplicative inverse in the base ring of ``self``'s parent, the
        the method returns an element of a suitably extended algebra.

        EXAMPLES::

          sage: R.<x> = QQ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: (x*Dx + 1).monic()
          Dx + 1/x
          sage: _.parent()
          Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field
        
        """
        if self.is_monic():
            return self
        a = ~self.leading_coefficient()
        A = self.parent()
        if a.parent() != A.base_ring():
            S = A.base_extend(a.parent())
            return a*S(self)
        else:
            return a*self

    def content(self, proof=True):
        """
        Returns the content of ``self``.

        If the base ring of ``self``'s parent is a field, the method returns the base ring's one.

        If the base ring is not a field, then it is a polynomial ring. In this case,
        the method returns the greatest common divisor of the nonzero coefficients of
        ``self``.

        EXAMPLES::

           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (5*x^2*Dx + 10*x).content()
           5*x
           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (5*x^2*Dx + 10*x).content()
           x
           sage: R.<x> = QQ['x']
           sage: A.<Dx> = OreAlgebra(R.fraction_field(), 'Dx')
           sage: (5*x^2*Dx + 10*x).content()
           1
        
        """
        if self == 0 or self.is_zero():
            return self.parent().base_ring().one()
        if self.order() == 0:
            return self.constant_coefficient()

        Rbase = self.parent().base_ring()
        coeffs = self.coefficients()

        if proof:
            cont = lambda x: gcd([x(c) for c in coeffs])
        else:
            cont = lambda x: gcd(x(coeffs.pop()),reduce(lambda y,z: x(y)+x(z),coeffs))

        if Rbase.is_field():
            try:
                return Rbase(cont(Rbase.base()))
            except:
                pass
            return Rbase.one()
        else:
            return cont(Rbase)

    def primitive_part(self):
        """
        Returns the primitive part of ``self``.

        It is obtained by dividing ``self`` from the left by ``self.content()``.

        EXAMPLES::

          sage: R.<x> = ZZ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: (5*x^2*Dx + 10*x).primitive_part()
          x*Dx + 2
        
        """
        if self.is_zero(): return self
        if self.parent().base_ring().is_field(): c = self.leading_coefficient()
        else: c = self.content()
        return self.map_coefficients(lambda p: p//c)


    def normalize(self):
        """
        Returns a normal form of an Ore operator.
        First, it takes the primitve part of the operator and then tries to normalize it by making it monic.
        If the leading coefficient is a polynomial, it tries to make its leading coefficient monic, unless this is a polynomial again.
        In the latter case, the process is repeated until the leading coefficient is a unit.
        """
        prim = self.primitive_part()
        c = prim.leading_coefficient()
        while (not c.is_unit()) and (c.parent()!=c.parent().base_ring()):
            c = c.leading_coefficient()
        if not c.is_unit():
            return prim
        return (~c)*prim

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
            return lcm([c.denominator() for c in self.coefficients()])


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

        if kwds.has_key("action"):
            D = kwds["action"]
        else:
            D = lambda p:p

        R = f.parent(); Dif = f; result = R(self[0])*f; 
        for i in xrange(1, self.order() + 1):
            Dif = D(Dif)
            result += R(self[i])*Dif
        
        return result

    # tests

    def __nonzero__(self):
        return self._poly.__nonzero__()

    def __eq__(self, other):

        if self.order() == 0:
            return self.constant_coefficient() == other
        elif not isinstance(other, OreOperator):
            return False
        elif self.parent() == other.parent():
            return self.polynomial() == other.polynomial()
        else:
            try:
                A, B = canonical_coercion(self, other)
                return A == B
            except:
                return False

    def _is_atomic(self):
        return self._poly._is_atomic()

    def is_monic(self):
        return self._poly.is_monic()

    def is_unit(self):
        return self._poly.is_unit()
       
    def is_gen(self):
        return self._poly.is_gen()
    
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
        raise NotImplementedError

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

        coeffs = self.coeffs()
        DiB = right.polynomial() # D^i * B, for i=0,1,2,...

        R = self.parent() # Ore algebra
        sigma = R.sigma(); delta = R.delta()
        A = DiB.parent() # associate commutative algebra
        D = A.gen() 
        res = coeffs[0]*DiB

        for i in xrange(1, len(coeffs)):

            DiB = DiB.map_coefficients(sigma)*D + DiB.map_coefficients(delta)
            res += coeffs[i]*DiB

        return R(res)

    def quo_rem(self, other, fractionFree=False):

        if other.is_zero(): 
            raise ZeroDivisionError, "other must be nonzero"

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
            cfquo = op(p.leading_coefficient(),qlcs[orddiff]) * D**(orddiff)
            quo = quo+cfquo
            p = p - cfquo*q
            orddiff = p.order() - q.order()
        return (quo,p)

    def gcrd(self, *other, **kwargs):
        """
        Returns the GCRD of self and other. 
        It is possible to specify which remainder sequence should be used.
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
        elif self.order() == 1 or other in self.base_ring():
            return self.parent().one()
        elif self.parent() is not other.parent():
            A, B = canonical_coercion(self, other)
            return A.gcrd(B)

        prs = kwargs["prs"] if kwargs.has_key("prs") else None

        r = (self,other)
        if (r[0].order()<r[1].order()):
            r=(other,self)

        if prs==None:
            if R.base_ring().is_field():
                prs = __classicPRS__
            else:
                prs = __improvedPRS__

        additional = []
        while not r[1].is_zero(): 
            r=prs(r,additional)[0]
        r=r[0]

        if not prs==__classicPRS__:
            r = r.primitive_part()

        return r

    
    def xgcrd(self, other,prs=None):
        """
        When called for two operators p,q, this will return their GCRD g together with 
        two operators s and t such that sp+tq=g. 
        It is possible to specify which remainder sequence should be used.
        """

        if self.is_zero(): return other
        if other.is_zero(): return self

        r = (self,other)
        if (r[0].order()<r[1].order()):
            r=(other,self)
        
        R = r[0].parent()
        RF = R.change_ring(R.base_ring().fraction_field())

        a11,a12,a21,a22 = RF.one(),RF.zero(),RF.zero(),RF.one()

        if prs==None:
            if R.base_ring().is_field():
                prs = __classicPRS__
            else:
                prs = __improvedPRS__

        additional = []

        while not r[1].is_zero():  
            (r,q,alpha,beta)=prs(r,additional)
            bInv = ~beta
            a11,a12,a21,a22 = a21,a22,bInv*(alpha*a11-q*a21),bInv*(alpha*a12-q*a22)

        r=r[0]

        if not prs==__classicPRS__:
            r = r.primitive_part()

        return (r,a11,a12)

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

        Through the optional argument ``solver``, a callable object can be
        provided which the function should use for computing the kernel of
        matrices with entries in the Ore algebra's base ring. 

        EXAMPLES::

            sage: R.<x> = ZZ['x']
            sage: Alg.<Dx> = OreAlgebra(R, 'Dx')
            sage: A = 5*(x+1)*Dx + (x - 7); B = (3*x+5)*Dx - (8*x+1)
            sage: L = A.lclm(B)
            (-645*x^4 - 2155*x^3 - 1785*x^2 + 475*x + 750)*Dx^2 + (1591*x^4 + 3696*x^3 + 3664*x^2 + 2380*x + 725)*Dx + 344*x^4 - 2133*x^3 - 2911*x^2 - 1383*x - 1285
            sage: A*B
            (15*x^2 + 40*x + 25)*Dx^2 + (-37*x^2 - 46*x - 25)*Dx - 8*x^2 + 15*x - 33
            sage: B.lclm(A*B)
            (-15*x^2 - 40*x - 25)*Dx^2 + (37*x^2 + 46*x + 25)*Dx + 8*x^2 - 15*x + 33
            sage: B.lclm(L, A*B)
            (15*x^2 + 40*x + 25)*Dx^2 + (-37*x^2 - 46*x - 25)*Dx - 8*x^2 + 15*x - 33
        
        """

        if len(other) != 1:
            return reduce(lambda p, q: p.lclm(q), other, self)
        elif len(other) == 0:
            return self

        other = other[0]
        if self.is_zero() or other.is_zero():
            return self.parent().zero()
        elif self.order() == 1:
            return other
        elif other in self.base_ring():
            return self
        elif self.parent() is not other.parent():
            A, B = canonical_coercion(self, other)
            return A.lclm(B)

        solver = kwargs["solver"] if kwargs.has_key("solver") else None
        
        if not isinstance(other, UnivariateOreOperator):
            raise TypeError, "unexpected argument in lclm"

        if self.parent() != other.parent():
            A, B = canonical_coercion(self, other)
            return A.lclm(B)

        A = self.numerator(); r = A.order()
        B = other.numerator(); s = B.order()
        D = self.parent().gen()

        t = max(r, s) # expected order of the lclm

        rowsA = [A]
        for i in xrange(t - r):
            rowsA.append(D*rowsA[-1])
        rowsB = [B]
        for i in xrange(t - s):
            rowsB.append(D*rowsB[-1])

        from sage.matrix.constructor import Matrix
        if solver == None:
            solver = A.parent()._solver()

        sys = Matrix(map(lambda p: p.coeffs(padd=t), rowsA + rowsB)).transpose()
        sol = solver(sys)

        while len(sol) == 0:
            t += 1
            rowsA.append(D*rowsA[-1]); rowsB.append(D*rowsB[-1])
            sys = Matrix(map(lambda p: p.coeffs(padd=t), rowsA + rowsB)).transpose()
            sol = solver(sys)

        U = A.parent()(list(sol[0])[:t+1-r])
        return self.parent()(U*A)

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

    def symmetric_product(self, other, solver=None):
        """
        Returns the symmetric product of ``self`` and ``other``.

        The symmetric product of two operators `A` and `B` is a minimal order
        operator `C` such that for all \"functions\" `f` and `g` with `A.f=B.g=0`
        we have `C.(fg)=0`.

        The function requires that a product rule is associated to the ore algebra
        where ``self`` and ``other`` live. (See docstring of OreAlgebra for information
        about product rules.)

        If no ``solver`` is specified, the the Ore algebra's solver is used.         

        EXAMPLES::

           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (Dx - 1).symmetric_product(x*Dx - 1)
           x*Dx - x - 1
           sage: (x*Dx - 1).symmetric_product(Dx - 1)
           x*Dx - x - 1
           sage: ((x+1)*Dx^2 + (x-1)*Dx + 8).symmetric_product((x-1)*Dx^2 + (2*x+3)*Dx + (8*x+5))
           (-29*x^8 + 4*x^7 + 55*x^6 + 34*x^5 + 23*x^4 - 80*x^3 - 95*x^2 + 42*x + 46)*Dx^4 + (-174*x^8 - 150*x^7 - 48*x^6 + 294*x^5 + 864*x^4 + 646*x^3 - 232*x^2 - 790*x - 410)*Dx^3 + (-783*x^8 - 1661*x^7 + 181*x^6 + 1783*x^5 + 3161*x^4 + 3713*x^3 - 213*x^2 - 107*x + 1126)*Dx^2 + (-1566*x^8 - 5091*x^7 - 2394*x^6 - 2911*x^5 + 10586*x^4 + 23587*x^3 + 18334*x^2 + 2047*x - 5152)*Dx - 2552*x^8 - 3795*x^7 - 8341*x^6 - 295*x^5 + 6394*x^4 + 24831*x^3 + 35327*x^2 + 23667*x + 13708
           sage: A.<Sx> = OreAlgebra(R, 'Sx')
           sage: (Sx - 2).symmetric_product(x*Sx - (x+1))
           x*Sx - 2*x - 2
           sage: (x*Sx - (x+1)).symmetric_product(Sx - 2)
           x*Sx - 2*x - 2
           sage: ((x+1)*Sx^2 + (x-1)*Sx + 8).symmetric_product((x-1)*Sx^2 + (2*x+3)*Sx + (8*x+5))
           (8*x^8 + 13*x^7 - 300*x^6 - 1640*x^5 - 3698*x^4 - 4373*x^3 - 2730*x^2 - 720*x)*Sx^4 + (-16*x^8 - 34*x^7 + 483*x^6 + 1947*x^5 + 2299*x^4 + 2055*x^3 + 4994*x^2 + 4592*x)*Sx^3 + (64*x^8 - 816*x^7 - 1855*x^6 + 21135*x^5 + 76919*x^4 + 35377*x^3 - 179208*x^2 - 283136*x - 125440)*Sx^2 + (-1024*x^7 - 1792*x^6 + 39792*x^5 + 250472*x^4 + 578320*x^3 + 446424*x^2 - 206528*x - 326144)*Sx + 32768*x^6 + 61440*x^5 - 956928*x^4 - 4897984*x^3 - 9390784*x^2 - 7923200*x - 2329600
        
        """
        if not isinstance(other, UnivariateOreOperator):
            raise TypeError, "unexpected argument in symmetric_product"

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
            return A.one()
        elif a == 1 and b > 1:
            A, B, a, b = B, A, b, a

        pr = Alg._product_rule()
        if pr is None:
            raise ValueError, "no product rule found"

        if b == 1:
            
            D = A.parent().gen(); D1 = D(R.one())
            h = -B[0]/B[1] # B = D - h
            if h == D1:
                return A            

            # define g such that (D - h)(u) == 0 iff (D - g)(1/u) == 0.
            g = (D1 - pr[0] - pr[1]*h)/(pr[1] + pr[2]*h)
            
            # define p, q such that "D*1/u == p*1/u*D + q*1/u" 
            p = (g - D1)/(D1 - h); q = g - p*D1
            sigma_u = lambda c: sigma(c)*p
            delta_u = lambda c: delta(c) + q

            # calculate L with L(u*v)=0 iff A(v)=0 and B(u)=0 using A(1/u * u*v) = 0
            coeffs = A.coeffs(); L = coeffs[0]; Dk = A.parent().one()
            for i in xrange(1, A.order() + 1):
                Dk = Dk.map_coefficients(sigma_u)*D + Dk.map_coefficients(delta_u)
                c = coeffs[i]
                if not c.is_zero():
                    L += c*Dk
            
            return A.parent()(L)

        # general case via linear algebra

        Ared = tuple(-A[i]/A[a] for i in xrange(a)); Bred = tuple(-B[j]/B[b] for j in xrange(b))

        if solver is None:
            solver = Alg._solver()

        # Dkuv[i][j] is the coefficient of D^i(u)*D^j(v) in the normal form of D^k(u*v) 
        Dkuv = [[zero for i in xrange(b + 1)] for j in xrange(a + 1)]; Dkuv[0][0] = one
        
        mat = [[Dkuv[i][j] for i in xrange(a) for j in xrange(b)]]

        from sage.matrix.constructor import Matrix
        sol = solver(Matrix(mat).transpose())

        while len(sol) == 0:

            # push
            for i in xrange(a - 1, -1, -1):
                for j in xrange(b - 1, -1, -1):
                    s = sigma(Dkuv[i][j])
                    Dkuv[i + 1][j + 1] += s*pr[2]
                    Dkuv[i][j + 1] += s*pr[1]
                    Dkuv[i + 1][j] += s*pr[1]
                    Dkuv[i][j] = delta(Dkuv[i][j]) + s*pr[0]

            # reduce
            for i in xrange(a + 1):
                if not Dkuv[i][b] == zero:
                    for j in xrange(b):
                        Dkuv[i][j] += Bred[j]*Dkuv[i][b]
                    Dkuv[i][b] = zero

            for j in xrange(b): # not b + 1
                if not Dkuv[a][j] == zero:
                    for i in xrange(a):
                        Dkuv[i][j] += Ared[i]*Dkuv[a][j]
                    Dkuv[a][j] = zero

            # solve
            mat.append([Dkuv[i][j] for i in xrange(a) for j in xrange(b)])
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
            raise TypeError, "unexpected exponent received in symmetric_power"
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
            raise TypeError, "unexpected exponent received in symmetric_power"

    def annihilator_of_associate(self, other, solver=None):
        """
        Computes an operator `L` with `L(other(f))=0` for all `f` with `self(f)=0`.

        EXAMPLES::

           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (Dx^2 + x*Dx + 5).annihilator_of_associate(Dx + 7*x+3)
           (-42*x^2 - 39*x - 7)*Dx^2 + (-42*x^3 - 39*x^2 + 77*x + 39)*Dx - 168*x^2 - 174*x - 61
           sage: A.<Sx> = OreAlgebra(R, 'Sx')
           (-42*x^2 - 88*x - 35)*Sx^2 + (-42*x^3 - 130*x^2 - 53*x + 65)*Sx - 210*x^2 - 860*x - 825

        """
        if not isinstance(other, UnivariateOreOperator):
            raise TypeError, "unexpected argument in symmetric_product"

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

        mat = [B.coeffs(padd=a-1)]

        from sage.matrix.constructor import Matrix
        sol = solver(Matrix(mat).transpose())

        while len(sol) == 0:
            B = (D*B) % A
            mat.append(B.coeffs(padd=a-1))
            sol = solver(Matrix(mat).transpose())

        L = A.parent()(list(sol[0]))
        return L

    # coefficient-related functions

    def companion_matrix(self):
        r"""
        If ``self`` is an operator of order `r`, returns an `r` by `r` matrix
        `M` such that for any sequence `c_i` annihilated by ``self``,
        `[c_{i+1}, c_{i+2}, \ldots, c_{i+r}]^T = M(i) [c_i, c_{i+1}, \ldots, c_{i+r-1}]^T`

        EXAMPLES::

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
            return infinity.infinity
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

    def coeffs(self, padd=-1):
        """
        Return the coefficient vector of this operator.

        If the degree is less than the number given in the optional
        argument, the list is padded with zeros so as to ensure that
        the output has length ``padd`` + 1.

        EXAMPLES::

           sage: A.<Sx> = OreAlgebra(ZZ['x'], 'Sx')
           sage: (5*Sx^3-4).coeffs()
           [-4, 0, 0, 5]
           sage: (5*Sx^3-4).coeffs(padd=5)
           [-4, 0, 0, 5, 0, 0]
           sage: (5*Sx^3-4).coeffs(padd=1)
           [-4, 0, 0, 5]
        
        """
        c = self.polynomial().coeffs()
        if len(c) <= padd:
            z = self.base_ring().zero()
            c = c + [z for i in xrange(padd + 1 - len(c))]
        return c

    def coefficients(self):
        return self.polynomial().coefficients()

    def exponents(self):
        return self.polynomial().exponents()


#############################################################################################################

class UnivariateOreOperatorOverUnivariateRing(UnivariateOreOperator):
    """
    Element of an Ore algebra with a single generator and a commutative rational function field as base ring.     
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperator, self).__init__(parent, *data, **kwargs)

    def degree(self):
        """
        maximum coefficient degree
        """
        raise NotImplementedError

    def polynomial_solutions(self):
        raise NotImplementedError

    def rational_solutions(self):
        raise NotImplementedError

    def desingularize(self, p):
        raise NotImplementedError

    def indicial_polynomial(self, p, var='lambda'):
        raise NotImplementedError

    def abramov_van_hoeij(self, other):
        """
        given other=a*D + b, find, if possible, an operator M such that rat*self = 1 - other*M
        for some rational function rat.
        """
        raise NotImplementedError


#############################################################################################################

class UnivariateDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[D], where D acts as derivation d/dx on K(x).
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):
        
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : p.derivative()

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_S(self, alg): # d2s
        """
        Returns a recurrence operator annihilating the coefficient sequence of
        every power series (about the origin) annihilated by ``self``.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_S()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the standard shift with respect to ``self.base_ring().gen()``.

        EXAMPLES::

            sage: R.<x> = ZZ['x']
            sage: A.<Dx> = OreAlgebra(R, 'Dx')
            sage: R2.<n> = ZZ['n']
            sage: A2.<Sn> = OreAlgebra(R2, 'Sn')
            sage: (Dx - 1).to_S(A2)
            (n + 1)*Sn - 1
            sage: ((1+x)*Dx^2 + Dx).to_S(A2)
            (n^2 + n)*Sn + n^2
            sage: ((x^3+x^2-x)*Dx + (x^2+1)).to_S(A2)
            (-n - 1)*Sn^2 + (n + 1)*Sn + n + 1

        """
        if type(alg) == str:
            R = self.base_ring(); x = R.gen(); one = R.one()
            rec_algebra = self.parent().change_var_sigma_delta(alg, {x:x+one}, {})
        elif not isinstance(alg, type(self.parent())) or not alg.is_S() \
             or alg.base_ring().base_ring() is not self.base_ring().base_ring():
            raise TypeError, "not an adequate algebra"
        else:
            rec_algebra = alg
        
        numer = self.numerator()
        coeffs = [list(c) for c in list(numer)]
        lengths = [len(c) for c in coeffs]

        r = len(coeffs) - 1
        d = max(lengths) - 1
        start = d + 1
        for k in range(r + 1):
            start = min(start, d - (lengths[k] - 1) + k)

        roots = [0] * r
        result = [[] for i in range(d + r + 1 - start)]

        def set_coeff(lst, i, x):
            while i >= len(lst):
                lst.append(0)
            lst[i] = x
            while lst and not lst[-1]:
                lst.pop()

        def from_newton_basis(coeffs, roots):
            n = len(coeffs)
            for i in range(n - 1, 0, -1):
                for j in range(i - 1, n - 1):
                    coeffs[j] -= coeffs[j + 1] * roots[i - 1]

        for k in range(start, d + r + 1):
            i = k - start
            result[i] = []

            for j in range(r + 1):
                v = d + j - k
                if v >= 0 and v < lengths[j]:
                    set_coeff(result[i], j, coeffs[j][v])

            if result[i]:
                from_newton_basis(result[i], range(-i, -i + r))

        return rec_algebra(result)

    def to_F(self, alg):
        """
        Returns a difference operator annihilating the coefficient sequence of
        every power series (about the origin) annihilated by ``self``.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_F()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the forward difference with respect to ``self.base_ring().gen()``.

        EXAMPLES::

            sage: R.<x> = ZZ['x']
            sage: A.<Dx> = OreAlgebra(R, 'Dx')
            sage: R2.<n> = ZZ['n']
            sage: A2.<Sn> = OreAlgebra(R2, 'Fn')
            sage: (Dx - 1).to_F(A2)
            (n + 1)*Fn + n
            sage: ((1+x)*Dx^2 + Dx).to_F(A2)
            (n^2 + n)*Fn + 2*n^2 + n
            sage: ((x^3+x^2-x)*Dx + (x^2+1)).to_F(A2)
            (-n - 1)*Fn^2 + (-n - 1)*Fn + n + 1

        """
        return self.to_S('S').to_F(alg)

    def to_T(self, alg): # d2theta
        """
        Rewrites ``self`` in terms of the eulerian derivation `x*d/dx`.

        If the base ring of the target algebra is not a field, the operator returned by the 
        method may not correspond exactly to ``self``, but only to a suitable left-multiple
        by a term `x^k`.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_T()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          an euler derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: R.<x> = ZZ['x']
          sage: R2.<y> = ZZ['y']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: (Dx^4).to_T(OreAlgebra(R2, 'Ty'))
          Ty^4 - 6*Ty^3 + 11*Ty^2 - 6*Ty
          sage: (Dx^4).to_T('Tx').to_D(A)
          x^4*Dx^4
          sage: _.to_T('Tx')
          Tx^4 - 6*Tx^3 + 11*Tx^2 - 6*Tx
        
        """
        R = self.base_ring(); one = R.one(); x = R.gen()
        
        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {}, {x:x})
        elif not isinstance(alg, type(self.parent())) or not alg.is_T() or \
             alg.base_ring().base_ring() is not R.base_ring():
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); alg2 = alg.change_ring(R); x = R.gen()

        theta = (1/x)*alg2.gen(); theta_k = alg2.one();
        c = self.coeffs(); out = alg2(R(c[0]))

        for i in xrange(self.order()):
            
            theta_k *= theta
            out += R(c[i + 1])*theta_k

        return out if alg.base_ring() is R else out.numerator()

    def annihilator_of_integral(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite integrals `\int f`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order. 

        EXAMPLES::

           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: ((x-1)*Dx - 2*x).annihilator_of_integral()
           (x-1)*Dx^2 - 2*x*Dx
           sage: _.annihilator_of_associate(Dx)
           (x-1)*Dx - 2*x
           
        """
        return self*self.parent().gen()

    def annihilator_of_composition(self, a, solver=None):
        """
        Returns an operator `L` which annihilates all the functions `f(a(x))`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order.

        INPUT:

        - ``a`` -- either an element of the base ring of the parent of ``self``,
          or an element of an algebraic extension of this ring.
        - ``solver`` (optional) -- a callable object which applied to a matrix
          with polynomial entries returns its kernel. 

        EXAMPLES::

           sage: R.<x> = ZZ['x']
           sage: K.<y> = R.fraction_field()['y']
           sage: K.<y> = R.fraction_field().extension(y^3 - x^2*(x+1))
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (x*Dx-1).annihilator_of_composition(y) # ann for x^(2/3)*(x+1)^(1/3)
           (3*x^2 + 3*x)*Dx - 3*x - 2
           sage: (x*Dx-1).annihilator_of_composition(y + 2*x) # ann for 2*x + x^(2/3)*(x+1)^(1/3)
           (-3*x^3 - 3*x^2)*Dx^2 + 2*x*Dx - 2
           sage: (Dx - 1).annihilator_of_composition(y) # ann for exp(x^(2/3)*(x+1)^(1/3))
           (243*x^6 + 810*x^5 + 999*x^4 + 540*x^3 + 108*x^2)*Dx^3 + (162*x^3 + 270*x^2 + 108*x)*Dx^2 + (-162*x^2 - 180*x - 12)*Dx - 243*x^6 - 810*x^5 - 1080*x^4 - 720*x^3 - 240*x^2 - 32*x
        
        """

        A = self.parent(); K = A.base_ring().fraction_field(); R = K['Y']
        if solver == None:
            solver = A._solver(K)

        if self == A.one():
            return self
        elif a in K:
            minpoly = R.gen() - K(a)
        else:
            try:
                minpoly = R(a.minpoly()).monic()
            except:
                raise TypeError, "argument not recognized as algebraic function over base ring"

        d = minpoly.degree(); r = self.order()

        # derivative of a
        Da = -minpoly.map_coefficients(lambda p: p.derivative())
        Da *= minpoly.xgcd(minpoly.derivative())[2]
        Da = Da % minpoly

        # self's coefficients with x replaced by a, denominators cleared, and reduced by minpoly.
        # have: (D^r f)(a) == sum( red[i]*(D^i f)a, i=0..len(red)-1 ) and each red[i] is a poly in Y of deg <= d.
        red = [ R(p.numerator().coeffs()) for p in self.numerator().change_ring(K).coeffs() ]
        lc = -minpoly.xgcd(red[-1])[2]
        red = [ (red[i]*lc) % minpoly for i in xrange(r) ]

        from sage.matrix.constructor import Matrix
        Dkfa = [R.zero() for i in xrange(r)] # Dkfa[i] == coeff of (D^i f)(a) in D^k (f(a))
        Dkfa[0] = R.one()
        mat = [[ q for p in Dkfa for q in p.padded_list(d) ]]; sol = []

        while len(sol) == 0:

            # compute coeffs of (k+1)th derivative
            next = [ (p.map_coefficients(lambda q: q.derivative()) + p.derivative()*Da) % minpoly for p in Dkfa ]
            for i in xrange(r - 1):
                next[i + 1] += (Dkfa[i]*Da) % minpoly
            for i in xrange(r):
                next[i] += (Dkfa[-1]*red[i]*Da) % minpoly
            Dkfa = next

            # check for linear relations
            mat.append([ q for p in Dkfa for q in p.padded_list(d) ])
            sol = solver(Matrix(K, mat).transpose())

        return self.parent()(list(sol[0]))

    def power_series_solutions(self, n):
        raise NotImplementedError

    def generalized_series_solutions(self, n):
        raise NotImplementedError

    def get_value(self, init, z, n):
        """
        If K is a subfield of CC, this computes an approximation of the solution of this operator
        wrt the given initial values at the point z to precision n.
        """
        raise NotImplementedError


#############################################################################################################

class UnivariateRecurrenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[S], where S is the shift x->x+1.
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):
        
        if type(f) in (tuple, list):

            r = self.order()
            R = self.parent().base_ring()
            K = R.base_ring()
            z = K.zero()
            c = self.numerator().coeffs()
            d = self.denominator()

            def fun(n):
                if f[n + r] is None:
                    return None
                else:
                    try:
                        return sum( c[i](n)*f[n + i] for i in xrange(r + 1) )/d(n)
                    except:
                        return None

            return type(f)(fun(n) for n in xrange(len(f) - r))

        x = self.parent().base_ring().gen()
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : p(x+1)

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_D(self, alg): # s2d
        """
        Returns a differential operator which annihilates every power series whose
        coefficient sequence is annihilated by ``self``.
        The output operator may not be minimal. 

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_D()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the standard derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
          sage: A.<Sn> = OreAlgebra(Rn, 'Sn')
          sage: B.<Dx> = OreAlgebra(Rx, 'Dx')
          sage: (Sn - 1).to_D(B)
          (-x + 1)*Dx - 1
          sage: ((n+1)*Sn - 1).to_D(B)
          x*Dx^2 + (-x + 1)*Dx - 1
          sage: (x*Dx-1).to_S(A).to_D(B)
          x*Dx - 1
        
        """
        R = self.base_ring(); x = R.gen(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {}, {x:one})
        elif not isinstance(alg, type(self.parent())) or not alg.is_D() \
             or alg.base_ring().base_ring() is not R.base_ring():
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); x = R.gen()
        alg_theta = alg.change_var_sigma_delta('T', {}, {x:x}).change_ring(R)

        S = alg_theta(~x); out = alg_theta.zero()
        coeffs = self.numerator().coeffs()

        for i in xrange(len(coeffs)):
            out += alg_theta([R(p) for p in coeffs[i].coeffs()])*(S**i)

        out = out.numerator().change_ring(alg.base_ring()).to_D(alg)
        out = alg.gen()**(len(coeffs)-1)*out

        return out

    def to_F(self, alg): # s2delta
        """
        Returns the difference operator corresponding to ``self``

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_F()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the forward difference with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: R.<x> = ZZ['x']
          sage: A.<Sx> = OreAlgebra(R, 'Sx')
          sage: (Sx^4).to_F(OreAlgebra(R, 'Fx'))
          Fx^4 + 4*Fx^3 + 6*Fx^2 + 4*Fx + 1
          sage: (Sx^4).to_F('Fx').to_S(A)
          Sx^4
        
        """
        R = self.base_ring(); x = R.gen(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:x+one}, {x:one})
        elif not isinstance(alg, type(self.parent())) or not alg.is_F() or \
             alg.base_ring().base_ring() is not R.base_ring():
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        delta = alg.gen() + alg.one(); delta_k = alg.one(); R = alg.base_ring()
        c = self.coeffs(); out = alg(R(c[0]))

        for i in xrange(self.order()):
            
            delta_k *= delta
            out += R(c[i + 1])*delta_k

        return out

    def to_T(self, alg):
        """
        Returns a differential operator, expressed in terms of the Euler derivation,
        which annihilates every power series (about the origin) whose coefficient
        sequence is annihilated by ``self``.
        The output operator may not be minimal. 

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_T()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the Euler derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
          sage: A.<Sn> = OreAlgebra(Rn, 'Sn')
          sage: B.<Tx> = OreAlgebra(Rx, 'Tx')
          sage: (Sn - 1).to_T(B)
          (-x + 1)*Tx - x
          sage: ((n+1)*Sn - 1).to_T(B)
          Tx^2 - x*Tx - x
          sage: (x*Tx-1).to_S(A).to_T(B)
          x*Tx^2 + (x - 1)*Tx
        
        """
        return self.to_D('D').to_T(alg)        

    def to_list(self, init, n, start=0, append=False, padd=False):
        """
        Computes the terms of some sequence annihilated by ``self``.

        INPUT:

        - ``init`` -- a vector (or list or tuple) of initial values.
          The components must be elements of ``self.base_ring().base_ring().fraction_field()``.
          If the length is more than ``self.order()``, we do not check whether the given
          terms are consistent with ``self``. 
        - ``n`` -- desired number of terms. 
        - ``start`` (optional) -- index of the sequence term which is represented
          by the first entry of ``init``. Defaults to zero.
        - ``append`` (optional) -- if ``True``, the computed terms are appended
          to ``init`` list. Otherwise (default), a new list is created.
        - ``padd`` (optional) -- if ``True``, the vector of initial values is implicitely
          prolonged to the left (!) by zeros if it is too short. Otherwise (default),
          the method raises a ``ValueError`` if ``init`` is too short.

        OUTPUT:

        A list of ``n`` terms whose `k` th component carries the sequence term with
        index ``start+k``.
        Terms whose calculation causes an error are represented by ``None``. 

        EXAMPLES::

           sage: R = ZZ['x']['n']; x = R('x'); n = R('n')
           sage: A.<Sn> = OreAlgebra(R, 'Sn')
           sage: L = ((n+2)*Sn^2 - x*(2*n+3)*Sn + (n+1))
           sage: L.to_list([1, x], 5)
           [1, x, (3*x^2 - 1)/2, (5*x^3 - 3*x)/2, (35*x^4 - 30*x^2 + 3)/8]
           sage: polys = L.to_list([1], 5, padd=True)
           [1, x, (3*x^2 - 1)/2, (5*x^3 - 3*x)/2, (35*x^4 - 30*x^2 + 3)/8]
           sage: L.to_list([polys[3], polys[4]], 8, start=3)
           [(5*x^3 - 3*x)/2, (35*x^4 - 30*x^2 + 3)/8, (63*x^5 - 70*x^3 + 15*x)/8, (231*x^6 - 315*x^4 + 105*x^2 - 5)/16, (429*x^7 - 693*x^5 + 315*x^3 - 35*x)/16]
           
           sage: ((n-5)*Sn - 1).to_list([1], 10)
           [1, 1/-5, 1/20, 1/-60, 1/120, -1/120, None, None, None, None]
        
        """
        return _rec2list(self, init, n, start, append, padd, self.parent().sigma(), ZZ)

    def annihilator_of_sum(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite sums `\sum_{k=0}^n a_k`
        where `a_n` runs through the sequences annihilated by ``self``.
        The output operator is not necessarily of smallest possible order. 

        EXAMPLES::

           sage: R.<x> = ZZ['x']
           sage: A.<Sx> = OreAlgebra(R, 'Sx')
           sage: ((x+1)*Sx - x).annihilator_of_sum() # constructs L such that L(H_n) == 0
           (x + 2)*Sx^2 + (-2*x - 3)*Sx + x + 1
           
        """
        A = self.parent()
        return self.map_coefficients(A.sigma())*(A.gen() - A.one())

    def annihilator_of_composition(self, a, solver=None):
        """
        Returns an operator `L` which annihilates all the sequences `f(floor(a(n)))`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order.

        INPUT:

        - `a` -- a polynomial `u*x+v` where `x` is the generator of the base ring,
          `u` and `v` are integers or rational numbers. If they are rational,
          the base ring of the parent of ``self`` must contain ``QQ``.
        - ``solver`` (optional) -- a callable object which applied to a matrix
          with polynomial entries returns its kernel. 

        EXAMPLES::

          sage: R.<x> = QQ['x']
          sage: A.<Sx> = OreAlgebra(R, 'Sx')
          sage: ((2+x)*Sx^2-(2*x+3)*Sx+(x+1)).annihilator_of_composition(2*x+5) 
          (16*x^3 + 188*x^2 + 730*x + 936)*Sx^2 + (-32*x^3 - 360*x^2 - 1340*x - 1650)*Sx + 16*x^3 + 172*x^2 + 610*x + 714
          sage: ((2+x)*Sx^2-(2*x+3)*Sx+(x+1)).annihilator_of_composition(1/2*x)
          (1/2*x^2 + 11/2*x + 15)*Sx^6 + (-3/2*x^2 - 25/2*x - 27)*Sx^4 + (3/2*x^2 + 17/2*x + 13)*Sx^2 - 1/2*x^2 - 3/2*x - 1
          sage: ((2+x)*Sx^2-(2*x+3)*Sx+(x+1)).annihilator_of_composition(100-x)
          (-x + 99)*Sx^2 + (2*x - 199)*Sx - x + 100
          
        """

        A = self.parent()
        
        if a in QQ:
            # a is constant => f(a) is constant => S-1 kills it
            return A.gen() - A.one()

        R = QQ[A.base_ring().gen()]

        try:
            a = R(a)
        except:
            raise ValueError, "argument has to be of the form u*x+v where u,v are rational"

        if a.degree() > 1:
            raise ValueError, "argument has to be of the form u*x+v where u,v are rational"

        try:
            u = QQ(a[1]); v = QQ(a[0])
        except:
            raise ValueError, "argument has to be of the form u*x+v where u,v are rational"

        r = self.order(); x = A.base_ring().gen()

        # special treatment for easy cases
        w = u.denominator().abs()
        if w > 1:
            w = w.lcm(v.denominator()).abs()
            p = self.polynomial()(A.associated_commutative_algebra().gen()**w)
            q = p = A(p.map_coefficients(lambda f: f(x/w)))
            for i in xrange(1, w):
                q = q.lclm(p.annihilator_of_composition(x - i), solver=solver)
            return q.annihilator_of_composition(w*u*x + w*v)
        elif v != 0:
            s = A.sigma(); v = v.floor()
            L = self.map_coefficients(lambda p: s(p, v))
            return L if u == 1 else L.annihilator_of_composition(u*x)
        elif u == 1:
            return self
        elif u < 0:
            c = [ p(-r - x) for p in self.coeffs() ]; c.reverse()
            return A(c).annihilator_of_composition(-u*x)

        # now a = u*x where u > 1 is an integer. 
        from sage.matrix.constructor import Matrix
        A = A.change_ring(A.base_ring().fraction_field())
        if solver == None:
            solver = A._solver()
        L = A(self)

        p = A.one(); Su = A.gen()**u # possible improvement: multiplication matrix. 
        mat = [ p.coeffs(padd=r) ]; sol = []

        while len(sol) == 0:

            p = (Su*p) % L
            mat.append( p.coeffs(padd=r) )
            sol = solver(Matrix(mat).transpose())

        return self.parent()(list(sol[0])).map_coefficients(lambda p: p(u*x))

    def annihilator_of_interlacing(self, *other):
        r"""
        Returns an operator `L` which annihilates any sequence which can be
        obtained by interlacing sequences annihilated by ``self`` and the
        operators given in the arguments.

        More precisely, if ``self`` and the operators given in the arguments are
        denoted `L_1,L_2,\dots,L_m`, and if `f_1(n),\dots,f_m(n)` are some
        sequences such that `L_i` annihilates `f_i(n)`, then the output operator
        `L` annihilates sequence
        `f_1(0),f_2(0),\dots,f_m(0),f_1(1),f_2(1),\dots,f_m(1),\dots`, the
        interlacing sequence of `f_1(n),\dots,f_m(n)`.

        The output operator is not necessarily of smallest possible order.

        The ``other`` operators must be coercible to the parent of ``self``.

        EXAMPLES::

          sage: R.<x> = QQ['x']
          sage: A.<Sx> = OreAlgebra(R, 'Sx')
          sage: (x*Sx - (x+1)).annihilator_of_interlacing(Sx - (x+1), Sx + 1)
          (-x^7 - 45/2*x^6 - 363/2*x^5 - 1129/2*x^4 - 45/2*x^3 + 5823/2*x^2 + 5751/2*x - 2349)*Sx^9 + (1/3*x^8 + 61/6*x^7 + 247/2*x^6 + 4573/6*x^5 + 14801/6*x^4 + 7173/2*x^3 + 519/2*x^2 - 3051*x + 756)*Sx^6 + (-7/2*x^6 - 165/2*x^5 - 1563/2*x^4 - 7331/2*x^3 - 16143/2*x^2 - 9297/2*x + 5535)*Sx^3 - 1/3*x^8 - 67/6*x^7 - 299/2*x^6 - 6157/6*x^5 - 22877/6*x^4 - 14549/2*x^3 - 10839/2*x^2 + 1278*x + 2430

        """
        A = self.parent(); A = A.change_ring(A.base_ring().fraction_field())
        ops = [A(self)] + map(A, list(other))
        S_power = A.associated_commutative_algebra().gen()**len(ops)
        x = A.base_ring().gen()

        for i in xrange(len(ops)):
            ops[i] = A(ops[i].polynomial()(S_power)\
                       .map_coefficients(lambda p: p(x/len(ops))))\
                       .annihilator_of_composition(x - i)

        return self.parent()(reduce(lambda p, q: p.lclm(q), ops).numerator())

    def generalized_series_solutions(self, n): # at infinity. 
        raise NotImplementedError


#############################################################################################################

class UnivariateQRecurrenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[S], where S is the shift x->q*x for some q in K.
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        if type(f) in (tuple, list):

            r = self.order()
            R = self.parent().base_ring()
            _, q = self.parent().is_Q()
            K = R.base_ring()
            z = K.zero()
            c = self.numerator().coeffs()
            d = self.denominator()

            def fun(n):
                if f[n + r] is None:
                    return None
                else:
                    try:
                        qn = q**n
                        return sum( c[i](qn)*f[n + i] for i in xrange(r + 1) )/d(qn)
                    except:
                        return None

            return type(f)(fun(n) for n in xrange(len(f) - r))

        R = self.parent(); x = R.base_ring().gen(); qx = R.sigma()(x)
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : p(qx)

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_J(self, alg): # q2j
        """
        Returns a q-differential operator which annihilates every power series (about the origin)
        whose coefficient sequence is annihilated by ``self``.
        The output operator may not be minimal. 

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_J()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the q-derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
          sage: A.<Qn> = OreAlgebra(Rn, 'Qn', q=2)
          sage: B.<Jx> = OreAlgebra(Rx, 'Jx', q=2)
          sage: (Qn - 1).to_J(B)
          (-2*x + 1)*Jx - 1
          sage: ((n+1)*Qn - 1).to_J(B)
          2*x*Jx^2 + (-4*x + 4)*Jx - 2
          sage: (x*Jx-1).to_Q(A).to_J(B) % (x*Jx - 1)
          0
        
        """
        R = self.base_ring(); K = R.base_ring(); x, q = self.parent().is_Q(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:q*x}, {x:one})
        elif not isinstance(alg, type(self.parent())) or not alg.is_J() or \
             alg.base_ring().base_ring() is not K or K(alg.is_J()[1]) != K(q):
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); x, q = alg.is_J()
        alg = alg.change_ring(R);

        Q = alg(~x); out = alg.zero()
        coeffs = self.numerator().coeffs()
        x_pows = {0 : alg.one(), 1 : ((q - R.one())*x)*alg.gen() + alg.one()}

        for i in xrange(len(coeffs)):
            term = alg.zero()
            c = coeffs[i].coeffs()
            for j in xrange(len(c)):
                if not x_pows.has_key(j):
                    x_pows[j] = x_pows[j - 1]*x_pows[1]
                term += c[j] * x_pows[j]
            out += term*(Q**i)

        return (alg.gen()**(len(coeffs)-1))*out.numerator().change_ring(alg.base_ring())

    def to_list(self, init, n, start=0, append=False, padd=False):
        """
        Computes the terms of some sequence annihilated by ``self``.

        INPUT:

        - ``init`` -- a vector (or list or tuple) of initial values.
          The components must be elements of ``self.base_ring().base_ring().fraction_field()``.
          If the length is more than ``self.order()``, we do not check whether the given
          terms are consistent with ``self``. 
        - ``n`` -- desired number of terms. 
        - ``start`` (optional) -- index of the sequence term which is represented
          by the first entry of ``init``. Defaults to zero.
        - ``append`` (optional) -- if ``True``, the computed terms are appended
          to ``init`` list. Otherwise (default), a new list is created.
        - ``padd`` (optional) -- if ``True``, the vector of initial values is implicitely
          prolonged to the left (!) by zeros if it is too short. Otherwise (default),
          the method raises a ``ValueError`` if ``init`` is too short.

        OUTPUT:

        A list of ``n`` terms whose `k` th component carries the sequence term with
        index ``start+k``.
        Terms whose calculation causes an error are represented by ``None``. 

        EXAMPLES::

           sage: R.<x> = QQ['x']; A.<Qx> = OreAlgebra(R, 'Qx', q=3)
           sage: (Qx^2-x*Qx + 1).to_list([1,1], 10)
           [1, 1, 0, -1, -9, -242, -19593, -4760857, -3470645160, -7590296204063]
           sage: (Qx^2-x*Qx + 1)(_)
           [0, 0, 0, 0, 0, 0, 0, 0]
        
        """
        _, q = self.parent().is_Q()
        return _rec2list(self, init, n, start, append, padd, self.parent().sigma(), lambda n: q**n)

    def annihilator_of_sum(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite sums `\sum_{k=0}^n a_k`
        where `a_n` runs through the sequences annihilated by ``self``.
        The output operator is not necessarily of smallest possible order. 

        EXAMPLES::

           sage: R.<x> = ZZ['q'].fraction_field()['x']
           sage: A.<Qx> = OreAlgebra(R, 'Qx')
           sage: ((x+1)*Qx - x).annihilator_of_sum()
           (q*x + 1)*Qx - q*x
           
        """
        A = self.parent()
        return self.map_coefficients(A.sigma())*(A.gen() - A.one())

    def annihilator_of_composition(self, u, v):
        """
        If self is such that self*f(n) = 0 and u, v are nonnegative rational numbers,
        this function returns an operator L such that L*f(floor(u*n+v)) = 0.
        """
        raise NotImplementedError

    def annihilator_of_interlacing(self, *other):
        """
        If ``self`` is an operator which annihilates a certain sequence `a(n)`
        and ``other`` an operator from the same algebra which annihilates some sequence `b(n)`,
        this returns an operator which annihilates the sequence `a(0),b(0),a(1),b(1),a(2),b(2),...`.

        Any number of operators can be given. For example, in the case of two arguments,
        the resulting operator will annihilate the sequence `a(0),b(0),c(0),a(1),...`,
        where `a(n),b(n),c(n)` are sequence annihilated by ``self`` and the to operators
        given as argument.         
        """
        raise NotImplementedError


#############################################################################################################

class UnivariateQDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[J], where J is the Jackson q-differentiation J f(x) = (f(q*x) - f(x))/(q*(x-1))
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        R = self.parent(); x = R.base_ring().gen(); qx = R.sigma()(x)
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : (p(qx) - p)/(x*(q-1))

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_Q(self, alg): # j2q
        """
        Returns a q-recurrence operator which annihilates the coefficient sequence
        of every power series (about the origin) annihilated by ``self``.
        The output operator may not be minimal. 

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_Q() == self.parent().is_J()``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the q-shift with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
          sage: A.<Jx> = OreAlgebra(Rx, 'Jx', q=2)
          sage: B.<Qn> = OreAlgebra(Rn, 'Qn', q=2)
          sage: (Jx - 1).to_Q(B)
          (2*n - 1)*Qn - 1
          sage: ((x+1)*Jx - 1).to_Q(B)
          (4*n - 1)*Qn^2 + (2*n - 2)*Qn
          sage: (n*Qn-1).to_J(A).to_Q(B) % (n*Qn - 1)
          0 
        
        """
        R = self.base_ring(); K = R.base_ring()
        x, q = self.parent().is_J(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:q*x}, {})
        elif not isinstance(alg, type(self.parent())) or not alg.is_Q() or \
             alg.base_ring().base_ring() is not R.base_ring() or K(alg.is_Q()[1]) != K(q) :
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); x, q = alg.is_Q()
        alg = alg.change_ring(R);

        Q = alg.gen(); J = ((q*x - R.one())/(q - R.one()))*Q; J_pow = alg.one()
        out = alg.zero(); 
        coeffs = self.numerator().coeffs()
        d = max( c.degree() for c in coeffs )

        for i in xrange(len(coeffs)):
            if i > 0:
                J_pow *= J
            c = coeffs[i].padded_list(d + 1)
            c.reverse()
            out += alg(map(R, c)) * J_pow            

        return ((q-1)**(len(coeffs)-1)*out).numerator().change_ring(alg.base_ring())

    def annihilator_of_integral(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite `q`-integrals `\int_q f`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order. 

        EXAMPLES::

           sage: R.<x> = ZZ['q'].fraction_field()['x']
           sage: A.<Jx> = OreAlgebra(R, 'Jx')
           sage: ((x-1)*Jx - 2*x).annihilator_of_integral()
           (x - 1)*Jx^2 - 2*x*Jx
           sage: _.annihilator_of_associate(Jx)
           (x - 1)*Jx - 2*x
           
        """
        return self*self.parent().gen()


#############################################################################################################

class UnivariateDifferenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[F], where F is the forward difference operator F f(x) = f(x+1) - f(x)
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        if type(f) in (tuple, list):
            return self.to_rec('n')(f, **kwargs)
            
        R = self.parent(); x = R.base_ring().gen(); qx = R.sigma()(x)
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : p(qx) - p

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_S(self, alg): # delta2s
        """
        Returns the differential operator corresponding to ``self``

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_S()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          a standard shift with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: R.<x> = ZZ['x']
          sage: A.<Fx> = OreAlgebra(R, 'Fx')
          sage: (Fx^4).to_S(OreAlgebra(R, 'Sx'))
          Sx^4 - 4*Sx^3 + 6*Sx^2 - 4*Sx + 1
          sage: (Fx^4).to_S('Sx')
          Sx^4 - 4*Sx^3 + 6*Sx^2 - 4*Sx + 1
        
        """
        R = self.base_ring(); x = R.gen(); one = R.one(); zero 

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:x+one}, {})
        elif not isinstance(alg, type(self.parent())) or not alg.is_S() or \
             alg.base_ring().base_ring() is not R.base_ring():
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        delta = alg.gen() - alg.one(); delta_k = alg.one(); R = alg.base_ring()
        c = self.coeffs(); out = alg(R(c[0]))

        for i in xrange(self.order()):
            
            delta_k *= delta
            out += R(c[i + 1])*delta_k

        return out

    def to_D(self, alg):
        """
        Returns a differential operator which annihilates every power series (about
        the origin) whose coefficient sequence is annihilated by ``self``.
        The output operator may not be minimal. 

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_D()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the standard derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
          sage: A.<Fn> = OreAlgebra(Rn, 'Fn')
          sage: B.<Dx> = OreAlgebra(Rx, 'Dx')
          sage: Fn.to_D(B)
          (-x + 1)*Dx - 1
          sage: ((n+1)*Fn - 1).to_D(B)
          (-x^2 + x)*Dx^2 + (-4*x + 1)*Dx - 2
          sage: (x*Dx-1).to_F(A).to_D(B)
          x*Dx - 1
        
        """
        return self.to_S('S').to_D(alg)

    def to_T(self, alg):
        """
        Returns a differential operator, expressed in terms of the Euler derivation,
        which annihilates every power series (about the origin) whose coefficient
        sequence is annihilated by ``self``.
        The output operator may not be minimal. 

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_T()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the Euler derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
          sage: A.<Fn> = OreAlgebra(Rn, 'Fn')
          sage: B.<Tx> = OreAlgebra(Rx, 'Tx')
          sage: Fn.to_T(B)
          (-x + 1)*Tx - x
          sage: ((n+1)*Fn - 1).to_T(B)
          (-x + 1)*Tx^2 - 3*x*Tx - 2*x
          sage: (x*Tx-1).to_F(A).to_T(B)
          x*Tx^2 + (x - 1)*Tx
        
        """
        return self.to_S('S').to_T(alg)

    def to_list(self, *args, **kwargs):
        return self.to_rec('S').to_list(*args, **kwargs)


#############################################################################################################

class UnivariateEulerDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[T], where T is the Euler differential operator T = x*d/dx
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        R = self.parent(); x = R.base_ring().gen(); 
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : x*p.derivative()

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_D(self, alg): # theta2d
        """
        Returns the differential operator corresponding to ``self``

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_D()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the standard derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: R.<x> = ZZ['x']
          sage: A.<Tx> = OreAlgebra(R, 'Tx')
          sage: (Tx^4).to_D(OreAlgebra(R, 'Dx'))
          x^4*Dx^4 + 6*x^3*Dx^3 + 7*x^2*Dx^2 + x*Dx
          sage: (Tx^4).to_D('Dx').to_T(A)
          Tx^4
        
        """
        R = self.base_ring(); x = R.gen(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {}, {x:one})
        elif not isinstance(alg, type(self.parent())) or not alg.is_D() or \
             alg.base_ring().base_ring() is not R.base_ring():
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring(); theta = R.gen()*alg.gen(); theta_k = alg.one(); 
        c = self.coeffs(); out = alg(R(c[0]))

        for i in xrange(self.order()):
            
            theta_k *= theta
            out += R(c[i + 1])*theta_k

        return out

    def to_S(self, alg):
        """
        Returns a recurrence operator annihilating the coefficient sequence of
        every power series (at the origin) annihilated by ``self``.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_S()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the standard shift with respect to ``self.base_ring().gen()``.

        EXAMPLES::

            sage: R.<x> = ZZ['x']
            sage: A.<Tx> = OreAlgebra(R, 'Tx')
            sage: R2.<n> = ZZ['n']
            sage: A2.<Sn> = OreAlgebra(R2, 'Sn')
            sage: (Tx - 1).to_S(A2)
            n - 1
            sage: ((1+x)*Tx^2 + Tx).to_S(A2)
            (n^2 + 3*n + 2)*Sn + n^2
            sage: ((x^3+x^2-x)*Tx + (x^2+1)).to_S(A2)
            Sn^3 + (-n - 2)*Sn^2 + (n + 2)*Sn + n

        """
        return self.to_D('D').to_S(alg)

    def to_F(self, alg):
        """
        Returns a difference operator annihilating the coefficient sequence of
        every power series (about the origin) annihilated by ``self``.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_F()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the forward difference with respect to ``self.base_ring().gen()``.

        EXAMPLES::

            sage: R.<x> = ZZ['x']
            sage: A.<Tx> = OreAlgebra(R, 'Tx')
            sage: R2.<n> = ZZ['n']
            sage: A2.<Fn> = OreAlgebra(R2, 'Fn')
            sage: (Tx - 1).to_F(A2)
            n - 1
            sage: ((1+x)*Tx^2 + Tx).to_F(A2)
            (n^2 + 3*n + 2)*Fn + 2*n^2 + 3*n + 2
            sage: ((x^3+x^2-x)*Tx + (x^2+1)).to_F(A2)
            Fn^3 + (-n + 1)*Fn^2 + (-n + 1)*Fn + n + 1

        """
        return self.to_D('D').to_F(alg)

#############################################################################################################


def _rec2list(L, init, n, start, append, padd, sigma, deform):
    """
    Common code for computing terms of holonomic and q-holonomic sequences.
    """
        
    r = L.order()
    terms = init if append else list(init)
    K = L.base_ring().base_ring().fraction_field()

    if len(terms) >= n:
        return terms
    elif len(terms) < r:

        if not padd:
            raise ValueError, "not enough initial values."
            
        z = K.zero(); padd = r - len(terms)
            
        if append:
            for i in xrange(padd):
                terms.insert(0, z)
            terms = _rec2list(L, terms, min(n, r) + padd, start - padd, True, False, sigma, deform)
            for i in xrange(padd):
                terms.remove(0)
        else:
            terms = _rec2list(L, [z]*padd + terms, min(n, r) + padd, start - padd, False, False, sigma, deform)[padd:]

        return _rec2list(L, terms, n, start, append, False, sigma, deform)

    for i in xrange(r):
        if terms[-i - 1] not in K:
            raise TypeError, "illegal initial value object"

    rec = L.numerator().coeffs()
    rec = tuple( -sigma(p, -r) for p in rec )
    lc = -rec[-1]

    for k in xrange(len(terms), n):

        lck = lc(deform(k + start))
        if lck.is_zero():
            for i in xrange(k, n):
                terms.append(None)
            return terms

        terms.append((~lck)*sum(terms[-r + k + i]*rec[i](deform(k + start)) for i in xrange(r)))

    return terms
    

#############################################################################################################

def __primitivePRS__(r,additional):
    """
    Computes one division step in the subresultant polynomial remainder sequence.
    """

    orddiff = r[0].order()-r[1].order()

    R = r[0].parent()

    alpha = R.sigma().factorial(r[1].leading_coefficient(),orddiff+1)
    newRem = (alpha*r[0]).quo_rem(r[1],fractionFree=True)
    beta = newRem[1].content()
    r2 = newRem[1].map_coefficients(lambda p: p//beta)
    return ((r[1],r2),newRem[0],alpha,beta)

def __classicPRS__(r,additional):
    """
    Computes one division step in the classic polynomial remainder sequence.
    """

    newRem = r[0].quo_rem(r[1])
    return ((r[1],newRem[1]),newRem[0],r[0].parent().base_ring().one(),r[0].parent().base_ring().one())

def __monicPRS__(r,additional):
    """
    Computes one division step in the monic polynomial remainder sequence.
    """

    newRem = r[0].quo_rem(r[1])
    return ((r[1],newRem[1].primitive_part()),newRem[0],r[0].parent().base_ring().one(),r[0].parent().base_ring().one())

def __improvedPRS__(r,additional):
    """
    Computes one division step in the improved polynomial remainder sequence.
    """

    d0 = r[0].order()
    d1 = r[1].order()
    orddiff = d0-d1

    R = r[0].parent()
    Rbase = R.base_ring()
    sigma = R.sigma()

    if (len(additional)==0):
        essentialPart = gcd(sigma(r[0].leading_coefficient(),-orddiff),r[1].leading_coefficient())
        phi = Rbase.one()
        beta = (-Rbase.one())**(orddiff+1)*R.sigma().factorial(sigma(phi,1),orddiff)
    else:
        (d2,oldalpha,k,essentialPart,phi) = (additional.pop(),additional.pop(),additional.pop(),additional.pop(),additional.pop())
        phi = oldalpha / R.sigma().factorial(sigma(phi,1),d2-d1-1)
        beta = ((-Rbase.one())**(orddiff+1)*R.sigma().factorial(sigma(phi,1),orddiff)*k)
        essentialPart = sigma(essentialPart,-orddiff)

    k = r[1].leading_coefficient()//essentialPart
    alpha = R.sigma().factorial(k,orddiff)
    alpha2=alpha*sigma(k,orddiff)
    newRem = (alpha2*r[0]).quo_rem(r[1],fractionFree=True)
    r2 = newRem[1].map_coefficients(lambda p: p//beta)
    additional.extend([phi,essentialPart,k,alpha,d1])

    return ((r[1],r2),newRem[0],alpha2,beta)

def __subresultantPRS__(r,additional):
    """
    Computes one division step in the subresultant polynomial remainder sequence.
    """

    d0 = r[0].order()
    d1 = r[1].order()
    orddiff = d0-d1

    R = r[0].parent()
    Rbase = R.base_ring()
    sigma = R.sigma()

    if (len(additional)==0):
        phi = -Rbase.one()
        beta = (-Rbase.one())*R.sigma().factorial(sigma(phi,1),orddiff)
    else:
        (d2,phi) = (additional.pop(),additional.pop())
        phi = R.sigma().factorial(-r[0].leading_coefficient(),d0-d1) / R.sigma().factorial(sigma(phi,1),d0-d1-1)
        beta = (-Rbase.one())*R.sigma().factorial(sigma(phi,1),orddiff)*r[0].leading_coefficient()

    alpha = R.sigma().factorial(r[1].leading_coefficient(),orddiff+1)
    newRem = (alpha*r[0]).quo_rem(r[1],fractionFree=True)
    r2 = newRem[1].map_coefficients(lambda p: p//beta)

    additional.extend([phi,d1])

    return ((r[1],r2),newRem[0],alpha,beta)
