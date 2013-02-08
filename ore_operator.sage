
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

        EXAMPLE::

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
        """
        """
        raise NotImplementedError

    def is_monic(self):
        """
        Returns True if this polynomial is monic. The zero operator is by
        definition not monic.
        """
        raise NotImplementedError

    def is_unit(self):
        r"""
        Return True if this polynomial is a unit.
        """
        raise NotImplementedError
       
    def is_gen(self):
        r"""
        Return True if this operator is the distinguished generator of
        the parent Ore algebra. 
                
        Important - this function doesn't return True if self equals the
        generator; it returns True if self *is* the generator.
        """
        raise NotImplementedError
        #return bool(self._is_gen)

    def prec(self):
        """
        Return the precision of this operator. This is always infinity,
        since operators are of infinite precision by definition (there is
        no big-oh).
        """
        return infinity.infinity
    
    # conversion

    def change_variable_name(self, var, n=0):
        """
        Return a new operator over the same base ring but in a different
        variable.
        """
        R = self.parent().change_var(var, n)
        return R(self.list())
        
    def change_ring(self, R):
        """
        Return a copy of this operator but with coefficients in R, if at
        all possible.
        """
        S = self.parent().change_ring(R)
        return S(self)

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
        to the base ring of its parent, and admits a multiplicative inverse there. 
        """
        return self.parent().one_element()/self

    def __div__(self, right):
        """
        Exact right division. Uses division with remainder, and returns the quotient if the
        remainder is zero. Otherwise a TypeError is raised.
        """
        Q, R = self.quo_rem(right)
        if R == self.parent().zero():
            return Q
        raise TypeError, "Cannot divide the given OreOperators"
                   
    def __floordiv__(self,right):
        """
        Quotient of quotient with remainder. 
        """
        Q, _ = self.quo_rem(right)
        return Q
        
    def __mod__(self, other):
        """
        Remainder of quotient with remainder. 
        """
        _, R = self.quo_rem(other)
        return R

    def quo_rem(self, other):
        """
        """
        raise NotImplementedError

    # base ring related functions
        
    def base_ring(self):
        """
        Return the base ring of the parent of self.
        """
        return self.parent().base_ring()

    def base_extend(self, R):
        """
        Return a copy of this operator but with coefficients in R, if
        there is a natural map from coefficient ring of self to R.
        """
        S = self.parent().base_extend(R)
        return S(self)

    # coefficient-related functions

    def __getitem__(self, n):
        raise NotImplementedError

    def __setitem__(self, n, value):
        raise IndexError, "Operators are immutable"

    def is_primitive(self, n=None, n_prime_divs=None):
        """
        Returns ``True`` if the operator is primitive.
        """
        raise NotImplementedError

    def is_monomial(self):
        """
        Returns True if self is a monomial, i.e., a power of the generator.
        """
        return len(self.exponents()) == 1 and self.leading_coefficient() == 1

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
        Does not change this operator. 
        """
        if self.is_monic():
            return self
        a = ~self.leading_coefficient()
        R = self.parent()
        if a.parent() != R.base_ring():
            S = R.base_extend(a.parent())
            return a*S(self)
        else:
            return a*self

    def content(self):
        """
        Return the content of ``self``.
        """
        raise NotImplementedError

    def map_coefficients(self, f, new_base_ring = None):
        """
        Returns the polynomial obtained by applying ``f`` to the non-zero
        coefficients of self.
        """
        raise NotImplementedError

    def subs(self, *x, **kwds):
        r"""
        Applies a substitution to each of the coefficients. 
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

    def denominator(self):
        """
        Return a denominator of self.

        First, the lcm of the denominators of the entries of self
        is computed and returned. If this computation fails, the
        unit of the parent of self is returned.
        """
        raise NotImplementedError

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

        D = self.parent().var();
        if kwds.has_key(D):
            D = kwds[D]
        else:
            D = lambda p:p

        R = f.parent(); Dif = f; result = R(self[0r])*f; 
        for i in xrange(1r, self.order() + 1r):
            Dif = D(Dif)
            result += R(self[i])*Dif
        
        return result

    # tests

    def __nonzero__(self):
        return self._poly.__nonzero__()

    def __eq__(self,other):
        return (isinstance(other, UnivariateOreOperator) and self.parent()==other.parent() and self.polynomial()==other.polynomial())

    def _is_atomic(self):
        return self._poly._is_atomic()

    def is_monic(self):
        """
        Returns True if this polynomial is monic. The zero operator is by definition not monic.
        """
        return self._poly.is_monic()

    def is_unit(self):
        r"""
        Return True if this polynomial is a unit.
        """
        return self._poly.is_unit()
       
    def is_gen(self):
        r"""
        Return True if this operator is the distinguished generator of
        the parent Ore algebra. 
                
        Important - this function doesn't return True if self equals the
        generator; it returns True if self *is* the generator.
        """
        return self._poly.is_gen()

    def prec(self):
        """
        Return the precision of this operator. This is always infinity,
        since operators are of infinite precision by definition (there is
        no big-oh).
        """
        return infinity.infinity
    
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

        coeffs = self.polynomial().coeffs()
        DiB = right.polynomial() # D^i * B, for i=0,1,2,...

        R = self.parent() # Ore algebra
        sigma = R.sigma(); delta = R.delta()
        A = DiB.parent() # associate commutative algebra
        D = A.gen() 
        res = coeffs[0r]*DiB

        for i in xrange(1r, len(coeffs)):

            DiB = DiB.map_coefficients(sigma)*D + DiB.map_coefficients(delta)
            res += coeffs[i]*DiB

        return R(res)

    def quo_rem(self, other):

        if other.is_zero(): 
            raise ZeroDivisionError, "other must be nonzero"
        if (self.order() < other.order()): return (self.parent().zero(),self)

        R = self.parent()
        K = R.base_ring()
        if not K.is_field(): R = R.change_ring(K.fraction_field())
        p = R(self)
        q = R(other)
        sigma = R.sigma()
        D = R.gen()
        orddiff = p.order() - q.order()
        cfquo = R.one()
        quo = R.zero()

        qlcs = [q.leading_coefficient()]
        for i in range(orddiff): qlcs.append(sigma(qlcs[len(qlcs)-1]))

        while(orddiff >= 0):
            cfquo = p.leading_coefficient()/qlcs[orddiff] * D**(orddiff)
            quo = quo+cfquo
            p = p - cfquo*q
            orddiff = p.order() - q.order()
        return (quo,p)

    def gcrd(self, other):
        """
        Returns the GCRD of self and other
        """
        if self.is_zero(): return other
        if other.is_zero(): return self

        r0 = self
        r1 = other
        if (r0.order()<r1.order()):
            r0,r1=r1,r0

        while not r1.is_zero(): r0,r1=r1,r0.quo_rem(r1)[1]
            
        return r0

    def xgcrd(self, other):
        """
        When called for two operators p,q, this will return their GCRD g together with two operators s and t such that sp+tq=g
        """
        if self.is_zero(): return other
        if other.is_zero(): return self

        r0 = self
        r1 = other
        if (r0.order()<r1.order()):
            r0,r1=r1,r0
        
        R = self.parent()

        a11,a12,a21,a22 = R.one(),R.zero(),R.zero(),R.one()

        while not r1.is_zero():        
            q = r0.quo_rem(r1)
            r0,r1=r1,q[1]
            a11,a12,a21,a22 = a21,a22,(a11-q[0]*a21),(a12-q[0]*a22)
        return (r0,a11,a12)

    def lclm(self, other):
        """
        """
        raise NotImplementedError

    def xlclm(self, other):
        """
        """
        raise NotImplementedError

    def symmetric_product(self, other, tensor_map):
        # tensor_map is meant to be a matrix [[a,b],[c,d]] such that
        # D(uv) = a u v + b Du v + c u Dv + d Du Dv  for "functions" u,v
        """
        """
        raise NotImplementedError

    def symmetric_power(self, exp, tensor_map):
        """
        """
        raise NotImplementedError

    def annihilator_of_operator_of_solution(self, other):
        """
        computes an operator M such that when self*f = 0, then M*(other*f)=0
        """
        raise NotImplementedError

    # coefficient-related functions

    def order(self):
        return self.polynomial().degree()

    def valuation(self):
        return min(self.exponents())

    def __getitem__(self, n):
        return self.polynomial()[n]

    def __setitem__(self, n, value):
        raise IndexError("Operators are immutable")

    def is_primitive(self, n=None, n_prime_divs=None):
        """
        Returns ``True`` if the operator is primitive.
        """
        return self.content().is_unit()

    def is_monomial(self):
        """
        Returns True if self is a monomial, i.e., a power of the generator.
        """
        return len(self.exponents()) == 1 and self.leading_coefficient() == 1

    def leading_coefficient(self):
        """
        Return the leading coefficient of this operator. 
        """
        return self.polynomial().leading_coefficient()

    def constant_coefficient(self):
        """
        Return the leading coefficient of this operator. 
        """
        return self.polynomial()[0]

    def content(self):
        return self.polynomial().content()

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

    def subs(self, *x, **kwds):
        r"""
        Applies a substitution to each of the coefficients. 
        """
        raise NotImplementedError

    def coefficients(self):
        """
        Return the coefficients of the monomials appearing in self.
        """
        return self.polynomial().coefficients()

    def coeffs(self):
        """
        Return the coefficient vector of this operator.
        """
        return self.polynomial().coeffs()

    def exponents(self):
        """
        Return the exponents of the monomials appearing in self.
        """
        return self.polynomial().exponents()


#############################################################################################################

class UnivariateOreOperatorOverRationalFunctionField(UnivariateOreOperator):
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

    # numerator and denominator

    def numerator(self):
        """
        Return a numerator of self computed as self.denominator() * self
        """
        return self.denominator() * self

    def denominator(self):
        """
        Return a denominator of self.

        First, the lcm of the denominators of the coefficient of self
        is computed and returned. If this computation fails, the
        unit of the base of the parent of self is returned.
        """
        R = self.base_ring()
        try:
            return lcm([ R(p.denominator()) for p in self.coefficients() ])
        except:
            return R.one_element()

#############################################################################################################

class UnivariateDifferentialOperatorOverRationalFunctionField(UnivariateOreOperatorOverRationalFunctionField):
    """
    Element of an Ore algebra K(x)[D], where D acts as derivation d/dx on K(x).
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverRationalFunctionField, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):
        
        D = self.parent().var();
        if not kwargs.has_key(D):
            kwargs[D] = lambda p : p.derivative()

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_recurrence(self, *args):
        raise NotImplementedError

    def to_rec(self, *args):
        return self.to_recurrence(args)

    def integrate(self):
        """
        If self is such that self*f = 0, this function returns an operator L such that L*int(f) = 0
        """
        raise NotImplementedError

    def compose(self, alg):
        """
        If self is such that self*f = 0 and alg is an algebraic function, this function returns an
        operator L such that L*(f circ alg) = 0.
        """

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

class UnivariateRecurrenceOperatorOverRationalFunctionField(UnivariateOreOperatorOverRationalFunctionField):
    """
    Element of an Ore algebra K(x)[S], where S is the shift x->x+1.
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverRationalFunctionField, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):
        
        D = self.parent().var();
        x = self.parent().base_ring().gen()
        if not kwargs.has_key(D):
            kwargs[D] = lambda p : p(x+1)

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_differential_equation(self, *args):
        raise NotImplementedError

    def to_deq(self, *args):
        return self.to_differential_equation(args)

    def sum(self):
        """
        If self is such that self*f = 0, this function returns an operator L such that L*sum(f) = 0
        """
        raise NotImplementedError

    def compose(self, u, v):
        """
        If self is such that self*f(n) = 0 and u, v are nonnegative rational numbers,
        this function returns an operator L such that L*f(floor(u*n+v)) = 0.
        """
        raise NotImplementedError

    def generalized_series_solutions(self, n): # at infinity. 
        raise NotImplementedError

    def get_data(self, init, n):
        raise NotImplementedError

#############################################################################################################

class UnivariateQRecurrenceOperatorOverRationalFunctionField(UnivariateOreOperatorOverRationalFunctionField):
    """
    Element of an Ore algebra K(x)[S], where S is the shift x->q*x for some q in K.
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverRationalFunctionField, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        R = self.parent()
        D = R.var()
        x = R.base_ring().gen()
        qx = R.sigma()(x)
        if not kwargs.has_key(D):
            kwargs[D] = lambda p : p(qx)

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def sum(self):
        """
        If self is such that self*f = 0, this function returns an operator L such that L*sum(f) = 0
        """
        raise NotImplementedError

    def compose(self, u, v):
        """
        If self is such that self*f(n) = 0 and u, v are nonnegative rational numbers,
        this function returns an operator L such that L*f(floor(u*n+v)) = 0.
        """
        raise NotImplementedError

    def get_data(self, init, n):
        raise NotImplementedError
