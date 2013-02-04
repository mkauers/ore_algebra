"""
OreOperator
===========

AUTHORS:

-  Maximilian Jaroschek, Fredrik Johansson, Manuel Kauers

"""

class OreOperator(RingElement):
    """
    An Ore operator. 
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

    def __call__(self, *x, **kwds):
        """
        Lets ``self`` act on ``x`` and returns the result.
        """
        raise NotImplementedError

    # tests

#   ???
#   def __richcmp__(left, right, int op):
#        return (<Element>left)._richcmp(right, op)

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

    def change_variable_name(self, var):
        """
        Return a new operator over the same base ring but in a different
        variable.
        """
        R = self.parent().base_ring()[var]
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
        """
        """
        raise NotImplementedError

    def __long__(self):
        """
        """
        raise NotImplementedError

    def _repr(self, name=None):
        """
        """
        raise NotImplementedError

    def _repr_(self):
        r"""
        """
        return self._repr()

    def _latex_(self, name=None):
        r"""
        """
        raise NotImplementedError
        
    def _sage_input_(self, sib, coerced):
        r"""
        """
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
        """
        return self.parent().one_element()/self

    def __div__(self, right):
        """
        Exact division. Works only if right can be casted to base ring and inverted there.
        """
        raise NotImplementedError

    def __pow__(self, right, modulus):
        """
        """
        raise NotImplementedError
        
    def _pow(self, right):
        """
        """
        raise NotImplementedError
                   
    def __floordiv__(self,right):
        """
        """
        Q, _ = self.quo_rem(right)
        return Q
        
    def __mod__(self, other):
        """
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
        """
        """
        raise IndexError("Operators are immutable")

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

    def numerator(self):
        """
        Return a numerator of self computed as self * self.denominator()
        """
        return self * self.denominator()

#############################################################################################################
    
class UnivariateOreOperator(OreOperator):
    """
    Element of an Ore algebra with a single generator and a commutative field as base ring.     
    """

    def __init__(self, parent, is_gen = False, construct=False): 
        OreOperator.__init__(self, parent, is_gen, construct)

    # action

    def __call__(self, *x, **kwds):
        """
        Lets ``self`` act on ``x`` and returns the result.
        """
        raise NotImplementedError

    # tests

    def __nonzero__(self):
        return self.order() >= 0

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
        """
        """
        raise NotImplementedError

    def __long__(self):
        """
        """
        raise NotImplementedError

    def _repr(self, name=None):
        """
        """
        raise NotImplementedError

    def _repr_(self):
        r"""
        """
        return self._repr()

    def _latex_(self, name=None):
        r"""
        """
        raise NotImplementedError
        
    def _sage_input_(self, sib, coerced):
        r"""
        """
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

    def _add_(self, right):
        raise NotImplementedError
    
    def _neg_(self):
        raise NotImplementedError

    def _lmul_(self, left):
        raise NotImplementedError
    
    def _rmul_(self, right):
        raise NotImplementedError

    def _mul_(self, right):
        raise NotImplementedError

    def __invert__(self):
        """
        """
        return self.parent().one_element()/self

    def __div__(self, right):
        """
        Exact division. Works only if right can be casted to base ring and inverted there.
        """
        raise NotImplementedError

    def __pow__(self, right, modulus):
        """
        """
        raise NotImplementedError
        
    def _pow(self, right):
        """
        """
        raise NotImplementedError
                   
    def __floordiv__(self,right):
        """
        """
        Q, _ = self.quo_rem(right)
        return Q
        
    def __mod__(self, other):
        """
        """
        _, R = self.quo_rem(other)
        return R

    def quo_rem(self, other):
        """
        """
        raise NotImplementedError

    def gcrd(self, other):
        """
        """
        raise NotImplementedError

    def xgcrd(self, other):
        """
        """
        raise NotImplementedError

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

    def annihilator_of_operator_of_solution(self, other):
        """
        computes an operator M such that when self*f = 0, then M*(other*f)=0
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

    def order(self):
        raise NotImplementedError

    def valuation(self, p):
        raise NotImplementedError

    def __getitem__(self, n):
        raise NotImplementedError

    def __setitem__(self, n, value):
        """
        """
        raise IndexError("Operators are immutable")

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

    def numerator(self):
        """
        Return a numerator of self computed as self * self.denominator()
        """
        return self * self.denominator()

#############################################################################################################

class UnivariateRationalOreOperator(UnivariateOreOperator):
    """
    Element of an Ore algebra with a single generator and a commutative rational function field as base ring.     
    """

    def __init__(self, parent, is_gen = False, construct=False): 
        UnivariateOreOperator.__init__(self, parent, is_gen, construct)

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

class UnivariateRationalDifferentialOperator(UnivariateRationalOreOperator):
    """
    Element of an Ore algebra K(x)[D], where D acts as derivation d/dx on K(x).
    """

    def __init__(self, parent, is_gen = False, construct=False): 
        UnivariateRationalOreOperator.__init__(self, parent, is_gen, construct)

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

class UnivariateRationalRecurrenceOperator(UnivariateRationalOreOperator):
    """
    Element of an Ore algebra K(x)[S], where S is the shift x->x+1.
    """

    def __init__(self, parent, is_gen = False, construct=False): 
        UnivariateRationalOreOperator.__init__(self, parent, is_gen, construct)

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

class UnivariateRationalQRecurrenceOperator(UnivariateRationalOreOperator):
    """
    Element of an Ore algebra K(x)[S], where S is the shift x->q*x for some q in K.
    """

    def __init__(self, parent, is_gen = False, construct=False): 
        UnivariateRationalOreOperator.__init__(self, parent, is_gen, construct)

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

