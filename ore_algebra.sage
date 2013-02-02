
from sage.rings.ring import Algebra
from sage.structure.element import RingElement

load("ore_operator.sage")

class OreAlgebra(sage.algebras.algebra.Algebra):
    """
    An Ore algebra is a noncommutative polynomial ring whose elements are
    interpreted as operators.

    An Ore algebra has the form `A=R[\partial_1,\partial_2,\dots,\partial_n]`
    where `R` is a commutative ring and `\partial_1,\dots,\partial_n` are
    indeterminates.  For each of them, there is an associated automorphism
    `\sigma:R\\rightarrow R` and a skew-derivation `\delta:R\\rightarrow R`
    satisfying `\delta(a+b)=\delta(a)+\delta(b)` and
    `\delta(ab)=\delta(a)b+\sigma(a)\delta(b)` for all `a,b\in R`.

    The generators `\partial_i` commute with each other, but not with elements
    of the base ring `R`. Instead, we have the commutation rules `\partial u =
    \sigma(u) \partial + \delta(u)` for all `u\in R`.

    A typical example of an Ore algebra is the ring of linear differential
    operators with rational function coefficients in one variable,
    e.g. `A=QQ[x][D]`. Here, `\sigma` is the identity and `\delta` is the
    standard derivation `d/dx`.

    To create an Ore algebra, supply a base ring and one or more
    generators. Each generator has to be given in form of a triple
    ``(name,sigma,delta)`` where ``name`` is the desired name of the variable
    (used for printout), ``sigma`` and ``delta`` are arbitrary callable objects
    which applied to the base ring return other base ring elements in accordance
    with the relevant laws. It is not checked whether they do.

    ::

      sage: R.<x> = QQ['x']
      sage: K = R.fraction_field()
      # This creates an Ore algebra of linear differential operators
      sage: A.<D> = OreAlgebra(K, ('D', lambda p: p, lambda p: p.derivative(x)))
      sage: A
      Univariate Ore algebra in D over Fraction Field of Univariate Polynomial Ring in x over Rational Field 
      # This creates an Ore algebra of linear recurrence operators
      sage: A.<S> = OreAlgebra(K, ('S', lambda p: p(x+1), lambda p: K.zero()))
      sage: A
      Univariate Ore algebra in S over Fraction Field of Univariate Polynomial Ring in x over Rational Field 

    For differential and shift operators (the most common cases), there are
    shortcuts: Supplying as generator a string ``'Dx'`` consisting of ``'D'``
    followed by the name of a generator of the base ring represents a derivation
    with respect to that generator (i.e., ``sigma=lambda p: p`` and
    ``delta=lambda p: p.derivative(x)``).  Likewise, a string ``'Sx'``
    consisting of ``'S'`` followed by the name of a generator of the base ring
    represents a shift with respect to that generator (i.e., ``sigma=lambda p:
    p(x+1)`` and ``delta=lambda p:K.zero()``).

    ::

      sage: R.<x> = QQ['x']
      sage: K = R.fraction_field()
      # This creates an Ore algebra of differential operators
      sage: A.<Dx> = OreAlgebra(K, 'Dx')
      sage: A
      Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field 
      # This creates an Ore algebra of linear recurrence operators
      sage: A.<Sx> = OreAlgebra(K, 'Sx')
      sage: A
      Univariate Ore algebra in Sx over Fraction Field of Univariate Polynomial Ring in x over Rational Field 

    Ore algebras support coercion from their base rings.  Furthermore, an Ore
    algebras `A` know how to coerce commutative polynomials `p` to elements of
    `A` if the generators of the parent of `p` have the same names as the
    generators of `A`, and the base ring of the parent of `p` admits a coercion
    to the base ring of `A`.

    """
    _no_generic_basering_coercion = False
    
    def __init__(self, base_ring, *gens, **kargs):
        """
        """
        R = self._base_ring = base_ring
        gens = self._gens = list(gens)

        # expand generator shortcuts
        for i in xrange(len(gens)):
            if type(gens[i]) == type(""):
                if gens[i][0] == 'D':
                    x = R(gens[i][1:])
                    gens[i] = (gens[i], lambda p: p, lambda p: p.derivative(x))
                elif gens[i][0] == 'S':
                    x = R(gens[i][1:]); z = R.zero()
                    gens[i] = (gens[i], lambda p: p.subs(x=x+1), lambda p: z)
                else:
                    raise TypeError, "unexpected generator declaration"
            elif len(gens[i]) != 3:
                raise TypeError, "unexpected generator declaration"

        # try to recognize standard operators
        is_shift = [False for q in gens]; is_qshift = [False for q in gens]; is_derivation = [False for q in gens]
        subgens = list(R.gens()); B = R.base_ring()
        try:
            while B.base_ring() is not B:
                for x in B.gens():
                    subgens.append(x)
                B = B.base_ring()
        except:
            pass
            
        for i in xrange(len(gens)):
            if all(gens[i][1](x) == x for x in subgens):
                deltas = [gens[i][2](x) for x in subgens]
                if sum(deltas) == 1 and all( (x==1 or x==0) for x in deltas):
                    is_derivation[i] = True
            if all(gens[i][2](x) == 0 for x in subgens):
                shifts = [gens[i][1](x) - x for x in subgens]
                shifts = [x for x in shifts if x != 0]
                if len(shifts) == 1 and shifts[0] == 1:
                    is_shift[i] = True
                else:
                    shifts = [gens[i][1](x)/x for x in subgens]
                    shifts = [x for x in shifts if x != 1]
                    if len(shifts) == 1 and gens[i][1](shifts[0]) == shifts[0]:
                        is_qshift[i] = True

        # select element class
        if kargs.has_key("element_class"):
            self._operator_class = kargs["element_class"]
        elif len(gens) > 1:
            raise NotImplementedError, "Multivariate Ore algebras still under construction"
        elif not R.is_field():
            raise NotImplementedError, "Ore algebras with non-fields as base ring still under construction"
        elif not R.is_integral_domain():
            raise TypeError, "Base rings of Ore algebras must be integral domains"
        elif not R == R.fraction_field() or not sage.rings.polynomial.polynomial_ring.is_PolynomialRing(R.base()):
            self._operator_class = UnivariateOreOperator
        elif is_qshift[0]:
            self._operator_class = UnivariateRationalQRecurrenceOperator
        elif is_shift[0]:
            self._operator_class = UnivariateRationalRecurrenceOperator
        elif is_derivation[0]:
            self._operator_class = UnivariateRationalDifferentialOperator
        else:
            self._operator_class = UnivariateRationalOreOperator

    # information extraction

    def is_integral_domain(self, proof = True):
        """
        Returns True because Ore algebras are always integral domains.
        """
        return True

    def is_noetherian(self):
        """
        Returns True because Ore algebras are always noetherian. 
        """
        return True

    def construction(self):
        from sage.categories.pushout import PolynomialFunctor
        return PolynomialFunctor(self.variable_name(), sparse=self.__is_sparse), self.base_ring()

    def _coerce_map_from_(self, P):
        """
        """
        raise NotImplementedError

    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce this value when
        evaluated.
        """
        raise NotImplementedError

    def _is_valid_homomorphism_(self, codomain, im_gens):
        """
        """
        raise NotImplementedError

    def __hash__(self):
        """
        """
        # should be faster than just relying on the string representation
        try:
            return self._cached_hash
        except AttributeError:
            pass
        h = self._cached_hash = hash((self.base_ring(),self.variable_name()))
        return h

    def _repr_(self):
        """
        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        raise NotImplementedError

    def _latex_(self):
        """
        """
        raise NotImplementedError


    def sigma(self, n=0):
        """
        """
        raise NotImplementedError

    def delta(self, n=0):
        """
        """
        raise NotImplementedError

    def var(self, n=0):
        """
        """
        raise NotImplementedError
        
    def variable_names_recursive(self, depth=sage.rings.infinity.infinity):
        r"""
        Returns the list of variable names of this and its base rings, as if
        it were a single multi-variate polynomial.
        
        EXAMPLES::
        
            sage: R = QQ['x']['y']['z']
            sage: R.variable_names_recursive()
            ('x', 'y', 'z')
            sage: R.variable_names_recursive(2)
            ('y', 'z')
        """
        if depth <= 0:
            return ()
        elif depth == 1:
            return self.variable_names()
        else:
            my_vars = self.variable_names()
            try:
               return self.base_ring().variable_names_recursive(depth - len(my_vars)) + my_vars
            except AttributeError:
                return my_vars
                
    def characteristic(self):
        """
        Return the characteristic of this Ore algebra, which is the
        same as that of its base ring.
        """
        return self.base_ring().characteristic()

    def gen(self, n=0):
        """
        Return the indeterminate generator(s) of this Ore algebra. 
        """
        if n < 0 or n >= self.ngens():
            raise IndexError("No such generator.")
        return self.gens()[n]

    def gens(self):
        """
        Return a list of generators of this Ore algebra. 
        """
        raise NotImplementedError

    def gens_dict(self):
        """
        Returns a dictionary whose keys are the variable names of this
        Algebra as strings and whose values are the corresponding
        generators.
        """
        raise NotImplementedError

    def is_finite(self):
        """
        Return False since Ore algebras are not finite (unless the base ring is 0).
        """
        R = self.base_ring()
        if R.is_finite() and R.order() == 1:
            return True
        return False

    def is_exact(self):
        """
        This algebra is exact iff its base ring is
        """
        return self.base_ring().is_exact()

    def is_field(self, proof = True):
        """
        Returns False since Ore algebras are not fields.
        """
        return False
        
    def krull_dimension(self):
        """
        Returns the Krull dimension of this algebra, which is the Kull dimension of the base ring
        plus the number of generators of this algebra. 
        """
        return self.base_ring().krull_dimension() + self.ngens()
        
    def ngens(self):
        """
        Return the number of generators of this Ore algebra
        """
        return len(self._gens)

    # generation of elements
        
    def _element_constructor_(self, x=None, check=True, is_gen = False, construct=False, **kwds):
        r"""
        Convert ``x`` into this algebra, possibly non-canonically.
        """
        raise NotImplementedError

    def one(self):
        """
        """
        raise NotImplementedError

    def zero(self):
        """
        """
        raise NotImplementedError
        
    def random_element(self, degree=2, *args, **kwds):
        r"""
        Return a random operator. 
        """
        raise NotImplementedError

    # generation of related parent objects

    def associated_commutative_algebra(self):
        """
        Returns a polynomial ring with the same base ring as this algebra and whose generators
        have the same name as this Ore algebra's generators.

        ::

           sage: R.<x> = QQ['x']
           sage: A = OreAlgebra(K.fraction_field(), "Dx")
           sage: A
           Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field
           sage: A.associated_commutative_algebra()
           Univariate Polynomial Ring in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field
        
        """
        try:
            return self._commutative_ring
        except AttributeError:
            pass
        R = self._commutative_ring = PolynomialRing(self.base_ring(), map(str, self.gens()))
        return R

    def base_extend(self, R):
        """
        """
        raise NotImplementedError

    def change_ring(self, R):
        """
        """
        raise NotImplementedError

    def change_var(self, var, n=0):
        r"""
        """
        raise NotImplementedError

    def change_sigma(self, sigma, n=0):
        """
        """
        raise NotImplementedError

    def change_delta(self, delta, n=0):
        """
        """
        raise NotImplementedError

    def change_var_sigma_delta(self, var, sigma, delta, n=0):
        """
        """
        raise NotImplementedError
        
##########################################################################################################

def guess_rec(data, n, S):
    """
    """
    raise NotImplementedError

def guess_deq(data, x, D):
    """
    """
    raise NotImplementedError


