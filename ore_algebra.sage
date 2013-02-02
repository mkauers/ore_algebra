
class OreAlgebra(sage.algebras.algebra.Algebra):
    """
    Univariate polynomial ring over a ring.
    """
    _no_generic_basering_coercion = False
    
    def __init__(self, base_ring, name=None, element_class=None, category=None):
        """
        """
        raise NotImplementedError
        
    def _element_constructor_(self, x=None, check=True, is_gen = False, construct=False, **kwds):
        r"""
        Convert ``x`` into this algebra, possibly non-canonically.
        """

    def is_integral_domain(self, proof = True):
        return True

    def is_noetherian(self):
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
        # should be faster than just relying on the string representation
        try:
            return self._cached_hash
        except AttributeError:
            pass
        h = self._cached_hash = hash((self.base_ring(),self.variable_name()))
        return h

    def _repr_(self):
        try:
            return self._cached_repr
        except AttributeError:
            pass
        raise NotImplementedError

    def _latex_(self):
        raise NotImplementedError

    def base_extend(self, R):
        raise NotImplementedError

    def change_ring(self, R):
        raise NotImplementedError

    def change_var(self, var):
        r"""
        """
        raise NotImplementedError

    def sigma(self, n=0):
        raise NotImplementedError

    def delta(self, n=0):
        raise NotImplementedError

    def change_sigma(self, sigma, n=0):
        raise NotImplementedError

    def change_delta(self, sigma, n=0):
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
        Return False since polynomial rings are not finite (unless the base
        ring is 0.)
        """
        R = self.base_ring()
        if R.is_finite() and R.order() == 1:
            return True
        return False

    def is_exact(self):
        return self.base_ring().is_exact()

    def is_field(self, proof = True):
        """
        """
        return False
        
    def is_sparse(self):
        """
        """
        return self.__is_sparse

    def krull_dimension(self):
        """
        """
        return self.base_ring().krull_dimension() + self.ngens()
        
    def ngens(self):
        """
        Return the number of generators of this Ore algebra
        """
        raise NotImplementedError
        
    def random_element(self, degree=2, *args, **kwds):
        r"""
        Return a random operator. 
        """
        raise NotImplementedError

##########################################################################################################

def guess_rec(data, n, S):
    raise NotImplementedError

def guess_deq(data, x, D):
    raise NotImplementedError


