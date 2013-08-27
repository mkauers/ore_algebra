
"""
dfinite_function
================

"""

#############################################################################
#  Copyright (C) 2013 Manuel Kauers (mkauers@gmail.com),                    #
#                     Maximilian Jaroschek (mjarosch@risc.jku.at),          #
#                     Fredrik Johansson (fjohanss@risc.jku.at).             #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

from sage.rings.ring import Algebra

class DFiniteFunctionRing(Algebra):
    """
    A Ring of Dfinite objects (functions or sequences)
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
        raise NotImplementedError

    def is_integral_domain(self, proof = True):
        """
        """
        return False

    def is_noetherian(self):
        """
        """
        return False
            
    def construction(self):
        """
        """
        raise NotImplementedError

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
        """
        """
        raise NotImplementedError

    def base_extend(self, R):
        """
        """
        raise NotImplementedError

    def ore_algebra(self):
        """
        """
        raise NotImplementedError
                
    def characteristic(self):
        """
        Return the characteristic of this Ore algebra, which is the
        same as that of its base ring.
        """
        return self.base_ring().characteristic()

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
        """
        """
        return self.base_ring().is_exact()

    def is_field(self, proof = True):
        """
        """
        return False
        
    def random_element(self, degree=2, *args, **kwds):
        r"""
        Return a random operator. 
        """
        raise NotImplementedError

####################################################################################################


class DFiniteFunction(RingElement):
    """
    An object depending on one or more differential and one or more discrete variables
    defined by an annihilating holonomic system and a suitable set of initial conditions. 
    """

    # constructor

    def __init__(self, parent, is_gen = False, construct=False): 
        """
        """
        self._is_gen = is_gen

    def __copy__(self):
        """
        Return a "copy" of self. This is just self, since D-finite functions are immutable. 
        """
        return self

    # action

    def __call__(self, *x, **kwds):
        """
        Lets ``self`` act on ``x`` and returns the result.
        ``x`` may be either a constant, then this computes an evaluation,
        or a (suitable) expression, then it represents composition and we return a new DFiniteFunction object.         
        """
        raise NotImplementedError

    # tests

#    ???
#    def __richcmp__(left, right, int op):
#        return (<Element>left)._richcmp(right, op)

    def __nonzero__(self):
        """
        """
        raise NotImplementedError

    def _is_atomic(self):
        """
        """
        raise NotImplementedError

    def is_unit(self):
        r"""
        Return True if this function is a unit.
        """
        raise NotImplementedError
       
    def is_gen(self):
        r"""
        Returns False; the parent ring is not finitely generated. 
        """
        return False
    
    def prec(self):
        """
        Return the precision of this object. 
        """
        return NotImplementedError
    
    # conversion

    def change_variable_name(self, var):
        """
        Return a new function over the same base ring but in a different
        variable.
        """
        return NotImplementedError
        
    def change_ring(self, R):
        """
        Return a copy of this function but with base ring R, if at all possible.
        """
        return NotImplementedError

    def __getitem__(self, n):
        raise NotImplementedError

    def __setitem__(self, n, value):
        """
        """
        raise IndexError("D-finite functions are immutable")

    def __iter__(self):
        return NotImplementedError

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
        raise NotImplementedError

    def list(self):
        raise NotImplementedError

    # arithmetic

    def __invert__(self):
        """
        works if 1/self is again d-finite. 
        """
        return NotImplementedError

    def __div__(self, right):
        """
        This is division, not division with remainder. Works only if 1/right is d-finite. 
        """
        return self*right.__invert__()

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
        raise NotImplementedError

    def __mod__(self, other):
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

    # part extraction functions

    def annihilator(self):
        """
        """
        raise NotImplementedError


#############################################################################################################
    
class UnivariateDFiniteFunction(DFiniteFunction):
    """
    D-finite function in a single (differentiable or discrete) variable.
    """

    def __init__(self, parent, is_gen=False, construct=False, cache=True): 
        DFiniteFunction.__init__(self, parent, is_gen, construct)

    # action

    def __call__(self, *x, **kwds):
        """
        """
        raise NotImplementedError

    # tests

    def __nonzero__(self):
        """
        """
        raise NotImplementedError

    def _is_atomic(self):
        """
        """
        raise NotImplementedError

    # conversion

    def __iter__(self):
        """
        """
        raise NotImplementedError

    def __float__(self):
        """
        """
        return NotImplementedError

    def __int__(self):
        """
        """
        return NotImplementedError

    def _integer_(self, ZZ):
        """
        """
        return NotImplementedError

    def _rational_(self):
        """
        """
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
        """
        """
        return self._repr()

    def _latex_(self, name=None):
        """
        """
        raise NotImplementedError
        
    def _sage_input_(self, sib, coerced):
        """
        """
        raise NotImplementedError

    def dict(self):
        raise NotImplementedError

    def list(self):
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
        raise NotImplementedError

    # evaluation

    def expand(self, n):
        # series coefficients or sequence terms
        raise NotImplementedError

    def evaluate(self, z, n):
        # numerically by analytic continuation
        raise NotImplementedError


