
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
from numpy.polynomial.polynomial import Polynomial
from numpy import poly1d

class DFiniteFunctionRing(Algebra):
    """
    A Ring of Dfinite objects (functions or sequences)
    """
    _no_generic_basering_coercion = False
    
    def __init__(self, ore_algebra, name=None, element_class=None, category=None):
        """
        """
        self._ore_algebra = ore_algebra 
        self._base_ring = ore_algebra.base_ring()

        
    def _element_constructor_(self, x=None, check=True, is_gen = False, construct=False, **kwds):
        r"""
        Convert ``x`` into this algebra, possibly non-canonically.
        """
        raise NotImplementedError

    def is_integral_domain(self, proof = True):
        """
        """
        return self._ore_algebra.is_integral_domain()


    def is_noetherian(self):
        """
        """
        return self.ore_algebra().is_noetherian()
            
    def construction(self):
        """
        """
        return self._ore_algebra.construction()

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
        r = "D-finite function ring defined by the following Ore Algebra: "
        r = r + self._ore_algebra._repr_()
        return r

    def _latex_(self): #gibt das Element in Latex Code aus
        """
        """
        raise NotImplementedError

    def base_extend(self, R):
        """
        Creates the Ore algebra obtained from ``self`` by replacing the base ring by `R`
        """
        return self._ore_algebra.base_extend(R)

    def base_ring(self):
        """
        Return the base ring over which the D-finite function ring is defined
        """
        return self._base_ring
    

    def ore_algebra(self):
        """
        Return the ore algebra which defines the D-finite function ring
        """
        return self._ore_algebra
                
    def characteristic(self):
        """
        Return the characteristic of this Ore algebra, which is the
        same as that of its base ring.
        """
        return self._base_ring.characteristic()

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
        return self._base_ring.is_exact()

    def is_field(self, proof = True):
        """
        """
        return False
        
    def random_element(self, typ = "polynomial", degree=2 ) : #, *args, **kwds):
        #still in process - does not work so far !!!
        r"""
        Return a random operator. 
        """
        A = self.ore_algebra().random_element()
        if typ == "polynomial" :
            B = A.polynomial_solutions([1,2],degree)
            C = list(min(B))
            p = poly1d(C)
            print p
            return p
        elif typ == "rational" :
            B = A.rational_solutions([1,2],degree)
            C = list(min(B))
            p = poly1d(C)
            print p
            return p
        else:
            raise NotImplementedError
        
    def change_base_ring(self,R):
        """
        Return a copy of "self" but with the base ring R
        """
        if R is self._base_ring:
            return self
        else:
            D = DFiniteFunctionRing(self._ore_algebra.change_ring(R))
            return D
        

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
        RingElement.__init__(self, parent)
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
        D = self.parent().change_base_ring(R)
        result = DFiniteFunction(D)
        return result
        

    def __getitem__(self, n):
        raise NotImplementedError

    def __setitem__(self, n, value):
        """
        """
        raise IndexError("D-finite functions are immutable")

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
        return self._repr_()

    def _repr_(self): #still in process - works just as an information for us
        return "some D-finite function"

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
    
    def initial_values(self):
        """
        return the initial values of self
        """
        raise NotImplementedError


#############################################################################################################
    
class UnivariateDFiniteFunction(DFiniteFunction):
    """
    D-finite function in a single (differentiable or discrete) variable.
    """

    def __init__(self, parent, ann, initial_val, is_gen=False, construct=False, cache=True):
        super(DFiniteFunction, self).__init__(parent)
        self._ann = parent._ore_algebra(ann)                #annihilation polynomial

        if len(initial_val) < self._ann.order():
            raise ValueError, "not enough initial values given"
        
        self._initial_values = initial_val
    # action
    

    def __call__(self, *x, **kwds):
        """
        """
        raise NotImplementedError

    # tests

    def __is_zero__(self):
        """
        """
        if all(x == 0 for x in self._initial_values):
            return true
        return false

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

    def _repr_(self, name=None):
        """
        """
        r = "Univariate D-finite function defined by the annihilating polynomial "
        r = r + self._ann._repr() + " and the initial values "
        r = r + '[%s]' % ', '.join(map(str, self._initial_values))
        return r

    def _repr_(self):
        """
        """
        return _repr_(self)

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
        """
        return the sum of two D-finite functions
        
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteFunction(C, "Sn**2 - Sn - 1", [0,1])
            sage: b = UnivariateDFiniteFunction(C, "(n**2+3)*Sn**3 + (4*n - 10)*Sn - 1", [0,1,2])
            sage: c = a+b
        """
        sum_ann = self._ann.lclm(right._ann)
        
        n = sum_ann.order()
        int_val_self = self._ann.to_list(self._initial_values,n)
        int_val_right = right._ann.to_list(right._initial_values,n)
        int_val_sum = [x+y for x, y in zip(int_val_self, int_val_right)]
        
        sum = UnivariateDFiniteFunction(self.parent(), sum_ann, int_val_sum)
        return sum
    
    def _neg_(self):
        """
        return the negative of a D-finite function
        """
        neg_int_val = [-x for x in self._initial_values]
        neg = UnivariateDFiniteFunction(self.parent(), self._ann, neg_int_val)
        return neg

    def _lmul_(self, left):#kann ich die nicht kommutativen FÃlle auch auf symmetric_product zurückführen?
        return left*self
    
    def _rmul_(self, right):
        return self*right

    def _mul_(self, right):
        """
        return the product of two D-finite functions
        
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteFunction(C, "Sn**2 - Sn - 1", [0,1])
            sage: b = UnivariateDFiniteFunction(C, "(n**2+3)*Sn**3 + (4*n - 10)*Sn - 1", [0,1,2])
            sage: a.to_list(10)
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
            sage: b.to_list(10)
            [0, 1, 2, 10/3, 13/4, 26/21, -19/72, -117/532, 977/7056, 3851/53352]
            sage: c = a+b
            sage: c.to_list(10)
            [0, 2, 3, 16/3, 25/4, 131/21, 557/72, 6799/532, 149153/7056, 1817819/53352]
        """
        prod_ann = self._ann.symmetric_product(right._ann)
        
        n = prod_ann.order()
        int_val_self = self._ann.to_list(self._initial_values,n)
        int_val_right = right._ann.to_list(right._initial_values,n)
        int_val_prod = [x*y for x, y in zip(int_val_self, int_val_right)]
        
        prod = UnivariateDFiniteFunction(self.parent(), prod_ann, int_val_prod)
        return prod

    def __invert__(self):
        raise NotImplementedError

    def interlace(self, right):
        """
        return the interlaced sequence of the two sequences self (e.g. a0,a1,a2,...)
        and right (e.g. b0,b1,b2,..). The result is then of the form a0,b0,a1,b1,....
        
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteFunction(C, "Sn**2 - Sn - 1", [0,1])
            sage: k = UnivariateDFiniteFunction(C, "Sn - 1", [4])
            sage: a.to_list(10)
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
            sage: k.to_list(10)
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
            sage: i = a.interlace(k)
            sage: i.to_list(10)
            [0, 4, 1, 4, 1, 4, 2, 4, 3, 4]
        """
        interlacing_ann = self._ann.annihilator_of_interlacing(right._ann)
        
        n = interlacing_ann.order()
        int_val_self = self._ann.to_list(self._initial_values,ceil(n/2))
        int_val_right = right._ann.to_list(right._initial_values,floor(n/2))
        int_val_interlacing =  result = [None]*n
        int_val_interlacing[::2] = int_val_self
        int_val_interlacing[1::2] = int_val_right
    
        interlacing = UnivariateDFiniteFunction(self.parent(), interlacing_ann, int_val_interlacing)
        return interlacing
 
    # evaluation
    
    def to_list(self,n):        #only makes sense for sequences
        """
        return the first n terms of the sequences self
        
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteFunction(C, "Sn**2 - Sn - 1", [0,1])
            sage: a.to_list(10)
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        """
        return self._ann.to_list(self._initial_values,n)
    
    def __getitem__(self,n):    #only makes sense for sequences
        """
        return the n-th term of the sequence self
        
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteFunction(C, "Sn**2 - Sn - 1", [0,1])
            sage: a[5]  #the 5th Fibonacci number
            5
            sage: a[10] #the 10th Fibnoacci number
            55
        """
        l = self.to_list(n+1)
        return l[n]

    

    def expand(self, n):
        # series coefficients or sequence terms
        raise NotImplementedError

    def evaluate(self, z, n):
        # numerically by analytic continuation
        raise NotImplementedError
    
    ########################## testing #########################
A = OreAlgebra(FractionField(ZZ['n']),'Sn')
C = DFiniteFunctionRing(A)      #C is our D-finite function ring
a = UnivariateDFiniteFunction(C,"Sn**2-Sn-1",[0,1]) #represents the Fibonacci Numbers
b = UnivariateDFiniteFunction(C, "(n**2+3)*Sn**3 + (4*n - 10)*Sn - 1", [0,1,2]) #some ugly sequence
k = UnivariateDFiniteFunction(C, "Sn -1", [4])      #constant sequence 4,4,4,...
print "First 10 terms of a: "
print a.to_list(10)
print "First 10 terms of b: "
print b.to_list(10)
print "First 10 terms of k: "
print k.to_list(10)

s = a+b
print "First 10 terms of a+b: "
print s.to_list(10)

p = a*k
print "First 10 terms of a*k: "
print p.to_list(10)

i = a.interlace(k)
print "First 10 terms of a interlaced with k: "
print i.to_list(10)





