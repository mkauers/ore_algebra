
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
from numpy import random
from math import factorial

class DFiniteFunctionRing(Algebra):
    """
    A Ring of Dfinite objects (functions or sequences)
    """
    
    _no_generic_basering_coercion = False
    
    def __init__(self, ore_algebra, domain = QQ, codomain = NN, name=None, element_class=None, category=None):
        """
        Input:
            - ore_algebra ... an OreAlgebra
            - domain ... (optional) the ring where the initial values live
            - codomain ... (optional) domain over which i consider the sequence 
                            if the codomain is ZZ there exist negative indices
                            if it is NN e.g. a(-3) doesn't exist

        Output:
        a DFiniteFunctionRing with an OreAlgebra, a domain and a codomain.
        
        """
        self._ore_algebra = ore_algebra
        self._base_ring = ore_algebra.base_ring()
        self._domain = domain
        if ore_algebra.is_D() and codomain == ZZ:
            raise NotImplementedError, "D-finite functions with negative powers are not implemented"
        self._codomain = codomain
        if codomain == NN:
            self._backward_calculation = false
        else:
            self._backward_calculation = true
    
    
    def _element_constructor_(self, x=None, check=True, is_gen = False, construct=False, **kwds):
        r"""
        Convert ``x`` into this algebra, possibly non-canonically.
        If ``x`` is a list of data it is interpreted as the first coefficients of
        a sequence or a power series and the method ``guess``is called in order to
        find a suitable Ore Operator
        If ``x`` can be converted into a floating point number it is intpreted as
        the constant sequence x,x,x,... or the constant function f(z) = x
        If ``x``can be converted into the base ring then it is interpreted as 
        the sequence (a_n) := x(n)
        """
        
        n = self.ore_algebra().is_S()
        
        #conversion for D-finite functions:
        if isinstance(x,DFiniteFunction):
            if self._coerce_map_from_(x.parent()):
                if n is not false:
                    if self._codomain == ZZ and x.parent()._codomain == NN:
                        raise TypeError, str(x) + " could not be converted - can't convert series over codomain NN into series over codomain ZZ"
                    else:
                        return UnivariateDFiniteSequence(self,x._ann,x.initial_conditions())
                else:
                    return UnivariateDFiniteFunction(self,x._ann,x.initial_conditions())
            else:
                raise TypeError, str(x) + " could not be converted - the underlying Ore Algebras don't match"
    
        #conversion for lists:
        elif type(x) == list:
            try:
                A = guess(x,self._ore_algebra)
            except:
                raise ValueError, "no relation found"

            int_val = x
    
            if n is not false:
                return UnivariateDFiniteSequence(self,A,int_val)
            else:
                return UnivariateDFiniteFunction(self,A,int_val)
                
        #conversion for numbers
        else:
            try:
                x = self._domain(x)
            except:
                #conversion for polynomials
                try:
                    x = self.base_ring()(x)
                except:
                    raise ValueError, str(x) + " could not be converted"
            
                if n is not false:
                    Sn = self.ore_algebra().gens()[0]
                    A = self.ore_algebra()(x(n)*Sn - x(n+1))
                    ord = A.order()
                    
                    sing = [a for a in range(ord)] + [a for a in A.singularities(self._backward_calculation)]
                    if sing:
                        s_max = max(sing) + ord
                        s_min = min(sing)
                    else:
                        s_max = ord
                        s_min = 0
                    int_val = {a:x(a) for a in range(s_min,s_max+1)}
                
                    return UnivariateDFiniteSequence(self,A,int_val)
                else:
                    y = self.ore_algebra().is_D()
                    Dy = self.ore_algebra().gens()[0]
                    R = self.ore_algebra().base_ring().change_var('n')
                    OreAlg = OreAlgebra(R,'Sn')
                    
                    A = x(y)*Dy - x.derivative()
                    A_S = A.to_S(OreAlg)
                    
                    sing = [a for a in A_S.singularities(false)]
                    if sing:
                        s = max(sing)
                    else:
                        s = 0

                    int_val = {a:(x.derivative(a)(0)/factorial(a)) for a in range(s + 2)}
                    seq = UnivariateDFiniteSequence(DFiniteFunctionRing(OreAlg,QQ,NN),A_S, int_val)
                    
                    return UnivariateDFiniteFunction(self,A,seq)
                        
            if n is not false:
                Sn = self.ore_algebra().gens()[0]
                return UnivariateDFiniteSequence(self,Sn-1,{0:x})
            else:
                Dy = self.ore_algebra().gens()[0]
                return UnivariateDFiniteFunction(self,Dy,{0:x,1:0})

    def __eq__(self,right):
        """
        tests if two DFiniteFunctionRings are equal
        Input: 
            -right ... a DFiniteFunctionRing
            
        """
        try:
            return (self.ore_algebra() == right.ore_algebra() and self.domain() == right.domain() and self._codomain == right._codomain)
        except:
            return false

    def is_integral_domain(self, proof = True):
        """
        - false for DFiniteFunctionRing in the sequential case
        - true for DFiniteFunctionRing in the differential case
        """
        if self.ore_algebra().is_S():
            return False
        elif self.ore_algebra().is_D():
            return True
        else:
            raise NotImplementedError

    def is_noetherian(self):
        """
        """
        raise NotImplementedError
    
    def is_commutative(self):
        """
        the function ring as well as the sequence ring are commutative
        """
        return True
            
    def construction(self):
        """
        """
        raise NotImplementedError

    def _coerce_map_from_(self, P):
        """
        If P is a DFiniteFunctionRing it is sufficient to ask to coerce the ore_algebras
        else it asks to coerce the ore_algebra from self to P
        """
        if isinstance(P,DFiniteFunctionRing):
            return self._ore_algebra._coerce_map_from_(P.ore_algebra())
        return self._ore_algebra._coerce_map_from_(P)

    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce this value when
        evaluated.
        """
        return sib.name('DFiniteFunctionRing')(sib(self.ore_algebra()),sib(self.domain()))

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
        h = self._cached_hash = hash((self.base_ring(),self.base_ring().variable_name()))
        return h

    def _repr_(self):
        """
        returns in words what object you are working with if you ask 
        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        r = "Ring of D-finite functions over "
        r = r + self._base_ring._repr_()
        return r

    def _latex_(self):
        """
        describes a DFiniteFunctionRing in latex
        """
        return "\mathcal{D}(" + self._base_ring._latex_() + ")"

    def base_extend(self, R):
        """
        Creates the Ore algebra obtained from ``self`` by replacing the base ring by `R`
        """
        D = DFiniteFunctionRing(self._ore_algebra.base_extend(R),self._domain)
        return D
    
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

    def domain(self):
        """
        Return the domain where the function values live
        """
        return self._domain
    
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
        R = self._base_ring
        if R.is_finite() and R.order() == 1:
            return True
        return False

    def is_exact(self):
        """
        is true if its OreAlgebra and its domain are exact
        """
        return (self.ore_algebra().is_exact()) & (self._domain.is_exact())

    def is_field(self, proof = True):
        """
        A DFiniteFunctionRing is not a field
        """
        return False
        
    def random_element(self, degree=2 ) : #, *args, **kwds):
        r"""
        Return a random D-finite object.
        
        Input:
            - degree ... (optional) the degree of the random ore operator
            
        Output:
            a random DFiniteSequence if the OreAlgebra of self is in S
            else a random DFiniteFunction
        
        """
        A = self.ore_algebra().random_element(degree)
        keys = [x for x in range(degree)] + [x for x in A.singularities(self._backward_calculation) if x >= degree or x<0]
        int_val = {k:randint(-100, 100) for k in keys}
        if self.ore_algebra().is_S():
            return UnivariateDFiniteSequence(self,A,int_val)
        else:
            return UnivariateDFiniteFunction(self,A,int_val)
    

    def change_base_ring(self,R):
        """
        Return a copy of "self" but with the base ring R
        """
        if R is self._base_ring:
            return self
        else:
            D = DFiniteFunctionRing(self._ore_algebra.change_ring(R), self._domain)
            return D



####################################################################################################


class DFiniteFunction(RingElement): #inherited from RingElement, abstract class
    """
    An object depending on one or more differential and one or more discrete variables
    defined by an annihilating holonomic system and a suitable set of initial conditions. 
    """

    # constructor

    def __init__(self, parent, ann, initial_val, is_gen = False, construct=False, cache=True):
        """
        Input:
            - parent ... a DFiniteFunctionRing
            - ann ... the operator in the corresponding OreAlgebra annihilating the sequence or the function
            - initial_val ... a list of initial values, determining the sequence or function
                              for sequences it has to be at least ord(ann) long (when no singularities occur)
                              if the sequence contains singularities then the initial values
                              have to be given in form of a dictionary containing the intial values and the singularities
                              for functions it has to be at least ord(ann) + 1 long
                                  and the i-th initial value has to be the i-th coefficient 
                                  of the taylor series of the function
                             for functions also a Dfinite sequence can be given as initial_val
                             
        Output:
        either 
            - a DFinite Sequence determined by ann and its initial values
        or 
            - a DFinite Function determined by ann and its initial values(saved as a DFinite Sequence)
                            
        """
        RingElement.__init__(self, parent)
        self._is_gen = is_gen
    
        self._ann = parent._ore_algebra(ann)
        ord = self._ann.order()
        s = [x for x in self._ann.singularities(self.parent()._backward_calculation) if x >= ord or x < 0]
        
        if parent.ore_algebra().is_S():
            if type(initial_val) == list:
                self._initial_values = {i:initial_val[i] for i in range(len(initial_val))}
            else:
                self._initial_values = initial_val
                
            if len(initial_val) < ord + len(s):
                if parent._backward_calculation is True:
                    raise ValueError, "not enough initial conditions given"
                
                #sequence comes from a d-finite function
                diff = ord + len(s) - len(initial_val)
                zeros = {i:0 for i in range(-diff,0)}
                self._initial_values.update(zeros)
                
        elif parent.ore_algebra().is_D():
            R = parent.ore_algebra().base_ring().change_var('n')
            A = OreAlgebra(R,'Sn')
            init_num = ord +1
            if isinstance(initial_val,UnivariateDFiniteSequence):
                self._initial_values = initial_val
                self.s_ann = initial_val._ann
            else:
                if len(initial_val) < init_num:
                    raise ValueError, "not enough initial conditions given"
                B = DFiniteFunctionRing(A,QQ,NN)
                self.s_ann = self._ann.to_S(A)
                self._initial_values = UnivariateDFiniteSequence(B,self.s_ann, initial_val)
        
        else:
            raise ValueError, "not a suitable D-finite function ring"

    
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
    
    def singularities(self,backwards = false):
        """
        returns the integer singularities of the annihilation operator of self,
        to be more precise:
        * If backwards is false, only the roots of the leading coefficient of the
        annihilator of self are returned shifted by its order
        * If backwards is true, additionally to the singularities from the leading
        coefficent, also the roots of the constant coefficient are returned (but not shifted)
        """
        return self._ann.singularities(backwards)

       # tests

#    ???
#    def __richcmp__(left, right, int op):
#        return (<Element>left)._richcmp(right, op)

    def __eq__(self,right):
        """
        asks if two DFiniteFunctions are equal
        """
        if (self.__is_zero__()) & (right.__is_zero__()):
            return true
        return (self._ann.monic() == right._ann.monic()) & (self.initial_conditions() == right.initial_conditions())
    
    def __ne__(self,right):
        """
        returns true if self and right are not equal, 
        else it returns false
        """
        return not self.__eq__(right)

    def __is_zero__(self):
        """
        Tests if self is the zero object 
        
        Output:
            - true if all initial values are zero
            - false else
        """
        return all(self.initial_conditions()[x] == 0 for x in self.initial_conditions())

    def _is_atomic(self):
        """
        """
        raise NotImplementedError

    def is_unit(self):  #TODO
        r"""
        Return True if this function is a unit.
        """
        if self._ann.order() == 1:
            return true
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
        return Infinity
    
    # conversion

    def change_variable_name(self, var):
        """
        Returns a new function over the same base ring but in a different
        variable.
        
        Input:
            - var ... the new variable; has to be given as a string
            
        Example:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: Sn = A.gen()
            sage: n = A.base_ring().gen()
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteSequence(C, Sn**2 - Sn - 1, [0,1]) #the Fibonacci numbers
            sage: c = a.change_variable_name('x')
            sage: a.annihilator()
            Sn^2 - Sn - 1
            sage: c.annihilator()
            x^2 - x - 1
        
        """
        D = DFiniteFunctionRing(self.parent().ore_algebra().change_var(var),self.parent()._domain, self.parent()._codomain)
        result = DFiniteFunction(D, self._ann, self._initial_values)
        return result
        
    def change_ring(self, R):
        """
        Return a copy of this function but with base ring R, if at all possible.
        """
        D = self.parent().change_base_ring(R)
        result = DFiniteFunction(D,self._ann,self._initial_values)
        return result

    def __getitem__(self, n):
        raise NotImplementedError

    def __setitem__(self, n, value):
        """
        """
        raise IndexError("D-finite functions are immutable")

    def __iter__(self):
        return NotImplementedError

    def __float__(self):    #TODO: differential case
        """
        tries to convert self into a float, this is possible if
        the sequence self is constant, i.e. self(n) = x for some x in RR 
        and for all n in NN
        
        - just in sequential case done
        """
        if self.parent().ore_algebra().is_S():
            i = self.initial_values()[0]
        
            if all(x == i for x in self.initial_values()):
                Sn = self.parent().ore_algebra().gens()[0]
                if self._ann.monic() == Sn**(self._ann.order())-1:
                    return float(i)
            raise TypeError, "no conversion possible"
            
        else:
            raise NotImplementedError
    
    def __int__(self):   #TODO: differential case
        """
        tries to convert self into an int, this is possible if
        the sequence self is constant, i.e. self(n) = k for some k in ZZ
        and for all n in NN

        - just in sequential case done
        """
        if self.parent().ore_algebra().is_S():
            i = self.initial_values()[0]
        
            if i in ZZ and all(x == i for x in self.initial_values()):
                Sn = self.parent().ore_algebra().gens()[0]
                if self._ann.monic() == Sn**(self._ann.order())-1:
                    return int(i)
        
            raise TypeError, "no conversion possible"
    
        else:
            raise NotImplementedError

    def _integer_(self, ZZ):
        """
        tries to convert self into a integer, this is possible if
        the sequence self is constant, i.e. self(n) = k for some k in ZZ 
        and for all n in NN

        - just in sequential case done
        """
        return ZZ(int(self))

    def _rational_(self):    #TODO: differential case
        """
        tries to convert self into a rational, this is possible if
        the sequence self is constant, i.e. self(n) = q for some q in QQ
        and for all n in NN
 
        - just in sequential case done
        """
        if self.parent().ore_algebra().is_S():
            i = self.initial_values()[0]
        
            if i in QQ and all(x == i for x in self.initial_values()):
                Sn = self.parent().ore_algebra().gens()[0]
                if self._ann.monic() == Sn**(self._ann.order())-1:
                    return QQ(i)
        
            raise TypeError, "no conversion possible"
    
        else:
            raise NotImplementedError

    def _symbolic_(self, R):
        raise NotImplementedError

    def __long__(self):          #TODO: differential case
        """
        tries to convert self into a long 
        - just in sequential case done
        """
        if self.parent().ore_algebra().is_S():
            i = self.initial_values()[0]
        
            if (all(x == i for x in self.initial_values())):
                Sn = self.parent().ore_algebra().gens()[0]
                if self._ann.monic() == Sn**(self._ann.order())-1:
                    return long(i)
            raise TypeError, "no conversion possible"
            
        else:
            raise NotImplementedError

    def _repr(self, name=None):
        return self._repr_()

    def _repr_(self):
        """
        returns in words what object you are working with if you ask 
        """
        r = "Univariate D-finite "
        if self.parent().ore_algebra().is_S():
            r = r + "Sequence"
        else:
            r = r + "function"
        r = r + " defined by the annihilating polynomial "
        r = r + self._ann._repr() + " and the initial conditions "
        r = r + str(self._initial_values)
        return r


    def _latex_(self, name=None):
        """
        describes a DFiniteFunction Sequence/Function in latex
        """
        
        if self.parent().ore_algebra().is_S():
            r = '\\text{D-finite sequence defined by the annihilating polynomial }'
        else:
            r = '\\text{D-finite function defined by the annihilating polynomial }'
        
        r = r + latex(self._ann) + '\\text{ and the initial conditions }'
        r = r + latex(self.initial_conditions())
        return r
        
    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce this value when
        evaluated.
        """
        par = self.parent()
        int_cond = self.initial_conditions()
        init = sib({sib.int(a):sib.int(int_cond[a]) for a in int_cond})
        if par.ore_algebra().is_S():
            result = sib.name('UnivariateDFiniteSequence')(sib(par),sib(self._ann),init)
        else:
            result = sib.name('UnivariateDFiniteFunction')(sib(par),sib(self._ann),init)
        return result

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

    def __pow__(self, n, modulus = None):
        """
        """
        return self._pow(n)
        
    def _pow(self, n):
        """
        returns self to the n-th power
        n has to be a natural number
        """
        if n == 0:
            return self.parent().one()
        if n == 1:
            return self

        return self*(self._pow(n-1))
                   
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
        return the annihilating operator of self
        """
        return self._ann
    
    def initial_values(self):
        """
        returns the initial values of ``self`` in form of a list
        in the differential case the i-th initial value is the value
        of the i-th derivative of self at 0
        """
        if self.parent().ore_algebra().is_S():
            if self.parent()._backward_calculation is False and min(self.initial_conditions()) < 0:
                m = min(self.initial_conditions())
                result = [self._initial_values[key] for key in range(m,self._ann.order()+m)]
            else:
                result = [self._initial_values[key] for key in range(self._ann.order())]
            return result
        elif self.parent().ore_algebra().is_D() :
            result = self._initial_values.expand(self._ann.order() + 1)
            result = [result[i] * factorial(i) for i in range(len(result))]
            return result
        else:
            return ValueError


    def initial_conditions(self):
        """
        return all initial conditions, i.e. the initial values plus possible singularities of
        ``self`` in form of a dictionary
        """
        if self.parent().ore_algebra().is_S():
            return self._initial_values
        else:
            return self._initial_values._initial_values

#############################################################################################################
    
class UnivariateDFiniteSequence(DFiniteFunction):
    """
    D-finite sequence in a single (differentiable or discrete) variable.
    """
    def __init__(self, parent, ann, initial_val, is_gen=False, construct=False, cache=True):
        if not parent.ore_algebra().is_S():
            raise TypeError, "Not the Shift Operator"
        super(UnivariateDFiniteSequence, self).__init__(parent, ann, initial_val, is_gen, construct, cache)

    # action

    def __call__(self, x):
        """
        If ``x`` is an integer (or a float, which then gets ``cut`` to an integer) the x-th sequence term
        is returned. This is also possible for negative x if the D-finite function ring is defined
        over the codomain ZZ (however in some cases when new singularities appear this computation can fail !!!)
        If ``x`` is a suitable expression, i.e. of the form x = u*n + v for 
        some u,v in QQ, it is interpreted as the composition self(floor(x(n)))
        
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: Sn = A.gen()
            sage: n = A.base_ring().gen()
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteSequence(D, Sn**2 - Sn - 1, [0,1]) #the Fibonacci numbers
            sage: a(3)
            2
            sage: a(2*n+3).expand(10) #the odd Fibonacci numbers staring with a_3
            [2, 5, 13, 34, 89, 233, 610, 1597, 4181, 10946]
        """
        try:
            n = int(x)
        except:
        
            n = self.parent().ore_algebra().is_S()
            x = QQ[n](x)
            A = self._ann.annihilator_of_composition(x)
            ord = A.order()
            sing = [a for a in range(ord)] + [a for a in A.singularities(self.parent()._backward_calculation) if 0 < a or a >= ord]
            int_val = {a:self[floor(x(a))] for a in sing}
            result = UnivariateDFiniteSequence(self.parent(), A, int_val)
            return result

        return self[n]

    def dict(self):
        raise NotImplementedError

    def list(self):
        raise NotImplementedError

    # arithmetic

    def _add_(self, right):
        """
        return the sum of two D-finite sequences
        
        _add_ uses lclm to get the new annihilator
        
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: Sn = A.gen()
            sage: n = A.base_ring().gen()
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteSequence(D, Sn**2 - Sn - 1, [0,1])
            sage: b = UnivariateDFiniteSequence(D, (n**2-3)*Sn**3 + (4*n - 10)*Sn - 1, [0,1,2])
            sage: c = a+b
            sage: c.expand(10)
            [0, 2, 3, -4/3, -7/2, 1/3, 173/18, 381/26, 361/18, 257507/7722]
            sage: [a(i) + b(i) for i in range(10)]
            [0, 2, 3, -4/3, -7/2, 1/3, 173/18, 381/26, 361/18, 257507/7722]
        """
        
        sum_ann = self._ann.lclm(right._ann)
        
        ord = sum_ann.order()
       
        sing_sum = [0] + [x for x in range(ord)] + [x for x in sum_ann.singularities(self.parent()._backward_calculation)]
        
        sing = sing_sum + [max(self.initial_conditions()), max(right.initial_conditions())]
        
        if sing:
            s_max = max(sing) + ord
            s_min = min(sing)
        else:
            s_max = ord
            s_min = 0
        
        int_val_sum = {a:self[a] + right[a] for a in range(s_min,s_max+1)}

        sum = UnivariateDFiniteSequence(self.parent(), sum_ann, int_val_sum)
        return sum
    
    def _neg_(self):
        """
        returns the negative of a D-finite sequence
        """
        neg_int_val = {key:(-self._initial_values[key]) for key in self._initial_values}
        neg = UnivariateDFiniteSequence(self.parent(), self._ann, neg_int_val)
        return neg

#   def _lmul_(self, left):
#       raise NotImplementedError
    
#   def _rmul_(self, right):
#       raise NotImplementedError

    def _mul_(self, right):
        """
        return the product of two D-finite sequences
        
        _mul_ uses the symmetric product to get the new annihilator
        
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: Sn = A.gen()
            sage: n = A.base_ring().gen()
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteSequence(D, Sn**2 - Sn - 1, [0,1])
            sage: b = UnivariateDFiniteSequence(D, (n**2-3)*Sn**3 + (4*n - 10)*Sn - 1, [0,1,2])
            sage: c = a*b
            sage: c.expand(10)
            [0, 1, 2, -20/3, -39/2, -70/3, 116/9, 43/2, -119/6, -85697/3861]
            sage: [a(i)*b(i) for i in range(10)]
            [0, 1, 2, -20/3, -39/2, -70/3, 116/9, 43/2, -119/6, -85697/3861]
        """
        if self.__is_zero__() or right.__is_zero__():
            return self.parent().zero()
        
        prod_ann = self._ann.symmetric_product(right._ann)
        
        ord = prod_ann.order()
        
        sing_prod = [0] + [x for x in range(ord)] + [x for x in prod_ann.singularities(self.parent()._backward_calculation)]
        
        sing = sing_prod + [max(self.initial_conditions()), max(right.initial_conditions())]
        if sing:
            s_max = max(sing) + ord
            s_min = min(sing)
        else:
            s_max = ord
            s_min = 0
        
        int_val_prod = {a:self[a] * right[a] for a in range(s_min,s_max+1)}
        
        prod = UnivariateDFiniteSequence(self.parent(), prod_ann, int_val_prod)
        return prod

    def __invert__(self):
        raise NotImplementedError
    
    def interlace(self, right):
        """
        returns the interlaced sequence of the two sequences self (e.g. a0,a1,a2,...)
        and right (e.g. b0,b1,b2,..). The result is then of the form a0,b0,a1,b1,....
        
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: Sn = A.gen()
            sage: n = A.base_ring().gen()
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteSequence(D, Sn**2 - Sn - 1, [0,1]) #the Fibonacci numbers
            sage: b = D(4) #the constant sequence ``4,4,4,...``
            sage: c = a.interlace(b)
            sage: c.expand(10)
            [0, 4, 1, 4, 1, 4, 2, 4, 3, 4]
        """
        interlacing_ann = self._ann.annihilator_of_interlacing(right._ann)
        
        ord = interlacing_ann.order()
        
        sing_interlacing = [0] + [x for x in range(ord)] + [x for x in interlacing_ann.singularities(self.parent()._backward_calculation)]
        
        sing = sing_interlacing + [max(self.initial_conditions()), max(right.initial_conditions())]
        if sing:
            s_max = max(sing) + ord
            s_min = min(sing)
        else:
            s_max = ord
            s_min = 0

        int_val_interlacing = {}
        for m in range(s_min,s_max+1):
            if m % 2 == 0:
                int_val_interlacing.update({m:self[m/2]})
            else:
                int_val_interlacing.update({m:right[floor(m/2)]})
        
        interlacing = UnivariateDFiniteSequence(self.parent(), interlacing_ann, int_val_interlacing)
        return interlacing

    # evaluation
    
    def expand(self, n):
        """
        returns the first n terms of the sequences "self"
        
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: Sn = A.gen()
            sage: n = A.base_ring().gen()
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteSequence(D, Sn**2 - Sn - 1, [0,1]) #the Fibonacci numbers
            sage: a.expand(10)
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

        """
        ord = self._ann.order()
        
        #check if self is coming from a d-finite function that contains added zeros:
        if self.parent()._backward_calculation is False and min(self.initial_conditions()) < 0:
            start = -min(self.initial_conditions())
            n = n + start 
        else:
            start = 0
            
        #1st case: n is smaller than the order - so all needed terms are already given
        if n < ord:
            return self.initial_values()[start:n]
        
        
        #2nd case: n is smaller than all relevant singularities - nothing to worry about
        s = [x for x in self._initial_values if ord <= x]
        if all(n < x for x in s):
            return self._ann.to_list(self.initial_values(),n, -start)[start:]

        #3rd case: there is at least one singularity in the first n terms of the sequence
        s = [x for x in self._initial_values if ord <= x < n]
        result = self.initial_values()
        while len(s) > 0:
            m = min(s)
            self._ann.to_list(result,m,-start,true)
            result.append(self._initial_values[m])
            s.remove(m)
        self._ann.to_list(result,n,-start,true)
        return result[start:]
        
    def __getitem__(self,n):
        """
        return the n-th term of the sequence ``self`` (starting with a0)
        It is also possible to calculate sequence terms for negative n but
        only if the D-finite function ring is defined over the codomain ZZ
            
        EXAMPLE:
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A)
            sage: a = UnivariateDFiniteSequence(D, "Sn**2 - Sn - 1", [0,1])
            sage: a[5]  #the 5th Fibonacci number
            5
            sage: a[10] #the 10th Fibnoacci number
            55
        """
        
        try:
            return self.initial_conditions()[n]
        except:
            pass

        #special case: n is negative - some special cases with singularities may cause problems
        if n < 0:
            
            if self.parent()._backward_calculation is False:
                raise TypeError, "Backward Calculation is not possible - the D-finite function ring is not suitable"
            try:
                ord = self._ann.order()
                num = -n
                n = self.parent().base_ring().gen()
                
                A = self._ann.annihilator_of_composition(ord-n)
                
                int_val = self.expand(ord+1)
            
                int_val.reverse()
                l = A.to_list(int_val,num+ord+1)
                return l[num+ord]
            except:
                raise ValueError, "term self(" + str(-num) + ") could not be computed"
    
        #normal case: n >= 0
        try:
            ord = self._ann.order()
            roots = [x - ord for x in self.singularities(backwards = false) if 0 <= x-ord <= n]
        
            if self.parent()._backward_calculation is false and min(self.initial_conditions) < 0:
                start = min(self.initial_conditions())
            else:
                start = 0
            int_val = self.initial_values()

            while len(roots) > 0:
                root = min(roots)
                Q,M = self._ann.forward_matrix_bsplit(root-start,start)
                v = Matrix([int_val]).transpose()/M
                result = Q * v
                if n < root + ord:
                    d = n - (root+ord)
                    return result[d][0]
                else:
                    int_val = [result[i][0] for i in range(1,result.nrows())] + [self.initial_conditions()[root+ord]]
                    start = root+1
                    roots.remove(root)

            Q,M = self._ann.forward_matrix_bsplit(n-start,start)
            v = Matrix([int_val]).transpose()/M
            result = Q * v
            return result[0][0]

        #TODO: getting rid of this exception - had no time left to do this
        except:
            return self.expand(n+1)[n]


###############################################################################################################
class UnivariateDFiniteFunction(DFiniteFunction):
    """
        D-finite function in a single (differentiable or discrete) variable.
        """
    def __init__(self, parent, ann, initial_val, is_gen=False, construct=False, cache=True):
        super(UnivariateDFiniteFunction, self).__init__(parent, ann, initial_val, is_gen, construct, cache)
        
    
    # action
    
    def __call__(self, x):
        """
        Lets ``self`` act on ``x`` and returns the result.
        ``x`` may be either a constant, then this computes an evaluation,
        or a (suitable) expression, then it represents composition and we return a new DFiniteFunction object.
        In this case ``a suitable expression`` means that ``x`` has to be a polynomial (either in explicit
        form or implicitly in form of a D-finite function) with no constant term
        
        EXAMPLE:
            sage: B = OreAlgebra(ZZ['x'],'Dx')
            sage: x = B.base_ring().gen()
            sage: Dx = B.gen()
            sage: D = DFiniteFunctionRing(B,QQ,NN)
            sage: sin = UnivariateDFiniteFunction(D, Dx^2 + 1, [0,1,0])
            sage: f = ZZ[x](3*x^3-5*x^2+5*x) #explicit form of the polynomial
            sage: g = D(3*x^3-5*x^2+5*x)     #implicit form as a D-finite function
            sage: sin(f)
            sage: sin(g)
            
            sage: sin(0.5)
            [0.479425538604203 +/- 1.06e-16]
            sage: g(14.3)
            7821.67100000000
        """
        try:
            x = float(x)
        except:
            if isinstance(x, UnivariateDFiniteFunction):
                x = x.to_polynomial()
            
            if x.constant_coefficient() != 0:
                    raise ValueError, "constant term has to be zero"
            
            A = self._ann.annihilator_of_composition(x)
            A_S = A.to_S(OreAlgebra(A.parent().base_ring().change_var('n'),'Sn'))
            ord = A_S.order()
            s = [a for a in A_S.singularities(backwards = false)]
            N = max(s) + ord
            
            a = self.expand(N)
            b = x.coefficients(false) +[0]
            for i in range(len(b)-1):
                b[i] = b[i+1]
            if len(b) < N:
                b = b + [0]*(N-len(b)-1)
                
            y = x.parent().gen()
            int_val = [None]*N
            
            for n in range(N):
                poly = x.parent()(0)
                B = y*(x.parent()(b[:n]))
                for i in range(n+1):
                    poly = poly + a[i]*B**i
                int_val[n] = poly[n]
        
            result = UnivariateDFiniteFunction(self.parent(), A, int_val)
            return result
        
        return self.evaluate(x,0)
        
    def to_polynomial(self):
        """
        Returns the explicit form of a polynomial if ``self`` represents a polynomial
        
        Example:
            sage: B = OreAlgebra(ZZ['x'],'Dx')
            sage: x = B.base_ring().gen()
            sage: Dx = B.gen()
            sage: D = DFiniteFunctionRing(B,QQ,NN)
            
            sage: f = D(3*x^3-5*x^2+5*x)
            sage: f.to_polynomial()
            3*x^3 - 5*x^2 + 5*x
            
            sage: g = UnivariateDFiniteFunction(D,(5*x^4-4*x+3)*Dx - (20*x^3-4),[3,-4])
            sage: g.to_polynomial()
            5*x^4 - 4*x + 3
        """
        
        R = self.parent().base_ring()
        x = R.gen()
        base = self._ann.polynomial_solutions()
        if len(base) == 0:
            raise TypeError, "the d-finite function is not a polynomial"
        vars = list(var('a_%d' % i) for i in range(len(base)))
        c = [0]*len(base)
        max_deg = 0
        for i in range(len(base)):
            base[i] = base[i][0]
            c[i] = base[i]*vars[i]
            if base[i].degree() > max_deg:
                max_deg = base[i].degree()
        coeffs = sum(c).expand().coefficients(x,false)
        int_val = self.expand(len(coeffs))
        eqs = list(coeffs[k] == int_val[k] for k in range(len(coeffs)))
        result = solve(eqs,vars)
        if type(result[0]) == list:
            result = result[0]
        coeffs_result = [0]*len(result)
        for i in range(len(result)):
            coeffs_result[i] = result[i].rhs()
        poly = sum(list(a*b for a,b in zip(coeffs_result,base)))
        return R(poly)
    

    
    def dict(self):
        raise NotImplementedError
    
    def list(self):
        raise NotImplementedError
    
    def expand(self, number, deriv = False):
        """
        returns a list of the first n coefficient of the taylor series of self if deriv = False
        else it returns the first n derivations of self in 0
        
        Example:
        
            sage: B = OreAlgebra(ZZ['x'],'Dx')
            sage: D = DFiniteFunctionRing(B,QQ,NN)
            sage: e = UnivariateDFiniteFunction(D, "Dx - 1", [1, 1, 1/2]) # exp(x)
            sage: e.expand(10)
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880]
            sage: e.expand(10,True)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        """
        result = self._initial_values.expand(number)
        if deriv == True:
            result = [result[i]* factorial(i) for i in range(len(result))]
        
        return result
    
    # arithmetic
    
    
    def _add_(self, right):
        """
        returns the sum of two dfinite functions
        _add_ uses lclm to get the new annihilator
        and uses the add of the dfinite sequences to get the initial values
        
        Example :
            sage: B = OreAlgebra(ZZ['x'],'Dx')
            sage: D = DFiniteFunctionRing(B,QQ,NN)
            sage: e = UnivariateDFiniteFunction(D, "Dx - 1", [1, 1, 1/2]) # exp(x)
            sage: f = UnivariateDFiniteFunction(D,"(3*x^2+1)*Dx - (6*x)",[1,0]) # 3x^2 + 1
            sage: c = e+f
            sage: f.expand(10)
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 0]
            sage: e.expand(10)
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880]
            sage: c.expand(10)
            [2, 1, 7/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880]
        """
        sum_ann = self._ann.lclm(right._ann)
        
        lseq = self._initial_values
        rseq = right._initial_values

        s_add = lseq + rseq
        sum = UnivariateDFiniteFunction(self.parent(), sum_ann, s_add)
       
        return sum
    
    
    def _neg_(self):
        """
        return the negative of a dfinite function respectively add
        """
        neg = UnivariateDFiniteFunction(self.parent(), self._ann,  -self._initial_values)
        return neg
    
    def _lmul_(self, left):
        raise NotImplementedError
    
    def _rmul_(self, right):
        raise NotImplementedError
    
    def _mul_(self, right):
        """
        returns the product of two dfinite functions
        _mul_ uses the symmetric product to get the new annihilator
        and uses  _mul_ of the dfinite sequences to get the initial values
        
        Example:
            sage: B = OreAlgebra(ZZ['x'],'Dx')
            sage: D = DFiniteFunctionRing(B,QQ,NN)
            sage: e = UnivariateDFiniteFunction(D, "Dx - 1", [1, 1, 1/2]) # exp(x)
            sage: f = UnivariateDFiniteFunction(D,"(3*x^2+1)*Dx - (6*x)",[1,0]) # 3x^2 + 1
            sage: c = e*f
            sage: c.expand(10)
            [1, 1, 7/2, 19/6, 37/24, 61/120, 91/720, 127/5040, 169/40320, 31/51840]
            sage: c.expand(10,True)
            [1, 1, 7, 19, 37, 61, 91, 127, 169, 217]
        """

        prod_ann = self._ann.symmetric_product(right._ann)
        init_num = prod_ann.order() + 1
        
        sing_prod = [0] + [a for a in range(init_num)] + [a for a in prod_ann.singularities(backwards = false)]
        sing = sing_prod + [max(self.initial_conditions()), max(right.initial_conditions())]
        if sing:
            s_max = max(sing) + init_num
        else:
            s_max = ord
        
        int_val_prod = {}
        for k in range(s_max):
            a = self.expand(k+1)
            b = right.expand(k+1)
            b.reverse()
            cauchy = sum([x*y for x,y in zip(a,b)])
            int_val_prod.update({k:cauchy})
        
        prod = UnivariateDFiniteFunction(self.parent(), prod_ann, int_val_prod)
        return prod
               
        
    def __invert__(self):
        raise NotImplementedError
    
    # evaluation
    
    
    def evaluate(self, z, n):
        """
        numerical evaluation of the n-th derivative of self at point z
        Input:
            - z ... the evaluation point
            - n ... number of derivations of self
            
        Example:
            sage: B = OreAlgebra(ZZ['x'],'Dx')
            sage: D = DFiniteFunctionRing(B,QQ,NN)
            sage: e = UnivariateDFiniteFunction(D, "Dx - 1", [1, 1, 1/2]) # exp(x)
            sage: f = UnivariateDFiniteFunction(D,"(3*x^2+1)*Dx - (6*x)",[1,0]) # 3x^2 + 1
            sage: e.evaluate(0,0)
            1.0000000000000000000000000000000000000000000000000000
            sage: e.evaluate(0,1)
            1.0000000000000000000000000000000000000000000000000000
            sage: e.evaluate(1,0)
            [2.718281828459045235360287471352662497757247093699960 +/- 6.43e-52]
            sage: e.evaluate(1,1)
            [2.718281828459045235360287471352662497757247093699960 +/- 6.43e-52]
            sage: f.evaluate(0,0)
            1.0000000000000000000000000000000000000000000000000000
            sage: f.evaluate(2,0)
            [13.000000000000000000000000000000000000000000000000000 +/- 6.94e-52]
            sage: f.evaluate(0,2)
            6.0000000000000000000000000000000000000000000000000000
            sage: f.evaluate(1,1)
            [6.000000000000000000000000000000000000000000000000000 +/- 4.68e-52]
            sage: f.evaluate(0,1)
            0
        """
        ini = self.initial_values()
        ini.pop()
        Dx = self.parent().ore_algebra().gen()
        for i in range(len(ini)):
            ini[i] = ini[i] / factorial(i)
        result = self._ann.numerical_solution(ini,[0,z], eps=1e-50, post_transform=Dx**n )
        return result

##################################################################################


