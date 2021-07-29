# coding: utf-8
r"""
Differentially finite functions and sequences

The dfinite_function module provides functionality for doing computations with D-finite functions and sequences.

A D-finite sequence can be represented by an ore operator which annihilates the sequence (for further information
about ore operators check the ``ore_algebra`` package), the operators singularities and a finite amount of initial values.

A D-finite function can be envisioned as a power series, therefore it can be represented via a D-finite sequence which
describes the coefficient sequence and an ore operator which annhilates the function.

D-finite sequences and functions are elements of D-finite function rings. D-finite function rings are ring objects created by the
function ``DFiniteFunctionRing`` as described below.

Depending on the particular parent ring, D-finite functions/sequences may support different functionality.
For example, for D-finite sequences, there is a method for computing an interlacing sequence, an operation 
which does not make sense for D-finite functions.
However basic arithmetic, such as addition and multiplication is implemented for both cases.

AUTHOR:

- Manuel Kauers, Stephanie Schwaiger, Clemens Hofstadler (2017-07-15)

"""

#############################################################################
#  Copyright (C) 2017 Manuel Kauers (mkauers@gmail.com),                    #
#                     Stephanie Schwaiger (stephanie.schwaiger97@gmail.com),#
#                     Clemens Hofstadler (clemens.hofstadler@liwest.at).    #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

from __future__ import absolute_import, division, print_function

import pprint

from copy import copy
from math import factorial
from operator import pow

from numpy import random

from sage.arith.all import gcd
from sage.calculus.var import var
from sage.functions.other import floor, ceil, binomial
from sage.matrix.constructor import matrix
from sage.matrix.constructor import Matrix
from sage.misc.all import prod, randint
from sage.rings.all import ZZ, QQ, CC
from sage.rings.ring import Algebra
from sage.rings.semirings.non_negative_integer_semiring import NN
from sage.structure.element import RingElement
from sage.symbolic.operators import add_vararg, mul_vararg
from sage.symbolic.relation import solve

from .ore_algebra import OreAlgebra
from .dfinite_symbolic import symbolic_database
from .guessing import guess

class DFiniteFunctionRing(Algebra):
    r"""
    A Ring of Dfinite objects (functions or sequences)
    """
    
# constructor
    
    def __init__(self, ore_algebra, domain = NN, name=None, element_class=None, category=None):
        r"""
        Constructor for a D-finite function ring.
        
        INPUT:
        
        - ``ore_algebra`` -- an Ore algebra over which the D-finite function ring is defined.
            Only ore algebras with the differential or the shift operator are accepted to
            define a D-finite function ring.
        - ``domain`` (default ``NN``) -- domain over which the sequence indices are considered,
            i.e. if the domain is ``ZZ``also negative sequence inidices exist.
            So far for d-finite sequences ``NN`` and ``ZZ`` are supported and for D-finite
            functions only ``NN``is supported.
        
        OUTPUT:
        
        A ring of either D-finite sequences or functions
            
        EXAMPLES::
        
            sage: from ore_algebra import *

            #This creates an d-finite sequence ring with indices in ``ZZ``
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: D
            Ring of D-finite sequences over Univariate Polynomial Ring in n over Integer Ring
            
            #This creates an d-finite function ring
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: E = DFiniteFunctionRing(B)
            sage: E
            Ring of D-finite functions over Univariate Polynomial Ring in x over Rational Field

        """
        if domain != ZZ and domain != NN:
            raise TypeError("Domain does not fit")
        
        self._ore_algebra = ore_algebra
        self._base_ring = ore_algebra.base_ring()
        
        if ore_algebra.is_D() and domain == ZZ:
            raise NotImplementedError("D-finite functions with negative powers are not implemented")
        self._domain = domain
        if domain == NN:
            self._backward_calculation = False
        else:
            self._backward_calculation = True

        self._populate_coercion_lists_()
    
#conversion

    def _element_constructor_(self, x=None, check=True, is_gen = False, construct=False, **kwds):
        r"""
        Convert ``x`` into this ring, possibly non-canonically.
        
        This is possible if:
        
        - ``x`` is already a d-finite object. Then it is converted into the new d-finite function ring if possible. See the
          method ``_construct_dfinite`` for further information
        - ``x`` is a list of data. Then it is interpreted as the first coefficients of
          a sequence or a power series and the method ``guess`` is called in order to
          find a suitable Ore Operator. See  the method ``_construct_list`` for further information
        - ``x`` can be converted into a rational number. Then it is intpreted as the constant sequence 
           x,x,x,\dots or the constant function f(z) = x depending on the D-finite function ring.
        - ``x``can be converted into the fraction field of the base ring of the D-finite function ring. Then it is interpreted as
          the sequence (a_n) := x(n) for the recurrence case or as the function f(z) = x(z) for the
          differential case. See the method ``_construct_rational`` for further information
        - ``x`` is a symbolic expression. Then ``x``is decomposed into atomic expressions which are then converted to D-finite objects if possible
          and put back together. See ``_construct_symbolic`` for further information.
        
        EXAMPLES::
    
        sage: from ore_algebra import *
        sage: A = OreAlgebra(QQ['n'],'Sn')
        sage: D1 = DFiniteFunctionRing(A)
        sage: B = OreAlgebra(QQ['x'],'Dx')
        sage: D2 = DFiniteFunctionRing(B)
        sage: n = A.base_ring().gen()
        sage: x = B.base_ring().gen()

        #conversion of a list of data
        sage: D1([0,1,1,2,3,5,8,13])
        Univariate D-finite sequence defined by the annihilating operator -Sn^2 + Sn + 1 and the initial conditions {0: 0, 1: 1}
        sage: D2([1,-1,1,-1,1,-1])
        Univariate D-finite function defined by the annihilating operator (x + 1)*Dx + 1 and the coefficient sequence defined by (n + 1)*Sn + n + 1 and {0: 1}
        
        #conversion of rational numbers
        sage: D1(3)
        Univariate D-finite sequence defined by the annihilating operator Sn - 1 and the initial conditions {0: 3}
        sage: D2(7.5)
        Univariate D-finite function defined by the annihilating operator Dx and the coefficient sequence defined by n and {0: 15/2}
        
        #conversion of rational functions
        sage: D1((n-2)/((n+1)*(n+5)))
        Univariate D-finite sequence defined by the annihilating operator (n^3 + 6*n^2 - 4*n - 24)*Sn - n^3 - 5*n^2 + n + 5 and the initial conditions {0: -2/5, 3: 1/32}
        sage: D2((x^2+4)/(x-1))
        Univariate D-finite function defined by the annihilating operator (x^3 - x^2 + 4*x - 4)*Dx - x^2 + 2*x + 4 and the coefficient sequence defined by (-4*n - 12)*Sn^3 + (4*n + 12)*Sn^2 + (-n + 1)*Sn + n - 1 and {0: -4, 1: -4, 2: -5}
    
        #conversion of symbolic expressions
        sage: D1(harmonic_number(n))
        Univariate D-finite sequence defined by the annihilating operator (n + 2)*Sn^2 + (-2*n - 3)*Sn + n + 1 and the initial conditions {0: 0, 1: 1}
        sage: D2(sin(x^2))
        Univariate D-finite function defined by the annihilating operator x*Dx^2 - Dx + 4*x^3 and the coefficient sequence defined by (n^8 + 9*n^7 + 21*n^6 - 21*n^5 - 126*n^4 - 84*n^3 + 104*n^2 + 96*n)*Sn^4 + 4*n^6 + 12*n^5 - 20*n^4 - 60*n^3 + 16*n^2 + 48*n and {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: -1/6}
        
        """
        n = self.ore_algebra().is_S()

        #conversion for D-finite functions:
        if isinstance(x,DFiniteFunction):
            return self._construct_dfinite(x,n)

        #conversion for lists:
        elif type(x) == list:
            return self._construct_list(x,n)

        #conversion for numbers
        elif x in QQ:
            if n:
                Sn = self.ore_algebra().gen()
                return UnivariateDFiniteSequence(self,Sn-1,{0:x})
            else:
                Dy = self.ore_algebra().gen()
                return UnivariateDFiniteFunction(self,Dy,{0:x})

        #conversion for rational functions
        elif x in self.base_ring().fraction_field():
               return self._construct_rational(x,n)
               
        #conversion for symbolic constants
        elif x.is_constant():
            if n:
                Sn = self.ore_algebra().gen()
                return UnivariateDFiniteSequence(self,Sn-1,{0:x})
            else:
                Dy = self.ore_algebra().gen()
                return UnivariateDFiniteFunction(self,Dy,{0:x})
        else:
        #conversion for symbolic expressions
            return self._construct_symbolic(x,n)

                
    def _construct_dfinite(self,x,n):
        r"""
        Convert a d-finite object ``x`` into this ring, possibly non-canonically
        
        This is possible if there is a coercion from the Ore algebra of the parent of ``x`` into the Ore algebra of ``self``.
        In the shift case, if ``x`` represents a sequence that is defined over ``NN`` then also the domain of ``self``
        has to be ``NN``(can not convert sequence over domain ``NN`` into sequence over domain ``ZZ``)
        
        INPUT:
        
        - ``x`` -- a d-finite object
        
        - ``n`` -- in the shift case the generator of the Ore algebra over which ``self`` is defined, in the differential case ``False``
        
        OUTPUT:
        
        The D-finite object ``x`` but now with ``self`` as parent
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: D2 = DFiniteFunctionRing(A,NN)
            sage: n = A.base_ring().gen()
            sage: a1 = D1(n)
            sage: a2 = D2(n)
            sage: a1
            Univariate D-finite sequence defined by the annihilating operator n*Sn - n - 1 and the initial conditions {-1: -1, 0: 0, 1: 1}
            sage: a2
            Univariate D-finite sequence defined by the annihilating operator n*Sn - n - 1 and the initial conditions {0: 0, 1: 1}
            sage: D2(a1)
            Univariate D-finite sequence defined by the annihilating operator n*Sn - n - 1 and the initial conditions {0: 0, 1: 1}

            #D1(a2) would not work since a2 is defined over ``NN`` but D1 has domain ``ZZ``
            
        """
        if self._coerce_map_from_(x.parent()):
            if n:
                if self._domain == ZZ and x.parent()._domain == NN:
                    raise TypeError("can not convert sequence over domain NN into sequence over domain ZZ")
                else:
                    return UnivariateDFiniteSequence(self,x.ann(),x.initial_conditions())
            else:
                return UnivariateDFiniteFunction(self,x.ann(),x.initial_conditions())
        else:
            raise TypeError(str(x) + " could not be converted - the underlying Ore Algebras don't match")

    def _construct_list(self,x,n):
        r"""
        Convert a list of data ``x`` into this ring, possibly non-canonically.
        
        This method may lead to problems when the D-finite object has singularities because these are not considered during
        the conversion. So in this case maybe not all sequence/power series term can be computed.
        
        INPUT:
        
        - ``x`` -- a list of rational numbers that are the first terms of a sequence or a power series
        
        - ``n`` -- in the shift case the generator of the Ore algebra over which ``self`` is defined, in the differential case ``False``
        
        OUTPUT:
        
        A D-finite object with initial values from the list ``x`` and an ore operator which annihilates this initial values
        
        EXAMPLES::
            
            #discrete case
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A)
            sage: D1([0,1,1,2,3,5,8])
            Univariate D-finite sequence defined by the annihilating operator -Sn^2 + Sn + 1 and the initial conditions {0: 0, 1: 1}

            #differential case
            sage: from ore_algebra import *
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: D2([1,-1,1,-1,1,-1])
            Univariate D-finite function defined by the annihilating operator (x + 1)*Dx + 1 and the coefficient sequence defined by (n + 1)*Sn + n + 1 and {0: 1}
            
        """
        try:
            ann = guess(x,self.ore_algebra())
        except:
            raise ValueError("no relation found")
    
        if n:
            return UnivariateDFiniteSequence(self,ann,x)
        else:
            return UnivariateDFiniteFunction(self,ann,x)

    def _construct_rational(self,x,n):
        r"""
        Convert a rational function ``x`` into this ring, possibly non-canonically.
        Pols of ``x`` will be represented as ``None`` entries.
        
        - ``x`` -- a list of rational numbers that are the first terms of a sequence or a power series
        
        - ``n`` -- in the shift case the generator of the Ore algebra over which ``self`` is defined, in the differential case ``False``
        
        OUTPUT:
        
        A D-finite object that either represents the sequence (a_n) = x(n) or the rational function x(z).
        
        EXAMPLES::
        
            #discrete case
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: n = A.base_ring().gen()
            sage: D1(1/n)
            Univariate D-finite sequence defined by the annihilating operator (n^3 + 3*n^2 + 2*n)*Sn - n^3 - 2*n^2 and the initial conditions {-2: -1/2, -1: -1, 0: None, 1: 1}

            #differential case
            sage: from ore_algebra import *
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: x = B.base_ring().gen()
            sage: D2(1/(x+1))
            Univariate D-finite function defined by the annihilating operator (x + 1)*Dx + 1 and the coefficient sequence defined by (n + 1)*Sn + n + 1 and {0: 1}
            
        """
        R = self.base_ring()
        x = R.fraction_field()(x)

        if n:
                
            #getting the operator
            A = self.ore_algebra()
            N = R.gen()
            Sn = A.gen()
            f = x.numerator()
            g = x.denominator()
            ann = A(g(n+1)*f(n)*Sn - g(n)*f(n+1))
                
            #getting the order of the operator
            ord = 1

            #initial values, singularities and pols of the new operator
            singularities_positive = ann.singularities()
            singularities_negative = set()
            if self._backward_calculation is True:
                singularities_negative = set([n for n in ann.singularities(True) if n < 0])

            initial_val = set(range(ord)).union(singularities_positive, singularities_negative)
            if self._backward_calculation is True:
                pols = set(r for (r,m) in g.roots() if r in ZZ)
            else:
                pols = set(r for (r,m) in g.roots() if r in NN)
                    
            int_val = {n:x(n) if n not in pols else None for n in initial_val}
            for n in pols:
                if self._backward_calculation is True and n-1 not in int_val:
                    int_val.update({n-1:x(n-1)})
                    ann = A(N - (n - 2))*ann
                    if n < 2:
                        int_val.update({n-2: x(n-2) if n-2 not in pols else None})
                if n+1 not in int_val:
                    int_val.update({n+1:x(n+1)})
                    ann = A(N - n)*ann

            return UnivariateDFiniteSequence(self,ann,int_val)
                    
        else:
            if R(x.denominator()).constant_coefficient() == 0:
                raise ValueError("The rational function you tried to convert has a pole at x = 0. Hence, it can not be converted into a D-finite formal power series object. ")
            y = self.ore_algebra().is_D()
            Dy = self.ore_algebra().gen()
            R = self.ore_algebra().base_ring().change_var('n')
            OreAlg = OreAlgebra(R,'Sn')
    
            ann = x.denominator()**2*(x*Dy - x.derivative())
            
            #getting the operator of the sequence
            s_ann = ann.to_S(OreAlg)
            ord = s_ann.order()
                    
            #initial values and singularities of the sequence operator
            singularities_positive = s_ann.singularities()

            initial_val = set(range(ord)).union(singularities_positive)
            int_val = {n:(x.derivative(n)(0)/factorial(n)) for n in initial_val}
            
            #getting the coefficient sequence
            seq = UnivariateDFiniteSequence(DFiniteFunctionRing(OreAlg,NN),s_ann, int_val)
                
            return UnivariateDFiniteFunction(self,ann,seq)
    
    def _construct_symbolic(self,exp,n):
        r"""
        Convert a symbolic expression ``exp`` into this ring, possibly non-canonically.
        
        In the shift case the symoblic expression can contain the following symbolic functions:
        ``harmonic_number(n)``, ``binomial(k,n)``, ``binomial(n,k)`` (where ``k`` is a fixed integer) and ``factorial(n)``.
        Of course all other functions that can be converted into a D-finite sequences (such as rational functions) can appear.
        Additionally addition and multiplication of these functions and composition of these functions with linear functions are
        supported.
        
        In the differential case the symbolic expression can contain several symbolic functions, including most trigonometric functions, square root, 
        logarithm, Airy functions, Bessel functions, error functions,\dots (for a detailed list see the documentation of ``dfinite_symbolic.py``). Of
        course all other functions that can be converrted into a D-finite function (such as rational functions) can appear. Additionally addition and
        multiplication of these functions and compostion of these functions with rational functions are supported.
        with linear inner functions
        
        INPUT:
        
        - ``exp`` -- a symbolic expression, i.e. an element of the ``Symbolic Ring``
        
        - ``n`` -- in the shift case the generator of the Ore algebra over which ``self`` is defined, in the differential case ``False``
        
        OUTPUT:
        
        A D-finite object that either represents the sequence (a_n) = exp(n) or the rational function exp(z).
        
        EXAMPLES::
        
        #discrete case
        sage: from ore_algebra import *
        sage: A = OreAlgebra(QQ['n'],'Sn')
        sage: D1 = DFiniteFunctionRing(A)
        sage: n = A.base_ring().gen()
        sage: D1(harmonic_number(3*n)+factorial(n+2))
        Univariate D-finite sequence defined by the annihilating operator (-2187*n^10 - 42282*n^9 - 364905*n^8 - 1845423*n^7 - 6031683*n^6 - 13246704*n^5 - 19679543*n^4 - 19390967*n^3 - 12024690*n^2 - 4192544*n - 615552)*Sn^3 + (2187*n^11 + 57591*n^10 + 663066*n^9 + 4430376*n^8 + 19123470*n^7 + 55966311*n^6 + 113079500*n^5 + 157190572*n^4 + 146606713*n^3 + 86797038*n^2 + 29096584*n + 4133152)*Sn^2 + (-4374*n^11 - 106434*n^10 - 1151901*n^9 - 7304094*n^8 - 30065373*n^7 - 84062052*n^6 - 162242521*n^5 - 215119824*n^4 - 190943363*n^3 - 107309404*n^2 - 34064948*n - 4573552)*Sn + 2187*n^11 + 51030*n^10 + 531117*n^9 + 3238623*n^8 + 12787326*n^7 + 34127424*n^6 + 62409725*n^5 + 77608795*n^4 + 63727617*n^3 + 32537056*n^2 + 9160908*n + 1055952 and the initial conditions {0: 2, 1: 47/6, 2: 529/20}
        
        #differential case
        sage: from ore_algebra import *
        sage: B = OreAlgebra(QQ['x'],'Dx')
        sage: D2 = DFiniteFunctionRing(B)
        sage: x = B.base_ring().gen()
        sage: D2(cos(1/(x+1)-1) + erf(x))
        Univariate D-finite function defined by the annihilating operator (x^10 + 8*x^9 + 49/2*x^8 + 31*x^7 - 11/2*x^6 - 68*x^5 - 357/4*x^4 - 52*x^3 - 10*x^2 + 3*x + 5/4)*Dx^4 + (2*x^11 + 16*x^10 + 53*x^9 + 88*x^8 + 38*x^7 - 164*x^6 - 823/2*x^5 - 450*x^4 - 453/2*x^3 - 5/2*x^2 + 47*x + 29/2)*Dx^3 + (12*x^10 + 84*x^9 + 198*x^8 + 78*x^7 - 479*x^6 - 986*x^5 - 1585/2*x^4 - 148*x^3 + 443/2*x^2 + 175*x + 173/4)*Dx^2 + (12*x^9 + 72*x^8 + 116*x^7 - 100*x^6 - 511*x^5 - 538*x^4 - 68*x^3 + 251*x^2 + 331/2*x + 29)*Dx and the coefficient sequence defined by (n^16 + 58*n^15 + 1586*n^14 + 27166*n^13 + 326302*n^12 + 2912554*n^11 + 19955316*n^10 + 106814918*n^9 + 450129433*n^8 + 1493422508*n^7 + 3874175930*n^6 + 7747309436*n^5 + 11664687464*n^4 + 12736187280*n^3 + 9474058368*n^2 + 4271719680*n + 875059200)*Sn^6 + (4*n^16 + 226*n^15 + 6024*n^14 + 100592*n^13 + 1177480*n^12 + 10234464*n^11 + 68209728*n^10 + 354734988*n^9 + 1450788084*n^8 + 4667051182*n^7 + 11732009336*n^6 + 22729682820*n^5 + 33164816192*n^4 + 35117569808*n^3 + 25364804352*n^2 + 11123953920*n + 2221516800)*Sn^5 + (6*n^16 + 332*n^15 + 8665*n^14 + 141575*n^13 + 1619527*n^12 + 13734455*n^11 + 89144987*n^10 + 450633211*n^9 + 1788177235*n^8 + 5572951365*n^7 + 13557803404*n^6 + 25406961342*n^5 + 35859170592*n^4 + 36752991608*n^3 + 25727020224*n^2 + 10954979712*n + 2129448960)*Sn^4 + (4*n^16 + 222*n^15 + 5800*n^14 + 94644*n^13 + 1078580*n^12 + 9088728*n^11 + 58465932*n^10 + 292217188*n^9 + 1144088016*n^8 + 3512007850*n^7 + 8404694540*n^6 + 15480277192*n^5 + 21465049560*n^4 + 21612626880*n^3 + 14866684128*n^2 + 6224622336*n + 1190868480)*Sn^3 + (n^16 + 64*n^15 + 1872*n^14 + 33490*n^13 + 412060*n^12 + 3705962*n^11 + 25218766*n^10 + 132382182*n^9 + 541045211*n^8 + 1724258998*n^7 + 4262072466*n^6 + 8068776120*n^5 + 11446089368*n^4 + 11738219056*n^3 + 8190152576*n^2 + 3465511488*n + 667895040)*Sn^2 + (8*n^15 + 396*n^14 + 9192*n^13 + 132528*n^12 + 1323888*n^11 + 9671720*n^10 + 53152704*n^9 + 222710256*n^8 + 713886792*n^7 + 1742615244*n^6 + 3199425048*n^5 + 4322678176*n^4 + 4146909952*n^3 + 2659625280*n^2 + 1017530496*n + 174804480)*Sn + 2*n^15 + 98*n^14 + 2246*n^13 + 31842*n^12 + 310942*n^11 + 2202770*n^10 + 11614202*n^9 + 46041790*n^8 + 137124544*n^7 + 303655500*n^6 + 489613472*n^5 + 554676448*n^4 + 415443392*n^3 + 183362112*n^2 + 35861760*n and {0: 1, 1: 2/sqrt(pi), 2: -1/2, 3: -2/3/sqrt(pi) + 1, 4: -35/24, 5: 1/5/sqrt(pi) + 11/6}
        
        sage: D2(sinh_integral(x+1)*exp(3*x^2))
        Univariate D-finite function defined by the annihilating operator (x + 1)*Dx^3 + (-18*x^2 - 18*x + 2)*Dx^2 + (108*x^3 + 108*x^2 - 43*x - 19)*Dx - 216*x^4 - 216*x^3 + 186*x^2 + 114*x - 12 and the coefficient sequence defined by (n^13 + 43*n^12 + 797*n^11 + 8255*n^10 + 51363*n^9 + 187089*n^8 + 313151*n^7 - 264475*n^6 - 2183264*n^5 - 3322832*n^4 - 298848*n^3 + 3391920*n^2 + 2116800*n)*Sn^7 + (n^13 + 42*n^12 + 761*n^11 + 7710*n^10 + 46923*n^9 + 166806*n^8 + 268043*n^7 - 261870*n^6 - 1937024*n^5 - 2863248*n^4 - 193104*n^3 + 2950560*n^2 + 1814400*n)*Sn^6 + (-18*n^12 - 631*n^11 - 9300*n^10 - 73715*n^9 - 326844*n^8 - 696633*n^7 + 121860*n^6 + 3938215*n^5 + 7051062*n^4 + 1419164*n^3 - 6836760*n^2 - 4586400*n)*Sn^5 + (-18*n^12 - 619*n^11 - 8945*n^10 - 69460*n^9 - 301044*n^8 - 620487*n^7 + 167115*n^6 + 3605710*n^5 + 6252812*n^4 + 1116856*n^3 - 6109920*n^2 - 4032000*n)*Sn^4 + (108*n^11 + 3138*n^10 + 36870*n^9 + 218520*n^8 + 625464*n^7 + 319914*n^6 - 2807010*n^5 - 6455220*n^4 - 2270472*n^3 + 5913648*n^2 + 4415040*n)*Sn^3 + (108*n^11 + 3102*n^10 + 35970*n^9 + 209880*n^8 + 587664*n^7 + 264726*n^6 - 2689830*n^5 - 5994780*n^4 - 1986072*n^3 + 5517072*n^2 + 4052160*n)*Sn^2 + (-216*n^10 - 5400*n^9 - 51840*n^8 - 226800*n^7 - 331128*n^6 + 703080*n^5 + 2762640*n^4 + 1706400*n^3 - 2379456*n^2 - 2177280*n)*Sn - 216*n^10 - 5400*n^9 - 51840*n^8 - 226800*n^7 - 331128*n^6 + 703080*n^5 + 2762640*n^4 + 1706400*n^3 - 2379456*n^2 - 2177280*n and {0: sinh_integral(1), 1: sinh(1), 2: 1/2*cosh(1) - 1/2*sinh(1) + 3*sinh_integral(1), 3: -1/3*cosh(1) + 7/2*sinh(1), 4: 43/24*cosh(1) - 15/8*sinh(1) + 9/2*sinh_integral(1), 5: -37/30*cosh(1) + 757/120*sinh(1), 6: 797/240*cosh(1) - 523/144*sinh(1) + 9/2*sinh_integral(1), 7: -663/280*cosh(1) + 39793/5040*sinh(1), 8: 173251/40320*cosh(1) - 28231/5760*sinh(1) + 27/8*sinh_integral(1), 9: -144433/45360*cosh(1) + 316321/40320*sinh(1)}

        """
        R = self.base_ring()
    
        try:
            operator = exp.operator()
        except:
            raise TypeError("no operator in this symbolic expression")

        operands = exp.operands()
        
        #add, mul
        if operator == add_vararg or operator == mul_vararg:
            while len(operands) > 1:
                operands.append( operator(self(operands.pop()), self(operands.pop())) )
            return operands[0]
        
       #pow
        elif operator == pow:
            exponent = operands[1]
            #pow
            if exponent in ZZ and exponent >= 0:
                return operator(self(operands[0]),ZZ(operands[1]))
            
            #sqrt - only works for sqrt(u*x+v) (linear inner function) - not implemented for sequences
            elif (not n) and (exponent - QQ(0.5) in ZZ) and (exponent >= 0):
                if R(operands[0]).degree() > 1:
                   raise ValueError("Sqrt implemented only for linear inner function")
                ann = symbolic_database(self.ore_algebra(),operator,operands[0])
                ord = ann.order()
                int_val = range(ord)
                initial_val = {i: operator(operands[0],QQ(0.5)).derivative(i)(x = 0)/factorial(i) for i in int_val}
                f = UnivariateDFiniteFunction(self,ann,initial_val)
                return operator(f,2*exponent)
            else:
                raise ValueError(str(exponent) + " is not a suitable exponent")

        #call
        else:
            if len(operands) == 1:
                inner = operands[0]
                k = None
            else:
                if operands[0] in QQ:
                    k = operands[0]
                    inner = operands[1]
                elif operands[1] in QQ:
                    k = operands[1]
                    inner = operands[0]
                #special case - binomial coefficient with linear functions in n
                elif operator == binomial:
                    if operands[0].derivative() not in QQ or operands[1].derivative() not in QQ:
                        raise TypeError("binomial coefficient only implemented for linear functions")
                    ann = symbolic_database(self.ore_algebra(),operator,R(operands[0]),R(operands[1]))
                    ord = ann.order()
                    singularities_positive = ann.singularities()
                    singularities_negative = set()
                    if self._backward_calculation is True:
                        singularities_negative = set([i for i in ann.singularities(True) if i < 0])
                    int_val = set(range(ord)).union(singularities_positive, singularities_negative)
                    initial_val = {i: exp(n = i) for i in int_val}
                    return UnivariateDFiniteSequence(self,ann,initial_val)
                else:
                    raise ValueError("one of the operands has to be in QQ")
                    
            #sequences
            if n:
                inner = R(inner)
                
                #check if inner is of the form u*n + v
                if inner.derivative() in QQ:
                    ann = symbolic_database(self.ore_algebra(),operator,n,k)
                    ord = ann.order()
                    singularities_positive = ann.singularities()
                    singularities_negative = set()
                    if self._backward_calculation is True:
                        singularities_negative = set([i for i in ann.singularities(True) if i < 0])
                    int_val = set(range(ord)).union(singularities_positive, singularities_negative)
                    initial_val = {i: exp.subs(inner == var('n'))(n = i) for i in int_val}
                    #if inner == n we are done
                    if inner == n:
                        return UnivariateDFiniteSequence(self,ann,initial_val)
                    #otherwise we need composition
                    else:
                        return UnivariateDFiniteSequence(self,ann,initial_val)(inner)
                else:
                    raise TypeError("inner argument has to be of the form u*x + v, with u,v rational")
            #functions
            else:
                x = R.gen()
                #check if inner is of the form u*x + v
                if inner.derivative() in QQ:
                    ann = symbolic_database(self.ore_algebra(),operator,inner,k)
                    ord = ann.order()
                    s_ann = ann.to_S(OreAlgebra(R.change_var('n'),"Sn"))
                    int_val = set(range(ord)).union(s_ann.singularities())
                    initial_val = {i: exp.derivative(i)(x = 0)/factorial(i) for i in int_val}
                    return UnivariateDFiniteFunction(self,ann,initial_val)
                else:
                    if len(operands) == 1:
                        return self(operator(x))(self(inner))
                    else:
                        return self(operator(k,x))(self(inner))
    
#testing and information retrieving

    def __eq__(self,right):
        r"""
        Tests if the two DFiniteFunctionRings ``self``and ``right`` are equal. 
        This is the case if and only if they are defined over equal Ore algebras and have the same domain
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: D2 = DFiniteFunctionRing(A,NN)
            sage: D3 = DFiniteFunctionRing(A,ZZ)
            sage: D1 == D2
            False
            sage: D1 == D3
            True
        
        """
        try:
            return (self.ore_algebra() == right.ore_algebra() and self.domain() == right.domain())
        except:
            return False

    def is_integral_domain(self, proof = True):
        r"""
        Returns whether ``self`` is an integral domain.
        In the discrete case this is False; in the differential case this is true.
        """
        if self.ore_algebra().is_S():
            return False
        elif self.ore_algebra().is_D():
            return True
        else:
            raise NotImplementedError

    def is_noetherian(self):
        r"""
        """
        raise NotImplementedError
    
    def is_commutative(self):
        r"""
        Returns whether ``self`` is commutative.
        This is true for the function ring as well as the sequence ring
        """
        return True
            
    def construction(self):
        r"""
        """
        raise NotImplementedError

    def _coerce_map_from_(self, P):
        r"""
        If `P` is a DFiniteFunctionRing, then a coercion from `P` to ``self`` is possible if there is a
        coercion from the Ore algebra of `P` to the Ore algebra of ``self``. If `P`is not a DFiniteFunctionRing,
        then it is sufficient to have a coercion from `P` itself to the Ore algebra from ``self``.
        """
        if isinstance(P,DFiniteFunctionRing):
            return self._ore_algebra._coerce_map_from_(P.ore_algebra())
        return self._ore_algebra._coerce_map_from_(P)

    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when
        evaluated.
        """
        if self.domain() == ZZ:
            return sib.name('DFiniteFunctionRing')(sib(self.ore_algebra()),sib(self.domain()))
        else:
            return sib.name('DFiniteFunctionRing')(sib(self.ore_algebra()),sib.name('NN'))

    def _is_valid_homomorphism_(self, domain, im_gens):
        r"""
        """
        raise NotImplementedError

    def __hash__(self):
        r"""
        """
        # should be faster than just relying on the string representation
        try:
            return self._cached_hash
        except AttributeError:
            pass
        h = self._cached_hash = hash((self.base_ring(),self.base_ring().variable_name()))
        return h

    def _repr_(self):
        r"""
        """
        try:
            return self._cached_repr
        except AttributeError:
            pass
        if self.ore_algebra().is_S():
            r = "Ring of D-finite sequences over "
        else:
            r = "Ring of D-finite functions over "
        r = r + self._base_ring._repr_()
        return r

    def _latex_(self):
        r"""
        """
        return r"\mathcal{D}(" + self._ore_algebra._latex_() + ")"

    def base_ring(self):
        r"""
        Return the base ring over which the Ore algebra of the DFiniteFunctionRing is defined
        """
        return self._base_ring

    def ore_algebra(self):
        r"""
        Return the Ore algebra over which the DFiniteFunctionRing is defined
        """
        return self._ore_algebra

    def domain(self):
        r"""
        Return the domain over which the DFiniteFunctionRing is defined
        """
        return self._domain

    def characteristic(self):
        r"""
        Return the characteristic of this DFiniteFunctionRing, which is the
        same as that of its base ring.
        """
        return self._base_ring.characteristic()

    def is_finite(self):
        r"""
        Return False since DFiniteFunctionRings are not finite (unless the base
        ring is 0.)
        """
        R = self._base_ring
        if R.is_finite() and R.order() == 1:
            return True
        return False

    def is_exact(self):
        r"""
        Return True if the Ore algebra over which the DFiniteFunctionRing is defined is exact
        """
        return self.ore_algebra().is_exact()

    def is_field(self, proof = True):
        r"""
        A DFiniteFunctionRing is not a field
        """
        return False
        
    def random_element(self, degree=2, *args, **kwds):
        r"""
        Return a random D-finite object.
        
        INPUT:
        
        -``degree`` (default 2) -- the degree of the ore operator of the random D-finite object
            
        OUTPUT:
        
        A D-finite sequence/function with a random ore operator of degree ``degree`` and random initial values constisting 
        of integers between -100 and 100.
        
        EXAMPLES::
        
            #discrete case
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: D1.random_element() # random
            Univariate D-finite sequence defined by the annihilating operator (-n^2 + n)*Sn^2 + (22/9*n - 1/622)*Sn - 5/6*n^2 - n - 1 and the initial conditions {0: -88, 1: 18, 2: -49, 3: -67}
        
            #differential case
            sage: from ore_algebra import *
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: D2.random_element(3) # random
            Univariate D-finite function defined by the annihilating operator 20*x*Dx^3 + (2/31*x^2 + 1/2*x + 1/2)*Dx^2 + (2*x^2 - 2*x + 1)*Dx + x^2 - 1/6 and the coefficient sequence defined by (20*n^3 + 361/2*n^2 + 1047/2*n + 486)*Sn^4 + (1/2*n^2 + 7/2*n + 6)*Sn^3 + (2/31*n^2 - 56/31*n - 751/186)*Sn^2 + (2*n + 2)*Sn + 1 and {0: -53, 1: 69, 2: -90, 3: -86}
            
        """
        #getting the operator
        ann = self.ore_algebra().random_element(degree)
        
        #initial values and singularities
        singularities_positive = ann.singularities()
        singularities_negative = set()
        if self._backward_calculation is True:
            singularities_negative = set([i for i in ann.singularities(True) if i < 0])
        
        initial_val = set(range(degree)).union(singularities_positive, singularities_negative)
        int_val = {n:randint(-100, 100) for n in initial_val}
        
        if self.ore_algebra().is_S():
            return UnivariateDFiniteSequence(self,ann,int_val)
        else:
            return UnivariateDFiniteFunction(self,ann,int_val)

    def _an_element_(self, *args, **kwds):
        r"""
        """
        return self.random_element()

#changing

    def change_base_ring(self,R):
        r"""
        Return a copy of ``self`` but with the base ring `R`
        """
        if R is self._base_ring:
            return self
        else:
            D = DFiniteFunctionRing(self._ore_algebra.change_ring(R), self._domain)
            return D

    def change_domain(self,R):
        r"""
        Return a copy of ``self`` but with the domain `R`
        """
        if R != NN and R != ZZ:
            raise TypeError("domain not supported")

        if self.domain() == R:
            return self

        return DFiniteFunctionRing(self.ore_algebra(), R)


####################################################################################################


class DFiniteFunction(RingElement):
    r"""
    An abstract class representing objects depending on one or more differential and one or more discrete variables
    defined by an annihilating holonomic system and a suitable set of initial conditions. 
    """

#constructor

    def __init__(self, parent, ann, initial_val, is_gen = False, construct=False, cache=True):
        r"""
        Constructor for D-finite sequences and functions
        
        INPUT:
        
        - ``parent`` -- a DFiniteFunctionRing
        
        - ``ann`` -- the operator in the corresponding Ore algebra annihilating the sequence or function
        
        - ``initial_val`` -- a list of initial values, determining the sequence or function, containing at least
          as many values as the order of ``ann`` predicts. For sequences these are the first sequence terms; for functions
          the first taylor coefficients. If the annhilating operator has singularities then ``initial_val`` has to be given
          in form of a dictionary containing the initial values and the singularities. For functions ``initial_val`` can also
          be a D-finite sequence representing the coefficient sequence of the function
          
                             
        OUTPUT:
        
        Either a D-finite sequence determined by ``ann`` and its initial conditions, i.e. initial values plus possible singularities
        or a D-finite function determined by ``ann`` and the D-finite sequence of its coefficients.
        
        For examples see the constructors of ``UnivariateDFiniteSequence`` and ``UnivariateDFiniteFunction`` respectively.
                            
        """
        RingElement.__init__(self, parent)
        self._is_gen = is_gen
    
        self.__ann = parent._ore_algebra(ann)
        ord = self.__ann.order()
        singularities = self.__ann.singularities()
        if parent._backward_calculation is True:
            singularities.update([a for a in self.__ann.singularities(True) if a < 0])
        
        #converting the initial values into sage rationals if possible
        if type(initial_val) == dict:
            initial_val = {key: QQ(initial_val[key]) if initial_val[key] in QQ else initial_val[key] for key in initial_val}
        elif type(initial_val) == list:
            initial_val = [QQ(k) if k in QQ else k for k in initial_val]
        
        
        initial_conditions = set(range(ord)).union(singularities)
    
        if parent.ore_algebra().is_S():
            #lists can only be given for sequences WITHOUT singularities (then the lists contains just the initial values)
            if type(initial_val) == list:
                self._initial_values = {i:initial_val[i] for i in range(min(ord,len(initial_val)))}
            else:
                if self.parent()._backward_calculation is False:
                    self._initial_values = {keys: initial_val[keys] for keys in initial_val if keys >= 0}
                else:
                    self._initial_values = initial_val
            
            if len(self._initial_values) < len(initial_conditions):
                if parent._backward_calculation is True:
                    print("Not enough initial values")
                
                #sequence comes from a d-finite function
                if parent._backward_calculation is False and len(self._initial_values) < ord:
                    diff = len(initial_conditions) - len(self._initial_values)
                    zeros = {i:0 for i in range(-diff,0)}
                    self._initial_values.update(zeros)
                
                
        elif parent.ore_algebra().is_D():
            if isinstance(initial_val,UnivariateDFiniteSequence):
                self._initial_values = initial_val
            
            else:
                if len(initial_val) < ord:
                    raise ValueError("not enough initial conditions given")
                R = parent.ore_algebra().base_ring().change_var('n')
                A = OreAlgebra(R,'Sn')
                D = DFiniteFunctionRing(A,NN)
                ann = self.__ann.to_S(A)
                self._initial_values = UnivariateDFiniteSequence(D, ann, initial_val)
        
        else:
            raise ValueError("not a suitable D-finite function ring")

    
    def __copy__(self):
        r"""
        Return a "copy" of ``self``. This is just ``self``, since D-finite functions are immutable.
        """
        return self

# action

    def compress(self):
        r"""
        Tries to compress the D-finite object ``self`` as much as
        possible by trying to find a smaller annihilating operator and deleting
        redundant initial conditions.
        
        OUTPUT:
        
        A D-finite object which is equal to ``self`` but may consist of a smaller operator
        (in terms of the order) and less initial conditions. In the worst case if no
        compression is possible ``self`` is returned.
        
        EXAMPLES::
            
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A)
            sage: UnivariateDFiniteSequence(D1,"((n-3)*(n-5))*(Sn^2 - Sn - 1)",{0:0,1:1,5:5,7:13})
            Univariate D-finite sequence defined by the annihilating operator (n^2 - 8*n + 15)*Sn^2 + (-n^2 + 8*n - 15)*Sn - n^2 + 8*n - 15 and the initial conditions {0: 0, 1: 1, 5: 5, 7: 13}
            sage: _.compress()
            Univariate D-finite sequence defined by the annihilating operator -Sn^2 + Sn + 1 and the initial conditions {0: 0, 1: 1}
            
            sage: from ore_algebra import *
            sage: B = OreAlgebra(QQ[x],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: D2(sin(x)^2*cos(x)^2)
            Univariate D-finite function defined by the annihilating operator Dx^5 + 20*Dx^3 + 64*Dx and the coefficient sequence defined by (n^14 + 10*n^13 + 5*n^12 - 250*n^11 - 753*n^10 + 1230*n^9 + 8015*n^8 + 5450*n^7 - 21572*n^6 - 35240*n^5 + 480*n^4 + 28800*n^3 + 13824*n^2)*Sn^4 + (20*n^12 + 60*n^11 - 560*n^10 - 1800*n^9 + 4260*n^8 + 16380*n^7 - 5480*n^6 - 49200*n^5 - 21280*n^4 + 34560*n^3 + 23040*n^2)*Sn^2 + 64*n^10 - 1920*n^8 + 17472*n^6 - 52480*n^4 + 36864*n^2 and {0: 0, 1: 0, 2: 1, 3: 0, 4: -4/3, 5: 0, 6: 32/45, 7: 0, 8: -64/315}
            sage: _.compress()
            Univariate D-finite function defined by the annihilating operator Dx^3 + 16*Dx and the coefficient sequence defined by (1/2*n^3 + 3/2*n^2 + n)*Sn^2 + 8*n and {0: 0, 1: 0, 2: 1}

        """
        A = self.parent().ore_algebra()
        d = self.__ann.degree()
        r = self.__ann.order()
        ini = copy(self.initial_conditions())
        
        if A.is_S():
            n = A.base_ring().gen()
        
            #special case r = 0, here we only compute the squarefree part of the operator
            if r == 0:
                return self.reduce_factors()
            
            #if the initial values contain symbolic expressions we can't use guessing - but we can
            #try to get rid of multiple common factors in the coefficients and redundant initial conditions
            elif not all(x in QQ for x in ini.values() if x != None):
                return self.reduce_factors()
            
            #if all initial conditions are in QQ we can try to guess a smaller operator
            else:
                #computing the data needed for guessing
                data = self.expand((r+1)*(d+2)+max(50,(r+1)*(d+2)))
                ann = guess(data,A,cut=None)
                
                #we did not find a better operator
                if ann.order() > r:
                    return self.reduce_factors()
            
                #order and minimal degree
                ord = ann.order()
                min_degree = next((index for index, coeff in enumerate(ann.list()) if coeff != 0), None)
        
                #checking if the singularities for forward calculation are really needed
                singularities_old = self.singularities()
                singularities_new = ann.singularities()
                singularities_missing = set([x for x in singularities_old.symmetric_difference(singularities_new) if x > max(0,ord-1)]).union(range(ord,r))
                
                if 0 in singularities_missing:
                    singularities_missing.remove(0)
            
                while len(singularities_missing) > 0:
                    k = min(singularities_missing)
                    
                    #taking care about NONE entries
                    if self[k] is None:
                        for l in range(k,k+ord+1):
                            ann = A(n - (l - ord))*ann
                            ini.update({l: self[l]})
                            
                            if self.parent()._backward_calculation is True and l < ord - min_degree:
                                ini.update({l-ord+min_degree: self[l-ord+min_degree]})
                            singularities_missing.remove(l)
                
                    #normal entries
                    else:
                        if self[k] == ann.to_list(self.expand(k-1)[k-ord:],ord+1,k-ord)[ord]:
                            if self.parent()._backward_calculation is True and k < ord - min_degree:
                                if self[k-ord+min_degree] == ann.to_list(self.expand(k-ord+min_degree-1)[k-ord+min_degree - ord:],ord+1,k-ord+min_degree - ord)[ord]:
                                    ini.pop(k)
                                    ini.pop(k-ord+min_degree)
                            else:
                                ini.pop(k)
                        else:
                            ann = A(n - (k - ord))*ann
                            ini.update({k: self[k]})
                            if self.parent()._backward_calculation is True and k < ord - min_degree:
                                ini.update({k-ord+min_degree: self[k-ord+min_degree]})
                        singularities_missing.remove(k)
                
                #checking if the singularities for backward calculation are really needed
                if self.parent()._backward_calculation is True:
                    singularities_old = self.singularities(True)
                    singularities_new = ann.singularities(True)
                    singularities_missing = set([x for x in singularities_old.symmetric_difference(singularities_new) if x < 0])
                    
                    #computing the operator for backward calculation
                    start = self.expand(ord-1)
                    start.reverse()
                    start.pop()
                    while len(singularities_missing) > 0:
                        k = max(singularities_missing)
                        #taking care about NONE entries
                        if self[k] is None:
                            for l in range(k-ord,k+1):
                                ann = A(n - (l - min_degree))*ann
                                ini.update({l: self[l]})
                                if l >= min_degree:
                                    ini.update({l-min_degree+ord: self[l-min_degree+ord]})
                                singularities_missing.remove(l)
                        #normal entries
                        else:
                            ann_backwards = ann.annihilator_of_composition((ord-1)-n)
                            if self[k] == ann_backwards.to_list(start + self.expand(k+1),ord-k)[ord-k-1]:
                                if k >= min_degree:
                                    if self[k-min_degree+ord] == ann_backwards.to_list(start + self.expand(k-min_degree+ord),-k-min_degree)[-k-min_degree-1]:
                                        ini.pop(k)
                                        ini.pop(k-min_degree+ord)
                                else:
                                    ini.pop(k)
                            else:
                                ann = A(n - (k - min_degree))*ann
                                ini.update({k: self[k]})
                            singularities_missing.remove(k)

                result = UnivariateDFiniteSequence(self.parent(),ann,ini)
                
                if self == result:
                    return result
                else:
                    return self.reduce_factors()
        
        else:
            #compress the coefficient sequence
            seq = ini.compress()
            
            #try to guess a smaller differential operator
            if all(x in QQ for x in ini.initial_conditions().values()):
                data = self.expand((r+1)*(d+2)+max(50,(r+1)*(d+2)))
                ann = guess(data,A,cut=None)
                
                #no better operator found
                if ann.order() > r:
                    ann = self.__ann
            else:
                ann = self.__ann
           
            result = UnivariateDFiniteFunction(self.parent(),ann,seq)
            
            if self == result:
                return result
            else:
                return UnivariateDFiniteFunction(self.parent(),self.__ann,seq)
                
                
    def reduce_factors(self):
        r"""
        Tries to delete factors of order 0 of the annihilating operator of ``self`` which appear more than
        once. Additionally this method tries to delete redundant initial conditions. This method is a subroutine
        of compress
        
        OUTPUT:
        
        A D-finite object which is equal to ``self`` but may consist of a smaller operator
        (in terms of the degree) and less initial conditions. In the worst case if no
        reduction is possible ``self`` is returned.
        
        EXAMPLES::

            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A)
            sage: UnivariateDFiniteSequence(D1,"((n-3)*(n-5))*(Sn^2 - Sn - 1)",{0:0,1:1,5:5,7:13})
            Univariate D-finite sequence defined by the annihilating operator (n^2 - 8*n + 15)*Sn^2 + (-n^2 + 8*n - 15)*Sn - n^2 + 8*n - 15 and the initial conditions {0: 0, 1: 1, 5: 5, 7: 13}
            sage: _.reduce_factors()
            Univariate D-finite sequence defined by the annihilating operator Sn^2 - Sn - 1 and the initial conditions {0: 0, 1: 1}

            sage: B = OreAlgebra(QQ[x],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: D2(sin(x)^2*cos(x)^2)
            Univariate D-finite function defined by the annihilating operator Dx^5 + 20*Dx^3 + 64*Dx and the coefficient sequence defined by (n^14 + 10*n^13 + 5*n^12 - 250*n^11 - 753*n^10 + 1230*n^9 + 8015*n^8 + 5450*n^7 - 21572*n^6 - 35240*n^5 + 480*n^4 + 28800*n^3 + 13824*n^2)*Sn^4 + (20*n^12 + 60*n^11 - 560*n^10 - 1800*n^9 + 4260*n^8 + 16380*n^7 - 5480*n^6 - 49200*n^5 - 21280*n^4 + 34560*n^3 + 23040*n^2)*Sn^2 + 64*n^10 - 1920*n^8 + 17472*n^6 - 52480*n^4 + 36864*n^2 and {0: 0, 1: 0, 2: 1, 3: 0, 4: -4/3, 5: 0, 6: 32/45, 7: 0, 8: -64/315}
            sage: _.reduce_factors()
            Univariate D-finite function defined by the annihilating operator Dx^5 + 20*Dx^3 + 64*Dx and the coefficient sequence defined by (n^9 + 20*n^8 + 170*n^7 + 800*n^6 + 2273*n^5 + 3980*n^4 + 4180*n^3 + 2400*n^2 + 576*n)*Sn^4 + (20*n^7 + 260*n^6 + 1340*n^5 + 3500*n^4 + 4880*n^3 + 3440*n^2 + 960*n)*Sn^2 + 64*n^5 + 640*n^4 + 2240*n^3 + 3200*n^2 + 1536*n and {0: 0, 1: 0, 2: 1, 3: 0, 4: -4/3}
        """
        A = self.parent().ore_algebra()
        n = A.is_S()
        ini = copy(self.initial_conditions())
        ann = self.__ann
                
        #order and minimal degree
        ord = ann.order()
        min_degree = next((index for index, coeff in enumerate(ann.list()) if coeff != 0), None)
                
        #killing multiple common factors in coefficients
        g = gcd(ann.coefficients())
        g_fac = g.factor()
        g_roots = [r for (r,m) in g.roots()]
        multiple_factors = prod([factor for (factor,power) in g_fac if power > 1])
        ann = A([coeff/multiple_factors for coeff in ann.coefficients(sparse = False)])
                
        if n:
            # checking if all positive initial conditions are really needed
            singularities_pos = set([x+ord for x in g_roots if x+ord > max(0,ord-1)])
            while len(singularities_pos) > 0:
                k = min(singularities_pos)
                #taking care about NONE entries
                if self[k] is None:
                    for l in range(k,k+ord+1):
                        singularities_pos.remove(l)
                
                #normal entries
                else:
                    ann = A([coeff/(n - (k - ord)) for coeff in ann.coefficients(sparse = False)])
                    if self[k] == ann.to_list(self.expand(k-1)[k-ord:],ord+1,k-ord)[ord]:
                        if self.parent()._backward_calculation is True and k < ord - min_degree:
                            if self[k-ord+min_degree] == ann.to_list(self.expand(k-ord+min_degree-1)[k-ord+min_degree -ord:],ord+1,k-ord+min_degree-ord)[ord]:
                                ini.pop(k)
                                ini.pop(k-ord+min_degree)
                            else:
                                ann = A(n - (k - ord))*ann
                        else:
                            ini.pop(k)
                    else:
                        ann = A(n - (k - ord))*ann
                    singularities_pos.remove(k)
        
            #checking if all negative initial conditions are really needed
            if self.parent()._backward_calculation is True:
                start = self.expand(ord-1)
                start.reverse()
                start.pop()
                singularities_neg = set([x+min_degree for x in g_roots if x+min_degree < 0])
                while len(singularities_neg) > 0:
                    k = max(singularities_neg)
                    #taking care of None entries
                    if self[k] is None:
                        for l in range(k-ord,k+1):
                            singularities_neg.remove(l)
                    #normal entries
                    else:
                        ann = A([coeff/(n - (k - min_degree)) for coeff in ann.coefficients(sparse = False)])
                        ann_backwards = ann.annihilator_of_composition((ord-1)-n)
                        if self[k] == ann_backwards.to_list(start + self.expand(k+1),ord-k)[ord-k-1]:
                            if k >= min_degree:
                                if self[k-min_degree+ord] == ann_backwards.to_list(start + self.expand(k-min_degree+ord),-k-min_degree)[-k-min_degree-1]:
                                    ini.pop(k)
                                    ini.pop(k-min_degree+ord)
                                else:
                                    ann = A(n - (k - min_degree))*ann
                            else:
                                ini.pop(k)
                        else:
                            ann = A(n - (k - min_degree))*ann
                        singularities_neg.remove(k)
        
            result = UnivariateDFiniteSequence(self.parent(),ann,ini)
    
        else:
            result = UnivariateDFiniteFunction(self.parent(), ann, self.initial_conditions().reduce_factors())
        
        #checking if the result is indeed equal to the input
        if self == result:
            return result
        else:
            return self

    def __call__(self, *x, **kwds):
        r"""
        Lets ``self`` act on ``x`` and returns the result.
        ``x`` may be either a constant, then this computes an evaluation,
        or a (suitable) expression, then it represents composition and we return a new DFiniteFunction object.
        """
        raise NotImplementedError
    
    def singularities(self, backwards = False):
        r"""
        Returns the integer singularities of the annihilating operator of ``self``.
        
        INPUT:
        
        - ``backwards`` (default ``False``) -- boolean value that decides whether the singularities needed for the forward calculation
          are returned or those for backward calculation.
          
        OUTPUT:
        
        - If ``backwards`` is ``False``, a set containing the roots of the leading coefficient of the annihilator of ``self`` shifted by 
          its order is returned
        - If ``backwards`` is ``True``, a set containing the roots of the coefficient corresponding to the term of minimal order 
          (regarding `Sn` or `Dx` respectively) is returned; shifted by the order of this term
          
        EXAMPLES::

            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: a = UnivariateDFiniteSequence(D,"(n-3)*(n+2)*Sn^3 + n^2*Sn^2 - (n-1)*(n+5)*Sn", {0:0,1:1,2:2,6:1,-4:1})
            sage: a.singularities()
            {1, 6}
            sage: a.singularities(True)
            {-4, 2}
        """
        return self.__ann.singularities(backwards)
    
    def critical_points(self, order = None, backwards = False):
        r"""
        Returns the singularities of ``self`` and the values around those singularities that can be affected.
        
        INPUT:
        
        - ``order`` (default: the order of the annihilating operator of ``self``) -- nonnegative integer that determines how many values
          after or before each singularity are returned
        
        - ``backwards`` (default ``False``) -- boolean value that determines whether we are interested in the critical points for forward calculation, 
          i.e. the singularities of the leading coefficient and ``order`` many values after each singularity, or in those for backward calculation, i.e.
          the singularities of the coefficient of minimal degree (regarding `Sn` or `Dx` respectively) and ``order`` many values before each singularity.
        
        OUTPUT:
        
        A set containing the critical points for forward calculation (if ``backwards`` is False) or those for backward calculation.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: a = UnivariateDFiniteSequence(D,"(n-3)*(n+2)*Sn^3 + n^2*Sn^2 - (n-1)*(n+5)*Sn", {0:0,1:1,2:2,6:1,-4:1})
            sage: a.critical_points()
            {1, 2, 3, 4, 6, 7, 8, 9}
            sage: a.critical_points(2,True)
            {-6, -5, -4}
            
        """
        if order is None:
            ord = self.__ann.order()
        else:
            ord = order
        
        critical_points = set()
        
        if backwards is False:
            singularities_positive = self.__ann.singularities()
            for n in singularities_positive:
                critical_points.update(range(n,n+ord+1))
        
        elif self.parent()._backward_calculation is True:
            singularities_negative = set([i for i in self.__ann.singularities(True) if i < 0])
            for n in singularities_negative:
                critical_points.update(range(n-ord,n+1))
        
        return critical_points

#tests

    def __is_zero__(self):
        r"""
        Return whether ``self`` is the zero sequence 0,0,0,\dots or the zero function f(x) = 0 \forall x, respectively.
        This is the case iff all the initial conditions are 0 or ``None``.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: a = D1(0)
            sage: a.__is_zero__()
            True
            sage: a = D1.random_element()
            sage: a.__is_zero__()
            False
        
        """
        if self.parent().ore_algebra().is_S():
            for x in self.initial_conditions():
                if self[x] != 0 and self[x] is not None:
                    return False
            return True
        else:
            return self.initial_conditions().__is_zero__()
    


    def __eq__(self,right):
        r"""
        Return whether the two DFiniteFunctions ``self`` and ``right`` are equal.
        More precicsely it is tested if the difference of ``self`` and ``right`` equals 0.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: a = D([0,1,1,2,3,5])
            sage: b = UnivariateDFiniteSequence(D,"Sn^2-Sn-1",[0,1])
            sage: a == b
            True
            
        """
        if self.parent() != right.parent():
            right = self.parent()(right)
        return (self.__add_without_compress__(-right)).__is_zero__()

    
    def __ne__(self,right):
        r"""
        Return ``True``if the DFiniteFunctions ``self`` and ``right`` are NOT equal; ``False`` otherwise
        
        """
        return not self.__eq__(right)

    def _is_atomic(self):
        r"""
        """
        raise NotImplementedError

    def is_unit(self):
        r"""
        Return ``True`` if ``self`` is a unit.

        This is the case if the annihialting operator of ``self`` has order 1.
        Otherwise we can not decide whether ``self`` is a unit or not.
        """
        if self.__ann.order() == 1:
            return True
        raise NotImplementedError
       
    def is_gen(self):
        r"""
        Return ``False``; the parent ring is not finitely generated.
        """
        return False
    
    def prec(self):
        r"""
        Return the precision of this object. 
        """
        return Infinity
    
    def change_variable_name(self, var):
        r"""
        Return a copy of ``self`` but with an Ore operator in the variable ``var``
        
        INPUT:
        
        - ``var`` -- the new variable
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(ZZ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: a = UnivariateDFiniteSequence(D, "Sn**2 - Sn - 1", [0,1])
            sage: c = a.change_variable_name('x')
            sage: a
            Univariate D-finite sequence defined by the annihilating operator Sn^2 - Sn - 1 and the initial conditions {0: 0, 1: 1}
            sage: c
            Univariate D-finite sequence defined by the annihilating operator x^2 - x - 1 and the initial conditions {0: 0, 1: 1}
        
        """
        D = DFiniteFunctionRing(self.parent().ore_algebra().change_var(var),self.parent()._domain)
        if self.parent().ore_algebra().is_S():
            return UnivariateDFiniteSequence(D, self.__ann, self._initial_values)
        else:
            return UnivariateDFiniteFunction(D,self.__ann, self._initial_values)
        
    def change_ring(self, R):
        r"""
        Return a copy of ``self`` but with an annihilating operator of an Ore algebra over ``R``
        
        """
        D = self.parent().change_base_ring(R)
        if self.parent().ore_algebra().is_S():
            return UnivariateDFiniteSequence(D, self.__ann, self._initial_values)
        else:
            return UnivariateDFiniteFunction(D,self.__ann, self._initial_values)

    def __getitem__(self, n):
        r"""
        """
        raise NotImplementedError

    def __setitem__(self, n, value):
        r"""
        """
        raise IndexError("D-finite functions are immutable")

    def __iter__(self):
        raise NotImplementedError

#conversion

    def __float__(self):
        r"""
        Tries to convert ``self`` into a float.
        This is possible iff ``self`` represents a constant sequence or constant function for some constant values in ``QQ``.
        If the conversion is not possible an error message is displayed.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: a = D1(3.4)
            sage: b = D2(4)
            sage: float(a)
            3.4
            sage: float(b)
            4.0
            
        """
        i = self._test_conversion_()
        if i is not None:
            return float(i)
        
        raise TypeError("no conversion possible")
    
    def __int__(self):
        r"""
        Tries to convert ``self`` into an integer.
        This is possible iff ``self`` represents a constant sequence or constant function for some constant value in ``ZZ``.
        If the conversion is not possible an error message is displayed.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: a = D1(3.4)
            sage: b = D2(4)
            sage: int(b) #int(a) would lead to an error message
            4

        """
        i = self._test_conversion_()
        if i is not None and i in ZZ:
            return int(i)
        
        raise TypeError("no conversion possible")

    def _integer_(self, ZZ):
        r"""
        Tries to convert ``self`` into a Sage integer.
        This is possible iff ``self`` represents a constant sequence or constant function for some constant value in ``ZZ``.
        If the conversion is not possible an error message is displayed.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: a = D1(3.4)
            sage: b = D2(4)
            sage: ZZ(b) #ZZ(a) would lead to an error message
            4

        """
        return ZZ(int(self))

    def _rational_(self):
        r"""
        Tries to convert ``self`` into a Sage rational.
        This is possible iff ``self`` represents a constant sequence or constant function for some constant value in ``QQ``.
        If the conversion is not possible an error message is displayed.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: a = D1(3.4)
            sage: b = D2(4)
            sage: QQ(a)
            17/5
            sage: QQ(b)
            4
            
        """
        i = self._test_conversion_()
        if i is not None and i in QQ:
            return QQ(i)
        
        raise TypeError("no conversion possible")
    
    def _symbolic_(self, R):
        raise NotImplementedError

#representation

    def _repr(self, name=None):
        return self._repr_()

    def _repr_(self):
        r"""
        """
        r = "Univariate D-finite "
        if self.parent().ore_algebra().is_S():
            r = r + "sequence defined by the annihilating operator "
            r = r + str(self.__ann) + " and the initial conditions "
            r = r + pprint.pformat(self._initial_values, width=2**30)
        else:
            r = r + "function defined by the annihilating operator "
            r = r + str(self.__ann) + " and the coefficient sequence defined by "
            r = r + str(self.initial_conditions().__ann) + " and "
            r = r + pprint.pformat(self.initial_conditions().initial_conditions(), width=2**30)
            
        return r

    def _latex_(self, name=None):
        r"""
        """
        if self.parent().ore_algebra().is_S():
            r = '\\text{D-finite sequence defined by the annihilating operator }'
            r = r + latex(self.__ann) + '\\text{ and the initial conditions }'
            r = r + latex(self.initial_conditions())
        else:
            r = '\\text{D-finite function defined by the annihilating operator }'
            r = r + latex(self.__ann) + '\\text{ and the coefficient sequence defined by }'
            r = r + latex(self.initial_conditions().__ann) + '\\text{ and }' + latex(self.initial_conditions().initial_conditions())

        return r
        
    def _sage_input_(self, sib, coerced):
        r"""
        Produce an expression which will reproduce ``self`` when evaluated.
        
        """
        par = self.parent()
        int_cond = self.initial_conditions()
        if par.ore_algebra().is_S():
            init = sib({sib.int(a):sib.int(int_cond[a]) for a in int_cond})
            result = sib.name('UnivariateDFiniteSequence')(sib(par),sib(self.__ann),init)
        else:
            result = sib.name('UnivariateDFiniteFunction')(sib(par),sib(self.__ann),sib(int_cond))
        return result

    def dict(self):
        raise NotImplementedError

    def list(self):
        raise NotImplementedError

# arithmetic

    def __invert__(self):
        r"""
        works if 1/self is again d-finite. 
        """
        raise NotImplementedError

    def __div__(self, right):
        r"""
        This is division, not division with remainder. Works only if 1/right is d-finite. 
        """
        return self*right.__invert__()

    def __pow__(self, n, modulus = None):
        r"""
        """
        return self._pow(n)
        
    def _pow(self, n):
        r"""
        Return ``self`` to the n-th power
        
        INPUT:
        
        - ``n`` -- a non-negative integer
        
        OUTPUT:
        
        self^n
        
        EXAMPLES::
        
            #discrete case
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: n = A.base_ring().gen()
            sage: a = D1(n)
            sage: a**3
            Univariate D-finite sequence defined by the annihilating operator (n^10 + 4*n^9 - 2*n^8 - 20*n^7 - 11*n^6 + 16*n^5 + 12*n^4)*Sn - n^10 - 7*n^9 - 13*n^8 + 13*n^7 + 73*n^6 + 79*n^5 - 7*n^4 - 73*n^3 - 52*n^2 - 12*n and the initial conditions {-3: -27, -2: -8, -1: -1, 0: 0, 1: 1, 2: 8, 3: 27}
            
            #differential case
            sage: from ore_algebra import *
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: x = B.base_ring().gen()
            sage: b = D2(x^2)
            sage: b**2
            Univariate D-finite function defined by the annihilating operator x*Dx - 4 and the coefficient sequence defined by n^2 - 6*n + 8 and {2: 0, 4: 1}
            
        """
        if n == 0:
            return self.parent().one()
        if n == 1:
            return self
        
        #for small n the traditional method is faster
        if n <= 10:
            return self * (self._pow(n-1))

        #for larger n we use repeated squaring
        else:
            result = self.parent().one()
            bit = bin(n)[2:] #binary representation of n
            for i in range(len(bit)):
                result = result * result
                if bit[i] == '1':
                    result = result * self
            return result
                   
    def __floordiv__(self,right):
        r"""
        """
        raise NotImplementedError

    def __mod__(self, other):
        r"""
        """
        raise NotImplementedError

#base ring related functions
        
    def base_ring(self):
        r"""
        Return the base ring of the parent of ``self``.
        
        """
        return self.parent().base_ring()

#part extraction functions

    def ann(self):
        r"""
        Return the annihilating operator of ``self``
        """
        return self.__ann
    
    def initial_values(self):
        r"""
        Return the initial values of ``self`` in form of a list.
        
        In the discrete case those are the first `r` sequence terms, where `r` is the order of the annihilating
        operator of ``self``. In the differential case those are the first `r` coefficients of ``self``, where
        `r` is again the order of the annihilating operator of ``self``.
        Singularities that might be saved will not be considered, unless they are within the first `r` terms. To get all saved
        values (initial values plus singularities) use the method ``initial_conditions``
        
        EXAMPLES::
        
            #discrete case
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: a = UnivariateDFiniteSequence(D1, "(n+3)*(n-2)*Sn^2 + Sn + 4*n", {0:0,1:1,4:3,-1:2})
            sage: a.initial_values()
            [0, 1]
            
            #differential case
            sage: from ore_algebra import *
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: b = UnivariateDFiniteFunction(D2, "(x-3)*Dx - 1", {0:-3})
            sage: b.initial_values()
            [-3]
    
        """
        if self.parent().ore_algebra().is_S():
            if self.parent()._backward_calculation is False and min(self.initial_conditions()) < 0:
                m = min(self.initial_conditions())
                result = [self._initial_values[key] for key in range(m,self.__ann.order()+m)]
            else:
                result = [self._initial_values[key] for key in range(self.__ann.order())]
            return result
        else:
             return self._initial_values.expand(self.__ann.order()-1)

    def initial_conditions(self):
        r"""
        Return all initial conditions of ``self``.
        
        In the discrete case the initial conditions are all values that are saved, i.e. the initial values and all singularities.
        In the differential case this method will return the coefficient sequence of ``self`` in form of a UnivariateDFiniteSequence object.
        To get all saved values of a UnivariateDFiniteFunction one has to call this method twice (see examples).
        
        EXAMPLES::
          
            #discrete case
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D1 = DFiniteFunctionRing(A,ZZ)
            sage: a = UnivariateDFiniteSequence(D1, "(n+3)*(n-2)*Sn^2 + Sn + 4*n", {0:0,1:1,4:3,-1:2})
            sage: a.initial_conditions()
            {-1: 2, 0: 0, 1: 1, 4: 3}
            
            #differential case
            sage: from ore_algebra import *
            sage: B = OreAlgebra(QQ['x'],'Dx')
            sage: D2 = DFiniteFunctionRing(B)
            sage: b = UnivariateDFiniteFunction(D2, "(x-3)*Dx - 1", {0:-3})
            sage: b.initial_conditions()
            Univariate D-finite sequence defined by the annihilating operator (-3*n - 3)*Sn + n - 1 and the initial conditions {0: -3}
            sage: b.initial_conditions().initial_conditions()
            {0: -3}
                        
        """
        return self._initial_values

#############################################################################################################
    
class UnivariateDFiniteSequence(DFiniteFunction):
    r"""
    D-finite sequence in a single discrete variable.
    """
    
#constructor

    def __init__(self, parent, ann, initial_val, is_gen=False, construct=False, cache=True):
        r"""
        Constructor for a D-finite sequence in a single discrete variable.
        
        INPUT:
        
        - ``parent`` -- a DFiniteFunctionRing defined over an OreAlgebra with the shift operator
        
        - ``ann`` -- an annihilating operator, i.e. an element from the OreAlgebra over which the DFiniteFunctionRing is defined,
           that defines a differential equation for the function ``self`` should represent.
           
        - ``initial_val`` -- either a dictionary (or a list if no singularities occur) which contains the first r sequence terms (and all singularities if
          there are some) of ``self``, where r is the order of ``ann``
          
        OUTPUT:
        
        An object consisting of ``ann`` and a dictionary that represents the D-finite sequence which is annihilated by ``ann``, has the initial values that 
        appear in the dictionary and at all singularities of ``ann`` has the values that the dictionary predicts.
       
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: n = A.base_ring().gen()
            sage: Sn = A.gen()
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: UnivariateDFiniteSequence(D,Sn^2 - Sn - 1, [0,1])
            Univariate D-finite sequence defined by the annihilating operator Sn^2 - Sn - 1 and the initial conditions {0: 0, 1: 1}
            sage: UnivariateDFiniteSequence(D, (n^2 - n)*Sn - n^2 - n, {0: 0, 1: 0, 2: 2, -1: 2})
            Univariate D-finite sequence defined by the annihilating operator (n^2 - n)*Sn - n^2 - n and the initial conditions {-1: 2, 0: 0, 1: 0, 2: 2}
            
        """
        if not parent.ore_algebra().is_S():
            raise TypeError("Not the Shift Operator")
        super(UnivariateDFiniteSequence, self).__init__(parent, ann, initial_val, is_gen, construct, cache)

#action

    def __call__(self, x):
        r"""
        Lets ``self`` act on `x`.
        
        If `x` is an integer (or a float, which then gets ``cut`` to an integer) the x-th sequence term
        is returned. This is also possible for negative `x` if the DFiniteFunctionRing is defined
        over the domain ZZ. If `x` is a suitable expression, i.e. of the form x = u*n + v for
        some u,v in QQ, it is interpreted as the composition self(floor(x(n)))
        
        EXAMPLES::

            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: n = A.base_ring().gen()
            sage: a = UnivariateDFiniteSequence(D, "Sn^2 - Sn - 1", [0,1]) #the Fibonacci numbers
            sage: a(-5)
            5
            sage: a(2*n+3).expand(10) #the odd Fibonacci numbers staring with a_3
            [2, 5, 13, 34, 89, 233, 610, 1597, 4181, 10946, 28657]

        """
        try:
            #x is a number
            n = int(x)
        except:
            #x is of the form u*n + v
            y = var('y')
            
            #getting the operator
            A = self.parent().ore_algebra()
            N = A.is_S()
            if isinstance(x,UnivariateDFiniteSequence):
                x = x.to_polynomial()
            else:
                x = QQ[N](x)
            ann = self.ann().annihilator_of_composition(x)
            
            #getting the largest and smallest degree of the new operator
            ord = ann.order()
            min_degree = next((index for index, coeff in enumerate(ann.list()) if coeff != 0), None)
            
            #initial values and singularities of the new operator
            singularities_positive = ann.singularities()
            singularities_negative = set()
            if self.parent()._backward_calculation is True:
                singularities_negative = set([i for i in ann.singularities(True) if i < 0])
        
            initial_val = set(range(ord)).union(singularities_positive, singularities_negative)
            int_val = {n:self[floor(x(n))] for n in initial_val}
                
            #critical points for forward calculation
            critical_points_positive = set()
            for n in singularities_positive:
                critical_points_positive.update(range(n+1,n+ord+1))
            
            for n in self.critical_points(ord):
                k = ceil(solve( n == x(y), y)[0].rhs())
                if n == floor(x(k)):
                    critical_points_positive.update([k])
            
            for n in critical_points_positive:
                int_val.update({n:self[floor(x(n))]})
                ann = A(N - (n - ord) )*ann
                if self.parent()._backward_calculation is True and n < ord - min_degree:
                    int_val.update({(n-ord)+min_degree: self[floor(x(n-ord+min_degree))]})
                
            #critical points for backward calculation
            critical_points_negative = set()
            for n in singularities_negative:
                critical_points_negative.update(range(n-ord,n))
            
            for n in self.critical_points(ord,True):
                k = ceil(solve( n == x(y), y)[0].rhs())
                if n == floor(x(k)):
                    critical_points_negative.update([k])
            
            for n in critical_points_negative:
                    int_val.update({n:self[floor(x(n))]})
                    ann = A(N - (n - min_degree) )*ann                                     
                    if n >= min_degree:
                        int_val.update({(n-min_degree)+ord : self[floor(x(n-min_degree+ord))]})

            return UnivariateDFiniteSequence(self.parent(), ann, int_val)
            
        return self[n]

    def _test_conversion_(self):
        r"""
        Test whether a conversion of ``self`` into an int/float/long/... is possible;
        i.e. whether the sequence is constant or not.
        
        OUTPUT:
        
        If ``self`` is constant, i.e. there exists a `k` in QQ, such that self(n) = k for all n in NN,
        then this value `k` is returned. If ``self`` is not constant ``None`` is returned.
        
        EXAMPLES::
            
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: n = A.base_ring().gen()
            sage: a = D(3)
            sage: b = D(n)
            sage: a._test_conversion_()
            3
            sage: b._test_conversion_() # returns None
            
            
        """
        ini = self.initial_values()
        if len(ini) > 0:
            i = self.initial_values()[0]
        else:
            i = 0
        if all(x == i for x in self.initial_values()):
            Sn = self.parent().ore_algebra().gen()
            if self.ann().quo_rem(Sn-1)[1].is_zero():
                return i
        return None
    
    def dict(self):
        r"""
        """
        raise NotImplementedError

    def list(self):
        r"""
        """
        raise NotImplementedError
    
    def to_polynomial(self):
        r"""
        Try to convert ``self`` into a polynomial.
        
        OUTPUT:
        
        Either a polynomial f(n) from the base ring of the OreAlgebra of the annihilating operator of ``self`` such that self(n) = f(n)
        for all n in NN or an error message if no such polynomial exists.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: n = A.base_ring().gen()
            sage: a = D(3*n^3 + 5*n - 10)
            sage: a.to_polynomial()
            3*n^3 + 5*n - 10
            sage: l = D(legendre_P(5,n))
            sage: l
            Univariate D-finite sequence defined by the annihilating operator (63/8*n^5 - 35/4*n^3 + 15/8*n)*Sn - 63/8*n^5 - 315/8*n^4 - 70*n^3 - 105/2*n^2 - 15*n - 1 and the initial conditions {-1: -1, 0: 0, 1: 1}
            sage: l.to_polynomial()
            63/8*n^5 - 35/4*n^3 + 15/8*n

        """
        #don`t want to care about None entries
        max_pol = max(self.initial_conditions()) + 1
        
        R = self.parent().base_ring()
        n = R.gen()
        
        if self.__is_zero__():
            return R.zero()
        
        #computing a base of the solution space
        base = self.ann().polynomial_solutions()
        if len(base) == 0:
            raise TypeError("the D-finite sequence does not come from a polynomial")
        
        #generating an equation system
        vars = list(var('x_%d' % i) for i in range(len(base)))
        c = [0]*len(base)
        for i in range(len(base)):
            base[i] = base[i][0]
            c[i] = base[i]*vars[i]
        poly = sum(c)
        eqs = list(poly(n = k) == self[k] for k in range(max_pol, len(base)+max_pol))
        
        #solving the system and putting results together
        result = solve(eqs,vars)
        if len(result) == 0:
            raise TypeError("the D-finite sequence does not come from a polynomial")
        if type(result[0]) == list:
            result = result[0]
        coeffs_result = [0]*len(result)
        for i in range(len(result)):
            coeffs_result[i] = result[i].rhs()
        result = sum(list(a*b for a,b in zip(coeffs_result,base)))
        
        #checking if the polynomial also yields the correct values for all singularities (except from pols)
        if all(result(n = k) == self[k] for k in self.initial_conditions() if self[k] is not None):
            return R(result)
        else:
            raise TypeError("the D-finite sequence does not come from a polynomial")

    def to_rational(self):
        r"""
        Try to convert ``self`` into a rational function.
        
        OUTPUT:
        
        Either a rational function r(n) from the fraction field of the base ring of the OreAlgebra of the annihilating 
        operator of ``self`` such that self(n) = r(n) for all n in NN (eventually except from pols) or an error message
        if no such rational function exists.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: n = A.base_ring().gen()
            sage: a = D((n^2+3*n-4)/(n^5+4*n^3+10*n))
            sage: a.to_rational()
            (n^2 + 3*n - 4)/(n^5 + 4*n^3 + 10*n)
            sage: l = D(legendre_P(5,n)/legendre_P(3,n))
            sage: l
            Univariate D-finite sequence defined by the annihilating operator (63/20*n^6 + 63/10*n^5 - 56/25*n^4 - 7*n^3 - 13/20*n^2 + 3/2*n + 3/10)*Sn - 63/20*n^6 - 63/5*n^5 - 1351/100*n^4 + 49/25*n^3 + 221/25*n^2 + 84/25*n + 6/25 and the initial conditions {0: -5/4}
            sage: l.to_rational()
            (63/20*n^4 - 7/2*n^2 + 3/4)/(n^2 - 3/5)
            
        """
        #don`t want to care about None entries
        max_pol = max(self.initial_conditions()) + 1
        
        R = self.parent().base_ring()
        n = R.gen()
        
        if self.__is_zero__():
            return R.fraction_field().zero()
        
        #computing a base of the solution space
        base = self.ann().rational_solutions()
        if len(base) == 0:
            raise TypeError("the D-finite sequence does not come from a rational function")
        
        #generating an equation system
        vars = list(var('x_%d' % i) for i in range(len(base)))
        c = [0]*len(base)
        for i in range(len(base)):
            base[i] = base[i][0]
            c[i] = base[i]*vars[i]
        rat = sum(c)
        num = rat.numerator()
        denom = rat.denominator()
        eqs = list(num.subs(n = k) == denom.subs(n = k)*self[k] for k in range(max_pol, len(base)+max_pol))
        
        #solving the system and putting results together
        result = solve(eqs,vars)
        if len(result) == 0:
            raise TypeError("the D-finite sequence does not come from a rational function")
        if type(result[0]) == list:
            result = result[0]
        coeffs_result = [0]*len(result)
        for i in range(len(result)):
            coeffs_result[i] = result[i].rhs()
        result = sum(list(a*b for a,b in zip(coeffs_result,base)))
        
        #checking if the ratinoal function also yields the correct values for all singularities (except from pols)
        if all(result(n = k) == self[k] for k in self.initial_conditions() if self[k] is not None):
            return R.fraction_field()(result)
        else:
            raise TypeError("the D-finite sequence does not come from a rational function")


    def generating_function(self):
        r"""
        """
        A = OreAlgebra(QQ['x'],'Dx')
        D = DFiniteFunctionRing(A)
        return UnivariateDFiniteFunction(D,self.ann().to_D(A),self)
    
    def __add_without_compress__(self,right):
        r"""
        Adds the D-finite sequences ``self`` and ``right`` without automatically trying
        to compress the result. This method is called whenever equality testing is done
        because in that case compressing the result would be unnecessary work.
        """
        #getting the operator
        N = self.parent().base_ring().gen()
        A = self.parent().ore_algebra()
        sum_ann = self.ann().lclm(right.ann())
        
        #getting the largest and smallest degree of the operator
        ord = sum_ann.order()
        min_degree = next((index for index, coeff in enumerate(sum_ann.list()) if coeff != 0), None)

        #initial values and singularities of the new operator
        singularities_positive = sum_ann.singularities()
        singularities_negative = set()
        if self.parent()._backward_calculation is True:
            singularities_negative = set([i for i in sum_ann.singularities(True) if i < 0])
    
        initial_val = set(range(ord)).union(singularities_positive, singularities_negative)
        int_val_sum = {n:self[n] + right[n] if (self[n] is not None and right[n] is not None) else None for n in initial_val}

        #critical points for forward calculation
        critical_points_positive = self.critical_points(ord).union( right.critical_points(ord) )
        for n in singularities_positive:
            critical_points_positive.update(range(n+1,n+ord+1))
        
        for n in critical_points_positive:
            int_val_sum.update({n:self[n] + right[n] if (self[n] is not None and right[n] is not None) else None})
            sum_ann = A(N - (n - ord) )*sum_ann
            if self.parent()._backward_calculation is True and n < ord - min_degree:
                int_val_sum.update({(n-ord)+min_degree: self[(n-ord)+min_degree] + right[(n-ord)+min_degree] if (self[(n-ord)+min_degree] is not None and right[(n-ord)+min_degree] is not None) else None})
        
        #critical points for backward calculation
        critical_points_negative = self.critical_points(ord,True).union( right.critical_points(ord,True) )
        for n in singularities_negative:
            critical_points_negative.update(range(n-ord,n))
            
        for n in critical_points_negative:
            int_val_sum.update({n:self[n] + right[n] if (self[n] is not None and right[n] is not None) else None})
            sum_ann = A(N - (n - min_degree) )*sum_ann
            if n >= min_degree:
                int_val_sum.update({(n-min_degree)+ord:self[(n-min_degree)+ord] + right[(n-min_degree)+ord] if (self[(n-min_degree)+ord] is not None and right[(n-min_degree)+ord] is not None) else None})
                
        return UnivariateDFiniteSequence(self.parent(), sum_ann, int_val_sum)

#arithmetic

    def _add_(self, right):
        r"""
        Return the sum of ``self`` and ``right``.
        
        ``_add_`` uses the method ``lclm`` from the OreAlgebra package to get the new annihilator.
        If ``self`` or ``right`` contains a ``None`` value at a certain position, then the sum will also 
        have a ``None`` entry at this position.
        Additionally the result is automatically compressed using the compress() method.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A.<Sn> = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A)
            sage: n = A.base_ring().gen()
            sage: a = UnivariateDFiniteSequence(D, Sn**2 - Sn - 1, [0,1])
            sage: b = D(harmonic_number(n))
            sage: c = a+b
            sage: c.expand(8)
            [0, 2, 5/2, 23/6, 61/12, 437/60, 209/20, 2183/140, 6641/280]
            sage: [a(i) + b(i) for i in range(9)]
            [0, 2, 5/2, 23/6, 61/12, 437/60, 209/20, 2183/140, 6641/280]
            
        """
        if self.__is_zero__():
            return right
        if right.__is_zero__():
            return self
        
        #getting the operator
        N = self.parent().base_ring().gen()
        A = self.parent().ore_algebra()
        sum_ann = self.ann().lclm(right.ann())
        
        #getting the largest and smallest degree of the operator
        ord = sum_ann.order()
        min_degree = next((index for index, coeff in enumerate(sum_ann.list()) if coeff != 0), None)

        #initial values and singularities of the new operator
        singularities_positive = sum_ann.singularities()
        singularities_negative = set()
        if self.parent()._backward_calculation is True:
            singularities_negative = set([i for i in sum_ann.singularities(True) if i < 0])
    
        initial_val = set(range(ord)).union(singularities_positive, singularities_negative)
        int_val_sum = {n:self[n] + right[n] if (self[n] != None and right[n] != None) else None for n in initial_val}

        #critical points for forward calculation
        critical_points_positive = self.critical_points(ord).union( right.critical_points(ord) )
        for n in singularities_positive:
            critical_points_positive.update(range(n+1,n+ord+1))
        
        for n in critical_points_positive:
            int_val_sum.update({n:self[n] + right[n] if (self[n] != None and right[n] != None) else None})
            sum_ann = A(N - (n - ord) )*sum_ann
            if self.parent()._backward_calculation is True and n < ord - min_degree:
                int_val_sum.update({(n-ord)+min_degree: self[(n-ord)+min_degree] + right[(n-ord)+min_degree] if (self[(n-ord)+min_degree] != None and right[(n-ord)+min_degree] != None) else None})
        
        #critical points for backward calculation
        critical_points_negative = self.critical_points(ord,True).union( right.critical_points(ord,True) )
        for n in singularities_negative:
            critical_points_negative.update(range(n-ord,n))
            
        for n in critical_points_negative:
            int_val_sum.update({n:self[n] + right[n] if (self[n] != None and right[n] != None) else None})
            sum_ann = A(N - (n - min_degree) )*sum_ann
            if n >= min_degree:
                int_val_sum.update({(n-min_degree)+ord:self[(n-min_degree)+ord] + right[(n-min_degree)+ord] if (self[(n-min_degree)+ord] != None and right[(n-min_degree)+ord] != None) else None})
        
        sum = UnivariateDFiniteSequence(self.parent(), sum_ann, int_val_sum)

        return sum.compress()
        
    def _neg_(self):
        r"""
        Return the negative of ``self``.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: n = A.base_ring().gen()
            sage: a = D(n)
            sage: -a
            Univariate D-finite sequence defined by the annihilating operator n*Sn - n - 1 and the initial conditions {-1: 1, 0: 0, 1: -1}
            sage: (-a).expand(10)
            [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]

        """
        neg_int_val = {key:(-self._initial_values[key]) if (self._initial_values[key] != None) else None for key in self._initial_values}
        return UnivariateDFiniteSequence(self.parent(), self.ann(), neg_int_val)

    def _mul_(self, right):
        r"""
        Return the product of ``self`` and ``right``
        
        The result is the termwise product (Hadamard product) of ``self`` and ``right``. To get the cauchy product
        use the method ``cauchy_product``.
        ``_mul_`` uses the method ``symmetric_product`` of the OreAlgebra package to get the new annihilator. If ``self``
        or ``right`` contains a ``None`` value at a certain position, then the product will also have a ``None`` entry at this position.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: n = A.base_ring().gen()
            sage: a = D(n)
            sage: b = D(1/n)
            sage: c = a*b
            sage: c.expand(10)
            [None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            
        """
        if self.__is_zero__() or right.__is_zero__():
            return self.parent().zero()
        
        #getting the operator
        N = self.parent().base_ring().gen()
        A = self.parent().ore_algebra()
        prod_ann = self.ann().symmetric_product(right.ann())
        
        #getting the largest and smallest degree of the operator
        ord = prod_ann.order()
        min_degree = next((index for index, coeff in enumerate(prod_ann.list()) if coeff != 0), None)

        #initial values and singularities of the new operator
        singularities_positive = prod_ann.singularities()
        singularities_negative = set()
        if self.parent()._backward_calculation is True:
            singularities_negative = set([i for i in prod_ann.singularities(True) if i < 0])
    
        initial_val = set(range(ord)).union(singularities_positive, singularities_negative)
        int_val_prod = {n:self[n] * right[n] if (self[n] != None and right[n] != None) else None for n in initial_val }
        
        #critical points for forward calculation
        critical_points_positive = self.critical_points(ord).union( right.critical_points(ord) )
        for n in singularities_positive:
            critical_points_positive.update(range(n+1,n+ord+1))
        
        for n in critical_points_positive:
            int_val_prod.update({n:self[n] * right[n] if (self[n] != None and right[n] != None) else None})
            prod_ann = A(N - (n - ord) )*prod_ann
            if self.parent()._backward_calculation is True and n < ord - min_degree:
                int_val_prod.update({(n-ord)+min_degree: self[(n-ord)+min_degree] * right[(n-ord)+min_degree] if (self[(n-ord)+min_degree] != None and right[(n-ord)+min_degree] != None) else None})
        
        #critical points for backward calculation
        critical_points_negative = self.critical_points(ord,True).union( right.critical_points(ord,True) )
        for n in singularities_negative:
            critical_points_negative.update(range(n-ord,n))
        
        for n in critical_points_negative:
            int_val_prod.update({n:self[n] * right[n] if (self[n] != None and right[n] != None) else None})
            prod_ann = A(N-(n-min_degree))*prod_ann
            if n >= min_degree:
                int_val_prod.update({(n-min_degree)+ord:self[(n-min_degree)+ord] * right[(n-min_degree)+ord] if (self[(n-min_degree)+ord] != None and right[(n-min_degree)+ord] != None) else None})
        
        prod = UnivariateDFiniteSequence(self.parent(), prod_ann, int_val_prod)
        return prod
        
        
    def cauchy_product(self, right):
        r"""
        Return the cauchy product of ``self`` and ``right``
        
        The result is the cauchy product of ``self`` and ``right``. To get the termwise product (Hadamard product)
        use the method ``_mul_``.
        This method uses the method ``symmetric_product`` (but in an OreAlgebra with the differential operator) of the 
        OreAlgebra package to get the new annihilator. If ``self`` or ``right`` contains a ``None`` value at a certain position, 
        then the cauchy product will have ``None`` entries at this position and all positions afterwards.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: a = UnivariateDFiniteSequence(D,"(n + 1)*Sn + n + 1", {0:1,-1:0}) #Taylor coefficients of 1/(x+1)
            sage: a.expand(10)
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
            sage: b = UnivariateDFiniteSequence(D,"(n + 1)*Sn + n - 1",{0:1,1:1}) #Taylor coefficients of x+1
            sage: b.expand(10)
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            sage: a.cauchy_product(b).expand(10)
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        """
        if self.__is_zero__() or right.__is_zero__():
            return self.parent().zero()
        
        #getting the operator
        R = self.parent().base_ring()
        N = R.gen()
        A = self.parent().ore_algebra()
        D = OreAlgebra(R.change_var('x'),'Dx')
        
        L = self.ann().to_D(D)
        M = right.ann().to_D(D)
        
        prod_ann = L.symmetric_product(M).to_S(A)
        
        #getting the largest and smallest degree of the operator
        ord = prod_ann.order()
        min_degree = next((index for index, coeff in enumerate(prod_ann.list()) if coeff != 0), None)
        
        #initial values and singularities of the new operator
        singularities_positive = prod_ann.singularities()
        singularities_negative = set()
        if self.parent()._backward_calculation is True:
            singularities_negative = set([i for i in prod_ann.singularities(True) if i < 0])
    
        initial_val = set(range(ord)).union(singularities_positive, singularities_negative)
        int_val_prod = {}
        for n in initial_val:
            a = self.expand(n)
            b = right.expand(n)
            b.reverse()
            if all(x != None for x in a) and all(y != None for y in b):
                cauchy = sum([x*y for x,y in zip(a,b)])
            else:
                cauchy = None
            int_val_prod.update({n:cauchy})
        
        #critical points for forward calculation
        critical_points_positive = self.critical_points(ord).union( right.critical_points(ord) )
        for n in singularities_positive:
            critical_points_positive.update(range(n+1,n+ord+1))
        
        for n in critical_points_positive:
            a = self.expand(n)
            b = right.expand(n)
            b.reverse()
            if all(x != None for x in a) and all(y != None for y in b):
                cauchy = sum([x*y for x,y in zip(a,b)])
            else:
                cauchy = None
            int_val_prod.update({n:cauchy})
            prod_ann = A(N - (n - ord) )*prod_ann
            if self.parent()._backward_calculation is True and n < ord - min_degree:
                a = self.expand((n-ord)+min_degree)
                b = right.expand((n-ord)+min_degree)
                b.reverse()
                if all(x != None for x in a) and all(y != None for y in b):
                    cauchy = sum([x*y for x,y in zip(a,b)])
                else:
                    cauchy = None
                int_val_prod.update({(n-ord)+min_degree:cauchy})

        
        #critical points for backward calculation
        critical_points_negative = self.critical_points(ord,True).union( right.critical_points(ord,True) )
        for n in singularities_negative:
            critical_points_negative.update(range(n-ord,n))
        
        for n in critical_points_negative:
            a = self.expand(n)
            b = right.expand(n)
            b.reverse()
            if all(x != None for x in a) and all(y != None for y in b):
                cauchy = sum([x*y for x,y in zip(a,b)])
            else:
                cauchy = None
            int_val_prod.update({n:cauchy})
            prod_ann = A(N-(n-min_degree))*prod_ann
            if n >= min_degree:
                a = self.expand((n-min_degree)+ord)
                b = right.expand((n-min_degree)+ord)
                b.reverse()
                if all(x != None for x in a) and all(y != None for y in b):
                    cauchy = sum([x*y for x,y in zip(a,b)])
                else:
                    cauchy = None
                int_val_prod.update({(n-min_degree)+ord:cauchy})
    
        return UnivariateDFiniteSequence(self.parent(), prod_ann, int_val_prod)
        
    def __invert__(self):
        r"""
        """
        raise NotImplementedError
    
    def interlace(self, right):
        r"""
        Return the interlaced sequence of ``self`` and ``right``.
        ``interlace`` uses the method ``annihilator_of_interlacing`` of the OreAlgebra package to get the new operator.
        
        OUTPUT:
        
        If ``self`` is of the form a_0,a_1,a_2,\dots and ``right`` is of the form b_0,b_1,b_2,\dots, then
        the result is a UnivariateDFiniteSequence object that represents the sequence a_0,b_0,a_1,b_1,a_2,b_2,\dots
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: n = A.base_ring().gen()
            sage: a = D(n)
            sage: b = D(1/n)
            sage: c = a.interlace(b)
            sage: c.expand(10)
            [0, None, 1, 1, 2, 1/2, 3, 1/3, 4, 1/4, 5]
            
        """
        #getting the operator
        N = self.parent().base_ring().gen()
        A = self.parent().ore_algebra()
        interlacing_ann = self.ann().annihilator_of_interlacing(right.ann())
        
        #getting the largest and smallest degree of the operator
        ord = interlacing_ann.order()
        min_degree = next((index for index, coeff in enumerate(interlacing_ann.list()) if coeff != 0), None)
        
        #initial values and singularities of the new operator
        singularities_positive = interlacing_ann.singularities()
        singularities_negative = set()
        if self.parent()._backward_calculation is True:
            singularities_negative = set([i for i in interlacing_ann.singularities(True) if i < 0])
        
        initial_val = set(range(ord)).union(singularities_positive, singularities_negative)
        int_val_interlacing = {}
        for n in initial_val:
            if n % 2 == 0:
                int_val_interlacing.update({n:self[n//2]})
            else:
                int_val_interlacing.update({n:right[n//2]})
        
        #critical points for forward calculation
        critical_points_positive = set()
        for n in singularities_positive:
            critical_points_positive.update(range(n+1,n+ord+1))
                
        for n in self.critical_points(ord):
                critical_points_positive.update([2*n])
        for n in right.critical_points(ord):
                critical_points_positive.update([2*n+1])

        for n in critical_points_positive:
            if n % 2 == 0:
                int_val_interlacing.update({n:self[n/2]})
            else:
                int_val_interlacing.update({n:right[floor(n/2)]})
            interlacing_ann = A(N -(n - ord))*interlacing_ann
            if self.parent()._backward_calculation is True and n < ord - min_degree:
                if (n-ord+min_degree) % 2 == 0:
                    int_val_interlacing.update({n-ord+min_degree:self[(n-ord+min_degree)/2]})
                else:
                    int_val_interlacing.update({n-ord:right[floor((n-ord+min_degree)/2)]})

        #critical points for backward calculation
        critical_points_negative = set()
        for n in singularities_negative:
            critical_points_negative.update(range(n-ord,n))

        for n in self.critical_points(ord,True):
            critical_points_negative.update([2*n])
        for n in right.critical_points(ord,True):
            critical_points_negative.update([2*n+1])

        for n in critical_points_negative:
            if n % 2 == 0:
                int_val_interlacing.update({n:self[n//2]})
            else:
                int_val_interlacing.update({n:right[n//2]})
            interlacing_ann = A(N - (n-min_degree) )*interlacing_ann
            if n >= min_degree:
                if (n+ord) % 2 == 0:
                    int_val_interlacing.update({(n-min_degree) + ord:self[((n-min_degree)+ ord)/2]})
                else:
                    int_val_interlacing.update({(n-min_degree) + ord:right[floor(((n-min_degree)+ ord)/2)]})
        
        return UnivariateDFiniteSequence(self.parent(), interlacing_ann, int_val_interlacing)
        
    
    def sum(self):
        r"""
        Return the sequence (s_n)_{n=0}^\infty with s_n = \sum_{k=0}^n self[k].
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A)
            sage: n = A.base_ring().gen()
            sage: a = D(1/(n+1))
            sage: a.sum()
            Univariate D-finite sequence defined by the annihilating operator (n + 3)*Sn^2 + (-2*n - 5)*Sn + n + 2 and the initial conditions {0: 1, 1: 3/2}
            sage: _ == D(harmonic_number(n+1))
            True
            
        """
        #only makes sense for sequences over NN
        if self.parent().domain() == ZZ:
            raise TypeError("domain of the DFiniteFunctionRing has to be NN")
        
        #getting the operator
        N = self.parent().base_ring().gen()
        A = self.parent().ore_algebra()
        sum_ann = self.ann().annihilator_of_sum()
    
        #getting the largest and smallest degree of the operator
        ord = sum_ann.order()
        min_degree = next((index for index, coeff in enumerate(sum_ann.list()) if coeff != 0), None)

        #initial values and singularities of the new operator
        singularities_positive = sum_ann.singularities()
        singularities_negative = set()
        if self.parent()._backward_calculation is True:
            singularities_negative = set([i for i in sum_ann.singularities(True) if i < 0])
    
        initial_val = set(range(ord)).union(singularities_positive, singularities_negative)
        int_val_sum = {n : sum(self.expand(n)) if all(self[k] != None for k in range(n+1)) else None for n in initial_val}

        #critical points for forward calculation
        critical_points_positive = self.critical_points(ord)
        for n in singularities_positive:
            critical_points_positive.update(range(n+1,n+ord+1))
    
        for n in critical_points_positive:
            int_val_sum.update({n : sum(self.expand(n)) if all(self[k] != None for k in range(n+1)) else None})
            sum_ann = A(N - (n - ord) )*sum_ann
            if self.parent()._backward_calculation is True and n < ord - min_degree:
                int_val_sum.update({(n-ord)+min_degree : sum(self.expand((n-ord)+min_degree)) if all(self[k] != None for k in range((n-ord)+min_degree+1)) else None})
        
        #critical points for backward calculation
        critical_points_negative = self.critical_points(ord,True)
        for n in singularities_negative:
            critical_points_negative.update(range(n-ord,n))
            
        for n in critical_points_negative:
            int_val_sum.update({n : sum(self.expand(n)) if all(self[k] != None for k in range(n+1)) else None})
            sum_ann = A(N - (n - min_degree) )*sum_ann
            if n >= min_degree:
                int_val_sum.update({(n-min_degree)+ord : sum(self.expand((n-min_degree)+ord)) if all(self[k] != None for k in range((n-min_degree)+ord+1)) else None})

        return UnivariateDFiniteSequence(self.parent(), sum_ann, int_val_sum)
    
#evaluation
    
    def expand(self, n):
        r"""
        Return all the terms of ``self`` between 0 and ``n``
        
        INPUT:
        
        - ``n`` -- an integer; if ``self`` is defined over the domain ZZ then ``n`` can also be negative
        
        OUTPUT:
        
        A list starting with the 0-th term up to the n-th term of ``self``.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: a = UnivariateDFiniteSequence(D, "Sn^2 - Sn - 1", [0,1]) #the Fibonacci numbers
            sage: a.expand(10)
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
            sage: a.expand(-10)
            [0, 1, -1, 2, -3, 5, -8, 13, -21, 34, -55]

        """
        ord = self.ann().order()
        start = 0
        
        if n >= 0:
            n = n+1
            #check if self is coming from a d-finite function that contains added zeros:
            if self.parent()._backward_calculation is False and min(self.initial_conditions()) < 0:
                start = -min(self.initial_conditions())
                n = n + start
            
            #1st case: n is smaller than the order - so all needed terms are already given
            if n < ord:
                return self.initial_values()[start:n]
        
        
            #2nd case: n is smaller than all relevant singularities - nothing to worry about
            s = [x for x in self.initial_conditions() if ord <= x]
            if all(n < x for x in s):
                return self.ann().to_list(self.initial_values(),n, -start)[start:]

            #3rd case: there is at least one singularity in the first n terms of the sequence
            s = set(x for x in self.initial_conditions() if ord <= x < n)
            r = self.initial_values()
            while s:
                m = min(s)
                if len(r) == m:
                    r.append(self._initial_values[m])
                else:
                    r2 = self.ann().to_list( r[len(r)-ord:], m-len(r)+ord, -start+len(r)-ord,True)
                    r = r + r2[ord:] + [self._initial_values[m]]
                s.remove(m)
            
            r2 = self.ann().to_list( r[len(r)-ord:], n-len(r)+ord, -start+len(r)-ord,True)
            r = r + r2[ord:]

            return r[start:]
      
        if n < 0:
            if self.parent()._backward_calculation is False:
                raise TypeError("Backward Calculation is not possible - the D-finite function ring is not suitable")
            
            if ord != 0:
                ord = self.ann().order()-1
            N = self.parent().base_ring().gen()
                
            A = self.ann().annihilator_of_composition(ord-N)
                
            int_val = {ord-i:self[i] for i in self.initial_conditions() if i <= ord}
            if int_val:
                b = UnivariateDFiniteSequence(self.parent().change_domain(NN), A, int_val)
                return b.expand(-(n+1)+ord+1)[ord:]
            else:
                return (-n+1)*[0]


    def __getitem__(self,n):
        r"""
        Return the n-th term of ``self``.
        
        INPUT:
        
        - ``n`` -- an integer; if ``self`` is defined over the domain ZZ then ``n`` can also be negative

        OUTPUT:
        
        The n-th sequence term of ``self`` (starting with the 0-th, i.e. to get the first term one has to call ``self[0]``)
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: D = DFiniteFunctionRing(A,ZZ)
            sage: a = UnivariateDFiniteSequence(D, "Sn^2 - Sn - 1", [0,1]) #the Fibonacci numbers
            sage: a[42]
            267914296
            sage: a[-100]
            -354224848179261915075
            
        """
        try:
            return self.initial_conditions()[n]
        except:
            pass
        
        ord = self.ann().order()
        if ord == 0:
            return 0
        
        #special case: n is negative
        if n < 0:
            return self.expand(n)[-n]
    
        #normal case: n >= 0
        if self.parent()._backward_calculation is False and min(self.initial_conditions()) < 0:
            start = min(self.initial_conditions())
        else:
            start = 0
        
        #handling None entries
        values = [self.initial_conditions()[i] for i in self.initial_conditions() if 0 <= i < n]
        if not all(x is not None for x in values):
            index = max([i for i in self.initial_conditions() if self.initial_conditions()[i] is None and 0 <= i < n])
            start += index+1
            int_val = [ self.initial_conditions()[i] for i in range(index+1,index+ord+1) ]
            roots = [ x - ord for x in self.singularities() if start <= x-ord <= n ]
        else:
            roots = [x - ord for x in self.singularities() if 0 <= x-ord <= n]
            int_val = self.initial_values()
        
        #handling singularities
        while len(roots) > 0:
            root = min(roots)
            Q,M = self.ann().forward_matrix_bsplit(ZZ(root-start),ZZ(start))
            v = Matrix([int_val]).transpose()/M
            result = Q * v
            if n < root + ord:
                d = n - (root+ord)
                return result[d][0]
            else:
                int_val = [result[i][0] for i in range(1,result.nrows())] + [self.initial_conditions()[root+ord]]
                start = root+1
                roots.remove(root)

        Q,M = self.ann().forward_matrix_bsplit(ZZ(n-start),ZZ(start))
        v = Matrix([int_val]).transpose()/M
        result = Q * v
        return result[0][0]

###############################################################################################################
class UnivariateDFiniteFunction(DFiniteFunction):
    r"""
    D-finite function in a single differentiable variable.
    """
    
#constructor
    
    def __init__(self, parent, ann, initial_val, is_gen=False, construct=False, cache=True):
        r"""
        Constructor for a D-finite function in a single differentiable variable.
        
         INPUT:
        
        - ``parent`` -- a DFiniteFunctionRing defined over an OreAlgebra with the differential operator
        
        - ``ann`` -- an annihilating operator, i.e. an element from the OreAlgebra over which the DFiniteFunctionRing is defined,
           that defines a differential equation for the function ``self`` should represent.
           
        - ``initial_val`` -- either a dictionary (or a list if no singularities occur) which contains the first r Taylor coefficients
          of ``self``, where r is the order of ``ann``, or a UnivariateDFiniteSequence which represents the Taylor sequence of ``self``
          
        OUTPUT:
        
        An object consisting of ``ann`` and a UnivariateDFiniteSequence that represents the D-finite function which is annihilated by ``ann``
        and has the Taylor sequence which is described by the UnivariateDFiniteSequence.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: x = A.base_ring().gen()
            sage: Dx = A.gen()
            sage: D1 = DFiniteFunctionRing(A)
            sage: UnivariateDFiniteFunction(D1,(3*x^2 + 4*x - 5)*Dx - 6*x - 4, {0:-5})
            Univariate D-finite function defined by the annihilating operator (3*x^2 + 4*x - 5)*Dx - 6*x - 4 and the coefficient sequence defined by (-5*n - 10)*Sn^2 + 4*n*Sn + 3*n - 6 and {-1: 0, 0: -5}
            sage: B = OreAlgebra(QQ['n'],'Sn')
            sage: D2 = DFiniteFunctionRing(B)
            sage: coeffs = D2([1,-1,1,-1,1,-1])  #a UnivariateDFiniteSequence
            sage: UnivariateDFiniteFunction(D1,(x + 1)*Dx + 1, coeffs )
            Univariate D-finite function defined by the annihilating operator (x + 1)*Dx + 1 and the coefficient sequence defined by Sn + 1 and {0: 1}
        
        """
        if not parent.ore_algebra().is_D():
            raise TypeError("Not the Differential Operator")
        super(UnivariateDFiniteFunction, self).__init__(parent, ann, initial_val, is_gen, construct, cache)
    
#action
    
    def __call__(self, r):
        r"""
        Lets ``self`` act on `r` and returns the result.
        `r` may be either a constant, then this method tries to evaluate ``self``at `r`. This evaluation might fail if there
        is a singularity of the annihilating operator of ``self`` between 0 and `r`. To then compute an evaluation use 
        ``evaluate`` and see the documentation there.
        `r` can also be a (suitable) expression, then the composition ``self(r)`` is computed. A suitable expression means 
        that `r` has to be a rational function (either in explicit form or in form of a UnivariateDFiniteFunction) whose first
        Taylor coefficient is 0.
        
        INPUT:
        
        - `r` -- either any data type that can be transformed into a float, or any data type that can be converted into a rational function
        
        OUTPUT:
        
        Either ``self`` evaluated at `r` (if possible) or the composition ``self(r)``
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: x = A.base_ring().gen()
            sage: D = DFiniteFunctionRing(A)
            sage: sin = D(sin(x))
            sage: f = 1/(x+1) - 1       #explicit rational function
            sage: g = D(1/(x+1) - 1)     #implicit form as a UnivariateDFiniteFunction
            sage: sin(f)
            Univariate D-finite function defined by the annihilating operator (-x^4 - 4*x^3 - 6*x^2 - 4*x - 1)*Dx^2 + (-2*x^3 - 6*x^2 - 6*x - 2)*Dx - 1 and the coefficient sequence defined by (-n^7 - 12*n^6 - 52*n^5 - 90*n^4 - 19*n^3 + 102*n^2 + 72*n)*Sn^4 + (-4*n^7 - 42*n^6 - 160*n^5 - 240*n^4 - 16*n^3 + 282*n^2 + 180*n)*Sn^3 + (-6*n^7 - 54*n^6 - 175*n^5 - 215*n^4 + 31*n^3 + 269*n^2 + 150*n)*Sn^2 + (-4*n^7 - 30*n^6 - 76*n^5 - 60*n^4 + 44*n^3 + 90*n^2 + 36*n)*Sn - n^7 - 6*n^6 - 10*n^5 + 11*n^3 + 6*n^2 and {0: 0, 1: -1, 2: 1, 3: -5/6, 4: 1/2, 5: -1/120}
            sage: sin(g)
            Univariate D-finite function defined by the annihilating operator (-x^4 - 4*x^3 - 6*x^2 - 4*x - 1)*Dx^2 + (-2*x^3 - 6*x^2 - 6*x - 2)*Dx - 1 and the coefficient sequence defined by (-n^7 - 12*n^6 - 52*n^5 - 90*n^4 - 19*n^3 + 102*n^2 + 72*n)*Sn^4 + (-4*n^7 - 42*n^6 - 160*n^5 - 240*n^4 - 16*n^3 + 282*n^2 + 180*n)*Sn^3 + (-6*n^7 - 54*n^6 - 175*n^5 - 215*n^4 + 31*n^3 + 269*n^2 + 150*n)*Sn^2 + (-4*n^7 - 30*n^6 - 76*n^5 - 60*n^4 + 44*n^3 + 90*n^2 + 36*n)*Sn - n^7 - 6*n^6 - 10*n^5 + 11*n^3 + 6*n^2 and {0: 0, 1: -1, 2: 1, 3: -5/6, 4: 1/2, 5: -1/120}
            sage: sin(pi)
            [+/- ...e-53]

        """
        if type(r) == list:
            return self.evaluate(r,0)
        
        if r in CC:
            return self.evaluate(r,0)
        
        else:
            if not isinstance(r, UnivariateDFiniteFunction):
                r = self.parent()(r)
            
            A = self.parent().ore_algebra()
            R = A.base_ring()
            x = R.gen()
            if r[0] != 0:
                    raise ValueError("constant term has to be zero")
        
            #getting the operator
            ann = self.ann().annihilator_of_composition(r.to_rational())
            S = OreAlgebra(R.change_var('n'),'Sn')
            s_ann = ann.to_S(S)
            ord = s_ann.order()
            
            #initial values and singularities of the new operator
            singularities_positive = s_ann.singularities()
                
            initial_val = set(range(ord)).union(singularities_positive)
            N = max(initial_val) + ord
            
            #computing the new coefficients
            B = sum( r[k]*x**k for k in range(1,N+2) )
            poly = sum( self[n]*B**n for n in range(N+1))
            
            int_val = {n:poly.derivative(n)(x=0)/factorial(n) for n in initial_val}

            #critical points for forward calculation
            critical_points_positive = self.critical_points(ord)
            for n in singularities_positive:
                critical_points_positive.update(range(n+1,n+ord+1))
        
            for n in critical_points_positive:
                int_val.update({ n : poly.derivative(n)(x=0)/factorial(n) })
                s_ann = S(s_ann.base_ring().gen() - (n - ord) )*s_ann
            
            seq = UnivariateDFiniteSequence(DFiniteFunctionRing(S,NN),s_ann,int_val)
        
            return UnivariateDFiniteFunction(self.parent(), ann, seq)

        
    def _test_conversion_(self):
        r"""
        Test whether a conversion of ``self`` into an int/float/long/... is possible;
        i.e. whether the function is constant or not.
        
        OUTPUT:
        
        If ``self`` is constant, i.e. all but the 0-th coefficient of ``self`` are 0, then the 0-th coefficient
        is returned (as an element of QQ). Otherwise ``None`` is returned.
        
        EXAMPLES::
            
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: D = DFiniteFunctionRing(A)
            sage: x = A.base_ring().gen()
            sage: a = D(3.4)
            sage: b = D(x)
            sage: a._test_conversion_()
            17/5
            sage: b._test_conversion_() # returns None            
            
        """
        ini = self.initial_conditions()
        
        if all(x == 0 for x in ini.initial_conditions() if x != 0):
            Dx = self.parent().ore_algebra().gen()
            if self.ann().quo_rem(Dx)[1].is_zero():
                return ini[0]
        return None
        
    def to_polynomial(self):
        r"""
        Try to convert ``self`` into a polynomial.
        
        OUTPUT:
        
        Either a polynomial f(x) from the base ring of the OreAlgebra of the annihilating operator of ``self`` such that
        f(x) is the explicit form of ``self`` (if ``self`` represents a polynomial) or an error message if no such polynomial exists.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: D = DFiniteFunctionRing(A)
            sage: x = A.base_ring().gen()
            sage: a = D(3*x^3 + 5*x - 10)
            sage: a.to_polynomial()
            3*x^3 + 5*x - 10
            sage: l = D(legendre_P(5,x))
            sage: l
            Univariate D-finite function defined by the annihilating operator (63/8*x^5 - 35/4*x^3 + 15/8*x)*Dx - 315/8*x^4 + 105/4*x^2 - 15/8 and the coefficient sequence defined by (15/8*n + 45/8)*Sn^4 + (-35/4*n + 35/4)*Sn^2 + 63/8*n - 315/8 and {0: 0, 1: 15/8, 2: 0, 3: -35/4}
            sage: l.to_polynomial()
            63/8*x^5 - 35/4*x^3 + 15/8*x

        """
        R = self.parent().base_ring()
        x = R.gen()
        
        if self.__is_zero__():
            return R.zero()
        
        #computing a base of the solution space
        base = self.ann().polynomial_solutions()
        if len(base) == 0:
            raise TypeError("the D-finite function is not a polynomial")
        
        #generating an equation system
        vars = list(var('x_%d' % i) for i in range(len(base)))
        c = [0]*len(base)
        for i in range(len(base)):
            base[i] = base[i][0]
            c[i] = base[i]*vars[i]
        coeffs = sum(c).coefficients(x,False)
        int_val = self.expand(len(coeffs)-1)
        eqs = list(coeffs[k] == int_val[k] for k in range(len(coeffs)))
        
        #solving the system and putting results together
        result = solve(eqs,vars)
        if len(result) == 0:
            raise TypeError("the D-finite function is not a polynomial")
        if type(result[0]) == list:
            result = result[0]
        coeffs_result = [0]*len(result)
        for i in range(len(result)):
            coeffs_result[i] = result[i].rhs()
        poly = sum(list(a*b for a,b in zip(coeffs_result,base)))
        
        if all(poly.derivative(k)(x=0)/factorial(k) == self[k] for k in self.initial_conditions().initial_conditions() if (self[k] != None and k>=0)):
            return R(poly)
        else:
            raise TypeError("the D-finite function is not a polynomial")

    def to_rational(self):
        r"""
        Try to convert ``self`` into a rational function.
        
        OUTPUT:
        
        Either a rational function r(x) from the fraction field of the base ring of the OreAlgebra of the annihilating
        operator of ``self`` such that r(x) is the explicit form of ``self`` (if ``self`` represents a rational function) or
        an error message if no such function exists.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: D = DFiniteFunctionRing(A)
            sage: x = A.base_ring().gen()
            sage: a = D((x^2+3*x-4)/(x^5+4*x^3+10))
            sage: a.to_rational()
            (x^2 + 3*x - 4)/(x^5 + 4*x^3 + 10)
            sage: l = D(legendre_P(5,x)/legendre_P(3,x))
            sage: l
            Univariate D-finite function defined by the annihilating operator (63/20*x^6 - 539/100*x^4 + 57/20*x^2 - 9/20)*Dx - 63/10*x^5 + 189/25*x^3 - 27/10*x and the coefficient sequence defined by (-9/20*n - 27/10)*Sn^6 + (57/20*n + 87/10)*Sn^4 + (-539/100*n - 161/50)*Sn^2 + 63/20*n - 63/10 and {0: -5/4, 1: 0, 2: 15/4, 3: 0, 4: 1, 5: 0}
            sage: l.to_rational()
            (63/20*x^4 - 7/2*x^2 + 3/4)/(x^2 - 3/5)
            
        """
        R = self.parent().base_ring()
        x = R.gen()
        
        if self.__is_zero__():
            return R.fraction_field().zero()
        
        #computing a base of the solution space
        base = self.ann().rational_solutions()
        if len(base) == 0:
            raise TypeError("the D-finite function is not a rational function")
        
        #generating an equation system
        vars = list(var('a_%d' % i) for i in range(len(base)))
        c = [0]*len(base)
        for i in range(len(base)):
            base[i] = base[i][0]
            c[i] = base[i]*vars[i]
        
        rat = sum(c)
        coeffs_num = rat.numerator().coefficients(x,False)
        coeffs_denom = rat.denominator().coefficients(x,False)
        
        eqs = []
        for k in range(len(coeffs_num)):
            eqs.append( coeffs_num[k] == sum(coeffs_denom[i]*self[k-i] for i in range(len(coeffs_denom))) )
        
        #solving the system and putting results together
        result = solve(eqs,vars)
        if len(result) == 0:
            raise TypeError("the D-finite function is not a rational function")
        if type(result[0]) == list:
            result = result[0]
        coeffs_result = list( result[i].rhs() for i in range(len(result)) )
        result = sum(list(a*b for a,b in zip(coeffs_result,base)))

        if all(result.derivative(k)(x=0)/factorial(k) == self[k] for k in self.initial_conditions().initial_conditions() if (self[k] != None and k >= 0)):
            return R.fraction_field()(result)
        else:
            raise TypeError("the D-finite function is not a rational function")

    def __add_without_compress__(self,right):
        r"""
        Adds the D-finite functions ``self`` and ``right`` without automatically trying
        to compress the result. This method is called whenever equality testing is done
        because there compressing the result would be unnecessary work.
        """
        sum_ann = self.ann().lclm(right.ann())
        
        lseq = self.initial_conditions()
        rseq = right.initial_conditions()

        seq = lseq.__add_without_compress__(rseq)
    
        return UnivariateDFiniteFunction(self.parent(), sum_ann, seq)


#evaluation

    def dict(self):
        raise NotImplementedError
    
    def list(self):
        raise NotImplementedError
    
    def expand(self, n, deriv = False):
        r"""
        Return a list of the first `n+1` coefficients of ``self`` if ``deriv``is False.
        If ``deriv`` is True the first `n+1` derivations of self at x=0 are returned.
        
        INPUT:
        
        - `n` -- a non-negative integer
        
        - ``deriv`` (default ``False``) -- boolean value. Determines whether the coefficients (default) or derivations of ``self`` are returned
    
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: D = DFiniteFunctionRing(A)
            sage: x = A.base_ring().gen()
            sage: e = D(exp(x))
            sage: e.expand(10)
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880, 1/3628800]
            sage: e.expand(10,True)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        """
        result = self._initial_values.expand(n)
        if deriv is True:
            result = [result[i]* factorial(i) for i in range(len(result))]
        
        return result
    
#arithmetic
    
    def _add_(self, right):
        r"""
        Returns the sum of ``self`` and ``right``
        ``_add_`` uses the method ``lclm`` from the OreAlgebra package to get the new annihilator.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: D = DFiniteFunctionRing(A)
            sage: x = A.base_ring().gen()
            sage: a = D(3*x^2 + 4)
            sage: e = D(exp(x))
            sage: s = a+e
            sage: a.expand(10)
            [4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0]
            sage: e.expand(10)
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880, 1/3628800]
            sage: s.expand(10)
            [5, 1, 7/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880, 1/3628800]
            
        """
        if self.__is_zero__():
            return right
        if right.__is_zero__():
            return self
        
        sum_ann = self.ann().lclm(right.ann())
        
        lseq = self.initial_conditions()
        rseq = right.initial_conditions()

        seq = lseq + rseq
    
        return UnivariateDFiniteFunction(self.parent(), sum_ann, seq).compress()
        
    
    def _neg_(self):
        r"""
        Return the negative of ``self``
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: D = DFiniteFunctionRing(A)
            sage: x = A.base_ring().gen()
            sage: a = D(1/(x-1))
            sage: a.expand(10)
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            sage: -a
            Univariate D-finite function defined by the annihilating operator (x - 1)*Dx + 1 and the coefficient sequence defined by (-n - 1)*Sn + n + 1 and {0: 1}
            sage: (-a).expand(10)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        """
        return UnivariateDFiniteFunction(self.parent(), self.ann(),  -self._initial_values)
    
    
    def _mul_(self, right):
        r"""
        Return the product of ``self`` and ``right``
        ``_mul_`` uses the method ``symmetric_product`` from the OreAlgebra package to get the new annihilator.
        Here we do not use the method ``cauchy_product`` from the class UnivariateDFiniteSequence, even though it would
        lead to the same (correct) result. But to use that method one would have to use (more) transformations of the annihilating operators
        between the differential and the shift OreAlgebra, which would increase their orders (even more) and would eventually lead to an increased
        computation time.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: D = DFiniteFunctionRing(A)
            sage: x = A.base_ring().gen()
            sage: a = D(1/(x+1))
            sage: b = D(x+1)
            sage: p = a*b
            sage: p
            Univariate D-finite function defined by the annihilating operator Dx and the coefficient sequence defined by n^2 and {0: 1}
            sage: p.to_polynomial()
            1
            
        """
        if self.__is_zero__() or right.__is_zero__():
            return self.parent().zero()
        
        if self == self.parent().one():
            return right
        if right == self.parent().one():
            return self
        
        lseq = self.initial_conditions()
        rseq = right.initial_conditions()
        
        #getting the new operators
        prod_ann = self.ann().symmetric_product(right.ann())
        A = OreAlgebra(self.parent().base_ring().change_var('n'),'Sn')
        N = A.base_ring().gen()
        s_ann = prod_ann.to_S(A)
        ord = s_ann.order()

        #initial values and singularities of the sequence operator
        singularities_positive = s_ann.singularities()
    
        initial_val = set(range(ord)).union(singularities_positive)
        int_val_prod = {}
        for n in initial_val:
            a = lseq.expand(n)
            b = rseq.expand(n)
            b.reverse()
            cauchy = sum([x*y for x,y in zip(a,b)])
            int_val_prod.update({n:cauchy})
        
        #critical points for forward calculation
        critical_points_positive = lseq.critical_points(ord).union( rseq.critical_points(ord) )
        for n in singularities_positive:
            critical_points_positive.update(range(n+1,n+ord+1))
        
        for n in critical_points_positive:
            a = lseq.expand(n)
            b = rseq.expand(n)
            b.reverse()
            cauchy = sum([x*y for x,y in zip(a,b)])
            int_val_prod.update({n:cauchy})
            s_ann = A(N - (n - ord) )*s_ann
        
        seq = UnivariateDFiniteSequence(DFiniteFunctionRing(A,NN),s_ann,int_val_prod)
    
        prod = UnivariateDFiniteFunction(self.parent(), prod_ann, seq)
        return prod
        
    def hadamard_product(self,right):
        r"""
        Return the D-finite function corresponding to the Hadamard product of ``self`` and ``right``.
        The Hadamard product of two formal power series a(x) = \sum_{n=0}^\infty a_n x^n and b(x) = \sum_{n=0}^\infty b_n x^n
        is defined as a(x) \odot b(x) := \sum_{n=0}^\infty a_nb_n x^n
        """
        seq = self.initial_conditions() * right.initial_conditions()
        return seq.generating_function()
        
    def __invert__(self):
        r"""
        """
        raise NotImplementedError
    
    def integral(self):
        r"""
        Return the D-finite function corresponding to the integral of ``self``.
        By integral the formal integral of a power series is meant, i.e. 
        \int a(x) = \int_0^x \sum_{n=0}^\infty a_n x^n = \sum_{n=0}^\infty \frac{a_n}{n+1} x^{n+1}
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ[x],'Dx')
            sage: D = DFiniteFunctionRing(A)
            sage: c = D(cos(x))
            sage: c.integral()
            Univariate D-finite function defined by the annihilating operator Dx^3 + Dx and the coefficient sequence defined by (n^7 + n^6 - 5*n^5 - 5*n^4 + 4*n^3 + 4*n^2)*Sn^2 + n^5 - 2*n^4 - n^3 + 2*n^2 and {0: 0, 1: 1, 2: 0, 3: -1/6, 4: 0}
            sage: _ == D(sin(x))
            True
        
        """
        #getting the new operators
        ann = self.ann().annihilator_of_integral()
        A = OreAlgebra(self.parent().base_ring().change_var('n'),'Sn')
        N = A.base_ring().gen()
        s_ann = ann.to_S(A)
        ord = s_ann.order()

        #initial values and singularities of the sequence operator
        singularities_positive = s_ann.singularities()
    
        initial_val = set(range(ord)).union(singularities_positive)
        int_val = {n:self[n-1]/QQ(n) for n in initial_val if n > 0}
        int_val.update({0:0})
        
        #critical points for forward calculation
        critical_points_positive = self.initial_conditions().critical_points(ord)
        for n in singularities_positive:
            critical_points_positive.update(range(n+1,n+ord+1))
        critical_points_positive.difference_update({0})
        
        for n in critical_points_positive:
            int_val.update({n:self[n-1]/n})
            s_ann = A(N - (n - ord) )*s_ann
        
        seq = UnivariateDFiniteSequence(DFiniteFunctionRing(A,NN),s_ann,int_val)
        integral = UnivariateDFiniteFunction(self.parent(), ann, seq)
        return integral
    
#evaluation

    def __getitem__(self, n):
        r"""
        Return the n-th coefficient of ``self`` (starting with 0).
        
        INPUT:
        
        - `n` -- an integer (`n` can also be negative)
        
        OUTPUT:
        
        If `n` is positive, then the n-th coefficient of ``self`` is returned (starting with the 0-th).
        If `n` is negative, then always 0 is returned.
        
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: D = DFiniteFunctionRing(A)
            sage: x = A.base_ring().gen()
            sage: f = D(3*x^113 + 5*x^2 + 13)
            sage: f[113]
            3
            sage: f[-12]
            0
        
        """
        if n >= 0:
            return self.initial_conditions()[n]
        else:
            return 0
    
    
    def evaluate(self, z, n = 0):
        r"""
        Tries to numerically evaluate the n-th derivative of ``self`` at  `z`
        
        INPUT:
        
        - `z` -- either a datatype that can be transformed into a float or a list of floating point numbers starting with 0 and ending
          with the value that which the derivation should be computed. The list should provide a path from 0 to the evaluation point, not 
          crossing any singularities of the annihilating operator of ``self`` (for further information see the documentation of the method
          ``numerical_solution`` of the OreAlgebra package).
          
        - `n` (default 0) -- a non-negative integer
        
        OUTPUT:
        
        The evaluation of the n-th derivative of ``self`` at `z` if `z` is a floating point number. If
        `z` is a list, then the evaluation of the n-th derivative of ``self`` at the last point of the list
        is computed.
            
        EXAMPLES::
        
            sage: from ore_algebra import *
            sage: A = OreAlgebra(QQ['x'],'Dx')
            sage: D = DFiniteFunctionRing(A)
            sage: x = A.base_ring().gen()
            sage: a = D(3*x^5 + 4*x^2 - 8*x +13)
            sage: sin = D(sin(x))
            sage: a.evaluate(0,0)
            13
            sage: a.evaluate(0,1)
            [-8.00000000000000000000000000000000000000000000000...]
            sage: sin.evaluate(pi/2,0)
            [1.000000000000000000000000000000000000000000000000...]
            sage: sin.evaluate(pi,1)
            [-1.00000000000000000000000000000000000000000000000...]
            
        """
        ini = self.initial_values()
        Dx = self.parent().ore_algebra().gen()
        
        if z == 0 and n == 0:
            return self[0]
        
        if z in CC:
            return self.ann().numerical_solution(ini,[0,z], eps=1e-50, post_transform=Dx**n)
        elif type(z) == list:
            return self.ann().numerical_solution(ini,z, eps=1e-50, post_transform=Dx**n)
        else:
            raise NotImplementedError("evaluation point has to be given in form of a single point or in form of a list")
