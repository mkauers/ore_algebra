# coding: utf-8
r"""
Database of annihilators

Database for annihilating operators, i.e. elements of certain OreAlgebras, for various discrete and differentiable
symbolic functions.

The following discrete functions are supported:
    - binomial(an+b, cn+d) where a,b,c,d are fixed rational numbers and n is the variable
    - factorial(n)
    - harmonic_number(n)

The following differentiable functions are supported:
    - Trigonometric functions: sin(x), cos(x), arcsin(x), arccos(x), arctan(x), arccsc(x), arcsec(x)
    - Hyperbolic functions: sinh(x), cosh(x), arcsinh(x), arctanh(x), arccsch(x)
    - Logarithmic functions: exp(x), log(x), dilog(x)
    - Airy functions: airy_ai(x), airy_bi(x), airy_ai_prime(x), airy_bi_prime(x)
    - Bessel functions: bessel_I(k,x), bessel_J(k,x), bessel_K(k,x), bessel_Y(k,x), spherical_bessel_J(k,x) where k is a fixed
      rational number and x is the variable
    - Error functions: erf(x), erfc(x), erfi(x)
    - Integrals: exp_integral_e(k,x), exp_integral_ei(x), sin_integral(x), cos_integral(x), sinh_integral(x), cosh_integral(x)
    - Other: sqrt(x), elliptic_ec(x), elliptic_kc(x)
    
AUTHOR:

- Clemens Hofstadler (2018-03-01)
    
"""

#############################################################################
#  Copyright (C) 2018, Clemens Hofstadler (clemens.hofstadler@liwest.at).   #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

from __future__ import absolute_import, division

import sage.functions.airy
import sage.functions.bessel
import sage.functions.error
import sage.functions.exp_integral
import sage.functions.hyperbolic
import sage.functions.log
import sage.functions.other
import sage.functions.special
import sage.functions.trig

from sage.rings.all import QQ

from operator import pow

def symbolic_database(A, f, inner = None, k = 0):
    r"""
    Tries to return an annihilating operator of a symbolic operator `f`, i.e. an element from a (suitable) OreAlgebra `A`
    that represents a differential/recurrence equation for the symoblic epxression ``f(x)``
    
    INPUT:
    
    - `A` -- an OreAlgebra from which the annihilating operator for `f` should come from
    
    - `f` - a symbolic operator. It is important that not the symbolic expression itself is handled to this method but only
        the operator (which can be accessed via the command ``.operator()``
            
    - ``inner`` (default ``None``) -- for differentiable functions an inner function can be handed over. Then not an operator
        for f(x) but for f(inner) is returned. However ``inner`` has to be a linear function, i.e. of the form u*x + v for
        u,v in QQ. For discrete functions ``inner`` does not have any impact.
        
    - `k` (default 0) -- a rational number that has to be handed over if the symolic operator `f` depends on two variables (such as binomial(k,n) or bessel_I(k,x) ) because only operators for univariate functions can be returned and 
        therefore one variable has to be assigned with a certain value `k`
    
    OUTPUT:
    
    An element from the OreAlgebra `A`, that represents a differential/recurrence equation which annihilates f(x) or if
     ``inner`` is not ``None`` an element from `A` which annihilates f(inner).
    
    EXAMPLES::
    
        #discrete case
        sage: from ore_algebra import *
        sage: from ore_algebra.dfinite_symbolic import symbolic_database
        sage: A = OreAlgebra(QQ['n'],'Sn')
        sage: n = var('n')
        sage: f = harmonic_number(n).operator()
        sage: symbolic_database(A,f)
        (n + 2)*Sn^2 + (-2*n - 3)*Sn + n + 1
        sage: g = binomial(5,n).operator()
        sage: symbolic_database(A,g,None,5)
        (n + 1)*Sn + n - 5
        
        #differential case
        sage: B = OreAlgebra(QQ['x'],'Dx')
        sage: x = var('x')
        sage: f = sin(x).operator()
        sage: symbolic_database(B,f)
        Dx^2 + 1
        sage: g = log(3*x+4).operator()
        sage: symbolic_database(B,g,3*x+4)
        (3*x + 4)*Dx^2 + 3*Dx
        sage: h = bessel_I(4,2*x+1).operator()
        sage: symbolic_database(B,h,2*x+1,4)
        (4*x^2 + 4*x + 1)*Dx^2 + (4*x + 2)*Dx - 16*x^2 - 16*x - 68
        
    """
    n = A.is_S()
    k = A(k)

    #sequences
    if n:
        Sn = A.gen()

        #factorial
        if isinstance(f,sage.functions.other.Function_factorial):
            return A(Sn - (n+1))
        #harmonic_number
        elif isinstance(f,sage.functions.log.Function_harmonic_number_generalized):
            return A( (n+2)*Sn**2 - (2*n+3)*Sn + (n+1) )
        #binomial
        elif isinstance(f,sage.functions.other.Function_binomial):
            #(k choose n) - k fixed, n variable
            if k in QQ:
                return A((n+1)*Sn - (k-n))
            #(a*n+b choose c*n+d) - a,b,c,d fixed, n variable
            else:
                k = k.constant_coefficient()
                f1 = prod(inner+i for i in range(1,inner[1]+1))
                f2 = prod(k+i for i in range(1,k[1]+1))
                f3 = prod(inner - k + i for i in range(1,inner[1]-k[1]+1))
            return A(f2*f3*Sn - f1)
        else:
            raise NotImplementedError
        
    #functions
    else:
        Dx = A.gen()

        if inner:
            x = A(inner)
            d = A(inner.derivative())
        else:
            x = A.base_ring().gen()
            d = 1 #x.derivative()
        
        #sin
        if isinstance(f,sage.functions.trig.Function_sin):
            return A(Dx**2 + d**2)
        #cos
        elif isinstance(f,sage.functions.trig.Function_cos):
            return A(Dx**2 + d**2)
        #tan
        elif isinstance(f,sage.functions.trig.Function_tan):
            raise TypeError("Tan is not D-finite")
        #arcsin
        elif isinstance(f,sage.functions.trig.Function_arcsin):
            return A((1-x**2)*Dx**2 - d*x*Dx)
        #arccos
        elif isinstance(f,sage.functions.trig.Function_arccos):
            return A((1-x**2)*Dx**2 - d*x*Dx)
        #arctan
        elif isinstance(f,sage.functions.trig.Function_arctan):
            return A((x**2+1)*Dx**2 + d*2*x*Dx)
        #sinh
        elif isinstance(f,sage.functions.hyperbolic.Function_sinh):
            return A(Dx**2 - d**2)
        #cosh
        elif isinstance(f,sage.functions.hyperbolic.Function_cosh):
            return A(Dx**2 - d**2)
        #arcsinh
        elif isinstance(f,sage.functions.hyperbolic.Function_arcsinh):
            return A((1+x**2)*Dx**2 + d*x*Dx)
        #arccosh
        elif isinstance(f,sage.functions.hyperbolic.Function_arccosh):
            raise TypeError("ArcCosh is not D-finite")
        #arctanh
        elif isinstance(f,sage.functions.hyperbolic.Function_arctanh):
            return A((1-x**2)*Dx**2 - 2*d*x*Dx)
        #exp
        elif isinstance(f,sage.functions.log.Function_exp):
            return A(Dx - d)
        #log
        elif isinstance(f,sage.functions.log.Function_log1):
            return A(x*Dx**2 + d*Dx)
        #sqrt
        elif f == pow:
            return A(-2*x*Dx + d)
        #airy_ai
        elif isinstance(f,sage.functions.airy.FunctionAiryAiSimple):
            return A(Dx**2 - d**2*x)
        #airy_ai_prime
        elif isinstance(f,sage.functions.airy.FunctionAiryAiPrime):
            return A(x*Dx**2 - d*Dx - d**2*x**2)
        #airy_bi
        elif isinstance(f,sage.functions.airy.FunctionAiryBiSimple):
            return A(Dx**2 - d**2*x)
        #airy_bi_prime
        elif isinstance(f,sage.functions.airy.FunctionAiryBiPrime):
            return A(x*Dx**2 - d*Dx - (d*x)**2)
        #arccsc
        elif isinstance(f,sage.functions.trig.Function_arccsc):
            return A(x*(1-x**2)*Dx**2 - (2*x**2-1)*d*Dx)
        #arccsch
        elif isinstance(f,sage.functions.hyperbolic.Function_arccsch):
            return A(x*(x**2+1)*Dx**2 + (2*x**2+1)*d*Dx)
        #arcsec
        elif isinstance(f,sage.functions.trig.Function_arcsec):
            return A(x*(1-x**2)*Dx**2 - (2*x**2-1)*d*Dx)
        #bessel_I
        elif isinstance(f,sage.functions.bessel.Function_Bessel_I):
            return A( x**2*Dx**2 + d*x*Dx - d**2*(x**2 + k**2) )
        #bessel_J
        elif isinstance(f,sage.functions.bessel.Function_Bessel_J):
            return A( x**2*Dx**2 + d*x*Dx + d**2*(x**2 - k**2) )
        #bessel_Y
        elif isinstance(f,sage.functions.bessel.Function_Bessel_Y):
            return A( x**2*Dx**2 + d*x*Dx + d**2*(x**2 - k**2) )
        #bessel_K
        elif isinstance(f,sage.functions.bessel.Function_Bessel_K):
            return A( x**2*Dx**2 + d*x*Dx - d**2*(x**2 + k**2) )
        #sherical_bessel_Jzz
        elif isinstance(f,sage.functions.bessel.SphericalBesselJ):
            return A( x**2*Dx**2 + 2*d*x*Dx + d**2*(x**2 - k*(k+1)) )
        #erf (error function)
        elif isinstance(f,sage.functions.error.Function_erf):
            return A( Dx**2 + 2*d*x*Dx )
        #erfc (complementary error function)
        elif isinstance(f,sage.functions.error.Function_erfc):
            return A( Dx**2 + 2*d*x*Dx )
        #erfi (imaginary error function)
        elif isinstance(f,sage.functions.error.Function_erfi):
            return A( Dx**2 - 2*d*x*Dx )
        #dilog
        elif isinstance(f,sage.functions.log.Function_dilog):
            return A( x*(1-x)*Dx**3 + d*(2-3*x)*Dx**2 - d**2*Dx )
        #exp_integral_e
        elif isinstance(f,sage.functions.exp_integral.Function_exp_integral_e):
            return A( x*Dx**2 + d*(x-k+2)*Dx + d**2*(1-k) )
        #exp_integral_ei (Ei)
        elif isinstance(f, sage.functions.exp_integral.Function_exp_integral):
            return A( x*Dx**3 + 2*d*Dx**2 - d**2*x*Dx )
        #sin_integral
        elif isinstance(f,sage.functions.exp_integral.Function_sin_integral):
            return A( x*Dx**3 + 2*d*Dx**2 + d**2*x*Dx )
        #cos_integral
        elif isinstance(f,sage.functions.exp_integral.Function_cos_integral):
            return A( x*Dx**3 + 2*d*Dx**2 + d**2*x*Dx )
        #sinh_integral
        elif isinstance(f,sage.functions.exp_integral.Function_sinh_integral):
            return A ( x*Dx**3 + 2*d*Dx**2 - d**2*x*Dx )
        #cosh_integral
        elif isinstance(f,sage.functions.exp_integral.Function_sinh_integral):
            return A( x*Dx**3 + 2*d*Dx**2 - d**2*x*Dx )
        #elliptic_ec (complete elliptic integral of second kind) - problems with computing the derivative
        elif isinstance(f,sage.functions.special.EllipticEC):
            return A( (1-x)*x*Dx**2 + d*(1-x)*Dx + d**2*QQ(1./4) )
        #elliptic_kc (complete elliptic integral of first kind) - problems with computing the derivative
        elif isinstance(f,sage.functions.special.EllipticKC):
            return A( (1-x)*x*Dx**2 + d*(1-2*x)*Dx - d**2*QQ(1./4) )
        else:
            raise NotImplementedError



