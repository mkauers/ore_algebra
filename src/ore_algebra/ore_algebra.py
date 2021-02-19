# coding: utf-8
"""
Ore algebras

The ore_algebra package provides functionality for doing computations with Ore polynomials.

Ore polynomials are operators which can be used to describe special functions or combinatorial
sequences. Typical examples are linear differential operators with polynomial coefficients.

Ore polynomials are elements of Ore algebras. Ore algebras are ring objects created by the
function ``OreAlgebra`` as described below. 

Depending on the particular parent algebra, Ore polynomials may support different functionality.
For example, for Ore polynomials representing recurrence operators, there is a method for 
computing interlacing operatos, an operation which does not make sense for differential operators.

The typical user will only need two functions defined in the package: 

 * ``OreAlgebra`` -- for creating a new Ore algebra object. 
 * ``guess`` -- for fitting an Ore polynomial to a given set of data.

Ore polynomials are created using ``OreAlgebra`` objects, and most of the functionality for
doing calculations with Ore polynomials is available in the methods attached to them. 

For examples and further information, see the docstring of ``OreAlgebra`` below, or the 
tutorial paper *Ore Polynomials in Sage* by the authors. 


AUTHOR:

 - Manuel Kauers, Maximilian Jaroschek, Fredrik Johansson (2013-06-15)

"""

#############################################################################
#  Copyright (C) 2013, 2014, 2017                                           #
#                Manuel Kauers (mkauers@gmail.com),                         #
#                Maximilian Jaroschek (mjarosch@risc.jku.at),               #
#                Fredrik Johansson (fjohanss@risc.jku.at).                  #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

from __future__ import absolute_import, division

"""
######### development mode ###########

def unload():
    for p in [k for k in sys.modules if k.startswith("ore_algebra")]: del sys.modules[p]

if True:

    # let load("ore_algebra") trigger reload of the modules in the list below
    for mod in ['nullspace', 'ore_operator', 'ore_operator_1_1', 'ore_operator_mult', 'generalized_series', 'tools', 'ideal']:
        try:
            del sys.modules[mod]
        except:
            pass

    # if sage version is 5.10.beta3 or later, use taylor shift of flint2
    import datetime
    try:
        v = version()
    except:
        v = "2013-01-01"    

    if datetime.date(int(v[-10:-6]), int(v[-5:-3]), int(v[-2:])) >= datetime.date(2013, 05, 15):
        try:
            load("shift.spyx") # cython code defining 'taylor_shift_univ_int_poly' and 'taylor_shift_univ_modp_poly'
        except:
            pass

#######################################
"""

def taylor_shift_univ_int_poly(p, i):
    ## assuming that p is an element of ZZ['x']
    return p(p.parent().gen() + i)
def taylor_shift_univ_int_ratfun(q, i):
    ## assuming that q is an element of Frac(ZZ['x'])
    num = taylor_shift_univ_int_poly(q.numerator(), i)
    den = taylor_shift_univ_int_poly(q.denominator(), i)
    return q.parent()(num, den, coerce=False, reduce=False)
def taylor_shift_univ_modp_poly(p, i):
    ## assuming that p is an element of GF(m)['x']
    return p(p.parent().gen() + i)
def taylor_shift_univ_modp_ratfun(q, i):
    ## assuming that q is an element of Frac(GF(m)['x'])
    num = taylor_shift_univ_modp_poly(q.numerator(), i)
    den = taylor_shift_univ_modp_poly(q.denominator(), i)
    return q.parent()(num, den, coerce=False, reduce=False)

from sage.misc.cachefunc import cached_method
from sage.structure.element import RingElement
from sage.structure.unique_representation import UniqueRepresentation
from sage.rings.ring import Algebra
from sage.rings.ring import Ring 
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing
from sage.rings.number_field.number_field import is_NumberField
from sage.rings.fraction_field import is_FractionField
from sage.rings.infinity import infinity
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.finite_rings.all import GF

from . import nullspace

def is_OreAlgebra(A):
    r"""
    Checks whether `A` is an Ore algebra object.     
    """
    return isinstance(A, OreAlgebra_generic)

def is_suitable_base_ring(R):
    r"""
    Checks whether `R` is suitable as base ring of an Ore algebra.
    This is the case if and only if: (a) `R` is one of `ZZ`, `QQ`,
    `GF(p)` for a prime `p`, or a finite algebraic extension of `QQ`,
    (b) `R` is a fraction field of a suitable base ring,
    (c) `R` is a (univariate or multivariate) polynomial ring over
    a suitable base ring.

    This function returns ``True`` or ``False``.

    EXAMPLES::

       sage: from ore_algebra.ore_algebra import is_suitable_base_ring
       sage: R = GF(1091)['x, y'].fraction_field()['z']
       sage: is_suitable_base_ring(R)
       True
       sage: is_suitable_base_ring(GF(9, 'a'))
       False
    
    """
    p = R.characteristic()
    if (p == 0 and (R is ZZ or R is QQ or is_NumberField(R))) or (p > 0 and R is GF(p)):
        return True
    elif is_FractionField(R):
        return is_suitable_base_ring(R.ring())
    elif is_PolynomialRing(R) or is_MPolynomialRing(R):
        return is_suitable_base_ring(R.base_ring())
    else:
        return False

class Sigma_class(object):
    r"""
    A ring endomorphism for suitable rings. 

    A sigma object is created by a ring `R` on which it operates, and some piece of defining the action.
    The action is defined through a dictionary which has generators of `R` on its left hand side and
    elements of `R` on its right hand side. Generators of `R` which are not contained in the dictionary
    are mapped to themselves.

    Instead of a dictionary, the constructor also accepts arbitrary callable objects. In this case, a
    dictionary is created based on the values this callable object produces when applied to the generators
    of `R`.     

    It is assumed without test that the ring `R` is \"suitable\".

    EXAMPLES::

       sage: from ore_algebra.ore_algebra import Sigma_class
       sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
       sage: sigma = Sigma_class(R, {x1:2*x1, x2:1-x2, x3:x3+1})
       sage: sigma(x1+x2+x3)
       2*x1 - x2 + x3 + 2
       sage: sigma = Sigma_class(R.fraction_field(), {x1:2*x1, x2:1-x2, x3:x3+1})
       sage: sigma(x1+x2+x3)
       2*x1 - x2 + x3 + 2

    Repeated application of a sigma object to some ring element can be specified by an optional second
    argument. There are also functions for computing sigma factorials, for constructing the compositional
    inverse of a (sufficiently simple) sigma object, and for converting a sigma object into a dictionary

    EXAMPLES::

        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: sigma = Sigma_class(R, {x1:2*x1, x2:1-x2, x3:x3+1})
        sage: sigma(x1+x2+x3, 5)
        32*x1 - x2 + x3 + 6
        sage: sigma.factorial(x1+x2+x3, 4).factor()
        (x1 + x2 + x3) * (2*x1 - x2 + x3 + 2) * (4*x1 + x2 + x3 + 2) * (8*x1 - x2 + x3 + 4)
        sage: sigma_inv = sigma.inverse()
        sage: sigma_inv(x1+x2+x3)
        1/2*x1 - x2 + x3
        sage: sigma(x1+x2+x3, -1)
        1/2*x1 - x2 + x3
        sage: sigma.inverse().inverse() == sigma
        True
        sage: sorted(sigma.dict().items(), key=str)
        [('x1', 2*x1), ('x2', -x2 + 1), ('x3', x3 + 1)]
    """

    def __init__(self, R, d):

        Rgens = R.gens(); my_dict = {}; is_id = True
        if type(d) != dict:
            for x in Rgens:
                dx = R(d(x))
                my_dict[str(x)] = dx
                if dx != x:
                    is_id = False
        else:         
            for x in d:
                if not R(x) in Rgens:
                    raise ValueError(str(x) + " is not a generator of " + str(R))
                if R(x) != d[x]:
                    my_dict[str(x)] = R(d[x])
                    is_id = False
            for x in Rgens:
                if not str(x) in my_dict:
                    my_dict[str(x)] = R(x)
        self.__R = R
        self.__dict = my_dict
        self.__is_identity = is_id
        self.__powers = {1: my_dict}

    def __call__(self, p, exp=1):

        if self.__is_identity:
            return p
        elif exp == 1:
            return self.__R(p)(**self.__dict)
        elif exp == 0:
            return p
        elif exp > 1:

            pows = self.__powers

            def merge(d1, d2):
                d = {}
                for x in d1:
                    d[x] = d1[x](**d2)
                return d
            
            def pow_dict(n):
                if n in pows:
                    return pows[n].copy()
                elif n % 2 == 0:
                    d = pow_dict(n/2)
                    pows[n] = d = merge(d, d)
                else:
                    d = pow_dict((n - 1)/2)
                    pows[n] = d = merge(merge(d, d), self.__dict)
                return d

            return self.__R(p)(**pow_dict(exp))
                
        elif exp < 0:
            return self.inverse()(p, -exp)
        else:
            raise ValueError("illegal sigma power " + str(exp))

    def set_call(self, fun):
        self.__call__ = fun 

    def is_identity(self):
        return self.__is_identity

    def dict(self):
        r"""
        Returns a dictionary representing ``self``
        """
        return self.__dict.copy()

    @cached_method
    def __hash__(self):
        gens = self.__R.gens()
        h = hash((self.__R, tuple(self(x) for x in gens)))
        return h

    def __eq__(self, other):

        try:
            for x in self.__R.gens():
                if self(x) != other(x):
                    return False
            for x in other.__R.gens():
                if self(x) != other(x):
                    return False
            return True
        except:
            return False

    def __neq__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "Endomorphism defined through " + str(self.dict())

    def ring(self):
        r"""
        Returns the ring for which this sigma object is defined
        """
        return self.__R

    def factorial(self, p, n):
        r"""
        Returns `p\sigma(p)...\sigma^{n-1}(p)` if `n` is nonnegative,
        and and `1/(\sigma(p)...\sigma^n(p)` otherwise.        
        """
        if n == 0:
            return self.__R.one()
        elif n == 1:
            return p
        elif n > 1:
            q = p; out = p
            for i in range(n - 1):
                q = self(q)
                out = out*q
            return out
        elif n < 0:
            s = self.inverse()
            q = ~s(p); out = q
            n = -n 
            for i in range(n - 1):
                q = s(q)
                out = out*q
            return out                
        else:
            raise ValueError("illegal argument to Sigma.factorial: " + str(n))

    def inverse(self):
        r"""
        Returns a sigma object which represents the compositional inverse of ``self``.

        The inverse can be constructed if `\sigma` is such that it maps every generator `x` of
        the base ring to a linear combination `a*x+b` where `a` and `b` belong to the base
        ring of the parent of `x`.

        If the method fails in constructing the inverse, it raises a ``ValueError``.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = QQ['x']
            sage: A.<Sx> = OreAlgebra(R.fraction_field(), "Sx")
            sage: sigma = A.sigma()
            sage: sigma_inverse = sigma.inverse()
            sage: sigma(x)
            x + 1
            sage: sigma_inverse(x)
            x - 1

        """
        # possible generalization in case of rings with more generators: each generator is
        # mapped to a linear combination of the other generators with coefficients in the
        # base ring.

        try:
            return self.__inverse
        except AttributeError:
            pass

        R = self.__R
        if is_FractionField(R):
            R = R.ring()
        C_one = R.base_ring().one()
        sigma = self
        sigma_inv_dict = {}
        for exp in range(R.ngens()):
            exp = tuple( (1 if i==exp else 0) for i in range(R.ngens()))
            if len(exp) == 1:
                x = R.gen()
            else:
                x = R({exp:C_one});
            sx = sigma(x)
            if sx == x:
                continue
            try:
                sx = R(sx) # may raise exception
                b = sx.constant_coefficient()
                if len(exp) == 1:
                    a = sx[1] # univariate poly ring
                else:
                    a = sx[tuple(exp)] # multivariate poly ring
                if sx != a*x + b:
                    raise ValueError # may raise exception
                sigma_inv_dict[x] = self.__R((x - b)/a) # may raise exception
            except:
                raise ValueError("unable to construct inverse of sigma")

        sigma_inv = Sigma_class(self.__R, sigma_inv_dict)
        self.__inverse = sigma_inv
        sigma_inv.__inverse = self
        return sigma_inv

class Delta_class(object):
    r"""
    A skew-derivation for suitable rings. 

    A delta object is created by a ring `R` on which it operates, some piece of information defining the action,
    and an associated Sigma object. 
    The action is defined through a dictionary which has generators of `R` on its left hand side and
    elements of `R` on its right hand side. Generators of `R` which are not contained in the dictionary
    are mapped to zero.

    Instead of a dictionary, the constructor also accepts arbitrary callable objects. In this case, a
    dictionary is created based on the values this callable object produces when applied to the generators
    of `R`.     

    It is assumed without test that the ring `R` is \"suitable\".

    EXAMPLES::

       sage: from ore_algebra.ore_algebra import *
       sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
       sage: sigma = Sigma_class(R, {x1:2*x1, x2:1-x2, x3:x3+1})
       sage: delta = Delta_class(R, {x1:1, x3:x3}, sigma)
       sage: delta(x1+x2+x3)
       x3 + 1
       sage: delta(x1*x2*x3)
       -2*x1*x2*x3 + 2*x1*x3 + x2*x3
       sage: sorted(delta.dict().items(), key=str)
       [('x1', 1), ('x3', x3)]

    """

    def __init__(self, R, d, s):

        if R is not s.ring():
            raise ValueError("delta constructor received incompatible sigma")

        Rgens = R.gens(); is_zero = True; zero = R.zero(); my_dict = {}

        for x in Rgens:
            my_dict[str(x), 0] = zero
            my_dict[str(x), 1] = zero

        if not isinstance(d, dict):
            for x in Rgens:
                dx = R(d(x))
                is_zero = is_zero and dx.is_zero()
                my_dict[str(x), 1] = R(dx)
        else:
            for x, dx in d.items():
                if not R(x) in Rgens:
                    raise ValueError(str(x) + " is not a generator of " + str(R))
                is_zero = is_zero and dx.is_zero()
                my_dict[str(x), 1] = R(dx)

        self.__is_zero = is_zero
        self.__R = R
        self.__dict = my_dict
        self.__sigma = s

    def __call__(self, p):

        if self.__is_zero:
            return self.__R.zero()

        R = self.__R; delta = self; sigma = self.__sigma; my_dict = self.__dict
        if p in R.base_ring():
            return R.zero()

        R0 = p.parent();
        if is_FractionField(R0):
            a = p.numerator(); b = p.denominator()
            # try/except cascade needed to work around sage's difficulties with rational function towers
            try:
                return (R0(delta(a)) - R0(sigma(a)/sigma(b))*R0(delta(b)))/R0(b)
            except TypeError:
                pass
            try:
                return R0(delta(a))/R0(b) - R0(sigma(a)/sigma(b))*(R0(delta(b))/R0(b))
            except TypeError:
                pass
            try:
                return R0(delta(a))/R0(b) - R0(delta(b)*sigma(a))/R0(b*sigma(b))
            except TypeError:
                pass
            return R0(delta(a)*sigma(b) - sigma(a)*delta(b))/R0(b*sigma(b))
        
        elif is_PolynomialRing(R0):
            x = R0.gen(); strx = str(x)
            if (strx, 0) not in my_dict:
                return R0.zero()
            if sigma.is_identity():
                return p.derivative()*my_dict[strx, 1]
            x = R(x)
            for i in range(2, p.degree() + 1):
                if (strx, i) not in my_dict:
                    my_dict[strx, i] = my_dict[strx, i - 1]*x + sigma(x**(i - 1))*my_dict[strx, 1]
            out = R0.zero()
            for i in range(p.degree() + 1):
                out += sigma(p[i])*my_dict[strx, i]
            return out
        elif is_MPolynomialRing(R0):
            Rgens = R0.gens()
            # special code for standard derivative
            if sigma.is_identity():
                out = R0.zero()
                for x in Rgens:
                    strx = str(x)
                    if (strx, 1) in my_dict and not my_dict[strx, 1].is_zero():
                        out += p.derivative(x)*my_dict[strx, 1]
                return out
            # fall-back code for general case
            for x in Rgens:
                strx = str(x)
                for i in range(2, p.degree(x) + 1):
                    if (strx, i) not in my_dict:
                        my_dict[strx, i] = my_dict[strx, i - 1]*x + sigma(x**(i - 1))*my_dict[strx, 1]
            out = R0.zero(); one = R0.one()
            for exp in p.exponents():
                # x1^e1 x2^e2 x3^e3
                # ==> delta(x1^e1)*x2^e2*x3^e3 + sigma(x1^e1)*delta(x2^e2)*x3^e3 + sigma(x1^e1)*sigma(x2^e2)*delta(x3^e3)
                for i in range(len(Rgens)):
                    term = one
                    for j in range(len(Rgens)):
                        if j < i:
                            term *= sigma(Rgens[j]**exp[j])
                        elif j == i:
                            term *= my_dict[str(Rgens[j]), exp[j]]
                        else:
                            term *= Rgens[j]**exp[j]
                    out += p[exp]*term
            return out
        else:
            raise TypeError("don't know how to apply delta to " + str(p))

    def set_call(self, fun):
        self.__call__ = fun

    def is_zero(self):
        return self.__is_zero

    def __repr__(self):
        return "Skew-derivation defined through " + str(self.dict()) + " for " + str(self.__sigma)

    @cached_method
    def __hash__(self):
        sigma = self.__sigma; gens = self.__R.gens()
        h = hash((self.__R,
            tuple(self(x) for x in gens),
            tuple(sigma(x) for x in gens)))
        return h

    def __eq__(self, other):

        try:
            for x in self.__R.gens():
                if self(x) != other(x):
                    return False
            for x in other.__R.gens():
                if self(x) != other(x):
                    return False
            return True
        except:
            return False

    def __neq__(self, other):
        return not self.__eq__(other)

    def ring(self):
        r"""
        Returns the ring for which this sigma object is defined
        """
        return self.__R

    def dict(self):
        r"""
        Returns a dictionary representing ``self``
        """

        R = self.__R

        try:
            Rgens = R.gens()
        except AttributeError:
            return {}
        
        d = {}; z = R.zero()
        for x in Rgens:
            dx = self(x)
            if dx != z:
                d[str(x)] = dx

        return d    

from sage.categories.pushout import ConstructionFunctor

class OreAlgebraFunctor(ConstructionFunctor):
    r"""
    Construction functor for Ore algebras.

    Such a functor is made from the same data as an Ore algebra, except for the base ring.
    In particular, Ore algebra functors contain sigmas and deltas, which do act on certain
    domains. The sigmas and deltas are represented by dictionaries. The functor is
    applicable to rings that contain generators named like the left hand sides of the
    sigmas and deltas, and to which the right hand sides can be casted.     
    """
    
    rank = 15 # less than polynomial ring

    def __init__(self, *gens):
        r"""
        INPUT:

        - gens: list (or tuple) of generators, each generator is specified by a tuple
          ``(a, b, c)`` where ``a`` is a variable name (string), ``b`` is a shift
          (specified as dictionary), and ``c`` is a sigma-derivation (specified as
          dictionary), or ``(a, b, c, d)`` where ``a,b,c`` are as before, and ``d``
          is a vector (w0,w1,w2) of base ring elements encoding the product rule
          for this generator: D(u*v) == w0*u*v + w1*(D(u)*v + u*D(v)) + w2*D(u)*D(v)

        The functor is only applicable to rings which are compatible with the given
        dictionaries. Applying the functor to another ring causes an error. 
        """
        from sage.categories.functor import Functor
        from sage.categories.rings import Rings
        Functor.__init__(self, Rings(), Rings())
        self.gens = tuple(tuple(g) for g in gens) 
        self.vars = [g[0] for g in gens]

    def _apply_functor(self, R):
        return OreAlgebra(R, *(self.gens))

    def __eq__(self, other):
        return (type(self) is type(other) and self.vars == other.vars
                and all(a[1:] == b[1:] for a, b in zip(self.gens, other.gens)))

    def merge(self, other):
        if self == other:
            return self
        else:
            return None

    def __str__(self):
        return "OreAlgebra" + str(list(self.vars))


def OreAlgebra(base_ring, *generators, **kwargs):
    r"""
    An Ore algebra is a noncommutative polynomial ring whose elements are
    interpreted as operators.
    
    An Ore algebra has the form `A=R[\partial_1,\partial_2,\dots,\partial_n]`
    where `R` is an integral domain and `\partial_1,\dots,\partial_n` are
    indeterminates.  For each of them, there is an associated automorphism
    `\sigma:R\\rightarrow R` and a skew-derivation `\delta:R\\rightarrow R`
    satisfying `\delta(a+b)=\delta(a)+\delta(b)` and
    `\delta(ab)=\delta(a)b+\sigma(a)\delta(b)` for all `a,b\in R`.

    The generators `\partial_i` commute with each other, but not with elements
    of the base ring `R`. Instead, we have the commutation rules `\partial u =
    \sigma(u) \partial + \delta(u)` for all `u\in R`.

    The base ring `R` must be suitable according to the following definition:
    `ZZ`, `QQ`, `GF(p)` for primes `p`, and finite algebraic extensions of `QQ`
    are suitable, and if `R` is suitable then so are `R[x]`, `R[x_1,x_2,...]`
    and `Frac(R)`. It is assumed that all the `\sigma` leave ``R.base_ring()`` fixed
    and all the `\delta` map ``R.base_ring()`` to zero. 

    A typical example of an Ore algebra is the ring of linear differential
    operators with rational function coefficients in one variable,
    e.g. `A=QQ[x][D]`. Here, `\sigma` is the identity and `\delta` is the
    standard derivation `d/dx`.

    To create an Ore algebra, supply a suitable base ring and one or more
    generators. Each generator has to be given in form of a triple
    ``(name,sigma,delta)`` where ``name`` is the desired name of the variable
    (used for printout), ``sigma`` and ``delta`` are arbitrary callable objects
    which applied to the base ring return other base ring elements in accordance
    with the relevant laws. It is not checked whether they do.

    ::

      sage: from ore_algebra import *

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

    Instead of a callable object for `\sigma` and `\delta`, also a dictionary can
    be supplied which for every generator of the base ring specifies the desired
    image. If some generator is not in the dictionary, it is understood that
    `\sigma` acts as identity on it, and that `\delta` maps it to zero.

    ::

      sage: U.<x, y> = ZZ['x', 'y']

      # here, the base ring represents the differential field QQ(x, e^x)
      sage: A.<D> = OreAlgebra(U, ('D', {}, {x:1, y:y}))

      # here, the base ring represents the difference field QQ(x, 2^x)
      sage: B.<S> = OreAlgebra(U, ('S', {x:x+1, y:2*y}, {}))

      # here too, but the algebra's generator represents the forward difference instead of the shift
      sage: C.<Delta> = OreAlgebra(U, ('Delta', {x:x+1, y:2*y}, {x:1, y:y}))

    For the most frequently needed operators, the constructor accepts their
    specification as a string only, without explicit statement of sigma or
    delta. The string has to start with one of the letters listed in the
    following table. The remainder of the string has to be the name of one
    of the generators of the base ring. The operator will affect this generator
    and leave the others untouched. 

       ============= ======================= ================ =============
       Prefix        Operator                `\sigma`         `\delta`
       ============= ======================= ================ =============
       C             Commutative variable    `\{\}`           `\{\}`
       D             Standard derivative     `\{\}`           `\{x:1\}`
       S             Standard shift          `\{x:x+1\}`      `\{\}`
       \u0394, F          Forward difference      `\{x:x+1\}`      `\{x:1\}`
       \u03B8, T, E       Euler derivative        `\{\}`           `\{x:x\}`
       Q             q-shift                 `\{x:q*x\}`      `\{\}`
       J             Jackson's q-derivative  `\{x:q*x\}`      `\{x:1\}`
       ============= ======================= ================ =============

    In the case of C, the suffix need not be a generator of the ground field but
    may be an arbitrary string. In the case of Q and J, either the base ring has
    to contain an element `q`, or the base ring element to be used instead has to
    be supplied as optional argument. 

    ::

      sage: R.<x, y> = QQ['x', 'y']
      sage: A = OreAlgebra(R, 'Dx') # This creates an Ore algebra of differential operators
      sage: A == OreAlgebra(R, ('Dx', {}, {x:1}))
      True
      sage: A == OreAlgebra(R, ('Dx', {}, {y:1})) # the Dx in A acts on x, not on y
      False 

      # This creates an Ore algebra of linear recurrence operators
      sage: A = OreAlgebra(R, 'Sx')
      sage: A == OreAlgebra(R, ('Sx', {x:x+1}, {}))
      True
      sage: A == OreAlgebra(R, ('Sx', {y:y+1}, {})) # the Sx in A acts on x, not on y
      False 
      sage: OreAlgebra(R, 'Qx', q=2)
      Univariate Ore algebra in Qx over Multivariate Polynomial Ring in x, y over Rational Field

    A generator can optionally be extended by a vector `(w_0,w_1,w_2)` of
    base ring elements which encodes the product rule for the generator:
    `D(u*v) == w_0*u*v + w_1*(D(u)*v + u*D(v)) + w_2*D(u)*D(v)`. This data
    is needed in the computation of symmetric products.     

    Ore algebras support coercion from their base rings. Furthermore, an Ore
    algebra `A` knows how to coerce commutative polynomials `p` to elements of
    `A` if the generators of the parent of `p` have the same names as the
    generators of `A`, and the base ring of the parent of `p` admits a coercion
    to the base ring of `A`. The ring of these polynomials is called the
    associated commutative algebra of `A`, and it can be obtained by calling
    ``A.associated_commutative_algebra()``.

    Elements of Ore algebras are called Ore operators. They can be constructed
    from the same data from which also elements of the associated commutative
    algebra can be constructed.

    The conversion from data to an Ore operator is equivalent to the conversion
    from the given data to an element of the associated commutative algebra, and
    from there to an Ore operator. This has the consequence that possible implicit
    information about multiplication order may be lost, for example when generating
    operators from strings:

    ::

       sage: A = OreAlgebra(QQ['x'], 'Dx')
       sage: A("Dx*x")
       x*Dx
       sage: A("Dx")*A("x")
       x*Dx + 1

    A safer way of creating operators is via a list of coefficients. These are then
    always interpreted as standing to the left of the respective algebra generator monomial.

    ::

       sage: R.<x> = QQ['x']
       sage: A.<Dx> = OreAlgebra(R, 'Dx')
       sage: A([x^2+1, 5*x-7, 7*x+18])
       (7*x + 18)*Dx^2 + (5*x - 7)*Dx + x^2 + 1
       sage: (7*x + 18)*Dx^2 + (5*x - 7)*Dx + x^2 + 1
       (7*x + 18)*Dx^2 + (5*x - 7)*Dx + x^2 + 1
       sage: _^2
       (49*x^2 + 252*x + 324)*Dx^4 + (70*x^2 + 180*x)*Dx^3 + (14*x^3 + 61*x^2 + 49*x + 216)*Dx^2 + (10*x^3 + 14*x^2 + 107*x - 49)*Dx + x^4 + 12*x^2 + 37

       sage: R.<x> = QQ['x']
       sage: A.<Sx> = OreAlgebra(QQ['x'], 'Sx')
       sage: A([x^2+1, 5*x-7, 7*x+18])
       (7*x + 18)*Sx^2 + (5*x - 7)*Sx + x^2 + 1
       sage: (7*x + 18)*Sx^2 + (5*x - 7)*Sx + x^2 + 1
       (7*x + 18)*Sx^2 + (5*x - 7)*Sx + x^2 + 1
       sage: _^2
       (49*x^2 + 350*x + 576)*Sx^4 + (70*x^2 + 187*x - 121)*Sx^3 + (14*x^3 + 89*x^2 + 69*x + 122)*Sx^2 + (10*x^3 - 4*x^2 + x - 21)*Sx + x^4 + 2*x^2 + 1
       
    """
    R = base_ring; gens = list(generators)
    zero = R.zero(); one = R.one()

    if not is_suitable_base_ring(R):
        raise TypeError("The base ring is not of the required form.")
    if len(gens) == 0:
        try:
            gens = list(kwargs['names'])
        except KeyError:
            raise TypeError("Algebra must have at least one generator")
    if len(gens) == 0:
        raise TypeError("Algebra must have at least one generator")

    product_rules = []
    for i in range(len(gens)):
        g = gens[i]
        if len(g) > 3:
            product_rules.append(tuple(g[3]))
            gens[i] = tuple(g[0:3])
        else:
            product_rules.append(None)

    # expand generator shortcuts, convert dictionaries to callables, and check that sigma(1)=1
    for i in range(len(gens)):
        if type(gens[i]) == str:
            head = gens[i][0]; 
            if head == 'C': # commutative
                s = Sigma_class(R, {}); d = Delta_class(R, {}, s)
                gens[i] = (gens[i], s, d)
                continue
            try:
                x = R(gens[i][1:])
            except:
                raise TypeError("base ring has no element '" + str(gens[i][1:]) + "'")
            if head == 'D': # derivative
                gens[i] = (gens[i], {}, {x:one})
            elif head == 'S': # shift
                gens[i] = (gens[i], {x:x + one}, lambda p: zero)
            elif head == 'F' or head == u'\u0394': # forward difference
                gens[i] = (gens[i], {x:x + one}, {x:one})
            elif head == 'T' or head == u'\u03B8' or head == 'E': # eulerian derivative
                gens[i] = (gens[i], {}, {x:x})
            elif head == 'Q': # q-shift
                if 'q' in kwargs:
                    q = R(kwargs['q'])
                else:
                    try:
                        q = R('q')
                    except:
                        raise TypeError("base ring has no element 'q'")
                if q.is_one():
                    raise ValueError("q must not be 1")
                gens[i] = (gens[i], {x:q*x}, {})
            elif head == 'J': # q-derivative
                if 'q' in kwargs:
                    q = kwargs['q']
                else:
                    try:
                        q = R('q')
                    except:
                        raise TypeError("base ring has no element 'q'")
                gens[i] = (gens[i], {x:q*x}, {x:one})
            else:
                raise TypeError("unexpected generator declaration")
        elif len(gens[i]) != 3:
            raise TypeError("unexpected generator declaration")
        s = Sigma_class(R, gens[i][1]) # assuming gens[i][1] is either a dict or a callable
        d = Delta_class(R, gens[i][2], s) # assuming gens[i][2] is either a dict or a callable
        if s(one) != one:
            raise ValueError("sigma(1) must be 1")
        gens[i] = (gens[i][0], s, d)

    # try to recognize standard operators
    is_shift = [False for q in gens]; is_qshift = [False for q in gens]; is_derivation = [False for q in gens]
    is_delta = [False for q in gens]; is_theta = [False for q in gens]; is_commutative = [False for q in gens]
    is_qderivation = [False for q in gens]
    Rgens = R.gens()

    for i in range(len(gens)):

        imgs = [(x, gens[i][1](x), gens[i][2](x)) for x in Rgens]
        imgs = [(x, u, v) for (x, u, v) in imgs if (u != x or v != zero) ]
        
        if len(imgs) == 0:
            is_commutative[i] = True
            continue
        elif len(imgs) > 1:
            continue

        x, sx, dx = imgs[0]

        try:
            if dx == one:
                if sx == x:
                    is_derivation[i] = True
                elif sx - x == one:
                    is_delta[i] = True
                elif gens[i][1](sx)*x == sx**2:
                    is_qderivation[i] = True
            elif dx == zero:
                if sx - x == one:
                    is_shift[i] = True
                    if R.base_ring() is ZZ and R.ngens() == 1:
                        if R.is_field():
                            gens[i][1].set_call(lambda q, k=1: taylor_shift_univ_int_ratfun(R(q), k))
                            gens[i][1].inverse().set_call(lambda q, k=1: taylor_shift_univ_int_ratfun(R(q), -k))
                        else:
                            gens[i][1].set_call(lambda p, k=1: taylor_shift_univ_int_poly(R(p), k))
                            gens[i][1].inverse().set_call(lambda p, k=1: taylor_shift_univ_int_poly(R(p), -k))
                    elif 0 < R.characteristic() < sys.maxsize and R.ngens() == 1:
                        if R.is_field():
                            gens[i][1].set_call(lambda q, k=1: taylor_shift_univ_modp_ratfun(R(q), k))
                            gens[i][1].inverse().set_call(lambda q, k=1: taylor_shift_univ_modp_ratfun(R(q), -k))
                        else:
                            gens[i][1].set_call(lambda p, k=1: taylor_shift_univ_modp_poly(R(p), k))
                            gens[i][1].inverse().set_call(lambda p, k=1: taylor_shift_univ_modp_poly(R(p), -k))
                elif gens[i][1](sx)*x == sx**2:
                    is_qshift[i] = True
            elif dx == x:
                if sx == x:
                    is_theta[i] = True
        except:
            pass

    for i in range(len(gens)):
        
        if product_rules[i] is not None:
            continue
        elif is_shift[i] or is_qshift[i] or is_commutative[i]:
            product_rules[i] = (zero, zero, one)
        elif is_derivation[i] or is_theta[i]:
            product_rules[i] = (zero, one, zero)

    # Select element class
    from . import ore_operator, ore_operator_1_1, ore_operator_mult
    if "element_class" in kwargs:
        operator_class = kwargs["element_class"]
    elif len(gens) > 1:
        operator_class = ore_operator_mult.MultivariateOreOperator
    elif len(Rgens) > 1:
        operator_class = ore_operator.UnivariateOreOperator
    elif is_shift[0]:
        operator_class = ore_operator_1_1.UnivariateRecurrenceOperatorOverUnivariateRing
    elif is_derivation[0]:
        operator_class = ore_operator_1_1.UnivariateDifferentialOperatorOverUnivariateRing
    elif is_qshift[0]:
        operator_class = ore_operator_1_1.UnivariateQRecurrenceOperatorOverUnivariateRing
    elif is_qderivation[0]:
        operator_class = ore_operator_1_1.UnivariateQDifferentialOperatorOverUnivariateRing
    elif is_delta[0]:
        operator_class = ore_operator_1_1.UnivariateDifferenceOperatorOverUnivariateRing
    elif is_theta[0]:
        operator_class = ore_operator_1_1.UnivariateEulerDifferentialOperatorOverUnivariateRing
    elif is_commutative[0]:
        operator_class = ore_operator_1_1.UnivariateOreOperatorOverUnivariateRing
    else:
        operator_class = ore_operator.UnivariateOreOperator

    # complain if we got any bogus keyword arguments
    for kw in kwargs:
        if kw not in ("solver", "element_class", "names", "q"):
            raise TypeError("OreAlgebra constructor got an unexpected keyword argument " + str(kw))

    alg = OreAlgebra_generic(base_ring, operator_class, tuple(gens), tuple(product_rules))

    # Select the linear system solver for matrices over the base ring
    if "solver" in kwargs:
        solvers = kwargs["solver"]
        if not isinstance(solvers, dict):
            solvers = {R : solvers}
        alg._set_solvers(solvers)

    return alg

def DifferentialOperators(base=QQ, var='x'):
    r"""
    Shorthand to construct an Ore algebra of differential operators.

    Return an Ore algebra of differential operators with polynomial
    coefficients, along with objects representing, x and d/dx.

    .. seealso:: :func:`OreAlgebra`

    INPUT:

    * ``base`` (default ``QQ``) - base ring of the polynomial coefficients
    * ``var`` (default ``x``) - variable name

    EXAMPLE::

        sage: from ore_algebra import *
        sage: Dops, x, Dx = DifferentialOperators()
        sage: Dops
        Univariate Ore algebra in Dx over Univariate Polynomial Ring in x over
        Rational Field
        sage: x*Dx + 1
        x*Dx + 1

        sage: DifferentialOperators(GF(2), 't')
        (Univariate Ore algebra in Dt over Univariate Polynomial Ring in t over
        Finite Field of size 2 (...),
        t, Dt)
    """
    var = str(var)
    Pol, x = PolynomialRing(base, var).objgen()
    Dop, Dx = OreAlgebra(Pol, 'D' + var).objgen()
    return Dop, x, Dx

class OreAlgebra_generic(UniqueRepresentation, Algebra):
    r"""
    """

    def __init__(self, base_ring, operator_class, gens, product_rules):
        # Attempt to preserve compatibility with sage < 9.1
        from sage.rings.polynomial.polynomial_ring import PolynomialRing_general
        if not hasattr(PolynomialRing_general, "_coerce_map_from_base_ring"):
            self._no_generic_basering_coercion = True
        super(self.__class__, self).__init__(base_ring)
        self._gens = gens
        self._operator_class = operator_class
        self.__solvers = {}
        self.__product_rules = product_rules

    # information extraction

    def is_integral_domain(self, proof = True):
        r"""
        Returns True because Ore algebras are always integral domains.
        """
        return True

    def is_noetherian(self):
        r"""
        Returns True because Ore algebras are always noetherian. 
        """
        return True

    def construction(self):
        r"""
        Returns a functorial description of this Ore algebra
        """
        R = self.base_ring()
        gens = tuple((str(x), self.sigma(x).dict(), self.delta(x).dict()) for x in self.gens())
        return (OreAlgebraFunctor(*gens), self.base_ring())

    def _coerce_map_from_base_ring(self):
        return None

    def _coerce_map_from_(self, P):
        r"""
        If `P` is an Ore algebra, then a coercion from `P` to self is possible
        if the base ring of `P` admits coercion to the base ring of ``self`` and
        the generators of `P` form a subset of the generators of ``self``.
        Corresponding generators are considered equal if they have the same
        name and the action of the associated `\sigma` and `\delta` agree on the
        generators of `P`'s base ring (including the base ring's base ring's
        generators and so on). 

        If `P` is not an Ore algebra, then a coercion from `P` to ``self`` is possible
        iff there is a coercion from `P` to the base ring of ``self`` or to the
        associated commutative algebra of ``self``.
        """
        if is_OreAlgebra(P):
            out = self.base_ring()._coerce_map_from_(P.base_ring())
            if out is None or out is False:
                return False
            for i in range(P.ngens()):
                found_match = False
                for j in range(self.ngens()):
                    if P.var(i) == self.var(j) and P.sigma(i) == self.sigma(j) and P.delta(i) == self.delta(j):
                        found_match = True; break
                if not found_match:
                    return False
            return True            
        else: # P is not an Ore algebra
            out = self.base_ring()._coerce_map_from_(P)
            if out is not None and out is not False:
                return True
            out = self.associated_commutative_algebra()._coerce_map_from_(P)
            if out is not None and out is not False:
                return True
            return False

    def _sage_input_(self, sib, coerced):
        # one for generators following the naming convention
        base = sib(self.base_ring())
        args = list(map(sib, [self.base_ring()] + [str(g) for g in self.gens()]))
        sie = sib.name('OreAlgebra')(*args)
        return sib.parent_with_gens(self, sie, self.variable_names(), 'R')

    def _is_valid_homomorphism_(self, codomain, im_gens):
        raise NotImplementedError

    def _repr_(self):
        try:
            return self._cached_repr
        except AttributeError:
            pass
        if self.ngens() == 1:
            r = "Univariate"
        else:
            r = "Multivariate"
        r = r + " Ore algebra in "
        for x in self._gens:
            r = r + x[0] + ", "
        r = r[:-2] + " over " + self.base_ring()._repr_()
        self._cached_repr = r
        return r

    def _latex_(self):
        try:
            return self._cached_latex
        except AttributeError:
            pass
        r = self.base_ring()._latex_() + "\\langle "
        for x in self.gens():
            r = r + x._latex_() + ", "
        r = r[:-2] + "\\rangle "
        self._cached_latex = r
        return r

    def var(self, n=0):
        r"""
        Returns the name of the `n` th generator of this algebra.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Dx> = OreAlgebra(QQ['x'].fraction_field(), 'Dx')
            sage: A.var()
            'Dx'
        """
        return self._gens[n][0]

    def _gen_to_idx(self, D):
        r"""
        If `D` is a generator of this algebra, given either as string or as an actual algebra element,
        return the index `n` such that ``self.gen(n) == self(D)``.
        If `D` is already an integer, return `D` itself. 
        An IndexError is raised if `gen` is not a generator of this algebra.
        """
        if D in ZZ:
            D = int(D)
            if D < 0 or D >= self.ngens():
                raise IndexError("No such generator.")
            return D
        D = str(D)
        for i in range(self.ngens()):
            if D == self.var(i):
                return i
        raise IndexError("No such generator.")

    def sigma(self, n=0):
        r"""
        Returns the sigma callable associated to the `n` th generator of this algebra.
        The generator can be specified by index (as integer), or by name (as string),
        or as algebra element.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Dx> = OreAlgebra(QQ['x'].fraction_field(), 'Dx')
            sage: A.sigma()
            Endomorphism defined through {'x': x}
            sage: A.sigma(0)
            Endomorphism defined through {'x': x}
            sage: A.sigma('Dx')
            Endomorphism defined through {'x': x}
            sage: A.sigma(Dx)
            Endomorphism defined through {'x': x}

        """
        return self._gens[self._gen_to_idx(n)][1]

    def delta(self, n=0):
        r"""
        Returns the delta callable associated to the `n` th generator of this algebra.
        The generator can be specified by index (as integer), or by name (as string),
        or as algebra element.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Dx> = OreAlgebra(QQ['x'].fraction_field(), 'Dx')
            sage: A.delta()
            Skew-derivation defined through {'x': 1} for Endomorphism defined through {'x': x}
            sage: A.delta(0)
            Skew-derivation defined through {'x': 1} for Endomorphism defined through {'x': x}
            sage: A.delta("Dx")
            Skew-derivation defined through {'x': 1} for Endomorphism defined through {'x': x}
            sage: A.delta(Dx)
            Skew-derivation defined through {'x': 1} for Endomorphism defined through {'x': x}
        """
        return self._gens[self._gen_to_idx(n)][2]

    def is_D(self, n=0):
        r"""
        Checks whether the `n` th generator of this algebra is the standard derivation `d/dx`
        for some generator `x` of the base ring. If so, it returns `x`, otherwise ``False``.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Dx> = OreAlgebra(ZZ['x'], 'Dx')
            sage: A.is_D()
            x
            sage: A.<Sx> = OreAlgebra(ZZ['x'], 'Sx')
            sage: A.is_D()
            False
        """
        n = self._gen_to_idx(n)
        try:
            return self.__is_D[n]
        except AttributeError:
            self.__is_D = {}
        except KeyError:
            pass

        sigma = self.sigma(n); delta = self.delta(n)
        one = self.base_ring().one()
        candidates = []
        
        for x in self.base_ring().gens():
            if sigma(x) == x and delta(x) == one:
                candidates.append(x)

        self.__is_D[n] = candidates[0] if len(candidates) == 1 else False
        
        return self.__is_D[n]        

    def is_S(self, n=0):
        r"""
        Checks whether the `n` th generator of this algebra is the standard shift `p(x)\rightarrow p(x+1)`
        for some generator `x` of the base ring. If so, it returns `x`, otherwise ``False``.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Sx> = OreAlgebra(ZZ['x'], 'Sx')
            sage: A.is_S()
            x
            sage: A.<Dx> = OreAlgebra(ZZ['x'], 'Dx')
            sage: A.is_S()
            False
        """
        n = self._gen_to_idx(n)
        try:
            return self.__is_S[n]
        except AttributeError:
            self.__is_S = {}
        except KeyError:
            pass

        sigma = self.sigma(n); delta = self.delta(n); R = self.base_ring()
        one = R.one(); zero = R.zero()
        candidates = []
        
        for x in R.gens():
            if sigma(x) == x + one and delta(x) == zero:
                candidates.append(x)

        self.__is_S[n] = candidates[0] if len(candidates) == 1 else False
        
        return self.__is_S[n]        

    def is_C(self, n=0):
        r"""
        Checks whether the `n` th generator of this algebra is a commutative variable.
        If so, it returns ``True``, otherwise ``False``.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<C> = OreAlgebra(ZZ['x'], 'C')
            sage: A.is_C()
            True
            sage: A.<Dx> = OreAlgebra(ZZ['x'], 'Dx')
            sage: A.is_C()
            False
        """
        n = self._gen_to_idx(n)
        try:
            return self.__is_C[n]
        except AttributeError:
            self.__is_C = {}
        except KeyError:
            pass

        sigma = self.sigma(n); delta = self.delta(n); R = self.base_ring()
        one = R.one(); zero = R.zero()

        self.__is_C[n] = all( (sigma(x)==x and delta(x)==zero) for x in R.gens() )
        
        return self.__is_C[n]

    def is_Delta(self, n=0):
        r"""
        Checks whether the `n` th generator of this algebra is the forward difference
        `p(x)\rightarrow p(x+1)-p(x)`
        for some generator `x` of the base ring. If so, it returns `x`, otherwise ``False``.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Fx> = OreAlgebra(ZZ['x'], 'Fx')
            sage: A.is_F()
            x
            sage: A.is_Delta()
            x
            sage: A.<Sx> = OreAlgebra(ZZ['x'], 'Sx')
            sage: A.is_F()
            False
            sage: A.is_Delta()
            False
        """
        return self.is_F(n)

    def is_F(self, n=0):
        r"""
        Checks whether the `n` th generator of this algebra is the forward difference
        `p(x)\rightarrow p(x+1)-p(x)`
        for some generator `x` of the base ring. If so, it returns `x`, otherwise ``False``.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Fx> = OreAlgebra(ZZ['x'], 'Fx')
            sage: A.is_F()
            x
            sage: A.is_Delta()
            x
            sage: A.<Sx> = OreAlgebra(ZZ['x'], 'Sx')
            sage: A.is_F()
            False
            sage: A.is_Delta()
            False
        """
        n = self._gen_to_idx(n)
        try:
            return self.__is_F[n]
        except AttributeError:
            self.__is_F = {}
        except KeyError:
            pass

        sigma = self.sigma(n); delta = self.delta(n); R = self.base_ring()
        one = R.one(); zero = R.zero()
        candidates = []
        
        for x in R.gens():
            if sigma(x) == x + one and delta(x) == one:
                candidates.append(x)

        self.__is_F[n] = candidates[0] if len(candidates) == 1 else False
        
        return self.__is_F[n]        

    def is_E(self, n=0):
        r"""
        Checks whether the `n` th generator of this algebra is the Euler derivation `x*d/dx`
        for some generator `x` of the base ring. If so, it returns `x`, otherwise ``False``.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Tx> = OreAlgebra(ZZ['x'], 'Tx')
            sage: A.is_T(), A.is_E()
            (x, x)
            sage: A.<Dx> = OreAlgebra(ZZ['x'], 'Dx')
            sage: A.is_T(), A.is_E()
            (False, False)
        """
        return self.is_T(n)

    def is_T(self, n=0):
        r"""
        Checks whether the `n` th generator of this algebra is the Euler derivation `x*d/dx`
        for some generator `x` of the base ring. If so, it returns `x`, otherwise ``False``.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Tx> = OreAlgebra(ZZ['x'], 'Tx')
            sage: A.is_T(), A.is_E()
            (x, x)
            sage: A.<Dx> = OreAlgebra(ZZ['x'], 'Dx')
            sage: A.is_T(), A.is_E()
            (False, False)
        """
        n = self._gen_to_idx(n)
        try:
            return self.__is_T[n]
        except AttributeError:
            self.__is_T = {}
        except KeyError:
            pass

        sigma = self.sigma(n); delta = self.delta(n); R = self.base_ring()
        one = R.one(); zero = R.zero(); candidates = []
        
        for x in R.gens():
            if sigma(x) == x and delta(x) == x:
                candidates.append(x)

        self.__is_T[n] = candidates[0] if len(candidates) == 1 else False
        
        return self.__is_T[n]        

    def is_Q(self, n=0):
        r"""
        Checks whether the `n` th generator of this algebra is the q-shift `p(x)\rightarrow p(q*x)`
        for some generator `x` of the base ring and some element `q` of the base ring's base ring.
        If so, it returns the pair `(x, q)`, otherwise ``False``.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Qx> = OreAlgebra(ZZ['x'], 'Qx', q=2)
            sage: A.is_Q()
            (x, 2)
            sage: A.<Sx> = OreAlgebra(ZZ['x'], 'Sx')
            sage: A.is_Q()
            False
        """
        n = self._gen_to_idx(n)
        try:
            return self.__is_Q[n]
        except AttributeError:
            self.__is_Q = {}
        except KeyError:
            pass

        sigma = self.sigma(n); delta = self.delta(n); R = self.base_ring()
        one = R.one(); zero = R.zero(); candidates = []
        
        for x in R.gens():
            try:
                sx = sigma(x); 
                if sx != x and sigma(sx)*x == sx**2 and delta(x) == zero:
                    candidates.append((x, R.base_ring()(sx(1))))
            except:
                pass

        self.__is_Q[n] = candidates[0] if len(candidates) == 1 else False
        
        return self.__is_Q[n]        

    def is_J(self, n=0):
        r"""
        Checks whether the `n` th generator of this algebra is the q-derivation `p(x)\rightarrow (p(q*x)-p(x))/(x*(q-1))`
        for some generator `x` of the base ring and some element `q`, different from 1, of the base ring's base ring.
        If so, it returns the pair `(x, q)`, otherwise ``False``.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Jx> = OreAlgebra(ZZ['x'], 'Jx', q=2)
            sage: A.is_J()
            (x, 2)
            sage: A.<Dx> = OreAlgebra(ZZ['x'], 'Dx')
            sage: A.is_J()
            False
            sage: A.<Sx> = OreAlgebra(ZZ['x'], 'Sx')
            sage: A.is_J()
            False
        """
        n = self._gen_to_idx(n)
        try:
            return self.__is_J[n]
        except AttributeError:
            self.__is_J = {}
        except KeyError:
            pass

        sigma = self.sigma(n); delta = self.delta(n); R = self.base_ring()
        one = R.one(); zero = R.zero(); candidates = []
        
        for x in R.gens():
            try:
                sx = sigma(x)
                if sx != x and sigma(sx)*x == sx**2 and delta(x) == one:
                    candidates.append((x, R(sx(one))))
            except:
                pass

        self.__is_J[n] = candidates[0] if len(candidates) == 1 else False
        
        return self.__is_J[n]        

    def variable_names(self):
        r"""
        Returns a tuple with the names (as strings) of the generators of this algebra. 

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A.<Dx> = OreAlgebra(QQ['x'], 'Dx')
            sage: A.variable_names()
            ('Dx',)
        """
        return tuple(x[0] for x in self._gens)
                        
    def characteristic(self):
        r"""
        Return the characteristic of this Ore algebra, which is the
        same as that of its base ring.
        """
        return self.base_ring().characteristic()

    def gen(self, n=0):
        r"""
        Return the indeterminate generator(s) of this Ore algebra. 
        """
        if n < 0 or n >= self.ngens():
            raise IndexError("No such generator.")
        return self.gens()[n]

    def gens(self):
        r"""
        Return a tuple of generators of this Ore algebra. 
        """
        try:
            return self._gensops
        except AttributeError:
            self._gensops = [ self(g) for g, _, _ in self._gens ]
        return self._gensops

    def gens_dict(self):
        r"""
        Returns a dictionary whose keys are the variable names of this
        Algebra as strings and whose values are the corresponding
        generators.
        """
        try:
            return self._gen_dict.copy()
        except AttributeError:
            pass
        d = {}
        for (x, _, _) in self._gens:
            d[x] = self(x)
        self._gen_dict = d.copy()
        return d

    def subalgebra(self, gens):
        r"""
        Returns the subalgebra of 'self' generated by the given generators.
        The given generators must be generators ("variables") of 'self'. 
        The base ring is not affected. 
        """
        selfgens = self.gens()
        targetgens = []
        for g in gens:
            g = selfgens[self._gen_to_idx(g)] # raises an error if g is not a generator
            targetgens.append((str(g), self.sigma(g).dict(), self.delta(g).dict(), self._product_rule(g)))

        return OreAlgebra(self.base_ring(), *targetgens)
    
    def is_zero(self):
        return self.base_ring().is_zero()
    
    def is_finite(self):
        r"""
        Return False since Ore algebras are not finite (unless the base ring is 0).
        """
        R = self.base_ring()
        if R.is_finite() and R.order() == 1:
            return True
        return False

    def is_exact(self):
        r"""
        This algebra is exact iff its base ring is
        """
        return self.base_ring().is_exact()

    def is_field(self, proof = True):
        r"""
        Returns False since Ore algebras are not fields (unless they have 0 generators and the base ring is a field)
        """
        return self.ngens() == 0 and self.base_ring().is_field()
        
    def krull_dimension(self):
        r"""
        Returns the Krull dimension of this algebra, which is the Krull dimension of the base ring
        plus the number of generators of this algebra. 
        """
        return self.base_ring().krull_dimension() + self.ngens()
        
    def ngens(self):
        r"""
        Return the number of generators of this Ore algebra
        """
        return len(self._gens)

    # generation of elements
        
    def _element_constructor_(self, *args, **kwds):
        r"""
        Create a new element based on the given arguments. 
        """
        return self._operator_class(self, *args, **kwds)
        
    def random_element(self, *args, **kwds):
        r"""
        Return a random operator. The random operator is constructed by coercing a random element
        of the associated commutative algebra to an element of this algebra. 
        """
        return self._element_constructor_(self.associated_commutative_algebra().random_element(*args, **kwds))

    def _an_element_(self, *args, **kwds):
        return self._element_constructor_(self.associated_commutative_algebra().an_element(*args, **kwds))

    # generation of related parent objects

    def associated_commutative_algebra(self):
        r"""
        Returns a polynomial ring with the same base ring as this algebra and whose generators
        have the same name as this Ore algebra's generators.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: R.<x> = QQ['x']
            sage: A = OreAlgebra(R.fraction_field(), "Dx")
            sage: A
            Univariate Ore algebra in Dx over Fraction Field of Univariate
            Polynomial Ring in x over Rational Field
            sage: A.associated_commutative_algebra()
            Univariate Polynomial Ring in Dx over Fraction Field of Univariate
            Polynomial Ring in x over Rational Field
        """
        try:
            return self._commutative_ring
        except AttributeError:
            pass
        R = self._commutative_ring = PolynomialRing(self.base_ring(), self.variable_names())
        return R

    def ideal(self, gens, coerce=True, is_known_to_be_a_groebner_basis=False):
        r"""
        Creates the left ideal generated by the given generators.

        The generators must be elements of self. 
        """
        from .ideal import OreLeftIdeal
        return OreLeftIdeal(self, gens, coerce, is_known_to_be_a_groebner_basis)

    def base_extend(self, R):
        r"""
        Creates the Ore algebra obtained from ``self`` by replacing the base ring by `R`
        """
        return self.change_ring(R)

    def _solver(self, R=None):
        r"""
        Returns this Ore algebra's preferred linear system solver.

        By default, the method returns a solver for matrices over the base ring of
        this algebra. To obtain a solver for matrices over some other ring, the
        ring can be supplied as optional argument.
        """
        if R is None:
            R = self.base_ring()

        try:
            return self.__solvers[R]
        except:
            pass

        # make a reasonable choice
        if R.is_prime_field() and R.characteristic() > 0:
            return nullspace.sage_native
        elif R is ZZ or R is QQ:
            return nullspace.sage_native
        elif is_NumberField(R):
            return nullspace.cra(nullspace.sage_native)
        elif not (is_MPolynomialRing(R) or is_PolynomialRing(R) or is_FractionField(R)):
            return nullspace.sage_native # for lack of better ideas. 

        B = R.base_ring(); field = R.is_field(); merge_levels = 0

        while (is_MPolynomialRing(B) or is_PolynomialRing(B) or is_FractionField(B)):
            field = field or B.is_field()
            merge_levels += 1
            B = B.base_ring()

        solver = nullspace.kronecker(nullspace.gauss()) # good for ZZ[x...] and GF(p)[x...]
        if B is QQ:
            solver = nullspace.clear(solver) # good for QQ[x...]
        # elif is_NumberField(B):
        #     solver = nullspace.galois(solver)
        #     # good for QQ(alpha)[x...] but not implemented
        elif not (B is ZZ or B.characteristic() > 0):
            solver = nullspace.sage_native # for lack of better ideas

        if field:
            solver = nullspace.clear(solver) # good for K(x...)

        solver = nullspace.quick_check(solver) 

        for i in range(merge_levels):
            solver = nullspace.merge(solver) # good for K(x..)(y..) 

        self.__solvers[R] = solver
        return solver

    def _set_solvers(self, solvers):
        r"""
        Defines a collection of linear system solvers which Ore algebra prefers.
        The argument is supposed to be a dictionary which some rings `R` to solvers
        for matrices with coefficients in `R`
        """
        self.__solvers = solvers.copy()

    def _product_rule(self, n=0):
        r"""
        Returns the product rule associated to the given generator.

        The product rule is a tuple `(w_0,w_1,w_2,w_3)` such that for the operator
        application we have `D(u*v) = w_0*u*v + w_1*D(u)*v + w_2*u*D(v) + w_3*D(u)*D(v)`.

        An algebra generator need not have a product rule associated to it.
        If there is none, this method returns ``None``.        
        """
        return self.__product_rules[self._gen_to_idx(n)]

    def _set_product_rules(self, rules, force=False):
        r"""
        Registers product rules for the generators of this algebra.

        A product rule for a generator `D` is a tuple `(w_0,w_1,w_2)` such that for the operator
        application we have `D(u*v) = w_0*u*v + w_1*(D(u)*v + u*D(v)) + w_2*D(u)*D(v)`.

        The input parameter ``rules`` is a list of length ``self.ngens()`` which at index ``i``
        carries either ``None`` or a coefficient tuple representing the rule.

        If ``force=False``, rules which are already registered are kept and only new rules are added
        to this algebra's set of rules. If ``force=True``, the current list of rules is discarded in
        favor of the given ``rules``.        
        """
        if force:
            self.__product_rules = list(rules)
        else:
            old = self.__product_rules; new = rules
            for i in range(self.ngens()):
                if new[i] is not None:
                    if old[i] is None:
                        old[i] = tuple(new[i])
                    else:
                        for j in range(3):
                            if old[i][j] != new[i][j]:
                                raise ValueError("inconsistent product rule specification")

    def change_ring(self, R):
        r"""
        Creates the Ore algebra obtained from ``self`` by replacing the base ring by `R`
        """
        if R is self.base_ring():
            return self
        else:
            try:
                return self.__alternate_base_rings[R]
            except AttributeError:
                self.__alternate_base_rings = {}
            except KeyError:
                pass
            A = OreAlgebra(R, *self._gens)
            self.__alternate_base_rings[R] = A
            return A

    def change_var(self, var, n=0):
        r"""
        Creates the Ore algebra obtained from ``self`` by renaming the `n` th generator to `var`
        """ 
        n = self._gen_to_idx(n)
        return self.change_var_sigma_delta(var, self._gens[n][1], self._gens[n][2], n)

    def change_sigma(self, sigma, n=0):
        r"""
        Creates the Ore algebra obtained from ``self`` by replacing the homomorphism associated to the
        `n` th generator to `\sigma`, which may be a callable or a dictionary.
        """
        n = self._gen_to_idx(n)
        return self.change_var_sigma_delta(self._gens[n][0], sigma, self._gens[n][2], n)

    def change_delta(self, delta, n=0):
        r"""
        Creates the Ore algebra obtained from ``self`` by replacing the skew-derivation associated to the
        `n` th generator to `\delta`, which may be a callable or a dictionary.
        """
        n = self._gen_to_idx(n)
        return self.change_var_sigma_delta(self._gens[n][0], self._gens[n][1], delta, n)

    def change_var_sigma_delta(self, var, sigma, delta, n=0):
        r"""
        Creates the Ore algebra obtained from ``self`` by replacing the
        `n` th generator and its associated homomorphism and skew-derivation
        by `var`, `\sigma`, and `\delta`, respectively.
        The maps `\sigma` and `\delta` may be specified as callables or dictionaries.
        """
        n = self._gen_to_idx(n)

        gens = list(self._gens)
        R = self.base_ring()
        sigma = Sigma_class(R, sigma)
        delta = Delta_class(R, delta, sigma)

        gens[n] = (var, sigma, delta)
        if var == self.var(n) and sigma == self.sigma(n) and delta == self.delta(n):
            return self
        else:
            return OreAlgebra(R, *gens)
        
