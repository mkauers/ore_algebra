
"""
generalized_series
==================

"""

from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.number_field.number_field import is_NumberField
from sage.structure.element import Element, RingElement, canonical_coercion
from sage.structure.parent import Parent
from sage.rings.infinity import infinity
from sage.rings.arith import gcd, lcm

import re

def GeneralizedSeriesMonoid(base, x):
    r"""
    Creates a monoid of generalized series objects.

    INPUT:

    - ``base`` -- constant field, may be either ``QQ`` or a number field. 
    - ``x`` -- name of the variable, must not contain the substring ``"log"``.

    The returned domain consists of series objects of the form 

    `\exp(\int_0^x \frac{p(t^{-1/r})}t dt)*q(x^{1/r},\log(x))`

    where

    * `r` is a positive integer (the object's "ramification")
    * `p` is in `K[x]` (the object's "exponential part")
    * `q` is in `K[[x]][y]` with `x\nmid q` unless `q` is zero (the object's "tail")
    * `K` is the base ring.

    Generalized series objects are created by providing `q`, `p` (optionally,
    defaults to `0`), and `r` (optionally, defaults to `1`). Note that the
    ramification is part of the element, not part of the parent. The parent only
    consists of the base ring and the name of the generator.

    Any two objects can be multiplied and differentiated. 

    Two objects are called "similar" if their exponential parts differ by an integer.
    Similar objects can be added.

    Also there is also a zero element which acts neutrally with respect to addition
    (it is considered similar to all other objects), and whose product with any other
    object is zero. In a strict mathematical sense, the set of all generalized series
    therefore does not form a monoid. 

    Nonzero objects involving no logariths (i.e., deg(q)==0) admit a multiplicative
    inverse.     

    Coercion is supported from constants, polynomials, power series and Laurent
    series and generalized series, provided that the respective coefficient
    domains support coercion.

    There are functions for lifting the coefficient field to some algebraic extension.

    EXAMPLES::

      sage: G = GeneralizedSeriesMonoid(QQ, 'x')
      sage: G
      Differential monoid of generalized series in x over Rational Field
      sage: G(x+2*x^3 + 4*x^4 + O(x^5))
      x*(1 + 2*x^2 + 4*x^3 + O(x^4))
      sage: G(x+2*x^3 + 4*x^4 + O(x^5), ramification=2)
      x^(1/2)*(1 + 2*x^(2/2) + 4*x^(3/2) + O(x^(4/2)))
      sage: G(x+2*x^3 + 4*x^4 + O(x^5), ramification=3)
      x^(1/3)*(1 + 2*x^(2/3) + 4*x^(3/3) + O(x^(4/3)))
      sage: _.derivative()
      x^(-2/3)*(1/3 + 2*x^(2/3) + 16/3*x^(3/3) + O(x^(4/3)))
      sage: _*__
      x^(-1/3)*(1/3 + 8/3*x^(2/3) + 20/3*x^(3/3) + O(x^(4/3)))
      sage: (G(1+x, ramification=2)*G(1+x, ramification=3)).ramification()
      6
      sage: K = QQ.extension(x^2-2, 'a'); a = K.gen()
      sage: a*G(x)
      a*x
      sage: _.parent()
      Differential monoid of generalized series in x over Number Field in a with defining polynomial x^2 - 2
      sage: G(x).base_extend(x^3+5, 'b')
      x
      sage: _.parent()
      Differential monoid of generalized series in x over Number Field in b with defining polynomial x^3 - 5

    """
    M = GeneralizedSeriesMonoid_class(base, x)

    # Check whether this algebra already exists.
    global _list_of_generalized_series_parents
    for m in _list_of_generalized_series_parents:
        if m == M:
            return m

    # It's new. register it and return it. 
    _list_of_generalized_series_parents.append(M)
    return M

_list_of_generalized_series_parents = []

from sage.categories.pushout import ConstructionFunctor
from sage.categories.functor import Functor
from sage.categories.rings import Rings

class GeneralizedSeriesFunctor(ConstructionFunctor):
    
    rank = 15

    def __init__(self, x):
        Functor.__init__(self, Rings(), Rings())
        self.x = x

    def _apply_functor(self, R):
        return GeneralizedSeriesMonoid(R, self.x)

    def __cmp__(self, other):
        c = cmp(type(self), type(other))
        return c if c != 0 else cmp(self.x, other.x)

    def merge(self, other):
        return self if self == other else None

    def __str__(self):
        return "GeneralizedSeries in " + str(self.x)


class GeneralizedSeriesMonoid_class(Parent):
    """
    Objects of this class represent parents of generalized series objects.
    They depend on a coefficient ring, which must be either QQ or a number field,
    and a variable name.     
    """

    def __init__(self, base, x):

        x = str(x)
        Parent.__init__(self, base=base, names=(x,), category=Rings())
        if base is not QQ and not is_NumberField(base):
            raise TypeError, "base ring must be QQ or a number field"
        if x.find("LOG") >= 0:
            raise ValueError, "generator name must not contain the substring 'LOG'"
        self.__exp_ring = base[x]
        self.__tail_ring = PowerSeriesRing(base, x)['LOG']
        self.__var = x

    def var(self):
        """
        Returns the variable name associated to this monoid. 
        """
        return self.__var

    def gen(self):
        """
        Returns the generator of this monoid
        """
        try:
            return self.__gen
        except AttributeError:
            pass
        self.__gen = self._element_constructor_(self.__tail_ring.one(), exp=self.__exp_ring.one())
        return self.__gen         

    def exp_ring(self):
        r"""
        Returns the ring which is used to store the exponential part of a generalized series.
        It is the univariate polynomial ring in over ``self.base()`` in the variable ``self.var()``.

        A polynomial `p` represents the exponential part `\exp(\int_0^x p(t^{1/r})/t dt)`,
        where `r` is the object's ramification.

        In particular, a constant polynomial `\alpha` represents \exp(\alpha\log(x))=x^\alpha`.
        """
        return self.__exp_ring

    def tail_ring(self):
        r"""
        Returns the ring which is used to store the non-exponential part (the tail) of a generalized series.
        It is the univariate polynomial ring in the variable "LOG" over the power series ring in ``self.var()``
        over ``self.base()``.

        A polynomial `p(x,y)` represents the tail `p(x^{1/r}, \log(x))`.        
        """
        return self.__tail_ring

    def __eq__(self, other):
        return isinstance(other, type(self)) \
               and self.base() is other.base() \
               and self.var() == other.var()

    def construction(self):
        return (GeneralizedSeriesFunctor(self.var()), self.base())

    def _coerce_map_from_(self, P):

        if isinstance(P, GeneralizedSeriesMonoid_class):
            if self.var() != P.var():
                return False
            m = self.base().coerce_map_from(P.base())
            return (m is not False) and (m is not None)

        m = self.base().coerce_map_from(P) # to const
        if (m is not False) and (m is not None):
            return True

        m = self.tail_ring().base_ring().coerce_map_from(P) # to power series
        if (m is not False) and (m is not None):
            return True

        return False

    def _element_constructor_(self, *args, **kwds):
        return GeneralizedSeries(self, *args, **kwds)

    def random_element(self):
        """
        Returns a random element of this monoid. 
        """
        return self._element_constructor_(self.__tail_ring.random_element(), exp=self.__exp_ring.random_element())
    
    def _repr_(self):
        return "Differential monoid of generalized series in " + self.var() + " over " + str(self.base())

    def _latex_(self):
        x = self.var()
        return r"\bigcup_{r>0}\bigcup_{p\in " + self.base()._latex_() + "[" + x + "]}" \
               + r"\exp\bigl(\int_x x^{-1} p(x^{-1/r})\bigr)" \
               + self.base()._latex_() + "[[" + x + r"^{1/r}]][\log(" + x + ")]"

    def is_exact(self):
        """
        Returns ``False``, because series objects are inherently approximate. 
        """
        return False

    def is_commutative(self):
        """
        Returns ``True``.
        """
        return True

    def base_extend(self, ext, name='a'):
        """
        Returns a monoid with an extended coefficient domain.

        INPUT:

        - ``ext`` -- either a univariate irreducible polynomial over ``self.base()`` or an algebraic
          extension field of ``self.base()``
        - ``name`` (optional) -- if ``ext`` is a polynomial, this is used as a name for the generator
          of the algebraic extension.

        EXAMPLES::

           sage: G = GeneralizedSeriesMonoid(QQ, 'x')
           sage: G
           Differential monoid of generalized series in x over Rational Field
           sage: x = ZZ['x'].gen()
           sage: G1 = G.base_extend(x^2 + 2, 'a')
           sage: G1
           Differential monoid of generalized series in x over Number Field in a with defining polynomial x^2 + 2
           sage: G2 = G1.base_extend(x^3 + 5, 'b')
           sage: G2
           Differential monoid of generalized series in x over Number Field in b with defining polynomial x^3 + 5 over its base field
           sage: G2(G1.random_element()).parent() is G2
           True
           sage: G1.random_element().parent() is G2
           False 
        
        """

        if not isinstance(ext, Parent):
            # assuming ext is an irreducible univariate polynomial over self.base()
            ext = self.base().extension(ext, name)

        return GeneralizedSeriesMonoid(ext, self.var())            


class GeneralizedSeries(RingElement):
    """
    Objects of this class represent generalized series objects.

    See the docstring of ``GeneralizedSeriesMonoid`` for further information.
    """

    def __init__(self, parent, tail, exp=0, ramification=1, make_monic=False):
        # tail: an element of K[[x]][y]
        # exp: an element of K[x]
        # ramification: a positive integer
        #
        # then the generated object represents the series
        # exp(int(exp(x^(-1/ramification)), x))*tail(x^(1/ramification), log(x))

        Element.__init__(self, parent)

        if isinstance(tail, GeneralizedSeries):
            self.__tail = parent.tail_ring()(tail.__tail)
            self.__exp = parent.exp_ring()(tail.__exp)
            self.__ramification = tail.__ramification
            if make_monic and not self.__tail.is_zero():
                self.__tail /= self.__tail.leading_coefficient().coefficients()[0]
            return

        ramification = ZZ(ramification)

        p = parent.tail_ring()(tail)

        if ramification not in ZZ or ramification <= 0:
            raise ValueError, "ramification must be a positive integer"
        elif p.is_zero():
            self.__exp = parent.exp_ring().zero()
            self.__ramification = ZZ(1)
            self.__tail = parent.tail_ring().zero()
        else:
            exp = parent.exp_ring()(exp)
            
            alpha = min(c.valuation()/ramification for c in p.coefficients())
            if not alpha.is_zero():
                # move part of the tail to the exponential part
                exp += alpha
                x = p.base_ring().gen()
                p = p.map_coefficients(lambda q: q//(x**(ramification*alpha)))
            
            new_ram = lcm([ (e/ramification).denominator() for e in exp.exponents() ])
            if new_ram < ramification:
                for c in p.coefficients():
                    new_ram = lcm([new_ram] +[ (e/ramification).denominator() for e in c.exponents() ])

            if new_ram != ramification:
                # the actual ramification is smaller than the specified one
                quo = ramification / new_ram
                ramification = new_ram

                exp_new = dict() 
                for e in exp.exponents():
                    exp_new[e/quo] = exp[e]
                exp = exp.parent()(exp_new) # exp = exp(x^(1/quo))

                p_new = []
                for c in p.coeffs():
                    c_new = dict()
                    for e in c.exponents():
                        c_new[int(e/quo)] = c[e]
                    p_new.append(c.parent()(c_new))
                p = p.parent()(p_new) # p = p(x^(1/quo), log(x))                

            if make_monic: 
                p /= p.leading_coefficient().coefficients()[0]
            
            self.__ramification = ramification
            self.__exp = exp
            self.__tail = p

    def __copy__(self):
        return self

    def base_extend(self, ext, name='a'):
        """
        Lifts ``self`` to a domain with an enlarged coefficient domain.

        INPUT:

        - ``ext`` -- either a univariate irreducible polynomial over ``self.base()`` or an algebraic
          extension field of ``self.base()``
        - ``name`` (optional) -- if ``ext`` is a polynomial, this is used as a name for the generator
          of the algebraic extension.

        EXAMPLES::

           sage: G = GeneralizedSeriesMonoid(QQ, 'x')
           sage: s = G(1+x+x^2, exp=3*x^2, ramification=3)
           sage: s.parent()
           Differential monoid of generalized series in x over Rational Field
           sage: x = ZZ['x'].gen()
           sage: s.base_extend(x^2 + 2, 'a')
           exp(-9/2*x^(-2/3))*(1 + x^(1/3) + x^(2/3))
           sage: _.parent()
           Differential monoid of generalized series in x over Number Field in a with defining polynomial x^2 + 2
           sage: s == s.base_extend(x^2 + 2, 'a')
           True
           sage: s is s.base_extend(x^2 + 2, 'a')
           False 
        
        """
        return self.parent().base_extend(ext, name=name)(self)

    def __inflate(self, s):
        """
        Write exp and tail of self as if it was a series with ramification s.
        s has to be a positive integer multiple of self's ramification.
        returns a pair (E, T), where E is the exponential part and T the tail
        such that GeneralizedSeries(self.parent(), T, exp=E, ramification=s) == self
        """
        r = self.ramification()
        if s == r:
            return (self.__exp, self.__tail)

        quo = s / r
        
        if r*quo != s or s <= 0:
            raise ValueError, "s must be a positive integer multiple of the ramification"

        exp = self.__exp
        x = exp.parent().gen()
        new_exp = exp(x**quo)

        tail = self.__tail
        x = tail.base_ring().gen()
        new_tail = tail.map_coefficients(lambda c: c(x**quo))

        return (new_exp, new_tail)

    def __eq__(self, other):

        if not isinstance(other, GeneralizedSeries) or self.parent() is not other.parent():
            A, B = canonical_coercion(self, other)
            return A == B

        return self.__ramification == other.__ramification and self.__exp == other.__exp \
               and self.__tail == other.__tail
        
    def _mul_(self, other):

        G = self.parent()
        s = lcm(self.ramification(), other.ramification())
        Ae, At = self.__inflate(s)
        Be, Bt = other.__inflate(s)
        
        return GeneralizedSeries(G, At*Bt, exp=Ae + Be, ramification=s)

    def _neg_(self):
        
        return GeneralizedSeries(self.parent(), \
                                 -self.__tail, \
                                 exp=self.__exp, \
                                 ramification=self.ramification())

    def _add_(self, other):

        if self.is_zero():
            return other
        elif other.is_zero():
            return self
        elif not self.similar(other):
            # could be generalized such as to support x^(1/3) + x^(1/2) = x^(1/2)*(1 + x^(1/6))
            raise ValueError, "can only add generalized series if they are \"similar\"."

        G = self.parent()
        s = lcm(self.ramification(), other.ramification())
        Ae, At = self.__inflate(s)
        Be, Bt = other.__inflate(s)

        exp_diff = ZZ(s*(Ae - Be))

        if exp_diff < 0:
            Ae, At, Be, Bt = Be, Bt, Ae, At
            exp_diff = -exp_diff

        x = At.base_ring().gen()
        return GeneralizedSeries(G, (x**(exp_diff))*At + Bt, exp=Be, ramification=s)

    def __invert__(self):

        if self.is_zero():
            raise ZeroDivisionError
        elif self.has_logarithms():
            raise ValueError, "generalized series involving logarithms are not invertible"
        else:
            return GeneralizedSeries(self.parent(), \
                                     ~self.__tail, \
                                     exp = -self.__exp, \
                                     ramification = self.ramification())

    def _repr_(self):

        x = str(self.__tail.base_ring().gen())
        r = self.ramification()

        x_ram = x if r == 1 else x + '^(1/' + str(r) + ')'

        T = self.__tail
        T_rep = T._repr_().replace(x, x_ram).replace('LOG', 'log(' + x + ')')

        E = self.__exp
        if E.is_zero():
            rep = T_rep
        else:

            alpha = E.constant_coefficient()
            E -= alpha

            if E.is_zero():
                E_rep = x
                if alpha.is_one():
                    pass
                elif alpha in ZZ and alpha > 0:
                    E_rep += "^" + str(alpha)
                else:
                    E_rep += "^(" + str(alpha) + ")"
            else:
                E = E.parent()([E[0]] + [-r/ZZ(i)*E[i] for i in xrange(1, E.degree() + 1)])
                x_ram = '-1' if r == 1 else '-1/' + str(r)
                x_ram = x + '^(' + x_ram + ')'
                E_rep = "exp(" + E._repr_().replace(x, x_ram) + ")"
                if alpha.is_zero():
                    pass
                elif alpha.is_one():
                    E_rep += "*" + x
                elif alpha in ZZ and alpha > 0:
                    E_rep += "*" + x + "^" + str(alpha)
                else:
                    E_rep += "*" + x + "^(" + str(alpha) + ")"

            if T.is_one():
                rep = E_rep
            elif (-T).is_one():
                rep = '-' + E_rep
            elif T._is_atomic():
                rep = E_rep + "*" + T_rep
            else:
                rep = E_rep + "*(" + T_rep + ")"

        if r > 1:
            # x^{1/3}^{17} --> x^{17/3}
            rep = re.sub(r"\^\(1/(?P<den>[0-9]*)\)\^(?P<num>[0-9]*)", r"^(\g<num>/\g<den>)", rep)
            rep = re.sub(r"\^\(-1/(?P<den>[0-9]*)\)\^(?P<num>[0-9]*)", r"^(-\g<num>/\g<den>)", rep)
        else:
            # x^{-1}^{4} --> x^{-4}
            rep = re.sub(r"\^\(-1\)\^(?P<exp>[0-9]*)", r"^(-\g<exp>)", rep)

        return rep

    def _latex_(self):

        x = self.__tail.base_ring().gen()._latex_()
        r = self.ramification()

        x_ram = x if r == 1 else x + '^{1/' + str(r) + '}'

        T = self.__tail
        T_rep = T._latex_().replace(x, x_ram).replace('LOG', r'\log(' + x + ')')

        E = self.__exp
        if E.is_zero():
            rep = T_rep

        else:

            alpha = E.constant_coefficient()
            E -= alpha
            
            if E.is_zero():
                E_rep = x
                if not alpha.is_one():
                    E_rep += "^{" + alpha._latex_() + "}"
            else:
                E = E.parent()([E.base_ring().zero()] + [-r/ZZ(i)*E[i] for i in xrange(1, E.degree() + 1)])
                x_ram = '-1' if r == 1 else '-1/' + str(r)
                x_ram = x + '^{' + x_ram + '}'
                E_rep = r"\exp\Bigl(" + E._latex_().replace(x, x_ram) + r"\Bigr)"
                if alpha.is_zero():
                    pass
                elif alpha.is_one():
                    E_rep += "*" + x
                elif alpha in QQ:
                    E_rep += "*" + x + "^{" + str(alpha) + "}" # prefer 1/3 over \frac{1}{3} in exponents
                else:
                    E_rep += "*" + x + "^{" + alpha._latex_() + "}"

            if T.is_one():
                rep = E_rep
            elif (-T).is_one():
                rep = '-' + E_rep
            elif T._is_atomic():
                rep = E_rep + "*" + T_rep
            else:
                rep = E_rep + r"*\Bigl(" + T_rep + r"\Bigr)"

        if r > 1:
            # x^{1/3}^{17} --> x^{17/3}
            rep = re.sub(r"\^\{1/(?P<den>[0-9]*)\}\^\{(?P<num>[0-9]*)\}", r"^{\g<num>/\g<den>}", rep)
            # x^{-1/3}^{4} --> x^{-4/3}
            rep = re.sub(r"\^\{-1/(?P<den>[0-9]*)\}\^\{(?P<num>[0-9]*)\}", r"^{-\g<num>/\g<den>}", rep)
        else:
            # x^{-1}^{4} --> x^{-4}
            rep = re.sub(r"\^\{-1\}\^\{(?P<exp>[0-9]*)\}", r"^{-\g<exp>}", rep)

        return rep

    def exponential_part(self):
        """
        Returns the exponential part of this series.

        This is the series obtained from ``self`` by discarding the tail.

        EXAMPLES::

          sage: G = GeneralizedSeriesMonoid(QQ, 'x')
          sage: G(1+x+x^2, exp=2*x+x^2)
          exp(-1/2*x^(-2) - 2*x^(-1))*(1 + x + x^2)
          sage: _.exponential_part()
          exp(-1/2*x^(-2) - 2*x^(-1))
          sage: G(x^3+x^4+x^5)
          x^3*(1 + x + x^2)
          sage: _.exponential_part()
          x^3
        
        """
        return GeneralizedSeries(self.parent(), 1, exp=self.__exp, ramification=self.ramification())

    def has_exponential_part(self):
        """
        True if ``self`` has a nontrivial exponential part.

        Note that the exponential part may not show up in form of an \"exp\" term in the printout,
        but may also simply consist of some power `x^\alpha` with nonzero `\alpha`.

        EXAMPLES::

          sage: G = GeneralizedSeriesMonoid(QQ, 'x')
          sage: G(1+x+x^2).has_exponential_part()
          False
          sage: G(1+x+x^2, exp=2*x+x^2).has_exponential_part()
          True
          sage: G(x+x^2).has_exponential_part()
          True
          sage: G(x+x^2) == G(1+x, exp=1)
          True 
        
        """
        return not self.exponential_part().is_zero()

    def has_logarithms(self):
        """
        True if ``self`` contains logarithmic terms. 
        """
        self.__tail.degree() > 0

    def tail(self):
        """
        Returns the tail of this series.

        This is the series object which is obtained from ``self`` by dropping the exponential part.

        EXAMPLES::

          sage: G = GeneralizedSeriesMonoid(QQ, 'x')
          sage: G(1+x+x^2, exp=2*x+x^2)
          exp(-1/2*x^(-2) - 2*x^(-1))*(1 + x + x^2)
          sage: _.tail()
          1 + x + x^2
          sage: G(x+x^2)
          x*(1 + x)
          sage: _.tail()
          1 + x
        
        """
        return GeneralizedSeries(self.parent(), self.__tail, exp=0, ramification=self.ramification())

    def ramification(self):
        """
        Returns the ramification of this series object.

        This is the smallest positive integer `r` such that replacing `x` by `x^r` in the series
        clears the denominators of all exponents.

        EXAMPLES::

          sage: G = GeneralizedSeriesMonoid(QQ, 'x')
          sage: G(1+x+x^2, ramification=2)
          1 + x^(1/2) + x^(2/2)
          sage: _.ramification()
          2
          sage: G(1+x^2+x^4, ramification=2)
          1 + x + x^2
          sage: _.ramification()
          1
        
        """
        return self.__ramification

    def order(self):
        r"""
        Returns the order of this series.

        The order is defined as the maximal coefficient ring element `\alpha`
        such that for all terms `x^i\log(x)^j` appearing in this series we have
        `i - \alpha` is a nonnegative rational number whose denominator divides
        the ramification. Note that `\alpha` itself may be a complex number.

        The order is also the constant coefficient of the polynomial used to
        represent the exponential part. 

        The order of the zero series is infinity.

        EXAMPLES::

          sage: G = GeneralizedSeriesMonoid(QQ, 'x')
          sage: G(1+x+x^2,exp=17/24+5*x+7*x^2, ramification=9)
          exp(-63/2*x^(-2/9) - 45*x^(-1/9))*x^(17/24)*(1 + x^(1/9) + x^(2/9))
          sage: _.order()
          17/24
          sage: G(x^5+x^6, exp=-3)
          x^2*(1 + x)
          sage: _.order()
        
        """
        return infinity if self.is_zero() else self.__exp.constant_coefficient()

    def base(self):
        """
        Returns the parent's coefficient domain. 
        """
        return self.parent().base()

    def similar(self, other, reference=ZZ):
        r"""
        Checks whether ``self`` and ``other`` are similar.

        Similarity is defined as follows. Let `A` and `B` be two generalized series objects
        with exponential part `\exp(\int_0^x a(t^{1/r})/t dt)` and `\exp(\int_0^x b(t^{1/r})/t dt)`
        respectively. Then `A` and `B` are called similar if `r*(a-b)` is an integer.

        An alternative reference set can be specified as optional argument.

        EXAMPLE::

          sage: G = GeneralizedSeriesMonoid(QQ, 'x')
          sage: A = G(1+x+x^2, exp=1+x+x^2, ramification=2)
          sage: B = G(1+x+x^2, exp=-3/2+x+x^2, ramification=2)
          sage: A.similar(B)
          True
          sage: B.similar(A)
          True
          sage: C = G(1+x+x^2, exp=-2/3+x+x^2, ramification=2)
          sage: A.similar(C)
          False
          sage: A.similar(C, reference=QQ)
          True
          sage: D = G(1+x+x^2, exp=1+x^2+x^4, ramification=4)
          sage: A.similar(D)
          True
          
        """

        if not isinstance(other, GeneralizedSeries) or self.parent() is not other.parent():
            A, B = canonical_coercion(self, other)
            return A.similar(B, reference=reference)

        s = lcm(self.ramification(), other.ramification())
        Ae, _ = self.__inflate(s)
        Be, _ = other.__inflate(s)

        return s*(Ae - Be) in reference

    def is_zero(self):
        """
        True if ``self`` is the monoid's zero element.
        """
        return self.__tail.is_zero()

    def is_one(self):
        """
        True if ``self`` is the monoid's one element.
        """
        return not self.has_exponential_part() and self.__tail.is_one()

    def substitute(self, e):
        """
        Returns the series object obtained from ``self`` by replacing `x` by `x^e`, where `e` is
        a positive rational number.

        EXAMPLES::

          sage: G = GeneralizedSeriesMonoid(QQ, 'x')
          sage: G(1+x+x^2, ramification=2)
          1 + x^(1/2) + x^(2/2)
          sage: _.substitute(3/5)
          1 + x^(3/10) + x^(6/10)
          sage: _.substitute(10/3)
          1 + x + x^2
          sage: _.ramification()
          1
          sage: G(1, exp=1+x+x^2, ramification=2)
          exp(-x^(-2/2) - 2*x^(-1/2))*x
          sage: _.substitute(3/5)
          exp(-x^(-6/10) - 2*x^(-3/10))*x^(3/5)
          sage: G([1,x,x^2], ramification=2)
          x^(2/2)*log(x)^2 + x^(1/2)*log(x) + 1
          sage: _.substitute(3/5)
          9/25*x^(6/10)*log(x)^2 + 3/5*x^(3/10)*log(x) + 1

        """
        if not e in QQ or e <= 0:
            raise TypeError, "exponent must be a rational number"
        elif e == 1:
            return self
        
        e = QQ(e)
        G = self.parent()

        a = e.numerator(); b = e.denominator()

        xe = G.exp_ring().gen()
        log = G.tail_ring().gen()
        xt = G.tail_ring().base_ring().gen()

        exp = e*self.__exp(xe**a)
        tail = self.__tail(e*log).map_coefficients(lambda p: p(xt**a))

        return GeneralizedSeries(G, tail, exp=exp, ramification=b*self.ramification())

    def derivative(self):
        """
        Returns the derivative of ``self``

        EXAMPLE::

           sage: G = GeneralizedSeriesMonoid(QQ, 'x')
           sage: G(1+x+x^2, exp=1+x+x^2, ramification=2)
           exp(-x^(-2/2) - 2*x^(-1/2))*x*(1 + x^(1/2) + x^(2/2))
           sage: _.derivative()
           exp(-x^(-2/2) - 2*x^(-1/2))*x^(-1)*(1 + 2*x^(1/2) + 3*x^(2/2) + 5/2*x^(3/2) + 2*x^(4/2))
           sage: G([0,0,0,1])
           log(x)^3
           sage: _.derivative()
           x^(-1)*3*log(x)^2
        
        """

        T = self.__tail
        E = self.__exp
        r = self.ramification()
        xT = T.base_ring().gen()
        xE = E.parent().gen()

        # self = exp(int(E(x^(-1/r))/x))*T(x^(1/r), log(x))

        # D(self) = E(x^(-1/r))/x * self  <<<< part1
        #         + exp(..) * ( T_1(x^(1/r),log(x)) * 1/r * x^(1/r - 1)  +  1/x * T_2(x^(1/r), log(x)) ) <<<< part2

        part_1 = GeneralizedSeries(self.parent(), \
                                   ((xE**r)*E).reverse()*T, \
                                   exp = E - E.degree()/r - 1, \
                                   ramification = r)

        part_2 = GeneralizedSeries(self.parent(), \
                                   T.map_coefficients(lambda p: p.derivative()*xT/r) + T.derivative(), \
                                   exp = E - 1, \
                                   ramification = r)

        return part_1 + part_2

    def prec(self):
        """
        The precision of ``self`` is the minimum of the precisions of all the power series objects
        contained in it. 
        """
        t = self.__tail

        if t.is_zero():
            return infinity
        else:
            return min(c.prec() for c in t.coefficients())
