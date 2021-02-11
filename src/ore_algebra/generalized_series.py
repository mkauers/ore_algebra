r"""
Generalized series found in expansions at singularities
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

from sage.arith.all import gcd, lcm
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RR
from sage.functions.log import log as LOG
from sage.functions.log import exp as EXP
from sage.rings.number_field.number_field_base import NumberField
from sage.structure.element import Element, RingElement, canonical_coercion
from sage.structure.parent import Parent
from sage.structure.unique_representation import UniqueRepresentation
from sage.rings.infinity import infinity
from sage.rings.qqbar import QQbar

import re

from sage.categories.pushout import ConstructionFunctor
from sage.categories.functor import Functor
from sage.categories.rings import Rings

from .tools import generalized_series_term_valuation, generalized_series_default_iota

class GeneralizedSeriesFunctor(ConstructionFunctor):
    
    rank = 15

    def __init__(self, x, type):
        Functor.__init__(self, Rings(), Rings())
        self.x = x
        self.type = type

    def _apply_functor(self, R):
        return GeneralizedSeriesMonoid(R, self.x, self.type)

    def __eq__(self, other):
        return type(self) is type(other) and self.x == other.x

    def merge(self, other):
        return self if self == other else None

    def __str__(self):
        return "GeneralizedSeries in " + str(self.x)


class GeneralizedSeriesMonoid(UniqueRepresentation, Parent):
    """
    Objects of this class represent parents of generalized series objects.
    They depend on a coefficient ring, which must be either QQ or a number field,
    and a variable name. The type must be \"continuous\" or \"discrete\"
    """

    @staticmethod
    def __classcall__(cls, base, x, type="continuous"):
        if not (any(base is P for P in [ZZ, QQ, QQbar])
                or isinstance(base, NumberField)):
            raise TypeError("base ring must be ZZ, QQbar or a number field")
        x = str(x)
        if x.find("LOG") >= 0:
            raise ValueError("generator name must not contain the substring 'LOG'")
        type = str(type)
        if type != "continuous" and type != "discrete":
            raise ValueError("type must be either \"continuous\" or \"discrete\"")
        return super(GeneralizedSeriesMonoid, cls).__classcall__(cls, base, x, type)

    def __init__(self, base, x, type):
        r"""
        Creates a monoid of generalized series objects.

        INPUT:

        - ``base`` -- constant field, may be either ``QQ`` or a number field.
        - ``x`` -- name of the variable, must not contain the substring ``"log"``.
        - ``type`` (optional) -- either ``"continuous"`` or ``"discrete"``.

        If the type is ``"continuous"``, the domain contains series objects of the form

        `\exp(\int_0^x \frac{p(t^{-1/r})}t dt)*q(x^{1/r},\log(x))`

        where

        * `r` is a positive integer (the object's "ramification")
        * `p` is in `K[x]` (the object's "exponential part")
        * `q` is in `K[[x]][y]` with `x\nmid q` unless `q` is zero (the object's "tail")
        * `K` is the base ring.

        Any two such objects can be multiplied and differentiated.
        Objects whose exponential parts differ by an integer ("similar" series) can also be added.

        If the type is ``"discrete"``, the domain contains series objects of the form

        `(x/e)^{x u/v}\rho^x\exp\bigl(c_1 x^{1/(m*v)} +...+ c_{v-1} x^{1-1/(m*v)}\bigr)x^\alpha p(x^{-1/(m*v)},\log(x))`

        where

        * `e` is Euler's constant (2.71...)
        * `v` is a positive integer (the object's "ramification")
        * `u` is an integer; the term `(x/e)^(v/u)` is called the "superexponential part" of the solution
        * `\rho` is an element of an algebraic extension of the coefficient field `K`
        (the algebra's base ring's base ring); the term `\rho^x` is called the "exponential part" of
        the solution
        * `c_1,...,c_{v-1}` are elements of `K(\rho)`; the term `\exp(...)` is called the "subexponential
        part" of the solution
        * `m` is a positive integer
        * `\alpha` is an element of some algebraic extension of `K(\rho)`; the term `n^\alpha` is called
        the "polynomial part" of the solution (even if `\alpha` is not an integer)
        * `p` is an element of `K(\rho)(\alpha)[[x]][y]`. It is called the "expansion part" of the solution.

        Any two such objects can be multiplied and shifted.
        Objects with the same superexponential, exponential, and subexponential part can also be added.


        Also there is also a zero element which acts neutrally with respect to addition,
        and whose product with any other object is zero. In a strict mathematical sense,
        the set of all generalized series therefore does not form a monoid.

        Nonzero objects involving no logariths (i.e., deg(q)==0) admit a multiplicative
        inverse if the series part has finite precision.

        Coercion is supported from constants, polynomials, power series and Laurent
        series and other generalized series, provided that the respective coefficient
        domains support coercion.

        There are functions for lifting the coefficient field to some algebraic extension.

        EXAMPLES::

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
            sage: G = GeneralizedSeriesMonoid(QQ, 'x')
            sage: G
            Monoid of continuous generalized series in x over Rational Field
            sage: x = QQ['x'].gen()
            sage: G(x+2*x^3 + 4*x^4 + O(x^5))
            x*(1 + 2*x^2 + 4*x^3 + O(x^4))
            sage: G(x+2*x^3 + 4*x^4 + O(x^5), ramification=2)
            x^(1/2)*(1 + 2*x^(2/2) + 4*x^(3/2) + O(x^(4/2)))
            sage: G(x+2*x^3 + 4*x^4 + O(x^5), ramification=3)
            x^(1/3)*(1 + 2*x^(2/3) + 4*x^(3/3) + O(x^(4/3)))
            sage: f = _
            sage: f.derivative()
            x^(-2/3)*(1/3 + 2*x^(2/3) + 16/3*x^(3/3) + O(x^(4/3)))
            sage: _*f
            x^(-1/3)*(1/3 + 8/3*x^(2/3) + 20/3*x^(3/3) + O(x^(4/3)))
            sage: (G(1+x, ramification=2)*G(1+x, ramification=3)).ramification()
            6
            sage: K = QQ.extension(x^2-2, 'a'); a = K.gen()
            sage: a*G(x)
            x*a
            sage: _.parent()
            Monoid of continuous generalized series in x over Number Field in a with defining polynomial x^2 - 2
            sage: G(x).base_extend(x^3+5, 'b')
            x
            sage: _.parent()
            Monoid of continuous generalized series in x over Number Field in b with defining polynomial x^3 + 5
        """
        self.__type = type
        self.__exp_ring = base[x]
        self.__tail_ring = PowerSeriesRing(base, x)['LOG']
        self.__var = x
        self.Element = (ContinuousGeneralizedSeries if self.is_continuous()
                        else DiscreteGeneralizedSeries)
        Parent.__init__(self, base=base, names=(x,), category=Rings())

    def is_discrete(self):
        return self.__type == "discrete"

    def is_continuous(self):
        return self.__type == "continuous"        

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
        if self.is_continuous():
            self.__gen = self(self.__tail_ring.one(), exp=self.__exp_ring.one())
        else:
            self.__gen = self([QQ.zero(), ZZ.one(), self.base().one(),
                                self.__exp_ring.zero(), self.base().one(),
                                self.__tail_ring.one()])
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

    def construction(self):
        return (GeneralizedSeriesFunctor(self.var(), self.__type), self.base())

    def _coerce_map_from_(self, P):

        if isinstance(P, GeneralizedSeriesMonoid):
            return self.var() == P.var() and self.base().has_coerce_map_from(P.base())
        else:
            return (self.base().has_coerce_map_from(P) # to const
                    or self.tail_ring().base_ring().has_coerce_map_from(P)) # to power series

    def random_element(self):
        """
        Returns a random element of this monoid. 
        """
        if self.is_continuous():
            return self(self.__tail_ring.random_element(), exp=self.__exp_ring.random_element())
        else:
            raise NotImplementedError
    
    def _repr_(self):
        return "Monoid of " + self.__type + " generalized series in " + self.var() + " over " + str(self.base())

    def _latex_(self):
        x = self.var()
        if self.is_continuous():        
            return r"\bigcup_{r>0}\bigcup_{p\in " + self.base()._latex_() + "[" + x + "]}" \
                   + r"\exp\bigl(\int_x x^{-1} p(x^{-1/r})\bigr)" \
                   + self.base()._latex_() + "[[" + x + r"^{1/r}]][\log(" + x + ")]"
        else:
            return self._repr_()

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

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
            sage: G = GeneralizedSeriesMonoid(QQ, 'x', 'continuous')
            sage: G
            Monoid of continuous generalized series in x over Rational Field
            sage: x = ZZ['x'].gen()
            sage: G1 = G.base_extend(x^2 + 2, 'a')
            sage: G1
            Monoid of continuous generalized series in x over Number Field in a with defining polynomial x^2 + 2
            sage: G2 = G1.base_extend(x^3 + 5, 'b')
            sage: G2
            Monoid of continuous generalized series in x over Number Field in b with defining polynomial x^3 + 5 over its base field
            sage: G2(G1.random_element()).parent() is G2
            True
            sage: G1.random_element().parent() is G2
            False
        
        """

        if not isinstance(ext, Parent):
            # assuming ext is an irreducible univariate polynomial over self.base()
            ext = self.base().extension(ext, name)

        return GeneralizedSeriesMonoid(ext, self.var(), self.__type)


class ContinuousGeneralizedSeries(RingElement):
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

        if isinstance(tail, ContinuousGeneralizedSeries):
            self.__tail = parent.tail_ring()(tail.__tail)
            self.__exp = parent.exp_ring()(tail.__exp)
            self.__ramification = tail.__ramification
            if make_monic and not self.__tail.is_zero():
                self.__tail /= self.__tail.leading_coefficient().coefficients()[0]
            return

        ramification = ZZ(ramification)

        # Move negative exponents to the exponential part if needed
        # TODO: Update coercions in doc
        x = parent.tail_ring().base_ring().gen()
        y = parent.tail_ring().gen()
        laurent = parent.tail_ring().base_ring().fraction_field()

        # tail can have many forms, we try to handle all coercions here
        tail_ring_ext = parent.tail_ring().change_ring(laurent)
        # tail_ring_ext is a polynomial ring in y over Laurent series field in x

        try:
            tail2 = tail_ring_ext(tail)
        except:
            # The above fails if tail is an element of K(x)
            # FIXME: Make the conversion more robust
            tail2 = tail_ring_ext(laurent(tail))

        val = min((c.valuation() for c in tail2.coefficients()), default=0)

        if val in ZZ :
            tail = tail2*x**(-val)
            exp += val/ramification

        # The list of coefficients contains things like O(x^1) which cannot get
        # converted into power series rings. But they still test equal to 0 so
        # we just force it.
        tailcoefs = [c if c != 0 else 0 for c in tail.coefficients(sparse=False)]
        p = parent.tail_ring()(tailcoefs)

        if ramification not in ZZ or ramification <= 0:
            raise ValueError("ramification must be a positive integer")
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
                p = p.map_coefficients(lambda q: q/(x**ZZ(ramification*alpha)))
                
            new_ram = lcm([ (e/ramification).denominator() for e in exp.exponents() ])
            if new_ram < ramification:
                for c in p.coefficients():
                    new_ram = lcm([new_ram] + [ (e/ramification).denominator() for e in c.exponents() ])

            if new_ram != ramification:
                # the actual ramification is smaller than the specified one
                quo = ramification / new_ram
                ramification = new_ram

                exp_new = dict() 
                for e in exp.exponents():
                    exp_new[e/quo] = exp[e]
                exp = exp.parent()(exp_new) # exp = exp(x^(1/quo))

                p_new = []
                for c in p.coefficients(sparse=False):
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

    def __call__(self, arg):
        """
        Evaluate this generalized series approximately at some approximate nonzero real or complex number 
        """
        if arg.parent().is_exact() and (not self.__exp.is_zero() or self.__ramification > 1):
            R = RR; arg = R(arg)
        else:
            R = arg.parent()

        out = R.one()
        if not self.__exp.is_zero():
            p = self.__exp
            x = arg**(-R.one()/self.__ramification)
            q = p[0]*LOG(arg)
            for i in range(1, p.degree() + 1):
                q += -(p[i]/i) * x**i
            out *= EXP(q)

        nn = arg**(R.one()/self.__ramification)
        out *= self.__tail.map_coefficients(lambda p: p.polynomial()(nn), R)(LOG(arg))

        return out

    def __copy__(self):
        return self

    def __getitem__(self, key):
        """
        Returns a particular coefficient of the tail. 
        The tail is regarded as an element of C[[x^(1/r)]][log(x)].
        The input of the method is either a pair (a,b) where a is the 
        exponent of log(x) and b the exponent of x, or just a rational 
        number, which amounts to the same as choosing a=0.
        Note that the exponent b only refers to the tail, excluding the
        polynomial part alpha of the series. 
        """
        if key in QQ:
            return self.__tail[0][key*self.__ramification]
        elif len(key) == 2:
            return self.__tail[key[0]][key[1]*self.__ramification]
        else:
            raise KeyError            
    
    def base_extend(self, ext, name='a'):
        """
        Lifts ``self`` to a domain with an enlarged coefficient domain.

        INPUT:

        - ``ext`` -- either a univariate irreducible polynomial over ``self.base()`` or an algebraic
          extension field of ``self.base()``
        - ``name`` (optional) -- if ``ext`` is a polynomial, this is used as a name for the generator
          of the algebraic extension.

        EXAMPLES::

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
            sage: G = GeneralizedSeriesMonoid(QQ, 'x')
            sage: s = G(1+x+x^2, exp=3*x^2, ramification=3)
            sage: s.parent()
            Monoid of continuous generalized series in x over Rational Field
            sage: x = ZZ['x'].gen()
            sage: s.base_extend(x^2 + 2, 'a')
            exp(-9/2*x^(-2/3))*(1 + x^(1/3) + x^(2/3))
            sage: _.parent()
            Monoid of continuous generalized series in x over Number Field in a with defining polynomial x^2 + 2
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
        such that ContinuousGeneralizedSeries(self.parent(), T, exp=E, ramification=s) == self
        """
        r = self.ramification()
        if s == r:
            return (self.__exp, self.__tail)

        quo = ZZ(s / r)
        
        if r*quo != s or s <= 0:
            raise ValueError("s must be a positive integer multiple of the ramification")

        exp = self.__exp
        x = exp.parent().gen()
        new_exp = exp(x**quo)

        tail = self.__tail
        x = tail.base_ring().gen()
        new_tail = tail.map_coefficients(lambda c: c(x**quo))

        return (new_exp, new_tail)

    def __eq__(self, other):

        if not isinstance(other, ContinuousGeneralizedSeries) or self.parent() is not other.parent():
            try:
                A, B = canonical_coercion(self, other)
                return A == B
            except:
                return False

        return self.__ramification == other.__ramification and self.__exp == other.__exp \
               and self.__tail == other.__tail
        
    def _mul_(self, other):

        if self.is_zero() or other.is_one():
            return self
        elif other.is_zero() or self.is_one():
            return other
        
        G = self.parent()
        s = lcm(self.ramification(), other.ramification())
        Ae, At = self.__inflate(s)
        Be, Bt = other.__inflate(s)
        
        return ContinuousGeneralizedSeries(G, At*Bt, exp=Ae + Be, ramification=s)

    def _neg_(self):
        
        return ContinuousGeneralizedSeries(self.parent(), \
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
            raise ValueError("can only add generalized series if they are \"similar\".")

        G = self.parent()
        s = lcm(self.ramification(), other.ramification())
        Ae, At = self.__inflate(s)
        Be, Bt = other.__inflate(s)

        exp_diff = ZZ(s*(Ae - Be))

        if exp_diff < 0:
            Ae, At, Be, Bt = Be, Bt, Ae, At
            exp_diff = -exp_diff

        x = At.base_ring().gen()
        return ContinuousGeneralizedSeries(G, (x**(exp_diff))*At + Bt, exp=Be, ramification=s)

    def __invert__(self):

        if self.is_zero():
            raise ZeroDivisionError
        elif self.has_logarithms():
            raise ValueError("generalized series involving logarithms are not invertible")
        else:
            return ContinuousGeneralizedSeries(self.parent(), \
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
                E = E.parent()([E[0]] + [-r/ZZ(i)*E[i] for i in range(1, E.degree() + 1)])
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
                E = E.parent()([E.base_ring().zero()] + [-r/ZZ(i)*E[i] for i in range(1, E.degree() + 1)])
                x_ram = '-1' if r == 1 else '-1/' + str(r)
                x_ram = x + '^{' + x_ram + '}'
                E_rep = r"\exp\Bigl(" + E._latex_().replace(x, x_ram) + r"\Bigr)"
                if alpha.is_zero():
                    pass
                elif alpha.is_one():
                    E_rep += "\\cdot " + x
                elif alpha in QQ:
                    E_rep += "\\cdot " + x + "^{" + str(alpha) + "}" # prefer 1/3 over \frac{1}{3} in exponents
                else:
                    E_rep += "\\cdot " + x + "^{" + alpha._latex_() + "}"

            if T.is_one():
                rep = E_rep
            elif (-T).is_one():
                rep = '-' + E_rep
            elif T._is_atomic():
                rep = E_rep + "\\cdot " + T_rep
            else:
                rep = E_rep + r"\cdot\Bigl(" + T_rep + r"\Bigr)"

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

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
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
        return ContinuousGeneralizedSeries(self.parent(), 1, exp=self.__exp, ramification=self.ramification())

    def has_exponential_part(self):
        """
        True if ``self`` has a nontrivial exponential part.

        Note that the exponential part may not show up in form of an \"exp\" term in the printout,
        but may also simply consist of some power `x^\alpha` with nonzero `\alpha`.

        EXAMPLES::

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
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
        return not self.__exp.is_zero()

    def has_logarithms(self):
        """
        True if ``self`` contains logarithmic terms. 
        """
        return self.__tail.degree() > 0

    def tail(self):
        """
        Returns the tail of this series.

        This is the series object which is obtained from ``self`` by dropping the exponential part.

        EXAMPLES::

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
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
        return ContinuousGeneralizedSeries(self.parent(), self.__tail, exp=0, ramification=self.ramification())

    def ramification(self):
        """
        Returns the ramification of this series object.

        This is the smallest positive integer `r` such that replacing `x` by `x^r` in the series
        clears the denominators of all exponents.

        EXAMPLES::

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
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

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
            sage: G = GeneralizedSeriesMonoid(QQ, 'x')
            sage: G(1+x+x^2,exp=17/24+5*x+7*x^2, ramification=9)
            exp(-63/2*x^(-2/9) - 45*x^(-1/9))*x^(17/24)*(1 + x^(1/9) + x^(2/9))
            sage: _.order()
            17/24
            sage: G(x^5+x^6, exp=-3)
            x^2*(1 + x)
            sage: _.order()
            2
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

        EXAMPLES::

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
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

        if not isinstance(other, ContinuousGeneralizedSeries) or self.parent() is not other.parent():
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

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
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
            raise TypeError("exponent must be a rational number")
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

        return ContinuousGeneralizedSeries(G, tail, exp=exp, ramification=b*self.ramification())

    def derivative(self):
        """
        Returns the derivative of ``self``

        EXAMPLES::

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
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

        part_1 = ContinuousGeneralizedSeries(self.parent(), \
                                             ((xE**r)*E).reverse()*T, \
                                             exp = E - E.degree()/r - 1, \
                                             ramification = r)

        part_2 = ContinuousGeneralizedSeries(self.parent(), \
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

    def initial_exponent(self):
        r"""
        Return the constant coefficient of the exponential part of this series.

        This is the exponent `\alpha` such that the series is `c x^\alpha e^{\ldots} (1 + \ldots)`.

        EXAMPLES::
        #TODO
        """
        return self.__exp.constant_coefficient()
        
    def tail_support(self):
        """
        Return the support of the tail.
        
        OUTPUT:

            L : the list of all tuples (i,j) such that `c x^j log(x)^i`, with
        `c \neq 0`, is a term of the polynomial part of ``self``.

        """
        cc = self.__tail.coefficients(sparse=False)
        L = []
        for i in range(len(cc)):
            for j in range(cc[i].degree()+1):
                if cc[i][j] != 0 :
                    L.append((i,j))
        return L

    def valuation(self, base=QQ, iota=None):
        r"""
        Return the valuation of this generalized series.

        INPUT: 

        - ``base`` (default: `QQ`) - a field. The initial exponent of ``self``
          must be coercible in that field. For the default value of ``iota``,
          ``base`` must be a subfield of `CC` with computable real part.

        - ``iota`` (default: None) - a function from ``base \times NN`` to
          ``base``, with the following additional properties:
        
            - `iota(z,j)` lies in `z + ZZ`

            - `iota(z,j) = iota(z+k,j)` for `k \in \ZZ`

            - `iota(z1,j1) + iota(z2,j2) - iota(z1+z2,j1+j2) \geq 0`

            - `iota(0,j)=j`

          If not provided, ``iota`` is the function ``QQ \times NN \to QQ``
          where `iota(z,j)` is the smallest `w=z+k` with `k \in ZZ` such that
          `x^w \log(x)^j` is bounded in a neighborhood of 0.

        OUTPUT:
        
        The valuation of this series, defined as the smallest value of
        ``z-iota(z,j)`` for all terms ``x^z log(x)^j`` in the expansion of the
        series.

        With the default value of `iota`, the valuation of the series is
        non-negative if and only if the series is bounded in a neighborhood of
        0.


        
        TODO examples

        """
        z = base(self.initial_exponent())
        # NOTE: Non optimal, we list too many terms
        t = self.tail_support()
        if len(t) == 0:
            return infinity
        else:
            return min(generalized_series_term_valuation(
                        z,j,i,iota=iota)
                       for i,j in t)

    def non_integral_terms(self, base=QQ, iota=None, cutoff=0):
        """
        List all terms in the support of self which are not integral

        INPUT:

        - ``base`` (default: `QQ`) - a field, with the same properties as in `:meth:valuation`.

        - ``iota`` (default: None) - a function from `base \times ZZ` to `base`, with the same additional properties and default value as in `:meth:valuation`. 

        - ``cutoff`` (default: 0) - an integer.

        OUTPUT:

        The list of all terms `(i,j)` of ``self`` which have valuation strictly smaller than ``cutoff``.
        If ``cutoff`` is 0, this is the list of all non-integral terms of ``self``.
        """
        z = base(self.initial_exponent())
        t = self.tail_support()
        return [(i,j+z) for i,j in t
                if generalized_series_term_valuation(z,j,i,iota=iota) < cutoff]

    def coefficient(self,a,b):
        # same as __getitem__ but takes into account the polynomial part
        # TODO: Should it be merged with __getitem__?
        z = self.initial_exponent()
        if not (b-z).is_integer():
            return 0
        else:
            try:
                return self[(a,int(ZZ(b-z)))]
            except IndexError:
                return 0

    def is_fuchsian(self,base):
        r"""
        Test whether this series is Fuchsian over the constant field `base`.
        """
        z = self.initial_exponent()
        return z in base
        

############################################################################################################

class DiscreteGeneralizedSeries(RingElement):
    """
    Objects of this class represent generalized series objects.

    See the docstring of ``GeneralizedSeriesMonoid`` for further information.
    """

    def __init__(self, parent, data, make_monic=False):
        """
        Input: data = [u/v, ram, rho, coeffs, alpha, [coeffs, coeffs, ...]]

        creates (n/e)^(u/v) * rho^n * exp(p(n^(1/ram))) * n^alpha * (p(n^(-1/ram)) + p(..)*log(n) + ...)        
        """

        Element.__init__(self, parent)

        if isinstance(data, DiscreteGeneralizedSeries):
            self.__gamma = data.__gamma
            self.__ramification = data.__ramification
            self.__rho = data.__rho
            self.__subexp = data.__subexp
            self.__alpha = data.__alpha
            self.__expansion = data.__expansion
            if make_monic and not self.__expansion.is_zero():
                self.__tail /= self.__expansion.leading_coefficient().coefficients()[0]
            return

        if type(data) != list:
            K = parent.base(); R = parent.exp_ring(); B = parent.tail_ring().base_ring()
            data = R(data)
            if data.is_zero():
                data = [QQ.zero(), ZZ.one(), K.one(), parent.exp_ring().zero(), K.zero(), R.zero()]
            else:
                alpha = data.degree(); data = data.reverse(); R = parent.tail_ring()
                data = [QQ.zero(), ZZ.one(), K.one(), parent.exp_ring().zero(), alpha, R([B(data)]) ]                

        K = parent.base()
        gamma, ram, rho, subexp, alpha, expansion = data

        self.__ramification = ZZ(ram)
        self.__gamma = QQ(gamma)
        self.__rho = K(rho)
        self.__subexp = subexp if subexp in parent.exp_ring() else parent.exp_ring()([K.zero()] + subexp)
        self.__alpha = K(alpha)

        if rho.is_zero() or ram <= 0:
            raise ValueError        

        R = parent.tail_ring(); PS = R.base_ring()
        if type(expansion) == list:
            for i in range(len(expansion)):
                if type(expansion[i]) == list:
                    expansion[i] = PS(expansion[i], len(expansion[i]))
        
        self.__expansion = parent.tail_ring()(expansion)

        if self.__expansion.is_zero():
            self.__ramification = ZZ.one()
            self.__gamma = ZZ.zero()
            self.__rho = K.zero()
            self.__subexp = parent.exp_ring().zero()
            self.__alpha = K.zero()
            return

        # normalize alpha
        x = PS.gen()
        diff = min(c.valuation() for c in self.__expansion.coefficients(sparse=False))
        if diff != 0:
            self.__alpha -= diff/self.__ramification
            self.__expansion = self.__expansion.map_coefficients(lambda p: p/(x**diff))

        # normalize ramification
        ram = self.__ramification
        new_ram = self.__gamma.denominator()
        new_ram = lcm([new_ram] + [ (e/ram).denominator() for e in self.__subexp.exponents() ])
        for c in self.__expansion.coefficients():
            new_ram = lcm([new_ram] + [ (e/ram).denominator() for e in c.exponents() ])

        if new_ram < ram:

            quo = ram/new_ram

            subexp_new = dict() 
            for e in self.__subexp.exponents():
                subexp_new[e/quo] = self.__subexp[e]
            self.__subexp = self.__subexp.parent()(subexp_new) 

            expansion_new = []
            for c in self.__expansion.coefficients(sparse=False):
                c_new = dict()
                for e in c.exponents():
                    c_new[int(e/quo)] = c[e]
                expansion_new.append(c.parent()(c_new, c.prec()/quo))
            self.__expansion = self.__expansion.parent()(expansion_new)

            self.__ramification = new_ram

        # make monic if requested
        if make_monic: 
            self.__expansion /= self.__expansion.leading_coefficient().coefficients()[0]

    def __copy__(self):
        return self

    def __call__(self, arg):
        """
        Evaluates this expansion at some approximate real number, or composes it with a polynomial
        of the form `n+i` where `n` is the generator of the parent of ``self`` and `i` is a nonnegative integer.

        EXAMPLES::

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
            sage: n = QQ['n'].gen()
            sage: G = GeneralizedSeriesMonoid(QQ, 'n', 'discrete')
            sage: f = G([0, 1, 1, [], 7, [[1,0,0,0,0,0]]])
            sage: f(n + 5)
            n^7*(1 + 35*n^(-1) + 525*n^(-2) + 4375*n^(-3) + 21875*n^(-4) + 65625*n^(-5) + O(n^(-6)))
        """
        E = self.parent().exp_ring()
        if arg in E and not arg in E.base_ring():
            arg = E(arg)
            if arg.degree() != 1 or arg[1] != 1 or arg[0] not in ZZ or arg[0] < 0:
                raise ValueError
            return self.shift(ZZ(arg[0]))
        elif arg in RR:
            if arg.parent().is_exact() and (self.__gamma != 0 or self.__ramification > 1 or \
                                            self.__alpha not in ZZ or self.has_logarithms()):
                R = RR; arg = R(arg)
            else:
                R = arg.parent()

            out = R.one()
            if self.__gamma != 0:
                out *= (arg/EXP(R.one()))**(self.__gamma*arg)
            if self.__rho != 1:
                out *= self.__rho**arg
            if not self.__subexp.is_zero():
                out *= EXP(self.__subexp(arg**(R.one()/self.__ramification)))
            if not self.__alpha.is_zero():
                out *= arg**self.__alpha

            nn = arg**(-R.one()/self.__ramification)
            out *= self.__expansion.map_coefficients(lambda p: p.polynomial()(nn), R)(LOG(arg))

            return out            
        else:
            raise ValueError("don't know how to evaluate discrete generalized series at" + str(arg))


    def __getitem__(self, key):
        """
        Returns a particular coefficient of the tail. 
        The tail is regarded as an element of C[[x^(-1/r)]][log(x)].
        The input of the method is either a pair (a,b) where a is the 
        exponent of log(x) and b the exponent of x, or just a rational 
        number, which amounts to the same as choosing a=0.
        Note that the exponent b only refers to the tail, excluding the
        polynomial part alpha of the series. 
        """
        if key in QQ:
            return self.__expansion[0][-key*self.__ramification]
        elif len(key) == 2:
            return self.__expansion[key[0]][-key[1]*self.__ramification]
        else:
            raise KeyError            
        
        
    def subs(self, *args, **kwargs):
        raise NotImplementedError
        
    def base_extend(self, ext, name='a'):
        """
        Lifts ``self`` to a domain with an enlarged coefficient domain.

        INPUT:

        - ``ext`` -- either a univariate irreducible polynomial over ``self.base()`` or an algebraic
          extension field of ``self.base()``
        - ``name`` (optional) -- if ``ext`` is a polynomial, this is used as a name for the generator
          of the algebraic extension.

        EXAMPLES::

            sage: from ore_algebra.generalized_series import GeneralizedSeriesMonoid
            sage: G = GeneralizedSeriesMonoid(QQ, 'x')
            sage: s = G(1+x+x^2, exp=3*x^2, ramification=3)
            sage: s.parent()
            Monoid of continuous generalized series in x over Rational Field
            sage: x = ZZ['x'].gen()
            sage: s.base_extend(x^2 + 2, 'a')
            exp(-9/2*x^(-2/3))*(1 + x^(1/3) + x^(2/3))
            sage: _.parent()
            Monoid of continuous generalized series in x over Number Field in a with defining polynomial x^2 + 2
            sage: s == s.base_extend(x^2 + 2, 'a')
            True
            sage: s is s.base_extend(x^2 + 2, 'a')
            False
        """
        return self.parent().base_extend(ext, name=name)(self)

    def __inflate(self, s):

        quo = s/self.__ramification

        if quo == 1:
            return (self.__subexp, self.__expansion)

        sub = self.__subexp
        sub = sub(sub.parent().gen()**quo)

        exp = self.__expansion
        x = exp.base_ring().gen()
        exp = exp.map_coefficients(lambda p: p(x**quo))

        return (sub, exp)        

    def __eq__(self, other):

        if not isinstance(other, DiscreteGeneralizedSeries) or self.parent() is not other.parent():
            try:
                A, B = canonical_coercion(self, other)
                return A == B
            except:
                return False

        return self.__gamma == other.__gamma and self.__ramification == other.__ramification \
                   and self.__rho == other.__rho and self.__subexp == other.__subexp \
                   and self.__alpha == other.__alpha and self.__expansion == other.__expansion

    def _mul_(self, other):

        if self.is_zero() or other.is_one():
            return self
        elif other.is_zero() or self.is_one():
            return other
        
        ram = lcm(self.__ramification, other.__ramification)

        Asub, Aexp = self.__inflate(ram)
        Bsub, Bexp = other.__inflate(ram)
        gamma = self.__gamma + other.__gamma

        return DiscreteGeneralizedSeries(self.parent(), \
                                         [self.__gamma + other.__gamma, \
                                          ram,
                                          self.__rho * other.__rho, \
                                          Asub + Bsub, \
                                          self.__alpha + other.__alpha, \
                                          Aexp*Bexp])

    def _neg_(self):

        return DiscreteGeneralizedSeries(self.parent(), \
                                         [self.__gamma, self.__ramification, self.__rho, self.__subexp, \
                                          self.__alpha, -self.__expansion])
    
    def _add_(self, other):

        if self.is_zero():
            return other
        elif other.is_zero():
            return self
        elif not self.similar(other):
            # could be generalized such as to support x^(1/3) + x^(1/2) = x^(1/2)*(1 + x^(1/6))
            raise ValueError("can only add generalized series if they are \"similar\".")

        ram = lcm(self.__ramification, other.__ramification)

        Asub, Aexp = self.__inflate(ram)
        Bsub, Bexp = other.__inflate(ram)

        diff = QQ(self.__alpha - other.__alpha)

        if diff < 0:
            alpha = other.__alpha
            Aexp *= Aexp.base_ring().gen()**(-ram*diff)
        elif diff > 0:
            alpha = self.__alpha
            Bexp *= Bexp.base_ring().gen()**(ram*diff)
        else:
            alpha = self.__alpha

        return DiscreteGeneralizedSeries(self.parent(), \
                                         [self.__gamma, self.__ramification, self.__rho, Asub, alpha, Aexp + Bexp])

    def __invert__(self):

        if self.is_zero():
            raise ZeroDivisionError
        elif self.has_logarithms():
            raise ValueError("generalized series involving logarithms are not invertible")

        return DiscreteGeneralizedSeries(self.parent(), \
                                         [-self.__gamma, self.__ramification, ~self.__rho, -self.__subexp, \
                                          -self.__alpha, ~self.__expansion])                                          
            
    def _repr_(self):

        if self.is_zero():
            return self.__expansion._repr_()

        out = ""; n = str(self.parent().exp_ring().gen());
        r = self.__ramification
        n_ram = n if r == 1 else n + '^(1/' + str(r) + ')'

        gamma = self.__gamma
        if gamma == 1:
            out += "(" + n + "/e)^" + n + "*"
        elif gamma == -1:
            out += "(" + n + "/e)^(-" + n + ")*"
        elif gamma != 0:
            out += "(" + n + "/e)^(" + str(gamma) + "*" + n + ")*"

        rho = self.__rho
        if rho != 1:
            out += "(" + str(rho) + ")^" + n + "*"

        subexp = self.__subexp
        if not subexp.is_zero():
            out += "exp(" + str(subexp).replace(n, n_ram) + ")*"

        alpha = self.__alpha
        if alpha == 0:
            out += ""
        elif alpha == 1:
            out += n + "*"
        else:
            out += n + "^(" + str(alpha) + ")*"

        n_ram = n + "^(-1)" if r == 1 else n + '^(-1/' + str(r) + ')'
        exp = str(self.__expansion).replace(n, n_ram).replace('LOG', 'log(' + n + ')')
        out += exp if len(out) == 0 else "(" + exp + ")"

        # x^{1/3}^{17} --> x^{17/3}
        out = re.sub(r"\^\(1/(?P<den>[0-9]*)\)\^(?P<num>[0-9]*)", r"^(\g<num>/\g<den>)", out)
        # x^{-1/3}^{17} --> x^{-17/3}
        out = re.sub(r"\^\(-1/(?P<den>[0-9]*)\)\^(?P<num>[0-9]*)", r"^(-\g<num>/\g<den>)", out)
        # x^{-1}^{4} --> x^{-4}
        out = re.sub(r"\^\(-1\)\^(?P<exp>[0-9]*)", r"^(-\g<exp>)", out)
        # (5) --> 5
        out = re.sub(r"\((?P<exp>[0-9]*)\)", r"\g<exp>", out)
        
        return out 

    def _latex_(self):

        if self.is_zero():
            return self.__expansion._latex_()

        out = ""; n = str(self.parent().exp_ring().gen());
        r = self.__ramification
        n_ram = n if r == 1 else n + '^{1/' + str(r) + '}'

        gamma = self.__gamma
        if gamma == 1:
            out += "(" + n + "/\\mathrm{e})^n"
        elif gamma == -1:
            out += "(" + n + "/\\mathrm{e})^{-n}"
        elif gamma != 0:
            out += "(" + n + "/\\mathrm{e})^{" + str(gamma) + "n}"

        rho = self.__rho
        if rho != 1:
            out += "(" + str(rho) + ")^{" + n + "}"

        subexp = self.__subexp
        if not subexp.is_zero():
            out += "\\exp\\Bigl(" + subexp._latex_().replace(n, n_ram) + "\\Bigr)"

        alpha = self.__alpha
        if alpha == 0:
            out += ""
        elif alpha == 1:
            out += n 
        elif alpha in QQ:
            out += n + "^{" + str(alpha) + "}"
        else:
            out += n + "^{" + alpha._latex_() + "}"

        n_ram = n + "^{-1}" if r == 1 else n + '^{-1/' + str(r) + '}'
        exp = self.__expansion._latex_().replace(n, n_ram).replace('\\mbox{LOG}', '\\log(n)')
        out += exp if len(out) == 0 else "\\Bigl(" + exp + "\\Bigr)"

        # x^{1/3}^{17} --> x^{17/3}
        out = re.sub(r"\^\{1/(?P<den>[0-9]*)\}\^\{(?P<num>[0-9]*)\}", r"^{\g<num>/\g<den>}", out)
        # x^{-1/3}^{17} --> x^{-17/3}
        out = re.sub(r"\^\{-1/(?P<den>[0-9]*)\}\^\{(?P<num>[0-9]*)\}", r"^{-\g<num>/\g<den>}", out)
        # x^{-1}^{4} --> x^{-4}
        out = re.sub(r"\^\{-1\}\^\{(?P<exp>[0-9]*)\}", r"^{-\g<exp>}", out)
        
        return out 

    def superexponential_part(self):
        return self.__gamma

    def exponential_part(self):
        return self.__rho

    def has_logarithms(self):
        return self.__expansion.degree() > 0

    def ramification(self):
        return self.__ramification

    def leading_exponent(self):
        return self.__alpha

    def base(self):
        return self.parent().base()

    def similar(self, other, reference=ZZ):

        if not isinstance(other, DiscreteGeneralizedSeries) or self.parent() is not other.parent():
            try:
                A, B = canonical_coercion(self, other)
                return A.similar(B)
            except:
                return False

        return self.__gamma == other.__gamma and self.__rho == other.__rho and \
               self.__ramification == other.__ramification and self.__subexp == other.__subexp and \
               self.__ramification*(self.__alpha - other.__alpha) in reference

    def is_zero(self):
        return self.__expansion.is_zero()

    def is_one(self):
        return self.__gamma == 0 and self.__rho == 1 and self.__subexp.is_zero() and self.__alpha == 0 \
               and self.__expansion.is_one()

    def shift(self, i=1):

        if self.is_zero():
            return self

        prec = min(c.prec() for c in self.__expansion.coefficients()); x = self.parent().exp_ring().gen()
        gamma = self.__gamma; rho = self.__rho; subexp = self.__subexp; alpha = self.__alpha
        subexp = [subexp[j] for j in range(1, max(subexp.degree(), gamma.denominator()) + 1)]
        ram = self.__ramification
        
        factor = _generalized_series_shift_quotient(x, prec=prec + 1, shift=i, gamma=gamma, rho=rho, \
                                                    subexp=subexp, ramification=ram, alpha=alpha).reverse()

        # (x+i)^(-1/ram) = x^(-1/ram) * (1+i/x)^(-1/ram) 
        x_shifted = x*sum(_binomial(-~ram, k)*(i*x**ram)**k for k in range(prec + 1))

        PS = self.parent().tail_ring().base_ring()
        expansion = self.__expansion.map_coefficients(lambda p: PS(factor*p(x_shifted), prec))

        logx_shifted = expansion.parent().gen() - sum((-i*x**ram)**k/QQ(k) for k in range(1, prec + 1))
        expansion = expansion(logx_shifted)

        return DiscreteGeneralizedSeries(self.parent(), [self.__gamma, ram, self.__rho, self.__subexp, \
                                                         self.__alpha + self.__gamma*i, expansion])
    
    def prec(self):
        """
        The precision of ``self`` is the minimum of the precisions of all the power series objects
        contained in it. 
        """
        t = self.__expansion

        if t.is_zero():
            return infinity
        else:
            return min(c.prec() for c in t.coefficients())/self.__ramification


############################################################################################################

def _binomial(lam, j): # works also when lambda is not an integer
    if type(lam) == int:
        lam = ZZ(lam)
    b = one = lam.parent().one()
    for jj in range(j):
        b *= lam/(j - jj)
        lam -= one
    return b # checked.

def _super_expansion(gamma, i, n, prec):
    # (1 + i/n)^(gamma*n)
    # = exp(gamma*i) * sum( 1/k! * sum( gamma*i^(l+1)/((l+1)*(-n)^l) , l=1..infty)^k , k=0..infty )
    inner = gamma*i*sum( (-i*n)**l/(l+1) for l in range(1, prec + 1) )
    outer = inner_pow = n.parent().one()
    for k in range(1, prec + 1):
        inner_pow = (inner_pow * inner) % (n**(prec + 1))
        outer += inner_pow/ZZ(k).factorial()
    coeffs = outer.padded_list(prec + 1)
    coeffs.reverse()
    return n.parent()(coeffs) # checked. 

def _sub_expansion(coeffs, ram, i, n, prec):
    # exp( c_1*((n+i)^(1/ram) - n^(1/ram)) + .. + c_{ram-1} ((n+i)^((ram-1)/ram) - n^((ram-1)/ram)) )
    # = prod( sum( 1/k! * sum(c_l*binom(l/ram, j)*i^j*n^(-j+l/ram), j=1..infty)^k, k=0..infty ), l=1..ram-1 )
    prod = one = n.parent().one()
    for l in range(1, len(coeffs) + 1):
        inner = coeffs[l - 1]*sum(_binomial(ZZ(l)/ram, j)* i**j * n**(ram*j - l) for j in range(1, prec + 1))
        outer = inner_pow = one
        for k in range(1, ram*prec + 1):
            inner_pow = (inner_pow * inner) % (n**(ram*prec + 1))
            outer += inner_pow/ZZ(k).factorial()
        prod = (prod*outer) % (n**(ram*prec + 1))
    coeffs = prod.padded_list(int(ram*prec + 1))
    coeffs.reverse()
    return n.parent()(coeffs) 

def _generalized_series_shift_quotient(x, prec=5, shift=1, gamma=0, rho=1, subexp=None, ramification=None, alpha=0):
    r"""
    Computes a series expansion for the shift quotient of some discrete generalized series.

    INPUT:

    - ``x`` -- generator of the univariate polynomial ring in terms of which the output should be expressed.
      The coefficient domain has to contain `\rho` and `\alpha`. 
    - ``prec`` -- a positive integer
    - ``shift`` -- a positive integer
    - ``gamma`` -- a rational number `u/v`
    - ``rho`` -- an algebraic number
    - ``subexp`` -- a list `[c_1,...,c_{m-1}]` where the `c_i` algebraic numbers and `m` is the ramification.
    - ``ramification`` -- the ramification of the series. If it is not specified, it is taken to be one more
      than the length of ``subexp``. If also ``subexp`` is not given, it is taken to be the denominator of
      ``gamma``. In any case, the ramification must be a positive integer multiple of the denominator of
      ``gamma``.
    - ``alpha`` -- an algebraic number

    OUTPUT:

    - a polynomial `p` such that

      `f(x+i)/f(x) = p(x^{1/ram})/x^{prec} + O(x^{-prec})`

      where

      `f(x) = `(x/e)^(x*u/v)\rho^x\exp\bigl(c_1 x^{1/ram} + ... + c_{v-1} x^{(v-1)/ram}\bigr)x^\alpha`

    """
    R = x.parent(); K = R.base_ring(); gamma = QQ(gamma)
    ram = ramification
    if ram is None:
        ram = gamma.denominator() if subexp is None else len(subexp) + 1 

    ram = ZZ(ram)
    if not (ram % gamma.denominator()).is_zero():
        raise ValueError("ramification must be a multiple of the denominator of gamma.")

    if shift == 0:
        return x**(ram*prec)
    elif shift < 0:
        raise ValueError("only nonnegative shifts are allowed")

    if gamma != 0:
        
        gamma = QQ(gamma); u = gamma.numerator(); v = gamma.denominator()

        if prec + shift*gamma < 0:
            raise ValueError("insufficient precision")
        
        # n^(i*gamma) (1+i/n)^(gamma*i) (1+i/n)^(gamma*n) 
        cert = sum(_binomial(shift*gamma, k) * shift**k * x**(ram*(prec + shift*gamma - k)) \
                   for k in range(prec + (shift*gamma).floor() + 1))
        cert = (cert * _super_expansion(gamma, shift, x, prec)(x**ram)).shift(-ram*prec)
        
    else:
        u = ZZ.zero(); v = ZZ.one(); cert = x**(ram*prec)

    if rho != 1:
        cert *= rho**shift

    if subexp is not None and not all(c == 0 for c in subexp):
        cert = (cert*_sub_expansion(subexp, ram, shift, x, prec)).shift(-ram*prec)

    if alpha != 0:
        cert = (cert*sum(_binomial(alpha, k)*(shift**k)*(x**(ram*(prec - k))) for k in range(prec + 1))).shift(-ram*prec)

    return x.parent()(cert)
