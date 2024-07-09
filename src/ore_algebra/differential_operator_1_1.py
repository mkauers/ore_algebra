"""
Univariate differential operators over univariate rings
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

import logging

from functools import reduce

import sage.functions.log as symbolic_log

from sage.arith.all import gcd, lcm, nth_prime
from sage.arith.misc import valuation
from sage.matrix.constructor import matrix
from sage.misc.cachefunc import cached_method
from sage.numerical.mip import MixedIntegerLinearProgram
from sage.rings.fraction_field import FractionField_generic
from sage.rings.infinity import infinity
from sage.rings.integer_ring import ZZ
from sage.rings.number_field import number_field_base
from sage.rings.number_field.number_field import NumberField
from sage.rings.finite_rings.finite_field_constructor import FiniteField as GF
from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.rational_field import QQ
from sage.structure.element import canonical_coercion, get_coercion_model
from sage.structure.factorization import Factorization
from sage.structure.formal_sum import FormalSum, FormalSums
from sage.symbolic.all import SR

from .generalized_series import GeneralizedSeriesMonoid, _binomial
from .ore_algebra import OreAlgebra_generic, OreAlgebra
from .ore_operator_1_1 import UnivariateOreOperatorOverUnivariateRing
from .ore_operator import UnivariateOreOperator
from .tools import clear_denominators, make_factor_iterator, shift_factor, _rec2list, _power_series_solutions
from . import nullspace

#############################################################################################################

class UnivariateDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
    Element of an Ore algebra K(x)[D], where D acts as derivation d/dx on K(x).
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        if "action" not in kwargs:
            kwargs["action"] = lambda p: p.derivative()

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_S(self, alg):  # d2s
        """
        Returns a recurrence operator annihilating the coefficient sequence of
        every power series (about the origin) annihilated by ``self``.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_S()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the standard shift with respect to ``self.base_ring().gen()``.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = ZZ['x']
            sage: A.<Dx> = OreAlgebra(R, 'Dx')
            sage: R2.<n> = ZZ['n']
            sage: A2.<Sn> = OreAlgebra(R2, 'Sn')
            sage: (Dx - 1).to_S(A2)
            (n + 1)*Sn - 1
            sage: ((1+x)*Dx^2 + Dx).to_S(A2)
            (n^2 + n)*Sn + n^2
            sage: ((x^3+x^2-x)*Dx + (x^2+1)).to_S(A2)
            (-n - 1)*Sn^2 + (n + 1)*Sn + n + 1
            sage: ((x+1)*Dx^3 + Dx^2).to_S(A2)
            (n^3 - n)*Sn + n^3 - 2*n^2 + n
        """
        if isinstance(alg, str):
            R = self.base_ring()
            x = R.gen()
            one = R.one()
            rec_algebra = self.parent().change_var_sigma_delta(alg, {x: x + one}, {})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_S():
            raise TypeError("target algebra is not adequate")
        else:
            rec_algebra = alg

        if self.is_zero():
            return rec_algebra.zero()

        numer = self.numerator()
        coeffs = [list(c) for c in list(numer)]
        lengths = [len(c) for c in coeffs]

        r = len(coeffs) - 1
        d = max(lengths) - 1
        start = d + 1
        for k in range(r + 1):
            start = min(start, d - (lengths[k] - 1) + k)

        result = [[] for i in range(d + r + 1 - start)]

        def set_coeff(lst, i, x):
            while i >= len(lst):
                lst.append(0)
            lst[i] = x
            while lst and not lst[-1]:
                lst.pop()

        def from_newton_basis(coeffs, roots):
            n = len(coeffs)
            for i in range(n - 1, 0, -1):
                for j in range(i - 1, n - 1):
                    coeffs[j] -= coeffs[j + 1] * roots[i - 1]

        for k in range(start, d + r + 1):
            i = k - start
            result[i] = []

            for j in range(r + 1):
                v = d + j - k
                if v >= 0 and v < lengths[j]:
                    set_coeff(result[i], j, coeffs[j][v])

            if result[i]:
                from_newton_basis(result[i], list(range(-i, -i + r)))

        rec = rec_algebra(result)
        sigma = rec_algebra.sigma()
        v = rec.valuation()
        return rec_algebra([sigma(p, -v) for p in list(rec)[v:]])

    def to_F(self, alg):
        r"""
        Returns a difference operator annihilating the coefficient sequence of
        every power series (about the origin) annihilated by ``self``.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_F()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the forward difference with respect to ``self.base_ring().gen()``.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = ZZ['x']
            sage: A.<Dx> = OreAlgebra(R, 'Dx')
            sage: R2.<n> = ZZ['n']
            sage: A2.<Sn> = OreAlgebra(R2, 'Fn')
            sage: (Dx - 1).to_F(A2)
            (n + 1)*Fn + n
            sage: ((1+x)*Dx^2 + Dx).to_F(A2)
            (n^2 + n)*Fn + 2*n^2 + n
            sage: ((x^3+x^2-x)*Dx + (x^2+1)).to_F(A2)
            (-n - 1)*Fn^2 + (-n - 1)*Fn + n + 1

        """
        return self.to_S('S').to_F(alg)

    def to_T(self, alg):  # d2theta
        """
        Rewrites ``self`` in terms of the eulerian derivation `x*d/dx`.

        If the base ring of the target algebra is not a field, the operator returned by the
        method may not correspond exactly to ``self``, but only to a suitable left-multiple
        by a term `x^k`.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_T()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          an euler derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']
          sage: R2.<y> = ZZ['y']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: (Dx^4).to_T(OreAlgebra(R2, 'Ty'))
          Ty^4 - 6*Ty^3 + 11*Ty^2 - 6*Ty
          sage: (Dx^4).to_T('Tx').to_D(A)
          x^4*Dx^4
          sage: _.to_T('Tx')
          Tx^4 - 6*Tx^3 + 11*Tx^2 - 6*Tx
        """
        R = self.base_ring()
        x = R.gen()

        if isinstance(alg, str):
            alg = self.parent().change_var_sigma_delta(alg, {}, {x: x})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_T():
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        ord = self.order()
        z = ZZ.zero()
        stirling = [[z for j in range(ord+1)] for i in range(ord+1)]
        stirling[0][0] = ZZ.one()
        for i in range(ord):
            for j in range(ord):
                stirling[i+1][j+1] = i*stirling[i][j+1] + stirling[i][j]

        out = [R.zero() for _ in range(ord+1)]
        for i, c in enumerate(self):
            for j in range(i + 1):
                out[j] += (-1 if (i+j) % 2 else 1)*stirling[i][j]*c << (ord-i)
        val = min(pol.valuation() for pol in out)
        out = alg([pol >> val for pol in out])
        return out

    def annihilator_of_integral(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite integrals `\int f`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: ((x-1)*Dx - 2*x).annihilator_of_integral()
           (x - 1)*Dx^2 - 2*x*Dx
           sage: _.annihilator_of_associate(Dx)
           (x - 1)*Dx - 2*x

        """
        return self*self.parent().gen()

    def annihilator_of_composition(self, a, solver=None, with_transform=False):
        r"""
        Returns an operator `L` which annihilates all the functions `f(a(x))` where
        `f` runs through the functions annihilated by ``self``, and, optionally,
        a map from the quotient by ``self`` to the quotient by `L` commuting
        with the composition by `a`.

        The output operator `L` is not necessarily of smallest possible order.

        INPUT:

        - ``a`` -- either an element of the base ring of the parent of ``self``,
          or an element of an algebraic extension of this ring.
        - ``solver`` (optional) -- a callable object which applied to a matrix
          with polynomial entries returns its kernel.
        - ``with_transform`` (optional) -- if `True`, also return a
          transformation map between the quotients

        OUTPUT:

        - ``L`` -- an Ore operator such that for all ``f`` annihilated by
          ``self``, ``L`` annihilates ``f \circ a``.
        - ``conv`` -- a function which takes as input an Ore operator ``P`` and
          returns an Ore operator ``Q`` such that for all functions ``f``
          annihilated by ``self``, ``P(f)(a(x)) = Q(f \circ a)(x)``.

        EXAMPLES:

            sage: from ore_algebra import *
            sage: R.<x> = ZZ['x']
            sage: K.<y> = R.fraction_field()['y']
            sage: K.<y> = R.fraction_field().extension(y^3 - x^2*(x+1))
            sage: A.<Dx> = OreAlgebra(R, 'Dx')
            sage: (x*Dx-1).annihilator_of_composition(y) # ann for x^(2/3)*(x+1)^(1/3)
            (3*x^2 + 3*x)*Dx - 3*x - 2
            sage: (x*Dx-1).annihilator_of_composition(y + 2*x) # ann for 2*x + x^(2/3)*(x+1)^(1/3)
            (3*x^3 + 3*x^2)*Dx^2 - 2*x*Dx + 2
            sage: (Dx - 1).annihilator_of_composition(y) # ann for exp(x^(2/3)*(x+1)^(1/3))
            (-243*x^6 - 810*x^5 - 999*x^4 - 540*x^3 - 108*x^2)*Dx^3 + (-162*x^3 - 270*x^2 - 108*x)*Dx^2 + (162*x^2 + 180*x + 12)*Dx + 243*x^6 + 810*x^5 + 1080*x^4 + 720*x^3 + 240*x^2 + 32*x

        If composing with a rational function, one can also compute the
        transformation map between the quotients.

            sage: L = x*Dx^2 + 1
            sage: LL, conv = L.annihilator_of_composition(x+1, with_transform=True)
            sage: print(LL)
            (x + 1)*Dx^2 + 1
            sage: print(conv(Dx))
            Dx
            sage: print(conv(x*Dx))
            (x + 1)*Dx
            sage: print(conv(L))
            0
            sage: LL, conv = L.annihilator_of_composition(1/x, with_transform=True)
            sage: print(LL)
            -x^3*Dx^2 - 2*x^2*Dx - 1
            sage: print(conv(Dx))
            -x^2*Dx
            sage: print(conv(x*Dx))
            -x*Dx
            sage: print(conv(conv(x*Dx))) # identity since 1/1/x = x
            x*Dx
            sage: LL, conv = L.annihilator_of_composition(1+x^2, with_transform=True)
            sage: print(LL)
            (-x^3 - x)*Dx^2 + (x^2 + 1)*Dx - 4*x^3
            sage: print(conv(Dx))
            1/(2*x)*Dx
            sage: print(conv(x*Dx))
            ((x^2 + 1)/(2*x))*Dx

        """

        A = self.parent()
        K = A.base_ring().fraction_field()
        A = A.change_ring(K)
        R = K['Y']
        if solver is None:
            solver = A._solver(K)

        if self == A.one() or a == K.gen():
            if with_transform:
                return self, lambda x: x
            else:
                return self
        elif a in K.ring() and K.ring()(a).degree() == 1:
            # special handling for easy case  a == alpha*x + beta
            a = K.ring()(a)
            alpha, beta = a[1], a[0]
            x = self.base_ring().gen()
            D = A.associated_commutative_algebra().gen()
            L = A(self.polynomial()(D/alpha).map_coefficients(lambda p: p(alpha*x + beta)))
            L = L.normalize()

            if with_transform:
                def make_conv_fun(self, a, alpha, beta, D, Dif):
                    def conv_fun(A):
                        A = A.quo_rem(self)[1]
                        return Dif(A.polynomial()(D/alpha).map_coefficients(lambda p: p(alpha*x + beta)))
                    return conv_fun
                conv_fun = make_conv_fun(self, a, alpha, beta, D, A)
                return L, conv_fun
            else:
                return L

        elif a in K:
            minpoly = R.gen() - K(a)
        else:
            if with_transform:
                # FIXME: Can we do better?
                raise NotImplementedError("transformation map not implemented for algebraic functions")
            try:
                minpoly = R(a.minpoly()).monic()
            except (TypeError, ValueError, AttributeError):
                raise TypeError("argument not recognized as algebraic function over base ring")

        d = minpoly.degree()
        r = self.order()

        # derivative of a
        Da = -minpoly.map_coefficients(lambda p: p.derivative())
        Da *= minpoly.xgcd(minpoly.derivative())[2]
        Da = Da % minpoly

        # self's coefficients with x replaced by a, denominators cleared, and reduced by minpoly.
        # have: (D^r f)(a) == sum( red[i]*(D^i f)a, i=0..len(red)-1 ) and each red[i] is a poly in Y of deg <= d.
        red = [R(p.numerator().coefficients(sparse=False))
               for p in self.numerator().change_ring(K).coefficients(sparse=False)]
        lc = -minpoly.xgcd(red[-1])[2]
        red = [(red[i]*lc) % minpoly for i in range(r)]

        from sage.matrix.constructor import Matrix
        Dkfa = [R.zero() for i in range(r)]  # Dkfa[i] == coeff of (D^i f)(a) in D^k (f(a))
        Dkfa[0] = R.one()
        mat = [[q for p in Dkfa for q in p.padded_list(d)]]
        sol = []

        while len(sol) == 0:

            # compute coeffs of (k+1)th derivative
            next = [(p.map_coefficients(lambda q: q.derivative()) + p.derivative()*Da) % minpoly for p in Dkfa]
            for i in range(r - 1):
                next[i + 1] += (Dkfa[i]*Da) % minpoly
            for i in range(r):
                next[i] += (Dkfa[-1]*red[i]*Da) % minpoly
            Dkfa = next

            # check for linear relations
            mat.append([q for p in Dkfa for q in p.padded_list(d)])
            sol = solver(Matrix(K, mat).transpose())

        LL = self.parent()(list(sol[0]))

        if with_transform:
            from sage.modules.free_module_element import vector
            conv_mtx = Matrix(K, mat[:-1]).transpose().inverse()

            def make_conv_fun(self, conv_mtx, a, Dif):
                def conv_fun(A):
                    l = conv_mtx.ncols()
                    A = A.quo_rem(self)[1]
                    ring = A.parent().base_ring().fraction_field()
                    # Dif = Dif.change_ring(ring)
                    coefs = (A.coefficients(sparse=False)+[0]*l)[:l]
                    coefs = [ring(c)(a) for c in coefs]
                    return Dif((conv_mtx*vector(coefs)).list())
                return conv_fun
            conv_fun = make_conv_fun(self, conv_mtx, a, A)
            return LL, conv_fun
        else:
            return LL

    def borel_transform(self):
        r"""
        Compute the Borel transform of this operator.

        This is an operator annihilating the formal Borel transform
        `\sum_n \frac{f_{n+1}}{n!} z^n`
        of every series solution
        `\sum_n f_n z^n`
        of this operator (and the formal Borel transform of formal
        logarithmic solutions as well).

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<z> = QQ[]
            sage: Dop.<Dz> = OreAlgebra(Pol)
            sage: (Dz*(1/z)).borel_transform()
            z*Dz
            sage: Dz.borel_transform()
            z
            sage: Dop(z).borel_transform()
            1
            sage: (-z^3*Dz^2 + (-z^2-z)*Dz + 1).borel_transform()
            (-z^2 - z)*Dz - z

        TESTS::

            sage: Dop(0).borel_transform()
            0

            sage: dop = (z*Dz)^4 - 6/z*(z*Dz)^3 - 1/z^2*(z*Dz) - 7/z^2
            sage: ref = (z^4 - 6*z^3)*Dz^4 + (10*z^3 - 54*z^2 - z)*Dz^3 + (25*z^2 - 114*z - 10)*Dz^2 + (15*z - 48)*Dz + 1 # van der Hoeven 2007, Figure 3.4
            sage: ref.quo_rem(dop.borel_transform())
            (Dz, 0)

            sage: Pol0.<t> = QQ[i][]
            sage: Pol.<z> = Pol0[]
            sage: Dop.<Dz> = OreAlgebra(Pol)
            sage: (i*t*z^2*Dz).borel_transform()
            I*t*z
        """
        # Left-multiply by a suitable power of `z`;
        # substitute `z` for `z^2 D_z` and `D_z` for `1/z`.
        Dop, Pol, _, dop = self._normalize_base_ring()
        Dz, z = Dop.gen(), Pol.gen()
        z2Dz = z**2*Dz
        coeff = []
        while not dop.is_zero():
            cor, dop, rem = dop.pseudo_quo_rem(z2Dz)
            for i in range(len(coeff)):
                coeff[i] *= cor
            coeff.append(rem[0])
        deg = max((pol.degree() for pol in coeff), default=0)
        return sum(pol.reverse(deg)(Dz)*z**i for i, pol in enumerate(coeff))

    def power_series_solutions(self, n=5):
        r"""
        Computes the first few terms of the power series solutions of this operator.

        The method raises an error if Sage does not know how to factor univariate polynomials
        over the base ring's base ring.

        The base ring has to have characteristic zero.

        INPUT:

        - ``n`` -- minimum number of terms to be computed

        OUTPUT:

        A list of power series of the form `x^\alpha + ...` with pairwise distinct
        exponents `\alpha` and coefficients in the base ring's base ring's fraction field.
        All expansions are computed up to order `k` where `k` is obtained by adding the
        maximal `\alpha` to the maximum of `n` and the order of ``self``.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']
          sage: A.<Dx> = OreAlgebra(R, 'Dx')
          sage: ((1-x)*Dx - 1).power_series_solutions(10) # geometric series
          [1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + O(x^9)]
          sage: (Dx - 1).power_series_solutions(5) # exp(x)
          [1 + x + 1/2*x^2 + 1/6*x^3 + O(x^4)]
          sage: (Dx^2 - Dx + x).power_series_solutions(5) # a 2nd order equation
          [x + 1/2*x^2 + 1/6*x^3 - 1/24*x^4 + O(x^5), 1 - 1/6*x^3 - 1/24*x^4 + O(x^5)]
          sage: (2*x*Dx - 1).power_series_solutions(5) # sqrt(x) is not a power series
          []

        """
        return _power_series_solutions(self, self.to_S('S'), n, ZZ)

    def generalized_series_solutions(self, n=5, base_extend=True, ramification=True, exp=True):
        r"""
        Returns the generalized series solutions of this operator.

        These are solutions of the form

          `\exp(\int_0^x \frac{p(t^{-1/s})}t dt)*q(x^{1/s},\log(x))`

        where

        * `s` is a positive integer (the object's "ramification")
        * `p` is in `K[x]` (the object's "exponential part")
        * `q` is in `K[[x]][y]` with `x\nmid q` unless `q` is zero (the object's "tail")
        * `K` is some algebraic extension of the base ring's base ring.

        An operator of order `r` has exactly `r` linearly independent solutions of this form.
        This method computes them all, unless the flags specified in the arguments rule out some
        of them.

        At present, the method only works for operators where the base ring's base ring is either
        QQ or a number field (i.e., no finite fields, no formal parameters).

        INPUT:

        - ``n`` (default: 5) -- minimum number of terms in the series expansions to be computed
          in addition to those needed to separate all solutions from each other.
        - ``base_extend`` (default: ``True``) -- whether or not the coefficients of the solutions may
          belong to an algebraic extension of the base ring's base ring.
        - ``ramification`` (default: ``True``) -- whether or not the exponential parts of the solutions
          may involve fractional exponents.
        - ``exp`` (default: ``True``) -- set this to ``False`` if you only want solutions that have no
          exponential part (viz `\deg(p)\leq0`). If set to a positive rational number `\alpha`,
          the method returns all those solutions whose exponential part involves only terms `x^{-i/r}`
          with `i/r<\alpha`.

        OUTPUT:

        - a list of ``ContinuousGeneralizedSeries`` objects forming a fundamental system for this operator.

        .. NOTE::

          - Different solutions may require different algebraic extensions. Thus in the list returned
            by this method, the coefficient fields of different series typically do not coincide.
          - If a solution involves an algebraic extension of the coefficient field, then all its
            conjugates are solutions, too. But only one representative is listed in the output.

        ALGORITHM:

        - Ince, Ordinary Differential Equations, Chapters 16 and 17
        - Kauers/Paule, The Concrete Tetrahedron, Section 7.3

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = QQ['x']; A.<Dx> = OreAlgebra(R, 'Dx')
          sage: L = (6+6*x-3*x^2) - (10*x-3*x^2-3*x^3)*Dx + (4*x^2-6*x^3+2*x^4)*Dx^2
          sage: L.generalized_series_solutions()
          [x^3*(1 + 3/2*x + 7/4*x^2 + 15/8*x^3 + 31/16*x^4 + O(x^5)), x^(1/2)*(1 + 3/2*x + 7/4*x^2 + 15/8*x^3 + 31/16*x^4 + O(x^5))]
          sage: list(map(L, _))
          [0, 0]

          sage: L = (1-24*x+96*x^2) + (15*x-117*x^2+306*x^3)*Dx + (9*x^2-54*x^3)*Dx^2
          sage: L.generalized_series_solutions(3)
          [x^(-1/3)*(1 + x + 8/3*x^2 + O(x^3)), x^(-1/3)*((1 + x + 8/3*x^2 + O(x^3))*log(x) + x - 59/12*x^2 + O(x^3))]
          sage: list(map(L, _))
          [0, 0]

          sage: L = 216*(1+x+x^3) + x^3*(36-48*x^2+41*x^4)*Dx - x^7*(6+6*x-x^2+4*x^3)*Dx^2
          sage: L.generalized_series_solutions(3)
          [exp(3*x^(-2))*x^(-2)*(1 + 91/12*x^2 + O(x^3)), exp(-2*x^(-3) + x^(-1))*x^2*(1 + 41/3*x + 2849/36*x^2 + O(x^3))]
          sage: list(map(L, _))
          [0, 0]

          sage: L = 9 - 49*x - 2*x^2 + 6*x^2*(7 + 5*x)*Dx + 36*(-1 + x)*x^3*Dx^2
          sage: L.generalized_series_solutions()
          [exp(x^(-1/2))*x^(4/3)*(1 + x^(2/2) + x^(4/2)), exp(-x^(-1/2))*x^(4/3)*(1 + x^(2/2) + x^(4/2))]
          sage: L.generalized_series_solutions(ramification=False)
          []

          sage: L = 2*x^3*Dx^2 + 3*x^2*Dx-1
          sage: L.generalized_series_solutions()
          [exp(a_0*x^(-1/2))]
          sage: _[0].base_ring()
          Number Field in a_0 with defining polynomial x^2 - 2

        """

        R = self.base_ring()
        D = self.parent().gen()

        if self.is_zero():
            raise ZeroDivisionError("infinite dimensional solution space")
        elif self.order() == 0:
            return []
        elif R.characteristic() > 0:
            raise TypeError("cannot compute generalized solutions for this coefficient domain")
        elif R.is_field() or not R.base_ring().is_field():
            return self._normalize_base_ring()[-1].generalized_series_solutions(n, base_extend, ramification, exp)
        elif not (R.base_ring() is QQ or isinstance(R.base_ring(), number_field_base.NumberField)):
            raise TypeError("cannot compute generalized solutions for this coefficient domain")

        solutions = []

        # solutions with exponential parts

        if exp is True:
            exp = QQ(self.degree() * self.order())  # = infinity
        elif exp is False:
            exp = QQ.zero()
        if exp not in QQ:
            raise ValueError("illegal option value encountered: exp=" + str(exp))

        # search for a name which is not yet used as generator in (some subfield of) R.base_ring()
        # for in case we need to make algebraic extensions.

        # TODO: use QQbar as constant domain.

        K = R.base_ring()
        names = []
        while K is not QQ:
            names.append(str(K.gen()))
            K = K.base_ring()
        i = 0
        newname = 'a_0'
        while newname in names:
            i = i + 1
            newname = 'a_' + str(i)

        x = self.base_ring().gen()

        if exp > 0:

            c = self.coefficients(sparse=False)
            # points = [(QQ(i), QQ(c[i].valuation()))   # UNUSED !
            #           for i in range(self.order() + 1)
            #           if not c[i].is_zero()]

            y = R.base_ring()['x'].gen()  # variable name changed from y to x to avoid PARI warning
            x = R.gen()
            K = R.base_ring()
            for s, p in self.newton_polygon(x):
                e = 1 - s
                if e > 0 or -e >= exp or not (ramification or e in ZZ):
                    continue
                for q, _ in p(e*y).factor():
                    if q == y:
                        continue
                    elif q.degree() == 1:
                        c = -K(q[0]/q[1])
                    elif base_extend:
                        c = K.extension(q, newname).gen()
                    else:
                        continue
                    a = e.numerator()
                    b = e.denominator()
                    G = GeneralizedSeriesMonoid(c.parent(), x, "continuous")
                    s = G(R.one(), exp=e*c*(x**(-a)), ramification=b)
                    L = self.annihilator_of_composition(x**b).symmetric_product(x**(1-a)*D + a*c)
                    sol = L.generalized_series_solutions(n, base_extend, ramification, -a)
                    solutions = solutions + [s*f.substitute(~b) for f in sol]

        # tails
        indpoly = self.indicial_polynomial(R.gen(), 's')
        s = indpoly.parent().gen()
        x = R.gen()

        for c, e in shift_factor(indpoly):

            if c.degree() == 1:
                K = R.base_ring()
                alpha = -c[0]/c[1]
                L = self
            else:
                K = c.base_ring().extension(c, newname)
                alpha = K.gen()
                L = self.base_extend(K[x])

            from sage.rings.power_series_ring import PowerSeriesRing
            PS = PowerSeriesRing(K, str(x))
            G = GeneralizedSeriesMonoid(K, x, "continuous")

            if len(e) == 1 and e[0][1] == 1:
                # just a power series, use simpler code for this case

                coeffs = _rec2list(L.to_S('S'), [K.one()], n, alpha, False, True, lambda p: p)
                solutions.append(G(PS(coeffs, n), exp=alpha))

            else:
                # there may be logarithms, use general code
                L = L.base_extend(K[s].fraction_field()[x]).to_S('S')

                f = f0 = indpoly.parent().one()
                for a, b in e:
                    f0 *= c(s + a)**b

                for i in range(e[-1][0]):
                    f *= f0(s + i + 1)

                coeffs = _rec2list(L, [f], n, s, False, True, lambda p: p)

                # If W(s, x) denotes the power series with the above coefficient array,
                # then [ (d/ds)^i ( W(s, x)*x^s ) ]_{s=a} is a nonzero solution for every
                # root a = alpha - e[j][0] of f0 and every i=0..e[j][1]-1.

                # D_s^i (W(s, x)*x^s) = (D_s^i W + i*log(x)*D_s^(i-1) W + binom(i,2)*log(x)^2 D_s^(i-2) W + ... )*x^s.

                m = sum([ee[1] for ee in e])
                der = [coeffs]
                while len(der) < m:
                    der.append([g.derivative() for g in der[-1]])

                accum = 0
                for a, b in e:
                    der_a = {}
                    for i in range(accum + b):
                        der_a[i] = PS([g(alpha - a) for g in der[i]], len(der[i]))
                    for i in range(accum, accum + b):
                        sol = []
                        for j in range(i + 1):
                            sol.append(_binomial(i, j)*der_a[j])
                        sol.reverse()
                        solutions.append(G(sol, exp=alpha - a, make_monic=True))
                    accum += b

        return solutions

    def indicial_polynomial(self, p, var='alpha'):
        r"""
        Compute the indicial polynomial of this operator at (a root of) `p`.

        If `x` is the generator of the base ring, the input may be either
        irreducible polynomial in `x` or the rational function `1/x`.

        The output is a univariate polynomial in the given variable ``var``
        with coefficients in the base ring's base ring. It has the following
        property: for every nonzero series solution of ``self`` in rising
        powers of `p`, i.e. `p_0 p^\alpha + p_1 p^{\alpha+1} + ...`, the
        minimal exponent `\alpha` is a root of the indicial polynomial. The
        converse may not hold.

        When `p` has degree one (but not in general), the degree of the
        indicial polynomial is equal to the order of the operator if and only
        if the root of `p` is an ordinary or regular singular point of the
        operator.

        INPUT:

        - ``p`` -- an irreducible polynomial in the base ring of the operator
          algebra, or `1/x`.
        - ``var`` (optional) -- the variable name to use for the indicial
          polynomial.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = ZZ['x']; A.<Dx> = OreAlgebra(R, 'Dx');
            sage: L = (x*Dx-5).lclm((x^2+1)*Dx - 7*x).lclm(Dx - 1)
            sage: L.indicial_polynomial(x).factor()
            5 * 2^2 * (alpha - 5) * (alpha - 1) * alpha
            sage: L.indicial_polynomial(1/x).factor()
            (-1) * 2 * (alpha - 7) * (alpha - 5)
            sage: L.indicial_polynomial(x^2+1).factor()
            5 * 7 * (alpha - 1) * alpha * (2*alpha - 7)

        The indicial polynomial at `p` is not always the same as the indicial
        polynomial at a root of `p`::

            sage: from ore_algebra.examples import cbt
            sage: dop = cbt.dop[4]; dop
            (-z^3 + 6*z^2 - 5*z + 1)*Dz^5 + (2*z^3 - 18*z^2 + 40*z - 15)*Dz^4 +
            (-z^3 + 16*z^2 - 54*z + 41)*Dz^3 + (-4*z^2 + 22*z - 24)*Dz^2 +
            (-2*z + 3)*Dz
            sage: lc = dop.leading_coefficient()
            sage: dop.indicial_polynomial(lc)
            alpha^4 - 6*alpha^3 + 11*alpha^2 - 6*alpha
            sage: K.<s> = QQ.extension(lc)
            sage: z = dop.base_ring().gen()
            sage: dop.change_ring(K['z']).indicial_polynomial(z-s)
            7*alpha^5 + (-3*s - 50)*alpha^4 + (18*s + 125)*alpha^3 +
            (-33*s - 130)*alpha^2 + (18*s + 48)*alpha

        TESTS::

            sage: A(x^3 - 2).indicial_polynomial(x^2 + 1)
            1

            sage: P.<x> = QQ[]; Q.<y> = Frac(P)[]; Dops.<Dy> = OreAlgebra(Q)
            sage: dop = ((x+1)*(y*Dy)^3-x)*((y*Dy)^2+2*x*y*Dy+1)
            sage: dop.indicial_polynomial(y).factor()
            (x + 1) * (alpha^2 + 2*x*alpha + 1) * (alpha^3 - x/(x + 1))
            sage: dop = ((((3*x - 5)/(-2*x^2 - 3/4*x - 2/41))*y^2
            ....:     + ((-1/8*x^2 + 903/2*x)/(-x^2 - 1))*y
            ....:     + (-1/59*x^2 - 1/6*x - 5)/(-8*x^2 + 1/2))*Dy^2
            ....:     + (((x^2 - 1/2*x + 2)/(x + 1/3))*y^2
            ....:     + ((2*x^2 + 19*x + 1/2)/(-5*x^2 + 21/4*x - 1/2))*y
            ....:     + (1/5*x^2 - 26*x - 3)/(-x^2 - 1/3*x + 1/3))*Dy
            ....:     + ((3*x^2 + 2/5*x + 1/2)/(-139*x^2 + 2))*y^2
            ....:     + ((1/2*x^2 + 1/20*x + 1)/(4/3*x^2 + 1/6*x + 4))*y
            ....:     + (3/5*x - 3)/(-1/2*x^2))
            sage: dop.indicial_polynomial(y)
            (1/472*x^2 + 1/48*x + 5/8)*alpha^2 + (-1/472*x^2 - 1/48*x - 5/8)*alpha
            sage: dop.indicial_polynomial(dop.leading_coefficient())
            alpha

            sage: Pol.<u> = QQ[]
            sage: Dop.<Du> = OreAlgebra(Pol)
            sage: dop = ((-96040000*u^18 + 64038100*u^17 - 256116467*u^16 +
            ....: 224114567*u^15 - 32034567*u^14 + 128040267*u^13 +
            ....: 448194834*u^12 - 352189134*u^11 + 352189134*u^10 -
            ....: 448194834*u^9 - 128040267*u^8 + 32034567*u^7 - 224114567*u^6 +
            ....: 256116467*u^5 - 64038100*u^4 + 96040000*u^3)*Du^3 +
            ....: (240100000*u^17 + 96010600*u^16 - 288129799*u^15 +
            ....: 1008488600*u^14 - 2641222503*u^13 + 2593354404*u^12 -
            ....: 2977470306*u^11 + 1776857604*u^10 + 720290202*u^9 -
            ....: 1632885804*u^8 + 2977475205*u^7 - 2737326204*u^6 +
            ....: 1680832301*u^5 - 1056479200*u^4 + 288124900*u^3 -
            ....: 48020000*u^2)*Du^2 + (-480200000*u^16 - 672221200*u^15 +
            ....: 4033758398*u^14 - 5186718602*u^13 + 13062047620*u^12 -
            ....: 10757577620*u^11 + 11813792216*u^10 - 7971790408*u^9 +
            ....: 2977494796*u^8 - 2593079996*u^7 - 384081598*u^6 -
            ....: 2304950206*u^5 - 191923200*u^4 - 1344540400*u^3 - 96049800*u^2
            ....: + 96040000*u)*Du + 480200000*u^15 + 1152421200*u^14 -
            ....: 8931857198*u^13 + 6916036404*u^12 - 18344443640*u^11 +
            ....: 7588296828*u^10 - 16615302196*u^9 + 673240380*u^8 -
            ....: 14694120024*u^7 + 3650421620*u^6 - 8356068006*u^5 +
            ....: 4802156800*u^4 - 1248500400*u^3 - 96059600*u^2 + 96049800*u -
            ....: 96040000)
            sage: dop.indicial_polynomial(70*u^2 + 69*u + 70)
            alpha^3 - 3*alpha^2 + 2*alpha
            """

        x = p.parent().gen()

        if (x*p).is_one() or p == x:
            return UnivariateOreOperatorOverUnivariateRing.indicial_polynomial(self, p, var=var)

        coeff, _ = clear_denominators(self)
        op = self.parent(coeff)

        L = op.parent().base_ring()  # k[x]
        if L.is_field():
            L = L.ring()
        K = PolynomialRing(L.base_ring(), var)  # k[alpha]

        if op.is_zero():
            return K.zero()
        if op.order() == 0:
            return K.one()

        r = op.order()
        d = op.degree()

        L = L.change_ring(K)  # FF(k[alpha])[x]
        alpha = L([K.gen()])

        ffac = [L.one()]  # falling_factorial(alpha, i)
        for i in range(r + 1):
            ffac.append(L([ffac[-1][0]*(alpha - i)]))

        xpowmodp = [p.parent().one()]
        for j in range(d + 1):
            xpowmodp.append((x*xpowmodp[-1]) % p)
        for j in range(d + 1):
            xpowmodp[j] = xpowmodp[j].change_ring(op.base_ring().base_ring())

        for k in range(d + r + 1):
            algind = L.zero()
            for i in range(min(k, r) + 1):
                j0 = k - i
                coeff = 0
                for j in range(d - j0 + 1):
                    a = ZZ(j0+j).binomial(j) * op[r-i][j0+j]
                    a = xpowmodp[j]._lmul_(a)
                    coeff += a
                coeff = coeff.change_ring(K)
                algind += coeff*ffac[r-i]
            if not algind.is_zero():
                break
        else:
            assert False

        ind = K(gcd(algind.coefficients()).numerator())
        try:  # facilitate factorization
            den = lcm([p.denominator() for p in ind])
            ind *= den
        except (TypeError, ValueError, NotImplementedError):
            pass
        return ind

    def _desingularization_order_bound(self):

        m = 0
        for p, _ in self.numerator().leading_coefficient().factor():

            ip = self.indicial_polynomial(p)
            nn = 0
            for q, _ in ip.change_ring(ip.base_ring().fraction_field()).factor():
                if q.degree() == 1:
                    try:
                        nn = max(nn, ZZ(-q[0] / q[1]))
                    except (TypeError, ValueError):
                        pass
            if nn > 0:
                ip = gcd(ip, reduce(lambda p, q: p*q, [ip.parent().gen() - i for i in range(nn)]))
                m = max(m, nn - ip.degree())

        return m

    def _coeff_list_for_indicial_polynomial(self):
        return self.coefficients(sparse=False)

    def spread(self, p=0):
        L = self.numerator()
        if L[0].is_zero():
            return [infinity]
        elif L[0].gcd(L.leading_coefficient()).degree() > 0:
            return [0]
        else:
            return []

    spread.__doc__ = UnivariateOreOperatorOverUnivariateRing.spread.__doc__

    def _denominator_bound(self):
        r"""
        Denominator bounding based on indicial polynomial.

        TESTS::

            sage: from ore_algebra import *
            sage: P.<x> = QQ[]; Q.<y> = Frac(P)[]; Dops.<Dy> = OreAlgebra(Q)
            sage: u = 1/(x^2 + y)
            sage: v = 1/((y+1)*(y-1))
            sage: dop = (Dy - Dy(u)/u).lclm(Dy - Dy(v)/v)
            sage: dop._denominator_bound()
            (y - 1) * (y + 1) * (y + x^2)
            sage: dop.rational_solutions()
            [(1/(y^2 - 1),), (1/(y + x^2),)]
            sage: dop = (Dy - Dy(u)/u).lclm(Dy^2 - y)
            sage: dop._denominator_bound()
            y + x^2
            sage: dop.rational_solutions()
            [(1/(y + x^2),)]
        """
        if self.is_zero():
            raise ZeroDivisionError("unbounded denominator")

        A, R, K, L = self._normalize_base_ring()

        r = L.order()

        lc = L.leading_coefficient()
        try:  # facilitate factorization
            den = lcm([p.denominator() for p in lc])
            lc = lc.map_coefficients(lambda p: den*p)
        except (TypeError, ValueError, AttributeError):
            pass
        fac = [p for p, _ in lc.factor()]

        # specialize additional variables
        K1, vars = _tower(K)
        K1 = K1.fraction_field()
        L1, fac1 = L, fac
        if vars and K1 is QQ:
            R1 = R.change_ring(K1)
            A1 = A.change_ring(R1)
            for _ in range(5):
                subs = {x: K1(nth_prime(5 + _) + nth_prime(15 + i)) for i, x in enumerate(vars)}
                L1 = A1([R1([c(**subs) for c in p]) for p in L])
                fac1 = [R1([c(**subs) for c in p]) for p in fac]
                if any(p1.degree() != p.degree() for p, p1 in zip(fac, fac1)):
                    continue
                if any(L1[i].valuation() != L[i].valuation() for i in range(L.order() + 1)):
                    continue
                break
        else:
            L1, fac1 = L, fac

        bound = []
        for p, p1 in zip(fac, fac1):
            e = 0
            for j in range(r + 1):  # may be needed for inhomogeneous part
                if not L1[j].is_zero():
                    e = max(e, L1[j].valuation(p1) - j)
            for q, _ in L1.indicial_polynomial(p1).factor():  # contribution for homogeneous part
                if q.degree() == 1:
                    try:
                        e = max(e, ZZ(q[0] / q[1]))
                    except (TypeError, ValueError):
                        pass
            bound.append((p, e))

        return Factorization(bound)

    def _powerIndicator(self):
        return self.leading_coefficient()

    def finite_singularities(self):

        R = self.parent().base_ring().fraction_field().base()
        R = R.change_ring(R.base_ring().fraction_field())
        A = self.parent().change_ring(R)
        L = A(self.normalize())
        assert(not L.is_zero())

        for p in make_factor_iterator(R, False)(L.leading_coefficient()):
            pass

        raise NotImplementedError

    finite_singularities.__doc__ = UnivariateOreOperatorOverUnivariateRing.finite_singularities.__doc__

    def local_basis_monomials(self, point):
        r"""
        Leading monomials of the local basis of “regular” solutions
        (logarithmic series solutions) used in the definition of initial values
        and transition matrices.

        INPUT:

        ``point`` -- Point where the local basis should be computed. (May be an
        irregular singular point, but the output only covers regular
        solutions.)

        OUTPUT:

        A list of expressions of the form ``(x-point)^λ*log(x-point)^k/k!``
        where ``λ`` is a root of the :meth:`indicial polynomial <indicial_polynomial>`
        (over the algebraic numbers) of the operator at ``point``, and ``k`` is
        a nonnegative integer less than the multiplicity of that root.

        If ``point`` is an ordinary point, the output is ``[1, x, x^2, ...]``.

        More generally, a solution of the operator is characterized by the
        coefficients in its logarithmic power series expansion at ``point`` of
        the monomials returned by this method. The basis of solutions
        consisting of the local solutions in which exactly one of the monomials
        appears (with a coefficient equal to one), ordered as in the output of
        this method, is used in several functions of this package to specify
        vectors of “generalized initial values” at regular singular points.
        (The order is essentially that of asymptotic dominance as ``x`` tends
        to ``point``, with oscillating functions being ordered in an arbitrary
        but consistent way.) Note that this basis is not the usual Frobenius
        basis and may not coincide with the one computed by
        :meth:`generalized_series_solutions`.

        .. SEEALSO::

            :meth:`local_basis_expansions`,
            :meth:`numerical_solution`,
            :meth:`numerical_transition_matrix`

        EXAMPLES::

            sage: from ore_algebra import DifferentialOperators
            sage: Dops, x, Dx = DifferentialOperators()
            sage: ((x+1)*Dx^4+Dx-x).local_basis_monomials(0)
            [1, x, x^2, x^3]
            sage: ((x^2 + 1)*Dx^2 + 2*x*Dx).local_basis_monomials(i)
            [log(x - I), 1]
            sage: (4*x^2*Dx^2 + (-x^2+8*x-11)).local_basis_monomials(0)
            [x^(-1.232050807568878?), x^2.232050807568878?]
            sage: (x^3*Dx^4+3*x^2*Dx^3+x*Dx^2+x*Dx+1).local_basis_monomials(0)
            [1, 1/2*x*log(x)^2, x*log(x), x]

        A local basis whose elements all start with pure monomials (without
        logarithmic part) can nevertheless involve logarithms. In particular,
        the leading monomials are not enough to decide if a given solution is
        analytic::

            sage: dop = (x^2 - x)*Dx^2 + (x - 1)*Dx + 1
            sage: dop.local_basis_monomials(1)
            [1, x - 1]
            sage: dop.annihilator_of_composition(1 + x).generalized_series_solutions(3)
            [x*(1 - x + 5/6*x^2 + O(x^3)),
             (x - x^2 + O(x^3))*log(x) - 1 + 1/2*x^2 + O(x^3)]

        An irregular case::

            sage: dop = -x^3*Dx^2 + (-x^2-x)*Dx + 1
            sage: dop.local_basis_monomials(0)
            [x]

        TESTS::

            sage: ((x+1/3)*Dx^4+Dx-x).local_basis_monomials(-1/3)
            [1, x + 1/3, 1/9*(3*x + 1)^2, 1/27*(3*x + 1)^3]

            sage: ((x^2 - 2)^3*Dx^4+Dx-x).local_basis_monomials(sqrt(2))
            [1, (x - sqrt(2))^0.978..., (x - sqrt(2))^2.044...,
            (x - sqrt(2))^2.977...]

            sage: dop = (Dx^3 + ((24*x^2 - 4*x - 12)/(8*x^3 - 8*x))*Dx^2 +
            ....:   ((32*x^2 + 32*x - 16)/(32*x^4 + 32*x^3 - 32*x^2 - 32*x))*Dx)
            sage: dop.local_basis_monomials(0)
            [1, sqrt(x), x]
        """
        from .analytic.differential_operator import DifferentialOperator
        from .analytic.local_solutions import simplify_exponent
        from .analytic.path import Point
        dop = DifferentialOperator(self)
        struct = Point(point, dop).local_basis_structure(critical_monomials=False)
        x = SR(dop.base_ring().gen()) - point
        return [x**simplify_exponent(sol.valuation)
                * symbolic_log.log(x, hold=True)**sol.log_power
                / sol.log_power.factorial()
                for sol in struct]

    # TODO: Add a version that returns DFiniteFunction objects
    def local_basis_expansions(self, point, order=None, ring=None):
        r"""
        Generalized series expansions of the local basis of “regular” solutions
        (logarithmic series solutions) used in the definition of initial values
        and transition matrices.

        INPUT:

        * ``point`` -- Point where the local basis is to be computed. (May
          be an irregular singular point, but this method only computes regular
          solutions.)

        * ``order`` (optional) -- Number of terms to compute, starting from
          each “leftmost” valuation of a group of solutions with valuations
          differing by integers. (Thus, the absolute truncation order will be
          the same for all solutions in such a group, with some solutions
          having more actual coefficients computed that others.)

          The default is to choose the truncation order in such a way that the
          structure of the basis is apparent, and in particular that logarithmic
          terms appear if logarithms are involved at all in that basis. The
          corresponding order may be very large in some cases.

        * ``ring`` (optional) -- Ring into which to coerce the coefficients of the
          expansion

        OUTPUT:

        A list of ``sage.structure.formal_sum.FormalSum` objects. Each term of
        each sum is a monomial of the form ``dx^n*log(dx)^k``  for some ``dx``,
        ``n``, and ``k``, multiplied by a coefficient belonging to ``ring``.
        See below for examples of how to access these parameters.

        .. SEEALSO::

            :meth:`local_basis_monomials`,
            :meth:`numerical_solution`,
            :meth:`numerical_transition_matrix`

        EXAMPLES::

            sage: from ore_algebra import *
            sage: Dops, x, Dx = DifferentialOperators(QQ, 'x')

            sage: (Dx - 1).local_basis_expansions(0)
            [1 + x + 1/2*x^2 + 1/6*x^3]

            sage: from ore_algebra.examples import ssw
            sage: ssw.dop[1,0,0].local_basis_expansions(0)
            [t^(-4) + 24*t^(-2)*log(t) - 48*log(t) - 96*t^2*log(t) - 88*t^2,
             t^(-2),
             1 + 2*t^2]

            sage: dop = (x^2*(x^2-34*x+1)*Dx^3 + 3*x*(2*x^2-51*x+1)*Dx^2
            ....:     + (7*x^2-112*x+1)*Dx + (x-5))
            sage: dop.local_basis_expansions(0, order=3)
            [1/2*log(x)^2 + 5/2*x*log(x)^2 + 12*x*log(x) + 73/2*x^2*log(x)^2
            + 210*x^2*log(x) + 72*x^2,
            log(x) + 5*x*log(x) + 12*x + 73*x^2*log(x) + 210*x^2,
            1 + 5*x + 73*x^2]

            sage: roots = dop.leading_coefficient().roots(AA)
            sage: basis = dop.local_basis_expansions(roots[1][0], order=3)
            sage: basis
            [1 - (-239/12*a+169/6)*(x - 0.02943725152285942?)^2,
             (x - 0.02943725152285942?)^(1/2) - (-203/32*a+9)*(x - 0.02943725152285942?)^(3/2) + (-24031/160*a+1087523/5120)*(x - 0.02943725152285942?)^(5/2),
             (x - 0.02943725152285942?) - (-55/6*a+13)*(x - 0.02943725152285942?)^2]
            sage: basis[0].base_ring()
            Number Field in a with defining polynomial y^2 - 2 with a = -1.414...
            sage: RR(basis[0].base_ring().gen())
            -1.41421356237309
            sage: basis[0][-1]
            (239/12*a - 169/6, (x - 0.02943725152285942?)^2)

            sage: dop.local_basis_expansions(roots[1][0], order=3, ring=QQbar)
            [1 - 56.33308678393081?*(x - 0.02943725152285942?)^2,
             (x - 0.02943725152285942?)^(1/2) - 17.97141728630432?*(x - 0.02943725152285942?)^(3/2) + 424.8128741711741?*(x - 0.02943725152285942?)^(5/2),
             (x - 0.02943725152285942?) - 25.96362432175337?*(x - 0.02943725152285942?)^2]

        Programmatic access to the coefficients::

            sage: dop = ((x*Dx)^2 - 2)*(x*Dx)^3 + x^4
            sage: sol = dop.local_basis_expansions(0, ring=ComplexBallField(10))

            sage: sol[0]
            1.00*x^(-1.414213562373095?) + [-0.0123+/-6.03e-5]*x^2.585786437626905?
            sage: c, mon = sol[0][1]
            sage: c
            [-0.0123 +/- 6.03e-5]
            sage: mon.n, mon.k
            (2.585786437626905?, 0)
            sage: (mon.expo, mon.shift)
            (-1.414213562373095?, 4)
            sage: mon.expo + mon.shift == mon.n
            True

        Note that (in contrast with the definition of initial values) there is
        no ``1/k!`` in the monomial part::

            sage: sol[1]
            0.500*log(x)^2 + [-0.00056+/-3.06e-6]*x^4*log(x)^2
            + [0.00147+/-6.29e-6]*x^4*log(x) + [-0.00118+/-2.56e-6]*x^4
            sage: c, mon = sol[1][1]
            sage: c, mon.n, mon.k
            ([-0.00056 +/- 3.06e-6], 4, 2)

        The local basis at an irregular singular point has fewer elements than
        the order of the operator::

            sage: dop = -x^3*Dx^2+(-x^2-x)*Dx+1
            sage: dop.local_basis_expansions(0)
            [x - x^2 + 2*x^3 - 6*x^4 + 24*x^5]

        TESTS::

            sage: (4*x^2*Dx^2 + (-x^2+8*x-11)).local_basis_expansions(0, 2)
            [x^(-1.232050807568878?) + (-1/11*a+4/11)*x^(-0.2320508075688773?),
            x^2.232050807568878? - (-1/11*a)*x^3.232050807568878?]

            sage: ((27*x^2+4*x)*Dx^2 + (54*x+6)*Dx + 6).local_basis_expansions(0, 2)
            [x^(-1/2) + 3/8*x^(1/2), 1 - x]

            sage: dop = (Dx^3 + ((24*x^2 - 4*x - 12)/(8*x^3 - 8*x))*Dx^2 +
            ....:   ((32*x^2 + 32*x - 16)/(32*x^4 + 32*x^3 - 32*x^2 - 32*x))*Dx)
            sage: dop.local_basis_expansions(0, 3)
            [1, x^(1/2) - 1/6*x^(3/2) + 3/40*x^(5/2), x - 1/6*x^2]

        Thanks to Armin Straub for this example::

            sage: dop = ((81*x^4 + 14*x^3 + x^2)*Dx^3
            ....:       + (486*x^3 + 63*x^2 + 3*x)*Dx^2
            ....:       + (567*x^2 + 48*x + 1)*Dx + 81*x + 3)
            sage: dop.local_basis_expansions(QQbar((4*sqrt(2)*I-7)/81), 2)
            [1,
             (x + 0.0864197530864198? - 0.06983770678385654?*I)^(1/2) + (365/96*a^3+365/96*a+13/3)*(x + 0.0864197530864198? - 0.06983770678385654?*I)^(3/2),
             (x + 0.0864197530864198? - 0.06983770678385654?*I)]

        and to Emre Sertöz for this one::

            sage: ode = (Dx^2 + (2*x - 7/4)/(x^2 - 7/4*x + 3/4)*Dx
            ....:       + 3/16/(x^2 - 7/4*x + 3/4))
            sage: ode.local_basis_expansions(1, 3)[1]
            1 - 3/4*(x - 1) + 105/64*(x - 1)^2
        """
        from .analytic.differential_operator import DifferentialOperator
        from .analytic.local_solutions import (log_series, LocalExpansions,
                                               LogMonomial)
        from .analytic.path import Point
        dop = DifferentialOperator(self)
        mypoint = Point(point, dop)
        ldop = dop.shift(mypoint)
        sols = LocalExpansions(ldop, order).run()
        x = SR.var(dop.base_ring().variable_name())
        dx = x if point == 0 else x.add(-point, hold=True)
        if ring is None:
            cm = get_coercion_model()
            ring = cm.common_parent(
                dop.base_ring().base_ring(),
                mypoint.value.parent(),
                *(sol.leftmost.as_number_field_element() for sol in sols))
        res = [FormalSum(
                    [(c/ZZ(k).factorial(),
                      LogMonomial(dx, sol.leftmost.as_number_field_element(), n, k))
                        for n, vec in enumerate(sol.value)
                        for k, c in reversed(list(enumerate(vec)))
                        if not c.is_zero()],
                    FormalSums(ring),
                    reduce=False)
               for sol in sols]
        return res

    def numerical_solution(self, ini, path, eps=1e-16, post_transform=None, **kwds):
        r"""
        Evaluate an analytic solution of this operator at a point of its Riemann
        surface.

        INPUT:

        - ``ini`` (iterable) - initial values, in number equal to the order `r`
          of the operator
        - ``path`` - a path on the complex plane, specified as a list of
          vertices `z_0, \dots, z_n`
        - ``eps`` (floating-point number or ball, default 1e-16) - approximate
          target accuracy
        - ``post_transform`` (default: identity) - differential operator to be
          applied to the solutions, see examples below
        - see :class:`ore_algebra.analytic.context.Context` for advanced
          options

        OUTPUT:

        A real or complex ball *enclosing* the value at `z_n` of the solution `y`
        defined in the neighborhood of `z_0` by the initial values ``ini`` and
        extended by analytic continuation along ``path``.

        When `z_0` is an ordinary point, the initial values are defined as the
        first `r` coefficients of the power series expansion at `z_0` of the
        desired solution `f`. In other words, ``ini`` must be equal to

        .. math:: [f(z_0), f'(z_0), f''(z_0)/2, \dots, f^{(r-1)}(z_0)/(r-1)!].

        Generalized initial conditions at regular singular points are also
        supported. If `z_0` is a regular point, the entries of ``ini`` are
        interpreted as the coefficients of the monomials `(z-z_0)^n
        \log(z-z_0)^k/k!` returned by :meth:`local_basis_monomials` in the
        logarithmic series expansion of `f` at `z_0`. This definition reduces
        to the previous one when `z_0` is an ordinary point.

        The accuracy parameter ``eps`` is used as an indication of the
        *absolute* error the code should aim for. The diameter of the result
        will typically be of the order of magnitude of ``eps``, but this is not
        guaranteed to be the case. (It is a bug, however, if the returned ball
        does not contain the exact result.)

        See :mod:`ore_algebra.analytic` for more information, and
        :mod:`ore_algebra.examples` for additional examples.

        .. SEEALSO:: :meth:`numerical_transition_matrix`

        EXAMPLES:

        First a very simple example::

            sage: from ore_algebra import DifferentialOperators
            sage: Dops, x, Dx = DifferentialOperators()
            sage: (Dx - 1).numerical_solution(ini=[1], path=[0, 1], eps=1e-50)
            [2.7182818284590452353602874713526624977572470936999...]

        Evaluation points can be complex and can depend on symbolic constants::

            sage: (Dx - 1).numerical_solution([1], [0, i + pi])
            [12.5029695888765...] + [19.4722214188416...]*I

        They can even be real or complex balls. In this case, the result
        contains the image of the ball::

            sage: (Dx - 1).numerical_solution([1], [0, CBF(1+i).add_error(0.01)])
            [1.5 +/- 0.0693] + [2.3 +/- 0.0506]*I

        Here, we use a more complicated analytic continuation path in order to
        evaluate the branch of the complex arctangent function obtained by
        turning around its singularity at `i` once::

            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
            sage: dop.numerical_solution([0, 1], [0, i+1, 2*i, i-1, 0])
            [3.14159265358979...] + [+/- ...]*I

        In some cases, this method is also able to compute limits of solutions
        at regular singular points. This only works when all solutions of the
        differential equation tend to finite values at the evaluation point::

            sage: dop = (x - 1)^2*Dx^3 + Dx + 1
            sage: dop.local_basis_monomials(1)
            [1,
            (x - 1)^(1.500000000000000? - 0.866025403784439?*I),
            (x - 1)^(1.500000000000000? + 0.866025403784439?*I)]
            sage: dop.numerical_solution(ini=[1, 0, 0], path=[0, 1])
            [0.6898729110219401...] + [+/- ...]*I

            sage: dop = -(x+1)*(x-1)^3*Dx^2 + (x+3)*(x-1)^2*Dx - (x+3)*(x-1)
            sage: dop.local_basis_monomials(1)
            [x - 1, (x - 1)^2]
            sage: dop.numerical_solution([1,0], [0,1])
            0

            sage: (Dx*x*Dx).numerical_solution(ini=[1,0],path=[1,0])
            Traceback (most recent call last):
            ...
            ValueError: solution may not have a finite limit at evaluation
            point 0 (try using numerical_transition_matrix())

        To obtain the values of the solution at several points in a single run,
        enclose the corresponding points of the path in length-one lists. The
        output then changes to a list of (point, solution value) pairs::

            sage: (Dx - 1).numerical_solution([1], [[i/3] for i in range(4)])
            [(0, 1.00...), (1/3, [1.39...]), (2/3, [1.94...]), (1, [2.71...])]

            sage: (Dx - 1).numerical_solution([1], [0, [1]])
            [(1, [2.71828182845904...])]

        The ``post_transform`` parameter can be used to compute derivatives or
        linear combinations of derivatives of the solution. Here, we use this
        feature to evaluate the tenth derivative of the Airy `Ai` function::

            sage: ini = [1/(3^(2/3)*gamma(2/3)), -1/(3^(1/3)*gamma(1/3))]
            sage: (Dx^2-x).numerical_solution(ini, [0,2], post_transform=Dx^10)
            [2.34553207877...]
            sage: airy_ai(10, 2.)
            2.345532078777...

        A similar, slightly more complicated example::

            sage: (Dx^2 - x).numerical_solution(ini, [0, 2],
            ....:                               post_transform=1/x + x*Dx)
            [-0.08871870365567...]
            sage: t = SR.var('t')
            sage: (airy_ai(t)/t + t*airy_ai_prime(t))(t=2.)
            -0.08871870365567...

        Some notable examples of incorrect input::

            sage: (Dx - 1).numerical_solution([1], [])
            Traceback (most recent call last):
            ...
            ValueError: empty path

            sage: ((x - 1)*Dx + 1).numerical_solution([1], [0, 2])
            Traceback (most recent call last):
            ...
            ValueError: Step 0 --> 2 passes through or too close to singular
            point 1 (to compute the connection to a singular point, make it a
            vertex of the path)

            sage: Dops.zero().numerical_solution([], 1)
            Traceback (most recent call last):
            ...
            ValueError: operator must be nonzero

            sage: (Dx - 1).numerical_solution(ini=[], path=[0, 1])
            Traceback (most recent call last):
            ...
            ValueError: incorrect initial values: []

            sage: (Dx - 1).numerical_solution([1], ["a"])
            Traceback (most recent call last):
            ...
            TypeError: unexpected value for point: 'a'

        TESTS::

            sage: (Dx - 1).numerical_solution([1], [[0], 1])
            [(0, 1.0000000000000000)]
        """
        from .analytic import analytic_continuation as ancont
        from .analytic import local_solutions, utilities
        from .analytic.differential_operator import DifferentialOperator
        from .analytic.polynomial_root import PolynomialRoot
        dop = DifferentialOperator(self)
        post_transform = ancont.normalize_post_transform(dop, post_transform)
        post_mat = matrix(1, dop.order(),
                          lambda i, j: ZZ(j).factorial()*post_transform[j])
        ctx = ancont.Context(**kwds)
        sol = ancont.analytic_continuation(dop, path, eps, ctx, ini=ini,
                                           post=post_mat,
                                           return_local_bases=True)
        val = []
        asycst = local_solutions.sort_key_by_asympt(
            (PolynomialRoot.make(QQ.zero()), 0, ZZ.zero()))
        for sol_at_pt in sol:
            pt = sol_at_pt["point"]
            mat = sol_at_pt["value"]
            if dop.order() == 0:
                val.append((pt, mat.base_ring().zero()))
                continue
            asympt = local_solutions.sort_key_by_asympt(sol_at_pt["structure"][0])
            if asympt > asycst:
                val.append((pt, mat.base_ring().zero()))
            elif asympt == asycst:
                val.append((pt, mat[0][0]))
            else:
                raise ValueError("solution may not have a finite limit at "
                                 f"evaluation point {pt} "
                                 "(try using numerical_transition_matrix())")
        if isinstance(path, list) and any(isinstance(pt, list) for pt in path):
            return val
        else:
            assert len(val) == 1
            return val[0][1]

    def numerical_transition_matrix(self, path, eps=1e-16, **kwds):
        r"""
        Compute a transition matrix along a path drawn in the complex plane.

        INPUT:

        - ``path`` - a path on the complex plane, specified as a list of
          vertices `z_0, \dots, z_n`
        - ``eps`` (floating-point number or ball) - target accuracy
        - see :class:`ore_algebra.analytic.context.Context` for advanced
          options

        OUTPUT:

        When ``self`` is an operator of order `r`, this method returns an `r×r`
        matrix of real or complex balls. The returned matrix maps a vector of
        “initial values at `z_0`” (i.e., the coefficients of the decomposition
        of a solution in a certain canonical local basis at `z_0`) to “initial
        values at `z_n`” that define the same solution, extended by analytic
        continuation along the path ``path``.

        The “initial values” are the coefficients of the monomials returned by
        :meth:`local_basis_monomials` in the local logarithmic power series
        expansions of the solution at the corresponding point. When `z_i` is an
        ordinary point, the corresponding vector of initial values is simply

        .. math:: [f(z_i), f'(z_i), f''(z_i)/2, \dots, f^{(r-1)}(z_i)/(r-1)!].

        The accuracy parameter ``eps`` is used as an indication of the
        *absolute* error that the code should aim for. The diameter of each
        entry of the result will typically be of the order of magnitude of
        ``eps``, but this is not guaranteed to be the case. (It is a bug,
        however, if the returned ball does not contain the exact result.)

        See :mod:`ore_algebra.analytic` for more information, and
        :mod:`ore_algebra.examples` for additional examples.

        .. SEEALSO:: :meth:`numerical_solution`

        EXAMPLES:

        We can compute `\exp(1)` as the only entry of the transition matrix from
        `0` to `1` for the differential equation `y' = y`::

            sage: from ore_algebra import DifferentialOperators
            sage: Dops, x, Dx = DifferentialOperators()
            sage: (Dx - 1).numerical_transition_matrix([0, 1])
            [[2.7182818284590452 +/- 3.54e-17]]

        Now consider a second-order operator that annihilates `\arctan(x)` and the
        constants. A basis of solutions is formed of the constant `1`, of the
        form `1 + O(x^2)` as `x \to 0`, and the arctangent function, of the form
        `x + O(x^2)`. Accordingly, the entries of the transition matrix from the
        origin to `1 + i` are the values of these two functions and their first
        derivatives::

            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
            sage: dop.numerical_transition_matrix([0, 1+i], 1e-10)
            [ [1.00...] + [+/- ...]*I  [1.017221967...] + [0.4023594781...]*I]
            [ [+/- ...] + [+/- ...]*I  [0.200000000...] + [-0.400000000...]*I]

        By making loops around singular points, we can compute local monodromy
        matrices::

            sage: dop.numerical_transition_matrix([0, i + 1, 2*i, i - 1, 0])
            [ [1.00...] + [+/- ...]*I  [3.141592653589793...] + [+/-...]*I]
            [ [+/- ...] + [+/- ...]*I  [1.000000000000000...] + [+/-...]*I]

        Then we compute a connection matrix to the singularity itself::

            sage: dop.numerical_transition_matrix([0, i], 1e-10)
            [            ...           [+/-...] + [-0.50000000...]*I]
            [ ...1.000000...  [0.7853981634...] + [0.346573590...]*I]

        Note that a path that crosses the branch cut of the complex logarithm
        yields a different result::

            sage: dop.numerical_transition_matrix([0, i - 1, i], 1e-10)
            [     [+/-...] + [+/-...]*I         [+/-...] + [-0.5000000000...]*I]
            [ [1.00000...] + [+/-...]*I [-2.356194490...] + [0.3465735902...]*I]

        In general, if the operator has rational coefficients, its singular
        points are algebraic numbers. In connection problems such as the above,
        they need to be specified exactly. Here is a way to do it::

            sage: dop = (x^2 - 2)*Dx^2 + x + 1
            sage: dop.numerical_transition_matrix([0, 1, QQbar(sqrt(2))], 1e-10)
            [         [2.49388146...] + [+/-...]*I          [2.40894178...] + [+/-...]*I]
            [[-0.203541775...] + [6.68738570...]*I  [0.204372067...] + [6.45961849...]*I]

        The operator itself may be defined over a number field (with a complex
        embedding)::

            sage: K.<zeta7> = CyclotomicField(7)
            sage: (Dx - zeta7).numerical_transition_matrix([0, 1])
            [[1.32375209616333...] + [1.31434281345999...]*I]

        Some notable examples of incorrect input::

            sage: (Dx - 1).numerical_transition_matrix([])
            Traceback (most recent call last):
            ...
            ValueError: empty path

            sage: ((x - 1)*Dx + 1).numerical_transition_matrix([0, 2])
            Traceback (most recent call last):
            ...
            ValueError: Step 0 --> 2 passes through or too close to singular
            point 1 (to compute the connection to a singular point, make it a
            vertex of the path)

            sage: Dops.zero().numerical_transition_matrix([0, 1])
            Traceback (most recent call last):
            ...
            ValueError: operator must be nonzero
        """
        from .analytic import analytic_continuation as ancont
        from .analytic.differential_operator import DifferentialOperator
        dop = DifferentialOperator(self)
        ctx = ancont.Context(**kwds)
        sol = ancont.analytic_continuation(dop, path, eps, ctx)
        if isinstance(path, list) and any(isinstance(pt, list) for pt in path):
            return [(s["point"], s["value"]) for s in sol]
        else:
            assert len(sol) == 1
            return sol[0]["value"]

    def _initial_integral_basis(self, place=None):
        r"""
        
        TESTS::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = x*(x-1)*Dx^3 - 1
            sage: L._initial_integral_basis()
            [1, (x^2 - x)*Dx, (x^4 - 2*x^3 + x^2)*Dx^2]
            sage: L._initial_integral_basis(place=x)
            [1, x*Dx, x^2*Dx^2]

        The default place is also correct if the operator has a denominator or if the operator has no singularities:

            sage: L = x*Dx^2 - 1/(x-1)
            sage: L._initial_integral_basis()
            [1, (x^2 - x)*Dx]
            sage: L = Dx^2 - 1
            sage: L._initial_integral_basis()
            [1, Dx]

        """
        r = self.order()
        ore = self.parent()
        DD = ore.gen()
        if place is None:
            poly = (self.denominator()*self.leading_coefficient()).numerator()
            if poly.degree() > 0:
                place = poly.radical().monic()
            else:
                place = 1
        return [place**i * DD**i for i in range(r)]

    def _normalize_make_valuation_place_args(self, f, iota=None, prec=None, sols=None,
                                             infolevel=0, **kwargs):
        return (f,iota,prec, None if sols is None else tuple(sols))

    @cached_method(key=_normalize_make_valuation_place_args)
    def _make_valuation_place(self, f, iota=None, prec=None, sols=None, infolevel=0, **kwargs):
        r"""
        Compute value functions for the place ``f``.

        INPUT:

        - ``f`` - a place, that is an irreducible polynomial in the base ring of
          the ambient Ore algebra

        - ``iota`` (default: None) - a function allowing to compute the valuation of logarithmic
          terms of a series. ``iota(z,j)``, for z in ``\CC`` and j in ``\NN``,
          should be an element ``z+k`` in ``z + \ZZ``. Furthermore,
          ``iota(0,j)=j`` and ``iota(z1,j1)+iota(z2,j2)-iota(z1+z2,j1+j2) \geq
          0`` must hold.

          If ``iota`` is not provided, the function returns the element of
          ``z+\ZZ`` with real part between 0 (exclusive) and 1 (inclusive) if
          ``j=0``, and the element with real part between 0 (inclusive) and 1
          (exclusive) otherwise.

        - ``prec`` (default: None) - how many terms to compute in the series
          solutions to prepare the functions. If not provided, the default of
          :meth:``generalized_series_solutions`` is used.

        - ``sols`` (default: None) - if given, use those solutions at the place
          ``f`` instead of computing new ones. The validity of the solutions is
          not checked. The value of the parameter should be a tuple of
          generalized series solutions at the given place.

        - ``infolevel`` (default: 0) - verbosity flag

        OUTPUT:

        A tuple composed of ``f``, a suitable function for ``value_function`` at
        ``f`` and a suitable function for ``raise_value`` at ``f``.

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = x*(x-1)*Dx^2 - 1
            sage: f, v, rv = L._make_valuation_place(x-1)
            sage: (f, v, rv) # random
            (x - 1,
             <function UnivariateDifferentialOperatorOverUnivariateRing._make_valuation_place.<locals>.get_functions.<locals>.val_fct at 0x7ff14825a0c0>,
             <function UnivariateDifferentialOperatorOverUnivariateRing._make_valuation_place.<locals>.get_functions.<locals>.raise_val_fct at 0x7ff14825a020>)

        We verify that the functions behave as expected.

            sage: v(Dx)
            -1
            sage: v(x*Dx)
            -1
            sage: v((x-1)*Dx)
            0
            sage: rv([Ore(1), (x-1)*Dx])
            sage: rv([(x-1)*Dx, x*(x-1)*Dx])
            (-1, 1)

        If one already knows solutions to the operator and wants to use them for
        computing an integral basis, it is possible to do it by using this method.

            sage: from ore_algebra import *
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = x*(x-1)*Dx^2 - 1
            sage: f1,f2 = L.annihilator_of_composition(x+1).generalized_series_solutions()
            sage: L3 = L.symmetric_power(3)
            sage: sols = [f1^i * f2^(3-i) for i in range(4)]

        Now `sols` are a basis of solutions of `L3` at `-1`.
        We prepare the functions using this basis of solutions.
        
            sage: place = L3._make_valuation_place(x-1, sols=sols, infolevel=1)
            Preparing place at x - 1
            Using precomputed solutions
            sage: _ = L3._make_valuation_place(x-1, sols=None, infolevel=1)
            Preparing place at x - 1
            Computing generalized series solutions... done
            sage: f, v, rv = place

        This output can then be passed to the integral basis methods, skipping
        the computation of new solutions.
        
            sage: L3.global_integral_basis(places=[place], basis=[Ore(1),Dx,Dx^2, Dx^3])
            [1, (x - 1)*Dx, (x - 1)*Dx^2, (x - 1)*Dx^3 - 7*Dx + 3/(x - 1)]
            sage: L3.local_integral_basis(f, val_fct=v, raise_val_fct=rv)
            [1, (x - 1)*Dx, (x - 1)*Dx^2, (x - 1)*Dx^3 - 7*Dx + 3/(x - 1)]

        """

        print1 = print if infolevel >= 1 else lambda *a, **k: None
        
        print1("Preparing place at {}"
               .format(f if f.degree() < 10
                       else "{} + ... + {}".format(f[f.degree()]*f.monomials()[0], f[0])))

        r = self.order()
        ore = self.parent()
        base = ore.base_ring()
        f = f.numerator()

        if sols is None:
            need_sols = True
        else:
            print1("Using precomputed solutions")
            need_sols = False
            
        C = base.base_ring()
        if f.degree() > 1:
            if need_sols:
                print1("Computing field extension... ", end="", flush=True)
                FF = NumberField(f, "xi")
                xi = FF.gen()
                print1("done")
            else:
                FF = sols[0].base_ring()
                xi = FF.gen()
        else:
            FF = C
            xi = -f[0]/f[1]

        # Next lines because there is no change_ring() method for a fraction
        # field, so we need to proceed in two steps.
        if base.is_field():
            base = base.ring()
        x = base.gen()
        pol_ext = base.change_ring(FF)
        ore_ext = ore.change_ring(pol_ext.fraction_field())

        reloc = ore_ext([c(x=x+xi) for c in self.coefficients(sparse=False)])
        if need_sols:
            print1("Computing generalized series solutions... ", end="", flush=True)
            if prec is None:
                sols = reloc.generalized_series_solutions(exp=False)
            else:
                sols = reloc.generalized_series_solutions(prec, exp=False)
            print1("done")

        # if any(True for s in sols if s.ramification()>1):
        #     raise NotImplementedError("Some generalized series solutions have ramification")

        if len(sols) < r or any(not s.is_fuchsian(C) for s in sols):
            raise ValueError("The operator has non Fuchsian series solutions")

        # Capture the objects
        def get_functions(xi, sols, x, ore_ext):
            # In both functions the second argument `place` is ignored because
            # captured

            def val_fct(op,base=C, iota=None, infolevel=0, **kwargs):
                op = ore_ext([c(x=x+xi)
                              for c in op.coefficients(sparse=False)])
                vect = [op(s).valuation(base=C, iota=iota) for s in sols]
                if infolevel>=1:
                    print("Value function", vect)
                return min(vect)

            def raise_val_fct(ops, dim=None, base=C, iota=None,
                              infolevel=0, **kwargs):
                # TODO: Is it okay that we don't use dim?
                ops = [ore_ext([c(x=x+xi)
                                for c in op.coefficients(sparse=False)])
                       for op in ops]
                ss = [[op(s) for s in sols] for op in ops]
                if infolevel >= 2:
                    print(ss)
                cands = set()
                r = len(sols)
                for k in range(r):
                    for i in range(len(ops)):
                        for t in ss[i][k].non_integral_terms(
                                base=C,
                                iota=iota, cutoff=1):
                            cands.add(t)

                mtx = [[] for i in range(len(ops))]
                for t in cands:
                    if infolevel >= 2:
                        print(" [raise_val_fct] Processing term x^({}) log(x)^{}".format(t[1], t[0]))
                    for i in range(len(ops)):
                        for s in ss[i]:
                            mtx[i].append(s.coefficient(*t))
                    if infolevel >= 3:
                        print(" [raise_val_fct] Current matrix:\n{}".format(mtx))

                M = matrix(mtx)
                K = M.left_kernel().basis()
                if K:
                    return (1/K[0][-1])*K[0]
                else:
                    return None

            return val_fct, raise_val_fct

        val_fct, raise_val_fct = get_functions(xi, sols, x, ore_ext)
        return f, val_fct, raise_val_fct
    
    def find_candidate_places(self, infolevel=0, iota=None, prec=None, **kwargs):
        r"""

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = x*(x-1)*Dx^2 - 1
            sage: places = L.find_candidate_places()
            sage: places # random
            [(x - 1,
            <function UnivariateDifferentialOperatorOverUnivariateRing._make_valuation_place.<locals>.get_functions.<locals>.val_fct at 0x7f309097a5f0>,
            <function UnivariateDifferentialOperatorOverUnivariateRing._make_valuation_place.<locals>.get_functions.<locals>.raise_val_fct at 0x7f308324a440>),
            (x,
            <function UnivariateDifferentialOperatorOverUnivariateRing._make_valuation_place.<locals>.get_functions.<locals>.val_fct at 0x7f3083249900>,
            <function UnivariateDifferentialOperatorOverUnivariateRing._make_valuation_place.<locals>.get_functions.<locals>.raise_val_fct at 0x7f3083249fc0>)]
            sage: [p[0] for p in places]
            [x - 1, x]
            sage: f,v,rv = places[0]; f
            x - 1
            sage: v(Dx)
            -1
            sage: v((x-1)*Dx)
            0
            sage: rv([(x-1)*Dx, x*(x-1)*Dx])
            (-1, 1)

        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = x*(x-1)*Dx^2 - 1
            sage: [p[0] for p in L.find_candidate_places()]
            [x - 1, x]
            sage: L = x*Dx^2 - 1/(x-1)
            sage: [p[0] for p in L.find_candidate_places()]
            [x - 1, x]
            sage: L = Dx^2 - 1
            sage: [p[0] for p in L.find_candidate_places()]
            []

        
        """
        lr = (self.leading_coefficient()*self.denominator()).numerator().monic()
        fact = list(lr.factor())
        places = []
        for f, m in fact:
            places.append(self._make_valuation_place(f,
                                                     prec=m+1 if prec is None else prec,
                                                     infolevel=infolevel,
                                                     iota=None))
        return places

    def value_function(self, op, place, iota=None, **kwargs):
        val = self._make_valuation_place(place, iota=iota, **kwargs)[1]
        return val(op)

    def raise_value(self, basis, place, dim=None, iota=None, **kwargs):
        fct = self._make_valuation_place(place, iota=iota, **kwargs)[2]
        return fct(basis, dim)

    def _normalize_local_integral_basis_args(self, basis=None, iota=None, sols=None, infolevel=0, prec=None,
                                             **kwargs):
        return (prec,
                None if sols is None else tuple(sols),
                None if basis is None else tuple(basis))
    
    @cached_method(key=_normalize_local_integral_basis_args)
    def local_integral_basis_at_infinity(self, basis=None, iota=None,
                                         sols=None,
                                         infolevel=0, **val_kwargs):
        r"""
        Compute a local integral basis at infinity

        A basis of the quotient algebra by self is local at infinity if the
        change of variable 1/x makes it local at 0.

        INPUT:

        - ``basis`` (default: None) an initial basis for the
          computation

        - ``iota`` (default: None) a function used to filter terms of
          generalized series solutions which are to be considered
          integral. For the conditions that this function must satisfy, see
          :meth:`ContinuousGeneralizedSeries.valuation`

        -- ``infolevel`` (default: 0) verbosity level to use in the computations

        -- ``sols`` (default: None) if given, a basis of solutions at infinity
           to use for computing the valuations.

        OUTPUT:

        A local integral basis at infinity.

        EXAMPLES:

        Example 6 in https://arxiv.org/abs/2302.06396::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = (x^2-x)*Dx^2 + (31/24*x - 5/6)*Dx + 1/48
            sage: L.local_integral_basis_at_infinity()
            [1, -x*Dx]

        Example 15::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = (x^2-x)*Dx^2 + (49/6*x - 7/3)*Dx + 12
            sage: L.local_integral_basis_at_infinity()
            [x^2, -x^5*Dx - 8/3*x^4 + 64/15*x^3]

        Example use of the optional parameter ``sols``:

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = (x^2-x)*Dx^2 + (49/6*x - 7/3)*Dx + 12
            sage: f1, f2 = L.annihilator_of_composition(1/x).generalized_series_solutions()
            sage: d = 2
            sage: L2 = L.symmetric_power(d)
            sage: sols = [f1^i * f2^(d-i) for i in range(d+1)]
            sage: L2.local_integral_basis_at_infinity(sols=sols)
            [x^5,
            -x^8*Dx - 16/3*x^7 + 128/15*x^6,
            x^11*Dx^2 + (27/2*x^10 - 27/10*x^9)*Dx + 344/9*x^9 - 968/45*x^8 + 6592/225*x^7 + 128/3*x^6]

        """
        x = self.base_ring().gen()
        place = x.numerator()
        Linf, conv = self.annihilator_of_composition(1/x, with_transform=True)
        f, v, rv = Linf._make_valuation_place(place, iota=iota, sols=sols, **val_kwargs)
        if basis:
            basis = [conv(b) for b in basis]
        wwinf = Linf.local_integral_basis(f, val_fct=v, raise_val_fct=rv,
                                          basis=basis, infolevel=infolevel, **val_kwargs)
        vv = [conv(w) for w in wwinf]
        return vv

    def _normalize_basis_at_infinity(self, uu, vv, infolevel=0, solver=None, modulus=None):
        r"""
        Compute an integral basis normal at infinity

        An integral basis `w_1,...,w_r` is called normal at infinity if there
        exist integers `\tau_1`, ..., `\tau_r` such that `\{x^{\tau_1}w_1, ...,
        x^{\tau_r}w_r\}` is a local integral basis at infinity.

        INPUT:

        - ``uu`` a global integral basis
        - ``vv`` a local integral basis at infinity
        - ``modulus`` (default: None) if given, perform the computations modulo
          that number

        OUTPUT:

        A tuple composed of an integral basis local at infinity and the suitable
        values of `\tau`.

        EXAMPLES:

        Example 15 in https://arxiv.org/abs/2302.06396::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = (x^2-x)*Dx^2 + (49/6*x - 7/3)*Dx + 12
            sage: uu = L.global_integral_basis(); uu
            [x^7 - 5*x^6 + 10*x^5 - 10*x^4 + 5*x^3 - x^2,
             (x^2 - x)*Dx + 1085/46*x^6 - 5355/46*x^5 + 5250/23*x^4 - 5075/23*x^3 + 4725/46*x^2 - 34/3*x - 4/3]
            sage: vv = L.local_integral_basis_at_infinity(); vv
            [x^2, -x^5*Dx - 8/3*x^4 + 64/15*x^3]
            sage: L._normalize_basis_at_infinity(uu,vv)
            ([(-23/35*x^4 + 23/35*x^3)*Dx - 641/210*x^3 - 13/105*x^2,
              (23/301*x^6 - 1104/1505*x^5 - 279427/12040*x^4 + 305877/12040*x^3 - 9867/6020*x^2 + 299/3010*x)*Dx + 184/903*x^5 - 1472/645*x^4 - 2708719/24080*x^3 + 10751/2580*x^2 - 598/645*x + 598/4515],
             [-1, -1])

        """
        if infolevel >= 1:
            print("Normalizing the basis")

        r = self.order()
        x = self.base_ring().gen()
        from sage.matrix.constructor import matrix
        ww = uu[:]

        if modulus:
            K = GF(modulus)
            Dif = self.parent().change_constant_ring(K)
            Pol = Dif.base_ring()
            x = Pol.gen()
            ww = [w.change_constant_ring(K) for w in ww]
            vv = [v.change_constant_ring(K) for v in vv]
            if solver is None:
                solver = nullspace.sage_native
        else:
            K = QQ
            Pol = self.base_ring()
            if solver is None:
                solver = nullspace.cra(nullspace.sage_native)
        
        def pad_list(ll, d):
            # add 0s to reach length d
            return ll+[0]*(d - len(ll))

        def tau_value(f):
            # largest t st x^t f can be evaluated at infinity
            if f == 0:
                return infinity
            if f in QQ:
                return 0
            else:
                num = f.numerator().degree()
                den = f.denominator().degree()
                return den - num

        def eval_inf(f):
            # value of f at infinity
            if f in K:
                return f
            else:
                if f.denominator().degree() > f.numerator().degree():
                    return 0
                elif f.denominator().degree() < f.numerator().degree():
                    raise ZeroDivisionError
                else:
                    # casting in Pol for the nullspace solver which expects
                    # polynomial coefficients
                    return Pol(f.numerator().leading_coefficient() / f.denominator().leading_coefficient())

        D_to_vv = matrix([pad_list(b.coefficients(sparse=False), r)
                          for b in vv]).inverse()

        first = True
        while first or B.determinant() == 0:
            if first:
                first = False
            else:
                import time
                t = time.perf_counter()
                # a = B.kernel()
                a = solver(B.transpose())
                tt = time.perf_counter() - t
                if infolevel >= 1 and modulus is None:
                    print(f"max coef size={max([len(str(c)) for r in B for c in r])} ; kernel time={tt}")
                # a = a.basis()[0] # breaking for profiling
                a = a[0]
                l = min([i for i in range(r) if a[i] != 0],
                        key=lambda i: tau[i])
                ww[l] = sum(a[i]*x**(tau[i]-tau[l])*ww[i] for i in range(r))

            ww_to_D = matrix([pad_list(b.coefficients(sparse=False), r)
                              for b in ww])
            mm = ww_to_D * D_to_vv
            # print(mm)
            tau = [min(tau_value(m) for m in row) for row in mm.rows()]

            B = matrix([[eval_inf(x**tau[i]*mm[i,j]) for j in range(r)]
                    for i in range(r)])
            if infolevel >= 1: print(f"{tau=}")
            if infolevel >= 2: print(f"{B=}")
                    
        #breakpoint()
        return ww, tau

    def normal_global_integral_basis(self, basis=None, iota=None, infolevel=0, **val_kwargs):
        r"""
        Compute a normal global integral basis

        An global integral basis `w_1,...,w_r` is called normal at infinity if
        there exist integers `\tau_1`, ..., `\tau_r` such that `\{x^{\tau_1}w_1,
        ..., x^{\tau_r}w_r\}` is a local integral basis at infinity.

        INPUT:

        - ``basis`` (default: None) an initial basis for the
          computation

        - ``iota`` (default: None) a function used to filter terms of
          generalized series solutions which are to be considered
          integral. For the conditions that this function must satisfy, see
          :meth:`ContinuousGeneralizedSeries.valuation`

        -- ``infolevel`` (default: 0) verbosity level to use in the computations

        OUTPUT:

        A normal global integral basis.

        EXAMPLES:

        Example 6 in https://arxiv.org/abs/2302.06396::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = (x^2-x)*Dx^2 + (31/24*x - 5/6)*Dx + 1/48
            sage: L.normal_global_integral_basis()
            [1, (x^2 - x)*Dx]

        Example 15::

            sage: L = (x^2-x)*Dx^2 + (49/6*x - 7/3)*Dx + 12
            sage: L.normal_global_integral_basis()
            [(-23/35*x^4 + 23/35*x^3)*Dx - 641/210*x^3 - 13/105*x^2,
             (23/301*x^6 - 1104/1505*x^5 - 279427/12040*x^4 + 305877/12040*x^3 - 9867/6020*x^2 + 299/3010*x)*Dx + 184/903*x^5 - 1472/645*x^4 - 2708719/24080*x^3 + 10751/2580*x^2 - 598/645*x + 598/4515]

        """
        ww = self.global_integral_basis(basis=basis, iota=iota, infolevel=infolevel, **val_kwargs)
        vv = self.local_integral_basis_at_infinity(iota=iota, infolevel=infolevel, **val_kwargs)

        ww, _ = self._normalize_basis_at_infinity(ww, vv, infolevel=infolevel)
        return ww

    def pseudoconstants(self, iota=None, infolevel=0, solver=None, **val_kwargs):
        r"""
        Compute pseudoconstants.

        A pseudoconstant is an operator which is integral at all points
        including infinity, and which is not a constant.

        INPUT:

        - ``iota`` (default: None) a function used to filter terms of
          generalized series solutions which are to be considered
          integral. For the conditions that this function must satisfy, see
          :meth:`ContinuousGeneralizedSeries.valuation`

        -- ``infolevel`` (default: 0) verbosity level to use in the computations

        OUTPUT:

        A basis of the vector space of pseudoconstants.

        EXAMPLES:

        Example 6 in https://arxiv.org/abs/2302.06396::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Dx> = OreAlgebra(Pol)
            sage: L = (x^2-x)*Dx^2 + (31/24*x - 5/6)*Dx + 1/48
            sage: L.pseudoconstants()
            [1]

        Example 15::

            sage: L = (x^2-x)*Dx^2 + (49/6*x - 7/3)*Dx + 12
            sage: L.pseudoconstants()
            []
            sage: L2 = L.symmetric_power(2)
            sage: pc = L2.pseudoconstants() # long time (3 seconds)
            sage: [p.order() for p in pc] # long time (previous)
            [2]

        """
        ww = self.global_integral_basis(iota=iota, infolevel=infolevel, **val_kwargs)
        vv = self.local_integral_basis_at_infinity(iota=iota, infolevel=infolevel, **val_kwargs)

        ww, tau = self._normalize_basis_at_infinity(ww, vv, solver=solver, infolevel=infolevel)
        x = self.base_ring().gen()
        res = []
        for i in range(len(ww)):
            if tau[i] >= 0:
                res.extend(x**j * ww[i] for j in range(tau[i] + 1))
        Dx = self.parent().gen()
        return [r for r in res if (Dx * r).quo_rem(self)[1] != 0]

    def is_fuchsian(self):
        r"""
        Test if this operator is Fuchsian (i.e. regular at each point of the
        Riemann sphere).

        EXAMPLES::

            sage: from ore_algebra.examples import fcc
            sage: fcc.dop4.is_fuchsian()
            True
        """
        coeffs = self.coefficients()
        fac = coeffs.pop().factor()
        for f, m in fac:
            for k, ak in enumerate(coeffs):
                mk = valuation(ak, f)
                if mk - m < k - self.order():
                    return False

        dop = self.annihilator_of_composition(1/self.base_ring().gen())
        for k, frac in enumerate(dop.monic().coefficients()[:-1]):
            d = (self.base_ring().gen()**(self.order() - k)*frac).denominator()
            if d(0) == 0:
                return False

        return True

    def factor(self, *, verbose=False):
        r"""
        Decompose this operator as a product of irreducible operators.

        This method computes a decomposition of this operator as a product of
        irreducible operators (potentially introducing algebraic extensions of
        the field of constants).

        The termination of this method is currently not guaranteed if the
        operator is not Fuchsian.

        .. SEEALSO::

            :meth:`right_factor`

        INPUT:

        - ``verbose`` (boolean, default: ``False``) -- if set to ``True``,
          this method prints some messages about the progress of the
          computation.

        OUTPUT:

        A list of irreducible operators such that the product of its elements
        is equal to the operator ``self``.

        ALGORITHM:

        Seminumeric algorithm as described in:

        - Around the Numeric-Symbolic Computation of Differential Galois Groups,
          van der Hoeven, 2007
        - Symbolic-Numeric Factorization of Differential Operators, Chyzak,
          Goyer, Mezzarobba, 2022

        EXAMPLES:

        Reducible case::

            sage: from ore_algebra.examples import ssw
            sage: ssw.dop[13,0,0].factor()
            [(2688*t^9 + 2048*t^7 - 594*t^5 + 45*t^3 - t)*Dt^2 + (24192*t^8 + 8192*t^6 - 2330*t^4 + 167*t^2 - 3)*Dt + 40320*t^7 - 672*t^3 + 48*t,
             t*Dt + 1,
             t*Dt + 2]

        Irreducible case::

            sage: from ore_algebra.examples import fcc
            sage: fcc.dop4.factor() == [fcc.dop4] # irreducible case
            True

        Case that requires an algebraic extension::

            sage: from ore_algebra import DifferentialOperators
            sage: Diffops, z, Dz = DifferentialOperators(QQ, 'z')
            sage: dop = 1 + z*Dz + z^2*Dz^2 + z^3*Dz^3
            sage: dop.factor()
            [z*Dz - 20/28181*a0^5 + 701/56362*a0^4 + 334/28181*a0^3 + 2382/28181*a0^2 + 20005/56362*a0 - 14161/28181,
             z*Dz + 20/84543*a0^5 - 701/169086*a0^4 - 334/84543*a0^3 - 794/28181*a0^2 - 76367/169086*a0 - 70382/84543,
             z*Dz + 40/84543*a0^5 - 701/84543*a0^4 - 668/84543*a0^3 - 1588/28181*a0^2 + 8176/84543*a0 - 56221/84543]
            sage: _[0].parent()
            Univariate Ore algebra in Dz over Fraction Field of Univariate Polynomial Ring in z over Number Field in a0 with defining polynomial y^6 + 2*y^5 + 11*y^4 + 48*y^3 + 63*y^2 + 190*y + 1108 with a0 = -2.883024910498311? - 1.202820819285479?*I
        """
        from .analytic.factorization import factor
        fac = factor(self, verbose=verbose)
        return fac

    def right_factor(self, *, verbose=False):
        r"""
        Find a right-hand factor of this operator.

        The termination of this method is currently not guaranteed if the
        operator is not Fuchsian.

        .. SEEALSO::

            :meth:`is_provably_irreducible`,
            :meth:`factor`

        INPUT:

        - ``verbose`` (boolean, default: ``False``) -- if set to ``True``, this
          method prints some messages about the progress of the computation.

        OUTPUT:

        - ``None`` if the operator is irreducible
        - a proper right-hand factor (potentially with a larger field of
          constants) otherwise.

        EXAMPLES:

        Reducible case with a right-hand factor coming from a rational solution::

            sage: from ore_algebra.analytic.examples.facto import hypergeo_dop
            sage: dop = hypergeo_dop(1,1,1); dop
            (-z^2 + z)*Dz^2 + (-3*z + 1)*Dz - 1
            sage: dop.right_factor().monic()
            Dz + 1/(z - 1)

        Reducible case with a right-hand factor coming from monodromy::

            sage: from ore_algebra.analytic.examples.facto import hypergeo_dop
            sage: dop = hypergeo_dop(1/2,1/3,1/3); dop
            (-z^2 + z)*Dz^2 + (-11/6*z + 1/3)*Dz - 1/6
            sage: dop.right_factor().monic()
            Dz + 1/2/(z - 1)

        Reducible case without rational solution, without monodromy (non Fuchsian)::

            sage: from ore_algebra.examples.stdfun import dawson
            sage: dawson.dop.right_factor()
            1/2*Dx + x

        Irreducible case::

            sage: from ore_algebra.examples import fcc
            sage: dop = fcc.dop4; dop
            (9*z^10 + 186*z^9 + 1393*z^8 + 4608*z^7 + 6156*z^6 - 256*z^5 - 7488*z^4 - 4608*z^3)*Dz^4 + (126*z^9 + 2304*z^8 + 15322*z^7 + 46152*z^6 + 61416*z^5 + 15584*z^4 - 39168*z^3 - 27648*z^2)*Dz^3 + (486*z^8 + 7716*z^7 + 44592*z^6 + 119388*z^5 + 151716*z^4 + 66480*z^3 - 31488*z^2 - 32256*z)*Dz^2 + (540*z^7 + 7248*z^6 + 35268*z^5 + 80808*z^4 + 91596*z^3 + 44592*z^2 + 2688*z - 4608)*Dz + 108*z^6 + 1176*z^5 + 4584*z^4 + 8424*z^3 + 7584*z^2 + 3072*z
            sage: dop.right_factor() is None # irreducible operator
            True

        Case that requires an algebraic extension::

            sage: from ore_algebra import DifferentialOperators
            sage: Diffops, z, Dz = DifferentialOperators(QQ, 'z')
            sage: dop = 1 + z*Dz + z^2*Dz^2 + z^3*Dz^3
            sage: dop.right_factor()
            z*Dz + a - 1
            sage: _.parent()
            Univariate Ore algebra in Dz over Fraction Field of Univariate Polynomial Ring in z over Number Field in a with defining polynomial y^3 - y^2 + y - 2 with a = -0.1766049820996622? - 1.202820819285479?*I
        """
        from .analytic.factorization import right_factor
        rfac = right_factor(self, verbose=verbose)
        return rfac

    def is_provably_irreducible(self, prec=None, max_prec=100000, *, verbose=False):
        r"""
        Attempt to prove that this operator is irreducible.

        If the operator is Fuchsian and irreducible then this method succeeds to
        prove the irreducibility when the permitted precision is large enough.

        .. WARNING::

            Unlike :meth:`right_factor`, this method cannot conclude when the
            operator is reducible. However, it is faster.

        .. SEEALSO::

            :meth:`is_provably_minimal_annihilator`

        INPUT:

        - ``prec`` (integer, optional) -- initial working precision
        - ``max_prec`` (integer, default: 100000) -- maximum working precision
        - ``verbose`` (boolean, default: ``False``) -- if set to ``True``, this
          method prints some messages about the progress of the computation.

        OUTPUT:

        - ``True`` if the method could verify that the operator is irreducible
        - ``False`` if it reached the precision limit without being able to
          conclude

        EXAMPLES::

            sage: from ore_algebra import DifferentialOperators
            sage: Diffops, z, Dz = DifferentialOperators(QQ, 'z')
            sage: red_dop = z^2*Dz^2 - 2*z*Dz + 2 # reducible operator
            sage: irred_dop = (z^2 - 1)*Dz^2 + Dz + 1 # irreducible operator
            sage: red_dop.is_provably_irreducible()
            False
            sage: irred_dop.is_provably_irreducible()
            True
            sage: irred_dop.is_provably_irreducible(max_prec=50) # insufficient precision
            False

        An example coming from Face-Centered Cubic Lattices
        (see http://www.koutschan.de/data/fcc/)::

            sage: from ore_algebra.examples import ssw, fcc
            sage: fcc.dop4.is_provably_irreducible()
            True
        """
        from .analytic.factorization import is_provably_irreducible
        return is_provably_irreducible(self, verbose=verbose, prec=prec, max_prec=max_prec)

    def is_provably_minimal_annihilator(self, initial_conditions, prec=None, max_prec=100000, *, verbose=False):
        r"""
        Attempt to prove that this operator is the minimal annihilator of a
        given solution.

        The initial conditions are the coefficients of the monomials returned
        by ``self.local_basis_monomials(0)`` (see :meth:`local_basis_monomials`).
        If 0 is an ordinary point, this is simply
        :math:`[f(0), f'(0), f''(0)/2, ..., f^{(r-1)}(0)/(r-1)!]`.

        If the operator is Fuchsian and minimal for the given solution then this
        method succeeds to prove the minimality when the permitted precision is
        large enough.

        .. SEEALSO::

            :meth:`is_provably_irreducible`

        INPUT:

        - ``initial_conditions`` -- list of complex numbers
        - ``prec`` (integer, optional) -- initial working precision
        - ``max_prec`` (integer, default: 100000) -- maximum working precision
        - ``verbose`` (boolean, default: ``False``) -- if set to ``True``, this
          function prints some messages about the progress of the computation.

        OUTPUT:

        - ``True`` if the method could verify minimality
        - ``False`` if it reached the precision limit without being able to
          conclude

        EXAMPLES::

            sage: from ore_algebra import DifferentialOperators
            sage: Diffops, z, Dz = DifferentialOperators(QQ, 'z')
            sage: dop = Dz*z*Dz # annihilator of 1 and log
            sage: dop.local_basis_monomials(0)
            [log(z), 1]
            sage: dop.is_provably_minimal_annihilator([0, 1]) # not minimal for 1
            False
            sage: dop.is_provably_minimal_annihilator([1, 0]) # minimal for log
            True
            sage: dop.is_provably_minimal_annihilator([1, 0], max_prec=50) # insufficient precision
            False

        A nontrivial example::

            sage: from ore_algebra.analytic.examples.facto import beukers_vlasenko_dops
            sage: dop = beukers_vlasenko_dops[1]; dop
            (16*z^3 - z)*Dz^2 + (48*z^2 - 1)*Dz + 16*z
            sage: dop.is_provably_minimal_annihilator([0, 1])
            True
        """
        from .analytic.factorization import is_provably_minimal_annihilator
        return is_provably_minimal_annihilator(self, initial_conditions, verbose=verbose, prec=prec, max_prec=max_prec)

    def _valuation_bound(self):

        ''' 
        INPUT:

        OUTPUT:
        
        The value of max(Z_{this operator}) as decribe in the research paper 'Minimisation of differential equations and algebraic values of E-functioncs'
        written by Alin BOSTIAN, Tanguy RIVOAL, and Bruno SALVY, on page 7.

        EXAMPLES::
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: dop = (1) * Dx^5 + ((-5128108531200*x^5+89585591787360*x^4+28064493964392*x^3-4948326183852076*x^2+4892914911799558*x+41370383855939061)/(44980531200*x^6-381397420800*x^5-4676947564608*x^4+40304971055712*x^3+86646529572240*x^2-834609791595624*x+241535292878880)) * Dx^4 + ((35056170970573399621632000*x^10-434789329773835743178752000*x^9-3305648457917553664136033280*x^8+22792665869942700841571013120*x^7+465108882635341293309549139392*x^6-1753053890698517839871570697408*x^5-18538013647205533683445279361392*x^4+72414214947553729513331730338048*x^3+118709159502187973554538821380836*x^2-762208621587711835892072253657660*x+2377928695823413036816695548892219)/(7081368654619607040000*x^12-120088210101257502720000*x^11-963477126879176164147200*x^10+25177003225114181820825600*x^9-3765200727574963366373376*x^8-1813645336239967268370511872*x^7+5153305346717529650097867264*x^6+51125138126663956004482043904*x^5-217103325945909740110961421696*x^4-438066783002485834276994006400*x^3+2584504629075704917655487861216*x^2-1411114043168410296798390147840*x+204187541971302188734225190400)) * Dx^3 + ((-612051226940523129584311228830842880000*x^15+10324427076269127450956438437201182720000*x^14-49933317436012214710964939439108115660800*x^13-462398619437540465245364797677913677004800*x^12+6728602980027558925980237222783478279987200*x^11+39337236081028039200696445567476074225817600*x^10-425925566835883335355767099081097720627201536*x^9-4654928501598374372148140815533663982807253760*x^8+30991396791188548655433440466850885639347772416*x^7+140202857718924018262369373404786637638641722624*x^6-1097502990502246490552626498705068161802477547968*x^5+269110712624929497288604629812645050570860646176*x^4+11170125853205024119056722640122174960826359768608*x^3-21918377244998160863069183799494910968611942069320*x^2-82167839593933099911499431801722354136152862768750*x+223864649760312097807101767749471466220511235629185)/(6051950750448565913294733312000000*x^18-153946497214535395419434778624000000*x^17-582455815717350862498692323082240000*x^16+44593062523523075425683017359687680000*x^15-180352724044835623060254452437116518400*x^14-4807808272930945609517250107542221619200*x^13+37512485557684305686403452720087864573952*x^12+220805995655277560717674604814220154044416*x^11-2812703173665757400762154893370006678011904*x^10-2208371167862206282964315679499280790650880*x^9+101053414584623222279252066020215629485817856*x^8-153139365772647315166132372068315121826537472*x^7-1699218781212162993045571377301772337982119936*x^6+5059288353965785374429601244051420058637584384*x^5+9106418471846965156647320437717599070640401920*x^4-45161096585432447272206363982560354021840804096*x^3+34573745616304558247197011575621643533032258560*x^2-9713764545512201149476460727450861018788147200*x+937051457292884456614066928443353210636288000)) * Dx^2 + ((521036245868079542239645442824948649664970752000000*x^20-3803417047856570661383091369316039458300100608000000*x^19-150005275536183381844529548193286337409733812551680000*x^18+706215589918660855979489135382024567450053294161920000*x^17+29945411254467973534757166508835548458582357149614080000*x^16-158785064762488030567462913700518834911010734070169600000*x^15-2935163609060241903287459233437043451241867566050236825600*x^14+17388956102427859095438322433303704399674850388319569510400*x^13+157170824736711889146885451092209710896642072825630182117376*x^12-907138337185034657457300293109193654901134869579601258635264*x^11-5612881416903653630120646027424243809633283399502607205963776*x^10+25092656919740490884289848422758823435481824851127737760094208*x^9+165610407678721264241040109318670973344212492227267787785123584*x^8-517681023584445900530512597617644126542912712700056016023124992*x^7-3237555231691775470650174671463380196925217704131319022064037632*x^6+10615767254383484776344206998834323555069981936031440878264120320*x^5+11112457812946964974476479210607096273510464025608077073260278896*x^4-52222072933934061057702129518111071764278742354910855233770300640*x^3+49212230692554838263617438095595516789875307271433564209952812840*x^2+187749938002028739453991746311705923381995225556788785768962128600*x-388071097517798585430870130161305856442539915150489771008864152695)/(544439919102830266116420493072190668800000000*x^24-18465587256237659859115261723365133516800000000*x^23+8421937327768909839122505021080293343232000000*x^22+6383780362084631377023215900731365012799488000000*x^21-56152707277658288981077058047717984839212728320000*x^20-795793866557554794944564047109738312490056417280000*x^19+12482637536789640763530120395889351870901522740019200*x^18+30308671066600432940678362224613593038602067823820800*x^17-1265879298491201494467983595848414824415938913482309632*x^16+2394761309683901384356132910218154391516120901214011392*x^15+69323113717835713711961624438351780098662059518992580608*x^14-323609510907769734120835812584194796795762142173863084032*x^13-1997183774118448443388899822017157189059903177442311798784*x^12+15748642449231153747743589892267023264509496864340388610048*x^11+20349160474003658381980578082260798864654161516819647561728*x^10-391592176440181459741211471966414826379958689438057529081856*x^9+370183740082324528972410232075534260731288737232339401900032*x^8+4768385190401304751226136279041594515474007825499576870764544*x^7-11644165359112475408542605454239801014889110985455773678534656*x^6-17705588849326635643393631190566290212011824959316712257781760*x^5+84982428789318175938789064253613383174957329088726400481579008*x^4-81134962938168953596270025612380702618361893070899500290211840*x^3+33078350066195905935762193325462086791592574428486517912371200*x^2-6256578571884720474563455827155327829243800939422691229696000*x+452661996359636323072220556196159130520935195280233594880000)) * Dx^1 + ((-1711556981257167358660943961493854380165831994675165344563200000*x^25+15878842159232600956279416531888462454645289660596773244108800000*x^24+160843957999504169182957780358165864441505549047731024966975488000*x^23-4943517621097874961968163167971715772024265865910388382112940032000*x^22+65204179307964560949247804644939130703867015047091625498362511360000*x^21+6425085796080511073532087719492319428606636646513316645425905664000*x^20-5367063561439336020382187762247501608922702729856326050748463841280000*x^19+1139946794760929015098098726866672333260860317955416414415224832000000*x^18+13786322014617093553495688839262406672424029522072784913847872520192000*x^17+4484324041321078066689066293667961616167698208929424202465060770529280000*x^16+2346756292288342786913914602538081499436830984628539409010432887954898944*x^15-492216116472357807205800651901520699492107820689939262120879990943867125760*x^14+909976155407779426144671348969156350556061367185912804269543273492120576000*x^13+21395365215606868516773104448131873036117191598970850381868750856736406999040*x^12-64646052247079978701549176788371073862238401328273148004253858272283414517760*x^11-410950567918099238850233809548865651332381818984223380399065118884480628145152*x^10+1179412940419675095096459194990511962698145277057938818237560501315841683182080*x^9+5320587951999766816847939794337132161669228614779834580308250784313520624021760*x^8-2436650678687825489153812953885271342494826107433111999153812954866742258832000*x^7-138149850594284798121659073800245847318225112655101832249673213874072604803664960*x^6+413639639931664661913717169684760245990379500463935227134572628834903417705923808*x^5+371763378142585805466820515342247615578564577539325238549218843560933267175297200*x^4-4532516268456881444671627646524125285916819069316127054727635582444868338507885000*x^3+3217210082496068227026286487440148702174283452566073833772306730265630286802115500*x^2+21183849236156711918837256587734295321531493109255080107056307246562131568107313750*x-34702037287245803422976784945540540609637281732067549910009824505574785203557570075)/(519428752494490742932718094360214824253359390720000000000*x^30-22021614819297680455585027542146607653241382502400000000000*x^29+103406270867472244544846871978342745715158837886976000000000*x^28+8319615677729575183439086343982580359973313917747200000000000*x^27-120836496879060266664702387796640802605158522412523847680000000*x^26-974629350257429032244107267352529101572208017127610777600000000*x^25+29719840320259048002936657985031462364134868295212200047411200000*x^24-29636395181755004015769604438749928340604537153716194639872000000*x^23-3587676338630990384765151864581213384799402669366246353512955904000*x^22+19754121335987475489774265714312204549392388550368878422403317760000*x^21+234992855303269946421635632123516355004488467650477959734342264553472*x^20-2358639981173758307487990135156022028793151045884885548473376613335040*x^19-6916228754051775442704302822230140440344863659572304648153153069383680*x^18+149513158592181308655730270357001015972043904379210377684818492338995200*x^17-107990473784591900718455108891391447120501512715062945691718156151685120*x^16-5617525635290816583753427442875150329816208384662263843056579544296194048*x^15+17379110526359201702798837125919170713908409714104208986186746978054963200*x^14+120437546983286098648450177286078387701790252942173347344015326631920926720*x^13-672798515366052206334459588265611478494471508554950722997918113035725045760*x^12-1078480038806297722693446286095589130232128778435454845952941484815081799680*x^11+13172607326276152078861575924256669049783202879119837028215606775384118394880*x^10-8758875956425744523347307214292583018536273898777749691573514465746508840960*x^9-126794492484229520036963422074795165383758698713031438791925041229590805872640*x^8+278445867232961144547229615378621993633965218320990735195775446351811222568960*x^7+337369796019708308882505950640280097322326091646593263411978676667952072294400*x^6-1715325885796662913472674430695347835546361933571738193772045227960804860002304*x^5+1927063836331851101559549154301508604957849494211890461744713035547610293862400*x^4-1012343173032021322820714759389581823508015376021645602010988251055447474176000*x^3+281052460894244050063907778112668259789944237080500453726507334334606213120000*x^2-40066274258243420548967284249304845964924214146620243441697483656082227200000*x+2319028457365415709581460905401833033752830213070688965313853123775692800000))
            sage: dop = dop.numerator()
            sage: dop._valuation_bound()
            4
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: dop = (1) * Dx^3 + ((-37276100316*x^4-8601296521971*x^3+149474526562354*x^2-851516260119981*x+1606375172887730)/(60207507360*x^5-1357738101720*x^4+10766894818680*x^3-33663219309720*x^2+21109635466920*x+50367182065200)) * Dx^2 + ((-244962605945443932*x^8+45859894027452154377*x^7-1007721565559420627511*x^6+7688451401051418251751*x^5-10700444172772190981212*x^4-151584123526871575963626*x^3+812508614558115570090512*x^2-1394044793550319515881430*x+523936206539527555046735)/(14645054712768480*x^10-660520578153355920*x^9+12685635710012415150*x^8-134497429354272364380*x^7+847929913011990138330*x^6-3135724106883297930000*x^5+5862200627940909582330*x^4-1360052817241898920140*x^3-11899750655303006826330*x^2+8591086400229176243400*x+10249083020317568427000)) * Dx^1 + ((3612630465937377271296*x^12-34847198210004251666553504*x^11+1214323652964968262407229084*x^10-18389563613810827299100150875*x^9+158046460793874663922760730966*x^8-826155207999895832163818812791*x^7+2505450484359110983431547996570*x^6-3180870458747487002695340817081*x^5-3436514987049513841913427774018*x^4+12416246770727538978764204120859*x^3+2523223218251925254166023092170*x^2-20023104718177006637248729222320*x-8670024295507734979387367314900)/(143931597717088621440*x^15-9737394363136772972640*x^14+296805518829532843570440*x^13-5374736569789980282940050*x^12+64117800033505201966052070*x^11-526993531541075189435464620*x^10+3020206887409287886072602840*x^9-11842329693864967921225061460*x^8+29534459320186329611791388700*x^7-35715306059567401841457843960*x^6-20824869529453481354980517400*x^5+121727941389987133586763768390*x^4-81381731516357672782477425330*x^3-124551810125977847709358161900*x^2+105950131451442330918186321000*x+84264988358975155511950170000))
            sage: dop = dop.numerator()
            sage: dop._valuation_bound()
            2
        '''
        x_ = self.base_ring().gen()
        m = max(self.indicial_polynomial(x_).roots(ZZ, multiplicities = False), default = 0)
        return m

    def _extend_series_solution(self, t, p):
        
        ''' 
        INPUT:

        -''t'' -- a truncated power series solution of this operator.

        - ''p'' -- a integer.

        OUTPUT:
        
        A truncated power series solution of this operator with at least a degree of p.
        
        EXAMPLES:

        Fibonacci sequence::
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: dop = (-x^3-x^2+x)*Dx-x^2-1
            sage: t = [0, 1]
            sage: t = dop._extend_series_solution(t, 50)
            sage: list_fibonacci_sequence = [fibonacci(i) for i in range(50)]
            sage: t == list_fibonacci_sequence #example <1s hand check
            True
            
        Exponential and reciprical of factorial::
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: dop = Dx-1
            sage: t = [1, 1]
            sage: t = dop._extend_series_solution(t, 50)
            sage: list_reciprical_factorial = [1/factorial(i) for i in range(50)]
            sage: t == list_reciprical_factorial #example <1s hand check
            True        
        '''
        R = self.base_ring().base_ring()
        if len(t) < p:
            rop_ = self.to_S(OreAlgebra(R['n'], 'Sn')) # convertion of this operator into a recurrence operator
            t = rop_.to_list(t, p) # computation of the missing term of the list annihilated by this operator
        return t

    def _list_factor_singularity_data(self):
        
        ''' 
        INPUT:

        OUTPUT:
        
        A list of Factor_Singularity_Data with the classification of the root(s) of the factor and the needed information about it(them).
        '''
        x_ = self.base_ring().gen()
        list_fac = [(1/x_, 1)] + [(fac, mult) for fac, mult in  self.leading_coefficient().factor()]
        list_factor_singularity_data = [_Factor_Singularity_Data(fac, self) for fac, _ in list_fac]
        return list_factor_singularity_data

    def _right_factor_apparent_singularities_bound(self, r_order, list_factor_singularity_data=None):
        
        ''' 
        INPUT:

        -''r_order'' -- a natural number corresponding to a potential right factor of this operator.
        
        -''list_factor_singularity_data'' -- a list of Factor_Singularity_Data which contain all needed information to solve this problem 
       
        OUTPUT:
        
        A bound on the number of apparent singularities of a potential right factor of this operator.

        EXAMPLES::
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: p = (x^6 - 6*x^5 + 13*x^4 - 12*x^3 + 4*x^2)*Dx^2 + (-46*x^5 + 217*x^4 - 365*x^3 + 266*x^2 - 72*x)*Dx + 496*x^4 - 5223/4*x^3 + 5305/4*x^2 - 2127/2*x + 352
            sage: q = (x^6 - 6*x^5 + 13*x^4 - 12*x^3 + 4*x^2)*Dx^2 + (-19*x^5 + 104*x^4 - 213*x^3 + 196*x^2 - 68*x)*Dx - 224*x^4 + 6391/12*x^3 + 537/4*x^2 - 4181/6*x + 260
            sage: qpp = q*p*p
            sage: list_factor_singularity_data = qpp._list_factor_singularity_data()
            sage: qpp._right_factor_apparent_singularities_bound(4, list_factor_singularity_data) #example <1s hand check
            70 
            sage: qpp._right_factor_apparent_singularities_bound(2, list_factor_singularity_data) #example <1s hand check
            68
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: pol = -x^5 + 2*x^3 + 1/2*x^2 + x - 2
            sage: dop_pol = pol*Dx - pol.derivative()
            sage: q = (1) * Dx^2 + ((-4302*x-33686)/(180*x^2-905*x-4785)) * Dx^1 + ((7128324*x^2+170561664*x-914731699)/(129600*x^4-1303200*x^3-3614300*x^2+34643400*x+91584900))
            sage: q = q.numerator()
            sage: dop = q*dop_pol
            sage: dop._right_factor_apparent_singularities_bound(2) #example <1s hand check 
            35
            sage: dop._right_factor_apparent_singularities_bound(1) #example <1s hand check
            29
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: dop = (1) * Dx^4 + ((-8789*x-250083)/(1683*x^2-11052*x+3105)) * Dx^3 + ((-573876259*x^2+5184664947*x-1245897126)/(944163*x^4-12400344*x^3+44199378*x^2-22877640*x+3213675)) * Dx^2 + ((116088953763904*x^3-121641558287931*x^2-5636597753470860*x+12405624767176527)/(19068315948*x^6-375656021136*x^5+2572413374124*x^4-6785971942752*x^3+4745896331940*x^2-1278631299600*x+119741530500)) * Dx^1 + ((3761488140623395072*x^4-2796968328358578264*x^3-331614221956180131573*x^2+1432286768863927980036*x-2245937427043850033919)/(128367902961936*x^8-3371888445716736*x^7+34161316701057216*x^6-164070061344389376*x^5+363892995858420576*x^4-302696102480290560*x^3+116275864918737600*x^2-21174134321376000*x+1487189808810000))
            sage: dop = dop.numerator()
            sage: dop._right_factor_apparent_singularities_bound(2) #example <1s hand check
            59
            sage: dop._right_factor_apparent_singularities_bound(1) #example <1s hand check
            Traceback (most recent call last):
            ...
            MIPSolverException: PPL : There is no feasible solution
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ) 
            sage: A.<Dx>=OreAlgebra(R)
            sage: dop = (1) * Dx^5 + ((24645979552480*x^3-414806362828799*x^2-6140380543550081*x-6807970887923660)/(687429943200*x^4+228462294060*x^3-77245433114850*x^2-243810309252510*x+179137993834800)) * Dx^4 + ((39480875588243076416000*x^6+9004334923464026624912720*x^5+57348739278644381535502648*x^4+437905499904253879966481524*x^3+3430687887998087028160505643*x^2+4458586592872170141958312920*x+16450811126586577150322962140)/(660922974556580736000*x^8+439305795325895097600*x^7-148460772555796239498960*x^6-518182310198893164224400*x^5+8533907208139565097665820*x^4+52794850945735552025572200*x^3+44431165112749165306696140*x^2-122169761330732357416574400*x+44881707461757822111456000)) * Dx^3 + ((-402253835580037799210378240000*x^9-1871760751299048495706176012800*x^8+35366864727176365869737027313680*x^7-329248283349614581644685233516262*x^6-7316325084699689719016093175754974*x^5-18710451747784043548921564638802416*x^4+149350570273829052240306182942040927*x^3+1455436299826856363153253804395728020*x^2+6536736734973787060215978100497450300*x+8080926451869911179294697500462495500)/(68048629460345552578560000*x^12+67846387030131238873344000*x^11-22917007589238825119382124800*x^10-87649371017562372839193653280*x^9+2580220984203261078474116322000*x^8+17155950139255521293341465463640*x^7-77412420868986074042558470383300*x^6-947399546294583942949178925074940*x^5-2212531796006159252402319810968700*x^4+1209000517026597119360184763694580*x^3+5134093813469043175883662676944800*x^2-4916802676797802007585931667104000*x+1204195959150245540692093520640000)) * Dx^2 + ((-1408244176348017185177028920320000*x^12-113307167125184244872088961878860800*x^11-2023970487364409457999053493395916800*x^10-34191194664143302286640771358266700160*x^9-261124545701604411848530588658345971092*x^8-13525082639479340305106227507024448112*x^7+3239929125723180040534344616416176412528*x^6-22186334011720259897145627274075090089672*x^5-68711716381058675159445908389938711427137*x^4+74758072615350661487138424212032151954040*x^3-2109163819586025959854179321275039195747100*x^2-4452741360150523496770216948071050269755000*x-4372889182868609096239111314940270838536050)/(77847632102635312149872640000*x^16+103488355683293516361474048000*x^15-34938878778553050650751104102400*x^14-145315712458960498620639875681280*x^13+5857176658968246783973873411083744*x^12+41193301508811067683259192134214080*x^11-384992156473889264093748343089997296*x^10-4391679024198196090448490552970900080*x^9+868311098675445761000163181217108574*x^8+158853540455244689875169496047335595160*x^7+652074542662890403790315172596993351364*x^6+456192197314549777127809491060529197000*x^5-1810610014852412041276463562624418487586*x^4-1090183192697823476192516598959627570880*x^3+3370708599493843529109864517263175353600*x^2-1954370062395004724391316806493834752000*x+358990082558034599608964151999114240000)) * Dx^1 + ((43888135231738960167609796858265600000*x^15+2745991295001642206167981620243777536000*x^14+87027679413114261177317195810335851520000*x^13+1629607197737960372591827875078540083200000*x^12+14931831935339715568174333544849248872230400*x^11+55688492267012709532563476548505536405692864*x^10+194933639632012349787469798235901468193103280*x^9+2123811321547684149325060404422898910430424240*x^8+3586366021915833158806299268014042249628165560*x^7-25993269050791222693462053074823540559328262930*x^6+176227717367534240239788147481161963336541639761*x^5+176878019323523638954796039745304746754374982200*x^4-10580895443546795403721240743668643427794083568000*x^3-40923805675211589453288563750285621123920294732800*x^2-80065792010549570600491971728065154640222644736000*x-60499225396935182385758430666053784725524822794240)/(268586687521092245220576460800000*x^20+446314067288108704689087283200000*x^19-150606682448162027994638887034880000*x^18-676805163641812202803250894152704000*x^17+33530339103525366922930885685028518400*x^16+248023313692267199967365283786678634944*x^15-3405408178752769947709774145165531488800*x^14-38861498867445089073768889934449007253120*x^13+102076621309087569294549867223135282702800*x^12+2759808746819783013358987764173511843264540*x^11+7123080175826440478542240835763247296878550*x^10-64275131332467264207936519989249024121529590*x^9-452128819755491433568143743173454071419941700*x^8-837797139938820551482435553488991616196873680*x^7+740373518089517214271543637198534839225335150*x^6+3045508271741071672053605867901046674980136906*x^5-1601652666551615085397328028988103942240504400*x^4-4346682197741918461257761022574264177370976000*x^3+5282855303922093597736621493658978696948480000*x^2-2196418794622626059507181392978096186035200000*x+322760803426277747816427489779363630899200000))
            sage: dop = dop.numerator()
            sage: dop._right_factor_apparent_singularities_bound(3) #example 17s
            28
            
        '''
        import contextlib
        import io
        if list_factor_singularity_data is None:
            list_factor_singularity_data = self._list_factor_singularity_data()
        
        def lcm_denominator(list_factor_singularity_data):
            list_denominator_ = [1]
            for fsd in list_factor_singularity_data:
                for r_ in fsd.ind_pol_fac_sum_roots:
                    list_denominator_.append(r_.denominator())
            lcm_denominator_ = lcm(list_denominator_)
            return lcm_denominator_

        lcm_denominator_ = lcm_denominator(list_factor_singularity_data)
        p = MixedIntegerLinearProgram(maximization=True, solver="PPL")
        c = p.new_variable(binary=True)
        d = p.new_variable(integer=True, nonnegative=True)
        for i, fsd in enumerate(list_factor_singularity_data):
            if not fsd.is_apparent:
                p.add_constraint(sum(c[i, j]*d for j, d in enumerate(fsd.ind_pol_fac_degree)) == r_order)
        p.add_constraint(lcm_denominator_*d[0] == lcm_denominator_*(-r_order*(r_order - 1) 
                                 - sum(fsd.fac_degree*(sum (c[i, j]*e 
                                    for j, e in enumerate(fsd.ind_pol_fac_sum_roots)) -r_order*(r_order - 1)/2)
                                 for i, fsd in enumerate(list_factor_singularity_data) if not fsd.is_apparent)))
        p.set_objective(d[0])
        with contextlib.redirect_stdout(io.StringIO()) as f:
            p.show()
        logging.debug(f.getvalue())
        b = p.solve()
        return b

    def _right_factor_degree_bound(self, r_order, list_factor_singularity_data=None):
        
        ''' 
        INPUT:

       -''r_order'' -- a natural number corresponding to a potential right factor of this operator.
       
        OUTPUT:
        
        A bound on the degree of the potential right factor of this operator.

        EXAMPLES::
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ) 
            sage: A.<Dx>=OreAlgebra(R)
            sage: p = (x^6 - 6*x^5 + 13*x^4 - 12*x^3 + 4*x^2)*Dx^2 + (-46*x^5 + 217*x^4 - 365*x^3 + 266*x^2 - 72*x)*Dx + 496*x^4 - 5223/4*x^3 + 5305/4*x^2 - 2127/2*x + 352
            sage: q = (x^6 - 6*x^5 + 13*x^4 - 12*x^3 + 4*x^2)*Dx^2 + (-19*x^5 + 104*x^4 - 213*x^3 + 196*x^2 - 68*x)*Dx - 224*x^4 + 6391/12*x^3 + 537/4*x^2 - 4181/6*x + 260
            sage: qpp = q*p*p
            sage: qpp._right_factor_degree_bound(4) #example <1s hand check
            444
            sage: qpp._right_factor_degree_bound(2) #example <1s hand check
            432
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: pol = -x^5 + 2*x^3 + 1/2*x^2 + x - 2
            sage: dop_pol = pol*Dx - pol.derivative()
            sage: q = (1) * Dx^2 + ((-4302*x-33686)/(180*x^2-905*x-4785)) * Dx^1 + ((7128324*x^2+170561664*x-914731699)/(129600*x^4-1303200*x^3-3614300*x^2+34643400*x+91584900))
            sage: q = q.numerator()
            sage: dop = q*dop_pol
            sage: dop._right_factor_degree_bound(2) #example <1s hand check
            129
            sage: dop._right_factor_degree_bound(1) #example <1s hand check
            111

            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: dop = (1) * Dx^4 + ((-8789*x-250083)/(1683*x^2-11052*x+3105)) * Dx^3 + ((-573876259*x^2+5184664947*x-1245897126)/(944163*x^4-12400344*x^3+44199378*x^2-22877640*x+3213675)) * Dx^2 + ((116088953763904*x^3-121641558287931*x^2-5636597753470860*x+12405624767176527)/(19068315948*x^6-375656021136*x^5+2572413374124*x^4-6785971942752*x^3+4745896331940*x^2-1278631299600*x+119741530500)) * Dx^1 + ((3761488140623395072*x^4-2796968328358578264*x^3-331614221956180131573*x^2+1432286768863927980036*x-2245937427043850033919)/(128367902961936*x^8-3371888445716736*x^7+34161316701057216*x^6-164070061344389376*x^5+363892995858420576*x^4-302696102480290560*x^3+116275864918737600*x^2-21174134321376000*x+1487189808810000))
            sage: dop = dop.numerator()
            sage: dop._right_factor_degree_bound(3) #example <1s hand check
            Traceback (most recent call last):
            ...
            MIPSolverException: PPL : There is no feasible solution
            sage: dop._right_factor_degree_bound(2)
            248
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: dop = (1) * Dx^5 + ((24645979552480*x^3-414806362828799*x^2-6140380543550081*x-6807970887923660)/(687429943200*x^4+228462294060*x^3-77245433114850*x^2-243810309252510*x+179137993834800)) * Dx^4 + ((39480875588243076416000*x^6+9004334923464026624912720*x^5+57348739278644381535502648*x^4+437905499904253879966481524*x^3+3430687887998087028160505643*x^2+4458586592872170141958312920*x+16450811126586577150322962140)/(660922974556580736000*x^8+439305795325895097600*x^7-148460772555796239498960*x^6-518182310198893164224400*x^5+8533907208139565097665820*x^4+52794850945735552025572200*x^3+44431165112749165306696140*x^2-122169761330732357416574400*x+44881707461757822111456000)) * Dx^3 + ((-402253835580037799210378240000*x^9-1871760751299048495706176012800*x^8+35366864727176365869737027313680*x^7-329248283349614581644685233516262*x^6-7316325084699689719016093175754974*x^5-18710451747784043548921564638802416*x^4+149350570273829052240306182942040927*x^3+1455436299826856363153253804395728020*x^2+6536736734973787060215978100497450300*x+8080926451869911179294697500462495500)/(68048629460345552578560000*x^12+67846387030131238873344000*x^11-22917007589238825119382124800*x^10-87649371017562372839193653280*x^9+2580220984203261078474116322000*x^8+17155950139255521293341465463640*x^7-77412420868986074042558470383300*x^6-947399546294583942949178925074940*x^5-2212531796006159252402319810968700*x^4+1209000517026597119360184763694580*x^3+5134093813469043175883662676944800*x^2-4916802676797802007585931667104000*x+1204195959150245540692093520640000)) * Dx^2 + ((-1408244176348017185177028920320000*x^12-113307167125184244872088961878860800*x^11-2023970487364409457999053493395916800*x^10-34191194664143302286640771358266700160*x^9-261124545701604411848530588658345971092*x^8-13525082639479340305106227507024448112*x^7+3239929125723180040534344616416176412528*x^6-22186334011720259897145627274075090089672*x^5-68711716381058675159445908389938711427137*x^4+74758072615350661487138424212032151954040*x^3-2109163819586025959854179321275039195747100*x^2-4452741360150523496770216948071050269755000*x-4372889182868609096239111314940270838536050)/(77847632102635312149872640000*x^16+103488355683293516361474048000*x^15-34938878778553050650751104102400*x^14-145315712458960498620639875681280*x^13+5857176658968246783973873411083744*x^12+41193301508811067683259192134214080*x^11-384992156473889264093748343089997296*x^10-4391679024198196090448490552970900080*x^9+868311098675445761000163181217108574*x^8+158853540455244689875169496047335595160*x^7+652074542662890403790315172596993351364*x^6+456192197314549777127809491060529197000*x^5-1810610014852412041276463562624418487586*x^4-1090183192697823476192516598959627570880*x^3+3370708599493843529109864517263175353600*x^2-1954370062395004724391316806493834752000*x+358990082558034599608964151999114240000)) * Dx^1 + ((43888135231738960167609796858265600000*x^15+2745991295001642206167981620243777536000*x^14+87027679413114261177317195810335851520000*x^13+1629607197737960372591827875078540083200000*x^12+14931831935339715568174333544849248872230400*x^11+55688492267012709532563476548505536405692864*x^10+194933639632012349787469798235901468193103280*x^9+2123811321547684149325060404422898910430424240*x^8+3586366021915833158806299268014042249628165560*x^7-25993269050791222693462053074823540559328262930*x^6+176227717367534240239788147481161963336541639761*x^5+176878019323523638954796039745304746754374982200*x^4-10580895443546795403721240743668643427794083568000*x^3-40923805675211589453288563750285621123920294732800*x^2-80065792010549570600491971728065154640222644736000*x-60499225396935182385758430666053784725524822794240)/(268586687521092245220576460800000*x^20+446314067288108704689087283200000*x^19-150606682448162027994638887034880000*x^18-676805163641812202803250894152704000*x^17+33530339103525366922930885685028518400*x^16+248023313692267199967365283786678634944*x^15-3405408178752769947709774145165531488800*x^14-38861498867445089073768889934449007253120*x^13+102076621309087569294549867223135282702800*x^12+2759808746819783013358987764173511843264540*x^11+7123080175826440478542240835763247296878550*x^10-64275131332467264207936519989249024121529590*x^9-452128819755491433568143743173454071419941700*x^8-837797139938820551482435553488991616196873680*x^7+740373518089517214271543637198534839225335150*x^6+3045508271741071672053605867901046674980136906*x^5-1601652666551615085397328028988103942240504400*x^4-4346682197741918461257761022574264177370976000*x^3+5282855303922093597736621493658978696948480000*x^2-2196418794622626059507181392978096186035200000*x+322760803426277747816427489779363630899200000))
            sage: dop = dop.numerator()
            sage: dop._right_factor_degree_bound(3) #example 17s
            165

        
        '''
        if list_factor_singularity_data is None:
            list_factor_singularity_data = self._list_factor_singularity_data()
        r = self.order()
        s_dop = sum([fsd.fac_degree for fsd in list_factor_singularity_data if not fsd.is_apparent]) # computation of the number of non apparent singularities of this operator
        b = self._right_factor_apparent_singularities_bound(r_order, list_factor_singularity_data) 
        N = r*(b + s_dop) # computation of a bound on the degree of the coefficient of a right factor right_fac_dop of this operator of order r_order, in the research paper 
        #'Explicit degree bounds for right factors of linear differential operators' written by Alin BOSTIAN, Tanguy RIVOAL, and Bruno SALVY see on pages 6 and 7.
        return N 
        
    def minimal_annihilator(self, initial_conditions):
        
        ''' 
        INPUT:
       -''initial_conditions'' -- a list of coefficient which represented a truncated power serie solution of this operator which determined a unique power serie solution of this operator.
       
        OUTPUT:
        
        The minimal order operator which annihilate initial_conditions.
        
        This code is a try of implementation of the algorithm presented in the research paper 
        'Minimisation of differential equations and algebraic values of E-functioncs' written by Alin BOSTIAN, Tanguy RIVOAL, and Bruno SALVY.
        This code only implemented the case where the differential operator is Fuchsian.
        Some function can be improuve for instance :
            -the determination if a point is a apparent singularity if the point is a root of a polynomial of high degree,
            the methode power_series_solutions takes a lot of time.
            -the function _right_factor_apparent_singularities_bound in some cases can take a lot of time.
            -the function guess in some cases can take a lot of time.

        EXAMPLES::
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: pol = -x^5 + 2*x^3 + 1/2*x^2 + x - 2
            sage: dop_pol = pol*Dx - pol.derivative()
            sage: q = (1) * Dx^2 + ((-4302*x-33686)/(180*x^2-905*x-4785)) * Dx^1 + ((7128324*x^2+170561664*x-914731699)/(129600*x^4-1303200*x^3-3614300*x^2+34643400*x+91584900))
            sage: q = q.numerator()
            sage: dop = q*dop_pol
            sage: initial_conditions = list(pol) + [0 for _ in range(50)]
            sage: dop_min = dop.minimal_annihilator(initial_conditions)
            sage: dop_pol.monic() == dop_min.monic() #example <1s
            True
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: pol = -3/26*x^10 - 1/2*x^9 - 7*x^8 + 34*x^7 + x^5 - 3/2*x^4 + x^3 - 162*x^2 - 29/51*x + 1
            sage: dop_pol = pol*Dx-pol.derivative()
            sage: e = (1) * Dx^4 + ((3348152203*x^2-31637133076*x+59125377396)/(75315240*x^3-1303422120*x^2+6899452560*x-11001070080)) * Dx^3 + ((-322596484080077900*x^4+15988727042629656667*x^3-231730571492254215098*x^2+1274479487856801877152*x-2356655368804738543008)/(3502819174654800*x^6-121241119183984800*x^5+1690882851101043600*x^4-12129907155140174400*x^3+47104852136916974400*x^2-93741424638762700800*x+74734623717020467200)) * Dx^2 + ((-4944775196347932477070*x^6+96246681870541809327831*x^5-632950856064133846487028*x^4+1334050468347601932646402*x^3+3222997729871087195102772*x^2-22316961365697458203852824*x+34776180106511727325739760)/(366044603751426600*x^9-19004545432089617400*x^8+429494309943265967400*x^7-5539645199758602933000*x^6+44896840115193463597200*x^5-236914340999728269477600*x^4+813426786960396801192000*x^3-1751552688077007546931200*x^2+2146303658529110797516800*x-1140749296416600411340800)) * Dx^1 + ((-17483283827572888736888400*x^8+89323443588786794725249395*x^7+6027729750374375677821936804*x^6-118685781917169083324382695614*x^5+1112622460040899341209884074958*x^4-6323380727201323628141021864232*x^3+22047940443205483377192041931420*x^2-43367858299121997001379196583080*x+36959585891499041335544730527712)/(351915282046621533240*x^12-24361293304547944224480*x^11+761355391311403516227600*x^10-14196991050674080962112800*x^9+175828583426543752660595640*x^8-1522970839368971326255521600*x^7+9455918973122537925993023040*x^6-42388134777081857232134265600*x^5+136115535491916678212760685440*x^4-305297941208861119092244070400*x^3+453976854359579592885043200000*x^2-401870462937137059149769605120*x+160194054796627495844094935040))
            sage: e = e.numerator()       
            sage: dop = e*dop_pol
            sage: initial_conditions = list(pol) + [0 for _ in range(50)]
            sage: dop_min = dop.minimal_annihilator(initial_conditions)
            sage: dop_pol.monic() == dop_min.monic() #example 15s
            True
            
            sage: from ore_algebra import *
            sage: R.<x>=PolynomialRing(QQ)
            sage: A.<Dx>=OreAlgebra(R)
            sage: p = (x^6 - 6*x^5 + 13*x^4 - 12*x^3 + 4*x^2)*Dx^2 + (-46*x^5 + 217*x^4 - 365*x^3 + 266*x^2 - 72*x)*Dx + 496*x^4 - 5223/4*x^3 + 5305/4*x^2 - 2127/2*x + 352
            sage: q = (x^6 - 6*x^5 + 13*x^4 - 12*x^3 + 4*x^2)*Dx^2 + (-19*x^5 + 104*x^4 - 213*x^3 + 196*x^2 - 68*x)*Dx - 224*x^4 + 6391/12*x^3 + 537/4*x^2 - 4181/6*x + 260
            sage: qpp = q*p*p
            sage: list_factor_singularity_data = qpp._list_factor_singularity_data()
            sage: initial_conditions = list((p.power_series_solutions(50)[0]))
            sage: qpp_min = qpp.minimal_annihilator(initial_conditions)
            sage: p.monic() == qpp_min.monic() #example <1s
            True     
        '''
        
        from ore_algebra.guessing import guess
        from sage.numerical.mip import MIPSolverException
        import time
        FORMAT = '%(created)f -- %(message)s'
        logging.basicConfig(format=FORMAT)
                 
        t1 = time.time()
        right_fac_dop = self.numerator() #creation of the current right factor of this operator of minimal order which annihilate initial_conditions with coefficient in R[x]
        t = initial_conditions
        r = self.order()
        r_order = r # creation of the current order tested of a right factor
        p = self._valuation_bound() + r
        R = self.base_ring()
        list_factor_singularity_data = self._list_factor_singularity_data()
        while r_order > 1:
            r_order = r_order - 1 # we test if a right factor of this operator of order r_order-1 exists
            #logging.debug('r_order : %s' % (r_order))
            try:
                t2 = time.time()
                N = self._right_factor_degree_bound(r_order, list_factor_singularity_data)
                #logging.debug('N', 'r_order : %s', 'succeed', 'time : %f' % (r_order, time.time()-t2))
            except MIPSolverException:
                #logging.debug('N', 'r_order : %s', 'fail', 'time : %f' % (r_order, time.time()-t2))
                continue # There is no right factor of this operator of order r_order
            while True:
                t3 = time.time()
                t = self._extend_series_solution(t, p + r_order)
                #logging.debug('_extend_series_solution', 'r_order : %s', 'time : %f' % (r_order, time.time()-t3))
                k = p//(r_order + 1)
                try:
                    t4 = time.time()
                    guess_dop = guess(t, self.parent(), min_order = r_order, max_order = r_order, max_degree = k) # we can either set min_order as 1 or as r_order
                    #logging.debug('guess', 'r_order : %s', 'succeed', 'time : %f' % (r_order, time.time()-t4))
                    #logging.debug(guess_dop)
                except ValueError:
                    #logging.debug('guess', 'r_order : %s', 'fail', 'time : %f' % (r_order, time.time()-t4))
                    pass
                else:
                    g_guess_dop = min([valuation(a_i) - i for i, a_i in enumerate(self)], default = 0) #g_{guess_dop} as decribed in the research paper 'Minimisation of differential equations and algebraic values of E-functioncs' written by Alin BOSTIAN, Tanguy RIVOAL, and Bruno SALVY, on page 6.
                    if valuation(guess_dop(R(t))) >= self._valuation_bound() + max(0, g_guess_dop) + 1:
                        right_fac_dop = guess_dop
                        r_order = guess_dop.order()
                        break # guess_dop is a right factor of this operator with a lower order than right_fac_dop
                if p >= (r_order + 1)*(N + 1):
                        break # There is no right factor of this operator of order r_order
                p = 2*p # increation of p for it to be greater than (r_order + 1)*(N + 1)
        #logging.debug('time : %f' % (time.time()-t1))
        return right_fac_dop

    
#############################################################################################################

class UnivariateEulerDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
    Element of an Ore algebra K(x)[T], where T is the Euler differential operator T = x*d/dx
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        R = self.parent()
        x = R.base_ring().gen()
        if "action" not in kwargs:
            kwargs["action"] = lambda p: x*p.derivative()

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_D(self, alg):  # theta2d
        """
        Returns the differential operator corresponding to ``self``

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_D()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the standard derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']
          sage: A.<Tx> = OreAlgebra(R, 'Tx')
          sage: (Tx^4).to_D(OreAlgebra(R, 'Dx'))
          x^4*Dx^4 + 6*x^3*Dx^3 + 7*x^2*Dx^2 + x*Dx
          sage: (Tx^4).to_D('Dx').to_T(A)
          Tx^4

        """
        R = self.base_ring()
        x = R.gen()
        one = R.one()

        if isinstance(alg, str):
            alg = self.parent().change_var_sigma_delta(alg, {}, {x: one})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_D():
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring()
        theta = R.gen()*alg.gen()
        theta_k = alg.one()
        c = self.coefficients(sparse=False)
        out = alg(R(c[0]))

        for i in range(self.order()):

            theta_k *= theta
            out += R(c[i + 1])*theta_k

        return out

    def to_S(self, alg):
        r"""
        Returns a recurrence operator annihilating the coefficient sequence of
        every power series (at the origin) annihilated by ``self``.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_S()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the standard shift with respect to ``self.base_ring().gen()``.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = ZZ['x']
            sage: A.<Tx> = OreAlgebra(R, 'Tx')
            sage: R2.<n> = ZZ['n']
            sage: A2.<Sn> = OreAlgebra(R2, 'Sn')
            sage: (Tx - 1).to_S(A2)
            n - 1
            sage: ((1+x)*Tx^2 + Tx).to_S(A2)
            (n^2 + 3*n + 2)*Sn + n^2
            sage: ((x^3+x^2-x)*Tx + (x^2+1)).to_S(A2)
            Sn^3 + (-n - 2)*Sn^2 + (n + 2)*Sn + n

        """
        return self.to_D('D').to_S(alg)

    def to_F(self, alg):
        r"""
        Returns a difference operator annihilating the coefficient sequence of
        every power series (about the origin) annihilated by ``self``.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_F()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the forward difference with respect to ``self.base_ring().gen()``.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = ZZ['x']
            sage: A.<Tx> = OreAlgebra(R, 'Tx')
            sage: R2.<n> = ZZ['n']
            sage: A2.<Fn> = OreAlgebra(R2, 'Fn')
            sage: (Tx - 1).to_F(A2)
            n - 1
            sage: ((1+x)*Tx^2 + Tx).to_F(A2)
            (n^2 + 3*n + 2)*Fn + 2*n^2 + 3*n + 2
            sage: ((x^3+x^2-x)*Tx + (x^2+1)).to_F(A2)
            Fn^3 + (-n + 1)*Fn^2 + (-n + 1)*Fn + n + 1

        """
        return self.to_D('D').to_F(alg)

    def power_series_solutions(self, *args, **kwargs):
        return self.to_D('D').power_series_solutions(*args, **kwargs)

    power_series_solutions.__doc__ = UnivariateDifferentialOperatorOverUnivariateRing.power_series_solutions.__doc__

    def spread(self, p=0):
        return self.to_D().spread(p)

    spread.__doc__ = UnivariateDifferentialOperatorOverUnivariateRing.spread.__doc__

    def _coeff_list_for_indicial_polynomial(self):
        return self.to_D()._coeff_list_for_indicial_polynomial()

    def _denominator_bound(self):
        return self.to_D()._denominator_bound()

    def symmetric_product(self, other, solver=None):

        if not isinstance(other, UnivariateOreOperator):
            raise TypeError("unexpected argument in symmetric_product")

        if self.parent() != other.parent():
            A, B = canonical_coercion(self, other)
            return A.symmetric_product(B, solver=solver)

        A = self.to_D('D')
        B = other.to_D(A.parent())
        return A.symmetric_product(B, solver=solver).to_T(self.parent())

    symmetric_product.__doc__ = UnivariateOreOperator.symmetric_product.__doc__


#############################################################################################################

def _tower(dom):
    if is_PolynomialRing(dom) or is_MPolynomialRing(dom):
        base, vars = _tower(dom.base_ring())
        return base, vars.union(set(dom.variable_names()))
    elif isinstance(dom, FractionField_generic):
        return _tower(dom.ring())
    else:
        return dom, set()
        
        
class _Factor_Singularity_Data:

    '''

    EXAMPLES::

        sage: from ore_algebra import *
        sage: from ore_algebra.differential_operator_1_1 import _Factor_Singularity_Data
        sage: R.<x>=PolynomialRing(QQ)
        sage: A.<Dx>=OreAlgebra(R)
        sage: dop = (1) * Dx^3 + ((-381277225*x^2-788431755*x+18325709424)/(5890500*x^3-20196000*x^2-395942580*x+1799463600)) * Dx^2 + ((-5421494604745625*x^4+48954833592924000*x^3+158868817738775325*x^2-2237565203895017700*x+4543644347878027392)/(3832653825000*x^6-26281054800000*x^5-470186842554000*x^4+4108179397608000*x^3+9287996212071720*x^2-157398483670084800*x+357669086840208000)) * Dx^1 + ((222834891756421875*x^6-5580440718303193750*x^5+46401236528146447500*x^4-281575459198101293250*x^3+1854138529385081693325*x^2-6637179953341173564900*x+6340899106132497425376)/(111785736562500*x^9-1149796147500000*x^8-18599595251737500*x^7+252513536868450000*x^6+547719056317075500*x^5-17763039204417936000*x^4+44566625258768552940*x^3+355568864800739475600*x^2-2103630734250683352000*x+3186831563746253280000))
        sage: dop = dop.numerator()
        sage: fsd_1 = _Factor_Singularity_Data(x - 33/5, dop)
        sage: fsd_1.fac #example <1s check
        x - 33/5
        sage: fsd_1.is_regular #example <1s check
        True
        sage: fsd_1.is_apparent #example <1s check
        False
        sage: fsd_1.ind_pol #example <1s check
        alpha^3 - 1811/51*alpha^2 + 28213/68*alpha - 27027/17
        sage: fsd_1.ind_pol_fac #example <1s check
        [alpha - 27/2, alpha - 77/6, alpha - 156/17]
        sage: fsd_1.ind_pol_fac_degree #example <1s check
        [1, 1, 1]
        sage: fsd_1.ind_pol_fac_sum_roots #example <1s check
        [27/2, 77/6, 156/17]
        sage: fsd_2 = _Factor_Singularity_Data(x - 27/5, dop)
        sage: fsd_2.fac #example <1s check
        x - 27/5
        sage: fsd_2.is_regular #example <1s check
        True
        sage: fsd_2.is_apparent #example <1s check
        False
        sage: fsd_2.ind_pol #example <1s check
        alpha^3 - 5917/180*alpha^2 + 18917/90*alpha + 47177/60
        sage: fsd_2.ind_pol_fac #example <1s check
        [alpha - 191/9, alpha - 57/4, alpha + 13/5]
        sage: fsd_2.ind_pol_fac_degree #example <1s check
        [1, 1, 1]
        sage: fsd_2.ind_pol_fac_sum_roots #example <1s check
        [191/9, 57/4, -13/5]
        
        sage: from ore_algebra import *
        sage: from ore_algebra.differential_operator_1_1 import _Factor_Singularity_Data
        sage: R.<x>=PolynomialRing(QQ)
        sage: A.<Dx>=OreAlgebra(R)
        sage: dop = (x*Dx-2)*(x*Dx-1)
        sage: dop = dop.numerator()
        sage: fsd_0 = _Factor_Singularity_Data(1/x, dop)
        sage: fsd_0.fac #example <1s check
        1/x
        sage: fsd_0.is_regular #example <1s check
        True
        sage: fsd_0.is_apparent #example <1s check
        False
        sage: fsd_0.ind_pol #example <1s check
        alpha^2 + 3*alpha + 2
        sage: fsd_0.ind_pol_fac #example <1s check
        [alpha + 1, alpha + 2]
        sage: fsd_0.ind_pol_fac_degree #example <1s check
        [1, 1]
        sage: fsd_0.ind_pol_fac_sum_roots #example <1s check
        [-1, -2]
        sage: fsd_1 = _Factor_Singularity_Data(x, dop)
        sage: fsd_1.fac #example <1s check
        x
        sage: fsd_1.is_regular #example <1s check
        True
        sage: fsd_1.is_apparent #example <1s check
        True
        
        sage: from ore_algebra import *
        sage: from ore_algebra.differential_operator_1_1 import _Factor_Singularity_Data
        sage: R.<x>=PolynomialRing(QQ)
        sage: A.<Dx>=OreAlgebra(R)
        sage: dop = (x^2 + 1)*Dx -1
        sage: dop = dop.numerator()
        sage: fsd_1 = _Factor_Singularity_Data(x^2 + 1, dop)
        sage: fsd_1.fac #example <1s check
        x^2 + 1
        sage: fsd_1.is_regular #example <1s check
        True
        sage: fsd_1.is_apparent #example <1s check
        False
        sage: fsd_1.ind_pol #example <1s check
        alpha + 1/2*s
        sage: fsd_1.ind_pol_fac #example <1s check
        [alpha + 1/2*s]
        sage: fsd_1.ind_pol_fac_degree #example <1s check
        [1]
        sage: fsd_1.ind_pol_fac_sum_roots #example <1s check
        [-1/2*s]
    '''
    
    def __init__(self, fac, dop):

        self.fac = fac
        self.classify_singularity(dop)
        self.degree_roots_sum_indicial_polynomial_irreducible_factor(dop)

    def classify_singularity(self, dop):    
        
        ''' 
        INPUT:
    
        -''dop'' -- a linear differential operator.
    
        OUTPUT:
    
        There is no output, this method update the classification of the factor i.e. if the root(s) of the factor is a singularity or not, regular or not, apparent or not. 
        
        '''
        from sage.misc.misc_c import prod
        
        r = dop.order()
        R = dop.base_ring()
        x_ = dop.base_ring().gen()
        if self.fac == 1/x_:
            self.fac_degree = 1
            dop = dop.annihilator_of_composition(self.fac)
            pol = x_
            pol_= x_
        elif self.fac.degree() == 1:
            self.fac_degree = 1
            pol = self.fac
            pol_= self.fac[1]*x_-self.fac[0]
        else :
            self.fac_degree = self.fac.degree()
            K = R.base_ring().extension(self.fac, 's')
            s = K.gen()
            dop = dop.change_ring(K[str(x_)])
            pol = x_-s
            pol_ = x_+s
        ind_pol = dop.indicial_polynomial(pol, 'alpha')
        alpha = ind_pol.parent().gen()
        if ind_pol.degree() < r:
            self.is_regular = False
            raise NotImplementedError(f'{dop} is not Fuchsian')
        else:
            self.is_regular = True
            ind_pol = ind_pol.monic()
            self.ind_pol = ind_pol
            ind_pol_roots = ind_pol.roots(multiplicities = False)
            ind_pol_roots = [r_ for r_ in ind_pol_roots if (r_.is_integer() == True and r_ >= 0)]
            if len(ind_pol_roots) != r: #testing if the indicial polynomial in one of the root of the i-th irreducible factor as r natural roots if not all the roots of the i-th irreducible factor are non apparent singularities
                self.is_apparent = False
            elif ind_pol == prod([alpha-i for i in range(r)]): #testing if the roots of indicial polynomial in one of the root of the i-th irreducible factor are different from 0, 1, ..., r-1 if not all the roots of the i-th irreducible factor are non apparent singularities
                self.is_apparent = False
            else:
                dop_c = dop.annihilator_of_composition(pol_)
                if len(dop_c.power_series_solutions()) < r: # testing if there is a base of power serie of cardinal r of the differential operator in one of the root of the i-th irreducible factor if not all the roots of the i-th irreducible factor are non apparent singularities
                    self.is_apparent = False
                else:
                    self.is_apparent = True
                    
    def degree_roots_sum_indicial_polynomial_irreducible_factor(self, dop):
        
        ''' 
        INPUT:
    
        -''dop'' -- a linear differential operator.
    
        
        OUTPUT:
        
        There is no output, this method update the needed information about the root(s) of the factor.
        In the case where the root(s) of the factor is an apparent regular  singularity, we don't need information.
        In the case where the root(s) of the factor is a non-apparent regular factor sngularity, we need to know the irreducibles factors, their degrees, their root's sum, of the indicial polynomial in the factor.
        '''
        
        if self.is_regular:
            if not self.is_apparent: 
                ind_pol = self.ind_pol
                ind_pol_fac_ = list(ind_pol.factor())
                ind_pol_fac = []
                for fac_, mult_ in ind_pol_fac_:
                    ind_pol_fac += [fac_]*mult_ 
                self.ind_pol_fac = ind_pol_fac
                self.ind_pol_fac_degree = [fac_.degree() for fac_ in ind_pol_fac]
                self.ind_pol_fac_sum_roots = [-fac_[fac_.degree()-1]/fac_[fac_.degree()] for fac_ in ind_pol_fac]

