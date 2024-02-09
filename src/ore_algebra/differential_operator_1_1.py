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

from functools import reduce

import sage.functions.log as symbolic_log

from sage.arith.all import gcd, lcm, nth_prime
from sage.arith.misc import valuation
from sage.matrix.constructor import matrix
from sage.misc.cachefunc import cached_method
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
            points = [(QQ(i), QQ(c[i].valuation()))   # UNUSED !
                      for i in range(self.order() + 1)
                      if not c[i].is_zero()]

            y = R.base_ring()['x'].gen()  # variable name changed from y to x to avoid PARI warning
            x = R.gen()
            K = R.base_ring()
            for (s, p) in self.newton_polygon(x):
                e = 1 - s
                if e > 0 or -e >= exp or not (ramification or e in ZZ):
                    continue
                for (q, _) in p(e*y).factor():
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

        for (c, e) in shift_factor(indpoly):

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
                        nn = max(nn, ZZ(-q[0]/q[1]))
                    except:
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
                subs = {x: K1(nth_prime(5 + _) + nth_prime(15 + i)) for (i, x) in enumerate(vars)}
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
        for (p, p1) in zip(fac, fac1):
            e = 0
            for j in range(r + 1):  # may be needed for inhomogeneous part
                if not L1[j].is_zero():
                    e = max(e, L1[j].valuation(p1) - j)
            for (q, _) in L1.indicial_polynomial(p1).factor():  # contribution for homogeneous part
                if q.degree() == 1:
                    try:
                        e = max(e, ZZ(q[0]/q[1]))
                    except:
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
        max_coef_size = None
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
        for (f, m) in fac:
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
