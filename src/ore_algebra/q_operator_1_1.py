"""
Univariate q-recurrence and q-differential operators over univariate rings
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
#  https://www.gnu.org/licenses/                                             #
#############################################################################

from sage.arith.misc import GCD as gcd
from sage.misc.misc_c import prod
from sage.rings.infinity import infinity
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.element import canonical_coercion
from sage.structure.factorization import Factorization

from .ore_algebra import OreAlgebra_generic
from .ore_operator_1_1 import UnivariateOreOperatorOverUnivariateRing
from .ore_operator import UnivariateOreOperator
from .tools import q_log, make_factor_iterator, shift_factor, _rec2list, _power_series_solutions


#############################################################################################################

class UnivariateQRecurrenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
    Element of an Ore algebra K(x)[S], where S is the shift x->q*x for some q in K.
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        if isinstance(f, (tuple, list)):

            r = self.order()
            R = self.parent().base_ring()
            _, q = self.parent().is_Q()
            c = self.numerator().coefficients(sparse=False)
            d = self.denominator()

            def fun(n):
                if f[n + r] is None:
                    return None
                else:
                    try:
                        qn = q**n
                        return sum( c[i](qn)*f[n + i] for i in range(r + 1) )/d(qn)
                    except:
                        return None

            return type(f)(fun(n) for n in range(len(f) - r))

        R = self.parent()
        x = R.base_ring().gen()
        qx = R.sigma()(x)
        if "action" not in kwargs:
            kwargs["action"] = lambda p : p.subs({x:qx})

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_J(self, alg):  # q2j
        """
        Returns a q-differential operator which annihilates every power series (about the origin)
        whose coefficient sequence is annihilated by ``self``.

        The output operator may not be minimal.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_J()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the q-derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
            sage: A.<Qn> = OreAlgebra(Rn, 'Qn', q=2)
            sage: B.<Jx> = OreAlgebra(Rx, 'Jx', q=2)
            sage: (Qn - 1).to_J(B)
            (-2*x + 1)*Jx - 1
            sage: ((n+1)*Qn - 1).to_J(B)
            2*x*Jx^2 + (-4*x + 4)*Jx - 2
            sage: (x*Jx-1).to_Q(A).to_J(B) % (x*Jx - 1)
            0

        """
        R = self.base_ring()
        K = R.base_ring()
        x, q = self.parent().is_Q()
        one = R.one()

        if isinstance(alg, str):
            alg = self.parent().change_var_sigma_delta(alg, {x:q*x}, {x:one})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_J() or \
             alg.base_ring().base_ring() is not K or K(alg.is_J()[1]) != K(q):
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field()
        x, q = alg.is_J()
        alg = alg.change_ring(R)

        Q = alg(~x)
        out = alg.zero()
        coeffs = self.numerator().coefficients(sparse=False)
        x_pows = {0 : alg.one(), 1 : ((q - R.one())*x)*alg.gen() + alg.one()}

        for i in range(len(coeffs)):
            term = alg.zero()
            c = coeffs[i].coefficients(sparse=False)
            for j in range(len(c)):
                if j not in x_pows:
                    x_pows[j] = x_pows[j - 1]*x_pows[1]
                term += c[j] * x_pows[j]
            out += term*(Q**i)

        return (alg.gen()**(len(coeffs)-1))*out.numerator().change_ring(alg.base_ring())

    def to_list(self, init, n, start=0, append=False, padd=False):
        r"""
        Computes the terms of some sequence annihilated by ``self``.

        INPUT:

        - ``init`` -- a vector (or list or tuple) of initial values.
          The components must be elements of ``self.base_ring().base_ring().fraction_field()``.
          If the length is more than ``self.order()``, we do not check whether the given
          terms are consistent with ``self``.
        - ``n`` -- desired number of terms.
        - ``start`` (optional) -- index of the sequence term which is represented
          by the first entry of ``init``. Defaults to zero.
        - ``append`` (optional) -- if ``True``, the computed terms are appended
          to ``init`` list. Otherwise (default), a new list is created.
        - ``padd`` (optional) -- if ``True``, the vector of initial values is implicitly
          prolonged to the left (!) by zeros if it is too short. Otherwise (default),
          the method raises a ``ValueError`` if ``init`` is too short.

        OUTPUT:

        A list of ``n`` terms whose `k` th component carries the sequence term with
        index ``start+k``.
        Terms whose calculation causes an error are represented by ``None``.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = QQ['x']; A.<Qx> = OreAlgebra(R, 'Qx', q=3)
            sage: (Qx^2-x*Qx + 1).to_list([1,1], 10)
            [1, 1, 0, -1, -9, -242, -19593, -4760857, -3470645160, -7590296204063]
            sage: (Qx^2-x*Qx + 1)(_)
            [0, 0, 0, 0, 0, 0, 0, 0]

        """
        _, q = self.parent().is_Q()
        return _rec2list(self, init, n, start, append, padd, lambda n: q**n)

    def annihilator_of_sum(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite sums `\sum_{k=0}^n a_k`
        where `a_n` runs through the sequences annihilated by ``self``.

        The output operator is not necessarily of smallest possible order.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = ZZ['q'].fraction_field()['x']
            sage: A.<Qx> = OreAlgebra(R, 'Qx')
            sage: ((x+1)*Qx - x).annihilator_of_sum()
            (q*x + 1)*Qx^2 + (-2*q*x - 1)*Qx + q*x

        """
        A = self.parent()
        return self.map_coefficients(A.sigma())*(A.gen() - A.one())

    def annihilator_of_composition(self, a, solver=None):
        r"""
        Returns an operator `L` which annihilates all the sequences `f(a(n))`
        where `f` runs through the functions annihilated by ``self``.

        The output operator is not necessarily of smallest possible order.

        INPUT:

        - ``a`` -- a polynomial `u*x+v` where `x` is the generator of the base ring,
          `u` and `v` are integers.
        - ``solver`` (optional) -- a callable object which applied to a matrix
          with polynomial entries returns its kernel.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = QQ['x']
            sage: A.<Qx> = OreAlgebra(R, 'Qx', q=3)
            sage: L = (x+3)*Qx^2 - (5*x+3)*Qx + 2*x-1
            sage: data = L.to_list([1,2], 11)
            sage: data
            [1, 2, 15/4, 115/12, 1585/48, 19435/144, 2387975/4032, 188901875/70848, 488427432475/40336128, 1461633379710215/26500836096, 14580926901721431215/57983829378048]
            sage: L2 = L.annihilator_of_composition(2*x)
            sage: L2.to_list([1,15/4], 5)
            [1, 15/4, 1585/48, 2387975/4032, 488427432475/40336128]
            sage: Lrev = L.annihilator_of_composition(10 - x)
            sage: Lrev.to_list([data[10], data[9]], 11)
            [14580926901721431215/57983829378048, 1461633379710215/26500836096, 488427432475/40336128, 188901875/70848, 2387975/4032, 19435/144, 1585/48, 115/12, 15/4, 2, 1]

        """
        # ugly code duplication: the following is more or less the same as
        # UnivariateRecurrenceOperatorOverUnivariateRing.annihilator_of_composition :-(

        A = self.parent()

        if a in ZZ:
            # a is constant => f(a) is constant => Q-1 kills it
            return A.gen() - A.one()

        R = ZZ[A.base_ring().gen()]

        try:
            a = R(a)
        except:
            raise ValueError("argument has to be of the form u*x+v where u,v are integers")

        if a.degree() > 1:
            raise ValueError("argument has to be of the form u*x+v where u,v are integers")

        try:
            u = ZZ(a[1])
            v = ZZ(a[0])
        except:
            raise ValueError("argument has to be of the form u*x+v where u,v are rational")

        A = A.change_ring(A.base_ring().fraction_field())
        L = A(self)
        s = A.sigma()
        r = self.order()
        x, q = A.is_Q()

        # special treatment for easy cases
        if v != 0:
            L = self.map_coefficients(lambda p: s(p, v))
            return L if u == 1 else L.annihilator_of_composition(u*x)
        elif u == 1:
            return self
        elif u < 0:
            c = [ p(q**(-r)/x) for p in self.coefficients(sparse=False) ]
            c.reverse()
            return A(c).numerator().annihilator_of_composition(-u*x)

        # now a = u*x where u > 1
        from sage.matrix.constructor import Matrix
        if solver is None:
            solver = A._solver()

        p = A.one()
        Qu = A.gen()**u  # possible improvement: multiplication matrix.
        mat = [ p.coefficients(sparse=False, padd=r) ]
        sol = []

        while len(sol) == 0:

            p = (Qu*p) % L
            mat.append( p.coefficients(sparse=False, padd=r) )
            sol = solver(Matrix(mat).transpose())

        return self.parent()(list(sol[0])).map_coefficients(lambda p: p(x**u))

    def spread(self, p=0):

        op = self.normalize()
        A = op.parent()
        R = A.base_ring()
        sigma = A.change_ring(R.change_ring(R.base_ring().fraction_field())).sigma()
        s = []
        r = op.order()
        _, q = A.is_Q()

        if op.order() == 0:
            return []
        elif op[0].is_zero():
            return [infinity]

        if R.is_field():
            R = R.ring()  # R = k[x]
            R = R.change_ring(R.base_ring().fraction_field())

        try:
            # first try to use shift factorization. this seems to be more efficient in most cases.
            all_facs = [sigma(u, -1) for u, _ in shift_factor(sigma(op[0].gcd(p), r)*op[r], 1, q)]
            tc = [ u[1:] for _, u in shift_factor(prod(all_facs)*sigma(op[0].gcd(p), r), 1, q) ]
            lc = [ u[1:] for _, u in shift_factor(prod(all_facs)*op[r], 1, q) ]
            for u, v in zip(tc, lc):
                s.extend(j[0] - i[0] for i in u for j in v)
            return sorted(set(s))
        except:
            pass

        K = PolynomialRing(R.base_ring(), 'y').fraction_field()  # F(k[y])
        R = R.change_ring(K)  # FF(k[y])[x]

        y = R(K.gen())
        x, q = op.parent().is_Q()
        x = R(x)
        q = K(q)

        s = []
        r = op.order()
        for p, _ in (R(op[r])(x*(q**(-r))).resultant(gcd(R(p), R(op[0]))(x*y))).numerator().factor():
            if p.degree() == 1:
                try:
                    s.append(q_log(q, K(-p[0]/p[1])))
                except:
                    pass

        s = list(set(s))  # remove duplicates
        s.sort()
        return s

    spread.__doc__ = UnivariateOreOperatorOverUnivariateRing.spread.__doc__

    def __to_J_literally(self, gen='J'):
        r"""
        Rewrites ``self`` in terms of `J`
        """
        A = self.parent()
        R = A.base_ring()
        x, q = A.is_Q()
        one = R.one()
        A = A.change_var_sigma_delta(gen, {x:q*x}, {x:one})

        if self.is_zero():
            return A.zero()

        Q = (q - 1)*x*A.gen() + 1
        Q_pow = A.one()
        c = self.coefficients(sparse=False)
        out = A(R(c[0]))

        for i in range(self.order()):

            Q_pow *= Q
            out += R(c[i + 1])*Q_pow

        return out

    def _coeff_list_for_indicial_polynomial(self):
        return self.__to_J_literally().coefficients(sparse=False)

    def _denominator_bound(self):

        A, R, _, L = self._normalize_base_ring()
        x = R.gen()

        # primitive factors (anything but powers of x)
        u = UnivariateOreOperatorOverUnivariateRing._denominator_bound(L)

        quo, rem = R(u).quo_rem(x)
        while rem.is_zero():
            quo, rem = quo.quo_rem(x)

        # special factors (powers of x)
        e = 0
        for q, _ in L.indicial_polynomial(x).factor():
            if q.degree() == 1:
                try:
                    e = min(e, ZZ(-q[0] / q[1]))
                except (TypeError, ValueError):
                    pass

        return Factorization([(quo*x + rem, 1), (x, -e)])

    def _powerIndicator(self):
        return self.coefficients(sparse=False)[0]

    def _local_data_at_special_points(self):
        r"""
        Returns information about the local behaviour of this operator's solutions at x=0 and
        at x=infinity.

        The output is a list of all tuples ``(gamma, phi, beta, alpha)`` such that for every
        q-hypergeometric solution `f` of this operator (over the same constant field) there
        is a tuple such that
        `f(q*x)/f(x) = phi * x^gamma * rat(q*x)/rat(x) * \prod_m (1-a_m*x)^{e_m}`
        with `\sum_m e_m = beta` and `q^(deg(num(rat)) - deg(den(rat)))*\prod_m (-a_m)^{e_m} = alpha`.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = QQ['x']; A.<Qx> = OreAlgebra(R, q=2)
            sage: ((2*x+3)*Qx - (8*x+3)).lclm(Qx-1)._local_data_at_special_points()
            [(0, 2, 0, 2), (0, 2, 0, 1/2), (0, 1, 0, 4), (0, 1, 0, 1)]

        """

        Q = self.parent().gen()
        x, qq = self.parent().is_Q()
        factors = make_factor_iterator(x.parent(), multiplicities=False)

        out = []
        for gamma, poly in self.newton_polygon(x):
            if gamma in ZZ:
                for p in factors(poly):
                    if p.degree() == 1:
                        phi = -p[0]/p[1]
                        L = self.symmetric_product(phi*x**max(-gamma, 0)*Q - x**max(gamma, 0))
                        for beta, qoly in L.newton_polygon(~x):
                            if beta in ZZ:
                                for q in factors(qoly(x*qq**beta) + (qq**beta-1)*qoly[0]):  # is this right?
                                    if q.degree() == 1 and q[0] != 0:
                                        out.append((-gamma, phi, beta, -q[0]/q[1]))

        return out


####################################################################################

class UnivariateQDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
    Element of an Ore algebra K(x)[J], where J is the Jackson q-differentiation J f(x) = (f(q*x) - f(x))/(q*(x-1))
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        A = self.parent()
        x, q = A.is_J()
        qx = A.sigma()(x)
        if "action" not in kwargs:
            kwargs["action"] = lambda p : (p.subs({x:qx}) - p)/(x*(q-1))

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_Q(self, alg):  # j2q
        """
        Returns a q-recurrence operator which annihilates the coefficient sequence
        of every power series (about the origin) annihilated by ``self``.

        The output operator may not be minimal.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_Q() == self.parent().is_J()``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the q-shift with respect to ``self.base_ring().gen()``.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
            sage: A.<Jx> = OreAlgebra(Rx, 'Jx', q=2)
            sage: B.<Qn> = OreAlgebra(Rn, 'Qn', q=2)
            sage: (Jx - 1).to_Q(B)
            (2*n - 1)*Qn - 1
            sage: ((x+1)*Jx - 1).to_Q(B)
            (4*n - 1)*Qn^2 + (2*n - 2)*Qn
            sage: (n*Qn-1).to_J(A).to_Q(B) % (n*Qn - 1)
            0
        """
        R = self.base_ring()
        K = R.base_ring()
        x, q = self.parent().is_J()

        if isinstance(alg, str):
            alg = self.parent().change_var_sigma_delta(alg, {x:q*x}, {})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_Q() or \
             alg.base_ring().base_ring() is not R.base_ring() or K(alg.is_Q()[1]) != K(q) :
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field()
        x, q = alg.is_Q()
        alg = alg.change_ring(R)

        Q = alg.gen()
        J = ((q*x - R.one())/(q - R.one()))*Q
        J_pow = alg.one()
        out = alg.zero()
        coeffs = self.numerator().coefficients(sparse=False)
        d = max( c.degree() for c in coeffs )

        for i in range(len(coeffs)):
            if i > 0:
                J_pow *= J
            c = coeffs[i].padded_list(d + 1)
            c.reverse()
            out += alg(list(map(R, c))) * J_pow

        return ((q-1)**(len(coeffs)-1)*out).numerator().change_ring(alg.base_ring())

    def annihilator_of_integral(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite `q`-integrals `\int_q f`
        where `f` runs through the functions annihilated by ``self``.

        The output operator is not necessarily of smallest possible order.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = ZZ['q'].fraction_field()['x']
            sage: A.<Jx> = OreAlgebra(R, 'Jx')
            sage: ((x-1)*Jx - 2*x).annihilator_of_integral()
            (x - 1)*Jx^2 - 2*x*Jx
            sage: _.annihilator_of_associate(Jx)
            (x - 1)*Jx - 2*x

        """
        return self*self.parent().gen()

    def power_series_solutions(self, n=5):
        r"""
        Computes the first few terms of the power series solutions of this operator.

        The method raises an error if Sage does not know how to factor univariate polynomials
        over the base ring's base ring.

        The base ring has to have characteristic zero.

        INPUT:

        - ``n`` -- minimum number of terms to be computed

        OUTPUT:

        A list of power series of the form `x^{\alpha} + ...` with pairwise distinct
        exponents `\alpha` and coefficients in the base ring's base ring's fraction field.
        All expansions are computed up to order `k` where `k` is obtained by adding the
        maximal `\alpha` to the maximum of `n` and the order of ``self``.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R.<x> = QQ['x']
            sage: A.<Jx> = OreAlgebra(R, 'Jx', q=2)
            sage: (Jx-1).lclm((1-x)*Jx-1).power_series_solutions()
            [x^2 + x^3 + 3/5*x^4 + 11/35*x^5 + O(x^6), 1 + x - 2/7*x^3 - 62/315*x^4 - 146/1395*x^5 + O(x^6)]

        """
        _, q = self.parent().is_J()
        return _power_series_solutions(self, self.to_Q('Q'), n, lambda n: q**n)

    def __to_Q_literally(self, gen='Q'):
        r"""
        This computes the q-recurrence operator which corresponds to ``self`` in the sense
        that `J` is rewritten to `1/(q-1)/x * (Q - 1)`
        """
        x, q = self.parent().is_J()

        alg = self.parent().change_var_sigma_delta(gen, {x:q*x}, {})
        alg = alg.change_ring(self.base_ring().fraction_field())

        if self.is_zero():
            return alg.zero()

        J = ~(q-1)*(~x)*(alg.gen() - alg.one())
        J_k = alg.one()
        R = alg.base_ring()
        c = self.coefficients(sparse=False)
        out = alg(R(c[0]))

        for i in range(self.order()):

            J_k *= J
            out += R(c[i + 1])*J_k

        return out.numerator().change_ring(R.ring())

    def spread(self, p=0):
        return self.__to_Q_literally().spread(p)

    spread.__doc__ = UnivariateOreOperatorOverUnivariateRing.spread.__doc__

    def _coeff_list_for_indicial_polynomial(self):
        return self.coefficients(sparse=False)

    def _denominator_bound(self):
        return self.__to_Q_literally()._denominator_bound()

    def symmetric_product(self, other, solver=None):

        if not isinstance(other, UnivariateOreOperator):
            raise TypeError("unexpected argument in symmetric_product")

        if self.parent() != other.parent():
            A, B = canonical_coercion(self, other)
            return A.symmetric_product(B, solver=solver)

        A = self.__to_Q_literally()
        B = other.__to_Q_literally()

        C = A.symmetric_product(B, solver=solver)._normalize_base_ring()[-1]
        C = C._UnivariateQRecurrenceOperatorOverUnivariateRing__to_J_literally(str(self.parent().gen()))

        try:
            return self.parent()(C.numerator().coefficients(sparse=False))
        except:
            return C

    symmetric_product.__doc__ = UnivariateOreOperator.symmetric_product.__doc__
