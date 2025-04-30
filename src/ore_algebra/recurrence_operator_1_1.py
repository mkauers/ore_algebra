"""
Univariate recurrence operators over univariate rings
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

from functools import reduce

from sage.arith.misc import GCD as gcd
from sage.functions.all import floor
from sage.misc.misc_c import prod
from sage.misc.cachefunc import cached_method
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.infinity import infinity
from sage.rings.qqbar import QQbar
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.structure.element import canonical_coercion

from .tools import shift_factor, _vect_val_fct, _vect_elim_fct, roots_at_integer_distance, _rec2list
from .ore_algebra import OreAlgebra_generic
from .ore_operator import UnivariateOreOperator
from .ore_operator_1_1 import UnivariateOreOperatorOverUnivariateRing
from .generalized_series import GeneralizedSeriesMonoid, _generalized_series_shift_quotient, _binomial


##########################################################################

class UnivariateRecurrenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
    Element of an Ore algebra K(x)[S], where S is the shift x->x+1.
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):
        if isinstance(f, (tuple, list)):

            r = self.order()
            c = self.numerator().coefficients(sparse=False)
            d = self.denominator()

            def fun(n):
                if f[n + r] is None:
                    return None
                else:
                    try:
                        return sum( c[i](n)*f[n + i] for i in range(r + 1) )/d(n)
                    except (ValueError, TypeError, KeyError):
                        return None

            return type(f)(fun(n) for n in range(len(f) - r))

        if "action" not in kwargs:
            x = self.parent().base_ring().gen()

            def shift(p):
                try:
                    return p.subs({x: x + 1})
                except (TypeError, AttributeError, ValueError):
                    return p(x + 1)

            kwargs["action"] = shift

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_D(self, alg): # s2d
        """
        Returns a differential operator which annihilates every power series whose
        coefficient sequence is annihilated by ``self``.
        The output operator may not be minimal.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_D()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the standard derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
          sage: A.<Sn> = OreAlgebra(Rn, 'Sn')
          sage: B.<Dx> = OreAlgebra(Rx, 'Dx')
          sage: (Sn - 1).to_D(B)
          (-x + 1)*Dx - 1
          sage: ((n+1)*Sn - 1).to_D(B)
          x*Dx^2 + (-x + 1)*Dx - 1
          sage: (x*Dx-1).to_S(A).to_D(B)
          x*Dx - 1

        """
        R = self.base_ring()
        x = R.gen()
        one = R.one()

        if isinstance(alg, str):
            alg = self.parent().change_var_sigma_delta(alg, {}, {x:one})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_D():
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field()
        x = R.gen()
        alg_theta = alg.change_var_sigma_delta('T', {}, {x:x}).change_ring(R)

        S = alg_theta(~x)
        out = alg_theta.zero()
        coeffs = self.numerator().coefficients(sparse=False)

        for i in range(len(coeffs)):
            out += alg_theta([R(p) for p in coeffs[i].coefficients(sparse=False)])*(S**i)

        out = out.numerator().change_ring(alg.base_ring()).to_D(alg)
        out = alg.gen()**(len(coeffs)-1)*out

        return out

    def to_F(self, alg): # s2delta
        """
        Returns the difference operator corresponding to ``self``

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
          sage: A.<Sx> = OreAlgebra(R, 'Sx')
          sage: (Sx^4).to_F(OreAlgebra(R, 'Fx'))
          Fx^4 + 4*Fx^3 + 6*Fx^2 + 4*Fx + 1
          sage: (Sx^4).to_F('Fx').to_S(A)
          Sx^4

        """
        R = self.base_ring()
        x = R.gen()
        one = R.one()

        if isinstance(alg, str):
            alg = self.parent().change_var_sigma_delta(alg, {x:x+one}, {x:one})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_F():
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        delta = alg.gen() + alg.one()
        delta_k = alg.one()
        R = alg.base_ring()
        c = self.coefficients(sparse=False)
        out = alg(R(c[0]))

        for i in range(self.order()):

            delta_k *= delta
            out += R(c[i + 1])*delta_k

        return out

    def to_T(self, alg):
        r"""
        Returns a differential operator, expressed in terms of the Euler derivation,
        which annihilates every power series (about the origin) whose coefficient
        sequence is annihilated by ``self``.
        The output operator may not be minimal.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_T()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the Euler derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
          sage: A.<Sn> = OreAlgebra(Rn, 'Sn')
          sage: B.<Tx> = OreAlgebra(Rx, 'Tx')
          sage: (Sn - 1).to_T(B)
          (-x + 1)*Tx - x
          sage: ((n+1)*Sn - 1).to_T(B)
          Tx^2 - x*Tx - x
          sage: (x*Tx-1).to_S(A).to_T(B)
          x*Tx^2 + (x - 1)*Tx

        """
        return self.to_D('D').to_T(alg)

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
           sage: R = ZZ['x']['n']; x = R('x'); n = R('n')
           sage: A.<Sn> = OreAlgebra(R, 'Sn')
           sage: L = ((n+2)*Sn^2 - x*(2*n+3)*Sn + (n+1))
           sage: L.to_list([1, x], 5)
           [1, x, (3*x^2 - 1)/2, (5*x^3 - 3*x)/2, (35*x^4 - 30*x^2 + 3)/8]
           sage: polys = L.to_list([1], 5, padd=True)
           sage: polys
           [1, x, (3*x^2 - 1)/2, (5*x^3 - 3*x)/2, (35*x^4 - 30*x^2 + 3)/8]
           sage: L.to_list([polys[3], polys[4]], 8, start=3)
           [(5*x^3 - 3*x)/2,
            (35*x^4 - 30*x^2 + 3)/8,
            (63*x^5 - 70*x^3 + 15*x)/8,
            (231*x^6 - 315*x^4 + 105*x^2 - 5)/16,
            (429*x^7 - 693*x^5 + 315*x^3 - 35*x)/16,
            (6435*x^8 - 12012*x^6 + 6930*x^4 - 1260*x^2 + 35)/128,
            (12155*x^9 - 25740*x^7 + 18018*x^5 - 4620*x^3 + 315*x)/128,
            (46189*x^10 - 109395*x^8 + 90090*x^6 - 30030*x^4 + 3465*x^2 - 63)/256]
           sage: ((n-5)*Sn - 1).to_list([1], 10)
           [1, 1/-5, 1/20, 1/-60, 1/120, -1/120, None, None, None, None]

        """
        return _rec2list(self, init, n, start, append, padd, ZZ)

    def forward_matrix_bsplit(self, n, start=0):
        r"""
        Uses division-free binary splitting to compute a product of ``n``
        consecutive companion matrices of ``self``.

        If ``self`` annihilates some sequence `c` of order `r`, this
        allows rapidly computing `c_n, \ldots, c_{n+r-1}` (or just `c_n`)
        without generating all the intermediate values.

        INPUT:

        - ``n`` -- desired number of terms to move forward
        - ``start`` (optional) -- starting index. Defaults to zero.

        OUTPUT:

        A pair `(M, Q)` where `M` is an `r` by `r` matrix and `Q`
        is a scalar, such that `M / Q` is the product of the companion
        matrix at `n` consecutive indices.

        We have `Q [c_{s+n}, \ldots, c_{s+r-1+n}]^T = M [c_s, c_{s+1}, \ldots, c_{s+r-1}]^T`,
        where `s` is the initial position given by ``start``.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: R = ZZ
            sage: Rx.<x> = R[]
            sage: Rxk.<k> = Rx[]
            sage: Rxks = OreAlgebra(Rxk, 'Sk')
            sage: ann = Rxks([1+k, -3*x - 2*k*x, 2+k])
            sage: initial = Matrix([[1], [x]])
            sage: M, Q = ann.forward_matrix_bsplit(5)
            sage: (M * initial).change_ring(QQ['x']) / Q
            [               63/8*x^5 - 35/4*x^3 + 15/8*x]
            [231/16*x^6 - 315/16*x^4 + 105/16*x^2 - 5/16]

            sage: Matrix([[legendre_P(5, x)], [legendre_P(6, x)]])
            [               63/8*x^5 - 35/4*x^3 + 15/8*x]
            [231/16*x^6 - 315/16*x^4 + 105/16*x^2 - 5/16]


            sage: Sk = Rxks.gen()
            sage: (Sk^2 - 1).forward_matrix_param_rectangular(1, 10)
            (
            [1 0]
            [0 1], 1
            )

        TODO: this should detect if the base coefficient ring is QQ (etc.)
        and then switch to ZZ (etc.) internally.
        """
        from sage.matrix.matrix_space import MatrixSpace
        n = ZZ(n)
        start = ZZ(start) # exact division below fails if n or start are in QQ, as reported by Clemens Hofstadler 2018-03-14.
        assert n >= 0
        r = self.order()
        scalar_ring = self.base_ring().base_ring()
        matrix_ring = MatrixSpace(scalar_ring, r, r)
        coeffs = list(self)
        def bsplit(a, b):
            if b - a == 0:
                return matrix_ring.one(), scalar_ring.one()
            elif b - a == 1:
                M = matrix_ring()
                Q = coeffs[r](a)
                for i in range(r-1):
                    M[i, i+1] = Q
                for i in range(r):
                    M[r-1, i] = -coeffs[i](a)
                return M, Q
            else:
                m = a + (b - a) // 2
                M1, Q1 = bsplit(a, m)
                M2, Q2 = bsplit(m, b)
                return M2 * M1, Q2 * Q1
        return bsplit(start, start + n)

    def _delta_matrix(self, m):

        from sage.matrix.matrix_space import MatrixSpace

        m = ZZ(m) # exact division below fails if n or start are in QQ, as reported by Clemens Hofstadler 2018-03-14.

        r = self.order()

        delta_ring = self.base_ring()
        delta_matrix_ring = MatrixSpace(delta_ring, r, r)
        k = delta_ring.gen()

        coeffs = list(self)

        def bsplit(a, b, shift):
            if b - a == 0:
                return delta_matrix_ring.one(), delta_ring.one()
            elif b - a == 1:
                M = delta_matrix_ring()
                Q = coeffs[r](k + shift + a)
                for i in range(r-1):
                    M[i, i+1] = Q
                for i in range(r):
                    M[r-1, i] = -coeffs[i](k + shift + a)
                return M, Q
            else:
                m = a + (b - a) // 2
                M1, Q1 = bsplit(a, m, shift)
                M2, Q2 = bsplit(m, b, shift)
                return M2 * M1, Q2 * Q1

        delta_M1, delta_Q1 = bsplit(0, m, m)
        delta_M2, delta_Q2 = bsplit(0, m, 0)

        delta_M = delta_M1 - delta_M2
        delta_Q = delta_Q1 - delta_Q2

        return delta_M, delta_Q

    def forward_matrix_param_rectangular(self, value, n, start=0, m=None):
        r"""
        Assuming the coefficients of self are in `R[x][k]`,
        computes the nth forward matrix with the parameter `x`
        evaluated at ``value``, using rectangular splitting
        with a step size of `m`.

        TESTS::

            sage: from ore_algebra import *
            sage: R = ZZ
            sage: Rx = R['x']; x = Rx.gen()
            sage: Rxk = Rx['k']; k = Rxk.gen()
            sage: Rxks = OreAlgebra(Rxk, 'Sk')
            sage: V = QQ
            sage: Vks = OreAlgebra(V['k'], 'Sk')
            sage: for i in range(1000): # long time (1.9 s)
            ....:     A = Rxks.random_element(randrange(1,4))
            ....:     r = A.order()
            ....:     v = V.random_element()
            ....:     initial = [V.random_element() for i in range(r)]
            ....:     start = randrange(0,5)
            ....:     n = randrange(0,30)
            ....:     m = randrange(0,10)
            ....:     B = Vks(list(A.polynomial()(x=v)))
            ....:     M, Q = A.forward_matrix_param_rectangular(v, n, m=m, start=start)
            ....:     if Q != 0:
            ....:         V1 = M * Matrix(initial).transpose() / Q
            ....:         values = B.to_list(initial, n + r, start)
            ....:         V2 = Matrix(values[-r:]).transpose()
            ....:         if V1 != V2:
            ....:             raise ValueError

        """
        from sage.matrix.matrix_space import MatrixSpace

        assert n >= 0
        r = self.order()

        indexed_ring = self.base_ring()
        parametric_ring = indexed_ring.base_ring()
        scalar_ring = parametric_ring.base_ring()

        coeffs = list(self)
        param_degree = max(d.degree() for c in coeffs for d in c)

        # Step size
        if m is None:
            m = floor(n ** 0.25)
        m = max(m, 1)
        m = min(m, n)

        delta_M, delta_Q = self._delta_matrix(m)

        # Precompute all needed powers of the parameter value
        # TODO: tighter degree bound (by inspecting the matrices)
        eval_degree = m * param_degree
        num_powers = eval_degree + 1

        power_table = [0] * num_powers
        for i in range(num_powers):
            if i == 0:
                power_table[i] = value ** 0
            elif i == 1:
                power_table[i] = value
            elif i % 2 == 0:
                power_table[i] = power_table[i // 2] * power_table[i // 2]
            else:
                power_table[i] = power_table[i - 1] * power_table[1]

        def evaluate_using_power_table(poly):
            if not poly:
                return scalar_ring.zero()
            s = poly[0]
            for i in range(1, poly.degree() + 1):
                s += poly[i] * power_table[i]
            return s

        # TODO: check if transposing the polynomials gives better
        # performance

        # TODO: if the denominator does not depend on the parameter,
        # we might want to avoid the ring of the parameter value for
        # the denominator
        value_ring = (scalar_ring.zero() * value).parent()
        value_matrix_ring = MatrixSpace(value_ring, r, r)

        value_M = value_matrix_ring.one()
        value_Q = scalar_ring.one()

        def baby_steps(VM, VQ, a, b):
            for j in range(a, b):
                M = value_matrix_ring()
                Q = evaluate_using_power_table(coeffs[r](start + j))
                for i in range(r-1):
                    M[i, i+1] = Q
                for i in range(r):
                    M[r-1, i] = evaluate_using_power_table(-coeffs[i](start + j))
                VM = M * VM
                VQ = Q * VQ
            return VM, VQ

        # Baby steps
        value_M, value_Q = baby_steps(value_M, value_Q, 0, m)

        if m != 0:
            step_M = value_M
            step_Q = value_Q

            # Giant steps
            for j in range(m, n - m + 1, m):
                v = start + j - m
                M = value_matrix_ring()
                Q = evaluate_using_power_table(delta_Q(v))
                for row in range(r):
                    for col in range(r):
                        M[row, col] = evaluate_using_power_table(delta_M[row, col](v))
                step_M = step_M + M
                step_Q = step_Q + Q
                value_M = step_M * value_M
                value_Q = step_Q * value_Q

            # Fill in if n is not a multiple of m
            remainder = n % m
            value_M, value_Q = baby_steps(value_M, value_Q, n-remainder, n)

        return value_M, value_Q

    def annihilator_of_sum(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite sums `\sum_{k=0}^n a_k`
        where `a_n` runs through the sequences annihilated by ``self``.
        The output operator is not necessarily of smallest possible order.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ['x']
           sage: A.<Sx> = OreAlgebra(R, 'Sx')
           sage: ((x+1)*Sx - x).annihilator_of_sum() # constructs L such that L(H_n) == 0
           (x + 2)*Sx^2 + (-2*x - 3)*Sx + x + 1

        """
        A = self.parent()
        return self.map_coefficients(A.sigma())*(A.gen() - A.one())

    def annihilator_of_composition(self, a, solver=None):
        r"""
        Returns an operator `L` which annihilates all the sequences `f(floor(a(n)))`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order.

        INPUT:

        - ``a`` -- a polynomial `u*x+v` where `x` is the generator of the base ring,
          `u` and `v` are integers or rational numbers. If they are rational,
          the base ring of the parent of ``self`` must contain ``QQ``.
        - ``solver`` (optional) -- a callable object which applied to a matrix
          with polynomial entries returns its kernel.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = QQ['x']
          sage: A.<Sx> = OreAlgebra(R, 'Sx')
          sage: ((2+x)*Sx^2-(2*x+3)*Sx+(x+1)).annihilator_of_composition(2*x+5)
          (16*x^3 + 188*x^2 + 730*x + 936)*Sx^2 + (-32*x^3 - 360*x^2 - 1340*x - 1650)*Sx + 16*x^3 + 172*x^2 + 610*x + 714
          sage: ((2+x)*Sx^2-(2*x+3)*Sx+(x+1)).annihilator_of_composition(1/2*x)
          (x^2 + 11*x + 30)*Sx^6 + (-3*x^2 - 25*x - 54)*Sx^4 + (3*x^2 + 17*x + 26)*Sx^2 - x^2 - 3*x - 2
          sage: ((2+x)*Sx^2-(2*x+3)*Sx+(x+1)).annihilator_of_composition(100-x)
          (-x + 99)*Sx^2 + (2*x - 199)*Sx - x + 100
        """

        A = self.parent()

        if a in QQ:
            # a is constant => f(a) is constant => S-1 kills it
            return A.gen() - A.one()

        K = a.parent().base_ring()
        R = K[A.base_ring().gen()]

        try:
            a = R(a)
        except (TypeError, ValueError):
            raise ValueError("argument has to be of the form u*x+v where u,v are rational")

        if a.degree() > 1:
            raise ValueError("argument has to be of the form u*x+v where u,v are rational")

        try:
            u = QQ(a[1])
            v = QQ(a[0])
        except (TypeError, ValueError):
            raise ValueError("argument has to be of the form u*x+v where u,v are rational")

        r = self.order()
        x = A.base_ring().gen()

        # special treatment for easy cases
        w = u.denominator().abs()
        if w > 1:
            w = w.lcm(v.denominator()).abs()
            p = self.polynomial()(A.associated_commutative_algebra().gen()**w)
            q = p = A(p.map_coefficients(lambda f: f(x/w)))
            for i in range(1, w):
                q = q.lclm(p.annihilator_of_composition(x - i), solver=solver)
            return q.annihilator_of_composition(w*u*x + w*v)
        elif v != 0:
            s = A.sigma()
            v = v.floor()
            L = self.map_coefficients(lambda p: s(p, v))
            return L if u == 1 else L.annihilator_of_composition(u*x)
        elif u == 1:
            return self
        elif u < 0:
            c = [ p(-r - x) for p in self.coefficients(sparse=False) ]
            c.reverse()
            return A(c).annihilator_of_composition(-u*x)

        # now a = u*x where u > 1 is an integer.
        u = u.numerator()
        from sage.matrix.constructor import Matrix
        A = A.change_ring(A.base_ring().fraction_field())
        if solver is None:
            solver = A._solver()
        L = A(self)

        p = A.one()
        Su = A.gen()**u # possible improvement: multiplication matrix.
        mat = [ p.coefficients(sparse=False, padd=r) ]
        sol = []

        while len(sol) == 0:

            p = (Su*p) % L
            mat.append( p.coefficients(sparse=False, padd=r) )
            sol = solver(Matrix(mat).transpose())

        return self.parent()(list(sol[0])).map_coefficients(lambda p: p(u*x))

    def annihilator_of_interlacing(self, *other):
        r"""
        Returns an operator `L` which annihilates any sequence which can be
        obtained by interlacing sequences annihilated by ``self`` and the
        operators given in the arguments.

        More precisely, if ``self`` and the operators given in the arguments are
        denoted `L_1,L_2,\dots,L_m`, and if `f_1(n),\dots,f_m(n)` are some
        sequences such that `L_i` annihilates `f_i(n)`, then the output operator
        `L` annihilates sequence
        `f_1(0),f_2(0),\dots,f_m(0),f_1(1),f_2(1),\dots,f_m(1),\dots`, the
        interlacing sequence of `f_1(n),\dots,f_m(n)`.

        The output operator is not necessarily of smallest possible order.

        The ``other`` operators must be coercible to the parent of ``self``.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = QQ['x']
          sage: A.<Sx> = OreAlgebra(R, 'Sx')
          sage: (x*Sx - (x+1)).annihilator_of_interlacing(Sx - (x+1), Sx + 1)
          (x^3 + 17/2*x^2 + 5/2*x - 87/2)*Sx^9 + (-1/3*x^4 - 11/2*x^3 - 53/2*x^2 - 241/6*x + 14)*Sx^6 + (7/2*x^2 + 67/2*x + 205/2)*Sx^3 + 1/3*x^4 + 13/2*x^3 + 77/2*x^2 + 457/6*x + 45
        """
        A = self.parent()
        A = A.change_ring(A.base_ring().fraction_field())
        ops = [A(self)] + list(map(A, list(other)))
        S_power = A.associated_commutative_algebra().gen()**len(ops)
        x = A.base_ring().gen()
        xQ = QQ[x].gen()

        for i in range(len(ops)):
            ops[i] = A(ops[i].polynomial()(S_power)\
                       .map_coefficients(lambda p: p(x/len(ops))))\
                       .annihilator_of_composition(xQ - i)

        return self.parent()(reduce(lambda p, q: p.lclm(q), ops).numerator())

    def _coeff_list_for_indicial_polynomial(self):
        d = self.degree() # assuming coeffs are polynomials, not ratfuns.
        r = self.order()
        if d > max(20, r + 2):
            # throw away coefficients which have no chance to influence the indicial polynomial
            q = self.base_ring().gen()**(d - (r + 2))
            return self.map_coefficients(lambda p: p // q).to_F('F').coefficients(sparse=False)
        else:
            return self.to_F('F').coefficients(sparse=False)

    def spread(self, p=0):
        r"""
        Returns the spread of this operator.

        This is the set of integers `i` such that ``sigma(self[0], i)`` and ``sigma(self[r], -r)``
        have a nontrivial common factor, where ``sigma`` is the shift of the parent's algebra and `r` is
        the order of ``self``.

        If the optional argument `p` is given, the method is applied to ``gcd(self[0], p)`` instead of ``self[0]``.

        The output set contains `\infty` if the constant coefficient of ``self`` is zero.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']; A.<Sx> = OreAlgebra(R, 'Sx');
          sage: ((x+5)*Sx - x).spread()
          [4]
          sage: ((x+5)*Sx - x).lclm((x+19)*Sx - x).spread()
          [3, 4, 17, 18]

        """
        op = self#.normalize(); // don't kill
        A = op.parent()
        R = A.base_ring()
        sigma = A.change_ring(R.change_ring(R.base_ring().fraction_field())).sigma()
        s = set()
        r = op.order()

        if op.is_zero():
            return []
        elif op[0].is_zero():
            return [infinity]

        if R.is_field():
            R = R.ring() # R = k[x]
            R = R.change_ring(R.base_ring().fraction_field())

        try:
            # first try to use shift factorization. this seems to be more efficient in most cases.
            all_facs = [sigma(u, -1) for u, _ in shift_factor(sigma(op[0].gcd(p), r)*op[r])]
            tc = [ u[1:] for _, u in shift_factor(prod(all_facs)*sigma(op[0].gcd(p), r)) ]
            lc = [ u[1:] for _, u in shift_factor(prod(all_facs)*op[r]) ]
            for u, v in zip(tc, lc):
                s = s.union([j[0] - i[0] for i in u for j in v])
            return sorted(s)
        except:
            pass

        # generic fall back code with using the resultant.

        K = R.base_ring()
        R0 = R
        R = R.change_ring(K.fraction_field()) # FF(k[y])[x]
        A = A.change_ring(R)

        y = R(K.gen())
        x = R.gen()

        for q, _ in R(gcd(R0(p), R0(op[r])))(x - r).resultant(R(op[0])(x + y)).numerator().factor():
            if q.degree() == 1:
                try:
                    s.add(ZZ(-q[0]/q[1]))
                except:
                    pass

        return sorted(s)

    def generalized_series_solutions(self, n=5, dominant_only=False, real_only=False, infolevel=0):
        r"""
        Returns the generalized series solutions of this operator.

        These are solutions of the form

          `(x/e)^{x u/v}\rho^x\exp\bigl(c_1 x^{1/m} +...+ c_{v-1} x^{1-1/m}\bigr)x^\alpha p(x^{-1/m},\log(x))`

        where

        * `e` is Euler's constant (2.71...)
        * `v` is a positive integer
        * `u` is an integer; the term `(x/e)^(v/u)` is called the "superexponential part" of the solution
        * `\rho` is an element of an algebraic extension of the coefficient field `K`
          (the algebra's base ring's base ring); the term `\rho^x` is called the "exponential part" of
          the solution
        * `c_1,...,c_{v-1}` are elements of `K(\rho)`; the term `\exp(...)` is called the "subexponential
          part" of the solution
        * `m` is a positive integer multiple of `v`, it is called the object's "ramification"
        * `\alpha` is an element of some algebraic extension of `K(\rho)`; the term `n^\alpha` is called
          the "polynomial part" of the solution (even if `\alpha` is not an integer)
        * `p` is an element of `K(\rho)(\alpha)[[x]][y]`. It is called the "expansion part" of the solution.

        An operator of order `r` has exactly `r` linearly independent solutions of this form.
        This method computes them all, unless the flags specified in the arguments rule out
        some of them.

        Generalized series solutions are asymptotic expansions of sequences annihilated by the operator.

        At present, the method only works for operators where `K` is some field which supports
        coercion to ``QQbar``.

        INPUT:

        - ``n`` (default: 5) -- minimum number of terms in the expansions parts to be computed.
        - ``dominant_only`` (default: False) -- if set to True, only compute solution(s) with maximal
          growth.
        - ``real_only`` (default: False) -- if set to True, only compute solution(s) where `\rho,c_1,...,c_{v-1},\alpha`
          are real.
        - ``infolevel`` (default: 0) -- if set to a positive integer, the methods prints some messages
          about the progress of the computation.

        OUTPUT:

        - a list of ``DiscreteGeneralizedSeries`` objects forming a fundamental system for this operator.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<n> = QQ['n']; A.<Sn> = OreAlgebra(R, 'Sn')
          sage: (Sn - (n+1)).generalized_series_solutions()
          [(n/e)^n*n^(1/2)*(1 + 1/12*n^(-1) + 1/288*n^(-2) - 139/51840*n^(-3) - 571/2488320*n^(-4) + O(n^(-5)))]
          sage: list(map(Sn - (n+1), _))
          [0]

          sage: L = ((n+1)*Sn - n).annihilator_of_sum().symmetric_power(2)
          sage: L.generalized_series_solutions()
          [1 + O(n^(-5)),
           (1 + O(n^(-5)))*log(n) + 1/2*n^(-1) - 1/12*n^(-2) + 1/120*n^(-4) + O(n^(-5)),
           (1 + O(n^(-5)))*log(n)^2 + (n^(-1) - 1/6*n^(-2) + 1/60*n^(-4) + O(n^(-5)))*log(n) + 1/4*n^(-2) - 1/12*n^(-3) + 1/144*n^(-4) + O(n^(-5))]
          sage: list(map(L, _))
          [0, 0, 0]

          sage: L = n^2*(1-2*Sn+Sn^2) + (n+1)*(1+Sn+Sn^2)
          sage: L.generalized_series_solutions() # long time (1.4 s)
          [exp(3.464101615137755?*I*n^(1/2))*n^(1/4)*(1 - 2.056810333988042?*I*n^(-1/2) - 1107/512*n^(-2/2) + (0.?e-19 + 1.489453749877895?*I)*n^(-3/2) + 2960239/2621440*n^(-4/2) + (0.?e-19 - 0.926161373412572?*I)*n^(-5/2) - 16615014713/46976204800*n^(-6/2) + (0.?e-20 + 0.03266142931818572?*I)*n^(-7/2) + 16652086533741/96207267430400*n^(-8/2) + (0.?e-20 - 0.1615093987591473?*I)*n^(-9/2) + O(n^(-10/2))), exp(-3.464101615137755?*I*n^(1/2))*n^(1/4)*(1 + 2.056810333988042?*I*n^(-1/2) - 1107/512*n^(-2/2) + (0.?e-19 - 1.489453749877895?*I)*n^(-3/2) + 2960239/2621440*n^(-4/2) + (0.?e-19 + 0.926161373412572?*I)*n^(-5/2) - 16615014713/46976204800*n^(-6/2) + (0.?e-20 - 0.03266142931818572?*I)*n^(-7/2) + 16652086533741/96207267430400*n^(-8/2) + (0.?e-20 + 0.1615093987591473?*I)*n^(-9/2) + O(n^(-10/2)))]

          sage: L = guess([(-3)^k*(k+1)/(2*k+4) - 2^k*k^3/(k+3) for k in range(500)], A)
          sage: L.generalized_series_solutions()
          [2^n*n^2*(1 - 3*n^(-1) + 9*n^(-2) - 27*n^(-3) + 81*n^(-4) + O(n^(-5))), (-3)^n*(1 - n^(-1) + 2*n^(-2) - 4*n^(-3) + 8*n^(-4) + O(n^(-5)))]
          sage: L.generalized_series_solutions(dominant_only=True)
          [(-3)^n*(1 - n^(-1) + 2*n^(-2) - 4*n^(-3) + 8*n^(-4) + O(n^(-5)))]

        TESTS::

            sage: rop = (-8 -12*Sn + (n^2+5*n+6)*Sn^3)
            sage: rop
            (n^2 + 5*n + 6)*Sn^3 - 12*Sn - 8
            sage: rop.generalized_series_solutions(1) # long time (7 s)
            [(n/e)^(-2/3*n)*2^n*exp(3*n^(1/3))*n^(-2/3)*(1 + 3/2*n^(-1/3) + 9/8*n^(-2/3) + O(n^(-3/3))),
            (n/e)^(-2/3*n)*(-1.000000000000000? + 1.732050807568878?*I)^n*exp((-1.500000000000000? + 2.598076211353316?*I)*n^(1/3))*n^(-2/3)*(1 + (-0.750000000000000? - 1.299038105676658?*I)*n^(-1/3) + (-0.562500000000000? + 0.974278579257494?*I)*n^(-2/3) + O(n^(-3/3))),
            (n/e)^(-2/3*n)*(-1.000000000000000? - 1.732050807568878?*I)^n*exp((-1.500000000000000? - 2.598076211353316?*I)*n^(1/3))*n^(-2/3)*(1 + (-0.750000000000000? + 1.299038105676658?*I)*n^(-1/3) + (-0.562500000000000? - 0.974278579257494?*I)*n^(-2/3) + O(n^(-3/3)))]
        """
        K = QQbar

        try:
            origcoeffs = coeffs = [c.change_ring(K) for c in self.numerator().primitive_part().coefficients(sparse=False) ]
        except:
            raise TypeError("unexpected coefficient domain: " + str(self.base_ring().base_ring()))

        if len(coeffs) == 0:
            raise ZeroDivisionError("everything is a solution of the zero operator")
        elif len(coeffs) == 1:
            return []

        def info(level, msg):
            if level <= infolevel:
                print(" "*3*(level - 1) + msg)

        r = len(coeffs) - 1
        x = coeffs[0].parent().gen()
        subs = _generalized_series_shift_quotient
        w_prec = r + 1

        # 1. superexponential parts
        deg = max(c.degree() for c in coeffs if c!=0)

        solutions = []
        for s, _ in self.newton_polygon(~x):
            if s == 0:
                newcoeffs = [c.shift(w_prec - deg) for c in coeffs ]
            else:
                v = s.denominator()
                underflow = int(max(0, -v*r*s))
                newdeg = max([ coeffs[i].degree() + i*s for i in range(len(coeffs)) if coeffs[i] != 0 ])
                newcoeffs = [(coeffs[i](x**v)*subs(x, prec=w_prec + underflow, shift=i, gamma=s))
                             .shift(-v*(newdeg + underflow)) for i in range(len(coeffs))]
            solutions.append( [s, newcoeffs ] )

        if dominant_only:
            max_gamma = max( [g for g, _ in solutions ] )
            solutions = [s for s in solutions if s[0]==max_gamma]

        info(1, "superexponential parts isolated: " + str([g for g, _ in solutions]))

        # 2. exponential parts
        refined_solutions = []
        for gamma, coeffs in solutions:
            info(2, "determining exponential parts for gamma=" + str(gamma))
            deg = max([p.degree() for p in coeffs])
            v = gamma.denominator()
            char_poly = K['rho']([ c[deg] for c in coeffs ])
            for cp, e in char_poly.factor():
                rho = -cp[0]/cp[1] # K is algebraically closed, so all factors are linear.
                if not rho.is_zero() and (not real_only or rho.imag().is_zero()):
                    info(3, "found rho=" + str(rho))
                    refined_solutions.append([gamma, rho, [coeffs[i]*(rho**i) for i in range(len(coeffs))], e*v])

        if dominant_only:
            max_rho = max( [abs(rho) for _, rho, _, _ in refined_solutions ] )
            refined_solutions = [s for s in refined_solutions if abs(s[1])==max_rho]

        info(1, "exponential parts isolated: " + str([(gamma, rho) for gamma, rho, _, _ in refined_solutions]))

        # 3. subexponential parts
        solutions = refined_solutions
        refined_solutions = []
        for gamma, rho, coeffs, ram in solutions:

            info(2, "determining subexponential parts for (gamma,rho)=" + str((gamma, rho)))

            if ram == 1:
                refined_solutions.append([gamma, rho, [], ram, coeffs])
                continue

            def mysubs(x, prec, shift, subexp, ramification=ram):
                return subs(x, prec, shift, subexp=subexp, ramification=ram)

            KK = K['s'].fraction_field()
            X = x.change_ring(KK)
            v = gamma.denominator()
            e = ram/v
            cc = [ c(x**e).change_ring(KK) for c in coeffs ]
            subexpvecs = [ [K.zero()]*(ram - 1) ]

            for i in range(ram - 1, 0, -1):
                old = subexpvecs
                subexpvecs = []
                for sub in old:
                    sub[i - 1] = KK.gen()
                    rest = sum((cc[j]*mysubs(X, e, j, sub)) for j in range(r + 1))
                    for p, _ in rest.leading_coefficient().factor():
                        c = -p[0]/p[1]
                        if not real_only or c.imag().is_zero():
                            vec = list(sub)
                            vec[i - 1] = c
                            subexpvecs.append(vec)
                info(3, "after " + str(ram - i) + " of " + str(ram - 1) + " iterations: " + str(subexpvecs))

            for sub in subexpvecs:
                if all(ee.is_zero() for ee in sub):
                    refined_solutions.append([gamma, rho, sub, gamma.denominator(), coeffs])
                elif False:
                    # possible improvement: check whether ramification can be reduced.
                    pass
                else:
                    newcoeffs = [ (coeffs[j](x**e)*mysubs(x, w_prec, j, sub)).shift(-ram*w_prec) for j in range(r + 1) ]
                    refined_solutions.append([gamma, rho, sub, ram, newcoeffs])

        info(1, "subexponential parts completed; " + str(len(refined_solutions)) + " solutions separated.")

        # 4. polynomial parts and expansion
        solutions = refined_solutions
        refined_solutions = []
        for gamma, rho, subexp, ram, coeffs in solutions:

            info(2, "determining polynomial parts for (gamma,rho,subexp)=" + str((gamma, rho, subexp)))

            KK = K['s'].fraction_field()
            s = KK.gen()
            X = x.change_ring(KK)
            rest = sum(coeffs[i].change_ring(KK)*subs(X, w_prec, i, alpha=s)(X**ram) for i in range(len(coeffs)))
            for p, e in shift_factor(rest.leading_coefficient().numerator(), ram):
                e.reverse()
                alpha = -p[0]/p[1]
                if alpha in QQ: # cause conversion to explicit rational
                    pass
                if (not real_only or alpha.imag().is_zero()):
                    info(3, "found alpha=" + str(alpha))
                    refined_solutions.append([gamma, rho, subexp, ram, alpha, e, 2*ram*w_prec - rest.degree()])

        info(1, "polynomial parts completed; " + str(len(refined_solutions)) + " solutions separated.")

        # 5. expansion and logarithmic terms
        solutions = refined_solutions
        refined_solutions = []
        G = GeneralizedSeriesMonoid(K, x, 'discrete')
        prec = n + w_prec
        PS = PowerSeriesRing(K, 'x')

        info(2, "preparing computation of expansion terms...")
        max_log_power = max([sum(b for _, b in e[5]) for e in solutions])
        poly_tails = [[x**(ram*prec)]*(ram*prec)]
        log_tails = [[x**(ram*prec)]*max_log_power]
        for l in range(1, r + 1):

            # (n+l)^(-1/ram) = n^(-1/ram)*sum(bin(-1/ram, i)*(l/n)^i, i=0...)
            # poly_tails[l][k] = expansion of (n+l)^(-k/ram)/n^(-k/ram)
            p = sum(_binomial(-1/ram, i)*(l*x**ram)**i for i in range(prec + 1))
            pt = [x.parent().one()]
            while len(pt) <= ram*prec:
                pt.append((pt[-1]*p) % x**(ram*prec + 1))
            poly_tails.append([x**(ram*prec - p.degree())*p.reverse() for p in pt])

            # log(n+l) = log(n) - sum( (-l/n)^i/i, i=1...)
            # log_tails[l][k] = (log(n+l) - log(n))^k
            p = -sum((-l*x**ram)**i/QQ(i) for i in range(1, prec + 1))
            lt = [x.parent().one()]
            while len(lt) < max_log_power:
                lt.append((lt[-1]*p) % x**(prec*ram + 1))
            log_tails.append([x**(ram*prec - p.degree())*p.reverse() for p in lt])

        for gamma, rho, subexp, ram, alpha, e, degdrop in solutions:

            info(2, "determining expansions for (gamma,rho,subexp,alpha)=" + str((gamma, rho, subexp,alpha)))

            underflow = int(max(0, -ram*r*gamma))
            coeffs = [(origcoeffs[i](x**ram)*subs(x, prec + underflow, i, gamma, rho, subexp, ram)).shift(-underflow)\
                          for i in range(r + 1)]
            deg = max([c.degree() for c in coeffs])
            coeffs = [coeffs[i].shift(ram*prec - deg) for i in range(r + 1)]
            sols = { a: [] for a, b in e }

            for a, b in e:

                s = alpha - a/ram
                # (n+l)^s/n^s = sum(binom(s,i) (l/n)^i, i=0...)
                spoly_tails = [sum(_binomial(s, i)*(j**i)*(x**(ram*(prec-i))) for i in range(prec)) for j in range(r+1)]

                def operator_applied_to_term(k, l=0):
                    # computes L( n^(s-k/ram) log(n)^l ) as list of length l+1
                    # whose i-th component contains the polynomial terms corresponding to log(n)^i
                    out = []
                    for i in range(l + 1):
                        # [log(n)^i] (n+j)^(s-k/ram)log(n+j)^l
                        # = binom(l, i)*log_tails[j][l - i]*poly_tails[j][k]*spoly_tails[j]
                        contrib = x-x #=0
                        for j in range(r + 1):
                            if i != l and j == 0: # [log(n)^i] log(n)^l
                                continue
                            contrib += ((coeffs[j]*log_tails[j][l - i]).shift(-ram*prec)* \
                                        (poly_tails[j][k]*spoly_tails[j]).shift(-ram*prec)).shift(-ram*prec - k)
                        out.append(_binomial(l, i)*contrib)

                    return out

                while len(sols[a]) < b:

                    info(3, str(len(sols[a])) + " of " + str(sum([bb for _, bb in e])) + " solutions...")

                    newsol = [[K.zero()] for i in range(len(sols[a]))] + [[K.one()]]
                    rest = operator_applied_to_term(0, len(sols[a]))
                    sols[a].append(newsol)

                    for k in range(1, ram*n):
                        info(4, str(k) + " of " + str(ram*n - 1) + " terms...")
                        for l in range(len(rest) - 1, -1, -1):
                            # determine coeff of log(n)^l*n^(s - k/ram) in newsol so as to kill
                            # coeff log(n)^l*n^(s - degdrop - k/ram) of rest
                            tokill = rest[l][ram*prec - k - degdrop]
                            if tokill.is_zero():
                                newsol[l].append(K.zero())
                                continue
                            adjustment = operator_applied_to_term(k, l)
                            killer = adjustment[l][ram*prec - k - degdrop]
                            dl = 0
                            # determine appropriate log power for getting nonzero killer
                            while killer.is_zero():
                                dl += 1
                                adjustment = operator_applied_to_term(k, l + dl)
                                killer = adjustment[l + dl][ram*prec - degdrop - k]
                            # update solution
                            while len(newsol) < l + dl:
                                newsol[-1].append(K.zero())
                                newsol.append([K.zero()]*(k - 1))
                            newcoeff = -tokill/killer
                            newsol[l + dl].append(newcoeff)
                            # update remainder
                            while len(rest) < len(adjustment):
                                rest.append(x.parent().zero())
                            for i in range(len(adjustment)):
                                rest[i] += newcoeff*adjustment[i]

            for a in sols.keys():
                for eexp in sols[a]:
                    refined_solutions.append(G([gamma, ram, rho, subexp, alpha - a/ram, [PS(p, len(p)) for p in eexp]]))

        return refined_solutions

    def _powerIndicator(self):
        return self.coefficients(sparse=False)[0]

    def _infinite_singularity(self):
        r"""
        Simplified version of generalized_series_solutions, without subexponential parts, without
        logarithms, and without extensions of the constant field.

        This function is used in the hypergeometric solver.

        OUTPUT:

           A list of all triples (gamma, phi, alpha) such that 'self' has a local
           solution at infinity of the form Gamma(x)^gamma phi^x x^alpha
           series(1/x), where gamma is in ZZ and phi and alpha are in the constant
           field of this operator's parent algebra.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ[]
           sage: A.<Sx> = OreAlgebra(R)
           sage: (Sx - x).lclm(x^2*Sx - 2).lclm((x+1)*Sx - (x-1/2))._infinite_singularity()
           [[-2, 2, 0], [0, 1, -3/2], [1, 1, 0]]

        """
        S = self.parent().gen()
        n = self.parent().base_ring().gen()
        R = self.base_ring().base_ring().fraction_field()[n]
        # coeffs = list(map(R, self.normalize().coefficients(sparse=False)))
        r = self.order()

        # determine the possible values of gamma and phi
        # points = list(filter(lambda p: p[1] >= 0, [ (i, coeffs[i].degree()) for i in range(len(coeffs)) ]))
        output = []

        for s, np in self.newton_polygon(~n):
            if s in ZZ:
                for p, _ in R(np).factor():
                    if p.degree() == 1 and not p[0].is_zero():
                        phi = -p[0]/p[1]
                        L = self.symmetric_product(phi*n**max(0, s)*S - n**max(0, -s)).normalize().change_ring(R)
                        d = max(r + 3, max(p.degree() for p in L if not p.is_zero()))
                        for q, _ in L.map_coefficients(lambda p: p//n**(d - (r + 3)))\
                                .indicial_polynomial(~n).factor():
                            if q.degree() == 1:
                                output.append([s, phi, -q[0]/q[1]])

        return output

    def _normalize_make_valuation_places_args(self,f,Nmin,Nmax,prec=None, infolevel=0):
        return (f,Nmin,Nmax,prec)

    @cached_method(key=_normalize_make_valuation_places_args)
    def _make_valuation_places(self,f,Nmin,Nmax,prec=None,infolevel=0):
        r"""
        Compute value functions for the place ``f``.

        INPUT:

        - ``f`` - a place, that is an irreducible polynomial in the base ring of
          the ambient Ore algebra

        - ``Nmin`` - an integer

        - ``Nmax`` - an integer

        - ``prec`` (default: None) - precision at which to compute the deformed
          solutions. If not provided, the default precision of a power series
          ring is used.

        - ``infolevel`` (default: None) - verbosity flag

        OUTPUT:

        A list of places corresponding to the shifted positions associated to
        ``f``.  More precisely, if ``xi`` is a root of ``f``, the places
        correspond to the points ``xi+Nmin, \ldots, xi+Nmax``.

        Each place is a tuple composed of ``f(x+k)``, a suitable function for
        ``value_function`` and a suitable function for ``raise_value``.

        EXAMPLES: see `find_candidate_places`
        """

        print1 = print if infolevel >= 1 else lambda *a, **k: None
        print2 = print if infolevel >= 2 else lambda *a, **k: None

        print1(f" [make_places] At (root of {f}) + Nmin={Nmin}, Nmax={Nmax}"
               )

        r = self.order()
        Ore = self.parent()
        Pol = Ore.base_ring()
        nn = Pol.gen()
        Coef = Pol.base_ring()

        # TODO: Do we have to choose a name?
        FF = Coef.extension(f,"xi")
        xi = FF.gen()

        Laur = LaurentSeriesRing(FF,'q',default_prec=prec)
        qq = Laur.gen()
        Frac_q = Pol.change_ring(Laur).fraction_field()

        coeffs_q = [Frac_q(c) for c in self.coefficients(sparse=False)]

        # Variable convention: k is a list index in the whole sequence, n is an
        # actual shift compared to xi, so k=n-Nmin, and the value at index k corresponds to the
        # values of the sequence at position xi+n = xi+k+Nmin.

        def prolong(l,n):
            # Given the values of a function at ...xi+n-r...xi+n-1, compute the
            # value at xi+n
            assert(len(l) >= r)
            l.append(-sum(l[-r+i]*coeffs_q[i](qq+xi+n-r) for i in range(r))
                     / coeffs_q[-1](qq+xi+n-r))

        def call(op,l,n):
            # Given another operator, and given the values l of a function at xi+n,...,xi+n+r,
            # apply its deformed version to l and compute the value at xi+n
            r = op.order()
            assert(len(l) > r)
            coeffs_q = [Frac_q(c) for c in op.coefficients(sparse=False)]
            return sum(l[i]*coeffs_q[i](qq+xi+n) for i in range(r+1))

        sols = [[1 if i==j else 0 for i in range(r)] for j in range(r)]
        for n in range(Nmin+r,Nmax+r):
            for i in range(r):
                prolong(sols[i],n)

        print2(" [make_places] sols")
        print2(sols)

        # Capture the relevant variables in the two functions
        def get_functions(xi,n,Nmin,sols,call):

            # In both functions the second argument `place` is ignored because captured
            def val_fct(op,**kwargs):
                # n-Nmin is the index of the value of the function at xi+n in
                # the list seq
                vect = [call(op,seq[n-Nmin:n-Nmin+r+1],n) for seq in sols]
                return _vect_val_fct(vect)
            def raise_val_fct(ops,dim=None,**kwargs):
                mat = [[call(op,seq[n-Nmin:n-Nmin+r+1],n) for seq in sols]
                       for op in ops]
                #if infolevel >= 2: print(mat)
                return _vect_elim_fct(mat,place=None,dim=dim,infolevel=infolevel)
            return val_fct, raise_val_fct# , sols, call

        res = []
        for n in range(Nmin+r,Nmax+1):
            print1(f" [make_places] preparing place at {xi}+{n} (min poly = {f(nn-n)})"
                   )
            val_fct, raise_val_fct = get_functions(xi,n,Nmin,sols,call)
            res.append((f(nn-n),val_fct,raise_val_fct# , sols, call
            ))
        return res

    def find_candidate_places(self, Zmax = None, infolevel=0, **kwargs):
        r"""

        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Ore.<Sx> = OreAlgebra(Pol)
            sage: L = x*(x-1)*Sx^2 - 1
            sage: places = L.find_candidate_places()
            sage: [p[0] for p in places]
            [x - 1, x - 2, x - 3, x - 4]
            sage: f,v,rv = places[0]
            sage: v((x-1)*Sx)
            0
            sage: rv([(x-1)*Sx, x*(x-1)*Sx])
            (-1, 1)
        """
        # Helpers
        print1 = print if infolevel >= 1 else lambda *a, **k: None

        coeffs = self.coefficients(sparse=False)

        r = self.order()
        i = next(i for i in range(r + 1) if coeffs[i] != 0)
        # Should we replace r with r-i when counting solutions?
        lr = coeffs[-1]
        l0 = coeffs[i]
        l0lr = l0*lr

        # Find the points of interest
        fact0 = list(lr.factor()) + list(l0.factor())

        print1(f"Factors (non unique): {fact0}")

        # Cleanup the list
        fact = []
        for f, m in fact0 :
            if f.degree() == 0:
                pass
            try:
                idx = next(iter(i for i, facti in enumerate(fact)
                                if facti[0].degree() == f.degree()
                                and roots_at_integer_distance(facti[0], f)))
            except StopIteration:
                fact.append([f, m])
            else:
                # f is a shift of a factor already seen
                fact[idx][1] += m

        print1(f"Factors (unique): {fact}")

        places = []
        for f, m in fact:
            print1(f"Computing places for {f}")

            # Finding the actual indices of interest
            inds = roots_at_integer_distance(l0lr, f)
            print1(f"Integer distances between roots: {inds}")
            Nmin = min(inds)
            Nmax = max(inds) + r
            Nmin = Nmin - r
            if Zmax :
                Nmax = min(Nmax,Zmax)
                # Else the default max is Nmax
                # TODO: Should we also update Nmin if Zmax < Nmax?
            print1(f"Nmin={Nmin} Nmax={Nmax}")

            places += self._make_valuation_places(f, Nmin, Nmax, prec=m + 1,
                                                  infolevel=infolevel)
            # TODO: is +1 needed?

        return places

    def value_function(self, op, place, **kwargs):
        val = self._make_valuation_places(place,0,0)[0][1]
        return val(op,place)

    def raise_value(self, basis, place, dim, **kwargs):
        fct = self._make_valuation_places(place,0,0)[0][2]
        return fct(basis, place, dim)

#############################################################################################################

class UnivariateDifferenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
    Element of an Ore algebra K(x)[F], where F is the forward difference operator F f(x) = f(x+1) - f(x)
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        if isinstance(f, (tuple, list)):
            return self.to_S('S')(f, **kwargs)

        R = self.parent()
        x = R.base_ring().gen()
        qx = R.sigma()(x)
        if "action" not in kwargs:
            kwargs["action"] = lambda p : p.subs({x:qx}) - p

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_S(self, alg): # delta2s
        """
        Returns the differential operator corresponding to ``self``

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_S()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          a standard shift with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']
          sage: A.<Fx> = OreAlgebra(R, 'Fx')
          sage: (Fx^4).to_S(OreAlgebra(R, 'Sx'))
          Sx^4 - 4*Sx^3 + 6*Sx^2 - 4*Sx + 1
          sage: (Fx^4).to_S('Sx')
          Sx^4 - 4*Sx^3 + 6*Sx^2 - 4*Sx + 1
        """
        R = self.base_ring()
        x = R.gen()
        one = R.one()

        if isinstance(alg, str):
            alg = self.parent().change_var_sigma_delta(alg, {x:x+one}, {})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_S():
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        delta = alg.gen() - alg.one()
        delta_k = alg.one()
        R = alg.base_ring()
        c = self.coefficients(sparse=False)
        out = alg(R(c[0]))

        for i in range(self.order()):

            delta_k *= delta
            out += R(c[i + 1])*delta_k

        return out

    def to_D(self, alg):
        r"""
        Returns a differential operator which annihilates every power series (about
        the origin) whose coefficient sequence is annihilated by ``self``.
        The output operator may not be minimal.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_D()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the standard derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
          sage: A.<Fn> = OreAlgebra(Rn, 'Fn')
          sage: B.<Dx> = OreAlgebra(Rx, 'Dx')
          sage: Fn.to_D(B)
          (-x + 1)*Dx - 1
          sage: ((n+1)*Fn - 1).to_D(B)
          (-x^2 + x)*Dx^2 + (-4*x + 1)*Dx - 2
          sage: (x*Dx-1).to_F(A).to_D(B)
          x*Dx - 1

        """
        return self.to_S('S').to_D(alg)

    def to_T(self, alg):
        r"""
        Returns a differential operator, expressed in terms of the Euler derivation,
        which annihilates every power series (about the origin) whose coefficient
        sequence is annihilated by ``self``.
        The output operator may not be minimal.

        INPUT:

        - ``alg`` -- the Ore algebra in which the output should be expressed.
          The algebra must satisfy ``alg.base_ring().base_ring() == self.base_ring().base_ring()``
          and ``alg.is_T()`` is not ``False``.
          Instead of an algebra object, also a string can be passed as argument.
          This amounts to specifying an Ore algebra over ``self.base_ring()`` with
          the Euler derivation with respect to ``self.base_ring().gen()``.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: Rn.<n> = ZZ['n']; Rx.<x> = ZZ['x']
          sage: A.<Fn> = OreAlgebra(Rn, 'Fn')
          sage: B.<Tx> = OreAlgebra(Rx, 'Tx')
          sage: Fn.to_T(B)
          (-x + 1)*Tx - x
          sage: ((n+1)*Fn - 1).to_T(B)
          (-x + 1)*Tx^2 - 3*x*Tx - 2*x
          sage: (x*Tx-1).to_F(A).to_T(B)
          x*Tx^2 + (x - 1)*Tx

        """
        return self.to_S('S').to_T(alg)

    def to_list(self, *args, **kwargs):
        return self.to_S('S').to_list(*args, **kwargs)

    to_list.__doc__ = UnivariateRecurrenceOperatorOverUnivariateRing.to_list.__doc__

    def indicial_polynomial(self, *args, **kwargs):
        return self.to_S('S').indicial_polynomial(*args, **kwargs)

    indicial_polynomial.__doc__ = UnivariateRecurrenceOperatorOverUnivariateRing.indicial_polynomial.__doc__

    def spread(self, p=0):
        return self.to_S().spread(p)

    spread.__doc__ = UnivariateRecurrenceOperatorOverUnivariateRing.spread.__doc__

    def _coeff_list_for_indicial_polynomial(self):
        return self.coefficients(sparse=False)

    def _denominator_bound(self):
        return self.to_S()._denominator_bound()

    def symmetric_product(self, other, solver=None):

        if not isinstance(other, UnivariateOreOperator):
            raise TypeError("unexpected argument in symmetric_product")

        if self.parent() != other.parent():
            A, B = canonical_coercion(self, other)
            return A.symmetric_product(B, solver=solver)

        A = self.to_S('S')
        B = other.to_S(A.parent())
        return A.symmetric_product(B, solver=solver).to_F(self.parent())

    symmetric_product.__doc__ = UnivariateOreOperator.symmetric_product.__doc__
