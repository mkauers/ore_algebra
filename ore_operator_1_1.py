
"""
ore_operator_1_1
================

Special classes for operators living in algebras with one generator with base rings that also have
one generator.

"""

from sage.structure.element import RingElement, canonical_coercion
from sage.rings.arith import gcd, lcm
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.infinity import infinity

from ore_operator import *

class UnivariateOreOperatorOverUnivariateRing(UnivariateOreOperator):
    """
    Element of an Ore algebra with a single generator and a commutative rational function field as base ring.     
    """
    # Overview of dependencies between degree and denominator bounding functions:
    #
    #   * _degree_bound()          ==> requires indicial_polynomial(1/x)
    #
    #   * _denominator_bound()     ==> requires dispersion() and implements Abramov's algorithm
    #                                  Not universally correct. May have to be adapted by subclasses.
    #
    #   * indicial_polynomial(p)   ==> requires _coeff_list_for_indicial_polynomial when p is x or 1/x
    #                                  ABSTRACT for other arguments p.
    #
    #   * dispersion(p)            ==> requires spread(p)
    #
    #   * _coeff_list_for_ind...   ==> ABSTRACT
    #
    #   * spread(p)                ==> ABSTRACT
    #

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperator, self).__init__(parent, *data, **kwargs)

    def _normalize_base_ring(self):
        """
        Rewrites ``self`` into an operator from an algebra whose base ring is a univariate
        polynomial ring over a field.

        Returns a tuple ``(A, R, K, L)`` where

         * ``L`` is the new operator
 
         * ``A`` is the parent of ``L``

         * ``R`` is the base ring of ``A``

         * ``K`` is the base ring of ``R``

        """
        A = self.parent(); R = A.base_ring(); K = R.base_ring()

        if R.is_field():
            L = self.numerator()
            R = R.ring()

        if not K.is_field():
            R.change_ring(K.fraction_field())

        L = L.change_ring(R)
        return L.parent(), R, K, L

    def degree(self):
        """
        Returns the maximum degree among the coefficients of ``self``

        The degree of the zero operator is `-1`.

        If the base ring is not a polynomial ring, this causes an error.
        """
        if self.is_zero():
            return -1
        else:
            R = self.base_ring()
            if R.is_field():
                R = R.ring()
            return max( R(p).degree() for p in self.coefficients() )                

    def polynomial_solutions(self, rhs=(), degree=None, solver=None):
        """
        ...
        """
        raise NotImplementedError

    def rational_solutions(self, rhs=(), denominator=None, degree=None, solver=None):
        """
        ...
        """
        raise NotImplementedError

    def _degree_bound(self):
        """
        Computes a degree bound for the polynomial solutions of this operator.

        This is an integer `d` such that every polynomial solution of this operator
        has degree `d` or less. 
        """

        if self.is_zero():
            raise ZeroDivisionError, "unbounded degree"
        
        R = self.base_ring()
        d = -1

        for (p, _) in self.indicial_polynomial(~R.fraction_field()(R.gen())).factor():
            if p.degree() == 1:
                try:
                    d = max(d, ZZ(-p[0]/p[1]))
                except:
                    pass

        return d        

    def _denominator_bound(self):
        """
        Computes a denominator bound for the rational solutions of this operator.

        This is a polynomial `q` such that every rational solution of this operator
        can be written in the form `p/q` for some other polynomial `p` (not necessarily
        coprime with `q`)

        The default implementation is Abramov's algorithm, which depends on the existence
        of an implementation of ``dispersion``. Subclasses for algebras where this is not
        appropriate must override this method. 
        """

        if self.is_zero():
            raise ZeroDivisionError, "unbounded denominator"

        A, R, k, L = self._normalize_base_ring()
        sigma = A.sigma()
        r = L.order()

        n = L.dispersion()
        A = sigma(L[r], -r)
        B = L[0]
        u = L.base_ring().one()

        for i in xrange(n, -1, -1):
            d = A.gcd(sigma(B, i))
            if d.degree() > 0:
                A //= d
                for j in xrange(i):
                    u *= d
                    d = sigma(d, -1)
                u *= d
                B //= d

        return self.base_ring()(u.numerator())

    def dispersion(self, p=0):
        """
        Returns the dispersion of this operator.

        This is the maximum nonnegative integer `i` such that ``sigma(self[0], i)`` and ``sigma(self[r], -r)``
        have a nontrivial common factor, where ``sigma`` is the shift of the parent's algebra and `r` is
        the order of ``self``.

        An output `-1` indicates that there are no such integers `i` at all.

        If the optional argument `p` is given, the method is applied to ``gcd(self[0], p)`` instead of ``self[0]``.

        The output is `\infty` if the constant coefficient of ``self`` is zero.

        EXAMPLES::

          sage: R.<x> = ZZ['x']; A.<Sx> = OreAlgebra(R, 'Sx');
          sage: ((x+5)*Sx - x).dispersion()
          4
        
        """
        s = self.spread(p)
        return max(max(s), -1) if len(s) > 0 else -1

    def spread(self, p=0):
        """
        Returns the spread of this operator.

        This is the set of integers `i` such that ``sigma(self[0], i)`` and ``sigma(self[r], -r)``
        have a nontrivial common factor, where ``sigma`` is the shift of the parent's algebra and `r` is
        the order of ``self``.

        If the optional argument `p` is given, the method is applied to ``gcd(self[0], p)`` instead of ``self[0]``.

        The output set contains `\infty` if the constant coefficient of ``self`` is zero.

        This method is a stub and may not be implemented for every algebra. 

        EXAMPLES::

          sage: R.<x> = ZZ['x']; A.<Sx> = OreAlgebra(R, 'Sx');
          sage: ((x+5)*Sx - x).spread()
          [4]
          sage: ((x+5)*Sx - x).lclm((x+19)*Sx - x).spread()
          [3, 4, 17, 18]
        
        """
        raise NotImplementedError # abstract

    def indicial_polynomial(self, p, var='lambda'):
        """
        Computes the indicial polynomial of ``self`` at (a root of) ``p``.

        The indicial polynomial is a polynomial in the given variable ``var`` with coefficients
        in the fraction field of the base ring's base ring. 

        The precise meaning of this polynomial may depend on the parent of ``self``. A minimum
        requirement is that if ``self`` has a rational solution whose denominator contains
        ``sigma.factorial(p, e)`` but neither ``sigma(p, -1)*sigma.factorial(p, e)`` nor
        ``sigma.factorial(p, e + 1)``, then ``-e`` is a root of this polynomial.

        Applied to `p=1/x`, the maximum integer root of the output should serve as a degree bound
        for the polynomial solutions of ``self``. 

        This method is a stub. Depending on the particular subclass, restrictions on ``p`` may apply.
        """

        x = self.base_ring().gen()

        if op.is_zero():
            return self.base_ring().base_ring()[var].zero()
        
        elif op.order() == 0:
            return self.base_ring().base_ring()[var].one()
        
        elif (x*p).is_one():
            # at infinity
            deg = lambda q: q.degree()

        elif x == p:
            # at zero
            deg = lambda q: q.valuation()

        else:
            # leave this case to the subclass
            raise NotImplementedError

        op = self._coeff_list_for_indicial_polynomial()
        R = self.base_ring().base_ring()[var]
        y = R.gen()

        b = deg(op[0])
        for j in xrange(1, len(op) + 1):
            b = max(b, deg(op[j]) - j)

        s = R.zero(); y_ff_i = R.one()
        for i in xrange(len(op) + 1):
            s = s + op[i][b + i]*y_ff_i
            y_ff_i *= y - i

        return s

    def _coeff_list_for_indicial_polynomial(self):
        """
        Computes a list of polynomials such that the usual algorithm for computing indicial
        polynomials applied to this list gives the desired result.

        For example, for differential operators, this is simply the coefficient list of ``self``,
        but for recurrence operators, it is the coefficient list of ``self.to_F()``.

        This is an abstract method.         
        """
        raise NotImplementedError

    def desingularize(self, p):
        """
        ...
        """
        raise NotImplementedError

    def abramov_van_hoeij(self, other):
        """
        given other=a*D + b, find, if possible, an operator M such that rat*self = 1 - other*M
        for some rational function rat.
        """
        raise NotImplementedError


#############################################################################################################

class UnivariateDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[D], where D acts as derivation d/dx on K(x).
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):
        
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : p.derivative()

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_S(self, alg): # d2s
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

        """
        if type(alg) == str:
            R = self.base_ring(); x = R.gen(); one = R.one()
            rec_algebra = self.parent().change_var_sigma_delta(alg, {x:x+one}, {})
        elif not isinstance(alg, type(self.parent())) or not alg.is_S() \
             or alg.base_ring().base_ring() is not self.base_ring().base_ring():
            raise TypeError, "not an adequate algebra"
        else:
            rec_algebra = alg
        
        numer = self.numerator()
        coeffs = [list(c) for c in list(numer)]
        lengths = [len(c) for c in coeffs]

        r = len(coeffs) - 1
        d = max(lengths) - 1
        start = d + 1
        for k in range(r + 1):
            start = min(start, d - (lengths[k] - 1) + k)

        roots = [0] * r
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
                from_newton_basis(result[i], range(-i, -i + r))

        return rec_algebra(result)

    def to_F(self, alg):
        """
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

    def to_T(self, alg): # d2theta
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
        R = self.base_ring(); one = R.one(); x = R.gen()
        
        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {}, {x:x})
        elif not isinstance(alg, type(self.parent())) or not alg.is_T() or \
             alg.base_ring().base_ring() is not R.base_ring():
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); alg2 = alg.change_ring(R); x = R.gen()

        theta = (1/x)*alg2.gen(); theta_k = alg2.one();
        c = self.coeffs(); out = alg2(R(c[0]))

        for i in xrange(self.order()):
            
            theta_k *= theta
            out += R(c[i + 1])*theta_k

        return out if alg.base_ring() is R else out.numerator()

    def annihilator_of_integral(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite integrals `\int f`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order. 

        EXAMPLES::

           sage: R.<x> = ZZ['x']
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: ((x-1)*Dx - 2*x).annihilator_of_integral()
           (x-1)*Dx^2 - 2*x*Dx
           sage: _.annihilator_of_associate(Dx)
           (x-1)*Dx - 2*x
           
        """
        return self*self.parent().gen()

    def annihilator_of_composition(self, a, solver=None):
        """
        Returns an operator `L` which annihilates all the functions `f(a(x))`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order.

        INPUT:

        - ``a`` -- either an element of the base ring of the parent of ``self``,
          or an element of an algebraic extension of this ring.
        - ``solver`` (optional) -- a callable object which applied to a matrix
          with polynomial entries returns its kernel. 

        EXAMPLES::

           sage: R.<x> = ZZ['x']
           sage: K.<y> = R.fraction_field()['y']
           sage: K.<y> = R.fraction_field().extension(y^3 - x^2*(x+1))
           sage: A.<Dx> = OreAlgebra(R, 'Dx')
           sage: (x*Dx-1).annihilator_of_composition(y) # ann for x^(2/3)*(x+1)^(1/3)
           (3*x^2 + 3*x)*Dx - 3*x - 2
           sage: (x*Dx-1).annihilator_of_composition(y + 2*x) # ann for 2*x + x^(2/3)*(x+1)^(1/3)
           (-3*x^3 - 3*x^2)*Dx^2 + 2*x*Dx - 2
           sage: (Dx - 1).annihilator_of_composition(y) # ann for exp(x^(2/3)*(x+1)^(1/3))
           (243*x^6 + 810*x^5 + 999*x^4 + 540*x^3 + 108*x^2)*Dx^3 + (162*x^3 + 270*x^2 + 108*x)*Dx^2 + (-162*x^2 - 180*x - 12)*Dx - 243*x^6 - 810*x^5 - 1080*x^4 - 720*x^3 - 240*x^2 - 32*x
        
        """

        A = self.parent(); K = A.base_ring().fraction_field(); R = K['Y']
        if solver == None:
            solver = A._solver(K)

        if self == A.one():
            return self
        elif a in K:
            minpoly = R.gen() - K(a)
        else:
            try:
                minpoly = R(a.minpoly()).monic()
            except:
                raise TypeError, "argument not recognized as algebraic function over base ring"

        d = minpoly.degree(); r = self.order()

        # derivative of a
        Da = -minpoly.map_coefficients(lambda p: p.derivative())
        Da *= minpoly.xgcd(minpoly.derivative())[2]
        Da = Da % minpoly

        # self's coefficients with x replaced by a, denominators cleared, and reduced by minpoly.
        # have: (D^r f)(a) == sum( red[i]*(D^i f)a, i=0..len(red)-1 ) and each red[i] is a poly in Y of deg <= d.
        red = [ R(p.numerator().coeffs()) for p in self.numerator().change_ring(K).coeffs() ]
        lc = -minpoly.xgcd(red[-1])[2]
        red = [ (red[i]*lc) % minpoly for i in xrange(r) ]

        from sage.matrix.constructor import Matrix
        Dkfa = [R.zero() for i in xrange(r)] # Dkfa[i] == coeff of (D^i f)(a) in D^k (f(a))
        Dkfa[0] = R.one()
        mat = [[ q for p in Dkfa for q in p.padded_list(d) ]]; sol = []

        while len(sol) == 0:

            # compute coeffs of (k+1)th derivative
            next = [ (p.map_coefficients(lambda q: q.derivative()) + p.derivative()*Da) % minpoly for p in Dkfa ]
            for i in xrange(r - 1):
                next[i + 1] += (Dkfa[i]*Da) % minpoly
            for i in xrange(r):
                next[i] += (Dkfa[-1]*red[i]*Da) % minpoly
            Dkfa = next

            # check for linear relations
            mat.append([ q for p in Dkfa for q in p.padded_list(d) ])
            sol = solver(Matrix(K, mat).transpose())

        return self.parent()(list(sol[0]))

    def power_series_solutions(self, n=5):
        """
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
        L = self.numerator()
        
        factors = L.indicial_polynomial(self.base_ring().gen()).factor()
        orders = []

        for (p, e) in factors:
            if p.degree() == 1:
                try:
                    alpha = ZZ(-p[0]/p[1])
                    if alpha >= 0:
                        orders.append(alpha)
                except:
                    pass

        if len(orders) == 0:
            return orders # no power series solutions

        r = self.order()
        maxexp = max(orders) + max(n, r)
        K = self.base_ring().base_ring().fraction_field(); zero = K.zero(); one = K.one()
        
        from sage.rings.power_series_ring import PowerSeriesRing
        R = PowerSeriesRing(K, str(self.base_ring().gen()))

        x = R.gen()
        rec = L.to_S('S')
        
        sols = []
        for alpha in orders:

            p = _rec2list(rec, [one], maxexp - alpha, alpha, False, True, ZZ, lambda k: zero)
            p = x**(alpha) * R(p, maxexp - alpha - 1)

            if L(p).is_zero(): # L(p) nonzero indicates series involving logarithms. 
                sols.append(p)

        return sols

    def generalized_series_solutions(self, n, exp=True, log=True):
        """
        ...
        """
        raise NotImplementedError

    def get_value(self, init, z, n):
        """
        If K is a subfield of CC, this computes an approximation of the solution of this operator
        wrt the given initial values at the point z to precision n.
        """
        raise NotImplementedError

    def indicial_polynomial(self, p, var='lambda'):
        """
        Computes the indicial polynomial of this operator at (a root of) `p`.

        If `x` is the generator of the base ring, the input may be either irreducible polynomial in `x`
        or the rational function `1/x`.

        The output is a univariate polynomial in the given variable ``var`` with coefficients in the
        base ring's base ring. It has the following property: for every nonzero series solution
        of ``self`` in rising powers of `p`, i.e. `p_0 p^\alpha + p_1 p^{\alpha+1} + ...`, the
        minimal exponent `\alpha` is a root of the indicial polynomial.
        The converse may not hold. 

        INPUT:

        - ``p`` -- an irreducible polynomial in the base ring of the operator algebra, or `1/x`.
        - ``var`` (optional) -- the variable name to use for the indicial polynomial.

        EXAMPLES::
        
          sage: R.<x> = ZZ['x']; A.<Dx> = OreAlgebra(R, 'Dx');
          sage: L = (x*Dx-5).lclm((x^2+1)*Dx - 7*x).lclm(Dx - 1)
          sage: L.indicial_polynomial(x).factor()
          (-1) * 5 * 2^2 * (lambda - 5) * (lambda - 1) * lambda
          sage: L.indicial_polynomial(1/x).factor()
          2 * (lambda - 7) * (lambda - 5)
          sage: L.indicial_polynomial(x^2+1).factor()
          5 * 7 * 2^2 * (lambda - 1) * lambda * (2*lambda - 7)
        
        """

        x = p.parent().gen()

        if (x*p).is_one() or p == x:
            return super(UnivariateOreOperatorOverUnivariateRing, self).indicial_polynomial(p, var=var)

        op = self.numerator()

        R = op.parent()
        L = R.base_ring() # k[x]
        if L.is_field():
            L = L.ring()
        K = PolynomialRing(L.base_ring(),var) # k[lambda]
        L = L.change_ring(K.fraction_field()) # FF(k[lambda])[x]

        if op.is_zero():
            return L.zero()
        if op.order() == 0:
            return L.one()

        try: 
            p = L(p)
        except:
            raise ValueError, "p has to be a polynomial or 1/" + str(p.parent().gen())

        s = L.zero()
        y = L(K.gen())

        r = self.order()
        pder = self.parent().delta()(p)
        currentder = p.parent().one() # = (p')^i mod p
        currentinv = p**r # = p^(ord - i)
        y_ff_i = y # = falling_factorial(y, i)
        s = op[0]*currentinv

        for i in range(1, op.order() + 1):
            currentder = (currentder * pder) % p
            currentinv = currentinv // p
            s += op[i]*currentinv*currentder*y_ff_i
            y_ff_i *= y - i

        q, r = s.quo_rem(p)
        while r.is_zero() and not q.is_zero():
            q, r = q.quo_rem(p)

        if r.is_zero():
            raise ValueError, "p not irreducible?"
        else:
            return K(gcd(r.coefficients()).numerator())

    def _coeff_list_for_indicial_polynomial(self):
        return self.coeffs()

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
        """
        Denominator bounding based on indicial polynomial. 
        """
        if self.is_zero():
            raise ZeroDivisionError, "unbounded denominator"        

        A, R, _, L = self._normalize_base_ring()

        bound = R.one()
        for (p, _) in L.leading_coefficient().factor():
            e = 0
            for (q, _) in L.indicial_polynomial(p).factor():
                if q.degree() == 1:
                    try:
                        e = min(e, ZZ(-q[0]/q[1]))
                    except:
                        pass
            bound *= p**(-e)

        return bound         


#############################################################################################################

class UnivariateRecurrenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[S], where S is the shift x->x+1.
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):
        
        if type(f) in (tuple, list):

            r = self.order()
            R = self.parent().base_ring()
            K = R.base_ring()
            z = K.zero()
            c = self.numerator().coeffs()
            d = self.denominator()

            def fun(n):
                if f[n + r] is None:
                    return None
                else:
                    try:
                        return sum( c[i](n)*f[n + i] for i in xrange(r + 1) )/d(n)
                    except:
                        return None

            return type(f)(fun(n) for n in xrange(len(f) - r))

        x = self.parent().base_ring().gen()
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : p(x+1)

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
        R = self.base_ring(); x = R.gen(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {}, {x:one})
        elif not isinstance(alg, type(self.parent())) or not alg.is_D() \
             or alg.base_ring().base_ring() is not R.base_ring():
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); x = R.gen()
        alg_theta = alg.change_var_sigma_delta('T', {}, {x:x}).change_ring(R)

        S = alg_theta(~x); out = alg_theta.zero()
        coeffs = self.numerator().coeffs()

        for i in xrange(len(coeffs)):
            out += alg_theta([R(p) for p in coeffs[i].coeffs()])*(S**i)

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

          sage: R.<x> = ZZ['x']
          sage: A.<Sx> = OreAlgebra(R, 'Sx')
          sage: (Sx^4).to_F(OreAlgebra(R, 'Fx'))
          Fx^4 + 4*Fx^3 + 6*Fx^2 + 4*Fx + 1
          sage: (Sx^4).to_F('Fx').to_S(A)
          Sx^4
        
        """
        R = self.base_ring(); x = R.gen(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:x+one}, {x:one})
        elif not isinstance(alg, type(self.parent())) or not alg.is_F() or \
             alg.base_ring().base_ring() is not R.base_ring():
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        delta = alg.gen() + alg.one(); delta_k = alg.one(); R = alg.base_ring()
        c = self.coeffs(); out = alg(R(c[0]))

        for i in xrange(self.order()):
            
            delta_k *= delta
            out += R(c[i + 1])*delta_k

        return out

    def to_T(self, alg):
        """
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
        """
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
        - ``padd`` (optional) -- if ``True``, the vector of initial values is implicitely
          prolonged to the left (!) by zeros if it is too short. Otherwise (default),
          the method raises a ``ValueError`` if ``init`` is too short.

        OUTPUT:

        A list of ``n`` terms whose `k` th component carries the sequence term with
        index ``start+k``.
        Terms whose calculation causes an error are represented by ``None``. 

        EXAMPLES::

           sage: R = ZZ['x']['n']; x = R('x'); n = R('n')
           sage: A.<Sn> = OreAlgebra(R, 'Sn')
           sage: L = ((n+2)*Sn^2 - x*(2*n+3)*Sn + (n+1))
           sage: L.to_list([1, x], 5)
           [1, x, (3*x^2 - 1)/2, (5*x^3 - 3*x)/2, (35*x^4 - 30*x^2 + 3)/8]
           sage: polys = L.to_list([1], 5, padd=True)
           [1, x, (3*x^2 - 1)/2, (5*x^3 - 3*x)/2, (35*x^4 - 30*x^2 + 3)/8]
           sage: L.to_list([polys[3], polys[4]], 8, start=3)
           [(5*x^3 - 3*x)/2, (35*x^4 - 30*x^2 + 3)/8, (63*x^5 - 70*x^3 + 15*x)/8, (231*x^6 - 315*x^4 + 105*x^2 - 5)/16, (429*x^7 - 693*x^5 + 315*x^3 - 35*x)/16]
           
           sage: ((n-5)*Sn - 1).to_list([1], 10)
           [1, 1/-5, 1/20, 1/-60, 1/120, -1/120, None, None, None, None]
        
        """
        return _rec2list(self, init, n, start, append, padd, ZZ)

    def annihilator_of_sum(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite sums `\sum_{k=0}^n a_k`
        where `a_n` runs through the sequences annihilated by ``self``.
        The output operator is not necessarily of smallest possible order. 

        EXAMPLES::

           sage: R.<x> = ZZ['x']
           sage: A.<Sx> = OreAlgebra(R, 'Sx')
           sage: ((x+1)*Sx - x).annihilator_of_sum() # constructs L such that L(H_n) == 0
           (x + 2)*Sx^2 + (-2*x - 3)*Sx + x + 1
           
        """
        A = self.parent()
        return self.map_coefficients(A.sigma())*(A.gen() - A.one())

    def annihilator_of_composition(self, a, solver=None):
        """
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

          sage: R.<x> = QQ['x']
          sage: A.<Sx> = OreAlgebra(R, 'Sx')
          sage: ((2+x)*Sx^2-(2*x+3)*Sx+(x+1)).annihilator_of_composition(2*x+5) 
          (16*x^3 + 188*x^2 + 730*x + 936)*Sx^2 + (-32*x^3 - 360*x^2 - 1340*x - 1650)*Sx + 16*x^3 + 172*x^2 + 610*x + 714
          sage: ((2+x)*Sx^2-(2*x+3)*Sx+(x+1)).annihilator_of_composition(1/2*x)
          (1/2*x^2 + 11/2*x + 15)*Sx^6 + (-3/2*x^2 - 25/2*x - 27)*Sx^4 + (3/2*x^2 + 17/2*x + 13)*Sx^2 - 1/2*x^2 - 3/2*x - 1
          sage: ((2+x)*Sx^2-(2*x+3)*Sx+(x+1)).annihilator_of_composition(100-x)
          (-x + 99)*Sx^2 + (2*x - 199)*Sx - x + 100
          
        """

        A = self.parent()
        
        if a in QQ:
            # a is constant => f(a) is constant => S-1 kills it
            return A.gen() - A.one()

        R = QQ[A.base_ring().gen()]

        try:
            a = R(a)
        except:
            raise ValueError, "argument has to be of the form u*x+v where u,v are rational"

        if a.degree() > 1:
            raise ValueError, "argument has to be of the form u*x+v where u,v are rational"

        try:
            u = QQ(a[1]); v = QQ(a[0])
        except:
            raise ValueError, "argument has to be of the form u*x+v where u,v are rational"

        r = self.order(); x = A.base_ring().gen()

        # special treatment for easy cases
        w = u.denominator().abs()
        if w > 1:
            w = w.lcm(v.denominator()).abs()
            p = self.polynomial()(A.associated_commutative_algebra().gen()**w)
            q = p = A(p.map_coefficients(lambda f: f(x/w)))
            for i in xrange(1, w):
                q = q.lclm(p.annihilator_of_composition(x - i), solver=solver)
            return q.annihilator_of_composition(w*u*x + w*v)
        elif v != 0:
            s = A.sigma(); v = v.floor()
            L = self.map_coefficients(lambda p: s(p, v))
            return L if u == 1 else L.annihilator_of_composition(u*x)
        elif u == 1:
            return self
        elif u < 0:
            c = [ p(-r - x) for p in self.coeffs() ]; c.reverse()
            return A(c).annihilator_of_composition(-u*x)

        # now a = u*x where u > 1 is an integer. 
        from sage.matrix.constructor import Matrix
        A = A.change_ring(A.base_ring().fraction_field())
        if solver == None:
            solver = A._solver()
        L = A(self)

        p = A.one(); Su = A.gen()**u # possible improvement: multiplication matrix. 
        mat = [ p.coeffs(padd=r) ]; sol = []

        while len(sol) == 0:

            p = (Su*p) % L
            mat.append( p.coeffs(padd=r) )
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

          sage: R.<x> = QQ['x']
          sage: A.<Sx> = OreAlgebra(R, 'Sx')
          sage: (x*Sx - (x+1)).annihilator_of_interlacing(Sx - (x+1), Sx + 1)
          (-x^7 - 45/2*x^6 - 363/2*x^5 - 1129/2*x^4 - 45/2*x^3 + 5823/2*x^2 + 5751/2*x - 2349)*Sx^9 + (1/3*x^8 + 61/6*x^7 + 247/2*x^6 + 4573/6*x^5 + 14801/6*x^4 + 7173/2*x^3 + 519/2*x^2 - 3051*x + 756)*Sx^6 + (-7/2*x^6 - 165/2*x^5 - 1563/2*x^4 - 7331/2*x^3 - 16143/2*x^2 - 9297/2*x + 5535)*Sx^3 - 1/3*x^8 - 67/6*x^7 - 299/2*x^6 - 6157/6*x^5 - 22877/6*x^4 - 14549/2*x^3 - 10839/2*x^2 + 1278*x + 2430

        """
        A = self.parent(); A = A.change_ring(A.base_ring().fraction_field())
        ops = [A(self)] + map(A, list(other))
        S_power = A.associated_commutative_algebra().gen()**len(ops)
        x = A.base_ring().gen()

        for i in xrange(len(ops)):
            ops[i] = A(ops[i].polynomial()(S_power)\
                       .map_coefficients(lambda p: p(x/len(ops))))\
                       .annihilator_of_composition(x - i)

        return self.parent()(reduce(lambda p, q: p.lclm(q), ops).numerator())

    def _coeff_list_for_indicial_polynomial(self):
        return self.to_F('F').coeffs()

    def spread(self, p=0):
        """
        Returns the spread of this operator.

        This is the set of integers `i` such that ``sigma(self[0], i)`` and ``sigma(self[r], -r)``
        have a nontrivial common factor, where ``sigma`` is the shift of the parent's algebra and `r` is
        the order of ``self``.

        If the optional argument `p` is given, the method is applied to ``gcd(self[0], p)`` instead of ``self[0]``.

        The output set contains `\infty` if the constant coefficient of ``self`` is zero.

        EXAMPLES::

          sage: R.<x> = ZZ['x']; A.<Sx> = OreAlgebra(R, 'Sx');
          sage: ((x+5)*Sx - x).spread()
          [4]
          sage: ((x+5)*Sx - x).lclm((x+19)*Sx - x).spread()
          [3, 4, 17, 18]
        
        """

        op = self.numerator(); A = op.parent(); R = A.base_ring()
        if R.is_field():
            R = R.ring() # R = k[x]

        K = PolynomialRing(R.base_ring(), 'y') # k[y]
        R = R.change_ring(K.fraction_field()) # FF(k[y])[x]

        y = R(K.gen())
        x = R.gen()

        if op.order()==0:
            return []
        elif op[0].is_zero():
            return [infinity]
        else:
            s = []; r = op.order()
            for (p, _) in (R(op[r])(x - r).resultant(gcd(R(p), R(op[0]))(x + y))).numerator().factor():
                if p.degree() == 1:
                    try:
                        s.append(ZZ(-p[0]/p[1]))
                    except:
                        pass
            s = list(set(s)) # remove duplicates
            s.sort()
            return s

    def generalized_series_solutions(self, n): # at infinity. 
        """
        ...
        """
        raise NotImplementedError


#############################################################################################################

class UnivariateQRecurrenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[S], where S is the shift x->q*x for some q in K.
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        if type(f) in (tuple, list):

            r = self.order()
            R = self.parent().base_ring()
            _, q = self.parent().is_Q()
            K = R.base_ring()
            z = K.zero()
            c = self.numerator().coeffs()
            d = self.denominator()

            def fun(n):
                if f[n + r] is None:
                    return None
                else:
                    try:
                        qn = q**n
                        return sum( c[i](qn)*f[n + i] for i in xrange(r + 1) )/d(qn)
                    except:
                        return None

            return type(f)(fun(n) for n in xrange(len(f) - r))

        R = self.parent(); x = R.base_ring().gen(); qx = R.sigma()(x)
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : p(qx)

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_J(self, alg): # q2j
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
        R = self.base_ring(); K = R.base_ring(); x, q = self.parent().is_Q(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:q*x}, {x:one})
        elif not isinstance(alg, type(self.parent())) or not alg.is_J() or \
             alg.base_ring().base_ring() is not K or K(alg.is_J()[1]) != K(q):
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); x, q = alg.is_J()
        alg = alg.change_ring(R);

        Q = alg(~x); out = alg.zero()
        coeffs = self.numerator().coeffs()
        x_pows = {0 : alg.one(), 1 : ((q - R.one())*x)*alg.gen() + alg.one()}

        for i in xrange(len(coeffs)):
            term = alg.zero()
            c = coeffs[i].coeffs()
            for j in xrange(len(c)):
                if not x_pows.has_key(j):
                    x_pows[j] = x_pows[j - 1]*x_pows[1]
                term += c[j] * x_pows[j]
            out += term*(Q**i)

        return (alg.gen()**(len(coeffs)-1))*out.numerator().change_ring(alg.base_ring())

    def to_list(self, init, n, start=0, append=False, padd=False):
        """
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
        - ``padd`` (optional) -- if ``True``, the vector of initial values is implicitely
          prolonged to the left (!) by zeros if it is too short. Otherwise (default),
          the method raises a ``ValueError`` if ``init`` is too short.

        OUTPUT:

        A list of ``n`` terms whose `k` th component carries the sequence term with
        index ``start+k``.
        Terms whose calculation causes an error are represented by ``None``. 

        EXAMPLES::

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

           sage: R.<x> = ZZ['q'].fraction_field()['x']
           sage: A.<Qx> = OreAlgebra(R, 'Qx')
           sage: ((x+1)*Qx - x).annihilator_of_sum()
           (q*x + 1)*Qx - q*x
           
        """
        A = self.parent()
        return self.map_coefficients(A.sigma())*(A.gen() - A.one())

    def annihilator_of_composition(self, a, solver=None):
        """
        Returns an operator `L` which annihilates all the sequences `f(a(n))`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order.

        INPUT:

        - ``a`` -- a polynomial `u*x+v` where `x` is the generator of the base ring,
          `u` and `v` are integers. 
        - ``solver`` (optional) -- a callable object which applied to a matrix
          with polynomial entries returns its kernel. 

        EXAMPLES::

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
            raise ValueError, "argument has to be of the form u*x+v where u,v are integers"

        if a.degree() > 1:
            raise ValueError, "argument has to be of the form u*x+v where u,v are integers"

        try:
            u = ZZ(a[1]); v = ZZ(a[0])
        except:
            raise ValueError, "argument has to be of the form u*x+v where u,v are rational"

        A = A.change_ring(A.base_ring().fraction_field())
        L = A(self); s = A.sigma(); 
        r = self.order(); x, q = A.is_Q()

        # special treatment for easy cases
        if v != 0:
            L = self.map_coefficients(lambda p: s(p, v))
            return L if u == 1 else L.annihilator_of_composition(u*x)
        elif u == 1:
            return self
        elif u < 0:
            c = [ p(q**(-r)/x) for p in self.coeffs() ]; c.reverse()
            return A(c).numerator().annihilator_of_composition(-u*x)

        # now a = u*x where u > 1 
        from sage.matrix.constructor import Matrix
        if solver == None:
            solver = A._solver()

        p = A.one(); Qu = A.gen()**u # possible improvement: multiplication matrix. 
        mat = [ p.coeffs(padd=r) ]; sol = []

        while len(sol) == 0:

            p = (Qu*p) % L
            mat.append( p.coeffs(padd=r) )
            sol = solver(Matrix(mat).transpose())

        return self.parent()(list(sol[0])).map_coefficients(lambda p: p(x**u))

    def spread(self, p=0):

        op = self.numerator(); A = op.parent(); R = A.base_ring()

        if op.order()==0:
            return []
        elif op[0].is_zero():
            return [infinity]

        if R.is_field():
            R = R.ring() # R = k[x]

        K = PolynomialRing(R.base_ring(), 'y').fraction_field() # F(k[y])
        R = R.change_ring(K) # FF(k[y])[x]

        y = R(K.gen())
        x, q = op.parent().is_Q()
        x = R(x); q = K(q)

        # hack: we find integers n with poly(q,q^n)==0 by comparing the roots of poly(q,Y)==0
        # against a finite set of precomputed powers of q. 
        q_pows = {K.one() : ZZ(0)}; qq = K.one()
        for i in xrange(1, 513):
            qq *= q
            q_pows[qq] = ZZ(i)
            if qq.is_one():
                raise ValueError, "q must not be a root of unity"
        try:
            qq = K.one()
            for i in xrange(1, 513):
                qq /= q
                q_pows[qq] = ZZ(-i)
        except:
            pass

        s = []; r = op.order()
        for (p, _) in (R(op[r])(x*(q**(-r))).resultant(gcd(R(p), R(op[0]))(x*y))).numerator().factor():
            if p.degree() == 1:
                try:
                    s.append(q_pows[K(-p[0]/p[1])])
                except:
                    pass

        s = list(set(s)) # remove duplicates
        s.sort()
        return s

    spread.__doc__ = UnivariateOreOperatorOverUnivariateRing.spread.__doc__

    def _coeff_list_for_indicial_polynomial(self):
        # rewrite self wrt "q-delta"

        R = self.base_ring(); x, q = self.parent().is_Q(); one = R.one()

        if self.is_zero():
            return []

        alg = self.parent().change_var_sigma_delta('q_delta', {x:q*x}, {x:x})

        delta = alg.gen() + alg.one(); delta_k = alg.one(); R = alg.base_ring()
        c = self.coeffs(); out = alg(R(c[0]))

        for i in xrange(self.order()):
            
            delta_k *= delta
            out += R(c[i + 1])*delta_k

        return out.coeffs()

    def _denominator_bound(self):

        A, R, _, L = self._normalize_base_ring()
        x = R.gen(); 

        # primitive factors (anything but powers of x)
        u = super(UnivariateOreOperatorOverUnivariateRing, L)._denominator_bound()

        quo, rem = R(u).quo_rem(x)
        while rem.is_zero():
            quo, rem = quo.quo_rem(x)

        # special factors (powers of x)
        e = 0
        for (q, _) in L.indicial_polynomial(x).factor():
            if q.degree() == 1:
                try:
                    e = min(e, ZZ(-q[0]/q[1]))
                except:
                    pass

        return (quo*x + rem)*x**(-e)

#############################################################################################################

class UnivariateQDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[J], where J is the Jackson q-differentiation J f(x) = (f(q*x) - f(x))/(q*(x-1))
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        R = self.parent(); x = R.base_ring().gen(); qx = R.sigma()(x)
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : (p(qx) - p)/(x*(q-1))

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_Q(self, alg): # j2q
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
        R = self.base_ring(); K = R.base_ring()
        x, q = self.parent().is_J(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:q*x}, {})
        elif not isinstance(alg, type(self.parent())) or not alg.is_Q() or \
             alg.base_ring().base_ring() is not R.base_ring() or K(alg.is_Q()[1]) != K(q) :
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); x, q = alg.is_Q()
        alg = alg.change_ring(R);

        Q = alg.gen(); J = ((q*x - R.one())/(q - R.one()))*Q; J_pow = alg.one()
        out = alg.zero(); 
        coeffs = self.numerator().coeffs()
        d = max( c.degree() for c in coeffs )

        for i in xrange(len(coeffs)):
            if i > 0:
                J_pow *= J
            c = coeffs[i].padded_list(d + 1)
            c.reverse()
            out += alg(map(R, c)) * J_pow            

        return ((q-1)**(len(coeffs)-1)*out).numerator().change_ring(alg.base_ring())

    def annihilator_of_integral(self):
        r"""
        Returns an operator `L` which annihilates all the indefinite `q`-integrals `\int_q f`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order. 

        EXAMPLES::

           sage: R.<x> = ZZ['q'].fraction_field()['x']
           sage: A.<Jx> = OreAlgebra(R, 'Jx')
           sage: ((x-1)*Jx - 2*x).annihilator_of_integral()
           (x - 1)*Jx^2 - 2*x*Jx
           sage: _.annihilator_of_associate(Jx)
           (x - 1)*Jx - 2*x
           
        """
        return self*self.parent().gen()

    def __to_Q_literally(self, gen='Q'):
        """
        This computes the q-recurrence operator which corresponds to ``self`` in the sense
        that `J` is rewritten to `1/(q-1)/x * (Q - 1)`
        """
        x, q = self.parent().is_Q()
        
        alg = self.parent().change_var_sigma_delta(gen, {x:q*x}, {})
        alg = alg.change_ring(self.base_ring().fraction_field())

        if self.is_zero():
            return alg.zero()

        J = ~(q-1)*(~x)*(alg.gen() - alg.one()); J_k = alg.one(); R = alg.base_ring()
        c = self.coeffs(); out = alg(R(c[0]))

        for i in xrange(self.order()):
            
            J_k *= J
            out += R(c[i + 1])*J_k

        return out

    def spread(self, p=0):
        return self.__to_Q_literally().spread(p)

    spread.__doc__ = UnivariateOreOperatorOverUnivariateRing.spread.__doc__

    def _coeff_list_for_indicial_polynomial(self):
        return self.__to_Q_literally()._coeff_list_for_indicial_polynomial()

    def _denominator_bound(self):
        return self.__to_Q_literally()._denominator_bound()


#############################################################################################################

class UnivariateDifferenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[F], where F is the forward difference operator F f(x) = f(x+1) - f(x)
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        if type(f) in (tuple, list):
            return self.to_S('S')(f, **kwargs)
            
        R = self.parent(); x = R.base_ring().gen(); qx = R.sigma()(x)
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : p(qx) - p

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

          sage: R.<x> = ZZ['x']
          sage: A.<Fx> = OreAlgebra(R, 'Fx')
          sage: (Fx^4).to_S(OreAlgebra(R, 'Sx'))
          Sx^4 - 4*Sx^3 + 6*Sx^2 - 4*Sx + 1
          sage: (Fx^4).to_S('Sx')
          Sx^4 - 4*Sx^3 + 6*Sx^2 - 4*Sx + 1
        
        """
        R = self.base_ring(); x = R.gen(); one = R.one(); zero 

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:x+one}, {})
        elif not isinstance(alg, type(self.parent())) or not alg.is_S() or \
             alg.base_ring().base_ring() is not R.base_ring():
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        delta = alg.gen() - alg.one(); delta_k = alg.one(); R = alg.base_ring()
        c = self.coeffs(); out = alg(R(c[0]))

        for i in xrange(self.order()):
            
            delta_k *= delta
            out += R(c[i + 1])*delta_k

        return out

    def to_D(self, alg):
        """
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
        """
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
        return self.coeffs()

    def _denominator_bound(self):
        return self.to_S()._denominator_bound()

#############################################################################################################

class UnivariateEulerDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    """
    Element of an Ore algebra K(x)[T], where T is the Euler differential operator T = x*d/dx
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        R = self.parent(); x = R.base_ring().gen(); 
        if not kwargs.has_key("action"):
            kwargs["action"] = lambda p : x*p.derivative()

        return UnivariateOreOperator.__call__(self, f, **kwargs)

    def to_D(self, alg): # theta2d
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

          sage: R.<x> = ZZ['x']
          sage: A.<Tx> = OreAlgebra(R, 'Tx')
          sage: (Tx^4).to_D(OreAlgebra(R, 'Dx'))
          x^4*Dx^4 + 6*x^3*Dx^3 + 7*x^2*Dx^2 + x*Dx
          sage: (Tx^4).to_D('Dx').to_T(A)
          Tx^4
        
        """
        R = self.base_ring(); x = R.gen(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {}, {x:one})
        elif not isinstance(alg, type(self.parent())) or not alg.is_D() or \
             alg.base_ring().base_ring() is not R.base_ring():
            raise TypeError, "target algebra is not adequate"

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring(); theta = R.gen()*alg.gen(); theta_k = alg.one(); 
        c = self.coeffs(); out = alg(R(c[0]))

        for i in xrange(self.order()):
            
            theta_k *= theta
            out += R(c[i + 1])*theta_k

        return out

    def to_S(self, alg):
        """
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
        """
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

#############################################################################################################

def _rec2list(L, init, n, start, append, padd, deform, singularity_handler=None):
    """
    Common code for computing terms of holonomic and q-holonomic sequences.
    """
        
    r = L.order(); sigma = L.parent().sigma()
    terms = init if append else list(init)
    K = L.base_ring().base_ring().fraction_field()

    if len(terms) >= n:
        return terms
    elif len(terms) < r:

        if not padd:
            raise ValueError, "not enough initial values."
            
        z = K.zero(); padd = r - len(terms)
            
        if append:
            for i in xrange(padd):
                terms.insert(0, z)
            terms = _rec2list(L, terms, min(n, r) + padd, start - padd, True, False, deform, singularity_handler)
            for i in xrange(padd):
                terms.remove(0)
        else:
            terms = _rec2list(L, [z]*padd + terms, min(n, r) + padd, start - padd, False, False, deform, singularity_handler)[padd:]

        return _rec2list(L, terms, n, start, append, False, deform, singularity_handler)

    for i in xrange(r):
        if terms[-i - 1] not in K:
            raise TypeError, "illegal initial value object"

    rec = L.numerator().coeffs(); sigma = L.parent().sigma()
    rec = tuple( -sigma(p, -r) for p in rec )
    lc = -rec[-1]

    for k in xrange(len(terms), n):

        lck = lc(deform(k + start))
        
        if not lck.is_zero():
            terms.append((~lck)*sum(terms[-r + k + i]*rec[i](deform(k + start)) for i in xrange(r)))
        elif singularity_handler is None:
            for i in xrange(k, n):
                terms.append(None)
                return terms
        else:
            terms.append(singularity_handler(k + start))

    return terms
    


