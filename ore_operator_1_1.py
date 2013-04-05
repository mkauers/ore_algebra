
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
from generalized_series import *
from generalized_series import _generalized_series_shift_quotient, _binomial ## why not implied by the previous line?

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
        L = self; A = L.parent(); R = A.base_ring(); K = R.base_ring()

        if R.is_field():
            L = L.numerator()
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
        Computes the polynomial solutions of this operator.

        INPUT:
        
        - ``rhs`` (optional) -- a list of base ring elements
        - ``degree`` (optional) -- bound on the degree of interest.
        - ``solver`` (optional) -- a callable for computing the right kernel
          of a matrix over the base ring's base ring.

        OUTPUT:

        A list of tuples `(p, c_0,...,c_r)` such that `self(p) == c_0*rhs[0] + ... + c_r*rhs[r]`,
        where `p` is a polynomial and `c_0,...,c_r` are constants.

        .. NOTE::

        - Even if no ``rhs`` is given, the output will be a list of tuples ``[(p1,), (p2,),...]``
          and not just a list of plain polynomials.
        - If no ``degree`` is given, a basis of all the polynomial solutions is returned.
          This feature may not be implemented for all algebras. 

        EXAMPLES::

          sage: R.<n> = ZZ['n']; A.<Sn> = OreAlgebra(R, 'Sn')
          sage: L = 2*Sn^2 + 3*(n-7)*Sn + 4
          sage: L.polynomial_solutions((n^2+4*n-8, 4*n^2-5*n+3))
          [(-70*n + 231, 242, -113)]
          sage: L(-70*n + 231)
          -210*n^2 + 1533*n - 2275
          sage: 242*(n^2+4*n-8) - 113*(4*n^2-5*n+3)
          -210*n^2 + 1533*n - 2275

          sage: R.<x> = ZZ['x']; A.<Dx> = OreAlgebra(R, 'Dx')
          sage: L = (x*Dx - 19).lclm( x*Dx - 4 )
          sage: L.polynomial_solutions()
          [(x^4,), (x^19,)]
        
        """
        A = self.parent(); R = A.base_ring(); 
        R_field = R.fraction_field()
        R_ring = R_field.ring()
        K = R_ring.base_ring()
        if not K.is_field():
            R_ring = K[R.gens()]
            K = R_ring.base_ring()

        # clear denominators
        if len(rhs) == 0:
            den = R_ring.one()
        else:
            den = lcm(map(lambda p: R_field(p).denominator(), rhs))
        den = den.lcm(self.denominator())

        L = (den*self).change_ring(R_ring); rhs = tuple(R_ring(den*r) for r in rhs)

        if degree is None:
            degree = L._degree_bound()

        if len(rhs) > 0:
            degree = max(degree, max(map(lambda p: L.order() + p.degree(), rhs)))

        if degree < 0:
            return []

        from sage.matrix.constructor import matrix

        x = R.gen()
        sys = [-L(x**i) for i in xrange(degree + 1)] + list(rhs)
        neqs = max(1, max(map(lambda p: R_ring(p).degree() + 1, sys)))
        sys = map(lambda p: R_ring(p).padded_list(neqs), sys)
        
        if solver is None:
            solver = A._solver(R_ring.base_ring())

        sol = solver(matrix(K, zip(*sys)))

        for i in xrange(len(sol)):
            s = list(sol[i])
            sol[i] = tuple([R_ring(s[:degree+1])] + s[degree+1:])

        return sol         

    def rational_solutions(self, rhs=(), denominator=None, degree=None, solver=None):
        """
        Computes the rational solutions of this operator.

        INPUT:
        
        - ``rhs`` (optional) -- a list of base ring elements
        - ``denominator`` (optional) -- bound on the degree of interest.
        - ``degree`` (optional) -- bound on the degree of interest.
        - ``solver`` (optional) -- a callable for computing the right kernel
          of a matrix over the base ring's base ring.

        OUTPUT:

        A list of tuples `(r, c_0,...,c_r)` such that `self(r) == c_0*rhs[0] + ... + c_r*rhs[r]`,
        where `r` is a rational function and `c_0,...,c_r` are constants.

        .. NOTE::

        - Even if no ``rhs`` is given, the output will be a list of tuples ``[(p1,), (p2,),...]``
          and not just a list of plain polynomials.
        - If no ``denominator`` is given, a basis of all the rational solutions is returned.
          This feature may not be implemented for all algebras. 
        - If no ``degree`` is given, a basis of all the polynomial solutions is returned.
          This feature may not be implemented for all algebras. 

        EXAMPLES::

          sage: R.<x> = ZZ['x']; A.<Dx> = OreAlgebra(R, 'Dx')
          sage: L = ((x+3)*Dx + 2).lclm(x*Dx + 3).symmetric_product((x+4)*Dx-2)
          sage: L.rational_solutions()
          [((x^2 + 8*x + 16)/x^3,), ((x^2 + 8*x + 16)/(x^2 + 6*x + 9),)]
          sage: L.rational_solutions((1, x))
          [((7*x^5 + 21*x^4 + 73*x^2 + 168*x + 144)/(x^5 + 6*x^4 + 9*x^3), 5184, 756), ((4*x^2 + 14*x + 1)/(x^2 + 6*x + 9), 2592, 378), ((7*x^2 + 24*x)/(x^2 + 6*x + 9), 4608, 672)]
          sage: L(_[0][0]) == _[0][1] + _[0][2]*x
          True

          sage: R.<n> = ZZ['n']; A.<Sn> = OreAlgebra(R, 'Sn');
          sage: L = ((n+3)*Sn - n).lclm((2*n+5)*Sn - (2*n+1))
          sage: L.rational_solutions()
          [((-4*n^3 - 8*n^2 + 3)/(4*n^5 + 20*n^4 + 35*n^3 + 25*n^2 + 6*n),), (1/(4*n^2 + 8*n + 3),)]

          sage: L = (2*n^2 - n - 2)*Sn^2 + (-n^2 - n - 1)*Sn + n^2 - 14
          sage: y = (-n + 1)/(n^2 + 2*n - 2)
          sage: L.rational_solutions((L(y),))
          [((-n + 1)/(n^2 + 2*n - 2), 1)]          
        
        """
        A = self.parent(); R = A.base_ring(); 
        R_field = R.fraction_field()
        R_ring = R_field.ring()
        A = A.change_ring(R_field)

        # clear denominators
        if len(rhs) == 0:
            den = R_ring.one()
        else:
            den = lcm(map(lambda p: R_field(p).denominator(), rhs))
        den = R_field(den.lcm(self.denominator()))

        L = den*self; rhs = tuple(den*r for r in rhs)

        if denominator is None:
            denominator = L._denominator_bound()

        sol = (L * A(~R_field(denominator))).polynomial_solutions(rhs, degree=degree, solver=solver)

        for i in xrange(len(sol)):
            sol[i] = tuple([sol[i][0]/denominator] + list(sol[i][1:]))

        return sol

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
            p = R(p)
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

    def indicial_polynomial(self, p, var='alpha'):
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

        if self.is_zero():
            return self.base_ring().base_ring()[var].zero()
        
        elif self.order() == 0:
            return self.base_ring().base_ring()[var].one()
        
        elif (x*p).is_one():
            # at infinity
            inf = 10*(max(1, self.degree()) + max(1, self.order()))
            deg = lambda q: -inf if q.is_zero() else q.degree()
            m = max

        elif x == p:
            # at zero
            inf = 10*(max(1, self.degree()) + max(1, self.order()))
            deg = lambda q: inf if q.is_zero() else q.valuation()
            m = min

        else:
            raise NotImplementedError # leave this case to the subclass
        
        op = self.numerator()._coeff_list_for_indicial_polynomial()
        R = op[0].parent().base_ring()[var]
        y = R.gen()

        A = self.parent()
        q = A.is_Q()
        if not q:
            q = A.is_J()
        if not q:
            my_int = lambda n : n # we are in the ordinary case
        else:
            q = R(q[1]) # we are in the q-case
            my_int = lambda n : (q**n - 1)/(q - 1)

        b = deg(op[0])
        for j in xrange(1, len(op)):
            b = m(b, deg(op[j]) - j)

        s = R.zero(); y_ff_i = R.one()
        for i in xrange(len(op)):
            s = s + op[i][b + i]*y_ff_i
            y_ff_i *= y - my_int(i)

        return s

    def _coeff_list_for_indicial_polynomial(self):
        """
        Computes a list of polynomials such that the usual algorithm for computing indicial
        polynomials applied to this list gives the desired result.

        For example, for differential operators, this is simply the coefficient list of ``self``,
        but for recurrence operators, it is the coefficient list of ``self.to_F()``.

        This is an abstract method.         
        """
        raise NotImplementedError # abstract

    def desingularize(self, p):
        """
        If self has polynomial coefficients, this computes a list of operators [Q1,Q2,...] 
        such that the product B=Qi*self again has polynomial coefficients and the multiplicity 
        of ``p`` in the leading coefficient of B is one less than in the leading coefficient of self.

        INPUT:

        - `p` -- a power of some irreducible polynomial

        EXAMPLES:

          sage: R.<x> = PolynomialRing(QQ); A.<Sx> = OreAlgebra(R, 'Sx')
          sage: L = (3+x)*(9+7*x+x^2)-(33+70*x+47*x^2+12*x^3+x^4)*Sx + (2+x)^2*(3+5*x+x^2)*Sx^2
          sage: p = x+2
          sage: Q = L.desingularize(p)[0]; B=Q*L
          sage: B.denominator().is_one()
          True
          sage: (B.leading_coefficient() / (p+Q.order())).denominator().is_one()
          True
          sage: (B.leading_coefficient() / (p+Q.order())^2).denominator().is_one()
          False

          sage: R.<x> = PolynomialRing(QQ); A.<Sx> = OreAlgebra(R, 'Sx')
          sage: L = (4*x-4)*Sx^2 + (2-4*x^2)*Sx +x*(2*x-1)
          sage: p = x-1
          sage: Q = L.desingularize(p)[0]; B=Q*L
          sage: B.denominator().is_one()
          sage: (B.leading_coefficient() / (p+Q.order())).denominator().is_one()
          False

          sage: L = (-2/45*x^2 + 1/2*x)*Sx^2 + (-2*x^2 - 1/4)*Sx + x^2 + 2/3*x - 6
          sage: p = L.leading_coefficient()
          sage: L.desingularize(p)[0]
          []
        """
        raise NotImplementedError

    def associate_solutions(self, D, p):
        r"""
        If ``self`` is `P`, this returns a list of pairs `(M, m)` such that `D*M = p + m*P`

        INPUT:

        - `D` -- a first order operator with the same parent as ``self``.
          Depending on the algebra, this operator may be constrained to certain choices.
          For example, for differential operators, it can only be `D` (corresponding to
          integration), and for recurrence operators, it can only be `S - 1` (corresponding
          to summation).         
        - `p` -- a nonzero base ring element

        OUTPUT:

        - `M` -- an operator of order ``self.order() - 1`` with rational function coefficients.
        - `m` -- a nonzero rational function.

        Intended application: Express indefinite sums or integrals of holonomic functions in
        terms of the summand/integrand. For example, with `D=S-1` and `P=S^2-S-1` and `p` some
        polynomial, the output `M` is such that

          `\sum_{k=0}^n p(k) F_k = const + M(F_n)`

        where `F_k` denotes the Fibonacci sequence. The rational function `m` does not appear
        in the closed form, it can be regarded as a certificate.         

        This function may not be implemented for every algebra.

        EXAMPLES::

          sage: R.<x> = QQ['x']; A.<Dx> = OreAlgebra(R, 'Dx');
          sage: L = x*Dx^2 + Dx; p = 1  ## L(log(x)) == 0
          sage: L.associate_solutions(Dx, p)
          [(-x^2*Dx + x, -x)]
          sage: (M, m) = _[0]
          sage: Dx*M == p + m*L  ## this implies int(log(x)) == M(log(x)) = x*log(x) - x
          True

          sage: R.<x> = QQ['x']; A.<Dx> = OreAlgebra(R, 'Dx');
          sage: L = x^2*Dx^2 + x*Dx + (x^2 - 1); p = 1  ## L(bessel(x)) == 0
          sage: L.associate_solutions(Dx, p)
          [(-Dx + 1/-x, 1/-x^2)]
          sage: (M, m) = _[0]
          sage: Dx*M == p + m*L  ## this implies int(bessel(x)) == -bessel'(x) -1/x*bessel(x)
          True

          sage: R.<n> = QQ['n']; A.<Sn> = OreAlgebra(R, 'Sn');
          sage: L = Sn^2 - Sn - 1; p = 1  ## L(fib(n)) == 0
          sage: L.associate_solutions(Sn - 1, p)
          [(Sn, 1)]
          sage: (M, m) = _[0]
          sage: (Sn-1)*M == p + m*L  ## this implies sum(fib(n)) == fib(n+1)
          True

          sage: R.<n> = QQ['n']; A.<Sn> = OreAlgebra(R, 'Sn');
          sage: L = Sn^3 - 2*Sn^2 - 2*Sn + 1; p = 1  ## L(fib(n)^2) == 0
          sage: L.associate_solutions(Sn - 1, p)
          [(1/2*Sn^2 - 1/2*Sn - 3/2, 1/2)]
          sage: (Sn-1)*M == p + m*L  ## this implies sum(fib(n)^2) == 1/2*fib(n+2)^2 - 1/2*fib(n+1)^2 - 3/2*fib(n)^2
          True
          
        """
        P = self; A = P.parent(); R = A.base_ring()

        if not isinstance(D, OreOperator) or D.parent() is not A:
            raise TypeError, "operators must live in the same algebra"
        elif p not in R.fraction_field():
            raise TypeError, "p must belong to the base ring"
        elif D.order() != 1:
            raise TypeError, "D must be a first order operator"
        elif self.order() <= 0:
            raise ValueError, "P must have at least order 1"
        elif A.is_F():
            sols = P.to_S('S').associate_solutions(D.to_S('S'), p)
            return [ (M.to_F(str(A.gen())), m) for (M, m) in sols]
        elif A.is_S() is not False or A.is_Q() is not False:
            S = A.gen()
            if not D == S - A.one():
                raise NotImplementedError, "unsupported choice of D: " + str(D)
            # adjoint = sum( (sigma^(-1) - 1)^i * a[i] ), where a[i] is the coeff of D^i in P
            adjoint = A.zero(); coeffs = P.to_F('F').coeffs(); r = P.order()
            for i in xrange(len(coeffs)):
                adjoint += S**(r-i)*(A.one() - S)**i * coeffs[i]
        elif A.is_D() is not False or A.is_T() is not False:
            if D != A.gen():
                raise NotImplementedError, "unsupported choice of D: " + str(D)
            # adjoint = sum( (-D)^i * a[i] ), where a[i] is the coeff of D in P
            adjoint = A.zero(); coeffs = P.coeffs()
            for i in xrange(len(coeffs)):
                adjoint += (-D)**i * coeffs[i]
        else:
            raise NotImplementedError

        sol = adjoint.rational_solutions((-p,))
        A = A.change_ring(A.base_ring().fraction_field())
        sigma = A.sigma(); delta = A.delta()

        for i in xrange(len(sol)):
            if sol[i][1].is_zero():
                sol[i] = None; continue
            rat = sol[i][0]/sol[i][1]
            DM = p + rat*P; M = A.zero()
            while DM.order() > 0:
                r = DM.order()
                a = DM.leading_coefficient()
                # DM = a*D^r + ...
                #    = (a*D)*D^(r-1) + ...
                #    = (D*s(a, -1) - d(s(a, -1)))*D^(r-1) + ...
                M += sigma(a, -1)*(D**(r-1))
                DM -= D*sigma(a, -1)*(D**(r-1))
            sol[i] = (M, rat)

        return filter(lambda p: p is not None, sol)

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

        A = self.parent(); K = A.base_ring().fraction_field(); A = A.change_ring(K); R = K['Y']
        if solver == None:
            solver = A._solver(K)

        if self == A.one() or a == K.gen():
            return self
        elif a in K.ring() and K.ring()(a).degree() == 1:
            # special handling for easy case  a == alpha*x + beta
            a = K.ring()(a); alpha, beta = a[1], a[0]
            x = self.base_ring().gen(); D = A.associated_commutative_algebra().gen()
            L = A(self.polynomial()(D/alpha).map_coefficients(lambda p: p(alpha*x + beta)))
            return L.normalize()
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

          sage: R.<x> = QQ['x']; A.<Dx> = OreAlgebra(R, 'Dx')
          sage: L = (6+6*x-3*x^2) - (10*x-3*x^2-3*x^3)*Dx + (4*x^2-6*x^3+2*x^4)*Dx^2
          sage: L.generalized_series_solutions()
          [x^3*(1 + 3/2*x + 7/4*x^2 + 15/8*x^3 + 31/16*x^4 + O(x^5)), x^(1/2)*(1 + 3/2*x + 7/4*x^2 + 15/8*x^3 + 31/16*x^4 + O(x^5))]
          sage: map(L, _)
          [0, 0]

          sage: L = (1-24*x+96*x^2) + (15*x-117*x^2+306*x^3)*Dx + (9*x^2-54*x^3)*Dx^2
          sage: L.generalized_series_solutions(2)
          sage: [x^(-1/3)*(1 + x + 8/3*x^2 + O(x^3)), x^(-1/3)*((1 + x + 8/3*x^2 + O(x^3))*log(x) + x - 59/12*x^2 + O(x^3))]
          sage: map(L, _)
          [0, 0]

          sage: L = 216*(1+x+x^3) + x^3*(36-48*x^2+41*x^4)*Dx - x^7*(6+6*x-x^2+4*x^3)*Dx^2
          sage: L.generalized_series_solutions(3)
          [exp(3*x^(-2))*x^(-2)*(1 + 91/12*x^2 + O(x^3)), exp(-2*x^(-3) + x^(-1))*x^2*(1 + 41/3*x + 2849/36*x^2 + O(x^3))]
          sage: map(L, _)
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
          Number Field in a_0 with defining polynomial c^2 - 2

        """

        R = self.base_ring()
        D = self.parent().gen()
        
        if self.is_zero():
            raise ZeroDivisionError, "infinite dimensional solution space"
        elif self.order() == 0:
            return []
        elif R.characteristic() > 0:
            raise TypeError, "cannot compute generalized solutions for this coefficient domain"
        elif R.is_field() or not R.base_ring().is_field():
            return self._normalize_base_ring()[-1].generalized_series_solutions(n, base_extend, ramification, exp)
        elif not (R.base_ring() is QQ or is_NumberField(R.base_ring())):
            raise TypeError, "cannot compute generalized solutions for this coefficient domain"

        solutions = []

        # solutions with exponential parts

        if exp is True:
            exp = QQ(self.degree() * self.order()) # = infinity
        elif exp is False:
            exp = QQ.zero()
        if exp not in QQ:
            raise ValueError, "illegal option value encountered: exp=" + str(exp)

        # search for a name which is not yet used as generator in (some subfield of) R.base_ring()
        # for in case we need to make algebraic extensions.

        # TODO: should we flatten multiple extensions? 

        K = R.base_ring(); names = []
        while K is not QQ:
            names.append(str(K.gen()))
            K = K.base_ring()
        i = 0; newname = 'a_0'
        while newname in names:
            i = i + 1; newname = 'a_' + str(i)

        if exp > 0:

            points = []
            c = self.coeffs()
            for i in xrange(self.order() + 1):
                if not c[i].is_zero():
                    points.append((QQ(i), QQ(c[i].valuation())))

            exponents = []; y = R.base_ring()['y'].gen(); k = 0
            while k < len(points) - 1:
                (i1, j1) = points[k]; p = None; s = self.degree() + 1; 
                for l in xrange(k + 1, len(points)):
                    (i2, j2) = points[l]; s2 = (j2 - j1)/(i2 - i1)
                    if s2 < s:
                        s = s2; k = l; p = c[i1][j1]*(y**i1) + c[i2][j2]*(y**i2);
                    elif s2 == s:
                        k = l; p += c[i2][j2]*(y**i2);
                e = 1 - s; 
                if e < 0 and -e < exp and p is not None and (ramification or e in ZZ):
                    exponents.append((e, p(e*y)))

            x = R.gen(); K = R.base_ring(); 
            for (e, p) in exponents:
                for (q, _) in p.factor():
                    if q == y:
                        continue
                    elif q.degree() == 1:
                        c = -K(q[0]/q[1])
                    elif base_extend:
                        c = K.extension(q, newname).gen()
                    else:
                        continue
                    a = e.numerator(); b = e.denominator()
                    G = GeneralizedSeriesMonoid(c.parent(), x, "continuous")
                    s = G(R.one(), exp = e*c*(x**(-a)), ramification = b)
                    L = self.annihilator_of_composition(x**b).symmetric_product(x**(1-a)*D + a*c)
                    sol = L.generalized_series_solutions(n, base_extend, ramification, -a)
                    solutions = solutions + map(lambda f: s*f.substitute(~b), sol)

        # tails
        indpoly = self.indicial_polynomial(R.gen(), 's')
        s = indpoly.parent().gen()
        x = R.gen()

        for (c, e) in _shift_factor(indpoly):

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

                coeffs = _rec2list(L.to_S('S'), [K.one()], n, alpha, False, True, lambda p:p)
                solutions.append(G(PS(coeffs, n), exp=alpha))

            else:
                # there may be logarithms, use general code
                L = L.base_extend(K[s].fraction_field()[x]).to_S('S')

                f = f0 = indpoly.parent().one()
                for (a, b) in e:
                    f0 *= c(s + a)**b

                for i in xrange(e[-1][0]):
                    f *= f0(s + i + 1) 

                coeffs = _rec2list(L, [f], n, s, False, True, lambda p:p)

                # If W(s, x) denotes the power series with the above coefficient array,
                # then [ (d/ds)^i ( W(s, x)*x^s ) ]_{s=a} is a nonzero solution for every
                # root a = alpha - e[j][0] of f0 and every i=0..e[j][1]-1.

                # D_s^i (W(s, x)*x^s) = (D_s^i W + i*log(x)*D_s^(i-1) W + binom(i,2)*log(x)^2 D_s^(i-2) W + ... )*x^s.

                m = sum([ee[1] for ee in e])
                der = [coeffs]
                while len(der) < m:
                    der.append(map(lambda g: g.derivative(), der[-1]))

                accum = 0
                for (a, b) in e:
                    der_a = dict()
                    for i in xrange(accum + b):
                        der_a[i] = PS(map(lambda g: g(alpha - a), der[i]), len(der[i]))
                    for i in xrange(accum, accum + b):
                        sol = []
                        for j in xrange(i + 1):
                            sol.append(_binomial(i, j)*der_a[j])
                        sol.reverse()
                        solutions.append(G(sol, exp=alpha - a, make_monic=True))
                    accum += b
        
        return solutions

    def indicial_polynomial(self, p, var='alpha'):
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
          (-1) * 5 * 2^2 * (alpha - 5) * (alpha - 1) * alpha
          sage: L.indicial_polynomial(1/x).factor()
          2 * (alpha - 7) * (alpha - 5)
          sage: L.indicial_polynomial(x^2+1).factor()
          5 * 7 * 2^2 * (alpha - 1) * alpha * (2*alpha - 7)
        
        """

        x = p.parent().gen()

        if (x*p).is_one() or p == x:
            return UnivariateOreOperatorOverUnivariateRing.indicial_polynomial(self, p, var=var)

        op = self.numerator()

        R = op.parent()
        L = R.base_ring() # k[x]
        if L.is_field():
            L = L.ring()
        K = PolynomialRing(L.base_ring(),var) # k[alpha]
        L = L.change_ring(K.fraction_field()) # FF(k[alpha])[x]

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

    def companion_matrix(self):
        r"""
        If ``self`` is an operator of order `r`, returns an `r` by `r` matrix
        `M` such that for any sequence `c_i` annihilated by ``self``,
        `[c_{i+1}, c_{i+2}, \ldots, c_{i+r}]^T = M(i) [c_i, c_{i+1}, \ldots, c_{i+r-1}]^T`

        EXAMPLES::

            sage: R.<n> = QQ['n']
            sage: A.<Sn> = OreAlgebra(R, 'Sn')
            sage: M = ((-n-4)*Sn**2 + (5+2*n)*Sn + (3+3*n)).companion_matrix()
            sage: M
            [                0                 1]
            [(3*n + 3)/(n + 4) (2*n + 5)/(n + 4)]
            sage: initial = Matrix([[1],[1]])
            sage: [prod(M(k) for k in range(n, -1, -1)) * initial for n in range(10)]
            [
            [1]  [2]  [4]  [ 9]  [21]  [ 51]  [127]  [323]  [ 835]  [2188]
            [2], [4], [9], [21], [51], [127], [323], [835], [2188], [5798]
            ]

        """
        from sage.matrix.constructor import Matrix
        ring = self.base_ring().fraction_field()
        r = self.order()
        M = Matrix(ring, r, r)
        for i in range(r-1):
            M[i, i+1] = 1
        for j in range(r):
            M[r - 1, j] = self[j] / (-self[r])
        return M

    def forward_matrix_bsplit(self, n, start=0):
        r"""
        Uses division-free binary splitting to compute a product of ``n``
        consecutive companion matrices of ``self``.

        If ``self`` annihilates some sequence `c` of order `r`, this
        allows rapidly computing `c_n, \ldots, c_{n+r-1}` (or just `c_n`)
        without generating all the intermediate values.

        INPUT::

        - ``n`` -- desired number of terms to move forward
        - ``start`` (optional) -- starting index. Defaults to zero.

        OUTPUT::

        A pair `(M, Q)` where `M` is an `r` by `r` matrix and `Q`
        is a scalar, such that `M / Q` is the product of the companion
        matrix at `n` consecutive indices.

        We have `Q [c_{s+n}, \ldots, c_{s+r-1+n}]^T = M [c_s, c_{s+1}, \ldots, c_{s+r-1}]^T`,
        where `s` is the initial position given by ``start``.

        EXAMPLES::

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

        TODO: this should detect if the base coefficient ring is QQ (etc.)
        and then switch to ZZ (etc.) internally.
        """
        from sage.matrix.matrix_space import MatrixSpace
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


    def delta_matrix(self, m):
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
        """
        Assuming the coefficients of self are in `R[x][k]`,
        computes the nth forward matrix with the parameter `x`
        evaluated at ``value``, using rectangular splitting
        with a step size of `m`.

        TESTS:

            sage: from sage.all import Matrix, randrange
            sage: R = ZZ
            sage: Rx = R['x']; x = Rx.gen()
            sage: Rxk = Rx['k']; k = Rxk.gen()
            sage: Rxks = OreAlgebra(Rxk, 'Sk')
            sage: V = QQ
            sage: Vks = OreAlgebra(V['k'], 'Sk')
            sage: for i in range(1000):
            sage:     A = Rxks.random_element(randrange(1,4))
            sage:     r = A.order()
            sage:     v = V.random_element()
            sage:     initial = [V.random_element() for i in range(r)]
            sage:     start = randrange(0,5)
            sage:     n = randrange(0,30)
            sage:     m = randrange(0,10)
            sage:     B = Vks(list(A.polynomial()(x=v)))
            sage:     singular = any([B[r](i) == 0 for i in range(n+r)])
            sage:     M, Q = A.forward_matrix_param_rectangular(v, n, m=m, start=start)
            sage:     if Q == 0:
            sage:         assert singular
            sage:     else:
            sage:         V1 = M * Matrix(initial).transpose() / Q
            sage:         values = B.to_list(initial, n + r, start)
            sage:         V2 = Matrix(values[-r:]).transpose()
            sage:         if V1 != V2:
            sage:             raise ValueError

        """
        from sage.matrix.matrix_space import MatrixSpace

        assert n >= 0
        r = self.order()

        indexed_ring = self.base_ring()
        parametric_ring = indexed_ring.base_ring()
        scalar_ring = parametric_ring.base_ring()

        coeffs = list(self)
        param_degree = max(max(d.degree() for d in c) for c in coeffs)

        # Step size
        if m is None:
            m = floor(n ** 0.25)
        m = max(m, 1)
        m = min(m, n)

        delta_M, delta_Q = self.delta_matrix(m)

        # Precompute all needed powers of the parameter value
        # TODO: tighter degree bound (by inspecting the matrices)
        eval_degree = m * param_degree
        num_powers = eval_degree + 1

        power_table = [0] * num_powers
        for i in xrange(num_powers):
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
            for i in xrange(1, poly.degree() + 1):
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
            for j in xrange(a, b):
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

        K = R.base_ring()['y'] # k[y]
        R0 = R
        R = R.change_ring(K.fraction_field()) # FF(k[y])[x]
        A = A.change_ring(R)

        y = R(K.gen())
        x = R.gen()

        if op.order()==0:
            return []
        elif op[0].is_zero():
            return [infinity]
        else:
            s = []; r = op.order()
            for (q, _) in R(gcd(R0(p), R0(op[r])))(x - r).resultant(R(op[0])(x + y)).numerator().factor():
                if q.degree() == 1:
                    try:
                        s.append(ZZ(-q[0]/q[1]))
                    except:
                        pass
            s = list(set(s)) # remove duplicates
            s.sort()
            return s

    def generalized_series_solutions(self, n=5, superexponential_part=True, exponential_part=True, \
                                     subexponential_part=True, polynomial_part=True, log_part=True, \
                                     base_extend=True, working_precision=3): 
        r"""
        Returns the generalized series solutions of this operator.

        These are solutions of the form

          `(x/e)^(x*u/v)\rho^x\exp\bigl(c_1 x^{1/m} +...+ c_{v-1} x^{1-1/m}\bigr)x^\alpha p(x^{-1/m},\log(x))`

        where

        * `e` is Euler's constant (2.71...)
        * `v` is a positive integer (the object's "ramification")
        * `u` is an integer; the term `(x/e)^(v/u)` is called the "superexponential part" of the solution
        * `\rho` is an element of an algebraic extension of the coefficient field `K`
          (the algebra's base ring's base ring); the term `\rho^x` is called the "exponential part" of
          the solution
        * `c_1,...,c_{v-1}` are elements of `K(\rho)`; the term `\exp(...)` is called the "subexponential
          part" of the solution
        * `m` is a positive integer multiple of `v`
        * `\alpha` is an element of some algebraic extension of `K(\rho)`; the term `n^\alpha` is called
          the "polynomial part" of the solution (even if `\alpha` is not an integer)
        * `p` is an element of `K(\rho)(\alpha)[[x]][y]`. It is called the "expansion part" of the solution.

        An operator of order `r` has exactly `r` linearly independent solutions of this form.
        This method computes them all, unless the flags specified in the arguments rule out
        some of them.

        Generalized series solutions are asymptotic expansions of sequences annihilated by the operator. 

        At present, the method only works for operators where `K` is either
        QQ or a number field (i.e., no finite fields, no formal parameters). 

        INPUT:

        - ``n`` (default: 5) -- minimum number of terms in the expansions parts to be computed
          in addition to those needed to separate all solutions from each other.
        - ``superexponential_part`` (default: True) -- if set to False, only compute solutions where `u=0`
        - ``exponential_part`` (default: True) -- if set to False, only compute solutions where `\rho=1`
        - ``subexponential_part`` (default: True) -- if set to False, only compute solutions where `c_i=0`
          for all `i`
        - ``polynomial_part`` (default: True) -- if set to False, only compute solutions where `\alpha=0`
        - ``log_part`` (default: True) -- if set to False, only compute solutions without logarithmic terms
        - ``base_extend`` (default: ``True``) -- if set to False, only compute solutions where `\rho` and
          `\alpha` belong to `K`. 
        - ``working_precision`` (default: 5) -- number of terms the algorithm should use in addition
          to ``n`` during the computation in order to account for possible cancellations. This option
          only affects the efficiency but not the correctness. If it turns out during the calculation
          that the value was to low, the calculation will be restarted with the this value doubled.

        OUTPUT:

        - a list of ``DiscreteGeneralizedSeries`` objects forming a fundamental system for this operator. 

        .. NOTE::

          - Different solutions may require different algebraic extensions. Thus in the list returned
            by this method, the coefficient fields of different series typically do not coincide.
          - If a solution involves an algebraic extension of the coefficient field, then all its
            conjugates are solutions, too. But only one representative is listed in the output.

        EXAMPLES::

          sage: R.<n> = QQ['n']; A.<Sn> = OreAlgebra(R, 'Sn')
          sage: (Sn - (n+1)).generalized_series_solutions()
          [(n/e)^n*n^(1/2)*(1 + 1/12*n^(-1) + 1/288*n^(-2) - 139/51840*n^(-3) - 571/2488320*n^(-4) + O(n^(-5)))]
          sage: ((n + 2)*Sn^2 + (-2*n - 3)*Sn + n + 1).generalized_series_solutions()
          [1 + O(n^(-5)), (1 + O(n^(-5)))*log(n) + 1/2*n^(-1) - 1/12*n^(-2) + 1/120*n^(-4) + O(n^(-5))]
          
        """
        L = self.numerator().primitive_part()
        origcoeffs = coeffs = L.coeffs()
        r = len(coeffs) - 1
        if len(coeffs) == 0:
            raise ZeroDivisionError, "everything is a solution of the zero operator"
        elif len(coeffs) == 1:
            return []

        K = coeffs[0].base_ring()
        if K is not QQ and not is_NumberField(K):
            raise TypeError, "unexpected coefficient domain: " + str(K)

        x = coeffs[0].parent().gen()
        solutions = []
        subs = _generalized_series_shift_quotient
        w_prec = max(working_precision, 2*L.order())

        # 1. superexponential parts
        points = filter(lambda p: p[1] >= 0, [ (i, coeffs[i].degree()) for i in xrange(len(coeffs)) ])
        deg = max(map(lambda p: p[1], points))
        degdiff = deg - min(map(lambda p: p[1], points))

        k = 0
        while k < len(points) - 1:
            (i1, j1) = points[k]; s = -QQ(deg + 5) # = -infinity
            for l in xrange(k + 1, len(points)):
                (i2, j2) = points[l]; s2 = QQ(j2 - j1)/QQ(i2 - i1)
                if s2 >= s:
                    s = s2; k = l
            if s.is_zero():
                solutions.append( [QQ.zero(), [c.shift(w_prec - deg) for c in coeffs ]] )
            elif superexponential_part:
                v = s.denominator(); newcoeffs = []
                newdeg = max([ coeffs[i].degree() - i*s for i in xrange(len(coeffs)) if coeffs[i] != 0 ])
                for i in xrange(len(coeffs)):
                    c = (coeffs[i](x**v)*subs(x, prec=w_prec, shift=i, gamma=-s)).shift(-v*newdeg)
                    newcoeffs.append(c)
                solutions.append( [-s, newcoeffs ] )

        # 2. exponential parts
        refined_solutions = []
        for (gamma, coeffs) in solutions:
            deg = max([p.degree() for p in coeffs])
            char_poly = K['rho']([ c[deg] for c in coeffs ])
            if exponential_part:
                for (cp, _) in char_poly.factor():
                    if cp.degree() == 1:
                        K = cp[0].parent(); rho = -cp[0]/cp[1]
                    elif base_extend:
                        K = cp.base_ring().extension(cp, 'rho'); rho = K.gen()
                    if not rho.is_zero():
                        refined_solutions.append( [gamma, rho, [ coeffs[i].change_ring(K)*(rho**i) for i in xrange(len(coeffs)) ]] )
                
            elif char_poly(K.one()).is_zero():
                refined_solutions.append( [gamma, K.one(), coeffs] )

        # 3. subexponential parts
        solutions = refined_solutions; refined_solutions = []
        for (gamma, rho, coeffs) in solutions:

            v = gamma.denominator(); zero = rho.parent().zero(); one = rho.parent().zero()
            x = x.change_ring(rho.parent())

            if v.is_one():
                refined_solutions.append( [gamma, rho, ([], ZZ.one()), coeffs] )
                continue

            subexp = [zero] * (v - 1);

            mult = ZZ.zero(); extra = 0;
            while extra == 0:
                mult += 1
                extra = sum(j**mult * coeffs[j][v*w_prec] for j in xrange(1, len(coeffs)))
            if mult > 1:
                for j in xrange(len(coeffs)):
                    coeffs[j] = coeffs[j](x**mult)

            ram = v*mult
            deg = w_prec*ram 

            for i in xrange(v - 1, 0, -1):
                tokill = sum(c[deg - v + i] for c in coeffs)
                killer = [zero] * (v - 1);
                subexp[i - 1] = killer[i - 1] = -tokill/((_binomial(i/ram, mult)*extra))
                for j in xrange(1, len(coeffs)):
                    coeffs[j] = (coeffs[j]*subs(x, prec=w_prec, shift=j, subexp=(killer, mult))(x**v)).shift(-ram*w_prec)

            if subexponential_part or all(c == 0 for c in subexp):
                refined_solutions.append( [gamma, rho, (subexp, mult), coeffs] )

        # 4. polynomial parts and expansion 
        solutions = refined_solutions; refined_solutions = []
        for (gamma, rho, subexp, coeffs) in solutions:
            
            K = rho.parent(); ram = gamma.denominator()*subexp[1]
            K = K['s'].fraction_field(); s = K.gen(); x = x.change_ring(K)
            rest = sum(coeffs[i].change_ring(K)*subs(x, w_prec, i, alpha=s)(x**ram) for i in xrange(len(coeffs)))
            ind_poly = rest.leading_coefficient().numerator()
            for (p, e) in _shift_factor(ind_poly, ram):
                alpha = -p[0]/p[1] if p.degree() == 1 else p.base_ring().extension(p, 'alpha').gen()
                if polynomial_part or alpha in ZZ:
                    e.reverse()
                    refined_solutions.append([gamma, rho, subexp, alpha, e])

        # 5. expansion        
        solutions = refined_solutions; refined_solutions = []
        for (gamma, rho, (subexp, mult), alpha, e) in solutions:

            K = alpha.parent(); ram = ZZ(gamma.denominator()*mult); prec = n + w_prec
            G = GeneralizedSeriesMonoid(K, self.base_ring().gen(), 'discrete')
            K = K['s'].fraction_field(); s = K.gen(); z = K.zero();
            x = x.change_ring(K); 

            coeffs = [(origcoeffs[i](x**ram)*subs(x, prec, i, gamma, rho, (subexp, mult))) for i in xrange(r + 1)]
            deg = max(map(lambda p: p.degree(), coeffs))
            coeffs = [coeffs[i].shift(ram*prec - deg).change_ring(K) for i in xrange(r + 1)]

            f = f0 = K.one();

            for (a, b) in e:
                f0 *= (s  - (alpha - a))**b

            for i in xrange(ram*e[0][0]):
                f *= f0(s - (i + 1)/ram)

            def normalize_ratfun(rat): # sage deficiency work around
                lc = ~rat.denominator().leading_coefficient()
                return (lc*rat.numerator())/(lc*rat.denominator())

            exp = [f] # coeffs of n^s, n^(s-1/ram), n^(s-2/ram), ...
            rest = sum((coeffs[i]*subs(x, prec, i, alpha=s)(x**ram)).shift(-ram*prec) for i in xrange(r + 1))
            #    = L(n^s) = indpoly(s)*n^(s - r) + ...
            f0 = rest.leading_coefficient()
            rest *= f
            rr = prec - rest.degree()/ram

            for k in xrange(1, ram*n):
                # determine coeff of n^(s - k/ram) in exp so as to kill coeff of n^(s - rr - k/ram) of rest
                newcoeff = -rest[ram*(prec - rr) - k]/f0(s - k/ram)
                rest += sum((newcoeff*coeffs[i]*subs(x, prec, i, alpha=s - k/ram)(x**ram)).shift(-ram*prec - k) \
                            for i in xrange(r + 1))
                rest = rest.map_coefficients(normalize_ratfun)
                exp.append(newcoeff)

            m = sum([ee[1] for ee in e])
            der = [exp]
            while len(der) < m:
                der.append(map(lambda g: g.derivative(), der[-1]))

            accum = 0
            for (a, b) in e:
                der_a = dict()
                for i in xrange(accum + b):
                    der_a[i] = map(lambda g: g(alpha - a), der[i])
                for i in xrange(accum, accum + b):
                    sol = []
                    for j in xrange(i + 1):
                        bin = _binomial(ZZ(i), ZZ(j))
                        sol.append(map(lambda q: q*bin, der_a[j]))
                    sol.reverse()
                    refined_solutions.append(G([gamma, gamma.denominator()*mult, rho, subexp, alpha - a, sol], \
                                               make_monic=True))
                accum += b

        return refined_solutions


    def desingularize(self,p):
        op = self.numerator()

        R = op.parent().base_ring()
        if R.is_field():
            R = R.ring()
        K = PolynomialRing(R.base_ring(), 'y').fraction_field()
        RY = R.change_ring(K)
        y = RY(K.gen())
        S = op.parent().gen()

        # compute denominator bound
        u = op.leading_coefficient()
        v = [0,0]
        for i in RY(u).resultant(RY(p)+y).factor():
            if i[0].degree()==1 and i[0][0].denominator()==1 and i[0][0]>0 and i[1] > v[1]:
                v = i
        v = v[1]

        # compute order bound
        k=1
        try:
            r = -(max(op.spread(p))+op.order())
        except:
            return []
        e = k+r*v
        neqs = op.order()+r

        # solve linear system
        q = op.parent().sigma()(p,r)**e
        I = R.ideal([q])
        L = R.quotient_ring(I)

        sys = [((S**i)*op).polynomial().padded_list(neqs+1) for i in range(r+1)]
        sys= map(lambda i: map(lambda p: L(p),i), sys)

        sol = op.parent()._solver(L)(matrix(zip(*sys)))

        # assemble solutions
        for i in range(len(sol)):
            d = 0
            s = sol[i]
            for j in range(len(s)):
                d = d+(s[j].lift()/q)*S**j
            sol[i] = d
        return sol

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

    def __to_J_literally(self, gen='J'):
        """
        Rewrites ``self`` in terms of `J`
        """
        A = self.parent()
        R = A.base_ring(); x, q = A.is_Q(); one = R.one()
        A = A.change_var_sigma_delta(gen, {x:q*x}, {x:one})

        if self.is_zero():
            return A.zero()

        Q = (q - 1)*x*A.gen() + 1; Q_pow = A.one(); 
        c = self.coeffs(); out = A(R(c[0]))

        for i in xrange(self.order()):

            Q_pow *= Q
            out += R(c[i + 1])*Q_pow

        return out

    def _coeff_list_for_indicial_polynomial(self):
        return self.__to_J_literally().coeffs()

    def _denominator_bound(self):

        A, R, _, L = self._normalize_base_ring()
        x = R.gen(); 

        # primitive factors (anything but powers of x)
        u = UnivariateOreOperatorOverUnivariateRing._denominator_bound(L)

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

        A = self.parent(); x, q = A.is_J(); qx = A.sigma()(x)
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

          sage: R.<x> = QQ['x']
          sage: A.<Jx> = OreAlgebra(R, 'Jx', q=2)
          sage: (Jx-1).lclm((1-x)*Jx-1).power_series_solutions()
          [x^2 + x^3 + 3/5*x^4 + 11/35*x^5 + O(x^6), 1 + x - 2/7*x^3 - 62/315*x^4 - 146/1395*x^5 + O(x^6)]
    
        """
        _, q = self.parent().is_J()
        return _power_series_solutions(self, self.to_Q('Q'), n, lambda n: q**n)

    def __to_Q_literally(self, gen='Q'):
        """
        This computes the q-recurrence operator which corresponds to ``self`` in the sense
        that `J` is rewritten to `1/(q-1)/x * (Q - 1)`
        """
        x, q = self.parent().is_J()
        
        alg = self.parent().change_var_sigma_delta(gen, {x:q*x}, {})
        alg = alg.change_ring(self.base_ring().fraction_field())

        if self.is_zero():
            return alg.zero()

        J = ~(q-1)*(~x)*(alg.gen() - alg.one()); J_k = alg.one(); R = alg.base_ring()
        c = self.coeffs(); out = alg(R(c[0]))

        for i in xrange(self.order()):
            
            J_k *= J
            out += R(c[i + 1])*J_k

        return out.numerator().change_ring(R.ring())

    def spread(self, p=0):
        return self.__to_Q_literally().spread(p)

    spread.__doc__ = UnivariateOreOperatorOverUnivariateRing.spread.__doc__

    def _coeff_list_for_indicial_polynomial(self):
        return self.coeffs()

    def _denominator_bound(self):
        return self.__to_Q_literally()._denominator_bound()

    def symmetric_product(self, other, solver=None):

        if not isinstance(other, UnivariateOreOperator):
            raise TypeError, "unexpected argument in symmetric_product"

        if self.parent() != other.parent():
            A, B = canonical_coercion(self, other)
            return A.symmetric_product(B, solver=solver)

        A = self.__to_Q_literally(); B = other.__to_Q_literally()

        C = A.symmetric_product(B, solver=solver)._normalize_base_ring()[-1]
        C = C._UnivariateQRecurrenceOperatorOverUnivariateRing__to_J_literally(str(self.parent().gen()))

        try:
            return self.parent()(C.numerator().coeffs())
        except:
            return C

    symmetric_product.__doc__ = UnivariateOreOperator.symmetric_product.__doc__

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

    def symmetric_product(self, other, solver=None):

        if not isinstance(other, UnivariateOreOperator):
            raise TypeError, "unexpected argument in symmetric_product"

        if self.parent() != other.parent():
            A, B = canonical_coercion(self, other)
            return A.symmetric_product(B, solver=solver)

        A = self.to_S('S'); B = other.to_S(A.parent())
        return A.symmetric_product(B, solver=solver).to_F(self.parent())

    symmetric_product.__doc__ = UnivariateOreOperator.symmetric_product.__doc__

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

    def symmetric_product(self, other, solver=None):

        if not isinstance(other, UnivariateOreOperator):
            raise TypeError, "unexpected argument in symmetric_product"

        if self.parent() != other.parent():
            A, B = canonical_coercion(self, other)
            return A.symmetric_product(B, solver=solver)

        A = self.to_D('D'); B = other.to_D(A.parent())
        return A.symmetric_product(B, solver=solver).to_T(self.parent())

    symmetric_product.__doc__ = UnivariateOreOperator.symmetric_product.__doc__

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

    if None in terms:
        for k in xrange(len(terms), n):
            terms.append(None)
        return terms

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
    
def _power_series_solutions(op, rec, n, deform):
    """
    Common code for computing terms of holonomic and q-holonomic power series.
    """

    L = op.numerator()
    factors = L.indicial_polynomial(L.base_ring().gen()).factor()
    orders = []

    for (p, _) in factors:
        if p.degree() == 1:
            try:
                alpha = ZZ(-p[0]/p[1])
                if alpha >= 0:
                    orders.append(alpha)
            except:
                pass

    if len(orders) == 0:
        return orders # no power series solutions

    r = L.order()
    maxexp = max(orders) + max(n, r)
    K = L.base_ring().base_ring().fraction_field(); zero = K.zero(); one = K.one()
        
    from sage.rings.power_series_ring import PowerSeriesRing
    R = PowerSeriesRing(K, str(L.base_ring().gen()))
    x = R.gen()

    sols = []
    for alpha in orders:

        p = _rec2list(rec, [one], maxexp - alpha, alpha, False, True, deform, lambda k: zero)
        p = (x**alpha) * R(p, maxexp - alpha - 1)

        if L(p).is_zero(): # L(p) nonzero indicates series involving logarithms. 
            sols.append(p)

    return sols

def _shift_factor(p, ram=ZZ.one()):
    """
    Returns the roots of p in an appropriate extension of the base field, sorted according to
    shift equivalence classes.

    INPUT:

    - ``p`` -- a univariate polynomial over QQ or a number field
    - ``ram`` (optional) -- positive integer

    OUTPUT:

    A list of pairs (q, e) where

    - q is an irreducible factor of p
    - e is a tuple of pairs (a, b) of nonnegative integers 
    - p = c*prod( q(x+a/ram)^b for (q, e) in output list for (a, b) in e ) for some nonzero constant c
    - e[0][0] == 0, and e[i][0] < e[i+1][0] for all i 
    - any two distinct q have no roots at integer distance.

    Note: rootof(q) is the largest root of every class. The other roots are given by rootof(q) - e[i][0]/ram.
        
    """

    classes = []

    for (q, b) in p.factor():

        d = q.degree()
        if d < 1:
            continue

        q0, q1 = q[d], q[d - 1]

        # have we already seen a member of the shift equivalence class of q? 
        new = True; 
        for i in xrange(len(classes)):
            u = classes[i][0]
            if u.degree() != d:
                continue
            u0, u1 = u[d], u[d - 1]
            a = ram*(u1*q0 - u0*q1)/(u0*q0*d)
            if a not in ZZ:
                continue
            # yes, we have: p(x+a) == u(x); u(x-a) == p(x)
            # register it and stop searching
            a = ZZ(a); new = False
            if a < 0:
                classes[i][1].append((-a, b))
            elif a > 0:
                classes[i][0] = q
                classes[i][1] = [(n+a,m) for (n,m) in classes[i][1]]
                classes[i][1].append((0, b))
            break

        # no, we haven't. this is the first.
        if new:
            classes.append( [q, [(0, b)]] )

    for c in classes:
        c[1].sort(key=lambda e: e[0])

    return classes
        
    
