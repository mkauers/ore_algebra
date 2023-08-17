"""
Univariate operators over univariate rings

Special classes for operators living in algebras with one generator with base rings that also have
one generator.

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

from sage.arith.all import previous_prime as pp
from sage.arith.all import gcd, lcm, srange
from sage.matrix.constructor import matrix
from sage.misc.all import prod
from sage.misc.cachefunc import cached_method
from sage.misc.lazy_import import lazy_import
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.structure.factorization import Factorization

from .tools import clear_denominators, q_log, shift_factor
from .ore_operator import OreOperator, UnivariateOreOperator

# re-export classes moved to separate modules for backward compatibility
lazy_import("ore_algebra.differential_operator_1_1", [
    "UnivariateDifferentialOperatorOverUnivariateRing",
    "UnivariateEulerDifferentialOperatorOverUnivariateRing",
])
lazy_import("ore_algebra.q_operator_1_1", [
    "UnivariateQDifferentialOperatorOverUnivariateRing",
    "UnivariateQRecurrenceOperatorOverUnivariateRing",
])
lazy_import("ore_algebra.recurrence_operator_1_1", [
    "UnivariateRecurrenceOperatorOverUnivariateRing",
    "UnivariateDifferenceOperatorOverUnivariateRing",
])

#############################################################################################################

class UnivariateOreOperatorOverUnivariateRing(UnivariateOreOperator):
    r"""
    Element of an Ore algebra with a single generator and a commutative rational function field as base ring.     

    TESTS::

        sage: from ore_algebra import OreAlgebra
        sage: R.<C> = OreAlgebra(GF(2)['x'])
        sage: type(C)
        <class 'ore_algebra.ore_operator_1_1.UnivariateOreOperatorOverUnivariateRing'>
        sage: C.list()
        [0, 1]
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
        super(self.__class__, self).__init__(parent, *data, **kwargs)

    def _normalize_base_ring(self):
        r"""
        Rewrites ``self`` into an operator from an algebra whose base ring is a univariate
        polynomial ring over a field.

        Returns a tuple ``(A, R, K, L)`` where

         * ``L`` is the new operator
 
         * ``A`` is the parent of ``L``

         * ``R`` is the base ring of ``A``

         * ``K`` is the base ring of ``R``

        """
        L = self
        A = L.parent()
        R = A.base_ring()
        K = R.base_ring()

        if R.is_field():
            L = L.numerator()
            R = R.ring()

        if not K.is_field():
            R = R.change_ring(K.fraction_field())

        L = L.change_ring(R)
        return L.parent(), R, K, L

    def degree(self):
        r"""
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
        r"""
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

          sage: from ore_algebra import *
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
        A = self.parent()
        R = A.base_ring()
        R_field = R.fraction_field()
        R_ring = R_field.ring()
        K = R_ring.base_ring()
        if not K.is_field():
            R_ring = K[R.gens()]
            K = R_ring.base_ring()

        [L, rhs], _ = clear_denominators([self.change_ring(R_field), [R_field(a) for a in rhs]])

        if degree is None:
            degree = L._degree_bound()

        if rhs:
            degree = max(degree, max(list(map(lambda p: L.order() + p.degree(), rhs))))

        if degree < 0 and not rhs:
            return []

        x = L.base_ring().gen()
        sys = [-L(x**i) for i in range(degree + 1)] + list(rhs)
        neqs = max(1, max(list(map(lambda p: p.degree() + 1, sys))))
        sys = list(map(lambda p: p.padded_list(neqs), sys))
        
        if solver is None:
            solver = A._solver(K)

        sol = solver(matrix(K, zip(*sys)))

        for i in range(len(sol)):
            s = list(sol[i])
            sol[i] = tuple([R_ring(s[:degree+1])] + s[degree+1:])

        return sol

    def rational_solutions(self, rhs=(), denominator=None, degree=None, solver=None):
        r"""
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
            and not just a list of plain rational functions.
          - If no ``denominator`` is given, a basis of all the rational solutions is returned.
            This feature may not be implemented for all algebras. 
          - If no ``degree`` is given, a basis of all the polynomial solutions is returned.
            This feature may not be implemented for all algebras. 

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']; A.<Dx> = OreAlgebra(R, 'Dx')
          sage: L = ((x+3)*Dx + 2).lclm(x*Dx + 3).symmetric_product((x+4)*Dx-2)
          sage: L.rational_solutions()
          [((-x^2 - 8*x - 16)/x^3,),
           ((-x^5 + 96*x^3 + 584*x^2 + 1344*x + 1152)/(x^5 + 6*x^4 + 9*x^3),)]
          sage: L.rational_solutions((1, x))
          [((x^2 + 8*x + 16)/(x^2 + 6*x + 9), 0, 0),
           ((x^5 + 7*x^4 + 2*x^3 - 73*x^2 - 168*x - 144)/(x^5 + 6*x^4 + 9*x^3), 0, 0),
           ((-2*x - 7)/(x^2 + 6*x + 9), 288, 42)]
          sage: L(_[0][0]) == _[0][1] + _[0][2]*x
          True

          sage: (x*(x*Dx-5)).rational_solutions([1])
          [(-x^5, 0), (1/x, -6)]

          sage: R.<n> = ZZ['n']; A.<Sn> = OreAlgebra(R, 'Sn');
          sage: L = ((n+3)*Sn - n).lclm((2*n+5)*Sn - (2*n+1))
          sage: L.rational_solutions()
          [(-1/(n^3 + 3*n^2 + 2*n),),
           ((-n^3 + n^2 + 6*n + 3)/(4*n^5 + 20*n^4 + 35*n^3 + 25*n^2 + 6*n),)]

          sage: L = (2*n^2 - n - 2)*Sn^2 + (-n^2 - n - 1)*Sn + n^2 - 14
          sage: y = (-n + 1)/(n^2 + 2*n - 2)
          sage: L.rational_solutions((L(y),))
          [((n - 1)/(n^2 + 2*n - 2), -1)]
        
        """
        A = self.parent()
        R = A.base_ring()
        R_field = R.fraction_field()

        [L, rhs], _ = clear_denominators([self.change_ring(R_field), [R_field(a) for a in rhs]])

        if denominator is None:
            denominator = L._denominator_bound()
        elif not isinstance(denominator, Factorization):
            denominator = Factorization([(denominator, 1)])

        L1, opden = L._apply_denominator_bound(denominator)
        sol = L1.polynomial_solutions([opden*c for c in rhs], degree=degree, solver=solver)

        denominator = denominator.expand()
        for i in range(len(sol)):
            sol[i] = tuple([sol[i][0]/denominator] + list(sol[i][1:]))

        return sol

    def _degree_bound(self):
        r"""
        Computes a degree bound for the polynomial solutions of this operator.

        This is an integer `d` such that every polynomial solution of this operator
        has degree `d` or less. 
        """

        if self.is_zero():
            raise ZeroDivisionError("unbounded degree")
        
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
        r"""
        Computes a denominator bound for the rational solutions of this operator.

        This is a polynomial `q` such that every rational solution of this operator
        can be written in the form `p/q` for some other polynomial `p` (not necessarily
        coprime with `q`)

        The default implementation is Abramov's algorithm, which depends on the existence
        of an implementation of ``dispersion``. Subclasses for algebras where this is not
        appropriate must override this method. 
        """

        if self.is_zero():
            raise ZeroDivisionError("unbounded denominator")

        A, R, k, L = self._normalize_base_ring()
        sigma = A.sigma()
        r = L.order()

        n = L.dispersion()
        A = sigma(L[r], -r)
        B = L[0]
        u = Factorization([])

        for i in range(n, -1, -1):
            d = A.gcd(sigma(B, i))
            if d.degree() > 0:
                A //= d
                for j in range(i):
                    u *= d.numerator()
                    d = sigma(d, -1)
                u *= d.numerator()
                B //= d

        return u.base_change(self.base_ring())

    def _apply_denominator_bound(self, den):
        sigma = self.parent().sigma()
        delta = self.parent().delta()
        if sigma.is_identity():
            r = self.order()
            opnum = list(self)
            opden = self.base_ring().one()
            for fac, m in den:
                fac, _ = clear_denominators(fac)
                delta_fac = delta(fac)
                dnum = [fac.parent().one()] # δ^k(1/fac^m) = dnum[k]/fac^(m+k)
                facpow = [fac.parent().one()]
                for k in range(1, r + 1):
                    dnum.append(delta(dnum[-1])*fac - (m+k-1)*dnum[-1]*delta_fac)
                    facpow.append(facpow[-1]*fac)
                opnum = [sum(k.binomial(i)*opnum[k]*dnum[k-i]*facpow[r-k+i]
                               for k in srange(i, r + 1))
                           for i in range(r+1)]
                opden *= fac**(m+r)
                g = gcd([opden] + opnum)
                opnum = [c//g for c in opnum]
                opden //= g
        else:
            op = self*~self.base_ring().fraction_field()(den.expand())
            opnum, opden = clear_denominators(op)
        return self.parent()(opnum), opden

    def dispersion(self, p=0):
        r"""
        Returns the dispersion of this operator.

        This is the maximum nonnegative integer `i` such that ``sigma(self[0], i)`` and ``sigma(self[r], -r)``
        have a nontrivial common factor, where ``sigma`` is the shift of the parent's algebra and `r` is
        the order of ``self``.

        An output `-1` indicates that there are no such integers `i` at all.

        If the optional argument `p` is given, the method is applied to ``gcd(self[0], p)`` instead of ``self[0]``.

        The output is `\infty` if the constant coefficient of ``self`` is zero.

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']; A.<Sx> = OreAlgebra(R, 'Sx');
          sage: ((x+5)*Sx - x).dispersion()
          4
        
        """
        s = self.spread(p)
        return max(max(s), -1) if len(s) > 0 else -1

    def spread(self, p=0):
        r"""
        Returns the spread of this operator.

        This is the set of integers `i` such that ``sigma(self[0], i)`` and ``sigma(self[r], -r)``
        have a nontrivial common factor, where ``sigma`` is the shift of the parent's algebra and `r` is
        the order of ``self``.

        If the optional argument `p` is given, the method is applied to ``gcd(self[0], p)`` instead of ``self[0]``.

        The output set contains `\infty` if the constant coefficient of ``self`` is zero.

        This method is a stub and may not be implemented for every algebra. 

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<x> = ZZ['x']; A.<Sx> = OreAlgebra(R, 'Sx');
          sage: ((x+5)*Sx - x).spread()
          [4]
          sage: ((x+5)*Sx - x).lclm((x+19)*Sx - x).spread()
          [3, 4, 17, 18]
        
        """
        raise NotImplementedError # abstract

    def newton_polygon(self, p):
        r"""
        Computes the Newton polygon of ``self`` at (a root of) ``p``.

        INPUT:

          - ``p`` -- polynomial at whose root the Newton polygon is to be determined. 
            ``p`` must be an element of the parent's base ring (or its fraction field).
            The value `p=1/x` represents the point at infinity.
        
        OUTPUT:

           A list of pairs ``(gamma, q)`` such that ``gamma`` is a slope in the Newton
           polygon and ``q`` is the associated polynomial, as elements of the base ring.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ[]; A.<Dx> = OreAlgebra(R); 
           sage: L = (x^3*Dx - 1+x).lclm(x*Dx^2-1)
           sage: L.newton_polygon(x)
           [(1/2, x^2 - 1), (3, -x + 1)]
           sage: L.newton_polygon(~x)
           [(-2, -x - 1), (-1/2, x^2 - 1)]
           sage: A.<Sx> = OreAlgebra(R); L = (x*Sx - 5).lclm(Sx-x^3); L.newton_polygon(~x)
           [(-1, -x + 5), (3, x - 1)]

        Depending on the algebra in which this operator lives, restrictions on ``p`` may apply.
        """
        assert(not self.is_zero())

        coeffs = self.change_ring(self.parent().base_ring().fraction_field().ring()).normalize().coefficients(sparse=False)
        x = coeffs[0].parent().gen()

        if (x*p).is_one():
            points = [ (QQ(i), QQ(coeffs[i].degree())) for i in range(len(coeffs)) if coeffs[i]!=0 ]
            coeffs = dict( (i, coeffs[i].leading_coefficient()) \
                           for i in range(len(coeffs)) if coeffs[i]!=0 )
            flip = -1
        else:
            points = [ (QQ(i), QQ(coeffs[i].valuation(p))) for i in range(len(coeffs)) if coeffs[i]!=0 ]
            coeffs = dict( (i, (coeffs[i]//p**coeffs[i].valuation(p))(0)) \
                           for i in range(len(coeffs)) if coeffs[i]!=0 )
            flip = 1 

        output = []
        k = 0
        infty = max([j for _, j in points]) + 2
        while k < len(points) - 1:
            (i1, j1) = points[k]
            m = infty
            poly = coeffs[i1]
            for l in range(k + 1, len(points)):
                (i2, j2) = points[l]
                m2 = flip*(j2 - j1)/(i2 - i1)
                if m2 == m:
                    k = l
                    poly += coeffs[i2]*x**(i2 - i1)
                elif m2 < m:
                    m = m2
                    k = l
                    poly = coeffs[i1] + coeffs[i2]*x**(i2 - i1)
            output.append((m, poly))
        
        return output

    def indicial_polynomial(self, p, var='alpha'):
        r"""
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
        R = PolynomialRing(op[0].parent().base_ring(), var)
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
        for j in range(1, len(op)):
            b = m(b, deg(op[j]) - j)

        s = R.zero()
        y_ff_i = R.one()
        for i in range(len(op)):
            s = s + op[i][b + i]*y_ff_i
            y_ff_i *= y - my_int(i)

        try: ## facilitate factorization
            den = lcm( [ p.denominator() for p in s ] )
            s = s.map_coefficients(lambda p: den*p)
        except:
            pass                
            
        return s

    def _coeff_list_for_indicial_polynomial(self):
        r"""
        Computes a list of polynomials such that the usual algorithm for computing indicial
        polynomials applied to this list gives the desired result.

        For example, for differential operators, this is simply the coefficient list of ``self``,
        but for recurrence operators, it is the coefficient list of ``self.to_F()``.

        This is an abstract method.         
        """
        raise NotImplementedError # abstract

    def _desingularization_order_bound(self):
        r"""
        Computes a number `m` such that there exists an operator ``Q`` of order `m` such that ``Q*self``
        is completely desingularized. 

        This method returns per default the maximum element of the elements of spread times `-1`.
        This is the right choice for many algebras. Other algebras have to override this method appropriately. 
        """
        s = self.spread()
        return 0 if len(s) == 0 else max(0, max([-k for k in s]))

    def desingularize(self, m=-1):
        r"""
        Returns a left multiple of ``self`` whose coefficients are polynomials and whose leading
        coefficient does not contain unnecessary factors.

        INPUT:

        - `m` (optional) -- If the order of ``self`` is `r`, the output operator will have order `r+m`.
          In order to ensure that all removable factors of the leading coefficient are removed in the 
          output, `m` has to be chosen sufficiently large. If no `m` is given, a generic upper bound
          is determined. This feature may not be available for every class.

        OUTPUT:
        
          A left multiple of ``self`` whose coefficients are polynomials, whose order is `m` more than
          ``self``, and whose leading coefficient has as low a degree as possible under these conditions.

          The output is not unique. With low probability, the leading coefficient degree in the output
          may not be minimal. 

        EXAMPLES::

          sage: from ore_algebra import *
          sage: R.<n> = ZZ['n']
          sage: A.<Sn> = OreAlgebra(R, 'Sn')
          sage: P = (-n^3 - 2*n^2 + 6*n + 9)*Sn^2 + (6*n^3 + 8*n^2 - 20*n - 30)*Sn - 8*n^3 - 12*n^2 + 20*n + 12
          sage: Q = P.desingularize()
          sage: Q.order() # random
          3
          sage: Q.leading_coefficient().degree()
          1

        """

        L = self.numerator()
        A = L.parent()
        if A.base_ring().is_field():
            A = A.change_base(A.base_ring().base())
            L = A(L)
        R = A.base_ring()
        C = R.base_ring()
        sub = m - 1

        if m < 0:
            m = L._desingularization_order_bound()
            sub = 0
        
        if m <= 0:
            return L

        deg = None
        Dold = A.zero()

        for k in range(m, sub, -1):
            D = A.zero()
            while D.order() != L.order() + k:
                # this is only probabilistic, it may fail to remove some removable factors with low probability.
                T = A([R.random_element() for i in range(k)] + [R.one()])
                if not T[0].is_zero():
                    D = L.lclm(T)
            L0 = (A.gen()**k)*L
            _, u, v = L0.leading_coefficient().xgcd(D.leading_coefficient())
            D = (u*L0 + v*D).normalize()
            if k == m:
                deg = D.leading_coefficient().degree() 
            elif deg < D.leading_coefficient().degree():
                return Dold
            Dold = D
        
        return D                

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

        The method returns the empty list if and only if no nontrivial solutions exist. 

        This function may not be implemented for every algebra.

        EXAMPLES::

          sage: from ore_algebra import *
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
          [(-Dx - 1/x, -1/x^2)]
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
          sage: (M, m) = _[0]
          sage: (Sn-1)*M == p + m*L  ## this implies sum(fib(n)^2) == 1/2*fib(n+2)^2 - 1/2*fib(n+1)^2 - 3/2*fib(n)^2
          True
          
        """
        P = self
        A = P.parent()
        R = A.base_ring()

        if not isinstance(D, OreOperator) or D.parent() is not A:
            raise TypeError("operators must live in the same algebra")
        elif p not in R.fraction_field():
            raise TypeError("p must belong to the base ring")
        elif D.order() != 1:
            raise TypeError("D must be a first order operator")
        elif self.order() <= 0:
            raise ValueError("P must have at least order 1")
        elif A.is_F():
            sols = P.to_S('S').associate_solutions(D.to_S('S'), p)
            return [ (M.to_F(str(A.gen())), m) for (M, m) in sols]
        elif A.is_S() is not False or A.is_Q() is not False:
            S = A.gen()
            if not D == S - A.one():
                raise NotImplementedError("unsupported choice of D: " + str(D))
            # adjoint = sum( (sigma^(-1) - 1)^i * a[i] ), where a[i] is the coeff of D^i in P
            adjoint = A.zero()
            coeffs = P.to_F('F').coefficients(sparse=False)
            r = P.order()
            for i in range(len(coeffs)):
                adjoint += S**(r-i)*(A.one() - S)**i * coeffs[i]
        elif A.is_D() is not False or A.is_T() is not False:
            if D != A.gen():
                raise NotImplementedError("unsupported choice of D: " + str(D))
            # adjoint = sum( (-D)^i * a[i] ), where a[i] is the coeff of D in P
            adjoint = A.zero()
            coeffs = P.coefficients(sparse=False)
            for i in range(len(coeffs)):
                adjoint += (-D)**i * coeffs[i]
        else:
            raise NotImplementedError

        sol = adjoint.rational_solutions((-p,))
        A = A.change_ring(A.base_ring().fraction_field())
        sigma = A.sigma()
        delta = A.delta()

        for i in range(len(sol)):
            if sol[i][1].is_zero():
                sol[i] = None
                continue
            rat = sol[i][0]/sol[i][1]
            DM = p + rat*P
            M = A.zero()
            while DM.order() > 0:
                r = DM.order()
                a = DM.leading_coefficient()
                # DM = a*D^r + ...
                #    = (a*D)*D^(r-1) + ...
                #    = (D*s(a, -1) - d(s(a, -1)))*D^(r-1) + ...
                M += sigma(a, -1)*(D**(r-1))
                DM -= D*sigma(a, -1)*(D**(r-1))
            sol[i] = (M, rat)

        return [p for p in sol if p is not None]

    def center(self,oBound,dBound):
        r"""
        Returns a Q-vector space of Ore polynomials that commute with this operator.

        INPUT:

        - ``oBound`` -- The maximal order of the operators in the center.
        - ``dBound`` -- The maximal coefficient degree of the operators in the center.

        OUTPUT:

        A subspace of Q^((oBound+1)*(dBound+1)). Each entry of a vector
        corresponds to a coefficient of an Ore polynomial that commutes with
        ``self``. To translate a vector to its corresponding Ore polynomial,
        call _listToOre

        Note: This method only works for operators over Q[n].
        """

        R = self.parent()
        K = R.base_ring()
        Q = K.base_ring()
        R2 = R.change_ring(PolynomialRing(PolynomialRing(Q,[('c'+str(i)+str(j)) for i in range(oBound) for j in range(dBound)]),K.gen()))
        L = reduce(lambda x,y: x+y,[reduce(lambda x,y: x+y,[R2.base_ring().base_ring().gens()[i+j*dBound]*R2.base_ring().gen()**i for i in range(dBound)])*R2.gen()**j for j in range(oBound)])
        C=L*self-self*L
        SYS=[]
        for sC in C.coefficients(sparse=False):
            for nC in sC.coefficients(sparse=False):
                l=[]
                for cC in R2.base_ring().base_ring().gens():
                    l.append(Q(nC.coefficient(cC)))
                SYS.append(l)
        return Matrix(SYS).right_kernel()

    def radical(self):
        r"""
        Computes the radical of an Ore polynomial P, i.e. an operator L and an integer k such that P=L^k and k is maximal among all the integers for which such an L exists.

        OUTPUT:

        A tuple (L,k) such that self is equal to L^k and there is no larger integer k' for which such an L exists.

        Note: This method only works for operators over Q[x].

        """
        if self.order()==0:
            return _commutativeRadical(self.leading_coefficient())
        if self.degree()==0:
            return _commutativeRadical(PolynomialRing(self.parent().base_ring().base_ring(),self.parent().gen())(self.polynomial()))
        M = [a for a in self._radicalExp() if self.order()%a==0 and self.degree()%a==0]
        R = self.parent()
        K = R.base_ring()
        Q = K.base_ring()
        for i in range(len(M)-1,-1,-1):
            a = M[i]
            oBound = self.order()//a+1
            dBound = self.degree()//a+1
            cen = self.center(oBound,dBound).basis()
            if len(cen)>1:
                R2 = R.change_ring(PolynomialRing(PolynomialRing(Q,[('c'+str(i)) for i in range(len(cen))]),K.gen()))
                L = reduce(lambda x,y: x+y,[R2.base_ring().base_ring().gens()[i]*_listToOre(cen[i],oBound,R2) for i in range(len(cen))])
                L2 = L**(self.order()//(oBound-1))-self
                dictionary = dict(zip(R2.base_ring().base_ring().gens(), _orePowerSolver(L2)))
                sol = L.map_coefficients(lambda x: x.map_coefficients(lambda y: y.subs(dictionary)))
                if sol!=L:
                    return (sol,self.order()/sol.order())

    def _radicalExp(self):
        r"""
        For an Ore polynomial `P`, this method computes candidates for possible
        powers `k` such that there exists an operator `L` with `P=L^k`.

        OUTPUT:

        A list of integers `k` such that there possibly exists an
        operator `L` such that ``self`` equals `L^k`.

        Note: This method only works for operators over Q[n].

        """
        p = self._powerIndicator()
        exponents=[divisors(d) for (c,d) in p.squarefree_decomposition()]
        M=[]
        for a in exponents[0]:
            contained = True
            for i in range(1,len(exponents)):
                contained = contained and a in exponents[i]
            if contained:
                M.append(a)
        return M

    def _powerIndicator(self):
        r"""
        Returns the coefficient of an Ore polynomial P that is of the form p^k,
        where p is an element from the base ring and k is such that P=L^k where
        L is the radical of P.
        """
        raise NotImplementedError
    
    def singularities(self, backwards = False):
        r"""
        Returns the integer singularities of the operator ``self``.
        
        INPUT:
        
        - ``backwards`` (default ``False``) -- boolean value that decides whether the singularities of the leading coefficient are returned
          (when ``backwards`` is ``False``) or those of the coefficient with minimal degree (regarding ``Sn`` or ``Dx``)
          
        OUTPUT:
        
        - If ``backwards`` is ``False``, a set containing the roots of the leading coefficient of the annihilator of ``self`` shifted by 
          its order are returned
        - If ``backwards`` is ``True``, a set containing the roots of the coefficient with minimal degree (regarding `Sn` or `Dx` respectively) 
          are returned; shifted by the degree of this coefficient
          
        EXAMPLES::

            sage: from ore_algebra import OreAlgebra
            sage: A = OreAlgebra(QQ['n'],'Sn')
            sage: a = A("(n-3)*(n+2)*Sn^3 + n^2*Sn^2 - (n-1)*(n+5)*Sn")
            sage: a.singularities()
            {1, 6}
            sage: a.singularities(True)
            {-4, 2}

        return the integer singularities of the Ore Operator ``self``, i.e. the roots of the
        leading coefficient shifted by the order of the operator if ``backwards``is false; 
        when``backwards`` is true then the roots of the smallest non-zero term (concerning the degree)
        are returned (shifted by the degree of this term)
        """
        if self == 0:
            return {ZZ}
        
        S = self.parent().is_S()
        result = set()
        ord = self.order()
        min_degree = 0
        
        #no backward singularities needed
        if not backwards:
            lc = self.leading_coefficient()
            roots = lc.numerator().roots()
            for i in range(len(roots)):
                r = roots[i][0]
                if (r in ZZ) and (r >= -ord):
                    if S:
                        result.add(ZZ(r + ord))
                    else:
                        result.add(ZZ(r))
        
        #backward singularities are also needed
        else:
            min_degree = next((index for index, coeff in enumerate(self.list()) if coeff.numerator() != 0), 0)
            coeff = self.list()[min_degree]
            roots = coeff.numerator().roots()
            for i in range(len(roots)):
                r = roots[i][0]
                if (r in ZZ) and (r <= ord - min_degree):
                    result.add(ZZ(r + min_degree))

        return result

    def finite_singularities(self):
        r"""
        Returns a list of all the finite singularities of this operator. 

        OUTPUT:

           For each finite singularity of the operator, the output list contains a pair (p, u) where

           * p is an irreducible polynomial, representing the finite singularity rootof(p)+ZZ

           * u is a list of pairs (v, dim, bound), where v is an integer that appears as valuation growth
             among the solutions of the operator, and bound is a polynomial (or rational function) such 
             that all the solutions of valuation growth v can be written f/bound*Gamma(x-rootof(p))^v 
             where f has minimal valuation almost everywhere. dim is a bound for the number of distinct
             hypergeometric solutions that may have this local behaviour at rootof(p)+ZZ.

        This is a generic implementation for the case of shift and q-shift
        recurrences. Subclasses for other kinds of operators may need to
        override this method.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ['x']; A.<Sx> = OreAlgebra(R)
           sage: (x^2*(x+1)*Sx + 3*(x+1/2)).finite_singularities()
           [(x + 1/2, [[1, 1, 1]]), (x, [[-3, 1, x^4 - 3*x^3 + 3*x^2 - x]])]

           sage: C.<q> = ZZ[]; R.<x> = C['x']; A.<Qx> = OreAlgebra(R)
           sage: ((q^2*x-1)*Qx-(x-1)).finite_singularities()
           [(-x + 1, [[0, 1, q*x^2 + (-q - 1)*x + 1]])]
        
        """
        from sage.matrix.constructor import matrix
        from sage.rings.finite_rings.all import GF

        R = self.parent().base_ring().fraction_field().base()
        R = R.change_ring(R.base_ring().fraction_field())
        A = self.parent().change_ring(R)
        L = A(self.normalize())
        assert(not L.is_zero())

        sigma = L.parent().sigma()
        coeffs = L.coefficients(sparse=False)
        while coeffs[0].is_zero(): # make trailing coefficient nonzero
            coeffs = [sigma(coeffs[i], -1) for i in range(1, len(coeffs))]
        L = L.parent()(coeffs)
        r = L.order()

        imgs = dict( (y, hash(y)) for y in R.base_ring().gens_dict_recursive() )
        R_img = QQ # coefficient ring after evaluation of parameters
        ev = (lambda p: p) if len(imgs) == 0 else (lambda p: p(**imgs))
        x = R.gen()

        if A.is_Q():
            _, q = A.is_Q()
            sf = lambda p: shift_factor(p, q=q)
            def make_sigma_mod(C):
                R_mod = R.change_ring(C)
                q_mod = C(ev(q))
                x_mod = R_mod.gen()
                return lambda p, n=1: R_mod(p)((1 + x_mod)*q_mod**n)
            def change_of_variables(C, xi, r):
                R_mod = R.change_ring(C)
                q_inv_mod = C(ev(q))**(-r)
                x_mod = R_mod.gen()
                return lambda p: R_mod(p.map_coefficients(ev, R_img))(xi*x_mod*q_inv_mod)

        elif A.is_S():
            sf = shift_factor
            def make_sigma_mod(C):
                R_mod = R.change_ring(C)
                x_mod = R_mod.gen()
                return lambda p, n=1: R_mod(p)(x_mod + n)
            def change_of_variables(C, xi, r):
                R_mod = R.change_ring(C)
                x_mod = R_mod.gen()
                return lambda p: R_mod(p.map_coefficients(ev, R_img))(x_mod + xi - r)
        else:
            raise NotImplementedError

        output = []
        lctc_factors = sf(L[0]*L[r])
        tc_factor_dict = dict( (u, sum(w for _, w in v) - 1) for u, v in 
                               sf(prod(u for u, _ in lctc_factors)*L[0]) )
        lc_factor_dict = dict( (u, sum(w for _, w in v) - 1) for u, v in 
                               sf(prod(u for u, _ in lctc_factors)*L[r]) )

        for pol, e in lctc_factors:

            # left-most critical point is rootof(pol) - e[-1][0], right-most critical point is rootof(pol)

            # search for a prime such that pol has a root xi in C:=GF(prime). 
            if pol.degree() == 0:
                continue
            elif pol.degree() == 1:
                C = GF(pp(2**23))
                xi = C(ev(-pol[0]/pol[1]))
            else:
                modulus = 2**23
                done = False
                while not done:
                    modulus = pp(modulus)
                    C = GF(modulus)
                    for u, _ in ev(pol).change_ring(C).factor():
                        if u.degree() == 1:
                            xi = -u[0]/u[1]
                            done = True
                            break

            # valuation growth can get at most val_range_bound much more than min 
            val_range_bound = lc_factor_dict[pol] + tc_factor_dict[pol] + 1
            R = LaurentSeriesRing(C, str(A.base_ring().gen()), default_prec=val_range_bound)

            # A. GOING FROM LEFT TO RIGHT
            coeffs = list(map(change_of_variables(C, xi, r), L.coefficients(sparse=False)))
            coeffs.reverse()
            coeffs[0] = -coeffs[0]

            # compute a C-basis of the left-to-right solutions in C((eps))^ZZ 
            sigma_mod = make_sigma_mod(C)
            def prolong(l, n):
                ## given a list of values representing the values of a laurent series sequence solution
                ## at ..., xi+n-2, xi+n-1, this appends the value at xi+n to the list l.
                ## the list l has to have at least r elements. 
                ## --- recycling the symbol x as epsilon here. 
                l.append(sum(l[-i]*sigma_mod(coeffs[i], n) for i in range(1, r + 1))/sigma_mod(coeffs[0], n))

            sols = []
            for i in range(r):
                sol = [ R.zero() for j in range(r) ]
                sol[i] = R.one()
                sols.append(sol)
                for n in range(-e[-1][0], r + 1):
                    prolong(sol, n)

            vg_min = min( s[-i].valuation() for s in sols for i in range(1, r + 1) if not s[-i].is_zero() )

            den = 1
            for n in range(r, len(sols[0])):
                k = min( [s[n].valuation() for s in sols if not s[n].is_zero()] + [val_range_bound] )
                den *= sigma(pol, e[-1][0] - n + r)**max(0, (-k + (vg_min if n > len(sols[0]) - r else 0)))

            # B. GOING FROM RIGHT TO LEFT
            coeffs = list(map(change_of_variables(C, xi, 0), L.coefficients(sparse=False)))
            coeffs[0] = -coeffs[0]

            sols = []
            for i in range(r):
                sol = [ R.zero() for j in range(r) ]
                sol[i] = R.one()
                sols.append(sol)
                for n in range(e[-1][0] + r + 1):
                    prolong(sol, -n)

            vg_max = min( s[-i].valuation() for s in sols for i in range(1, r + 1) if not s[-i].is_zero() )

            # record information for this singularity
            valuation_growths = [[i, r, den] for i in range(vg_min, -vg_max + 1)]
            output.append( (pol, valuation_growths) )

        return output

    def left_factors(self, order=1, early_termination=False, infolevel=0):
        r"""
        Returns a list of left-hand factors of this operator.

        This is a convenience method which simply returns the adjoints of the right factors
        of the adjoint of self. See docstring of adjoint and right_factors for further information.
        The method works only in algebras for which adjoint and right_factors are implemented.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<x> = ZZ[]; A.<Sx> = OreAlgebra(R)
           sage: (((x+1)*Sx + (x+5))*(2*x*Sx + 3)).left_factors()
           [[(-x - 1)*Sx - x - 5]]
           sage: (((x+1)*Sx + (x+5))*(2*x*Sx + 3)).right_factors()
           [[x*Sx + 3/2]]

        """
        return [[f.adjoint() for f in F] for F in 
                self.adjoint().right_factors(order, early_termination, infolevel)]

    def right_factors(self, order=1, early_termination=False, infolevel=0):
        r"""
        Returns a list of right hand factors of this operator. 

        INPUT:

        - ``order`` (default=1) -- only determine right factors of at most this
          order

        - ``early_termination`` (optional) -- if set to ``True``, the search for
          factors will be aborted as soon as one factor has been found. A list
          containing this single factor will be returned (or the empty list if
          there are no first order factors). If set to ``False`` (default), a
          complete list will be computed.  

        - ``infolevel`` (optional) -- nonnegative integer specifying the amount
          of progress reports that should be printed during the
          calculation. Defaults to 0 for no output.

        OUTPUT:

        A list of bases for all vector spaces of first-order operators living in the parent 
        of ``self`` of which ``self`` is a left multiple. 

        Note that this implementation does not construct factors that involve
        algebraic extensions of the constant field.

        This is a generic implementation for the case of shift and q-shift
        recurrences. Subclasses for other kinds of operators may need to
        override this method.

        EXAMPLES::

           sage: from ore_algebra import *
           sage: R.<n> = ZZ['n']; A.<Sn> = OreAlgebra(R, 'Sn');
           sage: L = (-25*n^6 - 180*n^5 - 584*n^4 - 1136*n^3 - 1351*n^2 - 860*n - 220)*Sn^2 + (50*n^6 + 560*n^5 + 2348*n^4 + 5368*n^3 + 7012*n^2 + 4772*n + 1298)*Sn - 200*n^5 - 1540*n^4 - 5152*n^3 - 8840*n^2 - 7184*n - 1936
           sage: L.right_factors()
           [[(n^2 + 6/5*n + 9/25)*Sn - 2*n^2 - 32/5*n - 128/25], [(n^2 + 2*n + 1)*Sn - 4*n - 2]]
           sage: ((Sn - n)*(n*Sn - 1)).right_factors()
           [[n*Sn - 1]]
           sage: ((Sn - n).lclm(n*Sn - 1)).right_factors()
           [[n*Sn - 1], [Sn - n]]
           sage: (Sn^2 - 2*Sn + 1).right_factors()
           [[Sn - 1, n*Sn - n - 1]]

           sage: R.<x> = QQ['x']; A.<Qx> = OreAlgebra(R, q=2) 
           sage: ((2*x+3)*Qx - (8*x+3)).lclm(x*Qx-2*(x+5)).right_factors()
           [[(x + 3/2)*Qx - 4*x - 3/2], [x*Qx - 2*x - 10]]
           sage: (((2*x-1)*Qx-(x-1)).lclm(Qx-(x-3))).right_factors()
           [[(x - 1/2)*Qx - 1/2*x + 1/2], [Qx - x + 3]]
           sage: (((2*x-1)*Qx-(x-1))*(Qx-(x-3))).right_factors()
           [[Qx - x + 3]]
           sage: (((2*x-1)*Qx-(x-1))*(x^2*Qx-(x-3))).right_factors()
           [[x^2*Qx - x + 3]]

        """

        if self.is_zero():
            raise ZeroDivisionError

        if order > 1:
            raise NotImplementedError

        coeffs = self.normalize().coefficients(sparse=False)
        R = self.base_ring().fraction_field().base()
        R = R.change_ring(R.base_ring().fraction_field()).fraction_field()
        A = self.parent().change_ring(R)
        S = A.gen()
        sigma = A.sigma()
        assert(R.characteristic() == 0)

        if A.is_Q():
            q_case = True
            x, q = A.is_Q()
        elif A.is_S():
            q_case = False
            x = R.gen()
        else:
            raise NotImplementedError

        # shift back such as to make the trailing coefficient nonzero
        min_r = 0
        while coeffs[min_r].is_zero():
            min_r += 1
        if min_r > 0:
            coeffs = [sigma(coeffs[i], -min_r) for i in range(min_r, len(coeffs))]

        # handle trivial cases
        factors = [] if min_r == 0 else [[A.gen()]]
        if len(coeffs) == 1:
            return factors
        elif len(coeffs) == 2: 
            return factors + [[A(coeffs)]]

        SELF = A([R(c) for c in coeffs])
        r = SELF.order()

        def info(i, msg):
            if i <= infolevel:
                print((i - 1)*"  " + msg)

        # 1. determine the local behaviour at the finite singularities (discard apparent ones)
        finite_local_data = [u for u in SELF.finite_singularities()
                            if len(u[1])>1 or u[1][0][0]!=0 or u[1][0][2]!=1]
        info(1, "Analysis of finite singularities completed. There are " + 
             str(len(list(finite_local_data))) + " of them.")
        # precompute some data that is handy to have available later during the big loop
        for p, u in finite_local_data:
            d = p.degree()
            for v in u:
                v.append(sigma(v[2])/v[2] * p**(-v[0])) # idx 3 : reciprocal shift quotient
                v.append(v[0]*d) # idx 4 : exponent
                # idx 5 : alpha, taking into account contribution from the denominator bound
                delta = R(v[2]).numerator().degree() - R(v[2]).denominator().degree()
                if q_case:
                    v.append((p[d]/p[0])**v[0] * q**(-delta))
                else:
                    v.append(v[0]*p[d - 1]/p[d] - delta)

        # 2. determine the local behaviour at infinity.

        # reorganize data in the form {valg:[[gamma,phi,max_alpha,dim],[gamma,phi,max_alpha,dim],...], ...}
        if q_case:
            special_local_data = SELF._local_data_at_special_points()
            def equiv(a, b):
                try:
                    return q_log(q, a/b) in ZZ
                except:
                    return False
            def merge(a, b):
                e = q_log(q, a/b)
                return a if e > 0 else b
        else:
            special_local_data = SELF._infinite_singularity()
            special_local_data = [ (0, phi, gamma, alpha) for gamma, phi, alpha in special_local_data]
            equiv = lambda a, b: a - b in ZZ
            merge = lambda a, b: max(a, b)

        spec = {}
        for gamma, phi, beta, alpha in special_local_data:
            try:
                u = [u for u in spec[beta] if u[1]==phi and equiv(u[2], alpha)]
            except:
                u = spec[beta] = []
            if len(u) == 0:
                spec[beta].append([gamma, phi, alpha, 1])
            else:
                u[0][2] = merge(u[0][2], alpha)
                u[0][3] += 1

        special_local_data = spec
        info(1, "Local data at infinity (for each val-growth, list of triples [gamma,phi,max_alpha,dim]): " 
             + str(special_local_data))

        # 3. search for factors without valuation growth if there is no combination corresponding to this case.
        if len(finite_local_data) == 0 or not all(any(v[0]==0 for v in u) for _, u in finite_local_data):
            info(1, "Searching for factors with singularities only at special points.")
            for gamma, phi, d, _ in special_local_data.setdefault(0, []):
                if d in ZZ and d >= 0:
                    f = []
                    for p in SELF.symmetric_product(phi*x**gamma*S - 1).polynomial_solutions(degree=d):
                        f.append( (p[0]*S - phi*x**gamma*sigma(p[0])).normalize() )
                    if len(f) > 0:
                        factors.append(f)
                        if early_termination:
                            return factors
            if len(finite_local_data) == 0:
                return factors

        # 4. construct iterator that enumerates all combinations of all local singular solutions
        def combs(local_data):
            idx = [0 for i in range(len(local_data))]
            while idx[0] < len(local_data[0][1]):
                yield [(local_data[j][0], local_data[j][1][idx[j]]) for j in range(len(idx))]
                idx[-1] += 1
                for j in range(len(local_data) - 1, 0, -1):
                    if idx[j] == len(local_data[j][1]):
                        idx[j] = 0
                        idx[j - 1] += 1
                    else:
                        break

        # 5. for all combinations of local solutions determine the polynomial factors. 
        #    this is the heavy loop.
        stat = [prod(len(u[1]) for u in finite_local_data), 0, 0, 0, 0]
        for c in combs(finite_local_data):

            if stat[1] > 0 and stat[1] % 1000 == 0:
                info(2, "%i/%i combinations completed (%.2f%%)" % (stat[1], stat[0], 100.0*stat[1]/stat[0]))
                info(3, "%.2f%% disc. by dimension, %.2f%% disc. by Fuchs-relation, %.4f%% disc. by degree, %.4f%% actually solved" % tuple(map(lambda u: 100.0*u/stat[1], [stat[2], stat[3], stat[4], stat[1] - (stat[2]+stat[3]+stat[4])])))

            stat[1] += 1

            # determine valg, gamma, alpha, dim for this combination
            valg = 0
            dim = r
            alpha = 1 if q_case else 0
            for _, u in c:
                valg += u[4]
                dim = min(dim, u[1])
                if q_case: 
                    alpha *= u[5]
                else:
                    alpha += u[5]
            if dim == 0: # all solutions with this finite local behaviour have already been identified
                stat[2] += 1
                continue
            
            # possible phi's are those that meet the current gamma and alpha+ZZ
            gamma_phis = [u for u in special_local_data.setdefault(valg, []) if equiv(u[2], alpha)]
            if len(gamma_phis) == 0: # Fuchs filter
                stat[3] += 1
                continue

            # check whether all solutions with this behaviour at infinity have already been found
            gamma_phis = [u for u in gamma_phis if u[3] > 0]
            if len(gamma_phis) == 0:
                stat[2] += 1 
                continue

            rat = prod( u[3] for _, u in c )
            for gamma_phi_d_dim in gamma_phis:

                gamma, phi, d, _ = gamma_phi_d_dim

                # determine degree bound 
                d = q_log(q, d/alpha) if q_case else (d - alpha)
                if d < 0 and not q_case:
                    stat[4] += 1
                    continue 

                # find polynomial solutions 
                sols = SELF.symmetric_product(x**gamma*phi*S - rat ).polynomial_solutions(degree = d)
                if len(sols) == 0:
                    continue

                # register solutions found 
                info(1, "Factor found.")
                for u in c:
                    u[1][1] -= len(sols) 
                gamma_phi_d_dim[3] -= len(sols)
                factors.append( [ (rat*p[0]*S - phi*x**gamma*sigma(p[0])).normalize() for p in sols ] )
                if early_termination:
                    return factors

        info(1, "%i combinations have been investigated in total. Of them:" % stat[0])
        stat[1] -= stat[2] + stat[3] + stat[4]
        info(1, "--  %i were discarded by dimension arguments (%.4f%%)" % (stat[2], 100.0*stat[2]/stat[0] ))
        info(1, "--  %i were discarded by the Fuchs-relation (%.4f%%)" % (stat[3], 100.0*stat[3]/stat[0] ))
        info(1, "--  %i were discarded by negative degree bound (%.4f%%)" % (stat[4], 100.0*stat[4]/stat[0] ))
        info(1, "--  %i the polynomial solver was called on (%.4f%%)" % (stat[1], 100.0*stat[1]/stat[0] ))
        info(1, "We have found %i factors." % sum(len(f) for f in factors))

        return factors

    # FIXME: Find a better name, this one is ambiguous
    def value_function(self, op, place, **kwargs):
        r"""
        Compute the value of the operator ``op`` in the algebra quotient of the ambient Ore algebra by `self`, at the place ``place``.

        INPUT:

        - ``op`` -- an Ore operator

        - ``place`` -- the place at which to compute the value. It should be an irreducible polynomial in the base ring of the Ore algebra

        - Implementations of this method can interpret further named arguments.

        OUTPUT:

        The value of ``op`` at the place ``place``

        EXAMPLES::
        #TODO
        
        """
        raise NotImplementedError # abstract

    def raise_value(self, vectors, place, dim=None, **kwargs):
        r"""
        Given a list of vectors in the quotient of the ambient Ore algebra by this operator, find a linear combination of those vectors which has higher value at ``place`` than the last element of the list.

        It is assumed that all vectors have value 0.

        Given ``[b_1, ..., b_n]``, the function computes ``a_1, ..., a_n`` in the coefficient field such that ``a_n = 1`` and

            val(a_1*b_1 + ... + a_n*b_n) > 0, 

        If no such combination exists, the function returns None.

        INPUT:

        - ``vectors`` -- a list of vectors in the quotient of the ambient Ore algebra by ``self``

        - ``place`` -- the place at which to consider the value function

        - ``dim`` (default: None) -- the dimension of the quotient of the ambient Ore algebra by ``self``. If not provided, it is assumed that ``vectors`` form a basis of that quotient. This may lead to incorrect results.

        - Implementations of this method can interpret further named arguments.

        OUTPUT:

        A linear combination as described if it exists, and None otherwise.

        EXAMPLES::

        #TODO
        """
        raise NotImplementedError # abstract

    def _normalize_local_integral_basis_args(
            self,x,basis=None, val_fct=None, raise_val_fct=None,
            infolevel=0,**args):
        """
        Normalize the arguments in a call to `local_integral_basis`.

        INPUT: same as `local_integral_basis`

        OUTPUT: a hashable object formed with the arguments, ensuring that the
        result of `local_integral_basis` only depends on the value of this
        object, and not on the choice of the specific set of arguments.

        EXAMPLES:
        #TODO
        """
        if basis:
            basis = tuple(basis)
        args = list(args.items())
        args.sort()
        args = tuple(args)
        return (x,basis,args)

    def _initial_integral_basis(self, place=None, **kwargs):
        r"""
        Return the default basis to use for computing an integral basis

        This method is provided so that instances can overload it.
        """
        r = self.order()
        ore = self.parent()
        DD = ore.gen()
        return [DD**i for i in range(r)]

    
    def _normalize_local_integral_basis_args(
            self,a,basis=None, val_fct=None, raise_val_fct=None,
            infolevel=0,**args):
        if basis:
            basis = tuple(basis)
        args = list(args.items())
        args.sort()
        args = tuple(args)
        return (a,basis,args)
    
    @cached_method(key=_normalize_local_integral_basis_args)
    def local_integral_basis(
            self, a, basis=None, val_fct=None, raise_val_fct=None,
            infolevel=0, **val_kwargs):
        r"""
        Return a basis of the quotient algebra by self which is an integral basis at ``a``.

        Let ``A=K(x)<y>`` be the parent Ore algebra, and ``L`` be the operator
        `self`.  An element of ``A/L`` is integral at the place `a` if
        it has non-negative valuation at `a`.  The set of integral elements forms a
        ``K[x]``-module, and an integral basis is a basis of that module.

        The definition of the valuation depends on the type of Ore operators,
        and on some parameters left to the user. 

        The results of this method are cached, additional keywords can be
        supplied to force a new result to be regenerated.

        INPUT:

        - ``a`` -- the place at which to compute an integral basis. ``a`` should
          be an irreducible polynomial in the base ring of the Ore algebra.

        - ``basis`` (default: None) -- starting basis. If provided, the output of the algorithm
          is guaranteed to be integral at all places where ``basis`` was already
          a local integral basis.

        - ``val_fct`` (default: None) -- a function computing the value of an
          operator at the place a. It should have the same interface as the
          generic method ``value_function``. If not provided, the algorithm
          calls ``self.value_function``.

        - ``raise_val_fct`` (default: None) -- a function computing a linear
          combination of operators with higher value. It should have the same
          interface as the generic method ``raise_value``. If not provided, the
          algorithm calls ``self.raise_value``.

        - ``infolevel`` (default:0) -- verbosity flag

        - All remaining named arguments are passed to the functions ``val_fct`` and ``raise_val_fct``.

        If values are given for `val_fct` or `raise_val_fct`, it is the
        responsibility of the user to ensure that those functions are suitable
        for computing an integral basis at place `a`.


        OUTPUT:

        An basis of the quotient of the parent Ore algebra by this operator,
        which is integral at the place ``a``.  If a starting basis was provided,
        the resulting basis is also integral at all places where the starting
        basis was integral.

        EXAMPLES::
        # TODO

        """

        # Helpers
        prefix_base=f"[local {a}]"
        prefix = prefix_base
        
        def print_with_prefix(*args, **kwargs):
            print(prefix, *args, **kwargs)
            
        print1 = print_with_prefix if infolevel >= 1 else lambda *a, **k: None
        print2 = print_with_prefix if infolevel >= 2 else lambda *a, **k: None
        print3 = print_with_prefix if infolevel >= 3 else lambda *a, **k: None

        print1(f"Computing local basis at {a}")
        
        if val_fct is None:
            val_fct = self.value_function
        if raise_val_fct is None:
            raise_val_fct = self.raise_value

        r = self.order()
        ore = self.parent()
        DD = ore.gen()
        if basis is None:
            basis = self._initial_integral_basis(place=a)

        k = ore.base_ring()

        F = a.parent().base_ring()
        deg = a.degree() # Requires a to be the minimal polynomial in extension cases
        Fvar = a.parent().gen(0)

        res = []
        r = len(basis)
        for d in range(r):
            # print1("d={}".format(d))
            prefix = prefix_base + f" {d=}"
            print2("Processing {}".format(basis[d]))
            v = val_fct(basis[d],place=a,**val_kwargs)
            print1("Valuation: {}".format(v))
            res.append(a**(-v) * basis[d])
            print2("Basis element after normalizing: {}".format(res[d]))
            done = False
            while not done:
                alpha = raise_val_fct(res,place=a,dim=r,infolevel=infolevel,**val_kwargs)
                if alpha is None:
                    done = True
                else:
                    print1("Relation found")
                    print2(alpha)

                    alpha_rep = [None for i in range(d+1)]
                    if deg > 1: # Should be harmless even otherwise (then Fvar=1), if we also force the cast to k
                        for i in range(d+1):
                            alpha_rep[i] = sum(alpha[i][j]*Fvar**j for j in range(deg))
                    else:
                        for i in range(d+1):
                            alpha_rep[i] = k(alpha[i])
                    print2("In base field: {}".format(alpha_rep))
                    # __import__("pdb").set_trace()
                    
                    res[d] = sum(alpha_rep[i]*res[i] for i in range(d+1))
                    val = val_fct(res[d],place=a,infolevel=infolevel,**val_kwargs)
                    print1("Valuation raised by {}".format(val))
                    res[d] = a**(-val)*res[d]
                    print2("Basis element after combination: {}".format(res[d]))
                    print1("Valuation after combination: {}".format(
                           val_fct(res[d],place=a,infolevel=infolevel,**val_kwargs)))
        return res

    def find_candidate_places(self, **kwargs):
        r"""
        Compute all places at which an operator in the quotient of the ambient Ore algebra with `self` may not be integral.

        INPUT:

        - Implementations of this virtual method may interpret named arguments.

        OUTPUT:

        Let ``\partial`` be the generator of the Ore algebra and by ``r`` the order of ``self``.
        The function returns a list ``L`` of places such that for any operator ``\partial^k``, ``0 \leq k < r``, in the quotient algebra, and for any place ``a`` not in ``L``, ``\partial^k`` is integral at ``a``.

        Such a list is not unique, since adding finitely many elements to it does not break the specification.
        The caller in global_integral_basis does not require that the list is minimal in any sense.

        Each place may be output as either an irreducible polynomial in the base ring of the parent Ore algebra, or a 3-tuple composed of such a function, as well as suitable functions `value_function` and `raise_valuation`.

        This can be useful in situations where computing the value function involves non-trivial calculations. Defining the functions here allows to capture the relevant data in the function and to minimize the cost at the time of calling.

        EXAMPLES::
        # TODO
        """
        raise NotImplementedError # abstract

    def _normalize_global_integral_basis_args(
            self, places=None, basis=None,
            infolevel=0,**args):
        """
        Normalize the arguments in a call to `global_integral_basis`.

        INPUT: same as `global_integral_basis`

        OUTPUT: a hashable object formed with the arguments, ensuring that the
        validity of an output of `global_integral_basis` only depends on the
        value of this object, and not on the choice of the specific set of
        arguments.

        EXAMPLES: see ``global_integral_basis``

        """
        if basis:
            basis = tuple(basis)
        if places:
            places.sort()
            places = tuple(places)
        args = list(args.items())
        args.sort()
        args = tuple(args)
        return (basis,places,args)

    @cached_method(key=_normalize_global_integral_basis_args)
    def global_integral_basis(self, places=None, basis=None, infolevel=0, **val_kwargs):
        r"""
        Compute a global integral basis of the quotient of the ambient Ore algebra
        with this operator.

        INPUT:

        - ``places`` (default: None) -- list of places. Each place is either an
          irreducible polynomial in the base ring of the Ore algebra, or a
          3-tuple composed of such a polynomial, as well as suitable functions
          `value_function` and `raise_value`.

        - ``basis`` (default: None) -- a basis of the quotient space. If provided, the output of the function is such that the first `i` elements of the integral basis generate the same vector space as the first `i` elements of ``basis``
        
        - ``infolevel`` (default: 0) -- verbosity flag

        All remaining named arguments are passed to the value functions.

        In the differential case, the function takes an additional optional
        argument:

        - ``iota`` (default: None) - a function used to filter terms of
          generalized series solutions which are to be considered
          integral. For the conditions that this function must satisfy, see
          :meth:`ContinuousGeneralizedSeries.valuation`.

        In the recurrence case, the function takes an additional optional
        argument:

        - ``Zmax`` (default: None) - an integer, used to determine an upper
          bound for points at which to eliminate poles. If not provided, uses a
          value guaranteed to be an upper bound for all new poles.

        # TODO: Better phrasing
        # TODO: Rename argument 

        OUTPUT:

        A basis of the quotient algebra which is integral everywhere, or at all
        places in ``places`` if provided.

        It requires that at least the method ``find_candidate_places``, as well
        as ``value_function`` and ``raise_value`` if not provided by
        ``find_candidate_places``, be implemented for that Ore operator.

        EXAMPLES:

        Integral bases can be computed for differential and recurrence operators.

        In the differential case, an operator is integral if, applied to a
        generalized series solution of ``self`` without any pole (except
        possibly at infinity), the resulting series again does not have any
        pole. ::
        
            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = PolynomialRing(QQ)
            sage: OreD.<Dx> = OreAlgebra(Pol)
            sage: L = x*Dx+1
            sage: S = L.generalized_series_solutions(); S
            [x^(-1)]
            sage: B = L.global_integral_basis(); B
            [x]
            sage: [OreD(1)(s) for s in S]
            [x^(-1)]
            sage: [B[0](s) for s in S]
            [1 + O(x^5)] 

            sage: L = Dx+x
            sage: L.generalized_series_solutions()
            [1 - 1/2*x^2 + 1/8*x^4 + O(x^5)]
            sage: L.global_integral_basis()
            [1]

            sage: L = x-1 + Dx - x*Dx^2
            sage: L.generalized_series_solutions(2)
            [x^2*(1 - 1/3*x + O(x^2)), 1 + x + O(x^2)]
            sage: B = L.global_integral_basis(); B
            [1, 1/x*Dx - 1/x]

            sage: L = (-1+2*x) + (1-4*x)*Dx + 2*x*Dx^2
            sage: L.generalized_series_solutions(2)
            [x^(1/2)*(1 + x + O(x^2)), 1 + x + O(x^2)]
            sage: B = L.global_integral_basis(); B
            [1, x*Dx]

            sage: L = x^3*Dx^3 + x*Dx - 1
            sage: L.generalized_series_solutions(2)
            [x, x*((1 + O(x^2))*log(x)), x*((1 + O(x^2))*log(x)^2)]
            sage: B = L.global_integral_basis(); B
            [1, x*Dx, x*Dx^2 - Dx + 1/x]

            sage: L = 24*x^3*Dx^3 - 134*x^2*Dx^2 + 373*x*Dx - 450
            sage: L.generalized_series_solutions(2)
            [x^(15/4), x^(10/3), x^(3/2)]
            sage: B = L.global_integral_basis(); B
            [1/x, 1/x^2*Dx - 3/2/x^3, 1/x*Dx^2 - 3/4/x^3]

        Poles may appear outside of 0.  This example is the same as the previous
        one after a change of variable. ::

            sage: L = 24*(x-2)^3*Dx^3 - 134*(x-2)^2*Dx^2 + 373*(x-2)*Dx - 450
            sage: L.generalized_series_solutions(2)
            [x^2*(1 - 67/72*x + O(x^2)), x*(1 - 103/48*x + O(x^2)), 1 - 139/24*x + O(x^2)]
            sage: B = L.global_integral_basis(); B
            [1/(x - 2),
             (1/(x^2 - 4*x + 4))*Dx - 3/2/(x^3 - 6*x^2 + 12*x - 8),
             (1/(x - 2))*Dx^2 - 3/4/(x^3 - 6*x^2 + 12*x - 8)]

        Poles may appear at non-rational points. ::
        
            sage: L = ((-x + x^3 + 3*x^4 - 6*x^5 + 3*x^6) * Dx^2
            ....:      + (-2 + 4*x + 4*x^2 - 9*x^6 + 18*x^7 - 9*x^8) * Dx
            ....:      + (4 + 2*x - 18*x^4 + 18*x^6 - 18*x^7))
            sage: a = (x^4 - x^3 + 1/3*x + 1/3).any_root(ComplexField(20)); a
            -0.39381 - 0.38222*I
            sage: L[L.order()](a) # abs tol 1e-6
            -1.9398e-7 + 9.5133e-7*I
            sage: L.local_basis_expansions(a,2)
            [1.00000*1, 1.00000*(x + 0.39381 + 0.38222*I)]
            sage: bb = L.global_integral_basis()
            sage: bb = [1/b.leading_coefficient().numerator().leading_coefficient() * b for b in bb]
            sage: bb
            [x^3 - 2*x^2 + x,
             ((x^2 - x)/(x^4 - x^3 + 1/3*x + 1/3))*Dx + (-3*x^6 + 9*x^5 - 9*x^4 + 2*x^3 + x^2 + 3*x - 1)/(x^4 - x^3 + 1/3*x + 1/3)]

        Integrality is not defined for non-Fuchsian operators, that is operators
        for which some generalized series solutions have non-rational exponents
        or a non-trivial exponential part. ::

            sage: L = x^2*Dx^2 + x*Dx + 1
            sage: L.local_basis_expansions(0)
            [x^(-1*I), x^(1*I)]
            sage: L.global_integral_basis()
            Traceback (most recent call last):
            ...
            ValueError: The operator has non Fuchsian series solutions

            sage: L = (x^2-2)*Dx + 1
            sage: L.local_basis_expansions(sqrt(2),1)
            [(x - sqrt(2))^(-0.3535533905932738?)]
            sage: L.global_integral_basis()
            Traceback (most recent call last):
            ...
            ValueError: The operator has non Fuchsian series solutions

        The definition of integral bases in the differential case depends on the
        choice of a function `\iota` evaluating the contribution of each term of
        the generalized series solution.  For the conditions that this function
        must satisfy, see :meth:`ContinuousGeneralizedSeries.valuation`.

        The default value of that function is such that a series is considered
        integral if and only if it is bounded in a neighborhood of 0.

        Different `\iota` functions give different integral bases.  It can only
        make a difference if there are logarithmic terms in a fundamental system
        of solutions, or if the initial exponent is irrational. ::

            sage: L = x*Dx^2 + Dx
            sage: L.generalized_series_solutions(1)
            [1 + O(x), (1 + O(x))*log(x)]
            sage: B = L.global_integral_basis(); B
            [x, x*Dx]
            sage: B = L.global_integral_basis(iota = lambda i,j : j); B
            [x, x*Dx]
            sage: B = L.global_integral_basis(iota = lambda i,j : j if i==0 else 0); B
            [1, x*Dx]
            sage: B = L.global_integral_basis(iota = lambda i,j : j if i==0 else 1); B
            [x, x*Dx]
            sage: B = L.global_integral_basis(iota = lambda i,j : j if i==0 else -1); B
            [1/x, x*Dx]

        Optionally, we can supply a list of points at which we want to compute
        an integral basis. Each point is given by its minimal polynomial in the
        base polynomial ring.

        
        In the recurrence case, we consider deformed operators: given a linear
        recurrence operator `L \in \QQ[x]\<Sx\>`, the deformed operator `L_q` is
        the operator `L(x+q) \in \QQ[q][x]\<Sx\>`.  Such an operator with order
        `r` always admits `r` linearly independent solutions in
        `QQ((q))^(z+\ZZ)` for `z \in \CC`.

        Fix `N_{max} \in \ZZ`.  Such a solution `f` is said to be integral at
        `z` if for all `k \in \ZZ` with `k \leq N_{max}`, `f(z+k) \in \QQ[[q]]`.
        An operator `B` in the algebra quotient by `L` is integral at `z` if for
        all solutions `f` of `L_q`, `B_q(f)` is integral at `z`. ::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = PolynomialRing(QQ)
            sage: Rec.<Sx> = OreAlgebra(Pol)
            sage: (x*Sx+1).global_integral_basis()
            [x - 1]
            sage: (Sx+x).global_integral_basis()
            [1/(x - 1)]
        
        If a solution has larger valuation in `q` towards `+\infty` than towards
        `-\infty`, the algorithm uses `N_{max}` as a cutoff value. In this case,
        different values of `N_{max}` yield different results, which differ by a
        rational factor. ::

            sage: L = ((x+2)^2 + x*Sx^2 + (x+2)*Sx^3)
            sage: B = L.global_integral_basis(); B
            [x - 1,
             1/x*Sx + (x - 2)/x^2,
             (1/(x^3 - x^2 - x + 1))*Sx^2 + (1/(x^3 + x^2 - x - 1))*Sx + (1/4*x + 1/4)/(x - 1)]
            sage: B = L.global_integral_basis(Zmax=2); B
            [1, 1/x*Sx + (x - 2)/x^2, (1/(x + 1))*Sx^2 + ((x - 1)/(x^2 + 2*x + 1))*Sx]

        The results of this function are cached. ::

            sage: R.<x> = QQ['x']
            sage: A.<Dx> = OreAlgebra(R, 'Dx')
            sage: L = x*Dx^2 + 1
            sage: places1 = [x+1,x^2+1]
            sage: L.global_integral_basis.is_in_cache(basis=None, places=places1)
            False
            sage: L.global_integral_basis(basis=None, places=places1)
            [1, x*Dx]
            sage: L.global_integral_basis.is_in_cache(basis=None, places=places1)
            True

        If provided, the functions for computing and raising the valuation at
        each place are also part of the caching key. ::

            sage: dummy_val = lambda op,place,**kwargs : 0
            sage: dummy_raise = lambda vects, place, **kwargs : None
            sage: places2 = [(x+1,dummy_val,dummy_raise)]
            sage: L.global_integral_basis.is_in_cache(places=places2)
            False
            sage: L.global_integral_basis(places=places2)
            [1, x*Dx]
            sage: L.global_integral_basis.is_in_cache(places=places2)
            True

        If the functions use global variables, changing those variables without
        redefining the function will not invalidate the cache. ::

            sage: dummy_val2 = lambda op,place,**kwargs : 1/a -1
            sage: places3 = [(x+1,dummy_val2,dummy_raise)]
            sage: L.global_integral_basis.is_in_cache(places=places3)
            False
            sage: a=1
            sage: L.global_integral_basis(places=places3)
            [1, x*Dx]
            sage: L.global_integral_basis.is_in_cache(places=places3)
            True
            sage: a=0
            sage: dummy_val2(None,None)
            Traceback (most recent call last):
            ...
            ZeroDivisionError: rational division by zero
            sage: L.global_integral_basis(places=places2) # invalid cache!
            [1, x*Dx]

        Changing the verbosity level is ignored. ::
        
            sage: L.global_integral_basis.is_in_cache(places=places1, infolevel=2)
            True

        All other arguments, including the initial basis, can give a different result. ::
        
            sage: basis2 = [1,x^2*Dx]
            sage: L.global_integral_basis.is_in_cache(basis=basis2, places=places1)
            False

        It is possible to bypass the cached value by passing additional
        parameters to the method. ::
        
            sage: L.global_integral_basis.is_in_cache(unused_arg=15)
            False
            sage: L.global_integral_basis(unused_arg=15)
            [1, x*Dx]
            sage: L.global_integral_basis.is_in_cache(unused_arg=15)
            True

        Note that the subroutine ``local_integral_basis`` also caches its
        results, so if one needs to clear the cache of
        ``global_integral_basis``, one should also clear the cache of
        ``local_integral_basis``.

        TESTS::

            sage: Pol.<x> = PolynomialRing(QQ)
            sage: Rec.<Sx> = OreAlgebra(Pol)
            sage: L = x*Sx+1
            sage: L.global_integral_basis.is_in_cache()
            False
            sage: L.global_integral_basis()
            [x - 1]
            sage: L.global_integral_basis.is_in_cache()
            True
        
        """
        if places is None:
            places = self.find_candidate_places(infolevel=infolevel,**val_kwargs)

        r = self.order()
        ore = self.parent()
        DD = ore.gen()
        if basis is None:
            res = self._initial_integral_basis(place=None)
        else:
            res = basis

        if len(places) == 0 :
            return [self.parent()(1)]
            
        for p in places :
            if not isinstance(p,tuple) :
                x = p
                val_fct = raise_val_fct = None
            else:
                x, val_fct, raise_val_fct = p
                
            res = self.local_integral_basis(x,basis=res,
                                            val_fct = val_fct,
                                            raise_val_fct = raise_val_fct,
                                            infolevel=infolevel,
                                            **val_kwargs) 
        return res

#############################################################################################################

def _commutativeRadical(p):
    r"""
    Computes the radical in degenerate cases. Used by radical(self)
    """

    if p.degree()==0:
        p = p.parent().base_ring()(p)
        for i in range(min(log(p.numerator()),log(p.denominator()))+1,2,-1):
            try:
                return (p.nth_root(i),i)
            except:
                pass
        return (p,1)
    sqf=p.squarefree_decomposition()
    exponents=[d for (c,d) in sqf]
    prad=1
    d = gcd(exponents)
    for i in range(len(sqf)):
        prad=prad*sqf[i][0]**(exponents[i]/d)
    sgn=p.leading_coefficient().sign()
    return (p.parent()(sgn*(sgn*p.leading_coefficient())**(1/d)/prad.leading_coefficient())*prad,d)

def _orePowerSolver(P):
    r"""
    Solver for special algebraic systems used in radical computation
    """

    R = P.parent()
    K = R.base_ring().base_ring()
    Q = K.base_ring()
    n = R.base_ring().gen()
    gens = list(K.gens())
    c = gens.pop()
    for i in range(P.order()+1):
        cS = P.coefficients(sparse=False)[P.order()-i]
        for j in range(cS.degree()+1):
            cN = cS.coefficients(sparse=False)[cS.degree()-j]
            if (cN.degree()==0):
                return []
            if (len(gens)==0) or (cN.degree(c) == cN.total_degree()):
                sols=PolynomialRing(Q,c)(cN).roots()
                for s in sols:
                    sol=s[0]
                    if len(gens)>0:
                        K2=PolynomialRing(Q,gens)
                    else:
                        K2=Q
                    K3=PolynomialRing(K2,n)
                    P2=P.map_coefficients(lambda x: x.map_coefficients(lambda y: y.subs({c:sol}),K2),K3)
                    if len(gens)==0:
                        if P2==0:
                            return [sol]
                        return []
                    recSol=_orePowerSolver(P2)
                    if not len(recSol)==0:
                        recSol.append(sol)
                        return recSol
    return []

def _listToOre(l,order,R):
    r"""
    Converts a list of values into an Ore polynomial in R. l[0] will be used for the leading coefficient, l[len(l)-1] for the trailing coefficient.
    
    INPUT:

    - ``l`` -- a list with values in R.base_ring().base_ring().
    - ``order`` -- the order of the Ore operator. Has to be a divisor of len(l).
    - ``R`` -- an Ore algebra.

    """
    S = R.gen()
    n = R.base_ring().gen()
    res = 0
    d = len(l)//order
    for i in range(len(l)):
        res = res+l[i]*n**(i%d)*S**(i//d)
    return res
