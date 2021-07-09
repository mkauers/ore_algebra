# coding: utf-8
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

from __future__ import absolute_import, division, print_function

from functools import reduce

import sage.functions.log as symbolic_log

from sage.arith.all import previous_prime as pp
from sage.arith.all import gcd, lcm, nth_prime, srange
from sage.functions.all import floor
from sage.matrix.constructor import matrix
from sage.misc.all import prod
from sage.rings.fraction_field import FractionField_generic
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.infinity import infinity
from sage.rings.number_field.number_field import NumberField
from sage.rings.number_field.number_field_base import is_NumberField
from sage.rings.qqbar import QQbar
from sage.rings.qqbar import QQbar, AA
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.structure.element import RingElement, canonical_coercion, get_coercion_model
from sage.structure.factorization import Factorization
from sage.structure.formal_sum import FormalSum, FormalSums
from sage.symbolic.all import SR

from .tools import clear_denominators, q_log, make_factor_iterator, shift_factor, _vect_val_fct, _vect_elim_fct, roots_at_integer_distance
from .ore_algebra import OreAlgebra_generic
from .ore_operator import OreOperator, UnivariateOreOperator
from .generalized_series import GeneralizedSeriesMonoid, _generalized_series_shift_quotient, _binomial

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
        L = self; A = L.parent(); R = A.base_ring(); K = R.base_ring()

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
        A = self.parent(); R = A.base_ring(); 
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

        from sage.matrix.constructor import matrix

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
          [((x^2 + 8*x + 16)/x^3,), ((x^2 + 8*x + 16)/(x^2 + 6*x + 9),)]
          sage: L.rational_solutions((1, x))
          [((7*x^5 + 21*x^4 + 73*x^2 + 168*x + 144)/(x^5 + 6*x^4 + 9*x^3), 5184, 756),
           ((4*x^2 + 14*x + 1)/(x^2 + 6*x + 9), 2592, 378),
           ((7*x^2 + 24*x)/(x^2 + 6*x + 9), 4608, 672)]
          sage: L(_[0][0]) == _[0][1] + _[0][2]*x
          True

          sage: (x*(x*Dx-5)).rational_solutions([1])
          [(1/x, -6), (x^5, 0)]

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

        output = []; k = 0; infty = max([j for _, j in points]) + 2
        while k < len(points) - 1:
            (i1, j1) = points[k]; m = infty
            poly = coeffs[i1]
            for l in range(k + 1, len(points)):
                (i2, j2) = points[l]; m2 = flip*(j2 - j1)/(i2 - i1)
                if m2 == m:
                    k = l; poly += coeffs[i2]*x**(i2 - i1)
                elif m2 < m:
                    m = m2; k = l; poly = coeffs[i1] + coeffs[i2]*x**(i2 - i1)
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

        s = R.zero(); y_ff_i = R.one()
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
          sage: Q.order()
          3
          sage: Q.leading_coefficient().degree()
          1

        """

        L = self.numerator()
        A = L.parent()
        if A.base_ring().is_field():
            A = A.change_base(A.base_ring().base())
            L = A(L)
        R = A.base_ring(); C = R.base_ring()
        sub = m - 1

        if m < 0:
            m = L._desingularization_order_bound()
            sub = 0
        
        if m <= 0:
            return L

        deg = None; Dold = A.zero()

        for k in range(m, sub, -1):
            D = A.zero(); 
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
        P = self; A = P.parent(); R = A.base_ring()

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
            adjoint = A.zero(); coeffs = P.to_F('F').coefficients(sparse=False); r = P.order()
            for i in range(len(coeffs)):
                adjoint += S**(r-i)*(A.one() - S)**i * coeffs[i]
        elif A.is_D() is not False or A.is_T() is not False:
            if D != A.gen():
                raise NotImplementedError("unsupported choice of D: " + str(D))
            # adjoint = sum( (-D)^i * a[i] ), where a[i] is the coeff of D in P
            adjoint = A.zero(); coeffs = P.coefficients(sparse=False)
            for i in range(len(coeffs)):
                adjoint += (-D)**i * coeffs[i]
        else:
            raise NotImplementedError

        sol = adjoint.rational_solutions((-p,))
        A = A.change_ring(A.base_ring().fraction_field())
        sigma = A.sigma(); delta = A.delta()

        for i in range(len(sol)):
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
                if sol!=L: return (sol,self.order()/sol.order())

    def _radicalExp(self):
        r"""
        For an Ore polynomial P, this method computes candidates for possible
        powers k such that there exists an operator L with P=L^k.

        OUTPUT:

        A list of integers k such that there possibly exists an operator L such that `self' equals L^k.

        Note: This method only works for operators over Q[n].

        """
        p = self._powerIndicator()
        exponents=[divisors(d) for (c,d) in p.squarefree_decomposition()]
        M=[]
        for a in exponents[0]:
            contained = True
            for i in range(1,len(exponents)):
                contained = contained and a in exponents[i]
            if contained: M.append(a)
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
        from sage.rings.laurent_series_ring import LaurentSeriesRing

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
                C = GF(pp(2**23)); xi = C(ev(-pol[0]/pol[1]))
            else:
                modulus = 2**23; done = False
                while not done:
                    modulus = pp(modulus); C = GF(modulus)
                    for u, _ in ev(pol).change_ring(C).factor():
                        if u.degree() == 1:
                            xi = -u[0]/u[1]; done = True; break

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
                sol = [ R.zero() for j in range(r) ]; sol[i] = R.one()
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
                sol = [ R.zero() for j in range(r) ]; sol[i] = R.one()
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
        S = A.gen(); sigma = A.sigma()
        assert(R.characteristic() == 0)

        if A.is_Q():
            q_case = True; x, q = A.is_Q()
        elif A.is_S():
            q_case = False; x = R.gen()
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

        SELF = A([R(c) for c in coeffs]); r = SELF.order()

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
                u[0][2] = merge(u[0][2], alpha); u[0][3] += 1

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
                        idx[j] = 0; idx[j - 1] += 1
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
            valg = 0; dim = r; alpha = 1 if q_case else 0
            for _, u in c:
                valg += u[4]; dim = min(dim, u[1])
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
                for u in c: u[1][1] -= len(sols) 
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

    def local_integral_basis(self, x, basis=None,
                             val_fct=None, raise_val_fct=None,
                             infolevel=0,
                             **val_kwargs):
        r"""
        Compute a local integral basis at x of the vector obtained by taking the
        quotient of the parent Ore algebra by this operator.

        INPUT:

        - ``x`` -- the place at which to compute an integral basis. ``x`` should
          be an irreducible polynomial in the base ring of the Ore algebra.

        - ``basis`` (default: None) -- starting basis. If provided, the output of the algorithm
          is guaranteed to be integral at all places where ``basis`` was already
          a local integral basis.

        - ``val_fct`` (default: None) -- a function computing the value of an
          operator at the place x. It should have the same interface as the
          generic method ``value_function``. If not provided, the algorithm
          calls ``self.value_function``.

        - ``raise_val_fct`` (default: None) -- a function computing a linear combination of operators with higher value. It should have the same interface as the
          generic method ``raise_value``. If not provided, the algorithm
          calls ``raise_value``.

        - ``infolevel`` (default:0) -- verbosity flag

        - All remaining named arguments are passed to the functions ``val_fct`` and ``raise_val_fct``.

        OUTPUT:

        An basis of the quotient of the parent Ore algebra by this operator, which is integral at the place ``x``.
        If a starting basis was provided, the resulting basis is also integral at all places where the starting basis was integral.

        EXAMPLES::
        # TODO
        """

        # Helpers
        print1 = print if infolevel >= 1 else lambda *a, **k: None
        print2 = print if infolevel >= 2 else lambda *a, **k: None
        print3 = print if infolevel >= 3 else lambda *a, **k: None

        print1(" [local] Computing local basis at {}".format(x))
        
        if val_fct is None: val_fct = self.value_function
        if raise_val_fct is None: raise_val_fct = self.raise_value

        r = self.order()
        ore = self.parent()
        DD = ore.gen()
        if basis is None: basis = [DD**i for i in range(r)]

        k = ore.base_ring()

        F = x.parent().base_ring()
        deg = x.degree() # Requires x to be the minimal polynomial in extension cases
        Fvar = x.parent().gen(0)

        res = []
        r = len(basis)
        for d in range(r):
            print1(" [local] d={}".format(d))
            print1(" [local] Processing {}".format(basis[d]))
            v = val_fct(basis[d],place=x,**val_kwargs)
            print1(" [local] Valuation: {}".format(v))
            res.append(x**(-v) * basis[d])
            print1(" [local] Basis element after normalizing: {}".format(res[d]))
            done = False
            while not done:
                alpha = raise_val_fct(res,place=x,dim=r,infolevel=infolevel,**val_kwargs)
                if alpha is None:
                    done = True
                else:
                    print1(" [local] Relation found: {}".format(alpha))

                    alpha_rep = [None for i in range(d+1)]
                    if deg > 1: # Should be harmless even otherwise (then Fvar=1), if we also force the cast to k
                        for i in range(d+1):
                            alpha_rep[i] = sum(alpha[i][j]*Fvar**j for j in range(deg))
                    else:
                        for i in range(d+1):
                            alpha_rep[i] = k(alpha[i])
                    print2(" [local] In base field: {}".format(alpha_rep))
                    # __import__("pdb").set_trace()
                    
                    res[d] = sum(alpha_rep[i]*res[i] for i in range(d+1))
                    res[d] = x**(- val_fct(res[d],place=x,**val_kwargs))*res[d]
                    print1(" [local] Basis element after combination: {}".format(res[d]))
        return res

    def find_candidate_places(self, **kwargs):
        r"""
        Compute all places at which an operator in the quotient of the ambient Ore algebra with `self` may not be integral.

        INPUT:

        - Implementations of this virtual method may interpret named arguments.

        OUTPUT:

        Let ``\partial`` be the generator of the Ore algebra and by ``r`` the order of ``self``.
        The function returns a list ``L`` of places such that for any operator ``\partial^k``, ``0 \leq k < r``, in the quotient algebra, and for any place ``z`` not in ``L``, ``\partial^k`` is integral at ``z``.

        Such a list is not unique, since adding finitely many elements to it does not break the specification.
        The caller in global_integral_basis does not require that the list is minimal in any sense.

        Each place may be output as either an irreducible polynomial in the base ring of the parent Ore algebra, or a 3-tuple composed of such a function, as well as suitable functions `value_function` and `raise_valuation`.

        This can be useful in situations where computing the value function involves non-trivial calculations. Defining the functions here allows to capture the relevant data in the function and to minimize the cost at the time of calling.

        EXAMPLES::
        # TODO
        """
        raise NotImplementedError # abstract

    def global_integral_basis(self, places=None, infolevel=0, **val_kwargs):
        r"""
        Compute a global integral basis of the quotient of the ambient Ore algebra
        with this operator.

        INPUT:

        - ``places`` (default: None) -- list of places. Each place is either an
          irreducible polynomial in the base ring of the Ore algebra, or a
          3-tuple composed of such a polynomial, as well as suitable functions
          `value_function` and `raise_value`.

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

        EXAMPLES::

        Integral bases can be computed for differential and recurrence operators.

        In the differential case, an operator is integral if, applied to a
        generalized series solution of ``self`` without any pole (except
        possibly at infinity), the resulting series again does not have any
        pole.
        
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
        one after a change of variable.

            sage: L = 24*(x-2)^3*Dx^3 - 134*(x-2)^2*Dx^2 + 373*(x-2)*Dx - 450
            sage: L.generalized_series_solutions(2)
            [x^2*(1 - 67/72*x + O(x^2)), x*(1 - 103/48*x + O(x^2)), 1 - 139/24*x + O(x^2)]
            sage: B = L.global_integral_basis(); B
            [1/(x - 2),
             (1/(x^2 - 4*x + 4))*Dx - 3/2/(x^3 - 6*x^2 + 12*x - 8),
             (1/(x - 2))*Dx^2 - 3/4/(x^3 - 6*x^2 + 12*x - 8)]

        Poles may appear at non-rational points.
        
            sage: L = ((-x + x^3 + 3*x^4 - 6*x^5 + 3*x^6) * Dx^2
            ....:      + (-2 + 4*x + 4*x^2 - 9*x^6 + 18*x^7 - 9*x^8) * Dx
            ....:      + (4 + 2*x - 18*x^4 + 18*x^6 - 18*x^7))
            sage: a = (x^4 - x^3 + 1/3*x + 1/3).any_root(ComplexField(20)); a
            -0.39381 - 0.38222*I
            sage: L[L.order()](a) # abs tol 1e-6
            -1.9398e-7 + 9.5133e-7*I
            sage: L.local_basis_expansions(a,2)
            [1.00000*1, 1.00000*(x + 0.39381 + 0.38222*I)]
            sage: L.global_integral_basis()
            [x^3 - 2*x^2 + x,
             ((x^2 - x)/(x^4 - x^3 + 1/3*x + 1/3))*Dx + (-3*x^6 + 9*x^5 - 9*x^4 + 2*x^3 + x^2 + 3*x - 1)/(x^4 - x^3 + 1/3*x + 1/3)]

        Integrality is not defined for non-Fuchsian operators, that is operators
        for which some generalized series solutions have non-rational exponents
        or a non-trivial exponential part.

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
        of solutions, or if the initial exponent is irrational.

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
            [x, x^2*Dx]
            sage: B = L.global_integral_basis(iota = lambda i,j : j if i==0 else -1); B
            [1/x, Dx]

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
        all solutions `f` of `L_q`, `B_q(f)` is integral at `z`.

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
        rational factor.

            sage: L = ((x+2)^2 + x*Sx^2 + (x+2)*Sx^3)
            sage: B = L.global_integral_basis(); B
            [x - 1,
             1/x*Sx + (x - 2)/x^2,
             (1/(x^3 - x^2 - x + 1))*Sx^2 + (1/(x^3 + x^2 - x - 1))*Sx + (1/4*x + 1/4)/(x - 1)]
            sage: B = L.global_integral_basis(Zmax=2); B
            [1, 1/x*Sx + (x - 2)/x^2, (1/(x + 1))*Sx^2 + ((x - 1)/(x^2 + 2*x + 1))*Sx]

        """
        # sage: ((x+2)^2 + x*Sx^2 + (x+2)*Sx^3).global_integral_basis(Zmax=3)
        # [1, 1/x*Sx + (x - 2)/x^2, (1/(x + 1))*Sx^2 + ((x - 1)/(x^2 + 2*x + 1))*Sx]
        # sage: ((x^2+2)^2 + x*Sx^2 + (x^2+2)*Sx^3).global_integral_basis(Zmax=3)
        # [1,
        #  (1/(x^2 - 4*x + 6))*Sx + (x - 2)/(x^4 - 8*x^3 + 28*x^2 - 48*x + 36),
        #  (1/(x^2 - 2*x + 3))*Sx^2 + ((x - 1)/(x^4 - 4*x^3 + 10*x^2 - 12*x + 9))*Sx]

        if places is None:
            places = self.find_candidate_places(infolevel=infolevel,**val_kwargs)

        if len(places) == 0 :
            return [self.parent()(1)]
            
        res = None
        for p in places :
            if len(p) == 1 :
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

class UnivariateDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
    Element of an Ore algebra K(x)[D], where D acts as derivation d/dx on K(x).
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):
        
        if not "action" in kwargs:
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
        if type(alg) == str:
            R = self.base_ring(); x = R.gen(); one = R.one()
            rec_algebra = self.parent().change_var_sigma_delta(alg, {x:x+one}, {})
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
        R = self.base_ring(); one = R.one(); x = R.gen()

        if isinstance(alg, str):
            alg = self.parent().change_var_sigma_delta(alg, {}, {x:x})
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
                out[j] += (-1 if (i+j)%2 else 1)*stirling[i][j]*c << (ord-i)
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

    def annihilator_of_composition(self, a, solver=None):
        r"""
        Returns an operator `L` which annihilates all the functions `f(a(x))`
        where `f` runs through the functions annihilated by ``self``.
        The output operator is not necessarily of smallest possible order.

        INPUT:

        - ``a`` -- either an element of the base ring of the parent of ``self``,
          or an element of an algebraic extension of this ring.
        - ``solver`` (optional) -- a callable object which applied to a matrix
          with polynomial entries returns its kernel. 

        EXAMPLES::

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
                raise TypeError("argument not recognized as algebraic function over base ring")

        d = minpoly.degree(); r = self.order()

        # derivative of a
        Da = -minpoly.map_coefficients(lambda p: p.derivative())
        Da *= minpoly.xgcd(minpoly.derivative())[2]
        Da = Da % minpoly

        # self's coefficients with x replaced by a, denominators cleared, and reduced by minpoly.
        # have: (D^r f)(a) == sum( red[i]*(D^i f)a, i=0..len(red)-1 ) and each red[i] is a poly in Y of deg <= d.
        red = [ R(p.numerator().coefficients(sparse=False)) for p in self.numerator().change_ring(K).coefficients(sparse=False) ]
        lc = -minpoly.xgcd(red[-1])[2]
        red = [ (red[i]*lc) % minpoly for i in range(r) ]

        from sage.matrix.constructor import Matrix
        Dkfa = [R.zero() for i in range(r)] # Dkfa[i] == coeff of (D^i f)(a) in D^k (f(a))
        Dkfa[0] = R.one()
        mat = [[ q for p in Dkfa for q in p.padded_list(d) ]]; sol = []

        while len(sol) == 0:

            # compute coeffs of (k+1)th derivative
            next = [ (p.map_coefficients(lambda q: q.derivative()) + p.derivative()*Da) % minpoly for p in Dkfa ]
            for i in range(r - 1):
                next[i + 1] += (Dkfa[i]*Da) % minpoly
            for i in range(r):
                next[i] += (Dkfa[-1]*red[i]*Da) % minpoly
            Dkfa = next

            # check for linear relations
            mat.append([ q for p in Dkfa for q in p.padded_list(d) ])
            sol = solver(Matrix(K, mat).transpose())

        return self.parent()(list(sol[0]))

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
        elif not (R.base_ring() is QQ or is_NumberField(R.base_ring())):
            raise TypeError("cannot compute generalized solutions for this coefficient domain")

        solutions = []

        # solutions with exponential parts

        if exp is True:
            exp = QQ(self.degree() * self.order()) # = infinity
        elif exp is False:
            exp = QQ.zero()
        if exp not in QQ:
            raise ValueError("illegal option value encountered: exp=" + str(exp))

        # search for a name which is not yet used as generator in (some subfield of) R.base_ring()
        # for in case we need to make algebraic extensions.

        # TODO: use QQbar as constant domain.

        K = R.base_ring(); names = []
        while K is not QQ:
            names.append(str(K.gen()))
            K = K.base_ring()
        i = 0; newname = 'a_0'
        while newname in names:
            i = i + 1; newname = 'a_' + str(i)

        x = self.base_ring().gen()

        if exp > 0:

            points = []
            c = self.coefficients(sparse=False)
            for i in range(self.order() + 1):
                if not c[i].is_zero():
                    points.append((QQ(i), QQ(c[i].valuation())))

            y = R.base_ring()['x'].gen(); # variable name changed from y to x to avoid PARI warning
            x = R.gen(); K = R.base_ring(); 
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
                    a = e.numerator(); b = e.denominator()
                    G = GeneralizedSeriesMonoid(c.parent(), x, "continuous")
                    s = G(R.one(), exp = e*c*(x**(-a)), ramification = b)
                    L = self.annihilator_of_composition(x**b).symmetric_product(x**(1-a)*D + a*c)
                    sol = L.generalized_series_solutions(n, base_extend, ramification, -a)
                    solutions = solutions + list(map(lambda f: s*f.substitute(~b), sol))

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

                coeffs = _rec2list(L.to_S('S'), [K.one()], n, alpha, False, True, lambda p:p)
                solutions.append(G(PS(coeffs, n), exp=alpha))

            else:
                # there may be logarithms, use general code
                L = L.base_extend(K[s].fraction_field()[x]).to_S('S')

                f = f0 = indpoly.parent().one()
                for (a, b) in e:
                    f0 *= c(s + a)**b

                for i in range(e[-1][0]):
                    f *= f0(s + i + 1) 

                coeffs = _rec2list(L, [f], n, s, False, True, lambda p:p)

                # If W(s, x) denotes the power series with the above coefficient array,
                # then [ (d/ds)^i ( W(s, x)*x^s ) ]_{s=a} is a nonzero solution for every
                # root a = alpha - e[j][0] of f0 and every i=0..e[j][1]-1.

                # D_s^i (W(s, x)*x^s) = (D_s^i W + i*log(x)*D_s^(i-1) W + binom(i,2)*log(x)^2 D_s^(i-2) W + ... )*x^s.

                m = sum([ee[1] for ee in e])
                der = [coeffs]
                while len(der) < m:
                    der.append(list(map(lambda g: g.derivative(), der[-1])))

                accum = 0
                for (a, b) in e:
                    der_a = dict()
                    for i in range(accum + b):
                        der_a[i] = PS(list(map(lambda g: g(alpha - a), der[i])), len(der[i]))
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

        L = op.parent().base_ring() # k[x]
        if L.is_field():
            L = L.ring()
        K = PolynomialRing(L.base_ring(), var) # k[alpha]

        if op.is_zero():
            return K.zero()
        if op.order() == 0:
            return K.one()

        r = op.order()
        d = op.degree()

        L = L.change_ring(K) # FF(k[alpha])[x]
        alpha = L([K.gen()])

        ffac = [L.one()] # falling_factorial(alpha, i)
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
        try: ## facilitate factorization
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
        try: ## facilitate factorization
            den = lcm( [ p.denominator() for p in lc ])
            lc = lc.map_coefficients(lambda p: den*p)
        except:
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
                if any (p1.degree() != p.degree() for p, p1 in zip(fac, fac1)):
                    continue
                if any(L1[i].valuation() != L[i].valuation() for i in range(L.order() + 1)):
                    continue
                break
        else:
            L1, fac1 = L, fac

        bound = []
        for (p, p1) in zip(fac, fac1):
            e = 0
            for j in range(r + 1): ## may be needed for inhomogeneous part
                if not L1[j].is_zero():
                    e = max(e, L1[j].valuation(p1) - j)
            for (q, _) in L1.indicial_polynomial(p1).factor(): ## contribution for homogeneous part
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

        local_data = []
        for p in make_factor_iterator(R, False)(L.leading_coefficient()):
            pass

        raise NotImplementedError

    finite_singularities.__doc__ = UnivariateOreOperatorOverUnivariateRing.finite_singularities.__doc__

    def local_basis_monomials(self, point):
        r"""
        Return the leading logarithmic monomials of a local basis of solutions.

        INPUT:

        ``point`` - a regular point of this operator

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
        but consistent way.) Note that this basis may not coincide with the one
        computed by :meth:`generalized_series_solutions`.

        .. seealso::

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
        struct = Point(point, dop).local_basis_structure()
        x = SR(dop.base_ring().gen()) - point
        return [x**simplify_exponent(sol.valuation)
                    *symbolic_log.log(x, hold=True)**sol.log_power
                    /sol.log_power.factorial()
                for sol in struct]

    # TODO: Add a version that returns DFiniteFunction objects
    def local_basis_expansions(self, point, order=None, ring=None):
        r"""
        Generalized series expansions of the local basis.

        INPUT:

        * point - Point where the local basis is to be computed

        * order (optional) - Number of terms to compute, *starting from each
          “leftmost” valuation of a group of solutions with valuations differing by
          integers*. (Thus, the absolute truncation order will be the same for all
          solutions in such a group, with some solutions having more actual
          coefficients computed that others.)

          The default is to choose the truncation order in such a way that the
          structure of the basis is apparent, and in particular that logarithmic
          terms appear if logarithms are involved at all in that basis. The
          corresponding order may be very large in some cases.

        * ring (optional) - Ring into which to coerce the coefficients of the
          expansion

        OUTPUT:

        A list of ``sage.structure.formal_sum.FormalSum` objects. Each term of
        each sum is a monomial of the form ``dx^n*log(dx)^k``  for some ``dx``,
        ``n``, and ``k``, multiplied by a coefficient belonging to ``ring``.
        See below for examples of how to access these parameters.

        .. seealso::

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

        TESTS::

            sage: (4*x^2*Dx^2 + (-x^2+8*x-11)).local_basis_expansions(0, 2)
            [x^(-1.232050807568878?) + (-4/11*a+2/11)*x^(-0.2320508075688773?),
            x^2.232050807568878? - (-4/11*a-2/11)*x^3.232050807568878?]

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
        from .analytic.local_solutions import (log_series, LocalBasisMapper,
                simplify_exponent, LogMonomial)
        from .analytic.path import Point
        mypoint = Point(point, self)
        dop = DifferentialOperator(self)
        ldop = dop.shift(mypoint)
        if order is None:
            ind = ldop.indicial_polynomial(ldop.base_ring().gen())
            order = max(dop.order(), ind.dispersion()) + 3
        class Mapper(LocalBasisMapper):
            def fun(self, ini):
                return log_series(ini, self.shifted_bwrec, order)
        sols = Mapper(ldop).run()
        x = SR.var(dop.base_ring().variable_name())
        dx = x if point == 0 else x.add(-point, hold=True)
        if ring is None:
            cm = get_coercion_model()
            ring = cm.common_parent(
                    dop.base_ring().base_ring(),
                    mypoint.value.parent(),
                    *(sol.leftmost for sol in sols))
        res = [FormalSum(
                    [(c/ZZ(k).factorial(), LogMonomial(dx, sol.leftmost, n, k))
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

        .. seealso:: :meth:`numerical_transition_matrix`

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
        from .analytic import analytic_continuation as ancont, local_solutions
        from .analytic.differential_operator import DifferentialOperator
        dop = DifferentialOperator(self)
        post_transform = ancont.normalize_post_transform(dop, post_transform)
        post_mat = matrix(1, dop.order(),
                lambda i, j: ZZ(j).factorial()*post_transform[j])
        ctx = ancont.Context(**kwds)
        sol = ancont.analytic_continuation(dop, path, eps, ctx, ini=ini,
                                         post=post_mat, return_local_bases=True)
        val = []
        asycst = local_solutions.sort_key_by_asympt(QQbar.zero(), ZZ.zero())
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

        .. seealso:: :meth:`numerical_solution`

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

    
    def _make_valuation_place(self, f, iota=None, prec=None, infolevel=0):
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

        - ``infolevel`` (default: 0) - verbosity flag

        OUTPUT:

        A tuple composed of ``f``, a suitable function for ``value_function`` at
        ``f`` and a suitable function for ``raise_value`` at ``f``.
        
        EXAMPLES::

        # TODO
        
        """

        if infolevel >= 1: print("Preparing place at {}"
                                 .format(f if f.degree() < 10
                                         else "{} + ... + {}".format(f[f.degree()]*f.monomials()[0],f[0])))

        r = self.order()
        ore = self.parent()
        x = ore.base_ring().gen()
        C = ore.base_ring().base_ring()
        if f.degree() > 1:
            FF = NumberField(f,"xi")
            xi = FF.gen()
        else:
            FF = C
            xi = -f[0]/f[1]
        ore_ext = ore.change_ring(ore.base_ring().change_ring(FF).fraction_field())
        reloc = ore_ext([c(x=x+xi) for c in self.coefficients(sparse=False)])
        if prec is None:
            sols = reloc.generalized_series_solutions(exp=False)
        else:
            sols = reloc.generalized_series_solutions(prec, exp=False)

        # if any(True for s in sols if s.ramification()>1):
        #     raise NotImplementedError("Some generalized series solutions have ramification")

        if len(sols) < r or any(not s.is_fuchsian(C) for s in sols):
            raise ValueError("The operator has non Fuchsian series solutions")
        
        # Capture the objects
        def get_functions(xi,sols,x,ore_ext):
            # In both functions the second argument `place` is ignored because
            # captured
            def val_fct(op,place,base=C, iota=None):
                op = ore_ext([c(x=x+xi)
                              for c in op.coefficients(sparse=False)])
                vect = [op(s).valuation(base=C,iota=iota) for s in sols]
                return min(vect)
            def raise_val_fct(ops,place,dim=None,base=C,iota=None,
                              infolevel=0):
                # TODO: Is it okay that we don't use dim?
                ops = [ore_ext([c(x=x+xi)
                                for c in op.coefficients(sparse=False)])
                       for op in ops]
                ss = [[op(s) for s in sols] for op in ops]
                if infolevel >= 2: print(ss)
                cands = set()
                r = len(sols)
                for k in range(r):
                    for i in range(len(ops)):
                        for t in ss[i][k].non_integral_terms(
                                base=C,
                                iota=iota,cutoff=1):
                            cands.add(t)

                mtx = [[] for i in range(len(ops))]
                for t in cands:
                    if infolevel >= 2:
                        print(" [raise_val_fct] Processing term x^({}) log(x)^{}".format(t[1],t[0]))
                    for i in range(len(ops)):
                        for s in ss[i]:
                            mtx[i].append(s.coefficient(*t))
                    if infolevel >= 2:
                        print(" [raise_val_fct] Current matrix:\n{}".format(mtx))

                M = matrix(mtx)
                K = M.left_kernel().basis()
                if K:
                    return (1/K[0][-1])*K[0]
                else:
                    return None
            
            return val_fct, raise_val_fct

        val_fct, raise_val_fct = get_functions(xi,sols,x,ore_ext)
        return f,val_fct, raise_val_fct

    def find_candidate_places(self, infolevel=0, iota=None):
        lr = self.coefficients()[-1]
        fact = list(lr.factor())
        places = []
        for f,m in fact:
            places.append(self._make_valuation_place(f,prec=m+1,
                                                     infolevel=infolevel,
                                                     iota=None))
        return places

    def value_function(self, op, place, iota=None):
        val = self._make_valuation_place(place,iota=iota)[1]
        return val(op)

    def raise_value(self, basis, place, dim=None, iota=None):
        fct = self._make_valuation_place(place,iota=iota)[2]
        return fct(basis, place, dim)

    
    
    
#############################################################################################################

class UnivariateRecurrenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
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
            c = self.numerator().coefficients(sparse=False)
            d = self.denominator()

            def fun(n):
                if f[n + r] is None:
                    return None
                else:
                    try:
                        return sum( c[i](n)*f[n + i] for i in range(r + 1) )/d(n)
                    except:
                        return None

            return type(f)(fun(n) for n in range(len(f) - r))

        sigma = self.parent().sigma()
        if not "action" in kwargs:
            x = self.parent().base_ring().gen()
            def shift(p):
                try:
                    return p.subs({x:x+1})
                except:
                    return p(x+1)
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
        R = self.base_ring(); x = R.gen(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {}, {x:one})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_D():
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); x = R.gen()
        alg_theta = alg.change_var_sigma_delta('T', {}, {x:x}).change_ring(R)

        S = alg_theta(~x); out = alg_theta.zero()
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
        R = self.base_ring(); x = R.gen(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:x+one}, {x:one})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_F():
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        delta = alg.gen() + alg.one(); delta_k = alg.one(); R = alg.base_ring()
        c = self.coefficients(sparse=False); out = alg(R(c[0]))

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
        n = ZZ(n); start = ZZ(start); # exact division below fails if n or start are in QQ, as reported by Clemens Hofstadler 2018-03-14.
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

            sage: from sage.all import Matrix, randrange
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
        except:
            raise ValueError("argument has to be of the form u*x+v where u,v are rational")

        if a.degree() > 1:
            raise ValueError("argument has to be of the form u*x+v where u,v are rational")

        try:
            u = QQ(a[1]); v = QQ(a[0])
        except:
            raise ValueError("argument has to be of the form u*x+v where u,v are rational")

        r = self.order(); x = A.base_ring().gen()

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
            s = A.sigma(); v = v.floor()
            L = self.map_coefficients(lambda p: s(p, v))
            return L if u == 1 else L.annihilator_of_composition(u*x)
        elif u == 1:
            return self
        elif u < 0:
            c = [ p(-r - x) for p in self.coefficients(sparse=False) ]; c.reverse()
            return A(c).annihilator_of_composition(-u*x)

        # now a = u*x where u > 1 is an integer. 
        u = u.numerator()
        from sage.matrix.constructor import Matrix
        A = A.change_ring(A.base_ring().fraction_field())
        if solver == None:
            solver = A._solver()
        L = A(self)

        p = A.one(); Su = A.gen()**u # possible improvement: multiplication matrix. 
        mat = [ p.coefficients(sparse=False, padd=r) ]; sol = []

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
        A = self.parent(); A = A.change_ring(A.base_ring().fraction_field())
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
        A = op.parent(); R = A.base_ring(); 
        sigma = A.change_ring(R.change_ring(R.base_ring().fraction_field())).sigma()
        s = set(); r = op.order()

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

        K = R.base_ring(); R0 = R; R = R.change_ring(K.fraction_field()) # FF(k[y])[x]
        A = A.change_ring(R)

        y = R(K.gen()); x = R.gen()

        for (q, _) in R(gcd(R0(p), R0(op[r])))(x - r).resultant(R(op[0])(x + y)).numerator().factor():
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
        degdiff = deg - min(c.degree() for c in coeffs if c!=0)

        solutions = []
        for s, _ in self.newton_polygon(~x):
            if s == 0:
                newcoeffs = [c.shift(w_prec - deg) for c in coeffs ]
            else:
                v = s.denominator(); underflow = int(max(0, -v*r*s))
                newdeg = max([ coeffs[i].degree() + i*s for i in range(len(coeffs)) if coeffs[i] != 0 ])
                newcoeffs = [(coeffs[i](x**v)*subs(x, prec=w_prec + underflow, shift=i, gamma=s))
                             .shift(-v*(newdeg + underflow)) for i in range(len(coeffs))]
            solutions.append( [s, newcoeffs ] )

        if dominant_only:
            max_gamma = max( [g for (g, _) in solutions ] )
            solutions = [s for s in solutions if s[0]==max_gamma]

        info(1, "superexponential parts isolated: " + str([g for g, _ in solutions]))

        # 2. exponential parts
        refined_solutions = []
        for (gamma, coeffs) in solutions:
            info(2, "determining exponential parts for gamma=" + str(gamma))
            deg = max([p.degree() for p in coeffs]); v = gamma.denominator()
            char_poly = K['rho']([ c[deg] for c in coeffs ])
            for (cp, e) in char_poly.factor():
                rho = -cp[0]/cp[1] # K is algebraically closed, so all factors are linear.
                if not rho.is_zero() and (not real_only or rho.imag().is_zero()):
                    info(3, "found rho=" + str(rho))
                    refined_solutions.append([gamma, rho, [coeffs[i]*(rho**i) for i in range(len(coeffs))], e*v])

        if dominant_only:
            max_rho = max( [abs(rho) for (_, rho, _, _) in refined_solutions ] )
            refined_solutions = [s for s in refined_solutions if abs(s[1])==max_rho]

        info(1, "exponential parts isolated: " + str([(gamma, rho) for (gamma, rho, _, _) in refined_solutions]))

        # 3. subexponential parts
        solutions = refined_solutions; refined_solutions = []
        for (gamma, rho, coeffs, ram) in solutions:

            info(2, "determining subexponential parts for (gamma,rho)=" + str((gamma, rho)))

            if ram == 1:
                refined_solutions.append([gamma, rho, [], ram, coeffs])
                continue

            def mysubs(x, prec, shift, subexp, ramification=ram):
                return subs(x, prec, shift, subexp=subexp, ramification=ram)
            
            KK = K['s'].fraction_field(); X = x.change_ring(KK); v = gamma.denominator(); e = ram/v
            cc = [ c(x**e).change_ring(KK) for c in coeffs ]
            subexpvecs = [ [K.zero()]*(ram - 1) ]

            for i in range(ram - 1, 0, -1):
                old = subexpvecs; subexpvecs = []
                for sub in old:
                    sub[i - 1] = KK.gen(); rest = sum((cc[j]*mysubs(X, e, j, sub)) for j in range(r + 1))
                    for (p, _) in rest.leading_coefficient().factor():
                        c = -p[0]/p[1]
                        if not real_only or c.imag().is_zero():
                            vec = [ee for ee in sub]; vec[i - 1] = c; subexpvecs.append(vec)
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
        solutions = refined_solutions; refined_solutions = []
        for (gamma, rho, subexp, ram, coeffs) in solutions:

            info(2, "determining polynomial parts for (gamma,rho,subexp)=" + str((gamma, rho, subexp)))

            KK = K['s'].fraction_field(); s = KK.gen(); X = x.change_ring(KK)
            rest = sum(coeffs[i].change_ring(KK)*subs(X, w_prec, i, alpha=s)(X**ram) for i in range(len(coeffs)))
            for (p, e) in shift_factor(rest.leading_coefficient().numerator(), ram):
                e.reverse()
                alpha = -p[0]/p[1]
                if alpha in QQ: # cause conversion to explicit rational 
                    pass
                if (not real_only or alpha.imag().is_zero()):
                    info(3, "found alpha=" + str(alpha))
                    refined_solutions.append([gamma, rho, subexp, ram, alpha, e, 2*ram*w_prec - rest.degree()])

        info(1, "polynomial parts completed; " + str(len(refined_solutions)) + " solutions separated.")

        # 5. expansion and logarithmic terms
        solutions = refined_solutions; refined_solutions = []
        G = GeneralizedSeriesMonoid(K, x, 'discrete'); prec = n + w_prec
        PS = PowerSeriesRing(K, 'x')

        info(2, "preparing computation of expansion terms...")
        max_log_power = max([sum(b for (_, b) in e[5]) for e in solutions])
        poly_tails = [[x**(ram*prec)]*(ram*prec)]; log_tails = [[x**(ram*prec)]*max_log_power]
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

        for (gamma, rho, subexp, ram, alpha, e, degdrop) in solutions:

            info(2, "determining expansions for (gamma,rho,subexp,alpha)=" + str((gamma, rho, subexp,alpha)))

            underflow = int(max(0, -ram*r*gamma))
            coeffs = [(origcoeffs[i](x**ram)*subs(x, prec + underflow, i, gamma, rho, subexp, ram)).shift(-underflow)\
                          for i in range(r + 1)]
            deg = max([c.degree() for c in coeffs])
            coeffs = [coeffs[i].shift(ram*prec - deg) for i in range(r + 1)]            
            sols = dict( (a, []) for (a, b) in e )

            for (a, b) in e:

                s = alpha - a/ram
                # (n+l)^s/n^s = sum(binom(s,i) (l/n)^i, i=0...)
                spoly_tails = [sum(_binomial(s, i)*(j**i)*(x**(ram*(prec-i))) for i in range(prec)) for j in range(r+1)];

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
                            killer = adjustment[l][ram*prec - k - degdrop]; dl = 0
                            # determine appropriate log power for getting nonzero killer
                            while killer.is_zero():
                                dl += 1
                                adjustment = operator_applied_to_term(k, l + dl)
                                killer = adjustment[l + dl][ram*prec - degdrop - k]
                            # update solution
                            while len(newsol) < l + dl:
                                newsol[-1].append(K.zero())
                                newsol.append([K.zero()]*(k - 1))
                            newcoeff = -tokill/killer; newsol[l + dl].append(newcoeff)
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

        S = self.parent().gen(); n = self.parent().base_ring().gen()
        R = self.base_ring().base_ring().fraction_field()[n]
        coeffs = list(map(R, self.normalize().coefficients(sparse=False)))
        r = self.order()

        # determine the possible values of gamma and phi
        points = list(filter(lambda p: p[1] >= 0, [ (i, coeffs[i].degree()) for i in range(len(coeffs)) ]))
        deg = max(list(map(lambda p: p[1], points)))
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

        TODO: Rephrase
        
        - ``infolevel`` (default: None) - verbosity flag

        OUTPUT:

        A list of places corresponding to the shifted positions associated to
        ``f``.  More precisely, if ``xi`` is a root of ``f``, the places
        correspond to the points ``xi+Nmin, \ldots, xi+Nmax``.

        Each place is a tuple composed of ``f(x+k)``, a suitable function for
        ``value_function`` and a suitable function for ``raise_value``.
        
        EXAMPLES::

        # TODO
        """

        print1 = print if infolevel >= 1 else lambda *a, **k: None
        print2 = print if infolevel >= 2 else lambda *a, **k: None
        print3 = print if infolevel >= 3 else lambda *a, **k: None

        print1(" [make_places] At (root of {}) + Nmin={}, Nmax={}"
               .format(f,Nmin,Nmax))
        
        FF = NumberField(f,"xi")
        # TODO: Do we have to choose a name?
        xi = FF.gen()
        r = self.order() 
        Ore = self.parent()
        SS = Ore.gen()
        Pol = Ore.base_ring()
        nn = Pol.gen()
        Coef = Pol.base_ring()

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

        # TODO: Refactor, not the most efficient
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

        print1(" [make_places] sols")
        print1(sols)

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
            print1(" [make_places] preparing place at {}+{} (min poly = {})"
                   .format(xi,n,f(nn-n)))
            val_fct, raise_val_fct = get_functions(xi,n,Nmin,sols,call)
            res.append((f(nn-n),val_fct,raise_val_fct# , sols, call
            ))
        return res
    
    
    def find_candidate_places(self, Zmax = None, infolevel=0):
        # TODO doc

        # Helpers
        print1 = print if infolevel >= 1 else lambda *a, **k: None
        print2 = print if infolevel >= 2 else lambda *a, **k: None
        print3 = print if infolevel >= 3 else lambda *a, **k: None

        coeffs = self.coefficients(sparse=False)
        
        r = self.order()
        i = min(i for i in range(r+1) if coeffs[i] != 0)
        # Should we replace r with r-i when counting solutions?
        lr = coeffs[-1]
        l0 = coeffs[i]
        l0lr = l0*lr

        # Find the points of interest
        fact0 = list(lr.factor())+list(l0.factor())

        print1("Factors (non unique): {}".format(fact0))
        
        # Cleanup the list
        fact = []
        for f,m in fact0 :
            if f.degree() == 0:
                pass
            elif any(True for i in range(len(fact))
                     if (fact[i][0].degree() == f.degree()
                         and roots_at_integer_distance(fact[i][0],f) != [])):
                # f is a shift of a factor already seen
                fact[i][1] += m
            else:
                fact.append([f,m])

        print1("Factors (unique): {}".format(fact))

        places = []
        for f, m in fact:
            print1("Computing places for {}".format(f))
        
            # Finding the actual indices of interest
            inds = roots_at_integer_distance(l0lr,f)
            print1("Integer distances between roots: {}".format(inds))
            Nmin = min(inds)
            Nmax = max(inds)+r
            Nmin = Nmin - r
            if Zmax :
                Nmax = min(Nmax,Zmax)
                # Else the default max is Nmax
                # TODO: Should we also update Nmin if Zmax < Nmax?
            print1("Nmin={} Nmax={}".format(Nmin,Nmax))

            places += self._make_valuation_places(f, Nmin, Nmax, prec=m+1,
                                                  infolevel=infolevel)
            # TODO: is +1 needed?

        return places

    def value_function(self, op, place):
        val = self._make_valuation_places(place,0,0)[0][1]
        return val(op,place)

    def raise_value(self, basis, place, dim):
        fct = self._make_valuation_places(place,0,0)[0][2]
        return fct(basis, place, dim)
    


#############################################################################################################

class UnivariateQRecurrenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
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

        R = self.parent(); x = R.base_ring().gen(); qx = R.sigma()(x)
        if not "action" in kwargs:
            kwargs["action"] = lambda p : p.subs({x:qx})

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
        R = self.base_ring(); K = R.base_ring(); x, q = self.parent().is_Q(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:q*x}, {x:one})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_J() or \
             alg.base_ring().base_ring() is not K or K(alg.is_J()[1]) != K(q):
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); x, q = alg.is_J()
        alg = alg.change_ring(R);

        Q = alg(~x); out = alg.zero()
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
            u = ZZ(a[1]); v = ZZ(a[0])
        except:
            raise ValueError("argument has to be of the form u*x+v where u,v are rational")

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
            c = [ p(q**(-r)/x) for p in self.coefficients(sparse=False) ]; c.reverse()
            return A(c).numerator().annihilator_of_composition(-u*x)

        # now a = u*x where u > 1 
        from sage.matrix.constructor import Matrix
        if solver == None:
            solver = A._solver()

        p = A.one(); Qu = A.gen()**u # possible improvement: multiplication matrix. 
        mat = [ p.coefficients(sparse=False, padd=r) ]; sol = []

        while len(sol) == 0:

            p = (Qu*p) % L  
            mat.append( p.coefficients(sparse=False, padd=r) )
            sol = solver(Matrix(mat).transpose())

        return self.parent()(list(sol[0])).map_coefficients(lambda p: p(x**u))

    def spread(self, p=0):

        op = self.normalize(); A = op.parent(); R = A.base_ring()
        sigma = A.change_ring(R.change_ring(R.base_ring().fraction_field())).sigma()
        s = []; r = op.order(); _, q = A.is_Q()

        if op.order()==0:
            return []
        elif op[0].is_zero():
            return [infinity]

        if R.is_field():
            R = R.ring() # R = k[x]
            R = R.change_ring(R.base_ring().fraction_field())

        try:
            # first try to use shift factorization. this seems to be more efficient in most cases.
            all_facs = [sigma(u, -1) for u, _ in shift_factor(sigma(op[0].gcd(p), r)*op[r], 1, q)]
            tc = [ u[1:] for _, u in shift_factor(prod(all_facs)*sigma(op[0].gcd(p), r), 1, q) ]
            lc = [ u[1:] for _, u in shift_factor(prod(all_facs)*op[r], 1, q) ]
            for u, v in zip(tc, lc):
                s = union(s, [j[0] - i[0] for i in u for j in v])
            s.sort()
            return s
        except:
            pass

        K = PolynomialRing(R.base_ring(), 'y').fraction_field() # F(k[y])
        R = R.change_ring(K) # FF(k[y])[x]

        y = R(K.gen())
        x, q = op.parent().is_Q()
        x = R(x); q = K(q); 

        s = []; r = op.order()
        for p, _ in (R(op[r])(x*(q**(-r))).resultant(gcd(R(p), R(op[0]))(x*y))).numerator().factor():
            if p.degree() == 1:
                try:
                    s.append(q_log(q, K(-p[0]/p[1])))
                except:
                    pass

        s = list(set(s)) # remove duplicates
        s.sort()
        return s

    spread.__doc__ = UnivariateOreOperatorOverUnivariateRing.spread.__doc__

    def __to_J_literally(self, gen='J'):
        r"""
        Rewrites ``self`` in terms of `J`
        """
        A = self.parent()
        R = A.base_ring(); x, q = A.is_Q(); one = R.one()
        A = A.change_var_sigma_delta(gen, {x:q*x}, {x:one})

        if self.is_zero():
            return A.zero()

        Q = (q - 1)*x*A.gen() + 1; Q_pow = A.one(); 
        c = self.coefficients(sparse=False); out = A(R(c[0]))

        for i in range(self.order()):

            Q_pow *= Q
            out += R(c[i + 1])*Q_pow

        return out

    def _coeff_list_for_indicial_polynomial(self):
        return self.__to_J_literally().coefficients(sparse=False)

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

        return Factorization([(quo*x + rem, 1), (x, -e)])

    def _powerIndicator(self):
        return self.coefficients(sparse=False)[0]

    def _local_data_at_special_points(self):
        r"""
        Returns information about the local behaviour of this operator's solutions at x=0 and
        at x=infinity.

        The output is a list of all tuples `(gamma, phi, beta, alpha)` such that for every
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

        Q = self.parent().gen(); x, qq = self.parent().is_Q()
        factors = make_factor_iterator(x.parent(), multiplicities=False)

        out = []
        for gamma, poly in self.newton_polygon(x):
            if gamma in ZZ:
                for p in factors(poly):
                    if p.degree() == 1:
                        phi = -p[0]/p[1]; L = self.symmetric_product(phi*x**max(-gamma, 0)*Q - x**max(gamma, 0))
                        for beta, qoly in L.newton_polygon(~x):
                            if beta in ZZ:
                                for q in factors(qoly(x*qq**beta) + (qq**beta-1)*qoly[0]): # is this right?
                                    if q.degree() == 1 and q[0] != 0:
                                        out.append((-gamma, phi, beta, -q[0]/q[1]))

        return out

#############################################################################################################

class UnivariateQDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
    Element of an Ore algebra K(x)[J], where J is the Jackson q-differentiation J f(x) = (f(q*x) - f(x))/(q*(x-1))
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        A = self.parent(); x, q = A.is_J(); qx = A.sigma()(x)
        if not "action" in kwargs:
            kwargs["action"] = lambda p : (p.subs({x:qx}) - p)/(x*(q-1))

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
        R = self.base_ring(); K = R.base_ring()
        x, q = self.parent().is_J(); one = R.one()

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:q*x}, {})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_Q() or \
             alg.base_ring().base_ring() is not R.base_ring() or K(alg.is_Q()[1]) != K(q) :
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring().fraction_field(); x, q = alg.is_Q()
        alg = alg.change_ring(R);

        Q = alg.gen(); J = ((q*x - R.one())/(q - R.one()))*Q; J_pow = alg.one()
        out = alg.zero(); 
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

        J = ~(q-1)*(~x)*(alg.gen() - alg.one()); J_k = alg.one(); R = alg.base_ring()
        c = self.coefficients(sparse=False); out = alg(R(c[0]))

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

        A = self.__to_Q_literally(); B = other.__to_Q_literally()

        C = A.symmetric_product(B, solver=solver)._normalize_base_ring()[-1]
        C = C._UnivariateQRecurrenceOperatorOverUnivariateRing__to_J_literally(str(self.parent().gen()))

        try:
            return self.parent()(C.numerator().coefficients(sparse=False))
        except:
            return C

    symmetric_product.__doc__ = UnivariateOreOperator.symmetric_product.__doc__

#############################################################################################################

class UnivariateDifferenceOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
    Element of an Ore algebra K(x)[F], where F is the forward difference operator F f(x) = f(x+1) - f(x)
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        if type(f) in (tuple, list):
            return self.to_S('S')(f, **kwargs)
            
        R = self.parent(); x = R.base_ring().gen(); qx = R.sigma()(x)
        if not "action" in kwargs:
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
        R = self.base_ring(); x = R.gen(); one = R.one(); 

        if type(alg) == str:
            alg = self.parent().change_var_sigma_delta(alg, {x:x+one}, {})
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_S():
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        delta = alg.gen() - alg.one(); delta_k = alg.one(); R = alg.base_ring()
        c = self.coefficients(sparse=False); out = alg(R(c[0]))

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

        A = self.to_S('S'); B = other.to_S(A.parent())
        return A.symmetric_product(B, solver=solver).to_F(self.parent())

    symmetric_product.__doc__ = UnivariateOreOperator.symmetric_product.__doc__

#############################################################################################################

class UnivariateEulerDifferentialOperatorOverUnivariateRing(UnivariateOreOperatorOverUnivariateRing):
    r"""
    Element of an Ore algebra K(x)[T], where T is the Euler differential operator T = x*d/dx
    """

    def __init__(self, parent, *data, **kwargs):
        super(UnivariateOreOperatorOverUnivariateRing, self).__init__(parent, *data, **kwargs)

    def __call__(self, f, **kwargs):

        R = self.parent(); x = R.base_ring().gen(); 
        if not "action" in kwargs:
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

          sage: from ore_algebra import *
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
        elif not isinstance(alg, OreAlgebra_generic) or not alg.is_D():
            raise TypeError("target algebra is not adequate")

        if self.is_zero():
            return alg.zero()

        R = alg.base_ring(); theta = R.gen()*alg.gen(); theta_k = alg.one(); 
        c = self.coefficients(sparse=False); out = alg(R(c[0]))

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

        A = self.to_D('D'); B = other.to_D(A.parent())
        return A.symmetric_product(B, solver=solver).to_T(self.parent())

    symmetric_product.__doc__ = UnivariateOreOperator.symmetric_product.__doc__

#############################################################################################################

def _rec2list(L, init, n, start, append, padd, deform, singularity_handler=None):
    r"""
    Common code for computing terms of holonomic and q-holonomic sequences.
    """
        
    r = L.order(); sigma = L.parent().sigma()
    terms = init if append else list(init)
    K = L.base_ring().base_ring().fraction_field()

    if len(terms) >= n:
        return terms
    
    elif len(terms) < r:

        if not padd:
            raise ValueError("not enough initial values.")
            
        z = K.zero(); padd = r - len(terms)
            
        if append:
            for i in range(padd):
                terms.insert(0, z)
            terms = _rec2list(L, terms, min(n, r) + padd, start - padd, True, False, deform, singularity_handler)
            for i in range(padd):
                terms.remove(0)
        else:
            terms = _rec2list(L, [z]*padd + terms, min(n, r) + padd, start - padd, False, False, deform, singularity_handler)[padd:]

        return _rec2list(L, terms, n, start, append, False, deform, singularity_handler)

    if None in terms:
        for k in range(len(terms), n):
            terms.append(None)
        return terms

    #for i in range(r):
    #    if terms[-i - 1] not in K:
    #        raise TypeError("illegal initial value object")

    rec = L.numerator().coefficients(sparse=False); sigma = L.parent().sigma()
    rec = tuple( -sigma(p, -r) for p in rec )
    lc = -rec[-1]

    for k in range(len(terms), n):

        lck = lc(deform(k + start))
        
        if not lck.is_zero():
            terms.append((~lck)*sum(terms[-r + k + i]*rec[i](deform(k + start)) for i in range(r)))
        elif singularity_handler is None:
            for i in range(k, n):
                terms.append(None)
            return terms
        else:
            terms.append(singularity_handler(k + start))

    return terms
    
def _power_series_solutions(op, rec, n, deform):
    r"""
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
            if (cN.degree()==0): return []
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
                        if P2==0: return [sol]
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

def _tower(dom):
    if is_PolynomialRing(dom) or is_MPolynomialRing(dom):
        base, vars = _tower(dom.base_ring())
        return base, vars.union(set(dom.variable_names()))
    elif isinstance(dom, FractionField_generic):
        return _tower(dom.ring())
    else:
        return dom, set()
