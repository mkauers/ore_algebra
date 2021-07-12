"""
Guessing tools

TESTS::

    sage: from ore_algebra import OreAlgebra, guess
    sage: guess([SR(1/(i+1)) for i in range(10)], OreAlgebra(QQ['n'], 'Sn'))
    (-n - 2)*Sn + n + 1
"""

#############################################################################
#  Copyright (C) 2013, 2014                                                 #
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

######### development mode ###########
"""
try:
    if 'ore_algebra' in sys.modules:
        del sys.modules['ore_algebra']
except:
    pass
"""
#######################################

import math
from datetime import datetime

from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.finite_rings.all import GF
from sage.rings.finite_rings.finite_field_base import is_FiniteField
from sage.matrix.constructor import Matrix, matrix
from sage.matrix.matrix_space import MatrixSpace
from sage.misc.lazy_string import lazy_string
from sage.arith.misc import xgcd
from sage.parallel.decorate import parallel
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.modules.free_module_element import vector
from sage.sets.primes import Primes

from . import nullspace
from .nullspace import _hermite
from .ore_algebra import OreAlgebra

def guess_rec(data, n, S, **kwargs):
    """
    Shortcut for ``guess`` applied with an Ore algebra of shift operators in `S` over `K[n]`
    where `K` is the parent of ``data[0]``.

    See the docstring of ``guess`` for further information.    
    """
    R = data[0].parent()[n]; x = R.gen()
    return guess(data, OreAlgebra(R, (S, {n:n+R.one()}, {})), **kwargs)

def guess_deq(data, x, D, **kwargs):
    """
    Shortcut for ``guess`` applied with an Ore algebra of differential operators in `D` over `K[x]`
    where `K` is the parent of ``data[0]``.

    See the docstring of ``guess`` for further information.    
    """
    R = data[0].parent()[x]; x = R.gen()
    return guess(data, OreAlgebra(R, (D, {}, {x:R.one()})), **kwargs)

def guess_qrec(data, qn, Q, q, **kwargs):
    """
    Shortcut for ``guess`` applied with an Ore algebra of `q`-recurrence operators in `Q` over `K[qn]`
    where `K` is the parent of `q`.

    See the docstring of ``guess`` for further information.    
    """
    R = q.parent()[qn]; x = R.gen()
    return guess(data, OreAlgebra(R, (Q, {qn:q*qn}, {qn:R.one()})), **kwargs)

def guess(data, algebra, **kwargs):
    """
    Searches for an element of the algebra which annihilates the given data.

    INPUT:

    - ``data`` -- a list of elements of the algebra's base ring's base ring `K` (or at least
      of objects which can be casted into this ring). If ``data`` is a string, it is assumed
      to be the name of a text file which contains the terms, one per line, encoded in a way
      that can be interpreted by the element constructor of `K`. 
    - ``algebra`` -- a univariate Ore algebra over a univariate polynomial ring whose
      generator is the standard derivation, the standard shift, the forward difference,
      a q-shift, or a commutative variable. 

    Optional arguments:

    - ``cut`` -- if `N` is the minimum number of terms needed for some particular
      choice of order and degree, and if ``len(data)`` is more than ``N+cut``,
      use ``data[:N+cut]`` instead of ``data``. This must be a nonnegative integer
      or ``None``. Default: ``None``.
    - ``ensure`` -- if `N` is the minimum number of terms needed for some particular
      choice of order and degree, and if ``len(data)`` is less than ``N+ensure``,
      raise an error. This must be a nonnegative integer. Default: 0.
    - ``ncpus`` -- number of processors to be used. Default: 1.
    - ``order`` -- bounds the order of the operators being searched for.
      Default: infinity.
    - ``min_order`` -- smallest order to be considered in the search. The output
      may nevertheless have lower order than this bound. Default: 1
    - ``degree`` -- bounds the degree of the operators being searched for.
      The method may decide to overrule this setting if it thinks this may speed up
      the calculation. Default: infinity.
    - ``min_degree`` -- smallest degree to be considered in the search. The output
      may nevertheless have lower degree than this bound. Default: 0
    - ``path`` -- a list of pairs `(r, d)` specifying which orders and degrees
      the method should attempt. If this value is equal to ``None`` (default), a
      path is chosen which examines all the `(r, d)` which can be tested with the
      given amount of data. 
    - ``solver`` -- function to be used for computing the right kernel of a matrix
      with elements in `K`. 
    - ``infolevel`` -- an integer specifying the level of details of progress
      reports during the calculation. 
    - ``method`` -- either "linalg" (for linear algebra) or "hp" (for Hermite-Pade) or "automatic" 
      (for the default choice), or a callable with the specification of a raw guesser.

    OUTPUT:

    - An element of ``algebra`` which annihilates the given ``data``.

    An error is raised if no such element is found. 

    .. NOTE::

        - This method is designed to find equations for D-finite objects. It
          may exhibit strange behaviour for objects which are holonomic but not
          D-finite. 
        - When the generator of the algebra is a commutative variable, the
          method searches for algebraic equations.

    EXAMPLES::

      sage: from ore_algebra import *
      sage: rec = guess([(2*i+1)^15 * (1 + 2^i + 3^i)^2 for i in range(1000)], OreAlgebra(ZZ['n'], 'Sn')) # long time (2.9 s)
      sage: rec.order(), rec.degree() # long time
      (6, 90)
      sage: R.<t> = QQ['t']
      sage: rec = guess([1/(i+t) + t^i for i in range(100)], OreAlgebra(R['n'], 'Sn'))
      sage: rec
      ((-t + 1)*n^2 + (-2*t^2 - t + 2)*n - t^3 - 2*t^2)*Sn^2 + ((t^2 - 1)*n^2 + (2*t^3 + 3*t^2 - 2*t - 1)*n + t^4 + 3*t^3 + t^2 - t)*Sn + (-t^2 + t)*n^2 + (-2*t^3 + t)*n - t^4 - t^3 + t^2

      sage: R.<C> = OreAlgebra(ZZ['x'])
      sage: cat = [binomial(2*n,n) // (n+1) for n in range(10)]
      sage: guess(cat, R)
      -x*C^2 + C - 1
    """

    A = algebra; R = A.base_ring(); K = R.base_ring(); x = R.gen()

    if A.ngens() > 1 or R.ngens() > 1:
        return guess_mult(data, algebra, **kwargs)
    
    if type(data) == str:
        with open(data, 'r') as f:
            data = [ K(line) for line in f ]

    if (data[0] == 0 or data[1] == 0) and (A.is_C() or A.is_S()):
        
        if all( d == 0 for d in data ):
            return A.one()
        for i in range(len(data)):
            if data[i] != 0:
                a = i
                break
        m = None; b = ZZ(a)
        for i in range(b + 1, len(data)):
            if data[i] != 0:
                m = i - b if m is None else m.gcd(i - b)
                b = i
                if m.is_one():
                    break

        if m is not None and m > 1:
            a = a % m
            if 'infolevel' in kwargs and kwargs['infolevel'] >= 1:
                print("Recognized that only 1 out of " + str(m) + " terms is nonzero; removing zeros...")
            eq = guess([data[m*k + a] for k in range(((len(data) - a)/m).floor())], A, **kwargs)
            if 'infolevel' in kwargs and kwargs['infolevel'] >= 1:
                print("Adjusting equation to restore deleted zeros...")
            x = R.gen()
            if A.is_S():
                ops = [A.one()]*m; ops[a] = eq
                return ops[0].annihilator_of_interlacing(*(ops[1:]))
            else:
                eq = eq.polynomial().map_coefficients(lambda p: p(x**m))
                if a != 0:
                    eq = eq(x**(-a)*eq.parent().gen())*x**(a*eq.degree())
                return A(eq)

    def to_A(obj):
        if isinstance(obj, tuple):
            return (A(obj[0]), *obj[1:])
        else:
            return A(obj)

    if R.is_field():
        return to_A(guess(data, A.change_ring(R.ring()), **kwargs))
                
    elif A.is_F() is not False:
        # reduce to shift case; note that this does not alter order or degrees
        if 'infolevel' in kwargs and kwargs['infolevel'] >= 1:
            print("Translating problem to shift case...")
        A0 = OreAlgebra(R, ('S', {x:x+K.one()}, {}))
        return guess(data, A0, **kwargs).change_ring(R).to_F(A)

    elif (not A.is_S() and not A.is_D() and not A.is_Q() and not A.is_C()):
        raise TypeError("unexpected algebra: " + str(A))

    elif K.is_prime_field() and K.characteristic() > 0:
        return _guess_via_gcrd(data, A, **kwargs)

    elif K is ZZ:
        # CRA
        return _guess_via_hom(data, A, _word_size_primes(), lambda mod : GF(mod), **kwargs)

    elif is_PolynomialRing(K) and K.base_ring().is_prime_field() and K.characteristic() > 0:  # K == GF(p)[t]
        # eval/interpol
        mod = _linear_polys(K.gen(), 7, K.characteristic())
        to_hom = lambda mod : (lambda pol : pol(-mod[0]))
        return _guess_via_hom(data, A, mod, to_hom, **kwargs)

    elif is_PolynomialRing(K) and K.base_ring() is ZZ:  # K == ZZ[t]
        # CRA + eval/interpol

        KK = QQ[K.gens()].fraction_field() ## all elements of 'data' must be coercible to KK
        KK2 = ZZ[K.gens()].fraction_field() ## rewrite them as elements of KK2

        def cleanup(rat):
            rat = KK(rat)
            n, d = rat.numerator(), rat.denominator() # live in QQ[t]
            nn, nd = n.numerator(), n.denominator()
            dn, dd = d.numerator(), d.denominator()
            return KK2(K(nn*dd)/K(nd*dn))

        data = list(map(cleanup, data))

        def to_hom(mod):
            KK3 = GF(mod); KK4 = KK3[K.gens()]; KK5 = KK4.fraction_field()
            return lambda rat: KK5(KK4(rat.numerator()).map_coefficients(KK3, KK3) / \
                                   KK4(rat.denominator()).map_coefficients(KK3, KK3))

        return _guess_via_hom(data, A, _word_size_primes(), to_hom, **kwargs)

    elif K is QQ:
        return to_A(guess(data, A.change_ring(ZZ[x]), **kwargs))

    elif K.is_field():
        return to_A(guess(data, A.change_ring(K.ring()[x]), **kwargs))

    elif is_PolynomialRing(K) and K.base_ring() is QQ:
        return to_A(guess(data, A.change_ring(ZZ[K.gens()][x]), **kwargs))

    else:
        raise TypeError("unexpected coefficient domain: " + str(K))

###########################################################################################

def guess_raw(data, A, order=-1, degree=-1, lift=None, solver=None, cut=25, ensure=0, infolevel=0):
    """
    Guesses recurrence or differential equations for a given sample of terms.

    INPUT:

    - ``data`` -- list of terms
    - ``A`` -- an Ore algebra of recurrence operators, differential operators,
      or q-differential operators. 
    - ``order`` -- maximum order of the sought operators
    - ``degree`` -- maximum degree of the sought operators
    - ``lift`` (optional) -- a function to be applied to the terms in ``data``
      prior to computation
    - ``solver`` (optional) -- a function to be used to compute the nullspace
      of a matrix with entries in the base ring of the base ring of ``A``
    - ``cut`` (optional) -- if `N` is the minimum number of terms needed for
      the the specified order and degree and ``len(data)`` is more than ``N+cut``,
      use ``data[:N+cut]`` instead of ``data``. This must be a nonnegative integer
      or ``None``.
    - ``ensure`` (optional) -- if `N` is the minimum number of terms needed
      for the specified order and degree and ``len(data)`` is less than ``N+ensure``,
      raise an error. This must be a nonnegative integer.
    - ``infolevel`` (optional) -- an integer indicating the desired amount of
      progress report to be printed during the calculation. Default: 0 (no output).

    OUTPUT:

    A basis of the ``K``-vector space of all the operators `L` in ``A`` of order
    at most ``order`` and degree at most ``degree`` such that `L` applied to
    ``data`` gives an array of zeros. (resp. `L` applied to the truncated power
    series with ``data`` as terms gives the zero power series) 

    An error is raised in the following situations:

    * the algebra ``A`` has more than one generator, or its unique generator
      is neither a standard shift nor a q-shift nor a standard derivation.
    * ``data`` contains some item which does not belong to ``K``, even after
      application of ``lift``
    * if the condition on ``ensure`` is violated. 
    * if the linear system constructed by the method turns out to be
      underdetermined for some other reason, e.g., because too many linear
      constraints happen to be trivial.

    ALGORITHM:

    Ansatz and linear algebra.

    .. NOTE::

      This is a low-level method. Don't call it directly unless you know what you
      are doing. In usual applications, the right method to call is ``guess``.

    EXAMPLES::

      sage: from ore_algebra import *
      sage: K = GF(1091); R.<n> = K['n']; A = OreAlgebra(R, 'Sn')
      sage: data = [(5*n+3)/(3*n+4)*fibonacci(n)^3 for n in range(200)]
      sage: guess_raw(data, A, order=5, degree=3, lift=K)
      [(n^3 + 546*n^2 + 588*n + 786)*Sn^5 + (356*n^3 + 717*n^2 + 381*n + 449)*Sn^4 + (8*n^3 + 569*n^2 + 360*n + 214)*Sn^3 + (31*n^3 + 600*n^2 + 784*n + 287)*Sn^2 + (1078*n^3 + 1065*n^2 + 383*n + 466)*Sn + 359*n^3 + 173*n^2 + 503, (n^3 + 1013*n^2 + 593*n + 754)*Sn^5 + (797*n^3 + 56*n^2 + 7*n + 999)*Sn^4 + (867*n^3 + 1002*n^2 + 655*n + 506)*Sn^3 + (658*n^3 + 834*n^2 + 1036*n + 899)*Sn^2 + (219*n^3 + 479*n^2 + 476*n + 800)*Sn + 800*n^3 + 913*n^2 + 280*n]
    
    """

    if min(order, degree) < 0:
        return [] 

    R = A.base_ring(); K = R.base_ring(); q = A.is_Q()

    def info(bound, msg):
        if bound <= infolevel:
            print(msg)

    info(1, lazy_string(lambda: datetime.today().ctime() + ": raw guessing started."))
    info(1, "len(data)=" + str(len(data)) + ", algebra=" + str(A._latex_()))

    if A.ngens() > 1 or (not A.is_S() and not A.is_Q() and not A.is_D() and not A.is_C()):
        raise TypeError("unexpected algebra")

    alg_case = True if A.is_C() else False
    diff_case = True if A.is_D() else False
    deform = (lambda n: q[1]**n) if q is not False else (lambda n: n)
    min_len_data = (order + 1)*(degree + 2 - (1 if alg_case else 0))

    if cut is not None and len(data) > min_len_data + cut:
        data = data[:min_len_data + cut]

    if len(data) < min_len_data + ensure:
        raise ValueError("not enough terms")

    if lift is not None:
        data = list(map(lift, data))

    if not all(p in K for p in data):
        raise ValueError("illegal term in data list")

    if solver is None:
        solver = A._solver(K)
        
    if solver is None:
        solver = nullspace.sage_native

    sys = {(0,0):data}
    nn = [deform(n) for n in range(len(data))]
    z = [K.zero()]

    if alg_case:
        # sys[i, j] contains x^i * series(data)^j 
        series = R(data)
        sys[0, 0] = R.one()
        for j in range(order):
            sys[0, j + 1] = (sys[0, j]*series).truncate(len(data))
        for j in range(order + 1):
            sys[0, j] = sys[0, j].padded_list(len(data))
            for i in range(degree):
                sys[i + 1, j] = z + sys[i, j]
    elif diff_case:
        # sys[i, j] contains ( x^i * D^j ) (data)
        nn = nn[1:]
        for j in range(order):
            sys[0, j + 1] = list(map(lambda a,b: a*b, sys[0, j][1:], nn))
            nn.pop(); 
        for i in range(degree):
            for j in range(order + 1):
                sys[i + 1, j] = z + sys[i, j]
    else:
        # sys[i, j] contains ( (n+j)^i * S^j ) (data)
        for i in range(degree):
            sys[i + 1, 0] = list(map(lambda a,b: a*b, sys[i, 0], nn))
        for j in range(order):
            for i in range(degree + 1):
                sys[i, j + 1] = sys[i, j][1:]

    sys = [sys[i, j] for j in range(order + 1) for i in range(degree + 1) ]

    trim = min(len(c) for c in sys)
    for i in range(len(sys)):
        if len(sys[i]) > trim:
            sys[i] = sys[i][:trim]

    sys = matrix(K, zip(*sys))

    dims = sys.dimensions()
    info(2, lazy_string(lambda: datetime.today().ctime() + ": matrix construction completed. size=" + str(dims)))
    sol = solver(sys, infolevel=infolevel - 2)
    del sys 
    info(2, lazy_string(lambda: datetime.today().ctime() + ": nullspace computation completed. size=" + str(len(sol))))

    sigma = A.sigma()
    for l in range(len(sol)):
        c = []; s = list(sol[l])
        for j in range(order + 1):
            c.append(sigma(R(s[j*(degree + 1):(j + 1)*(degree + 1)]), j))
        sol[l] = A(c)
        sol[l] *= ~sol[l].leading_coefficient().leading_coefficient()

    return sol

###########################################################################################

def guess_hp(data, A, order=-1, degree=-1, lift=None, cut=25, ensure=0, infolevel=0):
    """
    Guesses differential equations or algebraic equations for a given sample of terms.

    INPUT:

    - ``data`` -- list of terms
    - ``A`` -- an Ore algebra of differential operators or ordinary polynomials. 
    - ``order`` -- maximum order of the sought operators
    - ``degree`` -- maximum degree of the sought operators
    - ``lift`` (optional) -- a function to be applied to the terms in ``data``
      prior to computation
    - ``cut`` (optional) -- if `N` is the minimum number of terms needed for
      the the specified order and degree and ``len(data)`` is more than ``N+cut``,
      use ``data[:N+cut]`` instead of ``data``. This must be a nonnegative integer
      or ``None``.
    - ``ensure`` (optional) -- if `N` is the minimum number of terms needed
      for the specified order and degree and ``len(data)`` is less than ``N+ensure``,
      raise an error. This must be a nonnegative integer.
    - ``infolevel`` (optional) -- an integer indicating the desired amount of
      progress report to be printed during the calculation. Default: 0 (no output).

    OUTPUT:

    A basis of the ``K``-vector space of all the operators `L` in ``A`` of order
    at most ``order`` and degree at most ``degree`` such that `L` applied to
    the truncated power series with ``data`` as terms gives the zero power series.

    An error is raised in the following situations:

    * the algebra ``A`` has more than one generator, or its unique generator
      is neither a standard derivation nor a commutative variable. 
    * ``data`` contains some item which does not belong to ``K``, even after
      application of ``lift``
    * if the condition on ``ensure`` is violated. 

    ALGORITHM:

    Hermite-Pade approximation.

    .. NOTE::

      This is a low-level method. Don't call it directly unless you know what you
      are doing. In usual applications, the right method to call is ``guess``.

    EXAMPLES::

      sage: from ore_algebra import *
      sage: from ore_algebra.guessing import guess_hp
      sage: K = GF(1091); R.<x> = K['x'];
      sage: data = [binomial(2*n, n)*fibonacci(n)^3 for n in range(2000)]
      sage: guess_hp(data, OreAlgebra(R, 'Dx'), order=4, degree=4, lift=K)
      [(x^4 + 819*x^3 + 136*x^2 + 17*x + 635)*Dx^4 + (14*x^3 + 417*x^2 + 952*x + 605)*Dx^3 + (598*x^2 + 497*x + 99)*Dx^2 + (598*x + 794)*Dx + 893]
      sage: len(guess_hp(data, OreAlgebra(R, 'C'), order=16, degree=64, lift=K))
      1
    """

    if min(order, degree) < 0:
        return [] 

    R = A.base_ring(); K = R.base_ring()

    def info(bound, msg):
        if bound <= infolevel:
            print(msg)

    info(1, lazy_string(lambda: datetime.today().ctime() + ": Hermite/Pade guessing started."))
    info(1, "len(data)=" + str(len(data)) + ", algebra=" + str(A._latex_()))

    if A.ngens() > 1 or (not A.is_C() and not A.is_D() ):
        raise TypeError("unexpected algebra")

    diff_case = True if A.is_D() else False
    min_len_data = (order + 1)*(degree + 2)

    if cut is not None and len(data) > min_len_data + cut:
        data = data[:min_len_data + cut]

    if len(data) < min_len_data + ensure:
        raise ValueError("not enough terms")

    if lift is not None:
        data = list(map(lift, data))

    if not all(p in K for p in data):
        raise ValueError("illegal term in data list")

    if diff_case:
        series = [R(data)]
        for i in range(order):
            series.append(series[-1].derivative())
        truncate = len(data) - order 
        series = [s.truncate(truncate) for s in series]
    else:
        truncate = len(data)
        series = [R.one(), R(data)]
        for i in range(order - 1):
            series.append((series[1]*series[-1]).truncate(truncate))

    info(2, lazy_string(lambda: datetime.today().ctime() + ": matrix construction completed."))
    sol = _hermite(True, matrix(R, [series]), [degree], infolevel - 2, truncate = truncate - 1)
    info(2, lazy_string(lambda: datetime.today().ctime() + ": hermite pade approximation completed."))

    sol = [A(list(map(R, s))) for s in sol]
    sol = [(~L.leading_coefficient().leading_coefficient())*L for L in sol]

    return sol    

###########################################################################################

def _guess_via_hom(data, A, modulus, to_hom, **kwargs):
    """
    Implementation of guessing via homomorphic images.

    INPUT:

    - ``data``: list of terms
    - ``A``: an algebra of the form K[x][X]
    - ``modulus``: an iterator which produces appropriate moduli
    - ``to_hom``: a callable which turns a given modulus to a map from K to some hom image domain

    OUTPUT:

    - ``L`` in ``A``, the guessed operator.
    - if the option ``return_short_path`` is given and ``True``, return the pair ``(L, path)``.
    
    Covers three cases:

    1. K == ZZ ---> GF(p) and back via CRA.
       In this case, ``modulus`` is expected to iterate over primes `p`
    2. K == GF(p)[t] --> GF(p) and back via interpolation.
       In this case, ``modulus`` is expected to iterate over linear polynomial in `K`
    3. K == ZZ[t] --> GF(p)[t] and back via CRA.
       In this case, ``modulus`` is expected to iterate over primes `p`. The method
       produces problem instances of type 2, which are handled recursively.     

    """

    if 'infolevel' in kwargs:
        infolevel = kwargs['infolevel']
        kwargs['infolevel'] = infolevel - 2
    else:
        infolevel = 0
        
    def info(bound, msg):
        if bound <= infolevel:
            print(msg)

    R = A.base_ring(); x = R.gen(); K = R.base_ring(); 
    atomic = not ( is_PolynomialRing(K) and K.base_ring() is ZZ )

    info(1, lazy_string(lambda: datetime.today().ctime() + ": guessing via homomorphic images started."))
    info(1, "len(data)=" + str(len(data)) + ", algebra=" + str(A._latex_()))

    L = A.zero()
    mod = K.one() if atomic else ZZ.one()
    order_adjustment = None

    nn = 0; path = []; ncpus = 1
    return_short_path = 'return_short_path' in kwargs and kwargs['return_short_path'] is True

    def op2vec(L, r, d):
        # convert an operator L of order <=r and degree <=d to a vector of dimension (r+1)*(d+1).
        c = []
        for i in range(r + 1):
            p = L[i]
            for j in range(d + 1):
                c.append(p[j])
        return vector(K, c)

    def vec2op(v, r, d):
        # convert a vector of dimension (r+1)*(d+1) into an operator of order <=r and degree <=d.
        c = []
        for i in range(r + 1):
            c.append(R([v[(d + 1)*i + j] for j in range(d + 1)]))
        return A(c)

    while mod != 0:

        nn += 1 # iteration counter

        if nn == 1:
            # 1st iteration: use the path specified by the user (or a default path)
            kwargs['return_short_path'] = True

        elif nn == 2 and atomic and path[0][0] >= Lp.order() + 2:
            # 2nd iteration: try to optimize the path obtained in the 1st iteration 
            r0 = Lp.order(); d0 = Lp.degree(); r1, d1 = path[0]
            # determine the hyperbola through (r0,d0) and (r1,d1) and 
            # choose (r2,d2) as the point on this hyperbola for which (r2+1)*(d2+1) is minimized
            try: 
                r2 = r0 - 1 + math.sqrt(abs((d0-d1)*r0*(r0-1.-r1)/(d0+r0+d1*(r0-1.-r1)-r1)))
                d2 = (d1*(r0-1-r1)*(r0-r2) + d0*(r1-r2))/((r0-r1)*(r0-1-r2))
                r2 = int(math.ceil(r2)); d2 = int(math.ceil(d2))
                if abs(r2 - r1) >= 2 and abs(d2 - d1) >= 2:
                    path = [ (i, d2 + ((d1-d2)*(i-r2))//(r1-r2)) for i in range(r2, r1, 1 if r1 >= r2 else -1) ] + path
                    kwargs['path'] = path
                else:
                    del kwargs['return_short_path']
            except:
                del kwargs['return_short_path']

            if A.is_C():
                kwargs['path'] = [(Lp.order(), Lp.degree())] # there is no curve for algebraic equations

        elif 'return_short_path' in kwargs:
            # subsequent iterations: stick to the path we have.                 
            del kwargs['return_short_path']

        if not 'path' in kwargs:
            kwargs['return_short_path'] = True

        if ncpus == 1:
            # sequential version 

            imgs = [];
            for i in range(max(1, nn - 3)): # do several imgs before proceeding with a reconstruction attempt
                
                data_mod = None
                while data_mod is None:
                    p = next(modulus); hom = to_hom(p)
                    info(2, "modulus = " + str(p))
                    try:
                        data_mod = list(map(hom, data))
                    except ArithmeticError:
                        info(2, "unlucky modulus discarded.")

                qq = A.is_Q()
                if not qq:
                    Lp = guess(data_mod, A.change_ring(hom(K.one()).parent()[x]), **kwargs)
                else:
                    qq = hom(qq[1])
                    Lp = guess(data_mod, OreAlgebra(hom(K.one()).parent()[x], (A.var(), {x:qq*x}, {}), q=qq), **kwargs)

                if type(Lp) is tuple and len(Lp) == 2:  ## this implies nn < 3  
                    Lp, path = Lp
                    kwargs['path'] = path

                imgs.append((Lp, p))

            if len(imgs) == 1:
                Lp, p = imgs[0]
                r = Lp.order(); d = Lp.degree()
            else:
                Lp = A.zero(); p = K.one()
                for (Lpp, pp) in imgs:
                    try:
                        Lp, p = _merge_homomorphic_images(op2vec(Lp, r, d), p, op2vec(Lpp, r, d), pp, reconstruct=False)
                        Lp = vec2op(Lp, r, d)
                    except:
                        info(2, "unlucky modulus " + str(pp) + " discarded")

        else:
            # we can assume at this point that nn >= 3 and 'return_short_path' is switched off.
            primes = [next(modulus) for i in range(ncpus)]
            info(2, "moduli = " + str(primes))
            primes = [ (p, to_hom(p)) for p in primes ]
            primes = [ (p, hom, A.change_ring(hom(K.one()).parent()[x])) for (p, hom) in primes ]
            Lp = A.zero(); p = K.one()
            out = [ (arg[0][0], arg[0][2], Lpp) for (arg, Lpp) in forked_guess(primes) ]
            for (pp, alg, Lpp) in out:
                Lpp = alg(Lpp)
                try:
                    Lp, p = _merge_homomorphic_images(op2vec(Lp, r, d), p, vec2op(Lpp, r, d), pp, reconstruct=False)
                    Lp = vec2op(Lp, r, d)
                except:
                    info(2, "unlucky modulus " + str(pp) + " discarded")

        if nn == 1:
            r = Lp.order(); d = Lp.degree()
            info(2, "solution of order " + str(r) + " and degree " + str(d) + " predicted")

        elif nn == 2 and 'ncpus' in kwargs and kwargs['ncpus'] > 1:
            info(2, "Switching to multiprocessor code.")
            ncpus = kwargs['ncpus']
            del kwargs['ncpus']
            kwargs['infolevel'] = 0
            
            @parallel(ncpus=ncpus)
            def forked_guess(p, hom, alg):
                try:
                    return guess(list(map(hom, data)), alg, **kwargs).polynomial()
                except ArithmeticError:
                    return None
                
        elif nn == 3 and 'infolevel' in kwargs:
            kwargs['infolevel'] = kwargs['infolevel'] - 2

        if not Lp.is_zero():
            info(2, "Reconstruction attempt...")
            s = Lp.parent().sigma()
            if not s.is_identity() and mod.parent() is ZZ:
                try:
                    if order_adjustment is None:
                        order_adjustment = Lp.order() // ZZ(2)
                    Lp = Lp.map_coefficients(lambda p: s(p, -order_adjustment))
                except:
                    L = A.zero(); mod = K.one() if atomic else ZZ.one(); order_adjustment = 0

            L, mod = _merge_homomorphic_images(op2vec(L, r, d), mod, op2vec(Lp, r, d), p)
            L = vec2op(L, r, d)

    if order_adjustment:
        s = L.parent().sigma()
        L = L.map_coefficients(lambda p: s(p, order_adjustment))

    return (L, path) if return_short_path else L

###########################################################################################

def _guess_via_gcrd(data, A, **kwargs):
    """
    Implementation of guessing by taking gcrd of small equations. 

    INPUT:

    - ``data``: list of terms
    - ``A``: an algebra of the form GF(p)[x][X]

    OUTPUT:

    - ``L`` in ``A``, the guessed operator.

    raises an error if no equation is found.
    """

    if 'infolevel' in kwargs:
        infolevel = kwargs['infolevel']
        kwargs['infolevel'] = infolevel - 2
    else:
        infolevel = 0
        
    def info(bound, msg):
        if bound <= infolevel:
            print(msg)

    R = A.base_ring(); x = R.gen(); K = R.base_ring(); 

    info(1, lazy_string(lambda: datetime.today().ctime() + ": guessing via gcrd started."))
    info(1, "len(data)=" + str(len(data)) + ", algebra=" + str(A._latex_()))

    if 'ncpus' in kwargs:
        del kwargs['ncpus']

    if 'return_short_path' in kwargs:
        return_short_path = True
        del kwargs['return_short_path']
    else:
        return_short_path = False
        
    ensure = kwargs['ensure'] if 'ensure' in kwargs else 0

    N = len(data) - ensure
    
    if 'path' in kwargs:
        path = kwargs['path']; del kwargs['path']
        sort_key = lambda p: (p[0] + 1)*(p[1] + 1)
        prelude = []
    else:
        r2d = lambda r: (N - 2*r - 2)//(r + 1) # python integer division intended.
        path = [(r, r2d(r)) for r in range(1, N)] 
        path = [p for p in path if min(p[0] - 1, p[1]) >= 0]
        (r, d) = (1, 1); prelude = []
        while d <= r2d(r):
            prelude.append((r, d))
            (r, d) = (d, r + d)
        path = prelude + path
        sort_key = lambda p: 2*p[0] + (p[0] + 1)*(p[1] + 1) # give some preference to small orders

    max_deg = max_ord = len(data); min_deg = 0; min_ord = 1;

    if 'degree' in kwargs:
        max_deg = kwargs['degree']; del kwargs['degree']
    elif 'max_degree' in kwargs:
        max_deg = kwargs['max_degree']; del kwargs['max_degree']

    if 'min_degree' in kwargs:
        min_deg = kwargs['min_degree']; del kwargs['min_degree']

    if 'order' in kwargs:
        max_ord = kwargs['order']; del kwargs['order']
    elif 'max_order' in kwargs:
        max_ord = kwargs['max_order']; del kwargs['max_order']

    if 'min_order' in kwargs:
        min_ord = kwargs['min_order']; del kwargs['min_order']

    subguesser = guess_hp if A.is_C() else guess_raw # default = hp for algeqs and raw for other
    if 'method' in kwargs:
        if kwargs['method'] == 'linalg':
            subguesser = guess_raw
        elif kwargs['method'] == 'hp':
            subguesser = guess_hp
        elif kwargs['method'] == 'automatic' or kwargs['method'] == 'default':
            pass # same as when no method is specified
        else:
            subguesser = kwargs['method'] # callable
        del kwargs['method']
        
    path = [p for p in path if min_ord <= p[0] and p[0] <= max_ord and min_deg <= p[1] and p[1] <= max_deg]

    path.sort(key=sort_key)
    # autoreduce
    for i in range(len(prelude), len(path)):
        (r, d) = path[i]
        for j in range(len(path)):
            if i != j and path[j] is not None and path[j][0] >= r and path[j][1] >= d:
                path[i] = None                    
    path = [p for p in path if p is not None]

    if 'max_path_length' in kwargs:
        b = kwargs['max_path_length']
        if b > len(path):
            path = path[:b]
    
    info(2, "Going through a path with " + str(len(path)) + " points")

    # search equation

    neg_probes = []
    def probe(r, d):
        if (r, d) in neg_probes:
            return []        
        kwargs['order'], kwargs['degree'] = r, d
        sols = subguesser(data, A, **kwargs)
        info(2, str(len(sols)) + " sols for (r, d)=" + str((r, d)))
        if len(sols) == 0:
            neg_probes.append((r, d))
        return sols

    L = []; short_path = []; 
    
    for i in range(len(path)):

        r, d = path[i]
        for (r1, d1) in short_path:
            if r >= r1:
                d = min(d, d1 - 1)

        if d < 0:
            continue

        sols = probe(r, d)
        
        while return_short_path and d > 0 and len(sols) > 1:
            new = probe(r, d - 1)
            if len(new) == 0:
                break
            m = len(sols) - len(new) 
            if m == 0:
                # assuming subsolver returned minimal degrees (as does, e.g., a h/p solver)
                d = max(p.degree() for p in sols)
                break
            d2 = max(int(math.ceil(d - len(sols)*1.0/m)), 0)
            sols = probe(r, d2) if d2 < d - 1 else new
            d = d2
            if len(sols) == 0:
                while len(sols) == 0:
                    d += 1; sols = probe(r, d)
                break

        if len(sols) > 0:
            short_path.append((r, d))
            L = L + sols
        if len(L) >= 2:
            break

    info(2, lazy_string(lambda: datetime.today().ctime() + ": search completed."))

    if len(L) == 0:
        raise ValueError("No relations found.")
    elif len(L) == 1:
        L = L[0]
    else:
        L = L[0].gcrd(L[1])
        info(2, lazy_string(lambda: datetime.today().ctime() + ": gcrd completed."))

    L = (~L.leading_coefficient().leading_coefficient())*L

    return (L, short_path) if return_short_path else L

###########################################################################################

from sage.arith.multi_modular import MAX_MODULUS
from sage.arith.all import previous_prime as pp

def _word_size_primes(init=2**23, bound=1000):
    """
    returns an iterator which enumerates the primes smaller than ``init`` and bigger than ``bound``,
    in decreasing order. 
    """
    p = pp(init)
    while p > bound:
        yield p
        p = pp(p)

def _linear_polys(x, init=7, bound=None):
    """
    returns an iterator which enumerates the polynomials x-a for a ranging within the given bounds
    """
    p = x - init; step = -x.parent().one()
    if bound is not None:
        bound = x - bound
    while p != bound:
        yield p; p += step

###########################################################################################

def _merge_homomorphic_images(v, mod, vp, p, reconstruct=True):
    """
    Interpolation or chinese remaindering on the coefficients of operators.

    INPUT:

    - ``v`` -- a vector over R
    - ``mod`` -- an element of R such that there is some hypothetical
      vector over of which ``v`` can be obtained by taking its
      coefficients mod ``mod``.
    - ``vp`` -- an vector over r
    - ``p`` -- an element of R such that the hypothetical vector
      gives `vp` when its coeffs are reduced mod ``p``.
    - ``reconstruct`` (default: ``True``) -- if set to ``False``, only 
      do Chinese remaindering, but no rational reconstruction.       

    OUTPUT:

    A pair `(M, m)` where

    - `M` is a vector over R and obtained from `L` and `Lp`
      by chinese remindering or interpolation, possibly followed by rational
      reconstruction.
    - `m` is either `mod*p` or `0`, depending on whether rational reconstruction
      succeeded.

    If `v` is the zero vector, the method returns ``(Lp, p)``. Otherwise, if
    the dimensions of `v` and `vp` don't match, an exception is raised. An 
    exception is also raised if `vp` is zero. 

    Possible ground rings:

    - R=ZZ, r=GF(p). The method will apply chinese remaindering 
    - R=ZZ[q], r=GF(p)[q]. The method will apply chinese remaindering on the coefficients 
    - R=GF(p)[q], r=GF(p)[q]. The method will apply interpolation 

    """

    B = v.base_ring()
    R = mod.parent()
    r = vp.base_ring()
    if r is not B:
        vp = vp.change_ring(B)

    atomic = (B is ZZ) or (B.characteristic() > 0)
    poly = not atomic     

    if mod == 0:
        return v, R.zero()

    elif v.is_zero():
        vmod, mod = vp, R(p)

    elif len(v) != len(vp) or vp.is_zero():
        raise ValueError

    else:

        p = R(p); mod = R(mod)    
        if poly:
            p = R.base_ring()(p)
            mod = R.base_ring()(mod)
            R = r

        # cra / interpolation

        (_, mod0, p0) = p.xgcd(mod)
        mod0 = R(mod0*p); p0 = R(p0*mod)

        coords = []
        for i in range(len(v)):
            coords.append(mod0*v[i] + p0*B(vp[i]))

        vmod = vector(R, coords)
        mod *= p

    if not reconstruct:
        return vmod, mod

    # rational reconstruction attempt

    if R.characteristic() == 0:
        mod2 = mod // ZZ(2)
        adjust = lambda c : ((c + mod2) % mod) - mod2 
    else:
        if mod.degree() <= 5: # require at least 5 evaluation points
            return vmod, mod        
        adjust = lambda c : c % mod

    coords = list(vmod)

    try:
        d = R.one()
        for i in range(len(coords) - 1, -1, -1):
            c = coords[i]
            if poly:
                for l in range(c.degree(), -1, -1): 
                    d *= _rat_recon(d*c[l], mod)[1]
            else:
                d *= _rat_recon(d*c, mod)[1]
    except (ArithmeticError, ValueError):
        return vmod, mod # reconstruction failed

    # rat recon succeeded, the common denominator is d. clear it and normalize numerators.

    for i in range(len(coords)):
        c = d*coords[i]
        if adjust is not None:
            c = c.map_coefficients(adjust) if poly else adjust(c)
        coords[i] = c

    return v.parent()(coords), R.zero()

###########################################################################################

def _rat_recon(a, m, u=None):
    """
    if m.parent() is ZZ:

      find (p, q) such that a == p/q mod m and abs(p*q) < m/1000
      if u is not None, require abs(q) <= u.
      raises ArithmeticError if no p/q is found.
      if m < 1000000, we use sage's builtin

    if m.parent() is GF(p)[t]: 
    
      find (p, q) such that a == p/q mod m and deg(p) + deg(q) < deg(m) - 3
      if u is not None, require deg(q) <= deg(u).
      raises ArithmeticError if no p/q is found.
      if deg(m) < 6, we use sage's builtin

    """
    
    K = m.parent() # GF(p)[t] or ZZ
    
    if K is ZZ:
        score_fun = lambda p, q: abs(p*q)
        bound = m // ZZ(10000)
        early_termination_bound = m // ZZ(1000000)
        if u is None:
            u = m
    else:
        score_fun = lambda p, q: p.degree() + q.degree()
        bound = m.degree() - 3
        early_termination_bound = m.degree() - 6
        if u is None:
            u = m.degree()
    
    zero = K.zero(); one = K.one(); mone = -one    
    
    if a in (zero, one, mone):
        return a, one
    elif early_termination_bound <= 0:
        out = a.rational_reconstruct(m)
        return (a.numerator(), a.denominator())

    # p = q*a + r*m for some r
    p = K(a) % m; q = one;   
    pp = m;       qq = zero; 
    out = (p, one); score = score_fun(p, one)
    if K is ZZ:
        mp = m - p; mps = score_fun(mp, one)
        if mps < score:
            out = (mp, -ZZ.one()); score = mps
        if score < early_termination_bound:
            return out

    while True:

        quo = pp // p
        (pp, qq, p, q) = (p, q, pp - quo*p, qq - quo*q)

        if p.is_zero() or score_fun(q, one) > u:
            break

        s = score_fun(p, q)
        if s < score:
            out = (p, q); score = s
            if score < early_termination_bound:
                break

    if score < bound:
        if K is ZZ:
            return out
        else:
            lc = out[1].leading_coefficient()
            return (out[0]/lc, out[1]/lc)
    else:
        raise ArithmeticError

###########################################################################################

_ff_cache = dict()
def _ff_factory(domain):
    characteristic = domain.characteristic() # that's the only information that matters here
    try:
        return _ff_cache[characteristic]
    except:
        characteristic = int(characteristic)
        c = dict()
        if characteristic == 0: 
            one = ZZ.one()
            def ff(u, v): ## computation in ZZ; first argument must be integer too!
                try:
                    return c[u, v]
                except:
                    if v == 0:
                        return one
                    else:
                        cc = ZZ(u*ff(u - 1, v - 1))
                        c[u, v] = cc
                        return cc
        else:                
            def ff(u, v): ## computations with Python integers
                try:
                    return c[u, v]
                except:
                    if v == 0:
                        return 1
                    else:
                        cc = int((int(u)*ff(u - 1, v - 1)) % characteristic)
                        c[u, v] = cc
                        return cc
        _ff_cache[characteristic] = ff
        return ff

_power_cache = dict()
def _power_factory(domain):
    try:
        return _power_cache[domain]
    except:
        c = dict()
        domain = domain.fraction_field()
        if domain.characteristic() > 0: 
            characteristic = domain.characteristic()
            def power(u, v): # using Python integers
                try:
                    return c[u, v]
                except:
                    if v == 0:
                        return 1
                    elif v < 0:
                        return int(~(GF(characteristic)(power(u, -v))))
                    else:
                        cc = (int(u)*power(u, v - 1)) % characteristic
                        c[u, v] = cc
                        return cc
        else:
            one = domain.one()
            def power(u, v): # using the actual domain (because u might contain q)
                try:
                    return c[u, v]
                except:
                    if v == 0:
                        return one
                    elif v < 0:
                        return ~domain(power(u, -v))
                    else:
                        cc = domain(u)*power(u, v - 1)
                        c[u, v] = cc
                        return cc
        _power_cache[domain] = power
        return power    

###########################################################################################

def guess_mult(data, algebra, **kwargs):
    """
    Searches for elements of the algebra which annihilates the given data.

    INPUT:

    - ``data`` -- a nested list of elements of the algebra's base ring's base ring `K` (or at least
      of objects which can be casted into this ring). 
      The depth of the nesting must match the number of generators of the algebra.
    - ``algebra`` -- an Ore algebra over a polynomial ring all of whose generators are
      the standard derivation, the standard shift, or a q-shift. 

    Optional arguments: 

    - ``cut`` -- if `N` is the minimum number of terms needed for some particular
      choice of order and degree, and if ``len(data)`` is more than ``N+cut``,
      use ``data[:N+cut]`` instead of ``data``. This must be a nonnegative integer
      or ``None``. Default: 100.
    - ``ensure`` -- if `N` is the minimum number of terms needed for some particular
      choice of order and degree, and if ``len(data)`` is less than ``N+ensure``,
      raise an error. This must be a nonnegative integer. Default: 0.
    - ``order`` -- maximum degree of the algebra generators in the sought operators. 
      Alternatively: a list or tuple specifying individual degree bounds for each 
      generator of the algebra. Default: 2
    - ``degree`` -- maximum total degree of the polynomial coefficients in the sought 
      operators. Default: 3
    - ``point_filter`` -- a callable such that index tuples of data array for which 
      the callable returns 'False' will not be used. Default: None (everything allowed).
    - ``term_filter`` -- a callable such that operators containing power products of 
      the algebra generators for which the callable returns 'False' are excluded.
      Default: None (everything allowed).
    - ``solver`` -- function to be used for computing the right kernel of a matrix
      with elements in `K`. 
    - ``infolevel`` -- an integer specifying the level of details of progress
      reports during the calculation. 

    OUTPUT:

    - The left ideal of ``algebra`` generated by all the operators of the specified order and degree
      that annihilate the given ``data``. It may be the zero ideal. 

    .. NOTE::

        This method is designed to find equations for D-finite objects. It may
        exhibit strange behaviour for objects which are holonomic but not
        D-finite. 

    EXAMPLES::

      sage: from ore_algebra import *
      sage: from ore_algebra.guessing import guess_mult
      sage: data = [[binomial(n,k) for n in range(10)] for k in range(10)]
      sage: guess_mult(data, OreAlgebra(ZZ['n','k'], 'Sn', 'Sk'), order=1, degree=0)
      Left Ideal (Sn*Sk - Sn - 1) of Multivariate Ore algebra in Sn, Sk over Fraction Field of Multivariate Polynomial Ring in n, k over Integer Ring
      sage: guess_mult(data, OreAlgebra(ZZ['x','y'], 'Dx', 'Dy'), order=1, degree=1)
      Left Ideal ((x + 1)*Dx + (-y)*Dy) of Multivariate Ore algebra in Dx, Dy over Fraction Field of Multivariate Polynomial Ring in x, y over Integer Ring
      sage: guess_mult(data, OreAlgebra(ZZ['n','y'], 'Sn', 'Dy'), order=1, degree=1)
      Left Ideal ((-y + 1)*Sn*Dy - Sn + (-y)*Dy - 1, (-n - 1)*Sn + y*Dy - n, (-y + 1)*Sn - y) of Multivariate Ore algebra in Sn, Dy over Fraction Field of Multivariate Polynomial Ring in n, y over Integer Ring
      sage: guess_mult(data, OreAlgebra(ZZ['x','k'], 'Dx', 'Sk'), order=1, degree=1)
      Left Ideal (Dx*Sk + (-x - 1)*Dx - 1, x*Dx*Sk + (x + 1)*Dx + (-k)*Sk - x, (x + 1)*Dx - k, (x + 1)*Dx*Sk + (-k - 1)*Sk) of Multivariate Ore algebra in Dx, Sk over Fraction Field of Multivariate Polynomial Ring in x, k over Integer Ring

    """

    infolevel = kwargs.setdefault('infolevel', 0)
    def info(bound, msg):
        if bound <= infolevel:
            print(msg)

    # 1. extract configuration from options and check input for plausibility
    l = data; dims = []
    while type(l) in (list, tuple):
        dims.append(len(l))
        l = l[0]
    dim = len(dims)
    range_dim = list(range(dim))

    deg = kwargs.setdefault('degree', 3)
    ord = kwargs.setdefault('order', 2)
    if ord in ZZ: 
        ord = [ord for i in range_dim]
    assert(dim == len(ord))

    gens = list(algebra.gens()); vars = algebra.base_ring().gens(); assert(dim == len(vars) == len(gens))

    info(1, lazy_string(lambda: datetime.today().ctime() + ": multivariate guessing started."))
    info(1, "dim(data)=" + str(dim) + ", algebra=" + str(algebra._latex_()))

    # 2. prepare polynomial terms, operator terms (structure set), and all terms
    from itertools import combinations_with_replacement, product
    pol_terms = [tuple(0 for i in range_dim)] # exponent vector (0,...,0)
    for d in range(1, deg + 1): # create exponent vectors for terms of total degree d
        for p in combinations_with_replacement(vars, d): 
            pol_terms.append( tuple(p.count(x) for x in vars) )

    op_terms = []
    f = kwargs.setdefault('term_filter', lambda x: True)
    for o in product(*[list(range(ord[i] + 1)) for i in range_dim]):
        if f(o) or f(prod(gens[i]**o[i] for i in range_dim)):
            op_terms.append(o)

    terms = [(p, o) for o in op_terms for p in pol_terms]
    info(1, str(len(terms)) + " terms.")

    offset = [0]*dim # left offset needed in the index space
    A = [] # A(n,u,v)
    B = [] # B(n,u,v)
    ## e.g.: [x^n] x^u D^v sum(a[n]x^n, n=0..infty) = power(*A(n,u,v)) * a[B(n,u,v)]
    ##        n^u S^v a[n] = power(*A(n,u,v)) a[B[n,u,v]]

    for i in range_dim:
        if algebra.is_D(i):
            offset[i] = -ord[i]; A.append(lambda n, u, v: (n - u + v, v)); B.append(lambda n, u, v: n - u + v)
        elif algebra.is_S(i):
            A.append(lambda n, u, v: (n, u)); B.append(lambda n, u, v: n + v)
        elif algebra.is_Q(i):
            _, q = algebra.is_Q(i); A.append(lambda n, u, v: (q, n*u)); B.append(lambda n, u, v: n + v)
        else:
            raise TypeError("unexpected algebra generator: " + str(gens[i]))

    # 3. prepare evaluation points
    f = kwargs.setdefault('point_filter', lambda *x: True)
    points = [p for p in product(*[list(range(offset[i], dims[i] - ord[i]))
                                   for i in range_dim]) if f(*p)]
    info(1, str(len(points)) + " points.")

    cut = kwargs.setdefault("cut", 100)
    if cut is not None and len(points) > len(terms) + cut:
        points = points[:len(terms) + cut]
        info(1, "keeping " + str(len(points)) + " points.")
    else:
        info(1, "keeping all " + str(len(points)) + " points.")

    if len(terms) + kwargs.setdefault("ensure", 0) >= len(points):
        raise ValueError("not enough data"    )

    C = algebra.base_ring().base_ring().fraction_field() ### constant field 

    if C.characteristic() in Primes() and C is GF(C.characteristic()): ### constant field is GF(p) --> raw guessing

        power = []
        for i in range_dim:
            if algebra.is_D(i):
                power.append(_ff_factory(C)); 
            elif algebra.is_S(i):
                power.append(_power_factory(C))
            elif algebra.is_Q(i):
                power.append(_power_factory(C))

        sol = guess_mult_raw(C, data, terms, points, power, A, B, **kwargs)

    elif C is QQ or is_PolynomialRing(C.base()) and len(C.base().gens()) == 1 and C.base_ring() is GF(C.characteristic()): 
        ### C == QQ or C == GF(p)(t) --> plain chinese remaindering (resp interpolation) plus rational reconstruction

        modulus_generator = _word_size_primes() if C is QQ else _linear_polys(C.base().gen(), 7, C.characteristic())
        to_hom = ( lambda mod : GF(mod) ) if C is QQ else ( lambda mod : (lambda pol: C(pol)(-mod[0])) )
        R = ZZ if C is QQ else C.base()
        mod = [R.one()]
        sol = None
        power = [None]*dim
        imgs = []
        kwargs['infolevel'] = infolevel - 2 

        while not all(m.is_zero() for m in mod):

            p = next(modulus_generator)
            info(1, "modulus = " + str(p))
            C_mod = GF(p) if C is QQ else C.base_ring()
            phi = to_hom(p)
            
            for i in range_dim:
                if algebra.is_D(i):
                    power[i] = _ff_factory(C_mod)
                elif algebra.is_S(i):
                    power[i] = _power_factory(C_mod)
                elif algebra.is_Q(i):
                    _, q = algebra.is_Q(i); A[i] = lambda n, u, v: (phi(q), n*u)
                    power[i] = _power_factory(C_mod)

            ## compute modular image 
            kwargs['phi'] = phi
            solp = guess_mult_raw(C_mod, data, terms, points, power, A, B, **kwargs)
            del kwargs['phi']

            if sol is None: ## initialization

                ## early termination check
                if len(solp) == 0:
                    info(1, lazy_string(lambda: datetime.today().ctime() + " : multivariate guessing completed by early termination."))
                    return algebra.ideal([])
                
                ## extract support of solutions
                for i in range(len(terms)):
                    if all(v[i].is_zero() for v in solp):
                        terms[i] = None

                sol = [[] for i in range(len(solp))]
                new_terms = []
                for i in range(len(terms)):
                    if terms[i] is not None:
                        new_terms.append(terms[i])
                        for j in range(len(solp)):
                            sol[j].append(R(solp[j][i]))
                terms = new_terms
                sol = [vector(R, s) for s in sol]
                mod = [p]*len(sol)

                if cut is not None and len(points) > len(terms) + cut:
                    points = points[:len(terms) + cut]

            else: ## subsequent iterations

                try: ## save
                    imgs[imgs.index(None)] = ([vector(R, s) for s in solp], p)
                except: ## merge, merge, and reconstruct

                    p = [p]*len(solp); solp = [vector(R, s) for s in solp]
                    for (solpp, pp) in imgs:
                        for i in range(len(solp)):
                            try:
                                solp[i], p[i] = _merge_homomorphic_images(solp[i], p[i], solpp[i], pp, reconstruct=False)
                            except:
                                info(2, "unlucky modulus " + str(pp) + " discarded")

                    imgs = [None]*(len(imgs) + 1)

                    for i in range(len(sol)):
                        try:
                            # if all mod[i] are zero in the end, this will terminate the while loop
                            sol[i], mod[i] = _merge_homomorphic_images(sol[i], mod[i], solp[i], p[i], reconstruct=True)
                        except:
                            info(2, "unlucky modulus " + str(p[i]) + " discarded")

    elif C.base_ring().fraction_field() is QQ and is_PolynomialRing(C.base()) and len(C.base().gens()) == 1:
        ### C = QQ(t)

        raise NotImplementedError 
        
    else: 
        raise NotImplementedError("unexpected constant domain")
        
    info(1, lazy_string(lambda: datetime.today().ctime() + " : " + str(len(sol)) + " solutions."))

    if kwargs.setdefault('_return_raw_vectors', False):
        return sol

    # 5. convert to operators
    basis = []; 
    R = algebra.base_ring(); C = R.base_ring().fraction_field().base(); R1 = PolynomialRing(C.fraction_field(), R.gens())
    for v in sol:
        coeffs = dict( (o, dict()) for o in op_terms )
        d = C.one()
        for i in range(len(terms)):
            coeffs[terms[i][1]][terms[i][0]] = v[i]
            try:
                d = lcm(d, C(v[i].denominator()))
            except:
                pass
        for o in op_terms:
            coeffs[o] = R(d*R1(coeffs[o]))
        basis.append(algebra(coeffs))

    info(1, lazy_string(lambda: datetime.today().ctime() + " : multivariate guessing completed."))
    return algebra.ideal(basis)

def guess_mult_raw(C, data, terms, points, power, A, B, **kwargs):
    """
    Low-level multivariate guessing function. Do not call this method unless you know what you are doing.
    In most situations, you will want to call the function `guess` instead.
    
    INPUT:

    - `data` -- a nested list of elements of C
    - `terms` -- a list of pairs of tuples (u, v) specifying exponent vectors u, v representing terms x^u D^v
    - `points` -- a list of tuples specifying indices of the data array
    - `power` -- a list of functions f mapping triples (n, u, v) of nonnegative integers to elements of C
    - `A` -- a list of functions mapping triples (n, u, v) to integers
    - `B` -- a list of functions mapping triples (n, u, v) to integers

    OUTPUT:
    
    A list of vectors generating the space of all vectors in C^len(terms) for which 
    all(sum(prod(f[i][A[i][n[i],u[i],v[i]]]*a[B[i][n[i],u[i],v[i]]] for i in range(len(A))) 
    for (u,v) in terms) == 0 for n in points)

    SIDE EFFECT: 

    Elements of the list `points` which lead to a zero equation will be discarded.
    """

    infolevel = kwargs.setdefault('infolevel', 0)
    def info(bound, msg):
        if bound <= infolevel:
            print(msg)

    phi = kwargs.setdefault('phi', lambda x: x)
    C = C.fraction_field()
    mat = []
    info(1, lazy_string(lambda: datetime.today().ctime() + " : setting up modular system..."))
    monomial_cache = dict()
    range_dim = list(range(len(A)))

    for k, n in enumerate(points):

        row = []
        for u, v in terms:

            idx = tuple(B[i](n[i], u[i], v[i]) for i in range_dim)
            if min(idx) < 0:
                row.append(phi(C.zero()))
            else:
                exp = tuple(A[i](n[i], u[i], v[i]) for i in range_dim)
                d = data
                try:
                    factor = monomial_cache[exp]
                    for i in idx: 
                        d = d[i]
                except KeyError:
                    factor = phi(C.one())
                    for p, i, e in zip(*(power, idx, exp)):
                        d = d[i]
                        factor *= p(e[0], e[1])
                    monomial_cache[exp] = factor
                row.append(phi(d) * factor)

        if all(e.is_zero() for e in row):
            points[k] = None
        else:
            mat.append(row)

    try:
        while True:
            points.remove(None) # in place
    except ValueError:
        pass

    monomial_cache.clear()
    if len(terms) + kwargs.setdefault('ensure') >= len(mat):
        raise ValueError("not enough data, or too many zeros")

    info(1, lazy_string(lambda: datetime.today().ctime() + " : solving modular system..."))
    sol = MatrixSpace(C, len(points), len(terms))(mat).right_kernel().basis()

    info(1, lazy_string(lambda: datetime.today().ctime() + " : " + str(len(sol)) + " solutions detected."))
    return sol
