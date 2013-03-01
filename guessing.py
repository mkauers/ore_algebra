
"""
guessing
========

"""

######### development mode ###########

try:
    if sys.modules.has_key('ore_algebra'):
        del sys.modules['ore_algebra']
except:
    pass

#######################################

from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.finite_rings.all import GF
from sage.matrix.constructor import Matrix, matrix

from ore_algebra import *

from datetime import datetime
import nullspace
import math

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
      of objects which can be casted into this ring).
    - ``algebra`` -- a univariate Ore algebra over a univariate polynomial ring whose
      generator is the standard derivation, the standard shift, the forward difference,
      the euler derivation, or a q-shift.

    Optional arguments:

    - ``cut`` -- if `N` is the minimum number of terms needed for some particular
      choice of order and degree, and if ``len(data)`` is more than ``N+cut``,
      use ``data[:N+cut]`` instead of ``data``. This must be a nonnegative integer
      or ``None``. Default: ``None``.
    - ``ensure`` -- if `N` is the minimum number of terms needed for some particular
      choice of order and degree, and if ``len(data)`` is less than ``N+ensure``,
      raise an error. This must be a nonnegative integer. Default: 0.
    - ``order`` -- bounds the order of the operators being searched for.
      Default: infinity.
    - ``degree`` -- bounds the degree of the operators being searched for.
      The method may decide to overrule this setting if it thinks this may speed up
      the calculation. Default: infinity.
    - ``path`` -- a list of pairs `(r, d)` specifying which orders and degrees
      the method should attempt. If this value is equal to ``None`` (default), a
      path is chosen which examines all the `(r, d)` which can be tested with the
      given amount of data. 
    - ``solver`` -- function to be used for computing the right kernel of a matrix
      with elements in `K`. 
    - ``lift`` -- function which maps the given objects in ``data`` to elements of `K`.
    - ``infolevel`` -- an integer specifying the level of details of progress
      reports during the calculation. 

    OUTPUT:

    - An element of ``algebra`` which annihilates the given ``data``.

    An error is raised if no such element is found. 

    .. NOTE::

    This method is designed to find equations for D-finite objects. It may exhibit strange
    behaviour for objects which are holonomic but not D-finite. 

    EXAMPLES::

      sage: ...
    
    """

    A = algebra; R = A.base_ring(); K = R.base_ring()

    if kwargs.has_key('infolevel'):
        infolevel = kwargs['infolevel']
        kwargs['infolevel'] = infolevel - 2
    else:
        infolevel = 0
        
    def info(bound, msg):
        if bound <= infolevel:
            print msg

    info(1, datetime.today().ctime() + ": guessing started.")
    info(1, "len(data)=" + str(len(data)) + ", algebra=" + str(A._latex_()))

    if A.ngens() > 1 or R.ngens() > 1:
        raise TypeError, "unexpected algebra"
    elif A.is_F() is not False:
        # reduce to shift case; note that this does not alter order or degrees
        info(1, "Translating problem to shift case...")
        x = A.base_ring().gen()
        A0 = OreAlgebra(A.base_ring(), ('S', {x:x+K.one()}, {}))
        return guess(data, A0, **kwargs).to_F(A)
    elif A.is_T() is not False:
        # reduce to shift case; note that this does not alter order or degrees (apart from switching them)
        info(1, "Translating problem to shift case...")
        x = A.base_ring().gen()
        A0 = OreAlgebra(A.base_ring(), ('S', {x:x+K.one()}, {}))
        order = degree = None
        if kwargs.has_key('order'):
            order = kwargs['order']
            del kwargs['order']
        if kwargs.has_key('degree'):
            degree = kwargs['degree']
            del kwargs['degree']
        if order is not None:
            kwargs['degree'] = order
        if degree is not None:
            kwargs['order'] = degree
        if kwargs.has_key('path'):
            kwargs['path'] = [ (d, r) for (r, d) in kwargs['path'] ]
        L0 = guess(data, A0, **kwargs)
        # (n, S) <==> (theta, x)
        x = A(R.gen()); T = A.gen(); L = A.zero()
        for i in xrange(L0.order() + 1):
            L = L + A(L0[i].coeffs())*(x**i)
        return L            
    elif (not A.is_S() and not A.is_D() and not A.is_Q()):
        raise TypeError, "unexpected algebra"
    elif R.is_field():
        return guess(data, A.change_ring(R.ring()), **kwargs)
    elif K.characteristic() == 0:
        # homomorphic images and rational reconstruction
        info(2, "Going to use chinese remaindering...")

        # 1. first hom image: brute force search, and catch path.

        # 2. 2nd hom image: try a refined path based on a guessed curve.

        # 3. further hom images using refined path until it stabelizes. 
        
        raise NotImplementedError
    elif K.base_ring() is not K:
        # homomorphic images and rational reconstruction
        info(2, "Going to use evaluation/interpolation...")
        raise NotImplementedError

    # At this point, A = GF(p)[x]<X> where X is S or D or Q.

    # create path

    ensure = kwargs['ensure'] if kwargs.has_key('ensure') else 0
    
    if kwargs.has_key('path'):
        path = kwargs['path']; del kwargs['path']
    else:
        N = len(data) - ensure
        r2d = lambda r: (N - 2*r - 2)/(r + 1) # python integer division intended.
        d2r = lambda d: (N - d - 2)/(d + 2) # dto.
        if A.is_D() is not False:
            (r2d, d2r) = (d2r, r2d)
        path = [(r, r2d(r)) for r in xrange(N)] + [(d2r(d), d) for d in xrange(N)]
        path = filter(lambda p: min(p[0], p[1]) >= 0, path)
        path.sort(key=lambda p: N*p[0] - p[1])
        d = r = 2*len(data)
        for i in xrange(len(path)):
            if path[i][0] == r or path[i][1] >= d:
                path[i] = None
            else:
                r, d = path[i]
        path = filter(lambda p: p is not None, path)
        prelude = [ (1, 1) ]
        while True:
            (r, d) = prelude[-1]; (r, d) = (d, r + d)
            if d > r2d(r):
                break
            prelude.append((r, d))
        path = prelude + path
        info(2, "constructed path with " + str(len(path)) + " points")

    if kwargs.has_key('degree'):
        degree = kwargs['degree']; del kwargs['degree']
        path = filter(lambda p: p[1] <= degree, path)

    if kwargs.has_key('order'):
        order = kwargs['order']; del kwargs['order']
        path = filter(lambda p: p[0] <= order, path)

    # search equation

    if kwargs.has_key('return_short_path'):
        return_short_path = True
        del kwargs['return_short_path']
    else:
        return_short_path = False

    def probe(r, d):
        kwargs['order'], kwargs['degree'] = r, d
        sols = guess_raw(data, A, **kwargs)
        info(2, str(len(sols)) + " sols for (r, d)=" + str((r, d)))
        return sols

    L = []; short_path = []
    
    for i in xrange(len(path)):
        
        r, d = path[i]; sols = probe(r, d)
        while return_short_path and d > 0 and len(sols) > 1:
            new = probe(r, d - 1)
            if len(new) == 0:
                break
            m = len(sols) - len(new) # this is > 0
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

    info(2, datetime.today().ctime() + ": search completed.")

    if len(L) == 0:
        raise ValueError, "No relations found."
    elif len(L) == 1:
        L = L[0]
    else:
        L = L[0].gcrd(L[1])
        info(2, datetime.today().ctime() + ": gcrd completed.")

    L = (~L.leading_coefficient().leading_coefficient())*L

    return (L, short_path) if return_short_path else L

#######################################

def guess_raw(data, A, order=-1, degree=-1, lift=None, solver=None, cut=None, ensure=0, infolevel=0):
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

      sage: K = GF(1091); R.<n> = K['n']; A = OreAlgebra(R, 'Sn')
      sage: data = [(5*n+3)/(3*n+4)*fibonacci(n)^3 for n in xrange(200)]
      sage: guess_raw(data, A, order=5, degree=3, lift=K)
      [(n^3 + 546*n^2 + 588*n + 786)*Sn^5 + (356*n^3 + 717*n^2 + 381*n + 449)*Sn^4 + (8*n^3 + 569*n^2 + 360*n + 214)*Sn^3 + (31*n^3 + 600*n^2 + 784*n + 287)*Sn^2 + (1078*n^3 + 1065*n^2 + 383*n + 466)*Sn + 359*n^3 + 173*n^2 + 503, (n^3 + 1013*n^2 + 593*n + 754)*Sn^5 + (797*n^3 + 56*n^2 + 7*n + 999)*Sn^4 + (867*n^3 + 1002*n^2 + 655*n + 506)*Sn^3 + (658*n^3 + 834*n^2 + 1036*n + 899)*Sn^2 + (219*n^3 + 479*n^2 + 476*n + 800)*Sn + 800*n^3 + 913*n^2 + 280*n]
    
    """

    if min(order, degree) < 0:
        return [] 

    R = A.base_ring(); K = R.base_ring(); q = A.is_Q()

    def info(bound, msg):
        if bound <= infolevel:
            print msg

    info(1, datetime.today().ctime() + ": raw guessing started.")
    info(1, "len(data)=" + str(len(data)) + ", algebra=" + str(A._latex_()))

    if A.ngens() > 1 or (not A.is_S() and not A.is_Q() and not A.is_D() ):
        raise TypeError, "unexpected algebra"

    diff_case = True if A.is_D() else False
    deform = (lambda n: q[1]**n) if q is not False else (lambda n: n)
    min_len_data = (order + 1)*(degree + 2)

    if cut is not None and len(data) > min_len_data + cut:
        data = data[:min_len_data + cut]

    if len(data) < min_len_data + ensure:
        raise ValueError, "not enough terms"

    if lift is not None:
        data = map(lift, data)

    if not all(p in K for p in data):
        raise ValueError, "illegal term in data list"

    if solver is None:
        solver = A._solver(K)
        
    if solver is None:
        solver = nullspace.sage_native

    sys = {(0,0):data}
    nn = [deform(n) for n in xrange(len(data))]
    z = [K.zero()]

    if diff_case:
        # sys[i, j] contains ( x^i * D^j ) (data)
        nn = nn[1:]
        for j in xrange(order):
            sys[0, j + 1] = map(lambda a,b: a*b, sys[0, j][1:], nn)
            nn.pop(); 
        for i in xrange(degree):
            for j in xrange(order + 1):
                sys[i + 1, j] = z + sys[i, j]
    else:
        # sys[i, j] contains ( (n+j)^i * S^j ) (data)
        for i in xrange(degree):
            sys[i + 1, 0] = map(lambda a,b: a*b, sys[i, 0], nn)
        for j in xrange(order):
            for i in xrange(degree + 1):
                sys[i, j + 1] = sys[i, j][1:]

    sys = [sys[i, j] for j in xrange(order + 1) for i in xrange(degree + 1) ]

    trim = min(len(c) for c in sys)
    for i in xrange(len(sys)):
        if len(sys[i]) > trim:
            sys[i] = sys[i][:trim]

    info(2, datetime.today().ctime() + ": matrix construction completed. size=" + str((len(sys[0]), len(sys))))
    sol = solver(matrix(K, zip(*sys)), infolevel=infolevel-2)
    info(2, datetime.today().ctime() + ": nullspace computation completed. size=" + str(len(sol)))

    sigma = A.sigma()
    for l in xrange(len(sol)):
        c = []; s = list(sol[l])
        for j in xrange(order + 1):
            c.append(sigma(R(s[j*(degree + 1):(j+1)*(degree + 1)]), j))
        sol[l] = A(c)
        sol[l] *= ~sol[l].leading_coefficient().leading_coefficient()

    return sol
