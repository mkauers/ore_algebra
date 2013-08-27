
"""

nullspace
=========

This module is about computing the right kernel of matrices with polynomial entries.
It provides various general purpose tools for constructing **solvers**. 

By a **solver**, we mean a function of the following form:

  .. function:: solver(mat, degrees=[], infolevel=0)

  INPUT:
  
  - ``mat`` -- a matrix `A` over some polynomial ring
  - ``degrees`` -- list of nonnegative integers, providing, for each variable `x`, a bound 
    on the degree with which `x` is going to appear in the output. This data is optional. 
    If set to ``[]``, the solver has to figure out the degrees by itself.
  - ``infolevel`` -- a nonnegative integer indicating the desired verbosity (default=0), or
    a pair ``(u, v)`` where ``u`` indicates the desired verbosity and ``v`` specifies how many
    leading spaces should be added to the printed lines (default=0).
  
  OUTPUT:

  - list of vectors `v` such that `A*v=0`

Depending on the application from which a polynomial matrix arises, the preferred way of computing its
nullspace may be different. The purpose of this module is not to provide a single solver which is
good for every possible input, but instead, it provides a collection of functions by which solvers 
can be composed. 

For example, gauss_ is a function for constructing a solver using fraction free Gaussian elimination.
It works for most polynomial domains. 

::
  
  sage: my_solver = gauss()
  sage: A = MatrixSpace(ZZ['x'], 4, 5).random_element()
  sage: V = my_solver(A)
  sage: A*V[0]
  (0, 0, 0, 0)
  
  sage: A = MatrixSpace(ZZ['x', 'y'], 4, 5).random_element()
  sage: V = my_solver(A)
  sage: A*V[0]
  (0, 0, 0, 0)
  
  sage: A = MatrixSpace(GF(1093)['x', 'y'], 4, 5).random_element()
  sage: V = my_solver(A)
  sage: A*V[0]
  (0, 0, 0, 0)

Several other functions create solvers which reduce the nullspace problem to one or more nullspace problems
over simpler domains which are then solved by another solver. This solver can be chosen by the user. 

For example, kronecker_ is a function for constructing a solver for matrices of multivariate polynomials.
It requires as parameter a solver for matrices of univariate polynomials, for instance a solver created
by gauss_.

::
  
  sage: my_solver = kronecker(gauss())
  sage: A = MatrixSpace(ZZ['x', 'y'], 4, 5).random_element()
  sage: V = my_solver(A)
  sage: A*V[0]
  (0, 0, 0, 0)

For large random examples, the solver created by ``kronecker(gauss())`` is likely to be significantly faster
than the solver created by ``gauss()``. Even faster, for matrices of polynomials with integer coefficients,
is a solver using chinese remaindering combined with kronecker substitution combined with gaussian elimination:

::
  
  sage: my_solver = cra(kronecker(gauss()))
  sage: A = MatrixSpace(ZZ['x', 'y'], 4, 5).random_element()
  sage: V = my_solver(A)
  sage: A*V[0]
  (0, 0, 0, 0)

Alternatively:

::

  sage: my_solver = kronecker(cra(gauss()))
  sage: A = MatrixSpace(ZZ['x', 'y'], 4, 5).random_element()
  sage: V = my_solver(A)
  sage: A*V[0]
  (0, 0, 0, 0)

Here is the same example with a variant of the same solver (not necessarily faster).

::
  
  sage: my_solver = cra(kronecker(lagrange(sage_native, start_point=5), newton(sage_native)), max_modulus = 2^16, proof=True, ncpus=4)
  sage: A = MatrixSpace(ZZ['x', 'y'], 4, 5).random_element()
  sage: V = my_solver(A)
  sage: A*V[0]
  (0, 0, 0, 0)

A particular solver will typically only be applicable to matrices with entries in a domain for which it
was designed for. Below is an overview of the functions provided in this module, together with the supported
input domains, and the input domains of the subsolvers they require.
The notation `K[x,...]` refers to univariate or multivariate polynomial ring, understanding the same
reading (unvariate vs. multivariate) in corresponding rows of the 2nd and 3rd column.

  =============== ================================================== ========================
  method           input domain                                      requires subsolver for
  =============== ================================================== ========================
  cra_            `K[x,...]` where `K` is `ZZ`, `QQ`, or `GF(p)`     `GF(p)[x,...]`
  galois_         `QQ(alpha)[x,...]`                                 `GF(p)[x,...]`
  clear_          `K(x,...)`                                         `K[x,...]`
  clear_          `K[x,...]` where `K` is the fraction field of `R`  `R[x,...]`
  compress_       `K[x,...]` or `K(x,...)`                           same domain and `GF(p)`
  kronecker_      `K[x,...]`                                         `K[x]` and `GF(p)[x]`
  gauss_          `K[x,...]`                                         None
  wiedemann_      `K[x,...]` or `K(x,...)`                           None
  lagrange_       `K[x]` or `K(x)` where `K` is a field              `K`
  hermite_        `K[x]` where `K` is a field                        None
  newton_         `K[x]` where `K` is a field                        `K`
  merge_          `K[x,...][y,...]`                                  `K[x,...,y,...]`
  `quick_check`_  `K[x,...]` where `K` is `ZZ`, `QQ`, or `GF(p)`     same domain and `GF(p)`
  `sage_native`_  `K[x,...]` or `K(x,...)` or `K`                    None
  =============== ================================================== ========================

AUTHOR:

 - Manuel Kauers (2012-09-16)

.. _cra : #nullspace.cra
.. _clear : #nullspace.clear
.. _compress : #nullspace.compress
.. _kronecker : #nullspace.kronecker
.. _gauss : #nullspace.gauss
.. _hermite : #nullspace.hermite
.. _newton : #nullspace.newton
.. _lagrange : #nullspace.lagrange
.. _`sage_native` : #nullspace.sage_native
.. _wiedemann : #nullspace.wiedemann
.. _merge : #nullspace.merge
.. _galois : #nullspace.galois
.. _`quick_check` : #nullspace.quick_check
 
"""

#############################################################################
#  Copyright (C) 2013 Manuel Kauers (mkauers@gmail.com).                    #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################


""" todo:

* hensel: Z[x..] to Z_p[x..] with p-adic lifting and subsolver
* galois: QQ(alpha)[x..] to Zp[x..]
* schur: block-wise recursive gaussian elimination?
* preconditioners?

bug in kronecker when ground domain of input is ZZ? (segfault)

extend lagrange to multivariate polynomials

add handling of exceptional cases (empty matrices etc.) 

parallelism in : lagrange, gauss, hermite

systematic benchmarking with random matrices and meaningful matrices

testsuite

"""

from sage.rings.arith import previous_prime as pp
from sage.rings.arith import CRT_basis, xgcd, gcd, lcm
from sage.misc.misc import prod
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.fraction_field import FractionField
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.finite_rings.all import GF
from sage.matrix.berlekamp_massey import berlekamp_massey
from sage.ext.multi_modular import MAX_MODULUS 
from sage.parallel.decorate import parallel
from sage.matrix.constructor import Matrix, matrix
from sage.modules.free_module_element import vector
from datetime import datetime
import math

#####################
####### tools #######
#####################

class NoSolution(ArithmeticError):
    # used in some solvers to indicate empty solution spaces
    def __init__(self):
        pass
    
    def __str__(self):
        return "no solution"

def heuristic_row_content(row, ring):
    # used in nullspace_gauss

    n = len(row)

    if n == 0:
        return ring.zero()
    elif n <= 5:
        return gcd(row)

    degrees = [p.degree() for p in row]

    if sum(degrees) < 100*n:
        return gcd(row)

    # taking gcd of least-degree element, the second-least-element element,
    # the sum of elements at odd indices, and the sum of the other elements
    m1 = row[0]; m2 = row[1]; d1 = degrees[0]; d2 = degrees[1]
    if d2 < d1:
        m1, m2, d1, d2 = m2, m1, d2, d1
    for i in xrange(2, n):
        if degrees[i] < d2:
            m2, d2 = row[i], degrees[i]
            if d2 < d1:
                m1, m2, d1, d2 = m2, m1, d2, d1
    
    g = gcd([m1, m2, sum(row[i] for i in xrange(n) if i%2==0),
                sum(row[i] for i in xrange(n) if i%2==1)])
    
    return g

### good pivot selection strategy:
#  1. measure nz_fillin by rows^EXPONENT*cols
#  2. select candidates for which nz_fillin < ALPHA*(min_nz_fillin + BETA)
#  3. for each candidate, estimate term fillin caused by pivot multiplication
#  4. take the candidate for which GAMMA*(nz_fillin/avg_nz_fillin)^2 + (term_fillin/avg_term_fillin)^2 is smallest
#  best known parameter values:
#     EXPONENT=2, ALPHA=2, BETA=10, GAMMA=1

def _pivot(mat, r, n, c, m, zero):
    """
    Given a matrix, find a \"good\" pivot element in the index range (r..n-1) x (c..m-1).
    Returns the index (i, j) of the suggested pivot. The return value is None iff the
    matrix is the zero matrix. The last argument provides the zero element of the ring. 
    """

    EXPONENT, ALPHA, BETA, GAMMA = 2, 2, 10, 1

    ## throughout this function, matrix indices (i, j) are understood relative to (r, c).
    ## if (i, j) is a pivot candidate, the final pivot will be (i + r, j + c).
    n = n - r; m = m - c

    nz_in_row = [ 0 for i in xrange(n) ]  # number of nonzero elements in row i 
    nz_in_col = [ 0 for j in xrange(m) ]  # number of nonzero elements in col j
    zero_matrix = True                    # any nonzero elements at all?

    # number of nonzero elements in those rows i for which mat[i][j] is nonzero
    nz_in_rows_for_col = [ 0 for j in xrange(m) ]  

    for i in xrange(n):
        mati = mat[i + r]
        for j in xrange(m):
            if mati[j + c] != zero:
                nz_in_row[i] += 1; nz_in_col[j] += 1; zero_matrix = False

    if zero_matrix:
        return None # early termination: zero matrix

    for i in xrange(n):
        mati = mat[i + r]
        for j in xrange(m):
            if mati[j + c] != zero:
                if nz_in_row[i] == 1 or nz_in_col[j] == 1:
                    return (i + r, j + c) # early termination: row or column with only one nonzero entry
                nz_in_rows_for_col[j] += nz_in_row[i]
    
    piv_cand = []; min_nz_fillin = 1e1000; bound = ALPHA*(min_nz_fillin + BETA)
    for i in xrange(n):
        mati = mat[i + r]
        for j in xrange(m):
            if mati[j + c] != zero:
                w = (nz_in_row[i] - 1)**EXPONENT*(nz_in_col[j] - 1) # expected fillin
                if w < min_nz_fillin:
                    min_nz_fillin = w; bound = ALPHA*(min_nz_fillin + BETA)
                    piv_cand.append((i, j, w, nz_in_row[i] - 1))
                elif w <= bound:
                    piv_cand.append((i, j, w, nz_in_row[i] - 1))
    
    piv_cand = [ cand for cand in piv_cand if cand[2] <= bound ]

    if len(piv_cand) == 1:
        return (piv_cand[0][0] + r, piv_cand[0][1] + c)

    # At this point, len(piv_cand) > 1 and all candidates have w > 0

    R = zero.parent(); 
    if len(R.gens()) == 1:
        K = R.base_ring()
        if K == ZZ:
            elsize = dict()
            def size(ij):
                if not elsize.has_key(ij):
                    pol = mat[ij[0] + r][ij[1] + c].coefficients()
                    m = float(max(abs(cc) for cc in pol))
                    # the constant in the line below is 53*log(2)
                    elsize[ij] = (1 + math.floor(math.log(m)/36.7368))*len(pol)
                return elsize[ij]
        elif K.is_finite():
            def size(ij):
                pol = mat[ij[0] + r][ij[1] + c]
                return pol.degree() - pol.ord()
        else:
            elsize = dict()
            def size(ij):
                if not elsize.has_key(ij):
                    elsize[ij] = len(mat[ij[0] + r][ij[1] + c].coefficients())
                return elsize[ij]
    else:
        elsize = dict()
        def size(ij):
            if not elsize.has_key(ij):
                elsize[ij] = len(mat[ij[0] + r][ij[1] + c].coefficients())
            return elsize[ij]

    # determine the expected term-fillin for each pivot candidate
    avg_nz_fillin = float(0); avg_term_fillin = float(0); 
    for k in xrange(len(piv_cand)):
        i, j, nz_fillin, nz_row = piv_cand[k] 
        pivot_fillin = size((i, j))*(nz_in_rows_for_col[j] - nz_in_col[j] - nz_in_row[i] + 1)
        # The elimination rule being  ``mat[l][k] = mat[i][j]*mat[l][k] - mat[i][k]*mat[l][j]'', we
        # count only the fillin caused by the multiplication with mat[i][j] and neglect the fillin
        # caused by the additive term mat[i][k]*mat[l][j], although it ought to be of roughtly the
        # same size. In experiments, taking also an estimate for the other part into account has
        # led to poorer performance (perhaps because we can't predict the fillin for this part
        # sufficiently accurately).
        avg_nz_fillin += nz_fillin; avg_term_fillin += pivot_fillin
        piv_cand[k] = (i, j, nz_fillin, nz_row, pivot_fillin)

    l = float(len(piv_cand)); avg_nz_fillin /= l; avg_term_fillin /= l

    # select the candidate where both the term-fillin and the non-zero-fillin are small
    alpha = 1/max(.01, avg_nz_fillin); gamma = 1/max(.01, avg_term_fillin)
    pivot = min(piv_cand, key=lambda (i, j, w1, w2, w3): (alpha*w1)**2 + (gamma*w3)**2)
    del piv_cand
    return (pivot[0] + r, pivot[1] + c)

def _leading_coefficient(p):
    try:
        return p.leading_coefficient() ## good for univariate polynomials
    except:
        return p.lc() ## good for multivariate polynomials

def _normalize(sol):
    "make solution vectors monic"
    
    if len(sol) == 0:
        return sol

    R = sol[0][0].parent()
    if R.is_field():
        one = R.one(); zero = R.zero()
        for v in sol:
            j = 0
            while v[j] == zero:
                j += 1
            piv = one/v[j]
            for k in xrange(j, len(v)):
                v[k] *= piv
    elif R.base_ring().is_field():
        one = R.one(); zero = R.zero()
        for v in sol:
            j = 0
            while v[j] == zero:
                j += 1
            piv = one/_leading_coefficient(v[j])
            for k in xrange(j, len(v)):
                v[k] *= piv
        
    return sol

def _select_regular_square_submatrix(V0, n, m, dim, one, zero):
    # Determine the indices i[1], i[2], ..., i[dim] so that when viewing the vectors in V0 as columns of a matrix,
    # picking the rows with these indices yields the identity matrix.
    # A value error is raised if a different format is encountered.
    # This function is used by newton and wiedemann.
    idxB = [ -1 for k in xrange(dim) ]
    for k in xrange(dim):
        ek = [(one if i==k else zero) for i in xrange(dim)]
        for i in xrange(m):
            if [ V0[j][i] for j in xrange(dim) ] == ek:
                idxB[k] = i; break # i-th row is k-th unit vector
    if set(idxB) != set(range(dim)):
        raise ValueError
    idxA = [ i for i in xrange(m) if idxB.count(i) == 0  ]

    return idxA, idxB

def _info(infolevel, *message, **kwargs):
    if infolevel in ZZ:
        infolevel = (infolevel, 0)
    alter = kwargs['alter'] if kwargs.has_key('alter') else (0, 0)
    if alter in ZZ:
        alter = (alter, 0)
    if infolevel[0] + alter[0] <= 0:
        return
    for i in xrange(infolevel[1] + alter[1]):
        print " ",
    for m in message:
        print m,
    print ""

def _alter_infolevel(infolevel, dlevel, dprefix):
    if infolevel in ZZ:
        infolevel = (infolevel, 0)
    return (infolevel[0] + dlevel, infolevel[1] + dprefix)

def _launch_info(infolevel, name, dim=None, deg=None, domain=None):
    message = datetime.today().ctime() + ": " + name + " called";
    if not dim == None:
        message = message + ", dim=" + str(dim)
    if not deg == None:
        message = message + ", deg=" + str(deg)
    if not domain == None:
        try:
            message = message + ", domain=" + domain._latex_()
        except:
            message = message + ", domain=" + str(domain)
    message = message + "."    
    _info(infolevel, message)

########################################
####### solvers and transformers #######
########################################

def sage_native(mat, degrees=[], infolevel=0):
    r"""
    Computes the nullspace of the given matrix using Sage's native method.
    
    INPUT:
    
    - ``mat`` -- any matrix
    - ``degrees`` -- ignored
    - ``infolevel`` -- a nonnegative integer indicating the desired verbosity.
    
    OUTPUT:
    
    - a list of vectors that form a basis of the right kernel of ``mat``

    EXAMPLES:

    ::

       sage: A = MatrixSpace(GF(1093)['x'], 4, 7).random_element(degree=3)
       sage: V = sage_native(A)
       sage: A*V[0]
       (0, 0, 0, 0)

    ALGORITHM: just calls ``right_kernel_matrix()`` on ``mat`` and converts its output to a list of vectors

    """
    _launch_info(infolevel, "sage_native", dim=mat.dimensions(), domain=mat.parent().base_ring())
    return _normalize([v for v in mat.right_kernel_matrix()])

def gauss(pivot=_pivot, ncpus=1, fun=None):
    r"""
    Creates a solver based on fraction free gaussian elimination.
 
    INPUT:

    - ``pivot`` -- a function which takes as input: a matrix ``mat`` (as list of list of ring elements),
      four integers `r`, `n`, `c`, `m` specifying the index range ``mat[r:n][c:m]``, and the ``zero``
      element of the ring, and returns and a pair `(i, j)` such that ``mat[i][j]!=zero`` and `(i, j)`
      is a good choice for a pivot. The function returns ``None`` iff there are no nonzero elements
      in the given range of the matrix.

      The default function chooses `(i,j)` such that ``mat`` has many zeros in row `i` and column `j`,
      and ``mat[i][j]`` has a small number of terms. 
      
    - ``ncpus`` -- maximum number of cpus that may be used in parallel by this solver
    
    - ``fun`` -- if different from ``None``, at the beginning of each iteration of the outer loops of
      forward and backward elimination, the solver calls ``fun(mat, idx)``, where ``mat`` is the current
      matrix (as list of lists of elements) and ``idx`` is a counter. This functionality is intended
      for analyzing the elimination process, e.g., for inspecting how well the solver succees in maintaining
      the sparsity of the matrix. The solver assumes that the function won't modify the matrix. 

    OUTPUT:

    - A solver based on fraction free gaussian elimination.

    EXAMPLES:

    ::

       sage: A = MatrixSpace(GF(1093)['x'], 4, 7).random_element(degree=3)
       sage: my_solver = gauss()
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0)

    ALGORITHM: fraction-free gaussian elimination with heuristic content removal and Markoviz pivot search.
    """
    def gauss_solver(mat, degrees=[], infolevel=0):
        r"""See docstring of gauss() for further information"""
        
        return _gauss(pivot, ncpus, fun, mat, degrees, infolevel)
    return gauss_solver    

def _gauss(pivot, ncpus, fun, mat, degrees, infolevel):
    """
    Internal version of nullspace.gauss_
    """

    n, m = mat.dimensions(); R = mat.parent().base_ring(); x = R.gen(); zero = R.zero(); one = R.one()
    _launch_info(infolevel, "gauss", dim=(n, m), domain=R)

    if ncpus > 1:
        print "Parallel gaussian elimination not yet implemented. Proceeding sequentially..."

    if n == 0:
        return [vector(R, v) for v in VectorSpace(QQ, m).basis()]
    mat = filter(any, [ [ R(el) for el in row ] for row in mat ] ) # discard zero rows.
    n = len(mat)
    
    r = 0; # current row
    cancel_constants = (R.characteristic() == 0)

    col_perm = {} # col_perm[i] == j means that the current column i corresponds to the original column j
    for j in xrange(m):
        col_perm[j] = j

    _info(infolevel, "forward elimination...", alter = -1)

    # forward elimination
    for c in xrange(m):

        _info(infolevel, "column", c, "out of", m, "...", alter = -2)

        if fun is not None:
            fun(mat, c)

        # 1. choose a "good" pivot 
        p = pivot(mat, r, n, c, m, zero)
        if p is None:
            break
        (pr, pc) = p
        
        mat[r], mat[pr] = mat[pr], mat[r]
        for i in xrange(n):
            mati = mat[i]
            mati[c], mati[pc] = mati[pc], mati[c]
        col_perm[c], col_perm[pc] = col_perm[pc], col_perm[c]
        
        # 2. perform elimination
        affected_rows = []
        for i in xrange(r + 1, n):
            if mat[i][c] != zero:
                affected_rows.append(i); matr = mat[r]; piv = matr[c]; mati = mat[i]; elim = mati[c]
                g = gcd(piv, elim)
                if g != one:
                    piv //= g; elim //= g
                del g
                for j in xrange(c + 1, m):
                    mati[j] = (piv*mati[j] - elim*matr[j])
                mati[c] = zero

        # 3. cancel common content of all affected rows
        g = heuristic_row_content([mat[i][j] for i in affected_rows for j in xrange(c + 1, m) if mat[i][j] != zero], R)
        if g != zero and g != one and (cancel_constants or not g.is_constant()):
            for i in affected_rows:
                mati = mat[i]
                for j in xrange(c + 1, m):
                    if mati[j] != zero:
                        mati[j] //= g
        del g

        # 4. cancel remaining content of individual rows
        for i in affected_rows:
            mati = mat[i]
            g = heuristic_row_content([mati[j] for j in xrange(m) if mati[j] != zero], R)
            if g != zero and g != one and (cancel_constants or not g.is_constant()):
                for j in xrange(c + 1, m):
                    mati[j] //= g
            del g

        r = r + 1

    dim = m - r # dimension of the solution space

    if dim == 0:
        _info(infolevel, "No solution.", alter = -1)
        return [] # no solution

    _info(infolevel, "Constructing", dim, "nullspace basis vectors.", alter = -1)

    sol = [[ zero for i in xrange(m) ] for  j in xrange(dim) ]
    for i in xrange(dim):
        sol[-i-1][-i-1] = one

    for i in xrange(r - 1, -1, -1):
        _info(infolevel, "Coordinate", i, alter = -2)
        mati = mat[i]
        for j in xrange(dim):
            solj = sol[j]
            num = -sum(mati[k]*solj[k] for k in xrange(i + 1, m) if mati[k] != zero and solj[k] != zero)
            if num == zero:
                continue
            den = mati[i] 
            # now solj[i] = num/den, but we do it without rational functions
            g = gcd(num, den)
            if g != one:
                num //= g; den //= g
            del g
            if den == -one:
                den = one; num = -num
            if den != one:
                for k in xrange(i + 1, m):
                    solj[k] *= den
            solj[i] = num

    col_perm_inv = {}
    for j in xrange(m):
        col_perm_inv[col_perm[j]] = j

    sol = [[v[col_perm_inv[i]] for i in xrange(m)] for v in sol]

    for v in sol:
        g = heuristic_row_content([p for p in v if p != zero], R)
        if g != one and g != zero:
            for j in xrange(m):
                v[j] //= g

    return _normalize([vector(R, v) for v in sol])

def hermite(early_termination=True):
    """
    Creates a solver which computes a nullspace basis of minimal degree.

    INPUT:

    - ``early_termination`` -- a boolean value. If set to ``True`` (default), the calculation is aborted
      as soon as the first solution vector has been found. If set to ``False``, the calculation continues
      up to some (potentially rather pessimistic) worst case bound on the possible degrees of the solution
      vectors. If degree information is supplied to the solver, the ``early_termination`` setting is ignored.

    OUTPUT:

    - A solver based on Hermite-Pade approximation

    EXAMPLES:

    ::

       sage: A = MatrixSpace(GF(1093)['x'], 4, 7).random_element(degree=3)
       sage: my_solver = hermite()
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0)

    ALGORITHM: Hermite-Pade approximation
    """
    def hermite_solver(mat, degrees=[], infolevel=0):
        r"""See docstring of hermite() for further information"""
        return _hermite(early_termination, mat, degrees, infolevel)
    return hermite_solver

def _hermite(early_termination, mat, degrees, infolevel, truncate=None):
    """
    internal version of nullspace.hermite_.
    """
    
    n, m = mat.dimensions(); matdeg = max( mat[i,j].degree() for i in xrange(n) for j in xrange(m) )
    _launch_info(infolevel, "hermite", dim=(n,m), deg=matdeg, domain=mat.parent().base_ring())

    if truncate is not None:
        deg = truncate
        early_termination = False
    elif len(degrees) < 1:
        deg = (min(n, m) + 1)*matdeg
    else:
        deg = degrees[0] + matdeg 
        early_termination = False
    R = mat.parent().base_ring() # expected to be univariate polynomial ring over a field
    V, done = _hermite_rec(early_termination, R, mat, deg + 1, [0 for i in xrange(m) ], \
                           _alter_infolevel(infolevel, -1, 1))
    V = V.transpose()
    if not done:
        V = [ v for v in V if max(p.degree() for p in v) <= degrees[0] ]
    # if the coefficient domain is a field, make the lowest-indexed nonzero component of each vector monic
    if R.base_ring().is_field():
        one = R.base_ring().one()
        for v in V:
            j = 0
            while v[j].degree() < 0:
                j += 1
            piv = one/_leading_coefficient(v[j])
            for k in xrange(j, m):
                v[k] *= piv
    return [v for v in V]

def _hermite_base(early_termination, R, A, u, D):
    ### this should be in cython. 
    r"""
    
    Base case of Hermite-Pade (iterative version):

    INPUT:

    - ``early_termination`` -- whether or not to abort early once a solution vector is found
    - ``R`` -- a univariate polynomial ring over a field ``k``
    - ``A`` -- a matrix, encoded as list of lists, with elements in ``R``, encoded as coefficient lists
    - ``u`` -- an integer
    - ``D`` -- a list with ``len(A[0])`` elements

    OUTPUT: a polynomial square matrix ``V`` of size ``len(A[0])`` such that
    
    - :math:`\det(V)\neq0`
    - :math:`A\cdot V = 0 \bmod x^u`
    - :math:`\max_{i,j}( \deg(V_{i,j}) + D[i] )` is as small as possible

    and a boolean value ``done`` indicating whether early termination has been detected during the
    calculation.

    .. NOTE::
    
      - The matrix ``A`` will be overwritten during the calculation
      - The ``D`` vector will be updated to :math:`[\max_i( \deg V_{i,j}) + D[i] ), j=0,\dots,n-1]`
    """

    n = len(A); m = len(A[0]); x = R.gen(); one = R.one(); zero = R.zero()
    V = [ [ (one if i==j else zero) for i in xrange(m) ] for j in xrange(m) ]
    infinity = max(D) + u + 1 # larger than the largest possible value in D throughout this calculation
    one = R.base_ring().one(); zero = R.base_ring().zero(); 

    for k in xrange(u):
        # consider the coefficient of x^k in the entries of A
        if early_termination and k < u - 1:
            # if V[j] is already a solution vector, then the j-th row of A must be zero.
            candidates = filter(lambda j : not any(A[i][j][k] for i in xrange(n)), range(m))
            # for the candidates, check whether also the higher degree coefficients are zero
            candidates = filter(lambda j : not any(A[i][j][l] for i in xrange(n) for l in xrange(k+1, u)), candidates)
            if len(candidates) > 0:
                return Matrix(R, [[v[c] for c in candidates] for v in V ]), True
        for i in xrange(n):
            row = A[i]
            # pivot: among the indices j where A[i,j]!=0, pick one where D[j] is minimal
            piv = -1; d = infinity
            for j in xrange(m):
                if D[j] < d and row[j][k] != zero:
                    piv = j; d = D[j]
            if piv == -1:
                continue # kth coeff of ith row of A is already zero
            # elimination
            piv_element = -one/row[piv][k]; 
            for j in xrange(m):
                if j != piv and row[j][k] != zero:
                    q = piv_element*row[j][k]
                    for v in V:
                        v[j] += q*v[piv]
                    for l in xrange(n):
                        Alj = A[l][j]; Alpiv = A[l][piv]
                        for c in xrange(k, u):
                            Alj[c] += q*Alpiv[c]
            # multiplication and degree update
            for v in V:
                v[piv] *= x
            for a in A:
                a[piv].insert(0, 0)
            D[piv] += 1

    return Matrix(R, V), False

def _hermite_rec(early_termination, R, A, cut, offset, infolevel):
    r"""
    Recursive step of Hermite-Pade (divide and conquer):

    INPUT:

    - ``early_termination`` -- whether or not to abort as soon as one solution vector was found
    - ``R`` -- a univariate polynomial ring over a field
    - ``A`` -- a matrix with elements in ``R``
    - ``cut`` -- an integer
    - ``offset`` -- a vector of integers
    - ``infolevel`` -- integer indicating the desired verbosity

    OUTPUT: a polynomial square matrix ``V`` of size ``A.ncols()`` with

    - :math:`\det(V)\neq0`
    - :math:`A\cdot V=0\bmod x^{\mathrm{cut}}`
    - :math:`\max_{i,j}( \deg(V_{i,j}) + \mathrm{offset}[i] )` is minimal

    and a boolean value ``done`` indicating whether early termination has been detected somewhere
    down the recursion tree. 

    .. NOTE::

       The ``offset`` vector will be updated to :math:`[\max_i( \deg(V_{i,j}) + \mathrm{offset}[i] ), j=0,\dots,n-1]`
    """

    # 0. if cut is small, switch to direct method
    if cut <= 64:
        # B = low degree coeffs of A.
        _info(infolevel, "base case: switching to direct method.")
        B = [ [ A[i,j].coeffs()[:cut] for j in xrange(A.ncols()) ] for i in xrange(A.nrows()) ]
        z = R.base_ring().zero()
        for row in B:
            for pol in row:
                for k in xrange(cut - len(pol)):
                    pol.append(z)
        return _hermite_base(early_termination, R, B, cut, offset)
    
    # 1. write A = A0 + A1 x^ceil(k/2) with deg(A0), deg(A1) < ceil(k/2)
    cut2 = int(math.ceil(cut/2))

    _info(infolevel, "decending into first recursive call...")
    # 2. compute V0 such that A0*V0 == 0 mod x^ceil(k/2) recursively
    V0, done = _hermite_rec(early_termination, R, A, cut2, offset, _alter_infolevel(infolevel, -1, 1))
    _info(infolevel, "...done")
    if done: # we don't check for false alarm
        return V0, done

    # 3. set B=A1*V0 rem x^ceil(k/2)
    B = (A*V0).apply_map(lambda p : p.shift(-cut2))
    
    # 4. compute V1 such that B*V1 == 0 mod x^ceil(k/2) recursively
    _info(infolevel, "decending into second recursive call...")
    V1, done = _hermite_rec(early_termination, R, B, cut - cut2, offset, _alter_infolevel(infolevel, -1, 1))
    _info(infolevel, "...done")
    
    # 5. return V0*V1
    return V0*V1, done

def kronecker(subsolver, presolver=None):
    r"""
    Creates a solver for matrices of multivariate polynomials over some domain `K`,
    based on a given solver for matrices over univariate polynomials

    INPUT:

    - ``subsolver`` -- a solver for univariate polynomial matrices over `K`
    - ``presolver`` -- a solver for univariate polynomial matrices over prime fields. If ``None`` is given,
      the ``presolver`` will be set to ``subsolver``

    OUTPUT:

    - a solver for matrices with entries in `K[x,y,...]` where K is some domain such that
      the given ``subsolver`` can solve matrices in `K[x]`. If the solver is called without
      ``degree`` information about the solution vectors, the ``presolver`` is called on various
      matrices with entries in `GF(p)[x]` to determine the degrees. In this case if `K` is not
      a prime field, its elements must allow for coercion to prime field elements.

    EXAMPLE:

    ::

       sage: A = MatrixSpace(GF(1093)['x','y'], 4, 7).random_element(degree=3)
       sage: mysolver = kronecker(gauss())
       sage: V = mysolver(A)
       sage: A*V[0]
       (0, 0, 0, 0)

    ALGORITHM:

    #. If applied to a matrix of univariate polynomials, the function will delegate the whole problem to the
       subsolver, and return its result. 
    #. When no degree information is specified, the presolver will be applied to homomorphic images of the matrix
       with all the variables except one set to some ground field element and reduce the coefficients modulo a
       word size prime (unless the coefficient field already is a prime field). The degrees seen in these result
       are taken as the degrees of the respective variables in the final result.
    #. By Kronecker-substitution, reduce the task to a nullspace-problem for a matrix over ``K[x]``, apply the
       subsolver to it, undo the Kronecker-substitution, and return the result.     

    """
    def kronecker_solver(mat, degrees=[], infolevel=0):
        """See docstring of kronecker() for further information."""
        return _kronecker(subsolver, presolver, mat, degrees, infolevel)
    return kronecker_solver

def _kronecker(subsolver, presolver, mat, degrees, infolevel):
    r"""
    Internal version of nullspace.kronecker_.
    """
    _launch_info(infolevel, "kronecker", dim=mat.dimensions(), domain=mat.parent().base_ring())

    R = mat.parent().base_ring(); x = R.gens(); x0 = x[0]
    K = R.base_ring()
    if len(x) == 1:
        return subsolver(mat, degrees=degrees, infolevel=_alter_infolevel(infolevel, -1, 1))
    if not K.is_finite() and not K == ZZ:
        raise TypeError    
    if not R == PolynomialRing(K, x):
        raise TypeError 
    
    # 1. for each variable (except the last), determine the maximal degree of the nullspace basis
    Kimg = GF(pp(MAX_MODULUS)) if K.characteristic() == 0 else K
    Rimg = Kimg[x0]
    if len(degrees) < len(x) - 1:
        _info(infolevel, "probing for output degrees...", alter = -1)
        if presolver == None:
            presolver = subsolver
        degrees = []; evaluator = dict( (x[j], Rimg(59 + 17*j)) for j in xrange(len(x)) )
        for i in xrange(len(x)):
            myev = evaluator.copy(); myev[x[i]] = Rimg(x0)
            sol = presolver(mat.apply_map(lambda p: p.subs(myev), Rimg), infolevel=_alter_infolevel(infolevel, -1, 1))
            if len(sol) == 0:
                return []
            degrees.append(max(max(p.degree() for p in v) for v in sol) + 3)
        _info(infolevel, "... done. Expecting degree vector to be ", degrees, alter = -1)
    else:
        degrees = [ d + 3 for d in degrees ]

    # 2. kronecker substitution: x[i] |--> x[0]^(deg[0]*deg[1]*...*deg[i-1])
    # the first variable is translated by some offset in order to make it unlikely that
    # we get solutions like (x^2,y) which after kronecker substitution become (x^2,x^1000)
    # but are returned by the subsolver as (1,x^998).
    Rimg = K[x0]; z = K.zero(); shift = {x0 : R(x0 - 5556)}; 
    def phi(poly): ##### MOST TIME IS SPENT IN THIS FUNCTION (in particular by .subs and .dict)
        terms = {}; poly = poly.subs(shift).dict(); 
        for exp in poly.keys():
            n = exp[0]; d = 1;
            for i in xrange(len(degrees) - 1):
                d *= degrees[i]; n += d*exp[i+1]
            terms[n] = poly[exp] + (terms[n] if terms.has_key(n) else z)
        return Rimg(terms)
    
    # 3. subsolver in k[x]
    sol = subsolver(mat.apply_map(phi, Rimg), degrees=[prod(degrees)], infolevel=_alter_infolevel(infolevel, -1, 1))

    # 4. undo kronecker substitution x^u |--> prod(x[i]^(u quo degprod[i-1] rem deg[i]), i=0..len(x))
    _info(infolevel, "undo substitution.", alter = -1)
    unshift = {x[0] : R(x[0] + 5556)}
    def unphi(p):
        exp = [0 for i in xrange(len(x))]; d = {}
        for c in p.coeffs():
            if c != z:
                d[tuple(exp)] = c
            exp[0] += 1
            for i in xrange(len(degrees) - 1):
                if exp[i] >= degrees[i]:
                    exp[i] = 0; exp[i+1] += 1
                else:
                    break
        return R(d).subs(unshift)
    sol = [ v.apply_map(unphi, R) for v in sol ]

    if R.base_ring().is_field():
        one = R.base_ring().one(); zero = R.zero()
        for v in sol:
            j = 0
            while v[j] == zero:
                j += 1
            piv = one/v[j].lc()
            for k in xrange(j, len(v)):
                v[k] *= piv
    
    # 5. done
    return sol


def lagrange(subsolver, start_point=10, ncpus=1):
    r"""
    Creates a solver for matrices of univariate polynomials or rational functions over some field `K`,
    based on a given solver for matrices over `K`, using evaluation+interpolation.

    INPUT:

    - ``subsolver`` -- a solver for matrices over `K`
    - ``start_point`` -- first evaluation point to be used
    - ``ncpus`` -- maximum number of cpus that may be used in parallel by the solver (default=1).

    OUTPUT:

    - a solver for matrices with entries in `K[x]` or `K(x)` where K is some field such that the
      given ``subsolver`` can solve matrices with entries in `K`.

    EXAMPLE:

    ::

       sage: A = MatrixSpace(GF(1093)['x'], 4, 7).random_element(degree=3)
       sage: my_solver = lagrange(sage_native)
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0)

    ALGORITHM:

    #. For ``x`` replaced by ``start_point``, ``start_point+1``, ``start_point+2``, ..., compute the
       nullspace by the given ``subsolver``.
    #. Use (fast) interpolation and rational reconstruction to combine the images to a nullspace basis for
       the original matrix over ``K[x]``

    .. NOTE::

       - The solver raises a ``ValueError`` if the ground field is too small.
       - It is assumed that the ``subsolver`` returns a normal form of the nullspace basis, so that results of
         calls for different evaluations have some common preimage in the matrices over `K(x)`

    """
    def lagrange_solver(mat, degrees=[], infolevel=0):
        """See docstring of lagrange() for further information."""
        return _lagrange(subsolver, start_point, ncpus, mat, degrees, infolevel)
    return lagrange_solver

def _lagrange(subsolver, start_point, ncpus, mat, degrees, infolevel):
    r"""
    Internal version of nullspace.lagrange_.
    """
    n, m = mat.dimensions(); R = mat.parent().base_ring(); 
    _launch_info(infolevel, "lagrange", dim=(n, m), domain=R)

    if R is R.fraction_field():
        rational = True
        R = R.base()
    else:
        rational = False

    x = R.gen(); K = R.base_ring(); char = K.characteristic(); one = R.one(); zero = R.base_ring().zero(); 
    
    degree_known = (len(degrees) > 0) 
    bound =  2*degrees[0] + 3 if degree_known else 16

    if char > 0 and char < start_point + bound:
        raise ValueError, "not enough evaluation points"

    points = map(lambda p: R(start_point + p), range(bound))
    M = product_tree(x, points, 0, bound); mod = M[0]; Mprime = []
    multipoint_evaluate(mod.derivative(), points, 0, bound, M, Mprime)

    if rational:
        mymat = mat.apply_map(lambda p: (p.numerator() * p.denominator().inverse_mod(mod)) % mod, R)
    else:
        mymat = mat

    try:
        V = _lagrange_rec(mod, mymat, Mprime, 0, bound, M, subsolver, _alter_infolevel(infolevel, -1, 1))
    except NoSolution:
        return []
    
    if degree_known:
        
        # rational reconstruction and normalization
        split = bound//2; 
        for v in V:
            d = one
            for p in v:
                d = d.lcm(p.rational_reconstruct(mod, split, split)[1])
            for j in xrange(len(v)):
                v[j] = (v[j]*d) % mod
            j = 0
            while v[j] == zero:
                j += 1
            piv = one/_leading_coefficient(v[j])
            for k in xrange(j, m):
                v[k] *= piv
                
        return [ vector(R, v) for v in V ]

    # double the number of evaluation points until a solution is reached.
        
    modulus = mod; done = False

    while not done:

        start_point += bound; bound *= 2
        _info(infolevel, "Taking", bound, "more interpolation points...", alter = -1)
        if start_point + bound > char:
            raise ValueError, "not enough evaluation points"
        points = map(lambda p: R(start_point + p), range(bound))
        M = product_tree(x, points, 0, bound); mod = M[0]; Mprime = []
        multipoint_evaluate(mod.derivative(), points, 0, bound, M, Mprime)

        if rational:
            try:
                mymat = mat.apply_map(lambda q: (q.numerator() * q.denominator().inverse_mod(mod)) % mod, R)
            except ValueError:
                _info(infolevel, "Unlucky evaluation point encountered.", alter = -1)
                if bound > 128:
                    bound /= 4; start_point -= bound
                continue
        else:
            mymat = mat

        Vnew = _lagrange_rec(mod, mymat, Mprime, 0, bound, M, subsolver, _alter_infolevel(infolevel, -2, 1))

        _info(infolevel, "Combining with previous partial solution...", alter = -1)
        inv = xgcd(modulus, mod)[1]*modulus
        for i in xrange(len(V)):
            for j in xrange(len(V[i])):
                V[i][j] = V[i][j] + (Vnew[i][j] - V[i][j])*inv
        modulus *= mod

        # at this point V is correct mod 'modulus'

        # check for termination
        _info(infolevel, "Checking for termination...", alter = -1)
        split = modulus.degree()//2; W = []; done = True
        for v in V:
            try:
                d = one
                for p in v:
                    d = d.lcm(p.rational_reconstruct(modulus, split, split)[1])
            except ValueError:
                done = False; break
            w = [ (p*d) % modulus for p in v ]; W.append(w)
            if any(mat*vector(R, w)):
                done = False; break
            j = 0
            while w[j] == zero:
                j += 1
            piv = one/_leading_coefficient(w[j])
            for k in xrange(j, m):
                w[k] *= piv
    
    return [ vector(R, v) for v in W ]
            

def product_tree(x, points, a, b):
    # if points=[1,2,3,4,5,6,7,8], this constructs
    #  ( (x-1)...(x-8),  ( (x-1)..(x-4), ( (x-1)(x-2), .... ),  ( (x-3)(x-4), ....) ) ,
    #                    ( (x-5)..(x-8), ( (x-5)(x-6), .... ),  ( (x-7)(x-8), ....) ) )

    if b - a == 1:
        return (x - points[a], None, None)
    
    split = int(math.ceil((a+b)/2))
    left = product_tree(x, points, a, split)
    right = product_tree(x, points, split, b)
    return (left[0]*right[0], left, right)

def multipoint_evaluate(poly, points, a, b, product_tree, L):
    # alg 10.5 from vzgathen/gerhard

    if b - a == 1:
        L.append(poly[0])
        return
    
    split = int(math.ceil((a+b)/2))
    
    r0 = poly % product_tree[1][0]
    multipoint_evaluate(r0, points, a, split, product_tree[1], L)
    
    r1 = poly % product_tree[2][0]
    multipoint_evaluate(r1, points, split, b, product_tree[2], L)

def _lagrange_base(mat, MprimeA, subsolver, infolevel):
    # base case of interpolation solver (a separate function in order to facilitate profiling)

    R = mat[0][0].parent(); K = R.base_ring()
    V = subsolver(Matrix(K, [[ p[0] for p in v ] for v in mat]), infolevel=infolevel)
    if len(V) == 0:
        raise NoSolution
    return [[ R(p/MprimeA) for p in v ] for v in V]

def _lagrange_rec(mod, mat, Mprime, a, b, product_tree, subsolver, infolevel):
    # recursive step of interpolation solver

    if b - a == 1:
        return _lagrange_base(mat, Mprime[a], subsolver, infolevel)
    
    split = int(math.ceil((a + b)/2))
    
    M_left = product_tree[1][0]; M_right = product_tree[2][0]

    mymat = [ [ p % M_left for p in v ] for v in mat ] 
    V_left = _lagrange_rec(mod, mymat, Mprime, a, split, product_tree[1], subsolver, infolevel)
    del mymat

    mymat = [ [ p % M_right for p in v ] for v in mat ]
    V_right = _lagrange_rec(mod, mymat, Mprime, split, b, product_tree[2], subsolver, infolevel)
    del mymat
    
    return [ map(lambda v_l, v_r: M_right*v_l + M_left*v_r, V_left[i], V_right[i]) for i in xrange(len(V_left)) ]

def galois(subsolver, max_modulus=MAX_MODULUS, proof=False):
    r"""
    Creates a subsolver based on chinese remaindering for matrices over `K[x]` or `K[x,y,..]` where
    `K` is a single algebraic extension of `QQ`

    INPUT:

    - ``subsolver`` -- a solver for matrices over `GF(p)[x]` (if the original matrix contains univariate
      polynomails) or `GF(p)[x,y,...]` (if it contains multivariate polynomials)
    - ``max_modulus`` -- a positive integer. The solver will iterate over the primes less than this number
      in decreasing order. Defaults to the largest word size integer for which we expect Sage to use hardware
      arithmetic.
    - ``proof`` -- a boolean value. If set to ``False`` (default), a termination is only tested in a
      homomorphic image, which saves much time but may, with a very low probability, lead to a wrong output.

    OUTPUT:

    - a solver for matrices with entries in `QQ(\alpha)[x,...]` based on a ``subsolver`` for matrices with
      entries in `GF(p)[x,...]`.

    EXAMPLES:

    ::

       sage: R.<x> = QQ['x']
       sage: K.<a> = NumberField(x^3-2, 'a')
       sage: A = MatrixSpace(K['x'], 4, 5).random_element(degree=3)
       sage: my_solver = galois(gauss())
       sage: ##V = my_solver(A) 
       sage: ##A*V[0]
       ##(0, 0, 0, 0)

    ALGORITHM:

    #. If the coefficient domain is a finite field, the problem is delegated to the subsolver and we return
       whatever we obtain from it.
    #. For various word-size primes `p`, reduce the coefficients in ``mat`` modulo `p` and call the
       subsolver on the resulting matrix. Primes are chosen in such a way that the minimal polynomial of
       the generator of the number field splits into linear factors modulo `p`.
    #. The generator of the number field is mapped in turn to each
       of the roots of the linear factors of the minimal polynomial mod `p`, the results are then combined
       by interpolation to a polynomial over `GF(p)`, and chinese remaindering and rational reconstruction
       are used to lift it back to the original number field. 
    #. If this solution candidate is correct, return it and stop. If ``proof`` is set to ``True``, this check
       is performed rigorously, otherwise (default) only modulo some new prime.
    #. If the solution candidate is not correct, consider some more primes and try again.

    """
    def galois_solver(mat, degrees=[], infolevel=0):
        """See docstring of galois() for further information."""
        return _galois(subsolver, max_modulus, proof, mat, degrees, infolevel)
    return galois_solver

def _galois(subsolver, max_modulus, proof, mat, degrees, infolevel):
    raise NotImplementedError    

def cra(subsolver, max_modulus=MAX_MODULUS, proof=False, ncpus=1):
    r"""
    Creates a subsolver based on chinese remaindering for matrices over `K[x]` or `K[x,y,..]` where
    `K` is `ZZ` or `QQ` or `GF(p)`.

    INPUT:

    - ``subsolver`` -- a solver for matrices over `GF(p)[x]` (if the original matrix contains univariate
      polynomails) or `GF(p)[x,y,...]` (if it contains multivariate polynomials)
    - ``max_modulus`` -- a positive integer. The solver will iterate over the primes less than this number
      in decreasing order. Defaults to the largest word size integer for which we expect Sage to use hardware
      arithmetic.
    - ``proof`` -- a boolean value. If set to ``False`` (default), a termination is only tested in a
      homomorphic image, which saves much time but may, with a very low probability, lead to a wrong output.
    - ``ncpus`` -- number of cpus that may be used in parallel by the solver (default=1).

    OUTPUT:

    - a solver for matrices with entries in `K[x,...]` based on a ``subsolver`` for matrices with
      entries in `GF(p)[x,...]`.

    EXAMPLES:

    ::

       sage: A = MatrixSpace(ZZ['x', 'y'], 4, 7).random_element(degree=3)
       sage: my_solver = cra(kronecker(gauss()))
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0)

    ALGORITHM:

    #. If the coefficient domain is a finite field, the problem is delegated to the subsolver and we return
       whatever we obtain from it.
    #. For various word-size primes `p`, reduce the coefficients in ``mat`` modulo `p` and call the
       subsolver on the resulting matrix.
    #. Using the Chinese Reminder Algorithm and rational reconstruction, combine these results to a solution
       candidate with integer coefficients.
    #. If this solution candidate is correct, return it and stop. If ``proof`` is set to ``True``, this check
       is performed rigorously, otherwise (default) only modulo some prime.
    #. If the solution candidate is not correct, consider some more primes and try again.

    """
    def cra_solver(mat, degrees=[], infolevel=0) :
        """See docstring of cra() for further information."""
        return _cra(subsolver, max_modulus, proof, ncpus, mat, degrees, infolevel)
    return cra_solver

def _cra(subsolver, max_modulus, proof, ncpus, mat, degrees, infolevel):
    """
    Internal version of nullspace.cra_ 
    """
    R = mat.parent().base_ring(); x = R.gens(); K = R.base_ring()
    _launch_info(infolevel, "cra", dim=mat.dimensions(), domain=R)    

    if K.is_prime_field():
        return subsolver(mat, degrees=degrees)
    
    if not (K == QQ or K == ZZ):
        raise TypeError

    R = ZZ[x]

    # precomputed material used in the homomorphic termination check
    check_prime = pp(max_modulus)
    check_eval = dict(zip(x, [ 17*j + 13 for j in xrange(len(x)) ]))
    check_field = GF(check_prime)
    check_mat = mat.apply_map( lambda pol : pol.substitute(check_eval), check_field )
    
    V = None; M = 1; p = check_prime

    if ncpus > 1:
        @parallel(ncpus=ncpus)
        def forked_subsolver(Zp):
            return subsolver(mat.apply_map(Zp, Zp), degrees=degrees, infolevel=_alter_infolevel(infolevel, -2, 1))
        
    while True:

        _info(infolevel, math.floor(math.log(M, 10)/2), " decimal digits completed.", alter = -1)
        
        # compute solution(s) modulo p
        try:
            if ncpus == 1:
                p = pp(p); Zp = GF(p)[x]
                 # MAIN WORK, SEQUENTIALLY
                Vp = subsolver(mat.apply_map(Zp, Zp), degrees=degrees, infolevel=_alter_infolevel(infolevel, -2, 1))
                m = p
            else:
                Zp = []
                while len(Zp) < ncpus:
                    p = pp(p); Zp.append(GF(p)[x])
                Vpp = [ (u[0][0].characteristic(), v) for (u, v) in forked_subsolver(Zp) ] # MAIN WORK, DONE IN PARALLEL
                # combine them, provided all the Vp have the same length (viz. none of the primes was unlucky)
                primes = [ u[0] for u in Vpp ];  Vpp = [ u[1] for u in Vpp ];
                if any( len(u) - len(Vpp[0]) for u in Vpp ):
                    raise ArithmeticError # solution spaces have different sizes
                basis = map(R, CRT_basis(primes))
                Vp = [ v.apply_map(lambda u: basis[0]*R(u)) for v in Vpp[0] ]
                for i in xrange(1, len(Vpp)):
                    for j in xrange(len(Vp)):
                        Vp[j] += Vpp[i][j].apply_map(lambda u: basis[i]*R(u))
                m = prod(primes)
                # now Vp is a solution mod m, and m is the product of 'ncpus' many primes.
        except ArithmeticError: # unlucky prime may cause division by zero when mapping QQ --> Z_p
            _info(infolevel, "unlucky modulus ", m, " discarded (division by zero)", alter = -1)
            continue

        # degenerate situations
        if len(x) == 1:
            true_degrees = [max(max(e.degree() for e in v) for v in Vp)]
        else:
            true_degrees = [max(max(e.degree(x0) for e in v) for v in Vp) for x0 in x]
            
        if len(degrees) != len(x):
            degrees = [-1 for i in x]
        
        if V == None or len(V) > len(Vp) or any(degrees[i] > true_degrees[i] for i in xrange(len(x))):
            # initialization, or all previous primes were unlucky
            if len(Vp) == 0:
                return []
            V = [ [R(e) for e in v] for v in Vp ]; M = m;
            degrees = true_degrees
            _info(infolevel, "expecting solution degrees ", degrees, alter = -1)
        elif len(V) < len(Vp): # this prime is unlucky, skip it
            _info(infolevel, "unlucky modulus ", m, " discarded (dimension defect)", alter = -1)
            continue
        elif any(degrees[i] < true_degrees[i] for i in xrange(len(x))): # this prime is unlucky, skip it
            _info(infolevel, "unlucky modulus ", m, " discarded (degree missmatch: ", true_degrees, ")", alter = -1)
            continue
        
        # combine the new solution with the known partial solution
        if M != m: # (i.e., always except in the first iteration)
            (g, M0, p0) = xgcd(m, M); (M0, p0) = (R(M0*m), R(p0*M)); M *= m
            for i in xrange(len(V)):
                Vi = V[i]; Vpi = Vp[i]
                for j in xrange(len(V[i])):
                    Vi[j] = Vi[j]*M0 + R(Vpi[j])*p0
        
        # rational reconstruction and check for termination
        try:
            sol = []; m = M//2
            for v in V:
                d = ZZ.one()
                for e in v:
                    for c in e.coefficients():
                        d *= (d*c).rational_reconstruction(M).denominator()
                w = vector(R, [e.map_coefficients( lambda c: ((d*c + m) % M) - m , ZZ ) for e in v ])
                if (not proof and any(check_mat * vector(check_field, [e.substitute(check_eval) for e in w]))) or \
                       (proof and any(mat * w) ):
                    raise ArithmeticError # more primes needed
                sol.append(w)
            return sol # if no error was raised for any of the v in V, then we are done        
        except (ValueError, ArithmeticError):
            pass

def newton(subsolver, inverse=lambda mat:mat.inverse()):
    r"""
    Constructs a solver based on x-adic lifting for matrices with entries over `K[x]` where `K` is a field.

    INPUT:

    - ``subsolver`` -- a solver for matrices over `K`
    - ``inverse`` -- a function for computing the inverse of a given regular square matrix with entries in `K`.
      Defaults to ``lambda mat : mat.inverse()``

    OUTPUT:

    - a solver for matrices with entries in `K[x]` based on a ``subsolver`` for matrices with
      entries in `K`.

    EXAMPLES:

    ::

       sage: A = MatrixSpace(GF(1093)['x'], 4, 7).random_element(degree=3)
       sage: my_solver = newton(sage_native)
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0)

    ALGORITHM:

    #. First call the subsolver on a homomorphic image to get an idea about the rank of the matrix and the
       shape of the reduced echolon form of the solution matrix.
    #. Based on this information, rewrite the original homogeneous system :math:`A\cdot X=0` into an
       inhomogeneous system :math:`U\cdot Y+V=0` where :math:`U` is invertible.
    #. Compute the inverse :math:`U(x=0)^{-1}` of :math:`U(x=0)` using the given ``inverse`` function.
    #. Using Newton-iteration, lift :math:`U(x=0)^{-1}` to a polynomial matrix :math:`W` with
       :math:`U\cdot W=1\bmod x^{2k}` where :math:`k` exceeds the specified degree bound (or a worst-case
       bound, if no degree bound was specified).
    #. Compute :math:`W\cdot V`, rewrite it as nullspace basis mod :math:`x^{2k}`, apply rational
       reconstruction, clear denominators, and return the result. 

    .. NOTE::

       It is assumed that the subsolver returns a nullspace basis which is normalized so that when the
       basis vectors are viewed as the columns of a matrix, this matrix contains every possible unit
       vector as row
    """
    def newton_solver(mat, degrees=[], infolevel=0) :
        """See docstring of newton() for further information."""
        return _newton(subsolver, inverse, mat, degrees, infolevel)
    return newton_solver

def _newton(subsolver, inverse, mat, degrees, infolevel):
    r"""
    Internal version of nullspace.newton_.
    """
    n, m = mat.dimensions(); R = mat.parent().base_ring(); x = R.gen();
    matdeg = max( mat[i,j].degree() for i in xrange(n) for j in xrange(m) )
    _launch_info(infolevel, "newton", dim = (n, m), deg = matdeg, domain = R)
    
    if len(degrees) < 1:
        bound = 2*(min(n, m) + 1)*matdeg 
    else:
        bound = 2*degrees[0] + 1

    # move some "arbitrary" point to the origin
    from sage.categories.homset import Hom
    mat = mat.apply_map(Hom(R, R)([x + 1324]))

    # transform homogeneous system to inhomogeneous one
    K = R.base_ring(); one = K.one(); zero = K.zero()
    V0 = subsolver(mat.apply_map(lambda p: p[0], K), infolevel=_alter_infolevel(infolevel, -1, 1))
    dim = len(V0); rank = m - dim
    if dim == 0:
        return [vector(R, v) for v in V0]

    idxA, idxB = _select_regular_square_submatrix(V0, n, m, dim, one, zero)

    A = Matrix(R, [ [ v[idxA[i]] for i in xrange(len(idxA)) ] for v in mat ] )
    B = Matrix(R, [ [ v[idxB[i]] for i in xrange(len(idxB)) ] for v in mat ] )
    X = Matrix(R, [ [ R(v[idxA[i]]) for i in xrange(len(idxA)) ] for v in V0 ] ).transpose()
    # have: A*X + B = 0 mod x^1. want: A*X + B = 0 mod x^bound

    if n > rank:
        A = MatrixSpace(R, rank, n).random_element()*A # now A is a square matrix
    Ainv = inverse(A.apply_map(lambda p: p[0], K))
    Ainv = Ainv.change_ring(R)

    # lifting of Ainv
    xk = x; k = 1; 
    while k < bound:
        _info(infolevel, "lifting completed mod ", xk, alter = -1)
        #Ainv -= (Ainv*(A*Ainv).apply_map(lambda p: p.shift(-k))).apply_map(lambda p: (p%xk).shift(k))
        U = A._mul_(Ainv) # MOST COSTLY STEP
        for i in xrange(rank):
            for j in xrange(rank):
                U[i,j] = U[i,j].shift(-k)
        U = Ainv._mul_(U) # MOST COSTLY STEP
        for i in xrange(rank):
            for j in xrange(rank):
                Ainv[i,j] -= (U[i,j]%xk).shift(k)        
        k *= 2; xk *= xk
    _info(infolevel, "lifting completed.", alter = -1)

    # solution of the system
    X = -Ainv._mul_(B)
    
    # put back unit vectors
    zero = R.zero(); one = R.one(); V = [ [zero for i in xrange(m)] for j in xrange(dim) ]
    for i in xrange(dim):
        for j in xrange(len(idxA)):
            V[i][idxA[j]] = X[j][i]
        V[i][idxB[i]] = one

    # rational reconstruction and undo offset
    phi = Hom(R, R)([x - 1324]); split = bound//2; xk = x**bound
    for v in V:
        d = one
        for p in v:
            d = d.lcm(p.rational_reconstruct(xk, split, split)[1])
        for j in xrange(len(v)):
            v[j] = phi((v[j]*d) % xk)
        j = 0
        while v[j].degree() < 0:
            j += 1
        piv = one/_leading_coefficient(v[j])
        for k in xrange(j, m):
            v[k] *= piv
    
    return [ vector(R, v) for v in V ]

def clear(subsolver):
    """
    Constructs a solver which clears denominators in a given matrix over `FF(R)[x..]` or `FF(R[x..])`
    and then calls a subsolver on a matrix over `R[x..]`

    INPUT:

    - ``subsolver`` -- a solver for matrices over `R[x..]` 

    OUTPUT:

    - a solver for matrices over `FF(R)[x..]` or `FF(R[x..])`

    EXAMPLE:

    ::

       sage: A = MatrixSpace(ZZ['x'].fraction_field(), 4, 5).random_element()
       sage: my_solver = clear(gauss())
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0)

       sage: A = MatrixSpace(QQ['x'], 4, 5).random_element()
       sage: my_solver = clear(gauss())
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0)

    For turning a matrix over `QQ(x)` to one over `ZZ[x]`, you need to ``clear`` twice: once the
    denominator of the rational functions, and then once more the denominators of the coefficients

    ::
    
       sage: A = MatrixSpace(QQ['x'].fraction_field(), 4, 5).random_element()
       sage: my_solver = clear(clear(gauss()))
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0)

    ALGORITHM: clears denominators and then applies the subsolver.

    .. WARNING::

       For unstructured matrices, clearing denominators may significantly increase the size
       of the system. In such situations, consider using nullspace.lagrange_ .
      
    """
    def clear_solver(mat, degrees=[], infolevel=0):
        """See docstring of clear() for further information."""
        return _clear(subsolver, mat, degrees, infolevel)
    return clear_solver

def _clear(subsolver, mat, degrees, infolevel):

    R = mat.parent().base_ring()
    _launch_info(infolevel, "clear", dim=mat.dimensions(), domain=R)

    if R.fraction_field() == R:
        # entries live in some fraction field.
        try:
            newR = R.ring_of_integers()
        except AttributeError:
            newR = R.ring()
        def common_denominator(row):
            den = newR.one()
            for p in row:
                den = den.lcm(p.denominator())
            return den
    elif R.base_ring().fraction_field()[R.gens()] == R:
        # entries are polynomials over some fraction field
        oldK = R.base_ring(); # e.g. QQ
        try:
            newK = oldK.ring_of_integers(); # e.g. ZZ
        except AttributeError:
            newK = oldK.ring()
        newR = newK[R.gens()]; # e.g. ZZ[x]
        def common_denominator(row):
            den = newK.one()
            try:
                for p in row:
                    for c in p.coeffs():
                        den = den.lcm(c.denominator())
            except AttributeError:
                for p in row:
                    for c in p.coefficients():
                        den = den.lcm(c.denominator())
            return newR(den)
    else:
        # unexpected ground domain
        raise TypeError

    newmat = []
    for row in mat:
        den = common_denominator(row)
        newmat.append( row.apply_map(lambda p: den*p, newR) )
    newmat = Matrix(newR, newmat)
    
    return subsolver(newmat, degrees=degrees, infolevel=_alter_infolevel(infolevel, -1, 1))

def merge(subsolver):
    """
    Constructs a solver which first merges towers of polynomial or rational function extensions
    into a single one and then applies the subsolver.

    INPUT:

    - ``subsolver`` -- a solver for matrices over the target ring

    OUTPUT:

    - a solver for matrices over base rings which are obtained from some ring `R` by extending
      twice by polynomials or rational functions, i.e., `R=B[x..][y..]` or `R=F(B[x..])[y..]`
      or `R=F(B[x..][y..])` or `R=F(F(B[x..])[y..])`. In the first case, the target ring will
      be `B[x..,y..]`, in the remaining cases it will be `F(B[x..,y..])`.
      If `R` is not of one of the four forms listed above, the matrix is passed to the
      subsolver without modification. 

    EXAMPLES::

      sage: my_solver = merge(kronecker(gauss()))
      sage: A = MatrixSpace(ZZ['x']['y'], 3, 4).random_element()
      sage: V = my_solver(A)
      sage: A*V[0]
      (0, 0, 0)
      sage: my_solver = merge(clear(kronecker(gauss())))
      sage: A = MatrixSpace(ZZ['x'].fraction_field()['y'], 3, 4).random_element()
      sage: V = my_solver(A)
      sage: A*V[0]
      (0, 0, 0)

    """
    def merge_solver(mat, degrees=[], infolevel=0):
        """See docstring of merge() for further information."""
        return _merge(subsolver, mat, degrees, infolevel)
    return merge_solver

def _merge(subsolver, mat, degrees, infolevel):

    _launch_info(infolevel, "merge", dim=mat.dimensions(), domain=mat.parent().base_ring())

    try: 

        R = mat.parent().base_ring()

        B = R; field = B.is_field()
        upper_gens = B.gens()
        B = B.base_ring()
        field = field or B.is_field()
        lower_gens = B.gens()

        B = PolynomialRing(B.base_ring(), lower_gens + upper_gens)

        if field:
            B = B.fraction_field()

        mat = mat.change_ring(B)

        _info(infolevel, "changing base ring to ", str(B), alter=-1)

    except: # ring was not of expected form, or conversion failed
        _info(infolevel, "leaving base ring as it is: ", str(R), alter=-1)

    return subsolver(mat, degrees=degrees, infolevel=_alter_infolevel(infolevel, -2, 1))

def quick_check(subsolver, modsolver=sage_native, modulus=MAX_MODULUS):
    """
    Constructs a solver which first tests in a homomorphic image whether the nullspace is empty
    and applies the subsolver only if it is not.

    INPUT:

    - ``subsolver`` -- a solver
    - ``modsolver`` -- a solver for matrices over `GF(p)`, defaults to `sage_native`_
    - ``modulus`` -- modulus used for the precomputation in the homomorphic image if `R.characteristic()==0`.
      If this number is not a prime, it will be replaced by the next smaller integer which is a prime.

    OUTPUT:

    - a solver for matrices over `K[x..]` or `K(x..)` with `K` one of `ZZ`, `QQ`, `GF(p)`, and which
      can be handled by the given subsolver.

    EXAMPLE::

       sage: A = MatrixSpace(ZZ['x'], 4, 5).random_element()
       sage: my_solver = quick_check(gauss())
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0)
       sage: V = my_solver(A.transpose())
       sage: len(V)
       0
    
    """
    def quick_check_solver(mat, degrees=[], infolevel=0):
        """See docstring of quick_check() for further information."""
        return _quick_check(subsolver, modsolver, modulus, mat, degrees, infolevel)
    return quick_check_solver

def _quick_check(subsolver, modsolver, modulus, mat, degrees, infolevel):

    R = mat.base_ring()
    _launch_info(infolevel, "quick_check", dim=mat.dimensions(), domain=R)

    if R.is_prime_field() and R.characteristic() > 0:
        return modsolver(mat, degrees=degrees, infolevel=_alter_infolevel(infolevel, -1, 1))

    K = R.base_ring(); x = R.gens()

    if not (K == QQ or K == ZZ):
        _info(infolevel, "Unexpected ring encountered; skipping quick check.", alter = -1)
        return subsolver(mat, degrees=degrees, infolevel=_alter_infolevel(infolevel, -1, 1))
    
    check_prime = pp(modulus) if R.characteristic() == 0 else R.characteristic()
    check_eval = dict(zip(x, [ 17*j + 13 for j in xrange(len(x)) ]))
    check_field = GF(check_prime)
    check_mat = mat.apply_map( lambda pol : pol.substitute(check_eval), check_field )

    _info(infolevel, "Starting modular solver...", alter = -1)
    check_sol = modsolver(check_mat)

    _info(infolevel, "Modular solver predicts " + str(len(check_sol)) + " solutions", alter = -1)

    if len(check_sol) == 0:
        return []
    else:
        return subsolver(mat, degrees=degrees, infolevel=_alter_infolevel(infolevel, -2, 1))

def compress(subsolver, presolver=sage_native, modulus=MAX_MODULUS):
    """
    Constructs a solver which throws away unnecessary rows and columns and then applies the subsolver

    INPUT:

    - ``subsolver`` -- a solver for matrices over `R[x..]`
    - ``presolver`` -- a solver for matrices over `GF(p)`, defaults to `sage_native`_
    - ``modulus`` -- modulus used for the precomputation in the homomorphic image if `R.characteristic()==0`.
      If this number is not a prime, it will be replaced by the next smaller integer which is prime. 

    OUTPUT:

    - a solver for matrices over `R[x..]`

    EXAMPLE:

    ::

       sage: A = MatrixSpace(ZZ['x'], 7, 4).random_element()*MatrixSpace(ZZ['x'], 4, 5).random_element()
       sage: my_solver = compress(gauss())
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0, 0, 0, 0)

    ALGORITHM:

    #. Compute the homomorphic image of the solution space using the presolver. Discard all those columns
       of the matrix where the corresponding row of every solution vector contains a zero. 

    #. Starting from the heaviest row down to the lightest (measuring the weight of a row by counting the
       number of nonzero entries and the number of monomials), check whether the solution space changes
       if the row is dropped. Keep on dropping rows until the difference between the number of rows and
       the number of columns agrees with the dimension of the solution space.

    #. Apply the subsolver to the resulting matrix. Fill in zeros into the output according to the columns
       deleted in step one. Then return the result. 

    """
    def compress_solver(mat, degrees=[], infolevel=0):
        """See docstring of compress() for further information."""
        return _compress(subsolver, presolver, modulus, mat, degrees, infolevel)
    return compress_solver

def _compress(subsolver, presolver, modulus, mat, degrees, infolevel):
    r"""Internal version of compress solver"""
    
    R = mat.parent().base_ring(); x = R.gens(); p = R.characteristic(); (n, m) = mat.dimensions(); zero = R.zero()
    _launch_info(infolevel, "compress", dim=(n, m), domain=R)

    if p == 0:
        p = pp(modulus+1)

    phi = dict(zip(x, [GF(p)(53 + k*17) for k in xrange(len(x))])); z = GF(p).zero()
    matp = mat.apply_map(lambda q: q.subs(phi), GF(p))
    Vp = presolver(matp, degrees=[], infolevel=_alter_infolevel(infolevel, -2, 1))

    if len(Vp) == 0:
        return []

    _info(infolevel, len(Vp), "solutions expected", alter=-1)

    # determine indices of unnecessary columns, if there are any.
    useless_columns = []
    for j in xrange(m):
        if not any(v[j] for v in Vp):
            useless_columns.append(j)
    
    if len(useless_columns) > 0:
        _info(infolevel, "discarding", len(useless_columns), "columns", alter=-1)
        mat = mat.delete_columns(useless_columns)

    # determine row weights
    row_idx = [ 10**13*sum(1 for p in row if p != zero) + sum(p.degree() for p in row if p != zero) for row in mat ]
    row_idx = zip(range(n), row_idx)
    row_idx.sort(key=lambda p: -p[1])
    row_idx = map(lambda p: p[0], row_idx)
    # now row_idx[0] is the heaviest row, row_idx[1], the second heaviest, etc., until row_idx[-1] being the lightest.

    # remove unnecessary rows in descending order of weight
    useless_rows = []
    for r in row_idx:
        _info(infolevel, "considering row", r, "for deletion...", alter=-2)
        row = [q for q in matp[r]]; matp.set_row(r, [z for q in xrange(m)]) # replace r-th row by zeros
        if len(presolver(matp, degrees=[], infolevel=_alter_infolevel(infolevel, -3, 1))) == len(Vp):
            useless_rows.append(r) # not needed; keep the zeros and proceed
        else:
            matp.set_row(r, row) # needed; put back original elements
        if n - len(useless_rows) == m - len(useless_columns) - len(Vp):
            break # all other rows are needed for dimension reasons

    if len(useless_rows) > 0:
        _info(infolevel, "discarding", len(useless_rows), "rows", alter=-1)
        mat = mat.delete_rows(useless_rows)

    # call subsolver on reduced matrix
    V = subsolver(mat, degrees = degrees, infolevel = _alter_infolevel(infolevel, -1, 1))

    # put back zeros into solution vectors
    if len(useless_columns) > 0:
        _info(infolevel, "inserting zero rows into solution vector", alter=-1)
        V = [ list(v) for v in V ]
        for v in V:
            for j in useless_columns:
                v.insert(j, zero)
        V = [ vector(R, v) for v in V ]
                
    return V

def wiedemann():
    r"""
    Constructs a solver using Wiedemann's algorithm

    INPUT:

    - none.

    OUTPUT:

    - a solver for matrices over `R[x..]`

    EXAMPLE:

    ::

       sage: A = MatrixSpace(ZZ['x'], 4, 5).random_element()
       sage: my_solver = wiedemann()
       sage: V = my_solver(A)
       sage: A*V[0]
       (0, 0, 0, 0)

    ALGORITHM: Wiedemann's algorithm.

    .. NOTE::

       #. If the matrix is not a square matrix, it will be left-multiplied by its transpose before the
          algorithm is started. 

       #. If the matrix is a square matrix, it need not actually be a Sage matrix object. Instead, it can be
          any Python object ``A`` which provides the following functions:
          ``A.dimensions()``, ``A.parent()``, and ``A*v`` for a vector ``v``.

       #. The solver returns at most one solution vector, even if the nullspace has higher dimension.
          If it returns the empty list, it only means that the nullspace is probably empty. 

    """
    def wiedemann_solver(mat, degrees=[], infolevel=0):
        """See docstring of wiedemann() for further information"""
        return _wiedemann(mat, degrees, infolevel)
    return wiedemann_solver

def _berlekamp_massey(data):
    """finds a c-finite recurrence of order len(data)/2 for the given terms. """

    R = data[0].parent()
    M = berlekamp_massey(data) #### REIMPLEMENT!

    try:
        d = lcm([p.denominator() for p in M])
    except:
        d = R.one()

    return [-R(d*p) for p in M] 

def _wiedemann(A, degrees, infolevel):
    
    R = A.parent().base_ring(); x = R.gens(); p = R.characteristic(); (n, m) = A.dimensions(); zero = R.zero()
    _launch_info(infolevel, "wiedemann", dim=(n, m), domain=R)

    if p == 0:
        p = pp(2**16+1)

    if n != m:
        _info(infolevel, "Bringing matrix into square form", alter=-1)
        A = A.transpose() * A
        (n, m) = A.dimensions()

    x_base = vector(R, [R.random_element() for i in xrange(m) ])
    x = A*x_base; y = vector(R, [R.random_element() for i in xrange(m) ])

    _info(infolevel, "Computing Krylov basis", alter=-1)
    data = [y*x]
    for i in xrange(2*n + 1):
        _info(infolevel, "power", i, alter=-2)
        x = A*x ###### MOST EXPENSIVE STEP (if matrix is big and entries are small)
        data.append(y*x)

    _info(infolevel, "Computing minimal polynomial", alter=-1)
    M = _berlekamp_massey(data) ###### MOST EXPENSIVE STEP (if matrix is small and entries are big)

    _info(infolevel, "Computing solution vector", alter=-1)
    xi = x_base; x = M[0]*xi
    for i in xrange(1, len(M)):
        _info(infolevel, "power", i, alter=-2)
        xi = A*xi  ###### MOST EXPENSIVE STEP (if matrix is big and entries are small)
        x += M[i]*xi

    if not any(x):
        return []
    else:
        return _normalize([ vector(R, x) ])


#################################################################################################################

#def take_picture(mat, idx):
#    f = open("/scratch/mkauers/picbignaive" + str(idx) + ".m", "w")
#    f.write("{")
#    for i in xrange(len(mat) - 1):
#        f.write("{"); mati = mat[i]
#        for j in xrange(len(mati) - 1):
#            f.write(str(len(mati[j].coefficients())) + ",")
#        f.write(str(len(mati[-1].coefficients())) + "},\n")
#    f.write("{"); mati = mat[-1]
#    for j in xrange(len(mati) - 1):
#        f.write(str(len(mati[j].coefficients())) + ",")
#    f.write(str(len(mati[-1].coefficients())) + "}}\n")
#    f.close()
    
