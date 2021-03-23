# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of convergent D-finite series by binary splitting

TESTS::

    sage: from ore_algebra import DifferentialOperators
    sage: Dops, x, Dx = DifferentialOperators(QQ, 'x')

    sage: import logging
    sage: logging.basicConfig()
    sage: logger = logging.getLogger('ore_algebra.analytic.binary_splitting')

    sage: logger.setLevel(logging.DEBUG)

    sage: ((x^2 + 1)*Dx^2 + 2*x*Dx).numerical_solution([0, 1],
    ....:         [0, i+1, 2*i, i-1, 0], algorithm="binsplit")
    INFO:...
    DEBUG:ore_algebra.analytic.binary_splitting:coefficients in: ... Complex
    ball field ...
    [3.14159265358979...] + [+/- ...]*I

    sage: NF.<sqrt2> = QuadraticField(2)
    sage: dop = (x^2 - 3)*Dx^2 + x + 1
    sage: dop.numerical_transition_matrix([0, sqrt2], 1e-10, algorithm="binsplit")
    INFO:...
    DEBUG:ore_algebra.analytic.binary_splitting:evaluation point in: ... Number
    Field ...
    [[1.669017372...] [1.809514316...]]
    [[1.556515516...] [2.286697055...]]

    sage: (Dx - 1).numerical_solution([1], [0, i + pi], algorithm="binsplit")
    INFO:...
    DEBUG:ore_algebra.analytic.binary_splitting:coefficients in: ... Complex
    ball field ...
    [12.5029695888765...] + [19.4722214188416...]*I

    sage: logger.setLevel(logging.WARNING)

    sage: from ore_algebra.analytic.examples.misc import koutschan1
    sage: koutschan1.dop.numerical_solution(koutschan1.ini, [0, 84], algorithm="binsplit")
    [0.011501537469552017...]

Note that the zeros here shouldn't be exact unless we have proved that the
corresponding series do not continue::

    sage: ((x + 1)*Dx^2 + Dx).numerical_transition_matrix([0,1/2], algorithm='binsplit')
    [ [1.00000000000000...] [0.405465108108164...]]
    [             [+/- ...] [0.666666666666666...]]

    sage: ((x + 1)*Dx^3 + Dx).numerical_transition_matrix([0,1/2], algorithm='binsplit')
    [  [1.000000000000000...]  [0.4815453970799961...]  [0.2456596136789682...]]
    [               [+/- ...]  [0.8936357901691244...]  [0.9667328760004665...]]
    [               [+/- ...] [-0.1959698689702905...]  [0.9070244207738327...]]

    sage: ((x + 1)*Dx^3 + Dx^2).numerical_transition_matrix([0,1/2], algorithm='binsplit')
    [ [1.000000000000000...] [0.5000000000000000...] [0.2163953243244931...]]
    [              [+/- ...]  [1.000000000000000...] [0.8109302162163287...]]
    [              [+/- ...]               [+/- ...] [0.6666666666666666...]]

More examples::

    sage: from ore_algebra.examples import fcc
    sage: fcc.dop5.numerical_solution( # long time (3.6 s)
    ....:          [0, 0, 0, 0, 1, 0], [0, 1/5+i/2, 1],
    ....:          1e-60, algorithm='binsplit', bit_burst_thr=1000)
    [1.04885235135491485162956376369999275945402550465206640...] + [+/- ...]*I

    sage: QQi.<i> = QuadraticField(-1)
    sage: (Dx - i).numerical_solution([1], [sqrt(2), sqrt(3)], algorithm="binsplit")
    [0.9499135278648561...] + [0.3125128630622157...]*I

Regular singular connection problems
====================================

Elementary examples::

    sage: logger.setLevel(logging.DEBUG)
    sage: (x*Dx^2 + x + 1).numerical_transition_matrix([0, 1], 1e-10,
    ....:                                             algorithm="binsplit")
    INFO:...
    DEBUG:ore_algebra.analytic.binary_splitting:coefficients in: ... Complex
    ball field ...
    [[-0.006152006884...]    [0.4653635461...]]
    [   [-2.148107776...]  [-0.05672090833...]]
    sage: logger.setLevel(logging.WARNING)

    sage: dop = (Dx*(x*Dx)^2).lclm(Dx-1)
    sage: dop.numerical_solution([2, 0, 0, 0], [0, 1/4], algorithm="binsplit")
    [1.921812055672805...]

Connection to an algebraic endpoint::

    sage: NF.<sqrt2> = QuadraticField(2)
    sage: dop = (x^2 - 2)*Dx^2 + x + 1
    sage: dop.numerical_transition_matrix([0, sqrt2], 1e-10, algorithm="binsplit")
    [ [2.4938814...] +      [+/- ...]*I  [2.4089417...] +       [+/- ...]*I]
    [[-0.2035417...] + [6.6873857...]*I  [0.2043720...] + [6.45961849...]*I]

Bessel, with an algebraic point of order > 2::

    sage: alg = QQbar(-20)^(1/3)
    sage: dop = x*Dx^2 + Dx + x
    sage: dop.numerical_transition_matrix([0, alg], 1e-8, algorithm="binsplit")
    [ [3.7849872...] +  [1.7263190...]*I  [1.3140884...] + [-2.3112610...]*I]
    [ [1.0831414...] + [-3.3595150...]*I  [-2.0854436...] + [-0.7923237...]*I]

Ci(sqrt(2))::

    sage: dop = x*Dx^3 + 2*Dx^2 + x*Dx
    sage: ini = [1, CBF(euler_gamma), 0]
    sage: dop.numerical_solution(ini, path=[0, sqrt(2)], algorithm="binsplit")
    [0.46365280236686...]

Whittaker functions with irrational exponents::

    sage: dop = 4*x^2*Dx^2 + (-x^2+8*x-11)
    sage: dop.numerical_transition_matrix([0, 10], algorithm="binsplit")
    [[-3.829367993175840...] + [+/-...]*I  [7.857756823216...] + [+/-...]*I]
    [[-1.135875563239369...] + [+/-...]*I  [1.426170676718...] + [+/-...]*I]

Various mixes of algebraic exponents and evaluation points::

    sage: ((x*Dx)^3-2-x).numerical_transition_matrix([0,1], 1e-5, algorithm="binsplit")
    [[0.75335...] + [-0.09777...]*I  [0.75335...] + [0.09777...]*I  [1.1080...] + [+/- ...]*I]
    [[-0.78644...] + [-0.8794...]*I [-0.78644...] + [0.8794...]*I  [1.5074...] + [+/- ...]*I]
    [ [0.10146...] + [1.2165...]*I  [0.10146...] + [-1.2165...]*I  [0.32506...] + [+/- ...]*I]

    sage: ((x*Dx)^2-2-x).numerical_transition_matrix([0,i], 1e-5, algorithm="binsplit")
    [[-1.4237...] + [0.0706...]*I [-0.7959...] + [0.6169...]*I]
    [[1.4514...] + [-0.1681...]*I  [0.6742...] + [1.2971...]*I]

    sage: ((x*Dx)^3-2-x).numerical_transition_matrix([0,i], 1e-8, algorithm="binsplit")
    [  [1.94580...] + [-5.61860...]*I [0.04040867...] + [-0.16436364...]*I    [-0.491906...] + [0.873265...]*I]
    [   [0.66185...] + [8.46096...]*I  [0.1372584...] + [-0.08686559...]*I     [1.052794...] + [0.713283...]*I]
    [ [-6.63615...] + [-4.75084...]*I    [0.1614174...] + [0.0994372...]*I  [0.1969810...] + [-0.0803104...]*I]

    sage: ((x*Dx)^3-2-x).numerical_transition_matrix([0,QQbar(3)^(1/3)], 1e-8, algorithm="binsplit")
    [[0.4434002...] + [-0.31995...]*I   [0.44340...] + [0.31995...]*I  [1.8368349...] + [+/- ...]*I]
    [ [-0.59869...] + [-0.24297...]*I  [-0.59869...] + [0.24297...]*I  [1.7859825...] + [+/- ...]*I]
    [ [0.235894...] + [0.393421...]*I  [0.23589...] + [-0.39342...]*I  [0.3084151...] + [+/- ...]*I]

    sage: (((x-i)*Dx)^2+1-x).numerical_transition_matrix([i, 2*i], algorithm="binsplit")
    [ [2.582900...] + [-2.3669708...]*I [0.13647705...] + [0.1829854...]*I]
    [[-0.831495...] + [5.21969896...]*I [0.30502246...] + [0.1440307...]*I]

    sage: (((x-i)*Dx)^2+1-x).numerical_transition_matrix([i, sqrt(2)], algorithm="binsplit")
    [ [0.69261048...] + [0.16794895...]*I  [2.70326173...] + [-0.78385034...]*I]
    [[0.2516164...] + [-0.10376355...]*I   [0.68776459...] + [1.07708951...]*I]

    sage: (Dx - i).numerical_solution([1], [0, sqrt(2)], algorithm="binsplit")
    [0.1559436947653...] + [0.9877659459927...]*I

An interesting “real-world” example, where one of the local exponents is
irrational, but in the same extension as the singularity itself::

    sage: from ore_algebra.examples import cbt
    sage: s = cbt.dop[4].leading_coefficient().roots(AA, multiplicities=False)[0]
    sage: cbt.dop[4].local_basis_monomials(s)
    [1,
    z - 0.3079785283699041?,
    (z - 0.3079785283699041?)^1.274847940729959?,
    (z - 0.3079785283699041?)^2,
    (z - 0.3079785283699041?)^3]
    sage: (8 + 3*s)/7
    1.274847940729959?
    sage: cbt.dop[4].numerical_transition_matrix([0,s], 1e-8, algorithm="binsplit")
    [ [1.0000...] + [+/- ...]*I  [0.3075...] + [+/- ...]*I  [0.1022...] + [+/- ...]*I  [-0.006...] + [+/- ...]*I  [0.050...] + [+/- ...]*I]
    [ [+/- ...] + [+/- ...]*I  [0.9658...] + [+/- ...]*I  [1.1474...] + [+/- ...]*I  [-2.265...] + [+/- ...]*I  [2.878...] + [+/- ...]*I]
    [ [+/- ...] + [+/- ...]*I [0.04553...] + [-0.0532...]*I  [-0.7074...] + [0.8274...]*I  [3.381...] + [-3.954...]*I  [-3.586...] + [4.194...]*I]
    [ [+/- ...] + [+/- ...]*I [0.09709...] + [+/- ...]*I  [-0.5020...] + [+/- ...]*I  [8.028...] + [+/- ...]*I  [-6.548...] + [+/- ...]*I]
    [ [+/- ...] + [+/- ...]*I  [0.1263...] + [+/- ...]*I  [-1.9369...] + [+/- ...]*I  [9.949...] + [+/- ...]*I  [-6.564...] + [+/- ...]*I]

Recurrences of order zero::

    sage: (x*Dx + 1).numerical_transition_matrix([0,2], algorithm="binsplit")
    [0.50000000000000000]

    sage: (x*Dx + 1).numerical_transition_matrix([0,2], algorithm="binsplit",
    ....:                                        force_algorithm=True)
    Traceback (most recent call last):
    ...
    NotImplementedError: recurrence of order zero

Some other corner cases::

    sage: (x*Dx + 1).numerical_transition_matrix([i, i], algorithm="binsplit")
    [1.0000000000000000]

    sage: (Dx - (x - 1)).numerical_solution([1], [0, 1], algorithm="binsplit")
    [0.6065306597126334...]

Nonstandard branches::

    sage: from ore_algebra.examples.iint import f, w, diffop, iint_value
    sage: iint_value(diffop([f[1/4], w[1], f[1]]),
    ....:            [0, 0, 16/9*i, -16/27*i*(-3*i*pi+8)],
    ....:            algorithm="binsplit")
    [-3.445141853366...]

Miscellaneous examples::

    sage: dop = ((x*Dx-3/2)^2).lclm(Dx-1)
    sage: dop.numerical_solution([2, 3, 7], [0, -1], 1e-8, algorithm="binsplit")
    [10.160536...] + [-7.0000000...]*I
    sage: CBF(2*exp(-1) + 3*(-1)^(3/2)*log(-1) + 7*(-1)^(3/2))
    [10.1605368431122...] - 7.000000000000000*I

    sage: (x*(x-1)*Dx^2 - x).numerical_transition_matrix([0,1], 1e-6, algorithm="binsplit")
    [     [0.22389...] + [+/- ...]*I      [0.57672...] + [+/- ...]*I]
    [[-1.56881...] + [-0.70337...]*I  [0.42531...] + [-1.81183...]*I]

    sage: QQi.<i> = QuadraticField(-1)
    sage: (Dx - i).numerical_solution([1], [0,1], algorithm="binsplit")
    [0.5403023058681397...] + [0.8414709848078965...]*I

    sage: logger.setLevel(logging.DEBUG)
    sage: ((x*Dx - 3/7)).lclm(Dx - 1).numerical_transition_matrix([0,1/3], algorithm="binsplit")
    INFO:...
    DEBUG:ore_algebra.analytic.binary_splitting:coefficients in: ... Complex
    ball field ...
    [[1.395612425086089...] + [+/- ...]*I  [0.6244813348581596...]]
    [[1.395612425086089...] + [+/- ...]*I   [0.802904573389062...]]
    sage: logger.setLevel(logging.WARNING)
"""

# Copyright 2015, 2016, 2017, 2018, 2019 Marc Mezzarobba
# Copyright 2015, 2016, 2017, 2018, 2019 Centre national de la recherche scientifique
# Copyright 2015, 2016, 2017, 2018 Université Pierre et Marie Curie
# Copyright 2019 Sorbonne Université
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

from __future__ import print_function
from six.moves import range

import copy
import logging
import pprint

import sage.rings.polynomial.polynomial_element as polyelt
import sage.rings.polynomial.polynomial_ring as polyring
import sage.rings.polynomial.polynomial_ring_constructor as polyringconstr

from sage.arith.all import gcd
from sage.matrix.constructor import matrix
from sage.matrix.matrix_space import MatrixSpace
from sage.modules.free_module_element import vector
from sage.rings.all import ZZ, QQ, RLF, CLF, RealBallField, ComplexBallField
from sage.rings.number_field.number_field import is_NumberField, NumberField
from sage.structure.coerce_exceptions import CoercionException

from . import accuracy, bounds, utilities

from .context import dctx
from .local_solutions import (bw_shift_rec, FundamentalSolution,
        LocalBasisMapper, log_series_values)
from .safe_cmp import *

logger = logging.getLogger(__name__)

def PolynomialRing(base, var):
    if is_NumberField(base) and base is not QQ:
        return polyring.PolynomialRing_field(base, var,
                element_class=polyelt.Polynomial_generic_dense)
    else:
        return polyringconstr.PolynomialRing(base, var)

class StepMatrix(object):
    r"""
    A structured matrix that maps a vector of s coefficients and a partial sum
    (both around some truncation index n) of a D-finite series to a similar
    vector corresponding to the partial sum truncated at order n + p for some p.
    The partial sum (but not the coefficients) typically depends on a
    perturbation parameter δ, making it possible to compute several derivatives
    of the series at once.
    """

    def __init__(self, rec, n0, n1, ord_log):

        self.idx_start = n0
        self.idx_end = n1
        self.ord_diff = rec.derivatives
        self.ord_log = ord_log

        # updated by _init_range
        self.rec_den = rec.AlgInts_rec.one()
        self.pow_num = rec.pow_num.parent().one()
        self.pow_den = rec.pow_den.parent().one()

        if n0 == n1:
            self._init_identity(rec, n1, ord_log)
        elif n0 is not None:
            self._init_range(rec, n0, n1, ord_log)

    def _init_identity(self, rec, n, ord_log):
        self.rec_mat = rec.Mat_rec.one()
        self.zero_sum, self.sums_row = rec.Series_sums(ord_log)

    def _init_range(self, rec, n0, n1, ord_log):
        r"""
        Naïve unrolling.
        """

        self.zero_sum, self.sums_row, seqs = self._seq_init(rec, ord_log)

        for n in range(n0+1, n1+1):

            bwrec_n, rec_den_n = self._coeff_series_num_den(rec, n, ord_log)
            den = rec.pow_den*rec_den_n

            for (num, psum) in zip(seqs, self.sums_row):
                # separate to allow for a cython version
                self._seq_next(num, psum, bwrec_n, rec_den_n, den, ord_log)

            self.pow_num = self.pow_num._mul_trunc_(rec.pow_num, self.ord_diff)
            self.pow_den *= rec.pow_den
            self.rec_den *= rec_den_n

        # Polynomial of matrices.
        rec_mat = []
        Mat = rec.Mat_rec.base_ring()
        for k in range(ord_log):
            mat = Mat.matrix()
            for j, u in enumerate(seqs):
                for i in range(1, rec.ordrec + 1):
                    mat[-i,j] = u[-i][k]
            rec_mat.append(mat)
        self.rec_mat = rec.Mat_rec.element_class(rec.Mat_rec, rec_mat,
                                                 check=False)

        # TODO: find and remove common factors?
        # self.simplify()

        # May not always hold, but convenient for checking that we are producing
        # exact balls in simple cases
        # self.assert_exact()
        pass

    @staticmethod
    def _seq_init(rec, ord_log):
        zero, row = rec.Series_sums(ord_log)
        seqs = [[rec.Pols_rec.zero()]*rec.ordrec for _ in range(rec.ordrec)]
        for k in range(1, rec.ordrec + 1):
            seqs[-k][-k] = rec.Pols_rec.one()
        return zero, row, seqs

    def _coeff_series_num_den(self, rec, n, ord_log):
        return rec.coeff_series_num_den(n, ord_log)

    def _seq_next(self, num, psum, bwrec_n, rec_den_n, den, ord_log):

        for q in range(ord_log):
            for p in range(self.ord_diff):
                psum[q][p] += self.pow_num[p]*num[-1][q]
                psum[q][p] *= den

        u_n = -sum(bwrec_n[k]._mul_trunc_(num[-k], ord_log)
                    for k in range(1, len(bwrec_n)))
        num[:] = [rec_den_n*term for term in num[1:]]
        num.append(u_n)

    # TODO: try caching the powers of (pow_num/pow_den)? this probably
    # won't change anything for algebraic evaluation points, but it might
    # make a difference when the evaluation point is more complicated

    def compute_sums_row(low, high):

        # sums_row = high.sums_row*low.rec_mat*low.pow_num
        #             δ, Sk, row     Sk, mat      δ
        #
        #            + high.rec_den*high.pow_den*low.sums_row
        #                 cst (nf)        cst        δ, Sk, row

        ordrec = high.rec_mat.base_ring().nrows()
        Scalars = high.zero_sum.parent()

        high_den = Scalars(high.rec_den*high.pow_den)
        use_sum_of_products = (hasattr(Scalars, "_sum_of_products")
                and low.pow_num.base_ring() is Scalars
                and low.rec_mat.base_ring().base_ring() is Scalars)

        # TODO: maybe try introducing matrix-matrix multiplications

        res1 = [None]*len(high.sums_row)
        for j in range(ordrec):
            res2 = [None]*high.ord_log
            for q in range(high.ord_log):
                res3 = [None]*low.ord_diff
                for p in range(low.ord_diff):
                    # one coefficient of one entry the first term
                    # high.sums_row*low.rec_mat*low.pow_num
                    if use_sum_of_products:
                        # Even with this, perhaps ~2/3 of the time goes in
                        # (unnecessary) object creations/deletions/copies...
                        t = Scalars._sum_of_products(
                               ( high.sums_row[k][v][u],
                                 low.pow_num[p-u],
                                 low.rec_mat[q-v][k,j] )
                               for k in range(ordrec)
                               for u in range(p + 1)
                               for v in range(q + 1))
                    else:
                        t = sum(
                               high.sums_row[k][v][u]
                                   * Scalars(low.pow_num[p-u]
                                               * low.rec_mat[q-v][k,j])
                               for k in range(ordrec)
                               for u in range(p + 1)
                               for v in range(q + 1))
                    # same for the second term
                    # high.rec_den*pow_den.rec_den*low.sums_row
                    if q < low.ord_log: # usually true, but low might be
                                        # an optimized SolutionColumn
                        t += high_den*low.sums_row[j][q][p]
                    res3[p] = t
                res2[q] = res3
            res1[j] = res2
        return res1

    def imulleft(low, high): # pylint: disable=no-self-argument
        assert high.idx_start == low.idx_end
        # logger.debug("(%s->%s)*(%s->%s)", high.idx_start, high.idx_end,
        #                                   low.idx_start, low.idx_end)

        low.sums_row = low.compute_sums_row(high) # must come early
        low.rec_mat = high.rec_mat._mul_trunc_(low.rec_mat, high.ord_log)
        low.pow_num = low.pow_num._mul_trunc_(high.pow_num, low.ord_diff)
        low.pow_den *= high.pow_den
        low.rec_den *= high.rec_den

        low.idx_end = high.idx_end
        return low

    def copy(self):
        new = copy.copy(self)
        new.rec_mat = copy.copy(self.rec_mat)
        new.sums_row = copy.copy(self.sums_row)
        return new

    def __mul__(high, low): # pylint: disable=no-self-argument
        return low.copy().imulleft(high)

    def __repr__(self):
        res = "{{ indices={}->{}, ord_diff={}, ord_log={},\n".format(
                self.idx_start, self.idx_end, self.ord_diff, self.ord_log)
        res += "pow_den={}, pow_num={},\n".format(self.pow_den, self.pow_num)
        Pol = PolynomialRing(self.rec_mat.base_ring().base_ring(), 'Sk')
        Sk = Pol.gen()
        rec_mat = sum(m*Sk**k for k, m in enumerate(self.rec_mat))
        res += "rec_den={}, rec_mat=\n{}\n".format(self.rec_den, rec_mat)
        res += "sums_row={} }}".format(self.sums_row)
        return res

# TODO: get rid of these classes (unless they prove useful again!)

class StepMatrix_generic(StepMatrix):

    def assert_exact(self):
        return

class StepMatrix_arb(StepMatrix):

    def assert_exact(self):
        r"""
        Assert that all the balls used to represent this matrix are exact.

        This may not always be true in real computations.
        """
        for m in self.rec_mat:
            for a in m.list():
                assert a.is_exact()
        assert self.rec_den.is_exact()
        for a in self.pow_num:
            assert a.is_exact()
        assert self.pow_den.is_exact()
        for p in self.sums_row:
            for a in p:
                for b in a:
                    assert b.is_exact()

class SolutionColumn(object):
    r"""
    Partially “unrolled” local canonical solutions.

    Each solution is represented during the computation by a StepMatrix mat
    whose application to the length-one vector [ log(z)^k/k! ] where
    k = mat.ord_log - 1 yields the coefficients and partial sums of the series.
    Equivalently, entries of the SolutionColumn can be viewed as coefficient
    sequences of (scaled) powers of log(z), with the coefficients of
    Sk^(ord_log-1), Sk^(ord_log-2), ... corresponding to those of 1, log(z), ...

    Mathematically, mat should be a column matrix, but for technical reasons (in
    order to represent it as a polynomial in Sk with matrix coefficients), it is
    actually represented as the last column of a square matrix otherwise filled
    with zeros.

    The ord_log field is meant to be initialized to k+1 ≤ μ for a solution of
    leading monomial z^ν*log(z)^k/k! where ν is a singular index of
    multiplicity μ, and then increased (if necesssary) when crossing other
    singular indices.

    Applying a StepMatrix (viewed as a matrix of operators) to a SolutionColumn
    almost amounts to a usual StepMatrix multiplication. The difference is that
    the product must be truncated (wrt Sk) to the order of the SolutionColumn's
    ord_log, while products of StepMatrixes normally use the ord_log field of
    the left factor. (It is possible to some extent to perform a standard
    multiplication and call fix_product_and_shift_logs() to truncate the result
    afterwards.)
    """

    def __init__(self, rec, n, log_power):
        # n0 = None: can be multiplied on left, not on the right
        self.v = rec.StepMatrix_class(rec, None, n, log_power + 1)
        # square matrix because we want to use it as a polynomial coefficient
        mat = rec.Mat_rec().base_ring()()
        mat[-1,-1] = 1
        self.v.rec_mat = rec.Mat_rec(mat)
        # ordrec columns for compatibility with the square matrix; only the last
        # column is really used
        self.v.zero_sum, self.v.sums_row = rec.Series_sums(self.v.ord_log)

    def __repr__(self):
        from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
        Scalars = self.v.rec_mat.base_ring().base_ring()
        Pol, LOG = LaurentPolynomialRing(Scalars, 'LOG').objgen()
        s = len(self.v.sums_row)
        vec = vector(Pol, s,
                [sum(
                    (mat[i,-1]/self.v.rec_den)*LOG**(self.v.ord_log-1-k)
                    for k, mat in enumerate(self.v.rec_mat))
                for i in range(s)])
        return repr(vec)

    def assert_well_formed(self):
        assert self.v.rec_mat.degree() < self.v.ord_log
        assert len(self.v.sums_row[-1]) == self.v.ord_log

    def iapply(self, fwd, shift):
        r"""
        In-place application to self of a StepMatrix followed by a shift of the
        last coefficient row.

        Note that applying Sk as part of fwd and then shifting by the action of
        “Sk⁻¹” is not the identity operation, as it replaces the coefficient
        of Sk^(ord_log-1) (corresponding to log^0) by 0.
        """
        assert fwd.ord_log >= self.v.ord_log
        self.assert_well_formed()
        logger.log(logging.DEBUG-1, "*  %s", self)
        self.v.imulleft(fwd)
        logger.log(logging.DEBUG-1, "-> %s", self)
        # well-formedness may be temporarily broken at this point
        self.fix_product_and_shift_logs(shift)
        logger.log(logging.DEBUG-1, "-> %s", self)
        self.assert_well_formed()

    def fix_product_and_shift_logs(self, m):
        r"""
        In-place shift by Sk^-m of the “current” coefficient, i.e. the last
        entry of the coefficient vectors.

        Note that other entries, as well as sums_row, represent series
        coefficients *before* the exceptional index and should be left alone.

        This method can be used on non-well-formed SolutionColumns and truncates
        them as appropriate. In particular, it can be called it with m = 0 to
        fix the result of a left multiplication by a StepMatrix of larger
        ord_log.

        TESTS::

            sage: from ore_algebra import DifferentialOperators
            sage: Dops, x, Dx = DifferentialOperators(QQ, 'x')
            sage: dop = ((x*Dx)**2-x).lclm(x*Dx-1)
            sage: dop.numerical_transition_matrix([0,1/2], algorithm="binsplit")
            [[-1.28621962063266...]  [1.06608292975635...] [0.5000000000000...]]
            [ [1.42056764645230...] [0.271723456312137...]  [1.000000000000...]]
            [[-1.70678726708496...] [0.294359473444213...]                [...]]

            sage: from ore_algebra.examples import fcc
            sage: fcc.dop4.numerical_solution([0, 1, 0, 0], [1, 1/2],
            ....:                             algorithm="binsplit")
            [0.6757976924611368...] + [-2.987541622123742...]*I
            sage: ref = fcc.dop4.numerical_transition_matrix([1, 1/2])
            sage: mat = fcc.dop4.numerical_transition_matrix([1, 1/2],
            ....:                                          algorithm="binsplit")
            sage: all(c.contains_zero() and c.rad() < 1e-10
            ....:     for c in (mat - ref).list())
            True
        """
        # If some of the high-degree coefficients wrt log(z) happen to be zero,
        # e.g., at ordinary points, we don't need the full precision. This will
        # also be used to reduce the precision in the computation of recurrence
        # matrices.
        z = 0
        for k in range(m):
            # Only the last coefficient counts, because it is the one that is
            # going to be shifted.
            if self.v.rec_mat[k][-1][-1].is_zero():
                z += 1
            else:
                break
        Mat = self.v.rec_mat.base_ring()
        new_mats = [Mat() for _ in range(m - z)]
        new_mats.extend(self.v.rec_mat[k].__copy__()
                        for k in range(self.v.ord_log))
        for k in range(self.v.ord_log):
            new_mats[k][-1,-1] = self.v.rec_mat[k + z][-1,-1]
        for mat in new_mats[self.v.ord_log-z:]:
            mat[-1,-1] = 0
        self.v.rec_mat = self.v.rec_mat.parent()(new_mats)
        zeros = [[self.v.zero_sum]*self.v.ord_diff for _ in range(m - z)]
        for p in self.v.sums_row[:-1]:
            p[self.v.ord_log:] = [] # useful?
            p.extend(zeros)
        self.v.sums_row[-1][self.v.ord_log:] = []
        self.v.sums_row[-1][:0] = zeros
        self.v.ord_log += m - z

    def normalized_residual(self, maj, abstract_alg, alg, n):
        r"""
        Compute the normalized residual associated with the fundamental
        solution.

        TESTS::

            sage: from ore_algebra import *
            sage: DOP, t, D = DifferentialOperators()
            sage: ode = D + 1/4/(t - 1/2)
            sage: ode.numerical_transition_matrix([0,1+I,1], 1e-110,
            ....:                                 algorithm='binsplit')
            [[0.707...2078...] + [0.707...]*I]
        """
        # WARNING: this residual must correspond to the operator stored in
        # maj.dop, which typically isn't rec.diffop (but an operator in Θx
        # equal to x^k·rec.diffop for some k).

        assert alg == maj.leftmost

        IC = bounds.IC

        # Specialize abstract algebraic exponent
        alg_to_IC = _specialization_map(self.v.rec_mat.base_ring().base_ring(),
                                        IC, abstract_alg, alg)

        # last = [[u(n-1) for log⁰, log¹, ...], ..., [u(n-s) for all logs]]
        rec_den = IC(self.v.rec_den)
        last = [[alg_to_IC(self.v.rec_mat[self.v.ord_log-1-k][-j,-1])/rec_den # (?)
                 for k in range(self.v.ord_log)
                ] for j in range(self.v.rec_mat.base_ring().nrows())]
        res = maj.normalized_residual(n, last)
        return res

    def partial_sum(self, Jets, dz, abstract_alg, alg, shift, tail_bound):
        # Extract the (abstract numerator of) series part
        op_k = self.v.sums_row[-1] # K[λ][δ][Sk]
        numer = list(reversed(op_k))
        Scalars = Jets.base_ring()
        # Specialize abstract algebraic exponent and add error term
        # (overestimation: we are using a single bound for all subseries)
        specialize = _specialization_map(self.v.zero_sum.parent(), Scalars,
                                         abstract_alg, alg)
        denom = specialize(self.v.rec_den)*Scalars(self.v.pow_den)
        val = vector([Jets([(specialize(c)/denom).add_error(tail_bound)
                            for c in ser])
                     for ser in numer])
        assert self.v.ord_diff == dz.jet_order
        [val] = log_series_values(Jets, alg + shift, val, dz)
        return val

    def error_estimate(self):
        def IC_est(c):
            try:
                return bounds.IC(c) # arb, QQ
            except TypeError:
                return bounds.IC(c[0]) # NF, would work for QQ
        zero = bounds.IR.zero()
        num1 = max([zero] + [abs(IC_est(m[-1, -1])) for m in self.v.rec_mat])
        num2 = sum(abs(IC_est(a)) for a in self.v.pow_num)
        den = abs(IC_est(self.v.rec_den))*bounds.IR(self.v.pow_den)
        return num1*num2/den

class MatrixRec(object):
    r"""
    A matrix recurrence simultaneously generating the coefficients and partial
    sums of solutions of an ODE (with exponents in a certain ℤ-coset), and
    possibly derivatives of these solutions.

    CONVENTIONS:

    Objects and domains:

    - K₀ is a number field (/order) containing the coefficient field of the deq
      and the expansion point (/numerators of...); K₁ = idem with the evaluation
      point; K contains both.

    - λ is (the numerator of) an (abstract) algebraic root of the indicial
      polynomial

    - δ is a dummy variable used to compute derivatives via truncated series,
      and r' (typically = orddeq) is the number of derivatives to compute

    - Sk is an operator that shifts coefficient sequences wrt log(z)^k/k!

    The recurrence reads::

        [ u(n-s+1)·z^n ]   [ 0 z     |   ]  [ u(n-s)·z^(n-1) ]
        [      ⋮       ]   [     ⋱   |   ]  [       ⋮        ]
        [              ]   [       z | 0 ]  [                ]
        [ u(n)    ·z^n ] = [ * * ⋯ * |   ]  [ u(n-1)·z^(n-1) ]
        [ ------------ ]   [ --------+-- ]  [ -------------- ]
        [     σ(n)     ]   [ 0 ⋯ 0 1 | 1 ]  [     σ(n-1)     ]

              U(n)       =      B(n)            U(n-1)

        [so that U(n) = B(n)···B(1)·U(0)]

    where::

               [ (rec_mat/rec_den)·(pow_num/pow_den) | 0  ]
        B(n) = [ ------------------------------------+--- ]
               [      sums_row·rec_den·pow_den       | 1  ]

                       1            [ rec_mat·pow_num |        0         ]
              = ----------------- · [ ----------------+----------------- ]
                 rec_den·pow_den    [     sums_row    | rec_den·pow_den  ]

        rec_mat  ∈ K₀[λ]^(s×s)[Sk]/(Sk^τ)   = Mat_rec/(Sk^τ)
        rec_den  ∈ K₀[λ]                    = AlgInts_rec
        pow_num  ∈ K₁[δ]/(δ^r')             = Series_pow/(δ^r')
        pow_den  ∈ ℤ
        sums_row ∈ ((K[λ][δ]/(δ^r'))[Sk]/(Sk^τ))^(1×s)
                                            ≈ (Series_sums/(δ^r',Sk^τ))^(1×s)

    In the code::

        AlgInts_rec = K₀[λ]
        AlgInts_pow = K₁
        AlgInts_sums = K[λ]
        Series_pow = K₁[δ]
        Mat_rec = K₀[λ]^(s×s)[Sk]
        Series_sums = K[λ][δ][Sk], represented as a list of list of lists
                                   for efficiency reasons

    NOTES:

    - Mathematically, the recurrence matrix has the structure of a StepMatrix
      depending on parameters. However, this class does not derive from
      StepMatrix as the data structure is different.

    - rec_mat is a polynomial of matrices in order to leverage existing matrix
      multiplication and polynomial short product, but sums_row (sort of) has to
      be a matrix of polynomials because the matrix is not square.

    - The scalar domains are called AlgInts_* although they are actually number
      fields or numeric (ball) fields because the elements we are working with
      are algebraic integers.
    """

    def __init__(self, dop, shift, singular_indices,
                 dz, derivatives, prec, binsplit_threshold):

        # TODO: perhaps dynamically optimize the representation when there are
        # no logs, algebraic exponents, etc.

        self.singular_indices = singular_indices

        # Choose computation domains
        E = dz.parent()
        deq_Scalars = dop.base_ring().base_ring()
        assert deq_Scalars is E or deq_Scalars != E
        # Sets self.AlgInts_{rec,pow,sums}, self.pow_{num,den} (pow_num will be
        # modified later), and self.shift.
        if _can_use_CBF(E, deq_Scalars, shift.parent()):
            # Work with arb balls and matrices, when possible with entries in ZZ
            # or ZZ[i]. Round the entries that are larger than the target
            # precision (+ some guard digits) in the upper levels of the tree.
            #
            # Working precision ≈ max(prec + some room for rounding errors,
            # space needed for representing the leaves exactly), assuming prec ≈
            # number of terms. (If the coefficients are so large that we want to
            # round them, we probably shouldn't be using binary splitting.)
            h = dop._naive_height()
            rs = dop.degree()*dop.order()
            wp = max(prec, h*ZZ(rs).nbits()) + (rs + 2)*(ZZ(prec).nbits() + 8)
            self._init_CBF(deq_Scalars, shift, E, dz, wp)
        else:
            try:
                self._init_generic(deq_Scalars, shift, E, dz)
            except CoercionException:
                # Not great, but allows us to handle a few combination of
                # algebraic points that we couldn't otherwise...
                dop, dz, shift = dop.extend_scalars(dz, shift)
                deq_Scalars = dop.base_ring().base_ring()
                E = dz.parent()
                self._init_generic(deq_Scalars, shift, E, dz)
        self.dz = dz

        bwrec = bw_shift_rec(dop, shift=self.shift)
        # Also store an exact version of the leading coefficient to be able to
        # compute the inverse exactly (and simplify the result) when working
        # with arb balls. (Would it be enough to increase the working precision
        # as in the Cython version?)
        self.exact_lc = bwrec.lc_as_rec()
        # separate step because Ore ops cannot have ball coefficients
        self.bwrec = bwrec.change_base(self.AlgInts_rec)
        if self.bwrec.order == 0:
            # not sure what to do in this case
            raise NotImplementedError("recurrence of order zero")
        self.ordrec = self.bwrec.order

        self.binsplit_threshold = max(binsplit_threshold, self.ordrec)

        Mat_rec0 = MatrixSpace(self.AlgInts_rec, self.ordrec)
        self.Mat_rec = PolynomialRing(Mat_rec0, 'Sk')
        self.Pols_rec = PolynomialRing(self.AlgInts_rec, 'Sk')
        logger.debug("coefficients in: %s", self.Pols_rec)

        assert self.bwrec[0].base_ring() is self.AlgInts_rec # uniqueness
        assert self.bwrec[0](0).parent() is self.AlgInts_rec #   issues...

        # Power of dz. Note that this part does not depend on n.
        Series_pow = PolynomialRing(self.AlgInts_pow, 'delta')
        self.pow_num = Series_pow([self.pow_num, self.pow_den])
        self.derivatives = derivatives
        logger.debug("evaluation point in: %s", Series_pow)

    def Series_sums(self, ord_log):
        zero = self.AlgInts_sums.zero()
        row = [[[zero]*self.derivatives for _ in range(ord_log)]
               for _ in range(self.ordrec)]
        return zero, row

    def _init_CBF(self, deq_Scalars, shift, E, dz, prec):
        try:
            from . import binary_splitting_arb
            self.StepMatrix_class = binary_splitting_arb.StepMatrix_arb
        except ImportError:
            logger.warning("Cython implementation unavailable, "
                           "falling back to slower Python implementation")
            self.StepMatrix_class = StepMatrix_arb
        if _can_use_RBF(E, deq_Scalars, shift):
            dom = RealBallField(prec)
        else:
            dom = ComplexBallField(prec)
        if is_NumberField(E):
            pow_den = dz.denominator()
            self.pow_num = dom(pow_den*dz) # mul must be exact
            self.pow_den = dom(pow_den)
        else:
            self.pow_num = dom(dz)
            self.pow_den = dom.one()
        # we are going to use self.shift to build an Ore op, it needs to be
        # exact
        self.shift = shift
        self.AlgInts_rec = self.AlgInts_pow = self.AlgInts_sums = dom

    def _init_generic(self, deq_Scalars, shift, E, dz):
        self.StepMatrix_class = StepMatrix_generic

        if is_NumberField(E): # includes QQ
            # In fact we should probably do something similar for dz in any
            # finite-dimensional Q-algebra. (But how?)
            NF_pow, AlgInts_pow = utilities.number_field_with_integer_gen(E)
            self.pow_den = NF_pow(dz).denominator()
        else:
            # This includes the case E = ZZ, but dz could live in pretty
            # much any algebra over deq_Scalars (including matrices,
            # intervals...). Then the computation of sums_row may take time,
            # but we still hope to gain something on the computation of the
            # coefficients and/or limit interval blow-up thanks to the use
            # of binary splitting.
            AlgInts_pow = E
            self.pow_den = ZZ.one()
        self.pow_num = self.pow_den*dz

        # Reduce to the case of a number field generated by an algebraic
        # integer. This is mainly intended to avoid computing gcds (due to
        # denominators in the representation of number field elements) in
        # the product tree.
        NF_deq, _ = utilities.number_field_with_integer_gen(deq_Scalars)
        if deq_Scalars.has_coerce_map_from(shift.parent()):
            AlgInts_rec = NF_rec = NF_deq
            # We need a parent containing both the coefficients of the operator
            # and the evaluation point.
            AlgInts_sums = utilities.mypushout(AlgInts_rec, AlgInts_pow)
            self.shift = shift
        else:
            pol = shift.parent().polynomial()
            den = shift.denominator()
            assert pol.is_monic()
            assert den*shift == shift.parent().gen()
            name = str(shift.parent().gen())
            AlgInts_sums = utilities.mypushout(NF_deq, AlgInts_pow).extension(pol, name)
            AlgInts_rec = AlgInts_sums # for now at least
            self.shift = AlgInts_rec.gen()/den

        # Guard against various problems related to number field embeddings and
        # uniqueness
        assert AlgInts_pow is AlgInts_rec or AlgInts_pow != AlgInts_rec
        assert AlgInts_sums is AlgInts_rec or AlgInts_sums != AlgInts_rec
        assert AlgInts_sums is AlgInts_pow or AlgInts_sums != AlgInts_pow

        self.AlgInts_rec = AlgInts_rec
        self.AlgInts_pow = AlgInts_pow
        self.AlgInts_sums = AlgInts_sums

    def mult(self, n):
        for n1, m in self.singular_indices:
            if n1 == n:
                return m
        return 0

    def coeff_series_num_den(self, n, ord_log):

        mult = self.mult(n)
        bwrec_n = self.bwrec.eval_series(self.bwrec.Scalars, n,
                                         ord_log + mult)
        assert all(bwrec_n[0][i].is_zero() or bwrec_n[0][i].contains_zero()
                   for i in range(mult))
        assert bwrec_n[0][0].parent() is self.AlgInts_rec
        bwrec_n[0] = None
        # slooow conversion when we are creating arb polys
        bwrec_n = [self.Pols_rec.element_class(self.Pols_rec, c, check=False)
                   for c in bwrec_n]

        # We must compute the (shifted) series inverse exactly even with balls.

        # Returns a polynomial in the wrong variable (but that's ok)
        invlc = self.exact_lc.eval_inv_lc_series(n, ord_log + mult, mult)
        den = invlc.denominator()
        invlc = den*invlc
        invlc = self.Pols_rec.element_class(self.Pols_rec,
                [self.AlgInts_rec(a) for a in invlc], check=False)
        den = self.AlgInts_rec(den)

        for i in range(self.ordrec):
            bwrec_n[1+i] = bwrec_n[1+i]._mul_trunc_(invlc, ord_log)

        if self.AlgInts_rec is QQ: # TODO: generalize (den might be a ball!)
            g = gcd([den] + [c for p in bwrec_n[1:] for c in p])
            den //= g
            if not g.is_one():
                for i in range(self.ordrec):
                    bwrec_n[1+i] = bwrec_n[1+i]/g

        return bwrec_n, den

    def __call__(self, n, ord_log):
        return self.StepMatrix_class(self, n-1, n, ord_log)

    def step_matrix_basecase(self, low, high, ord_log):
        return self.StepMatrix_class(self, low, high, ord_log)

    def step_matrix_binsplit(self, low, high, ord_log):
        r"""
        Compute R(high)·R(high-1)···R(low+1) by binary splitting.
        """
        if high - low <= self.binsplit_threshold:
            mat = self.StepMatrix_class(self, low, high, ord_log)
        else:
            mid = (low + high) // 2
            mat = self.step_matrix_binsplit(low, mid, ord_log)
            if high - low > 400000//self.ordrec**3:
                logger.info("(n=%s)", mid)
            mat.imulleft(self.step_matrix_binsplit(mid, high, ord_log))
        assert mat.idx_start == low and mat.idx_end == high
        return mat

    def __repr__(self):
        return pprint.pformat(self.__dict__)

class MatrixRecsUnroller(LocalBasisMapper):

    def __init__(self, dop, pt, eps, derivatives, ctx=dctx):
        super(self.__class__, self).__init__(dop)
        self.pt = pt
        self.eps = eps
        self.derivatives = derivatives
        self.ctx = ctx
        self._est_terms, _ = dop.est_terms(pt, utilities.prec_from_eps(eps))

    def process_decomposition(self):
        int_expos = (len(self.sl_decomp) == 1
                and self.sl_decomp[0][0].degree() == 1
                and self.sl_decomp[0][0][0].is_zero())
        is_real = (int_expos
                and utilities.is_real_parent(self.dop.base_ring().base_ring())
                and utilities.is_real_parent(self.pt.pt.parent()))
        # enough to represent individual series, but real jets may still become
        # complex later if there are logs
        Intervals = utilities.ball_field(self.eps, is_real)
        self.Jets = PolynomialRing(Intervals, 'delta')

    def process_irred_factor(self):
        # Instead of iterating over the roots as process_irred_factor() does in
        # the superclass, this version only calls process_modZ_class() once,
        # with an abstract algebraic number. It then computes all possible
        # embeddings of the resulting abstract solutions.

        self.edop = self.dop
        if self.irred_factor.degree() == 1:
            # exponent is in the ground field, but not necessarily in QQ, since
            # the differential equation can have coefficients in an embedded
            # number field)
            self.leftmost = -self.irred_factor[0]/self.irred_factor[1]
        else:
            # Represent the exponent as an element of a number field defined by
            # a monic polynomial. Variant of number_field_with_integer_gen().
            name = "lambda" + str(self.nontrivial_factor_index)
            lam = NumberField(self.irred_factor, name).gen()
            assert self.irred_factor().is_monic()
            den = self.irred_factor().denominator()
            monic = (den*lam).minpoly()
            name = "x" + str(den) + name
            self.leftmost = NumberField(monic, name).gen()/den

        self.process_modZ_class()

        # Replace the abstract solutions currently in irred_factor_cols by
        # concrete ones
        assert len(self.irred_factor_cols) == sum(m for _, m in self.shifts)
        self.irred_factor_cols = self.modZ_class_partial_sums(self.Jets)
        # logger.debug("concrete partial sums: %s", self.irred_factor_cols)
        assert (len(self.irred_factor_cols)
                == sum(m for _, m in self.shifts)*self.irred_factor.degree())

    def process_modZ_class(self):

        logger.info(
                r"solutions z^(%s+n)·log(z)^k/k!+···, n=%s, est.#terms=%s",
                self.leftmost, ", ".join(str(s) for s, _ in self.shifts),
                self._est_terms)

        self.modZ_class_tail_bound = None

        # Generic recurrence matrix
        self.matrix_rec = MatrixRec(self.dop, self.leftmost, self.shifts,
                self.pt.pt, self.derivatives, utilities.prec_from_eps(self.eps),
                min(self.ctx.binsplit_thr, self._est_terms))

        # Majorants
        maj = {rt: bounds.DiffOpBound(self.dop, rt, self.shifts,
                                      bound_inverse="solve")
               for rt in self.roots}

        wrapper = bounds.MultiDiffOpBound(maj.values())
        # TODO: switch to fast_fail=True?
        stop = accuracy.StoppingCriterion(wrapper, self.eps, fast_fail=False)

        class BoundCallbacks(accuracy.BoundCallbacks):
            def get_residuals(_): # “self” refers to the MatrixRecsUnroller
                # Will need updating if we want to support very large singular
                # indices efficiently. For the time being, we wait to pass the
                # last singular index before checking for convergence.
                assert len(self.irred_factor_cols) == sum(m for _, m in
                                                                    self.shifts)
                return {(rt, sol.shift, sol.log_power):
                        sol.value.normalized_residual(maj[rt], self.leftmost,
                                                      rt, self.shift)
                        for rt in self.roots
                        for sol in self.irred_factor_cols}
            def get_bound(_, residuals):
                r"""
                Bound the Frobenius norm of the matrix of tail majorants.

                In particular, this is a common bound on the tails of all power
                series coefficients of the solution associated to the current
                irreducible factor of the indicial equation.

                Note that this bound currently does not account for the singular
                prefactors. This doesn't affect the correctness of the final
                result, since the series sums and singular factors will be
                assembled using interval arithmetic. However, it might lead to
                pessimistic enclosures or unnecessary computations in some
                cases.
                """
                sqbound = bounds.IR.zero()
                for rt in self.roots:
                    myres = [residuals[rt,s,k] for (s,m) in self.shifts
                                               for k in range(m)]
                    # Tail majorant valid for the power series in front of the
                    # logs for all solutions associated to rt
                    tmaj = maj[rt].tail_majorant(self.shift, myres)
                    sqbound += tmaj.bound(self.pt.rad, rows=self.derivatives,
                                          cols=len(myres))**2
                return sqbound.sqrtpos()
        cb = BoundCallbacks()

        first_singular_index = min(s for s, _ in self.shifts)
        last_singular_index = max(s for s, _ in self.shifts)
        si = 0
        assert first_singular_index >= 0
        est, tail_bound = None, bounds.IR('inf')
        prev, done = first_singular_index, False
        ord_log = 0
        while True:
            # Below the expected truncation point, aim roughly for it. Above,
            # start with a small stride and roughly double the number of terms
            # every time. Always stop at exceptional indices in any case.
            stride = self._est_terms - prev
            if stride > 2*self.ctx.binsplit_thr:
                resolution = ZZ(self.ctx.binsplit_thr).nbits()
                stride = ((stride >> resolution)) << resolution
            elif stride <= 0:
                stride = max(1 << ZZ(stride).nbits(), self.ctx.binsplit_thr)
                stride = min(prev, stride)
            self.shift = prev + stride
            self.mult = 0
            if si < len(self.shifts) and self.shift >= self.shifts[si][0]:
                self.shift, self.mult = self.shifts[si]
                si += 1
            # Unroll by binary splitting, automatically handling exceptional
            # indices as necessary
            fwd = self.matrix_rec.step_matrix_binsplit(prev, self.shift, ord_log)
            # Extend known solutions
            logger.log(logging.DEBUG-1, "n=%s, extending known solutions",
                       self.shift)
            for sol in self.irred_factor_cols:
                sol.value.iapply(fwd, self.mult)
            # Append “new” solutions starting at the current valuation
            if self.mult > 0: # 'if' only for clarity
                self.process_valuation()
                ord_log = max(sol.value.v.ord_log for sol in self.irred_factor_cols)
            # Check if we have converged
            if self.shift > last_singular_index:
                est = max(sol.value.error_estimate()
                          for sol in self.irred_factor_cols)
                done, tail_bound = stop.check(cb, False, self.shift, tail_bound,
                               est, next_stride=self.shift-first_singular_index)
            if self.shift > 16 or self.mult > 0:
                logger.log(logging.INFO if self.shift > 1000 else logging.DEBUG,
                        "n=%s, logs=%s, est=%s, tb=%s",
                        self.shift, fwd.rec_mat.degree(), est, tail_bound)
            if done:
                break
            prev = self.shift
        logger.info("summed %d terms, tails <= %s", self.shift, tail_bound)
        # logger.debug("abstract partial sums:\n* %s",
        #         '\n* '.join(str(sol) for sol in self.irred_factor_cols))
        self.modZ_class_tail_bound = tail_bound

    def modZ_class_partial_sums(self, Jets):
        # overestimation: partial_sum() only requires a common bounds on all
        # involved series, we are passing it a bound on the Frobenius norm of a
        # a matrix of such series
        return [FundamentalSolution(
                leftmost = rt,
                shift = sol.shift,
                log_power = sol.log_power,
                value = sol.value.partial_sum(Jets, self.pt, self.leftmost, rt,
                                         sol.shift, self.modZ_class_tail_bound))
            for rt in self.roots
            for sol in self.irred_factor_cols]

    def fun(self, ini):
        # *Abstract* solution, will be replaced by one or several concrete
        # solutions later on.
        return SolutionColumn(self.matrix_rec, self.shift, self.log_power)

def fundamental_matrix_regular(dop, pt, eps, fail_fast, effort, ctx=dctx):
    rows = pt.jet_order
    cols = MatrixRecsUnroller(dop, pt, eps, rows, ctx).run()
    return matrix([sol.value for sol in cols]).transpose()

def _can_use_CBF(*doms):
    return all((isinstance(dom, (RealBallField, ComplexBallField))
                    or dom is QQ or utilities.is_QQi(dom)
                    or dom is RLF or dom is CLF)
                for dom in doms)

def _can_use_RBF(*doms):
    return all(isinstance(dom, RealBallField) or dom is QQ or dom is RLF
               for dom in doms)

def _specialization_map(source, dest, abstract_alg, alg):
    r"""
    Replace abstract_alg = gen/den by alg

    Given an abstract number field source and an element abstract_alg of the
    form source.gen()/q, return a map to dest (≈ ℂ) that maps abstract_alg to
    alg.
    """
    if alg == abstract_alg:
        return dest
    assert is_NumberField(abstract_alg.parent())
    den = abstract_alg.denominator()
    assert den*abstract_alg == abstract_alg.parent().gen()
    Homset = source.Hom(dest)
    base = source.base_ring()
    if base is QQ:
        hom = Homset([dest(den*alg)], check=False)
    else:
        base_hom = base.hom([dest(base.gen())], check=False)
        try:
            hom = Homset([dest(den*alg)], base_map=base_hom, check=False)
        except TypeError: # temporary kludge for sage < 9.0
            hom = Homset([dest(den*alg)], base_hom=base_hom, check=False)
    return hom
