r"""
Lower-level reimplementation of key subroutines of binary_splitting
"""

from __future__ import print_function

from sage.libs.arb.types cimport *
from sage.libs.arb.acb cimport *
from sage.libs.arb.acb_poly cimport *
from sage.libs.arb.acb_mat cimport *

from sage.matrix.matrix_complex_ball_dense cimport Matrix_complex_ball_dense
from sage.rings.complex_arb cimport ComplexBall
from sage.rings.polynomial.polynomial_complex_arb cimport Polynomial_complex_arb
from sage.rings.polynomial.polynomial_element cimport Polynomial

import cython

from . import binary_splitting

cdef inline acb_mat_struct* acb_mat_poly_coeff_ptr(Polynomial p, long i):
    return (<Matrix_complex_ball_dense> (p.get_coeff_c(i))).value

# may become a cdef class (and no longer inherit from the Python version) in
# the future
class StepMatrix_arb(binary_splitting.StepMatrix_arb):

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_sums_row(low, high):

        cdef size_t j, q, p, k, u, v
        cdef acb_ptr a, b, c
        cdef ComplexBall entry
        cdef acb_mat_struct *mat
        cdef list l

        Scalars = high.zero_sum.parent()

        cdef unsigned int ordrec = high.rec_mat.base_ring().nrows()
        cdef unsigned long prec = Scalars.precision()

        cdef acb_poly_struct *low_pow_num # why isn't this working with _ptr?
        low_pow_num = (<Polynomial_complex_arb> (low.pow_num)).__poly
        cdef Polynomial low_rec_mat = <Polynomial> (low.rec_mat)

        cdef acb_t high_den
        acb_init(high_den)
        acb_mul(high_den, (<ComplexBall> high.rec_den).value,
                          (<ComplexBall> high.pow_den).value, prec)

        # sums_row = high.sums_row*low.rec_mat*low.pow_num
        #             δ, Sk, row     Sk, mat      δ
        #
        #            + high.rec_den*high.pow_den*low.sums_row
        #                 cst (nf)        cst        δ, Sk, row

        cdef acb_t t
        acb_init(t)

        res1 = [None]*len(high.sums_row)
        for j in xrange(ordrec):
            res2 = [None]*high.ord_log
            for q in xrange(high.ord_log):
                res3 = [None]*low.ord_diff
                for p in xrange(low.ord_diff):

                    entry = ComplexBall.__new__(ComplexBall)
                    entry._parent = Scalars
                    acb_zero(entry.value)

                    # explicit casts to acb_ptr and friends are there as a
                    # workaround for cython bug #1984

                    # one coefficient of one entry the first term
                    # high.sums_row*low.rec_mat*low.pow_num
                    # TODO: try using acb_dot? how?
                    for k in xrange(ordrec):
                        for u in xrange(p + 1):
                            for v in xrange(q + 1):
                                l = <list> (high.sums_row)
                                l = <list> (l[k])
                                l = <list> (l[v])
                                a = <acb_ptr> (<ComplexBall> l[u]).value
                                b = acb_poly_get_coeff_ptr(low_pow_num, p-u)
                                if b == NULL:
                                    continue
                                mat = acb_mat_poly_coeff_ptr(low_rec_mat, q-v)
                                c = acb_mat_entry(mat, k, j)
                                acb_mul(t, b, c, prec)
                                acb_addmul(entry.value, t, a, prec)

                    # same for the second term
                    # high.rec_den*pow_den.rec_den*low.sums_row
                    if q < low.ord_log: # usually true, but low might be
                                        # an optimized SolutionColumn
                        a = <acb_ptr> ((<ComplexBall> low.sums_row[j][q][p]).value)
                        acb_addmul(entry.value, high_den, a, prec)

                    res3[p] = entry
                res2[q] = res3
            res1[j] = res2

        acb_clear(high_den)
        acb_clear(t)

        return res1
