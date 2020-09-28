# cython: language=c++
# cython: language_level=3
r"""
Lower-level reimplementation of key subroutines of naive_sum
"""

# Copyright 2018 Marc Mezzarobba
# Copyright 2018 Centre national de la recherche scientifique
# Copyright 2018 Université Pierre et Marie Curie
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

from __future__ import print_function

from libc.stdlib cimport malloc, free
from sage.libs.arb.types cimport *
from sage.libs.arb.mag cimport *
from sage.libs.arb.arb cimport *
from sage.libs.arb.acb cimport *
from sage.libs.arb.acb_poly cimport *
from sage.rings.complex_arb cimport ComplexBall
from sage.rings.polynomial.polynomial_complex_arb cimport Polynomial_complex_arb
from sage.rings.real_arb cimport RealBall
from sage.structure.parent cimport Parent

cdef extern from "acb.h":
    void acb_dot(acb_t res, const acb_t s, bint subtract, acb_srcptr x, long
            xstep, acb_srcptr y, long ystep, long len, long prec)

import cython

from . import accuracy, naive_sum

class PartialSum(naive_sum.PartialSum):

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def next_term(self, py_n, py_mult, py_bwrec_n not None, py_cst, jetpow, squash):

        cdef ssize_t a, b, i, j, k, p, dot_length
        cdef list bwrec_n_i, last_i, last_0
        cdef ComplexBall ball
        cdef RealBall err
        cdef Polynomial_complex_arb poly

        cdef ssize_t n = py_n
        cdef ssize_t mult = py_mult
        cdef list bwrec_n = <list> py_bwrec_n
        cdef ssize_t log_prec = self.log_prec
        cdef ssize_t ordrec = self.ordrec
        cdef list psum = self.psum
        cdef ssize_t prec = self.Intervals.precision()
        cdef Parent Intervals = self.Intervals
        cdef Parent IR = accuracy.IR
        cdef ComplexBall cst = <ComplexBall> py_cst

        cdef object last = self.last

        assert n == self.trunc
        last.rotate(1)
        self.trunc += 1

        zero = Intervals.zero()

        if mult > 0:
            self.last[0] = [zero for _ in range(self.log_prec + mult)]

        last_0 = <list> (self.last[0])

        dot_length = log_prec*(ordrec + 1) + mult
        cdef acb_struct *left  = <acb_struct *> malloc(dot_length*sizeof(acb_struct))
        cdef acb_struct *right = <acb_struct *> malloc(dot_length*sizeof(acb_struct))

        err = None
        for p in range(log_prec - 1, -1, -1):
            k = 0
            for i in range(ordrec, -1, -1):
                bwrec_n_i = <list> (bwrec_n[i])
                last_i = <list> (self.last[i])
                if i != 0:
                    a, b = 0, log_prec - p
                else:
                    a, b = mult + 1, mult + log_prec - p
                for j in range(a, b):
                    left[k] = ((<ComplexBall> bwrec_n_i[j]).value)[0]
                    right[k] = ((<ComplexBall> last_i[p+j]).value)[0]
                    k += 1
            ball = <ComplexBall> ComplexBall.__new__(ComplexBall)
            ball._parent = Intervals
            acb_zero(ball.value)
            acb_dot(ball.value, ball.value, False, left, 1, right, 1, k, prec)
            acb_mul(ball.value, cst.value, ball.value, prec)
            if mult == p == 0 and squash:
                err = <RealBall> RealBall.__new__(RealBall)
                err._parent = IR
                acb_get_rad_ubound_arf(arb_midref(err.value), ball.value, MAG_BITS)
                mag_zero(arb_radref(acb_realref(ball.value)))
                mag_zero(arb_radref(acb_imagref(ball.value)))
            last_0[mult + p] = ball

        free(right)
        free(left)

        for p in range(mult - 1, -1, -1):
            last_0[p] = Intervals(self.ini.shift[n][p])

        if mult > 0:
            self.handle_singular_index(n, mult)
            log_prec = self.log_prec

        if log_prec == mult == 0:
            return accuracy.IR.zero()

        for k in range(log_prec):
            # XXX reuse existing object?
            poly = <Polynomial_complex_arb> Polynomial_complex_arb.__new__(Polynomial_complex_arb)
            poly._parent = (<Polynomial_complex_arb> (psum[k]))._parent
            acb_poly_scalar_mul(
                    poly.__poly,
                    (<Polynomial_complex_arb> jetpow).__poly,
                    (<ComplexBall> last_0[k]).value,
                    prec)
            acb_poly_add(
                    poly.__poly,
                    (<Polynomial_complex_arb> (psum[k])).__poly,
                    poly.__poly,
                    prec)
            psum[k] = poly

        return err
