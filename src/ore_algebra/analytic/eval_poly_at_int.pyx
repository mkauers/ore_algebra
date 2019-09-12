# cython: language=c++
# cython: language_level=3
r"""
Evaluation of polynomials at Python integers with limited overhead
"""

from cpython.int cimport PyInt_AsLong

from sage.libs.arb.acb cimport *
from sage.libs.arb.acb_poly cimport *
from sage.libs.arb.arb cimport *
from sage.libs.flint.fmpz cimport *
from sage.libs.flint.fmpz_poly cimport *
from sage.libs.gmp.mpz cimport *

from sage.rings.complex_arb cimport ComplexBall
from sage.rings.number_field.number_field_element_quadratic cimport NumberFieldElement_quadratic
from sage.rings.polynomial.polynomial_complex_arb cimport Polynomial_complex_arb
from sage.rings.polynomial.polynomial_element cimport Polynomial
from sage.rings.polynomial.polynomial_integer_dense_flint cimport Polynomial_integer_dense_flint
from sage.rings.ring cimport Ring
from sage.structure.parent cimport Parent

import cython

def cbf(pol, n, tgt):
    cdef Polynomial_complex_arb _pol = (<Polynomial_complex_arb> pol)
    cdef long _n = PyInt_AsLong(n)
    cdef long i

    cdef ComplexBall res = <ComplexBall> (ComplexBall.__new__(ComplexBall))
    res._parent = <Parent> tgt

    cdef long prec = _pol._parent._base._prec

    acb_zero(res.value)
    for i in range(acb_poly_degree(_pol.__poly), -1, -1):
        acb_mul_si(res.value, res.value, _n, prec)
        acb_add(res.value, res.value, acb_poly_get_coeff_ptr(_pol.__poly, i), prec)

    return res

cdef NumberFieldElement_quadratic _qnf(pol, n):

    cdef Polynomial _pol = (<Polynomial> pol)
    cdef unsigned long _n = PyInt_AsLong(n)
    cdef unsigned long i

    cdef NumberFieldElement_quadratic res
    res = (<NumberFieldElement_quadratic> (<Ring> _pol._parent._base)._zero_element)._new()
    mpz_set_ui(res.a, 0)
    mpz_set_ui(res.b, 0)
    mpz_set_ui(res.denom, 1)

    cdef NumberFieldElement_quadratic c

    for i in range(pol.degree(), -1, -1):
        mpz_mul_ui(res.a, res.a, _n)
        mpz_mul_ui(res.b, res.b, _n)
        c = _pol.get_unsafe(i)
        # In general, quadratic algebraic integers can have c != 1, but the way
        # we choose the generators of our quadratic number fields should avoid
        # that.
        assert mpz_cmp_ui(c.denom, 1) == 0
        mpz_add(res.a, res.a, c.a)
        mpz_add(res.b, res.b, c.b)

    return res

def qnf(pol, n, tgt):
    assert tgt is (<Polynomial> pol)._parent._base
    return _qnf(pol, n)

def qnf_to_cbf(pol, n, tgt):
    cdef NumberFieldElement_quadratic val = _qnf(pol, n)
    return val._acb_(tgt)

@cython.boundscheck(False)
def qqi_to_cbf(zzpols, n, tgt):

    cdef list _pols = <list> zzpols
    cdef fmpz_t _n

    fmpz_init(_n)
    fmpz_set_ui(_n, PyInt_AsLong(n))

    cdef ComplexBall res = ComplexBall.__new__(ComplexBall)
    res._parent = tgt

    cdef Polynomial_integer_dense_flint pol
    cdef fmpz_t tmp
    fmpz_init(tmp)

    pol = <Polynomial_integer_dense_flint> (zzpols[0])
    fmpz_poly_evaluate_fmpz(tmp, pol.__poly, _n)
    arb_set_fmpz(acb_realref(res.value), tmp)

    pol = <Polynomial_integer_dense_flint> (zzpols[1])
    fmpz_poly_evaluate_fmpz(tmp, pol.__poly, _n)
    arb_set_fmpz(acb_imagref(res.value), tmp)

    fmpz_clear(tmp)
    fmpz_clear(_n)

    return res
