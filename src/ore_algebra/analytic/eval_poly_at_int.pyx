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
from sage.rings.integer cimport Integer
from sage.rings.number_field.number_field_element_quadratic cimport NumberFieldElement_quadratic
from sage.rings.polynomial.polynomial_complex_arb cimport Polynomial_complex_arb
from sage.rings.polynomial.polynomial_element cimport Polynomial, Polynomial_generic_dense
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

cdef int _qnf(mpz_t a, mpz_t b, Polynomial_generic_dense pol, n) except -1:

    cdef unsigned long _n = PyInt_AsLong(n)
    cdef long i

    cdef NumberFieldElement_quadratic c

    mpz_set_ui(a, 0)
    mpz_set_ui(b, 0)

    for i in range(len(pol.__coeffs) - 1, -1, -1):
        mpz_mul_ui(a, a, _n)
        mpz_mul_ui(b, b, _n)
        c = pol.get_unsafe(i)
        # In general, quadratic algebraic integers can have denom != 1, but we
        # choose the generators of our quadratic number fields should avoid
        # that in most cases, and exceptional cases where we do have to work
        # with a "bad" number fields should not take this code path.
        assert mpz_cmp_ui(c.denom, 1) == 0
        mpz_add(a, a, c.a)
        mpz_add(b, b, c.b)

def qnf(pol, n, tgt):
    cdef Polynomial _pol = (<Polynomial_generic_dense> pol)
    assert tgt is _pol._parent._base

    cdef NumberFieldElement_quadratic res
    res = (<NumberFieldElement_quadratic> (<Ring> _pol._parent._base)._zero_element)._new()
    mpz_set_ui(res.denom, 1)
    _qnf(res.a, res.b, _pol, n)
    return res

def qnf_to_cbf(pol, n, tgt):
    # Adapted from the implementation of NumberFieldElement_quadratic._a[rc]_ in Sage.
    cdef Polynomial _pol = <Polynomial_generic_dense> pol
    cdef Integer D = <Integer> _pol._parent._base._D
    cdef bint standard_embedding = _pol._parent._base._standard_embedding

    cdef mpz_t a, b
    mpz_init(a)
    mpz_init(b)
    _qnf(a, b, _pol, n)

    cdef ComplexBall res = ComplexBall.__new__(ComplexBall)
    res._parent = tgt
    cdef fmpz_t tmp
    fmpz_init(tmp)

    fmpz_set_mpz(tmp, a)
    arb_set_fmpz(acb_realref(res.value), tmp)

    fmpz_set_mpz(tmp, D.value)
    cdef long prec = tgt._prec
    cdef long prec2 = prec
    if mpz_sgn(D.value) < 0:
        fmpz_neg(tmp, tmp)
    if mpz_sgn(a)*mpz_sgn(b) > 0 ^ standard_embedding:
        # we expect a lot of cancellation in the subtraction that follows
        prec2 = max(prec, <long> mpz_sizeinbase(b, 2))
    arb_sqrt_fmpz(acb_imagref(res.value), tmp, prec2)
    if not standard_embedding:
        arb_neg(acb_imagref(res.value), acb_imagref(res.value))

    fmpz_set_mpz(tmp, b)
    if mpz_sgn(D.value) < 0:
        arb_mul_fmpz(acb_imagref(res.value), acb_imagref(res.value), tmp, prec)
    else:
        arb_addmul_fmpz(acb_realref(res.value), acb_imagref(res.value), tmp, prec)
        arb_zero(acb_imagref(res.value))

    fmpz_clear(tmp)
    mpz_clear(b)
    mpz_clear(a)
    return res

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
