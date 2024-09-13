# cython: language_level=3
# vim: tw=80
r"""
Divide-and-conquer summation of convergent D-finite series
"""

from libc.stdlib cimport malloc, free, calloc

from sage.libs.flint.types cimport *
from sage.libs.flint.acb cimport *
from sage.libs.flint.acb_mat cimport *
from sage.libs.flint.acb_poly cimport *
from sage.libs.flint.arb cimport *
from sage.libs.flint.fmpq_poly cimport *
from sage.libs.flint.fmpz cimport *
from sage.libs.flint.fmpz_mat cimport *
from sage.libs.flint.fmpz_poly cimport *
from sage.libs.flint.fmpz_vec cimport *
from sage.libs.flint.gr cimport *
from sage.libs.flint.mag cimport *

from sage.libs.flint.gr_mat cimport gr_mat_pascal

cdef extern from "flint_wrap.h":
    void gr_ctx_init_fmpz(gr_ctx_t ctx) noexcept
    void GR_MUST_SUCCEED(int status) noexcept
    mp_limb_t FLINT_BIT_COUNT(mp_limb_t x) noexcept

from sage.rings.complex_arb cimport ComplexBall
from sage.rings.polynomial.polynomial_complex_arb cimport Polynomial_complex_arb
from sage.rings.real_arb cimport RealBall
from sage.structure.parent cimport Parent


import cython
import logging

from itertools import count

from . import accuracy

from .context import dctx


logger = logging.getLogger(__name__)


cdef slong APPLY_DOP_INTERPOLATION_MAX_POINTS = 256


# MAYBE-TODO:
#   - Decouple the code for computing (i) coefficients, (ii) partial sums,
# (iii) sums with error bounds. Could be done by subclassing DACUnroller and/or
# introducing additional classes for sums etc. Use this when computing formal
# log-series solutions.
#   - Add support for inhomogeneous equations with polynomial rhs. (Always
# sum at least up to deg(rhs) when doing error control).
#   - Could keep only the last deg coeffs even in the DAC part (=> slightly
# different indexing conventions).


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DACUnroller:
    r"""
    Several methods work in place on ``self.series``. The arguments ``low``,
    ``mid``, ``high``, and ``n`` are shifts wrt λ; the index into ``series``
    corresponding to ``n`` is ``n - base``. In other words,
    ``self.series[k][n-base]`` stores the coefficient of `x^{λ+n}·\log(x)^k/k!`
    (both on input and on output, in the solution and in the rhs).
    """

    cdef slong dop_order
    cdef slong dop_degree
    cdef slong numpts
    cdef slong jet_order
    cdef slong prec         # general working precision (bits)
    cdef slong bounds_prec  # bit precision for error bounds

    # max length wrt log of any of the coefficients of the solution, the rhs,
    # or the sum
    cdef slong max_log_prec  # future
    cdef slong log_prec      # to date

    cdef acb_poly_struct *dop_coeffs
    cdef fmpz_poly_struct *dop_coeffs_fmpz  # TODO: coeffs in ℤ[i]
    cdef bint dop_is_exact

    cdef acb_poly_t ind  # indicial polynomial
    cdef acb_t leftmost  # aka λ, exponent of the _group_ of solutions

    cdef dict ini          # n -> coeff of x^{λ+n}·log(x)^k/k!, 0≤k<mult(λ+n)
    cdef slong last_ini_n  # position of last initial value (shift wrt λ)

    cdef acb_ptr evpts  # evaluation points x[i], 0 < i < numpts
    cdef arb_t rad      # ≥ abs(evaluation points), for error bounds

    # main shared buffers

    cdef acb_ptr pows   # x[i]^n

    # Unlike the old Python version, which creates/destroys/returns lists of
    # polynomials, this version acts on shared data buffers.
    #
    # The coefficient of a given log(x)^k/k! is represented as a contiguous
    # array of coefficients (= FLINT polynomial), and subroutines that operate
    # on slices of coefficients take as input offsets in this array (as opposed
    # to direct pointers), with the same offset typically applying for all k.

    # vector of polynomials in x holding the coefficients wrt log(x)^k/k! of the
    # last few (???) terms of the series solution
    cdef acb_poly_struct *series

    # vector of polynomials in δ (perturbation of ξ = evaluation point) holding
    # the jets of coefficients wrt log(ξ)^k/k! of the partial sums:
    # self.sums + j*self.max_log_prec + k is the jet of order self.jet_order
    # of the coeff of log^k/k! in the sum at the point of index j
    cdef acb_poly_struct *sums

    # internal data -- next_sum

    cdef fmpz *binom_n  # binom(n, j) for j < jet_order

    # internal data -- apply_dop_basecase

    cdef fmpz_mat_t binom  # binom(t, k) for k ≤ t < dop_order

    # internal data -- apply_dop_interpolation

    cdef acb_mat_struct *tinterp_cache_num
    cdef fmpz *tinterp_cache_den
    cdef slong tinterp_cache_size
    cdef slong apply_dop_interpolation_max_len

    # internal data -- error bounds

    cdef slong _rhs_offset

    # internal data -- remaining python code

    cdef readonly object py_evpts, IR, IC
    cdef object Ring, Pol, Reals, Pol_IC

    # auxiliary outputs ("cdef readonly" means read-only _from Python_!)

    cdef readonly object Jets
    cdef readonly bint real
    cdef readonly dict critical_coeffs


    def __cinit__(self, dop_T, ini, py_evpts, *args, **kwds):
        cdef slong i

        cdef size_t sz_poly = sizeof(acb_poly_struct)
        self.dop_order = dop_T.order()
        self.dop_degree = dop_T.degree()
        self.numpts = len(py_evpts)
        self.jet_order = py_evpts.jet_order

        self.dop_coeffs = <acb_poly_struct *> malloc((self.dop_order+1)*sz_poly)
        self.dop_coeffs_fmpz = (<fmpz_poly_struct *> malloc((self.dop_order + 1)
                                                     *sizeof(fmpz_poly_struct)))
        for i in range(self.dop_order + 1):
            acb_poly_init(self.dop_coeffs + i)
            fmpz_poly_init(self.dop_coeffs_fmpz + i)

        # using dop_order as a crude bound for max log prec
        # (needs updating to support inhomogeneous equations)
        self.max_log_prec = self.dop_order
        self.log_prec = 0
        self.series = <acb_poly_struct *> malloc(self.max_log_prec*sz_poly)
        for i in range(self.max_log_prec):
            acb_poly_init(self.series + i)

        self.evpts = _acb_vec_init(self.numpts)
        self.pows =  _acb_vec_init(self.numpts)
        self.binom_n = _fmpz_vec_init(self.jet_order)

        self.sums = <acb_poly_struct *> malloc(self.numpts
                                               *self.max_log_prec*sz_poly)
        for i in range(self.numpts*self.max_log_prec):
            acb_poly_init(self.sums + i)
            acb_poly_fit_length(self.sums + i, self.jet_order)

        acb_poly_init(self.ind)
        acb_init(self.leftmost)
        arb_init(self.rad)

        self.init_binom(self.dop_order + 1)

        self.tinterp_cache_num = NULL
        self.tinterp_cache_den = NULL
        self.tinterp_cache_size = 0


    def __dealloc__(self):
        cdef slong i

        self.clear_binom()

        arb_clear(self.rad)
        acb_clear(self.leftmost)
        acb_poly_clear(self.ind)

        for i in range(self.numpts*self.max_log_prec):
            acb_poly_clear(self.sums + i)
        free(self.sums)

        _fmpz_vec_clear(self.binom_n, self.jet_order)
        _acb_vec_clear(self.pows, self.numpts)
        _acb_vec_clear(self.evpts, self.numpts)

        for i in range(self.max_log_prec):
            acb_poly_clear(self.series + i)
        free(self.series)

        for i in range(self.dop_order + 1):
            acb_poly_clear(self.dop_coeffs + i)
            fmpz_poly_clear(self.dop_coeffs_fmpz + i)
        free(self.dop_coeffs)
        free(self.dop_coeffs_fmpz)


    def __init__(self, dop_T, ini, py_evpts, Ring, *, ctx=dctx):

        cdef slong i, j
        cdef acb_poly_struct *p

        assert dop_T.parent().is_T()

        ## Parents

        self.Ring = Ring
        self.Reals = Ring.base()
        # Slices of series solns (w/o logs); must contain λ.
        self.Pol = dop_T.base_ring().change_ring(Ring)
        # Values (of the evaluation point, of partial sums). Currently limited
        # to using a ball field as a base ring, but this may change.
        self.Jets = None  # see below
        # Error bounds
        self.IR = ctx.IR
        self.IC = self.IR.complex_field()
        self.Pol_IC = self.Pol.change_ring(self.IC)

        ## Internal data

        # maybe change this to a plain c data structure
        self.ini = {k: tuple(Ring(a) for a in v)
                     for k, v in ini.shift.items()}
        self.last_ini_n = ini.last_index()

        self.dop_is_exact = True
        for i, pol in enumerate(dop_T):
            p = self.dop_coeffs + i
            # This truncates the coefficients to self.prec
            acb_poly_swap(p, (<Polynomial_complex_arb?> (self.Pol(pol)))._poly)
            for j in range(acb_poly_length(p)):
                self.dop_is_exact = (self.dop_is_exact
                                     and acb_is_exact(_coeffs(p) + j))
            self.dop_is_exact = (self.dop_is_exact
                                 and acb_poly_get_unique_fmpz_poly(
                                                   self.dop_coeffs_fmpz + i, p))

        # ini.expo is a PolynomialRoot, consider doing part of the work
        # with its exact value instead of an interval approximation
        leftmost = Ring(ini.expo)
        acb_set(self.leftmost, (<ComplexBall?> leftmost).value)

        acb_poly_fit_length(self.ind, self.dop_order + 1)
        _acb_poly_set_length(self.ind, self.dop_order + 1)
        for i in range(self.dop_order + 1):  # (unnecessary copies)
            acb_poly_get_coeff_acb(_coeffs(self.ind) + i, self.dop_coeffs + i, 0)
        _acb_poly_normalise(self.ind)

        self.prec = Ring.precision()
        self.bounds_prec = self.IR.precision()

        self.py_evpts = py_evpts
        Jets, jets = py_evpts.jets(Ring)
        for i in range(self.numpts):
            assert jets[i].degree() == 0 or jets[i].degree() == 1 and jets[i][1].is_one()
            acb_poly_get_coeff_acb(self.evpts + i,
                                   (<Polynomial_complex_arb?> jets[i])._poly,
                                   0)
        arb_set(self.rad, (<RealBall?> py_evpts.rad).value)

        ## Auxiliary output (some also used internally)

        self.critical_coeffs = {}
        self.Jets = Jets
        # At the moment Ring must in practice be a complex ball field (other
        # rings do not support all required operations); this flag signals that
        # the series (incl. singular part) and evaluation points are real.
        self.real = (py_evpts.is_real_or_symbolic
                     and ini.is_real(dop_T.base_ring().base_ring()))


    cdef acb_poly_struct *sum_ptr(self, int j, int k) noexcept:
        return self.sums + j*self.max_log_prec + k


    ## Main summation loop


    # Maybe get rid of this and use sum_dac only?
    def sum_blockwise(self, stop):
        cdef slong i, j, k
        cdef acb_ptr c
        cdef arb_t est, tb
        cdef arb_t radpow, radpow_blk
        cdef mag_t coeff_rad

        # Block size must be >= deg. Power-of-two factors may be beneficial when
        # using apply_dop_interpolation.
        cdef slong blksz = max(1, self.dop_degree)
        # cdef slong blksz = 1 << (self.dop_degree - 1).bit_length()
        cdef slong blkstride = max(1, 32//blksz)

        arb_init(radpow)
        arb_pow(radpow, self.rad, acb_imagref(self.leftmost), self.bounds_prec)
        arb_init(radpow_blk)
        arb_pow_ui(radpow_blk, self.rad, blksz, self.bounds_prec)

        arb_init(est)
        arb_pos_inf(est)
        arb_init(tb)
        arb_pos_inf(tb)
        mag_init(coeff_rad)

        self.apply_dop_interpolation_max_len = min(  # threshold TBI
            APPLY_DOP_INTERPOLATION_MAX_POINTS,
            self.prec,
            2*blksz)
        self.tinterp_cache_init(self.apply_dop_interpolation_max_len//2)

        for i in range(self.numpts):
            acb_one(self.pows + i)
        for k in range(self.max_log_prec):
            acb_poly_zero(self.series + k)
            acb_poly_fit_length(self.series + k, 2*blksz)
        for i in range(self.numpts*self.max_log_prec):
            acb_poly_zero(self.sums + i)
            _acb_poly_set_length(self.sums + i, self.jet_order)

        cdef bint done = False
        cdef slong b = 0
        while True:
            self.sum_dac(b*blksz, b*blksz, (b+1)*blksz)
            self.apply_dop(b*blksz, b*blksz, (b+1)*blksz, (b+2)*blksz)

            # Support stopping in the middle of a block when dop_degree is
            # large? Would need the ability to compute the high part of the
            # residual (to low precision).
            if b % blkstride == 0:
                if self.check_convergence(stop, (b+1)*blksz, blksz,
                                          est, tb, radpow, blkstride*blksz):
                    break

            for k in range(self.log_prec):
                acb_poly_shift_right(self.series + k, self.series + k, blksz)

            arb_mul(radpow, radpow, radpow_blk, self.bounds_prec)

            b += 1

        self.fix_sums()

        psums = [[None]*self.log_prec for _ in range(self.numpts)]
        cdef Polynomial_complex_arb psum
        for j in range(self.numpts):  # psum in psums
            for k in range(self.log_prec):  # jet in psum
                for i in range(self.jet_order):
                    c = _coeffs(self.sum_ptr(j, k)) + i
                    arb_add_error(acb_realref(c), tb)
                    if self.real:
                        assert arb_is_zero(acb_imagref(c))
                    else:
                        arb_add_error(acb_imagref(c), tb)
                    mag_max(coeff_rad, coeff_rad, arb_radref(acb_realref(c)))
                    mag_max(coeff_rad, coeff_rad, arb_radref(acb_imagref(c)))
                psum = Polynomial_complex_arb.__new__(Polynomial_complex_arb)
                psum._parent = self.Jets
                acb_poly_swap(psum._poly, self.sum_ptr(j, k))
                psums[j][k] = psum

        self._report_stats((b+1)*blksz, est, tb, coeff_rad)

        self.tinterp_cache_clear()
        arb_clear(tb)
        arb_clear(est)
        arb_clear(radpow_blk)
        arb_clear(radpow)
        mag_clear(coeff_rad)

        return psums


    cdef void sum_dac(self, slong base, slong low, slong high) noexcept:
        r"""
        Compute the chunk ``y[λ+low:λ+high]`` of the solution of ``L(y) = rhs``
        for a given rhs itself of support contained in ``λ+low:λ+high``.
        Works in place on ``self.series[:][low-base:high-base]``.
        """
        # XXX should it be L(y) = -rhs in the above docstring?

        assert base <= low <= high

        if high == low:
            return

        if high == low + 1:
            self.next_term(base, low)
            self.next_sum(base, low)
            return

        mid = (high + low)//2
        # mid = (high + low + 1)//2

        self.sum_dac(base, low, mid)
        resid_len = min(high - mid, self.dop_degree + 1)
        self.apply_dop(base, low, mid, mid + resid_len)
        self.sum_dac(base, mid, high)


    cdef void _report_stats(self, slong n, arb_t est, arb_t tb,
                            mag_t coeff_rad):
        cdef RealBall _est = RealBall.__new__(RealBall)
        _est._parent = self.IR
        arb_swap(_est.value, est)
        cdef RealBall _tb = RealBall.__new__(RealBall)
        _tb._parent = self.IR
        arb_swap(_tb.value, tb)
        cdef RealBall _coeff_rad = RealBall.__new__(RealBall)
        _coeff_rad._parent = self.IR
        arb_set_interval_mag(_coeff_rad.value, coeff_rad, coeff_rad, MAG_BITS)
        logger.info("summed %d terms, tail bound = %s (est = %s), max rad = %s",
                    n, _tb, _est, _coeff_rad)
        arb_swap(tb, _tb.value)
        arb_swap(est, _est.value)


    ## Computation of individual terms


    cdef void next_term(self, slong base, slong n) noexcept:
        r"""
        Write to ``self.series[:][n-base]`` the coefficient of ``x^(λ+n)`` in
        the solution. Also store the corresponding critical coefficients in the
        Python dict ``self.critical_coeffs``.
        """

        assert n >= base

        cdef slong k
        cdef ComplexBall tmp_ball

        cdef tuple ini = self.ini.get(n, ())
        cdef slong mult = len(ini)

        # Probably don't need the max(len(rhs), ...) here since we initialize
        # series once and for all as a table of length max_log_prec.
        cdef slong rhs_len = self.log_prec
        while (rhs_len > 0
               and acb_is_zero(_coeffs(self.series + rhs_len - 1) + n - base)):
            rhs_len -= 1
        assert rhs_len + mult <= self.max_log_prec

        # Evaluate the indicial polynomial at λ + n + δ
        # XXX cache for the benefit of get_residuals? use multipoint evaluation?
        cdef acb_poly_t ind_n
        acb_poly_init(ind_n)
        self.eval_ind(ind_n, n, rhs_len + mult)

        cdef fmpz_t lc
        fmpz_init(lc)
        cdef acb_t invlc
        acb_init(invlc)

        # Compute the high-order (wrt log) part of the new term from the
        # coefficient of x^{λ+n} on the rhs and the indicial polynomial.

        # Working on copies allows us to use acb_dot.
        cdef acb_struct *new_term  # XXX reuse?
        new_term = <acb_struct *> malloc(rhs_len*sizeof(acb_struct))
        for k in range(rhs_len):
            acb_init(new_term + k)

        for k in range(rhs_len - 1, -1, -1):
            # combin = rhs[k][0] + sum(ind_n[mult+1+j]*new_term[k+1+j], j≥0)
            acb_dot(new_term + k,  # result
                    _coeffs(self.series + k) + n - base,  # initial value
                    False,  # subtract the sum from the initial value
                    _coeffs(ind_n) + mult + 1, 1,  # first vec, step
                    new_term + k + 1, 1,  # second vec, step
                    rhs_len - k - 1,  # terms
                    self.prec)
            if acb_is_zero(new_term + k):
                continue
            if acb_is_zero(invlc):
                if (acb_is_exact(_coeffs(ind_n) + mult)
                        and acb_get_unique_fmpz(lc, _coeffs(ind_n) + mult)):
                    acb_indeterminate(invlc)
                else:
                    acb_inv(invlc, _coeffs(ind_n) + mult, self.prec)
            if not fmpz_is_zero(lc):
                acb_div_fmpz(new_term + k, new_term + k, lc, self.prec)
            else:
                acb_mul(new_term + k, new_term + k, invlc, self.prec)
            acb_neg(new_term + k, new_term + k)

        # Write the new term to self.series

        for k in range(mult):
            acb_poly_set_coeff_acb(self.series + k, n - base,
                                   (<ComplexBall?> ini[k]).value)
        for k in range(mult, rhs_len + mult):
            if acb_poly_length(self.series + k) <= n - base:  # XXX crucial, but is this the right place to do it?
                _acb_poly_set_length(self.series + k, n - base + 1)
            acb_swap(_coeffs(self.series + k) + n - base,
                     new_term + k - mult)
        for k in range(rhs_len + mult, self.log_prec):
            acb_poly_set_coeff_si(self.series + k, n - base, 0)

        # Store the critical coefficients

        crit = None
        if mult > 0:
            crit = [None]*(rhs_len + mult)
            for k in range(rhs_len + mult):
                tmp_ball = <ComplexBall> ComplexBall.__new__(ComplexBall)
                acb_set(tmp_ball.value, _coeffs(self.series + k) + n - base)
                tmp_ball._parent = self.Ring
                crit[k] = tmp_ball
            self.critical_coeffs[n] = crit

        # Update log-degree

        self.log_prec = max(self.log_prec, rhs_len + mult)

        for k in range(rhs_len):
            acb_clear(new_term + k)
        free(new_term)
        acb_clear(invlc)
        fmpz_clear(lc)
        acb_poly_clear(ind_n)


    # - cache (cf. get_residuals)?
    # - use multi-point evaluation?
    cdef void eval_ind(self, acb_poly_t ind_n, slong n, int order) noexcept:
        cdef slong i
        cdef acb_t expo
        cdef acb_ptr c
        acb_poly_fit_length(ind_n, order)
        if order == 1 and acb_is_zero(self.leftmost):  # covers ordinary points
            c = _coeffs(ind_n)
            acb_zero(c)
            for i in range(acb_poly_degree(self.ind), -1, -1):
                acb_mul_si(c, c, n, self.prec)
                acb_add(c, c, _coeffs(self.ind) + i, self.prec)
            _acb_poly_set_length(ind_n, 1)
            _acb_poly_normalise(ind_n)
        else:  # improvable...
            acb_init(expo)
            acb_add_si(expo, self.leftmost, n, self.prec)
            acb_poly_taylor_shift(ind_n, self.ind, expo, self.prec)
            acb_poly_truncate(ind_n, order)
            acb_clear(expo)


    cdef void next_sum(self, slong base, slong n) noexcept:
        r"""
        Add to each entry of `self.sums` a term corresponding to the
        current `n`.

        WARNING: Repeated calls to this function compute x^i*f^(i)(x) instead of
        f^(i)(x). Call `fix_sums()` to fix the result.
        """
        cdef slong i, j, k
        if n == 0:
            _fmpz_vec_zero(self.binom_n, self.jet_order)
            fmpz_one(self.binom_n)
        else:
            for i in range(self.jet_order - 1, 0, -1):
                fmpz_add(self.binom_n+i, self.binom_n+i, self.binom_n+i-1)
        cdef acb_t tmp
        acb_init(tmp)
        for j in range(self.numpts):
            for k in range(self.log_prec):
                acb_mul(tmp,
                        _coeffs(self.series + k) + n - base,
                        self.pows + j,
                        self.prec)
                for i in range(self.jet_order):
                    acb_addmul_fmpz(_coeffs(self.sum_ptr(j, k)) + i,
                                    tmp, self.binom_n + i, self.prec)
            acb_mul(self.pows + j, self.pows + j, self.evpts + j, self.prec)
        acb_clear(tmp)


    cdef void fix_sums(self) noexcept:
        r"""
        Fix the partial sums computed by `next_sum` by dividing the terms
        corresponding to derivatives by suitable powers of the corresponding
        evaluation points.
        """
        cdef slong i, j, k
        cdef acb_t inv, invpow
        acb_init(inv)
        acb_init(invpow)
        for j in range(self.numpts):
            acb_inv(inv, self.evpts + j, self.prec)
            acb_one(invpow)
            for i in range(1, self.jet_order):
                acb_mul(invpow, invpow, inv, self.prec)
                for k in range(self.log_prec):
                    acb_mul(_coeffs(self.sum_ptr(j, k)) + i,
                            _coeffs(self.sum_ptr(j, k)) + i,
                            invpow,
                            self.prec)
        acb_clear(invpow)
        acb_clear(inv)


    ## Image of a polynomial


    cdef void apply_dop(self, slong base, slong low, slong mid, slong high) noexcept:
        r"""
        *Add* to ``self.series[:][mid-base:high-base]`` the coefficients of
        ``self.dop(y[λ+low:λ+mid])``, where the input is given in
        ``self.series[:][low-base:mid-base]``.
        """
        cdef slong k

        if False:
            # too slow at the moment; may/should be useful in some range of
            # (high-low)/dop_order?
            self.apply_dop_interpolation(base, low, mid, high)
        elif False:
            # really slow, but might be useful for huge (high-low)/dop_order when
            # the bitlength coefficients of the operator is comparable to the
            # working precision
            self.apply_dop_polmul(base, low, mid, high)
        elif self.dop_is_exact and acb_is_zero(self.leftmost):
            self.apply_dop_basecase_exact(base, low, mid, high)
        else:
            self.apply_dop_basecase(base, low, mid, high)


    # Compared to a version of apply_dop that uses fast polynomial
    # multiplication (and being able to exploit fast multiplication is the
    # main point of the D&C algorithm!), this one performs amounts to a
    # naïve, quadratic-time middle product.
    #
    # However, instead of performing a separate multiplication for each
    # derivative up to the order of dop, it computes the ℤ-linear
    # combination of coefficients of dop that gives the cofactor of a given
    # term of the input series in the output series and multiplies the
    # resulting quantity by the corresponding term of the input series. This
    # is beneficial because the coefficients of dop are often exact balls of
    # bit length significantly smaller than the working precision.
    #
    # These linear combinations are polynomial evaluations at n=0,1,2,....
    # Moreover, assuming no logs for simplicity, each pair (coeff of output
    # series, degree in dop) occurs exactly once over all recursive calls.
    # So in the end this does essentially the same computation as naïve
    # recurrence unrolling, just in a different order.
    #
    # Approximate cost for high - mid = mid - low = d, log_prec = 1,
    # leftmost = 0:
    #
    # d²/2·(M(h,p) + p) + r·d²/2·h
    #
    # ((r, d, h) = (order, degree, height) of dop, p = working prec).
    # With the pure acb version, the O(r·d²·h) term can easily dominate in
    # practice, even for small r...
    cdef void apply_dop_basecase(self, slong base, slong low, slong mid,
                                 slong high) noexcept:
        cdef slong i, j, k, n, t, j0, length
        cdef acb_ptr b, c

        cdef acb_t expo
        acb_init(expo)

        cdef acb_ptr cofac = _acb_vec_init(mid - low)

        # TODO: Also optimize for rational exponents.
        cdef bint leftmost_is_zero = acb_is_zero(self.leftmost)

        for k in range(self.log_prec):
            if acb_poly_length(self.series + k) < high - base:
                acb_poly_fit_length(self.series + k, high - base)
                _acb_poly_set_length(self.series + k, high - base)

        for n in range(mid, high):

            j0 = n - mid + 1
            length = min(n - low, self.dop_degree) + 1 - j0

            for t in range(self.log_prec):

                # TODO This part could be shared between several solutions
                for j in range(j0, j0 + length):

                    # c = cofactor of current coeff of self.series in expression
                    # of current output coeff (=> collects contributions from
                    # all terms of dop while often staying exact and of moderate
                    # bit length)
                    #
                    # Each triple (n, t, j) occurs only once in the whole
                    # computation (over all recursive calls etc.). So computing
                    # all values of cofac is basically the same as evaluating
                    # all coefficients of the recurrence associated to dop at
                    # all n.

                    if not leftmost_is_zero:
                        acb_add_si(expo, self.leftmost, n - j, self.prec)

                    # - Use some kind of fast multi-point evaluation??
                    # - Use acb_poly_taylor_shift instead of looping on t???
                    # - Precompute dop coeff*binom?

                    c = cofac + j - j0
                    acb_zero(c)
                    for i in range(self.dop_order, t - 1, -1):  # Horner
                        if leftmost_is_zero:
                            acb_mul_si(c, c, n - j, self.prec)
                        else:
                            acb_mul(c, c, expo, self.prec)
                        if j >= acb_poly_length(self.dop_coeffs + i) or t > i:
                            continue
                        b = _coeffs(self.dop_coeffs + i) + j
                        if t == 0:
                            acb_add(c, c, b, self.prec)
                        elif t == 1:
                            acb_addmul_si(c, b, i, self.prec)
                        else:
                            acb_addmul_fmpz(c, b,
                                            fmpz_mat_entry(self.binom, i, t),
                                            self.prec)

                # We could perform a dot product of length log_prec*that
                # (looping over t in addition to j), but this does not seem
                # worth the additional complexity at the moment.
                for k in range(self.log_prec - t):
                    acb_dot(
                        _coeffs(self.series + k) + n - base,
                        _coeffs(self.series + k) + n - base,
                        False,
                        cofac, 1,
                        _coeffs(self.series + k + t) + mid - 1 - base, -1,
                        length,
                        self.prec)


        _acb_vec_clear(cofac, mid - low)
        acb_clear(expo)


    # Same as of apply_dop_basecase but using fmpz, for operators with exact
    # integer coefficients
    #
    # TODO: support coefficients in ℤ[i]
    cdef void apply_dop_basecase_exact(self, slong base, slong low, slong mid,
                                       slong high) noexcept:

        cdef slong i, j, k, n, t, j0, length
        cdef fmpz *b
        cdef fmpz *c

        cdef fmpz *cofac = _fmpz_vec_init(mid - low)

        for k in range(self.log_prec):
            if acb_poly_length(self.series + k) < high - base:
                acb_poly_fit_length(self.series + k, high - base)
                _acb_poly_set_length(self.series + k, high - base)

        for n in range(mid, high):

            j0 = n - mid + 1
            length = min(n - low, self.dop_degree) + 1 - j0

            for t in range(self.log_prec):

                # TODO This part could be shared between several solutions
                for j in range(j0, j0 + length):

                    c = cofac + j - j0

                    fmpz_zero(c)
                    for i in range(self.dop_order, t - 1, -1):  # Horner
                        fmpz_mul_si(c, c, n - j)
                        if (j >= fmpz_poly_length(self.dop_coeffs_fmpz + i)
                                or t > i):
                            continue
                        # Here special-casing t ∈ {0,1} does not help much.
                        fmpz_addmul(c,
                                    (self.dop_coeffs_fmpz + i).coeffs + j,
                                    fmpz_mat_entry(self.binom, i, t))

                for k in range(self.log_prec - t):
                    acb_dot_fmpz(
                        _coeffs(self.series + k) + n - base,
                        _coeffs(self.series + k) + n - base,
                        False,
                        _coeffs(self.series + k + t) + mid - 1 - base, -1,
                        cofac, 1,
                        length,
                        self.prec)

        _fmpz_vec_clear(cofac, mid - low)


    cdef void init_binom(self, slong s) noexcept:
        cdef gr_ctx_t fmpz
        gr_ctx_init_fmpz(fmpz)
        fmpz_mat_init(self.binom, s, s)
        GR_MUST_SUCCEED(gr_mat_pascal(<gr_mat_struct *> self.binom, -1, fmpz))
        gr_ctx_clear(fmpz)


    cdef void clear_binom(self) noexcept:
        fmpz_mat_clear(self.binom)


    # Good asymptotic complexity wrt diffop degree, but slow in practice on
    # typical inputs. Might behave better than the basecase version for large
    # degree + large integer/ball coeffs.
    cdef void apply_dop_polmul(self, slong base, slong low, slong mid,
                               slong high) noexcept:

        cdef slong i, j, k

        cdef acb_poly_t tmp
        acb_poly_init(tmp)
        acb_poly_fit_length(tmp, high - low)

        # To compute derivatives, we need a copy of a chunk of `series`
        cdef acb_poly_struct *curder
        curder = <acb_poly_struct *> malloc(self.log_prec*sizeof(acb_poly_struct))
        for k in range(self.log_prec):
            acb_poly_init(curder + k)  # XXX maybe reuse between calls
            acb_poly_shift_right(curder + k, self.series + k, low - base)
            # typically already satisfied (?)
            acb_poly_truncate(curder + k, mid - low)

        for i in range(self.dop_order + 1):

            for k in range(self.log_prec):

                # rhs[k] ← rhs[k] + self.dop(chunk)[k]

                # This should be a mulmid. In practice mullow_classical or a
                # quadratic-time naïve mulmid is faster on typical inputs...
                # XXX Try with a Karatsuba mulmid?
                # XXX Should ignore constant coefficients
                acb_poly_mullow(tmp, self.dop_coeffs + i, curder + k,
                                high - low, self.prec)
                acb_poly_shift_right(tmp, tmp, mid - low)
                if acb_poly_length(self.series + k) < high - base:
                    acb_poly_fit_length(self.series + k, high - base)
                    _acb_poly_set_length(self.series + k, high - base)
                _acb_poly_add(_coeffs(self.series + k) + mid - base,
                              _coeffs(self.series + k) + mid - base,
                              high - mid,
                              _coeffs(tmp),
                              acb_poly_length(tmp),
                              self.prec)

                # curder[k] ← (d/dx)(previous curder)[k]

                # XXX Optimize case of rational `leftmost`. Maybe fuse some
                # operations.
                acb_poly_scalar_mul(tmp, curder + k, self.leftmost, self.prec)
                for j in range(acb_poly_length(curder + k)):
                    acb_mul_ui(_coeffs(curder + k) + j,
                               _coeffs(curder + k) + j,
                               low + j,
                               self.prec)
                acb_poly_add(curder + k, curder + k, tmp, self.prec)
                if k + 1 < self.log_prec:
                    acb_poly_add(curder + k, curder + k, curder + k + 1,
                                 self.prec)

        for k in range(self.log_prec):
            acb_poly_clear(curder + k)
        free(curder)

        acb_poly_clear(tmp)


    # Variant of ``apply_dop`` using a polynomial middle product based on
    # transposed interpolation at small integers.
    #
    # Approximate cost (partly heuristic) for high - mid = mid - low = d,
    # log_prec = 1, leftmost = 0:
    #
    # 2·r·d·M(p, h + d·log(d)) + 2·r·d²·p  (+ precomputation)
    #
    # ((r, d, h) = (order, degree, height) of dop, p = working prec).
    #
    # Compared to the basecase version, we have only ~4rd “big” multiplications
    # instead of ~d²/2, but with slightly larger operands, and some additional
    # overhead. The number of operations of cost O(p) is still quadratic in d,
    # again with an additional r factor and a larger constant.
    #
    # XXX Unfortunately, at the moment, this version seems to slow to be useful.
    #
    # XXX Also, this implementation is maybe more complicated than necessary:
    # - Using points at infinity makes the code more complex, for an unclear
    #   performance benefit.
    # - Since the low-degree third of the full deg-3N result is known in
    #   advance, we could maybe use a plain evaluation+interpolation scheme at
    #   ~2N points, without relying on transposed multiplication? [Thanks to
    #   Anne Vaugon for this remark!]
    cdef void apply_dop_interpolation(self, slong base, slong low, slong mid,
                                      slong high) noexcept:


        cdef slong i, j, k, n, p
        cdef acb_ptr dest

        cdef acb_poly_t tmp
        acb_poly_init(tmp)
        acb_poly_fit_length(tmp, high - low)

        cdef acb_t y
        acb_init(y)

        # To compute derivatives, we need a copy of a chunk of `series`
        cdef acb_poly_struct *curder
        curder = <acb_poly_struct *> malloc(self.log_prec
                                            *sizeof(acb_poly_struct))
        for k in range(self.log_prec):
            acb_poly_init(curder + k)
            acb_poly_shift_right(curder + k, self.series + k, low - base)
            # typically already satisfied (?)
            acb_poly_truncate(curder + k, mid - low)

        # We need at least high - low - 1 interpolation points. We round this
        # number to the next even integer to compute the transposed
        # interpolation matrix, and use its upper left block in the case of an
        # odd length.
        assert high - low < APPLY_DOP_INTERPOLATION_MAX_POINTS
        cdef slong halflen = (high - low)//2
        self.tinterp_cache_compute(halflen)
        cdef acb_mat_struct *tinterp_num = self.tinterp_cache_num + halflen
        cdef fmpz *tinterp_den = self.tinterp_cache_den + halflen

        # Each middle product decomposes as:
        # (i)   a transposed interpolation at high-low-1 points of a coefficient
        #       of dop, with the constant term omitted (precomputed, input
        #       length = output length = high-low-1);
        # (ii)  a direct evaluation at high-low-1 points of the reciprocal
        #       polynomial one of the entries of curder (input length mid-low,
        #       output length high-low-1);
        # (iii) pointwise multiplications between the results of (i) and (ii);
        # (iv)  a transposed evaluation in degree < high - mid of the output
        #       vector (input length high-low-1, output length high-mid).
        # Step (iv) is delayed until the contributions of all derivatives have
        # been accumulated.

        # XXX Not enough to avoid numerical issues. Why?
        cdef slong prec = self.prec + 2*(high-low)*FLINT_BIT_COUNT(high-low)

        cdef acb_ptr curderval = _acb_vec_init(2*halflen)
        cdef acb_mat_t prodval
        acb_mat_init(prodval, 2*halflen, self.log_prec)

        for i in range(self.dop_order + 1):

            for k in range(self.log_prec):

                # (ii) Direct evaluation
                #
                # XXX Often the dominant step in practice... (Try fast
                # multipoint evaluation at integers???)
                eval_reverse_stdpts(curderval, halflen, curder + k, mid - low,
                                    prec)

                # (iii) Pointwise multiplications, with accumulation in the
                # transformed domain
                for p in range(high - low - 1):
                    acb_addmul(acb_mat_entry(prodval, p, k),
                               acb_mat_entry(tinterp_num, p, i),
                               curderval + p,
                               prec)

                # curder[k] ← (d/dx)(previous curder)[k]

                # XXX Optimize case of rational `leftmost`. Maybe fuse some
                # operations.
                acb_poly_scalar_mul(tmp, curder + k, self.leftmost, prec)
                for j in range(acb_poly_length(curder + k)):
                    acb_mul_ui(_coeffs(curder + k) + j,
                               _coeffs(curder + k) + j,
                               low + j,
                               prec)
                acb_poly_add(curder + k, curder + k, tmp, prec)
                if k + 1 < self.log_prec:
                    acb_poly_add(curder + k, curder + k, curder + k + 1,
                                 prec)

        for k in range(self.log_prec):
            if acb_poly_length(self.series + k) < high - base:
                acb_poly_fit_length(self.series + k, high - base)
                _acb_poly_set_length(self.series + k, high - base)

        # (iv) Transposed Horner evaluation, adding to the values already
        # present in the high part of self.series.

        # (iv.a) Finite points. When high-low == 2*halfen, so that 2*halflen-2 <
        # high-low-1, the last row of the transposed interpolation matrix is not
        # used.
        for n in range(mid, high):
            for k in range(self.log_prec):
                # Would it be faster to compute the sum using acb_dot_ui with a
                # vector of ones?
                acb_zero(y)
                for p in range(2*halflen - 1):
                    acb_add(y, y,
                            acb_mat_entry(prodval, p, k),
                            prec)
                acb_div_fmpz(y, y, tinterp_den, prec)
                dest = _coeffs(self.series + k) + n - base
                acb_add(dest, dest, y, prec)
            if n == high - 1:
                break
            # Using acb_dot_ui with a vector of powers above should be faster
            # than doing a second pass here when the powers fit on a single
            # limb.
            for p in range(2*halflen - 1):
                for k in range(self.log_prec):
                    # there is no _acb_vec_scalar_mul_si
                    acb_mul_si(acb_mat_entry(prodval, p, k),
                               acb_mat_entry(prodval, p, k),
                               (p + 1)//2 if p % 2 == 1 else -p//2,
                               prec)
        # (iv.b) Infinity (last column of the transposed evaluation matrix). The
        # point at infinity only contributes to the coefficient of
        # x^{low+2*halfen-1}. In other words, it contributes to the coefficient
        # of x^{high-1} when high-low is odd, and not at all when it is even.
        if (high - low) % 2 == 1:
            for k in range(self.log_prec):
                acb_div_fmpz(y, acb_mat_entry(prodval, 2*halflen - 1, k),
                             tinterp_den, prec)
                dest = _coeffs(self.series + k) + high - 1 - base
                acb_add(dest, dest, y, prec)

        acb_mat_clear(prodval)
        _acb_vec_clear(curderval, 2*halflen)
        for k in range(self.log_prec):
            acb_poly_clear(curder + k)
        free(curder)
        acb_clear(y)
        acb_poly_clear(tmp)


    cdef void tinterp_cache_init(self, slong size):
        assert self.tinterp_cache_num == NULL
        self.tinterp_cache_size = size
        self.tinterp_cache_num = <acb_mat_struct *> calloc(size,
                                                         sizeof(acb_mat_struct))
        self.tinterp_cache_den = <fmpz *> malloc(size*sizeof(fmpz))


    cdef void tinterp_cache_clear(self):
        cdef slong i
        for i in range(self.tinterp_cache_size):
            if (<acb_ptr *> (self.tinterp_cache_num + i))[0] != NULL:
                acb_mat_clear(self.tinterp_cache_num + i)
                fmpz_clear(self.tinterp_cache_den + i)
        free(self.tinterp_cache_num)
        free(self.tinterp_cache_den)


    cdef void tinterp_cache_compute(self, slong n):
        r"""
        Precompute the images by the transposed interpolation operator at the 2n
        points ``0, 1, -1, ..., n-1, 1-n, ∞`` of the polynomials
        ``a₁ + a_2·x + ··· + a_{2n}·x^{2n-1}`` where ``a`` is one of the
        coefficients of ``dop``. Each value is represented as the quotient of a
        complex numerator stored in ``tinterp_cache_num`` by a common
        denominator stored in ``tinterp_cache_den``. In ``tinterp_cache_num``,
        rows correspond to evaluation points and columns correspond to
        coefficients of ``dop``.
        """
        cdef slong i, j, k, s
        cdef acb_poly_struct *p
        cdef fmpz *b
        cdef acb_ptr c
        cdef fmpz_t intpow

        assert n < self.tinterp_cache_size
        cdef acb_mat_struct *num = self.tinterp_cache_num + n
        cdef fmpz *den = self.tinterp_cache_den + n
        if (<acb_ptr *> num)[0] != NULL:  # initialized?
            return

        # TBI, see apply_dop_interpolation
        cdef slong prec = self.prec + 2*n*FLINT_BIT_COUNT(n)

        # cleared by tinterp_cache_clear
        # possible optimization: fmpz or fmpzi version for operators with exact
        # coefficients
        acb_mat_init(num, 2*n, self.dop_order + 1)
        fmpz_init(den)

        cdef fmpz_mat_t pitvdm_num
        fmpz_mat_init(pitvdm_num, n, 2*n-1)
        pitvdm_stdpts(pitvdm_num, den, n)
        for i in range(n):
            for s in range(2):
                if i == s == 0:
                    continue
                for j in range(self.dop_order + 1):
                    p = self.dop_coeffs + j
                    acb_dot_fmpz(acb_mat_entry(num, 2*i - 1 + s, j),
                                 NULL, False,
                                 _coeffs(p) + 1, 1,  # ignore cst coeff
                                 fmpz_mat_entry(pitvdm_num, i, 0), 1,
                                 min(2*n - 1, acb_poly_length(p) - 1),
                                 prec)
                for k in range(1, 2*n - 1, 2):
                    b = fmpz_mat_entry(pitvdm_num, i, k)
                    fmpz_neg(b, b)
        fmpz_mat_clear(pitvdm_num)

        # the row associated to the point at infinity
        for j in range(self.dop_order + 1):
            dest = acb_mat_entry(num, 2*n - 1, j)
            c = acb_poly_get_coeff_ptr(self.dop_coeffs + j, 2*n)
            if c != NULL:
                acb_mul_fmpz(dest, c, den, prec)
        fmpz_init(intpow)
        for i in range(n):
            fmpz_ui_pow_ui(intpow, i, 2*n - 1)
            for j in range(self.dop_order + 1):
                dest = acb_mat_entry(num, 2*n - 1, j)
                if i > 0:
                    acb_submul_fmpz(dest, acb_mat_entry(num, 2*i - 1, j),
                                    intpow, prec)
                acb_addmul_fmpz(dest, acb_mat_entry(num, 2*i, j), intpow,
                                prec)
        fmpz_clear(intpow)


    ## Error control and BoundCallbacks interface


    cdef bint check_convergence(self, object stop, slong n,
                                slong blksz,
                                arb_t est,         # W
                                arb_t tail_bound,  # RW
                                arb_srcptr radpow, slong next_stride):
        r"""
        Requires:
        last terms in low part (up to blksz) of self.series,
            XXX not really necessary, since this is use only to compute an
            estimate, and the residual is expected to have more or less the same
            value?
        residual (== rhs) in high part
        """
        # XXX The estimates computed here are sometimes more pessimistic than
        # the actual tail bounds. This can happen with naive_sum too, but seems
        # less frequent. While the two implementations are not using exactly the
        # same formulas for the estimates, I don't see why the results would
        # differ by more than a small multiplicative factor.

        cdef slong i, k
        cdef acb_ptr c
        cdef arb_t tmp
        arb_init(tmp)

        if n <= self.last_ini_n:
            arb_pos_inf(tail_bound)
            return False

        arb_zero(est)

        # Note that here radpow contains the contribution of z^λ.
        for k in range(self.log_prec):
            for i in range(blksz):
                # TODO Use a low-prec estimate instead (but keep reporting
                # accuracy information)
                c = _coeffs(self.series + k) + i
                arb_abs(tmp, acb_realref(c))
                arb_add(est, est, tmp, self.prec)
                arb_abs(tmp, acb_imagref(c))
                arb_add(est, est, tmp, self.prec)
        arb_mul_arf(est, est, arb_midref(radpow), self.prec)

        cdef RealBall _est = RealBall.__new__(RealBall)
        _est._parent = self.Reals
        arb_swap(_est.value, est)
        cdef RealBall _tb = RealBall.__new__(RealBall)
        _tb._parent = self.Reals
        arb_swap(_tb.value, tail_bound)
        self._rhs_offset = blksz  # only used by __check_residuals
        done, new_tail_bound = stop.check(self, n, _tb, _est, next_stride)
        arb_swap(tail_bound, (<RealBall?> new_tail_bound).value)
        arb_swap(est, _est.value)

        arb_clear(tmp)
        return done


    def get_residuals(self, stop, n):
        cdef Polynomial_complex_arb pol
        cdef slong d, k
        cdef acb_poly_t _ind
        acb_poly_init(_ind)
        cdef acb_t inv
        acb_init(inv)
        cdef acb_t tmp
        acb_init(tmp)

        cdef acb_struct *nres_term  # XXX share scratch space with next_term?
        nres_term = <acb_struct *> malloc(self.log_prec*sizeof(acb_struct))
        nres = [None]*self.log_prec
        for k in range(self.log_prec):
            acb_init(nres_term + k)  # coeff of x^d*log^k for varying d
            pol = Polynomial_complex_arb.__new__(Polynomial_complex_arb)
            pol._parent = self.Pol_IC
            acb_poly_fit_length(pol._poly, self.dop_degree)
            _acb_poly_set_length(pol._poly, self.dop_degree)
            nres[k] = pol
        for d in range(self.dop_degree):
            # This is very similar to the computation of an additional term
            # and should probably share code and/or intermediate results. At the
            # very least the evaluation of the indicial polynomial can be
            # shared.
            self.eval_ind(_ind, n + d, self.log_prec)  # not monic!
            acb_inv(inv, acb_poly_get_coeff_ptr(_ind, 0), self.bounds_prec)
            for k in reversed(range(self.log_prec)):
                # cst*self._residual[k][d]
                # (cst operand constant, could save a constant factor)
                acb_mul(nres_term + k,
                        _coeffs(self.dop_coeffs + self.dop_order),  # cst
                        _coeffs(self.series + k) + self._rhs_offset + d,
                        self.bounds_prec)
                # ... - sum(ind[u]*nres[k+u][d], 1 <= u < log_prec - k)
                acb_dot(nres_term + k,
                        nres_term + k,  # initial value
                        True,           # subtract
                        acb_poly_get_coeff_ptr(_ind, 1), 1,
                        nres_term + k + 1, 1,
                        self.log_prec - k - 1,
                        self.bounds_prec)
                # inv*(...)
                acb_mul(nres_term + k, nres_term + k, inv, self.bounds_prec)
            for k in range(self.log_prec):
                acb_swap(_coeffs((<Polynomial_complex_arb> nres[k])._poly) + d,
                         nres_term + k)
        for k in range(self.log_prec):
            _acb_poly_normalise((<Polynomial_complex_arb> nres[k])._poly)
            acb_clear(nres_term + k)
        free(nres_term)
        acb_clear(inv)
        acb_poly_clear(_ind)

        # self.__check_residuals(stop, n, nres)

        return [nres]


    def __check_residuals(self, stop, n, nres):
        r"""
        Debugging utility.

        Recompute the residual using the reference code in bounds.py.
        """
        cdef slong i, k
        cdef ComplexBall b
        last = []
        for i in range(self.dop_degree):
            cc = []
            for k in range(self.log_prec):
                b = <ComplexBall> ComplexBall.__new__(ComplexBall)
                b._parent = self.IC.zero().parent()
                acb_set(b.value, acb_poly_get_coeff_ptr(self.series + k,
                                                        self._rhs_offset-1-i))
                cc.append(b)
            last.append(cc)
        ref = stop.maj.normalized_residual(n, last)
        if not all(c.contains_zero() for p, q in zip(nres, ref) for c in p - q):
            print(f"{n=} {nres=}\n{n=} {ref=}")
            assert False


    def get_bound(self, stop, n, resid):
        if n <= self.last_ini_n:
            raise NotImplementedError
        maj = stop.maj.tail_majorant(n, resid)
        tb = maj.bound(self.py_evpts.rad, rows=self.py_evpts.jet_order)
        # XXX take log factors etc. into account (as in naive_sum)?
        return tb


## Subroutines for (transposed) evaluation/interpolation at small integers


cdef void pitvdm_stdpts(fmpz_mat_t matnum, fmpz_t matden, slong n) noexcept:
    r"""
    Partial inverse transpose Vandermonde matrix at integer nodes.

    Compute the rows associated to the nodes 0, 1, ..., n-1 of the inverse
    transpose Vandermonde matrix at the nodes 0, -1, 1, -2, 2, ..., 1-n, n-1.
    (The entry in column 0 ≤ j < 2n-1 of the row associated to -i is equal to
    (-1)^j times the corresponding entry in the row associated to i.)
    """
    cdef slong i, j
    assert n >= 1
    assert fmpz_mat_nrows(matnum) == n
    assert fmpz_mat_ncols(matnum) == 2*n-1
    cdef fmpz_t tmp
    fmpz_init(tmp)
    cdef fmpz *points = _fmpz_vec_init(2*n-1)
    cdef fmpz *values = _fmpz_vec_init(2*n-1)
    cdef fmpz *rowden = _fmpz_vec_init(n)
    fmpz_zero(points)
    for i in range(1, n):
        fmpz_set_si(points + 2*i - 1,  i)
        fmpz_set_si(points + 2*i,     -i)
    # FIXME This repeats essentially the same computation n times.
    # (And we could also share some work between different values of n by
    # working in the Newton basis...)
    for i in range(n):
        j = 0 if i == 0 else 2*i - 1
        fmpz_one(values + j)
        _fmpq_poly_interpolate_fmpz_vec(fmpz_mat_entry(matnum, i, 0),
                                        rowden + i, points, values, 2*n-1)
        fmpz_zero(values + j)
    _fmpz_vec_lcm(matden, rowden, n)
    for i in range(n):
        fmpz_divexact(tmp, matden, rowden + i)
        _fmpz_vec_scalar_mul_fmpz(fmpz_mat_entry(matnum, i, 0),
                                  fmpz_mat_entry(matnum, i, 0),
                                  2*n-1, tmp)
    _fmpz_vec_clear(rowden, n)
    _fmpz_vec_clear(values, 2*n-1)
    _fmpz_vec_clear(points, 2*n-1)
    fmpz_clear(tmp)


cdef void eval_reverse_stdpts(acb_ptr val, slong n, const acb_poly_struct *pol,
                              slong length, slong prec):
    r"""
    Evaluate at 0, 1, -1, 2, -2, ..., n-1, 1-n, ∞ the reciprocal polynomial of
    ``pol`` viewed as a polynomial of length ``length``.
    """
    cdef slong i, j, deg
    cdef acb_ptr c
    cdef acb_t even, odd
    acb_init(even)
    acb_init(odd)

    acb_poly_get_coeff_acb(val + 0, pol, length - 1)
    deg = acb_poly_degree(pol)
    for i in range(1, n):
        acb_zero(even)
        for j in range(1 - (length % 2), length, 2):
            acb_mul_si(even, even, i*i, prec)
            if j <= deg:
                acb_add(even, even, acb_poly_get_coeff_ptr(pol, j), prec)
        acb_zero(odd)
        for j in range(length % 2, length, 2):
            acb_mul_si(odd, odd, i*i, prec)
            if j <= deg:
                acb_add(odd, odd, acb_poly_get_coeff_ptr(pol, j), prec)
        acb_mul_si(odd, odd, i, prec)
        acb_add(val + 2*i - 1, even, odd, prec)
        acb_sub(val + 2*i, even, odd, prec)
    acb_poly_get_coeff_acb(val + 2*n - 1, pol, 0)

    acb_clear(even)
    acb_clear(odd)


## Debugging utilities


cdef Polynomial_complex_arb _make_constant_poly(acb_srcptr c, Parent parent):
    cdef Polynomial_complex_arb pol = Polynomial_complex_arb.__new__(Polynomial_complex_arb)
    pol._parent = parent
    acb_poly_set_acb(pol._poly, c)
    return pol

cdef Polynomial_complex_arb _make_poly(acb_poly_struct *p, Parent parent):
    cdef Polynomial_complex_arb pol = Polynomial_complex_arb.__new__(Polynomial_complex_arb)
    pol._parent = parent
    acb_poly_set(pol._poly, p)
    return pol

cdef _breakpoint():
    pass

cdef acb_ptr _coeffs(acb_poly_t pol) noexcept:
    # pol->coeffs is not accessible from sage, and acb_poly_get_coeff_ptr
    # returns null when the argument is out of bounds
    return (<acb_ptr *> pol)[0]
