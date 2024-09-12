# cython: language_level=3
# vim: tw=80

from libc.stdlib cimport malloc, free, calloc

from sage.libs.flint.flint cimport flint_printf
from sage.libs.flint.types cimport *
from sage.libs.flint.acb cimport *
from sage.libs.flint.acb_poly cimport *
from sage.libs.flint.arb cimport *
from sage.libs.flint.fmpz cimport *
from sage.libs.flint.fmpz_mat cimport *
from sage.libs.flint.fmpz_poly cimport *
from sage.libs.flint.fmpz_vec cimport *
from sage.libs.flint.gr cimport *
from sage.libs.flint.mag cimport *

from sage.libs.flint.gr_mat cimport gr_mat_pascal

cdef extern from "flint_wrap.h":
    void gr_ctx_init_fmpz(gr_ctx_t ctx) noexcept
    void GR_MUST_SUCCEED(int status)

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

    cdef int dop_order
    cdef int numpts
    cdef slong dop_degree

    cdef bint dop_is_exact
    cdef acb_poly_struct *dop_coeffs
    cdef fmpz_poly_struct *dop_coeffs_fmpz  # TODO: coeffs in ℤ[i]
    cdef acb_poly_t ind
    cdef acb_t leftmost
    cdef arb_t rad

    # Unlike the Python version, which creates/destroys/returns lists of
    # polynomials, this version acts on shared data buffers.
    #
    # The coefficient of a given log(x)^k/k! is represented as a contiguous
    # array of coefficients (= FLINT polynomial), and subroutines that operate
    # on slices of coefficients take as input offsets in this array (as opposed
    # to direct pointers), with the same offset typically applying for all k.

    # vector of polynomials in x holding the coefficients wrt log(x)^k/k! of the
    # last few (???) terms of the series solution
    cdef acb_poly_struct *series

    # vector of polynomials in δ (perturbation of ζ = evaluation point) holding
    # the jets of coefficients wrt log(ζ)^k/k! of the partial sums
    cdef acb_poly_struct *sums

    cdef slong prec
    cdef slong bounds_prec

    cdef int max_log_prec
    # max length wrt log of any of the coefficients (of the solution, the rhs,
    # or the sum) computed to date
    cdef int log_prec

    cdef int jet_order

    cdef acb_ptr evpts
    cdef acb_ptr pows
    cdef fmpz *binom_n  # binom(n, j) for j < jet_order

    cdef dict _ini
    cdef slong _last_ini

    cdef readonly dict critical_coeffs
    cdef int _rhs_offset

    cdef fmpz_mat_t binom

    ## used by remaining python code

    cdef readonly object dop_T, ini, py_evpts, IR, IC, real, _leftmost

    ## may or may not stay

    cdef object Ring, Pol, _Reals, Pol_IC
    cdef readonly object Jets


    def __cinit__(self, dop_T, ini, py_evpts, *args, **kwds):
        # XXX maybe optimize data layout
        cdef int i

        cdef size_t sz_poly = sizeof(acb_poly_struct)
        self.dop_order = dop_T.order()
        self.dop_degree = dop_T.degree()
        self.numpts = len(py_evpts)
        self.jet_order = py_evpts.jet_order

        self.dop_coeffs = <acb_poly_struct *> malloc((self.dop_order + 1)*sz_poly)
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

        self.sums = <acb_poly_struct *> malloc(self.numpts*self.max_log_prec*sz_poly)
        for i in range(self.numpts*self.max_log_prec):
            acb_poly_init(self.sums + i)
            acb_poly_fit_length(self.sums + i, self.jet_order)
            _acb_poly_set_length(self.sums + i, self.jet_order)

        acb_poly_init(self.ind)
        acb_init(self.leftmost)
        arb_init(self.rad)

        self.init_binom(self.dop_order + 1)


    def __dealloc__(self):
        cdef int i

        self.clear_binom()

        arb_clear(self.rad)
        acb_poly_clear(self.ind)
        acb_clear(self.leftmost)

        for i in range(self.numpts*self.max_log_prec):
            acb_poly_clear(self.sums + i)
        free(self.sums)

        _acb_vec_clear(self.evpts, self.numpts)
        _acb_vec_clear(self.pows, self.numpts)
        _fmpz_vec_clear(self.binom_n, self.jet_order)

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

        ### Kept for gradual Python to Cython conversion; would change or
        ### disappear in a pure Cython version

        self.dop_T = dop_T
        self.ini = ini  # LogSeriesInitialValues
        self.py_evpts = py_evpts

        # Parents

        self.Ring = Ring
        self._Reals = Ring.base()
        # Slices of series solns (w/o logs); must contain λ.
        self.Pol = dop_T.base_ring().change_ring(Ring)
        # Values (of the evaluation point, of partial sums). Currently limited
        # to using a ball field as a base ring, but this may change.
        self.Jets = None  # see below
        # Error bounds
        self.IR = ctx.IR
        self.IC = self.IR.complex_field()
        self.Pol_IC = self.Pol.change_ring(self.IC)

        ### Internal data used by the Cython part

        # maybe change this to a plain c data structure
        self._ini = {k: tuple(Ring(a) for a in v)
                     for k, v in ini.shift.items()}
        self._last_ini = self.ini.last_index()

        self.dop_is_exact = True
        for i, pol in enumerate(dop_T):
            p = self.dop_coeffs + i
            acb_poly_swap(p, (<Polynomial_complex_arb?> (self.Pol(pol)))._poly)
            for j in range(acb_poly_length(p)):
                self.dop_is_exact = (self.dop_is_exact
                                     and acb_is_exact(_coeffs(p) + j))
            self.dop_is_exact = (self.dop_is_exact
                                 and acb_poly_get_unique_fmpz_poly(
                                                   self.dop_coeffs_fmpz + i, p))

        self._leftmost = leftmost = Ring(ini.expo)
        acb_set(self.leftmost, (<ComplexBall?> leftmost).value)

        arb_set(self.rad, (<RealBall?> py_evpts.rad).value)

        acb_poly_fit_length(self.ind, self.dop_order + 1)
        _acb_poly_set_length(self.ind, self.dop_order + 1)
        for i in range(self.dop_order + 1):  # (unnecessary copies)
            acb_poly_get_coeff_acb(_coeffs(self.ind) + i, self.dop_coeffs + i, 0)
        _acb_poly_normalise(self.ind)

        self.prec = Ring.precision()
        self.bounds_prec = self.IR.precision()

        Jets, jets = py_evpts.jets(Ring)
        for i in range(self.numpts):
            assert jets[i].degree() == 0 or jets[i].degree() == 1 and jets[i][1].is_one()
            acb_poly_get_coeff_acb(self.evpts + i,
                                   (<Polynomial_complex_arb?> jets[i])._poly,
                                   0)

        ### Auxiliary output

        self.critical_coeffs = {}

        # Precomputed data, also available as auxiliary output

        self.Jets = Jets

        # At the moment Ring must in practice be a complex ball field (other
        # rings do not support all required operations); this flag signals that
        # the series (incl. singular part) and evaluation points are real.
        self.real = (py_evpts.is_real_or_symbolic
                     and ini.is_real(dop_T.base_ring().base_ring()))


    cdef void init_binom(self, slong s) noexcept:
        cdef gr_ctx_t fmpz
        gr_ctx_init_fmpz(fmpz)
        fmpz_mat_init(self.binom, s, s)
        GR_MUST_SUCCEED(gr_mat_pascal(<gr_mat_struct *> self.binom, -1, fmpz))
        gr_ctx_clear(fmpz)


    cdef void clear_binom(self) noexcept:
        fmpz_mat_clear(self.binom)


    def sum_blockwise(self, stop):
        cdef slong i, j, k
        cdef acb_ptr c
        cdef arb_t est, tb
        cdef arb_t radpow, radpow_blk
        cdef mag_t coeff_rad

        # Block size must be >= deg.
        cdef slong blksz = max(1, self.dop_degree)
        for k in range(self.max_log_prec):
            acb_poly_fit_length(self.series + k, 2*blksz)
        cdef slong blkstride = max(2, 32//blksz)

        arb_init(radpow)
        arb_pow(radpow, self.rad, acb_imagref(self.leftmost), self.bounds_prec)
        arb_init(radpow_blk)
        arb_pow_ui(radpow_blk, self.rad, blksz, self.bounds_prec)

        arb_init(est)
        arb_pos_inf(est)
        arb_init(tb)
        arb_pos_inf(tb)
        mag_init(coeff_rad)

        for i in range(self.numpts):
            acb_one(self.pows + i)

        cdef bint done = False
        cdef slong b = 0
        while True:
            self.sum_dac(b*blksz, b*blksz, (b+1)*blksz)
            self.apply_dop(b*blksz, b*blksz, (b+1)*blksz, (b+2)*blksz)

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

        arb_clear(tb)
        arb_clear(est)
        arb_clear(radpow_blk)
        arb_clear(radpow)
        mag_clear(coeff_rad)

        return psums


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


    cdef void sum_dac(self, slong base, slong low, slong high) noexcept:
        r"""
        Compute the chunk ``y[λ+low:λ+high]`` of the solution of ``L(y) = rhs``
        for a given rhs itself of support contained in ``λ+low:λ+high``.
        Works in place on ``self.series[:][low-base:high-base]``.
        """

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


    cdef void next_term(self, slong base, slong n) noexcept:
        r"""
        Write to ``self.series[:][n-base]`` the coefficient of ``x^(λ+n)`` in
        the solution. Also store the corresponding critical coefficients in the
        Python dict ``self.critical_coeffs``.
        """

        assert n >= base

        cdef slong k
        cdef ComplexBall tmp_ball

        cdef tuple ini = self._ini.get(n, ())
        cdef int mult = len(ini)

        # Probably don't need the max(len(rhs), ...) here since we initialize
        # series once and for all as a table of length max_log_prec.
        cdef int rhs_len = self.log_prec
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


    cdef acb_poly_struct *sum_ptr(self, int j, int k) noexcept:
        # self.sums + j*self.max_log_prec + k is the jet of order self.jet_order
        # of the coeff of log^k/k! in the sum at the point of index j
        return self.sums + j*self.max_log_prec + k


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



    cdef void apply_dop(self, slong base, slong low, slong mid, slong high) noexcept:
        r"""
        *Add* to ``self.series[:][mid-base:high-base]`` the coefficients of
        ``self.dop(y[λ+low:λ+mid])``, where the input is given in
        ``self.series[:][low-base:mid-base]``.
        """

        assert base <= low <= mid <= high
        if self.dop_is_exact and acb_is_zero(self.leftmost):
            self.apply_dop_basecase_exact(base, low, mid, high)
        else:
            self.apply_dop_basecase(base, low, mid, high)


    cdef void apply_dop_basecase(self, slong base, slong low, slong mid, slong high) noexcept:
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
        # d²/2·(M(h,p) + O(p)) + O(r·d²/2·h)
        #
        # ((r, d, h) = (order, degree, height) of dop, p = working prec).
        # With the pure acb version, the O(r·d²/2·h) term can easily dominate in
        # practice, even for small r...

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


    # Good asymptotic complexity wrt diffop degree, but slow in practice on
    # typical inputs. Might behave better than the basecase version for large
    # degree + large integer/ball coeffs.
    cdef void apply_dop_polmul(self, slong base, slong low, slong mid, slong high) noexcept:

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


    # Error control and BoundCallbacks interface


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

        if n <= self._last_ini:
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
        _est._parent = self._Reals
        arb_swap(_est.value, est)
        cdef RealBall _tb = RealBall.__new__(RealBall)
        _tb._parent = self._Reals
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
        assert self.dop_T == stop.maj.dop
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
        if n <= self.ini.last_index():
            raise NotImplementedError
        maj = stop.maj.tail_majorant(n, resid)
        tb = maj.bound(self.py_evpts.rad, rows=self.py_evpts.jet_order)
        # XXX take log factors etc. into account (as in naive_sum)?
        return tb


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

cdef acb_ptr _coeffs(acb_poly_t pol) noexcept:  # pol->coeffs is not accessible from sage
    return (<acb_ptr *> pol)[0]
