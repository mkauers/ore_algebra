# cython: language_level=3
# vim: tw=80
r"""
Divide-and-conquer summation of convergent D-finite series
"""

from libc.stdlib cimport malloc, free, calloc

from sage.libs.flint.flint cimport flint_printf
from sage.libs.flint.types cimport *
from sage.libs.flint.acb cimport *
from sage.libs.flint.acb_mat cimport *
from sage.libs.flint.acb_poly cimport *
from sage.libs.flint.arb cimport *
from sage.libs.flint.arf cimport arf_is_nan, arf_set_mag
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
    cdef slong WORD_MAX

from sage.rings.complex_arb cimport ComplexBall
from sage.rings.polynomial.polynomial_complex_arb cimport Polynomial_complex_arb
from sage.rings.real_arb cimport RealBall
from sage.structure.parent cimport Parent


import cython
import logging

from itertools import count

from . import accuracy

from sage.rings.complex_arb import ComplexBallField
from sage.rings.real_arb import RealBallField

from .context import dctx


logger = logging.getLogger(__name__)


cpdef enum ApplyDopAlgorithm:
    APPLY_DOP_AUTO
    APPLY_DOP_POLMUL
    APPLY_DOP_BASECASE_GENERIC
    APPLY_DOP_BASECASE_EXACT
    APPLY_DOP_INTERPOLATION

cdef slong APPLY_DOP_INTERPOLATION_MAX_POINTS = 256


# MAYBE-TODO:
#   - Decouple the code for computing (i) coefficients, (ii) partial sums,
# (iii) sums with error bounds. In particular, split Solution into a class for
# coefficients and one for partial sums. Maybe also subclass DACUnroller.
# Use this when computing formal log-series solutions.
#   - Add support for inhomogeneous equations with polynomial rhs. (Always
# sum at least up to deg(rhs) when doing error control).


# State of a solution currently being computed by :class:`DACUnroller`.
#
# This is a struct rather than an object because working with typed arrays of
# Python objects is cumbersome, and (for now at least) all actual operations are
# implemented in DACUnroller.
cdef struct Solution:

    # (i,k) -> coeff of x^{λ+n[i]}·log(x)^k/k!, 0 ≤ k < log_alloc, where the
    # n[i] are the integer shifts such that λ+n[i] is a root of the indicial
    # polynomial, in increasing order. Entries with k < mult(λ+n) must be
    # specified as initial conditions, remaining entries with n < truncation
    # order will be filled during the computation of the solution.
    acb_mat_t critical_coeffs

    # max length wrt log of any of the coefficients of the solution, the rhs,
    # or the sum computed to date
    slong log_prec

    # The coefficient of a given log(x)^k/k! is represented as a contiguous
    # array of coefficients (= FLINT polynomial), and routines that operate on
    # slices of coefficients take as input offsets in this array (as opposed to
    # direct pointers), with the same offset typically applying for all k.

    # vector of polynomials in x holding the coefficients wrt log(x)^k/k! of the
    # terms of the solution and/or of the rhs (= image by dop of the truncated
    # solution ≈ residual)
    acb_poly_struct *series

    acb_poly_struct *full_series

    # vector of polynomials in δ (perturbation of ξ = evaluation point) holding
    # the jets of coefficients wrt log(ξ)^k/k! of the partial sums:
    # self.sums + j*self.log_alloc + k is the jet of order self.jet_order
    # of the coeff of log^k/k! in the sum at the point of index j
    acb_poly_struct *sums

    # true if the series (including the singular part) and evaluation points are
    # known to be real
    #
    # XXX Useful? At the moment this is used to decide if the tail bounds need
    # to be added to the imaginary parts of the sum, and queried by the
    # SolMapper to choose a parent for its output. The latter could be done
    # based on the imaginary part of the sum, at least when computing a sum.
    # However, we might also use a flag like this to optimize the computation of
    # the sum.
    bint real

    # used to keep track of the sizes of various data structures
    slong _dop_order
    slong _log_alloc
    slong _numpts


cdef void init_solution(Solution *sol, slong dop_order, slong numshifts, slong
                        log_alloc, slong numpts, slong jet_order,
                        bint full_series):
    cdef slong i

    sol.log_prec = 0
    sol.real = False

    sol._dop_order = dop_order
    sol._log_alloc = log_alloc
    sol._numpts = numpts

    acb_mat_init(sol.critical_coeffs, numshifts, log_alloc)

    # XXX maybe using a single acb_vec would be more convenient after all?
    # (in particular, the coefficients of x^n·log(x)^k for k = 0, 1, ...
    # would then be regularly spaced, and we don't use non-underscore
    # acb_poly functions much)
    sol.series = _acb_poly_vec_init(log_alloc)
    if full_series:
        sol.full_series = _acb_poly_vec_init(log_alloc)
    else:
        sol.full_series = NULL

    sol.sums = _acb_poly_vec_init(numpts*log_alloc)
    for i in range(numpts*log_alloc):
        acb_poly_fit_length(sol.sums + i, jet_order)


cdef void clear_solution(Solution *sol):

    _acb_poly_vec_clear(sol.sums, sol._numpts*sol._log_alloc)
    _acb_poly_vec_clear(sol.series, sol._log_alloc)
    if sol.full_series != NULL:
        _acb_poly_vec_clear(sol.full_series, sol._log_alloc)
    acb_mat_clear(sol.critical_coeffs)


cdef struct AccuracyEstimates:
    double neglogterm
    double cvg_rate
    slong acc
    double loss_rate
    double terms_wanted
    double terms_accessible
    slong prec_wanted


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef class DACUnroller:
    r"""
    Tool for computing partial sums of solutions of an operator that may have a
    regular singular point at the origin.

    An instance of ``DACUnroller`` holds one or more instances of ``Solution``,
    corresponding to one or more solutions of the same differential equation to
    be computed simultaneously. These solutions must have exponents in the same
    coset of ℂ/ℤ.

    Each partial sum is evaluated at zero or more points (the same for all
    solutions), along with its first ``jet_order`` derivatives. (This is mainly
    intended for points of about the same absolute value, so the truncation
    order is the same for all points.)

    The main methods work in place on ``self.sol[:].series``. The arguments
    ``low``, ``mid``, ``high``, and ``n`` are shifts wrt λ; the index into
    ``series`` corresponding to ``n`` is ``n - base``. In other words,
    ``self.sol[m].series[k][n-base]`` stores the coefficient of
    `x^{λ+n}·\log(x)^k/k!` in the solution of index m (both on input and on
    output, in the solution and in the rhs).
    """

    cdef bint debug
    cdef ApplyDopAlgorithm apply_dop_algorithm

    cdef slong dop_order
    cdef slong dop_degree
    cdef slong numsols
    cdef slong numpts
    cdef slong jet_order

    # working precisions in bits
    cdef slong prec         # main (full) precision (series coefficients, etc.)
    cdef slong sums_prec    # evaluation points, sums (shorter dep chains)
    cdef slong bounds_prec  # error bounds

    cdef acb_poly_struct *dop_coeffs
    cdef fmpz_poly_struct *dop_coeffs_fmpz  # TODO: coeffs in ℤ[i]
    cdef bint dop_is_exact

    cdef acb_poly_t ind     # indicial polynomial
    cdef acb_t leftmost     # aka λ, exponent of the _group_ of solutions
    # XXX not as convenient as I thought: use (s, mult) pairs instead?
    cdef slong *ini_shifts  # s s.t. ind(λ+s)=0, multiple roots repeated,
                            # (-1)-terminated
    cdef slong ini_idx      # current index in ini_shifts
    cdef slong shift_idx    # same but ignoring multiplicities (= distinct elts
                            # elts seen so far, = row index in critical_coeffs)

    cdef acb_ptr evpts  # evaluation points x[j], 0 < j < numpts
    cdef acb_ptr pows   # x[j]^{n-i} for i < jet_order (i-major, cyclic on i);
                        # only the block i=0 is really used for nonzero points
    cdef mag_t rad      # ≥ abs(evaluation points), for error estimate
                        # (rigorous bounds currently use the python version)

    # the solutions we are computing
    cdef Solution *sol

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

    cdef slong __rhs_offset

    # internal data -- remaining python code

    cdef object py_evpts, Reals, Pol_IC


    # for some reason (related to typed memory views?) making this a method of
    # Solution does not seem to work
    cdef acb_poly_struct *sum_ptr(self, slong m, slong j, slong k) noexcept:
        return self.sol[m].sums + j*self.sol[m]._log_alloc + k


    def __cinit__(self, dop_T, inis, py_evpts, *args,
                  bint keep_series=False, **kwds):
        cdef slong m

        self.debug = False

        self.dop_order = dop_T.order()
        self.dop_degree = dop_T.degree()
        self.numsols = len(inis)
        self.numpts = len(py_evpts)
        self.jet_order = py_evpts.jet_order

        self.sol = <Solution *> malloc(self.numsols*sizeof(Solution))
        for m, ini in enumerate(inis):
            # using dop_order as a crude bound for max possible log prec
            # (needs updating to support inhomogeneous equations)
            init_solution(self.sol + m, self.dop_order, len(ini.shift),
                          self.dop_order, self.numpts, self.jet_order,
                          keep_series)

        self.dop_coeffs = _acb_poly_vec_init(self.dop_order + 1)
        self.dop_coeffs_fmpz = _fmpz_poly_vec_init(self.dop_order + 1)

        self.evpts = _acb_vec_init(self.numpts)
        self.pows =  _acb_vec_init(self.numpts*self.jet_order)
        self.binom_n = _fmpz_vec_init(self.jet_order)
        self.ini_shifts = <slong *> malloc((self.dop_order + 1)*sizeof(slong))

        acb_poly_init(self.ind)
        acb_init(self.leftmost)
        mag_init(self.rad)

        self.init_binom(self.dop_order + 1)

        self.tinterp_cache_num = NULL
        self.tinterp_cache_den = NULL
        self.tinterp_cache_size = 0


    def __dealloc__(self):
        cdef slong m

        self.clear_binom()

        mag_clear(self.rad)
        acb_clear(self.leftmost)
        acb_poly_clear(self.ind)

        free(self.ini_shifts)

        # __dealloc__ may be called before the object is fully initialized if
        # a python exception is raised in __cinit__. If dop_coeffs has been
        # initialized, we can reasonably assume that all subsequent
        # initializations have been done, but sol might be initialized even if
        # dop_coeffs is not.
        if self.dop_coeffs != NULL:
            _fmpz_vec_clear(self.binom_n, self.jet_order)
            _acb_vec_clear(self.pows, self.numpts*self.jet_order)
            _acb_vec_clear(self.evpts, self.numpts)

            _acb_poly_vec_clear(self.dop_coeffs, self.dop_order + 1)
            _fmpz_poly_vec_clear(self.dop_coeffs_fmpz, self.dop_order + 1)

        if self.sol != NULL:
            for m in range(self.numsols):
                clear_solution(self.sol + m)
        free(self.sol)


    def __init__(self, dop_T, inis, py_evpts, sums_prec, prec , *,
                 ctx=dctx, **kwds):

        cdef slong i, j, k, m
        cdef acb_poly_struct *p
        cdef ComplexBall b
        cdef Solution sol

        assert dop_T.parent().is_T()
        assert len(inis) > 0

        self.prec = prec
        self.sums_prec = sums_prec
        self.bounds_prec = ctx.IR.precision()

        self.apply_dop_algorithm = ApplyDopAlgorithm[ctx.apply_dop]

        ## Parents

        Coeffs = ComplexBallField(prec)
        Sums = ComplexBallField(sums_prec)
        # Slices of series solns (w/o logs); must contain λ.
        Pol = dop_T.base_ring().change_ring(Coeffs)
        # Error bounds
        self.Pol_IC = Pol.change_ring(ctx.IR.complex_field())
        self.Reals = RealBallField(prec)  # XXX temporary

        ## Solutions
        # XXX initialize outside and pass to DACUnroller, or along with
        # DACUnroller to some other function?

        for m, ini in enumerate(inis):
            for j, (_, vec) in enumerate(sorted(ini.shift.items())):
                for k, a in enumerate(vec):
                    b = Coeffs(a)
                    acb_swap(acb_mat_entry(self.sol[m].critical_coeffs, j, k),
                             b.value)
            self.sol[m].real = (py_evpts.is_real_or_symbolic and
                                ini.is_real(dop_T.base_ring().base_ring()))

        ## Internal data

        self.dop_is_exact = True
        for i, pol in enumerate(dop_T):
            p = self.dop_coeffs + i
            # This truncates the coefficients to self.prec
            acb_poly_swap(p, (<Polynomial_complex_arb?> (Pol(pol)))._poly)
            for j in range(acb_poly_length(p)):
                self.dop_is_exact = (self.dop_is_exact
                                     and acb_is_exact(_coeffs(p) + j))
            self.dop_is_exact = (self.dop_is_exact
                                 and acb_poly_get_unique_fmpz_poly(
                                                   self.dop_coeffs_fmpz + i, p))

        # expo is a PolynomialRoot, consider doing part of the work
        # with its exact value instead of an interval
        leftmost = inis[0].expo
        ini_shifts = inis[0].flat_shifts()
        for ini in inis[1:]:
            if ini.expo != leftmost or ini.flat_shifts() != ini_shifts:
                raise ValueError("incompatible initial conditions")
        acb_swap(self.leftmost, (<ComplexBall?> (Coeffs(leftmost))).value)
        for i, s in enumerate(ini_shifts):
            self.ini_shifts[i] = ini_shifts[i]
        self.ini_shifts[len(ini_shifts)] = -1
        self.ini_idx = 0
        self.shift_idx = 0

        acb_poly_fit_length(self.ind, self.dop_order + 1)
        _acb_poly_set_length(self.ind, self.dop_order + 1)
        for i in range(self.dop_order + 1):  # (unnecessary copies)
            acb_poly_get_coeff_acb(_coeffs(self.ind) + i, self.dop_coeffs + i, 0)
        _acb_poly_normalise(self.ind)

        self.py_evpts = py_evpts
        for i in range(self.numpts):
            b = py_evpts.approx(Sums, i)
            acb_swap(self.evpts + i, b.value)
        arb_get_mag(self.rad, (<RealBall?> py_evpts.rad).value)


    cdef slong max_log_prec(self) noexcept:
        cdef slong m
        cdef slong res = 0
        # cython seems to ignore the type of m if using range here?!
        for m in range(self.numsols):
            if self.sol[m].log_prec > res:
                res = self.sol[m].log_prec
        return res


    ## Main summation loop


    cdef void reset_solutions(self, slong series_length) noexcept:
        cdef slong i, k, m
        cdef acb_poly_struct *f
        for i in range(self.numpts):  # only set to one the first block
            acb_one(self.pows + i)
        for m in range(self.numsols):
            for k in range(self.sol[m]._log_alloc):
                f = self.sol[m].series + k
                acb_poly_zero(f)
                acb_poly_fit_length(f, series_length)
                _acb_poly_set_length(f, series_length)  # for printing
                if self.sol[m].full_series != NULL:
                    acb_poly_zero(self.sol[m].full_series + k)
            for i in range(self.sol[m]._numpts*self.sol[m]._log_alloc):
                f = self.sol[m].sums + i
                acb_poly_zero(f)
                _acb_poly_set_length(f, self.jet_order)


    # Maybe get rid of this and use sum_dac only?
    cpdef void sum_blockwise(self, object stop, slong max_terms=WORD_MAX):
        cdef slong k, m, base, low, high, acc
        cdef acb_ptr f
        cdef mag_t radpow, radpow_blk, re_leftmost, sum_rad
        cdef arb_t _radpow, est, tb

        max_terms = max(max_terms, 0)

        # Block size must be >= deg. Power-of-two factors may be beneficial when
        # using apply_dop_interpolation.
        cdef slong blksz = max(1, self.dop_degree)
        # cdef slong blksz = 1 << (self.dop_degree - 1).bit_length()
        cdef slong blkstride = max(1, 32//blksz)

        # initialize radpow >= rad^Re(λ) (neglecting Im(λ)...)
        arb_init(_radpow)
        arf_set_mag(arb_midref(_radpow), self.rad)
        arb_pow(_radpow, _radpow, acb_realref(self.leftmost), self.bounds_prec)
        arb_get_mag(radpow, _radpow)
        arb_clear(_radpow)

        mag_init(radpow_blk)
        mag_pow_ui(radpow_blk, self.rad, blksz)

        arb_init(est)
        arb_pos_inf(est)
        arb_init(tb)
        arb_pos_inf(tb)
        mag_init(sum_rad)

        # Final wanted interval accuracy
        cdef double tgt_acc = (self.prec if stop is None
                               else -float(stop.eps.log(2)))
        cdef AccuracyEstimates cvest = self.initial_accuracy_estimates()

        self.apply_dop_interpolation_max_len = min(  # threshold TBI
            APPLY_DOP_INTERPOLATION_MAX_POINTS,
            self.prec,
            2*blksz)
        self.tinterp_cache_init(self.apply_dop_interpolation_max_len//2 + 1)

        self.reset_solutions(2*blksz)
        cdef bint done = False
        cdef slong b = 0
        while True:
            base = low = b*blksz
            high = min(low + blksz, max_terms)
            self.sum_dac(base, low, high)

            if high >= max_terms:
                if stop is None:
                    arb_zero(tb)
                    break
                else:
                    raise NotImplementedError(
                        "reached max_terms with a StoppingCriterion object")

            self.apply_dop(base, low, high, high + blksz)

            mag_mul(radpow, radpow, radpow_blk)

            # - Support stopping in the middle of a block when dop_degree is
            # large? Would need the ability to compute the high part of the
            # residual (to low precision).
            # - It would be simpler to perform the convergence check after
            # shifting sol[:].series so that it contains the residual, but doing
            # it before allows us to check the computation using the low-degree
            # part in debug mode.
            if (b + 1) % blkstride == 0:
                self.term_estimate(est, radpow, high - base)

                if stop is not None and self.check_convergence(
                        stop, high, high - base, est, tb, blkstride*blksz):
                    break

                # Gradually decrease the working precision when [convergence
                # and] accuracy estimates suggest the current value is wasteful.
                # But we still want a larger initial working precision to result
                # in a larger effective working precision, so we cannot just set
                # it right away to the current best guess. Note that when the
                # terms of the series get small enough, one can have self.prec <
                # self.sums_prec.
                # (TBI: Currently done only for multiples of blkstride to avoid
                # bad interactions with the rigorous convergence check.)
                # NOTE: Moving this before apply_dop (without changing the
                # formulas used to estimate the speed of convergence etc.)
                # results in code that seems to perform significantly worse.
                cvest = self.update_accuracy_estimates(&cvest, tgt_acc, est,
                                                          high, blksz)
                if cvest.prec_wanted < self.prec:
                    self.prec -= (self.prec - cvest.prec_wanted)//2
                    logger.debug("prec_wanted=%s, new prec=%s",
                                 cvest.prec_wanted, self.prec)

            for m in range(self.numsols):
                for k in range(self.sol[m].log_prec):
                    f = _coeffs(self.sol[m].series + k)
                    _acb_poly_shift_right(f, f, high+blksz-base, high-base)
                    _acb_vec_zero(f + blksz, blksz)

            b += 1

        self.fix_sums()
        self.add_error_get_rad(sum_rad, tb)
        self._report_stats((b+1)*blksz, est, tb, sum_rad)

        self.tinterp_cache_clear()
        arb_clear(tb)
        arb_clear(est)
        mag_clear(radpow_blk)
        mag_clear(radpow)
        mag_clear(re_leftmost)
        mag_clear(sum_rad)


    cdef void sum_dac(self, slong base, slong low, slong high) noexcept:
        r"""
        Compute the chunk ``y[λ+low:λ+high]`` of the solution of ``L(y) = rhs``
        for a given rhs itself of support contained in ``λ+low:λ+high``.
        Works in place on ``self.sol[:].series[:][low-base:high-base]``.
        """
        # XXX should it be L(y) = -rhs in the above docstring?
        cdef slong mid

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


    cdef AccuracyEstimates initial_accuracy_estimates(self):
        cdef AccuracyEstimates cvest
        cvest.neglogterm = 0.
        cvest.cvg_rate = 0.
        cvest.terms_wanted = WORD_MAX
        cvest.acc = self.prec
        # We do not want to underestimate the loss rate as this could lead us to
        # cut too much working precision, but if we overestimate it too much, we
        # will essentially just set prec_wanted to the current accuracy all the
        # time. So we start with a rough guess that should at least ensure that
        # a few exact operations in the initial iterations do not reduce the
        # working precision too much.
        cvest.loss_rate = self.dop_order*self.dop_degree
        cvest.terms_accessible = float('inf')
        cvest.prec_wanted = ARF_PREC_EXACT


    cdef AccuracyEstimates update_accuracy_estimates(
            self,
            AccuracyEstimates *old,
            double tgt_acc, arb_srcptr est,
            slong terms, slong stride):

        # TODO: use these estimates to decide when to give up, whether
        # truncation bounds should be refined, etc.

        cdef AccuracyEstimates new

        # Estimated “known correct bits” of the result, for now with no attempt
        # to distinguish between absolute and relative accuracy
        cdef mag_t _est
        arb_get_mag(_est, est)
        new.neglogterm = -mag_get_d_log2_approx(_est)

        # Running estimate of convergence rate (bit/term), exponential
        # discounting
        cdef double cvg_rate_stride = (new.neglogterm - old.neglogterm)/stride
        new.cvg_rate = old.cvg_rate/2 + cvg_rate_stride/2

        # Number of terms still needed for full accuracy at the current rate of
        # convergence
        new.terms_wanted = (tgt_acc - new.neglogterm)/max(0., new.cvg_rate)

        # Current interval accuracy
        new.acc = arb_rel_accuracy_bits(est)

        # Running estimate of interval growth (lost bits/term), exponential
        # discounting. Note that self.prec may have changed since old.acc was
        # computed: putting the min(self.prec) here rather than in the
        # definition of acc is expected to reduce the impact of working
        # precision adjustments on the loss estimates.
        cdef double loss_stride = min(old.acc, self.prec) - new.acc
        cdef double loss_rate_stride = max(0., loss_stride)/stride
        new.loss_rate = old.loss_rate/2 + loss_rate_stride/2

        # Number of terms we should be able to compute before interval growth
        # makes it impossible to meet the specified tolerance
        new.terms_accessible = (new.neglogterm + new.acc - tgt_acc)/new.loss_rate

        if (new.terms_accessible <= new.terms_wanted
                or new.loss_rate >= old.loss_rate*1.01):
            new.prec_wanted = ARF_PREC_EXACT
        elif new.terms_wanted > 0:
            new.prec_wanted = <slong> (tgt_acc - 0.9*new.neglogterm
                                       + new.terms_wanted*new.loss_rate)
        else:
            new.prec_wanted = <slong> tgt_acc

        # print(f"n={terms} {tgt_acc=} {new.acc=} {new.neglogterm=}\n"
        #       f"  {cvg_rate_stride=} {new.cvg_rate=} \n"
        #       f"  {loss_rate_stride=} {new.loss_rate=} \n"
        #       f"  {new.terms_wanted=}({terms + new.terms_wanted}) "
        #       f"{new.terms_accessible=} \n"
        #       f"  {new.prec_wanted=}")

        return new


    cdef void add_error_get_rad(self, mag_ptr sum_rad, arb_ptr err):
        cdef slong i, j, k, m
        mag_zero(sum_rad)
        for m in range(self.numsols):
            for j in range(self.numpts):  # psum in psums
                for k in range(self.sol[m].log_prec):  # jet in psum
                    for i in range(self.jet_order):
                        c = _coeffs(self.sum_ptr(m, j, k)) + i
                        arb_add_error(acb_realref(c), err)
                        if self.sol[m].real:
                            assert (arb_is_zero(acb_imagref(c))
                                    or arf_is_nan(arb_midref(acb_imagref(c))))
                        else:
                            arb_add_error(acb_imagref(c), err)
                        mag_max(sum_rad, sum_rad,
                                arb_radref(acb_realref(c)))
                        mag_max(sum_rad, sum_rad,
                                arb_radref(acb_imagref(c)))


    cdef void _report_stats(self, slong n, arb_t est, arb_t tb,
                            mag_t sum_rad):
        if not logger.isEnabledFor(logging.INFO):
            return
        cdef Parent IR = RealBallField(self.bounds_prec)
        cdef RealBall _est = RealBall.__new__(RealBall)
        _est._parent = IR
        arb_swap(_est.value, est)
        cdef RealBall _tb = RealBall.__new__(RealBall)
        _tb._parent = IR
        arb_swap(_tb.value, tb)
        cdef RealBall _sum_rad = RealBall.__new__(RealBall)
        _sum_rad._parent = IR
        arb_set_interval_mag(_sum_rad.value, sum_rad, sum_rad, MAG_BITS)
        logger.info("summed %d terms, tail bound = %s (est = %s), "
                    "max rad = %s, final prec=%s",
                    n, _tb, _est, _sum_rad, self.prec)
        arb_swap(tb, _tb.value)
        arb_swap(est, _est.value)


    ## Interface for retrieving the results from Python


    def py_sums(self, Parent parent):
        cdef slong j, k, m
        cdef Polynomial_complex_arb psum
        if not issubclass(parent.Element, Polynomial_complex_arb):
            raise ValueError("bad parent")
        psums = [[[None]*self.sol[m].log_prec for _ in range(self.numpts)]
                 for m in range(self.numsols)]
        for m in range(self.numsols):
            for j in range(self.numpts):  # psum in psums
                for k in range(self.sol[m].log_prec):  # jet in psum
                    psum = Polynomial_complex_arb.__new__(Polynomial_complex_arb)
                    psum._parent = parent
                    acb_poly_set(psum._poly, self.sum_ptr(m, j, k))
                    psums[m][j][k] = psum
        return psums


    def py_series(self, slong m, Parent parent):
        cdef slong k
        cdef Polynomial_complex_arb ser
        if m < 0 or m > self.numsols or self.sol[m].full_series == NULL:
            raise IndexError(m)
        if not issubclass(parent.Element, Polynomial_complex_arb):
            raise ValueError("bad parent")
        res = []
        for k in range(self.sol[m].log_prec):
            ser = Polynomial_complex_arb.__new__(Polynomial_complex_arb)
            ser._parent = parent
            acb_poly_set(ser._poly, self.sol[m].full_series + k)
            res.append(ser)
        return res


    def py_critical_coeffs(self, slong m, Parent parent):
        cdef slong k
        cdef list l
        cdef ComplexBall b
        crit = {}
        cdef slong si = 0
        cdef slong ii = 0
        cdef slong n = -2
        if not issubclass(parent.Element, ComplexBall):
            raise ValueError("bad parent")
        while True:
            while self.ini_shifts[ii] == n:
                ii += 1
            n = self.ini_shifts[ii]
            if n == -1:
                break
            l = []
            for k in range(self.sol[m].log_prec):
                b = <ComplexBall> ComplexBall.__new__(ComplexBall)
                acb_set(b.value,
                        acb_mat_entry(self.sol[m].critical_coeffs, si, k))
                b._parent = parent
                l.append(b)
            crit[n] = l
            si += 1
        return crit


    def real(self):
        cdef slong m
        return all(self.sol[m].real for m in range(self.numsols))


    ## Computation of individual terms


    cdef void next_term(self, slong base, slong n) noexcept:
        r"""
        Write to ``self.sol[:].series[:][n-base]`` the coefficients of
        ``x^(λ+n)`` in the solutions. Also store the corresponding critical
        coefficients in the Python dicts ``self.sol[:].self.critical_coeffs``.
        """

        assert n >= base

        cdef slong k, m, rhs_len
        cdef acb_poly_struct *series
        cdef ComplexBall tmp_ball

        cdef slong mult = 0
        while self.ini_shifts[self.ini_idx + mult] == n:
            mult += 1

        cdef slong max_rhs_len = self.max_length_of_coeff_of_x(base, n)

        # Evaluate the indicial polynomial at λ + n + δ
        # XXX cache for the benefit of get_residuals? use multipoint evaluation?
        cdef acb_poly_t ind_n
        acb_poly_init(ind_n)
        self.eval_ind(ind_n, n, max_rhs_len + mult)

        cdef fmpz_t lc
        fmpz_init(lc)
        cdef acb_t invlc
        acb_init(invlc)

        # Compute the high-order (wrt log) part of the new term from the
        # coefficient of x^{λ+n} on the rhs and the indicial polynomial.

        # Working on copies allows us to use acb_dot.
        cdef acb_struct *new_term
        new_term = _acb_vec_init(max_rhs_len)

        for m in range(self.numsols):

            series = self.sol[m].series
            rhs_len = self.sol[m].log_prec
            while rhs_len > 0 and acb_is_zero(_coeffs(series + rhs_len - 1)
                                              + n - base):
                rhs_len -= 1

            for k in range(rhs_len - 1, -1, -1):
                # combin = rhs[k][0] + sum(ind_n[mult+1+j]*new_term[k+1+j], j≥0)
                acb_dot(new_term + k,
                        _coeffs(series + k) + n - base,
                        False,
                        _coeffs(ind_n) + mult + 1, 1,
                        new_term + k + 1, 1,
                        rhs_len - k - 1,
                        self.prec)
                if acb_is_zero(new_term + k):
                    continue
                # Only compute lc or invlc if (and when first) needed
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

            # Write the new term to sol[:].series, store new critical coeffs

            assert max_rhs_len + mult <= self.sol[m]._log_alloc

            for k in range(mult):  # initial conditions
                acb_set(_coeffs(series + k) +  n - base,
                        acb_mat_entry(self.sol[m].critical_coeffs,
                                      self.shift_idx, k))
            for k in range(mult, rhs_len + mult):
                acb_swap(_coeffs(series + k) + n - base, new_term + k - mult)
                # store new critical coeffs
                if mult > 0:
                    acb_set(acb_mat_entry(self.sol[m].critical_coeffs,
                                          self.shift_idx, k),
                            _coeffs(series + k) + n - base)
            for k in range(rhs_len + mult, self.sol[m].log_prec):
                acb_zero(_coeffs(series + k) + n - base)

            # Update log-degree

            self.sol[m].log_prec = max(self.sol[m].log_prec, rhs_len + mult)

            # Copy the new term to sol[:].full_series if desired

            if self.sol[m].full_series != NULL:
                for k in range(self.sol[m].log_prec):
                    acb_poly_set_coeff_acb(self.sol[m].full_series + k, n,
                                           _coeffs(series + k) + n - base)

        self.ini_idx += mult
        if mult > 0:
            self.shift_idx += 1

        _acb_vec_clear(new_term, rhs_len)
        acb_clear(invlc)
        fmpz_clear(lc)
        acb_poly_clear(ind_n)


    cdef slong max_length_of_coeff_of_x(self, slong base, slong n) noexcept:
        cdef slong m
        cdef slong length = self.max_log_prec()
        while length > 0:
            for m in range(self.numsols):
                if length <= self.sol[m].log_prec:
                    if not acb_is_zero(_coeffs(self.sol[m].series + length - 1)
                                       + n - base):
                        break
            else:
                length -= 1
                continue
            break
        return length


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
        Add to each entry of `self.sol[:].sums` a term corresponding to the
        current `n`.

        WARNING: For `i > 0` and `ξ ≠ 0`, and only in this case, the term added
        to the i-th derivative is `c·ξ^n` instead of `c·ξ^{n-i}` where `c` is
        the coefficient of `x^{n-i}` in the `i`th derivative, so that repeated
        calls to this function with `n = 0, 1, \dots` compute `ξ^i·f^(i)(ξ)`.
        Call `fix_sums()` at the end to fix the result.
        """
        cdef slong i, j, k, m, pows_offset
        cdef acb_ptr pows
        if n == 0:
            _fmpz_vec_zero(self.binom_n, self.jet_order)
            fmpz_one(self.binom_n)
        else:
            for i in range(self.jet_order - 1, 0, -1):
                fmpz_add(self.binom_n+i, self.binom_n+i, self.binom_n+i-1)
        cdef acb_t tmp
        acb_init(tmp)
        for j in range(self.numpts):
            for m in range(self.numsols):
                for k in range(self.sol[m].log_prec):
                    # We can trade some full-width muls for muls by small
                    # integer by multiplying everyone by ξ^n), but this
                    # optimization is valid only when ξ does not contain 0.
                    if acb_contains_zero(self.evpts + j):
                        for i in range(self.jet_order):
                            pows_offset = ((n + self.jet_order - i)
                                           % self.jet_order)*self.numpts
                            acb_mul(tmp,
                                    _coeffs(self.sol[m].series + k) + n - base,
                                    self.pows + pows_offset + j,
                                self.sums_prec)
                            acb_addmul_fmpz(_coeffs(self.sum_ptr(m, j, k)) + i,
                                            tmp, self.binom_n + i,
                                            self.sums_prec)
                    else:
                        pows_offset = (n % self.jet_order)*self.numpts
                        acb_mul(tmp,
                                _coeffs(self.sol[m].series + k) + n - base,
                                self.pows + pows_offset + j,
                                self.sums_prec)
                        for i in range(self.jet_order):
                            acb_addmul_fmpz(_coeffs(self.sum_ptr(m, j, k)) + i,
                                            tmp, self.binom_n + i, self.sums_prec)
            acb_mul(self.pows + ((n + 1) % self.jet_order)*self.numpts + j,
                    self.pows + (n       % self.jet_order)*self.numpts + j,
                    self.evpts + j,
                    self.sums_prec)
        acb_clear(tmp)


    cdef void fix_sums(self) noexcept:
        r"""
        Fix the partial sums computed by `next_sum` by dividing the terms
        corresponding to derivatives by suitable powers of the corresponding
        evaluation points.
        """
        cdef slong i, j, k, m
        cdef acb_t inv, invpow
        acb_init(inv)
        acb_init(invpow)
        for j in range(self.numpts):
            if acb_contains_zero(self.evpts + j):
                continue
            acb_inv(inv, self.evpts + j, self.sums_prec)
            acb_one(invpow)
            for i in range(1, self.jet_order):
                acb_mul(invpow, invpow, inv, self.sums_prec)
                for m in range(self.numsols):
                    for k in range(self.sol[m].log_prec):
                        acb_mul(_coeffs(self.sum_ptr(m, j, k)) + i,
                                _coeffs(self.sum_ptr(m, j, k)) + i,
                                invpow,
                                self.sums_prec)
        acb_clear(invpow)
        acb_clear(inv)


    ## Image of a polynomial


    cdef void apply_dop(self, slong base, slong low, slong mid,
                        slong high) noexcept:
        r"""
        *Add* to ``self.sol[:].series[:][mid-base:high-base]`` the coefficients
        of ``self.dop(y[λ+low:λ+mid])`` for each solution ``y``, where the input
        is given in ``self.sol[:].series[:][low-base:mid-base]``.
        """
        cdef slong k
        cdef ApplyDopAlgorithm algo

        if self.apply_dop_algorithm == APPLY_DOP_AUTO:
            # Crude threshold determined by experiments on the diffopdb.bkp.sel
            # family of operators.
            if high - mid >= 7*self.dop_order:
                algo = APPLY_DOP_POLMUL
            else:
                algo = APPLY_DOP_BASECASE_EXACT
        else:
            algo = self.apply_dop_algorithm
        if algo == APPLY_DOP_BASECASE_EXACT:
            if self.dop_is_exact and acb_is_zero(self.leftmost):
                self.apply_dop_basecase_exact(base, low, mid, high)
            else:
                self.apply_dop_basecase(base, low, mid, high)
        elif algo == APPLY_DOP_POLMUL:
            self.apply_dop_polmul(base, low, mid, high)
        elif algo == APPLY_DOP_BASECASE_GENERIC:
            self.apply_dop_basecase(base, low, mid, high)
        elif algo == APPLY_DOP_INTERPOLATION:
            # currently useless (slow and too precision-hungry)
            self.apply_dop_interpolation(base, low, mid, high)
        else:
            assert False


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
    # leftmost = 0, numsols = 1:
    #
    # d²/2·(M(h,p) + p) + r·d²/2·h
    #
    # ((r, d, h) = (order, degree, height) of dop, p = working prec).
    # With the pure acb version, the O(r·d²·h) term can easily dominate in
    # practice, even for small r...
    cdef void apply_dop_basecase(self, slong base, slong low, slong mid,
                                 slong high) noexcept:
        cdef slong i, j, k, m, n, t, j0, length
        cdef acb_ptr b, c, src, dest
        cdef acb_poly_struct *series

        cdef acb_t expo
        acb_init(expo)

        cdef acb_ptr cofac = _acb_vec_init(mid - low)

        # TODO: Also optimize for rational exponents.
        cdef bint leftmost_is_zero = acb_is_zero(self.leftmost)

        for n in range(mid, high):

            j0 = n - mid + 1
            length = min(n - low, self.dop_degree) + 1 - j0

            # - Use some kind of fast multi-point evaluation??
            # - Use acb_poly_taylor_shift instead of looping on t???
            # - Precompute dop coeff*binom?
            for t in range(self.max_log_prec()):

                # The main reason for computing the sum for several initial
                # conditions in parallel is to share this loop.
                for j in range(j0, j0 + length):

                    # c = cofactor of current coeff of sol[:].series in
                    # expression of current output coeff (=> collects
                    # contributions from all terms of dop while often staying
                    # exact and of moderate bit length)
                    #
                    # Each triple (n, t, j) occurs only once in the whole
                    # computation (over all recursive calls etc.). So computing
                    # all values of cofac is basically the same as evaluating
                    # all coefficients of the recurrence associated to dop at
                    # all n.

                    if not leftmost_is_zero:
                        acb_add_si(expo, self.leftmost, n - j, self.prec)

                    c = cofac + j - j0
                    acb_zero(c)
                    for i in range(self.dop_order, t - 1, -1):  # Horner
                        if leftmost_is_zero:
                            acb_mul_si(c, c, n - j, self.prec)
                        else:
                            acb_mul(c, c, expo, self.prec)
                        if j >= acb_poly_length(self.dop_coeffs + i):
                            continue
                        b = _coeffs(self.dop_coeffs + i) + j
                        self.acb_addmul_binom(c, b, i, t, self.prec)

                for m in range(self.numsols):
                    # We could perform a dot product of length log_prec*that
                    # (looping over t in addition to j), but this does not seem
                    # worth the additional complexity at the moment.
                    for k in range(self.sol[m].log_prec - t):
                        series = self.sol[m].series
                        dest = _coeffs(series + k) + n - base
                        src = _coeffs(series + k + t) + mid - 1 - base
                        acb_dot(dest, dest, False,
                                cofac, 1, src, -1, length,
                                self.prec)

        _acb_vec_clear(cofac, mid - low)
        acb_clear(expo)


    cdef void acb_addmul_binom(self, acb_ptr c, acb_srcptr b,
                               slong i, slong t, slong prec) noexcept:
        if t == 0:
            acb_add(c, c, b, prec)
        elif t == 1:
            acb_addmul_si(c, b, i, prec)
        else:
            acb_addmul_fmpz(c, b, fmpz_mat_entry(self.binom, i, t), prec)


    # Same as of apply_dop_basecase but using fmpz, for operators with exact
    # integer coefficients
    #
    # TODO: support coefficients in ℤ[i]
    cdef void apply_dop_basecase_exact(self, slong base, slong low, slong mid,
                                       slong high) noexcept:

        cdef slong i, j, k, m, n, t, j0, length
        cdef acb_ptr src, dest
        cdef acb_poly_struct *series
        cdef fmpz *b
        cdef fmpz *c

        cdef fmpz *cofac = _fmpz_vec_init(mid - low)

        for n in range(mid, high):

            j0 = n - mid + 1
            length = min(n - low, self.dop_degree) + 1 - j0

            for t in range(self.max_log_prec()):

                for j in range(j0, j0 + length):
                    c = cofac + j - j0
                    fmpz_zero(c)
                    for i in range(self.dop_order, t - 1, -1):  # Horner
                        fmpz_mul_si(c, c, n - j)
                        if j >= fmpz_poly_length(self.dop_coeffs_fmpz + i):
                            continue
                        # Here special-casing t ∈ {0,1} does not help much.
                        fmpz_addmul(c,
                                    (self.dop_coeffs_fmpz + i).coeffs + j,
                                    fmpz_mat_entry(self.binom, i, t))

                for m in range(self.numsols):
                    for k in range(self.sol[m].log_prec - t):
                        series = self.sol[m].series
                        dest = _coeffs(series + k) + n - base
                        src = _coeffs(series + k + t) + mid - 1 - base
                        acb_dot_fmpz(dest, dest, False,
                                     src, -1, cofac, 1, length,
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
        cdef slong i, j, k, m
        cdef acb_poly_struct *series

        cdef acb_poly_t tmp
        acb_poly_init(tmp)
        acb_poly_fit_length(tmp, high - low)

        cdef acb_poly_struct *curder = _acb_poly_vec_init(self.max_log_prec())

        for m in range(self.numsols):
            series = self.sol[m].series
            # To compute derivatives, we need a copy of a chunk of `series`
            _acb_poly_vec_set_block(curder, series, self.sol[m].log_prec,
                                    low - base, mid - low)

            for i in range(self.dop_order + 1):

                for k in range(self.sol[m].log_prec):

                    # rhs[k] ← rhs[k] + self.dop(chunk)[k]. This should be a
                    # mulmid, and ignore the constant coefficients.

                    # also tried using mullow_gauss, it seems worse
                    acb_poly_mullow(tmp, self.dop_coeffs + i, curder + k,
                                    high - low, self.prec)
                    acb_poly_shift_right(tmp, tmp, mid - low)
                    _acb_poly_add(_coeffs(series + k) + mid - base,
                                  _coeffs(series + k) + mid - base,
                                  high - mid,
                                  _coeffs(tmp),
                                  acb_poly_length(tmp),
                                  self.prec)

                    # curder[k] ← (d/dx)(previous curder)[k]
                    self.__diff_log_coeff(curder, low, k, self.sol[m].log_prec)

        _acb_poly_vec_clear(curder, self.max_log_prec())
        acb_poly_clear(tmp)


    cdef void __diff_log_coeff(self, acb_poly_struct *f, slong low,
                               slong k, slong log_prec) noexcept:
        cdef slong j
        cdef acb_poly_t tmp
        acb_poly_init(tmp)
        # XXX Optimize case of rational `leftmost`. Maybe fuse some
        # operations.
        acb_poly_scalar_mul(tmp, f + k, self.leftmost, self.prec)
        for j in range(acb_poly_length(f + k)):
            acb_mul_ui(_coeffs(f + k) + j,
                       _coeffs(f + k) + j,
                       low + j,
                       self.prec)
        acb_poly_add(f + k, f + k, tmp, self.prec)
        if k + 1 < log_prec:
            acb_poly_add(f + k, f + k, f + k + 1,
                         self.prec)
        acb_poly_clear(tmp)


    # Variant of ``apply_dop`` using a polynomial middle product based on
    # transposed interpolation at small integers.
    #
    # Approximate cost (partly heuristic) for high - mid = mid - low = d,
    # log_prec = 1, leftmost = 0, numsols = 1:
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
        cdef slong i, j, k, m, n, p
        cdef acb_ptr dest

        cdef acb_t y
        acb_init(y)

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

        cdef acb_poly_struct *curder = _acb_poly_vec_init(self.max_log_prec())
        cdef acb_ptr curderval = _acb_vec_init(2*halflen)
        cdef acb_mat_t prodval
        acb_mat_init(prodval, 2*halflen, self.max_log_prec())

        for m in range(self.numsols):
            _acb_poly_vec_set_block(curder, self.sol[m].series,
                                    self.sol[m].log_prec,
                                    low - base, mid - low)

            for i in range(self.dop_order + 1):

                for k in range(self.sol[m].log_prec):

                    # (ii) Direct evaluation
                    #
                    # XXX Often the dominant step in practice... (Try fast
                    # multipoint evaluation at integers???)
                    eval_reverse_stdpts(curderval, halflen, curder + k,
                                        mid - low, prec)

                    # (iii) Pointwise multiplications, with accumulation in the
                    # transformed domain
                    for p in range(high - low - 1):
                        acb_addmul(acb_mat_entry(prodval, p, k),
                                   acb_mat_entry(tinterp_num, p, i),
                                   curderval + p,
                                   prec)

                    # curder[k] ← (d/dx)(previous curder)[k]
                    self.__diff_log_coeff(curder, low, k, self.sol[m].log_prec)

            # (iv) Transposed Horner evaluation, adding to the values already
            # present in the high part of sol[m].series.

            # (iv.a) Finite points. When high-low == 2*halfen, so that
            # 2*halflen-2 < high-low-1, the last row of the transposed
            # interpolation matrix is not used.
            for n in range(mid, high):
                for k in range(self.sol[m].log_prec):
                    # Would it be faster to compute the sum using acb_dot_ui
                    # with a vector of ones?
                    acb_zero(y)
                    for p in range(2*halflen - 1):
                        acb_add(y, y,
                                acb_mat_entry(prodval, p, k),
                                prec)
                    acb_div_fmpz(y, y, tinterp_den, prec)
                    dest = _coeffs(self.sol[m].series + k) + n - base
                    acb_add(dest, dest, y, prec)
                if n == high - 1:
                    break
                # Using acb_dot_ui with a vector of powers above should be
                # faster than doing a second pass here when the powers fit on a
                # single limb.
                for p in range(2*halflen - 1):
                    for k in range(self.sol[m].log_prec):
                        # there is no _acb_vec_scalar_mul_si
                        acb_mul_si(acb_mat_entry(prodval, p, k),
                                   acb_mat_entry(prodval, p, k),
                                   (p + 1)//2 if p % 2 == 1 else -p//2,
                                   prec)
            # (iv.b) Infinity (last column of the transposed evaluation matrix).
            # The point at infinity only contributes to the coefficient of
            # x^{low+2*halfen-1}. In other words, it contributes to the
            # coefficient of x^{high-1} when high-low is odd, and not at all
            # when it is even.
            if (high - low) % 2 == 1:
                for k in range(self.sol[m].log_prec):
                    acb_div_fmpz(y, acb_mat_entry(prodval, 2*halflen - 1, k),
                                 tinterp_den, prec)
                    dest = _coeffs(self.sol[m].series + k) + high - 1 - base
                    acb_add(dest, dest, y, prec)

        acb_mat_clear(prodval)
        _acb_vec_clear(curderval, 2*halflen)
        _acb_poly_vec_clear(curder, self.max_log_prec())
        acb_clear(y)


    cdef void tinterp_cache_init(self, slong size) noexcept:
        assert self.tinterp_cache_num == NULL
        self.tinterp_cache_size = size
        self.tinterp_cache_num = <acb_mat_struct *> calloc(size,
                                                         sizeof(acb_mat_struct))
        self.tinterp_cache_den = <fmpz *> malloc(size*sizeof(fmpz))


    cdef void tinterp_cache_clear(self) noexcept:
        cdef slong i
        for i in range(self.tinterp_cache_size):
            if (<acb_ptr *> (self.tinterp_cache_num + i))[0] != NULL:
                acb_mat_clear(self.tinterp_cache_num + i)
                fmpz_clear(self.tinterp_cache_den + i)
        free(self.tinterp_cache_num)
        free(self.tinterp_cache_den)


    cdef void tinterp_cache_compute(self, slong n) noexcept:
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


    cdef term_estimate(self, arb_t est, mag_srcptr radpow, slong rhs_offset):
        cdef slong i, k, m
        cdef acb_ptr c
        cdef mag_t _c, cmag
        mag_init(_c)
        mag_init(cmag)

        arb_zero(est)
        cdef mag_ptr crad = arb_radref(est)

        # The code should still work (returning degraded estimates) when the
        # inequality is not satisfied, but this is not used at the moment.
        assert rhs_offset >= self.dop_degree

        # Note that here radpow contains the contribution of z^λ.
        for m in range(self.numsols):
            for k in range(min(self.sol[m].log_prec, rhs_offset)):
                for i in range(self.dop_degree):
                    c = _coeffs(self.sol[m].series + k) + rhs_offset - 1 - i
                    acb_get_mag(_c, c)
                    mag_add(cmag, cmag, _c)
                    mag_max(crad, crad, arb_radref(acb_realref(c)))
                    mag_max(crad, crad, arb_radref(acb_imagref(c)))
        mag_mul(cmag, cmag, radpow)
        mag_mul(crad, crad, radpow)
        # store both parts of the estimate in an arb object for compatibility
        # with existing code
        arf_set_mag(arb_midref(est), cmag)

        mag_clear(cmag)
        mag_clear(_c)


    cdef bint check_convergence(self, object stop, slong n,
                                slong rhs_offset,
                                arb_t est,         # W
                                arb_t tail_bound,  # RW
                                slong next_stride):
        r"""
        Requires:

        - rhs (≈ residuals, see ``get_residual`` for the difference) in
          ``sol[:].series`` at offset ``self.rhs_offset``,
        - last computed terms just before.
        """

        if n <= self.max_ini_shift():
            arb_pos_inf(tail_bound)
            return False

        self.__rhs_offset = rhs_offset  # used in callbacks
        cdef RealBall _est = RealBall.__new__(RealBall)
        _est._parent = self.Reals
        arb_swap(_est.value, est)
        cdef RealBall _tb = RealBall.__new__(RealBall)
        _tb._parent = self.Reals
        arb_swap(_tb.value, tail_bound)

        # When the convergence checker raises a PrecisionError,
        # analytic_continuation expects the result to be a bad but rigorous
        # enclosure. So we keep the old tail bound and return True to instruct
        # sum_blockwise to stop the computation.
        # TODO: Simplify!
        try:
            done, new_tail_bound = stop.check(self, n, _tb, _est, next_stride)
        except accuracy.PrecisionError:
            arb_swap(tail_bound, _tb.value)
            return True

        arb_swap(tail_bound, (<RealBall?> new_tail_bound).value)
        arb_swap(est, _est.value)

        return done


    def get_residuals(self, stop, slong n):
        nres = [self.get_residual(m, n) for m in range(self.numsols)]
        if self.debug:
            self.__check_residuals(stop, n, nres)
        return nres


    cdef object get_residual(self, slong m, slong n):
        cdef Polynomial_complex_arb pol
        cdef slong d, k
        cdef acb_poly_t _ind
        acb_poly_init(_ind)
        cdef acb_t inv
        acb_init(inv)
        cdef acb_t tmp
        acb_init(tmp)
        cdef slong log_prec = self.sol[m].log_prec

        cdef acb_struct *nres_term = _acb_vec_init(log_prec)
        nres = [None]*log_prec
        for k in range(log_prec):
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
            self.eval_ind(_ind, n + d, log_prec)  # not monic!
            acb_inv(inv, acb_poly_get_coeff_ptr(_ind, 0), self.bounds_prec)
            for k in reversed(range(log_prec)):
                # cst*self._residual[k][d]
                # (cst operand constant, could save a constant factor)
                acb_mul(nres_term + k,
                        _coeffs(self.dop_coeffs + self.dop_order),  # cst
                        _coeffs(self.sol[m].series + k) + self.__rhs_offset + d,
                        self.bounds_prec)
                # ... - sum(ind[u]*nres[k+u][d], 1 <= u < log_prec - k)
                acb_dot(nres_term + k,
                        nres_term + k,  # initial value
                        True,           # subtract
                        acb_poly_get_coeff_ptr(_ind, 1), 1,
                        nres_term + k + 1, 1,
                        log_prec - k - 1,
                        self.bounds_prec)
                # inv*(...)
                acb_mul(nres_term + k, nres_term + k, inv, self.bounds_prec)
            for k in range(log_prec):
                acb_swap(_coeffs((<Polynomial_complex_arb> nres[k])._poly) + d,
                         nres_term + k)
        for k in range(log_prec):
            _acb_poly_normalise((<Polynomial_complex_arb> nres[k])._poly)
        _acb_vec_clear(nres_term, log_prec)
        acb_clear(inv)
        acb_poly_clear(_ind)

        return nres


    def __check_residuals(self, stop, n, nres):
        r"""
        Debugging utility.

        Recompute the residual using the reference code in bounds.py.
        """
        cdef slong i, k, m
        cdef ComplexBall b
        cdef acb_ptr rhs
        cdef Parent IC = ComplexBallField(self.bounds_prec)
        if self.__rhs_offset < self.dop_degree:
            logger.info("n=%s cannot check residual", n)
        for m in range(self.numsols):
            last = []
            for i in range(self.dop_degree):
                cc = []
                for k in range(self.sol[m].log_prec):
                    b = <ComplexBall> ComplexBall.__new__(ComplexBall)
                    b._parent = IC
                    rhs = _coeffs(self.sol[m].series + k) + self.__rhs_offset
                    acb_set(b.value, rhs - 1 - i)
                    cc.append(b)
                last.append(cc)
            ref = stop.maj.normalized_residual(n, last)
            if not all(c.contains_zero()
                       for p, q in zip(nres[m], ref)
                       for c in p - q):
                logger.error("n=%s m=%s bad residual:\nnres=%s\nref=%s",
                            n, m, nres, ref)
                assert False


    def get_bound(self, stop, n, resid):
        if n <= self.max_ini_shift():
            raise NotImplementedError
        # Support separate tail bounds for individual series?
        # (This should not be difficult to do by moving est and tb to Solution,
        # but maybe not worth the repeated calls to tail_majorant, at least
        # while the code for tail bounds is so slow. But do not forget we may
        # also be evaluating the same series at several points.)
        maj = stop.maj.tail_majorant(n, resid)
        tb = maj.bound(self.py_evpts.rad, rows=self.jet_order)
        # XXX take values of x^λ and log(x) into account (as in naive_sum)?
        return tb


    cdef slong max_ini_shift(self):
        cdef slong i
        cdef slong res = -1
        for i in range(self.dop_order + 1):
            if self.ini_shifts[i] == -1:
                return res
            if self.ini_shifts[i] > res:
                res = self.ini_shifts[i]


cdef acb_ptr _coeffs(acb_poly_t pol) noexcept:
    # pol->coeffs is not accessible from sage, and acb_poly_get_coeff_ptr
    # returns null when the argument is out of bounds
    return (<acb_ptr *> pol)[0]


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


cdef acb_poly_struct *_acb_poly_vec_init(slong n) noexcept:
    cdef slong i
    cdef acb_poly_struct *vec
    vec = <acb_poly_struct *> malloc(n*sizeof(acb_poly_struct))
    for i in range(n):
        acb_poly_init(vec + i)
    return vec


cdef void _acb_poly_vec_clear(acb_poly_struct *vec, slong n) noexcept:
    cdef slong i
    for i in range(n):
        acb_poly_clear(vec + i)
    free(vec)


cdef void _acb_poly_vec_set_block(acb_poly_struct *tgt, acb_poly_struct *src,
                                  slong n, slong base, slong length):
    cdef slong k
    for k in range(n):
        acb_poly_fit_length(tgt + k, length)
        _acb_vec_set(_coeffs(tgt + k), _coeffs(src + k) + base, length)
        _acb_poly_set_length(tgt + k, length)
        _acb_poly_normalise(tgt + k)


cdef fmpz_poly_struct *_fmpz_poly_vec_init(slong n) noexcept:
    cdef slong i
    cdef fmpz_poly_struct *vec
    vec = <fmpz_poly_struct *> malloc(n*sizeof(fmpz_poly_struct))
    for i in range(n):
        fmpz_poly_init(vec + i)
    return vec


cdef void _fmpz_poly_vec_clear(fmpz_poly_struct *vec, slong n) noexcept:
    cdef slong i
    for i in range(n):
        fmpz_poly_clear(vec + i)
    free(vec)


## Debugging utilities


cdef void _print_solution(Solution *sol):
    cdef slong k
    flint_printf("SOL(log_prec=%d real=%d):\n", sol.log_prec, sol.real)
    flint_printf("ini/crit=\n%{acb_mat}\n", sol.critical_coeffs)
    flint_printf("series=\n")
    for k in range(sol._log_alloc):
        flint_printf("[%{acb_poly}]*LOG^%d\n", sol.series + k, k)
    flint_printf("\n")


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

