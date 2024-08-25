# vim: tw=80

from libc.stdlib cimport malloc, free, abort

from sage.libs.flint.flint cimport flint_printf
from sage.libs.flint.types cimport *
from sage.libs.flint.acb cimport *
from sage.libs.flint.acb_poly cimport *
from sage.libs.flint.arb cimport *

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

    cdef acb_poly_struct *dop_coeffs
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

    cdef acb_poly_struct *jets
    cdef acb_poly_struct *jetpows

    cdef dict _ini
    cdef slong _last_ini

    cdef readonly dict critical_coeffs
    cdef int _rhs_offset

    ## used by remaining python code

    cdef readonly object dop_T, ini, evpts, IR, IC, real, _leftmost

    ## may or may not stay

    cdef object Ring, Pol, _Reals, Pol_IC
    cdef readonly object Jets


    def __cinit__(self, dop_T, ini, evpts, *args, **kwds):
        # XXX maybe optimize data layout
        cdef int i

        cdef size_t sz_poly = sizeof(acb_poly_struct)
        self.dop_order = dop_T.order()
        self.dop_degree = dop_T.degree()
        self.numpts = len(evpts)
        self.jet_order = evpts.jet_order

        self.dop_coeffs = <acb_poly_struct *> malloc((self.dop_order + 1)*sz_poly)
        for i in range(self.dop_order + 1):
            acb_poly_init(self.dop_coeffs + i)

        # using dop_order as a crude bound for max log prec
        # (needs updating to support inhomogeneous equations)
        self.max_log_prec = self.dop_order
        self.log_prec = 0
        self.series = <acb_poly_struct *> malloc(self.max_log_prec*sz_poly)
        for i in range(self.max_log_prec):
            acb_poly_init(self.series + i)

        self.jets = <acb_poly_struct *> malloc(self.numpts*sz_poly)
        self.jetpows = <acb_poly_struct *> malloc(self.numpts*sz_poly)
        for i in range(self.numpts):
            acb_poly_init(self.jets + i)
            acb_poly_init(self.jetpows + i)

        self.sums = <acb_poly_struct *> malloc(self.numpts*self.max_log_prec*sz_poly)
        for i in range(self.numpts*self.max_log_prec):
            acb_poly_init(self.sums + i)
            acb_poly_fit_length(self.sums + i, self.jet_order)

        acb_poly_init(self.ind)
        acb_init(self.leftmost)
        arb_init(self.rad)


    def __dealloc__(self):
        cdef int i

        arb_clear(self.rad)
        acb_poly_clear(self.ind)
        acb_clear(self.leftmost)

        for i in range(self.numpts*self.max_log_prec):
            acb_poly_clear(self.sums + i)
        free(self.sums)

        for i in range(self.numpts):
            acb_poly_clear(self.jetpows + i)
            acb_poly_clear(self.jets + i)
        free(self.jetpows)
        free(self.jets)

        for i in range(self.max_log_prec):
            acb_poly_clear(self.series + i)
        free(self.series)

        for i in range(self.dop_order + 1):
            acb_poly_clear(self.dop_coeffs + i)
        free(self.dop_coeffs)


    def __init__(self, dop_T, ini, evpts, Ring, *, ctx=dctx):

        cdef int i

        assert dop_T.parent().is_T()

        ### Kept for gradual Python to Cython conversion; would change or
        ### disappear in a pure Cython version

        self.dop_T = dop_T
        self.ini = ini  # LogSeriesInitialValues
        self.evpts = evpts

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

        for i, pol in enumerate(dop_T):
            acb_poly_swap(self.dop_coeffs + i,
                          (<Polynomial_complex_arb?> (self.Pol(pol)))._poly)

        self._leftmost = leftmost = Ring(ini.expo)
        acb_set(self.leftmost, (<ComplexBall?> leftmost).value)

        arb_set(self.rad, (<RealBall?> evpts.rad).value)

        acb_poly_fit_length(self.ind, self.dop_order + 1)
        _acb_poly_set_length(self.ind, self.dop_order + 1)
        for i in range(self.dop_order + 1):  # (unnecessary copies)
            acb_poly_get_coeff_acb(_coeffs(self.ind) + i, self.dop_coeffs + i, 0)
        _acb_poly_normalise(self.ind)

        self.prec = Ring.precision()
        self.bounds_prec = self.IR.precision()

        Jets, jets = evpts.jets(Ring)
        for i in range(self.numpts):
            acb_poly_swap(self.jets + i,
                          (<Polynomial_complex_arb?> jets[i])._poly)

        ### Auxiliary output

        self.critical_coeffs = {}

        # Precomputed data, also available as auxiliary output

        self.Jets = Jets

        # At the moment Ring must in practice be a complex ball field (other
        # rings do not support all required operations); this flag signals that
        # the series (incl. singular part) and evaluation points are real.
        self.real = (evpts.is_real_or_symbolic
                     and ini.is_real(dop_T.base_ring().base_ring()))


    def sum_blockwise(self, stop):
        cdef slong i, j, k
        cdef acb_ptr c
        cdef arb_t tb
        cdef arb_t radpow, radpow_blk

        # Block size must be >= deg.
        cdef slong blksz = max(1, self.dop_degree)
        for k in range(self.max_log_prec):
            acb_poly_fit_length(self.series + k, 2*blksz)
        cdef slong blkstride = max(2, 32//blksz)

        arb_init(radpow)
        arb_pow(radpow, self.rad, acb_imagref(self.leftmost), self.bounds_prec)
        arb_init(radpow_blk)
        arb_pow_ui(radpow_blk, self.rad, blksz, self.bounds_prec)

        arb_init(tb)
        arb_pos_inf(tb)

        for i in range(self.numpts):
            acb_poly_one(self.jetpows + i)

        cdef bint done = False
        cdef slong b = 0
        while True:
            self.sum_dac(b*blksz, b*blksz, (b+1)*blksz)
            self.apply_dop(b*blksz, b*blksz, (b+1)*blksz, (b+2)*blksz)

            if b % blkstride == 0:
                if self.check_convergence(stop, (b+1)*blksz, blksz,
                                          tb, radpow, blkstride*blksz):
                    break

            for k in range(self.log_prec):
                acb_poly_shift_right(self.series + k, self.series + k, blksz)

            arb_mul(radpow, radpow, radpow_blk, self.bounds_prec)

            b += 1

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
                psum = Polynomial_complex_arb.__new__(Polynomial_complex_arb)
                psum._parent = self.Jets
                acb_poly_swap(psum._poly, self.sum_ptr(j, k))
                psums[j][k] = psum

        arb_clear(tb)
        arb_clear(radpow_blk)
        arb_clear(radpow)

        return psums


    cdef sum_dac(self, slong base, slong low, slong high):
        r"""
        Compute the chunk ``y[λ+low:λ+high]`` of the solution of ``L(y) = rhs``
        for a given rhs itself of support contained in ``λ+low:λ+high``.
        Works in place on ``self.series[:][low-base:high-base]``.
        """

        assert base <= low <= high

        if high == low:
            return [[]]*len(self.evpts)

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


    cdef next_term(self, slong base, slong n):
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

        cdef acb_t neginvlc
        acb_init(neginvlc)  # XXX reuse?

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
            # should probably use mul-by-rational's when λ ∈ ℚ and the
            # coefficients of ind are exact
            if acb_is_zero(neginvlc):
                acb_inv(neginvlc, _coeffs(ind_n) + mult, self.prec)
                acb_neg(neginvlc, neginvlc)

            acb_mul(new_term + k, new_term + k, neginvlc, self.prec)

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
        acb_clear(neginvlc)
        acb_poly_clear(ind_n)


    cdef acb_poly_struct *sum_ptr(self, int j, int k):
        # self.sums + j*self.max_log_prec + k is the jet of order self.jet_order
        # of the coeff of log^k/k! in the sum at the point of index j
        return self.sums + j*self.max_log_prec + k


    cdef next_sum(self, slong base, slong n):
        cdef int j, k
        cdef acb_poly_t tmp
        acb_poly_init(tmp)
        for j in range(self.numpts):
            for k in range(self.log_prec):
                acb_poly_scalar_mul(tmp, self.jetpows + j,
                                    _coeffs(self.series + k) + n - base,
                                    self.prec)
                acb_poly_add(self.sum_ptr(j, k), self.sum_ptr(j, k), tmp,
                             self.prec)

            acb_poly_mullow(self.jetpows + j, self.jetpows + j, self.jets + j,
                            self.jet_order, self.prec)
        acb_poly_clear(tmp)


    cdef apply_dop(self, slong base, slong low, slong mid, slong high):
        r"""
        *Add* to ``self.series[:][mid-base:high-base]`` the coefficients of
        ``self.dop(y[λ+low:λ+mid])``, where the input is given in
        ``self.series[:][low-base:mid-base]``.
        """

        assert base <= low <= mid <= high

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

                # This should be a mulmid.
                acb_poly_mullow(tmp, self.dop_coeffs + i, curder + k, high - low, self.prec)
                acb_poly_shift_right(tmp, tmp, mid - low)
                if acb_poly_length(self.series + k) < high - base:
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


    cdef eval_ind(self, acb_poly_t ind_n, slong n, int order):
        cdef acb_t expo
        acb_poly_init(ind_n)
        acb_init(expo)
        acb_add_si(expo, self.leftmost, n, self.prec)
        # XXX would compose_series be faster? or evaluate, evaluate2 when
        # relevant?
        acb_poly_taylor_shift(ind_n, self.ind, expo, self.prec)
        acb_poly_truncate(ind_n, order)
        acb_poly_fit_length(ind_n, order)
        acb_clear(expo)


    # Error control and BoundCallbacks interface


    cdef bint check_convergence(self, object stop, slong n,
                                slong blksz,
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
        cdef slong i, k
        cdef acb_ptr c

        if n <= self._last_ini:
            arb_pos_inf(tail_bound)
            return False

        # Note that here radpow contains the contribution of z^λ.
        cdef RealBall est = RealBall.__new__(RealBall)
        est._parent = self._Reals
        for k in range(self.log_prec):
            for i in range(blksz):
                # TODO Use a low-prec estimate instead (but keep reporting
                # accuracy information)
                c = _coeffs(self.series + k) + i
                arb_addmul_si(est.value, acb_realref(c),
                              arb_sgn_nonzero(acb_realref(c)), self.prec)
                arb_addmul_si(est.value, acb_imagref(c),
                              arb_sgn_nonzero(acb_imagref(c)), self.prec)
        arb_mul_arf(est.value, est.value, arb_midref(radpow), self.prec)

        cdef RealBall _tb = RealBall.__new__(RealBall)
        _tb._parent = self._Reals
        arb_swap(_tb.value, tail_bound)
        self._rhs_offset = blksz  # only used by __check_residuals
        done, new_tail_bound = stop.check(self, n, _tb, est, next_stride)
        arb_swap(tail_bound, (<RealBall?> new_tail_bound).value)
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
        tb = maj.bound(self.evpts.rad, rows=self.evpts.jet_order)
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
