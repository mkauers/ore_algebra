# vim: tw=80
"""
Divide-and-conquer summation of convergent D-finite series

TESTS::

    sage: from ore_algebra import DifferentialOperators
    sage: Dops, x, Dx = DifferentialOperators(QQ, 'x')

    sage: ((x^2 + 1)*Dx^2 + 2*x*Dx).numerical_solution([0, 1],
    ....:         [0, i+1, 2*i, i-1, 0], algorithm=["dac"])
    [3.14159265358979...] + [+/- ...]*I

    sage: NF.<sqrt2> = QuadraticField(2)
    sage: dop = (x^2 - 3)*Dx^2 + x + 1
    sage: dop.numerical_transition_matrix([0, sqrt2], 1e-10, algorithm=["dac"])
    [[1.669017372...] [1.809514316...]]
    [[1.556515516...] [2.286697055...]]

    sage: (Dx - 1).numerical_solution([1], [0, i + pi], algorithm=["dac"])
    [12.5029695888765...] + [19.4722214188416...]*I

    sage: from ore_algebra.analytic.examples.misc import koutschan1
    sage: koutschan1.dop.numerical_solution(koutschan1.ini, [0, 84],
    ....:         algorithm=["dac"], two_point_mode=False)
    [0.011501537469552017...]

    sage: from ore_algebra.examples import fcc
    sage: fcc.dop5.numerical_solution( # long time (4.8 s)
    ....:          [0, 0, 0, 0, 1, 0], [0, 1/5+i/2, 1],
    ....:          1e-60, algorithm=["dac"])
    [1.04885235135491485162956376369999275945402550465206640...] + [+/- ...]*I

Easy singular example::

    sage: (x*Dx^2 + x + 1).numerical_transition_matrix([0, 1], 1e-10,
    ....:                                             algorithm=["dac"])
    [[-0.006152006884...]    [0.4653635461...]]
    [   [-2.148107776...]  [-0.05672090833...]]

    sage: dop = (Dx*(x*Dx)^2).lclm(Dx-1)
    sage: dop.numerical_solution([2, 0, 0, 0], [0, 1/4], algorithm=["dac"])
    [1.921812055672805...]

Bessel function at an algebraic point::

    sage: alg = QQbar(-20)^(1/3)
    sage: dop = x*Dx^2 + Dx + x
    sage: dop.numerical_transition_matrix([0, alg], 1e-8, algorithm=["dac"])
    [ [3.7849872...] +  [1.7263190...]*I  [1.3140884...] + [-2.3112610...]*I]
    [ [1.0831414...] + [-3.3595150...]*I  [-2.0854436...] + [-0.7923237...]*I]

Ci(sqrt(2))::

    sage: dop = x*Dx^3 + 2*Dx^2 + x*Dx
    sage: ini = [1, CBF(euler_gamma), 0]
    sage: dop.numerical_solution(ini, path=[0, sqrt(2)], algorithm=["dac"])
    [0.46365280236686...]

Whittaker functions with irrational exponents::

    sage: dop = 4*x^2*Dx^2 + (-x^2+8*x-11)
    sage: dop.numerical_transition_matrix([0, 10], algorithm=["dac"])
    [[-3.829367993175840...]  [7.857756823216673...]]
    [[-1.135875563239369...]  [1.426170676718429...]]

An interesting “real-world” example, where one of the local exponents is
Recurrences of order zero::

    sage: (x*Dx + 1).numerical_transition_matrix([0,2], algorithm=["dac"])
    [0.50000000000000000]

Some other corner cases::

    sage: (x*Dx + 1).numerical_transition_matrix([i, i], algorithm=["dac"])
    [1.0000000000000000]

    sage: (Dx - (x - 1)).numerical_solution([1], [0, 1], algorithm=["dac"])
    [0.6065306597126334...]

Nonstandard branches::

    sage: from ore_algebra.examples.iint import f, w, diffop, iint_value
    sage: iint_value(diffop([f[1/4], w[1], f[1]]),
    ....:            [0, 0, 16/9*i, -16/27*i*(-3*i*pi+8)],
    ....:            algorithm=["dac"])
    [-3.445141853366...]

High degree::

    sage: (((x^200-1)//(x-1))*Dx^2 + 5*x*Dx + sum((i+3)*x**i for i in range(500))).numerical_transition_matrix([0,1/16], algorithm="dac")  # long time
    [ [0.99412284058692...] [0.062180605822178...]]
    [[-0.18799810365157...]  [0.98478282225787...]]

Miscellaneous examples::

    sage: QQi.<i> = QuadraticField(-1)
    sage: (Dx - i).numerical_solution([1], [sqrt(2), sqrt(3)], algorithm=["dac"])
    [0.9499135278648561...] + [0.3125128630622157...]*I
"""

import logging

from itertools import count, zip_longest
from types import SimpleNamespace

from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.misc.cachefunc import cached_method
from sage.rings.complex_arb import ComplexBallField
from sage.rings.infinity import infinity

from . import accuracy
from . import utilities

from .bounds import DiffOpBound
from .context import dctx
from .local_solutions import HighestSolMapper, log_series_values

from .dac_sum_c import DACUnroller


logger = logging.getLogger(__name__)


# MAYBE-TODO:
#   - Decouple the code for computing (i) coefficients, (ii) partial sums,
# (iii) sums with error bounds. Could be done by subclassing DACUnroller and/or
# introducing additional classes for sums etc. Use this when computing formal
# log-series solutions.
#   - Add support for inhomogeneous equations with polynomial rhs. (Always
# sum at least up to deg(rhs) when doing error control).
#   - Could keep only the last deg coeffs even in the DAC part (=> slightly
# different indexing conventions).


class PythonDACUnroller:

    def __init__(self, dop_T, ini, evpts, Ring, *, ctx=dctx):

        assert dop_T.parent().is_T()

        self.dop_T = dop_T
        self.ini = ini  # LogSeriesInitialValues
        self.evpts = evpts
        self.ctx = ctx

        # Parents

        # Slices of series solns (w/o logs); must contain λ.
        self.Pol = dop_T.base_ring().change_ring(Ring)
        # Values (of the evaluation point, of partial sums). Currently limited
        # to using a ball field as a base ring, but this may change.
        self.Jets = None  # see below
        # Error bounds
        self.IR = ctx.IR
        self.IC = ctx.IR.complex_field()

        # Precomputed data, also available as auxiliary output

        self.Jets, self.jets = self.evpts.jets(Ring)
        # At the moment Ring must in practice be a complex ball field (other
        # rings do not support all required operations); this flag signals that
        # the series (incl. singular part) and evaluation points are in real.
        self.real = (self.evpts.is_real_or_symbolic
                     and self.ini.is_real(dop_T.base_ring().base_ring()))
        # XXX self.ini.expo is a PolynomialRoot, consider doing part of the work
        # with its exact value instead of an interval approximation
        self.leftmost = Ring(self.ini.expo)  # aka λ in doc/comments
        self.ind = self.Pol([p[0] for p in self.dop_T])
        self.dop_coeffs = [self.Pol(p) for p in self.dop_T]

        # Auxiliary output

        self.critical_coeffs = {}

        self._last = None

        # Internal stuff

        self.__eval_ind_cache_key = None
        self.__eval_ind_cache_val = None
        self._residual = None


    # Maybe get rid of this and use sum_dac only?
    def sum_blockwise(self, stop):
        # Block size must be >= deg. Using the min possible size seems slightly
        # better in practice than rounding to the next power of two at the
        # moment. This may be because the other version does not truncate
        # intermediate operations properly and/or sums more terms in total.
        blksz = max(1, self.dop_T.degree())
        # blksz = 1 << (self.dop_T.degree() - 1).bit_length()
        blkstride = max(2, 32//blksz)
        radpow = self.evpts.rad**self.IC(self.leftmost).real()
        radpow = self.Jets.base_ring()(radpow)
        radpow_blk = self.Jets.base_ring()(self.evpts.rad**blksz)
        jetpows = [self.Jets.one()]*len(self.evpts)
        jetpows_blk = [self._jetpow(i, blksz) for i in range(len(self.evpts))]
        tb = self.IR(infinity)
        rhs = []
        psums = [[]]*len(self.evpts)
        done = False
        for b in count():
            blkseries, blksums = self.sum_dac(rhs, b*blksz, (b+1)*blksz)
            psums = self.combine_sums(psums, blksums, jetpows)
            rhs = self.apply_dop(blkseries, b*blksz, (b+1)*blksz, (b+2)*blksz)
            if b % blkstride == 0:
                done, tb = self.check_convergence(stop, (b+1)*blksz, blkseries,
                                                  rhs, tb, radpow,
                                                  blkstride*blksz, blksz)
            if done:
                break
            radpow *= radpow_blk
            # could save a bit of time by doing multiplying block _sums_ (not
            # recursively) by powers of the evaluation point, ideally with a √N
            # × √N split
            jetpows = [jet0._mul_trunc_(jet1, self.evpts.jet_order)
                       for (jet0, jet1) in zip(jetpows_blk, jetpows)]

        for j, psum in enumerate(psums):
            for k, jet in enumerate(psum):
                psums[j][k] = self.Jets([self._add_error(jet[i], tb)
                                         for i in range(self.evpts.jet_order)])
        return psums

    def sum_dac(self, rhs, low, high):
        r"""
        Solve ``dop(x^(λ+low)*y) = -x^(λ+low)*rhs`` to precision ``high - low``.

        Note the minus sign!

        This method views the coefficients of dop as truncated power series. It
        does not do anything special when ``high - low`` exceeds the degree of
        the coefficients and always computes the full sum.

        INPUT:

        * ``rhs`` - list ``[g0, g1, ...]`` of polys in ``x`` representing
          ``g0 + g1*log(x)/1! + ···``
        * ``high``, ``low`` - integers

        OUTPUT:

        * ``series`` - list ``[f0, f1, ...]`` of polys in ``x`` s.t. the
          log-series solution is ``(f0 + f1*log(x)/1! + ···) + Õ(x^(high-low))``
        * ``sums`` - ``[[f[i](a+δ)+O(δ^jet_order), 0≤i<jet_order], a∈evpts]``
          as a list of lists of polys in ``δ`` (jet variable),
        """

        if high <= low:
            return [], [[]]*len(self.evpts)

        if high - low == 1:
            new_coeff = self.next_term(low, rhs)
            series = [self.Pol([c]) for c in new_coeff]
            sums = [[self.Jets([c]) for c in new_coeff]]*len(self.evpts)
            return series, sums

        mid = (high + low)//2
        # XXX just above the leaves, truncating here is a bit wasteful
        rhs_low = list(f[:mid - low] for f in rhs)
        series_low, sums_low = self.sum_dac(rhs_low, low, mid)
        resid = self.apply_dop(series_low, low, mid, high)
        rhs_high = list((f >> (mid - low)) + g
                        for f, g in zip_longest(rhs, resid,
                                                fillvalue=self.Pol.zero()))
        series_high, sums_high = self.sum_dac(rhs_high, mid, high)

        series = self.combine_series(series_low, series_high, mid - low)
        jetpows = [self._jetpow(i, mid - low) for i in range(len(self.evpts))]
        sums = self.combine_sums(sums_low, sums_high, jetpows)


        return series, sums

    def next_term(self, n, rhs):
        r"""
        INPUT:

        * ``n`` - integer shift wrt ``λ``
        * ``rhs`` - list ``[p0, ..., p_{k-1}]`` of polynomials representing
          ``sum(x^{λ+n}·p_k(x)·log(x)^k/k!)``

        OUTPUT:

        * vector of coefficients of ``log(x)^k/k!``, 0 ≤ k < ``len(rhs)``
        """

        mult = len(self.ini.shift.get(n, ()))
        # max(len(rhs), len(rhs) - ctz(rhz) + mult), where the first case
        # ensures that log_prec is nondecreasing
        log_prec = len(rhs) + mult - utilities.ctz(rhs, mult)

        ind_n = self.eval_ind(n, log_prec)

        invlc = None
        new_term = vector(self.Pol.base_ring(), log_prec)
        for k in range(log_prec - mult - 1, -1, -1):
            # XXX turn that into a truncated mul???
            # XXX sums_of_products?
            combin = sum(ind_n[j]*new_term[k+j]
                          for j in range(mult + 1, log_prec - k)
                          if new_term[k+j])
            if k < len(rhs):
                # Note that rhs is (essentially) the current _residual_, i.e.,
                # the quantity we are computing is the “first” term of the
                # solution of dop(y) = -rhs, not dop(y) = rhs.
                combin += rhs[k][0]
            if combin:
                if invlc is None:
                    invlc = ~ind_n[mult]
                # pylint: disable=invalid-unary-operand-type
                new_term[mult + k] = -invlc * combin

        for p in range(mult - 1, -1, -1):
            new_term[p] = self.ini.shift[n][p]

        if mult > 0:
            self.critical_coeffs[n] = new_term


        return new_term

    # XXX could use multipoint evaluation
    def eval_ind(self, n, order):
        r"""
        Evaluate the indicial polynomial and its first few derivatives at λ+n,
        caching the last computed result.
        """
        if self.__eval_ind_cache_key == (n, order):
            return self.__eval_ind_cache_val
        # using Pol = C[x] here for convenience, but conceptually ind is a
        # polynomial in n and is being evaluated at some n + ε
        n_pert = self.Pol([self.leftmost + n, 1])
        # XXX maybe shift ind so that leftmost is in the coefficients
        ind_n = self.ind.compose_trunc(n_pert, order)
        self.__eval_ind_cache_key = (n, order)
        self.__eval_ind_cache_val = ind_n
        return ind_n

    def apply_dop(self, series, low, mid, high):
        r"""
        Compute ``rhs`` such that
        ``dop(x^(λ+low)*series) = x^(λ+mid)*rhs + Õ(x^(λ+high))``,
        assuming ``low + deg_x(series) < mid``.

        In the application, ``x^low*series = y[low:mid]`` where
        ``dop(x^λ*y) = 0``, so that the output satisfies::

            dop(x^λ*y[:mid])[:high] = x^λ*(dop(x^λ*y[:low])[mid:high] + x^mid*rhs)
                                           ^^^^^^^^^^^^^^^^

        where the underlined term is already known.

        INPUT:

        * ``series`` - list of polys in ``x`` (as in other methods)
        * ``low`` ≤ ``mid`` ≤ ``high`` - integers

        OUTPUT:

        * list of polys in ``x``
        """
        log_prec = len(series)
        rhs = [self.Pol.zero()]*log_prec
        shder = [(self.leftmost + n) for n in range(low, mid)]
        curder = list(series)
        for i, pol in enumerate(self.dop_coeffs):
            for k in range(log_prec):
                # We are only interested in the slice starting at mid, but have
                # no efficient way of computing only those coefficients.
                prod = pol._mul_trunc_(curder[k], high - low)
                rhs[k] += prod >> (mid - low)
                # XXX Will be slow in python.
                curder[k] = self.Pol([shder[n]*u
                                      for n, u in enumerate(curder[k])])
                if k + 1 < log_prec:
                    curder[k] += curder[k+1]
        return rhs

    def combine_series(self, series_low, series_high, shift):
        r"""
        Compute ``f + x^shift*g`` where ``f``, ``g`` are the log-polys
        represented by ``series_low``, ``series_high``.
        """
        ll, lh = len(series_low), len(series_high)
        lmin = min(ll, lh)
        series = [None]*max(ll, lh)
        for k, f in enumerate(series_high):
            series[k] = f << shift
        for k in range(lmin):
            series[k] += series_low[k]
        series[lmin:ll] = series_low[lmin:]
        return series

    # Partial sums

    def combine_sums(self, sums_low, sums_high, jetpows):
        r"""
        Like :meth:`combine_series` but for values.
        """
        sums = [None]*len(self.evpts)
        for i in range(len(self.evpts)):
            ll, lh = len(sums_low[i]), len(sums_high[i])
            lmin = min(ll, lh)
            sums[i] = [None]*max(ll, lh)
            for k, val in enumerate(sums_high[i]):
                sums[i][k] = jetpows[i]._mul_trunc_(val, self.evpts.jet_order)
            for k in range(lmin):
                sums[i][k] += sums_low[i][k]
            sums[i][lmin:ll] = sums_low[i][lmin:]
        return sums

    @cached_method
    def _jetpow(self, idx, expo):
        r"""
        Compute and cache powers of the variable, using the same splitting
        scheme as the main divide-and-conquer procedure.
        """
        if expo == 0:
            return self.Jets.one()
        elif expo == 1:
            return self.jets[idx]
        order = self.evpts.jet_order
        if expo % 2 == 0:
            return self._jetpow(idx, expo/2).power_trunc(2, order)
        else:
            prev = self._jetpow(idx, expo - 1)
            return prev.multiplication_trunc(self.jets[idx], order)

    # Error control and BoundCallbacks interface

    def _add_error(self, a, err):
        if self.real:
            assert a.imag().is_zero()
            return a.parent()(a.real().add_error(err))
        else:
            return a.add_error(err)

    def check_convergence(self, stop, n, series, residual, tail_bound, radpow,
                          next_stride, blksz):
        if n <= self.ini.last_index():
            return False, self.IR('inf')

        # Probable leftover from debugging session (why did I put this here?!)
        # deg = self.dop_T.degree()
        # if any(self.ini.shift.get(n+d, ()) for d in range(deg)):
        #     return False, self.IR('inf')

        # Note that here radpow contains the contribution of z^λ.
        est = sum(abs(c) for f in series for c in f)*radpow.squash()
        # used by callbacks to get_residuals etc.
        self._residual = residual
        self._last = [[f[blksz-1-i] for f in series]  ## temporary
                      for i in range(self.dop_T.degree())]
        done, tail_bound = stop.check(self, n, tail_bound, est, next_stride)
        return done, tail_bound

    def get_residuals(self, stop, n):
        Pol = self.Pol.change_ring(self.IC)

        deg = self.dop_T.degree()
        logs = max(1, len(self._residual))
        cst = self.IC.coerce(self.dop_T.leading_coefficient()[0])
        nres = [[None]*deg for _ in range(logs)]
        for d in range(deg):
            # improvable when, e.g., the last elts of _residual have val > 0
            ind = self.eval_ind(n + d, logs)  # not monic!
            ind = ind.change_ring(self.IC)
            inv = ~ind[0]
            for k in reversed(range(logs)):
                cor = sum(ind[u]*nres[k+u][d]
                          for u in range(1, logs-k))
                nres[k][d] = inv*(cst*self._residual[k][d] - cor)
        nres = [Pol(coefflist) for coefflist in nres]

        ## Temporary debugging stuff
        # assert self.dop_T == stop.maj.dop
        # ref = stop.maj.normalized_residual(n, self._last)
        # assert all(c.contains_zero() for p, q in zip(nres, ref) for c in p - q)

        return [nres]

    def get_bound(self, stop, n, resid):
        if n <= self.ini.last_index():
            raise NotImplementedError
        maj = stop.maj.tail_majorant(n, resid)
        tb = maj.bound(self.evpts.rad, rows=self.evpts.jet_order)
        # XXX take log factors etc. into account (as in naive_sum)?
        return tb


# DACUnroller = PythonDACUnroller

class SolutionAdapter:
    r"""
    Adapter for using HighestSolMapper.
    """

    class PSum:
        def update_downshifts(self, *_):
            pass

    def __init__(self, critical_coeffs, solns):
        # solns: list of list of values (jets); each inner list contains the
        # downshifts of the current solution at a given evaluation point
        self.cseq = SimpleNamespace()
        self.cseq.critical_coeffs = critical_coeffs
        self.psums = []
        for sol in solns:
            psum = self.PSum()
            psum.downshifts = sol
            self.psums.append(psum)


# XXX maybe refactor to share more code with HighestSolMapper_tail_bound
class HighestSolMapper_dac(HighestSolMapper):

    def __init__(self, dop, evpts, eps, fail_fast, effort, *, ctx):
        super().__init__(dop, evpts, ctx=ctx)
        self.eps = eps
        self.fail_fast = fail_fast
        self.effort = effort

        self.dop_T = dop.to_T(dop._theta_alg())

    def do_sum(self, inis):

        # XXX get rid of this if possible?
        maj = DiffOpBound(self.dop, self.leftmost,
                        special_shifts=(None if self.ordinary else self.shifts),
                        bound_inverse="solve",
                        pol_part_len=(4 if self.ordinary else None),
                        ind_roots=self.all_roots,
                        ctx=self.ctx)

        effort = self.effort

        sols = []
        for (_, mult), ini in zip(self.shifts, inis):
            unr, sums = self._sum_auto(ini, maj, effort)
            if unr.real:
                Jets = unr.Jets.change_ring(unr.Jets.base().base())
            else:
                Jets = unr.Jets
            downshifts = [
                log_series_values(
                    Jets,
                    self.leftmost, # ini.expo???
                    vector(Jets, psum),
                    self.evpts.approx(Jets.base_ring(), i),
                    self.evpts.jet_order,
                    self.evpts.is_numeric,
                    downshift=range(mult))
                for i, psum in enumerate(sums)]
            sols.append(SolutionAdapter(unr.critical_coeffs, downshifts))
            # if we computed at least one solution successfully, try a bit
            # harder to compute the remaining ones without splitting the
            # integration step
            effort = self.effort + 1

        return sols

    def _sum_auto(self, ini, maj, effort):
        # Adapted from naive_sum.RecUnroller_tail_bound.sum_auto, with a little
        # code duplication, but this version is much simpler without really
        # being a special case of the other one.

        stop = accuracy.StopOnRigorousBound(maj, self.eps)
        input_accuracy = max(0, min(self.evpts.accuracy, ini.accuracy()))
        # cf. stop.reset(...) below for the term involving dop
        bit_prec0 = utilities.prec_from_eps(self.eps) + 2*self.dop.order()
        bit_prec = 8 + bit_prec0*(1 + (self.dop_T.degree() - 2).nbits())*11//10
        max_prec = bit_prec + 2*input_accuracy  # = ∞ for exact input
        logger.info("initial working precision = %s bits", bit_prec)

        for attempt in count(1):
            logger.debug("attempt #%s (of max %s), bit_prec=%s",
                         attempt, effort + 1, bit_prec)
            ini_are_accurate = 2*input_accuracy > bit_prec
            # Ask for a bit more accuracy than we really require because
            # DACUnroller does not guarantee that the result will really be < ε
            # (and ignores some things such as log factors when deciding where
            # to stop). Without that, we would risk failing the accuracy check
            # below on the first attempt almost every time. Also, strictly
            # decrease self.eps at each new attempt to avoid situations where it
            # would be happy with the result and stop at the same point despite
            # the higher bit_prec.
            Ring = ComplexBallField(bit_prec)
            stop.reset(self.eps >> (self.dop.order() + 4*attempt),
                       stop.fast_fail and ini_are_accurate)
            unr = DACUnroller(self.dop_T, ini, self.evpts, Ring, ctx=self.ctx)
            try:
                psums = unr.sum_blockwise(stop)
            except accuracy.PrecisionError:
                if attempt > effort:
                    raise
            else:
                # estimated “total” error accounting for both method error (tail
                # bounds) and interval growth, but ignoring derivatives and with
                # just a rough estimate of the singular factors
                err = max((abs(jet[0]).rad_as_ball()  # CBF => no rad_as_ball()
                           for psum in psums for jet in psum),
                          default=unr.IR.zero())
                err *= self.evpts.rad**unr.IC(self.leftmost).real()
                logger.debug("bit_prec = %s, err = %s (tgt = %s)",
                             bit_prec, err, self.eps)
                if err < self.eps:
                    return unr, psums

            bit_prec *= 2
            if attempt <= effort and bit_prec < max_prec:
                logger.info("lost too much precision, restarting with %d bits",
                            bit_prec)
                continue
            if self.fail_fast:
                raise accuracy.PrecisionError
            else:
                logger.info("lost too much precision, giving up")
                return unr, psums


def fundamental_matrix_regular(dop, evpts, eps, fail_fast, effort, ctx=dctx):
    r"""
    Fundamental matrix at a possibly regular singular point
    """
    eps_col = ctx.IR(eps)/ctx.IR(dop.order()).sqrt()
    hsm = HighestSolMapper_dac(dop, evpts, eps_col, fail_fast, effort, ctx=ctx)
    cols = hsm.run()
    mats = [matrix([sol.value[i] for sol in cols]).transpose()
            for i in range(len(evpts))]
    return mats
