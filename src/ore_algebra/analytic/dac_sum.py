# vim: tw=80
r"""
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

from itertools import count
from types import SimpleNamespace

from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.rings.complex_arb import ComplexBallField

from . import accuracy
from . import utilities

from .bounds import DiffOpBound
from .context import dctx
from .local_solutions import HighestSolMapper, log_series_values

from .dac_sum_c import DACUnroller


logger = logging.getLogger(__name__)


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
        bit_prec = 8 + bit_prec0*(1 + (self.dop_T.degree() - 2).nbits())
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
