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

    sage: from ore_algebra.examples.periods import dop_140118_4
    sage: dop_140118_4.numerical_transition_matrix([0,1], 1e-200,
    ....:         assume_analytic=True, algorithm="dac")[3,3]
    [0.0037773116...2646238326...]

    sage: from ore_algebra.examples import fcc
    sage: fcc.dop5.numerical_solution( # long time (4.8 s)
    ....:          [0, 0, 0, 0, 1, 0], [0, 1/5+i/2, 1],
    ....:          1e-60, algorithm=["dac"])
    [1.04885235135491485162956376369999275945402550465206640...] + [+/- ...]*I

    sage: from ore_algebra.examples import polya
    sage: polya.dop[10].numerical_solution([0]*9+[1], [0,1/20], algorithm="dac")
    [1.05954374788826...] + [+/- ...]*I

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

Variants of ``apply_dop``::

    sage: from ore_algebra.analytic.dac_sum_c import ApplyDopAlgorithm
    sage: dop = (x*Dx)^4*(x*Dx-1/3)^4*((x*Dx)^2-2) + x^3*Dx + 5*x^2
    sage: ref = dop.numerical_solution(range(10), [0,1],
    ....:         post_transform=1+1/7*Dx^5, algorithm=["naive"])
    sage: ref
    [23.9550253138882...]
    sage: [(algo,
    ....:   dop.numerical_solution(range(10), [0,1], post_transform=1+1/7*Dx^5,
    ....:                          algorithm=["dac"], apply_dop=algo)
    ....:   .overlaps(ref))
    ....:  for algo in ApplyDopAlgorithm.__members__.keys()]
    [('APPLY_DOP_POLMUL', True),
     ('APPLY_DOP_BASECASE_GENERIC', True),
     ('APPLY_DOP_BASECASE_EXACT', True),
     ('APPLY_DOP_INTERPOLATION', True)]
"""

import logging

from itertools import count
from types import SimpleNamespace

from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.rings.complex_arb import ComplexBallField
from sage.rings.integer_ring import ZZ
from sage.structure.sequence import Sequence

from . import accuracy
from . import utilities

from .bounds import DiffOpBound
from .context import dctx
from .differential_operator import DifferentialOperator
from .local_solutions import (
    HighestSolMapper,
    log_series_values,
    LogSeriesInitialValues
)
from .path import EvaluationPoint_base, EvaluationPoint

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

        unr, allsums = self._sum_auto(inis, maj, effort)
        if unr.real():
            Jets = unr.Jets.change_ring(unr.Jets.base().base())
        else:
            Jets = unr.Jets
        sols = []
        for j, sums in enumerate(allsums):
            # fixed solution, entries of sums <-> eval pts
            mult = self.shifts[j][1]
            downshifts = [
                log_series_values(
                    Jets,
                    self.leftmost,  # ini.expo???
                    vector(Jets, psum),
                    self.evpts.approx(Jets.base_ring(), i),
                    self.evpts.jet_order,
                    self.evpts.is_numeric,
                    downshift=range(mult))
                for i, psum in enumerate(sums)]
            sols.append(SolutionAdapter(unr.py_critical_coeffs(j), downshifts))

        return sols

    def _sum_auto(self, inis, maj, effort):
        # Adapted from naive_sum.RecUnroller_tail_bound.sum_auto, with a little
        # code duplication, but this version is much simpler without really
        # being a special case of the other one.

        stop = accuracy.StopOnRigorousBound(maj, self.eps)
        input_accuracy = utilities.input_accuracy(self.evpts, inis)
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
            unr = DACUnroller(self.dop_T, inis, self.evpts, Ring, ctx=self.ctx)
            try:
                unr.sum_blockwise(stop)
            except accuracy.PrecisionError:
                if attempt > effort:
                    raise
            else:
                allsums = unr.py_sums()
                # estimated “total” error accounting for both method error (tail
                # bounds) and interval growth, but ignoring derivatives and with
                # just a rough estimate of the singular factors
                # TODO cythonize, arb --> mag, return unr only
                err = max((abs(jet[0]).rad_as_ball()  # CBF => no rad_as_ball()
                           for sol in allsums for psum in sol for jet in psum),
                          default=unr.IR.zero())
                err *= self.evpts.rad**unr.IC(self.leftmost).real()
                logger.debug("bit_prec = %s, err = %s (tgt = %s)",
                             bit_prec, err, self.eps)
                if err < self.eps:
                    return unr, allsums

            bit_prec *= 2
            if attempt <= effort and bit_prec < max_prec:
                logger.info("lost too much precision, restarting with %d bits",
                            bit_prec)
                continue
            if self.fail_fast:
                raise accuracy.PrecisionError
            else:
                logger.info("lost too much precision, giving up")
                return unr, allsums


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


def truncated_sum(dop, ini, evpts, bit_prec, terms):
    r"""
    Compute a partial sum of a logarithmic series at one or more points using
    ``DACUnroller``.

    EXAMPLES::

        sage: from ore_algebra import DifferentialOperators
        sage: from ore_algebra.analytic.local_solutions import LogSeriesInitialValues
        sage: Dops, x, Dx = DifferentialOperators(QQ, 'x')

        sage: from ore_algebra.analytic.dac_sum import truncated_sum
        sage: truncated_sum(Dx-1, [2], 1/2, 30, 4)
        [[3.2916666...]]
        sage: 2.*(1 + 1/2 + 1/2*(1/2)^2 + 1/6*(1/2)^3)
        3.29166666666667

        sage: truncated_sum(Dx^2 + 1, [1, 0], [RBF(pi), 1/2], 30, 30)
        [[-1.000000...], [0.877582...]]

        sage: truncated_sum(Dx-1, [2], 1/2, 30, 0)
        [0]

        sage: truncated_sum((x*Dx - 1/3)^3*(x*Dx - 7/3) + x,
        ....:     LogSeriesInitialValues(1/3, {0: (1,2,3), 2:(4,)}),
        ....:     [1/2], 30, 30)
        [[6.3444...]]
    """
    dop = DifferentialOperator(dop)
    dop_T = dop.to_T(dop._theta_alg())
    if not isinstance(ini, LogSeriesInitialValues):
        ini = LogSeriesInitialValues(ZZ.zero(), ini, dop)
    if not isinstance(evpts, EvaluationPoint_base):
        if isinstance(evpts, (list, tuple)):
            evpts = tuple(Sequence(evpts))
        evpts = EvaluationPoint(evpts)
    Ring = ComplexBallField(bit_prec)
    unr = DACUnroller(dop_T, [ini], evpts, Ring)
    unr.sum_blockwise(stop=None, max_terms=terms)
    [sums] = unr.py_sums()
    return [log_series_values(unr.Jets, ini.expo, vector(unr.Jets, psum),
                              evpts.approx(unr.Jets.base_ring(), i),
                              derivatives=1, is_numeric=True)[0][0]
            for i, psum in enumerate(sums)]

