# vim: tw=80
r"""
Divide-and-conquer summation of convergent D-finite series

The bulk of the implementation lives in ``dac_sum_c.pyx``; this module only
provides utilities for using ``DACUnroller`` from Python and interfacing it with
the rest of ``ore_algebra.analytic``.

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
    [1.059543747888...] + [+/- ...]*I

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

This used to return an incorrect enclosure due to Sage bug #38746::

    sage: (Dx^2 + x).numerical_solution([1, 0], [0,108], 1e-43, algorithm="dac")
    [0.273126153520200398594082901520988383972...]
"""

import logging

from itertools import count
from types import SimpleNamespace

from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.rings.complex_arb import ComplexBallField
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.real_arb import RealBallField
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


# Fundamental matrices with rigorous error bounds


# XXX maybe refactor to share more code with HighestSolMapper_tail_bound
class HighestSolMapper_dac(HighestSolMapper):

    def __init__(self, dop, evpts, eps, fail_fast, effort, *, ctx):
        super().__init__(dop, evpts, ctx=ctx)
        self.eps = eps
        self.fail_fast = fail_fast
        self.effort = effort

        self.dop_T = dop.to_T(dop._theta_alg())

        self.IR = ctx.IR
        self.IC = ctx.IC

    def do_sum(self, inis):

        # XXX get rid of this if possible?
        maj = DiffOpBound(self.dop, self.leftmost,
                        special_shifts=(None if self.ordinary else self.shifts),
                        bound_inverse="solve",
                        pol_part_len=(4 if self.ordinary else None),
                        ind_roots=self.all_roots,
                        ctx=self.ctx)

        effort = self.effort

        # Adapted from naive_sum.RecUnroller_tail_bound.sum_auto, with a little
        # code duplication, but this version is much simpler without really
        # being a special case of the other one.

        stop = accuracy.StopOnRigorousBound(maj, self.eps)
        input_accuracy = utilities.input_accuracy(self.evpts, inis)
        # cf. stop.reset(...) below for the term involving dop
        bit_prec0 = utilities.prec_from_eps(self.eps) + 2*self.dop.order()
        sums_prec = 8 + bit_prec0 + 4*bit_prec0.nbits()  # TBI
        bit_prec = 8 + bit_prec0*(1 + (self.dop_T.degree() - 2).nbits())
        max_prec = bit_prec + 2*input_accuracy  # = ∞ for exact input
        logger.info("initial working precision = %s bits", bit_prec)

        for attempt in count(1):
            logger.debug("attempt #%s (of max %s), bit_prec=%s, sums_prec=%s",
                         attempt, effort + 2, bit_prec, sums_prec)
            ini_are_accurate = 2*input_accuracy > bit_prec
            # Ask for a bit more accuracy than we really require because
            # DACUnroller does not guarantee that the result will really be < ε
            # (and ignores some things such as log factors when deciding where
            # to stop). Without that, we would risk failing the accuracy check
            # below on the first attempt almost every time. Also, strictly
            # decrease self.eps at each new attempt to avoid situations where it
            # would be happy with the result and stop at the same point despite
            # the higher bit_prec.
            stop.reset(self.eps >> (self.dop.order() + 4*attempt),
                       stop.fast_fail and ini_are_accurate)
            unr = DACUnroller(self.dop_T, inis, self.evpts, sums_prec, bit_prec,
                              ctx=self.ctx)
            CCp = ComplexBallField(bit_prec)
            Jets = PolynomialRing(CCp, 'delta')

            unr.sum_blockwise(stop)
            allsums = unr.py_sums(Jets)
            # estimated “total” error accounting for both method error (tail
            # bounds) and interval growth, but ignoring derivatives and with
            # just a rough estimate of the singular factors
            # TODO cythonize, arb --> mag
            err = max((abs(jet[0]).rad_as_ball()  # CBF => no rad_as_ball()
                        for sol in allsums for psum in sol for jet in psum),
                        default=self.IR.zero())
            err *= self.evpts.rad**self.IC(self.leftmost).real()
            logger.debug("bit_prec = %s, err = %s (tgt = %s)",
                            bit_prec, err, self.eps)
            if err < self.eps:
                break

            bit_prec *= 2
            # Attempt to deal with large humps in the magnitude of the series
            # terms. TODO This should really use information from the previous
            # runs to refine the guess for sums_prec, and there should probably
            # be some notion of _relative_ error tolerante in play somewhere.
            sums_prec = 6*sums_prec//5 if attempt < (effort+1)//2 else bit_prec
            # Try at least twice (attempts = effort + 2) so that the logic from
            # the previous line has a chance to run.
            if attempt <= effort + 1 and bit_prec < max_prec:
                logger.info("lost too much precision, restarting with %d bits "
                            "(%d bits for sums)",
                            bit_prec, sums_prec)
                continue
            if self.fail_fast:
                raise accuracy.PrecisionError
            else:
                logger.info("lost too much precision, giving up")
                break

        if unr.real():
            Jets = PolynomialRing(RealBallField(bit_prec), 'delta')

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
            # XXX should this use the real field when everything is real?
            crit = unr.py_critical_coeffs(j, CCp)
            sols.append(SolutionAdapter(crit, downshifts))

        return sols


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


# Truncated series, partial sums and truncated fundamental matrices
# (computed in interval arithmetic but without tail bounds)


def truncated_series(dop, inis, bit_prec, terms):
    r"""
    Compute truncated series solutions of ``dop``.

    The initial values must all define solutions in the same group.

    The series are returned as lists of polynomials, where the i-th entry is the
    coefficient of `\log(x)^i/i!`. Singular parts are omitted.

    EXAMPLES::

        sage: from ore_algebra import DifferentialOperators
        sage: from ore_algebra.analytic.local_solutions import LogSeriesInitialValues
        sage: Dops, x, Dx = DifferentialOperators(QQ, 'x')

        sage: from ore_algebra.analytic.dac_sum import truncated_series

        sage: truncated_series(Dx^2 - 1, [(1,0), (0,1)], 20, 6)
        [({[z^(0+0)·log(z)^0/0!] = 1, [z^(0+1)·log(z)^0/0!] = 0},
          [([0.041666...])*x^4 + 0.500000*x^2 + 1.00000]),
         ({[z^(0+0)·log(z)^0/0!] = 0, [z^(0+1)·log(z)^0/0!] = 1},
          [([0.008333...])*x^5 + ([0.16666...])*x^3 + x])]

        sage: truncated_series((x*Dx - 1/3)^3*(x*Dx - 7/3) + x,
        ....:        [LogSeriesInitialValues(1/3, {0: (1,2,3), 2:(4,)})], 20, 4)
        [({[z^(1/3+0)·log(z)^0/0!] = 1, [z^(1/3+0)·log(z)^1/1!] = 2,
        [z^(1/3+0)·log(z)^2/2!] = 3, [z^(1/3+2)·log(z)^0/0!] = 4},
          [([-0.47...])*x^3 + 4.00000*x^2 + ([9.0...])*x + 1.00000,
           ([0.206...])*x^3 + ([-2.4...])*x^2 + ([-4.00...])*x + 2.00000,
           ([-0.0671...])*x^3 + ([1.06...])*x^2 + ([3.00...])*x + 3.00000,
           ([0.0138...])*x^3 + ([-0.375...])*x^2])]

    TESTS::

        sage: QQi.<I> = QuadraticField(-1)
        sage: Dops, t, Dt = DifferentialOperators(QQi, 't')
        sage: truncated_series(Dt - I, [(1,)], 30, 3)
        [({[z^(0+0)·log(z)^0/0!] = 1}, [-0.500000000*t^2 + I*t + 1.00000000])]
    """
    dop = DifferentialOperator(dop)
    dop_T = dop.to_T(dop._theta_alg())
    if not inis:
        raise ValueError("need at least one set of initial values")
    for i in range(len(inis)):
        if not isinstance(inis[i], LogSeriesInitialValues):
            inis[i] = LogSeriesInitialValues(ZZ.zero(), inis[i], dop)
    evpts = EvaluationPoint([])
    unr = DACUnroller(dop_T, inis, evpts, bit_prec, bit_prec, keep_series=True)
    unr.sum_blockwise(stop=None, max_terms=terms)
    Series = dop.base_ring().change_ring(ComplexBallField(bit_prec))
    series = [(ini, unr.py_series(m, Series))
              for m, ini in enumerate(inis)]
    return series


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

        sage: truncated_sum(Dx-1, [2], 1/2, 30, -1)
        [0]

        sage: truncated_sum((x*Dx - 1/3)^3*(x*Dx - 7/3) + x,
        ....:     LogSeriesInitialValues(1/3, {0: (1,2,3), 2:(4,)}),
        ....:     [1/2], 30, 30)
        [[6.3444...]]

        sage: truncated_sum(Dx-1, [2], [], 30, 4)
        []
    """
    dop = DifferentialOperator(dop)
    dop_T = dop.to_T(dop._theta_alg())
    if not isinstance(ini, LogSeriesInitialValues):
        ini = LogSeriesInitialValues(ZZ.zero(), ini, dop)
    if not isinstance(evpts, EvaluationPoint_base):
        if isinstance(evpts, (list, tuple)):
            evpts = tuple(Sequence(evpts))
        evpts = EvaluationPoint(evpts)
    # XXX accept a sums_prec here? or compute it based on
    # utilities.input_accuracy(self.evpts, inis)?
    unr = DACUnroller(dop_T, [ini], evpts, bit_prec, bit_prec)
    unr.sum_blockwise(stop=None, max_terms=terms)
    CCp = ComplexBallField(bit_prec)
    Jets = PolynomialRing(CCp, 'delta')
    [sums] = unr.py_sums(Jets)
    return [log_series_values(Jets, ini.expo, vector(Jets, psum),
                              evpts.approx(CCp, i),
                              derivatives=1, is_numeric=True)[0][0]
            for i, psum in enumerate(sums)]


class HighestSolMapper_dac_truncated(HighestSolMapper):

    def __init__(self, dop, evpts, bit_prec, terms, *, ctx):
        super().__init__(dop, evpts, ctx=ctx)
        self.bit_prec = bit_prec
        self.terms = terms
        self.dop_T = dop.to_T(dop._theta_alg())

    def do_sum(self, inis):
        # XXX accept a sums_prec here? or compute it based on
        # utilities.input_accuracy(self.evpts, inis)?
        unr = DACUnroller(self.dop_T, inis, self.evpts,
                          self.bit_prec, self.bit_prec,
                          ctx=self.ctx)
        unr.sum_blockwise(stop=None, max_terms=self.terms)
        CCp = ComplexBallField(self.bit_prec)
        Jets = PolynomialRing(CCp, 'delta')
        # basically copied from HighestSolMapper_dac
        sols = []
        for j, sums in enumerate(unr.py_sums(Jets)):
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
            crit = unr.py_critical_coeffs(j, CCp)
            sols.append(SolutionAdapter(crit, downshifts))
        return sols


def fundamental_matrix_regular_truncated(dop, evpts, bit_prec, *,
                                         terms, ctx=dctx):
    r"""
    Fundamental matrices at the points in ``evtps``, truncated after ``terms``
    terms.

    The output is a list of matrices. Each list element corresponds to an
    evaluation point. Each column of a given list element corresponds to an
    element of the canonical basis of solutions of ``dop``, truncated after
    ``terms`` terms counted not from its own valuation but from that of leading
    solution in its group. The `i`th row contains the `i`th derivative (of the
    truncated series, not the truncated derivative of the infinite series)
    multiplied by `i!`.

    Unlike the function of the same name in ``naive_sum``, this function
    currently does not support truncating all series to the same absolute order.

    TESTS::

        sage: from ore_algebra import DifferentialOperators
        sage: Dops, x, Dx = DifferentialOperators(QQ, 'x')
        sage: from ore_algebra.analytic.dac_sum import fundamental_matrix_regular_truncated
        sage: m1, m2 = fundamental_matrix_regular_truncated((x*Dx - 1/3)^3*(x*Dx - 7/3) + x, [1/2, -1/2], 30, terms=30)
        sage: m1*vector((2, 3, 5, 7))
        ([7.5038...], [17.5924...], [9.394...], [-1.418...])
        sage: m2*vector((2, 3, 5, 7))
        ([-6.9906...] + [0.0230...]*I, [6.806...] + [-12.278...]*I,
        [18.406...] + [3.783...]*I, [7.993...] + [3.254...]*I)

        sage: fundamental_matrix_regular_truncated((x*Dx)^2 - 3 + x^2, [1/4], bit_prec=20, terms=1)
        [
        [ [11.03...] [0.0906...]]
        [ [-76.4...]  [0.627...]]
        ]
        sage: (1/4.)^(-sqrt(3.)), -sqrt(3.)*(1/4.)^(-sqrt(3.)-1)
        (11.0356646359636, -76.4573273791203)
        sage: (1/4.)^(sqrt(3)), sqrt(3.)*(1/4.)^(sqrt(3.)-1)
        (0.250000000000000^sqrt(3), 0.627801175445067)
        sage: (1/4.)^(sqrt(3.)), sqrt(3.)*(1/4.)^(sqrt(3.)-1)
        (0.0906152944101932, 0.627801175445067)

        sage: fundamental_matrix_regular_truncated((x*Dx)^2*(x*Dx-2), [1/2], bit_prec=20, terms=0)
        [
        [0 0 0]
        [0 0 0]
        [0 0 0]
        ]
        sage: fundamental_matrix_regular_truncated((x*Dx)^2*(x*Dx-2), [1/2], bit_prec=20, terms=2)
        [
        [[-0.6931...]                1.00000                      0]
        [     2.00000                      0                      0]
        [    -2.00000                      0                      0]
        ]
        sage: fundamental_matrix_regular_truncated((x*Dx)^2*(x*Dx-2), [1/2], bit_prec=20, terms=3)
        [
        [[-0.6931...]                1.00000               0.250000]
        [     2.00000                      0                1.00000]
        [    -2.00000                      0                1.00000]
        ]

        sage: fundamental_matrix_regular_truncated(Dx^2 - 1, [RBF(0, rad=.25)], 30, terms=3)
        [
        [[1.0 +/- 0.0313]      [+/- 0.251]]
        [     [+/- 0.251]       1.00000000]
        ]

        sage: fundamental_matrix_regular_truncated(Dx^2 - 1, [], 30, terms=3)
        []
    """
    dop = DifferentialOperator(dop)
    if not isinstance(evpts, EvaluationPoint_base):
        if isinstance(evpts, (list, tuple)):
            evpts = tuple(Sequence(evpts))
        evpts = EvaluationPoint(evpts, jet_order=dop.order())
    hsm = HighestSolMapper_dac_truncated(dop, evpts, bit_prec, terms, ctx=ctx)
    cols = hsm.run()
    mats = [matrix([sol.value[i] for sol in cols]).transpose()
            for i in range(len(evpts))]
    return mats

