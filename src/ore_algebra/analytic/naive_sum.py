# -*- coding: utf-8 - vim: tw=80
"""
Evaluation of convergent D-finite series by direct summation
"""

# TODO:
# - support summing a given number of terms rather than until a target accuracy
# is reached?
# - cythonize critical parts?

import collections, itertools, logging

from sage.categories.pushout import pushout
from sage.matrix.constructor import identity_matrix, matrix
from sage.misc.cachefunc import cached_method
from sage.modules.free_module_element import vector
from sage.rings.complex_arb import ComplexBallField, CBF, ComplexBall
from sage.rings.infinity import infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import NumberField_quadratic
from sage.rings.polynomial import polynomial_element
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import QQbar
from sage.rings.rational_field import QQ
from sage.rings.real_arb import RealBallField, RBF, RealBall
from sage.structure.sequence import Sequence

import ore_algebra.ore_algebra as ore_algebra

from . import accuracy, bounds, utilities
from .local_solutions import *
from .safe_cmp import safe_lt
from .shiftless import my_shiftless_decomposition

logger = logging.getLogger(__name__)

################################################################################
# Argument processing etc. (common to the ordinary and the regular case)
################################################################################

def backward_rec(dop, shift=ZZ.zero()):
    Pols_n = PolynomialRing(dop.base_ring().base_ring(), 'n') # XXX: name
    Rops = ore_algebra.OreAlgebra(Pols_n, 'Sn')
    # Using the primitive part here would break the computation of residuals!
    # TODO: add test (arctan); better fix?
    # rop = dop.to_S(Rops).primitive_part().numerator()
    rop = dop.to_S(Rops)
    ordrec = rop.order()
    coeff = [rop[ordrec-k](Pols_n.gen()-ordrec+shift)
             for k in xrange(ordrec+1)]
    return BackwardRec(coeff)

class BackwardRec(object):
    r"""
    A recurrence relation, written in terms of the backward shift operator.

    This class is mainly intended to provide reasonably fast evaluation in the
    context of naïve unrolling.
    """

    def __init__(self, coeff):
        assert isinstance(coeff[0], polynomial_element.Polynomial)
        self.coeff = coeff
        self.base_ring = coeff[0].parent()
        Scalars = self.base_ring.base_ring()
        self.order = len(coeff) - 1
        # Evaluating polynomials over ℚ[i] is slow...
        # TODO: perhaps do something similar for eval_series
        if (isinstance(Scalars, NumberField_quadratic)
                and list(Scalars.polynomial()) == [1,0,1]):
            QQn = PolynomialRing(QQ, 'n')
            self._re_im = [
                    (QQn([c.real() for c in pol]), QQn([c.imag() for c in pol]))
                    for pol in coeff]
            self.eval_int_ball = self._eval_qqi_cbf

    # efficient way to cache the last few results (without cython)?
    def eval(self, tgt, point):
        return [tgt(pol(point)) for pol in self.coeff]

    def _eval_qqi_cbf(self, tgt, point):
        return [tgt(re(point), im(point)) for re, im in self._re_im]

    # Optimized implementation of eval() when point is a Python
    # integer and iv is a ball field. Can be dynamically replaced by
    # one of the above implementations on initialization.
    eval_int_ball = eval

    @cached_method
    def _coeff_series(self, i, j):
        if j == 0:
            return self.coeff[i]
        else:
            return self._coeff_series(i, j - 1).diff()/j

    def eval_series(self, tgt, point, ord):
        return [[tgt(self._coeff_series(i,j)(point)) for j in xrange(ord)]
                for i in xrange(len(self.coeff))]

    def eval_inverse_lcoeff_series(self, tgt, point, ord):
        ser = self.base_ring( # polynomials, viewed as jets
                [self._coeff_series(0, j)(point) for j in xrange(ord)])
        inv = ser.inverse_series_trunc(ord)
        return [tgt(c) for c in inv]

    def __getitem__(self, i):
        return self.coeff[i]

    def shift(self, sh):
        n = self.coeff[0].parent().gen()
        return BackwardRec([pol(sh + n) for pol in self.coeff])

class EvaluationPoint(object):
    r"""
    A ring element (a complex number, a polynomial indeterminate, perhaps
    someday a matrix) where to evaluate the partial sum of a series, along with
    a “jet order” used to compute derivatives and a bound on the norm of the
    mathematical quantity it represents that can be used to bound the truncation
    error.
    """

    # XXX: choose a single place to set the default value for jet_order
    def __init__(self, pt, rad=None, jet_order=1):
        self.pt = pt
        self.rad = (bounds.IR.coerce(rad) if rad is not None
                    else bounds.IC(pt).above_abs())
        self.jet_order = jet_order

        self.is_numeric = utilities.is_numeric_parent(pt.parent())

    def __repr__(self):
        fmt = "{} + η + O(η^{}) (with |.| ≤ {})"
        return fmt.format(self.pt, self.jet_order + 1, self.rad)

    def jet(self, Intervals):
        base_ring = (Intervals if self.is_numeric
                     else pushout(self.pt.parent(), Intervals))
        Jets = utilities.jets(base_ring, 'eta', self.jet_order)
        return Jets([self.pt, 1])

    def is_real(self):
        return utilities.is_real_parent(self.pt.parent())

    def is_precise(self, eps):
        if self.pt.parent().is_exact():
            return True
        elif isinstance(self.pt.parent(), (RealBallField, ComplexBallField)):
            return safe_lt(bounds.IR(self.pt.rad()), eps)

class LogSeriesInitialValues(object):
    r"""
    Initial values defining a logarithmic series.

    - ``self.expo`` is an algebraic number representing the “valuation” of the
      log-series,
    - ``self.shift`` is a dictionary mapping an integer shift s to a tuple of
      initial values corresponding to the coefficients of x^s, x^s·log(x), ...,
      x^s·log(x)^k/k! for some k
    """

    def __init__(self, expo, values, dop=None, check=True):
        r"""
        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.naive_sum import *
            sage: Dops, x, Dx = DifferentialOperators()
            sage: LogSeriesInitialValues(0, {0: (1, 0)}, x*Dx^3 + 2*Dx^2 + x*Dx)
            Traceback (most recent call last):
            ...
            ValueError: invalid initial data for x*Dx^3 + 2*Dx^2 + x*Dx at 0
        """
        try:
            self.expo = ZZ.coerce(expo)
        except TypeError:
            self.expo = QQbar.coerce(expo)
        if isinstance(values, dict):
            all_values = sum(values.values(), ()) # concatenation of tuples
        else:
            all_values = values
            values = dict((n, (values[n],)) for n in xrange(len(values)))
        self.universe = Sequence(all_values).universe()
        if not utilities.is_numeric_parent(self.universe):
            raise ValueError("initial values must coerce into a ball field")
        self.shift = { s: tuple(self.universe(a) for a in ini)
                       for s, ini in values.iteritems() }

        try:
            if check and dop is not None and not self.is_valid_for(dop):
                raise ValueError("invalid initial data for {} at 0".format(dop))
        except TypeError: # coercion problems btw QQbar and number fields
            pass

    def __repr__(self):
        return ", ".join(
            "[z^({expo}+{shift})·log(z)^{log_power}/{log_power}!] = {val}"
            .format(expo=self.expo, shift=s, log_power=log_power, val=val)
            for s, ini in self.shift.iteritems()
            for log_power, val in enumerate(ini))

    def is_valid_for(self, dop):
        ind = dop.indicial_polynomial(dop.base_ring().gen())
        for sl_factor, shifts in my_shiftless_decomposition(ind):
            for k, (val_shift, _) in enumerate(shifts):
                if sl_factor(self.expo - val_shift).is_zero():
                    if len(self.shift) != len(shifts) - k:
                        return False
                    for shift, mult in shifts[k:]:
                        if len(self.shift.get(shift - val_shift, ())) != mult:
                            return False
                    return True
        return False

    def is_real(self, dop):
        r"""
        Try to detect cases where the coefficients of the series will be real.

        TESTS::

            sage: from ore_algebra import *
            sage: Dops, x, Dx = DifferentialOperators()
            sage: i = QuadraticField(-1, 'i').gen()
            sage: (x^2*Dx^2 + x*Dx + 1).numerical_transition_matrix([0, 1/2])
            [ [0.769238901363972...] + [0.638961276313634...]*I [0.769238901363972...] + [-0.6389612763136...]*I]
            sage: (Dx-i).numerical_transition_matrix([0,1])
            [[0.540302305868139...] + [0.841470984807896...]*I]
        """
        # We check that the exponent is real to ensure that the coefficients
        # will stay real. Note however that we don't need to make sure that
        # pt^expo*log(z)^k is real.
        return (utilities.is_real_parent(dop.base_ring().base_ring())
                and utilities.is_real_parent(self.universe)
                and self.expo.imag().is_zero())

    def is_precise(self, eps):
        if self.universe.is_exact():
            return True
        elif isinstance(self.universe, (RealBallField, ComplexBallField)):
            return all(safe_lt(bounds.IR(x.rad()), eps)
                       for val in self.shift.itervalues()
                       for x in val)
        else:
            return False

class PrecisionError(Exception):

    def __init__(self, suggested_new_prec):
        self.suggested_new_prec = suggested_new_prec

def series_sum(dop, ini, pt, tgt_error, maj=None, bwrec=None,
        stride=50, record_bounds_in=None, max_prec=100000):
    r"""
    EXAMPLES::

        sage: from sage.rings.real_arb import RealBallField, RBF
        sage: from sage.rings.complex_arb import ComplexBallField, CBF
        sage: QQi.<i> = QuadraticField(-1)

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.naive_sum import series_sum, EvaluationPoint
        sage: Dops, x, Dx = DifferentialOperators()

        sage: dop = ((4*x^2 + 3/58*x - 8)*Dx^10 + (2*x^2 - 2*x)*Dx^9 +
        ....:       (x^2 - 1)*Dx^8 + (6*x^2 - 1/2*x + 4)*Dx^7 +
        ....:       (3/2*x^2 + 2/5*x + 1)*Dx^6 + (-1/6*x^2 + x)*Dx^5 +
        ....:       (-1/5*x^2 + 2*x - 1)*Dx^4 + (8*x^2 + x)*Dx^3 +
        ....:       (-1/5*x^2 + 9/5*x + 5/2)*Dx^2 + (7/30*x - 12)*Dx +
        ....:       8/7*x^2 - x - 2)
        sage: ini = [CBF(-1/16, -2), CBF(-17/2, -1/2), CBF(-1, 1), CBF(5/2, 0),
        ....:       CBF(1, 3/29), CBF(-1/2, -2), CBF(0, 0), CBF(80, -30),
        ....:       CBF(1, -5), CBF(-1/2, 11)]

    Funny: on the following example, both the evaluation point and most of the
    initial values are exact, so that we end up with a significantly better
    approximation than requested::

        sage: series_sum(dop, ini, 1/2, RBF(1e-16))
        ([-3.575140703474456...] + [-2.2884877202396862...]*I)

        sage: import logging; logging.basicConfig()
        sage: series_sum(dop, ini, 1/2, RBF(1e-30))
        WARNING:ore_algebra.analytic.naive_sum:input intervals too wide for
        requested accuracy
        ...
        ([-3.5751407034...] + [-2.2884877202...]*I)

    In normal usage ``pt`` should be an object coercible to a complex ball or an
    :class:`EvaluationPoint` that wraps such an object. Polynomial with ball
    coefficients (wrapped in EvaluationPoints) are also supported to some extent
    (essentially, this is intended for use with polynomial indeterminates, and
    anything else that works does so by accident). ::

        sage: from ore_algebra.analytic.accuracy import AbsoluteError
        sage: series_sum(Dx - 1, [RBF(1)],
        ....:         EvaluationPoint(x.change_ring(RBF), rad=RBF(1), jet_order=2),
        ....:         AbsoluteError(1e-3), stride=1)
        (... + [0.0083...]*x^5 + [0.0416...]*x^4 + [0.1666...]*x^3
        + 0.5000...*x^2 + x + [1.0000 +/- ...e-5],
        ... + [0.0083...]*x^5 + [0.0416...]*x^4 + [0.1666...]*x^3
        + [0.5000...]*x^2 + x + [1.0000 +/- ...e-5])

    TESTS::

        sage: b = series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [RBF(0), RBF(1)],
        ....:                         7/10, RBF(1e-30))
        sage: b.parent()
        Vector space of dimension 1 over Real ball field with ... precision
        sage: b[0].rad().exact_rational() < 10^(-30)
        True
        sage: b[0].overlaps(RealBallField(130)(7/10).arctan())
        True

        sage: b = series_sum((x^2 + 1)*Dx^2 + 2*x*Dx, [CBF(0), CBF(1)],
        ....:                         (i+1)/2, RBF(1e-30))
        sage: b.parent()
        Vector space of dimension 1 over Complex ball field with ... precision
        sage: b[0].overlaps(ComplexBallField(130)((1+i)/2).arctan())
        True

        sage: series_sum(x*Dx^2 + Dx + x, [0], 1/2, 1e-10)
        Traceback (most recent call last):
        ...
        ValueError: invalid initial data for x*Dx^2 + Dx + x at 0
    """

    # The code that depends neither on the numeric precision nor on the
    # ordinary/regsing dichotomy goes here.

    if not isinstance(ini, LogSeriesInitialValues):
        ini = LogSeriesInitialValues(ZZ.zero(), ini, dop)

    if not isinstance(pt, EvaluationPoint):
        pt = EvaluationPoint(pt)

    if isinstance(tgt_error, accuracy.RelativeError) and pt.jet_order > 1:
        raise TypeError("relative error not supported when computing derivatives")
    if not isinstance(tgt_error, accuracy.StoppingCriterion):
        input_is_precise = pt.is_precise(tgt_error) and ini.is_precise(tgt_error)
        if not input_is_precise:
            logger.warn("input intervals too wide for requested accuracy")
        tgt_error = accuracy.AbsoluteError(tgt_error, input_is_precise)
    logger.log(logging.INFO - 1, "target error = %s", tgt_error)

    if maj is None:
        maj = bounds.DiffOpBound(dop, ini.expo,
                [] if dop.leading_coefficient().valuation() == 0
                else [(s, len(v)) for s, v in ini.shift.iteritems()])

    if bwrec is None:
        bwrec = backward_rec(dop, shift=ini.expo)

    ivs = (RealBallField
           if ini.is_real(dop) and (pt.is_real() or not pt.is_numeric)
           else ComplexBallField)
    doit = (series_sum_ordinary if dop.leading_coefficient().valuation() == 0
            else series_sum_regular)

    # Now do the actual computation, automatically increasing the precision as
    # necessary

    bit_prec = utilities.prec_from_eps(tgt_error.eps)
    # Roughly speaking, the computation of a new coefficient of the series
    # *multiplies* the diameter by the order of the recurrence, so it is not
    # unreasonable that the loss of precision is of the order of
    # log2(ordrec^nterms)... but this observation is far from explaining
    # everything; in particular, it completely ignores the size of the
    # coefficients of the recurrence. Anyhow, this formula seems to work well in
    # practice.
    bit_prec *= 1 + ZZ(bwrec.order - 2).nbits()
    logger.info("initial precision = %s bits", bit_prec)
    while True:
        try:
            psum = doit(ivs(bit_prec), dop, bwrec, ini, pt,
                    tgt_error, maj, stride, record_bounds_in)
            return psum
        except PrecisionError as err:
            if err.suggested_new_prec > max_prec:
                logger.info("lost too much precision, giving up")
                raise
            bit_prec = err.suggested_new_prec
            logger.info("lost too much precision, restarting with %d bits",
                        bit_prec)

################################################################################
# Ordinary points
################################################################################

def series_sum_ordinary(Intervals, dop, bwrec, ini, pt,
        tgt_error, maj, stride, record_bounds_in):

    if record_bounds_in:
        record_bounds_in[:] = []

    jet = pt.jet(Intervals).lift()
    Jets = jet.parent() # polynomial ring!
    ord = pt.jet_order
    jetpow = Jets.one()
    radpow = bounds.IR.one()

    ordrec = bwrec.order
    assert ini.expo.is_zero()
    last = collections.deque([Intervals.zero()]*(ordrec - dop.order() + 1))
    last.extend(Intervals(ini.shift[n][0])
                for n in xrange(dop.order() - 1, -1, -1))
    assert len(last) == ordrec + 1 # not ordrec!
    psum = Jets.zero()

    tail_bound = bounds.IR(infinity)
    def check_convergence(prev_tail_bound):
        # last[-1] since last[0] may still be "undefined" and last[1] may
        # not exist in degenerate cases
        est = (max([abs(a) for a in last])*radpow).above_abs()
        if pt.is_numeric:
            abs_sum = abs(psum[0])
            width = abs_sum.rad_as_ball()
        else:
            abs_sum = None
            width = max([abs(a).rad_as_ball() for a in last])*radpow
        ivs_too_wide = False
        if width > tgt_error.eps:
            if tgt_error.precise:
                raise PrecisionError(2*Intervals.precision())
            else:
                ivs_too_wide = True
        if (not tgt_error.reached(est, abs_sum) and record_bounds_in is None
                and not ivs_too_wide):
            return False, bounds.IR(infinity)
        # Warning: this residual must correspond to the operator stored in
        # maj.dop, which typically isn't the operator series_sum was called on
        # (but its to_T(), i.e. its product by a power of x).
        residual = bounds.residual(n, bwrec_nplus, list(last)[1:],
                                                       maj.Poly.variable_name())
        majeqrhs = maj.maj_eq_rhs([residual])
        for i in xrange(5):
            tail_bound = maj.matrix_sol_tail_bound(n, pt.rad, majeqrhs, ord)
            logger.debug("n=%s, i=%s, sum=%s, est=%s, rhs[.]=%s, tail_bound=%s",
                            n, i, psum[0], est, majeqrhs[0], tail_bound)
            if record_bounds_in is not None:
                record_bounds_in.append((n, psum, tail_bound))
            if tgt_error.reached(tail_bound, abs_sum):
                return True, tail_bound
            elif ivs_too_wide:
                if width > tail_bound:
                    return True, tail_bound
            elif (i == 1 and tail_bound.is_finite()
                    and not tail_bound <= prev_tail_bound.above_abs()):
                raise PrecisionError(2*Intervals.precision())
            elif not tgt_error.reached(tail_bound*
                    est**(QQ((maj._effort**2 + 2)*stride)/n)):
                maj.refine()
                continue
            break
        return False, tail_bound

    start = dop.order()
    # Evaluate the coefficients a bit in advance as we are going to need them to
    # compute the residuals. This is not ideal at high working precision, but
    # already saves a lot of time compared to doing the evaluations twice.
    bwrec_nplus = collections.deque(
            (bwrec.eval_int_ball(Intervals, start+i) for i in xrange(ordrec)),
            maxlen=ordrec)
    for n in range(start): # Initial values (“singular part”)
        last.rotate(1)
        term = Jets(last[0])._mul_trunc_(jetpow, ord)
        psum += term
        jetpow = jetpow._mul_trunc_(jet, ord)
        radpow *= pt.rad
    for n in itertools.count(start):
        last.rotate(1)
        #last[0] = None
        # At this point last[0] should be considered undefined (it will hold
        # the coefficient of z^n later in the loop body) and last[1], ...
        # last[ordrec] are the coefficients of z^(n-1), ..., z^(n-ordrec)
        if n%stride == 0:
            done, tail_bound = check_convergence(tail_bound)
            if done: break
        bwrec_n = (bwrec_nplus[0] if bwrec_nplus
                   else bwrec.eval_int_ball(Intervals, n))
        comb = sum(bwrec_n[k]*last[k] for k in xrange(1, ordrec+1))
        last[0] = -~bwrec_n[0]*comb
        # logger.debug("n = %s, [c(n), c(n-1), ...] = %s", n, list(last))
        term = Jets(last[0])._mul_trunc_(jetpow, ord)
        psum += term
        jetpow = jetpow._mul_trunc_(jet, ord)
        radpow *= pt.rad
        bwrec_nplus.append(bwrec.eval_int_ball(Intervals, n+bwrec.order))
    # Account for the dropped high-order terms in the intervals we return
    # (tail_bound is actually a bound on the Frobenius norm of the error matrix,
    # so there is some overestimation). WARNING: For symbolic x, the resulting
    # polynomials have to be interpreted with some care: in particular, it would
    # be incorrect to evaluate a polynomial result with real coefficients at a
    # complex point. Our current mechanism to choose whether to add a real or
    # complex error bound in this case is pretty fragile.
    tail_bound = tail_bound.abs()
    res = vector(_add_error(psum[i], tail_bound) for i in xrange(ord))
    logger.info("summed %d terms, tail <= %s, coeffwise error <= %s", n,
            tail_bound,
            max(x.rad() for x in res) if pt.is_numeric else "n/a")
    return res

# XXX: pass ctx (→ real/complex?)?
def fundamental_matrix_ordinary(dop, pt, eps, rows, maj, max_prec=None):
    eps_col = bounds.IR(eps)/bounds.IR(dop.order()).sqrt()
    evpt = EvaluationPoint(pt, jet_order=rows)
    inis = [
        LogSeriesInitialValues(ZZ.zero(), ini, dop, check=False)
        for ini in identity_matrix(dop.order())]
    cols = [
        series_sum(dop, ini, evpt, eps_col, maj=maj, max_prec=max_prec)
        for ini in inis]
    return matrix(cols).transpose()

################################################################################
# Regular singular points
################################################################################

# TODO: move parts not specific to the naïve summation algo elsewhere
def fundamental_matrix_regular(dop, pt, eps, rows):
    r"""
    TESTS::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.naive_sum import *
        sage: Dops, x, Dx = DifferentialOperators()

        sage: fundamental_matrix_regular(x*Dx^2 + (1-x)*Dx, 1, RBF(1e-10), 2)
        [[1.317902...] 1.000000...]
        [[2.718281...]           0]

        sage: dop = (x+1)*(x^2+1)*Dx^3-(x-1)*(x^2-3)*Dx^2-2*(x^2+2*x-1)*Dx
        sage: fundamental_matrix_regular(dop, 1/3, RBF(1e-10), 3)
        [ 1.0000000...  [0.321750554...]  [0.147723741...]]
        [            0  [0.900000000...]  [0.991224850...]]
        [            0  [-0.27000000...]  [1.935612425...]]

        sage: dop = (
        ....:     (2*x^6 - x^5 - 3*x^4 - x^3 + x^2)*Dx^4
        ....:     + (-2*x^6 + 5*x^5 - 11*x^3 - 6*x^2 + 6*x)*Dx^3
        ....:     + (2*x^6 - 3*x^5 - 6*x^4 + 7*x^3 + 8*x^2 - 6*x + 6)*Dx^2
        ....:     + (-2*x^6 + 3*x^5 + 5*x^4 - 2*x^3 - 9*x^2 + 9*x)*Dx)
        sage: fundamental_matrix_regular(dop, RBF(1/3), RBF(1e-10), 4)
        [ [3.1788470...] [-1.064032...]  [1.000...] [0.3287250...]]
        [ [-8.981931...] [3.2281834...]    [+/-...] [0.9586537...]]
        [  [26.18828...] [-4.063756...]    [+/-...] [-0.123080...]]
        [ [-80.24671...]  [9.190740...]    [+/-...] [-0.119259...]]

        sage: dop = x*Dx^3 + 2*Dx^2 + x*Dx
        sage: ini = [1, CBF(euler_gamma), 0]
        sage: dop.numerical_solution(ini, [0, RBF(1/3)], 1e-14)
        [-0.549046117782...]
    """
    evpt = EvaluationPoint(pt, jet_order=rows)
    eps_col = bounds.IR(eps)/bounds.IR(dop.order()).sqrt()
    col_tgt_error = accuracy.AbsoluteError(eps_col, evpt.is_precise(eps))
    bwrec = backward_rec(dop)
    ind = bwrec[0]
    n = ind.parent().gen()
    sl_decomp = my_shiftless_decomposition(ind)
    logger.debug("indicial polynomial = %s ~~> %s", ind, sl_decomp)

    cols = []
    for sl_factor, shifts in sl_decomp:
        for irred_factor, irred_mult in sl_factor.factor():
            assert irred_mult == 1
            # Complicated to do here and specialize, for little benefit
            #irred_nf = irred_factor.root_field("leftmost")
            #irred_leftmost = irred_nf.gen()
            #irred_bwrec = [pol(irred_leftmost + n) for pol in bwrec]
            for leftmost, _ in irred_factor.roots(QQbar):
                leftmost = utilities.as_embedded_number_field_element(leftmost)
                emb_bwrec = bwrec.shift(leftmost)
                maj = bounds.DiffOpBound(dop, leftmost, shifts)
                for shift, mult in shifts:
                    for log_power in xrange(mult):
                        logger.info("solution z^(%s+%s)·log(z)^%s/%s! + ···",
                                    leftmost, shift, log_power, log_power)
                        ini = LogSeriesInitialValues(
                            dop = dop,
                            expo = leftmost,
                            values = {s: tuple(ZZ.one()
                                               if (s, p) == (shift, log_power)
                                               else ZZ.zero()
                                               for p in xrange(m))
                                      for s, m in shifts},
                            check = False)
                        # XXX: inefficient if shift >> 0
                        value = series_sum(dop, ini, evpt, col_tgt_error,
                                maj=maj, bwrec=emb_bwrec)
                        sol = FundamentalSolution(
                            valuation = leftmost + shift,
                            log_power = log_power,
                            value = value)
                        logger.debug("sol=%s\n\n", sol)
                        cols.append(sol)
    cols.sort(key=sort_key_by_asympt)
    return matrix([sol.value for sol in cols]).transpose()

def _pow_trunc(a, n, ord):
    pow = a.parent().one()
    pow2k = a
    while n:
        if n & 1:
            pow = pow._mul_trunc_(pow2k, ord)
        pow2k = pow2k._mul_trunc_(pow2k, ord)
        n = n >> 1
    return pow

def log_series_value(Jets, derivatives, expo, psum, pt):
    log_prec = psum.length()
    if log_prec > 1 or expo not in ZZ:
        pt = pt.parent().complex_field()(pt)
        Jets = Jets.change_ring(Jets.base_ring().complex_field())
        psum = psum.change_ring(Jets)
    # hardcoded series expansions of log(pt) = log(a+η) and pt^λ = (a+η)^λ (too
    # cumbersome to compute directly in Sage at the moment)
    high = Jets([0] + [(-1)**(k+1)*~pt**k/k
                       for k in xrange(1, derivatives)])
    logpt = Jets([pt.log()]) + high
    logger.debug("logpt=%s", logpt)
    aux = high*expo
    logger.debug("aux=%s", aux)
    inipow = pt**expo*sum(_pow_trunc(aux, k, derivatives)/Integer(k).factorial()
                          for k in xrange(derivatives))
    logger.debug("inipow=%s", inipow)
    val = inipow.multiplication_trunc(
            sum(psum[p]._mul_trunc_(_pow_trunc(logpt, p, derivatives), derivatives)
                        /Integer(p).factorial()
                for p in xrange(log_prec)),
            derivatives)
    return val

# This function only handles the case of a “single” series, i.e. a series where
# all indices differ from each other by integers. But since we need logic to go
# past singular indices anyway, we can allow for general initial conditions (at
# roots of the indicial equation belonging to the same shift-equivalence class),
# not just initial conditions associated to canonical solutions.
def series_sum_regular(Intervals, dop, bwrec, ini, pt, tgt_error,
        maj, stride=50, record_bounds_in=None):
    r"""
    TESTS::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.naive_sum import *
        sage: Dops, x, Dx = DifferentialOperators()

    Test that we don't compute the zero solution when the valuation is large.
    TODO: In the presence of several solutions with very different valuations,
    we used to be able to stop before reaching the largest one if the initial
    values there were zero. Unfortunately, this is no longer the case with the
    simplified bounds on rational sequences::

        sage: #dop = (Dx-1).lclm(x*Dx-1000)
        sage: dop = (x^2-1000*x)*Dx^2 + (-x^2+999000)*Dx + 1000*x - 999000
        sage: logger = logging.getLogger('ore_algebra.analytic.naive_sum')
        sage: logger.setLevel(logging.INFO) # TBI
        sage: dop.numerical_transition_matrix([0,1]) # not tested - TODO
        INFO:ore_algebra.analytic.naive_sum:solution z^(0+0)·log(z)^0/0! + ···
        INFO:ore_algebra.analytic.naive_sum:summed 50 terms, tail <= ...
        ...
        [[2.7182818284590...] 1.0000000000000000]
        [[2.7182818284590...] 1000.0000000000000]
        sage: logger.setLevel(logging.WARNING)
        sage: series_sum(dop, {0: (1,), 1000: (1/1000,)}, 1, 1e-10)
        ([2.719281828459...])

    Some simple tests involving large non-integer valuations::

        sage: dop = (x*Dx-1001/2).symmetric_product(Dx-1)
        sage: dop = dop._normalize_base_ring()[-1]
        sage: (exp(CBF(1/2))/RBF(2)^(1001/2)).overlaps(dop.numerical_transition_matrix([0, 1/2], 1e-10)[0,0])
        True
        sage: (exp(CBF(2))/RBF(1/2)^(1001/2)).overlaps(dop.numerical_transition_matrix([0, 2], 1e-10)[0,0])
        True

        sage: dop = (x*Dx+1001/2).symmetric_product(Dx-1)
        sage: dop = dop._normalize_base_ring()[-1]
        sage: (CBF(1/2)^(-1001/2)*exp(CBF(1/2))).overlaps(dop.numerical_transition_matrix([0, 1/2], 1e-10)[0,0])
        True
        sage: (CBF(2)^(-1001/2)*exp(CBF(2))).overlaps(dop.numerical_transition_matrix([0, 2], 1e-10)[0,0])
        True

        sage: h = CBF(1/2)
        sage: #dop = (Dx-1).lclm(x^2*Dx^2 - x*(2*x+1999)*Dx + (x^2 + 1999*x + 1000^2))
        sage: dop = x^2*Dx^3 + (-3*x^2 - 1997*x)*Dx^2 + (3*x^2 + 3994*x + 998001)*Dx - x^2 - 1997*x - 998001
        sage: mat = dop.numerical_transition_matrix([0,1/2], 1e-5) # XXX: long time with the simplified bounds on rational functions
        sage: mat[0,0].overlaps(exp(h)) # long time
        True
        sage: mat[0,1].overlaps(exp(h)*h^1000*log(h)) # long time
        True
        sage: mat[0,2].overlaps(exp(h)*h^1000) # long time
        True

        sage: dop = (x^3 + x^2)*Dx^3 + (-1994*x^2 - 1997*x)*Dx^2 + (994007*x + 998001)*Dx + 998001
        sage: mat = dop.numerical_transition_matrix([0, 1/2], 1e-5)
        sage: mat[0,0].overlaps(1/(1+h))
        True
        sage: mat[0,1].overlaps(h^1000/(1+h)*log(h))
        True
        sage: mat[0,2].overlaps(h^1000/(1+h))
        True

    """

    jet = pt.jet(Intervals).lift()
    Jets = jet.parent()
    ord = pt.jet_order
    jetpow = Jets.one()
    radpow = bounds.IR.one() # bound on abs(pt)^n in the series part (=> starts
                             # at 1 regardless of ini.expo)

    log_prec = sum(len(v) for v in ini.shift.itervalues())
    last_special_index = max(ini.shift)
    last_index_with_ini = max([dop.order()]
            + [s for s, vals in ini.shift.iteritems()
                 if not all(v.is_zero() for v in vals)])
    last = collections.deque([vector(Intervals, log_prec)
                              for _ in xrange(bwrec.order + 1)])
    psum = vector(Jets, log_prec)

    # Every few iterations, heuristically check if we have converged and if
    # we still have enough precision. If it looks like the target error may
    # be reached, perform a rigorous check. Our stopping criterion currently
    # (1) only works at “generic” indices, and (2) assumes that the initial
    # values at exceptional indices larger than n are zero, so we also
    # ensure that we are in this case. (Both assumptions could be lifted,
    # (1) by using a slightly more complicated formula for the tail bound,
    # and (2) if we had code to compute lower bounds on coefficients of
    # series expansions of majorants.)
    tail_bound = bounds.IR(infinity)
    def check_convergence(prev_tail_bound):
        if n <= last_index_with_ini or mult > 0:
            return False, bounds.IR(infinity)
        est = max(abs(a) for log_jet in last for a in log_jet)*radpow.above_abs()
        if tgt_error.precise and any(abs(iv).rad_as_ball() > tgt_error.eps
                                     for log_jet in psum for iv in log_jet):
            raise PrecisionError(2*Intervals.precision())
        sum_est = bounds.IR(abs(psum[0][0]))
        # TODO: improve the automatic increase of precision for large x^λ:
        # currently we only check the series part (which would sort of make
        # sense in a relative error setting)
        if not tgt_error.reached(est, sum_est) and record_bounds_in is None:
            return False, bounds.IR(infinity)
        majeqrhs = bounds.maj_eq_rhs_with_logs(n, bwrec, bwrec_nplus,
                list(last)[1:], maj.Poly.variable_name(), log_prec)
        for i in xrange(5):
            tail_bound = maj.matrix_sol_tail_bound(n, pt.rad, majeqrhs,
                                                        ord=pt.jet_order)
            logger.debug("n=%d, sum[.]=%s, est=%s, rhs[.]=%s, tail_bound=%s",
                    n, psum[0][0], est, majeqrhs[0], tail_bound)
            if record_bounds_in is not None:
                # TODO: record all partial sums, not just [log(z)^0]
                # (requires improvements to plot_bounds)
                record_bounds_in.append((n, psum[0], tail_bound))
            if tgt_error.reached(tail_bound, sum_est):
                return True, tail_bound
            elif n < last_special_index: # some really bad bounds in this case
                break
            elif (i == 1 and tail_bound.is_finite()
                         and not tail_bound <= prev_tail_bound.above_abs()):
                # We likely lost all precision on the coefficients.
                raise PrecisionError(2*Intervals.precision())
            else:
                # We don't want to go too far beyond the optimal truncation to
                # improve tail_bound (we would lose too much precision), but
                # refining many times is really expensive.
                bound_est = tail_bound*est**(QQ((maj._effort**2 + 2)*stride)/n)
                if not tgt_error.reached(bound_est):
                    maj.refine()
                    continue
            break
        return False, tail_bound

    precomp_len = max(1, bwrec.order) # hack for recurrences of order zero
    bwrec_nplus = collections.deque(
            (bwrec.eval_series(Intervals, i, log_prec)
                for i in xrange(precomp_len)),
            maxlen=precomp_len)
    for n in itertools.count():
        last.rotate(1)
        logger.log(logging.DEBUG - 2, "n = %s, [c(n), c(n-1), ...] = %s", n, list(last))
        logger.log(logging.DEBUG - 1, "n = %s, sum = %s", n, psum)
        mult = len(ini.shift.get(n, ()))

        if n%stride == 0:
            done, tail_bound = check_convergence(tail_bound)
            if done: break

        for p in xrange(log_prec - mult - 1, -1, -1):
            combin  = sum(bwrec_nplus[0][i][j]*last[i][p+j]
                          for j in xrange(log_prec - p)
                          for i in xrange(bwrec.order, 0, -1))
            combin += sum(bwrec_nplus[0][0][j]*last[0][p+j]
                          for j in xrange(mult + 1, log_prec - p))
            last[0][mult + p] = - ~bwrec_nplus[0][0][mult] * combin
        for p in xrange(mult - 1, -1, -1):
            last[0][p] = ini.shift[n][p]
        psum += last[0]*jetpow
        jetpow = jetpow._mul_trunc_(jet, ord)
        radpow *= pt.rad
        bwrec_nplus.append(bwrec.eval_series(Intervals, n+precomp_len, log_prec))
    logger.info("summed %d terms, tail <= %s", n, tail_bound)

    # Add error terms accounting for the truncation (for each power of log and
    # each derivative), combine the series corresponding to each power of log,
    # return the vector of successive derivatives.
    tail_bound = tail_bound.abs()
    psum = vector(Jets, [[c.add_error(tail_bound) for c in t]
                         for t in psum])
    val = log_series_value(Jets, ord, ini.expo, psum, jet[0])
    return vector(val[i] for i in xrange(ord))

################################################################################
# Miscellaneous utilities
################################################################################

# Temporary: later on, polynomials with ball coefficients could implement
# add_error themselves.
def _add_error(approx, error):
    if isinstance(approx, polynomial_element.Polynomial):
        return approx[0].add_error(error) + ((approx >> 1) << 1)
    else:
        return approx.add_error(error)

def _random_ini(dop):
    import random
    from sage.all import VectorSpace, QQ
    ind = dop.indicial_polynomial(dop.base_ring().gen())
    sl_decomp = my_shiftless_decomposition(ind)
    pol, shifts = random.choice(sl_decomp)
    expo = random.choice(pol.roots(QQbar))[0]
    values = {
        shift: tuple(VectorSpace(QQ, mult).random_element(10))
        for shift, mult in shifts
    }
    return LogSeriesInitialValues(expo, values, dop)

def plot_bounds(dop, ini=None, pt=None, eps=None, **kwds):
    r"""
    EXAMPLES::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.naive_sum import *
        sage: Dops, x, Dx = DifferentialOperators()

        sage: plot_bounds(Dx - 1, [CBF(1)], CBF(i)/2, RBF(1e-20))
        Graphics object consisting of 5 graphics primitives

        sage: plot_bounds(x*Dx^3 + 2*Dx^2 + x*Dx, eps=1e-8)
        Graphics object consisting of 5 graphics primitives

        sage: dop = x*Dx^2 + Dx + x
        sage: plot_bounds(dop, eps=1e-8,
        ....:       ini=LogSeriesInitialValues(0, {0: (1, 0)}, dop))
        Graphics object consisting of 5 graphics primitives

        sage: dop = ((x^2 + 10*x + 50)*Dx^10 + (5/9*x^2 + 50/9*x + 155/9)*Dx^9
        ....: + (-10/3*x^2 - 100/3*x - 190/3)*Dx^8 + (30*x^2 + 300*x + 815)*Dx^7
        ....: + (145*x^2 + 1445*x + 3605)*Dx^6 + (5/2*x^2 + 25*x + 115/2)*Dx^5
        ....: + (20*x^2 + 395/2*x + 1975/4)*Dx^4 + (-5*x^2 - 50*x - 130)*Dx^3
        ....: + (5/4*x^2 + 25/2*x + 105/4)*Dx^2 + (-20*x^2 - 195*x - 480)*Dx
        ....: + 5*x - 10)
        sage: plot_bounds(dop, pol_part_len=2, bound_inverse="solve", eps=1e-10) # long time
        Graphics object consisting of 5 graphics primitives
    """
    import sage.plot.all as plot
    from sage.all import VectorSpace, QQ, RIF
    from ore_algebra.analytic.bounds import abs_min_nonzero_root
    if ini is None:
        ini = _random_ini(dop)
    if pt is None:
        rad = abs_min_nonzero_root(dop.leading_coefficient())
        pt = QQ(2) if rad == infinity else RIF(rad/2).simplest_rational()
    if eps is None:
        eps = RBF(1e-50)
    logger.info("point: %s", pt)
    logger.info("initial values: %s", ini)
    recd = []
    maj = bounds.DiffOpBound(dop, refinable=False, **kwds)
    series_sum(dop, ini, pt, eps, stride=1, record_bounds_in=recd, maj=maj)
    ref_sum = recd[-1][1][0].add_error(recd[-1][2])
    recd[-1:] = []
    # Note: this won't work well when the errors get close to the double
    # precision underflow threshold.
    error_plot_upper = plot.line(
            [(n, (psum[0]-ref_sum).abs().upper())
             for n, psum, _ in recd],
            color="lightgray", scale="semilogy")
    error_plot = plot.line(
            [(n, (psum[0]-ref_sum).abs().lower())
                for n, psum, _ in recd],
            color="black", scale="semilogy")
    bound_plot_lower = plot.line([(n, bound.lower()) for n, _, bound in recd],
                           color="lightblue", scale="semilogy")
    bound_plot = plot.line([(n, bound.upper()) for n, _, bound in recd],
                           color="blue", scale="semilogy")
    title = repr(dop) + " @ x=" + repr(pt)
    title = title if len(title) < 80 else title[:77]+"..."
    myplot = error_plot_upper + error_plot + bound_plot_lower + bound_plot
    ymax = myplot.ymax()
    if ymax < float('inf'):
        txt = plot.text(title, (myplot.xmax(), ymax),
                        horizontal_alignment='right', vertical_alignment='top')
        myplot += txt
    return myplot

