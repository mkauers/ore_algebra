# -*- coding: utf-8 - vim: tw=80
r"""
Local solutions
"""

import collections, logging

from sage.categories.pushout import pushout
from sage.functions.log import log as symbolic_log
from sage.misc.cachefunc import cached_method
from sage.modules.free_module_element import vector
from sage.rings.all import ZZ, QQ, QQbar, RBF, RealBallField, ComplexBallField
from sage.rings.number_field.number_field import NumberField_generic
from sage.rings.number_field.number_field_morphisms import NumberFieldEmbedding
from sage.rings.polynomial import polynomial_element
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.sequence import Sequence
from sage.symbolic.all import SR

from .. import ore_algebra
from . import utilities

from .shiftless import dispersion, my_shiftless_decomposition

logger = logging.getLogger(__name__)

##############################################################################
# Recurrence relations
##############################################################################

def bw_shift_rec(dop, shift=ZZ.zero()):
    Pols_n = PolynomialRing(dop.base_ring().base_ring(), 'n') # XXX: name
    Rops = ore_algebra.OreAlgebra(Pols_n, 'Sn')
    # Using the primitive part here would break the computation of residuals!
    # TODO: add test (arctan); better fix?
    # Other interesting cases: operators of the for P(Θ) (with constant
    # coefficients)
    #rop = dop.to_S(Rops).primitive_part().numerator()
    rop = dop.to_S(Rops)
    ordrec = rop.order()
    coeff = [rop[ordrec-k](Pols_n.gen()-ordrec+shift)
             for k in xrange(ordrec+1)]
    return BwShiftRec(coeff)

class BwShiftRec(object):
    r"""
    A recurrence relation, written in terms of the backward shift operator.

    This class is mainly intended to provide reasonably fast evaluation in the
    context of naïve unrolling.
    """

    def __init__(self, coeff):
        assert isinstance(coeff[0], polynomial_element.Polynomial)
        self.coeff = coeff
        self.base_ring = coeff[0].parent()
        self.Scalars = self.base_ring.base_ring()
        self.order = len(coeff) - 1
        # Evaluating polynomials over ℚ[i] is slow...
        # TODO: perhaps do something similar for eval_series
        if utilities.is_QQi(self.Scalars):
            QQn = PolynomialRing(QQ, 'n')
            self._re_im = [
                    (QQn([c.real() for c in pol]), QQn([c.imag() for c in pol]))
                    for pol in coeff]
            self.eval_int_ball = self._eval_qqi_cbf

    def __repr__(self):
        n = self.base_ring.variable_name()
        return " + ".join("({})*S{}^(-{})".format(c, n, j)
                          for j, c in enumerate(self.coeff))

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

    @cached_method
    def scalars_embedding(self, tgt):
        if isinstance(self.Scalars, NumberField_generic):
            # do complicated coercions via QQbar and CLF only once...
            Pol = PolynomialRing(tgt, 'x')
            x = tgt(self.Scalars.gen())
            return lambda elt: Pol([tgt(c) for c in elt._coefficients()])(x)
        else:
            return tgt

    def eval_series(self, tgt, point, ord):
        mor = self.scalars_embedding(tgt)
        point = self.Scalars(point)
        return [[mor(self._coeff_series(i,j)(point)) for j in xrange(ord)]
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
        return BwShiftRec([pol(sh + n) for pol in self.coeff])

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

    def accuracy(self):
        infinity = RBF.maximal_accuracy()
        if self.universe.is_exact():
            return infinity
        elif isinstance(self.universe, (RealBallField, ComplexBallField)):
            return min(infinity, *(x.accuracy()
                                   for val in self.shift.itervalues()
                                   for x in val))
        else:
            raise ValueError

def random_ini(dop):
    import random
    from sage.all import VectorSpace
    ind = dop.indicial_polynomial(dop.base_ring().gen())
    sl_decomp = my_shiftless_decomposition(ind)
    pol, shifts = random.choice(sl_decomp)
    expo = random.choice(pol.roots(QQbar))[0]
    values = {
        shift: tuple(VectorSpace(QQ, mult).random_element(10))
        for shift, mult in shifts
    }
    return LogSeriesInitialValues(expo, values, dop)

##############################################################################
# Structure of the local basis at a regular singular point
##############################################################################

_FundamentalSolution0 = collections.namedtuple(
    'FundamentalSolution',
    ['leftmost', 'shift', 'log_power', 'value'])

class FundamentalSolution(_FundamentalSolution0):
    @property
    @cached_method
    def valuation(self):
        return QQbar(self.leftmost + self.shift) # alg vs NFelt for re, im

def sort_key_by_asympt(sol):
    r"""
    Specify the sorting order for local solutions.

    Roughly speaking, they are sorted in decreasing order of asymptotic
    dominance: when two solutions are asymptotically comparable, the largest
    one as x → 0 comes first. In addition, divergent solutions, including
    things like `x^i`, always come before convergent ones.
    """
    re, im = sol.valuation.real(), sol.valuation.imag()
    return re, -sol.log_power, -im.abs(), im.sign()

class LocalBasisMapper(object):

    def run(self, dop):
        r"""
        Compute self.fun() for each element of the local basis at 0 of dop.

        Subclasses should define a fun() method that takes as input a
        LogSeriesInitialValues structure and can access the iteration variables
        as well as some derived quantities through the instance's field.
        Additionally, hooks are provided to share parts of the computation
        in a class of related solutions. The choice of unconditional
        computations, exported data and hooks is ad hoc.

        The output is a list of FundamentalSolution structures, sorted in the
        canonical order.
        """

        bwrec = bw_shift_rec(dop)
        ind = bwrec[0]
        sl_decomp = my_shiftless_decomposition(ind)
        logger.debug("indicial polynomial = %s ~~> %s", ind, sl_decomp)

        cols = []
        for self.sl_factor, self.shifts in sl_decomp:
            for self.irred_factor, irred_mult in self.sl_factor.factor():
                assert irred_mult == 1
                self.process_irred_factor()
                # Complicated to do here and specialize, for little benefit
                #irred_nf = irred_factor.root_field("self.leftmost")
                #irred_leftmost = irred_nf.gen()
                #irred_bwrec = [pol(irred_leftmost + n) for pol in bwrec]
                for leftmost, _ in self.irred_factor.roots(QQbar):
                    self.leftmost = utilities.as_embedded_number_field_element(leftmost)
                    self.emb_bwrec = bwrec.shift(self.leftmost)
                    self.process_modZ_class()
                    for self.shift, self.mult in self.shifts:
                        self.process_valuation()
                        for self.log_power in xrange(self.mult):
                            logger.info(r"solution z^(%s+%s)·log(z)^%s/%s! + ···",
                                        self.leftmost, self.shift,
                                        self.log_power, self.log_power)
                            ini = LogSeriesInitialValues(
                                dop = dop,
                                expo = self.leftmost,
                                values = {
                                    s: tuple(ZZ.one() if (s, p) == (self.shift,
                                                                 self.log_power)
                                             else ZZ.zero()
                                             for p in xrange(m))
                                    for s, m in self.shifts},
                                check = False)
                            # XXX: inefficient if self.shift >> 0
                            value = self.fun(ini)
                            sol = FundamentalSolution(
                                leftmost = self.leftmost,
                                shift = ZZ(self.shift),
                                log_power = ZZ(self.log_power),
                                value = value)
                            logger.debug("sol=%s\n\n", sol)
                            cols.append(sol)
        cols.sort(key=sort_key_by_asympt)
        return cols

    def process_irred_factor(self):
        pass

    def process_modZ_class(self):
        pass

    def process_valuation(self):
        pass

    def fun(self, ini):
        return None

def exponent_shifts(dop, leftmost):
    bwrec = bw_shift_rec(dop)
    ind = bwrec[0]
    sl_decomp = my_shiftless_decomposition(ind)
    cand = [shifts for fac, shifts in sl_decomp if fac(leftmost).is_zero()]
    assert len(cand) == 1
    shifts = [s for s in cand[0] if s >= 0]
    assert shifts[0][0] == 0
    return shifts

def log_series(ini, bwrec, order):
    Coeffs = pushout(bwrec.base_ring.base_ring(), ini.universe)
    log_prec = sum(len(v) for v in ini.shift.itervalues())
    precomp_len = max(1, bwrec.order) # hack for recurrences of order zero
    bwrec_nplus = collections.deque(
            (bwrec.eval_series(Coeffs, i, log_prec)
                for i in xrange(precomp_len)),
            maxlen=precomp_len)
    series = []
    for n in xrange(order):
        new_term = vector(Coeffs, log_prec)
        mult = len(ini.shift.get(n, ()))
        for p in xrange(log_prec - mult - 1, -1, -1):
            combin  = sum(bwrec_nplus[0][i][j]*series[-i][p+j]
                          for j in xrange(log_prec - p)
                          for i in xrange(min(bwrec.order, n), 0, -1))
            combin += sum(bwrec_nplus[0][0][j]*new_term[p+j]
                          for j in xrange(mult + 1, log_prec - p))
            new_term[mult + p] = - ~bwrec_nplus[0][0][mult] * combin
        for p in xrange(mult - 1, -1, -1):
            new_term[p] = ini.shift[n][p]
        series.append(new_term)
        bwrec_nplus.append(bwrec.eval_series(Coeffs, n+precomp_len, log_prec))
    return series
