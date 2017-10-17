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
from sage.structure.element import get_coercion_model
from sage.structure.formal_sum import FormalSum, FormalSums
from sage.structure.sequence import Sequence
from sage.symbolic.all import SR

from .. import ore_algebra
from . import utilities

from .shiftless import dispersion, my_shiftless_decomposition

logger = logging.getLogger(__name__)

##############################################################################
# Recurrence relations
##############################################################################

def backward_rec(dop, shift=ZZ.zero()):
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
            return NumberFieldEmbedding(self.Scalars, tgt,
                                        tgt(self.Scalars.gen()))
        else:
            return tgt

    def eval_series(self, tgt, point, ord):
        mor = self.scalars_embedding(tgt)
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
        return BackwardRec([pol(sh + n) for pol in self.coeff])

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

def map_local_basis(dop, fun, modZ_class_aux):
    r"""
    Compute fun(ini, bwrec, data) for each element of the local basis at 0
    of dop, where data is computed as modZ_class_aux(leftmost, shift) for each
    “group” (class modulo ℤ) of roots the indicial polynomial. Somewhat ad hoc.
    """
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
                modZ_class_data = modZ_class_aux(leftmost, shifts)
                for shift, mult in shifts:
                    for log_power in xrange(mult):
                        logger.info(r"solution z^(%s+%s)·log(z)^%s/%s! + ···",
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
                        value = fun(ini, emb_bwrec, **modZ_class_data)
                        sol = FundamentalSolution(
                            leftmost = leftmost,
                            shift = ZZ(shift),
                            log_power = ZZ(log_power),
                            value = value)
                        logger.debug("sol=%s\n\n", sol)
                        cols.append(sol)
    cols.sort(key=sort_key_by_asympt)
    return cols

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

# TODO: Implement (as a method of differential operators) a version that returns
# a list of DFiniteFunction objects

def local_basis(dop, point, order=None, ring=None):
    r"""
    Generalized series expansions the local basis.

    INPUT:

    * dop - Differential operator

    * point - Point where the local basis is to be computed

    * order (optional) - Number of terms to compute, **starting from each
      “leftmost” valuation of a group of solutions with valuations differing by
      integers**. (Thus, the absolute truncation order will be the same for all
      solutions in such a group, with some solutions having more actual
      coefficients computed that others.)

      The default is to choose the truncation order in such a way that the
      structure of the basis is apparent, and in particular that logarithmic
      terms appear if logarithms are involved at all in that basis. The
      corresponding order may be very large in some cases.

    * ring (optional) - Ring in which to coerce the coefficients of the
      expansion

    EXAMPLES::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.local_solutions import local_basis
        sage: Dops, x, Dx = DifferentialOperators(QQ, 'x')

        sage: local_basis(Dx - 1, 0)
        [1 + x + 1/2*x^2 + 1/6*x^3]

        sage: from ore_algebra.analytic.examples import ssw
        sage: local_basis(ssw.dop3, 0)
        [t^(-4) + 24*log(t)/t^2 - 48*log(t) - 96*t^2*log(t) - 88*t^2,
         t^(-2),
         1 + 2*t^2]

        sage: dop = (x^2*(x^2-34*x+1)*Dx^3 + 3*x*(2*x^2-51*x+1)*Dx^2
        ....:     + (7*x^2-112*x+1)*Dx + (x-5))
        sage: local_basis(dop, 0, order=3)
        [1/2*log(x)^2 + 5/2*x*log(x)^2 + 12*x*log(x) + 73/2*x^2*log(x)^2
         + 210*x^2*log(x) + 72*x^2,
         log(x) + 5*x*log(x) + 12*x + 73*x^2*log(x) + 210*x^2,
         1 + 5*x + 73*x^2]

        sage: roots = dop.leading_coefficient().roots(AA)
        sage: basis = local_basis(dop, roots[1][0], order=3)
        sage: basis
        [1 - (-239/12*a+169/6)*(x - 0.02943725152285942?)^2,
         sqrt(x - 0.02943725152285942?)
         - (-203/32*a+9)*(x - 0.02943725152285942?)^(3/2)
         + (-24031/160*a+1087523/5120)*(x - 0.02943725152285942?)^(5/2),
         x - 0.02943725152285942? - (-55/6*a+13)*(x - 0.02943725152285942?)^2]
        sage: basis[0].base_ring()
        Number Field in a with defining polynomial y^2 - 2
        sage: RR(basis[0].base_ring().gen())
        -1.41421356237309
        sage: basis[0][1]
        (239/12*a - 169/6, (x - 0.02943725152285942?)^2)

        sage: local_basis(dop, roots[1][0], order=3, ring=QQbar)
        [1 - 56.33308678393081?*(x - 0.02943725152285942?)^2,
         sqrt(x - 0.02943725152285942?)
         - 17.97141728630432?*(x - 0.02943725152285942?)^(3/2)
         + 424.8128741711741?*(x - 0.02943725152285942?)^(5/2),
         x - 0.02943725152285942?
         - 25.96362432175337?*(x - 0.02943725152285942?)^2]

    TESTS::

        sage: local_basis(4*x^2*Dx^2 + (-x^2+8*x-11), 0, 2)
        [x^(-sqrt(3) + 1/2) + (-4/11*a+2/11)*x^(-sqrt(3) + 3/2),
         x^(sqrt(3) + 1/2) - (-4/11*a-2/11)*x^(sqrt(3) + 3/2)]

        sage: local_basis((27*x^2+4*x)*Dx^2 + (54*x+6)*Dx + 6, 0, 2)
        [1/sqrt(x) + 3/8*sqrt(x), 1 - x]

    """
    from .path import Point
    mypoint = Point(point, dop)
    ldop = mypoint.local_diffop()
    if order is None:
        ind = ldop.indicial_polynomial(ldop.base_ring().gen())
        order = max(dop.order(), ind.dispersion()) + 3
    sols = map_local_basis(ldop,
            lambda ini, bwrec: log_series(ini, bwrec, order),
            lambda leftmost, shift: {})
    x = SR.var(dop.base_ring().variable_name())
    dx = x if point.is_zero() else x.add(-point, hold=True)
    # Working with symbolic expressions here is too complicated: let's try
    # returning FormalSums.
    def log_monomial(expo, n, k):
        expo = simplify_exponent(expo)
        return dx**(expo + n) * symbolic_log(dx, hold=True)**k
    if ring is None:
        cm = get_coercion_model()
        ring = cm.common_parent(
                dop.base_ring().base_ring(),
                mypoint.value.parent(),
                *(sol.leftmost for sol in sols))
    res = [FormalSum(
                [(c/ZZ(k).factorial(), log_monomial(sol.leftmost, n, k))
                    for n, vec in enumerate(sol.value)
                    for k, c in reversed(list(enumerate(vec)))],
                FormalSums(ring))
           for sol in sols]
    return res

def simplify_exponent(e): # work around sage bug #21758
    try:
        return ZZ(e)
    except (TypeError, ValueError):
        return e
