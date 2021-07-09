# -*- coding: utf-8 - vim: tw=80
r"""
Local solutions
"""

# Copyright 2016, 2017, 2018, 2019 Marc Mezzarobba
# Copyright 2016, 2017, 2018, 2019 Centre national de la recherche scientifique
# Copyright 2016, 2017, 2018 Université Pierre et Marie Curie
# Copyright 2019 Sorbonne Université
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

from six.moves import range

import collections, logging, warnings

from itertools import chain

import sage.functions.log as symbolic_log

from sage.arith.all import gcd, lcm
from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute
from sage.modules.free_module_element import vector, FreeModuleElement_generic_dense
from sage.rings.all import ZZ, QQ, AA, QQbar, RBF
from sage.rings.all import RealBallField, ComplexBallField
from sage.rings.complex_arb import ComplexBall
from sage.rings.integer import Integer
from sage.rings.number_field.number_field import (
        NumberField_absolute,
        NumberField_quadratic,
    )
from sage.rings.polynomial import polynomial_element
from sage.rings.polynomial.complex_roots import complex_roots
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.sequence import Sequence
from sage.symbolic.all import SR, pi
from sage.symbolic.constants import I

from .. import ore_algebra
from . import utilities

from .accuracy import IC
from .differential_operator import DifferentialOperator
from .shiftless import dispersion, my_shiftless_decomposition

logger = logging.getLogger(__name__)

##############################################################################
# Recurrence relations
##############################################################################

def bw_shift_rec(dop, shift=ZZ.zero()):
    Scalars = utilities.mypushout(dop.base_ring().base_ring(), shift.parent())
    if dop.parent().is_D():
        dop = DifferentialOperator(dop) # compatibility bugware
        rop = dop._my_to_S()
    else: # more compatibility bugware
        Pols_n = PolynomialRing(dop.base_ring().base_ring(), 'n')
        rop = dop.to_S(ore_algebra.OreAlgebra(Pols_n, 'Sn'))
    Pols_n, n = rop.base_ring().change_ring(Scalars).objgen()
    Rops = ore_algebra.OreAlgebra(Pols_n, 'Sn')
    ordrec = rop.order()
    rop = Rops([p(n-ordrec+shift) for p in rop])
    # Clear_denominators
    den = lcm([p.denominator() for p in rop])
    rop = den*rop
    # Remove constant common factors to make the recurrence smaller
    if Scalars is QQ:
        g = gcd(c for p in rop for c in p)
    # elif utilities.is_QQi(Scalars): # XXX: too slow (and not general enough)
    #     ZZi = Scalars.maximal_order()
    #     g = ZZi.zero()
    #     for c in (c1 for p in rop for c1 in p):
    #         g = ZZi(g).gcd(ZZi(c)) # gcd returns a nfe
    #         if g.is_one():
    #             g = None
    #             break
    else:
        g = None
    if g is not None:
        rop = (1/g)*rop
    coeff = [rop[ordrec-k] for k in range(ordrec+1)]
    return BwShiftRec(coeff)

class BwShiftRec(object):
    r"""
    A recurrence relation, written in terms of the backward shift operator.

    This class is mainly intended to provide reasonably fast evaluation in the
    context of naïve unrolling.
    """

    def __init__(self, coeff):
        assert isinstance(coeff[0], polynomial_element.Polynomial)
        # assert all(c.denominator().is_one() for c in coeff)
        self.coeff = coeff
        self.base_ring = coeff[0].parent()
        self.Scalars = self.base_ring.base_ring()
        self.order = len(coeff) - 1
        self._coeff_series_cache = [[[] for _ in self.coeff] for _ in [0,1]]
        self._ord = [0, 0]

    def __repr__(self):
        n = self.base_ring.variable_name()
        return " + ".join("({})*S{}^(-{})".format(c, n, j)
                          for j, c in enumerate(self.coeff))

    @lazy_attribute
    def ZZpol(self):
        return self.base_ring.change_ring(ZZ)

    @lazy_attribute
    def base_ring_degree(self):
        # intended for number fields
        return self.Scalars.degree()

    def _coeff_series(self, i, j, components):
        p = self.coeff[i]
        rng = range(p.degree() + 1 - j)
        bin = [ZZ(k+j).binomial(k) for k in rng]
        if components:
            pjplus = [a._coefficients() for a in list(p)[j:]]
            return [self.ZZpol([bin[k]*pjplus[k][l]
                                if len(pjplus[k]) > l else 0
                                for k in rng])
                    for l in range(self.base_ring_degree)]
        else:
            return self.base_ring([bin[k]*p[k+j] for k in rng])

    def _compute_coeff_series(self, ord, components):
        components = int(components)
        cache = self._coeff_series_cache[components]
        for j in range(self._ord[components], ord):
            for (i, c) in enumerate(cache):
                c.append(self._coeff_series(i, j, components))
        self._ord[components] = ord

    def scalars_embedding(self, tgt):
        if tgt is self.Scalars:
            return lambda elt: elt
        elif (isinstance(self.Scalars, NumberField_absolute)
                and self.Scalars.degree() > 2):
            # do complicated coercions via QQbar and CLF only once...
            Pol = PolynomialRing(tgt, 'x')
            x = tgt(self.Scalars.gen())
            def emb(elt):
                return Pol([tgt(c) for c in elt._coefficients()])(x)
            return emb
        else:
            return tgt

    @cached_method
    def poly_eval_strategy(self, tgt):
        r"""
        TESTS::

            sage: from ore_algebra import OreAlgebra
            sage: NF.<a> = NumberField(x^2-x+1, embedding = .5+.8*i)
            sage: Pol.<z> = NF[]
            sage: Dops.<Dz> = OreAlgebra(Pol)
            sage: dop = (z^2 - 4*z + 4)*Dz^2 + (z^2 - 2*z)*Dz + 1
            sage: dop.local_basis_expansions(2, 1)
            [(z - 2)^(-0.50000000000000000? - 0.866025403784439?*I),
            (z - 2)^(-0.50000000000000000? + 0.866025403784439?*I)]
        """

        mor = self.scalars_embedding(tgt)
        def generic_eval(pol, x, _tgt):
            assert _tgt is tgt
            return mor(pol(x))

        try:
            from . import eval_poly_at_int
        except ImportError:
            warnings.warn("Cython code not found")
            return generic_eval, False

        if isinstance(self.Scalars, ComplexBallField):
            if isinstance(tgt, ComplexBallField):
                return eval_poly_at_int.cbf, False
        elif (isinstance(self.Scalars, NumberField_quadratic)
                and self.Scalars.discriminant() % 4 != 1):
            self.Scalars.zero() # cache for direct cython access
            if tgt is self.Scalars:
                return eval_poly_at_int.qnf, False
            elif isinstance(tgt, ComplexBallField):
                if utilities.is_QQi(self.Scalars):
                    return eval_poly_at_int.qqi_to_cbf, True
                else:
                    return eval_poly_at_int.qnf_to_cbf, False
        else:
            return generic_eval, False

    def eval_series(self, tgt, point, ord):
        # typically ord << deg => probably not worth trying to use a fast Taylor
        # shift
        eval_poly, components = self.poly_eval_strategy(tgt)
        if self._ord[int(components)] < ord:
            self._compute_coeff_series(ord, components)
        coeff = self._coeff_series_cache[components]
        rng = range(ord)
        return [ [eval_poly(c[j], point, tgt) for j in rng] for c in coeff]

    def eval_inv_lc_series(self, point, ord, shift):
        eval_poly, components = self.poly_eval_strategy(self.Scalars)
        if self._ord[int(components)] < ord:
            self._compute_coeff_series(ord, components)
        c = self._coeff_series_cache[components][0]
        ser = self.base_ring.element_class(
                self.base_ring, # polynomials, viewed as jets
                [eval_poly(c[j], point, self.Scalars)
                    for j in range(shift, ord)],
                check=False)
        return ser.inverse_series_trunc(ord)

    def __getitem__(self, i):
        return self.coeff[i]

    def shift(self, sh):
        n = self.coeff[0].parent().gen()
        coeff = [pol(sh + n) for pol in self.coeff]
        den = lcm([p.denominator() for p in coeff])
        return BwShiftRec([den*c for c in coeff])

    def change_base(self, base):
        if base is self.base_ring:
            return self
        return BwShiftRec([pol.change_ring(base) for pol in self.coeff])

    def lc_as_rec(self):
        return BwShiftRec([self.coeff[0]])

class MultDict(dict):

    def __missing__(self, k):
        return 0

class LogSeriesInitialValues(object):
    r"""
    Initial values defining a logarithmic series.

    - ``self.expo`` is an algebraic number representing the “valuation” of the
      log-series,
    - ``self.shift`` is a dictionary mapping an integer shift s to a tuple of
      initial values corresponding to the coefficients of x^s, x^s·log(x), ...,
      x^s·log(x)^k/k! for some k
    """

    def __init__(self, expo, values, dop=None, check=True, mults=None):
        r"""
        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.naive_sum import *
            sage: from ore_algebra.analytic.differential_operator import DifferentialOperator
            sage: Dops, x, Dx = DifferentialOperators()
            sage: LogSeriesInitialValues(0, {0: (1, 0)},
            ....:         DifferentialOperator(x*Dx^3 + 2*Dx^2 + x*Dx))
            Traceback (most recent call last):
            ...
            ValueError: invalid initial data for x*Dx^3 + 2*Dx^2 + x*Dx at 0
        """

        try:
            self.expo = QQ.coerce(expo)
        except TypeError:
            try:
                self.expo = QQbar.coerce(expo)
            except TypeError:
                # symbolic; won't be sortable
                self.expo = expo

        if isinstance(values, dict):
            all_values = tuple(chain.from_iterable(
                            ini if isinstance(ini, tuple) else (ini,)
                            for ini in values.values()))
        else:
            all_values = values
            values = dict((n, (values[n],)) for n in range(len(values)))
        self.universe = Sequence(all_values).universe()
        if not utilities.is_numeric_parent(self.universe):
            raise ValueError("initial values must coerce into a ball field")

        self.shift = {}
        if mults is not None:
            for s, m in mults:
                self.shift[s] = [self.universe.zero()]*m
        for k, ini in values.items():
            if isinstance(k, tuple): # requires mult != None
                s, m = k
                s = int(s)
                self.shift[s][m] = self.universe(ini)
            else:
                s = int(k)
                self.shift[s] = tuple(self.universe(a) for a in ini)
        self.shift = { s: tuple(ini) for s, ini in self.shift.items() }

        try:
            if check and dop is not None and not self.is_valid_for(dop):
                raise ValueError("invalid initial data for {} at 0".format(dop))
        except TypeError: # coercion problems btw QQbar and number fields
            pass

    def __repr__(self):
        return ", ".join(
            "[z^({expo}+{shift})·log(z)^{log_power}/{log_power}!] = {val}"
            .format(expo=self.expo, shift=s, log_power=log_power, val=val)
            for s, ini in self.shift.items()
            for log_power, val in enumerate(ini)
            if ini)

    def is_valid_for(self, dop):
        ind = dop._indicial_polynomial_at_zero()
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
                                   for val in self.shift.values()
                                   for x in val))
        else:
            raise ValueError

    def last_index(self):
        return max(chain(iter((-1,)), (s for s, vals in self.shift.items()
                                        if not all(v.is_zero() for v in vals))))

    @cached_method
    def mult_dict(self):
        return MultDict((s, len(vals)) for s, vals in self.shift.items())

    def compatible(self, others):
        return all(self.mult_dict() == other.mult_dict() for other in others)

def random_ini(dop):
    import random
    from sage.all import VectorSpace
    ind = dop.indicial_polynomial(dop.base_ring().gen())
    sl_decomp = my_shiftless_decomposition(ind)
    pol, shifts = random.choice(sl_decomp)
    expo = random.choice(pol.roots(QQbar))[0]
    expo = utilities.as_embedded_number_field_element(expo)
    values = {}
    while all(a.is_zero() for v in values.values() for a in v):
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
    @lazy_attribute
    def valuation(self):
        return QQbar(self.leftmost + self.shift) # alg vs NFelt for re, im

class sort_key_by_asympt:
    r"""
    Specify the sorting order for local solutions.

    Roughly speaking, they are sorted in decreasing order of asymptotic
    dominance: when two solutions are asymptotically comparable, the largest
    one as x → 0 comes first. In addition, divergent solutions, including
    things like `x^i`, always come before convergent ones.
    """

    def __init__(self, x, y=None):
        if y is None:
            self.valuation = x.valuation
            self.log_power = x.log_power
        else:
            self.valuation = x
            self.log_power = y

    def __repr__(self):
        return f"x^({self.valuation})*log(x)^{self.log_power}"

    def __eq__(self, other):
        return (self.log_power == other.log_power
                and self.valuation == other.valuation)

    def __lt__(self, other):
        self_valuation_num = IC(self.valuation)
        other_valuation_num = IC(other.valuation)
        if self_valuation_num.real() < other_valuation_num.real():
            return True
        elif self_valuation_num.real() > other_valuation_num.real():
            return False
        elif self.valuation == other.valuation:
            pass
        elif self.valuation.conjugate() == other.valuation:
            pass
        elif self.valuation.real() < other.valuation.real():
            return True
        elif self.valuation.real() > other.valuation.real():
            return False
        # Valuations have equal real parts, compare powers of log
        if self.log_power > other.log_power:
            return True
        elif self.log_power < other.log_power:
            return False
        # Compare the imaginary parts in such a way that purely real valuations
        # come last
        if abs(self_valuation_num.imag()) > abs(other_valuation_num.imag()):
            return True
        elif abs(self_valuation_num.imag()) < abs(other_valuation_num.imag()):
            return False
        elif self.valuation.conjugate() == other.valuation:
            pass
        elif abs(self.valuation.imag()) > abs(other.valuation.imag()):
            return True
        elif abs(self.valuation.imag()) < abs(other.valuation.imag()):
            return False
        assert self.valuation.imag().sign() == -other.valuation.imag().sign()
        return self.valuation.imag() < 0

class LocalBasisMapper(object):
    r"""
    Utility class for iterating over the canonical local basis of solutions of
    an operator.

    Subclasses should define a fun() method that takes as input a
    LogSeriesInitialValues structure and can access the iteration variables
    as well as some derived quantities through the instance's field.

    The nested loops that iterate over the solutions are spread over several
    methods that can be overridden to share parts of the computation in a
    class of related solutions. The choice of unconditional computations,
    exported data and hooks is a bit ad hoc.
    """

    def __init__(self, dop):
        self.dop = dop

    def run(self):
        r"""
        Compute self.fun() for each element of the local basis at 0 of self.dop.

        The output is a list of FundamentalSolution structures, sorted in the
        canonical order.
        """

        self.bwrec = bw_shift_rec(self.dop) # XXX wasteful in binsplit case
        ind = self.bwrec[0]
        if self.dop.leading_coefficient()[0] != 0:
            n = ind.parent().gen()
            self.sl_decomp = [(-n, [(i, 1) for i in range(self.dop.order())])]
        else:
            self.sl_decomp = my_shiftless_decomposition(ind)

        self.process_decomposition()

        self.cols = []
        self.nontrivial_factor_index = 0
        for self.sl_factor, self.shifts in self.sl_decomp:
            for self.irred_factor, irred_mult in self.sl_factor.factor():
                assert irred_mult == 1
                if self.irred_factor.degree() == 1:
                    rt = -self.irred_factor[0]/self.irred_factor[1]
                    if rt.is_rational():
                        rt = QQ(rt)
                    self.roots = [rt]
                else:
                    roots = complex_roots(self.irred_factor,
                            skip_squarefree=True, retval='algebraic')
                    self.roots = [utilities.as_embedded_number_field_element(rt)
                                  for rt, _ in roots]
                logger.debug("indicial factor = %s, roots = %s",
                             self.irred_factor, self.roots)
                self.irred_factor_cols = []
                self.process_irred_factor()
                self.cols.extend(self.irred_factor_cols)
                if self.irred_factor.degree() >= 2:
                    self.nontrivial_factor_index += 1
        self.cols.sort(key=sort_key_by_asympt)
        return self.cols

    def process_decomposition(self):
        pass

    # The next three methods can be overridden to customize the iteration. Each
    # specialized implementation should set the same fields (self.leftmost,
    # etc.) as the original method does, and call the next method in the list,
    # or at least ultimately result in process_solution() being called with the
    # correct fields set.

    def process_irred_factor(self):
        for self.leftmost in self.roots:
            self.edop, self.leftmost = self.dop.extend_scalars(self.leftmost)
            if self.edop is self.dop:
                self.shifted_bwrec = self.bwrec.shift(self.leftmost)
            else:
                mybwrec = bw_shift_rec(self.edop)
                self.shifted_bwrec = mybwrec.shift(self.leftmost)
            self.process_modZ_class()

    def process_modZ_class(self):
        for self.shift, self.mult in reversed(self.shifts):
            self.process_valuation()

    def process_valuation(self):
        for self.log_power in reversed(range(self.mult)):
            self.process_solution()

    def process_solution(self):
        ini = LogSeriesInitialValues(
            dop = self.edop,
            expo = self.leftmost,
            values = { (self.shift, self.log_power): ZZ.one() },
            mults = self.shifts,
            check = False)
        # XXX: inefficient if self.shift >> 0
        value = self.fun(ini)
        sol = FundamentalSolution(
            leftmost = self.leftmost,
            shift = ZZ(self.shift),
            log_power = ZZ(self.log_power),
            value = value)
        self.irred_factor_cols.append(sol)

    def fun(self, ini):
        return None

def exponent_shifts(dop, leftmost):
    bwrec = bw_shift_rec(dop)
    ind = bwrec[0]
    sl_decomp = my_shiftless_decomposition(ind)
    cand = [shifts for fac, shifts in sl_decomp if fac(leftmost).is_zero()]
    assert len(cand) == 1
    shifts = cand[0]
    assert all(s >=0 for s, m in shifts)
    assert shifts[0][0] == 0
    return shifts

def log_series(ini, bwrec, order):
    Coeffs = utilities.mypushout(bwrec.base_ring.base_ring(), ini.universe)
    max_log_prec = sum(len(v) for v in ini.shift.values())
    log_prec = 0
    series = []
    for n in range(order):
        mult = len(ini.shift.get(n, ()))
        bwrec_n = bwrec.eval_series(Coeffs, n, log_prec + mult)
        invlc = None
        new_term = vector(Coeffs, max_log_prec)
        for p in range(log_prec - 1, -1, -1):
            combin  = sum(bwrec_n[i][j]*series[-i][p+j]
                          for j in range(log_prec - p)
                          for i in range(min(bwrec.order, n), 0, -1)
                          if series[-i][p+j])
            combin += sum(bwrec_n[0][j]*new_term[p+j]
                          for j in range(mult + 1, log_prec + mult - p)
                          if new_term[p+j])
            if combin:
                if invlc is None:
                    invlc = ~bwrec_n[0][mult]
                new_term[mult + p] = - invlc * combin
        for p in range(mult - 1, -1, -1):
            new_term[p] = ini.shift[n][p]
        for p in range(log_prec, log_prec + mult):
            if new_term[p]:
                log_prec = p + 1
        series.append(new_term)
    return series

def log_series_values(Jets, expo, psum, evpt, downshift=[0]):
    r"""
    Evaluate a logarithmic series, and optionally its downshifts.

    That is, compute the vectors (v[0], ..., v[r-1]) such that ::

        Σ[k=0..r] v[k] η^k
            = (pt + η)^expo * Σ_k (psum[d+k]*log(x + η)^k/k!) + O(η^r)

        (x = evpt.pt, r = evpt.jet_order)

    for d ∈ downshift, as an element of ``Jets``, optionally using a
    non-standard branch of the logarithm.

    Note that while this function computes ``pt^expo`` in ℂ, it does NOT
    specialize abstract algebraic numbers that might appear in ``psum``.
    """
    derivatives = evpt.jet_order
    log_prec = psum.length()
    assert all(d < log_prec for d in downshift) or log_prec == 0
    if not evpt.is_numeric:
        if expo != 0 or log_prec > 1:
            raise NotImplementedError("log-series of symbolic point")
        return [vector(psum[0][i] for i in range(derivatives))]
    pt = Jets.base_ring()(evpt.pt)
    if log_prec > 1 or expo not in ZZ or evpt.branch != (0,):
        pt = pt.parent().complex_field()(pt)
        Jets = Jets.change_ring(Jets.base_ring().complex_field())
        psum = psum.change_ring(Jets)
    high = Jets([0] + [(-1)**(k+1)*~pt**k/k
                       for k in range(1, derivatives)])
    aux = high*expo
    logger.debug("aux=%s", aux)
    val = [Jets.base_ring().zero() for d in downshift]
    for b in evpt.branch:
        twobpii = pt.parent()(2*b*pi*I)
        # hardcoded series expansions of log(a+η) and (a+η)^λ
        # (too cumbersome to compute directly in Sage at the moment)
        logpt = Jets([pt.log() + twobpii]) + high
        logger.debug("logpt[%s]=%s", b, logpt)
        if expo in ZZ and expo >= 0:
            # the general formula in the other branch does not work when pt
            # contains zero
            expo = int(expo)
            inipow = _pow_trunc(Jets([pt, 1]), expo, derivatives)
        else:
            inipow = ((twobpii*expo).exp()*pt**expo
                     *sum(_pow_trunc(aux, k, derivatives)/Integer(k).factorial()
                          for k in range(derivatives)))
        logger.debug("inipow[%s]=%s", b, inipow)
        logterms = [_pow_trunc(logpt, p, derivatives)/Integer(p).factorial()
                    for p in range(log_prec)]
        for d in downshift:
            val[d] += inipow.multiplication_trunc(
                    sum(psum[d+p]._mul_trunc_(logterms[p], derivatives)
                        for p in range(log_prec - d)),
                    derivatives)
    Vectors = Jets.base_ring()**derivatives
    l = len(evpt.branch)
    val = [FreeModuleElement_generic_dense(Vectors,
               [v[i] for i in range(derivatives)], coerce=False, copy=False)/l
           for v in val]
    return val

def _pow_trunc(a, n, ord):
    pow = a.parent().one()
    pow2k = a
    while n:
        if n & 1:
            pow = pow._mul_trunc_(pow2k, ord)
        pow2k = pow2k._mul_trunc_(pow2k, ord)
        n = n >> 1
    return pow

##############################################################################
# Human-readable representations that avoid various issues with symbolic
# expressions
##############################################################################

def simplify_exponent(e):
    r"""
    TESTS::

        sage: from ore_algebra.examples import cbt
        sage: lc = cbt.dop[10].leading_coefficient()
        sage: s = sorted(lc.roots(QQbar, multiplicities=False), key=abs)[0]
        sage: cbt.dop[10].local_basis_monomials(s)
        [1,
        z - 0.2651878342412026?,
        (z - 0.2651878342412026?)^2,
        (z - 0.2651878342412026?)^3,
        (z - 0.2651878342412026?)^4,
        (z - 0.2651878342412026?)^4.260514654474679?,
        (z - 0.2651878342412026?)^5,
        (z - 0.2651878342412026?)^6,
        (z - 0.2651878342412026?)^7,
        (z - 0.2651878342412026?)^8,
        (z - 0.2651878342412026?)^9]
        sage: cbt.dop[10].local_basis_expansions(s, 1) # long time (1.3 s)
        [1, 0, 0, 0, 0, (z - 0.2651878342412026?)^4.260514654474679?, 0, 0, 0,
        0, 0]
    """
    for dom in ZZ, QQ, AA, QQbar:
        try:
            return dom(e)
        except (TypeError, ValueError):
            pass
    return e

class LogMonomial(object):

    def __init__(self, dx, expo, shift, k):
        self.dx = dx
        self.expo = simplify_exponent(expo)
        self.shift = shift
        self.n = self.expo + shift
        self.k = k

    def __eq__(self, other):
        return (isinstance(other, LogMonomial)
                and self.__dict__ == other.__dict__)

    def __hash__(self):
        return hash((self.dx, self.expo, self.shift, self.k))

    def __repr__(self):
        dx = repr(self.dx)
        if self.n.is_zero():
            if self.k == 0:
                s = "1"
            else:
                s = ""
        else:
            s = dx if self.dx.operator() is None else "(" + dx + ")"
            if not self.n.is_one():
                n = repr(self.n)
                if (self.n in ZZ or self.n.parent() is AA) and self.n >= 0:
                    s += "^" + n
                else:
                    s += "^(" + n + ")"
        if self.k > 0:
            if not self.n.is_zero():
                s += "*"
            s += "log(" + dx + ")"
        if self.k > 1:
            s += "^" + repr(self.k)
        return s

    def _symbolic_(self):
        return dx**self.n*symbolic_log.log(x, hold=True)**self.k
