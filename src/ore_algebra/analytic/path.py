# -*- coding: utf-8 - vim: tw=80
"""
Analytic continuation paths
"""

# Copyright 2015, 2016, 2017, 2018, 2019 Marc Mezzarobba
# Copyright 2015, 2016, 2017, 2018, 2019 Centre national de la recherche scientifique
# Copyright 2015, 2016, 2017, 2018 Université Pierre et Marie Curie
# Copyright 2019 Sorbonne Université
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

import itertools, logging, sys

import sage.plot.all as plot
import sage.rings.all as rings
import sage.rings.number_field.number_field as number_field
import sage.rings.number_field.number_field_base as number_field_base
import sage.structure.coerce
import sage.symbolic.ring

from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute
from sage.rings.all import ZZ, QQ, CC, CIF, QQbar, RLF, CLF
from sage.rings.complex_arb import CBF, ComplexBallField, ComplexBall
from sage.rings.real_arb import RBF, RealBallField, RealBall
from sage.structure.element import coercion_model
from sage.structure.sage_object import SageObject

from .accuracy import IR, IC
from .context import dctx
from .deform import PathDeformer, PathDeformationFailed
from .differential_operator import DifferentialOperator
from .local_solutions import FundamentalSolution, LocalBasisMapper
from .safe_cmp import *
from .utilities import *

logger = logging.getLogger(__name__)

QQi = number_field.QuadraticField(-1, 'i')

class PathPrecisionError(Exception):
    pass

######################################################################
# Points
######################################################################

class Point(SageObject):
    r"""
    A point on the complex plane with an associated differential operator.

    A point can be exact (a number field element) or inexact (a real or complex
    interval or ball). It can be classified as ordinary, regular singular, etc.
    The main reason for making the operator part of the definition of Points is
    that this gives a convenient place to cache information that depend on both,
    with an appropriate lifetime. Note however that the point is considered to
    lie on the complex plane, not on the Riemann surface of the operator.
    """

    def __init__(self, point, dop=None, singular=None, detour_to=None, **kwds):
        """
        INPUT:

        - ``singular``: can be set to True to force this point to be considered
          a singular point, even if this cannot be checked (e.g. because we only
          have an enclosure)

        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = DifferentialOperators()
            sage: [Point(z, Dx)
            ....:  for z in [1, 1/2, 1+I, QQbar(I), RIF(1/3), CIF(1/3), pi,
            ....:  RDF(1), CDF(I), 0.5r, 0.5jr, 10r, QQbar(1), AA(1/3)]]
            [1, 1/2, I + 1, I, [0.333333333333333...], [0.333333333333333...],
            3.141592653589794?, ~1.0000, ~1.0000*I, ~0.50000, ~0.50000*I, 10,
            1, 1/3]
            sage: Point(sqrt(2), Dx).iv()
            [1.414...]
            sage: Point(RBF(0), (x-1)*x*Dx, singular=True).dist_to_sing()
            1.000000000000000
        """
        SageObject.__init__(self)

        from sage.rings.complex_double import ComplexDoubleField_class
        try:
            from sage.rings.complex_mpfr import ComplexField_class
        except ImportError:
            from sage.rings.complex_field import ComplexField_class
        from sage.rings.complex_interval_field import ComplexIntervalField_class
        from sage.rings.real_double import RealDoubleField_class
        from sage.rings.real_mpfi import RealIntervalField_class
        from sage.rings.real_mpfr import RealField_class

        point = sage.structure.coerce.py_scalar_to_element(point)
        try:
            parent = point.parent()
        except AttributeError:
            raise TypeError("unexpected value for point: " + repr(point))
        if isinstance(point, Point): # XXX useful?
            self.value = point.value
        elif isinstance(parent, (RealBallField, ComplexBallField)):
            self.value = point
        elif isinstance(parent, number_field_base.NumberField):
            _, hom = good_number_field(point.parent())
            self.value = hom(point)
        elif QQ.has_coerce_map_from(parent):
            self.value = QQ.coerce(point)
        elif QQbar.has_coerce_map_from(parent):
            alg = QQbar.coerce(point)
            NF, val, hom = alg.as_number_field_element()
            if NF is QQ:
                self.value = QQ.coerce(val) # parent may be ZZ
            else:
                embNF = number_field.NumberField(NF.polynomial(),
                                                NF.variable_name(),
                                                embedding=hom(NF.gen()))
                self.value = val.polynomial()(embNF.gen())
        elif isinstance(parent, (RealField_class, RealDoubleField_class,
                                 RealIntervalField_class)):
            self.value = RealBallField(point.prec())(point)
        elif isinstance(parent, (ComplexField_class, ComplexDoubleField_class,
                                 ComplexIntervalField_class)):
            self.value = ComplexBallField(point.prec())(point)
        elif parent is sage.symbolic.ring.SR:
            try:
                self.__init__(point.pyobject(), dop)
                return
            except TypeError:
                pass
            try:
                self.__init__(QQbar(point), dop)
                return
            except (TypeError, ValueError, NotImplementedError):
                pass
            try:
                self.value = RLF(point)
            except (TypeError, ValueError):
                self.value = CLF(point)
        else:
            try:
                self.value = RLF.coerce(point)
            except TypeError:
                self.value = CLF.coerce(point)

        parent = self.value.parent()
        assert (isinstance(parent, (number_field_base.NumberField,
                                    RealBallField, ComplexBallField))
                or parent is RLF or parent is CLF)

        if dop is None: # TBI XXX useful?
            if isinstance(point, Point):
                self.dop = point.dop
        else:
            self.dop = DifferentialOperator(dop.numerator())
        self._force_singular = bool(singular)
        self.detour_to = detour_to
        self.options = kwds

    def _repr_(self, size=False):
        """
        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = DifferentialOperators()
            sage: Point(10**20, Dx)
            ~1.0000e20
        """
        res = None
        if self.is_exact():
            try:
                len = (self.value.parent().precision()
                        if isinstance(self.value, (RealBall, ComplexBall))
                        else self.nbits())
                if len > 50:
                    res = repr(self.value.n(digits=5))
                    if size:
                        res = f"~[{self.nbits()}b]{res}"
                    else:
                        res = "~" + res
            except AttributeError:
                res = repr(self.value)
        if res is None:
            res = repr(self.value)
        if self.detour_to is not None:
            res += f" (.. {self.detour_to})"
        return res

    def keep_value(self):
        return bool(self.options.get("keep_value"))

    def nbits(self):
        if isinstance(self.value, (RealBall, ComplexBall)):
            return self.value.nbits()
        else:
            res = self.value.denominator().nbits()
            res += max(self.value.numerator().real().numerator().nbits(),
                        self.value.numerator().imag().numerator().nbits())
            return res

    def algdeg(self):
        if isinstance(self.value, rings.NumberFieldElement):
            return self.value.parent().degree()
        else:
            return 1

    @cached_method
    def is_fast(self):
        return isinstance(self.value, (RealBall, ComplexBall, rings.Integer,
                                 rings.Rational)) or is_QQi(self.value.parent())

    def bit_burst_bits(self, tgt_prec):
        if self.is_fast():
            return self.nbits()
        else:
            # RLF, CLF, other number fields (debatable!)
            return tgt_prec

    def conjugate(self):
        value = QQbar.coerce(self.value).conjugate()
        return Point(value, self.dop, **self.options)

    # Numeric representations

    @cached_method
    def iv(self):
        """
        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = DifferentialOperators()
            sage: [Point(z, Dx).iv()
            ....: for z in [1, 1/2, 1+I, QQbar(I), RIF(1/3), CIF(1/3), pi]]
            [1.000000000000000,
            0.5000000000000000,
            1.000000000000000 + 1.000000000000000*I,
            1.000000000000000*I,
            [0.333333333333333 +/- 3.99e-16],
            [0.333333333333333 +/- 3.99e-16],
            [3.141592653589793 +/- 7.83e-16]]
        """
        return IC(self.value)

    def exact(self):
        r"""
        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = DifferentialOperators()
            sage: QQi.<i> = QuadraticField(-1)
            sage: [Point(z, Dx).exact() for z in [1, 1/2, 1+i, QQbar(I)]]
            [1, 1/2, i + 1, I]
            sage: [Point(z, Dx).exact() for z in [RBF(3/4), RBF(1) + I]]
            [3/4, i + 1]
            sage: Point(RIF(1/3), Dx).exact()
            Traceback (most recent call last):
            ...
            ValueError
        """
        if self.value.parent().is_exact():
            return self
        elif isinstance(self.value, RealBall) and self.value.is_exact():
            return Point(QQ(self.value), self.dop, **self.options)
        elif isinstance(self.value, ComplexBall) and self.value.is_exact():
            value = QQi((QQ(self.value.real()), QQ(self.value.imag())))
            return Point(value, self.dop, **self.options)
        raise ValueError

    def _algebraic_(self, parent):
        return parent(self.exact().value)

    def approx_abs_real(self, prec):
        r"""
        Compute an approximation with absolute error about 2^(-prec).
        """
        if isinstance(self.value.parent(), RealBallField):
            return self.value
        elif self.value.is_zero():
            return RealBallField(max(2, prec)).zero()
        elif self.is_real():
            expo = ZZ(IR(self.value).abs().log(2).upper().ceil())
            rel_prec = max(2, prec + expo + 10)
            val = RealBallField(rel_prec)(self.value)
            return val
        else:
            raise ValueError("point may not be real")

    def is_real(self):
        return is_real_parent(self.value.parent())

    def is_exact(self):
        r"""
        Is this point exact in the sense that we can use it in the coefficients
        of an operator?
        """
        return (isinstance(self.value, (rings.Integer, rings.Rational,
                                        rings.NumberFieldElement))
                or isinstance(self.value, (RealBall, ComplexBall))
                    and self.value.is_exact())

    def rationalize(self):
        a = self.iv()
        if any(a.overlaps(s) for s in self.dop._singularities(IC)):
            raise PathPrecisionError
        else:
            return Point(_rationalize(a), self.dop)

    def truncate(self, prec, tgt_prec):
        Ivs = RealBallField if self.is_real() else ComplexBallField
        approx = Ivs(prec)(self.value).round()
        lc = self.dop.leading_coefficient()
        if lc(approx).contains_zero():
            raise PathPrecisionError # appropriate?
        approx = approx.squash()
        return Point(Ivs(tgt_prec)(approx), self.dop)

    def __complex__(self):
        return complex(self.value)

    # Point equality is identity

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __lt__(self, other):
        r"""
        Temporary kludge (Sage graphs require vertices to be comparable).
        """
        return id(self) < id(other)

    ### Methods that depend on dop

    @cached_method
    def is_ordinary(self):
        if self._force_singular:
            return False
        iv = self.iv()
        if not any(iv.overlaps(s) for s in self.dop._singularities(IC)):
            return True
        if self.is_exact():
            lc = self.dop.leading_coefficient()
            try:
                val = lc(self.value)
            except TypeError: # work around coercion weaknesses
                val = lc.change_ring(QQbar)(QQbar.coerce(self.value))
            return not val.is_zero()
        else:
            raise ValueError("can't tell if inexact point is singular")

    def is_singular(self):
        return not self.is_ordinary()

    @cached_method
    def is_regular(self):
        try:
            if self.is_ordinary():
                return True
        except ValueError:
            # we could handle balls containing no irregular singular point...
            raise NotImplementedError("can't tell if inexact point is regular")
        assert self.is_exact()
        # Fuchs criterion
        dop, pt = self.dop.extend_scalars(self.value)
        Pols = dop.base_ring()
        lin = Pols([pt, -1])
        ref = dop.leading_coefficient().valuation(lin) - dop.order()
        return all(coef.valuation(lin) - k >= ref for k, coef in enumerate(dop))

    def is_regular_singular(self):
        return not self.is_ordinary() and self.is_regular()

    def is_irregular(self):
        return not self.is_regular()

    def singularity_type(self, short=False):
        r"""
        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = DifferentialOperators()

            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
            sage: Point(1, dop).singularity_type()
            'ordinary point'
            sage: Point(i, dop).singularity_type()
            'regular singular point'
            sage: Point(0, x^2*Dx + 1).singularity_type()
            'irregular singular point'
            sage: Point(CIF(1/3), x^2*Dx + 1).singularity_type()
            'ordinary point'
            sage: Point(CIF(1/3)-1/3, x^2*Dx + 1).singularity_type()
            'point of unknown singularity type'
        """
        try:
            if self.is_ordinary():
                return "" if short else "ordinary point"
            elif self.is_regular():
                return "regular singular point"
            else:
                return "irregular singular point"
        except (ValueError, NotImplementedError):
            return "point of unknown singularity type"

    def descr(self):
        t = self.singularity_type(short=True)
        if t == "":
            return repr(self)
        else:
            return t + " " + repr(self)

    def dist_to_sing(self):
        """
        Distance of self to the singularities of self.dop *other than self*.

        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = DifferentialOperators()
            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
            sage: Point(1, dop).dist_to_sing()
            [1.41421356237309...]
            sage: Point(i, dop).dist_to_sing()
            2.00...
            sage: Point(1+i, dop).dist_to_sing()
            1.00...

        """
        sing = self.dop._singularities(IC)
        close, distant = split(lambda s: s.overlaps(self.iv()), sing)
        if (len(close) >= 2 or len(close) == 1 and not self.is_singular()):
            raise NotImplementedError # refine?
        dist = [(self.iv() - s).abs() for s in distant]
        min_dist = IR(rings.infinity).min(*dist)
        if min_dist.contains_zero():
            raise NotImplementedError # refine???
        return IR(min_dist.lower())

    def local_basis_structure(self):
        r"""
        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = DifferentialOperators()
            sage: Point(0, x*Dx^2 + Dx + x).local_basis_structure()
            [FundamentalSolution(leftmost=0, shift=0, log_power=1, value=None),
             FundamentalSolution(leftmost=0, shift=0, log_power=0, value=None)]
            sage: Point(0, Dx^3 + x*Dx + x).local_basis_structure()
            [FundamentalSolution(leftmost=0, shift=0, log_power=0, value=None),
             FundamentalSolution(leftmost=0, shift=1, log_power=0, value=None),
             FundamentalSolution(leftmost=0, shift=2, log_power=0, value=None)]
        """
        # TODO: provide a way to compute the first terms of the series. First
        # need a good way to share code with fundamental_matrix_regular. Or
        # perhaps modify generalized_series_solutions() to agree with our
        # definition of the basis?
        if self.is_ordinary(): # support inexact points in this case
            return [FundamentalSolution(QQbar.zero(), ZZ(expo), ZZ.zero(), None)
                    for expo in range(self.dop.order())]
        elif not self.is_regular():
            raise NotImplementedError("irregular singular point")
        return LocalBasisMapper(self.dop.shift(self)).run()

    def simple_approx(self, neighb, ctx=dctx):
        r"""
        Return an approximation of this point suitable as a starting point for
        the next analytic continuation step.

        For intermediate steps via simple points to reach points of large bit
        size or with irrational coordinates in the context of binary splitting,
        see bit_burst_split().

        For intermediate steps where thick balls are shrunk to their center,
        see exact_approx().
        """
        # Point options become meaningless (and are lost) when not returning
        # self.
        thr = ctx.simple_approx_thr
        if (self.is_singular()
                or self.is_fast() and self.is_exact() and self.nbits() <= thr):
            return self
        _self = self.iv().squash()
        rad = RBF.one().min(self.dist_to_sing())
        rad = rad.min(*(abs(_self - z.iv().squash()) for z in neighb))
        rad /= 32
        ball = self.iv().add_error(rad)
        try:
            for c in neighb:
                Step(Point(ball, self.dop), c).check_singularity()
        except ValueError:
            return self
        rat = _rationalize(ball, real=self.is_real())
        return Point(rat, self.dop, detour_to=self)

    @cached_method
    def exact_approx(self):
        if isinstance(self.value, (RealBall, ComplexBall)):
            if not self.value.is_exact():
                return Point(self.value.trim().squash(), self.dop)
        return self

class EvaluationPoint(object):
    r"""
    Series evaluation point(s)/jet(s).

    A tuple of elements of the same ring (complex numbers, polynomial
    indeterminates, perhaps someday a matrices) where to evaluate the partial
    sum of a series, along with a “jet order” used to compute derivatives and a
    bound on the norm of the mathematical quantity it represents that can be
    used to bound the truncation error.

    * ``branch`` - branch of the logarithm to use; ``(0,)`` means the standard
      branch, ``(k,)`` means log(z) + 2kπi, a tuple of length > 1 averages over
      the corresponding branches
    """

    # XXX: choose a single place to set the default value for jet_order
    def __init__(self, pts, jet_order=1, branch=(0,), rad=None ):
        pts = pts if isinstance(pts, tuple) else (pts,) # bwd compat
        # This is mainly to catch cases where the parents are number fields that
        # differ only in variable names, maybe also ball fields with different
        # precisions. A downside of coercing to a common parent is that we may
        # end up working in a larger extension than necessary if the evaluation
        # points are algebraic numbers belonging to different extensions;
        # however, algebraic numbers are mainly useful for evaluations at
        # singularities and therefore should typically appear only in
        # "expansion" position.
        try:
            self.parent = coercion_model.common_parent(*pts)
            self.pts = tuple(self.parent.coerce(pt) for pt in pts)
        except TypeError: # sigh...
            self.parent, self.pts = as_embedded_number_field_elements(
                list(QQbar.coerce(pt) for pt in pts))
            self.pts = tuple(self.pts)

        self.rad = max(IC(pt).above_abs() for pt in pts) if rad is None else rad
        self.jet_order = jet_order
        self.branch=branch

        self.is_numeric = is_numeric_parent(self.parent)
        self.is_real_or_symbolic = (is_real_parent(self.parent)
                                    or not self.is_numeric)
        self.accuracy = self._accuracy()

    @lazy_attribute
    def pt(self): # bwd compat
        assert len(self.pts) == 1
        return self.pts[0]

    def __repr__(self):
        fmt = "{} + η + O(η^{}) (with |.| ≤ {})"
        return fmt.format(self.pts, self.jet_order + 1, self.rad)

    def jets(self, Intervals):
        base_ring = (Intervals if self.is_numeric
                     else mypushout(self.parent, Intervals))
        Jets = PolynomialRing(base_ring, 'delta')
        jets = tuple(Jets([pt, 1]).truncate(self.jet_order)
                     for pt in self.pts)
        return Jets, jets

    def _accuracy(self):
        if self.parent.is_exact():
            return IR.maximal_accuracy()
        elif isinstance(self.parent, (RealBallField, ComplexBallField)):
            return min(pt.accuracy() for pt in self.pts) # debatable
        else:
            raise ValueError

######################################################################
# Paths
######################################################################

class Step(SageObject):
    r"""
    Analytic continuation step from a :class:`Point` to another

    EXAMPLES::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.path import Point, Step
        sage: QQi.<i> = QuadraticField(-1)
        sage: Dops, x, Dx = DifferentialOperators()

        sage: s1 = Step(Point(0, x*Dx-1), Point(i/7, x*Dx-1))
        sage: s2 = Step(Point(RIF(1/3), x*Dx-1), Point(pi, x*Dx-1))
        sage: s3 = Step(Point(-i, x*Dx-1), Point(i, x*Dx-1))

        sage: s1, s2
        (0 --> 1/7*i, [0.333333333333333 +/- 3.99e-16] --> 3.141592653589794?)

        sage: list(s1), list(s2)
        ([0, 1/7*i], [[0.333333333333333 +/- 3.99e-16], 3.141592653589794?])

        sage: s1.is_exact(), s2.is_exact()
        (True, False)

        sage: s1.delta(), s2.delta()
        (1/7*i, [2.80825932025646 +/- 1.56e-15])

        sage: s1.length(), s2.length()
        ([0.142857142857142...], [2.8082593202564...])

        sage: s1.check_singularity()
        sage: s2.check_singularity()
        sage: s3.check_singularity()
        Traceback (most recent call last):
        ...
        ValueError: Step -i --> i passes through or too close to singular point
        0 (to compute the connection to a singular point, make it a vertex of
        the path)

        sage: s2.check_convergence()
        Traceback (most recent call last):
        ...
        ValueError: Step ... escapes from the disk of (guaranteed)
        convergence of the solutions at ...

        sage: s2.plot()
        Graphics object consisting of 1 graphics primitive

    TESTS:

    Check that we can handle connections between points in ℚ[i] and in other
    complex number fields in spite of various weaknesses of the coercion system.
    Thanks to Armin Straub for the example::

        sage: dop = ((81*x^4 + 14*x^3 + x^2)*Dx^3 + (486*x^3 + 63*x^2 +
        ....: 3*x)*Dx^2 + (567*x^2 + 48*x + 1)*Dx + 81*x + 3)
        sage: dop.numerical_transition_matrix([0,QQbar((4*sqrt(2)*I-7)/81)])[0,0]
        [-3.17249673357...] + [-4.486587907205...]*I
    """

    def __init__(self, start, end, reversed=False, type=None, branch=None,
                 max_split=None):
        if not (isinstance(start, Point) and isinstance(end, Point)):
            raise TypeError
        if start.dop != end.dop:
            raise ValueError
        self.start = start
        self.end = end
        self.reversed = reversed
        self.type = type
        self.branch = (0,) if branch is None else branch
        self.max_split = 3 if max_split is None else max_split

    def _repr_(self):
        type = "" if self.type is None else "[{}] ".format(self.type)
        bb = (self.type == "bit-burst")
        start = self.start._repr_(size=bb)
        end = self.end._repr_(size=bb)
        arrow = " --> "
        if self.reversed:
            start, end = end, start
            arrow = " <-- "
        return type + start + arrow + end

    def __getitem__(self, i):
        if i == 0:
            return self.start
        elif i == 1:
            return self.end
        else:
            raise IndexError

    def is_exact(self):
        return self.start.is_exact() and self.end.is_exact()

    def algdeg(self):
        d0 = self.start.algdeg()
        d1 = self.end.algdeg()
        if d0 > 1 and d1 > 1:
            if self.start.value.parent() is not self.end.value.parent():
                return d0*d1
        return max(d0, d1)

    def delta(self):
        r"""
        TESTS::

            sage: from ore_algebra import *
            sage: Dops, x, Dx = DifferentialOperators()
            sage: (Dx - 1).numerical_solution([1], [0, RealField(10)(.33), 1])
            [2.71828182845904...]
        """
        z0, z1 = self.start.value, self.end.value
        if z0.parent() is z1.parent():
            return z1 - z0
        elif (isinstance(z0, (RealBall, ComplexBall))
                and isinstance(z1, (RealBall, ComplexBall))):
            p0, p1 = z0.parent().precision(), z1.parent().precision()
            real = isinstance(z0, RealBall) and isinstance(z1, RealBall)
            Tgt = (RealBallField if real else ComplexBallField)(max(p0, p1))
            return Tgt(z1) - Tgt(z0)
        else: # XXX not great when one is in a number field != QQ[i]
            if self.start.is_exact():
                z0 = self.start.exact().value
            if self.end.is_exact():
                z1 = self.end.exact().value
            try:
                d = z1 - z0
            except TypeError:
                # Should be coercions, but embedded number fields currently
                # don't coerce into QQbar...
                d = QQbar(z1) - QQbar(z0)
            # When z0, z1 are number field elements, we want another number
            # field element, not an element of QQbar or AA (even though z1-z0
            # may succeed and return such an element).
            if d.parent() is z0.parent() or d.parent() is z1.parent():
                return d
            else:
                return as_embedded_number_field_element(d)

    def evpt(self, order):
        return EvaluationPoint(self.delta(), order, branch=self.branch)

    def direction(self):
        delta = self.end.iv() - self.start.iv()
        return delta/abs(delta)

    def length(self):
        return IC(self.delta()).abs()

    def prec(self, tgt_prec):
        r"""
        Precision "at which this step contributes" to a result to be computed at
        precision tgt_prec, ~ lg(1/length) when < tgt_prec, ~ ∞ otherwise.
        """
        myIC = ComplexBallField(tgt_prec + 10) # not ideal...
        len = IC(myIC(self.end.value) - myIC(self.start.value)).abs()
        if len.contains_zero():
            return ZZ(sys.maxsize)
        else:
            return -ZZ(len.log(2).upper().ceil())

    def cvg_ratio(self):
        return self.length()/self.start.dist_to_sing()

    def split(self):
        # Ensure that the substeps correspond to convergent series when
        # splitting a singular step
        if self.max_split <= 0:
            raise ValueError
        assert not self.end.is_singular()
        if self.start.is_singular():
            mid = (self.start.iv() + 2*self.end.iv())/3
        else:
            mid = (self.start.iv() + self.end.iv())/2
        mid = Point(mid, self.start.dop)
        mid = mid.rationalize()
        s0 = Step(self.start, mid, type="split", branch=self.branch,
                  max_split=self.max_split-1)
        s1 = Step(mid, self.end, type="split", branch=None,
                  max_split=self.max_split-1)
        return (s0, s1)

    def bit_burst_split(self, tgt_prec, bit_burst_prec):
        r"""
        Try to split a step using a ~bit_burst_prec-bit approx of one of the
        ends as an intermediate point
        """
        z0, z1 = self
        if z0.is_singular() or z1.is_singular():
            return ()
        p0, p1 = z0.bit_burst_bits(tgt_prec), z1.bit_burst_bits(tgt_prec)
        if max(p0, p1) <= 2*bit_burst_prec:
            return ()
        elif p0 <= p1:
            z1_tr = z1.truncate(bit_burst_prec, tgt_prec)
            s0 = Step(z0, z1_tr, type="bit-burst",
                      branch=self.branch, max_split=0)
            s1 = Step(z1_tr, z1, type="bit-burst", max_split=0)
        else:
            z0_tr = z0.truncate(bit_burst_prec, tgt_prec)
            s0 = Step(z0, z0_tr, type="bit-burst",
                      branch=self.branch, max_split=0)
            s1 = Step(z0_tr, z1, type="bit-burst", max_split=0)
        return (s0, s1)

    def singularities(self):
        dop = self.start.dop
        sing = dop._singularities(IC)
        z0, z1 = IC(self.start.value), IC(self.end.value)
        sing = [s for s in sing if s != z0 and s != z1]
        res = []
        for s in sing:
            ds = s - self.start.iv()
            t = (self.end.iv() - self.start.iv())/ds
            if (ds.contains_zero() or t.imag().contains_zero()
                    and not safe_lt(t.real(), IR.one())):
                res.append(s)
        return res

    def check_singularity(self):
        r"""
        Raise an error if this step goes through a singular point or seems to do
        so at our working precision.

        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Point, Step
            sage: Dops, x, Dx = DifferentialOperators(); i = QuadraticField(-1, 'i').gen()
            sage: dop = (x^2 + 1)*Dx
            sage: Step(Point(0, dop), Point(0, dop)).check_singularity()
            sage: Step(Point(0, dop), Point(1, dop)).check_singularity()
            sage: Step(Point(1, dop), Point(1, dop)).check_singularity()
            sage: Step(Point(1, dop), Point(i, dop)).check_singularity()
            sage: Step(Point(i, dop), Point(0, dop)).check_singularity()
            sage: Step(Point(i, dop), Point(i, dop)).check_singularity()
            sage: Step(Point(2*i+1, dop), Point(-11/10, dop)).check_singularity()
            sage: Step(Point(2*i, dop), Point(0, dop)).check_singularity()
            Traceback (most recent call last):
            ...
            ValueError: Step 2*i --> 0 passes through or too close to singular
            point 1*I (to compute the connection to a singular point, make it a
            vertex of the path)
            sage: Step(Point(2*i+1, dop), Point(-1, dop)).check_singularity()
            Traceback (most recent call last):
            ...
            ValueError: Step 2*i + 1 --> -1 passes through or too close to
            singular point 1*I (to compute the connection to a singular point,
            make it a vertex of the path)
        """
        sing = self.singularities()
        if len(sing) > 0:
            plural = "" if len(sing) == 1 else "s"
            sings = ", ".join(str(self.start.dop._sing_as_alg(s)) for s in sing)
            raise ValueError(
                "Step {} passes through or too close to singular point{} {} "
                "(to compute the connection to a singular point, make it "
                "a vertex of the path)".format(self, plural, sings))

    def check_convergence(self):
        r"""
        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import *
            sage: Dops, x, Dx = DifferentialOperators()

            sage: Path([0, 1], x*(x^2+1)*Dx).check_convergence()
            Traceback (most recent call last):
            ...
            ValueError: Step 0 --> 1 escapes from the disk of (guaranteed)
            convergence of the solutions at regular singular point 0

            sage: path = Path([1, 0], x*(x^2+1)*Dx, two_point_mode=False)
            sage: path.check_convergence()
            Traceback (most recent call last):
            ...
            ValueError: Step 1 <-- 0 escapes from the disk of (guaranteed)
            convergence of the solutions at regular singular point 0
        """
        if not self.length() < self.start.dist_to_sing():
            raise ValueError("Step {} escapes from the disk of (guaranteed) "
                    "convergence of the solutions at {}"
                    .format(self, self.start.descr()))

    def plot(self):
        return plot.arrow2d(self.start.iv().mid(), self.end.iv().mid())

class Path(SageObject):
    """
    A path in ℂ or on the Riemann surface of some operator.

    Note that paths are not the only potentially interesting analytic
    continuation plans: we may reuse already computed transition matrices!

    EXAMPLES::

        sage: from ore_algebra import *
        sage: from ore_algebra.analytic.path import Path
        sage: from ore_algebra.analytic.differential_operator import DifferentialOperator
        sage: Dops, x, Dx = DifferentialOperators()
        sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

        sage: path = Path([0, 1+I, CBF(2*I)], DifferentialOperator(dop),
        ....:             two_point_mode=True)
        sage: path
        0 --> I + 1 --> ~2.0000*I
        sage: path.vert[0]
        0
        sage: len(path)
        2
        sage: list(path.steps())
        [0 <-- I + 1, I + 1 --> ~2.0000*I]
        sage: path.dop
        (x^2 + 1)*Dx^2 + 2*x*Dx

        sage: path.check_singularity()
        True
        sage: path.check_convergence()
        Traceback (most recent call last):
        ...
        ValueError: Step 0 <-- I + 1 escapes from the disk of (guaranteed)
        convergence of the solutions at I + 1
    """

    def __init__(self, vert, dop, two_point_mode=False):
        r"""
        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Path
            sage: Dops, x, Dx = DifferentialOperators()
            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

            sage: Path([], Dx)
            Traceback (most recent call last):
            ...
            ValueError: empty path
        """
        SageObject.__init__(self)
        self.dop = dop
        if not vert:
            raise ValueError("empty path")
        self.vert = []
        for v in vert:
            if isinstance(v, Point):
                pass
            elif isinstance(v, list) and len(v) == 1:
                v = Point(v[0], dop, keep_value=True)
            else:
                v = Point(v, dop)
            self.vert.append(v)
        self.two_point_mode = two_point_mode

    def steps_direct(self):
        for a, b in pairwise(self.vert):
            yield Step(a, b, branch=a.options.get("outgoing_branch"))

    def steps(self):
        r"""
        Iterate over the steps of this path.

        With ``orient=True``, steps with at least one non-singular end are
        oriented from the expansion point to the evaluation point.
        """
        # XXX At the moment, what we are doing here needs to be consistent with
        # the behavior of subdivide(). Consider having subdivide() provide
        # enough information to determine the orientation of all steps.
        reverse = self.two_point_mode
        for a, b in pairwise(self.vert):
            branch = a.options.get("outgoing_branch")
            if a.is_singular():
                # XXX Consider forbidding steps from a singular point to another
                # if b.is_singular():
                #     raise ValueError
                reverse = False
            elif b.is_singular():
                reverse = True
            if reverse:
                a, b = b, a
            assert a.options.get("action") in (None, "expand")
            assert b.options.get("action") in (None, "connect")
            yield Step(a, b, reversed=reverse, branch=branch)
            reverse = self.two_point_mode and not reverse

    def __len__(self):
        return len(self.vert) - 1

    def _repr_(self):
        return " --> ".join(str(v) for v in self.vert)

    def short_repr(self):
        arrow = " --> " if len(self.vert) < 2 else " --> ... --> "
        return repr(self.vert[0]) + arrow + repr(self.vert[-1])

    def plot(self, disks=False, **kwds):
        gr  = plot.point2d(self.dop._singularities(CC),
                           marker='*', color='red', **kwds)
        gr += plot.line([z.iv().mid() for z in self.vert])
        gr.set_aspect_ratio(1)
        if disks:
            for step in self.steps():
                z = step.start.iv().mid()
                gr += plot.circle((z.real(), z.imag()),
                                  step.start.dist_to_sing().lower(),
                                  linestyle='dotted', color='red')
                gr += plot.circle((z.real(), z.imag()),
                                  step.length().lower(),
                                  linestyle='dashed')
        return gr

    @cached_method
    def check_singularity(self):
        """
        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Path
            sage: Dops, x, Dx = DifferentialOperators()
            sage: QQi.<i> = QuadraticField(-1, 'i')
            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

            sage: Path([0], dop).check_singularity()
            True
            sage: Path([1,3], dop).check_singularity()
            True
            sage: Path([0, i], dop).check_singularity()
            True

            sage: Path([42, 1+i/2, -1+3*i/2], dop).check_singularity()
            Traceback (most recent call last):
            ...
            ValueError: Step 1/2*i + 1 --> 3/2*i - 1 passes through or too close
            to singular point 1*I (to compute the connection to a singular
            point, make it a vertex of the path)

        TESTS:

        Check that we detect additional singular points on path segments with
        regular singular endpoints. Adapted from a NumGfun bug found by
        Christoph Koutschan. ::

            sage: dop = (-8*x^3+4*x^4+5*x^2-x)*Dx + 10*x^2-4*x-8*x^3+1
            sage: dop.numerical_transition_matrix([0,1])
            Traceback (most recent call last):
            ...
            ValueError: ...

        Multiple singular points along a single edge::

            sage: (((x-1)*Dx-1)*((x-2)*Dx-2)).numerical_transition_matrix([0,3])
            Traceback (most recent call last):
            ...
            ValueError: Step 0 --> 3 passes through or too close to singular
            points 1, 2...
        """
        for step in self.steps_direct():
            step.check_singularity()
        return True # @cached_method doesn't cache None

    def check_convergence(self):
        """
        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Path
            sage: Dops, x, Dx = DifferentialOperators()
            sage: dop = x*(x^2 + 1)*Dx^2 + 2*x*Dx
            sage: Path([0, 1], dop).check_convergence()
            Traceback (most recent call last):
            ...
            ValueError: Step 0 --> 1 escapes from the disk of (guaranteed)
            convergence of the solutions at regular singular point 0
            sage: Path([0, 1/2], dop).check_convergence()
        """
        for step in self.steps():
            step.check_convergence()

    # Path rewriting

    def bypass_singularities(self):
        r"""
        TESTS::

            sage: from ore_algebra import *
            sage: Dops, x, Dx = DifferentialOperators()
            sage: ((x-1)*Dx - 1).numerical_solution([1], [0,2], assume_analytic=True)
            [-1.0000000000000...] + [+/- ...]*I

            sage: dop = ((x - 1)*Dx - 1)*((x - 2)*Dx - 2)
            sage: dop.numerical_solution([1, 0], [0, 3], assume_analytic=True)
            [-3.5000000000000...] + [+/- ...]*I

            sage: QQi.<i> = QuadraticField(-1)
            sage: dop = ((x - i - 1)*Dx - 1)*((x - 2*i - 2)*Dx - 2)
            sage: dop.numerical_solution([1, 0], [0, 3*i + 3], assume_analytic=True)
            [-3.5000000000000...] + [+/- ...]*I
        """
        new = []
        for a, b in pairwise(self.vert):
            new.append(a)
            step = Step(a, b)
            dir = step.direction()
            sings = step.singularities()
            sings.sort(key=lambda s: (s-a.iv()).abs())
            for s in sings:
                ds = Point(s, self.dop, singular=True).dist_to_sing()
                d0 = abs(s - a.iv())
                d1 = abs(s - b.iv())
                zs = []
                if not safe_lt(d0, ds):
                    zs.append(-1)
                zs.append(IC.gen(0))
                if not safe_lt(d1, ds):
                    zs.append(1)
                rad = (ds/2).min(d0, d1)
                new.extend([_rationalize(CIF(s + rad*z*dir)) for z in zs])
        new.append(self.vert[-1])
        new = Path(new, self.dop)
        return new

    def _intermediate_points(self, a, b, npoints, factor=IR(0.5),
                             rel_tol=IR(0.125)):
        r"""
        Find one or two intermediate points along [a,b] suitable for summing
        local solutions.

        * In one-point mode, we look for a point m such that the series at a is
          converges reasonably fast at m, and m is a good base for a new series
          expansion.

        * In two-point mode, we look for points m and c such that the series
          at m converges reasonably fast at both a and c.

        In both cases, the function may return fewer than npoints points if
        a and b are close.

        TESTS:

        Thanks to Eric Pichon for this example where subdivision used to fail::

            sage: from ore_algebra.analytic.examples.misc import pichon1_dop as dop
            sage: step = [-1, 0.14521345101433106 - 0.1393025865960824*I]
            sage: dop.numerical_transition_matrix(step, eps=2^(-100))[0,0]
            [4.1258139030954317085986778073...] + [-1.3139743385825164244545395830...]*I

        ...and for this one, showing that step subdivision could silently change
        the homotopy class of the path, leading to incorrect results::

            sage: l = [0, -0.05*I, -0.6-0.05*I, -0.8-0.05*I, -0.7-0.1*I,
            ....:      -0.6-0.05*I, -0.05*I, 0]
            sage: dop.numerical_transition_matrix(l + list(reversed(l)))
            [[1.000000000...] + [+/- ...]*I        [+/- ...] + [+/- ...]*I]
            [       [+/- ...] + [+/- ...]*I   [1.0000000...] + [+/- ...]*I]

        Simple examples demonstrating essentially the same issue::

            sage: from ore_algebra import OreAlgebra
            sage: Pol.<x> = QQ[]
            sage: Dop.<Dx> = OreAlgebra(Pol)
            sage: ((x - 1)*Dx - 1).numerical_transition_matrix([0, 100+i/10])
            [[-99.0000000000...] + [-0.100000000000...]*I]
            sage: ((x^2 + 1)*Dx - 1).numerical_transition_matrix([0, 200*i-1/7])
            [[0.207877720258...] + [0.0010394053945...]*I]
        """
        assert npoints in (1, 2)

        vec = b.iv() - a.iv()
        length = abs(vec)
        dir = vec/abs(vec)
        is_real = a.iv().imag().is_zero() and b.iv().imag().is_zero()

        t = ~factor
        if npoints == 1:
            newlength = factor*a.dist_to_sing()
        elif npoints == 2:
            den = t**2 - 1
            newlength = length

        channel_half_width = a.dist_to_sing()
        for s in self.dop._singularities(IC):
            h = (s - a.iv())/dir
            xs, ys = h.real(), h.imag()
            # Find the closest singularity (if any) that projects orthogonally
            # on the step that we are splitting...
            if not (IR.zero() > xs) and not (xs > length):
                channel_half_width = min(channel_half_width, ys.below_abs())
            # ...and the largest disk centered on [a, b], with a on the
            # boundary, that is free of singularities (newlength = factor*radius
            # of this disk)
            if npoints == 2:
                newlength1 = (((t*abs(h))**2 - ys**2).sqrt() - xs)/den
                if newlength1 < newlength:
                    newlength = newlength1

        # if npoints == 2:
        #     _m0 = a.iv() + newlength*dir
        #     _dist = min(abs(_m0-s) for s in self.dop._singularities(IC))
        #     assert (factor*_dist).overlaps(newlength)

        if npoints == 2 and newlength < length < 2*newlength:
            newlength = length/2

        res = []
        for p in range(1, npoints + 1):

            if not p*newlength*(1 + rel_tol) < length:
                # Better go straight to point b
                break

            m0 = a.iv() + p*newlength*dir

            # Attempt to make m0 real and/or a simple Gaussian rational by
            # perturbing it a little. To preserve the homotopy class of the
            # path, the angular perturbation needs to be small (absolute
            # perturbations <= channel_half_width/√2, up to numerical errors,
            # are safe). However, we allow for larger perturbations in the
            # direction of the step. This is interesting mainly for steps along
            # one of the axes.
            for i in reversed(range(3)):
                delta = dir*IC(p*newlength.add_error(rel_tol*newlength),
                                IR.zero().add_error(rel_tol*channel_half_width))
                rel_tol = rel_tol**2 if i else IR(0.)
                # The condition p*newlength*(1 + rel_tol) < length holds and
                # implies |δ| < length in exact arithmetic, but, due to the
                # wrapping effect in the multiplication by dir, the interval
                # version may not hold.
                if not abs(delta) < length:
                    continue
                m = a.iv() + delta
                r = _rationalize(m, is_real)
                if is_real and a.iv().real() < r < b.iv().real():
                    break
                # Check that the homotopy class of the path did not change
                c = Point(m0.union(r), self.dop)
                try:
                    Step(a, c).check_singularity()
                    Step(c, b).check_singularity()
                    break
                except ValueError:
                    logger.debug(
                        "homotopy check failed (m0=%s, m=%s, r=%s), "
                        "trying again with rel_tol=%s",
                        m0, m, r, rel_tol)
            else:
                raise ValueError(f"failed to subdivide (sub)step {a}-->{b}")

            # At the moment the action parameter is for debugging purposes; it
            # is not used to choose the actual action taken
            if npoints == 2:
                action = "connect" if p == 2 else "expand"
            else:
                action = None
            res.append(Point(r, self.dop, action=action))

        return res

    def subdivide(self, mode, thr=IR(0.6)):
        # TODO:
        # - support paths passing very close to singular points
        # - choose orientations in a more clever way (single-step paths, last
        #   step, point complexity...)
        # XXX Needs to stay in sync with steps().
        assert mode in (1, 2)
        new = [self.vert[0]]
        npoints = mode
        skip = self.vert[0].is_singular() # force npoints = 1 for 1st iteration
        logger.debug("subdividing, new path = %s (skip=%s) ...", new, skip)
        i = 1
        while i < len(self.vert):
            cur, next = new[-1], self.vert[i]
            logger.debug("(%s --> %s)", cur, next)
            dist_to_next = abs(next.iv() - cur.iv())
            # In two-point mode, the path alternates between "expansion" and
            # "connection" points. The last inserted point new[-1] at the
            # beginning of an iteration is usually a connection point, except,
            # in some cases, when skip is True (initially when the starting
            # point is singular, and at the beginning of a new edge when the
            # vertex is in "expansion" position).
            npoints = mode + 1 - npoints if skip else mode
            skip = False
            if (npoints == 1 and next.is_ordinary()
                    and dist_to_next <= thr*cur.dist_to_sing()
                 or cur.is_ordinary() and (next.is_singular() or npoints == 2)
                    and dist_to_next <= thr*next.dist_to_sing()):
                # We are very close to the target, do not insert any new point.
                # In two-point mode, this means that next will be inserted into
                # the new path in "expansion" position. Note that this is the
                # only way in which we can reach a singular vertex of the input
                # path.
                skip = True
            else:
                newpts = self._intermediate_points(cur, next, npoints)
                logger.debug("... + %s", newpts)
                new.extend(newpts)
            if skip or len(newpts) < npoints:
                logger.debug("... + %s (skip=%s)", next, skip)
                new.append(next)
                i += 1
        new = Path(new, self.dop, two_point_mode=(mode == 2))
        return new

    def simplify_points_add_detours(self, ctx=dctx):
        n = len(self.vert)
        new = [None]*n
        for i in range(n):
            neighb = []
            if i > 0:
                neighb.append(self.vert[i-1])
            if i < n - 1:
                neighb.append(self.vert[i+1])
            new[i] = self.vert[i].simple_approx(neighb, ctx=ctx)
        return Path(new, self.dop, two_point_mode=self.two_point_mode)

    def _sing_connect(self, sing, other, thr=IR(0.6)):
        if sing.is_ordinary():
            return sing
        rad = sing.dist_to_sing()
        dist = (sing.iv() - other.iv()).abs()
        if safe_le(dist, thr*rad):
            return other
        else:
            [via] = self._intermediate_points(sing, other, 1)
            return via

    def deform(self):

        # Sentinels for the rigorous homotopy check
        # (we need to use exactly the same list of sentinels for both crossing
        # sequences at each stage)
        sing = self.dop._singularities(IC)
        sentinels = [z.squash() for z in sing]
        sentinels.sort(key=lambda z: (-z.imag(), z.real()))

        new = [self.vert[0]]
        l = len(self.vert)
        i = 0
        # XXX slow vs fast points?
        while i < l - 1:
            j = i + 1

            # Split the path at singular vertices and at vertices that we have
            # to pass through to record the value of the solution
            while (j < l - 1 and self.vert[j].is_ordinary()
                   and not self.vert[j].keep_value()):
                j += 1
            subpath = self.vert[i:j+1]
            # Replace singular endpoints by intermediate points on the step
            # leading to or coming from them.
            subpath[0] = self._sing_connect(subpath[0], subpath[1])
            subpath[-1] = self._sing_connect(subpath[-1], subpath[-2])

            # Heuristic deformation of the subpath
            pdef = PathDeformer(subpath, self.dop)

            # Transform the floating-point vertices returned by the PathDeformer
            # into exact or interval points. For the extreme points of the
            # deformed path, use the corresponding input points instead, except
            # when we had to insert intermediate points to deal with
            # singularities.
            new_subpath = []
            for z in pdef.result:
                # XXX do we really want to rationalize the vertices?
                # XXX even if we do, consider doing that after the homotopy
                # check
                rad = min(abs(z - w) for w in pdef.sing)
                z = CBF(z).add_error(rad/16.)
                z = _rationalize(z, z.imag().contains_zero())
                new_subpath.append(Point(z, self.dop))

            # Rigorous homotopy check. We first check in interval arithmetic
            # that none of the “thick segments” with interval endpoints making
            # up our two paths goes through any of the singularities. If that is
            # the case, we can conclude by comparing crossing sequences computed
            # by replacing each of these intervals (both path vertices and
            # singularities) by a point contained in it.
            ini_iv = [z.iv() for z in subpath]
            new_iv = [z.iv() for z in new_subpath]
            for s in sing:
                for z0, z1 in itertools.chain(pairwise(ini_iv),
                                              pairwise(new_iv)):
                    if may_be_on_segment(z0, z1, s):
                        raise PathDeformationFailed("too close to singularities"
                                " for rigorous homotopy check")
            ini_cs = crossing_sequence(sentinels, ini_iv)
            new_cs = crossing_sequence(sentinels, new_iv)
            if ini_cs != new_cs:
                raise PathDeformationFailed("homotopy test failed", subpath,
                        new_subpath)

            if subpath[0] is not self.vert[i]:
                new.append(subpath[0])
            new.extend(new_subpath[1:-1])
            if subpath[-1] is not self.vert[j]:
                new.append(subpath[-1])
            new.append(self.vert[j])

            i = j

        return Path(new, self.dop)

    def deform_or_subdivide(self, mode):
        if mode == 2:
            raise NotImplementedError
        if not self.dop._singularities(IC) or len(self.vert) <= 2:
            return self.subdivide(mode)
        try:
            new = self.deform()
            logger.info("path homotopy succeeded, old=%s, new=%s, sub=%s", self, new, self.subdivide(mode))
            return new
        except PathDeformationFailed:
            logger.warning("path homotopy failed, falling back to subdivision")
            return self.subdivide(mode)

def orient2d_det(a, b, c):
    return ((b.real() - a.real())*(c.imag() - a.imag())
            - (c.real() - a.real())*(b.imag() - a.imag()))

def orient2d(a, b, c):
    det = orient2d_det(a, b, c)
    zero = det.parent().zero()
    if safe_gt(det, zero):
        return 1
    elif safe_lt(det, zero):
        return -1
    elif det.is_zero():
        return 0
    else:
        raise ValueError

def may_be_on_segment(a, b, z):
    if a.overlaps(b):
        return z.overlaps(a.union(b))
    if orient2d_det(a, b, z).is_nonzero():
        return False
    az, bz = z - a, z - b
    dot = az.real()*bz.real() + az.imag()*bz.imag()
    return not safe_gt(dot, dot.parent().zero())

def crossing_sequence(sentinels, path):
    # Compute the reduced crossing sequence with horizontal cuts to the left of
    # sentinel points at the center of each singular ball.
    seq = []
    def append(i):
        if seq and seq[-1] == -i:
            seq.pop()
        else:
            seq.append(i)
    path = [z.squash() for z in path]
    sentinels_enum = list(enumerate(sentinels, start=1))
    for z0, z1 in pairwise(path):
        if safe_ge(z0.imag(), z1.imag()):
            # crossing from above
            for i, w in sentinels_enum:
                if (safe_ge(z0.imag(), w.imag()) and safe_lt(z1.imag(), w.imag())
                        and orient2d(z0, z1, w) == +1):
                    append(i)
        else:
            # crossing from below
            for i, w in reversed(sentinels_enum):
                if (safe_lt(z0.imag(), w.imag()) and safe_ge(z1.imag(), w.imag())
                        and orient2d(z0, z1, w) == -1):
                    append(-i)
    return seq

def local_monodromy_path(sing):
    raise NotImplementedError

def polygon_around(point, size=17):
    # not ideal in the case of a single singularity...
    rad = (point.dist_to_sing()/2).min(1)
    polygon = []
    for k in range(size):
        x = point.iv() + rad*(CBF(2*k)/size).exppii()
        # XXX balls are not supported (or don't work well) as
        # starting/intermediate points
        if not Point(x, point.dop).is_ordinary():
            raise PathPrecisionError
        x = _rationalize(IC(x))
        polygon.append(Point(x, point.dop))
    return polygon

def _rationalize(civ, real=False):
    from sage.rings.real_mpfi import RealIntervalField
    my_RIF = RealIntervalField(civ.real().parent().precision())
    # XXX why can't we do this automatically when civ.imag().contains_zero()???
    if real or civ.imag().is_zero():
        return my_RIF(civ.real()).simplest_rational()
    else:
        return QQi([my_RIF(civ.real()).simplest_rational(),
                    my_RIF(civ.imag()).simplest_rational()])
