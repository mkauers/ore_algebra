# -*- coding: utf-8 - vim: tw=80
"""
Analytic continuation paths
"""

import logging

import sage.categories.pushout as pushout
import sage.plot.all as plot
import sage.rings.all as rings
import sage.rings.number_field.number_field as number_field
import sage.rings.number_field.number_field_base as number_field_base
import sage.structure.coerce
import sage.symbolic.ring

from sage.misc.cachefunc import cached_method
from sage.rings.all import ZZ, QQ, CC, RIF, CIF, QQbar, RLF, CLF, Integer
from sage.rings.complex_arb import CBF, ComplexBallField, ComplexBall
from sage.rings.real_arb import RBF, RealBallField, RealBall
from sage.structure.sage_object import SageObject

from .local_solutions import (FundamentalSolution, sort_key_by_asympt,
        LocalBasisMapper)
from .safe_cmp import *
from .utilities import *

logger = logging.getLogger(__name__)

IR, IC = RBF, CBF # TBI
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

    def __init__(self, point, dop=None, **kwds):
        """
        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = DifferentialOperators()
            sage: [Point(z, Dx)
            ....:  for z in [1, 1/2, 1+I, QQbar(I), RIF(1/3), CIF(1/3), pi,
            ....:  RDF(1), CDF(I), 0.5r, 0.5jr, 10r, QQbar(1), AA(1/3)]]
            [1, 1/2, I + 1, I, [0.333333333333333...], [0.333333333333333...],
            3.141592653589794?, 1.000000000000000, 1.000000000000000*I,
            0.5000000000000000, 0.5000000000000000*I, 10, 1, 1/3]
            sage: Point(sqrt(2), Dx).iv()
            [1.414...]
        """
        SageObject.__init__(self)

        from sage.rings.complex_double import ComplexDoubleField_class
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
        if isinstance(point, Point):
            self.value = point.value
        elif isinstance(parent, (
                number_field_base.NumberField,
                RealBallField, ComplexBallField)):
            self.value = point
        elif QQ.has_coerce_map_from(parent):
            self.value = QQ.coerce(point)
        # must come before QQbar, due to a bogus coerce map (#14485)
        elif parent is sage.symbolic.ring.SR:
            try:
                return self.__init__(point.pyobject(), dop)
            except TypeError:
                pass
            try:
                return self.__init__(QQbar(point), dop)
            except (TypeError, ValueError, NotImplementedError):
                pass
            try:
                self.value = RLF(point)
            except (TypeError, ValueError):
                self.value = CLF(point)
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
        else:
            try:
                self.value = RLF.coerce(point)
            except TypeError:
                self.value = CLF.coerce(point)
        parent = self.value.parent()
        assert (isinstance(parent, (number_field_base.NumberField,
                                    RealBallField, ComplexBallField))
                or parent is RLF or parent is CLF)

        self.dop = dop or point.dop
        self.options = kwds

    def _repr_(self):
        """
        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = DifferentialOperators()
            sage: Point(10**20, Dx)
            ~1.0000e20
        """
        try:
            len = (self.value.numerator().real().numerator().nbits() +
                   self.value.numerator().imag().numerator().nbits() +
                   self.value.denominator().nbits())
            if len > 50:
                return '~' + repr(self.value.n(digits=5))
        except AttributeError:
            pass
        return repr(self.value)

    # Numeric representations

    @cached_method
    def iv(self):
        """
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
        if self.is_exact():
            return self
        elif isinstance(self.value, RealBall) and self.value.is_exact():
            return Point(QQ(self.value), self.dop)
        elif isinstance(self.value, ComplexBall) and self.value.is_exact():
            value = QQi((QQ(self.value.real()), QQ(self.value.imag())))
            return Point(value, self.dop)
        raise ValueError

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
        # XXX: also include exact balls?
        return isinstance(self.value,
                (rings.Integer, rings.Rational, rings.NumberFieldElement))

    # Point equality is identity

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    ### Methods that depend on dop

    @cached_method
    def is_ordinary(self):
        lc = self.dop.leading_coefficient()
        if self.is_exact():
            return not _is_exact_sing(self, self.dop)
        elif not lc(self.iv()).contains_zero():
            return True
        else:
            raise ValueError("can't tell if inexact point is singular")

    def is_singular(self):
        return not is_ordinary(self)

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
        Pols = self.dop.base_ring().change_ring(self.value.parent())
        def val(pol):
            return Pols(pol).valuation(Pols([self.value, -1]))
        ref = val(self.dop.leading_coefficient()) - self.dop.order()
        return all(val(coef) - k >= ref for k, coef in enumerate(self.dop))

    def is_regular_singular(self):
        return not self.is_ordinary() and self.is_regular()

    def is_irregular(self):
        return not is_regular(self)

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
        # TODO - solve over CBF directly; perhaps with arb's own poly solver
        sing = dop_singularities(self.dop, CIF)
        sing = [IC(s) for s in sing]
        close, distant = split(lambda s: s.overlaps(self.iv()), sing)
        if (len(close) >= 2 or len(close) == 1
                               and not _is_exact_sing(self, self.dop)):
            raise NotImplementedError # refine?
        dist = [(self.iv() - s).abs() for s in distant]
        min_dist = IR(rings.infinity).min(*dist)
        if min_dist.contains_zero():
            raise NotImplementedError # refine???
        return IR(min_dist.lower())

    def local_diffop(self): # ?
        r"""
        TESTS::

            sage: from ore_algebra import DifferentialOperators
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = DifferentialOperators()
            sage: Point(1, x*Dx - 1).local_diffop()
            (x + 1)*Dx - 1
            sage: Point(RBF(1/2), x*Dx - 1).local_diffop()
            (x + 1/2)*Dx - 1
        """
        Pols_dop = self.dop.base_ring()
        # NOTE: pushout(QQ[x], K) doesn't handle embeddings well, and creates
        # an L equal but not identical to K. But then other constructors like
        # PolynomialRing(L, x) sometimes return objects over K found in cache,
        # leading to endless headaches with slow coercions. But the version here
        # may be closer to what I really want in any case.
        # XXX: This seems to work in the usual trivial case where we are looking
        # for a scalar domain containing QQ and QQ[i], but probably won't be
        # enough if we really have two different number fields with embeddings
        ex = self.exact()
        Scalars = pushout.pushout(Pols_dop.base_ring(), ex.value.parent())
        Pols = Pols_dop.change_ring(Scalars)
        A, B = self.dop.base_ring().base_ring(), ex.value.parent()
        C = Pols.base_ring()
        assert C is A or C != A
        assert C is B or C != B
        dop_P = self.dop.change_ring(Pols)
        return dop_P.annihilator_of_composition(Pols([ex.value, 1]))

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
        sols = LocalBasisMapper().run(self.local_diffop())
        sols.sort(key=sort_key_by_asympt)
        return sols

######################################################################
# Paths
######################################################################

# XXX: do we need special *Steps* for connections to singular points?
class Step(SageObject):
    r"""
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

    def __init__(self, start, end, branch=(0,)):
        if not (isinstance(start, Point) and isinstance(end, Point)):
            raise TypeError
        if start.dop != end.dop:
            raise ValueError
        self.start = start
        self.end = end
        self.branch = branch

    def _repr_(self):
        return repr(self.start) + " --> " + repr(self.end)

    def __getitem__(self, i):
        if i == 0:
            return self.start
        elif i == 1:
            return self.end
        else:
            raise IndexError

    def is_exact(self):
        return self.start.is_exact() and self.end.is_exact()

    def delta(self):
        try: # XXX: TBI
            return self.end.value - self.start.value
        except TypeError:
            z0 = QQbar(self.start.value)
            z1 = QQbar(self.end.value)
            return as_embedded_number_field_element(z1 - z0)

    def direction(self):
        delta = self.end.iv() - self.start.iv()
        return delta/abs(delta)

    def length(self):
        return IC(self.delta()).abs()

    def cvg_ratio(self):
        return self.length()/self.start.dist_to_sing()

    def singularities(self):
        dop = self.start.dop
        # TODO: solve over CBF directly?
        sing = [IC(s) for s in dop_singularities(dop, CIF)]
        z0, z1 = IC(self.start.value), IC(self.end.value)
        sing = [s for s in sing if s != z0 and s != z1]
        res = []
        for s in sing:
            ds = s - self.start.iv()
            t = self.delta()/ds
            if (ds.contains_zero() or t.imag().contains_zero()
                    and not safe_lt(t.real(), IR.one())):
                res.append(sing_as_alg(dop, s))
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
            sings = ", ".join(str(s) for s in sing)
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

            sage: Path([1, 0], x*(x^2+1)*Dx).check_convergence()
            Traceback (most recent call last):
            ...
            ValueError: Step 1 --> 0 escapes from the disk of (guaranteed)
            convergence of the solutions at regular singular point 0
        """
        ref = self.end if self.end.is_regular_singular() else self.start
        if self.length() >= ref.dist_to_sing(): # not < ?
            raise ValueError("Step {} escapes from the disk of (guaranteed) "
                    "convergence of the solutions at {}"
                    .format(self, ref.descr()))

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
        sage: Dops, x, Dx = DifferentialOperators()
        sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

        sage: path = Path([0, 1+I, CBF(2*I)], dop)
        sage: path
        0 --> I + 1 --> 2.000...*I
        sage: path[0]
        0 --> I + 1
        sage: path.vert[0]
        0
        sage: len(path)
        2
        sage: path.dop
        (x^2 + 1)*Dx^2 + 2*x*Dx

        sage: path.check_singularity()
        sage: path.check_convergence()
        Traceback (most recent call last):
        ...
        ValueError: Step 0 --> I + 1 escapes from the disk of (guaranteed)
        convergence of the solutions at 0
    """

    def __init__(self, vert, dop):
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
        self.vert = [v if isinstance(v, Point) else Point(v, dop)
                     for v in vert]

    def __getitem__(self, i):
        r"""
        Return the i-th step of self
        """
        if len(self.vert) < 2:
            raise IndexError
        else:
            branch = self.vert[i].options.get("outgoing_branch", (0,))
            return Step(self.vert[i], self.vert[i+1], branch)

    def __len__(self):
        return len(self.vert) - 1

    def _repr_(self):
        return " --> ".join(str(v) for v in self.vert)

    def short_repr(self):
        arrow = " --> " if len(self.vert) < 2 else " --> ... --> "
        return repr(self.vert[0]) + arrow + repr(self.vert[-1])

    def plot(self, disks=False):
        gr  = plot.point2d(dop_singularities(self.dop, CC),
                           marker='*', size=200, color='red')
        for step in self:
            gr += step.plot()
        gr.set_aspect_ratio(1)
        if disks:
            for step in self:
                z = step.start.iv().mid()
                gr += plot.circle((z.real(), z.imag()),
                                  step.start.dist_to_sing().lower(),
                                  linestyle='dotted', color='red')
                gr += plot.circle((z.real(), z.imag()),
                                  step.length().lower(),
                                  linestyle='dashed')
        return gr

    def check_singularity(self):
        """
        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Path
            sage: Dops, x, Dx = DifferentialOperators()
            sage: QQi.<i> = QuadraticField(-1, 'i')
            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

            sage: Path([0], dop).check_singularity()
            sage: Path([1,3], dop).check_singularity()
            sage: Path([0, i], dop).check_singularity()

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
        for step in self:
            step.check_singularity()

    def check_convergence(self):
        """
        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.path import Path
            sage: Dops, x, Dx = DifferentialOperators()
            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
            sage: Path([0, 1], dop).check_convergence()
            Traceback (most recent call last):
            ...
            ValueError: Step 0 --> 1 escapes from the disk of (guaranteed)
            convergence of the solutions at 0
            sage: Path([1, 0], dop).check_convergence()
        """
        for step in self:
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
        for step in self:
            new.append(step.start)
            dir = step.direction()
            sings = step.singularities()
            for s in sings:
                ds = Point(s, self.dop).dist_to_sing()
                d0 = abs(IC(s) - step.start.iv())
                d1 = abs(IC(s) - step.end.iv())
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

    def subdivide(self, threshold=IR(0.6), factor=IR(0.5)):
        # TODO:
        # - support paths passing very close to singular points
        new = [self.vert[0]]
        i = 1
        while i < len(self.vert):
            cur, next = new[-1], self.vert[i]
            rad = cur.dist_to_sing()
            dist_to_next = (next.iv() - cur.iv()).abs()
            if (dist_to_next <= threshold*rad if next.is_ordinary()
                else (cur.value == next.value
                      or cur.is_ordinary()
                         and dist_to_next <= threshold*next.dist_to_sing())):
                new.append(next)
                i += 1
            else:
                dir = (next.iv() - cur.iv())/dist_to_next
                interm = cur.iv() + factor*rad*dir
                is_real = interm.imag().is_zero()
                interm = interm.add_error(rad/8)
                Step(cur, Point(interm, self.dop)).check_singularity() # TBI
                interm = _rationalize(interm, is_real)
                new.append(Point(interm, self.dop))
                logger.debug("subdividing %s -> %s", cur, next)
        new = Path(new, self.dop)
        return new

    def find_loops(self): # ???
        raise NotImplementedError

    def optimize_by_homotopy(self):
        raise NotImplementedError

    def bit_burst(self, z0, z1):
        raise NotImplementedError

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
    if real or civ.imag().is_zero():
        return my_RIF(civ.real()).simplest_rational()
    else:
        return QQi([my_RIF(civ.real()).simplest_rational(),
                    my_RIF(civ.imag()).simplest_rational()])

def _is_exact_sing(pt, dop):
    lc = dop.leading_coefficient()
    try:
        val = lc(pt.value)
    except TypeError: # work around coercion weaknesses
        val = lc.change_ring(QQbar)(QQbar.coerce(pt.value))
    return val.is_zero()
