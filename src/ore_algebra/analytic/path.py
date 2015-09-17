# -*- coding: utf-8 - vim: tw=80
"""
Analytic continuation paths

Main class hierarchy for points:

    Point > RegularPoint > OrdinaryPoint

Auxiliary classes, for classification purposes: RegularSingularPoint,
IrregularSingularPoint.

(Algorithms for, say, regular singular points should typically work for ordinary
points as well even when a more specialized version exists. One could have
QuasiRegularPoints too, but combined with SingularPoints that would make the
whole thing too complicated.)

FIXME: silence deprecation warnings::

    sage: def ignore(*args): pass
    sage: sage.misc.superseded.warning=ignore
"""

import logging

import sage.categories.pushout as pushout
import sage.plot.all as plot
import sage.rings.all as rings
import sage.rings.number_field.number_field as number_field
import sage.rings.number_field.number_field_base as number_field_base
import sage.symbolic.ring

from sage.misc.cachefunc import cached_method
from sage.rings.all import QQ, CC, RIF, CIF, QQbar, RLF, CLF
from sage.rings.complex_ball_acb import CBF, ComplexBallField
from sage.rings.real_arb import RBF, RealBallField
from sage.structure.sage_object import SageObject

from ore_algebra.analytic.utilities import *

logger = logging.getLogger(__name__)

IR, IC = RBF, CBF # TBI
QQi = number_field.QuadraticField(-1, 'i')

######################################################################
# Points
######################################################################

class Point(SageObject):
    r"""
    A point on the complex plane, in relation with a differential operator.

    Though a differential operator is part of the definition of an instance,
    this class represents a point on the complex plane, not on the Riemann
    surface of the operator.

    A point can be exact (a number field element) or inexact (a real or complex
    interval or ball).

    It can be classified as ordinary, regular singular, etc.

    NOTES:

    - Decouple pure complex points from things that depend on the operator?
    """

    def __init__(self, point, dop=None):
        """
        TESTS::

            sage: from ore_algebra.analytic.ui import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = Diffops()
            sage: [Point(z, Dx) 
            ....:  for z in [1, 1/2, 1+I, QQbar(I), RIF(1/3), CIF(1/3), pi]]
            [1, 1/2, 1 + 1*I, I, 0.3333333333333334?, 0.3333333333333334?,
            3.141592653589794?]
        """
        SageObject.__init__(self)

        from sage.rings.real_mpfr import RealField_class
        from sage.rings.complex_field import ComplexField_class
        from sage.rings.real_mpfi import RealIntervalField_class
        from sage.rings.complex_interval_field import ComplexIntervalField_class
        parent = point.parent()
        if isinstance(point, Point):
            self.value = point.value
        elif isinstance(parent, (
                number_field_base.NumberField,
                RealBallField, ComplexBallField)):
            self.value = point
        elif QQ.has_coerce_map_from(parent):
            self.value = QQ.coerce(point)
        # must come before QQbar, due to a bogus coerce map
        elif parent is sage.symbolic.ring.SR:
            try:
                self.value = RLF(point)
            except TypeError:
                self.value = CLF(point)
        elif QQbar.has_coerce_map_from(parent):
            self.value = QQbar.coerce(point).as_number_field_element()[1]
        elif isinstance(parent, (RealField_class, RealIntervalField_class)):
            self.value = rings.RealBallField(point.prec())(point)
        elif isinstance(parent, (ComplexField_class,
                                 ComplexIntervalField_class)):
            self.value = rings.ComplexBallField(point.prec())(point)
        else:
            try:
                self.value = RLF.coerce(point)
            except TypeError:
                self.value = CLF.coerce(point)

        self.dop = dop or point.dop

        self.keep_value = False

    def _repr_(self):
        """
        TESTS::

            sage: from ore_algebra.analytic.ui import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = Diffops()
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
        # XXX: est-ce que je voudrais prendre une précision ici ?
        return IC(self.value)

    def exact(self):
        if self.is_exact():
            return self
        raise ValueError

    def n(self):
        try:
            return CC(self.value)
        except TypeError: pass
        return CC(self.value.center())

    ###

    def is_real(self):
        return RIF.has_coerce_map_from(self.value.parent())

    def is_exact(self):
        return isinstance(self.value, rings.NumberFieldElement)

    # Autre modèle possible, peut-être mieux : Point pourrait être juste un
    # point du plan complexe (ne pas connaître l'opérateur), et ce serait alors
    # les contextes de prolongement analytique qui auraient les méthodes qui
    # suivent. (Les points classifiés pourraient éventuellement connaître
    # l'opérateur... s'ils ont encore une raison d'être...)

    def dist_to_sing(self):
        """
        TESTS::

            sage: from ore_algebra.analytic.ui import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = Diffops()
            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
            sage: Point(1, dop).dist_to_sing()
            1.4142135623730950?
            sage: Point(i, dop).dist_to_sing()
            0
            sage: Point(1+i, dop).dist_to_sing()
            1

        """
        # TODO - solve over CBF directly; perhaps with arb's own poly solver
        dist = [(self.iv() - IC(s)).abs()
                for s in dop_singularities(self.dop, CIF)]
        min_dist = IR(rings.infinity).min(*dist)
        if min_dist.contains_zero():
            if self.dop.leading_coefficient()(self.value).is_zero():
                return IR(0) # ?
            else:
                raise NotImplementedError # refine???
        return IR(min_dist.lower())

    def local_diffop(self): # ?
        Pols_dop = self.dop.base_ring()
        # NOTE: pushout(QQ[x], K) doesn't handle embeddings well, and creates
        # an L equal but not identical to K. But then other constructors like
        # PolynomialRing(L, x) sometimes return objects over K found in cache,
        # leading to endless headaches with slow coercions. But the version here
        # may be closer to what I really want in any case.
        # XXX: This seems to work in the usual trivial case where we are looking
        # for a scalar domain containing QQ and QQ[i], but probably won't be
        # enough if we really have two different number fields with embeddings
        Scalars = pushout.pushout(Pols_dop.base_ring(), self.value.parent())
        Pols = Pols_dop.change_ring(Scalars)
        A, B = self.dop.base_ring().base_ring(), self.value.parent()
        C = Pols.base_ring()
        assert C is A or C != A
        assert C is B or C != B
        dop_P = self.dop.change_ring(Pols)
        return dop_P.annihilator_of_composition(Pols([self.value, 1]))

    def classify(self):
        r"""
        EXAMPLES::

            sage: from ore_algebra.analytic.ui import *
            sage: from ore_algebra.analytic.path import Point
            sage: Dops, x, Dx = Diffops()

            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx
            sage: type(Point(1, dop).classify())
            <class 'ore_algebra.analytic.path.OrdinaryPoint'>
            sage: type(Point(i, dop).classify())
            <class 'ore_algebra.analytic.path.RegularSingularPoint'>
            sage: type(Point(0, x^2*Dx + 1).classify())
            <class 'ore_algebra.analytic.path.IrregularSingularPoint'>

        TESTS::

            sage: type(Point(CIF(1/3), x^2*Dx + 1).classify())
            <class 'ore_algebra.analytic.path.OrdinaryPoint'>
            sage: type(Point(CIF(1/3)-1/3, x^2*Dx + 1).classify())
            <class 'ore_algebra.analytic.path.Point'>
        """
        if self.value.parent().is_exact():
            if self.dop.leading_coefficient()(self.value):
                return OrdinaryPoint(self)
            # Fuchs criterion
            Pols = self.dop.base_ring().change_ring(self.value.parent())
            def val(pol):
                return Pols(pol).valuation(Pols([self.value, -1]))
            ref = val(self.dop.leading_coefficient()) - self.dop.order()
            if all(val(coef) - k >= ref
                   for k, coef in enumerate(self.dop)):
                return RegularSingularPoint(self)
            else:
                return IrregularSingularPoint(self)
        elif is_interval_field(self.value.parent()):
            if not self.dop.leading_coefficient()(self.value).contains_zero():
                return OrdinaryPoint(self)
        return self

class RegularPoint(Point):

    def connect_to_ordinary(self):
        raise NotImplementedError

class OrdinaryPoint(RegularPoint):
    pass

class RegularSingularPoint(RegularPoint):
    pass

class IrregularSingularPoint(Point):
    pass

######################################################################
# Paths
######################################################################

class Step(SageObject):

    def __init__(self, start, end):
        if start.dop != end.dop:
            raise ValueError
        self.start = start
        self.end = end

    def __getitem__(self, i):
        if i == 0:
            return self.start
        elif i == 1:
            return self.end
        else:
            raise IndexError

    def _repr_(self):
        return repr(self.start) + " --> " + repr(self.end)

    def delta(self):
        return self.end.value - self.start.value

    def length(self):
        return IC(self.delta()).abs()

    def check_singularity(self):
        dop = self.start.dop
        # TODO: solve over CBF directly?
        sing = [IC(s) for s in dop_singularities(dop, CIF)]
        for s in sing:
            t = self.delta()/(s - self.start.iv())
            if t.imag().contains_zero() and not safe_lt(t.real(), IR.one()):
                raise ValueError(
                    "Step {} passes through or too close to singular point {} "
                    # "(to compute the connection to a singular point, specify "
                    # "it explicitly as a vertex)"
                    .format(self, sing_as_alg(dop, s)))

    def check_convergence(self):
        if self.length() >= self.start.dist_to_sing(): # not < ?
            raise ValueError("Step {} escapes from the disk of (guaranteed) "
                    "convergence of the solutions at {}"
                    .format(self, self.start))

    def plot(self):
        return plot.arrow2d(self.start.n(), self.end.n())

    def is_exact(self):
        return self.start.is_exact() and self.end.is_exact()

class Path(SageObject):
    """
    A path in ℂ or on the Riemann surface of some operator.

    Note that any analytic continuation plan is not necessarily a path (we may
    use a given transition matrix several times after computing it!).

    EXAMPLES::

        sage: from ore_algebra.analytic.ui import *
        sage: from ore_algebra.analytic.path import Path
        sage: Dops, x, Dx = Diffops()
        sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

        sage: path = Path([0, 1+I, CIF(2*I)], dop)
        sage: path
        0 --> 1 + 1*I --> 2*I
        sage: path[0]
        0 --> 1 + 1*I
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
        ValueError: Step 1 + 1*I --> 2*I escapes from the disk of (guaranteed)
        convergence of the solutions at 1 + 1*I
    """

    def __init__(self, vert, dop, classify=False):
        """
        TESTS::

            sage: from ore_algebra.analytic.ui import *
            sage: from ore_algebra.analytic.path import Path
            sage: Dops, x, Dx = Diffops()
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
        if classify:
            self.vert = [v.classify() for v in self.vert]

    def __getitem__(self, i):
        "Return the i-th step of self"
        if len(self.vert) < 2:
            raise IndexError
        else:
            return Step(self.vert[i], self.vert[i+1])

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
                z = step.start.n()
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

            sage: from ore_algebra.analytic.ui import *
            sage: from ore_algebra.analytic.path import Path
            sage: Dops, x, Dx = Diffops()
            sage: QQi.<i> = QuadraticField(-1, 'i')
            sage: dop = (x^2 + 1)*Dx^2 + 2*x*Dx

            sage: Path([0], dop).check_singularity()
            sage: Path([1,3], dop).check_singularity()

            sage: Path([42, 1+i/2, -1+3*i/2], dop).check_singularity()
            Traceback (most recent call last):
            ...
            ValueError: Step 1/2*i + 1 --> 3/2*i - 1 passes through or too close
            to singular point 1*I 

            sage: Path([0, i], dop).check_singularity()
            Traceback (most recent call last):
            ...
            ValueError: Step 0 --> i passes through or too close to singular
            point 1*I
        """
        for step in self:
            step.check_singularity()

    def check_convergence(self):
        """
        EXAMPLES::

            sage: from ore_algebra.analytic.ui import *
            sage: from ore_algebra.analytic.path import Path
            sage: Dops, x, Dx = Diffops()
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
    # - On veut garder le chemin initial et le chemin optimisé. Je ne sais pas
    #   si on veut optimiser en place ou faire une copie.

    def subdivide(self, threshold=IR(0.75), factor=IR(0.5)):
        # TODO:
        # - support paths passing very close to singular points
        # - create intermediate points of smaller bit size
        new = [self.vert[0]]
        i = 1
        while i < len(self.vert):
            cur, next = new[-1], self.vert[i]
            rad = cur.dist_to_sing()
            dist_to_next = (next.iv() - cur.iv()).abs()
            if dist_to_next <= threshold*rad:
                new.append(next)
                i += 1
            else:
                dir = (next.iv() - cur.iv())/dist_to_next
                interm = cur.iv() + factor*rad*dir
                # TBI
                Step(cur, Point(interm, self.dop)).check_singularity()
                if interm.imag().is_zero():
                    # interm = interm.real().simplest_rational()
                    interm = interm.real().mid().exact_rational()
                else:
                    # interm = QQi([interm.real().simplest_rational(),
                    #               interm.imag().simplest_rational()])
                    interm = QQi([interm.real().mid().exact_rational(),
                                  interm.imag().mid().exact_rational()])
                new.append(OrdinaryPoint(interm, self.dop))
        new = Path(new, self.dop)
        return new

    def find_loops(self): # ???
        raise NotImplementedError

    def optimize_by_homotopy(self):
        raise NotImplementedError

    def remove_duplicates(self):
        raise NotImplementedError

    def bit_burst(self, z0, z1):
        raise NotImplementedError

    def connect_sing(self, XXX): # ???
        raise NotImplementedError

def local_monodromy_path(sing):
    raise NotImplementedError
