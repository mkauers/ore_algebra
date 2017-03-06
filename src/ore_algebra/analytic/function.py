# -*- coding: utf-8 - vim: tw=80
r"""
D-Finite analytic functions

TESTS::

    sage: from ore_algebra import *
    sage: from ore_algebra.analytic.function import DFiniteFunction
    sage: DiffOps, x, Dx = DifferentialOperators()

    sage: f = DFiniteFunction((x^2 + 1)*Dx^2 + 2*x*Dx, [0, 1])

    sage: [f(10^i) for i in range(-3, 4)] # long time (5.6 s)
    [[0.0009999996666...], [0.0099996666866...], [0.0996686524911...],
     [0.7853981633974...], [1.4711276743037...], [1.5607966601082...],
     [1.5697963271282...]]

    sage: f(0)
    [+/- ...]

    sage: f(-1)
    [-0.78539816339744...]

    sage: (f(1.23456) - RBF(1.23456).arctan()).abs() < RBF(1)>>50
    True
    sage: R200 = RealBallField(200)
    sage: (f(1.23456, 100) - R200(1.23456).arctan()).abs() < (RBF(1) >> 99)
    True

    sage: plot(lambda x: RR(f(x, prec=10)), (-3, 3))
    Graphics object consisting of 1 graphics primitive

    sage: f.approx(0, post_transform=Dx)
    [1.00000000000000...]
    sage: f.approx(2, post_transform=x*Dx+1)
    [1.50714871779409...]
    sage: 2/(1+2^2) + arctan(2.)
    1.50714871779409

    sage: g = DFiniteFunction(Dx-1, [1])

    sage: [g(10^i) for i in range(-3, 4)]
    [[1.001000500166...],
     [1.010050167084...],
     [1.105170918075...],
     [2.718281828459...],
     [22026.46579480...],
     [2.688117141816...e+43...],
     [1.9700711140170...+434...]]

"""

import collections, logging, sys

import sage.plot.all as plot

from sage.plot.plot import generate_plot_points
from sage.rings.all import ZZ, QQ, RBF, CBF, RIF
from sage.rings.complex_arb import ComplexBall, ComplexBallField
from sage.rings.complex_number import ComplexNumber
from sage.rings.real_arb import RealBall, RealBallField
from sage.rings.real_mpfr import RealNumber

from . import analytic_continuation as ancont
from . import bounds
from . import polynomial_approximation as polapprox

from .path import Point
from .safe_cmp import *

logger = logging.getLogger(__name__)

    # NOTES:
    #
    # - Make it possible to “split” a disk (i.e. use non-maximal disks) when the
    #   polynomial approximations become too large???
    #
    # - Introduce separate "Cache" objects?

RealPolApprox = collections.namedtuple('RealPolApprox', ['pol', 'prec'])

class DFiniteFunction(object):
    r"""
    At the moment, this class just provides a simple caching mechanism for
    repeated evaluations of a D-Finite function on the real line. It may
    evolve to support evaluations on the complex plane, branch cuts, ring
    operations on D-Finite functions, and more. Do not expect any API stability.
    """

    # Stupid, but simple and deterministic caching strategy:
    #
    # - To any center c = m·2^k with k ∈ ℤ and m *odd*, we associate the disk of
    #   radius 2^k. Any two disks with the same k have at most one point in
    #   common.
    #
    # - Thus, in the case of an equation with at least one finite singular
    #   point, there is a unique largest disk of the previous collection that
    #   contains any given ordinary x ∈ ℝ \ ℤ·2^(-∞) while staying “far enough”
    #   from the singularities.
    #
    # - When asked to evaluate f at x, we actually compute and store a
    #   polynomial approximation on (the real trace of) the corresponding disk
    #   and/or a vector of initial conditions at its center, which can then be
    #   reused for subsequent evaluations.
    #
    # - We may additionally want to allow disks (of any radius?) centered at
    #   real regular singular points, and perhaps, as a special case, at 0.
    #   These would be used when possible, and one would revert to the other
    #   family otherwise.

    def __init__(self, dop, ini, name="dfinitefun",
                 max_prec=256, max_rad=RBF('inf')):
        self.dop = dop
        if not isinstance(ini, dict):
            ini = {0: ini}
        if len(ini) != 1:
            # In the future, we should support specifying several vectors of
            # initial values.
            raise NotImplementedError
        self.ini = ini
        self.name = name

        # Global maximum width for the approximation intervals. In the case of
        # equations with no finite singular point, we try to avoid cancellation
        # and interval blowup issues by taking polynomial approximations on
        # intervals on which the general term of the series doesn't grow too
        # large. The expected order of magnitude of the “top of the hump” is
        # about exp(κ·|αx|^(1/κ)) and doesn't depend on the base point. We also
        # let the user impose a maximum width, even in other cases.
        self.max_rad = RBF(max_rad)
        if dop.leading_coefficient().is_constant():
            kappa, alpha = _growth_parameters(dop)
            self.max_rad = self.max_rad.min(1/(alpha*RBF(kappa)**kappa))
        self.max_prec = max_prec

        self._inivecs = {}
        self._polys = {}

        self._sollya_object = None
        self._sollya_domain = RIF('-inf', 'inf')
        self._keep_all_derivatives = False
        self._update_approx_hook = (lambda *args: None)

    def __repr__(self):
        return self.name

    def _disk(self, pt):
        assert pt.is_real()
        # Since approximation disks satisfy 2·rad ≤ dist(center, sing), any
        # approximation disk containing pt must have rad ≤ dist(pt, sing)
        max_rad = pt.dist_to_sing().min(self.max_rad)
        # What we want is the largest such disk containing pt
        expo = ZZ(max_rad.log(2).upper().ceil()) # rad = 2^expo
        logger.log(logging.DEBUG-2, "max_rad = %s, expo = %s", max_rad, expo)
        while True:
            approx_pt = pt.approx_abs_real(-expo)
            mantissa = (approx_pt.squash() >> expo).floor()
            if ZZ(mantissa) % 2 == 0:
                mantissa += 1
            center = mantissa << expo
            dist = Point(center, pt.dop).dist_to_sing()
            rad = RBF.one() << expo
            logger.log(logging.DEBUG-2,
                    "candidate disk: approx_pt = %s, mantissa = %s, "
                    "center = %s, dist = %s, rad = %s",
                    approx_pt, mantissa, center, dist, rad)
            if safe_ge(dist >> 1, rad):
                break
            expo -= 1
        logger.debug("disk for %s: center=%s, rad=%s", pt, center, rad)
        # pt may be a ball with nonzero radius: check that it is contained in
        # our candidate disk
        log = approx_pt.abs().log(2)
        F = RealBallField(ZZ((expo - log).max(0).upper().ceil()) + 10)
        dist_to_center = (F(approx_pt) - F(center)).abs()
        if not safe_le(dist_to_center, rad):
            assert not safe_gt((approx_pt.squash() - center).squash(), rad)
            logger.info("check that |%s - %s| < %s failed",
                        approx_pt, center, rad)
            return None, None
        # exactify center so that subsequent computations are not limited by the
        # precision of its parent
        center = QQ(center)
        return center, rad

    def _path_to(self, dest, prec=None):
        r"""
        Find a path from a point with known "initial" values to pt
        """
        # TODO:
        # - attempt to start as close as possible to the destination
        #   [and perhaps add logic to change for a starting point with exact
        #   initial values if loosing too much precision]
        # - return a path passing through "interesting" points (and cache the
        #   associated initial vectors)
        start, ini = self.ini.items()[0]
        return ini, [start, dest]

    # Having the update (rather than the full test-and-update) logic in a
    # separate method is convenient to override it in subclasses.
    def _update_approx(self, center, rad, prec, derivatives):
        ini, path = self._path_to(center, prec)
        eps = RBF.one() >> prec
        ctx = ancont.Context(self.dop, path, eps, keep="all")
        pairs = ancont.analytic_continuation(ctx, ini=ini)
        for (vert, val) in pairs:
            known = self._inivecs.get(vert)
            if known is None or known[0].accuracy() < val[0][0].accuracy():
                self._inivecs[vert] = [c[0] for c in val]
        logger.info("computing new polynomial approximations: "
                    "ini=%s, path=%s, rad=%s, eps=%s, ord=%s",
                    ini, path, rad, eps, derivatives)
        polys = polapprox.doit(self.dop, ini=ini, path=path, rad=rad,
                eps=eps, derivatives=derivatives, x_is_real=True,
                economization=polapprox.chebyshev_economization)
        logger.info("...done")
        approx = self._polys.get(center, [])
        new_approx = []
        for ord, pol in enumerate(polys):
            if ord >= len(approx) or approx[ord].prec < prec:
                new_approx.append(RealPolApprox(pol, prec))
            else:
                new_approx.append(approx[ord])
        self._update_approx_hook(center, rad, polys)
        self._polys[center] = new_approx
        return polys

    def _sollya_annotate(self, center, rad, polys):
        import sollya
        logger = logging.getLogger(__name__ + ".sollya")
        logger.info("calling annotatefunction() on %s derivatives", len(polys))
        center = QQ(center)
        sollya_fun = self._sollya_object
        # sollya keeps all annotations, let's not bother with selecting
        # the best one
        for ord, pol0 in enumerate(polys):
            pol = ZZ(ord).factorial()*pol0
            sollya_pol = sum([c.center()*sollya.x**k
                              for k, c in enumerate(pol)])
            dom = RIF(center - rad, center + rad) # XXX: dangerous when inexact
            err_pol = pol.map_coefficients(lambda c: c - c.squash())
            err = RIF(err_pol(RBF.zero().add_error(rad)))
            with sollya.settings(display=sollya.dyadic):
                logger.debug("annotatefunction(%s, %s, %s, %s, %s);",
                        sollya_fun, sollya_pol, sollya.SollyaObject(dom),
                        sollya.SollyaObject(err), sollya.SollyaObject(center))
            sollya.annotatefunction(sollya_fun, sollya_pol, dom, err, center)
            sollya_fun = sollya.diff(sollya_fun)
        logger.info("...done")

    def approx(self, pt, prec=None, post_transform=None):
        r"""
        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.function import DFiniteFunction
            sage: DiffOps, x, Dx = DifferentialOperators()

            sage: h = DFiniteFunction(Dx^3-1, [0, 0, 1])
            sage: h.approx(0, post_transform=Dx^2)
            [2.0000000000000...]

            sage: f = DFiniteFunction((x^2 + 1)*Dx^2 + 2*x*Dx, [0, 1], max_prec=20)
            sage: f.approx(1/3, prec=10)
            [0.32...]
            sage: f.approx(1/3, prec=40)
            [0.321750554396...]
            sage: f.approx(1/3, prec=10, post_transform=Dx)
            [0.9...]
            sage: f.approx(1/3, prec=40, post_transform=Dx)
            [0.900000000000...]
            sage: f.approx(1/3, prec=10, post_transform=Dx^2)
            [-0.54...]
            sage: f.approx(1/3, prec=40, post_transform=Dx^2)
            [-0.540000000000...]

        """
        pt = Point(pt, self.dop)
        if prec is None:
            prec = _guess_prec(pt)
        derivatives = (post_transform.order() + 1 if post_transform is not None
                       else 1)
        post_transform = ancont.normalize_post_transform(self.dop,
                                                         post_transform)
        if not self._keep_all_derivatives:
            derivatives = post_transform.order() + 1
        if prec >= self.max_prec or not pt.is_real():
            logger.info("performing high-prec evaluation (pt=%s, prec=%s)",
                        pt, prec)
            ini, path = self._path_to(pt)
            eps = RBF.one() >> prec
            return self.dop.numerical_solution(ini, path, eps,
                    post_transform=post_transform)
        center, rad = self._disk(pt)
        if center is None:
            raise NotImplementedError
            #logger.info("falling back on generic evaluator")
            #ini, path = self._path_to(pt)
            #return self.dop.numerical_solution(ini=ini, path=path, eps=eps)
        approx = self._polys.get(center, [])
        Balls = RealBallField(prec)
        # due to the way the polynomials are recomputed, the precisions attached
        # to the successive derivatives are nonincreasing
        if (len(approx) < derivatives or approx[derivatives-1].prec < prec):
            polys = self._update_approx(center, rad, prec, derivatives)
        else:
            polys = [a.pol for a in approx]
        bpt = Balls(pt.value)
        reduced_pt = bpt - Balls(center)
        val = sum(ZZ(j).factorial()*coeff(bpt)*polys[j](reduced_pt)
                  for j, coeff in enumerate(post_transform))
        return val

    def __call__(self, x, prec=None):
        return self.approx(x, prec=prec)

    def plot(self, xrange, **options):
        r"""
        Plot this function.

        The plot is intended to give an idea of the accuracy of the evaluations
        that led to it. However, it may not be a rigorous enclosure of the graph
        of the function.

        EXAMPLES::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.function import DFiniteFunction
            sage: DiffOps, x, Dx = DifferentialOperators()

            sage: f = DFiniteFunction(Dx^2 - x,
            ....:         [1/(gamma(2/3)*3^(2/3)), -1/(gamma(1/3)*3^(1/3))])
            sage: plot(f, (-10, 5), color='black')
            Graphics object consisting of 1 graphics primitive
        """
        mids = generate_plot_points(
                lambda x: self.approx(x, 20).mid(),
                xrange, plot_points=200)
        ivs = [(x, self.approx(x, 20)) for x, _ in mids]
        bounds  = [(x, y.upper()) for x, y in ivs]
        bounds += [(x, y.lower()) for x, y in reversed(ivs)]
        options.setdefault('aspect_ratio', 'automatic')
        g = plot.polygon(bounds, thickness=1, **options)
        return g

    def plot_known(self):
        r"""
        TESTS::

            sage: from ore_algebra import *
            sage: from ore_algebra.analytic.function import DFiniteFunction
            sage: DiffOps, x, Dx = DifferentialOperators()

            sage: f = DFiniteFunction((x^2 + 1)*Dx^2 + 2*x*Dx, [0, 1])
            sage: f(-10, 100) # long time
            [-1.4711276743037345918528755717...]
            sage: f.approx(5, post_transform=Dx) # long time
            [0.038461538461538...]
            sage: f.plot_known() # long time
            Graphics object consisting of ... graphics primitives
        """
        g = plot.Graphics()
        for center, polys in self._polys.iteritems():
            center, rad = self._disk(Point(center, self.dop))
            xrange = (center - rad).mid(), (center + rad).mid()
            for i, a in enumerate(polys):
                # color palette copied from sage.plot.plot.plot
                color = plot.Color((0.66666+i*0.61803)%1, 1, 0.4, space='hsl')
                Balls = a.pol.base_ring()
                g += plot.plot(lambda x: a.pol(Balls(x)).mid(),
                               xrange, color=color)
                g += plot.text(str(a.prec), (center, a.pol(center).mid()),
                               color=color)
        for point, ini in self._inivecs.iteritems():
            g += plot.point2d((point, 0), size=50)
        return g

    def _sollya_(self):
        r"""
        EXAMPLES::

            sage: import logging; logging.basicConfig(level=logging.INFO)
            sage: logger = logging.getLogger("ore_algebra.analytic.function.sollya")
            sage: logger.setLevel(logging.DEBUG)

            sage: import ore_algebra
            sage: DiffOps, x, Dx = ore_algebra.DifferentialOperators()
            sage: from ore_algebra.analytic.function import DFiniteFunction
            sage: f = DFiniteFunction(Dx - 1, [1])

            sage: import sollya # optional - sollya
            sage: sollya.plot(f, sollya.Interval(0, 1)) # not tested
            ...

        """
        if self._sollya_object is not None:
            return self._sollya_object
        import sollya
        logger = logging.getLogger(__name__ + ".sollya")
        Dx = self.dop.parent().gen()
        def wrapper(pt, ord, prec):
            try:
                val = self.approx(pt, prec, post_transform=Dx**ord)
            except Exception:
                logger.info("pt=%s, ord=%s, prec=%s, error", pt, ord, prec,
                            exc_info=True)
                return RIF('nan')
            logger.debug("pt=%s, ord=%s, prec=%s, val=%s", pt, ord, prec, val)
            if not pt.overlaps(self._sollya_domain):
                backtrace = sollya.getbacktrace()
                logger.debug("%s not in %s", pt.str(style='brackets'),
                             self._sollya_domain.str(style='brackets'))
                logger.debug("sollya backtrace: %s",
                             [sollya.objectname(t.struct.called_proc)
                                 for t in backtrace])
            return val
        wrapper.__name__ = self.name
        self._sollya_object = sollya.sagefunction(wrapper)
        self._update_approx_hook = self._sollya_annotate
        return self._sollya_object

def _guess_prec(pt):
    if isinstance(pt, (RealNumber, ComplexNumber, RealBall, ComplexBall)):
        return pt.parent().precision()
    else:
        return 53

def _growth_parameters(dop):
    r"""
    Find κ, α such that the solutions of dop grow at most like
    sum(α^n*x^n/n!^κ) ≈ exp(κ*(α·x)^(1/κ)).

    EXAMPLES::

        sage: from ore_algebra import *
        sage: DiffOps, x, Dx = DifferentialOperators()
        sage: from ore_algebra.analytic.function import _growth_parameters
        sage: _growth_parameters(Dx^2 + 2*x*Dx) # erf(x)
        (1/2, [1.4...])
        sage: _growth_parameters(Dx^2 + 8*x*Dx) # erf(2*x)
        (1/2, [2.82 +/- 8.45e-3])
        sage: _growth_parameters(Dx^2 - x) # Airy
        (2/3, [1.0 +/- 3.62e-3])

        XXX: todo - add an example with several slopes

    """
    # Newton polygon. In terms of the coefficient sequence,
    # (S^(-j)·((n+1)S)^i)(α^n/n!^κ) ≈ α^(i-j)·n^(i+κ(j-i)).
    # In terms of asymptotics at infinity,
    # (x^j·D^i)(exp(κ·(α·x)^(1/κ))) ≈ α^(i/κ)·x^((i+κ(j-i))/κ)·exp(...).
    # The upshot is that we want the smallest κ s.t. i+κ(j-i) is max and reached
    # twice, and then the largest |α| with sum[edge](a[i,j]·α^(i/κ))=0.
    # (Note that the equation on α resulting from the first formulation
    # simplifies thanks to i+κ(j-i)=cst on the edge.)
    points = [(ZZ(j-i), ZZ(i), c) for (i, pol) in enumerate(dop)
                                  for (j, c) in enumerate(pol)
                                  if not c.is_zero()]
    h0, i0, _ = min(points, key=lambda (h, i, c): (h, -j))
    slope = max((i-i0)/(h-h0) for (h, i, c) in points if h > h0)
    Pol = dop.base_ring()
    eqn = Pol({i: c for (h, i, c) in points if i == i0 + slope*(h-h0)})
    expo_growth = bounds.abs_min_nonzero_root(eqn)**(-slope)
    return -slope, expo_growth
