# coding: utf-8
r"""
Path deformation

This module implements a *heuristic* method for deforming a given path into a
homotopic one along which analytic continuation by Taylor series can be expected
to be fast.

The method is based on the Voronoi diagram of the singularities. The general
idea is to “glue together” subpaths lying in a single Voronoi region in a
sequence corresponding to the shortest path on the Voronoi diagram homotopic to
the path we were given. The pieces of the path corresponding to individual
regions attempt to minimize the total number of terms of Taylor expansions
needed asymptotically as the working precision tends to infinity.

The main reason why the method is heuristic is that it works purely in
floating-point arithmetic, with no attempt at rigorous bounds on errors.
However, it is not hard to check a posteriori that the computed path is
homotopic to the original one.

Other limitations:

- When working with a full basis of solutions of a differential operator, it
  can be beneficial to reuse transition matrices along subpaths (and/or to take
  inverses) during analytic continuation along a given path. Typical examples
  include powers of loops, and conjugation of a loop by a path from a base
  point. Transformations of this kind are out of scope here.

- Apparent singularities are treated just like genuine (regular) singular
  points.

- Bad computational complexity (e.g., w.r.t. the number of singular points) --
  though this is unlikely to be an issue in practice.

TESTS::

    sage: from ore_algebra import DifferentialOperators
    sage: Dops, x, Dx = DifferentialOperators(QQ)

    sage: from ore_algebra.analytic.differential_operator import DifferentialOperator
    sage: from ore_algebra.analytic.path import Path
    sage: from ore_algebra.analytic.deform import PathDeformer

A fairly general singularity pattern::

    sage: from ore_algebra.examples import fcc
    sage: dop = DifferentialOperator(fcc.dop5)

    sage: path = [-20-I,-15+2*I,-12-I,-10+I,5*I,8/3*I+2,I,2,2-4*I,-5-5*I,-3-I]
    sage: PathDeformer(Path(path, dop)).check()
    True

    sage: path = [-20-I,-15+2*I,-12-I,-10+I,5*I,8/3*I+2,I,2,2-4*I,-5-5*I,-3-I,-3+I]
    sage: PathDeformer(Path(path, dop)).check()
    True

    sage: path = [-20-I,-15+2*I,-12-I,-10+I,5*I,8/3*I+2,I,2,2-4*I,-5-5*I,-3-I, -20+10*I]
    sage: PathDeformer(Path(path, dop)).check()
    True

    sage: path = [-3-I,-3+I,-8-I,6,I,-1-I,-3,-4,-3,2*I-1]
    sage: PathDeformer(Path(path, dop)).check()
    True

Multiple crossings of the infinite cut::

    sage: path = [-3-I, -16+16*I, -20-5*I, 6-5*I, 6+5*I, -16+5*I, -16-I]
    sage: PathDeformer(Path(path, dop)).check()
    True

    sage: path = [-3-I, -16+16*I, -20-5*I, 6-5*I, 6+5*I, -16+5*I, -16-I, -10+I, 2]
    sage: PathDeformer(Path(path, dop)).check()
    True

Multiple loops around a finite singularity::

    sage: path = [I,-1-I,-3,-4,-3,2*I-1,-1-3*I]
    sage: PathDeformer(Path(path, dop)).check()
    True

A single singularity::

    sage: dop = DifferentialOperator(x*Dx-1)
    sage: pdef = PathDeformer(Path([-1, i, 3, 1], dop))
    sage: pdef.analytic_path
    [(-1+0j), (-0.9307...+0.365...j), (-0.7325...+0.680...j),
    (-0.4328...+0.901...j), (-0.0731...+0.997...j), (0.29664...+0.954...j),
    (0.62534...+0.780...j), (0.86740...+0.497...j), (0.98929...+0.145...j),
    (1+0j)]
    sage: pdef.check()
    True
    sage: pdef.plot()
    Graphics object consisting of 11 graphics primitives

    sage: pdef = PathDeformer(Path([-1, -i, 3, 1], dop))
    sage: pdef.analytic_path
    [(-1+0j), (-0.930...-0.365...j), (-0.732...-0.680...j),
    (-0.432...-0.901...j), (-0.073...-0.997...j), (0.2966...-0.954...j),
    (0.6253...-0.780...j), (0.8674...-0.497...j), (0.9892...-0.145...j),
    (1+0j)]
    sage: pdef.check()
    True

    sage: dop = DifferentialOperator(x*Dx - 1)
    sage: loop0 = [1, i, -1, -i, 1]
    sage: loops = [loop0,
    ....:          list(reversed(loop0)),
    ....:          [a+1e-10j for a in loop0]]
    sage: for loop in loops:
    ....:     for k in range(len(loop) - 1):
    ....:         for path in [loop[:k] + loop[k:], loop + loop[1:k]]:
    ....:             if not PathDeformer(Path(path, dop)).check():
    ....:                 print("failed:", path)

    sage: PathDeformer(Path([i, -1, -i], dop)).check()
    True

    sage: PathDeformer(Path([-i, -1, i], dop)).check()
    True

    sage: PathDeformer(Path([i, 1, -i], dop)).check()
    True

    sage: PathDeformer(Path([-i, 1, i], dop)).check()
    True

    sage: PathDeformer(Path([1, -1+i, -i, 2], dop)).check()
    True

    sage: PathDeformer(Path([-2, -i, i+1, -1, -i], dop)).check()
    True

Two singularities::

    sage: dop = DifferentialOperator((x-1)*(x-2)*Dx)

    sage: path = [0, I, 3]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [0, 1+I, 2-I]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [0, 1+I, 2-I, 3, I, -3]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: dop = DifferentialOperator((x^2-1)*Dx)

    sage: path = [0, 2-I, 2+I, 0]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: dop = DifferentialOperator((x^2+1)*Dx)

    sage: path = [0, 1+2*I, -1+2*I, 0]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check() ### BUG
    True

    sage: QQi.<i> = QuadraticField(-1)
    sage: dop = DifferentialOperator((x+1+i)*(x-1-i)*Dx)

    sage: path = [0, 3, 3*I, 0]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [0, -3, -3*I, 0]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [0, 3*I, 3, 0]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check() ### BUG
    True

    sage: path = [0, 1+2*I, -1+2*I, -I + 1]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check() ### BUG
    True

Three singularities in general position::

    sage: dop = DifferentialOperator((x^2+1)*(x-2)*Dx)

    sage: path = [-1, 1]
    sage: PathDeformer(Path(path, dop)).analytic_path
    [(-1+0j), (-0.46...+0j), (-0.06...+0j), (0.29...+0j), (0.64...+0j),
    (0.75+0j), (1+0j)]

    sage: path = [-1,2*I,1]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [-1,-2*I,1]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [1, -1, -2*i]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [1, -1, -2*i, 1]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: dop = DifferentialOperator((x^2+1)*(x-2)*Dx)
    sage: path = [-2*i, -1, 1]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [-1,2*I,1,2*I,-1]
    sage: PathDeformer(Path(path, dop)).analytic_path
    [(-1+0j)]

    sage: path = [1, -1, -2*i, 1, 3j, -1]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [-1,2*I,1,-2*I,-1]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

More than two aligned singularities::

    sage: dop = DifferentialOperator((x-1)*(x-2)*(x-3)*Dx)
    sage: path = [0, 1+I, 2-I]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [0, 3-I, 2+I]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [0, 2+I, 3-I]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [0, 4+I, 4-I, 0]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [4, I, -I, 4]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [4, I, -I, 4, I, -I]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [4, I, 1-I, 2+I]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: dop = DifferentialOperator((x-1)*(x-2)*(x-3)*(x-4)*Dx)
    sage: path = [0, 2-I, 5, 3+I, 2-I, 1+I]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check()
    True

    sage: path = [3/2, 1+I, 5/2, 1-I]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check() ### BUG
    True

    sage: path = [3/2, 1+I, 5/2, 1-I, 3/2]
    sage: pdef = PathDeformer(Path(path, dop))
    sage: pdef.check() ### BUG
    True

Errors::

    sage: PathDeformer(Path([1, -1+i, 1-i, 1], DifferentialOperator(x*Dx))).check()
    Traceback (most recent call last):
    ...
    PathDeformationFailed: step [(-1+1j), (1-1j)] too close to singularity 0j
"""

# Copyright 2019, 2020 Marc Mezzarobba
# Copyright 2019, 2020 Centre national de la recherche scientifique
# Copyright 2019, 2020 Sorbonne Université
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

from __future__ import absolute_import, division, print_function

import collections, logging, cmath, math
import sage.plot.all as plot
import numpy
import scipy.optimize
import scipy.spatial.qhull

from itertools import combinations
from numpy import argmin

from sage.graphs.digraph import DiGraph
from sage.graphs.graph import Graph
from sage.misc.lazy_attribute import lazy_attribute
from sage.rings.all import CC

from .utilities import pairwise, split

logger = logging.getLogger(__name__)

eps = 1e-12
neg_inf = complex('-inf')

class PathDeformationFailed(Exception):
    pass

def reim(z):
    return z.real, z.imag

######################################################################
# Geometric primitives
######################################################################

# XXX Ideally, all geometric operations should be based on a small number of
# predicates, and we should use robust implementations (à la Shewchuk) of these
# basic bricks.

def sgn(x):
    if x < 0:
        return -1
    elif x > 0:
        return +1
    elif x == 0:
        return 0
    else:
        assert False

def dot(u, v):
    return u.real*v.real + u.imag*v.imag

def orient2d(a, b, c):
    r"""
    Positive when c lies to the left of [a,b].
    """
    if a.real == neg_inf:
        return sgn(c.imag - a.imag)
    # b != -∞ due to edge orientation convention
    elif c.real == neg_inf:
        return sgn(b.imag - a.imag)
    else:
        return sgn(((b.real - a.real)*(c.imag - a.imag)
                   - (c.real - a.real)*(b.imag - a.imag)))

def orient4(s, t, a, b, sticky):
    r"""
    Orientation of the path around s from a to b not crossing the half-line
    through t.

    sticky = 1 if the points on the cut [s, t] are considered to lie to its
    left, -1 to its right.
    """
    # due to limitations of orient2d regarding points at infinity, t cannot be
    # the second argument
    sta = orient2d(a, s, t)
    if sta == 0 and dot(a - s, t - s) >= 0.:
        return sticky
    sbt = orient2d(t, s, b)
    if sbt == 0 and dot(b - s, t - s) >= 0.:
        return -sticky
    sab = orient2d(b, s, a)
    return +1 if int(sta > 0) + int(sbt > 0) + int(sab > 0) >= 2 else -1

def sgn_inter(u, v):
    r"""
    Signed intersection of segments.

    One of u and v may also be a half-line of the form (-∞, z).
    Both u and v are sticky on their left side.

    TESTS::

        sage: from ore_algebra.analytic.deform import sgn_inter
        sage: cut = (complex("-inf"), complex(0.))
        sage: sgn_inter(cut, (-1.+0.jr, 1.jr))
        0
        sage: sgn_inter(cut, (1.jr, 1.+0.jr))
        0
        sage: sgn_inter(cut, (-1.jr, -1.+0.jr))
        -1
        sage: sgn_inter(cut, (-1.+0.jr, -1.jr))
        1
    """
    (a, b) = u
    (c, d) = v
    abd = orient2d(a, b, d)
    abc = orient2d(a, b, c)
    if (abd >= 0.) == (abc >= 0.):
        return 0
    cdb = orient2d(c, d, b)
    cda = orient2d(c, d, a) # = abd - abc + cdb
    if (cda >= 0.) == (cdb >= 0.):
        return 0
    return -1 if abd >= 0 else +1

def in_triangle(a, b, c, z):
    oa = orient2d(b, c, z)
    ob = orient2d(c, a, z)
    oc = orient2d(a, b, z)
    return oa >= 0 and ob >= 0 and oc >= 0 or oa <= 0 and ob <= 0 and oc <= 0

def circle_interval_intersections(zc, rad, za, zb, bounded):
    r"""
    Compute the intersection (if any) between a circle given by its center c
    and squared radius, and a line interval given by two points a, b and how
    many of these two points are bounds (2 = segment, 1 = half-line limited
    by a, 0 = full line).
    """
    # z = t·a + (1 - t)·b
    # r² = |z - c|² = (z - c)(zbar - cbar) =
    #               = (t a + (1-t) b - c)(t abar + (1-t) bbar - cbar)
    wa, wb, wc = za.conjugate(), zb.conjugate(), zc.conjugate()
    eq = [(za-zb)*(wa-wb), (za-zb)*(wb-wc) + (zb-zc)*(wa-wb),
          (zb-zc)*(wb-wc) - rad**2]
    rts = numpy.roots(eq)
    rts = [t.real for t in rts if -eps < t.imag < eps]
    if bounded >= 1:
        rts = [t for t in rts if t <= 1+eps]
    if bounded >= 2:
        rts = [t for t in rts if -eps <= t]
    return [t*za + (1-t)*zb for t in rts]

def loops_from_crossings(crossings, close):
    # XXX Vastly improvable, but good enough for now...
    if crossings == 0:
        return 0
    elif close:
        return abs(crossings)
    else:
        return abs(crossings) - 1

######################################################################
# Local path deformation
######################################################################

# XXX somehow account for the constant cost of adding a step?

def first_step(zs, z0, z1, orient, loops):
    r"""
    A first intermediate step from z0 to z1 going around zs with the given
    orientation.

    Should be good when z0, z1, and the returned point all lie in the Voronoi
    region of zs. Should still be acceptable when sz lies on the convex hull of
    the singularities, and z0, z1 are points outside the convex hull to which
    the closest singularity belonging to the convex hull is zs.

    TESTS::

        sage: from ore_algebra.analytic.deform import first_step
        sage: first_step(0, 1, .5, + 1, 0)
        0.500000000000000
        sage: first_step(0, 1, 4, + 1, 0)
        (1.3183657369408286+0j)
        sage: first_step(0, 1, complex(1,-0.001), + 1, 0)**17
        (0.99...+0.08...j)
        sage: first_step(0., 1., 1+.1j, + 1, 0)
        1.00000000000000 + 0.100000000000000*I
    """
    scaled_tgt = (z1 - zs)/(z0 - zs)
    angle = cmath.phase(scaled_tgt)
    if angle*orient < 0:
        loops += 1
    angle += orient*loops*2*math.pi
    dilat = abs(scaled_tgt)
    # we want the zero t ∈ [0;1] of this function
    # with t*angle ≈ π/4
    def f(t):
        rho = dilat**t
        phi = angle*t
        z = rho*cmath.exp(complex(0, phi))
        w = z - 1
        r = abs(w)
        v = r*math.log(r) + ((r/w)*z*cmath.log(z)).real
        return v
    tmin = .001
    tmax = math.pi/(4*abs(angle)) if 4*abs(angle) > math.pi else 1.
    if not f(tmin) <= 0:
        raise PathDeformationFailed
    if f(tmax) <= 0:
        return z1
    t0 = scipy.optimize.brentq(f, tmin, tmax)
    scaled_step = dilat**t0*cmath.exp(complex(0,angle*t0))
    return zs + scaled_step*(z0 - zs)

######################################################################
# Global path deformation
######################################################################

class DegenerateVoronoi(object):
    r"""
    Voronoi diagram of zero or more aligned points, in a format compatible (for
    our purposes) with the output of scipy.spatial.Voronoi.
    """

    def __init__(self, points):
        if not all(-eps < x0*y1 - x1*y0 < eps
                   for (x0, y0), (x1, y1) in combinations(points, 2)):
            raise ValueError("expected aligned points")
        self.points = points = numpy.array(points)
        self.vertices = numpy.ndarray([0,2])
        point_indices = list(range(len(points)))
        if len(points) > 2:
            p0 = points[0]
            dir = points[1] - p0
            def proj(i):
                p = points[i]
                return (p[0]-p0[0])*dir[0] + (p[1]-p0[1])*dir[1]
            point_indices.sort(key=proj)
        self.ridge_points = list(pairwise(point_indices))
        self.ridge_vertices = [[-1,-1] for _ in self.ridge_points]
        self.regions = [[-1] for _ in points]
        self.point_region = range(len(points))

        self.hull = point_indices + list(reversed(point_indices[1:-1]))

class VoronoiStep(object):
    r"""
    A “combinatorial” step.

    That is, a step whose endpoints can be vertices of the Vornoi diagram,
    virtual vertices of the extended Voronoi diagram, or the first or last
    point of the actual path. The data structure also records whether the step
    crosses a cut and how, so that the homotopy class of the step is
    unambiguous even in degenerate situations.
    """

    def __init__(self, ridge, v0, v1, orient=None):
        self.ridge = ridge # None when connecting to interior point
        self.v0 = v0 # None for first step
        self.v1 = v1 # None for last step
        self.orient = orient # None except when crossing a cut
        self.beacon = None # set by prepare_path_at_infinity (XXX not great)

    def __getitem__(self, i):
        if i == 0:
            return self.v0
        elif i == 1:
            return self.v1
        else:
            raise IndexError

    def __repr__(self):
        return "{}→{}({})".format(self.v0, self.v1, self.ridge)

class PathDeformer(object):

    def __init__(self, path, dop=None, max_subdivide=100):
        if dop is None: # then interpret path as a Path object
            dop = path.dop
            path = path.vert
        self.sing = [complex(z) for z in dop._singularities(CC)]
        if not self.sing:
            raise NotImplementedError("need at least one singularity")
        self.leftmost = int(argmin([z.real for z in self.sing]))
        self.sing.append(complex(float('-inf'), self.sing[self.leftmost].imag))
        self.input_path = [complex(v) for v in path]
        self.max_subdivide = max_subdivide
        # TODO should support the case where path[0], path[-1] are allowed to
        # be singular points
        self.check_input_path()

    def check_input_path(self):
        for z0, z1 in pairwise(self.input_path):
            if z0 == z1:
                continue
            for zs in self.sing:
                if (-eps < orient2d(z0, z1, zs) < eps
                        and 0. <= ((zs - z0)/(z1 - z0)).real <= 1.):
                    msg = "step " + str([z0, z1])
                    msg += " too close to singularity " + str(zs)
                    raise PathDeformationFailed(msg)

    @lazy_attribute
    def vert(self):
        return [complex(*xy) for xy in self.voronoi.vertices]

    # Voronoi diagram

    @lazy_attribute
    def _infinity_pos(self):
        z = self.sing[self.leftmost] - 3.
        return (z.real, z.imag)

    @lazy_attribute
    def voronoi(self):
        sing = self.sing[:-1]
        points = [(z.real, z.imag) for z in sing]
        try:
            vor = scipy.spatial.qhull.Voronoi(points)
        except (IndexError, scipy.spatial.qhull.QhullError):
            vor = DegenerateVoronoi(points)
        assert [complex(*xy) for xy in vor.points] == sing
        return vor

    @lazy_attribute
    def delaunay_graph(self):
        pos = dict(enumerate(self.voronoi.points))
        np = len(self.voronoi.points)
        if np <= 1:
            return Graph(np, pos=pos)
        edges = [(s0, s1, r) for r, (s0, s1)
                             in enumerate(self.voronoi.ridge_points)]
        return Graph(edges, format="list_of_edges", pos=pos)

    def other_region(self, r, s):
        r"""
        Return the region separated from s by ridge r, if any, None otherwise.
        """
        if r is None:
            return None
        s0, s1 = self.voronoi.ridge_points[r]
        if s0 == s:
            return s1
        elif s1 == s:
            return s0
        else:
            return None

    # Convex hull

    @lazy_attribute
    def hull(self):
        r"""
        Oriented convex hull of the singularities, starting with self.leftmost.
        """
        if isinstance(self.voronoi, DegenerateVoronoi):
            return self.voronoi.hull
        hull = [s for s in range(len(self.voronoi.points))
                if -1 in self.voronoi.regions[self.voronoi.point_region[s]]]
        z0 = self.sing[hull[0]]
        class Key(object):
            def __init__(key, i):
                key.i = i
            def __lt__(key0, key1, eps=eps):
                s = orient2d(z0, self.sing[key0.i], self.sing[key1.i])
                if s > eps:
                    # key1 to the left of the chord => key0 comes first
                    return True
                elif s < -eps:
                    return False
                d0 = abs(self.sing[key0.i] - z0)
                d1 = abs(self.sing[key1.i] - z0)
                return d0 <= d1
        hull.sort(key=Key)
        i0 = hull.index(self.leftmost)
        return hull[i0:] + hull[:i0]

    def oriented_hull_edge(self, s0, s1):
        r"""
        Edge oriented positively along the convex hull.

        WARNING: may not be consistent with oriented_cut().
        """
        l = len(self.hull)
        k0, k1 = self.hull.index(s0), self.hull.index(s1)
        if l == 2:
            return tuple(self.hull)
        elif (k1 + 1) % l == k0:
            return s1, s0
        else:
            assert (k0 + 1) % l == k1
            return s0, s1

    # Extended Voronoi diagram

    @lazy_attribute
    def virtual_offset(self):
        r"""
        Index of the first virtual offset.
        """
        return len(self.voronoi.vertices)

    @lazy_attribute
    def xvor(self):
        r"""
        The extended Voronoi diagram, as a Sage graph.

        This is the Vornoi diagram of the singularities, augmented with a
        ”virtual” vertex at infinity for each edge of the convex hull of the
        singularities. A convex hull reduced to a single point is understood to
        have one edge, and sides of segments forming the convex hull of a set
        of aligned points count as separate edges.

        All vertices have nonnegative integer indices. The true Voronoi
        vertices are those of index 0, ..., v-1 where v = self.virutal_offset.
        The virtual vertices have index v, v+1, ..., in cyclic order along the
        convex hull, starting with the vertex just after the unbounded cut.
        """
        edges = []
        pos = dict(enumerate(self.voronoi.vertices))
        voff = self.virtual_offset
        ns = len(self.voronoi.points)
        # Edges corresponding to Voronoi ridges
        for r, (v0, v1) in enumerate(self.voronoi.ridge_vertices):
            if v0 != -1 and v1 != -1: # finite ridge
                edges.append((v0, v1, r))
                continue
            s0, s1 = self.oriented_hull_edge(*self.voronoi.ridge_points[r])
            i0, i1 = self.hull.index(s0), self.hull.index(s1)
            vec = self.sing[s1] - self.sing[s0]
            vec = vec/abs(vec)
            if v0 == -1:
                v0, v1 = v1, v0
            # Virtual vertices. The index of a virtual vertex is voff + the
            # position of the corresponding edge (or edge side in the
            # degenerate case) on the oriented convex hull.
            if v0 != -1: # half-line
                edges.append((v0, voff + i0, r))
                pos[voff + i0] = reim(self.vert[v0] - 1.j*vec)
            else: # full line (degenerate case with all sing aligned)
                i1bis = 2*(ns - 1) - 1 - i0
                edges.append((voff + i0, voff + i1bis, r))
                mid = 0.5*(self.sing[s0] + self.sing[s1])
                pos[voff + i0] = reim(mid - 1.j*vec)
                pos[voff + i1bis] = reim(mid + 1.j*vec)
        # Additional edges between virtual vertices
        l = len(self.hull)
        edges.append((voff + l - 1, voff, -1)) # edge crossing the unbounded cut
        for i in range(l - 1):
            edges.append((voff + i, voff + (i + 1), -2-i))

        # Loops, multiedges because of degenerate cases, e.g., ”first” and
        # ”last” of aligned singularities
        return Graph(edges, format="list_of_edges", loops=True,
                     multiedges=True, pos=pos)

    @lazy_attribute
    def xvor_ridge_vertices(self):
        erv = [None]*len(self.voronoi.ridge_vertices)
        for v0, v1, r in self.xvor.edge_iterator():
            if r >= 0:
                erv[r] = [v0, v1] if v0 >= 0 else [v1, v0]
        return erv

    def is_virtual(self, v):
        return v is not None and v >= self.virtual_offset

    def virtual_vertices(self, s):
        r"""
        List of virtual vertices on the boundary of a given region.

        The corresponding singularity must belong to the convex hull. The
        vertices are returned in cyclic order along the convex hull, starting
        just before the unbounded edge.
        """
        voff = self.virtual_offset
        if s == self.leftmost:
            return [voff + len(self.hull) - 1, voff]
        i = self.hull.index(s)
        ns = len(self.voronoi.points)
        assert i >= 1
        if (not isinstance(self.voronoi, DegenerateVoronoi) or i == ns - 1):
            return [voff + i - 1, voff + i]
        else:
            off2 = 2*(ns - 1) - 1
            return [voff + i - 1, voff + i, off2 - i - 1, off2 - i]

    def concrete_ridge_to_virtual_vertex(self, v):
        # Note that while the returned ridge is concrete, in degenerate
        # situations, its other end may still be a virtual vertex.
        assert self.is_virtual(v)
        edges = self.xvor.edges_incident(v)
        cand = [r for _, _, r in edges if r >= 0]
        assert len(cand) == 1
        return cand[0]

    def virtual_ridge_region(self, r):
        if not r < 0:
            raise ValueError
        return self.hull[-r-1]

    # Cuts

    @lazy_attribute
    def cuts(self):
        r"""
        Euclidean minimum spanning tree of the singularities, augmented with an
        edge to infinity, so that the plane minus the tree is simply connected.

        Format: (sing index, sing index, ridge index).

        The node at infinity has index -1 (cf. scipy convention for unbounded
        Voronoi ridges). The unbounded edge corresponds to a horizontal cut to
        the left of self.leftmost, and has index -1 too.
        """
        def length(edge):
            return abs(self.sing[edge[0]] - self.sing[edge[1]])
        edges = self.delaunay_graph.min_spanning_tree(length)
        edges.append((-1, self.leftmost, -1))
        return edges

    @lazy_attribute
    def roadmap(self):
        r"""
        The extended Voronoi diagram minus the ridges crossing finite cuts.
        """
        con = Graph(self.xvor)
        cuts = set(r for _, _, r in self.cuts)
        for (v0, v1, r) in self.xvor.edge_iterator():
            if r in cuts:
                con.delete_edge(v0, v1, r)
        return con

    def oriented_cut(self, s0, s1):
        r"""
        Canonical orientation of the cut from s0 to s1.

        Horizontal edges must all be oriented in the same way so that our
        orientation predicate can respect the "stickiness" of the branch cut
        of the logarithm. The orientation of other edges is arbitrary.

        Note that, when s0, s1 lie on the convex hull, this may or may not
        correspond to the orientation of the convex hull (also used in the
        overall algorithm).
        """
        z0, z1 = self.sing[s0], self.sing[s1]
        if z0.real < z1.real or z0.real == z1.real and z0.imag < z1.imag:
            return s0, s1
        else:
            return s1, s0

    def a_cut(self, s):
        r"""
        The ridge index and the other end of a cut starting from s.

        Returns a bounded cut whenever possible.
        """
        for s0, s1, r in self.cuts:
            if s0 == s and s1 != -1:
                return r, s1
            if s1 == s and s0 != -1:
                return r, s0
        assert len(self.voronoi.points) == 1
        assert self.cuts == [(-1, s, -1)]
        return -1, -1

    # First and last step

    def connect_to_region(self, z):
        r"""
        Find the Vornoi region of z and a vertex (finite or virtual) of that
        region to which z can be connected in a straight line without crossing
        the cuts.
        """
        s = argmin([abs(z - w) for w in self.sing])
        for v in self.voronoi.regions[self.voronoi.point_region[s]]:
            if v == -1:
                continue
            for s0, s1, _ in self.cuts:
                if s == s0 or s == s1:
                    cut = (self.sing[s0], self.sing[s1])
                    if sgn_inter(cut, (z, self.vert[v])) != 0:
                        break
            else:
                return s, v
        # z lies in a region with no reachable finite vertex and must be
        # connected to a virtual vertex
        virt = self.virtual_vertices(s)
        if s == self.leftmost: # handle the unbounded cut
            assert virt[1] == self.virtual_offset
            if len(self.hull) == 1 or z.real <= self.sing[s].real:
                if z.imag >= self.sing[s].imag:
                    return s, virt[0]
                else:
                    return s, virt[1]
            else:
                s0, s1 = self.oriented_cut(s, self.hull[1])
        else:
            assert len(virt) == 2 or len(virt) == 4
            i = self.hull.index(s)
            assert i >= 1
            s0, s1 = self.oriented_cut(self.hull[i-1], s)
        if not isinstance(self.voronoi, DegenerateVoronoi):
            # two virtual vertices, both reachable
            # (typical example: 3 singularities with a single finite Voronoi
            # vertex in the middle)
            assert len(virt) == 2
            return s, virt[0]
        else:
            z0, z1 = self.sing[s0], self.sing[s1]
            if orient2d(z0, z1, z) < 0:
                return s, min(*virt)
            else:
                return s, max(*virt)

    @lazy_attribute
    def _sv_start(self):
        return self.connect_to_region(self.input_path[0])

    @lazy_attribute
    def s_start(self):
        return self._sv_start[0]

    @lazy_attribute
    def v_start(self):
        return self._sv_start[1]

    @lazy_attribute
    def _sv_end(self):
        return self.connect_to_region(self.input_path[-1])

    @lazy_attribute
    def s_end(self):
        return self._sv_end[0]

    @lazy_attribute
    def v_end(self):
        return self._sv_end[1]

    # Crossings

    def _inter_t(self, r, step):
        # The step and the ridge are assumed to intersect transversally.
        a, b = step
        if r == -1:
            t = (b.imag - self.sing[self.leftmost].imag)/(b.imag - a.imag)
        else:
            # z = t·a + (1-t)·b on the line through u, v
            # (z - u) ⟂ i·(v-u)
            u, v = (self.sing[s] for s in self.voronoi.ridge_points[r])
            orth = 1.j*(v - u)
            t = dot(orth, b-u)/dot(orth, b-a)
        assert -eps < t < 1. + eps
        return 1 - t

    def _crossings(self, path):
        # XXX many redundancies, bad complexity...
        seq = []
        for step in pairwise(path):
            inter = []
            for (s0, s1, r) in self.cuts:
                s0, s1 = self.oriented_cut(s0, s1)
                sgn = sgn_inter((self.sing[s0], self.sing[s1]), step)
                if sgn != 0:
                    inter.append((r, sgn))
            inter.sort(key=lambda k: self._inter_t(k[0], step))
            for r, sgn in inter:
                if seq and seq[-1] == (r, -sgn):
                    seq.pop()
                else:
                    seq.append((r, sgn))
        return seq

    @lazy_attribute
    def crossings(self):
        r"""
        Crossing sequence of the input path with the cuts (spanning tree).
        """
        return self._crossings(self.input_path)

    # Combinatorial path

    def roadmap_path(self, v0, v1):
        it = self.roadmap.shortest_simple_paths(v0, v1,
                by_weight=False,
                report_edges=True,
                labels=True)
        path0 = next(it)
        cur = v0
        path = []
        for v0, v1, r in path0:
            if v0 == cur:
                path.append(VoronoiStep(r, v0, v1))
                cur = v1
            elif v1 == cur:
                path.append(VoronoiStep(r, v1, v0))
                cur = v0
            else:
                assert False
        return path

    def path_to_critical_ridge(self, v_start, r_tgt, orient):
        r"""
        Path to a given oriented ridge crossing one of the cuts.

        INPUT: index of Vornoi vertex, ridge index, ridge orientation.
        OUTPUT: path to the other end of the ridge.
        """
        if r_tgt < 0:
            assert r_tgt == -1
            v0, v1 = self.virtual_vertices(self.leftmost)
            if orient == -1:
                v0, v1 = v1, v0
        else:
            s0, s1 = self.voronoi.ridge_points[r_tgt]
            s0, s1 = self.oriented_cut(s0, s1)
            v0, v1 = self.xvor_ridge_vertices[r_tgt]
            if self.is_virtual(v1):
                sign = +1 if self.oriented_hull_edge(s0, s1) == (s0, s1) else -1
                # In the degenerate case with two virtual vertices, the line
                # oriented from the virtual vertex of larger index to that of
                # smaller index always crosses the cut oriented based on its
                # first occurrence in the convex hull positively. In the case
                # of a half-line (v0 non-virtual), the half-line oriented from
                # v0 to v1 is outgoing, so we have nothing to do.
                if self.is_virtual(v0) and v1 >= v0:
                    sign = -sign
            else: # segment
                ridge = (self.vert[v0], self.vert[v1])
                sign = sgn_inter((self.sing[s0], self.sing[s1]), ridge)
                assert sign != 0 # because [s0,s1] is an edge of the EMST
            if orient*sign < 0:
                v0, v1 = v1, v0
        path = self.roadmap_path(v_start, v0)
        path.append(VoronoiStep(r_tgt, v0, v1, orient))
        return path

    @lazy_attribute
    def cpath(self):
        r"""
        “Combinatorial” path.

        A path on the Voronoi diagram such that the endpoints of the input path
        can be connected to the endpoints of the combinatorial path by straight
        lines not crossing the cuts, and the result is homotopic to the input
        path in the plane minus the singularities (with suitable conventions
        when the combinatorial path passes through infinity).
        """
        path = [VoronoiStep(None, None, self.v_start)]
        for ridge, orient in self.crossings:
            subpath = self.path_to_critical_ridge(path[-1].v1, ridge, orient)
            path.extend(subpath)
        path.extend(self.roadmap_path(path[-1].v1, self.v_end))
        path.append(VoronoiStep(None, self.v_end, None))
        return path

    # Analytic deformation

    def exit_index(self, s, i0):
        r"""
        Last step in/on the boundary of region s.

        Find the index of last step of the combinatorial path lying on the
        boundary of region s, starting from a step *on that boundary*.
        The output can be i0 itself.
        """
        region = self.voronoi.regions[self.voronoi.point_region[s]]
        try:
            j = next(i for i, step in enumerate(self.cpath[i0:], i0) if
                    # concrete ridge not on the boundary of this region
                    # (unlike the variant using voronoi.point_region, this test
                    # works with unbounded ridges)
                    step.ridge is not None
                       and step.ridge >= 0
                       and s not in self.voronoi.ridge_points[step.ridge]
                    # virtual ridge associated to another region
                    or step.ridge is not None
                       and step.ridge < 0
                       and self.virtual_ridge_region(step.ridge) != s
                    # exit to destination while we are not in the destination
                    # region
                    or step.v1 is None and s != self.s_end)
        except StopIteration:
            j = len(self.cpath)
        assert j >= i0
        return j - 1

    def beacon(self, i):
        v = self.cpath[i].v1
        if v is None:
            return self.input_path[-1]
        elif self.is_virtual(v):
            assert self.cpath[i].beacon is not None
            return self.cpath[i].beacon
        else:
            return complex(*self.voronoi.vertices[v])

    # XXX With the current implementation of this function, the intermediate
    # beacons seem to be too close to the convex hull. (Maybe see if the path
    # and beacon can be chosen so that the path is orthogonal to the unbounded
    # ridges it crosses?)
    def prepare_path_at_infinity(self, s0, z0, i0):
        r"""
        Set the beacons for the subpath at infinity starting at position i0.

        Modifies self.cpath.
        """

        i1 = i0 + 1
        path_sing = [s0]
        while self.cpath[i1].ridge is not None and self.cpath[i1].ridge < 0:
            path_sing.append(self.virtual_ridge_region(self.cpath[i1].ridge))
            i1 += 1
        assert self.is_virtual(self.cpath[i1].v0)
        if len(self.voronoi.ridge_points) > 0:
            reentry_ridge = self.concrete_ridge_to_virtual_vertex(
                                                              self.cpath[i1].v0)
            s1 = self.other_region(reentry_ridge, path_sing[-1])
        else:
            s1 = s0
        path_sing.append(s1)

        ea = (path_sing[0], path_sing[1])
        eb = self.oriented_hull_edge(*ea)
        orient = +1 if ea == eb else -1
        if isinstance(self.voronoi, DegenerateVoronoi):
            side2_offset = self.virtual_offset + len(self.voronoi.points) - 1
            if self.cpath[i0].v1 >= side2_offset:
                orient = -orient

        # Attempt to stay at a reasonable distance from the convex hull
        # both when the endpoints are far from singularities close to each
        # other and conversely. Maybe too complicated.
        sqrt2 = math.sqrt(2)
        spc = [float('-inf')]
        for s, t in pairwise(reversed(path_sing)):
            max_dist = max(spc[-1], abs(self.sing[s] - self.sing[t]))/sqrt2
            spc.append(max_dist)
        spc.reverse()

        # interpolate the distance to the singularity
        # XXX maybe better: do it in proportion to the angles...
        d0 = abs(z0 - self.sing[s0])
        if not self.is_virtual(self.cpath[i1].v1):
            d1 = abs(self.beacon(i1) - self.sing[s1])
        else:
            s10, s11 = self.voronoi.ridge_points[reentry_ridge]
            d1 = 0.5*abs(self.sing[s10] - self.sing[s11])
        dist = numpy.geomspace(d0, d1, len(path_sing) + 1)

        path = []
        for j in range(i1 - i0):
            s, t = path_sing[j], path_sing[j+1]
            mid = (self.sing[s] + self.sing[t])/2.
            if s != t:
                vec = self.sing[t] - self.sing[s]
                norm = -1.j*orient*vec/abs(vec)
            else:
                assert len(self.voronoi.vertices) == 0
                norm = z0 - self.sing[s]
                norm = norm/abs(norm)
            self.cpath[i0+j].beacon = mid + max(spc[j], dist[j+1])*norm
            self.cpath[i0+j].next_region = t

    def adjust_step(self, s, orient, vor_subpath, z0, z1, beacon, allow_escape):

        neighb = self.delaunay_graph.neighbors(s)
        if not neighb:
            return z1, s
        def dist_to_z1(s1):
            return abs(z1 - self.sing[s1])
        s_closest = min(neighb, key=dist_to_z1)
        zs_closest = self.sing[s_closest]

        # Do we escape region s? Test based on the singularities, not the
        # Voronoi vertices, because of unbounded ridges.
        zs = self.sing[s]
        if abs(z1 - zs_closest) >= (1 - eps)*abs(z1 - zs):
            return z1, s

        ridge_hit = self.delaunay_graph.edge_label(s, s_closest)
        logger.debug("hit ridge %s (= %s)", ridge_hit, [s, s_closest])

        # So z1 is on the wrong side of the ridge between s and s_closest. If
        # beacon lies on the other side of the ridge as well (and thus outside
        # the region), let the step escape the region.
        if abs(beacon - zs_closest) < abs(beacon - zs):
            logger.debug("escape to %s based on beacon position", s_closest)
            return z1, s_closest

        # Otherwise, we need to move z1 back into the region. Look for an
        # intersection of the circle centered at z0 passing through z1 with the
        # oriented boundary of the region. Note that there can be more than two
        # intersections in total, and that these intersections do not
        # necessarily happen on ridges corresponding to violated constraints.
        # Any intersection with the part of the boundary covered by the
        # combinatorial part should do. We take the farthest.

        def try_adjust(ridge, va, vb):

            if self.is_virtual(va):
                va, vb = vb, va
            pos = self.xvor.get_pos()
            za, zb = complex(*pos[va]), complex(*pos[vb])
            bounded = int(not self.is_virtual(va)) + int(not self.is_virtual(vb))
            rad = abs(z0 - z1)
            inter = circle_interval_intersections(z0, rad, za, zb, bounded)

            s1 = self.other_region(ridge, s)
            best_adjusted = None
            for zi in inter:
                if orient2d(zs, z0, zi) == orient:
                    logger.log(logging.DEBUG - 1,
                               "can adjust to %s on ridge %s", zi, [s, s1])
                    if not abs(zi - z0) < (1+eps)*abs(z1 - z0) < abs(z0 - zs):
                        logger.log(logging.DEBUG, "adjustment may have failed")
                        # raise PathDeformationFailed
                    best_adjusted = (zi, s1)

            return best_adjusted

        best_adjusted = None
        for step in vor_subpath:
            if step.ridge is None or step.ridge < 0:
                continue # ignore virtual ridges and endpoint connections
            adjusted = try_adjust(step.ridge, *step)
            if adjusted is not None:
                best_adjusted = adjusted
        if best_adjusted is not None:
            return best_adjusted

        # The step escapes the current region, but not by crossing any of the
        # ridges traversed by the combinatorial path. This should only happen
        # at infinity or when connecting from/to an interior point.
        elif allow_escape:
            logger.debug("escape to %s by special permission", s_closest)
            return z1, s_closest
        # Fall back to taking the intersection with the ridge we hit.
        else:
            va, vb = self.xvor_ridge_vertices[ridge_hit]
            adjusted = try_adjust(ridge_hit, va, vb)
            if adjusted is not None:
                return adjusted
        raise PathDeformationFailed

    @lazy_attribute
    def analytic_path(self):
        r"""
        Deform and subdivide a combinatorial path.

        Starting from a “combinatorial” path on the Voronoi diagram, build a
        homotopic “analytic” path with a choice of shape and step sizes that
        tries to minimize the running time of Taylor methods.
        """

        s = self.s_start        # index of the current singularity (region)
        z = self.input_path[0]  # ∈ ℂ, current point of path under construction
        i = 0                   # position in the combinatorial path
        j = None                # pos of last step associated to current region

        # Really a local variable of this function, but stored as an attribute
        # for easier inspection (e.g., for plotting incomplete paths).
        self._analytic_path = [z]

        # Iterate over regions of the Voronoi diagram on the boundary of which
        # the combinatorial path passes.
        for _ in self.cpath:

            j = self.exit_index(s, i)
            step_j = self.cpath[j]
            if self.is_virtual(step_j.v1) and step_j.beacon is None:
                self.prepare_path_at_infinity(s, z, j)
            beacon = self.beacon(j) # where we are heading at the moment

            # Distinguished cut used to count loops around s.
            rc, sc = self.a_cut(s)
            sc0, sc1 = self.oriented_cut(s, sc)
            zc0, zc1 = self.sing[sc0], self.sing[sc1]
            # We compute the signed crossing number of the combinatorial path
            # and the distinguished cut, then correct for the first step. This
            # is better than trying to replace the first combinatorial step by
            # a step from z to its end, or something like that, due to the
            # possible presence of virtual vertices.
            crossings = sum(step.orient for step in self.cpath[i:j+1]
                                        if step.ridge == rc)
            if self.cpath[i].ridge == rc:
                assert self.cpath[i].orient in [-1, 1]
                if self.cpath[i].orient*orient2d(zc0, zc1, z) < 0:
                    crossings -= self.cpath[i].orient
            if crossings != 0:
                orient = sgn(crossings)
                if sc == sc1:
                    orient = -orient
            else:
                sticky = +1 if sc0 == s else -1
                orient = orient4(self.sing[s], self.sing[sc], z, beacon, sticky)

            logger.debug("### s=%s %s orient=%s beacon=%s rc=%s crossings=%s",
                    s, self.cpath[i:j+1], orient, beacon, rc, crossings)

            # Go around the current singularity
            for _ in range(self.max_subdivide):

                # Have we reached the beacon? Then we are done with this region.
                dist = abs(z - beacon)
                if crossings == 0 and dist < eps:
                    logger.debug("hit beacon, j=%s, v=%s", j, step_j.v1)
                    if self.is_virtual(step_j.v1):
                        s = step_j.next_region
                        logger.debug("region switch at infinity to %s", s)
                    elif step_j.v1 is None:
                        assert beacon == self.input_path[-1]
                        return self._analytic_path
                    elif self.cpath[j+1].v1 is None: # XXX useful?
                        s = self.s_end
                        logger.debug("reached final region %s", s)
                    else:
                        # It can happen in degenerate cases that we need to
                        # switch to a region that shares no ridge with the
                        # current one and hence wasn't discovered yet.
                        s1, s2 = self.voronoi.ridge_points[self.cpath[j+1].ridge]
                        s = s1 if s1 != s else s2
                        logger.debug("degenerate region switch to %s", s)
                    i = j + 1
                    break

                # Compute the next step.
                loops = loops_from_crossings(crossings, dist < eps)
                candidate = first_step(self.sing[s], z, beacon, orient, loops)
                if not orient2d(self.sing[s], z, candidate)*orient >= 0:
                    raise PathDeformationFailed("inconsistent orientations")
                if (dist < abs(z - candidate) and crossings == 0
                        and orient2d(self.sing[s], z, beacon) == orient):
                    candidate = beacon
                logger.log(logging.DEBUG, "candidate step %s → %s ... %s",
                        z, candidate, beacon)
                allow_escape = step_j.ridge is None or step_j.ridge < 0
                z, s_switch = self.adjust_step(s, orient, self.cpath[i:j+1],
                                        z, candidate, beacon, allow_escape)
                if not abs(z - self._analytic_path[-1]) > eps:
                    raise PathDeformationFailed("no progress")
                self._analytic_path.append(z)

                # Did we cross the distinguished cut? Then update the counter.
                if len(self._analytic_path) >= 2:
                    sign = sgn_inter((zc0, zc1), self._analytic_path[-2:])
                    if sign:
                        if crossings == 0:
                            raise PathDeformationFailed
                        crossings -= sign
                        logger.log(logging.DEBUG-1, "crossings=%s", crossings)

                # If the step we just performed hit a ridge, we may want to
                # switch to the region across the ridge.
                if s_switch != s:
                    do_switch = False
                    if self.exit_index(s_switch, j) >= j + 1:
                        if crossings == 0:
                            do_switch = True
                        elif abs(crossings) == 1 and step_j.ridge == rc:
                            sign = orient2d(zc0, zc1, self._analytic_path[-2])
                            do_switch = (sign >= 0)
                            if crossings == -1:
                                do_switch = not do_switch
                    if do_switch:
                        logger.debug("z=%s hit exit ridge, region switch %s→%s",
                                     z, s, s_switch)
                        s = s_switch
                        i = j
                        break
                    else:
                        logger.debug("not switching to region %s", s_switch)

            else:
                raise PathDeformationFailed("subdivision limit reached")

    # Post-processing

    @lazy_attribute
    def smooth_analytic_path(self):
        r"""
        “Smoother” version of the analytic path obtained by local improvements.
        """

        if len(self.analytic_path) <= 2:
            return self.analytic_path

        sing = self.sing[:-1]

        def cost(z0, z1):
            rad = min(abs(z0 - w) for w in sing)
            dist = abs(z1 - z0)
            if dist >= rad:
                return float('inf')
            c = 1/math.log(rad/dist)
            assert c >= 0.
            return c

        def ok(x, y, z):
            return not any(in_triangle(x, y, z, w) for w in sing)

        path = [self.analytic_path[0]]
        for i in range(len(self.analytic_path) - 2):
            x = path[-1]
            y, z = self.analytic_path[i+1:i+3]
            c0 = cost(x, y) + cost(y, z)
            if cost(x, z) <= c0: # in particular, finite,  => ok
                # drop y
                continue
            m = (x + z)/2.
            if cost(x, m) + cost(m, z) <= c0 and ok(x, y, z):
                path.append(m)
                continue
            path.append(y)

        path.append(self.analytic_path[-1])

        return path

    @lazy_attribute
    def result(self):
        return self.smooth_analytic_path

    def check(self):
        r"""
        NON-RIGOROUS check that the output path is homotopic to the input path.
        """
        return self.crossings == self._crossings(self.result)

    # Plots

    def plot_voronoi(self, edge_labels=True, edge_colors=None, **kwds):
        vor = self.xvor
        if edge_colors is None:
            # roadmap_edges = self.roadmap.edges()
            # other_edges = list(set(vor.edges()) - set(roadmap_edges))
            vor_edges, virt_edges = split(lambda e: e[-1] >= 0, vor.edges())
            edge_colors = {
                    # "blue": roadmap_edges,
                    "black": vor_edges,
                    "lightgrey": virt_edges,
            }
        pl = vor.plot(edge_labels=edge_labels, edge_colors=edge_colors, **kwds)
        return pl

    def plot_cuts(self, vertex_shape="s", edge_labels=True, **kwds):
        edges = [self.oriented_cut(s0, s1) + (i,)
                 for (s0, s1, i) in self.cuts]
        pos = dict(enumerate(self.voronoi.points))
        pos[-1] = self._infinity_pos
        graph = DiGraph(edges, format="list_of_edges", pos=pos)
        return graph.plot(edge_labels=edge_labels, vertex_shape=vertex_shape,
                          **kwds)

    def plot_input(self, **kwds):
        pl = self.plot_voronoi(edge_labels=False, **kwds)
        pl += self.plot_cuts(edge_style="dotted")
        pl += plot.line([(z.real, z.imag) for z in self.input_path[:-1]],
                        linestyle= "dashed", thickness=2, color="green")
        pl += plot.arrow(*[(z.real, z.imag) for z in self.input_path[-2:]],
                        linestyle= "dashed", thickness=2, color="green")
        return pl

    def plot_cpath(self, thickness=3, **kwds):
        pl = self.plot_input()
        pos = self.xvor.get_pos()
        for step in self.cpath[1:-1]:
            if step.v0 != -1 and step.v1 != -1:
                seg = [pos[v] for v in step]
                pl += plot.line(seg, thickness=thickness, **kwds)
        return pl

    def plot(self, both=False, **kwds):
        pl = self.plot_input()
        failed = False
        try:
            path = self.result
            pl += plot.line([(z.real, z.imag) for z in path],
                            thickness=3, **kwds)
        except PathDeformationFailed as exn:
            # Then we plot the possibly incomplete path found in
            # self._analytic_path.
            pl.set_legend_options(title="FAILED")
            print("FAILED:", exn)
            pl.legend(True)
            both = True
        if both:
            pl += plot.line([(z.real, z.imag) for z in self._analytic_path],
                            thickness=2, **kwds)
        return pl
