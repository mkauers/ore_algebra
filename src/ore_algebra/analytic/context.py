# -*- coding: utf-8 -*-
r"""
Analytic continuation contexts
"""

# Copyright 2018 Marc Mezzarobba
# Copyright 2018 Centre national de la recherche scientifique
# Copyright 2018 Université Pierre et Marie Curie
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

import pprint

from sage.rings.real_arb import RealBallField


class Context(object):
    r"""
    Analytic continuation context

    Options:

    - ``algorithm`` (string) -- Which algorithm to try first for computing sums
      of local expansions. Unless ``force_algorithm`` is set to ``True``, the
      computation may still use other methods. Supported values: ``"naive"``
      (direct summation), ``"binsplit"`` (binary splitting).

    - ``assume_analytic`` (boolean) -- When ``True``, assume that the
      solution(s) of interest of the equation being solved are analytic at any
      singular point of the operator lying on the path (excluding the
      endpoints). The solver is then allowed to make any transformation (such as
      deforming the path) that preserves the values of solutions satisfying this
      assumption.

    - ``binsplit_thr`` (int) -- Threshold used in the binary splitting algorithm
      to determine when to use a basecase algorithm for a subproduct.

    - ``bit_burst_thr`` (int) -- Minimal bit size to consider using bit-burst
      steps instead of direct binary splitting.

    - ``bounds_prec`` (int) -- Working precision for the computation of error
      bounds and other internal low-precision calculations.

    - ``deform`` (boolean) -- (EXPERIMENTAL) Whether to attempt to automatically
      deform the analytic continuation path into a faster one. Enabling this
      should result in significantly faster integration for problems with many
      singularities, especially at high precision. It may be slower in simple
      cases, though.

    - ``force_algorithm`` (boolean) -- If ``True``, only use the algorithm
      specified by the ``algorithm`` option.

    - ``recorder`` -- An object that will be used to record various intermediate
      results for debugging and analysis purposes. At the moment recording just
      consists in writing data to some fields of the object. Look at the source
      code to see what fields are available; define those fields as properties
      to process the data.

    - ``simple_approx_thr`` (int) -- Bit size above which vertices of the
      analytic continuation path should be replaced by simpler approximations if
      possible.

    - ``squash_intervals`` (boolean) -- (EXPERIMENTAL) If ``True``, try to
      reduce the working precision in the direct summation algorithm, at the
      price of computing additional error bounds.

    - ``two_point_mode`` (boolean) -- If ``True``, when possible, compute series
      expansions at every second point of the integration path and evaluate each
      expansion at two points. If ``False``, prefer evaluating each expansion at
      the next expansion point only. (Note that the path will not be subdivided
      in the same way in both cases, and that the presence of singular vertices
      complicates the picture.)
    """

    def __init__(self, dop=None, path=None, eps=None, *, ctx=None, **kwds):

        # TODO: dop, path, eps...

        if ctx is None:
            self._set_options(**kwds)
        else:
            assert isinstance(ctx, Context)
            if kwds:
                raise ValueError("received both a Context object and keywords")
            self.__dict__.update(ctx.__dict__)

    def _set_options(self, *,
                     algorithm=None,
                     assume_analytic=False,
                     binsplit_thr=128,
                     bit_burst_thr=32,
                     bounds_prec=53,
                     deform=False,
                     force_algorithm=False,
                     recorder=None,
                     simple_approx_thr=64,
                     squash_intervals=False,
                     two_point_mode=None,
                     ):

        if not algorithm in [None, "naive", "binsplit"]:
            raise ValueError("algorithm", algorithm)
        self.algorithm = algorithm

        if not isinstance(assume_analytic, bool):
            raise TypeError("assume_analytic", type(assume_analytic))
        self.assume_analytic = assume_analytic

        self.binsplit_thr = int(binsplit_thr)

        self.bit_burst_thr = int(bit_burst_thr)

        self._set_interval_fields(bounds_prec)

        if not isinstance(deform, bool):
            raise TypeError("deform", type(deform))
        self.deform = deform

        if not isinstance(force_algorithm, bool):
            raise TypeError("force_algorithm", type(force_algorithm))
        self.force_algorithm = force_algorithm

        self.recorder = recorder

        self.simple_approx_thr = int(simple_approx_thr)

        if not isinstance(squash_intervals, bool):
            raise TypeError("squash_intervals", type(squash_intervals))
        self.squash_intervals = squash_intervals

        if two_point_mode is None:
            two_point_mode = not deform
        if not isinstance(two_point_mode, bool):
            raise TypeError("two_point_mode", type(two_point_mode))
        if deform and two_point_mode:
            raise NotImplementedError("deform == two_point_mode == True")
        self.two_point_mode = two_point_mode

    def _set_interval_fields(self, bounds_prec):
        bounds_prec = int(bounds_prec)
        self.IR = RealBallField(bounds_prec)
        self.IC = self.IR.complex_field()

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def increase_bounds_prec(self):
        self._set_interval_fields(2*self.IR.precision())

    def prefer_binsplit(self):
        return self.algorithm == "binsplit"

    def force_binsplit(self):
        return self.prefer_binsplit() and self.force_algorithm

    def prefer_naive(self):
        return self.algorithm == "naive"

    def force_naive(self):
        return self.prefer_naive() and self.force_algorithm


dctx = Context()  # default context
