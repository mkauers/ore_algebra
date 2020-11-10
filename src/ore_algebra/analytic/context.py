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

class Context(object):

    def __init__(self, dop=None, path=None, eps=None,
            algorithm=None,
            assume_analytic=False,
            force_algorithm=False,
            squash_intervals=False,
            deform=False,
            binsplit_thr=128,
            bit_burst_thr=32,
            simple_approx_thr=64,
            recorder=None,
        ):
        r"""
        Analytic continuation context

        Options:

        * ``deform`` -- Whether to attempt to automatically deform the analytic
          continuation path into a faster one. Enabling this should result in
          significantly faster integration for problems with many
          singularities, especially at high precision. It may be slower in
          simple cases, though.

        * ``recorder`` -- An object that will be used to record various
          intermediate results for debugging and analysis purposes. At the
          moment recording just consists in writing data to some fields of the
          object. Look at the source code to see what fields are available;
          define those fields as properties to process the data.

        * (other options still to be documented...)
        """

        # TODO: dop, path, eps...

        if not algorithm in [None, "naive", "binsplit"]:
            raise ValueError("algorithm", algorithm)
        self.algorithm = algorithm

        if not isinstance(assume_analytic, bool):
            raise TypeError("assume_analytic", type(assume_analytic))
        self.assume_analytic = assume_analytic

        if not isinstance(force_algorithm, bool):
            raise TypeError("force_algorithm", type(force_algorithm))
        self.force_algorithm = force_algorithm

        if not isinstance(squash_intervals, bool):
            raise TypeError("squash_intervals", type(squash_intervals))
        self.squash_intervals = squash_intervals

        if not isinstance(deform, bool):
            raise TypeError("deform", type(deform))
        self.deform = deform

        self.binsplit_thr = int(binsplit_thr)

        self.bit_burst_thr = int(bit_burst_thr)

        self.simple_approx_thr = int(simple_approx_thr)

        self.recorder = recorder

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def __call__(self, **kwds):
        # XXX Should check the new values, and maybe return a wrapper that
        # shadows some attributes rather than a copy.
        new = self.__new__(Context)
        new.__dict__ = self.__dict__.copy()
        new.__dict__.update(kwds)
        return new

    def prefer_binsplit(self):
        return self.algorithm == "binsplit"

    def force_binsplit(self):
        return self.prefer_binsplit() and self.force_algorithm

    def prefer_naive(self):
        return self.algorithm == "naive"

    def force_naive(self):
        return self.prefer_naive() and self.force_algorithm

dctx = Context() # default context
