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

class Context(object):

    def __init__(self, dop=None, path=None, eps=None,
            algorithm=None,
            assume_analytic=False,
            force_algorithm=False,
            keep="last",
            max_split=3,
            squash_intervals=False,
        ):

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

        if not keep in ["all", "last"]:
            raise ValueError("keep", keep)
        self.keep = keep

        if not isinstance(max_split, int):
            raise TypeError("max_split", type(max_split))
        self.max_split = max_split

        if not isinstance(squash_intervals, bool):
            raise TypeError("squash_intervals", type(squash_intervals))
        self.squash_intervals = squash_intervals

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def prefer_binsplit(self):
        return self.algorithm == "binsplit"

    def force_binsplit(self):
        return self.prefer_binsplit() and self.force_algorithm

    def prefer_naive(self):
        return self.algorithm == "naive"

    def force_naive(self):
        return self.prefer_naive() and self.force_algorithm

dctx = Context() # default context
