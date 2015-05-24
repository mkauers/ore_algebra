# -*- coding: utf-8 - vim: tw=80
"""
Abstract transition matrices
"""

import logging
logger = logging.getLogger(__name__)

class TransitionMatrixGroup(Group):

    def __init__(self, dop):
        self._dop = dop

class TransitionMatrix(MultiplicativeGroupElement):

    def __init__(self, dop, path):
        MultiplicativeGroupElement.__init__(self, TransitionMatrixGroup(dop))
        self._path = AnalyticContinuationPath(dop, path)

    def _repr_(self):
        return "Transition matrix along " + repr(self._path)

    def absolute_approx(self, binary_prec, terms=lambda _: None):
        return self._path.ordinary_transition_matrix(RIF(1.) >> binary_prec, terms)

    def _numerical_approx(self, prec):  # relative prec!
        pass

    def _mul_(self, other):
        pass

    # ...

