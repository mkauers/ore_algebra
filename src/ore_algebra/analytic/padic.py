# -*- coding: utf-8 - vim: tw=80
r"""
Evaluation of univariate D-finite functions at p-adic points
"""

import logging

from sage.matrix.constructor import identity_matrix
from sage.rings.number_field.number_field import NumberField
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

from . import accuracy
from . import binary_splitting

from .differential_operator import DifferentialOperator
from .path import EvaluationPoint
from .utilities import pairwise

logger = logging.getLogger(__name__)

def embedded_exact_field(Approx):
    Base = Approx.base_ring()
    if Base is Approx:
        return QQ
    else:
        assert Base.base_ring() is Base
        return NumberField(Approx._exact_modulus, names=Approx.variable_name(),
                           embedding=Approx.gen())

def p_digit_burst_precs(prec, r0=8, r1=2, size_thr=2):
    steps = [prec]
    # Finish with a single tiny step, presumably using a basecase algorithm.
    prec //= r0
    steps.append(prec)
    # Actual bit burst path
    while prec >= size_thr:
        prec //= r1
        steps.append(prec)
    if prec > 0:
        steps.append(0)
    return list(reversed(steps))

def p_digit_burst_eval(dop, pt, eps):

    assert pt.parent().prime() == eps.parent().prime()
    prime = accuracy.IR(eps.parent().prime())

    dop = DifferentialOperator(dop)
    orddeq = dop.order()

    prec = p_digit_burst_precs(eps.valuation())
    coord = list(pt.polynomial())
    NF = embedded_exact_field(pt.parent())
    # XXX handle denominators?
    # XXX consider *not* lifting the last point
    path = [NF([c.slice(p0, p1).lift() for c in coord])
            for p0, p1 in pairwise([0] + prec)]

    path_mat = identity_matrix(ZZ, dop.order())
    step_mats = []
    for val, (dz0, dz1) in zip(prec, pairwise(path)):
        dop = dop.shift(dz0)
        if dz1.is_zero():
            continue
        # XXX go back to p-adic elements (--> truncated binary splitting)???
        pt = EvaluationPoint(dz1, rad=prime**(-val), jet_order=orddeq)
        step_mat = binary_splitting.fundamental_matrix_regular(dop, pt, eps)
        step_mats.append(step_mat)
        path_mat = step_mat*path_mat

    return path_mat
