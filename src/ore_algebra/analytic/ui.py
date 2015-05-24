# -*- coding: utf-8 - vim: tw=80
"""
Some convenience functions for direct use of the features of this package

Ultimately, the typical way to use it should be through methods of objects such
as differential operators and D-finite functions, not through this module!
"""

from . import analytic_continuation as ancont

def transition_matrix(dop, path, eps):
    ctx = ancont.Context(dop, path, eps)
    pairs = ancont.analytic_continuation(ctx)
    assert len(pairs) == 1
    return pairs[0][1]

def transition_matrices(dop, path, eps):
    ctx = ancont.Context(dop, path, eps, keep="all")
    pairs = ancont.analytic_continuation(ctx)
    return pairs

def eval_diffeq(dop, ini, path, eps):
    ctx = ancont.Context(dop, path, eps)
    pairs = ancont.analytic_continuation(ctx, ini=ini)
    assert len(pairs) == 1
    _, mat = pairs[0]
    return mat[0][0]

def multi_eval_diffeq(dop, ini, path, eps):
    ctx = ancont.Context(dop, path, eps, keep="all")
    pairs = ancont.analytic_continuation(ctx, ini=ini)
    return [(point, mat[0][0]) for point, mat in pairs]

def polynomial_approximation_on_disk(dop, ini, path, rad, eps):
    raise NotImplementedError

def polynomial_approximation_on_interval(dop, ini, path, rad, eps):
    raise NotImplementedError

def make_proc(xxx): # ??? - ou object DFiniteFunction ?
    pass
