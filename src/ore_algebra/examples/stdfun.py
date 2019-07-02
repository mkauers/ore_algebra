# coding: utf-8
r"""
Elementary and special functions

This file contains a small collection of definitions of classical functions as
solutions of differential initial value problems.

::

    sage: from ore_algebra.examples import stdfun

    sage: stdfun.airy_ai.dop(airy_ai(x))
    0
    sage: stdfun.airy_ai.dop.numerical_solution(
    ....:         stdfun.airy_ai.ini, [stdfun.airy_ai.pt, 1])
    [0.135292416312881...]

    sage: stdfun.airy_bi.dop(airy_bi(x))
    0
    sage: stdfun.airy_bi.dop.numerical_solution(
    ....:         stdfun.airy_bi.ini, [stdfun.airy_bi.pt, 1])
    [1.207423594952871...]

    sage: stdfun.arctan.dop(arctan(x))
    0
    sage: stdfun.arctan.dop.numerical_solution(
    ....:         stdfun.arctan.ini, [stdfun.arctan.pt, 1])
    [0.7853981633974483...]

    sage: stdfun.cos.dop(cos(x))
    0
    sage: stdfun.cos.dop.numerical_solution(
    ....:         stdfun.cos.ini, [stdfun.cos.pt, pi/3])
    [0.500000000000000...]

    sage: stdfun.dawson.dop.numerical_solution(
    ....:         stdfun.dawson.ini, [stdfun.dawson.pt, 1])
    [0.538079506912768...]

    sage: stdfun.erf.dop(erf(x))
    0
    sage: stdfun.erf.dop.numerical_solution(
    ....:         stdfun.erf.ini, [stdfun.erf.pt, 1])
    [0.842700792949714...]

    sage: stdfun.erfi.dop(erfi(x))
    0
    sage: stdfun.erfi.dop.numerical_solution(
    ....:         stdfun.erfi.ini, [stdfun.erfi.pt, 1])
    [1.65042575879754...]

    sage: stdfun.exp.dop(exp(x))
    0
    sage: stdfun.exp.dop.numerical_solution(
    ....:         stdfun.exp.ini, [stdfun.exp.pt, 1])
    [2.71828182845904...]

    sage: stdfun.log.dop(log(x))
    0
    sage: stdfun.log.dop.numerical_solution(
    ....:         stdfun.log.ini, [stdfun.log.pt, 2])
    [0.69314718055994...]

    sage: e1 = stdfun.mittag_leffler_e(1)
    sage: e1.dop(exp(x))
    0
    sage: e1.dop.numerical_solution(
    ....:         e1.ini, [e1.pt, 1])
    [2.71828182845904...]

    sage: e2 = stdfun.mittag_leffler_e(2)
    sage: e2.dop(cosh(sqrt(x))).simplify_full()
    0
    sage: e2.dop.numerical_solution(e2.ini, [e2.pt, 2])
    [2.17818355660857...]

    sage: eh = stdfun.mittag_leffler_e(1/2)
    sage: eh.dop(exp(x^2)*(1+erf(x))).simplify_full()
    0
    sage: eh.dop.numerical_solution(eh.ini, [eh.pt, 2])
    [108.940904389977...]

    sage: ea = stdfun.mittag_leffler_e(7/3, 1/2)
    sage: ea.dop.numerical_solution(ea.ini, [ea.pt, 1])
    [1.17691287735093...]

    sage: stdfun.sin.dop(sin(x))
    0
    sage: stdfun.sin.dop.numerical_solution(
    ....:         stdfun.sin.ini, [stdfun.sin.pt, pi/3])
    [0.86602540378443...]
"""

import collections
import sage.functions.all as funs
from sage.all import pi, prod, QQ, ZZ
from ore_algebra import DifferentialOperators

Dops, x, Dx = DifferentialOperators(QQ, 'x')

# XXX: perhaps use DFiniteFunction once it gets stable enough.
IVP = collections.namedtuple("IVP", ["pt", "dop", "ini"])

airy_ai = IVP(
        0,
        Dx**2 - x,
        [ QQ(1)/3*3**(QQ(1)/3)/funs.gamma(QQ(2)/3),
         -QQ(1)/2*3**(QQ(1)/6)/pi*funs.gamma(QQ(2)/3)])

airy_bi = IVP(
        0,
        airy_ai.dop,
        [QQ(1)/3*3**(QQ(5)/6)/funs.gamma(QQ(2)/3),
         QQ(1)/2*3**(QQ(2)/3)/pi*funs.gamma(QQ(2)/3)])

arctan = IVP(
        0,
        (x**2 + 1)*Dx**2 + 2*x*Dx,
        [0, 1])

cos = IVP(
        0,
        Dx**2 + 1,
        [1, 0])

dawson = IVP(
        0,
        Dx**2 + 2*x*Dx + 2,
        [0, 1])

erf = IVP(
        0,
        Dx**2 + 2*x*Dx,
        [0, 2/funs.sqrt(pi)])

erfi = IVP(
        0,
        Dx**2 - 2*x*Dx,
        [0, 2/funs.sqrt(pi)])

exp = IVP(
        0,
        Dx - 1,
        [1])

log = IVP(
        1,
        Dx*x*Dx,
        [0, 1])

def mittag_leffler_e(alpha, beta=1):
    alpha = QQ.coerce(alpha)
    if alpha <= QQ.zero():
        raise ValueError
    num, den = alpha.numerator(), alpha.denominator()
    dop0 = prod(alpha*x*Dx + beta - num + t for t in range(num)) - x**den
    expo = dop0.indicial_polynomial(x).roots(QQ, multiplicities=False)
    pre = prod((x*Dx - k) for k in range(den) if QQ(k) not in expo)
    dop = pre*dop0 # homogenize
    dop = dop.numerator().primitive_part()
    expo = sorted(dop.indicial_polynomial(x).roots(QQ, multiplicities=False))
    assert len(expo) == dop.order()
    ini = [(1/funs.gamma(alpha*k + beta) if k in ZZ else 0) for k in expo]
    return IVP(0, dop, ini)

sin = IVP(
        0,
        cos.dop,
        [0, 1])


