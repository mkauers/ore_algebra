# coding: utf-8
r"""
Asymptotic enumeration of Compacted Binary Trees

after Genitrini et al.

This example shows how to compute rigorous enclosures of the constant factor
in front of the asymptotic expansion of a P-recursive sequence using
numerical_transition_matrix().

We consider the example of compacted binary trees of right height at most k, as
studied by Genitrini, Gittenberger, Kauers and Wallner (*Asymptotic Enumeration
of Compacted Binary Trees*, arXiv:1703.10031 [math.CO], 2017). Thanks to Michael
Wallner for the data and his explanations.

Our goal is to determine a numerical enclosure of the constant κ_{k} appearing
in Theorem 3.4 of the paper. Let us take for example k=10. Our starting point is
the operator M_{10} given in Theorem 8.6 of the paper, which annihilates the
exponential generating series of compacted binary trees of right height at
most 10::

    sage: from ore_algebra.examples import cbt
    sage: dop = cbt.dop[10]
    sage: dop
    (z^6 - 21*z^5 + 70*z^4 - 84*z^3 + 45*z^2 - 11*z + 1)*Dz^11 + ...

The singularities are::

    sage: dop.leading_coefficient().roots(QQbar, multiplicities=False)
    [0.2651878342412026?,
     0.3188643842942825?,
     0.4462147547781043?,
     0.7747192223207199?,
     1.988156536964752?,
     17.20685726740094?]

and the dominant one is::

    sage: s = _[0]
    sage: s
    0.2651878342412026?

The origin is an ordinary point, so any solution is characterized by the
coefficients in its series expansions of::

    sage: dop.local_basis_monomials(0)
    [1, z, z^2, z^3, z^4, z^5, z^6, z^7, z^8, z^9, z^10]

while at s, the relevant coefficients are those of::

    sage: dop.local_basis_monomials(s)
    [1,
     z - 0.2651878342412026?,
     (z - 0.2651878342412026?)^2,
     (z - 0.2651878342412026?)^3,
     (z - 0.2651878342412026?)^4,
     (z - 0.2651878342412026?)^4.260514654474679?,
     (z - 0.2651878342412026?)^5,
     (z - 0.2651878342412026?)^6,
     (z - 0.2651878342412026?)^7,
     (z - 0.2651878342412026?)^8,
     (z - 0.2651878342412026?)^9]

This is the matrix that sends the coefficients of a solution in the
basis (1 + O(z^11), z + O(z^11), ...) to its coefficients in the basis::

    (1 + O((z - 0.265)^10),
      ...,
     (z - 0.265...)^4.26... + + O((z - 0.265)^10),
     ...)

::

    sage: mat = dop.numerical_transition_matrix([0,s], 1e-30) # long time (1.5 s)

We are interested in the fifth row::

    sage: mat.row(5) # long time
    (..., [4.923936...e-6 +/- ...] + [-5.260478...e-6 +/- ...]*I, ...)

To get the constant corresponding to the particular solution we are interested
in, we need to multiply this (row) vector by the (column of) initial values at
the origin corresponding to the first 11 coefficients of the generating series
of compacted trees (of right height at most 10, but all compacted trees of size
at most 10 have right height at most 10)::

    sage: QQ[['z']](cbt.egf)
    1 + z + 3/2*z^2 + 5/2*z^3 + 37/8*z^4 + ...

This is an exponential generating series, but the
``numerical_transition_matrix()`` method uses the Taylor coefficients as
initial values, not the derivatives, so that we can simply set::

    sage: ini = list(cbt.egf)[:dop.order()]

The decomposition in the local basis at the dominant singularity s is
hence::

    sage: coef = mat*vector(ini) # long time

and we already saw that we were interested in the coefficient of
(z - s)^4.260514654474679? + O((z - s)^10), i.e. ::

    sage: coef[5] # long time
    [-15159.961154304779924257964877...] + [16196.11585885375838162771522...]*I

Sage's AsymptoticRing apparently doesn't digest the algebraic exponent yet, so
let's do the singularity analysis by hand. Slightly annoyingly, the local basis
used by the analytic continuation code is defined in terms of (z-s)^α, while
extraction of coefficients and application of transfer theorems are easier with
expressions of the form (1-z/s)^α. Since ::

    [z^n] (1-z/s)^α ~ (1/s)^n·n^(-α-1)/Γ(-α),

the constant κ_{10} we are looking for is given by ::

    sage: alpha = QQbar(dop.local_basis_monomials(s)[5].op[1])
    sage: C = ComplexBallField(100)
    sage: C(-s)^alpha*coef[5]/C(-alpha).gamma() # long time
    [645.8284296998659315345812...] + [+/- ...]*I

We glossed over a subtle point, though: in general, we may need to pay attention
to the choice of branch of the complex logarithm in the above transformation.
What happens in our case is the following. The element of the local basis at s
involving (z-s)^α has a branch cut to the left of s. Our analytic continuation
path [0,s] arrives at s “above” the branch cut, where log(z-s) = log(s-z) + iπ,
and hence::

    (z-s)^α = exp(α·i·π)·(s-z)^α = (-1)^α·(s-z)^α.

The rhs, with its branch cut now to the right of s, provides an analytic
continuation to a Δ-domain “to the left” of s of the branch of the local
solution to which the output of numerical_transition_matrix() refer.
This is exactly what we need in the context of singularity analysis, and
hence everything should be fine with the above formula. If however we had
used, say, an analytic continuation path of the form [0, -i, s], then
the required correction may have been something else than (-1)^α.

With a quick and dirty implementation of some cases of singularity analysis, we
can automate the process for other values of k::

    sage: def asy(dop, prec=100):
    ....:     real_sing = dop.leading_coefficient().roots(AA,
    ....:                                                  multiplicities=False)
    ....:     s = min(real_sing, key=abs)
    ....:     ini = list(cbt.egf[:dop.order()])
    ....:     mat = dop.numerical_transition_matrix([0, s], 10^(-prec))
    ....:     cmb = mat*vector(ini)
    ....:     loc = dop.local_basis_monomials(s)
    ....:     C = cmb[0].parent()
    ....:     res = []
    ....:     for pos, mon in enumerate(loc):
    ....:         if mon.nops() == 0 or mon.operator() is operator.add:
    ....:             pass
    ....:         elif mon.operator() is operator.pow:
    ....:             if mon.op[1] in ZZ:
    ....:                 pass
    ....:             else:
    ....:                 expo = mon.op[1].pyobject()
    ....:                 res.append(C(-s)^expo*cmb[pos]/C(-expo).gamma())
    ....:         elif mon.operator() is operator.mul:
    ....:             assert str(mon.op[1].operator()) == 'log'
    ....:             assert mon.op[0].operator is operator.pow
    ....:             expo = mon.op[0,1].pyobject()
    ....:             res.append(C(-s)^expo*cmb[pos]/C(-expo).gamma())
    ....:
    ....:     nonan = [pos for pos, mon in enumerate(loc)
    ....:              if not (mon.nops() == 0
    ....:                      or mon.operator() is not operator.pow
    ....:                      or mon.op[1] in ZZ)]
    ....:     if len(nonan) != 1:
    ....:         return CBF(NaN)
    ....:     pos = nonan[0]
    ....:     expo = loc[pos].op[1].pyobject()
    ....:     return (C(-s)^expo*cmb[pos]/C(-expo).gamma()).real()

We obtain::

    sage: for k, dop in cbt.dop.items(): # long time (4.6 s)
    ....:     print("{}\t{}".format(k, asy(dop, 50)))
    2  [0.5613226189564568270393235883810334361992061196...]
    3  [0.6049645385653542644762421366594344081594004532...]
    4  [0.8724609101215661991266866210277371543438236597...]
    5  [1.6248570260792824202355889447707451896274397227...]
    6  [3.7818686091669627122667627632166635874791894574...]
    7  [10.708084931657092542368755716629129442143758729...]
    8  [36.087875288239535234327556576384625828336477172...]
    9  [142.21543933025087303695985127830779667104241363...]
    10 [645.82842969986593153458120613640308394397361391...]
"""

from sage.all import ZZ, QQ
from ore_algebra import DifferentialOperators

Dops, z, Dz = DifferentialOperators(QQ, 'z')

egf = (1+z+(ZZ(3)/2)*z**2+(ZZ(5)/2)*z**3+(ZZ(37)/8)*z**4+(ZZ(373)/40)*z**5
        +(ZZ(4829)/240)*z**6+(ZZ(76981)/1680)*z**7+(ZZ(293057)/2688)*z**8
        +(ZZ(32536277)/120960)*z**9+(ZZ(827662693)/1209600)*z**10)

dop = {}

dop[2] = (z**2-3*z+1)*Dz**3+(-z**2+6*z-6)*Dz**2+(-2*z+3)*Dz
dop[3] = ((3*z**2-4*z+1)*Dz**4+(-4*z**2+18*z-10)*Dz**3+(z**2-12*z+14)*Dz**2
        +(z-3)*Dz)
dop[4] = ((-z**3+6*z**2-5*z+1)*Dz**5+(2*z**3-18*z**2+40*z-15)*Dz**4
        +(-z**3+16*z**2-54*z+41)*Dz**3+(-4*z**2+22*z-24)*Dz**2+(-2*z+3)*Dz)
dop[5] = ((-4*z**3+10*z**2-6*z+1)*Dz**6+(9*z**3-58*z**2+75*z-21)*Dz**5
        +(-6*z**3+78*z**2-184*z+95)*Dz**4+(z**3-33*z**2+141*z-110)*Dz**3
        +(3*z**2-32*z+40)*Dz**2+(z-3)*Dz)
dop[6] = ((z**4-10*z**3+15*z**2-7*z+1)*Dz**7
        +(-3*z**4+40*z**3-145*z**2+126*z-28)*Dz**6
        +(3*z**4-57*z**3+310*z**2-505*z+190)*Dz**5
        +(-z**4+34*z**3-261*z**2+636*z-375)*Dz**4
        +(-7*z**3+91*z**2-309*z+259)*Dz**3+(-10*z**2+54*z-60)*Dz**2+(-2*z+3)*Dz)
dop[7] = ((5*z**4-20*z**3+21*z**2-8*z+1)*Dz**8+(-16*z**4+140*z**3-306*z**2
    +196*z-36)*Dz**7+(18*z**4-280*z**3+1035*z**2-1182*z+343)*Dz**6+(-8*z**4
    +226*z**3-1344*z**2+2320*z-1050)*Dz**5+(z**4-72*z**3+741*z**2-1900*z
    +1189)*Dz**4+(6*z**3-154*z**2+648*z-550)*Dz**3+(7*z**2-72*z+92)*Dz**2
    +(z-3)*Dz)
dop[8] = ((-z**5+15*z**4-35*z**3+28*z**2-9*z+1)*Dz**9+(4*z**5-75*z**4+390*z**3
    -574*z**2+288*z-45)*Dz**8+(-6*z**5+146*z**4-1125*z**3+2931*z**2-2457*z
    +574)*Dz**7+(4*z**5-138*z**4+1436*z**3-5525*z**2+7176*z-2548)*Dz**6
    +(-z**5+63*z**4-913*z**3+4802*z**2-8875*z+4389)*Dz**5+(-11*z**4+278*z**3
    -2016*z**2+5056*z-3345)*Dz**4+(-31*z**3+376*z**2-1278*z+1109)*Dz**3
    +(-22*z**2+118*z-132)*Dz**2+(-2*z+3)*Dz)
dop[9] = ((-6*z**5+35*z**4-56*z**3+36*z**2-10*z+1)*Dz**10+(25*z**5-285*z**4
    +917*z**3-988*z**2+405*z-55)*Dz**9+(-40*z**5+750*z**4-3886*z**3+7231*z**2
    -4664*z+906)*Dz**8+(30*z**5-900*z**4+7035*z**3-19449*z**2+19397*z
    -5544)*Dz**7+(-10*z**5+525*z**4-6260*z**3+24640*z**2-34446*z+13797)*Dz**6
    +(z**5-135*z**4+2735*z**3-15730*z**2+29795*z-15729)*Dz**5+(10*z**4-510*z**3
    +4830*z**2-12760*z+8615)*Dz**4+(25*z**3-585*z**2+2445*z-2150)*Dz**3
    +(15*z**2-152*z+196)*Dz**2+(z-3)*Dz)
dop[10] = ((z**6-21*z**5+70*z**4-84*z**3+45*z**2-11*z+1)*Dz**11+(-5*
    z**6+126*z**5-875*z**4+1904*z**3-1593*z**2+550*z-66)*Dz**10+(10
    *z**6-310*z**5+3150*z**4-11606*z**3+15968*z**2-8244*z+1365)*Dz
    **9+(-10*z**6+400*z**5-5375*z**4+28736*z**3-60053*z**2+46936*z-
    11070)*Dz**8+(5*z**6-285*z**5+5000*z**4-35950*z**3+105408*z**2-
    115990*z+38199)*Dz**7+(-z**6+106*z**5-2595*z**4+24560*z**3-96425*z**2
    +142710*z-61740)*Dz**6+(-16*z**5+700*z**4-9135*z**3+47270*z**2
    -91925*z+50630)*Dz**5+(-75*z**4+1690*z**3-11925*z**2+30300*z-21075)*Dz**4
    +(-115*z**3+1351*z**2-4593*z+4051)*Dz**3+(-46*z**2+246*z-276)*Dz**2
    +(-2*z+3)*Dz)

