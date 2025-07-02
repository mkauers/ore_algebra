# -*- vim: tw=80
"""
Examples and testcases for operator factorization
"""

# Copyright 2021 Alexandre Goyer, Inria Saclay Ile-de-France
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

from sage.rings.rational_field import QQ
from ore_algebra import DifferentialOperators

Diffops, z, Dz = DifferentialOperators(QQ, 'z')

# Face-centred cubic lattices (http://koutschan.de/data/fcc1/)
# (fcc4, fcc5, fcc6 are already available in ore_algebra.examples)
fcc3 = 2*(-1+z)*z**2*(3+z)**2*Dz**3+3*z*(3+z)*(-6+5*z+5*z**2)*Dz**2+6*(-3+3*z+12*z**2+4*z**3)*Dz+6*z*(2+z)

# Fuchsian operators with linear factors but without rational solution:
# - the first one is the annihilator of sqrt(1+z) and sqrt(1+2z).
# - thanks to Emre Sertoz for the second one (arosing from actual computations).
# - the differential Galois group (= monodromy group) of the third one is composed by homotheties.
sqrt_dop = (4*z**2 + 6*z + 2)*Dz**2 + (4*z + 3)*Dz - 1
sertoz_dop = (-128*z**2 + 8)*(z*Dz)**3 + (-256*z**2-24)*(z*Dz)**2 + (32*z**2 + 10)*(z*Dz)+ 64*z**2
exact_guessing_dop = (2*z*Dz - 1).lclm(2*z*Dz - 3)

# Annihilator of the hypergeometric function 2F1(a,b;c;z).
hypergeo_dop = lambda a,b,c: z*(1 - z)*Dz**2 + (c - (a + b + 1)*z)*Dz - a*b

# Kummer confluent hypergeometric function 1F1. !Not Fuchsian!
# This equation admits f_{a, b} = sum_{n >= 0} ( (a)_n / (b)_n ) ( z^n / n! )
# as solution, where (x)_n = x (x + 1) ··· (x + n - 1).
#
# Note that f_{-N, b} is a polynomial of degree N for N a nonnegative integer.
# Therefore, the associated equation (of degree 1) admits a right-hand factor
# of arbitrary degree N.
hypergeo_1f1_dop = lambda a,b: z*Dz^2 + (b-z)*Dz - a

# The two following operators are particular cases of hypergeo_1f1_dop.
# - for the first one: b = 2, a = -k,
# reference: Explicit degree bounds ..., Bostan, Rivoal, Salvy, 2019
# - for the second one: b = 0, a = -n, z <-> n*z
# reference: PhD thesis of van Hoeij (section 3.1)
bostan_dop = lambda k: z*Dz**2 + (2-z)*Dz + k
vanhoeij_dop = lambda n: Dz**2 - (1/n)*Dz + n/z

# The following operators do not admit factorization in QQ(z)<Dz> (to be
# confirmed). However the factorisation in QQbar(z)<Dz> involves algebraic
# numbers of degree n!.
QQbar_dop = lambda n: sum(z**i*Dz**i for i in range(n+1))

# This reducible operator does not admit any factorization in QQ[z]<Dz> (the
# first Weyl algebra).
irr_weyl_dop = Dz**2 + (-z**2 + 2*z)*Dz + z - 2

# The following operator have 0 as only integer exponent (with multiplicity 1)
# but its adjoint has no power series solution (the only integer exponent is -1).
adjoint_exponent_dop = 2*z**3*Dz**2 + (5*z**2 + 6*z)*Dz + z

# This operator illustrates the newton polygon's definition.
# reference: Formal Solutions ..., van Hoeij, 1997 (Example 3.1)
Tz = z*Dz
newton_dop = z**6*Tz**9 + 2*z**5*Tz**8 + 3*z**4*Tz**7 + 2*z**3*Tz**6 + (z**2 + 2*z**4)*Tz**5 + (5*z**2 - 3*z)*Tz**3 + 3*z*Tz**2 + (2 + 2*z)*Tz + 7*z

# The algebraicity of this operator is still an open question.
# reference: Transcendence Certificates ..., Kauers, Koutschan, Verron, 2023 (Example 8)
kauers_dop = (z-2)**3*(z-1)**3*z**3*Dz**3 + QQ(19/5)*(z-2)**2*(z-1)**2*z**2*(z**2 - QQ(16547/9576)*z + QQ(2420/1197))*Dz**2 + QQ(99/80)*(z-2)*(z-1)*z*(z**4 + QQ(8816399/112266)*z**3 - QQ(8566381/37422)*z**2 + QQ(7980386/56133)*z - QQ(3200/6237))*Dz - QQ(9/20)*z**6 + QQ(5640547/68040)*z**5 - QQ(20050393/136080)*z**4 - QQ(2904319/30240)*z**3 + QQ(5167531/54432)*z**2 + QQ(1144387/19440)*z + QQ(320/63)

# In the following list, the k-th operator annihilates the power series whose
# general term is the constant term of (x1 + 1/x1 + ... + xk + 1/xk)**n.
# Note : the k-th operator is Fuchsian and of order k.
# Conjecture: for all k, the k-th operator is irreducible.
# reference: On p-Integrality of Instanton Numbers, Beukers, Vlasenko, 2021
beukers_vlasenko_dops = [(4*z**2 - 1)*Dz + 4*z, (16*z**3 - z)*Dz**2 + (48*z**2 - 1)*Dz + 16*z, (-144*z**6 + 40*z**4 - z**2)*Dz**3 + (-1296*z**5 + 240*z**3 - 3*z)*Dz**2 + (-2592*z**4 + 288*z**2 - 1)*Dz - 864*z**3 + 48*z, (-1024*z**7 + 80*z**5 - z**3)*Dz**4 + (-14336*z**6 + 800*z**4 - 6*z**2)*Dz**3 + (-55296*z**5 + 2048*z**3 - 7*z)*Dz**2 + (-61440*z**4 + 1344*z**2 - 1)*Dz - 12288*z**3 + 128*z, (14400*z**10 - 4144*z**8 + 140*z**6 - z**4)*Dz**5 + (360000*z**9 - 82880*z**7 + 2100*z**5 - 10*z**3)*Dz**4 + (2880000*z**8 - 515520*z**6 + 9268*z**4 - 25*z**2)*Dz**3 + (8640000*z**7 - 1158720*z**5 + 13608*z**3 - 15*z)*Dz**2 + (8640000*z**6 - 825600*z**4 + 5528*z**2 - 1)*Dz + 1728000*z**5 - 109440*z**3 + 320*z, (147456*z**11 - 12544*z**9 + 224*z**7 - z**5)*Dz**6 + (4866048*z**10 - 338688*z**8 + 4704*z**6 - 15*z**4)*Dz**5 + (55296000*z**9 - 3075840*z**7 + 31808*z**5 - 65*z**3)*Dz**4 + (265420800*z**8 - 11450880*z**6 + 82880*z**4 - 90*z**2)*Dz**3 + (530841600*z**7 - 17072640*z**5 + 78720*z**3 - 31*z)*Dz**2 + (371589120*z**6 - 8432640*z**4 + 21120*z**2 - 1)*Dz + 53084160*z**5 - 783360*z**3 + 768*z, (-2822400*z**14 + 826624*z**12 - 31584*z**10 + 336*z**8 - z**6)*Dz**7 + (-138297600*z**13 + 34718208*z**11 - 1105440*z**9 + 9408*z**7 - 21*z**5)*Dz**6 + (-2489356800*z**12 + 528711680*z**10 - 13751904*z**8 + 90384*z**6 - 140*z**4)*Dz**5 + (-20744640000*z**11 + 3670284800*z**9 - 76058880*z**7 + 367920*z**5 - 350*z**3)*Dz**4 + (-82978560000*z**10 + 12003174400*z**8 - 191870208*z**6 + 637488*z**4 - 301*z**2)*Dz**3 + (-149361408000*z**9 + 17261690880*z**7 - 203784192*z**5 + 417888*z**3 - 63*z)*Dz**2 + (-99574272000*z**8 + 8929320960*z**6 - 73200384*z**4 + 76960*z**2 - 1)*Dz - 14224896000*z**7 + 952627200*z**5 - 4935168*z**3 + 1792*z, (-37748736*z**15 + 3358720*z**13 - 69888*z**11 + 480*z**9 - z**7)*Dz**8 + (-2264924160*z**14 + 174653440*z**12 - 3075072*z**10 + 17280*z**8 - 28*z**6)*Dz**7 + (-51791265792*z**13 + 3421896704*z**11 - 50110464*z**9 + 223776*z**7 - 266*z**5)*Dz**6 + (-577102675968*z**12 + 32232701952*z**10 - 384334848*z**8 + 1312416*z**6 - 1050*z**4)*Dz**5 + (-3329438515200*z**11 + 154681344000*z**9 - 1461829632*z**7 + 3620160*z**5 - 1701*z**3)*Dz**4 + (-9766352977920*z**10 + 370055577600*z**8 - 2675785728*z**6 + 4449600*z**4 - 966*z**2)*Dz**3 + (-13317754060800*z**9 + 401552179200*z**7 - 2117173248*z**5 + 2094976*z**3 - 127*z)*Dz**2 + (-6849130659840*z**8 + 159205294080*z**6 - 571023360*z**4 + 271488*z**2 - 1)*Dz - 761014517760*z**7 + 13070499840*z**5 - 28606464*z**3 + 4096*z, (914457600*z**18 - 270648576*z**16 + 11059840*z**14 - 140448*z**12 + 660*z**10 - z**8)*Dz**9 + (74071065600*z**17 - 19486697472*z**15 + 696769920*z**13 - 7584192*z**11 + 29700*z**9 - 36*z**7)*Dz**8 + (2370274099200*z**16 - 550176012288*z**14 + 17038986624*z**12 - 156601632*z**10 + 498696*z**8 - 462*z**6)*Dz**7 + (38714476953600*z**15 - 7861661079552*z**13 + 208388936448*z**11 - 1587838560*z**9 + 3984288*z**7 - 2646*z**5)*Dz**6 + (348430292582400*z**14 - 61302652637184*z**12 + 1371240707328*z**10 - 8466726048*z**8 + 16052916*z**6 - 6951*z**4)*Dz**5 + (1742151462912000*z**13 - 262594912849920*z**11 + 4872727756800*z**9 - 23679448320*z**7 + 32006700*z**5 - 7770*z**3)*Dz**4 + (4645737234432000*z**12 - 592052526858240*z**10 + 8923673318400*z**8 - 32838948864*z**6 + 29054476*z**4 - 3025*z**2)*Dz**3 + (5973090729984000*z**11 - 633557545205760*z**9 + 7552421959680*z**7 - 19955195136*z**5 + 10089816*z**3 - 255*z)*Dz**2 + (2986545364992000*z**10 - 258683438039040*z**8 + 2355318466560*z**6 - 4133152512*z**4 + 935592*z**2 - 1)*Dz + 331838373888000*z**9 - 22924354682880*z**7 + 152023080960*z**5 - 156432384*z**3 + 9216*z, (15099494400*z**19 - 1381236736*z**17 + 31313920*z**15 - 261888*z**13 + 880*z**11 - z**9)*Dz**10 + (1434451968000*z**18 - 117405122560*z**16 + 2348544000*z**14 - 17022720*z**12 + 48400*z**10 - 45*z**8)*Dz**9 + (55037657088000*z**17 - 4003398942720*z**15 + 70018781184*z**13 - 434058240*z**11 + 1022736*z**9 - 750*z**7)*Dz**8 + (1108906868736000*z**16 - 71140560076800*z**14 + 1076366573568*z**12 - 5616568320*z**10 + 10682496*z**8 - 5880*z**6)*Dz**7 + (12785043898368000*z**15 - 717120208896000*z**13 + 9269412790272*z**11 - 39924943872*z**9 + 59245296*z**7 - 22827*z**5)*Dz**6 + (86299046313984000*z**14 - 4189825542389760*z**12 + 45579233525760*z**10 - 158120100864*z**8 + 174913200*z**6 - 42525*z**4)*Dz**5 + (335607402332160000*z**13 - 13938763707187200*z**11 + 125309288448000*z**9 - 339239956480*z**7 + 262498016*z**5 - 34105*z**3)*Dz**4 + (712309588623360000*z**12 - 24959281161830400*z**10 + 181267082772480*z**8 - 367154708480*z**6 + 180852320*z**4 - 9330*z**2)*Dz**3 + (739706111262720000*z**11 - 21504803733504000*z**9 + 122558195957760*z**7 - 175131279360*z**5 + 47196224*z**3 - 511*z)*Dz**2 + (301361749032960000*z**10 - 7121431166976000*z**8 + 30648175165440*z**6 - 28318801920*z**4 + 3168192*z**2 - 1)*Dz + 27396522639360000*z**9 - 512812803686400*z**7 + 1580138496000*z**5 - 819609600*z**3 + 20480*z]

# The following irreducible fuchsian operators are related to mirror symmetry.
# references:
# - Calabi-Yau operators, van Straten, 2017
# - Arithmetic and Topology of Differential Equations, Zagier, 2016 (section 10)
van_straten_dop = Tz**4 - 5*z*(5*Tz + 1)*(5*Tz + 2)*(5*Tz + 3)*(5*Tz + 4)
zagier_dop_a = 1800*z*(7*z - 62)*(z**2 + 50*z + 20)*Dz**2 + (30240*z**3 + 124560*z**2 - 10245600*z - 446400)*Dz + 6048*z**2 - 139453*z - 249550
zagier_dop_b = 90000*z**3*(2911*z+310)*(z**2+50*z+20)*Dz**4+18000*z**2*(154283*z**3+5185005*z**2+1675710*z+142600)*Dz**3+50*z*(147290778*z**3+2740219655*z**2+566777510*z+37497600)*Dz**2+(4599496440*z**3+28145233025*z**2+6744696050*z+53568000)*Dz+250881624*z**2-19383210*z+22459500

# The following irreducible fuchsian operators annihilate the power series of
# general term:
# ( (6n)! n! ) / ( (3n)! (2n)! (2n)! ) for the first one,
# ( (30n)! n! ) / ( (16n)! (10n)! (5n)! ) for the second one.
# These operators are algebraic.
# reference: Integral Ratios of Factorials ..., Villegas, 2007
simple_chebychev_dop = (216*z**2 - 2*z)*Dz**2 + (432*z - 1)*Dz + 30
chebychev_dop = (11337408000000000*z**8 - 11250*z**7)*Dz**8 + (362797056000000000*z**7 - 275625*z**6)*Dz**7 + (4043927462400000000*z**6 - 2223250*z**5)*Dz**6 + (19459527091200000000*z**5 - 7118750*z**4)*Dz**5 + (40810981455014400000*z**4 - 8665432*z**3)*Dz**4 + (33378063480115200000*z**3 - 3181944*z**2)*Dz**3 + (7972406431637760000*z**2 - 181392*z)*Dz**2 + (251637206929920000*z - 48)*Dz + 3726543300480

# Thanks to Thomas Cluzeau to propose me (A. G.) the following operator for
# experimenting the computation of the Lie algebra of the differential Galois
# group.
cluzeau_dop = 125*z**3*(1207249920*z+31)*(859963392000*z**2+10368000*z+1)*Dz**4+25*z**2*(55024109018331217920000*z**3+445891448733696000*z**2+34747522560*z+713)*Dz**3+60*z*(60799044988415508480000*z**3+272741734877184000*z**2+13602660240*z+217)*Dz**2+(2278309570579770900480000*z**3+3361648619556864000*z**2+194247246240*z+372)*Dz+124271431122532958208000*z**2-2315118197145600*z+646833600

# The Maple command DEtools[DFactor] fails with the following operator.
# Thanks to Bruno Salvy for reporting it. We suspect that the large exponent
# (=-972) at point 3 involves the resolution of a large system. !Not Fuchsian!
salvy_dop = (z**2*Dz + 3)*((z - 3)*Dz + 4*z**5)

# The Maple command DEtools[DFactor] fails to factor the operator QPP (it stops
# after >10min returning a warning message).
# The operators QPP and QPPR have the same local exponents structure, but QPPR
# is irreducible.
QPP = ( (48*z**6 - 288*z**5 + 624*z**4 - 576*z**3 + 192*z**2)*Dz**2 + (-2064*z**5 + 10752*z**4 - 20208*z**3 + 16320*z**2 - 4800*z)*Dz + 7680*z**4 - 53348*z**3 + 127788*z**2 - 111400*z + 29376 ) * ( (4*z**6 - 24*z**5 + 52*z**4 - 48*z**3 + 16*z**2)*Dz**2 + (-184*z**5 + 868*z**4 - 1460*z**3 + 1064*z**2 - 288*z)*Dz + 1984*z**4 - 5223*z**3 + 5305*z**2 - 4254*z + 1408 )**2
QPPR = QPP + 192*z**3 - 576*z**2 + 384*z
