# -*- coding: utf-8 - vim: tw=80
"""
Symbolic-numeric algorithm for the factorization of linear differential
operators.

TESTS

Examples related to Small Step Walks (https://specfun.inria.fr/chyzak/ssw/)::

    sage: from ore_algebra.examples import ssw
    sage: dop = ssw.dop[13,0,0]
    sage: fac = dop.factor()
    sage: [x.order() for x in fac], prod(fac)==dop
    ([2, 1, 1], True)

    sage: from ore_algebra.analytic.factorization import _tests_ssw
    sage: _tests_ssw() # long time (50s, 9a7e7c6f of facto)
    True

    sage: dop = ssw.aux_dop
    sage: fac = dop.factor() # long time (10s, 9a7e7c6f of facto)
    sage: [x.order() for x in fac], prod(fac) == dop # long time
    ([5, 1, 1], True)


Examples related to random walks (http://koutschan.de/data/fcc1/)::

    sage: from ore_algebra.analytic.examples import fcc
    sage: from ore_algebra.analytic.examples.facto import fcc3
    sage: fcc4, fcc5, fcc6 = fcc.dop4, fcc.dop5, fcc.dop6

    sage: fac = fcc3.factor()
    sage: len(fac), fac[0] == fcc3
    (1, True)

    sage: fac = fcc4.factor()
    sage: len(fac), fac[0] == fcc4
    (1, True)

    sage: fac = fcc5.factor() # (1.5s)
    sage: len(fac), fac[0] == fcc5
    (1, True)

    sage: fac = fcc6.factor() # long time (35s, 9a7e7c6f of facto)
    sage: len(fac), fac[0] == fcc6 # long time
    (1, True)

    sage: dop = fcc3**2
    sage: fac = dop.factor()
    sage: [x.order() for x in fac], prod(fac) == dop
    ([3, 3], True)

    sage: dop = fcc3*fcc4
    sage: fac = dop.factor() # long time (3s, 9a7e7c6f of facto)
    sage: [x.order() for x in fac], prod(fac) == dop # long time
    ([3, 4], True)

    sage: dop = fcc4*fcc3
    sage: fac = dop.factor() # long time (3s, 9a7e7c6f of facto)
    sage: [x.order() for x in fac], prod(fac) == dop # long time
    ([4, 3], True)

    sage: dop = fcc4**2
    sage: fac = dop.factor() # long time (2.5s, 9a7e7c6f of facto)
    sage: [x.order() for x in fac], prod(fac) == dop # long time
    ([4, 4], True)


Computing periods of rational integrals (v24.147 at https://pierre.lairez.fr/supp/periods/)::

    sage: from ore_algebra.analytic.examples.facto import z, Dz
    sage: dop = (21292817700*z**14 + 67398344860*z**13 + 94987355417*z**12 + 78506801252*z**11 + 42103272002*z**10 + 15265487382*z**9 + 3762840342*z**8 + 612043042*z**7 + 59827597*z**6 + 2515785*z**5 - 71162*z**4 - 8281*z**3)*Dz**4 + (340685083200*z**13 + 1038647876420*z**12 + 1400226908946*z**11 + 1099765534766*z**10 + 557107264294*z**9 + 189803233386*z**8 + 43803871110*z**7 + 6671139470*z**6 + 617643708*z**5 + 26717232*z**4 - 256438*z**3 - 49686*z**2)*Dz**3 + (1533082874400*z**12 +4495150057860*z**11 + 5780819437704*z**10 + 4294604549884*z**9 + 2040150857974*z**8 + 646242963697*z**7 + 137523612381*z**6 + 19182940768*z**5 + 1627435425*z**4 + 67706000*z**3 + 123578*z**2 - 57967*z)*Dz**2 + (2044110499200*z**11 + 5755179562440*z**10 + 7037979509820*z**9 + 4917287798460*z**8 + 2170385444000*z**7 + 630170966596*z**6 + 121015612776*z**5 + 14952652460*z**4 + 1099932789*z**3 + 39329199*z**2 + 234325*z - 8281)*Dz + 511027624800*z**10 + 1379206428600*z**9 +1598005558080*z**8 + 1042508677240*z**7 + 422269162452*z**6 + 110147546634*z**5 + 18479595006*z**4 + 1915723890*z**3 + 110445426*z**2 + 2649920*z
    sage: fac = dop.factor()
    sage: len(fac), fac[0] == dop
    (1, True)


QPP, QPP+R, PQP (see [Chyzak, Goyer, Mezzarobba, ISSAC'22])::

    sage: from ore_algebra.analytic.examples.facto import z, Dz
    sage: Q = (48*z**6 - 288*z**5 + 624*z**4 - 576*z**3 + 192*z**2)*Dz**2 + (-2064*z**5 + 10752*z**4 - 20208*z**3 + 16320*z**2 - 4800*z)*Dz + 7680*z**4 - 53348*z**3 + 127788*z**2 - 111400*z + 29376
    sage: P = (4*z**6 - 24*z**5 + 52*z**4 - 48*z**3 + 16*z**2)*Dz**2 + (-184*z**5 + 868*z**4 - 1460*z**3 + 1064*z**2 - 288*z)*Dz + 1984*z**4 - 5223*z**3 + 5305*z**2 - 4254*z + 1408
    sage: R = 192*z**3 - 576*z**2 + 384*z

    sage: dop = Q*P*P; fac = dop.factor() # long time (13s, 9a7e7c6f of facto)
    sage: [x.order() for x in fac], prod(fac) == dop # long time
    ([2, 2, 2], True)

    sage: dop = P*Q*P; fac = dop.factor() # long time (18s, 9a7e7c6f of facto)
    sage: [x.order() for x in fac], prod(fac) == dop # long time
    ([2, 2, 2], True)

    sage: dop = Q*P*P + R; fac = dop.factor() # long time (10s, 9a7e7c6f of facto)
    sage: len(fac), fac[0] == dop # long time
    (1, True)


Examples with algebraic extensions::

    sage: from ore_algebra.analytic.examples.facto import QQbar_dop
    sage: dop = QQbar_dop(2); dop
    z^2*Dz^2 + z*Dz + 1
    sage: fac = dop.factor(); fac
    [z*Dz + a, z*Dz - a]
    sage: fac[0].parent()
    Univariate Ore algebra in Dz over Fraction Field of Univariate Polynomial Ring in z over Number Field in a with defining polynomial y^2 + 1 with a = 1*I

    sage: dop = QQbar_dop(3); fac = dop.factor(); fac # long time (0.5s <-> 25s, 9a7e7c6f of facto) ---> problème à résoudre
    [z*Dz - 20/28181*a0^5 + 701/56362*a0^4 + 334/28181*a0^3 + 2382/28181*a0^2 + 20005/56362*a0 - 14161/28181,
     z*Dz + 20/84543*a0^5 - 701/169086*a0^4 - 334/84543*a0^3 - 794/28181*a0^2 - 76367/169086*a0 - 70382/84543,
     z*Dz + 40/84543*a0^5 - 701/84543*a0^4 - 668/84543*a0^3 - 1588/28181*a0^2 + 8176/84543*a0 - 56221/84543]
    sage: fac[0].parent() # long time
    Univariate Ore algebra in Dz over Fraction Field of Univariate Polynomial Ring in z over Number Field in a0 with defining polynomial y^6 + 2*y^5 + 11*y^4 + 48*y^3 + 63*y^2 + 190*y + 1108 with a0 = -2.883024910498311? - 1.202820819285479?*I


Examples from Periods of hypersurfaces (ore_algebra.examples.periods)::

    sage: from ore_algebra.examples.periods import lairez_sertoz
    sage: from ore_algebra.examples.periods import *
    sage: dop = lairez_sertoz # long time (28s, d0b5a297 of facto)
    sage: fac = dop.factor() # long time
    sage: len(fac), fac[0] == dop # long time
    (1, True)

    sage: dop = dop_140118_4 # long time (4s, d0b5a297 of facto)
    sage: fac = dop.factor() # long time
    sage: len(fac), fac[0] == dop # long time
    (1, True)

    sage: dop = allODEs[2][0] # long time (2s, d0b5a297 of facto)
    sage: fac = dop.factor() # long time
    sage: len(fac), fac[0] == dop # long time
    (1, True)

    sage: for i in range(21): # long time (4s, d0b5a297 of facto)
    ....:     dop = allODEs[0][i] # long time
    ....:     fac = dop.factor() # long time
    ....:     assert prod(fac) == dop # long time

    sage: for i in range(21): # long time (1min, d0b5a297 of facto)
    ....:     dop = allODEs[1][i] # long time
    ....:     fac = dop.factor() # long time
    ....:     assert prod(fac) == dop # long time


Examples from Iterated integrals (ore_algebra.examples.iint)::

    sage: from ore_algebra.examples.iint import diffop, f, h, w
    sage: dop = diffop([f[1], w[3]]) # (A.15)
    sage: fac = dop.factor()
    sage: [x.order() for x in fac], prod(fac) == dop
    ([1, 1, 1], True)

    sage: dop = diffop([f[1/4], w[1], f[1]]) # (A.8)
    sage: fac = dop.factor()
    sage: [x.order() for x in fac], prod(fac) == dop
    ([1, 1, 1, 1], True)

    sage: dop = diffop([w[29], w[8]]) # (A.23)
    sage: fac = dop.factor()
    sage: [x.order() for x in fac], prod(fac) == dop
    ([1, 1, 1], True)

    sage: dop = diffop([h[0], w[8], w[8], f[1], f[1]]) # (A.69)
    sage: fac = dop.factor()
    sage: [x.order() for x in fac], prod(fac) == dop
    ([1, 1, 1, 1, 1, 1], True)


Examples from Asymptotic enumeration of Compacted Binary Trees::

    sage: from ore_algebra.examples import cbt
    sage: dop = cbt.dop[2]; fac = dop.factor()
    sage: [x.order() for x in fac], prod(fac) == dop
    ([2, 1], True)


Examples from ore_algebra.examples.polya::

    sage: from ore_algebra.examples import polya
    sage: dop = polya.dop[3]; fac = dop.factor()
    sage: len(fac), fac[0] == dop
    (1, True)

    sage: dop = polya.dop[4]; fac = dop.factor()
    sage: len(fac), fac[0] == dop
    (1, True)

    sage: for i in range(5, 8): # long time (9s, d0b5a297 of facto)
    ....:     dop = polya.dop[i] # long time
    ....:     fac = dop.factor() # long time
    ....:     assert len(fac) == 1 and fac[0] == dop # long time


Examples from ore_algebra.examples.stdfun::

    sage: from ore_algebra.examples.stdfun import *
    sage: dawson.dop.factor()
    [2*Dx, 1/2*Dx + x]

    sage: dop = mittag_leffler_e(1/2).dop # not fuchsian + without monodromy BUT ok
    sage: dop.factor()
    [-Dx, -1/2*Dx + x]


Examples from ore_algebra.analytic.examples.misc::

    sage: from ore_algebra.analytic.examples.misc import *
    sage: dop = pichon1_dop
    sage: fac = dop.factor(); len(fac), fac[0] == dop
    (1, True)

    sage: dop = pichon2_dop
    sage: fac = dop.factor(); len(fac), fac[0] == dop # long time (6s, 9a7e7c6f of facto)
    (1, True)

    sage: dop = pichon3_dop
    sage: fac = dop.factor() # long time (3s, 9a7e7c6f of facto)
    sage: [x.order() for x in fac], prod(fac) == dop # long time
    ([3, 1], True)

    sage: dop = chyzak1_dop
    sage: fac = dop.factor(); len(fac), fac[0] == dop # long time (55s, 9a7e7c6f of facto)
    (1, True)

    sage: dop = rodriguez_villegas_dop
    sage: fac = dop.factor(); len(fac), fac[0] == dop # long time (1.5s, 9a7e7c6f of facto)
    (1, True)

    sage: dop = salvy1_dop
    sage: fac = dop.factor() # long time (10s, 9a7e7c6f of facto)
    sage: [x.order() for x in fac], prod(fac) == dop # long time
    ([5, 1], True)

    sage: dop = quadric_slice_dop
    sage: fac = dop.factor() # long time (2.5s, 9a7e7c6f of facto)
    sage: [x.order() for x in fac], prod(fac) == dop # long time
    ([3, 1], True)

    sage: iint_quadratic_alg[0].factor() # long time (2.5s, 9a7e7c6f of facto)
    [(8680468749131953125000000000000000000000*x^6 + (34722222218750000000000000000000*a - 8680555572048611109375000000000000000000)*x^5 - 17369715535481778446278125000000000000000*x^4 + (-69479556937496488750000000000000*a + 17369889269113900656248244375000000000000)*x^3 + 8689246786349825321278125000000000000000*x^2 + (34757334718746488750000000000000*a - 8689333697065289546873244375000000000000)*x)*Dx - 26041406247395859375000000000000000000000*x^5 + (-138888888875000000000000000000000*a + 34722222288194444437500000000000000000000)*x^4 + 34739431070963556892556250000000000000000*x^3 + (208438670812489466250000000000000*a - 52109667807341701968744733125000000000000)*x^2 - 8689246786349825321278125000000000000000*x - 69514669437492977500000000000000*a + 17378667394130579093746488750000000000000,
     (x^7 - 41680711662497191/13888888887500000*x^5 + 2084737833124719028985671/694444444375000000000000*x^3 - 695146694374859478985671/694444444375000000000000*x)*Dx + 3*x^6 - 138945068874991573/27777777775000000*x^4 + 25025281/25000000*x^2 + 695146694374859478985671/694444444375000000000000,
     Dx]


Examples from ore_algebra.analytic.examples.facto::

    sage: from ore_algebra.analytic.examples.facto import *
    sage: dop = sqrt_dop; fac = dop.factor()
    sage: [x.order() for x in fac], prod(fac) == dop
    ([1, 1], True)

    sage: dop = sertoz_dop; fac = dop.factor()
    sage: [x.order() for x in fac], prod(fac) == dop
    ([1, 1, 1], True)

    sage: dop = salvy_dop; fac = dop.factor() # long time (1.5s, 9a7e7c6f of facto)
    sage: [x.order() for x in fac], prod(fac) == dop # long time
    ([1, 1], True)

    sage: bostan_dop(5).factor()
    [(-720*z/(z^5 - 30*z^4 + 300*z^3 - 1200*z^2 + 1800*z - 720))*Dz + (720*z - 1440)/(z^5 - 30*z^4 + 300*z^3 - 1200*z^2 + 1800*z - 720),
     (-1/720*z^5 + 1/24*z^4 - 5/12*z^3 + 5/3*z^2 - 5/2*z + 1)*Dz + 1/144*z^4 - 1/6*z^3 + 5/4*z^2 - 10/3*z + 5/2]

    sage: hypergeo_dop(1,2,3).factor()
    [(-z + 1)*Dz - 1, z*Dz + 2]


Others::

    sage: dop = Dz*(z*(z - 1)*(z - 23))*Dz + z # kontsevitch_odesskii for t = 23
    sage: fac = dop.factor()
    sage: len(fac), fac[0] == dop
    (1, True)

    sage: dop = ((2*z + 2)*Dz - 1).lclm((2*z - 2)*Dz - 1)
    sage: fac = dop.factor()
    sage: [x.order() for x in fac], prod(fac) == dop
    ([1, 1], True)

    sage: dop = (23328*z**30 - 2962656*z**29 + 167481432*z**28 - 5571556704*z**27 + 121729674150*z**26 - 1851395926086*z**25 + 40700956322629/2*z**24 - 166404060748989*z**23 + 2075213903062521/2*z**22 - 5039429066440560*z**21 + 19398104528727162*z**20 - 59991326512462044*z**19 + 150581068219429838*z**18 - 308893191659376156*z**17 + 519871215900006075*z**16 - 718528419860924574*z**15 + 813818978377742103*z**14 - 751139117952342336*z**13 + 559450624042995962*z**12 - 331096536795373020*z**11 +152073433106343000*z**10 - 52250704431108750*z**9 + 25272275126878125/2*z**8 - 1918124754778125*z**7 + 274941996890625/2*z**6)*Dz**6 + (2122848*z**29 - 251277552*z**28 + 13229814240*z**27 - 410223377880*z**26 + 8378506293030*z**25 - 119780974597491*z**24 + 1247451935860499*z**23 - 9753741779831230*z**22 + 58673669994057235*z**21 - 277014369667643170*z**20 + 1042973990433429142*z**19 - 3170371723000704238*z**18 + 7851382961812157130*z**17 - 15937382846325142040*z**16 +26603165524579973420*z**15 - 36533478019847750354*z**14 + 41171988038741033396*z**13 - 37854313332296302670*z**12 + 28111149739081942810*z**11 - 16600614478473845250*z**10 + 7612966802875410000*z**9 - 2613143160673903125*z**8 + 631635634021640625*z**7 - 95873891621625000*z**6 + 6873549922265625*z**5)*Dz**5 + (1772928*z**28 - 191060208*z**27 + 9303985296*z**26 - 272881870056*z**25 + 5414176964808*z**24 - 77161260988311*z**23 + 816942000683035*z**22 - 6564352661122430*z**21 +40730888154463260*z**20 - 198233317321299601*z**19 + 767285193309750131*z**18 - 2389345325654892933*z**17 + 6040114702696303343*z**16 - 12473992563919793515*z**15 + 21122384911404177021*z**14 - 29352110878545889685*z**13 + 33402828296724828019*z**12 - 30958636402896387771*z**11 + 23142803494946381409*z**10 - 13741279316128878615*z**9 + 6329965071185425125*z**8 - 2180688690729701250*z**7 + 528635258525362500*z**6 - 80415682172915625*z**5 + 5773781934703125*z**4)*Dz**4 + (3079296*z**27 -288314640*z**26 + 12291128568*z**25 - 322741102428*z**24 + 5933901927222*z**23 - 81199522929312*z**22 + 846595466790247*z**21 - 6783280316776317*z**20 + 42134994404044533*z**19 - 205280872844283139*z**18 + 794368617371094252*z**17 - 2469813602975656377*z**16 + 6227877247789998128*z**15 - 12822890652430754931*z**14 + 21643972013267727882*z**13 - 29983117541423786699*z**12 + 34021065623944456350*z**11 - 31446507800544432561*z**10 + 23448703697621235732*z**9 - 13889978259690737871*z**8 +6383770909597906290*z**7 - 2194230662646093225*z**6 + 530727301376148375*z**5 - 80562533545372500*z**4 + 5773781934703125*z**3)*Dz**3 + (1632960*z**26 - 154326384*z**25 + 7338837168*z**24 - 225663323436*z**23 + 4801890449016*z**22 - 72874733229288*z**21 + 808011192676735*z**20 - 6687793265835490*z**19 + 42187300083028603*z**18 - 206817067662336884*z**17 + 801560426130937658*z**16 - 2490727387230197624*z**15 + 6272063547301139206*z**14 - 12895041179674817944*z**13 +21738287032668387484*z**12 - 30083767255776896928*z**11 + 34108882884431529192*z**10 - 31507843213748157876*z**9 + 23481458285447155422*z**8 - 13902375219613699716*z**7 + 6386678527821802506*z**6 - 2194528584495136680*z**5 + 530717365635608925*z**4 - 80561169859929750*z**3 + 5773781934703125*z**2)*Dz**2 + (2356128*z**25 - 172802160*z**24 + 7290746496*z**23 - 220329890208*z**22 + 4748895867996*z**21 - 72989477514576*z**20 + 815018141851579*z**19 - 6762783377980019*z**18 +42644822731963227*z**17 - 208681301740955672*z**16 + 807002082826244992*z**15 - 2502711432378058869*z**14 + 6293052390293003992*z**13 - 12925897748676398084*z**12 + 21777653697882335640*z**11 - 30126732277129911711*z**10 + 34146782819340442956*z**9 - 31532942478475275696*z**8 + 23492924047512296220*z**7 - 13905563328105795207*z**6 + 6387084187992321264*z**5 - 2194529579280600948*z**4 + 530717772906189585*z**3 - 80561238605140050*z**2 + 5773781934703125*z)*Dz + 1376352*z**24 -124268256*z**23 + 6537065472*z**22 - 214686219744*z**21 + 4751137968516*z**20 - 73617218440524*z**19 + 823687823266975*z**18 - 6829979407739124*z**17 + 42978937261434045*z**16 - 209811099775500026*z**15 + 809761480688798142*z**14 - 2508043312587912396*z**13 + 6302258618186925892*z**12 - 12940989910793711676*z**11 + 21799463094240316488*z**10 - 30151403872312102182*z**9 + 34166775655227923766*z**8 - 31543700462524877778*z**7 + 23496355856172136314*z**6 - 13906054545221671128*z**5 +6387084190303343280*z**4 - 2194529586021124812*z**3 + 530717777625867777*z**2 - 80561239688783970*z + 5773781934703125 # fuchsian6 chez Marc
    sage: fac = dop.factor() # long time (28s, d0b5a297 of facto)
    sage: len(fac), fac[0] == dop # long time
    (1, True)
"""

# Copyright 2021 Alexandre Goyer, Inria Saclay Ile-de-France
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

from ore_algebra import guess
from ore_algebra.analytic.accuracy import PrecisionError
from ore_algebra.analytic.differential_operator import DifferentialOperator
from ore_algebra.analytic.linear_algebra import (invariant_subspace,
                                                 row_echelon_form, ker,
                                                 gen_eigenspaces, orbit,
                                                 customized_accuracy)
from ore_algebra.analytic.monodromy import _monodromy_matrices
from ore_algebra.analytic.utilities import as_embedded_number_field_elements
from ore_algebra.examples import ssw
from sage.arith.functions import lcm
from sage.arith.misc import algdep, gcd
from sage.functions.other import binomial, factorial
from sage.matrix.constructor import matrix
from sage.matrix.matrix_dense import Matrix_dense
from sage.matrix.special import block_matrix, identity_matrix, diagonal_matrix
from sage.misc.misc_c import prod
from sage.modules.free_module_element import (vector,
                                              FreeModuleElement_generic_dense)
from sage.rings.integer_ring import ZZ
from sage.rings.qqbar import QQbar
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RealField
from sympy.core.numbers import oo

Radii = RealField(30)



def factor(dop, verbose=False):
    r"""
    See the documentation of the associated method :meth:`factor`.
    """

    fac = _factor(dop, verbose)
    K0, K1 = fac[0].base_ring().base_ring(), fac[-1].base_ring().base_ring()
    if K0 != K1:
        A = fac[0].parent()
        fac = [A(f) for f in fac]

    return fac

def right_factor(dop, verbose=False):
    r"""
    See the documentation of the associated method :meth:`.right_factor`.
    """
    z, r = dop.base_ring().gen(), dop.order()
    if r < 2:
        return None
    R = _try_rational(dop)
    if R is not None:
        return R

    if verbose:
        print("### Trying to factor an operator of order", r)

    s0, sings = QQ.zero(), DifferentialOperator(dop)._singularities(QQbar)
    while s0 in sings:
        s0 = s0 + QQ.one()
    dop = dop.annihilator_of_composition(z + s0).monic()
    R = right_factor_via_monodromy(dop, verbose=verbose)

    if R is None:
        return None
    return R.annihilator_of_composition(z - s0)

def check_irreducible(dop, verbose=False, prec=None, max_prec=100000):
    r"""
    See the documentation of the associated method :meth:`check_irreducible`.
    """
    if prec is None:
        prec = 40*dop.order()
    if prec > max_prec:
        return

    if verbose:
        print("Try with precision", prec)

    mono, it = [], _monodromy_matrices(dop, 0, eps=Radii.one()>>prec)
    try:
        for pt, m, scal in it:
            if not scal:
                mono.append(m)
                if verbose:
                    print(len(mono), "matrices computed")
                if len(mono)>1:
                    if invariant_subspace(mono) is None:
                        return True
    except PrecisionError:
        pass

    return check_irreducible(dop, verbose=verbose, prec=prec<<1)

def check_minimal_annihilator(dop, initial_conditions, verbose=False, prec=None, max_prec=100000):
    r"""
    See the documentation of the associated method
    :meth:`check_minimal_annihilator`.
    """
    if prec is None:
        prec = 40*dop.order()
    if prec > max_prec:
        return

    if verbose:
        print("Try with precision", prec)

    mono, it = [], _monodromy_matrices(dop, 0, eps=Radii.one()>>prec)
    try:
        for pt, m, scal in it:
            if not scal:
                mono.append(m)
                if verbose:
                    print(len(mono), "matrices computed")
                V = orbit(mono, vector(mono[0].base_ring(), initial_conditions))
                if len(V) == dop.order():
                    return True
    except PrecisionError:
        pass

    return check_minimal_annihilator(dop, initial_conditions, verbose=verbose, prec=prec<<1, max_prec=max_prec)

################################################################################
### Hybrid algorithm, see [Chyzak, Goyer, Mezzarobba, 2022] ####################
################################################################################

def right_factor_via_monodromy(dop, order=None, bound=None, alg_degree=None, precision=None, loss=0, verbose=False):
    r"""
    Same as right_factor but focused on the hybrid algorithm.

    EXAMPLES::

        sage: from ore_algebra.analytic.factorization import right_factor_via_monodromy
        sage: from ore_algebra.analytic.examples.facto import hypergeo_dop, z
        sage: dop = hypergeo_dop(1/2,1/3,1/3).annihilator_of_composition(z-1); dop
        (z^2 - 3*z + 2)*Dz^2 + (11/6*z - 13/6)*Dz + 1/6
        sage: right_factor_via_monodromy(dop)
        (2*z - 4)*Dz + 1
    """

    r = dop.order()

    if bound is None:
        bound = _degree_bound_for_right_factor(dop)
        if verbose:
            print("Degree bound for right factor =", bound)

    if order is None:
        deg_of_dop = DifferentialOperator(dop).degree()
        order = max(min( r*deg_of_dop, 100, bound*(r + 1) + 1 ), 1)
    if alg_degree is None:
        alg_degree = dop.base_ring().base_ring().degree()
    if precision is None:
        precision = 50*(r + 1)

    if verbose:
        print("Current order of truncation =", order)
        print("Current working precision =", precision, "(before monodromy computation)")
        print("Current algebraic degree =", alg_degree)
        print("Starting to compute the monodromy matrices")

    try:
        mono, it = [], _monodromy_matrices(dop, 0, eps=Radii.one()>>precision)
        for pt, mat, scal in it:
            if not scal:
                local_loss = max(0, precision - customized_accuracy(mat))
                if local_loss > loss:
                    loss = local_loss
                    if verbose:
                        print("loss =", loss)
                mono.append(mat)
                if verbose:
                    print(len(mono), "matrices computed")
                conclusive_method ="One_Dimensional"
                R = one_dimensional_eigenspaces(dop, mono, order, bound, alg_degree, verbose)
                if R =="NotGoodConditions":
                    conclusive_method ="Simple_Eigenvalue"
                    R = simple_eigenvalue(dop, mono, order, bound, alg_degree, verbose)
                    if R =="NotGoodConditions":
                        conclusive_method ="Multiple_Eigenvalue"
                        R = multiple_eigenvalue(dop, mono, order, bound, alg_degree, verbose)
                if R !="Inconclusive":
                    if verbose:
                        print("Concluded with " + conclusive_method + " method")
                    return R
        if mono == []:
            return right_factor_when_monodromy_is_trivial(dop, order, verbose)


    except (ZeroDivisionError, PrecisionError):
        precision += max(150, precision - loss)
        return right_factor_via_monodromy(dop, order, bound, alg_degree, precision, loss, verbose)

    precision += max(150, precision - loss)
    order = min( bound*(r + 1) + 1, order<<1 )
    return right_factor_via_monodromy(dop, order, bound, alg_degree + 1, precision, loss, verbose)

def right_factor_when_monodromy_is_trivial(dop, order, verbose=False):
    r"""
    Same as right_factor_via_monodromy but focused on the case where the monodromy
    group is generated by homotheties.

    EXAMPLES::

        sage: from ore_algebra.analytic.factorization import right_factor_when_monodromy_is_trivial
        sage: from ore_algebra.examples.stdfun import dawson
        sage: dawson.dop, right_factor_when_monodromy_is_trivial(dawson.dop, 10)
        (Dx^2 + 2*x*Dx + 2, 1/2*Dx + x)
    """
    if verbose:
        print("Galois algebra is trivial: symbolic HP approximants method at order", order)
    K, r = dop.base_ring().base_ring(), dop.order()
    S = PowerSeriesRing(K, default_prec=order + r)
    f = dop.local_basis_expansions(QQ.zero(), order + r)[0]
    f = _formal_finite_sum_to_power_series(f, S)

    der = [ f.truncate() ]
    for k in range(r - 1):
        der.append( der[-1].derivative() )
    mat = matrix(r, 1, der)
    min_basis = mat.minimal_approximant_basis(max(order//r, 1))
    rdeg = min_basis.row_degrees()
    i0 = min(range(len(rdeg)), key = lambda i: rdeg[i])
    R = dop.parent()(list(min_basis[i0]))
    if dop%R == 0:
        return R

    order = order<<1
    return right_factor_when_monodromy_is_trivial(dop, order, verbose)

def one_dimensional_eigenspaces(dop, mono, order, bound, alg_degree, verbose=False):
    """
    output: a nontrivial right factor R of dop, or None, or ``NotGoodConditions``,
    or ``Inconclusive``
    """
    mat = _random_combination(mono)
    id = mat.parent().one()
    Spaces = gen_eigenspaces(mat)
    conclusive = True
    goodconditions = True
    for space in Spaces:
        eigvalue = space["eigenvalue"]
        eigspace = ker(mat - eigvalue*id)
        if len(eigspace) > 1:
            goodconditions = False
            break
        R = annihilator(dop, eigspace[0], order, bound, alg_degree, mono, verbose)
        if R =="Inconclusive":
            conclusive = False
        if R != dop:
            return R
    if not goodconditions:
        return "NotGoodConditions"
    if conclusive:
        return None
    return "Inconclusive"

def simple_eigenvalue(dop, mono, order, bound, alg_degree, verbose=False):
    """
    output: a nontrivial right factor R of dop, or None, or ``NotGoodConditions``,
    or ``Inconclusive``

    Assumption: dop is monic.
    """
    mat = _random_combination(mono)
    id = mat.parent().one()
    Spaces = gen_eigenspaces(mat)
    goodconditions = False
    for space in Spaces:
        if space['multiplicity'] == 1:
            goodconditions = True
            ic = space['basis'][0]
            R = annihilator(dop, ic, order, bound, alg_degree, mono, verbose)
            if R !="Inconclusive" and R!=dop:
                return R
            adj_dop = myadjoint(dop)
            Q = _transition_matrix_for_adjoint(dop)
            adj_mat = Q * mat.transpose() * (~Q)
            adj_mono = [ Q * m.transpose() * (~Q) for m in mono ]
            eigspace = ker(adj_mat - space['eigenvalue']*id)
            if eigspace == []:
                return "Inconclusive"
            if len(eigspace) > 1:
                break # raise PrecisionError?
            adj_ic = eigspace[0]
            adj_Q = annihilator(adj_dop, adj_ic, order, bound, alg_degree, adj_mono, verbose)
            # the bound could have to be changed
            if adj_Q !="Inconclusive" and adj_Q != adj_dop:
                return myadjoint(adj_dop//adj_Q)
            if R == dop and adj_Q == adj_dop:
                return None
            break
    if not goodconditions:
        return "NotGoodConditions"
    return "Inconclusive"

def multiple_eigenvalue(dop, mono, order, bound, alg_degree, verbose=False):
    """
    OUTPUT: a nontrivial right factor R of dop, or None, or ``Inconclusive``
    """
    r = dop.order()
    invspace = invariant_subspace(mono)
    if invspace is None:
        return None
    R = annihilator(dop, invspace[0], order, bound, alg_degree, mono, verbose)
    if R !="Inconclusive" and R.order() < r:
        return R
    return "Inconclusive"

def annihilator(dop, ic, order, bound, alg_degree, mono=None, verbose=False):
    """
    OUTPUT: ``dop``, or a nontrivial right factor of ``dop``, or ``Inconclusive``
    """
    r, OA = dop.order(), dop.parent()
    d = r - 1
    base_field = OA.base_ring().base_ring()

    if mono is not None:
        orb = orbit(mono, ic)
        d = len(orb)
        if d == r:
            return dop
        ic = _reduced_row_echelon_form(matrix(orb))[0]

    symb_ic, K = _guess_symbolic_coefficients(ic, alg_degree, verbose=verbose)
    if symb_ic !="NothingFound":
        if base_field != QQ and K != QQ:
            K = K.composite_fields(base_field)[0]
            symb_ic = [K(x) for x in symb_ic]
        S = PowerSeriesRing(K, default_prec=order + d)
        sol_basis = dop.local_basis_expansions(QQ.zero(), order + d)
        sol_basis = [ _formal_finite_sum_to_power_series(sol, S) for sol in sol_basis ]
        f = vector(symb_ic) * vector(sol_basis)
        if K == QQ and base_field == QQ:
            v = f.valuation()
            try:
                R = _substitution_map(guess(f.list()[v:], OA, order=d), -v)
                if 0 < R.order() < r and dop%R == 0:
                    return R
            except ValueError:
                pass
        else:
            der = [ f.truncate() ]
            for k in range(d): der.append( der[-1].derivative() )
            mat = matrix(d + 1, 1, der)
            if verbose: print("Trying to guess an annihilator with HP approximants")
            min_basis = mat.minimal_approximant_basis(order)
            rdeg = min_basis.row_degrees()
            if max(rdeg) > 1 + min(rdeg):
                i0 = min(range(len(rdeg)), key=lambda i: rdeg[i])
                R, g = DifferentialOperator(dop).extend_scalars(K.gen())
                R = R.parent()(list(min_basis[i0]))
                if dop%R == 0:
                    return R

    if order > r*(bound + 1) and verbose:
        print("Ball Hermite--Padé approximants not implemented yet")

    return "Inconclusive"


################################################################################
### Tools ######################################################################
################################################################################

def _try_rational(dop):
    r"""
    Return a first order right-hand factor of ``dop`` as soon as it has rational
    solutions.

    INPUT:
     -- ``dop`` -- differential operator

    OUTPUT:
     -- ``right_factor`` -- differential operator, or ``None``

    EXAMPLES::

        sage: from ore_algebra.analytic.factorization import _try_rational
        sage: from ore_algebra.examples import ssw
        sage: dop = ssw.dop[1,0,0]; dop
        (16*t^4 - t^2)*Dt^3 + (144*t^3 - 9*t)*Dt^2 + (288*t^2 - 15)*Dt + 96*t
        sage: _try_rational(dop)
        t*Dt + 2
    """
    for (f,) in dop.rational_solutions():
        d = f.gcd(f.derivative())
        right_factor = (1/d)*(f*dop.parent().gen() - f.derivative())
        return right_factor

    return None

def _reduced_row_echelon_form(mat):
    R, p = row_echelon_form(mat, pivots=True)
    rows = list(R)
    for j in p.keys():
        for i in range(p[j]):
            rows[i] = rows[i] - rows[i][j]*rows[p[j]]
    return matrix(rows)

def _local_exponents(dop, multiplicities=True):
    ind_pol = dop.indicial_polynomial(dop.base_ring().gen())
    return ind_pol.roots(QQbar, multiplicities=multiplicities)

def _largest_modulus_of_exponents(dop):

    z = dop.base_ring().gen()
    dop = DifferentialOperator(dop)
    lc = dop.leading_coefficient()//gcd(dop.list())

    out = 0
    for pol, _ in list(lc.factor()) + [ (1/z, None) ]:
        local_exponents = dop.indicial_polynomial(pol).roots(QQbar, multiplicities=False)
        local_largest_modulus = max([x.abs().ceil() for x in local_exponents], default=QQbar.zero())
        out = max(local_largest_modulus, out)

    return out

def _degree_bound_for_right_factor(dop):

    r = dop.order() - 1
    #S = len(dop.desingularize().leading_coefficient().roots(QQbar)) # too slow (example: QPP)
    S = len(DifferentialOperator(dop).leading_coefficient().roots(QQbar))
    E = _largest_modulus_of_exponents(dop)
    bound = r**2*(S + 1)*E + r*S + r**2*(r - 1)*(S - 1)/2

    return ZZ(bound)

def _random_combination(mono):
    prec, C = customized_accuracy(mono), mono[0].base_ring()
    if prec < 10:
        raise PrecisionError
    ran = lambda : C(QQ.random_element(prec), QQ.random_element(prec))
    return sum(ran()*mat for mat in mono)


myadjoint = lambda dop: sum((-dop.parent().gen())**i*pi for i, pi in enumerate(dop.list()))

def _diffop_companion_matrix(dop):
    r = dop.order()
    A = block_matrix([[matrix(r - 1 , 1, [0]*(r - 1)), identity_matrix(r - 1)],\
                      [ -matrix([[-dop.list()[0]]]) ,\
                        -matrix(1, r - 1, dop.list()[1:-1] )]], subdivide=False)
    return A

def _transition_matrix_for_adjoint(dop):
    """
    Return an invertible constant matrix Q such that: if M is the monodromy of
    dop along a loop gamma, then the monodromy of the adjoint of dop along
    gamma^{-1} is equal to Q*M.transpose()*(~Q), where the monodromies are
    computed in the basis given by ".local_basis_expansions()" method.

    Assumptions: dop is monic, 0 is the base point, and 0 is not singular.
    """
    AT = _diffop_companion_matrix(dop).transpose()
    r = dop.order()
    B = [identity_matrix(dop.base_ring(), r)]
    for k in range(1, r):
        Bk = B[k - 1].derivative() - B[k - 1] * AT
        B.append(Bk)
    P = matrix([ B[k][-1] for k in range(r) ])
    Delta = diagonal_matrix(QQ, [1/factorial(i) for i in range(r)])
    Q = Delta * P(0) * Delta
    return Q


def _guess_symbolic_coefficients(vec, alg_degree, verbose=False):
    """
    Return a reasonable symbolic vector contained in the ball vector ``vec``
    and its field of coefficients if something reasonable is found, or
    ``NothingFound`` otherwise.

    INPUT:
     -- ``vec``          -- ball vector
     -- ``alg_degree``   -- positive integer

    OUTPUT:
     -- ``symb_vec`` -- vector with exact coefficients, or ``NothingFound``
     -- ``K``        -- QQ, or a number field, or None (if ``symb_vec``=``NothingFound``)


    EXAMPLES::

        sage: from ore_algebra.analytic.factorization import _guess_symbolic_coefficients
        sage: C = ComplexBallField(); err = C(0).add_error(RR.one()>>40)
        sage: vec = vector(C, [C(sqrt(2)) + err, 3 + err])
        sage: _guess_symbolic_coefficients(vec, 1)
        ('NothingFound', None)
        sage: _guess_symbolic_coefficients(vec, 2)
        ([a, 3],
         Number Field in a with defining polynomial y^2 - 2 with a = 1.414213562373095?)
    """
    if verbose:
        print("Trying to guess symbolic coefficients")

    # fast attempt that working well if rational
    v1, v2 = [], []
    for x in vec:
        if not x.imag().contains_zero(): break
        x, err = x.real().mid(), x.rad()
        err1, err2 = err, 2*err/3
        v1.append(x.nearby_rational(max_error=x.parent()(err1)))
        v2.append(x.nearby_rational(max_error=x.parent()(err2)))
    if len(v1) == len(vec) and v1 == v2:
        if verbose:
            print("Found rational coefficients")
        return v1, QQ

    p = customized_accuracy(vec)
    if p<30: return "NothingFound", None
    for d in range(2, alg_degree + 1):
        v1, v2 = [], []
        for x in vec:
            v1.append(algdep(x.mid(), degree=d, known_bits=p-10))
            v2.append(algdep(x.mid(), degree=d, known_bits=p-20))
        if v1 == v2:
            symb_vec = []
            for i, x in enumerate(vec):
                roots = v1[i].roots(QQbar, multiplicities=False)
                k = len(roots)
                i = min(range(k), key = lambda i: abs(roots[i] - x.mid()))
                symb_vec.append(roots[i])
            K, symb_vec = as_embedded_number_field_elements(symb_vec)
            if not all(symb_vec[i] in x for i, x in enumerate(vec)):
                return "NothingFound", None
            if verbose:
                print("Found algebraic coefficients in a number field of degree", K.degree())
            return symb_vec, K

    return "NothingFound", None

_frobenius_norm = lambda m: sum([x.abs().mid()**2 for x in m.list()]).sqrt()

def _formal_finite_sum_to_power_series(f, PSR):
    """ Assumtion: x is extended at 0 (you have to shift otherwise). """
    if isinstance(f, list):
        return [ _formal_finite_sum_to_power_series(g, PSR) for g in f ]

    out = PSR.zero()
    for constant, monomial in f:
        if constant != 0:
            out += constant*PSR.gen()**monomial.n

    return out

def _euler_representation(dop):
    r"""
    Return the list of the coefficients of dop with respect to the powers of
    z*Dz.
    """
    z, n = dop.base_ring().gen(), dop.order()
    output = [ dop[0] ] + [0]*n
    l = [0] # coefficients of T(T-1)...(T-k+1) (initial: k=0)

    for k in range(1, n+1):

        newl = [0]
        for i in range(1, len(l)):
            newl.append((-k+1)*l[i]+l[i-1])
        l = newl + [1]

        ck = dop[k]
        for j in range(1, k+1):
            output[j] += ck*z**(-k)*l[j]

    return output

def _substitution_map(dop, e):
    r"""
    Return the operator obtained from ``dop`` after the substitution map
    T --> T + e, where T = z*Dz denotes Euler operator.

    NOTE: this corresponds to multiply the solution space by z^(-e).

    EXAMPLES::

        sage: from ore_algebra.analytic.factorization import _substitution_map
        sage: from ore_algebra.examples import ssw
        sage: dop = ssw.dop[1,0,0]; dop
        (16*t^4 - t^2)*Dt^3 + (144*t^3 - 9*t)*Dt^2 + (288*t^2 - 15)*Dt + 96*t
        sage: f = dop.power_series_solutions(20)[0]
        sage: dop(f)
        O(t^18)
        sage: dop = _substitution_map(dop,-3)
        sage: dop(f.parent().gen()^3*f)
        O(t^21)
    """
    l = _euler_representation(DifferentialOperator(dop))
    for i, c in enumerate(l):
        for k in range(i):
            l[k] += binomial(i, k)*e**(i - k)*c
    T = dop.base_ring().gen()*dop.parent().gen()
    output = sum(c*T**i for i, c in enumerate(l))

    return output

def _factor(dop, verbose=False):

    R = right_factor(dop, verbose)
    if R is None:
        return [dop]
    OA = R.parent()
    OA = OA.change_ring(OA.base_ring().fraction_field())
    Q = OA(dop)//R
    fac_left = _factor(Q, verbose)
    fac_right = _factor(R, verbose)
    return fac_left + fac_right

def _tests_ssw(): # to test all ssw examples
    b = True
    for k in range(1,20):
        for i in range(2):
            for j in range(2):
                dop = ssw.dop[k,i,j]
                fac = dop.factor()
                if prod(fac)!=dop: b = False
    return b
