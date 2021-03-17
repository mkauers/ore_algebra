# coding: utf-8
r"""
Iterated integrals

After J. Ablinger, J. Blümlein, C. G. Raab, and C. Schneider,
*Iterated Binomial Sums and their Associated Iterated Integrals*,
Journal of Mathematical Physics 55(11), 2014,
<https://doi.org/10.1063/1.4900836>.

Thanks to Jakob Ablinger and Clemens Raab for the data.

We first import some data from the paper as well as utility functions defined
in the example.iint submodule::

    sage: from ore_algebra.examples.iint import *

Using this data, we compute numerical enclosures of some of the iterated
integrals studied in the paper::

    sage: dop = diffop([f[1], w[3]]) # (A.15)
    sage: dop.local_basis_monomials(1)
    [log(x - 1), 1, sqrt(x - 1)]
    sage: iint_value(dop, [0, 0, 4*i])
    [-4.934802200544679...]
    sage: n(pi^2/2)
    4.934802200544...

::

    sage: dop = diffop([f[1/4], w[1], f[1]]) # (A.8)
    sage: dop.local_basis_monomials(1)
    [1, x - 1, (x - 1)^(3/2)*log(x - 1), (x - 1)^(3/2)]
    sage: iint_value(dop, [0, 0, 16/9*i, -16/27*i*(-3*i*pi+8)])
    [-3.445141853366646...]
    sage: n(-pi^3/9)
    -3.445141853366...

::

    sage: dop = diffop([w[29], w[8]]) # (A.23)
    sage: dop.local_basis_monomials(1)
    [log(x - 1), 1, x - 1]
    sage: iint_value(dop, [0, 0, -4/3])
    [1.562604825792972...]
    sage: n(4/9*(psi(1,1/3)-2/3*pi^2))
    1.562604825792...

::

    sage: dop = diffop([h[0], w[8], w[8], f[1], f[1]]) # (A.69)
    sage: dop.local_basis_monomials(1)
    [log(x - 1),
    1,
    x - 1,
    1/2*(x - 1)^2*log(x - 1)^2,
    (x - 1)^2*log(x - 1),
    (x - 1)^2]
    sage: myini = [0, 0, 0, 1/3, -2/3-i*pi/3, 11/12+2*i*pi/3-pi^2/6]
    sage: iint_value(dop, myini)
    [0.97080469562493...]
    sage: iint_value(dop, myini, algorithm="binsplit")
    [0.97080469...]
    sage: iint_value(dop, myini, 1e-500, deform=True) # long time (2.8 s)
    [0.97080469562493...0383420...]

Here are the known exact values for reference::

    sage: psi1_a = RBF(10.09559712542709408179200409989251636051890, rad=1e-35)
    sage: Li_a = RBF(0.67795750683172255251567721037981473544402459, rad=1e-35)
    sage: Li_b = RBF(-0.4484142069236462024430644059157743208342699, rad=1e-35)
    sage: Li_c = RBF(-0.4725978446588968746186231931265476736649961, rad=1e-35)
    sage: Li_d = RBF(0.34882786115484008421425193538699285536624537, rad=1e-35)
    sage: Li_e = RBF(0.51747906167389938633075816189886294562237747, rad=1e-35)

    sage: ref = dict()
    sage: ref[1] = 4*catalan
    sage: ref[2] = pi/2*zeta(2)
    sage: ref[3] = -2*pi*ln(2)
    sage: ref[4] = pi*(2*ln(2)^2 + zeta(2))
    sage: ref[5] = 2*pi*ln(2)
    sage: ref[6] = pi*(4*ln(2)^2 - zeta(2))
    sage: ref[7] = 0
    sage: ref[8] = -2*pi/3*zeta(2)
    sage: ref[9] = -pi*(2/9*zeta(3) - 4/3*ln(2)*zeta(2)
    ....:               + 2*pi/(9*sqrt(3))*(4*zeta(2) - psi1_a))
    sage: ref[10] = -2*pi/3*zeta(3)
    sage: ref[11] = -2*pi*ln((sqrt(5)-1)/2)
    sage: ref[12] = 2*pi*(2*Li_a - 6/5*zeta(3) - 6/5*ln((sqrt(5)-1)/2)*zeta(2)
    ....:                 + 2/3*ln((sqrt(5)-1)/2)^3)
    sage: ref[13] = -pi*zeta(2)
    sage: ref[14] = -2*pi*(4/5*zeta(3) + 9/5*ln((sqrt(5)-1)/2)*zeta(2)
    ....:                  - 2/3*ln((sqrt(5)-1)/2)^3)
    sage: ref[15] = 3*zeta(2)
    sage: ref[16] = zeta(2)/3
    sage: ref[17] = 2*arccot(sqrt(7))^2
    sage: ref[18] = 1/4*CBF(1/8).hypergeometric([1]*5, [3/2, 2, 2, 2])
    sage: ref[19] = (1/3*ln(3/2)^3 + ln(3/2)*zeta(2) + ln(3/2)*Li_b
    ....:            - Li_c - 2*Li_d)
    sage: ref[20] = -1/3*ln(2)^2 + 4/9*zeta(2) - 2/3*Li_b
    sage: ref[21] = 1/2*ln(2)^2 + zeta(2) + Li_b
    sage: ref[22] = 1/24*ln(2)^4 - ln(2)^2*zeta(2) - 4/5*zeta(2)^2 + Li_e
    sage: ref[23] = 4/9*(psi1_a - 4*zeta(2))
    sage: ref[24] = sqrt(2)*(2/3*zeta(2) - 2*Li_b - ln(2)^2)

Finally, we compute all the integrals and compare with the exact values when
applicable::

    sage: for k in sorted(word.keys()): # long time (10.2 s)
    ....:     dop = diffop(word[k])
    ....:     val = iint_value(dop, ini[k], deform=True)
    ....:     ok = "ok" if k in ref and val in RBF(ref[k]).add_error(1e-10) else ""
    ....:     print("(A.{})\t{}\t{}".format(k, val, ok))
    (A.1)   [3.66386237670887...]  ok
    (A.2)   [2.58385639002498...]  ok
    (A.3)   [-4.3551721806072...]  ok
    (A.4)   [8.18648809789096...]  ok
    (A.5)   [4.35517218060720...]  ok
    (A.6)   [0.86983785563201...]  ok
    (A.7)   [+/- ...e-1...]        ok
    (A.8)   [-3.4451418533666...]  ok
    (A.9)   [8.38881875897492...]  ok
    (A.10)  [-2.5175820907753...]  ok
    (A.11)  [3.02354306885557...]  ok
    (A.12)  [4.9576404218637...]   ok
    (A.13)  [-5.1677127800499...]  ok
    (A.14)  [2.44339103708582...]  ok
    (A.15)  [4.93480220054467...]  ok
    (A.16)  [0.54831135561607...]  ok
    (A.17)  [0.26117239648121...]  ok
    (A.18)  [0.25268502737327...]  ok
    (A.19)  [0.28230892870993...]  ok
    (A.20)  [0.86987360746446...]  ok
    (A.21)  [1.43674636688368...]  ok
    (A.22)  [-2.4278628067547...]  ok
    (A.23)  [1.56260482579297...]  ok
    (A.24)  [2.13970244864910...]  ok
    (A.27)  [-0.5342475125153...]
    (A.28)  [0.14083166505781...]
    (A.29)  [0.08671923333638...]
    (A.30)  [9.29746030760470...]
    (A.31)  [-1.2068741628578...]
    (A.32)  [0.76813058387865...]
    (A.34)  [0.02493993273621...]
    (A.35)  [0.25758589176283...]
    (A.36)  [0.01419229300628...]
    (A.37)  [0.01752989857942...]
    (A.38)  [0.00931804685640...]
    (A.39)  [0.00269874086778...]
    (A.41)  [0.00909837539280...]
    (A.42)  [0.12315273017782...]
    (A.43)  [0.13779105121530...]
    (A.44)  [0.03130068493969...]
    (A.45)  [-0.0016512243669...]
    (A.46)  [0.93615701249494...]
    (A.47)  [-0.3836443021183...]
    (A.48)  [-3.9987993660442...]
    (A.49)  [-0.0145924441289...]
    (A.50)  [0.00042865207521...]
    (A.51)  [0.00722568081669...]
    (A.52)  [-0.0017644712040...]
    (A.53)  [5.86168742725...e-5 ...]
    (A.54)  [-0.0005385608328...]
    (A.55)  [-0.0007156119070...]
    (A.56)  [0.00319604460177...]
    (A.57)  [0.00241736336932...]
    (A.58)  [0.06418646378008...]
    (A.59)  [0.07359150737673...]
    (A.60)  [0.01646793507128...]
    (A.61)  [-0.0002687223080...]
    (A.62)  [-0.0003463692070...]
    (A.63)  [-0.0001223539098...]
    (A.64)  [-6.9281364974...e-5 ...]
    (A.65)  [0.00029942509011...]
    (A.66)  [0.00076351201665...]
    (A.67)  [0.34902731697236...]
    (A.68)  [-0.1404863030270...]
    (A.69)  [-0.9708046956249...]
    (A.70)  [-0.0018467475220...]
"""

import string

from sage.functions.all import log, sqrt
from sage.misc.misc_c import prod
from sage.rings.all import AA, QQ, ZZ
from sage.symbolic.all import SR, pi
from sage.symbolic.constants import I

from ore_algebra import DifferentialOperators
from ore_algebra.analytic.path import Point

Dops, x, Dx = DifferentialOperators()
Rat = Dops.base_ring().fraction_field()

def diffop(word):
    dlog = [Rat(log(a).diff().canonicalize_radical()) for a in word]
    factors = [(Dx - sum(dlog[:i])) for i in range(len(dlog)) ]
    dop = prod(reversed( [(Dx - sum(dlog[:i])) for i in range(len(dlog) + 1) ] ))
    dop = dop.numerator()
    return dop

# - Second part not robust, but seems to be enough for all our examples
# - The last call to series() MUST use :-series (as opposed to
#   MultiSeries:-series), for MultiSeries always assumes that the expansion
#   variable tends to zero along the positive real axis
_ini_Hstar_code = r"""
    proc(word, x, ord)
        local u, v, w, i, letter, ser, ini;
        ser := 1;
        for i from nops(word) by -1 to 1 do
            letter := subs(x=1-u, word[i]);
            ser := MultiSeries:-series(int(MultiSeries:-series(letter*ser,u),u),u)
                   assuming u > 0;
        end do;
        ser := series(subs(u=-v, ser), v) assuming v < 0;
        ser := subs([ln(1/v)=-ln(v)+2*I*pi, ln(-v)=ln(v)-I*pi], ser);
        ser := map(collect, ser, [v, ln(v)], expand);

        ser := op(1, ser);
        ser := subs(v=1, subs([ln(v)^2=2*w^2, ln(v)=w], ser));
        ini := PolynomialTools[CoefficientList](ser, w);
        ini := [0$(ord - nops(ini)), op(ListTools[Reverse](ini))];
    end proc
    """

def get_ini_Hstar():
    from sage.interfaces.maple import maple
    return maple(string.replace(_ini_Hstar_code, '\n', ' '))

def iint_value(dop, ini, eps=1e-16, **kwds):
    roots = dop.leading_coefficient().roots(AA, multiplicities=False)
    sing = list(reversed(sorted([s for s in roots if 0 < s < 1])))
    path = [1] + [Point(s, dop, outgoing_branch=(0,-1)) for s in sing] + [0]
    val = dop.numerical_solution(ini, path, eps, **kwds)
    return val.real()

_one = ZZ.one()

class _F(object):
    def __getitem__(self, a):
        if a == 1:
            return 1/(1-x)
        else:
            return QQ(1-a).sign()/(x-a)
f = _F()

h = dict()
h[0] = x/(x-1)
h[1] = sqrt(x)/((x-1)*sqrt(8+x)) # rename?
h[2] = sqrt(x)/((x+1)*sqrt(8-x)) # rename?
h[3] = x/(x+1)
h[4] = x/((x+1)*sqrt(x+_one/4))
h[5] = x/((x-1)*sqrt(x-_one/4))
h[6] = 1/(x+1)

w = dict()
w[1]  = 1/(sqrt(x)*sqrt(1-x))
w[2]  = 1/sqrt(x*(1+x))
w[3]  = 1/(x*sqrt(1-x))
w[4]  = 1/(x*sqrt(1+x))
w[5]  = 1/sqrt((1+x)*(2+x))
w[6]  = 1/sqrt((1-x)*(2-x))
w[7]  = 1/sqrt((1-x)*(2+x))
w[8]  = 1/(x*sqrt(x-_one/4))
w[9]  = 1/((1-x)*sqrt(x))
w[10] = 1/(x*sqrt(2-x))
w[11] = 1/(x*sqrt((1-x)*(2-x)))
w[12] = 1/sqrt(x*(8-x))
w[13] = 1/((2-x)*sqrt(x*(8-x)))
w[14] = 1/(x*sqrt(x+_one/4))
w[15] = 1/(x*sqrt((1+x)*(2+x)))
w[16] = 1/(x*sqrt(2+x))
w[17] = 1/sqrt(x*(8+x))
w[18] = 1/((2+x)*sqrt(x*(8+x)))
w[19] = 1/sqrt(x*(4-x))
w[20] = 1/sqrt(x*(4+x))
w[21] = 1/sqrt((4+x)*(8+x))
w[22] = 1/((2+x)*sqrt(x-_one/4))
w[23] = 1/((1+x)*sqrt(x*(4+x)))
w[24] = 1/((2-x)*sqrt(x+_one/4))
w[25] = 1/sqrt((4-x)*(8-x))
w[26] = 1/((1-x)*sqrt(x*(4-x)))
w[27] = 1/((x+_one/2)*sqrt(x*(4-x)))
w[28] = 1/(x*sqrt(x+_one/8))
w[29] = 1/((1-x)*sqrt(x-_one/4))

word = dict()
word[ 1] = [w[2], w[1]]
word[ 2] = [w[2], w[2], w[1]]
word[ 3] = [w[1], f[1]]
word[ 4] = [w[1], f[1], f[1]]
word[ 5] = [f[1], w[1]]
word[ 6] = [f[1], w[1], f[0]]
word[ 7] = [f[_one/4], w[1]]
word[ 8] = [f[_one/4], w[1], f[1]]
word[ 9] = [f[_one/4], w[1], f[1], f[1]]
word[10] = [f[_one/4], w[1], w[1], w[1]]
word[11] = [f[-_one/4], w[1]]
word[12] = [f[-_one/4], f[0], f[0], w[1]]
word[13] = [f[_one/4], f[0], w[1]]
word[14] = [f[-_one/4], w[1], w[1], w[1]]
word[15] = [f[1], w[3]]
word[16] = [f[4], w[3]]
word[17] = [f[8], w[3]]
word[18] = [f[8], f[0], f[0], w[3]]
word[19] = [f[-2], f[1], f[0]]
word[20] = [w[27], w[19]]
word[21] = [f[-_one/2], f[0]]
word[22] = [f[_one/2], f[0], f[0], f[0]]
word[23] = [w[29], w[8]]
word[24] = [f[-_one/2], w[28]]
word[27] = [h[0], w[8], w[8]]
word[28] = [h[3], w[14], w[14]]
word[29] = [h[6], f[-_one/2], f[0], f[0], f[0]]
word[30] = [f[_one/4], f[0], w[1], f[1]]
word[31] = [f[_one/4], w[2], w[2], w[1]]
word[32] = [f[-_one/4], w[2], w[2], w[1]]
word[34] = [h[2], w[13]]
word[35] = [h[4], w[14]]
word[36] = [h[2], w[25]]
word[37] = [h[2], w[12], f[0]]
word[38] = [h[2], w[12], f[2]]
word[39] = [h[2], w[25], w[19]]
# word[40] = word[9]
word[41] = [h[2], w[13], f[1], f[0]]
word[42] = [h[4], w[14], f[0], f[0]]
word[43] = [h[4], w[14], f[1], f[0]]
word[44] = [h[4], w[14], f[-1], f[0]]
word[45] = [h[1], w[18], f[-1], f[0]]
word[46] = [h[5], w[8], f[0], f[1]]
word[47] = [h[5], w[8], f[1], f[0]]
word[48] = [h[5], w[8], f[1], f[1]]
word[49] = [h[5], w[8], f[-1], f[0]]
word[50] = [h[2], w[25], w[19], w[19]]
word[51] = [h[2], w[12], f[0], f[1], f[0]]
word[52] = [h[1], w[17], f[0], f[-1], f[0]]
word[53] = [h[2], w[25], w[19], w[19], w[19]]
word[54] = [h[1], w[21], w[20], w[19]]
word[55] = [h[1], w[21], w[23], f[0]]
word[56] = [h[2], w[25], w[26], f[0]]
word[57] = [h[2], w[12], f[2], f[1], f[0]]
word[58] = [h[3], w[14], w[14], f[0], f[0]]
word[59] = [h[3], w[14], w[14], f[1], f[0]]
word[60] = [h[3], w[14], w[14], f[-1], f[0]]
word[61] = [h[1], w[17], f[-2], f[-1], f[0]]
word[62] = [h[1], w[21], w[20], f[0], f[0]]
word[63] = [h[1], w[21], w[20], f[-1], f[0]]
word[64] = [h[1], w[21], w[20], w[19], w[19]]
word[65] = [h[2], w[25], w[19], f[0], f[0]]
word[66] = [h[2], w[25], w[19], f[1], f[0]]
word[67] = [h[0], w[8], w[8], f[0], f[1]]
word[68] = [h[0], w[8], w[8], f[1], f[0]]
word[69] = [h[0], w[8], w[8], f[1], f[1]]
word[70] = [h[0], w[8], w[8], f[-1], f[0]]

ini = dict()

# Computed with Maple, using get_ini_Hstar and some semi-manual
# postprocessing
ini[1] = [ZZ(0), ZZ(0), ZZ(2)/ZZ(3)*I*ZZ(2)**(ZZ(1)/ZZ(2))]
ini[2] = [ZZ(0), ZZ(0), ZZ(0), -ZZ(4)/ZZ(15)*I]
ini[3] = [ZZ(0), -ZZ(2)*I, -ZZ(2)*pi+ZZ(4)*I]
ini[4] = [ZZ(0), -ZZ(2)*I, -ZZ(2)*pi+ZZ(4)*I, I*pi**ZZ(2)+ZZ(4)*pi-ZZ(8)*I]
ini[5] = [ZZ(0), ZZ(0), -ZZ(4)*I]
ini[6] = [ZZ(0), ZZ(0), ZZ(0), ZZ(4)/ZZ(9)*I]
ini[7] = [ZZ(0), ZZ(0), ZZ(16)/ZZ(9)*I]
ini[8] = [ZZ(0), ZZ(0), ZZ(16)/ZZ(9)*I, ZZ(16)/ZZ(9)*pi-ZZ(128)/ZZ(27)*I]
ini[10] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(32)/ZZ(45)*I]
ini[11] = [ZZ(0), ZZ(0), ZZ(16)/ZZ(15)*I]
ini[12] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(64)/ZZ(525)*I]
ini[13] = [ZZ(0), ZZ(0), ZZ(0), -ZZ(32)/ZZ(45)*I]
ini[14] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(32)/ZZ(75)*I]
ini[15] = [ZZ(0), ZZ(0), -ZZ(4)*I]
ini[16] = [ZZ(0), ZZ(0), ZZ(4)/ZZ(9)*I]
ini[17] = [ZZ(0), ZZ(0), ZZ(4)/ZZ(21)*I]
ini[18] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(16)/ZZ(735)*I]
ini[19] = [ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(6)]
ini[20] = [ZZ(0), ZZ(0), ZZ(1)/ZZ(9)]
ini[21] = [ZZ(0), ZZ(0), ZZ(1)/ZZ(3)]
ini[22] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(12)]
ini[23] = [ZZ(0), ZZ(0), -ZZ(4)/ZZ(3)]
ini[24] = [ZZ(0), ZZ(0), ZZ(1)/ZZ(27)*ZZ(9)**(ZZ(1)/ZZ(2))*ZZ(8)**(ZZ(1)/ZZ(2))]
ini[27] = [ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(3)]
ini[28] = [ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(15)]
ini[29] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(360)]
ini[30] = [ZZ(0), ZZ(0), ZZ(0), -ZZ(32)/ZZ(45)*I, -ZZ(32)/ZZ(45)*pi+ZZ(1472)/ZZ(675)*I]
ini[31] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(32)/ZZ(315)*I]
ini[32] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(32)/ZZ(525)*I]
ini[34] = [ZZ(0), ZZ(0), ZZ(1)/ZZ(28)]
ini[35] = [ZZ(0), ZZ(0), ZZ(1)/ZZ(5)]
ini[36] = [ZZ(0), ZZ(0), ZZ(1)/ZZ(588)*ZZ(7)**(ZZ(1)/ZZ(2))*ZZ(21)**(ZZ(1)/ZZ(2))]
ini[37] = [ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(84)]
ini[38] = [ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(84)]
ini[39] = [ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(5292)*ZZ(7)**(ZZ(1)/ZZ(2))*ZZ(21)**(ZZ(1)/ZZ(2))*ZZ(3)**(ZZ(1)/ZZ(2))]
ini[41] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(84)]
ini[42] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(60)]
ini[43] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(15)]
ini[44] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(120)]
ini[45] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(972)]
ini[46] = [ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(3), ZZ(1)/ZZ(3)*I*pi+ZZ(2)/ZZ(3)]
ini[47] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(3)]
ini[48] = [ZZ(0), ZZ(0), ZZ(4)/ZZ(3), -ZZ(4)/ZZ(3)*I*pi-ZZ(8)/ZZ(3), -ZZ(2)/ZZ(3)*pi**ZZ(2)+ZZ(8)/ZZ(3)*I*pi+ZZ(4)]
ini[49] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(27)]
ini[50] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(21168)*ZZ(7)**(ZZ(1)/ZZ(2))*ZZ(21)**(ZZ(1)/ZZ(2))]
ini[51] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(336)]
ini[52] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(1728)]
ini[53] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(317520)*ZZ(7)**(ZZ(1)/ZZ(2))*ZZ(21)**(ZZ(1)/ZZ(2))*ZZ(3)**(ZZ(1)/ZZ(2))]
ini[54] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(109350)*ZZ(9)**(ZZ(1)/ZZ(2))*ZZ(45)**(ZZ(1)/ZZ(2))*ZZ(5)**(ZZ(1)/ZZ(2))*ZZ(3)**(ZZ(1)/ZZ(2))]
ini[55] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(72900)*ZZ(9)**(ZZ(1)/ZZ(2))*ZZ(45)**(ZZ(1)/ZZ(2))*ZZ(5)**(ZZ(1)/ZZ(2))]
ini[56] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(5292)*ZZ(7)**(ZZ(1)/ZZ(2))*ZZ(21)**(ZZ(1)/ZZ(2))*ZZ(3)**(ZZ(1)/ZZ(2))]
ini[57] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(336)]
ini[58] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(300)]
ini[59] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(60)]
ini[60] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(600)]
ini[61] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(5184)]
ini[62] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(194400)*ZZ(9)**(ZZ(1)/ZZ(2))*ZZ(45)**(ZZ(1)/ZZ(2))*ZZ(5)**(ZZ(1)/ZZ(2))]
ini[63] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(388800)*ZZ(9)**(ZZ(1)/ZZ(2))*ZZ(45)**(ZZ(1)/ZZ(2))*ZZ(5)**(ZZ(1)/ZZ(2))]
ini[64] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(583200)*ZZ(9)**(ZZ(1)/ZZ(2))*ZZ(45)**(ZZ(1)/ZZ(2))*ZZ(5)**(ZZ(1)/ZZ(2))]
ini[65] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(105840)*ZZ(7)**(ZZ(1)/ZZ(2))*ZZ(21)**(ZZ(1)/ZZ(2))*ZZ(3)**(ZZ(1)/ZZ(2))]
ini[66] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(1)/ZZ(21168)*ZZ(7)**(ZZ(1)/ZZ(2))*ZZ(21)**(ZZ(1)/ZZ(2))*ZZ(3)**(ZZ(1)/ZZ(2))]
ini[67] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(2)/ZZ(27), -ZZ(2)/ZZ(27)*I*pi-ZZ(13)/ZZ(81)]
ini[68] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(2)/ZZ(27)]
ini[69] = [ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(3), ZZ(1)/ZZ(3)*I*pi+ZZ(2)/ZZ(3), ZZ(1)/ZZ(6)*pi**ZZ(2)-ZZ(2)/ZZ(3)*I*pi-ZZ(11)/ZZ(12)]
ini[70] = [ZZ(0), ZZ(0), ZZ(0), ZZ(0), ZZ(0), -ZZ(1)/ZZ(144)]
# Computed separately (get_ini_Hstar fails because of what looks like a Maple
# bug)
ini[9] = [0, 0, ZZ(16)/9*I, ZZ(16)/9*pi-ZZ(128)/27*I,
          ZZ(-128)/27*pi-ZZ(8)/9*I*pi**2 + ZZ(832)/81*I]

