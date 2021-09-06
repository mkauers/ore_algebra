# -*- coding: utf-8 - vim: tw=80
"""
Some examples arosing when I implemented the factorization of linear
differential operators.
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



# from http://koutschan.de/data/fcc1/. (fcc4, fcc5, fcc6 are available in
# ore_algebra's examples).
fcc3 = 2*(-1+z)*z**2*(3+z)**2*Dz**3+3*z*(3+z)*(-6+5*z+5*z**2)*Dz**2+6*(-3+3*z+12*z**2+4*z**3)*Dz+6*z*(2+z)

# Fuchsian operators with linear factors but without rational solution:
# - the first one is the annihilator of sqrt(1+z) and sqrt(1+2z).
# - thanks to Emre Sertoz for the second one (arosing from actual computations).
# - the differential Galois group (= monodromy group) of the third one is composed by homotheties.
sqrt_ex = (4*z**2 + 6*z + 2)*Dz**2 + (4*z + 3)*Dz - 1
sertoz_ex = (-128*z**2 + 8)*(z*Dz)**3 + (-256*z**2-24)*(z*Dz)**2 + (32*z**2 + 10)*(z*Dz)+ 64*z**2
exact_guessing_ex = (2*z*Dz - 1).lclm(2*z*Dz - 3)

# DEtools[DFactor] (of Maple, diffop package) fails with the following operator.
# Thanks to Bruno Salvy for reporting it. We suspect that the large exponent
# (=-972) at point 3 is the cause. !Not Fuchsian! (Update: 2020, Dec)
salvy_ex = (z**2*Dz + 3)*((z - 3)*Dz + 4*z**5)

# The only right factor of the following operator has a degree k (a parameter)
# while the degree of the full operator is 2. For more details, see the article
# "Explicit degree bounds for right factors of linear differential operators" by
# Alin Bostan, Tanguy Rivoal and Bruno Salvy (2020). !Not Fuchsian!
bostan_ex = lambda k: z*Dz**2 + (2-z)*Dz + k

# This example is from van Hoeij's phd thesis (section 3.1). It seems that its
# only right factor has a degree n^2. !Not Fuchsian!
vanhoeij_ex = lambda n: Dz**2 - (1/n)*Dz + n/z

# This reducible operator does not admits factorization with coefficients in QQ (to be confirmed).
# Factorization in QQbar: (z^2*Dz + (1 + i)*z)*(Dz - i/z).
QQbar_ex = z**2*Dz**2 + z*Dz + 1

# Annihilator of the hypergeometric function 2F1(a,b;c;z).
hypergeo = lambda a,b,c: z*(1 - z)*Dz**2 + (c - (a + b + 1)*z)*Dz - a*b

# This reducible operator admits no factorization in the first Weyl algebra (QQ[z]<Dz>).
irr_weyl = Dz**2 + (-z**2 + 2*z)*Dz + z - 2

# Test cases (see commit e75d04a4 of ore_algebra for more details).
# The second one is not Fuchsian.
test_ex1 = ((z**5 - z**4 + z**3)*Dz**3 + (QQ(27/8)*z**4 - QQ(25/9)*z**3 + 8*z**2)*Dz**2 + (QQ(37/24)*z**3 - QQ(25/9)*z**2 + 14*z)*Dz - 2*z**2 - QQ(3/4)*z + 4)*((z**5 - QQ(9/4)*z**4 + z**3)*Dz**3 + (QQ(11/6)*z**4 - QQ(31/4)*z**3 + 7*z**2)*Dz**2 + (QQ(7/30)*z**3 - QQ(101/20)*z**2 + 10*z)*Dz + QQ(4/5)*z**2 + QQ(5/6)*z + 2)
test_ex2 = ((QQ(-1/4)*z**10 + 8*z**9 + 2*z**8 - 2*z**7 - QQ(5/2)*z**6 - QQ(1/8)*z**5 - QQ(45/2)*z**3 - QQ(1/2)*z**2 - QQ(1/2)*z)*Dz**4 + (6*z**9 - QQ(3/17)*z**5 - QQ(3/2)*z**4 - 2*z**3 + QQ(21/2)*z**2 - z - 10)*Dz**3 + (QQ(5/2)*z**8 + z**6 + 10*z**5 - 3*z**4 + QQ(1/2)*z**3 + z**2 - QQ(1/4)*z)*Dz**2 + (23*z**7 - 3*z**6 + QQ(1/13)*z**5 + z**2 - z + 1)*Dz + QQ(2/13)*z**6 - QQ(1/2)*z**5 - QQ(1/3)*z**4 + 2*z**2 - z - 9)*((2*z**5 + 7*z**2)*Dz**3 + (QQ(1/5)*z**4 + 2*z**3 - 2*z + 1)*Dz**2 + (-QQ(3/2)*z**3 - QQ(1/2)*z**2 + 2*z)*Dz - QQ(1/521)*z**2 - 8*z + QQ(1/2))

# This operator is given as example to illustrate the newton polygon's
# definition in [Formal Solutions ..., van Hoeij, 1997] (Example 3.1).
Tz = z*Dz
newton_ex = z**6*Tz**9 + 2*z**5*Tz**8 + 3*z**4*Tz**7 + 2*z**3*Tz**6 + (z**2 + 2*z**4)*Tz**5 + (5*z**2 - 3*z)*Tz**3 + 3*z*Tz**2 + (2 + 2*z)*Tz + 7*z
