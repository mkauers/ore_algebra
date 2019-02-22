
r"""
Examples from the paper "Hypergeometric expressions for generating functions
of walks with small steps in the quarter plane" by Bostan, Chyzak, van Hoeij,
Kauers, and Pech (2017).

For the rational functions rat[1], ..., rat[19], the task consists in computing
a telescoper wrt Du and Dv. The variables x and y may be set to 1. 

Example: 

   sage: from ore_algebra.examples.smallsteps import rat
   sage: from ore_algebra import OreAlgebra
   sage: A.<Du,Dv,Dt> = OreAlgebra(rat[1](x=1,y=1).parent(), 'Du', 'Dv', 'Dt')
   
   sage: q = rat[1](x=1,y=1)
   sage: ct1 = A.ideal([q*D - D(q) for D in Du,Dv,Dt]).ct(Dv)[0]
   sage: ct2 = ct1[0].parent().ideal(ct1).ct(Du)[0]


"""

from sage.rings.all import PolynomialRing, ZZ

Pols, (u, v, x, y, t) = PolynomialRing(ZZ, ('u', 'v', 'x', 'y', 't')).objgens()

rat = {}

# indexed from 1 on in order to be consistent with the paper
rat[1] = (-1+u**2-u**2*v**2+v**2)/(-u*v+u*v**2*y+u**2*v*x-u**2*v**2*x*y+t*v-t*v**2*y-t*v*x*u+t*v**2*x*u*y+t*u-t*u*y*v-t*u**2*x+t*u**2*x*y*v+t*u**2*v-t*u**2*v**2*y-t*u**3*v*x+t*u**3*v**2*x*y+t*u*v**2-t*u*v**3*y-t*u**2*v**2*x+t*u**2*v**3*x*y)

rat[2] = (-1+u**2-u**2*v**2+v**2)/(-u*v+u*v**2*y+u**2*v*x-u**2*v**2*x*y+t-t*y*v-t*x*u+t*x*u*y*v+t*u**2-t*u**2*y*v-t*u**3*x+t*u**3*x*y*v+t*u**2*v**2-t*u**2*v**3*y-t*u**3*v**2*x+t*u**3*v**3*x*y+t*v**2-t*v**3*y-t*v**2*x*u+t*v**3*x*u*y)

rat[3] = (-1+u**2-u**2*v**2+v**2)/(t+t*u*v**2-t*u**2*v**2*x-u**2*v**2*x*y-t*u*y*v-t*u*v**3*y+t*u**2*v**2-u*v+t*u+u**2*v*x-t*u**2*x+u*v**2*y+t*u**2*x*y*v+t*u**2*v**3*x*y+t*u**2+t*v**2-t*u**2*y*v-t*u**2*v**3*y-t*v**2*x*u-t*u**3*v**2*x-t*y*v-t*v**3*y-t*x*u-t*u**3*x+t*x*u*y*v+t*u**3*x*y*v+t*u**3*v**3*x*y+t*v**3*x*u*y)

rat[4] = (-1+u**2-u**2*v**2+v**2)/(t+t*u**2*v+t*u*v**2-t*v*x*u-t*u**3*v*x-t*u**2*v**2*x-u**2*v**2*x*y-t*u*y*v-t*u**2*v**2*y-t*u*v**3*y+t*u**2*v**2-u*v+t*v+t*u+u**2*v*x-t*u**2*x+u*v**2*y-t*v**2*y+t*v**2*x*u*y+t*u**2*x*y*v+t*u**3*v**2*x*y+t*u**2*v**3*x*y+t*u**2+t*v**2-t*u**2*y*v-t*u**2*v**3*y-t*v**2*x*u-t*u**3*v**2*x-t*y*v-t*v**3*y-t*x*u-t*u**3*x+t*x*u*y*v+t*u**3*x*y*v+t*u**3*v**3*x*y+t*v**3*x*u*y)

rat[5] = (-1+u**4-u**3*v**2+u*v**2)/(t+t*u*v**2-t*u**2*v**2*x-u**2*v**2*x*y-t*u*v**3*y-u*v+u**2*v*x+u*v**2*y+t*u**2*v**3*x*y+2*t*u**2-2*t*u**2*y*v-t*y*v-u**5*t*x+u**3*v**2*y+t*u**3*v**2-t*x*u-2*t*u**3*x+t*x*u*y*v+2*t*u**3*x*y*v-t*u**3*v**3*y+u**4*v*x-t*u**4*v**2*x-u**4*v**2*x*y-u**4*t*y*v-u**3*v+u**4*t+u**5*t*x*y*v+t*u**4*v**3*x*y)

rat[6] = (-1+u**4-u**3*v**2+u*v**2)/(t+2*t*u**2*v+t*u*v**2-t*v*x*u-2*t*u**3*v*x-t*u**2*v**2*x-u**2*v**2*x*y-2*t*u**2*v**2*y-t*u*v**3*y-u*v+t*v+u**2*v*x+u*v**2*y-t*v**2*y+t*v**2*x*u*y+2*t*u**3*v**2*x*y+t*u**2*v**3*x*y+2*t*u**2-2*t*u**2*y*v-t*y*v-u**5*t*x+u**3*v**2*y+t*u**3*v**2-t*x*u-2*t*u**3*x+t*x*u*y*v+2*t*u**3*x*y*v-t*u**3*v**3*y+u**4*v*x-t*u**4*v**2*x-u**4*v**2*x*y-u**4*t*y*v+u**4*t*v-u**3*v+u**4*t+u**5*t*x*y*v+t*u**4*v**3*x*y-u**5*t*x*v-u**4*t*v**2*y+u**5*t*x*v**2*y)

rat[7] = (-u-1+u**4+u**3-u**3*v**2+u*v**2)/(t+t*u*v**2-t*u**2*v**2*x-u**2*v**2*x*y-2*t*u*y*v-t*u*v**3*y+u**2*v**2*y+u**3*v*x+t*u**2*v**2-u*v-u**2*v+2*t*u+u**2*v*x-2*t*u**2*x+u*v**2*y+2*t*u**2*x*y*v+t*u**2*v**3*x*y-u**3*v**2*x*y+3*t*u**2-3*t*u**2*y*v-t*u**2*v**3*y-t*u**3*v**2*x-t*y*v-2*t*u**4*x-u**5*t*x+u**3*v**2*y+t*u**3*v**2-t*x*u-3*t*u**3*x+t*x*u*y*v+3*t*u**3*x*y*v+t*u**3*v**3*x*y+2*t*u**3-2*t*u**3*y*v-t*u**3*v**3*y+u**4*v*x-t*u**4*v**2*x-u**4*v**2*x*y-u**4*t*y*v-u**3*v+u**4*t+u**5*t*x*y*v+t*u**4*v**3*x*y+2*t*u**4*x*y*v)

rat[8] = (-u-1+u**4+u**3-u**3*v**2+u*v**2)/(t+2*t*u**2*v+t*u*v**2-t*v*x*u-2*t*u**3*v*x-t*u**2*v**2*x-u**2*v**2*x*y-2*t*u*y*v-2*t*u**2*v**2*y-t*u*v**3*y+u**2*v**2*y+u**3*v*x+t*v*u+t*u**3*v+t*u**2*v**2-u*v-u**2*v+t*v+2*t*u+u**2*v*x-2*t*u**2*x+u*v**2*y-t*v**2*y+t*v**2*x*u*y+2*t*u**2*x*y*v+2*t*u**3*v**2*x*y+t*u**2*v**3*x*y-u**3*v**2*x*y+3*t*u**2-t*v**2*u*y-3*t*u**2*y*v-t*u**3*v**2*y-t*u**2*v**3*y-t*u**2*x*v-t*u**3*v**2*x-t*y*v-2*t*u**4*x-u**5*t*x+u**3*v**2*y+t*u**3*v**2-t*x*u-3*t*u**3*x+t*x*u*y*v+3*t*u**3*x*y*v+t*u**3*v**3*x*y+2*t*u**3-2*t*u**3*y*v-t*u**3*v**3*y+u**4*v*x-t*u**4*v**2*x-u**4*v**2*x*y-u**4*t*y*v+u**4*t*v-u**3*v+u**4*t+u**5*t*x*y*v+t*u**4*v**3*x*y-u**5*t*x*v-u**4*t*v**2*y+u**5*t*x*v**2*y+2*t*u**4*x*y*v-t*u**4*x*v+t*u**2*x*v**2*y+t*u**4*x*v**2*y)

rat[9] = (-u-1+u**4+u**3-u**4*v**2+v**2)/(t+t*u*v**2-t*u**2*v**2*x-u**2*v**2*x*y-2*t*u*y*v-t*u*v**3*y+u**2*v**2*y+u**3*v*x+2*t*u**2*v**2-u*v-u**2*v+2*t*u+u**2*v*x-2*t*u**2*x+u*v**2*y+2*t*u**2*x*y*v+t*u**2*v**3*x*y-u**3*v**2*x*y+3*t*u**2+t*v**2-3*t*u**2*y*v-2*t*u**2*v**3*y-t*v**2*x*u-2*t*u**3*v**2*x-t*y*v-t*v**3*y-2*t*u**4*x-u**5*t*x+u**3*v**2*y+t*u**3*v**2-t*x*u-3*t*u**3*x+t*x*u*y*v+3*t*u**3*x*y*v+2*t*u**3*v**3*x*y+t*v**3*x*u*y+2*t*u**3-2*t*u**3*y*v-t*u**3*v**3*y+u**4*v*x-t*u**4*v**2*x-u**4*v**2*x*y-u**4*t*y*v+t*u**4*v**2-u**3*v+u**4*t+u**5*t*x*y*v+t*u**4*v**3*x*y-t*u**4*v**3*y-u**5*t*x*v**2+2*t*u**4*x*y*v+u**5*t*x*v**3*y)

rat[10] = (-u-1+u**4+u**3-u**4*v**2+v**2)/(t+2*t*u**2*v+t*u*v**2-t*v*x*u-2*t*u**3*v*x-t*u**2*v**2*x-u**2*v**2*x*y-2*t*u*y*v-2*t*u**2*v**2*y-t*u*v**3*y+u**2*v**2*y+u**3*v*x+t*v*u+t*u**3*v+2*t*u**2*v**2-u*v-u**2*v+t*v+2*t*u+u**2*v*x-2*t*u**2*x+u*v**2*y-t*v**2*y+t*v**2*x*u*y+2*t*u**2*x*y*v+2*t*u**3*v**2*x*y+t*u**2*v**3*x*y-u**3*v**2*x*y+3*t*u**2+t*v**2-t*v**2*u*y-3*t*u**2*y*v-t*u**3*v**2*y-2*t*u**2*v**3*y-t*v**2*x*u-t*u**2*x*v-2*t*u**3*v**2*x-t*y*v-t*v**3*y-2*t*u**4*x-u**5*t*x+u**3*v**2*y+t*u**3*v**2-t*x*u-3*t*u**3*x+t*x*u*y*v+3*t*u**3*x*y*v+2*t*u**3*v**3*x*y+t*v**3*x*u*y+2*t*u**3-2*t*u**3*y*v-t*u**3*v**3*y+u**4*v*x-t*u**4*v**2*x-u**4*v**2*x*y-u**4*t*y*v+t*u**4*v**2+u**4*t*v-u**3*v+u**4*t+u**5*t*x*y*v+t*u**4*v**3*x*y-t*u**4*v**3*y-u**5*t*x*v-u**4*t*v**2*y+u**5*t*x*v**2*y-u**5*t*x*v**2+2*t*u**4*x*y*v-t*u**4*x*v+t*u**2*x*v**2*y+t*u**4*x*v**2*y+u**5*t*x*v**3*y)

rat[11] = (-u+u**3-u**4*v**2-u**3*v**2+u*v**2+v**2)/(-u**2*v+u**2*v**2*y+u**3*v*x-u**3*v**2*x*y+t*u**2-t*u**2*y*v-t*u**3*x+t*u**3*x*y*v+t*u**3*v**2-t*u**3*v**3*y-t*u**4*v**2*x+t*u**4*v**3*x*y+t*u**2*v**2-t*u**2*v**3*y-t*u**3*v**2*x+t*u**3*v**3*x*y+t*u*v**2-t*u*v**3*y-t*u**2*v**2*x+t*u**2*v**3*x*y)

rat[12] = (-u+u**3-u**4*v**2-u**3*v**2+u*v**2+v**2)/(t*u*v**2-t*u**2*v**2*x-t*u*v**3*y+u**2*v**2*y+u**3*v*x+t*v*u+t*u**3*v+t*u**2*v**2-u**2*v+t*u**2*v**3*x*y-u**3*v**2*x*y+t*u**2-t*v**2*u*y-t*u**2*y*v-t*u**3*v**2*y-t*u**2*v**3*y-t*u**2*x*v-t*u**3*v**2*x+t*u**3*v**2-t*u**3*x+t*u**3*x*y*v+t*u**3*v**3*x*y-t*u**3*v**3*y-t*u**4*v**2*x+t*u**4*v**3*x*y-t*u**4*x*v+t*u**2*x*v**2*y+t*u**4*x*v**2*y)

rat[13] = (-1+u**4-u**4*v**2-u**3*v**2+u*v**2+v**2)/(t+t*u*v**2-t*u**2*v**2*x-u**2*v**2*x*y-t*u*v**3*y+2*t*u**2*v**2-u*v+u**2*v*x+u*v**2*y+t*u**2*v**3*x*y+2*t*u**2+t*v**2-2*t*u**2*y*v-2*t*u**2*v**3*y-t*v**2*x*u-2*t*u**3*v**2*x-t*y*v-t*v**3*y-u**5*t*x+u**3*v**2*y+t*u**3*v**2-t*x*u-2*t*u**3*x+t*x*u*y*v+2*t*u**3*x*y*v+2*t*u**3*v**3*x*y+t*v**3*x*u*y-t*u**3*v**3*y+u**4*v*x-t*u**4*v**2*x-u**4*v**2*x*y-u**4*t*y*v+t*u**4*v**2-u**3*v+u**4*t+u**5*t*x*y*v+t*u**4*v**3*x*y-t*u**4*v**3*y-u**5*t*x*v**2+u**5*t*x*v**3*y)

rat[14] = (-1+u**4-u**4*v**2-u**3*v**2+u*v**2+v**2)/(t+2*t*u**2*v+t*u*v**2-t*v*x*u-2*t*u**3*v*x-t*u**2*v**2*x-u**2*v**2*x*y-2*t*u**2*v**2*y-t*u*v**3*y+2*t*u**2*v**2-u*v+t*v+u**2*v*x+u*v**2*y-t*v**2*y+t*v**2*x*u*y+2*t*u**3*v**2*x*y+t*u**2*v**3*x*y+2*t*u**2+t*v**2-2*t*u**2*y*v-2*t*u**2*v**3*y-t*v**2*x*u-2*t*u**3*v**2*x-t*y*v-t*v**3*y-u**5*t*x+u**3*v**2*y+t*u**3*v**2-t*x*u-2*t*u**3*x+t*x*u*y*v+2*t*u**3*x*y*v+2*t*u**3*v**3*x*y+t*v**3*x*u*y-t*u**3*v**3*y+u**4*v*x-t*u**4*v**2*x-u**4*v**2*x*y-u**4*t*y*v+t*u**4*v**2+u**4*t*v-u**3*v+u**4*t+u**5*t*x*y*v+t*u**4*v**3*x*y-t*u**4*v**3*y-u**5*t*x*v-u**4*t*v**2*y+u**5*t*x*v**2*y-u**5*t*x*v**2+u**5*t*x*v**3*y)

rat[15] = (-u+u**3-u**4*v**2+v**2)/(-u**2*v+u**2*v**2*y+u**3*v*x-u**3*v**2*x*y+t*u**2-t*u**2*y*v-t*u**3*x+t*u**3*x*y*v+t*u**3*v**2-t*u**3*v**3*y-t*u**4*v**2*x+t*u**4*v**3*x*y+t*u*v**2-t*u*v**3*y-t*u**2*v**2*x+t*u**2*v**3*x*y)

rat[16] = (-u+u**3-u**4*v**2+v**2)/(-u**2*v+u**2*v**2*y+u**3*v*x-u**3*v**2*x*y+t*v*u-t*v**2*u*y-t*u**2*x*v+t*u**2*x*v**2*y+t*u**2-t*u**2*y*v-t*u**3*x+t*u**3*x*y*v+t*u**3*v-t*u**3*v**2*y-t*u**4*x*v+t*u**4*x*v**2*y+t*u**3*v**2-t*u**3*v**3*y-t*u**4*v**2*x+t*u**4*v**3*x*y+t*u*v**2-t*u*v**3*y-t*u**2*v**2*x+t*u**2*v**3*x*y)

rat[17] = (-u*v+u**3-u**4*v+u**3*v**3-v**4*u+v**3)/(-u**2*v**2+u**2*v**3*y+u**3*v**2*x-u**3*v**3*x*y+t*u**2*v-t*u**2*v**2*y-t*u**3*v*x+t*u**3*v**2*x*y+t*u**3*v**2-t*u**3*v**3*y-t*u**4*v**2*x+t*u**4*v**3*x*y+t*u*v**3-u*v**4*t*y-t*u**2*v**3*x+u**2*v**4*t*x*y)

rat[18] = (-u*v+u**3-u**4*v+u**3*v**3-v**4*u+v**3)/(t*u**2*v+t*u*v**2-t*u**3*v*x-t*u**2*v**2*x-t*u**2*v**2*y-t*u*v**3*y+t*u**3*v-u**2*v**2+u**2*v**4*t*x*y+t*u**3*v**2*x*y+t*u**2*v**3*x*y-t*u**3*v**2*y-t*u**2*v**3*x+t*u**2*v**3+t*u*v**3-t*u**3*v**3*x+u**2*v**3*y+u**3*v**2*x+t*u**3*v**2-u**3*v**3*x*y-t*u**3*v**3*y-t*u**4*v**2*x+t*u**4*v**3*x*y+u**3*v**4*t*x*y-t*u**4*x*v+t*u**4*x*v**2*y-u**2*v**4*t*y-u*v**4*t*y)

rat[19] = (-u**2*v+u**4-u**6+u**6*v-u**4*v**3+v**4*u**2-v**4+v**3)/(-u**3*v**2+u**3*v**3*y+u**4*v**2*x-u**4*v**3*x*y+t*u**2*v**2-t*u**2*v**3*y-t*u**3*v**2*x+t*u**3*v**3*x*y+u**4*t*v-u**4*t*v**2*y-u**5*t*x*v+u**5*t*x*v**2*y+t*u**4*v**2-t*u**4*v**3*y-u**5*t*x*v**2+u**5*t*x*v**3*y+t*u**2*v**3-u**2*v**4*t*y-t*u**3*v**3*x+u**3*v**4*t*x*y)
