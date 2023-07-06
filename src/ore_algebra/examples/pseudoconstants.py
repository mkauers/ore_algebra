"""r

This module contains examples for computation of pseudoconstants and global
integral bases.

EXAMPLES::

    sage: from ore_algebra.examples.pseudoconstants import L9
    sage: L9.order(), L9.degree()
    (6,76)
    sage: L9.singularities()
    {0, 1, 2, 3, 4, 5, 6}

We also provide the pseudoconstant, and we show how to verify that it is correct

    sage: from ore_algebra.examples.pseudoconstants import L9_pc
    sage: prec=50
    sage: for (z,place) in [(s,x+s) for s in L9.singularities()] + [(Infinity,1/x)]:
    sage:     L9_shift, phi = L9.annihilator_of_composition(place, with_transform=True)
    sage:     sols = L9_shift.generalized_series_solutions(prec)
    sage:     L9_pc_shift = phi(L9_pc)
    sage:     print(f"Exponents at place {z}: {[L9_pc_shift(s).initial_exponent() for s in sols]}")
    Exponents at place 0: [25/7, 20/7, 15/7, 10/7, 0, 5/7]
    Exponents at place 1: [25/7, 20/7, 15/7, 10/7, 0, 5/7]
    Exponents at place 2: [25/7, 20/7, 15/7, 10/7, 0, 5/7]
    Exponents at place 3: [25/7, 20/7, 15/7, 10/7, 0, 5/7]
    Exponents at place 4: [25/7, 20/7, 15/7, 10/7, 0, 5/7]
    Exponents at place 5: [25/7, 20/7, 15/7, 10/7, 0, 5/7]
    Exponents at place 6: [25/7, 20/7, 15/7, 10/7, 0, 5/7]
    Exponents at place +Infinity: [5, 4, 3, 2, 1, 0]

"""

from sage.repl.preparse import preparse
from sage.rings.all import Frac, Integer, PolynomialRing, ZZ, QQ
from ore_algebra import OreAlgebra
import os

path = os.path.dirname(__file__)

Pol = PolynomialRing(QQ, "x")
x = Pol.gen()
Dif = OreAlgebra(Pol.fraction_field(), "Dx")
Dx = Dif.gen()

with open(os.path.join(path,"pseudoconstant_L9.sage"),"r") as f:
    lines = f.readlines()
    
L9 = Dif(eval("".join(preparse(l[:-1]) for l in lines)))

with open(os.path.join(path,"pseudoconstant_L9_pc.sage"),"r") as f:
    lines = f.readlines()

L9_pc = Dif(eval("".join(preparse(l[:-1]) for l in lines)))

