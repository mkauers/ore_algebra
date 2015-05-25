
"""
ideal
=====

"""

#############################################################################
#  Copyright (C) 2015                                                       #
#                Manuel Kauers (mkauers@gmail.com),                         #
#                                                                           #
#  Distributed under the terms of the GNU General Public License (GPL)      #
#  either version 2, or (at your option) any later version                  #
#                                                                           #
#  http://www.gnu.org/licenses/                                             #
#############################################################################

from ore_algebra import *
from ore_operator import *
from ore_operator_1_1 import *
from ore_operator_mult import *
import nullspace 

from sage.rings.noncommutative_ideals import *

class OreLeftIdeal(Ideal_nc):

    def __init__(self, ring, gens, coerce=True):
        Ideal_nc.__init__(self, ring, gens, coerce, "left")

