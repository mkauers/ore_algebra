import logging; logging.basicConfig(level=logging.DEBUG)

# from line_profiler import LineProfiler

from ore_algebra import *
import ore_algebra.analytic.analytic_continuation as ancont
import ore_algebra.analytic.path as path
import ore_algebra.analytic.naive_sum as naive_sum

import ore_algebra.analytic.binary_splitting

from ore_algebra.analytic.ui import *

from ore_algebra.analytic.path import Point
QQi.<i> = QuadraticField(-1)
Pol.<x> = QQ[]; Dop.<Dx> = OreAlgebra(Pol)
dop = (x^2 + 1)*Dx^2 + 2*x*Dx
#p = Point(1, dop)
#
#Point(1, dop).dist_to_sing()
#Point(i, dop).dist_to_sing()
#Point(1+i, dop).dist_to_sing()
#
#type(Point(1, dop).classify())
#type(Point(i, dop).classify())

from ore_algebra.analytic.path import Path

Path([1,2,3,i], dop)
path = Path([1,2,3], dop)
# path.check_singularity()
# path.check_convergence()
# Path([0,1], dop).check_convergence()
# Path([1, i, 2], dop).check_singularity()
# Path([0, 2*i], dop).check_singularity()
path = Path([0, 1+i, 2*i], dop)
# path.plot(disks=False))
# path.plot(disks=False)

path.subdivide()

dop = (x+1)*(x^2+1)*Dx^3-(x-1)*(x^2-3)*Dx^2-2*(x^2+2*x-1)*Dx # exp + arctan
print transition_matrix(dop, [0, 1+i], 1e-20) # XXX: cancellation ?
print eval_diffeq(dop, [1, 2, 1], [0, 1+i], 1e-20)
