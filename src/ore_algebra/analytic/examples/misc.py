r"""
Miscellaneous examples

An example kindly provided by Christoph Koutschan::

    sage: from ore_algebra.analytic.examples.misc import koutschan1
    sage: koutschan1.dop.numerical_solution(koutschan1.ini, [0, 84])
    [0.011501537469552017...]
"""
import collections

from sage.rings.rational_field import QQ
from ore_algebra import DifferentialOperators

IVP = collections.namedtuple("IVP", ["dop", "ini"])

DiffOps, a, Da = DifferentialOperators(QQ, 'a')

koutschan1 = IVP(
    dop = (1315013644371957611900*a**2+263002728874391522380*a+13150136443719576119)*Da**3
        + (2630027288743915223800*a**2+16306169190212274387560*a+1604316646133788286518)*Da**2
        + (1315013644371957611900*a**2-39881765316802329075320*a+35449082663034775873349)*Da
        + (-278967152068515080896550+6575068221859788059500*a),
    ini = [ QQ(5494216492395559)/3051757812500000000000000000000,
            QQ(6932746783438351)/610351562500000000000000000000,
            1/QQ(2) * QQ(1142339612827789)/19073486328125000000000000000 ]
)
