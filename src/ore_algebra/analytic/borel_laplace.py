# -*- coding: utf-8 - vim: tw=80
r"""
Borel-Laplace summation

EXAMPLES::

    sage: from ore_algebra import OreAlgebra
    sage: from ore_algebra.analytic.borel_laplace import *

    sage: Pol.<x> = QQ[]
    sage: Dop.<Dx> = OreAlgebra(Pol)

We start with several examples involving the (homogeneized) Euler equation.
(Compare, e.g., Thomann 1990, §5, §7.7, Thomann 1995, §3, §5.) We consider its
power series solution `E` with `E'(0) = 1`. The first terms are::

    sage: (-x^3*Dx^2+(-x^2-x)*Dx+1).local_basis_expansions(0)
    [x - x^2 + 2*x^3 - 6*x^4 + 24*x^5]

On a suitable domain (see Loday-Richaud 2016, Example 1.1.4 for details) is
equal to ``Ei(1,1/x)*exp(1/x)``. ::

    sage: def ref(x):
    ....:     x = ComplexBallField(100)(x)
    ....:     return (1/x).exp_integral_e(1)*(1/x).exp()

Thus::

    sage: ref(1/2)
    [0.3613286168882225846971616...]
    sage: borel_laplace(-x^3*Dx^2+(-x^2-x)*Dx+1, 1/2, 0, RBF(1e-20))
    [[0.361328616888222584...] + [+/- ...]*I]

    sage: ref(1/10)
    [0.0915633339397880818...]
    sage: borel_laplace(-x^3*Dx^2+(-x^2-x)*Dx+1, 1/10, 0, RBF(1e-50))
    [[0.091563333939788081876069815766438449226677369109...] + [+/- ...]*I]

We can change the direction of summation::

    sage: borel_laplace(-x^3*Dx^2+(-x^2-x)*Dx+1, 1/2, pi/4, RBF(1e-20))
    [[0.361328616888222584...] + [+/- ...]*I]

Standard direction, but close to the border of the associated sector... ::

    sage: borel_laplace(-x^3*Dx^2+(-x^2-x)*Dx+1, 1/100+1/2*i, RBF(0), RBF(1e-10))
    [[0.150323330...] + [0.395016510...]*I]

Attempting to evaluate on the border of the sector results in an error::

    sage: borel_laplace(-x^3*Dx^2+(-x^2-x)*Dx+1, 1/2*i, 0, RBF(1e-10))
    Traceback (most recent call last):
    ...
    ValueError: evaluation point not in the half-plane bisected by the direction
    (or too close to the border)

However, we can compute the analytic continuation of the sum by choosing a
more suitable direction::

    sage: borel_laplace(-x^3*Dx^2+(-x^2-x)*Dx+1, 1/2*i, pi/4, RBF(1e-20))
    [[0.144545303037332420...] + [0.399020988594183846...]*I]

    sage: borel_laplace(-x^3*Dx^2+(-x^2-x)*Dx+1, 1/2*i, pi/2, RBF(1e-20))
    [[0.144545303037332420...] + [0.399020988594183846...]*I]

    sage: ref(1/2*i)
    [0.14454530303733242045870285...] + [0.39902098859418384689266651...]*I

The negative real axis is a singular direction::

    sage: borel_laplace(-x^3*Dx^2+(-x^2-x)*Dx+1, -1/2, pi, RBF(1e-20))
    Traceback (most recent call last):
    ...
    ValueError: singular direction (or close)

Stokes phenomenon (see again Loday-Richaud 2016, Example 1.1.4 for a detailed
description of the situation)::

    sage: ref(-1/2)
    [-0.670482709790073281043223808...] + [-0.425168331587636328439122361...]*I
    sage: val_above = borel_laplace(-x^3*Dx^2+(-x^2-x)*Dx+1, -1/2, 3*pi/4, RBF(1e-20))
    sage: val_above
    [[-0.67048270979007328...] + [0.42516833158763632...]*I]
    sage: val_below = borel_laplace(-x^3*Dx^2+(-x^2-x)*Dx+1, -1/2, -3*pi/4, RBF(1e-20))
    sage: val_below
    [[-0.67048270979007328...] + [-0.42516833158763632...]*I]
    sage: val_above - val_below
    [[+/- ...] + [0.85033666317527265...]*I]
    sage: CBF(2*pi*i*exp(-2))
    [0.8503366631752...]*I

TESTS:

Monomials::

    sage: borel_laplace(x*Dx, 1/2, RBF(0), RBF(1e-20))
    [[1.00000000000000000...] + [+/- ...]*I]

    sage: borel_laplace(x*Dx*(x*Dx+1), 1/2, RBF(0), RBF(1e-20))
    [[2.000000000000000000...] + [+/- ...]*I  [1.000000000000000000...] + [+/- ...]*I]

    sage: borel_laplace(x*Dx-1, 1/2, RBF(0), RBF(1e-20))
    [[0.50000000000000000...] + [+/- ...]*I]

    sage: borel_laplace(x*Dx-3, 1/8, RBF(0), RBF(1e-20))
    [[0.00195312500000000...] + [+/- ...]*I]

    sage: borel_laplace(x*Dx-1/2, 1/2, RBF(0), RBF(1e-20))
    [[0.70710678118654752...] + [+/- ...]*I]

    sage: borel_laplace((x*Dx-1)^2, 1/100, RBF(0), RBF(1e-20))
    [[-0.0460517018598809...] + [+/- ...]*I  [0.01000000...] + [+/- ...]*I]

    sage: borel_laplace(x*Dx+1/2, 1/10, RBF(0), RBF(1e-20))
    [[3.16227766016837933...] + [+/- ...]*I]

    sage: borel_laplace(x*Dx+1/3, 1/10, RBF(0), RBF(1e-20))
    [[2.15443469003188372...] + [+/- ...]*I]

    sage: borel_laplace((x*Dx+1)^2, 1/10, RBF(0), RBF(1e-20))
    [[-23.025850929940456...] + [+/- ...]*I  [10.000000...] + [+/- ...]*I]

    sage: borel_laplace((x*Dx-1)*(x*Dx-2), 1/2, RBF(0), RBF(1e-10))
    [ [0.500000000...] + [+/- ...]*I [0.2500000000...] + [+/- ...]*I]

    sage: borel_laplace((x*Dx+1)^3, 1/2, RBF(0), RBF(1e-20))
    [ [0.48045301391820142...] + [+/- ...]*I [-1.38629436111989061...] + [+/- ...]*I  [2.00000000000000000...] + [+/- ...]*I]

    sage: borel_laplace((x*Dx+1)^3, 1/1000, RBF(0), RBF(1e-40))[0,0]
    [23858.541497152791047527498748673079612...] + [+/- ...]*I

Convergent series::

    sage: mat = borel_laplace(Dx-1, 1/2, RBF(0), RBF(1e-100)) # long time
    sage: RealBallField(400)(1/2).exp() in mat[0,0] # long time
    True

Fauvet-Thomann 2005, §4.2 (homogeneized). No reference values in the paper  but
the computed value agrees with the sum to the least term. ::

    sage: borel_laplace(
    ....:     -x^3*(x+1)*(x^2+1)*Dx^2-x*(2*x^4+4*x^3+x+1)*Dx-2*x^2+2*x+1,
    ....:     1/8, RBF(0), RBF(1e-65))
    [[0.13709092399032135599904213260371938637452001398221067...] + [+/- ...]*I]

    sage: all( # long time
    ....:     (borel_laplace(
    ....:          -x^3*(x+1)*(x^2+1)*Dx^2-x*(2*x^4+4*x^3+x+1)*Dx-2*x^2+2*x+1,
    ....:          1/8, RBF(0), RBF(10)^(-p))[0,0].rad() < 4*RR(10)^(-p))
    ....:     for p in range(10, 100, 10))
    True
"""

import logging

from sage.matrix.constructor import identity_matrix, matrix
from sage.matrix.special import companion_matrix
from sage.modules.free_module_element import vector
from sage.rings.complex_arb import CBF, ComplexBallField, ComplexBall
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.real_arb import RBF, RealBall
from sage.rings.real_mpfr import RealField
from sage.structure.element import coercion_model

from . import utilities

from .context import dctx
from .differential_operator import DifferentialOperator
from .local_solutions import (
    LocalBasisMapper,
    log_series,
)
from .path import Point

logger = logging.getLogger(__name__)

################################################################################
# Initial values for transformed operators
################################################################################

class IniMap(LocalBasisMapper):
    r"""
    Shared part of the code for computing initial values at regular singular
    points for transformed operators.

    Assumption: the coefficient of ``z^α`` in the transformed solution (a
    polynomial in ``log(z)``) is determined by the coefficients of ``z^β`` in
    the original series for _integer_ ``β - α ≤ max_shift``.
    """

    max_shift = 0

    def __init__(self, dop, dop1, ring, *, ctx=dctx):
        dop = DifferentialOperator(dop)
        super().__init__(dop, ctx=ctx)
        origin = Point(0, DifferentialOperator(dop1))
        self.struct1 = origin.local_basis_structure(critical_monomials=False)
        self.ring = ring

    def fun(self, ini):

        # determine how many terms of the current sol of the original
        # operator we need
        _leftmost = self.leftmost.as_ball(CBF)
        order = 0
        for sol1 in self.struct1:
            offset = sol1.leftmost.as_ball(CBF) - _leftmost
            if offset.contains_integer():
                # we need the term of index val(sol1) + self.max_shift
                order1 = offset.real().upper().floor()
                order1 += sol1.shift + self.max_shift + 1
                order = max(order, order1)

        # compute these terms
        # XXX there likely is some redundancy here, cf. HighestSolMapper
        shifted_bwrec = self.bwrec.shift_by_PolynomialRoot(self.leftmost)
        ser = log_series(ini, shifted_bwrec, order)

        # deduce the initial values of dop1 corresponding to the transform
        # of the current solution
        inicol = []
        leftmost = self.leftmost.as_algebraic()
        for sol1 in self.struct1:
            offset = sol1.leftmost.as_algebraic() - leftmost
            try:
                offset = ZZ(offset)
            except ValueError:
                inicol.append(self.ring.zero())
                continue
            ini1 = self.compute_coefficient(sol1, ser, offset)
            inicol.append(ini1)

        return inicol

    def compute_coefficient(self, sol1, ser, offset):
        r"""
        Compute the coefficient of ``z^n·log(z)^k/k!`` in the image of `f`,
        given the expansion of `f` truncated at order ``n + max_shift + 1``.

        INPUT:

        The parameters ``n`` and ``k`` are passed as a ``FundamentalSolution``
        structure ``sol1`` with no values.

        The series expansion of `f` is passed as a list ``ser`` where the
        coefficient of ``z^(sol1.leftmost + s)`` is in position ``offset + s``.
        """
        raise NotImplementedError

    def run(self):
        r"""
        Compute a matrix mapping vectors of initial conditions for the original
        operator to vectors of initial conditions for the transformed operator.

        In general, this is a “tall” matrix.
        """
        inicols = super().run()
        # explicit dimensions are necessary to determine the dimension of the
        # target space when inicols == []
        inimap = matrix(len(inicols), len(self.struct1),
                        [sol.value for sol in inicols]).transpose()
        assert inimap.ncols() <= self.dop.order()
        assert inimap.nrows() == len(self.struct1)
        return inimap

class BorelIniMap(IniMap):
    r"""
    Compute generalized initial values for a Borel transform.

    The origin may be an irregular singular point of the original operator.

    The output parent ``ring`` must be a ``ComplexBallField``.
    """

    # Borel(z^(σ+1)·log(z)^k/k!) = ζ^σ·∑ γ[i]·log(ζ)^(k-i)/(k-i)! where γ[i] =
    # [ε^i] 1/Γ(σ+1+ε) (vdH2007, §2.1, notation modified),
    # so the coefficient of a monomial ζ^σ·log(ζ)^k/k! in the transform is
    # determined by the coefficients of z^(σ+1)·log(z)^k'/k'!, k' ≥ k, in the
    # original function

    max_shift = 1

    def compute_coefficient(self, sol1, ser, offset):
        expo = sol1.valuation_as_ball(self.ring)
        ICeps = PolynomialRing(self.ring, 'eps')
        coeff = ser[offset + sol1.shift + 1]
        # XXX redundant computations when log_power > 0
        rgamma = ICeps([expo + 1, 1])._rgamma_series(len(coeff))
        # k-i = sol1.log_power
        return sum(rgamma[i]*c for i, c in enumerate(coeff[sol1.log_power:]))

class IntIniMap(IniMap):
    r"""
    Compute generalized initial values for an antiderivative.

    Currently requires ``ring`` to be a ``ComplexBallField``, but this is not
    essential.
    """

    max_shift = -1

    def compute_coefficient(self, sol1, ser, offset):
        if sol1.leftmost.try_integer() == -sol1.shift:
            return self.ring.zero()
        expo = sol1.valuation_as_ball(self.ring)
        coeff = ser[offset + sol1.shift - 1]
        s = sum((c if i % 2 == 0 else -c)/expo**(i+1)
                for i, c in enumerate(coeff[sol1.log_power:]))
        return s

class MulBySeriesIniMap(IniMap):

    def __init__(self, dop, dop1, ring, compute_cofactor, *, ctx=dctx):
        super().__init__(dop, dop1, ring, ctx=ctx)
        self.compute_cofactor = compute_cofactor
        self.__cof_ser = None

    def process_decomposition(self):
        # we need d + 1 terms of the cofactor where d is the dispersion of the
        # indicial polynomial of the original operator
        dispersion = max(shifts[-1][0] if shifts else 0
                         for _, shifts in self.sl_decomp)
        self.__cof_ser = self.compute_cofactor(dispersion + 1)

    def compute_coefficient(self, sol1, ser, offset):
        return sum(ser[offset+i][sol1.log_power]*self.__cof_ser[sol1.shift-i]
                   for i in range(-offset, sol1.shift + 1))

################################################################################
# Bounds on tails of integral transforms
################################################################################

# Note to self: 2022-08-24-A
# TODO: support bounds on a sector?
def bound_rats_on_ray(nums, poles, vcden, z0):
    r"""
    Bound rational functions with the same denominator
    on the ray ``[z0, z0*infinity)``.

    The common denominator is given by its roots and its valuation coefficient.
    """
    degden = sum(m for _, m in poles)
    if any(num.degree() > degden for num in nums):
        raise ValueError("unbounded")
    # Reduce to z0 == 1
    z = nums[0].parent().gen() if nums else None
    poles = [(s/z0, m) for s, m in poles]
    dencst = z0**degden # den(z0*z) = dest*prod(z-poles); we put the cst in num
    nums = [num(z0*z)/dencst for num in nums]
    # For each num, bound t^degden·num(1/t) where t = 1/z
    iv = RBF(0.5, 0.5)
    bnums = [(num.reverse(degree=degden)(iv)).above_abs() for num in nums]
    # Lower-bound t^degden·den(1/t) = vcden·∏(t-1/s, s in poles)
    bden = (vcden).below_abs()
    for s, mult in poles:
        if s.is_zero():
            continue
        u = ~s
        # u may overlap several of the regions corresponding to each case
        d = RBF('inf')
        if not (u.real() >= RBF.zero()):
            d = d.min(u.below_abs())
            logger.debug("case A: s=%s, u=%s, d=%s", s, u, d)
        if not (u.real() <= RBF.zero() or u.real() >= RBF.one()):
            d = d.min(u.imag().below_abs())
            logger.debug("case A: s=%s, u=%s, d=%s", s, u, d)
        if not (u.real() <= RBF.one()):
            d = d.min((u - 1).below_abs())
            logger.debug("case A: s=%s, u=%s, d=%s", s, u, d)
        bden *= d**mult
    return [bnum/bden for bnum in bnums]

class ExponentialBoundOnRay:
    r"""
    A positive function defined on the complex half-line ``[base, base*∞)``.
    """

    def __init__(self, cst, expcoeff, base):
        assert isinstance(cst, RealBall)
        self.cst = cst
        assert isinstance(expcoeff, RealBall)
        self.expcoeff = expcoeff
        assert isinstance(base, (RealBall, ComplexBall))
        self.base = base

    def __repr__(self):
        R = RealField(30)
        return f"{R(self.cst)}*exp({R(self.expcoeff)}*abs(z-({self.base})))"

    def __call__(self, z):
        return self.cst*(self.expcoeff*abs(z - self.base)).exp()

    def __mul__(self, other):
        assert isinstance(other, ExponentialBoundOnRay)
        # The rays must be contained in one another
        assert (self.base.contains_zero()
                or (other.base/self.base).real().contains_zero())
        if not other.base.is_zero():
            raise NotImplementedError
        return ExponentialBoundOnRay(self.cst*other(self.base),
                                     self.expcoeff + other.expcoeff,
                                     self.base)

    def integral(self):
        r"""
        Integrate this bound on the ray from ``base`` to ∞.
        """
        if self.expcoeff < 0:
            int_value = -1/self.expcoeff
        elif self.expcoeff > 0:
            int_value = RBF('inf')
        else:
            int_value = RBF('nan')
        return self.cst*int_value

def _frobenius_norm(mat):
    return sum((abs(c)**2 for c in mat.list()),
               start=abs(mat.base_ring().zero())).sqrtpos()

# XXX isolate the first terms (least term summation + bound on tail)???
def bound_fundamental_matrix_on_ray(dop, z0, ini):
    r"""
    Bound the Frobenius norm of the fundamental matrix of ``dop`` defined by
    ``F(z0) = ini`` on the ray ``[z0, z0*infinity)``.
    """
    # XXX maybe extend to support rectangular ini
    dop = DifferentialOperator(dop)
    sing = dop._singularities(CBF, multiplicities=True)
    lc = dop.leading_coefficient()
    c = CBF(lc[lc.valuation()])
    coeff_bounds = bound_rats_on_ray(list(dop)[:-1], sing, c, z0)
    mat = companion_matrix(coeff_bounds + [RBF.one()])
    mat = mat.change_ring(CBF) # work around sage bug #34691
    # Grönwall's inequality
    # Note to self: see 2022-10-27-A for ideas on how to improve these bounds
    expcoeff = _frobenius_norm(mat).above_abs()
    cst = _frobenius_norm(ini).above_abs()
    return ExponentialBoundOnRay(cst, expcoeff, z0)

################################################################################
# Laplace transform
################################################################################

def laplace_kernel_dop(Dop, pt, diff_order):
    r"""
    Differential operator annihilating ``(d/dp)^diff_order exp(-z/p)``.

    EXAMPLES::

        sage: from ore_algebra import OreAlgebra
        sage: from ore_algebra.analytic.borel_laplace import *
        sage: P0.<p> = QQ[]
        sage: P1.<z> = P0[]
        sage: Dop.<Dz> = OreAlgebra(P1)
        sage: laplace_kernel_dop(Dop, p, 2)
        (p*z^2 - 2*p^2*z)*Dz + z^2 - 4*p*z + 2*p^2
    """
    z = Dop.base_ring().gen()
    Pol, p = PolynomialRing(Dop.base_ring(), 'p').objgen()
    a = Pol.fraction_field().one() # cofactor of exp(...) in (d/dp)^k exp(...)
    b = 1/p  # idem in - (d/dp)^k (d/dz) exp(...)
    for _ in range(diff_order):
        a = a.derivative() + a*z/p**2
        b = b.derivative() + b*z/p**2
    quo = a/b
    return Dop([quo.denominator()(pt), quo.numerator()(pt)])

def trd_mat_from_int_mat(int_mat, ini_map, z1, zeta1):
    r"""
    Deduce the transition matrix for the annihilator of the Laplace
    “transformand” from a transition matrix for the annihilator of the truncated
    Laplace integral.

    INPUT:

    - ``int_mat`` - transition matrix for the truncated integral
    - ``ini_map`` - matrix mapping initial values for the transformand to
      initial values for the integral
    - ``z1`` - evaluation point of the Laplace transform
    - ``zeta1`` truncation point of the Laplace integral

    OUTPUT:

    The result is returned and the unevaluated product of a constant and a
    matrix.

    TESTS::

        sage: from ore_algebra import OreAlgebra
        sage: from ore_algebra.analytic.borel_laplace import *
        sage: Pol.<x> = QQ[]; Dop.<Dx> = OreAlgebra(Pol)

    Fauvet-Thomann 2005, §4.2, homogeneized::

        sage: dop = -x^3*(x+1)*(x^2+1)*Dx^2-x*(2*x^4+4*x^3+x+1)*Dx-2*x^2+2*x+1
        sage: z1 = 1/8; zeta1 = 5
        sage: bdop = dop.borel_transform()
        sage: ker_dop = z1*Dx + 1
        sage: itd_dop = bdop.symmetric_product(ker_dop).numerator()
        sage: int_dop = itd_dop*Dx
        sage: ini_map = IntIniMap(itd_dop, int_dop, CBF).run()
        sage: ini_map = ini_map*laplace_integrand_ini_map(itd_dop, z1, CBF)
        sage: ini_map
        [                 0                  0                  0]
        [ 1.000000000000000                  0                  0]
        [-1.000000000000000  1.000000000000000                  0]
        [ 1.750000000000000 -4.000000000000000 0.5000000000000000]
        sage: int_mat = int_dop.numerical_transition_matrix([0, zeta1], 1e-50)
        sage: bdop_mat = trd_mat_from_int_mat(int_mat, ini_map, RBF(z1), RBF(zeta1))
        sage: bdop_mat
        [ [-1.0006373340063...] [-0.0143480197420...]  [-0.367285793951...]]
        [  [-0.440309653236...]   [0.460498392700...]   [0.171999422527...]]
        [    [0.53698867828...]   [-0.03120085618...]    [0.16930961176...]]
        sage: ref = bdop.numerical_transition_matrix([0, zeta1])
        sage: all(ref[i,j] in bdop_mat[i,j] for i in range(3) for j in range(3))
        True
    """
    Ser, eps = PolynomialRing(int_mat.base_ring(), 'eps').objgen()
    order = int_mat.nrows() - 1
    # Transition matrix for the integral, but in the canonical basis of trd_dop
    int_mat1 = int_mat*ini_map
    # XXX extend and use ker_series?
    scaled_inv_ker_ser = (eps/z1)._exp_series(order)
    # Series expansions of the integrands. Taking the derivative here is not
    # exactly the same as deleting the first row of int_mat!
    itd_ser = [Ser(list(col)).derivative() for col in int_mat1.columns()]
    scaled_trd_ser = [ser.multiplication_trunc(scaled_inv_ker_ser, order)
                      for ser in itd_ser]
    scaled_trd_mat = matrix([ser.padded_list(order)
                             for ser in scaled_trd_ser])
    scaled_trd_mat = scaled_trd_mat.transpose()
    cst = (zeta1/z1).exp()
    return cst*scaled_trd_mat

def laplace_integrand_ini_map(itd_dop, z1, ring): # XXX derivatives
    zeta = PolynomialRing(ring, 'zeta').gen()
    ker_series = lambda order: (-zeta/z1)._exp_series(order)
    return MulBySeriesIniMap(itd_dop, itd_dop, ring, ker_series).run()

def _check_singular_direction(dop, dir):
    sing = dop._singularities(CBF, multiplicities=False)
    for s in sing:
        s /= dir
        if s.imag().contains_zero() and s.real() > 0:
            raise ValueError("singular direction (or close)")

# - It would be better to compute all required derivatives simultaneously, but
#   we do not have the tooling for that at the moment.
# - Ideally, we should work in a basis adapted to the decomposition
#   int_dop = Dz*itd_dop
def analytic_laplace(trd_dop, z1, theta, eps, derivative=0, *, ctx):

    IC = ComplexBallField(utilities.prec_from_eps(eps))
    Dz = trd_dop.parent().gen()

    ker_dop = laplace_kernel_dop(
        coercion_model.common_parent(trd_dop.parent(), z1),
        z1, derivative)
    itd_dop = trd_dop.symmetric_product(ker_dop)
    itd_dop = DifferentialOperator(itd_dop)
    itd_ini_map = laplace_integrand_ini_map(itd_dop, z1, IC)
    logger.debug("Integrand: %s", LazyDiffopInfo(itd_dop, itd_ini_map))
    int_dop = itd_dop*Dz
    int_ini_map = IntIniMap(itd_dop, int_dop, IC).run()
    logger.debug("Truncated Laplace transform: %s",
                 LazyDiffopInfo(int_dop, int_ini_map))
    ini_map = int_ini_map*itd_ini_map

    dir = IC(0, theta).exp()
    _dir = CBF(dir)
    _check_singular_direction(itd_dop, _dir)
    expcoeff = -(_dir/z1).real()
    if not expcoeff < 0:
        raise ValueError("evaluation point not in the half-plane bisected "
                         "by the direction (or too close to the border)")

    zeta0, zeta1 = 0, dir # XXX better initial zeta1? (|ζ₁| ≈ |z₁|?)
    int_mat = identity_matrix(IC, int_dop.order())
    prev_tail_bound = RBF('inf')
    while True:
        step_mat = int_dop.numerical_transition_matrix([zeta0, zeta1], eps/16,
                                                       ctx=ctx)
        int_mat = step_mat*int_mat

        # Bound on the fundamental matrix of trd_dop corresponding to unit
        # initial values at zero, valid on the ray [ζ₁, ζ₁·∞).
        # Computing the bound this way can lead to large overestimations at low
        # precision, especially for large ζ₁/z₁. On simple examples at least,
        # the overestimation seems already to be present in int_mat; the
        # conversion does not make it much worse.
        _zeta1 = CBF(zeta1)
        trd_mat = trd_mat_from_int_mat(int_mat, ini_map, z1, _zeta1)
        trd_bound = bound_fundamental_matrix_on_ray(trd_dop, _zeta1, trd_mat)
        # Bound on the kernel, on the same ray
        ker_bound = ExponentialBoundOnRay(RBF.one(), expcoeff, CBF.zero())
        # Bound the tail of the Laplace transform of each individual entry of
        # the fundamental matrix (but not on the norm of the whole matrix).
        # [If f is one of the elements of the basis, we have no particular
        # interest in L(f'), which anyhow is just (1/z₁)·Lf-f(0), but afaict the
        # bound is valid for it.]
        itg_bound = trd_bound*ker_bound
        tail_bound = itg_bound.integral()
        logger.info("integrand bounded by %s on [%s, %s∞)",
                    itg_bound, _zeta1, _zeta1)

        int_row = vector([val.add_error(tail_bound) for val in int_mat.row(0)])
        if (all(val.rad() < eps.lower() for val in int_row)
                # we expect tail_bound to decrease, val.rad() to increase
                or any(val.rad() > tail_bound.upper() for val in int_mat.row(0))
                # but tail_bound may also get worse
                or not tail_bound <= prev_tail_bound):
            return int_row*ini_map

        zeta0, zeta1 = zeta1, 2*zeta1

def _shift_exponents_to_right_hand_plane(dop):
    # Compute a change of unknown function that shifts all exponents to the open
    # right half-plane, so that the Laplace transform of each monomial
    # converges. (Thanks to the analytic continuation of Γ(z), it might actually
    # be enough to make all _integer_ exponents positive?)

    z = dop.base_ring().gen()
    Dz = dop.parent().gen()

    radind = dop.indicial_polynomial(z).radical()
    exponents = radind.roots(CBF, multiplicities=False)
    zshift = 1 - min(a.real().lower().ceil() for a in exponents)
    shifted_dop = dop.symmetric_product(z*Dz - zshift)

    return zshift, shifted_dop

def borel_laplace(dop, pt, theta, eps, derivatives=1, *, ctx=dctx):
    if derivatives != 1:
        raise NotImplementedError
    IC = ComplexBallField(utilities.prec_from_eps(eps))

    zshift, shifted_dop = _shift_exponents_to_right_hand_plane(dop)
    logger.debug("Shifted operator: %s",
                 LazyDiffopInfo(shifted_dop))

    borel_dop = DifferentialOperator(shifted_dop.borel_transform())
    borel_ini_map = BorelIniMap(shifted_dop, borel_dop, IC).run()
    logger.debug("Borel transform: %s",
                 LazyDiffopInfo(borel_dop, borel_ini_map))

    laplace_val = analytic_laplace(borel_dop, pt, theta, eps, ctx=ctx)
    assert len(laplace_val) == borel_dop.order()
    unshifted_val = pt**(-zshift)*laplace_val

    unshifted_mat = matrix(1, len(unshifted_val), unshifted_val)

    result_mat = unshifted_mat*borel_ini_map
    return result_mat

################################################################################
# Utilities
################################################################################

class LazyDiffopInfo:

    def __init__(self, dop, ini=None):
        self.dop = dop
        self.ini = ini

    def __repr__(self):
        s  = '\n' + str(self.dop)
        s += '\n' + str(self.dop.local_basis_expansions(0))
        if self.ini is not None:
            s += '\n' + str(self.ini)
        return s
