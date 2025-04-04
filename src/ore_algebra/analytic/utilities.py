# vim: tw=80
"""
Miscellaneous utilities
"""

# Copyright 2015, 2016, 2017, 2018 Marc Mezzarobba
# Copyright 2015, 2016, 2017, 2018 Centre national de la recherche scientifique
# Copyright 2015, 2016, 2017, 2018 Université Pierre et Marie Curie
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

import itertools
import sys
import warnings


import sage.rings.complex_arb
import sage.rings.real_arb

from sage.categories.pushout import pushout
from sage.matrix.matrix_complex_ball_dense import Matrix_complex_ball_dense
from sage.misc.cachefunc import cached_function
from sage.misc.misc import cputime
from sage.rings.qqbar import (qq_generator, AlgebraicNumber, ANExtensionElement)
from sage.rings.integer_ring import Z as ZZ
from sage.rings.rational_field import Q as QQ
from sage.rings.qqbar import QQbar
from sage.rings.complex_arb import CBF, ComplexBall, ComplexBallField
from sage.rings.number_field.number_field import (
    GaussianField,
    NumberField,
    NumberField_quadratic)
from sage.rings.real_arb import RealBall, RealBallField
from sage.rings.number_field import number_field_base
from sage.rings.number_field.number_field_element_quadratic import NumberFieldElement_quadratic
from sage.structure.coerce_exceptions import CoercionException
from sage.structure.element import coercion_model
from sage.structure.factorization import Factorization
from sage.structure.sequence import Sequence

######################################################################
# Timing
######################################################################

class Clock:
    def __init__(self, name="time"):
        self.name = name
        self._sum = 0.
        self._tic = None
    def __repr__(self):
        return "{} = {} s".format(self.name, self.total())
    def since_tic(self):
        return 0. if self._tic is None else cputime(self._tic)
    def total(self):
        return self._sum + self.since_tic()
    def tic(self, t=None):
        assert self._tic is None
        self._tic = cputime() if t is None else t
    def toc(self):
        self._sum += cputime(self._tic)
        self._tic = None

class Stats:
    def __repr__(self):
        return ", ".join(str(clock) for clock in self.__dict__.values()
                                    if isinstance(clock, Clock))

######################################################################
# Numeric fields
######################################################################

_RBFmin = sage.rings.real_arb.RealBallField(2)
_CBFmin = sage.rings.complex_arb.ComplexBallField(2)

def is_numeric_parent(parent):
    return _CBFmin.has_coerce_map_from(parent)

def is_real_parent(parent):
    return _RBFmin.has_coerce_map_from(parent)

def is_QQi(parent):
    return (isinstance(parent, NumberField_quadratic)
                and list(parent.polynomial()) == [1,0,1]
                and CBF(parent.gen()).imag().is_one())

def ball_field(eps, real):
    prec = prec_from_eps(eps)
    if real:
        return sage.rings.real_arb.RealBallField(prec)
    else:
        return sage.rings.complex_arb.ComplexBallField(prec)

def exactify_polynomial(pol):
    base = pol.base_ring()
    if base.is_exact():
        return pol
    elif isinstance(base, RealBallField):
        return pol.change_ring(QQ)
    elif isinstance(base, ComplexBallField):
        if all(c.is_real() for c in pol):
            return pol.change_ring(QQ)
        else:
            return pol.change_ring(GaussianField())
    else:
        raise NotImplementedError

################################################################################
# Number fields and orders
################################################################################

def internal_denominator(a):
    r"""
    Denominator of the internal representation of ``a``.

    TESTS::

        sage: from ore_algebra import OreAlgebra
        sage: Dx = OreAlgebra(PolynomialRing(QQ, 'x'), 'Dx').gen()
        sage: from ore_algebra.analytic.utilities import internal_denominator
        sage: K.<a> = QuadraticField(1/27)
        sage: internal_denominator(a)
        9
        sage: (Dx - a).local_basis_expansions(0)
        [1 + a*x + 1/54*x^2 + 1/162*a*x^3]
    """
    if isinstance(a, NumberFieldElement_quadratic):
        # return the denominator in the internal representation based on √disc
        return a.__reduce__()[-1][-1]
    elif isinstance(a, (RealBall, ComplexBall)):
        return a.parent().one()
    else:
        return a.denominator()

def as_embedded_number_field_elements(algs):
    # Adapted (in part) from sage's number_field_elements_from algebraics(),
    # because the latter loses too much time trying to detect if the numbers are
    # real.
    gen = qq_generator
    algs = [QQbar.coerce(a) for a in algs]
    for a in algs:
        a.simplify()
        gen = gen.union(a._exact_field())
    nf = gen._field
    if nf is not QQ:
        gen_emb = AlgebraicNumber(ANExtensionElement(gen, nf.gen()))
        nf = NumberField(nf.polynomial(), nf.variable_name(),
                         embedding=gen_emb)
        algs = [gen(a._exact_value()).polynomial()(nf.gen()) for a in algs]
    return nf, algs

def as_embedded_number_field_element(alg):
    return as_embedded_number_field_elements([alg])[1][0]

def number_field_with_integer_gen(K):
    r"""
    TESTS::

        sage: from ore_algebra.analytic.utilities import number_field_with_integer_gen
        sage: K = NumberField(6*x^2 + (2/3)*x - 9/17, 'a')
        sage: number_field_with_integer_gen(K)[0]
        Number Field in x306a with defining polynomial x^2 + 34*x - 8262 ...
    """
    if K is QQ:
        return QQ, ZZ
    den = K.polynomial().monic().denominator()
    if den.is_one():
        # Ensure that we return the same number field object (coercions can be
        # slow!)
        intNF = K
    else:
        intgen = K.gen() * den
        ### Attempt to work around various problems with embeddings
        emb = K.coerce_embedding()
        embgen = emb(intgen) if emb else intgen
        # Write K.gen() = α = β/q where q = den, and
        # K.polynomial() = q + p[d-1]·X^(d-1) + ··· + p[0].
        # By clearing denominators in P(β/q) = 0, one gets
        # β^d + q·p[d-1]·β^(d-1) + ··· + p[0]·q^(d-1) = 0.
        intNF = NumberField(intgen.minpoly(), "x" + str(den) + str(K.gen()),
                            embedding=embgen)
        assert intNF != K
    # Work around weaknesses in coercions involving order elements,
    # including #14982 (fixed). Used to trigger #14989 (fixed).
    #return intNF, intNF.order(intNF.gen())
    return intNF, intNF

def invert_order_element(alg):
    if alg in ZZ:
        return 1, alg
    else:
        Order = alg.parent()
        pol = alg.polynomial().change_ring(ZZ)
        modulus = Order.gen(1).minpoly()
        den, num, _ = pol.xgcd(modulus)  # hopefully fraction-free!
        return Order(num), ZZ(den)

def mypushout(X, Y):
    if X.has_coerce_map_from(Y):
        return X
    elif Y.has_coerce_map_from(X):
        return Y
    else:
        Z = pushout(X, Y)
        if (isinstance(X, number_field_base.NumberField)
                and isinstance(Y, number_field_base.NumberField)
                and not isinstance(Z, number_field_base.NumberField)):
            # we likely obtained a parent where both number fields have a
            # canonical embedding, typically QQbar...
            raise CoercionException
        return Z

def extend_scalars(Scalars, *pts):
    gen = Scalars.gen()
    try:
        # Largely redundant with the other branch, but may do a better job
        # in some cases, e.g. pushout(QQ, QQ(α)), where as_enf_elts() would
        # invent new generator names.
        NF = coercion_model.common_parent(Scalars, *pts)
        if not isinstance(NF, number_field_base.NumberField):
            raise CoercionException
        gen1 = NF.coerce(gen)
        pts1 = tuple(NF.coerce(pt) for pt in pts)
    except (CoercionException, TypeError):
        NF, val1 = as_embedded_number_field_elements((gen,)+pts)
        gen1, pts1 = val1[0], tuple(val1[1:])
    hom = Scalars.hom([gen1], codomain=NF)
    return (hom,) + pts1

def my_sequence(points):
    try:
        universe = coercion_model.common_parent(*points)
    except TypeError:
        universe, points = as_embedded_number_field_elements(
                                            [QQbar.coerce(pt) for pt in points])
    return Sequence(points, universe=universe)

######################################################################
# Sage features
######################################################################

@cached_function
def has_new_ComplexBall_constructor():
    from sage.rings.complex_arb import ComplexBall, CBF
    try:
        ComplexBall(CBF, QQ(1), QQ(1))
    except TypeError:
        return False
    else:
        return True

######################################################################
# Miscellaneous stuff
######################################################################

def prec_from_eps(eps):
    return -eps.lower().log2().floor() + 4

def input_accuracy(dop, evpts, inis):
    accuracy = min(evpts.accuracy, min(ini.accuracy() for ini in inis))
    R = dop.base_ring().base_ring()
    if not R.is_exact():
        if isinstance(R, (RealBallField, ComplexBallField)):
            accuracy = min(accuracy, min(c.accuracy() for p in dop for c in p))
        else:
            raise NotImplementedError
    return max(0, accuracy)

def split(cond, objs):
    matching, not_matching = [], []
    for x in objs:
        (matching if cond(x) else not_matching).append(x)
    return matching, not_matching

def short_str(obj, n=60):
    s = str(obj)
    if len(s) < n:
        return s
    else:
        return s[:n/2-2] + "..." + s[-n/2 + 2:]

# Adapted from itertools manual
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def myfactor_monic(pol):
    r"""
        sage: from ore_algebra.analytic.utilities import myfactor_monic
        sage: P.<x> = i.parent()[]
        sage: myfactor_monic((x^2+1)^2*(x^2-1)^3)
        [(x - 1, 3), (x + 1, 3), (x - I, 2), (x + I, 2)]
    """
    assert pol.is_monic()
    if pol.degree() == 1:
        return Factorization([(pol, 1)])
    Base = pol.base_ring()
    try:
        pol = pol.change_ring(QQ)
    except TypeError:
        return pol.factor()
    fac = []
    for f, m in pol.factor():
        f = f.change_ring(Base)
        if f.degree() == 1:
            fac.append((f, m))
        else:
            for f1, m1 in f.factor():
                fac.append((f1, m*m1))
    return fac

def binomial_coefficients(s):
    binom = [[0]*s for _ in range(s)]
    for n in range(s):
        binom[n][0] = 1
        for k in range(1, n + 1):
            binom[n][k] = binom[n-1][k-1] + binom[n-1][k]
    return binom

def ctz(vec, maxlen=sys.maxsize):
    z = 0
    for m in range(min(len(vec), maxlen)):
        if vec[-1 - m].is_zero():
            z += 1
        else:
            break
    return z

def warn_no_cython_extensions(logger, *, fallback=False):
    import sage.version
    msg = "Cython extensions not found."
    if fallback:
        msg += " Falling back to slower Python implementation."
    if list(map(int, sage.version.version.split('.')[:2])) < [10, 2]:
        msg += (
            f" (Hint: You are using SageMath version {sage.version.version}."
            " The Cython extensions in this version of ore_algebra require"
            " SageMath 10.2 or later."
            " Consider upgrading SageMath or downgrading ore_algebra to git"
            " commit 73a430aaf.)")
    warnings.warn(msg, stacklevel=2)

def invmat(mat):
    # inverting matrices over RBF yields nonsense results
    # (sage bug #38746)
    if isinstance(mat, Matrix_complex_ball_dense):
        return ~mat
    R = mat.base_ring()
    if isinstance(R, RealBallField):
        inv = ~(mat.change_ring(R.complex_field()))
        return inv.change_ring(R)
    assert R.is_exact(), "unexpected base ring in invmat"
    # mainly for identity matrices over ZZ
    return ~mat
