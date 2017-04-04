# -*- coding: utf-8 - vim: tw=80
r"""
Recurrence relations
"""

from sage.misc.cachefunc import cached_method
from sage.rings.all import ZZ, QQ
from sage.rings.number_field.number_field import NumberField_quadratic
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

from .. import ore_algebra

def backward_rec(dop, shift=ZZ.zero()):
    Pols_n = PolynomialRing(dop.base_ring().base_ring(), 'n') # XXX: name
    Rops = ore_algebra.OreAlgebra(Pols_n, 'Sn')
    # Using the primitive part here would break the computation of residuals!
    # TODO: add test (arctan); better fix?
    # rop = dop.to_S(Rops).primitive_part().numerator()
    rop = dop.to_S(Rops)
    ordrec = rop.order()
    coeff = [rop[ordrec-k](Pols_n.gen()-ordrec+shift)
             for k in xrange(ordrec+1)]
    return BackwardRec(coeff)

class BackwardRec(object):
    r"""
    A recurrence relation, written in terms of the backward shift operator.

    This class is mainly intended to provide reasonably fast evaluation in the
    context of naïve unrolling.
    """

    def __init__(self, coeff):
        assert isinstance(coeff[0], polynomial_element.Polynomial)
        self.coeff = coeff
        self.base_ring = coeff[0].parent()
        Scalars = self.base_ring.base_ring()
        self.order = len(coeff) - 1
        # Evaluating polynomials over ℚ[i] is slow...
        # TODO: perhaps do something similar for eval_series
        if (isinstance(Scalars, NumberField_quadratic)
                and list(Scalars.polynomial()) == [1,0,1]):
            QQn = PolynomialRing(QQ, 'n')
            self._re_im = [
                    (QQn([c.real() for c in pol]), QQn([c.imag() for c in pol]))
                    for pol in coeff]
            self.eval_int_ball = self._eval_qqi_cbf

    # efficient way to cache the last few results (without cython)?
    def eval(self, tgt, point):
        return [tgt(pol(point)) for pol in self.coeff]

    def _eval_qqi_cbf(self, tgt, point):
        return [tgt(re(point), im(point)) for re, im in self._re_im]

    # Optimized implementation of eval() when point is a Python
    # integer and iv is a ball field. Can be dynamically replaced by
    # one of the above implementations on initialization.
    eval_int_ball = eval

    @cached_method
    def _coeff_series(self, i, j):
        if j == 0:
            return self.coeff[i]
        else:
            return self._coeff_series(i, j - 1).diff()/j

    def eval_series(self, tgt, point, ord):
        return [[tgt(self._coeff_series(i,j)(point)) for j in xrange(ord)]
                for i in xrange(len(self.coeff))]

    def eval_inverse_lcoeff_series(self, tgt, point, ord):
        ser = self.base_ring( # polynomials, viewed as jets
                [self._coeff_series(0, j)(point) for j in xrange(ord)])
        inv = ser.inverse_series_trunc(ord)
        return [tgt(c) for c in inv]

    def __getitem__(self, i):
        return self.coeff[i]

    def shift(self, sh):
        n = self.coeff[0].parent().gen()
        return BackwardRec([pol(sh + n) for pol in self.coeff])


