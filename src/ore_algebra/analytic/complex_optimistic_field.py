# -*- vim: tw=80
"""
A version of ``ComplexBallField`` with an optimistic zero test
(``b==0`` as soon as ``b`` contains 0)
"""

# Copyright 2021 Marc Mezzarobba, ?
# Copyright 2021 Alexandre Goyer, Inria Saclay Ile-de-France
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

import sage.categories.fields

from ore_algebra.analytic.accuracy import PrecisionError
from sage.categories.functor import Functor
from sage.categories.homset import Hom
from sage.categories.morphism import SetMorphism
from sage.categories.pushout import ConstructionFunctor
from sage.rings.complex_arb import ComplexBallField, ComplexBall
from sage.rings.real_mpfr import RealField
from sage.rings.ring import Field
from sage.structure.element import Element, RingElement
from sage.structure.richcmp import (op_EQ, op_NE, op_LT, op_LE, op_GT, op_GE,
        rich_to_bool)
from sage.structure.unique_representation import UniqueRepresentation


Radii = RealField(30)

class OptimisticFunctor(ConstructionFunctor):
    rank = 6

    def __init__(self, eps):
        Functor.__init__(self, sage.categories.fields.Fields(),
                sage.categories.fields.Fields())
        self.eps = eps

    def _apply_functor(self, R):
        assert isinstance(R, ComplexBallField)
        return ComplexOptimisticField(R.precision(), self.eps)

class ComplexOptimisticBall(RingElement):

    def __init__(self, parent, z):
        if z.parent() is not parent._ball_field:
            raise TypeError
        RingElement.__init__(self, parent)
        self.value = z

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return repr(self.value)

    def _is_atomic(self):
        return self.value._is_atomic()

    def __nonzero__(self):
        return bool(self.value)

    def _richcmp_(left, right, op):
        eps = left.parent().eps
        if left.value.overlaps(right.value):
            if left.value.rad() <= eps and right.value.rad() <= eps:
                return rich_to_bool(op, 0)
            else:
                raise PrecisionError("Cannot compare this two numbers.")
        else:
            return left.value._richcmp_(right.value, op)

    def __neg__(self):
        return self.__class__(self.parent(), -self.value)

    def __invert__(self):
        return self.__class__(self.parent(), ~self.value)

    def _add_(self, other):
        return self.__class__(self.parent(), self.value + other.value)

    def _sub_(self, other):
        return self.__class__(self.parent(), self.value - other.value)

    def _mul_(self, other):
        return self.__class__(self.parent(), self.value * other.value)

    def _div_(self, other):
        return self.__class__(self.parent(), self.value / other.value)

    def contains_zero(self):
        return self.value.contains_zero()

    def is_nonzero(self):
        return not self == self.parent().zero()

    def rad(self):
        return self.value.rad()

    def mid(self):
        return self.value.mid()

    def below_abs(self):
        return self.value.below_abs()

    def above_abs(self):
        return self.value.above_abs()

    def real(self):
        return self.value.real()

    def imag(self):
        return self.value.imag()

    def accuracy(self):
        return self.value.accuracy()

    def add_error(self, arg):
        return self.value.add_error(arg)

class ComplexOptimisticField(UniqueRepresentation, Field):

    Element = ComplexOptimisticBall

    @staticmethod
    def __classcall__(cls, prec=53, eps=None):
        prec = int(prec)
        if eps is None:
            eps = Radii.one() >> (7*prec//8)
        else:
            eps = Radii.coerce(eps)
        return super(ComplexOptimisticField, cls).__classcall__(cls, prec, eps)

    def __init__(self, prec, eps):
        self.eps = eps
        Field.__init__(self,
                base_ring=self, # because we have no real analogue
                category=sage.categories.fields.Fields().Infinite())
        self._ball_field = ComplexBallField(prec)

        mor = SetMorphism(Hom(self, self._ball_field), lambda x: x.value)
        mor.register_as_conversion()

    def construction(self):
        return OptimisticFunctor(self.eps), self._ball_field

    def __repr__(self):
        return f"{self._ball_field} and approximate zero test to {self.eps}"

    def complex_field(self):
        return self

    def _coerce_map_from_(self, other):
        return self._ball_field.has_coerce_map_from(other)

    def _element_constructor_(self, z=None):
        if isinstance(z, ComplexOptimisticBall):
            z = z.value
        return self.element_class(self, self._ball_field(z))

    def _an_element_(self):
        return self(self._ball_field._an_element_())

    def precision(self):
        return self._ball_field().precision()

    def is_exact(self):
        return True
