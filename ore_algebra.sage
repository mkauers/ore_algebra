
from sage.structure.element import RingElement
from sage.rings.ring import Algebra
from sage.rings.polynomial.polynomial_ring import is_PolynomialRing
from sage.rings.polynomial.multi_polynomial_ring import is_MPolynomialRing
from sage.rings.number_field.number_field import is_NumberField
from sage.rings.fraction_field import is_FractionField

load("ore_operator.sage")

def is_OreAlgebra(A):
    """
    Checks whether `A` is an Ore algebra object.     
    """
    return isinstance(A, OreAlgebra_generic)

def _collect_generators(R):
    """
    An auxiliary function which collects the generators of the given ring `R`, its base ring,
    the base ring of its base ring, and so on. Returns these generators as elements of `R`.

    ::

       sage: R1.<x1> = QQ['x1']
       sage: R2.<x2> = R1['x2']
       sage: R3.<x3> = (R2.fraction_field())['x3']
       sage: _collect_generators(R3)
       [x3, x2, x1, 1]
    
    """
    gens = list(R.gens()); B = R.base_ring()
    try:
        while B.base_ring() is not B:
            for x in B.gens():
                gens.append(x)
            B = B.base_ring()
    except:
        pass
    gens.append(R.one_element())
    return map(R, gens)

def _dict_to_sigma(R, d):
    """
    Given a ring `R` and a dictionary `d` whose left hand sides are generators of (some base ring of) `R`,
    construct a callable object that acts on elements of `R` as the homomorphism defined by the dictionary.

    ::

       sage: R1.<x1> = QQ['x1']
       sage: R2.<x2> = R1['x2']
       sage: R3.<x3> = (R2.fraction_field())['x3']
       sage: sigma = _dict_to_sigma(R3, {x1:2*x1, x2:1-x2, x3:x3+1})
       sage: sigma(x1+x2+x3)
       x3 - x2 + 2*x1

    WARNING:

       Problems arise if a generator is mapped to some element which does not belong to the parent of
       the generator. For example, if `R=QQ(x)(y)(z)`, then the image of `x` must belong to `QQ(x)`,
       and the image of `y` to `QQ(x)(y)`. 
    
    """
    my_dict = {}
    for x in d:
        my_dict[R(x)] = d[x]
    def sigma(p):
        R0 = p.parent()
        if p in QQ:
            return p
        elif p in R0.base_ring():
            return sigma(R0.base_ring()(p))
        elif is_FractionField(R0):
            return sigma(p.numerator())/sigma(p.denominator())
        elif is_PolynomialRing(R0):
            x = R(R0.gen())
            q = p.map_coefficients(sigma)
            if my_dict.has_key(x) and my_dict[x] != x:
                q = q(my_dict[x])
            return q
        else:
            raise NotImplementedError, "NYI: sigma for multivariate polynomials"
    
    return sigma

def _sigma_to_dict(R, sigma):
    """
    Given a ring `R` and a callable object `sigma` representing a homomorphism from `R` to itself, 
    construct a dictionary with the values of `sigma` of the generators of `R` and its base rings.
    Generators on which `sigma` acts as identity are omitted. 

    ::

       sage: R1.<x1> = QQ['x1']
       sage: R2.<x2> = R1['x2']
       sage: R3.<x3> = (R2.fraction_field())['x3']
       sage: sigma = _dict_to_sigma(R3, {x1:2*x1, x2:1-x2, x3:x3+1})
       sage: _sigma_to_dict(R3, sigma)
       {x3:x3+1, x2:1-x2, x1:2*x1}
    
    """
    d = {}
    for x in _collect_generators(R):
        sx = sigma(x)
        if x != sx:
            d[x] = sx
    return d    

def _dict_to_delta(R, d, sigma):
    """
    Given a ring `R` and a dictionary `d` whose left hand sides are generators of (some base ring of) `R`,
    and a callable object `sigma` encoding a homomorphism on `R`,
    construct a callable object that acts on elements of `R` as the skew-derivation for `sigma` defined
    by the dictionary. Generators for which no image is specified are mapped to zero. 
    Unless `delta` is the zero map, we require that `R` be a tower of polynomial or rational-function
    extensions (univariate or multivariate) of ZZ or QQ or some number field on which `delta` is identically zero.
    If `R` does not have this form, a `TypeError` is raised. 

    ::

       sage: R1.<x1> = QQ['x1']
       sage: R2.<x2> = R1['x2']
       sage: R3.<x3> = (R2.fraction_field())['x3']
       sage: delta = _dict_to_delta(R3, {x1:0, x2:1, x3:0}, lambda p:p)
       sage: delta(x1+x2+x3)
       1

    WARNING:

       Problems arise if a generator is mapped to some element which does not belong to the parent of
       the generator. For example, if `R=QQ(x)(y)(z)`, then the image of `x` must belong to `QQ(x)`,
       and the image of `y` to `QQ(x)(y)`. 
    
    """
    # 1. is delta the zero map?
    is_zero = True; zero = R.zero(); my_dict = {}
    for x in d:
        if d[x] != zero:
            is_zero = False
        my_dict[R(x), 0r] = zero
        my_dict[R(x), 1r] = d[x]
    if is_zero:
        return lambda p: zero

    # 2. check whether R has the expected form
    B = R
    while B.base_ring() != B and not is_NumberField(B):
        if is_FractionField(B):
            B = B.ring()
        if not is_PolynomialRing(B) and not is_MPolynomialRing(B):
            raise TypeError, "unexpected ring encountered in construction of skew derivation"
        B = B.base_ring()

    if is_NumberField(B):
        # check that delta is identically zero on B
        is_zero = True
        for x in _collect_generators(B):
            if my_dict.has_key((R(x), 1r)) and my_dict[R(x), 1r] != B.zero():
                is_zero = False
        if not is_zero:
            raise TypeError, "expecting skew derivation to be identically zero on number field"

    # 3. define the map and return it.
    def delta(p):
        R0 = p.parent()
        if p in B:
            return R0.zero()
        elif p in R0.base_ring():
            return delta(R0.base_ring()(p))
        elif is_FractionField(R0):
            a = p.numerator(); b = p.denominator()
            return R0(delta(a))/R0(b) - R0(delta(b)*sigma(a))/R0(b*sigma(b)) # this needs sigma(1)=1
        elif is_PolynomialRing(R0):
            x = R(R0.gen())
            if not my_dict.has_key((x, 1r)) or my_dict[x, 1r] == R.zero():
                return p.map_coefficients(delta)
            if sigma(x) == x:
                return p.map_coefficients(delta) + p.derivative(x).map_coefficients(sigma)*my_dict[x, 1r]
            for i in xrange(2r, p.degree() + 1r):
                if not my_dict.has_key((x, i)):
                    my_dict[x, i] = my_dict[x, i-1r]*x + sigma(x^(i-1r))*my_dict[x, 1r]
            out = p.map_coefficients(delta)
            for i in xrange(p.degree() + 1r):
                out += sigma(p[i])*my_dict[x, i]
            return out
        else:
            raise NotImplementedError, "NYI: delta for elements of " + str(R0)
    
    return delta

def _delta_to_dict(R, delta):
    """
    Given a ring `R`, and a callable object `delta` representing a skew-derivation on `R`,
    construct a dictionary with the values of `delta` of the generators of `R` and its base rings.

    ::

       sage: R1.<x1> = QQ['x1']
       sage: R2.<x2> = R1['x2']
       sage: R3.<x3> = (R2.fraction_field())['x3']
       sage: delta = _dict_to_delta(R3, {x1:0, x2:1, x3:0}, lambda p:p)
       sage: _delta_to_dict(R3, delta)
       {x3:0, x2:1, x1:0, 1:0}
    
    """
    d = {}
    for x in _collect_generators(R):
        dx = delta(x)
        if x != dx:
            d[x] = dx
    return d    

def OreAlgebra(base_ring, *generators, **kwargs):
    """
    An Ore algebra is a noncommutative polynomial ring whose elements are
    interpreted as operators.
    
    An Ore algebra has the form `A=R[\partial_1,\partial_2,\dots,\partial_n]`
    where `R` is a commutative ring and `\partial_1,\dots,\partial_n` are
    indeterminates.  For each of them, there is an associated automorphism
    `\sigma:R\\rightarrow R` and a skew-derivation `\delta:R\\rightarrow R`
    satisfying `\delta(a+b)=\delta(a)+\delta(b)` and
    `\delta(ab)=\delta(a)b+\sigma(a)\delta(b)` for all `a,b\in R`.

    The generators `\partial_i` commute with each other, but not with elements
    of the base ring `R`. Instead, we have the commutation rules `\partial u =
    \sigma(u) \partial + \delta(u)` for all `u\in R`.

    A typical example of an Ore algebra is the ring of linear differential
    operators with rational function coefficients in one variable,
    e.g. `A=QQ[x][D]`. Here, `\sigma` is the identity and `\delta` is the
    standard derivation `d/dx`.

    To create an Ore algebra, supply a base ring and one or more
    generators. Each generator has to be given in form of a triple
    ``(name,sigma,delta)`` where ``name`` is the desired name of the variable
    (used for printout), ``sigma`` and ``delta`` are arbitrary callable objects
    which applied to the base ring return other base ring elements in accordance
    with the relevant laws. It is not checked whether they do.

    ::

      sage: R.<x> = QQ['x']
      sage: K = R.fraction_field()
      # This creates an Ore algebra of linear differential operators
      sage: A.<D> = OreAlgebra(K, ('D', lambda p: p, lambda p: p.derivative(x)))
      sage: A
      Univariate Ore algebra in D over Fraction Field of Univariate Polynomial Ring in x over Rational Field 
      # This creates an Ore algebra of linear recurrence operators
      sage: A.<S> = OreAlgebra(K, ('S', lambda p: p(x+1), lambda p: K.zero()))
      sage: A
      Univariate Ore algebra in S over Fraction Field of Univariate Polynomial Ring in x over Rational Field

    Instead of a callable object for `sigma` and `delta`, also a dictionary can
    be supplied which for every generator of the base ring specifies the desired
    image. If the base ring is a tower of polynomial ring or fraction field
    extensions, the dictionary may specify images for any selection of
    generators at any level of the tower. If some generator is not in the
    dictionary, it is understood that `sigma` acts as identity on it, and that
    `delta` maps it to zero.

    ::

      sage: U = QQ['x'].fraction_field(); x = U.gen()
      sage: V = U['y'].fraction_field(); y = V.gen()
      # here, the base ring represents the differential field QQ(x, y=e^x)
      sage: A.<D> = OreAlgebra(V, ('D', {}, {x:1, y:0}))
      # here, the base ring represents the difference field QQ(x, 2^x)
      sage: B.<S> = OreAlgebra(V, ('S', {x:x+1, y:2*y}, {}))
      # here too, but the algebra's generator represents the forward difference instead of the shift
      sage: C.<Delta> = OreAlgebra(V, ('Delta', {x:x+1, y:2*y}, {x:1, y:y}))

    For differential and shift operators (the most common cases), there are
    shortcuts: Supplying as generator a string ``'Dx'`` consisting of ``'D'``
    followed by the name of a generator of the base ring represents a derivation
    with respect to that generator (i.e., ``sigma=lambda p: p`` and
    ``delta=lambda p: p.derivative(x)``).  Likewise, a string ``'Sx'``
    consisting of ``'S'`` followed by the name of a generator of the base ring
    represents a shift with respect to that generator (i.e., ``sigma=lambda p:
    p(x+1)`` and ``delta=lambda p:K.zero()``).

    ::

      sage: R.<x> = QQ['x']
      sage: K = R.fraction_field()
      # This creates an Ore algebra of differential operators
      sage: A.<Dx> = OreAlgebra(K, 'Dx')
      sage: A
      Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field 
      # This creates an Ore algebra of linear recurrence operators
      sage: A.<Sx> = OreAlgebra(K, 'Sx')
      sage: A
      Univariate Ore algebra in Sx over Fraction Field of Univariate Polynomial Ring in x over Rational Field 

    Ore algebras support coercion from their base rings.  Furthermore, an Ore
    algebra `A` knows how to coerce commutative polynomials `p` to elements of
    `A` if the generators of the parent of `p` have the same names as the
    generators of `A`, and the base ring of the parent of `p` admits a coercion
    to the base ring of `A`. The ring of these polynomials is called the
    associated commutative algebra of `A`, and it can be obtained by calling
    ``A.associated_commutative_algebra()``.

    Elements of Ore algebras are called Ore operators. They can be constructed
    from the same data from which also elements of the associated commutative
    algebra can be constructed.

    The conversion from data to an Ore operator is equivalent to the conversion
    from the given data to an element of the associated commutative algebra, and
    from there to an Ore operator. This has the consequence that possible implicit
    information about multiplication order may be lost, for example when generating
    operators from strings:

    ::

       sage: A = OreAlgebra(QQ['x'].fraction_field(), 'Dx')
       sage: A("Dx*x")
       x*Dx
       sage: A("Dx")*A("x")
       x*Dx + 1

    A secure way of creating operators is via a list of coefficients. These are then
    always interpreted as standing to the left of some algebra generator monomial.

    ::

       sage: R.<x> = QQ['x']
       sage: A = OreAlgebra(R.fraction_field(), 'Dx')
       sage: A([x^2+1, 5*x-7, 7*x+18])
       (7*x + 18)*Dx^2 + (5*x - 7)*Dx + x^2 + 1
       sage: (7*x + 18)*Dx^2 + (5*x - 7)*Dx + x^2 + 1
       (7*x + 18)*Dx^2 + (5*x - 7)*Dx + x^2 + 1
       sage: _^2
       (49*x^2 + 252*x + 324)*Dx^4 + (70*x^2 + 180*x)*Dx^3 + (14*x^3 + 61*x^2 + 49*x + 216)*Dx^2 + (10*x^3 + 14*x^2 + 107*x - 49)*Dx + x^4 + 12*x^2 + 37
       
       sage: A.<Sx> = OreAlgebra(R.fraction_field(), 'Sx')
       sage: A([x^2+1, 5*x-7, 7*x+18])
       (7*x + 18)*Sx^2 + (5*x - 7)*Sx + x^2 + 1
       sage: (7*x + 18)*Sx^2 + (5*x - 7)*Sx + x^2 + 1
       (7*x + 18)*Sx^2 + (5*x - 7)*Sx + x^2 + 1
       sage: _^2
       (49*x^2 + 350*x + 576)*Sx^4 + (70*x^2 + 187*x - 121)*Sx^3 + (14*x^3 + 89*x^2 + 69*x + 122)*Sx^2 + (10*x^3 - 4*x^2 + x - 21)*Sx + x^4 + 2*x^2 + 1
       
    """
    R = base_ring; gens = list(generators)

    # expand generator shortcuts, convert dictionaries to callables, and check that sigma(1)=1
    for i in xrange(len(gens)):
        if type(gens[i]) == type(""):
            if gens[i][0] == 'D':
                x = R(gens[i][1:]); gens[i] = (gens[i], lambda p: p, {x:1})
            elif gens[i][0] == 'S':
                x = R(gens[i][1:]); gens[i] = (gens[i], {x:x+1}, lambda p: R.zero())
            else:
                raise TypeError, "unexpected generator declaration"
        elif len(gens[i]) != 3:
            raise TypeError, "unexpected generator declaration"
        if type(gens[i][1]) == dict:
            gens[i] = (gens[i][0], _dict_to_sigma(R, gens[i][1]), gens[i][2])
        if type(gens[i][2]) == dict:
            gens[i] = (gens[i][0], gens[i][1], _dict_to_delta(R, gens[i][2], gens[i][1]))
        if not gens[i][1](R.one()) == R.one():
            raise ValueError, "sigma(1) must be 1"
        gens[i] = tuple(gens[i])

    # try to recognize standard operators
    is_shift = [False for q in gens]; is_qshift = [False for q in gens]; is_derivation = [False for q in gens]
    subgens = _collect_generators(R)
            
    for i in xrange(len(gens)):
        if all(gens[i][1](x) == x for x in subgens):
            deltas = [gens[i][2](x) for x in subgens]
            if sum(deltas) == 1 and all( (x==1 or x==0) for x in deltas):
                is_derivation[i] = True
        if all(gens[i][2](x) == 0 for x in subgens):
            shifts = [gens[i][1](x) - x for x in subgens]
            shifts = [x for x in shifts if x != 0]
            if len(shifts) == 1 and shifts[0] == 1:
                is_shift[i] = True
            else:
                shifts = [gens[i][1](x)/x for x in subgens]
                shifts = [x for x in shifts if x != 1]
                if len(shifts) == 1 and gens[i][1](shifts[0]) == shifts[0]:
                    is_qshift[i] = True

    # Select element class
    if kwargs.has_key("element_class"):
        operator_class = kwargs["element_class"]
    elif len(gens) > 1:
        raise NotImplementedError, "Multivariate Ore algebras still under construction"
    elif not R.is_field():
        raise NotImplementedError, "Ore algebras with non-fields as base ring still under construction"
    elif not R.is_integral_domain():
        raise TypeError, "Base rings of Ore algebras must be integral domains"
    elif not R == R.fraction_field() or not sage.rings.polynomial.polynomial_ring.is_PolynomialRing(R.base()):
        operator_class = UnivariateOreOperator
    elif is_qshift[0]:
        operator_class = UnivariateQRecurrenceOperatorOverRationalFunctionField
    elif is_shift[0]:
        operator_class = UnivariateRecurrenceOperatorOverRationalFunctionField
    elif is_derivation[0]:
        operator_class = UnivariateDifferentialOperatorOverRationalFunctionField
    else:
        operator_class = UnivariateOreOperator

    # Check whether this algebra already exists.
    global _list_of_ore_algebras
    alg = OreAlgebra_generic(base_ring, operator_class, gens, **kwargs)
    for a in _list_of_ore_algebras:
        if a == alg:
            return a

    # It's new. register it and return it. 
    _list_of_ore_algebras.append(alg)    
    return alg

_list_of_ore_algebras = []

class OreAlgebra_generic(sage.algebras.algebra.Algebra):

    def __init__(self, base_ring, operator_class, gens, **kargs):
        self._base_ring = base_ring
        self._operator_class = operator_class
        self._gens = gens

    # information extraction

    def __eq__(self, other):
        if not is_OreAlgebra(other):
            return False
        if not self.base_ring() == other.base_ring():
            return False
        if not self.ngens() == other.ngens():
            return False
        if not self._operator_class == other._operator_class:
            return False 
        base_gens = _collect_generators(self.base_ring())
        for i in xrange(self.ngens()):
            if not self.var(i) == other.var(i):
                return False
            self_sigma = self.sigma(i); other_sigma = other.sigma(i)
            if not all((self_sigma(x) == other_sigma(x)) for x in base_gens):
                return False
            self_delta = self.delta(i); other_delta = other.delta(i)
            if not all((self_delta(x) == other_delta(x)) for x in base_gens):
                return False
        return True        

    def base_ring(self):
        """
        Returns this algebra's base ring
        """
        return self._base_ring

    def is_integral_domain(self, proof = True):
        """
        Returns True because Ore algebras are always integral domains.
        """
        return True

    def is_noetherian(self):
        """
        Returns True because Ore algebras are always noetherian. 
        """
        return True

    def _coerce_map_from_(self, P):
        """
        If `P` is an Ore algebra, then a coercion from `P` to self is possible
        if the base ring of `P` admits coercion to the base ring of `self` and
        the generators of `P` form a subset of the generators of `self`.
        Corresponding generators are considered equal if they have the same
        name and the action of the associated `sigma` and `delta` agree on the
        generators of `P`'s base ring (including the base ring's base ring's
        generators and so on). 

        If `P` is not an Ore algebra, then a coercion from `P` to `self` is possible
        iff there is a coercion from `P` to the base ring of `self` or to the
        associated commutative algebra of `self`.
        """
        if is_OreAlgebra(P):
            out = self.base_ring()._coerce_map_from_(P.base_ring())
            if out is None or out is False:
                return False
            P_base_gens = _collect_generators(P.base_ring())
            # TEST: forall i exists j st "P.gen(i) == self.gen(j)" (with matching sigma and delta)
            for i in xrange(P.ngens()):
                found_match = False
                for j in xrange(self.ngens()):
                    if P.var(i) == self.var(j):
                        Ps = P.sigma(i); Ss = self.sigma(j)
                        Pd = P.delta(i); Sd = self.delta(j)
                        if all((Ps(x) == Ss(x)) for x in P_base_gens) and \
                           all((Pd(x) == Sd(x)) for x in P_base_gens):
                            found_match = True; break
                if not found_match:
                    return False
            return True            
        else: # P is not an Ore algebra
            out = self.base_ring()._coerce_map_from_(P)
            if out is not None and out is not False:
                return True
            out = self.associated_commutative_algebra()._coerce_map_from_(P)
            if out is not None and out is not False:
                return True
            return False

    def _sage_input_(self, sib, coerced):
        """
        """
        raise NotImplementedError

    def _is_valid_homomorphism_(self, codomain, im_gens):
        """
        """
        raise NotImplementedError

    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            pass
        h = self._cached_hash = hash((self.base_ring(),tuple(self._gens)))
        return h

    def _repr_(self):
        try:
            return self._cached_repr
        except AttributeError:
            pass
        if self.ngens() == 1:
            r = "Univariate"
        else:
            r = "Multivariate"
        r = r + " Ore algebra in "
        for x in self._gens:
            r = r + x[0] + ", "
        r = r[:-2r] + " over " + self.base_ring()._repr_()
        self._cached_repr = r
        return r

    def _latex_(self):
        try:
            return self._cached_latex
        except AttributeError:
            pass
        r = self.base_ring()._latex_() + "\\langle "
        for x in self._gens:
            r = r + x[0] + ", "
        r = r[:-2r] + "\\rangle "
        self._cached_latex = r
        return r

    def var(self, n=0):
        """
        Returns the name of the `n`th generator of this algebra.

        EXAMPLES::

           sage: A.<Dx> = OreAlgebra(QQ['x'].fraction_field(), 'Dx')
           sage: A.var()
           Dx
        """
        return self._gens[n][0]

    def _gen_to_idx(self, D):
        """
        If `D` is a generator of this algebra, given either as string or as an actual algebra element,
        return the index `n` such that `self.gen(n) == self(D)`.
        If `D` is already an integer, return `D` itself. 
        An IndexError is raised if `gen` is not a generator of this algebra.
        """
        if D in ZZ:
            return D
        D = str(D)
        for i in xrange(self.ngens()):
            if D == self.var(i):
                return i
        raise IndexError, "no such generator"

    def sigma(self, n=0):
        """
        Returns the sigma callable associated to the `n`th generator of this algebra.
        The generator can be specified by index (as integer), or by name (as string),
        or as algebra element.         

        EXAMPLES::

           sage: A.<Dx> = OreAlgebra(QQ['x'].fraction_field(), 'Dx')
           sage: A.sigma() 
           <function <lambda> at 0x1be2bf7c> # random
           sage: A.sigma(0)
           <function <lambda> at 0x1be2bf7c> # random
           sage: A.sigma("Dx")
           <function <lambda> at 0x1be2bf7c> # random
           sage: A.sigma(Dx)
           <function <lambda> at 0x1be2bf7c> # random
           
        """
        return self._gens[self._gen_to_idx(n)][1]

    def sigma_inverse(self, n=0):
        """
        Returns a callable object which represents the compositional inverse of the sigma
        callable associated to the `n`th generator of this algebra.
        The generator can be specified by index (as integer), or by name (as string),
        or as algebra element.

        The inverse can be constructed if `sigma` is such that it maps every generator `x` of
        the base ring (or the base ring's base ring, etc.) to a linear combination `a*x+b`
        where `a` and `b` belong to the base ring of the parent of `x`.

        If the method fails in constructing the inverse, it raises a ``ValueError``.

        EXAMPLES::

           sage: R.<x> = QQ['x']
           sage: A.<Sx> = OreAlgebra(R.fraction_field(), "Sx")
           sage: sigma = A.sigma()
           sage: sigma_inverse = A.sigma_inverse()
           sage: sigma(x)
           x + 1
           sage: sigma_inverse(x)
           x - 1
        
        """
        # possible generalization in case of rings with more generators: each generator is
        # mapped to a linear combination of the other generators with coefficients in the
        # base ring.
        
        if not hasattr(self, "_sigma_inverses"):
            self._sigma_inverses = [ None for D in self.gens() ]

        n = self._gen_to_idx(n)
        sig_inv = self._sigma_inverses[n]
        if sig_inv is not None:
            return sig_inv

        R = self.base_ring()
        sigma = self.sigma(n)
        sigma_inv_dict = {}
        for x in _collect_generators(R):
            sx = sigma(x)
            if sx == x:
                continue
            B = R
            while x not in B.gens():
                B = B.base_ring()
            if is_FractionField(B):
                B = B.ring()
            try:
                sx = B(sx) # may raise exception
                if sx.degree() > 1r:
                    raise ValueError # may raise exception
                b, a = sx[0r], sx[1r];
                sigma_inv_dict[x] = R( (x - b)/a ) # may raise exception
            except:
                raise ValueError, "unable to construct inverse of sigma"

        sigma_inv = _dict_to_sigma(R, sigma_inv_dict)
        self._sigma_inverses[n] = sigma_inv
        return sigma_inv

    def delta(self, n=0):
        """
        Returns the delta callable associated to the `n`th generator of this algebra. 
        The generator can be specified by index (as integer), or by name (as string),
        or as algebra element.         

        EXAMPLES::

           sage: A.<Dx> = OreAlgebra(QQ['x'].fraction_field(), 'Dx')
           sage: A.delta() 
           <function <lambda> at 0x1be2bf7c> # random
           sage: A.delta(0)
           <function <lambda> at 0x1be2bf7c> # random
           sage: A.delta("Dx")
           <function <lambda> at 0x1be2bf7c> # random
           sage: A.delta(Dx)
           <function <lambda> at 0x1be2bf7c> # random
           
        """
        return self._gens[self._gen_to_idx(n)][2]

    def variable_names(self):
        """
        Returns a tuple with the names (as strings) of the generators of this algebra. 

        EXAMPLES::

           sage: A.<Dx> = OreAlgebra(QQ['x'].fraction_field(), 'Dx')
           sage: A.variable_names() 
           ('Dx',)
        """
        return tuple(x[0] for x in self._gens)
        
    def variable_names_recursive(self, depth=sage.rings.infinity.infinity):
        r"""
        Returns the list of variable names of this and its base rings, as if
        it were a single multi-variate polynomial.
        
        EXAMPLES::
        
            sage: R = QQ['x']['y']['z']
            sage: R.variable_names_recursive()
            ('x', 'y', 'z')
            sage: R.variable_names_recursive(2)
            ('y', 'z')
        """
        if depth <= 0:
            return ()
        elif depth == 1:
            return self.variable_names()
        else:
            my_vars = self.variable_names()
            try:
                return self.base_ring().variable_names_recursive(depth - len(my_vars)) + my_vars
            except AttributeError:
                return my_vars
                
    def characteristic(self):
        """
        Return the characteristic of this Ore algebra, which is the
        same as that of its base ring.
        """
        return self.base_ring().characteristic()

    def gen(self, n=0):
        """
        Return the indeterminate generator(s) of this Ore algebra. 
        """
        if n < 0 or n >= self.ngens():
            raise IndexError("No such generator.")
        return self.gens()[n]

    def gens(self):
        """
        Return a list of generators of this Ore algebra. 
        """
        return self.gens_dict().values()

    def gens_dict(self):
        """
        Returns a dictionary whose keys are the variable names of this
        Algebra as strings and whose values are the corresponding
        generators.
        """
        try:
            return self._gen_dict.copy()
        except AttributeError:
            pass
        d = {}
        for (x, _, _) in self._gens:
            d[x] = self(x)
        self._gen_dict = d.copy()
        return d

    def is_finite(self):
        """
        Return False since Ore algebras are not finite (unless the base ring is 0).
        """
        R = self.base_ring()
        if R.is_finite() and R.order() == 1:
            return True
        return False

    def is_exact(self):
        """
        This algebra is exact iff its base ring is
        """
        return self.base_ring().is_exact()

    def is_field(self, proof = True):
        """
        Returns False since Ore algebras are not fields (unless they have 0 generators and the base ring is a field)
        """
        return self.ngens() == 0 and self.base_ring().is_field()
        
    def krull_dimension(self):
        """
        Returns the Krull dimension of this algebra, which is the Krull dimension of the base ring
        plus the number of generators of this algebra. 
        """
        return self.base_ring().krull_dimension() + self.ngens()
        
    def ngens(self):
        """
        Return the number of generators of this Ore algebra
        """
        return len(self._gens)

    # generation of elements
        
    def _element_constructor_(self, *args, **kwds):
        """
        Create a new element based on the given arguments. 
        """
        return self._operator_class(self, *args, **kwds)
        
    def random_element(self, *args, **kwds):
        """
        Return a random operator. The random operator is constructed by coercing a random element
        of the associated commutative algebra to an element of this algebra. 
        """
        return self._element_constructor_(self.associated_commutative_algebra().random_element(*args, **kwds))

    # generation of related parent objects

    def associated_commutative_algebra(self):
        """
        Returns a polynomial ring with the same base ring as this algebra and whose generators
        have the same name as this Ore algebra's generators.

        ::

           sage: R.<x> = QQ['x']
           sage: A = OreAlgebra(R.fraction_field(), "Dx")
           sage: A
           Univariate Ore algebra in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field
           sage: A.associated_commutative_algebra()
           Univariate Polynomial Ring in Dx over Fraction Field of Univariate Polynomial Ring in x over Rational Field
        
        """
        try:
            return self._commutative_ring
        except AttributeError:
            pass
        R = self._commutative_ring = PolynomialRing(self.base_ring(), self.variable_names())
        return R

    def base_extend(self, R):
        """
        Creates the Ore algebra obtained from `self` by replacing the base ring by `R`
        """
        return self.change_ring(R)

    def change_ring(self, R):
        """
        Creates the Ore algebra obtained from `self` by replacing the base ring by `R`
        """
        return OreAlgebra(R, *self._gens)

    def change_var(self, var, n=0):
        """
        Creates the Ore algebra obtained from `self` by renaming the `n`th generator to `var`
        """ 
        n = self._gen_to_idx(n)
        if n < 0 or n >= self.ngens():
            raise IndexError("No such generator.")
        return self.change_var_sigma_delta(var, self._gens[n][1], self._gens[n][2], n)

    def change_sigma(self, sigma, n=0):
        """
        Creates the Ore algebra obtained from `self` by replacing the homomorphism associated to the
        `n`th generator to `sigma`, which may be a callable or a dictionary.
        """
        n = self._gen_to_idx(n)
        if n < 0 or n >= self.ngens():
            raise IndexError("No such generator.")
        return self.change_var_sigma_delta(self._gens[n][0], sigma, self._gens[n][2], n)

    def change_delta(self, delta, n=0):
        """
        Creates the Ore algebra obtained from `self` by replacing the skew-derivation associated to the
        `n`th generator to `delta`, which may be a callable or a dictionary.
        """
        n = self._gen_to_idx(n)
        if n < 0 or n >= self.ngens():
            raise IndexError("No such generator.")
        return self.change_var_sigma_delta(self._gens[n][0], self._gens[n][1], delta, n)

    def change_var_sigma_delta(self, var, sigma, delta, n=0):
        """
        Creates the Ore algebra obtained from `self` by replacing the
        `n`th generator and its associated homomorphism and skew-derivation
        by `var`, `sigma`, and `delta`, respectively.
        The maps `sigma` and `delta` may be specified as callables or dictionaries.
        """
        n = self._gen_to_idx(n)
        if n < 0 or n >= self.ngens():
            raise IndexError("No such generator.")

        gens = list(self._gens)
        if type(sigma) == dict:
            sigma = _dict_to_sigma(self.base_ring(), sigma)
        if type(delta) == dict:
            delta = _dict_to_delta(self.base_ring(), delta, sigma)

        gens[n] = (var, sigma, delta)
        return OreAlgebra(self.base_ring(), *gens)
        
##########################################################################################################

def guess_rec(data, n, S):
    """
    """
    raise NotImplementedError

def guess_deq(data, x, D):
    """
    """
    raise NotImplementedError
