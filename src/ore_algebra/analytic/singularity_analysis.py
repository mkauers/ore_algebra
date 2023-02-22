# coding: utf-8 - vim: tw=80
r"""
Bounds on sequences by singularity analysis

AUTHORS:

- Ruiwen Dong: initial version
- Marc Mezzarobba

EXAMPLES::

    sage: from ore_algebra import (OreAlgebra, DFiniteFunctionRing,
    ....:         UnivariateDFiniteFunction)
    sage: from ore_algebra.analytic.singularity_analysis import (
    ....:         bound_coefficients, check_seq_bound)
    sage: from ore_algebra.analytic.singularity_analysis import eval_bound

    sage: Pols_z.<z> = PolynomialRing(QQ)
    sage: Diffops.<Dz> = OreAlgebra(Pols_z)
    sage: DFR = DFiniteFunctionRing(Diffops)

A trivial example so that the warning does not appear with later tests::

    sage: bound_coefficients((z-1)*Dz+1, [1], order=2)
    doctest:...: FutureWarning: This class/method/function is marked as
    experimental. ...
    1.000...*(1.000... + B([...]*(...)^n, n >= ...))

Membrane example::

    sage: seqini = [72, 1932, 31248, 790101/2, 17208645/4, 338898609/8, 1551478257/4]
    sage: deq = (8388593*z^2*(3*z^4 - 164*z^3 + 370*z^2 - 164*z + 3)*(z + 1)^2*(z^2 - 6*z + 1)^2*(z - 1)^3*Dz^3
    ....: + 8388593*z*(z + 1)*(z^2 - 6*z + 1)*(66*z^8 - 3943*z^7 + 18981*z^6 - 16759*z^5 - 30383*z^4 + 47123*z^3 - 17577*z^2 + 971*z - 15)*(z - 1)^2*Dz^2
    ....: + 16777186*(z - 1)*(210*z^12 - 13761*z^11 + 101088*z^10 - 178437*z^9 - 248334*z^8 + 930590*z^7 - 446064*z^6 - 694834*z^5 + 794998*z^4 - 267421*z^3 + 24144*z^2 - 649*z + 6)*Dz
    ....: + 6341776308*z^12 - 427012938072*z^11 + 2435594423178*z^10 - 2400915979716*z^9 - 10724094731502*z^8 + 26272536406048*z^7 - 8496738740956*z^6 - 30570113263064*z^5 + 39394376229112*z^4 - 19173572139496*z^3 + 3825886272626*z^2 - 170758199108*z + 2701126946)

    sage: asy = bound_coefficients(deq, seqini, order=5, prec=200) # long time (5 s)
    sage: asy # long time
    1.000...*5.828427124746190?^n*(([8.0719562915...] + [+/- ...]*I)*n^3*log(n)
    + ([1.3714048996...82527...] + [+/- ...]*I)*n^3
    + ([50.509130873...07157...] + [+/- ...]*I)*n^2*log(n)
    + ([29.698551451...84781...] + [+/- ...]*I)*n^2
    + ...
    + ([-0.283779713...91869...] + [+/- ...]*I)*n^(-1)*log(n)
    + ([35.493938347...65227...] + [+/- ...]*I)*n^(-1)
    + B([...]*n^(-2)*log(n)^2, n >= ...))

    sage: ref = UnivariateDFiniteFunction(DFR, deq, seqini)
    sage: check_seq_bound(asy.expand(), ref, list(range(100)) + list(range(200, 230)) + [1000]) # long time

The option ``ignore_exponentially_small_term`` computes an enclosure up to an
error term that is exponentially smaller that the last computed term as n tends
to infinity. This is significantly faster than computing a complete bound::

    sage: bound_coefficients(deq, seqini, ignore_exponentially_small_term=True)
    1.000000000000000*5.828427124746190?^n*(([8.07195629151...]...)*n^3*log(n)
    + ([1.37140489963...]...)*n^3
    + ([50.5091308739...]...)*n^2*log(n) + ([29.6985514518...]...)*n^2
    + ([106.3602546895...]...)*n*log(n) + ([118.7598389584...]...)*n
    + B([...]*log(n)^2, n >= ...)
    + O(0.60...?^n))

Algebraic example::

    sage: deq = (4*z^4 - 4*z^3 + z^2 - 2*z + 1)*Dz + (-4*z^3 + 4*z^2 - z - 1)
    sage: asy = bound_coefficients(deq, [1], order=5) # long time (13 s)
    sage: asy # long time
    1.000...*2^n*([0.564189583547...]*n^(-1/2) + [-0.105785546915...]*n^(-3/2)
    + [-0.117906807499...]*n^(-5/2) + [-0.375001499318...]*n^(-7/2)
    + [-1.255580304110...]*n^(-9/2) + B([...]*n^(-11/2), n >= ...))

    sage: ref = UnivariateDFiniteFunction(DFR, deq, [1])
    sage: check_seq_bound(asy.expand(), ref, list(range(100)) + list(range(200, 230)) + [1000]) # long time

Diagonal example::

    sage: seqini = [1, -3, 9, -3, -279, 2997]
    sage: deq = (z^2*(81*z^2 + 14*z + 1)*Dz^3 + 3*z*(162*z^2 + 21*z + 1)*Dz^2
    ....:        + (21*z + 1)*(27*z + 1)*Dz + 3*(27*z + 1))
    sage: asy = bound_coefficients(deq, seqini, order=2) # long time (3.5 s)
    sage: asy # long time
    1.000...*9.000...?^n*(([0.30660...] + [0.14643...]*I)*(e^(I*arg(-0.77777...? + 0.62853...?*I)))^n*n^(-3/2)
    + ([-0.26554...] + [-0.03529...]*I)*(e^(I*arg(-0.77777...? + 0.62853...?*I)))^n*n^(-5/2)
    + ([0.30660...] + [-0.14643...]*I)*(e^(I*arg(-0.77777...? - 0.62853...?*I)))^n*n^(-3/2)
    + ([-0.26554...] + [0.03529...]*I)*(e^(I*arg(-0.77777...? - 0.62853...?*I)))^n*n^(-5/2)
    + B([...]*n^(-7/2), n >= ...))

    sage: ref = UnivariateDFiniteFunction(DFR, deq, seqini)
    sage: # not supported
    sage: #check_seq_bound(asy.expand(), ref, range(1000))
    sage: asy = bound_coefficients(deq, seqini, order=2, output='list') # long time (3.5 s)
    sage: all(eval_bound(asy[1], j).contains_exact(ref[j]) for j in range(asy[0], 150)) # long time
    True

Lattice path example from the paper::

    sage: dop = (z^2*(4*z - 1)*(4*z + 1)*Dz^3 + 2*z*(4*z+1)*(16*z-3)*Dz^2
    ....:        + 2*(112*z^2 + 14*z - 3)*Dz + 4*(16*z + 3))
    sage: bound_coefficients(dop, [1, 2, 6], order=3) # long time
    1.000000000000000*4^n*(([1.273239544735...] + [+/- ...]*I)*n^(-1)
    + ([-1.909859317102...] + [+/- ...]*I)*n^(-2)
    + ([3.023943918746...] + [+/- ...]*I)*n^(-3)
    + B([3432.0...]*n^(-4)*log(n), n >= 9)
    + [0.3183098861837...]*(e^(I*arg(-1)))^n*n^(-3))
    sage: _, bound = bound_coefficients(dop, [1, 2, 6], order=6, output='list') # long time
    sage: my_n = RBF(1000000); eval_bound(bound, my_n).real()/4^my_n # long time
    [1.27323763487919e-6 +/- ...]

Example from Y. Baryshnikov, S. Melczer, R. Pemantle, and A. Straub, "Diagonal
asymptotics for symmetric rational functions via ACSV", arXiv:1804.10929::

    sage: c = ZZ['c,z'].gen(0)
    sage: dop = ((z^2*(c^4*z^4 + 4*c^3*z^3 + 6*c^2*z^2 + 4*c*z - 256*z + 1)*(3*c*z - 1)^2*Dz^3
    ....:        + 3*z*(3*c*z - 1)*(6*c^5*z^5 + 15*c^4*z^4 + 8*c^3*z^3 - 6*c^2*z^2 - 384*c*z^2 - 6*c*z + 384*z - 1)*Dz^2
    ....:        + (c*z + 1)*(63*c^5*z^5 - 3*c^4*z^4 - 66*c^3*z^3 + 18*c^2*z^2 + 720*c*z^2 + 19*c*z - 816*z + 1)*Dz
    ....:        + (9*c^6*z^5 - 3*c^5*z^4 - 6*c^4*z^3 + 18*c^3*z^2 - 360*c^2*z^2 + 13*c^2*z - 384*c*z + c - 24)))

    sage: dop28 = Diffops([p(c=28) for p in dop])
    sage: bound_coefficients(dop28, [1, -4, -56], order=2, n0=50) # long time
    1.000000000000000*83.3254624124113?^n*(([0.0311212622056357...] + [-0.0345183803114027...]*I)*(e^(I*arg(0.9521089229254642? - 0.3057590536447162?*I)))^n*n^(-3/2)
    + ([-0.050269964085834...] + [-0.0298161277530909...]*I)*(e^(I*arg(0.9521089229254642? - 0.3057590536447162?*I)))^n*n^(-5/2)
    + ([0.0311212622056357...] + [0.0345183803114027...]*I)*(e^(I*arg(0.9521089229254642? + 0.3057590536447162?*I)))^n*n^(-3/2)
    + ([-0.050269964085834...] + [0.0298161277530909...]*I)*(e^(I*arg(0.9521089229254642? + 0.3057590536447162?*I)))^n*n^(-5/2)
    + B([6.10...]*n^(-7/2), n >= 50))

    sage: dop27 = Diffops([p(c=27) for p in dop]).primitive_part()
    sage: bound_coefficients(dop27,  [1, -3, 9], order=2, n0=50) # long time
    1.000000000000000*9.00000000000000?^n*(([0.306608607103967...] + [0.146433894558384...]*I)*(e^(I*arg(-0.7777777777777777? + 0.6285393610547089?*I)))^n*n^(-3/2)
    + ([-0.265549842772215...] + [-0.035298693487940...]*I)*(e^(I*arg(-0.7777777777777777? + 0.6285393610547089?*I)))^n*n^(-5/2)
    + ([0.306608607103967...] + [-0.146433894558384...]*I)*(e^(I*arg(-0.7777777777777777? - 0.6285393610547089?*I)))^n*n^(-3/2)
    + ([-0.265549842772215...] + [0.035298693487940...]*I)*(e^(I*arg(-0.7777777777777777? - 0.6285393610547089?*I)))^n*n^(-5/2)
    + B([50.0...]*n^(-7/2), n >= 50))

    sage: dop26 = Diffops([p(c=26) for p in dop])
    sage: bound_coefficients(dop26, [1, -2, 76], order=2, n0=50) # long time
    1.000000000000000*108.1021465879489?^n*(([0.0484997667050581...] + [...]*I)*n^(-3/2)
    + ([-0.068160009777454...] + [...]*I)*n^(-5/2)
    + B([8.40...]*n^(-7/2), n >= 50))

Complex exponents example::

    sage: deq = (z-2)^2*Dz^2 + z*(z-2)*Dz + 1
    sage: seqini = [1, 2, -1/8]
    sage: asy = bound_coefficients(deq, seqini, order=3) # long time (2 s)
    sage: asy # long time
    1.000...*(1/2)^n*(([1.124337...] + [0.462219...]*I)*n^(-0.500000...? + 0.866025...?*I)
    + ([1.124337...] + [-0.462219...]*I)*n^(-0.500000...? - 0.866025...?*I)
    + ([-0.400293...] + [0.973704...]*I)*n^(-1.500000...? + 0.866025...?*I)
    + ([-0.400293...] + [-0.973704...]*I)*n^(-1.500000...? - 0.866025...?*I)
    + ([0.451623...] + [-0.356367...]*I)*n^(-2.500000...? + 0.866025...?*I)
    + ([0.451623...] + [0.356367...]*I)*n^(-2.500000...? - 0.866025...?*I)
    + B([...]*n^(-7/2), n >= ...))

    sage: ref = UnivariateDFiniteFunction(DFR, deq, seqini)
    sage: # complex powers of n are not supported
    sage: #check_seq_bound(asy.expand(), ref, range(1000))
    sage: asy = bound_coefficients(deq, seqini, order=3, output='list') # long time (2 s)
    sage: all(eval_bound(asy[1], j).contains_exact(ref[j]) for j in range(asy[0], 150)) # long time
    True

We can often detect when the expansion in powers of n terminates, but currently
do not compute the exponentially smaller terms that follow in this case::

    sage: dop = ((z-1)*Dz + 1).lclm(((z-2)*Dz)^2)
    sage: dop((1/(1-z) + log(1/(1-z/2)))).simplify_full()
    0
    sage: seqini = list((1/(1-z) + log(1/(1-z/2))).series(z, 3).truncate().polynomial(QQ))
    sage: bound_coefficients(dop, seqini) # long time
    1.000...*([1.000...] + B([...]*(4/7)^n, n >= ...))

    sage: bound_coefficients((z-1)*Dz-z+2, [1]) # exp(z)/(1-z) # long time
    1.000...*([2.718281828459...] + B([...]*(4/7)^n, n >= ...))

It is possible, however, to use the option ``known_analytic`` to indicate that
the initial terms given on input define a solution that is analytic at a
particular singular point of the differential operator::

    sage: seqini = list(log(1/(1-z/2)).series(z, 3).truncate().polynomial(QQ))
    sage: bound_coefficients(dop, seqini) # long time
    1.000...*([+/-...] + B([...]*(4/7)^n, n >= ...))
    sage: bound_coefficients(dop, seqini, known_analytic=[1]) # long time
    1.000...*(1/2)^n*(([1.000...]...)*n^(-1) + B([...]*n^(-4)*log(n), n >= ...))

An example from M. Kauers and V. Pillwein, "When can we detect that a P-finite
sequence is positive?", ISSAC 2010, where the generating function of interest is
analytic at a non-apparent dominant singular point of the operator::

    sage: Pols_n.<n> = QQ[]
    sage: Recops.<Sn> = OreAlgebra(Pols_n)
    sage: dop = ((n+3)^2*Sn^2-(n+2)*(3*n+11)/2*Sn+(n+4)*(n+1)/2).to_D(Diffops)

    sage: bound_coefficients(dop, [1,1/4], order=2, n0=50, prec=1000)
    1.00...*([...] + [...]*I + B(2975.8...*(4/7)^n, n >= 50))

    sage: bound_coefficients(dop, [1,1/4], order=2, n0=50, known_analytic=[0,1])
    1.00...*(1/2)^n*(([1.00...] + [...]*I)*n^(-1)
    + ([-1.00...] + [...]*I)*n^(-2)
    + B([65.1...]*n^(-3)*log(n), n >= 50))

Singularities and exponents with multiple scales. Note that Sage is not always
able to correctly order the terms yet::

    sage: dop = (((z-1)*Dz)^2*((z-1)*Dz-1/3)*((z-1)*Dz-4/3)).lclm((z-1/2)*Dz-1).lclm((z+1)*Dz+1/2)
    sage: dop(log(1/(1-z))).simplify_full()
    0
    sage: dop((1-SR(z))^(1/3)).simplify_full()
    0
    sage: dop((1-SR(z))^(4/3)).simplify_full()
    0
    sage: dop(1/sqrt(1+z)).simplify_full()
    0

    sage: bound_coefficients(dop, [0, 1, 1/2, 1/3, 1/4, 1/5], order=3,
    ....:         ignore_exponentially_small_term=True)
    1.000000000000000*(([1.000000000000...] + [+/-...]*I)*n^(-1)
    + ([+/-...] + [+/-...]*I)*n^(-4/3)
    + ([+/-...] + [+/-...]*I)*n^(-7/3)
    + ([+/-...] + [+/-...]*I)*n^(-10/3)
    + B([...]*n^(-7/2), n >= ...)
    + O(...)
    + [+/-...]*(e^(I*arg(-1)))^n*n^(-1/2)
    + [+/-...]*(e^(I*arg(-1)))^n*n^(-3/2)
    + [+/-...]*(e^(I*arg(-1)))^n*n^(-5/2))
    sage: assert 10/3 < 7/2 == 1/2 + 3 < 11/3

    sage: bound_coefficients(dop, [0, 1, 1/2, 1/3, 1/4, 1/5], order=3,
    ....:         known_analytic=[-1], ignore_exponentially_small_term=True)
    1.000000000000000*(([1.000000000000...] + [+/-...]*I)*n^(-1)
    + ([+/-...] + [+/-...]*I)*n^(-4/3)
    + ([+/-...] + [+/-...]*I)*n^(-7/3)
    + ([+/-...] + [+/-...]*I)*n^(-10/3)
    + B([...]*n^(-4)*log(n), n >= 10) + O(...))

    sage: bound_coefficients(dop, [1, -1/3, -1/9, -5/81, -10/243, -22/729],
    ....:         known_analytic=[-1], ignore_exponentially_small_term=True)
    1.000000000000000*(([+/- ...]...)*n^(-1)
    + ([-0.2461627038738...]...)*n^(-4/3)
    + ([-0.0547028230830...]...)*n^(-7/3)
    + ([-0.02127332008786...]...)*n^(-10/3)
    + B([...]*n^(-4)*log(n), n >= 10) + O(...n))

    sage: bound_coefficients(dop, [1, -4/3, 2/9, 4/81, 5/243, 8/729], order=3,
    ....:         known_analytic=[-1], ignore_exponentially_small_term=True)
    1.000000000000000*(([+/-...] + [+/-...]*I)*n^(-1)
    + ([+/-...] + [+/-...]*I)*n^(-4/3)
    + ([0.328216938498...] + [+/-...]*I)*n^(-7/3)
    + ([0.510559682108...] + [+/-...]*I)*n^(-10/3)
    + B([...]*n^(-4)*log(n), n >= 10) + O(...))

    sage: bound_coefficients(dop, [1, -1/2, 3/8, -5/16, 35/128, -63/256],
    ....:         order=3, ignore_exponentially_small_term=True)
    1.000000000000000*(([+/-...] + [+/-...]*I)*n^(-1)
    + ([+/-...] + [+/-...]*I)*n^(-4/3)
    + ([+/-...] + [+/-...]*I)*n^(-7/3)
    + ([+/-...] + [+/-...]*I)*n^(-10/3)
    + B([...]*n^(-7/2), n >= 10) + O(...)
    + [0.5641895835477...]*(e^(I*arg(-1)))^n*n^(-1/2)
    + [-0.07052369794346...]*(e^(I*arg(-1)))^n*n^(-3/2)
    + [0.00440773112146...]*(e^(I*arg(-1)))^n*n^(-5/2))

    sage: bound_coefficients(dop, [1, -1/2, 3/8, -5/16, 35/128, -63/256],
    ....:         order=3, ignore_exponentially_small_term=True,
    ....:         known_analytic=[1])
    1.000000000000000*(e^(I*arg(-1)))^n*([0.5641895835477...]*n^(-1/2)
    + [-0.07052369794346...]*n^(-3/2)
    + [0.00440773112146...]*n^(-5/2)
    + B([...]*n^(-7/2), n >= 10) + O(...))

    sage: # 3*log(1/(1-z)) + 1/sqrt(1+z)
    sage: bound_coefficients(dop, [1, 5/2, 15/8, 11/16, 131/128, 453/1280],
    ....:         order=3, ignore_exponentially_small_term=True)
    1.000000000000000*(([3.000000000000...] + [+/-...]*I)*n^(-1)
    + ([+/-...] + [+/-...]*I)*n^(-4/3)
    + ([+/-...] + [+/-...]*I)*n^(-7/3)
    + ([+/-...] + [+/-...]*I)*n^(-10/3)
    + B([...]*n^(-7/2), n >= 10) + O(...)
    + [0.5641895835477...]*(e^(I*arg(-1)))^n*n^(-1/2)
    + [-0.07052369794346...]*(e^(I*arg(-1)))^n*n^(-3/2)
    + [0.00440773112146...]*(e^(I*arg(-1)))^n*n^(-5/2))

TESTS:

Standard asymptotic expansions (algebro-logarithmic singularities)::

    sage: from ore_algebra.analytic.singularity_analysis import test_monomial

General exponents::

    sage: test_monomial(alpha=3/2, beta=0)
    (True, ... + B([...]*n^(-7/2), n >= ...) + O(...^n)))
    sage: test_monomial(alpha=1/2, beta=0)
    (True, ... + B([...]*n^(-9/2), n >= ...) + O(...^n)))
    sage: test_monomial(alpha=1/3, beta=0)
    (True, ... + B([...]*n^(-14/3), n >= ...) + O(...^n)))
    sage: test_monomial(alpha=-1/3, beta=0)
    (True, ... + B([...]*n^(-16/3), n >= ...) + O(...^n)))
    sage: test_monomial(alpha=-1/2, beta=0)
    (True, ... + B([...]*n^(-11/2), n >= ...) + O(...^n)))
    sage: test_monomial(alpha=-3/2, beta=0)
    (True, ... + B([...]*n^(-13/2), n >= ...) + O(...^n)))

::

    sage: test_monomial(alpha=-3/2, beta=1)
    (True, ... + B([...]*n^(-13/2)*log(n), n >= ...) + O(...^n)))
    sage: test_monomial(alpha=2/3, beta=1)
    (True, ... + B([...]*n^(-13/3)*log(n), n >= ...) + O(...^n)))

Here our code disagrees with ``asympotic_expansions.SingularityAnalysis``, but
it does agree with ``gdev``, and numerical tests seem to confirm that the bug is
in ``SingularityAnalysis`` (see Sage bug #33994) ::

    sage: test_monomial(alpha=-3/2, beta=2) # long time
    (...,
    1.000...*(([0.4231421876608...]...)*n^(-5/2)*log(n)^2
    + ([-0.595070478381...]...)*n^(-5/2)*log(n)
    + ([-3.75954106470...]...)*n^(-5/2)
    + ([0.7933916018640...]...)*n^(-7/2)*log(n)^2
    + ([-2.808325897608...]...)*n^(-7/2)*log(n)
    + ([-5.43585635190...]...)*n^(-7/2)
    + ([1.272732361323...]...)*n^(-9/2)*log(n)^2
    + ([-6.62073373238...]...)*n^(-9/2)*log(n)
    + ([-4.57888923343...]...)*n^(-9/2)
    + ([1.952487145212...]...)*n^(-11/2)*log(n)^2
    + ([-13.05989942809...]...)*n^(-11/2)*log(n)
    + ([0.9099324035...]...)*n^(-11/2)
    + B([...]*n^(-13/2)*log(n)^2, n >= ...) + O((...)^n)))

    sage: test_monomial(alpha=3/2, beta=2) # long time
    (...,
    1.000...*(([1.1283791670955...]...)*n^(1/2)*log(n)^2
    + ([-0.0823490528905...]...)*n^(1/2)*log(n)
    + ([-1.05330887105...]...)*n^(1/2)
    + ([0.4231421876608...]...)*n^(-1/2)*log(n)^2
    + ([2.225877439357...]...)*n^(-1/2)*log(n)
    + ([0.651039287560...]...)*n^(-1/2)
    + ([-0.06170823570053...]...)*n^(-3/2)*log(n)^2
    + ([-0.183559730685...]...)*n^(-3/2)*log(n)
    + ([0.487607437620...]...)*n^(-3/2)
    + ([0.00991739502330...]...)*n^(-5/2)*log(n)^2
    + ([0.052169002484...]...)*n^(-5/2)*log(n)
    + ([-0.07289588912...]...)*n^(-5/2)
    + B([...]*n^(-7/2)*log(n)^2, n >= ...) + O((...)^n)))

Similar situation here. ``gdev`` gives the opposite sign, but, as far as I
understand, this is because it uses a nonstandard branch of the logarithm::

    sage: test_monomial(alpha=3/2, beta=3) # long time
    (...,
    1.000...*(([1.128379167095...]...)*n^(1/2)*log(n)^3
    + ([-0.1235235793358...]...)*n^(1/2)*log(n)^2
    + ([-3.15992661315...]...)*n^(1/2)*log(n)
    + ([1.050612156263...]...)*n^(1/2)
    + ([0.4231421876608...]...)*n^(-1/2)*log(n)^3
    + ([3.338816159035...]...)*n^(-1/2)*log(n)^2
    + ([1.95311786268...]...)*n^(-1/2)*log(n)
    + ([-2.88947063389...]...)*n^(-1/2)
    + ([-0.0617082357005...]...)*n^(-3/2)*log(n)^3
    + ([-0.275339596028...]...)*n^(-3/2)*log(n)^2
    + ([1.46282231286...]...)*n^(-3/2)*log(n)
    + ([2.41630885740...]...)*n^(-3/2)
    + ([0.00991739502330...]...)*n^(-5/2)*log(n)^3
    + ([0.07825350372...]...)*n^(-5/2)*log(n)^2
    + ([-0.21868766738...]...)*n^(-5/2)*log(n)
    + ([-0.76330866778...]...)*n^(-5/2)
    + B([...]*n^(-7/2)*log(n)^3, n >= ...) + O((...)^n)))

Integer exponents. In simple cases, we are able to return error terms of the
correct order of magnitude even when some of the series terminate::

    sage: test_monomial(alpha=1, beta=0)
    (True,
    1.000...*(1.000... + O((...)^n)))
    sage: test_monomial(alpha=2, beta=0)
    (True,
    1.000...*(1.000...*n + 1.000... + O((...)^n)))

::

    sage: test_monomial(alpha=10, beta=0)
    (True,
    1.000...*([2.75573192239...e-6 +/- ...]*n^9
    + [0.0001240079365079...]*n^8 + [0.00239748677248...]*n^7
    + [0.0260416666666...]*n^6 + B([...]*n^5, n >= ...) + O((...)^n)))
    sage: test_monomial(alpha=10, beta=0, big_circle=True) # long time
    (True,
    1.000...*([2.75573192239...e-6 +/- ...]*n^9
    + [0.0001240079365079...]*n^8 + [0.00239748677248...]*n^7
    + [0.0260416666666...]*n^6 + B([...]*n^5, n >= ...)))

::

    sage: test_monomial(alpha=1, beta=3) # long time
    (True, ... + B([...]*n^(-4)*log(n)^2, n >= ...) + O((...)^n)))
    sage: test_monomial(alpha=3, beta=2, order=2)
    (True, ... + B([...]*log(n)^2, n >= ...) + O((...)^n)))
    sage: test_monomial(alpha=3, beta=2, order=3) # long time
    (True, ... + B([...]*n^(-1)*log(n), n >= ...) + O((...)^n)))

::

    sage: test_monomial(alpha=0, beta=1)
    (True, 1.000...*([1.000...]*n^(-1) + O((...)^n)))
    sage: test_monomial(alpha=0, beta=2) # long time
    (True,
    1.000...*(([2.000...]...)*n^(-1)*log(n) + ...
    + B([...]*n^(-5), n >= ...) + O((...)^n)))

::

    sage: test_monomial(alpha=-1, beta=2) # long time
    (True, ... + B([...]*n^(-6)*log(n), n >= ...) + O((...)^n)))

Algebraic exponents::

    sage: sqrt2 = QuadraticField(2).gen()
    sage: test_monomial(alpha=sqrt2, beta=0)
    (True, ... + B([...]*n^(-3.585786437626905?), n >= ...) + O((...)^n)))
    sage: test_monomial(alpha=sqrt2, beta=2, compare=False, order=2)
    (None,
    1.000...*(([1.127927979999...]...)*n^(0.4142135623730951?)*log(n)^2
    + ([0.105821360001...]...)*n^(0.4142135623730951?)*log(n)
    + ([-1.13839359105...]...)*n^(0.4142135623730951?)
    + ([0.330362456651...]...)*n^(-0.5857864376269049?)*log(n)^2
    + ([2.09332847214...]...)*n^(-0.5857864376269049?)*log(n)
    + ([0.89124353934...]...)*n^(-0.5857864376269049?)
    + B([...]*n^(-1.585786437626905?)*log(n)^2, n >= 10)
    + O((...)^n)))

::

    sage: test_monomial(alpha=I, beta=2, compare=False, order=2)
    (None,
    1.000...*(([-0.569607641036...] + [1.830744396590...]*I)*n^(I - 1)*log(n)^2
    + ([7.71154584360...] + [2.019217722531...]*I)*n^(I - 1)*log(n)
    + ([-0.02823947920...] + [-7.57203529270...]*I)*n^(I - 1)
    + ([1.200176018813...] + [-0.630568377776...]*I)*n^(I - 2)*log(n)^2
    + ([-5.93804521268...] + [-7.83534146173...]*I)*n^(I - 2)*log(n)
    + ([-10.21649619212...] + [12.33281876488...]*I)*n^(I - 2)
    + B([...]*n^(-3)*log(n)^2, n >= ... + O((...)^n)))

::

    sage: dop = (falling_factorial(z - 1, 4) + z)*Dz + 1
    sage: asy = bound_coefficients(dop, [1], order=3, output='list') # long time
    sage: asy # long time
    (10,
    [(1.272845131361936? - 0.2284421234572305?*I,
    [([0.0990004637690...] + [0.2140635237159...]*I)*n^(-0.9424090224859841? + 0.3302957595950048?*I),
    ([-0.150112291716...] + [-0.201113316532...]*I)*n^(-1.942409022485985? + 0.3302957595950048?*I),
    ([-0.310200559727...] + [0.623654228068...]*I)*n^(-2.942409022485984? + 0.3302957595950048?*I),
    ([+/- ...] + [+/- ...]*I)/n^3.942409022485984?]),
    (1.272845131361936? + 0.2284421234572305?*I,
    [([0.0990004637690...] + [-0.2140635237159...]*I)*n^(-0.9424090224859841? - 0.3302957595950048?*I),
    ([-0.150112291716...] + [0.201113316532...]*I)*n^(-1.942409022485985? - 0.3302957595950048?*I),
    ([-0.310200559727...] + [-0.623654228068...]*I)*n^(-2.942409022485984? - 0.3302957595950048?*I),
    ([+/- ...] + [+/- ...]*I)/n^3.942409022485984?])])

    sage: ref = UnivariateDFiniteFunction(DFR, dop, [1]).expand(200) # long time
    sage: all(eval_bound(asy[1], j).contains_exact(ref[j]) for j in range(asy[0], len(ref))) # long time
    True

Varying the position of the singularity::

    sage: test_monomial(zeta=2, alpha=1/2, beta=1)
    (True, ... + B([...]*n^(-9/2)*log(n), n >= ...) + O((...)^n)))

::

    sage: test_monomial(zeta=1+I, alpha=1/2, beta=1, order=2, compare=False)
    (None,
    1.000...*(e^(I*arg(-1/2*I + 1/2)))^n*0.707...?^n*(([0.564...]...)*n^(-1/2)*log(n)
    + ([1.107791903872...]...)*n^(-1/2)
    + ([-0.07052369794346...]...)*n^(-3/2)*log(n)
    + ([-0.1384739879841...]...)*n^(-3/2)
    + B([...]*n^(-5/2)*log(n), n >= 8) + O(...^n)))

::

    sage: test_monomial(zeta=1+I, alpha=I/3+1, beta=1, order=2, compare=False) # long time
    (None,
    1.000...*(e^(I*arg(-1/2*I + 1/2)))^n*0.7071067811865475?^n*(([1.074944392622...] + [0.193783909890...]*I)*n^(1/3*I)*log(n)
    + ([0.588542135640...] + [-0.46215768679...]*I)*n^(1/3*I)
    + ([-0.092016451238...] + [0.1683916259986...]*I)*n^(1/3*I - 1)*log(n)
    + ([0.517207055500...] + [0.578972535470...]*I)*n^(1/3*I - 1)
    + B([...]*n^(-2)*log(n), n >= ...) + O(...^n)))

Apparent singularities::

    sage: dop = ((z-1)*Dz - 1).lclm((z+1)*Dz + 2)
    sage: dop((1+z)^(-2))
    0
    sage: bound_coefficients(dop, [1, -2], order=5, ignore_exponentially_small_term=True)
    1.000...*(e^(I*arg(-1)))^n*([1.000...]*n + [1.000...] + O(...^n))

    sage: dop = ((z-1)*Dz - 1).lclm((z-2)*Dz + 2)
    sage: bound_coefficients(dop, [1, 1], order=5, ignore_exponentially_small_term=True)
    1.000...*(1/2)^n*(([1.000...]...)*n + [1.000...]... + O(...^n))

Subtleties with branches of log::

    sage: dop = ((1-z)*Dz)^3
    sage: ref = list((ln(1/(1-SR(z)))^2).series(SR(z),1000).truncate().polynomial(QQ))
    sage: asy = bound_coefficients(dop, ref[:3]) # long time
    sage: asy # long time
    1.000...*(([2.000000000000...] + [+/- ...]*I)*n^(-1)*log(n)
    + ([1.154431329803...] + [+/- ...]*I)*n^(-1)
    + ([-1.000000000000...] + [+/- ...]*I)*n^(-2)
    + ([-0.1666666666666...] + [+/- ...]*I)*n^(-3) + B([...]*n^(-4), n >= 9))
    sage: check_seq_bound(asy.expand(), ref) # long time

Miscellaneous examples::

    sage: bound_coefficients((z-1)*Dz + 2, seqini=[1,2,3], name='a') # long time
    1.000...*(1.000...*a + 1.000... + B(..., a >= ...))
    sage: bound_coefficients((z-1)*z*Dz + 2, seqini=[42, 42, 1]) # long time
    1.000...*([1.000...]*n + [-1.000...] + B(..., n >= ...))

::

    sage: asy = bound_coefficients((z^2 - z)*Dz^2 + (3*z - 3)*Dz + 1, seqini=[1/2, 1/6], order=6) # long time
    sage: asy # long time
    1.000000000000000*(([1.000000000000...]...)*n^(-2)
    + ([-3.000000000000...]...)*n^(-3)
    + ([7.00000000000...]...)*n^(-4)
    + ([-15.0000000000...]...)*n^(-5)
    + ([31.000000000...]...)*n^(-6)
    + B([...]*n^(-7)*log(n), n >= 15))
    sage: check_seq_bound(asy.expand(), [1/((n+1)*(n+2)) for n in range(1000)]) # long time

::

    sage: bound_coefficients((z^2 + z - 2)*Dz^2 + (-z + 1)*Dz + 1, seqini=[1, 1, 1/2], order=8) # long time
    1.000000000000000*(([1.00000000000...]...)*n^(-2)
    + ([1.00000000000...]...)*n^(-3)
    + ([1.00000000000...]...)*n^(-4)
    + ([1.00000000000...]...)*n^(-5)
    + ([1.00000000000...]...)*n^(-6)
    + ([1.00000000000...]...)*n^(-7)
    + ([1.0000000000...]...)*n^(-8)
    + B([...]*n^(-9), n >= 19))

Exponentially small error terms with ``output="list"``::

    sage: bound_coefficients((z-1)*Dz+2, [1], output='list')
    (13, [(1, [1.000...*n, 1.000..., 0]), (1.75000..., [[...]])])

Incorrect input::

    sage: bound_coefficients(((z-1)*Dz+1)^2, [0])
    Traceback (most recent call last):
    ...
    ValueError: not enough initial values

    sage: from ore_algebra.analytic.singularity_analysis import *
    sage: bound_coefficients(Dz-1, [1])
    Traceback (most recent call last):
    ...
    NotImplementedError: no nonzero finite singularities

An sequence that is ultimately zero::

    sage: bound_coefficients((z-1)*Dz, [1], order=4) # long time
    Traceback (most recent call last):
    ...
    NotImplementedError: no nonzero finite singularities

REFERENCES:

    [D21] Ruiwen Dong. Asymptotic Expansions and Error Bounds of P-Recursive
    Sequences. M2 Internship Report, Master parisien de recherche en
    informatique, 2021.

    [DMM] Ruiwen Dong, Stephen Melczer, and Marc Mezzarobba. Computing Error
    Bounds for Asymptotic Expansions of Regular P-Recursive Sequences. In
    preparation.
"""

# Copyright 2021 Ruiwen Dong
# Copyright 2021, 2022 Marc Mezzarobba
# Copyright 2021, 2022 Centre national de la recherche scientifique

import collections
import logging
import warnings

from sage.arith.misc import (
    bernoulli,
    rising_factorial,
)
from sage.arith.srange import srange
from sage.categories.cartesian_product import cartesian_product
from sage.categories.homset import Hom
from sage.functions.gamma import gamma
from sage.functions.log import log
from sage.functions.other import (
    binomial,
    ceil,
    factorial,
)
from sage.misc.misc_c import prod
from sage.rings.asymptotic.asymptotic_expansion_generators import asymptotic_expansions
from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing
from sage.rings.asymptotic.growth_group import (
        ExponentialGrowthGroup,
        GrowthGroup,
        MonomialGrowthGroup,
        GenericNonGrowthElement,
)
from sage.rings.asymptotic.term_monoid import BTerm
from sage.rings.complex_arb import CBF, ComplexBallField
from sage.modules.free_module_element import vector
from sage.rings.infinity import infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring import polygen
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.qqbar import QQbar
from sage.rings.rational_field import QQ
from sage.rings.real_arb import RBF
from sage.symbolic.constants import pi
from sage.symbolic.ring import SR

from ..ore_algebra import OreAlgebra
from . import utilities
from .bounds import DiffOpBound
from .differential_operator import DifferentialOperator
from .local_solutions import (
        FundamentalSolution,
        LocalBasisMapper,
        log_series,
        LogSeriesInitialValues,
)
from .path import Point

logger = logging.getLogger(__name__)

################################################################################
# Path choice
#################################################################################

def _sing_in_disk(elts, rad, large_value):
    for j, x in enumerate(elts):
        mag = abs(x)
        if mag > rad:
            return elts[:j], mag
    return elts, large_value

def _choose_big_radius(all_exn_pts, dominant_sing, next_sing_rad):
    # [DMM, (11)]
    max_smallrad = min(abs(ex - ds) for ds in dominant_sing
                                    for ex in all_exn_pts
                                    if ex != ds)
    dom_rad = abs(dominant_sing[-1])
    rad = min(next_sing_rad*RBF(0.875) + dom_rad*RBF(0.125),
              dom_rad + max_smallrad*RBF(0.75))
    logger.info("radius of outer circle = %s", rad)
    return rad

def _check_big_radius(rad, dominant_sing):
    if not dominant_sing:
        raise ValueError("No singularity in the given disk")
    if not abs(dominant_sing[-1]) < rad:
        raise ValueError(f"Singularity {dominant_sing[-1]} is too close to "
                         "the border of the disk")
    rad0 = abs(CBF(dominant_sing[0]))
    bad_sing = [s for s in dominant_sing[1:] if abs(CBF(s)) > rad0]
    if bad_sing:
        warnings.warn("The given disk contains singular points of non-minimal "
                      f"modulus: {bad_sing}")

def _classify_sing(deq, known_analytic, rad):

    # Interesting points = all sing of the equation, plus the origin

    all_exn_pts = deq._singularities(QQbar, multiplicities=False)
    if not any(s.is_zero() for s in all_exn_pts):
        all_exn_pts.append(QQbar.zero())

    # Potential singularities of the function, sorted by magnitude

    singularities = deq._singularities(QQbar, apparent=False,
                                       multiplicities=False)
    singularities = [s for s in singularities if not s in known_analytic]
    singularities.sort(key=lambda s: abs(s)) # XXX wasteful
    logger.debug("potential singularities: %s", singularities)

    if not singularities:
        raise NotImplementedError("no nonzero finite singularities")

    # Dominant singularities

    # dominant_sing is the list of potential dominant singularities of the
    # function, not of "dominant singular points". It does not include singular
    # points of the equation lying in the disk where the function is known to be
    # analytic.
    if rad is None:
        dominant_sing, next_sing_rad = _sing_in_disk(singularities,
                abs(singularities[0]), abs(singularities[-1])*3)
        rad = _choose_big_radius(all_exn_pts, dominant_sing, next_sing_rad)
    else:
        rad = RBF(rad)
        dominant_sing, _ = _sing_in_disk(singularities, rad,
                abs(singularities[-1])*2 + rad*2)
        _check_big_radius(rad, dominant_sing)
    logger.info("dominant singularities: %s", dominant_sing)

    return all_exn_pts, dominant_sing, rad

################################################################################
# Contribution of a logarithmic monomial
# (variant with error bounds of Sage's SingularityAnalysis)
#################################################################################

def truncated_psi(m, n, invz):
    """
    Compute psi^(m)(z) (or, for m = 0, psi(z) - log(z)) truncated at z^(-m-2n-1)
    with an error bound of order z^(-m-2n)

    INPUT:

    - n: integer, non-negative
    - m: integer, non-negative
    - invz: element of polynomial ring, representing 1/z

    TESTS::

        sage: from ore_algebra.analytic.singularity_analysis import truncated_psi
        sage: Pol.<invz> = CBF[]
        sage: truncated_psi(0, 3, invz)
        ([+/- ...] + [+/- ...]*I)*invz^6
        + ([0.008333...])*invz^4 + ([-0.08333...])*invz^2 - 0.5000...*invz
        sage: truncated_psi(1, 3, invz)
        ([+/- ...] + [+/- ...]*I)*invz^7
        + ([-0.0333...])*invz^5 + ([0.1666...])*invz^3 + 0.5000...*invz^2 + invz
        sage: truncated_psi(2, 3, invz)
        ([+/- ...] + [+/- ...]*I)*invz^8
        + ([0.1666...])*invz^6 + ([-0.5000...])*invz^4 - invz^3 - invz^2
        sage: truncated_psi(2, 2, invz)
        ([+/- 0.16...] + [+/- ...]*I)*invz^6
        + ([-0.5000...])*invz^4 - invz^3 - invz^2
    """
    assert n >= 1
    CB = invz.parent().base_ring()
    err = abs(rising_factorial(2*n + 1, m - 1)*bernoulli(2*n))
    sign = (-1)**(m+1)
    ser = sign * (
        + gamma(m+1) * invz**(m+1) / 2
        + sum(bernoulli(2*k)*invz**(2*k+m)*rising_factorial(2*k+1, m-1)
              for k in range(1, n)))
    if m != 0:
        ser += sign * gamma(m) * invz**m
    res = ser + CB(0).add_error(err)*invz**(2*n+m)
    return res

def _generalized_bernoulli(Ring, sigma, count):
    t = polygen(Ring, 't')
    ser = t._exp_series(count + 1) >> 1       # (e^t - 1)/t
    ser = -2*sigma*ser._log_series(count)     # -2σ·log((e^t - 1)/t)
    ser = (ser >> 2) << 2                     # -2σ·log((e^t - 1)/t) + σt
    ser = ser._exp_series(count)              # ((e^t - 1)/t)^(-2σ) * e^(σt)
    bern = [ser[n]*ZZ(n).factorial() for n in range(count)]
    return bern

def truncated_gamma_ratio(alpha, order, u, s):
    """
    Find a truncated expression with error bound for Γ(n+α)/Γ(n+1)/(n+α/2)^(α-1)

    INPUT:

    - alpha: complex number α, !!cannot be negative integer or zero!!
    - order: order of truncation
    - u: element of polynomial ring, representing 1/(n+α/2)
    - s: positive number where n >= s*|alpha| is guaranteed, s > 2

    OUTPUT:

    - a polynomial in CB[u] such that Γ(n+α)/Γ(n+1)/(n+α/2)^(α-1)
      is in its range when n >= s*|alpha|
    """

    Pol = u.parent()
    CB = Pol.base_ring()
    n_gam = ceil((1+order)/2)
    gen_bern = _generalized_bernoulli(CB, alpha/2, 2*n_gam)
    gen_bern_abs = _generalized_bernoulli(CB, abs(alpha/2), 2*n_gam + 1)

    ratio_gamma_asy = Pol([CB(1 - alpha).rising_factorial(j) / factorial(j) * b
                           if j % 2 == 0 else 0
                           for j, b in enumerate(gen_bern)])
    half = RBF.one()/2
    _alpha = CBF(alpha)
    Rnw_bound = (abs((1 - _alpha.real()).rising_factorial(2*n_gam))
                 / factorial(2*n_gam)
                 * abs(gen_bern_abs[2*n_gam])
                 * (abs(_alpha.imag())*(half/s).arcsin()).exp()
                 * ((s+half)/(s-half))**(max(0, -alpha.real()+1+2*n_gam)))
    assert not Rnw_bound < 0
    ratio_gamma = ratio_gamma_asy + CB(0).add_error(Rnw_bound) * u**(2*n_gam)
    return ratio_gamma

def truncated_inverse(alpha, order, invn, s):
    r"""
    (n + α)^(-1) as a polynomial in invn = 1/n plus an error term O(invn^order)
    """
    CB = invn.parent().base_ring()
    err = abs(alpha)**order / (1 - 1/s)
    ser = sum((-alpha)**(j-1) * invn**j for j in range(1, order))
    return ser + CB(0).add_error(err) * invn**(order)

def truncated_log(alpha, order, invn, s):
    r"""
    log(1 + α/n) as a polynomial in invn = 1/n plus an error term O(invn^order)
    """
    CB = invn.parent().base_ring()
    err = (1 + 1/s).log()*abs(CBF(alpha))**order
    ser = - sum((-alpha)**j * invn**j / j for j in range(1, order))
    return ser + CB(0).add_error(err) * invn**(order)

def truncated_power(alpha, order, invn, s):
    """
    Compute a bound for a truncated (1 + α/2n)^(α-1)

    INPUT:

    - alpha: complex number α, !!cannot be negative integer or zero!!
    - order: order of truncation
    - invn: element of polynomial ring, representing 1/n
    - s: positive number where n >= s*|alpha| is guaranteed, s > 2

    OUTPUT:

    - trunc_power: a polynomial in CB[invn] such that (1 + α/2n)^(α-1) is in its
      range when n >= s*|alpha|
    """
    CB = invn.parent().base_ring()
    _alpha = CBF(alpha)
    a = alpha.real() - 1
    h = RBF(1)/2
    err = (abs(_alpha)**(order) / (1 - 1/s)
           * (abs(alpha.imag())/2).exp()
           * max((3*h)**a, h**a))
    assert not err < 0
    t = polygen(CB, 't')
    ser = (alpha - 1) * (1 + alpha/2 * t)._log_series(order)
    ser = ser._exp_series(order)
    return ser(invn) + CB(0).add_error(err) * invn**(order)

def trim_univariate(pol, order, varbound):
    r"""
    Replace each x^{order+j} with B(0, varbound)^j*x^order.
    """
    err = pol.base_ring().zero()
    for j, c in enumerate(list(pol)[order:]):
        err = err.add_error(abs(c)*varbound**j)
    coeff = pol.padded_list(order)
    coeff.append(err)
    return pol.parent()(coeff)

def trim_expr(f, order, n0):
    """
    Truncate and bound an expression f(1/n, log(n)) to a given degree

    Each 1/n^(order+t) (t > 0) is replaced by 1/n^order*CB(0).add_error(1/n0^t);
    factors log(n)^k are left untouched.
    """
    Expr = f.parent()
    invn = Expr.gen(0)
    CB = Expr.base_ring()
    g = Expr.zero()
    # TODO: simplify...
    for c, mon in f:
        deg = mon.degree(invn)
        if deg > order:
            tuple_mon_g = tuple(map(lambda x, y: x - y, mon.exponents()[0],
                                    (invn**(deg - order)).exponents()[0]))
            mon_g = prod(Expr.gens()[j]**(tuple_mon_g[j])
                         for j in range(len(tuple_mon_g)))
            c_g = ((c if c.mid() == 0 else CB(0).add_error(abs(c)))
                   / CB(n0**(deg - order)))
            g = g + c_g * mon_g
        else:
            g = g + c*mon
    return g

def trim_expr_series(f, order, n0):
    trimmed = f.parent()([trim_expr(c, order, n0) for c in f])
    return trimmed.add_bigoh(f.parent().default_prec())

def bound_gamma_ratio(Expr, exact_alpha, order, n0, s):
    # fg = n^(1-α) Γ(n+α)/Γ(n+1)
    CB = Expr.base_ring()
    invn, _ = Expr.gens()
    alpha = CB(exact_alpha)

    # for α = 0, fg is a polynomial, but of degree α, not α - 1
    if exact_alpha.parent() is ZZ and order >= exact_alpha >= 1:
        fg = prod(1 + k*invn for k in range(1, exact_alpha))
        logger.debug("    fg = %s", fg)
    else:
        # (n+α/2)^(1-α) * Γ(n+α)/Γ(n+1)
        u = polygen(CB, 'u') # u stands for 1/(n+α/2)
        f = truncated_gamma_ratio(alpha, order, u, s)
        truncated_u = truncated_inverse(alpha/2, order, invn, s)
        f = trim_expr(f(truncated_u), order, n0)
        logger.debug("    f = %s", f)
        # (1 + α/2n)^(α-1) = (n+α/2)^(α-1) * n^(1-α)
        g = truncated_power(alpha, order, invn, s)
        logger.debug("    g = %s", g)
        fg = trim_expr(f*g, order, n0)
    return fg

def bound_gamma_ratio_derivatives(Expr, exact_alpha, log_order, order, n0, s):
    CB = Expr.base_ring()
    invn, logn = Expr.gens()
    alpha = CB(exact_alpha)

    # Bound for 1/Γ(n+α) (d/dα)^k [Γ(n+α)/Γ(α)]
    # Use PowerSeriesRing because polynomials do not implement all the
    # "truncated" operations we need
    Pol_invz, invz = PolynomialRing(CB, 'invz').objgen() # z = n + α
    Series_z, eps = PowerSeriesRing(Pol_invz, 'eps', log_order).objgen()
    order_psi = max(1, ceil(order/2))
    if not (exact_alpha.parent() is ZZ and exact_alpha <= 0):
        pols = [(truncated_psi(m, order_psi, invz) - alpha.psi(m))
                / (m + 1).factorial() for m in srange(log_order)]
        p = Series_z([0] + pols)
        hh1 = (1/alpha.gamma())*p.exp()
    else:
        pols = [(truncated_psi(m, order_psi, invz)
                 + (-1)**(m+1)*(1-alpha).psi(m)) / (m + 1).factorial()
                for m in srange(log_order - 1)]
        p = Series_z([0] + pols)
        _pi = CB(pi)
        sine = (-1 if exact_alpha%2 else 1)*(_pi*eps).sin()
        hh1 = ((1-alpha).gamma()/_pi)*(p.exp()*sine)
    # XXX use s instead?
    invz_bound = ~(n0 - abs(alpha))
    hh1 = hh1.parent()([trim_univariate(c, order, invz_bound)
                        for k, c in enumerate(hh1)])

    Series, eps = PowerSeriesRing(Expr, 'eps', log_order).objgen()
    truncated_invz = truncated_inverse(alpha, order, invn, s)
    h1 = hh1.map_coefficients(Hom(Pol_invz, Expr)(truncated_invz)) # ∈ Expr[[z]]
    assert h1.parent() is Series
    h1 = trim_expr_series(h1, order, n0)

    # (n + α)^ε as a series in ε with coeffs in ℂ[logn][[invn]] (trunc+bounded)
    h2 = (truncated_log(alpha, order, invn, s)*eps).exp()
    h2 = trim_expr_series(h2, order, n0)
    h3 = (logn*eps).exp()

    logger.debug("    h1 = %s", h1)
    logger.debug("    h2 = %s", h2)
    logger.debug("    h3 = %s", h3)

    h = h1*h2*h3
    h = trim_expr_series(h, order, n0)
    return h

def bound_coeff_mono(Expr, alpha, log_order, order, n0, s):
    """
    Bound [z^n] (1-z)^(-α) * log(1/(1-z))^k by an expression
    of the form n^(α-1) * P(1/n, log(n)) for all k < log_order

    INPUT:

    - Expr: output ring, of the form C[invn, logn]
    - alpha: complex number, representing α
    - log_order: non-negative integer
    - order: expansion order wrt n
    - n0: integer > -alpha, lower bound of validity range
    - s: real number > 2 s.t. n >= s*|alpha| for all n >= n0

    OUTPUT:

    - a list of length log_order of polynomials P in invn, logn,
      corresponding to k = 0, ..., log_order - 1
    """
    # All entries can probably be deduced from the last one using Frobenius'
    # method, but I don't think it helps much computationally(?)

    if alpha.is_integer():
        alpha = ZZ(alpha)

    order = max(0, order)

    fg = bound_gamma_ratio(Expr, alpha, order, n0, s)
    h = bound_gamma_ratio_derivatives(Expr, alpha, log_order, order, n0, s)

    full_prod = fg * h
    full_prod = trim_expr_series(full_prod, order, n0)
    res = [ZZ(k).factorial()*c
           for k, c in enumerate(full_prod.padded_list(log_order))]
    for k, pol in enumerate(res):
        logger.debug("    1/(1-z)^(%s)*log(1/(1-z))^%s --> %s",
                     alpha, k, pol)
    return res

#################################################################################
# Contribution of a single regular singularity
#################################################################################

def _my_log_series(bwrec, inivec, leftmost, mults, struct, order):
    r"""
    Similar to log_series() called on a LogSeriesInitialValues object
    corresponding to the part with a given local exponent mod 1 of a local
    solution specified by a vector of initial conditions, but attempts to
    minimize interval swell by unrolling the recurrence in exact arithmetic
    (once per relevant initial value) and taking a linear combination.

    The output is a list of lists, not vectors.
    """
    log_len = sum(m for _, m in mults)
    res = [[inivec.base_ring().zero()]*log_len for _ in range(order)]
    for sol, c in zip(struct, inivec):
        if c.is_zero() or sol.leftmost.as_algebraic() != leftmost.as_algebraic():
            continue
        values = { (sol1.shift, sol1.log_power): QQ.zero()
                   for sol1 in struct if sol1.leftmost.as_algebraic() == leftmost.as_algebraic() }
        values[sol.shift, sol.log_power] = QQ.one()
        ini = LogSeriesInitialValues(expo=leftmost, values=values, mults=mults)
        ser = log_series(ini, bwrec, order)
        for i in range(order):
            for j, a in enumerate(ser[i]):
                res[i][j] += c*a
    return res

ExponentGroupData = collections.namedtuple('ExponentGroupData', [
    'val',  # exponent group (lefmost element mod ℤ)
    're_val', # real part of exponent group
    'bound',  # coefficient bound (explicit part + local error term)
    'initial_terms'  # explicit part of the local solution
                     # (will be reused to compute the global error term)
])

SingularityData = collections.namedtuple('SingularityData', [
    'rho',  # the singularity
    'expo_group_data',  # as above, for each exponent group
])

class SingularityAnalyzer(LocalBasisMapper):

    def __init__(self, dop, inivec, *, rho, rad, Expr, rel_order, n0):

        super().__init__(dop)

        self.inivec = inivec
        self.rho = rho
        self.rad = rad
        self.Expr = Expr
        self.n0 = n0
        self.rel_order = rel_order

    def run(self):
        # Redundant work; TBI.
        # (Cases where we really need this to detect non-analyticity are
        # rare...)
        self._local_basis_structure = Point(0, self.dop).local_basis_structure()

        nonanalytic = [sol for sol in self._local_basis_structure if not (
            sol.leftmost.as_algebraic().is_integer()
            and sol.leftmost.as_algebraic() + sol.shift >= 0
            and all(c.is_zero() for term in sol.value.values()
                    for c in term[1:]))]

        if not nonanalytic:
            logger.debug("found an apparent singularity")
            return []

        self.re_leftmost = QQbar(nonanalytic[0].leftmost).real()
        min_val = self.re_leftmost + nonanalytic[0].shift
        self.abs_order = self.rel_order + min_val
        return super().run()

    def process_modZ_class(self):

        logger.info("sing=%s, valuation=%s", self.rho, QQbar(self.leftmost))

        order = (self.abs_order - self.re_leftmost).ceil()
        # TODO: consider increasing order1 adaptively, like we do with
        # pol_part_len (via maj.refine()) in _bound_local_integral_of_tail
        order1 = 40 + 4*order + self.bwrec.order
        # XXX It works, and should be faster, to compute initial values and call
        # log_series, but doing so leads to worse bounds due to using the
        # recurrence in interval arithmetic
        shifted_bwrec = self.bwrec.shift(
            self.leftmost.as_number_field_element())
        ser = _my_log_series(shifted_bwrec, self.inivec,
                self.leftmost, self.shifts, self._local_basis_structure, order1)

        # All contributions to the bound depend on a parameter s > 2 and are
        # valid for n > n0 provided that n0 > s*|α| where α is the relevant
        # valuation, which is always of the form leftmost + shift for some
        # integer shift with 0 ≤ shift ≤ order.
        s = RBF(self.n0) / (abs(self.leftmost.as_ball(CBF)) + abs(order) + 1)
        logger.debug("s=%s", s)
        assert s > 2

        bound_lead_terms, initial_terms = _bound_local_integral_explicit_terms(
                self.Expr, self.rho, self.leftmost.as_algebraic(), order, s,
                self.n0, ser[:order])

        # TODO: move to _bound_local_integral_of_tail(?), simplify arguments,
        # avoid calling _bound_tail in analytic case
        smallrad = self.rad - CBF(self.rho).below_abs()
        vb = _bound_tail(self.dop, self.leftmost, order, self.all_roots,
                         self.shifts, smallrad, ser)

        # Bound degree in log(z) of the local expansion of the solution defined
        # by ini
        assert self.shifts[0][0] == 0 and order1 > 0
        kappa = max(k for shift, mult in self.shifts if shift < order1
                    for k, c in enumerate(ser[shift]) if not c.is_zero())
        # We could use the complete family of critical monomials for a tighter
        # bound... but in the future we may want to avoid computing it
        kappa += sum(mult for shift, mult in self.shifts if shift >= order1)

        bound_int_SnLn = _bound_local_integral_of_tail(self.Expr, self.rho,
                self.leftmost.as_algebraic(), order, s, self.n0, vb, kappa)

        logger.info("  explicit part = %s", bound_lead_terms)
        logger.info("  local error term = %s", bound_int_SnLn)

        data = ExponentGroupData(
            val = self.leftmost.as_exact(),
            re_val = QQbar(self.leftmost).real(),
            bound = bound_lead_terms + bound_int_SnLn,
            initial_terms = initial_terms)

        # WARNING: Abusing FundamentalSolution somewhat here.
        # The log_power field is not meaningful, but we need _some_ integer
        # value to please code that will try to sort the solutions.
        sol = FundamentalSolution(leftmost=self.leftmost, shift=ZZ.zero(),
                                  log_power=0, value=data)
        self.irred_factor_cols.append(sol)

def _bound_tail(dop, leftmost, order, ind_roots, shifts, smallrad, series):
    r"""
    Upper-bound the tail of order ``order`` of a logarithmic series solution of
    ``dop`` with exponents in ``leftmost`` + ℤ, on a disk of radius
    ``smallrad``, using ``order1`` ≥ ``order`` explicitly computed terms given
    as input in ``series`` and a bound based on the method of majorants for the
    terms of index ≥ ``order1``.
    """
    assert order <= len(series)
    maj = DiffOpBound(dop,
                      leftmost=leftmost,
                      special_shifts=(None if dop.leading_coefficient()[0] != 0
                                      else shifts),
                      bound_inverse="solve",
                      pol_part_len=10,
                      ind_roots=ind_roots)
    ordrec = maj.dop.degree()
    last = list(reversed(series[-ordrec:]))
    order1 = len(series)
    # Coefficients of the normalized residual in the sense of [Mez19, Sec. 6.3],
    # with the indexing conventions of [Mez19, Prop. 6.10]
    res = maj.normalized_residual(order1, last, Ring=CBF)
    oldtb = tb = RBF('inf')
    while True:
        # Majorant series of [the components of] the tail of the local expansion
        # of f at ρ. See [Mez19, Sec. 4.3] and [Mez19, Algo. 6.11].
        # WARNING: will be modified in-place
        tmaj = maj.tail_majorant(order1, [res])
        # Shift it (= factor out z^order) ==> majorant series of the tails
        # of the coefficients of log(z)^k/k!
        tmaj >>= -order
        # Bound on the *values* for |z| <= smallrad of the analytic functions
        # appearing as coefficients of log(z)^k/k! in the tail of order 'order1'
        # of the local expansion
        oldtb = tb
        tb = tmaj.bound(smallrad)
        if not tb < oldtb/16:
            break
        maj.refine()
    # Bound on the intermediate terms
    ib = sum(smallrad**n1 * max(c.above_abs() for c in vec)
            for n1, vec in enumerate(series[order:]))
    # Same as tb, but for the tail of order 'order'
    return tb + ib

def _bound_local_integral_of_tail(Expr, rho, val_rho, order, s, n0, vb, kappa):
    r"""
    Bound the integral over a small loop around the singularity ``rho``,
    connecting to the big circle, of 1/(2πi)*g(z)/z^{n+1} where g is the tail
    starting at z^{val_rho + order} of the local expansion of a certain
    solution of the differential equation.

    The bound is valid for ``n ≥ max(n0, s*(|val_rho| + order))`` provided that
    ``s > 2``.
    """

    assert s > 2

    # Analytic case (terminating expansion in powers of n).
    # (This is more general than the case of a terminating local expansion in
    # powers of (z-ρ), where vb would be zero anyway!)
    if kappa == 0 and val_rho in ZZ and val_rho + order >= 0:
        return Expr.zero()

    # Results in an error term of degree in logn too large by one unit when
    # val_rho is an integer and val_rho + order >= 0. This is no big deal here
    # (unlike above) since one can always increase the expansion order.

    invn, logn = Expr.gens()

    _pi = RBF(pi)
    _rho = CBF(rho)

    # Change representation from log(z-ρ) to log(1/(1 - z/ρ))
    # The h_i are cofactors of powers of log(z-ρ), not log(1/(1-z/ρ)).
    # Define the B polynomial in a way that accounts for that.
    ll = abs(CBF(-rho).log())
    B = vb*RBF['z']([
            sum([ll**(m - j) * binomial(m, j) / factorial(m)
                    for m in range(j, kappa + 1)])
            for j in range(kappa + 1)])

    beta = CBF(val_rho).real() + order

    A = ((abs(_rho.arg()) + 2*_pi)*abs(CBF(val_rho).imag())).exp()
    Bpi = B(_pi + logn)

    # Sub polynomial factor for bound on S(n)
    assert isinstance(n0, Integer)
    err_S = abs(_rho)**beta * A * CBF(1 - 1/n0)**(-n0-1)
    bound_S = CBF(0).add_error(err_S)*Bpi
    # Sub polynomial factor for bound on L(n)
    if beta <= 0:
        C_nur = 1
    else:
        C_nur = 2 * (CBF(1).exp()*(s - 2)/(2*s*beta))**beta
    # This constant differs from the one in the paper because we are working
    # with powers of (z-ρ) instead of (1-z/ρ). (?)
    err_L = C_nur/_pi * abs(_rho)**beta * A
    bound_L = CBF(0).add_error(err_L)*Bpi

    return Expr(bound_S + bound_L) * invn**order

def _bound_local_integral_explicit_terms(Expr, rho, val_rho, order, s, n0, ser):
    r"""
    Bound the coefficient of z^n in the expansion at the origin of the initial
    terms of the local expansion at ρ whose coefficients are given in ser.

    The first element of the output is a polynomial p(invn, logn), with an error
    term of order Õ(invn^order), such that n^val_rho*p(1/n, log(n)) is the
    desired bound.

    The bound is valid for ``n ≥ n0``. It is required that s > 2 and
    ``n0 > s*(|val_rho| + order))``.

    The function additionally returns a polynomial ℓ(Z, L) such that the series
    to whose coefficients the previous bound applies is equal to
    ℓ(z-ρ, log(1-z/ρ)).
    """

    assert s > 2
    assert order >= 0
    assert n0 > s*(abs(val_rho) + order)

    invn, _ = Expr.gens()
    CB = Expr.base_ring()

    # Rewrite the local expansion in terms of new variables Z = z - ρ,
    # L = log(1/(1-z/rho))

    PolL, L = PolynomialRing(CB, 'L').objgen()
    PolL, Z = PolynomialRing(PolL, 'Z').objgen()
    mylog = CB.coerce(-rho).log() - L # = log(z - ρ) for Im(z) ≥ 0
    locf_ini_terms = sum(c/ZZ(k).factorial() * mylog**k * Z**shift
                         for shift, vec in enumerate(ser)
                         for k, c in enumerate(vec))

    bound_lead_terms = Expr.zero()
    _minus_val_rho = QQbar(-val_rho)
    for degZ, slice in enumerate(locf_ini_terms):
        logger.debug("  (z - %s)^(%s + %s)*(...)...",
                     rho, _minus_val_rho, degZ)
        # XXX could be shared between singularities with common exponents...
        # (=> tie to an object and add @cached_method decorator?)
        coeff_bounds = bound_coeff_mono(Expr, _minus_val_rho - degZ,
                                        slice.degree() + 1, order - degZ, n0, s)
        new_term = (CB(-rho)**CB(val_rho+degZ) * invn**(degZ)
                    * sum(c*coeff_bounds[degL] for degL, c in enumerate(slice)))
        bound_lead_terms += new_term
        logger.debug("  (z - %s)^(%s)*(%s) --> %s",
                     rho, _minus_val_rho + degZ, slice, new_term)

    return bound_lead_terms, locf_ini_terms

def contribution_single_singularity(deq, ini, rho, rad, Expr, rel_order, n0):
    r"""
    Bound the integral over a small loop around ρ, connecting to the big circle
    of radius rad, of 1/(2πi)*f(z)/z^{n+1} where f is the solution of deq
    corresponding to the initial conditions at 0 given in ini.

    The result is a collection of contributions of the same form as the output
    of _bound_local_integral_explicit_terms(). The bound is valid for all
    n >= n0.

    Some ancillary data is returned for each contribution, including an
    expression of the associated local expansion at ρ of a component of f (see
    _bound_local_integral_explicit_terms()).
    """

    logger.info("singular point %s, computing transition matrix...", rho)
    eps = RBF.one() >>  Expr.base_ring().precision() + 13
    tmat = deq.numerical_transition_matrix([0, rho], eps, assume_analytic=True)
    coord_all = tmat*ini
    logger.info("done")

    ldop = deq.shift(Point(rho, deq))

    # Split the local expansion of f according to the local exponents mod ℤ. For
    # each group (ℤ-coset) of exponents, compute coefficient asymptotics (and
    # some auxiliary data). Again: each element of the output corresponds to a
    # whole ℤ-coset of exponents, already incorporating initial values.
    analyzer = SingularityAnalyzer(dop=ldop, inivec=coord_all, rho=rho, rad=rad,
                                   Expr=Expr, rel_order=rel_order, n0=n0)
    data = analyzer.run()

    data1 = SingularityData(
        rho = rho,
        expo_group_data = [sol.value for sol in data],
    )

    return data1

################################################################################
# Exponentially small error term
################################################################################

def numerical_sol_big_circle(deq, ini, dominant_sing, rad, halfside):
    """
    Bound the values of a solution `f` on the large circle

    INPUT:

    - ``deq``: a linear ODE that the generating function satisfies
    - ``ini``: coefficients of `f` in the basis at zero
    - ``dominant_sing``: list of dominant singularities
    - ``rad``: radius of big circle
    - ``halfside``: half of side length of covering squares

    OUTPUT:

    A list of (square, value) pairs where the squares cover the large circle
    and, for each square, the corresponding value is a complex interval
    containing the image of the square by the analytic continuation of `f` to a
    multi-slit disk.
    """
    logger.info("starting to compute values on outer circle...")
    clock = utilities.Clock()
    clock.tic()

    I = CBF.gen(0)
    # TODO: eps could be chosen as a function of halfside or something...
    eps = RBF.one() >> 100
    halfside = RBF(halfside)

    sings = [CBF(s) for s in dominant_sing]
    sings.sort(key=lambda s: s.arg())
    num_sings = len(sings)
    pairs = []
    for j0 in range(num_sings):
        j1 = (j0 + 1) % num_sings
        arg0 = sings[j0].arg()
        arg1 = sings[j1].arg()
        if j1 == 0:
            # last arc is a bit special: we need to add 2*pi to the argument of
            # the end
            arg1 += 2*RBF.pi()

        # Compute initial values at a point on the large circle, halfway between
        # two adjacent dominant singularities
        hub = rad * ((arg0 + arg1)/2 * I).exp()
        tmat_hub = deq.numerical_transition_matrix([0, hub], eps,
                                                   assume_analytic=True)
        ini_hub = tmat_hub*vector(ini)

        # From there, walk along the large circle in both directions
        halfarc = (arg1 - arg0)/2
        np = ZZ(((halfarc*rad / (2*halfside)).above_abs()).ceil()) + 2

        logger.info("  sector %d, %d squares of half-side %s", j0, np, halfside)
        # TODO: optimize case of real coefficients
        # TODO: check path correctness (plot?) in complex cases
        for side in [1, -1]:
            squares = [[(hub*(side*halfarc*k/np*I).exp()).add_error(halfside)]
                       for k in range(np+1)]
            path = [hub] + squares
            pairs += deq.numerical_solution(ini_hub, path, eps)

    clock.toc()
    logger.info("...done, %s", clock)
    return pairs

def max_big_circle(deq, ini, dominant_sing, sing_data, rad, halfside):

    pairs = numerical_sol_big_circle(deq, ini, dominant_sing, rad, halfside)
    covering, f_big_circle = zip(*pairs)

    sum_g = [
        sum((_z-_rho)**(CBF(edata.val))
                # some of the _z may lead to arguments of log that cross the
                # branch cut, but that's okay
                * edata.initial_terms(Z=_z-_rho, L=(~(1-_z/_rho)).log())
            for sdata in sing_data for _rho in (CBF(sdata.rho),)
            for edata in sdata.expo_group_data)
        for j, _z in enumerate(covering)]
    res = RBF.zero().max(*((s - vv).above_abs()
                           for s, vv in zip(sum_g, f_big_circle)))
    return res

def absorb_exponentially_small_term(CB, cst, ratio, beta, final_kappa, n0, n):
    if beta <= n0 * ratio.log():
        _beta = RBF(beta)
        cst1 = _beta.exp()*(_beta/ratio.log())**(-_beta)
    else:
        cst1 = ratio**n0 * CBF(n0)**(-beta)
    rad_err = cst*cst1 / CBF(n0).log()**final_kappa
    return (CB(0).add_error(rad_err) * n**QQbar(beta) * log(n)**final_kappa)

def add_error_term(bound, rho, term, n):
    for rho1, local_bound in bound:
        if rho1 == rho:
            # We know that the last term is an error term with the same
            # power of n and log(n) as error_term_big_circle
            local_bound[-1] = (local_bound[-1] + term).collect(n)
        break
    else:
        bound.append([rho, term])

################################################################################
# Conversion to an asymptotic expansion
################################################################################

class FormalProduct:
    r"""
    A formal product of a main exponential factor ``(1/ρ)^n`` and an asymptotic
    expansion (which may contain terms involving ``ω^n`` for some ``ω`` as well,
    typically with ``|ω| = 1``).
    """

    def __init__(self, exponential_factor, series_factor):
        self._exponential_factor = exponential_factor
        self._series_factor = series_factor

    def __repr__(self):
        return f"{self._exponential_factor}*({self._series_factor})"

    def exponential_factor(self):
        return self._exponential_factor

    def series_factor(self):
        return self._series_factor

    def expand(self):
        return self._exponential_factor*self._series_factor

def _remove_non_growth_factors(growth):
    G = growth.parent()
    return G([g for g in growth.factors()
              if not isinstance(g, GenericNonGrowthElement)])

def to_asymptotic_expansion(Coeff, name, term_data, n0, beta, kappa, rad,
                            cst_big_circle):
    r"""
    Convert a list ``term_data`` of contribution to a Sage asymptotic expansion.

    As of Sage 9.6, may fail for some inputs because not everything we need is
    implemented in Sage yet.

    INPUT:

    - ``Coeff``: coefficient ring (typically a complex ball field)
    - ``name``: variable name
    - ``term_data``: list of lists [ρ, [m0, m1, ...]], representing terms of the
      form ρ^(-n)*mk
    - ``n0``, ``beta``, ``kappa``, ``rad``: parameters (validity, exponent of n,
      exponent of log(n)) of the desired error term; ``beta`` can be -∞,
      indicating an exponentially small error term ``O(rad^(-n))``
    - ``cst_big_circle``: coefficient of the _exponentially small_ error term;
      if ``None``, the error term in the result will be an O-term instead of a
      B-term; otherwise, ignored except when ``beta`` is -∞.
    """

    from sage.categories.cartesian_product import cartesian_product
    from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing

    n = SR.var(name)

    # TODO: detect cases where we can use 1 or ±1 or U as Arg
    Exp, Arg = ExponentialGrowthGroup.factory(QQbar, name, return_factors=True)
    # AsymptoticRing does not split MonomialGrowthGroups with non-real
    # exponent groups in growth*non-growth parts, presumably because this has no
    # impact on term ordering. Let us do the same.
    Pow = MonomialGrowthGroup(QQbar, name)
    Log = MonomialGrowthGroup(ZZ, f"log({name})")
    Growth = cartesian_product([Arg, Exp, Pow, Log])
    Asy = AsymptoticRing(Growth, coefficient_ring=Coeff)
    ET = Asy.term_monoid('exact')
    BT = Asy.term_monoid('B').change_parameter(
            coefficient_ring=Coeff._real_field())
    OT = Asy.term_monoid('O')

    def make_arg_factor(dir):
        if dir.imag().is_zero() and dir.real() >= 0:
            return ET.one()
        else:
            return ET(Arg(raw_element=dir))

    if not term_data:
        exp_factor = ET(Exp(raw_element=~rad))
        if cst_big_circle is None:
            terms = [OT(1)]
        else:
            terms = [BT(1, coefficient=cst_big_circle, valid_from={name: n0})]
        return FormalProduct(Asy(exp_factor), Asy(terms))

    rho0 = term_data[0][0]
    mag0 = abs(rho0)
    exp_factor = ET(Exp(raw_element=~mag0))
    if all(rho == rho0 for rho, _ in term_data[1:]):
        exp_factor *= make_arg_factor(~rho0)
    else:
        rho0 = mag0

    if beta == -infinity:
        alg_error_growth = None
    else:
        alg_error_growth = ET(n**beta*log(n)**kappa).growth

    terms = []
    alg_error_coeff = Coeff.zero()
    for rho, symterms in term_data:
        dir = rho0/rho
        # need an additional growth factor if this is not the case
        assert RBF(abs(dir)).contains_exact(1)
        arg_factor = make_arg_factor(dir)
        for symterm in symterms:
            if symterm.is_trivial_zero():
                continue
            term = arg_factor*ET(symterm.subs(n=n))
            term_growth = _remove_non_growth_factors(term.growth)
            if alg_error_growth is not None and term_growth == alg_error_growth:
                assert term.coefficient.contains_zero()
                alg_error_coeff += term.coefficient.above_abs()
            else:
                terms.append(term)
    if not alg_error_coeff.is_zero():
        terms.append(BT(alg_error_growth, coefficient=alg_error_coeff,
                        valid_from={name: n0}))

    if cst_big_circle is None:
        terms.append(OT(Exp(raw_element=(mag0/rad))))
    elif beta == -infinity:
        terms.append(BT((Exp(raw_element=(mag0/rad))),
                        coefficient=cst_big_circle,
                        valid_from={name: n0}))

    return FormalProduct(Asy(exp_factor), Asy(terms))

################################################################################
# Final bound
################################################################################

def _coeff_zero(seqini, deq):
    """
    Find coefficients of generating function in the basis with these expansions
    at the origin

    INPUT:

    - seqini : list, initial terms of the sequence
    - deq : a linear ODE that the generating function satisfies

    OUTPUT:

    - coeff : vector, coefficients of generating function in the basis with
      these expansions at the origin
    """

    list_basis = deq.local_basis_expansions(0)
    list_coeff = []
    for basis in list_basis:
        mon = next(m for c, m in basis if not c == 0)
        if mon.k == 0 and mon.n >= 0:
            if mon.n >= len(seqini):
                raise ValueError("not enough initial values")
            list_coeff.append(seqini[mon.n])
        else:
            list_coeff.append(0)
    return vector(list_coeff)

def _bound_validity_range(n0, dominant_sing, order):

    # Make sure the disks B(ρ, |ρ|/n) contain no other singular point

    if len(dominant_sing) > 1:
        min_dist = min(s0.dist_to_sing() for s0 in dominant_sing)
        n1 = ceil(2*abs(QQbar(dominant_sing[-1]))/min_dist)
    else:
        n1 = 0

    # Make sure that n0 > 2*|α| for all exponents α we encounter

    # TODO: avoid redundant computation...
    max_abs_val = max(abs(sol.leftmost.as_ball(CBF))
                      for s0 in dominant_sing
                      for sol in s0.local_basis_structure())
    # XXX couldn't we limit ourselves to the exponents of non-analytic terms?
    n2 = (RBF(21)/10*(max_abs_val + order + 1)).above_abs().ceil()
    n0 = ZZ(max(n0, n1, n2))

    logger.debug("n1=%s, n2=%s, n0=%s", n1, n2, n0)
    return n0

def truncate_tail_SR(val, re_val, f, beta, kappa, n0, n):
    """
    Convert a polynomial f ∈ C[invn, logn] to the symbolic expression
    n^val*p(1/n, log(n)), truncating it to O(n^β*log(n)^κ) on the fly.

    1/n^(beta+t), t>=0 is replaced by cst*logn^kappa/n^beta

    INPUT:

    - val: algebraic number
    - re_val: real part of val
    - f: polynomial in invn (representing 1/n) and logn (representing log(n))
    - beta, kappa: parameters of the new error term
    - n0: integer, result valid for n >= n0
    - n : symbolic variable

    OUTPUT:

    a list of symbolic expressions whose sum forms the new bound
    """
    Expr = f.parent()
    CB = Expr.base_ring()
    invn, logn = Expr.gens()
    g = []
    error_term = SR.zero()
    for deg_invn in range(f.degree(invn) + 1):
        for c, mon in list(f.coefficient({invn: deg_invn})):
            deg_logn = mon.degree(logn)
            if re_val - deg_invn > beta:
                g.append(c * n**(val - deg_invn) * log(n)**deg_logn)
            else:
                c_g = (((c if c.mid() == 0 else CB(0).add_error(abs(c)))
                        / CB(n0)**(beta + deg_invn - re_val))
                       * CB(n0).log()**(deg_logn - kappa))
                error_term += c_g * n**beta * log(n)**kappa
    g.append(error_term)
    return g

def bound_coefficients(deq, seqini, name='n', order=3, prec=53, n0=0, *,
                       known_analytic=[0], rad=None, halfside=None,
                       output='asymptotic_expansion',
                       ignore_exponentially_small_term=False):
    """
    Compute a bound for the n-th element of a holonomic sequence

    INPUT:

    - ``deq``: a linear ODE that the generating function satisfies
    - ``seqini``: list, initial elements of sequence, long enough to determine
      the sequence
    - `name` (optional): variable name to be used in the formula
    - ``order`` (optional): expansion order, counting only powers of the
      variable (as opposed to log factors)
    - ``prec`` (optional): numeric working precision (in bits)
    - ``n0`` (optional): restrict to n >= n0 (note that the final validity range
      may be smaller)
    - ``known_analytic`` (optional, default `[0]`): list of points where the
      generating function is known to be analytic, default is [0]
    - ``rad`` (optional, default: automatic choice): radius of the outer part of
      the integration contour (corresponding to an exponentially small error
      term)
    - ``halfside`` (optional, default: automatic choice): resolution parameter
      used in the computation of the exponentially small error term
    - ``output`` (optional, default: ``"asymptotic_expansion"``): set to
      ``"list"`` to get the results as a list of terms instead of an
      ``AsymptoticRingElement``
    - ``ignore_exponentially_small_term``: skip computation of an exponentially
      small contribution to the error term; with
      ``output="asymptotic_expansion"``, the resulting expansion will contain an
      exponentially small O-term (usually in addition to a B-term)

    OUTPUT:

    - when ``output='asymptotic_expansion'``: an ``AsymptoticRingElement`` with
      a B-term encoding the error bound

    - when ``output='list'``: a pair ``(N0, bound)`` where:

        - ``N0`` is an integer such that the bound is valid when ``n >= N0``
        - ``bound`` is list of lists of the form ``[rho, ser]`` where ``rho`` is
          in ``QQbar`` and ``ser`` is a symbolic expression

      such that the sum of ``rho**(-n) * ser`` is a bound for the `n`-th element
      of the input sequence
    """
    if output not in ['asymptotic_expansion', 'list']:
        raise ValueError(f"unknown output format: {output}")

    CB = ComplexBallField(prec)

    deq = DifferentialOperator(deq)

    while True:

        # Identify dominant singularities, choose big radius
        all_exn_pts, dominant_sing, rad1 = _classify_sing(deq, known_analytic, rad)

        # Compute validity range
        # TODO: also use Points elsewhere when relevant
        _dominant_sing = [Point(s, deq) for s in dominant_sing]
        n0 = _bound_validity_range(n0, _dominant_sing, order)

        # Convert initial sequence terms to solution coordinates in the basis at 0
        ini = _coeff_zero(seqini, deq)

        # Contribution of each singular point

        Expr = PolynomialRing(CB, ['invn', 'logn'], order='lex')
        invn, logn = Expr.gens()

        sing_data = [contribution_single_singularity(deq, ini, rho, rad1, Expr,
                                                    order, n0)
                    for rho in dominant_sing]

        if all(sdata.expo_group_data == [] for sdata in sing_data):
            known_analytic.extend(dominant_sing)
        else:
            rad = rad1
            break

    # All error terms will be reduced to the form cst*n^β*log(n)^final_kappa
    ref_val = min((edata.re_val for sdata in sing_data
                                for edata in sdata.expo_group_data
                                if not edata.bound.is_zero()),
                  default=infinity)
    beta = - ref_val - 1 - order
    final_kappa = -1
    for sdata in sing_data:
        for edata in sdata.expo_group_data:
            shift = edata.re_val - ref_val
            if shift not in ZZ: # explicit terms + o(n^β)
                continue
            final_kappa = max([
                final_kappa,
                *(edata.bound.coefficient({invn: d}).degree(logn)
                  for d in range(order - ZZ(shift),
                                 edata.bound.degree(invn) + 1))])
    if final_kappa < 0: # all expansions were exact
        beta = -infinity

    n = SR.var(name)
    bound = [(sdata.rho,
              [term for edata in sdata.expo_group_data
               for term in truncate_tail_SR(-edata.val-1, -edata.re_val-1,
                                            edata.bound, beta,
                                            final_kappa, n0, n)])
             for sdata in sing_data]

    # Exponentially small error term

    if ignore_exponentially_small_term:
        cst_big_circle = None
    else:
        # TODO: use a non-uniform covering
        if halfside is None:
            halfside = min(abs(abs(ex) - rad) for ex in all_exn_pts)/10
        cst_big_circle = max_big_circle(deq, ini, dominant_sing, sing_data, rad,
                                        halfside)
        if beta != -infinity:
            mag_dom = abs(dominant_sing[0])
            error_term_big_circle = absorb_exponentially_small_term(CB,
                        cst_big_circle, mag_dom/rad, beta, final_kappa, n0, n)
            logger.info("global error term = %s*%s^(-%s) ∈ %s*%s^(-%s)",
                        cst_big_circle, rad, name, error_term_big_circle,
                        mag_dom, name)
            add_error_term(bound, mag_dom, error_term_big_circle, n)

    if output == 'list':
        if beta == -infinity and not ignore_exponentially_small_term:
            bound.append((rad, [cst_big_circle]))
        return n0, bound
    else:
        try:
            asy = to_asymptotic_expansion(CB, name, bound, n0,
                                          beta, final_kappa, QQ(rad.lower()),
                                          cst_big_circle)
        except (ImportError, ValueError):
            raise RuntimeError(f"conversion of bound {bound} to an asymptotic "
                               "expansion failed, try with output='list' or a "
                               "newer Sage version")
        return asy

################################################################################
# Tests
################################################################################

def eval_bound(bound, n_num, prec=53):
    r"""
    Evaluation of a bound in "list" form
    """
    CBFp = ComplexBallField(prec)
    list_eval = [rho**(-n_num) * sum(term.subs(n=n_num) for term in ser)
                 for rho, ser in bound]
    return CBFp(sum(list_eval))

# TODO: better test the tester
def check_seq_bound(asy, ref, indices=None, *, verbose=False, force=False):
    r"""
    Test that an asymptotic expansion with error bound contains a reference
    sequence
    """
    Coeff = asy.parent().coefficient_ring
    myCBF = Coeff.complex_field()
    myRBF = myCBF.base()
    # An asymptotic ring with exponents etc. in CBF instead of QQbar, to make it
    # possible to evaluate a^n, n^b
    BGG = cartesian_product([
        ExponentialGrowthGroup.factory(myCBF, 'n',
                                       extend_by_non_growth_group=True),
        # factors like n^(RBF*I) do not work (#32452)
        MonomialGrowthGroup.factory(myRBF, 'n'),
                                    # extend_by_non_growth_group=True),
        GrowthGroup('log(n)^ZZ')])
    if any(isinstance(factor.parent(), MonomialGrowthGroup)
           and factor.exponent.imag() != 0
           for term in asy.summands
           for factor in term.growth.factors()):
        raise NotImplementedError("need monomial non-growth group")
    BAsy = AsymptoticRing(BGG, myCBF)
    basy = BAsy(asy, simplify=False)
    exact_part = basy.exact_part()
    error_part = basy.error_part()
    assert len(error_part.summands) == 1
    bterm = error_part.summands.pop()
    assert isinstance(bterm, BTerm)
    error_ball = myCBF.zero().add_error(bterm.coefficient)
    (name,) = asy.variable_names()
    validity = bterm.valid_from[name]
    if indices is None:
        try:
            rb = int(len(ref))
        except TypeError:
            rb = validity + 10
        indices = range(validity, rb)
    for n in indices:
        if n < validity and not force:
            continue
        bn = myRBF(n)
        one = myRBF.one()
        refval = ref[n]
        asyval = exact_part.substitute({name: bn})
        err0 = bterm.growth._substitute_({name: bn, '_one_': one})
        relbound = (asyval - refval)/err0
        if relbound not in error_ball:
            absbound = asyval + error_ball*err0
            if absbound.contains_exact(refval):
                # this may be a false warning due to overestimation at
                # evaluation time
                print(f"{name} = {n}, magnitude of interval rescaled "
                        f"error {relbound} is larger than {bterm.coefficient}")
            else:
                # definitely wrong
                msg = (f"{name} = {n}, computed enclosure "
                        f"{absbound} does not contain reference value {refval}"
                        f" ≈ {myCBF(refval)}")
                if force:
                    print(msg)
                else:
                    raise AssertionError(msg)
        else:
            if verbose:
                print(f"{name} = {n}, {relbound} contained in {error_ball}")

def test_monomial(alpha, beta, zeta=1, order=4, compare=True, big_circle=False):
    Pol, z = PolynomialRing(QQ, 'z').objgen()
    Dz = OreAlgebra(Pol, 'Dz').gen()
    dop = ((z - zeta)*Dz + alpha)**(1 + beta)
    rat = 1/(1-SR(z)/zeta)
    expr = rat**alpha*log(rat)**beta
    ini = list(expr.series(SR(z), 1+beta).truncate().polynomial(QQbar))
    assert CBF(dop(expr)(z=pi)).contains_zero()
    asy = bound_coefficients(dop, ini, order=order,
                             ignore_exponentially_small_term=not big_circle)

    if compare:
        ref = asymptotic_expansions.SingularityAnalysis('n', zeta=zeta,
                    alpha=alpha, beta=beta, normalized=False,
                    precision=order*(beta+1))
        ref = ref.parent().change_parameter(coefficient_ring=CBF)(ref)
        ok = all(term.coefficient.contains_zero()
                 for term in (asy.expand() - ref).summands if term.is_exact())
    else:
        ok = None
    return ok, asy
