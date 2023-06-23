# vim: tw=80
r"""
Geometric predicates
"""

def orient2d_interval(a, b, c):
    im = ((c - a)/(b - a)).imag()
    z = im.parent().zero()
    if im > z:
        return +1
    elif im < z:
        return -1
    elif im.is_zero():
        return 0
    raise ValueError("unable to determine orientation")

def in_triangle(orient2d, a, b, c, z):
    try:
        oa = orient2d(b, c, z)
    except ValueError:
        oa = None
    try:
        ob = orient2d(c, a, z)
    except ValueError:
        ob = None
    try:
        oc = orient2d(a, b, z)
    except ValueError:
        oc = None
    if oa is not None and ob is not None and oc is not None:
        return (   oa >= 0 and ob >= 0 and oc >= 0
                or oa <= 0 and ob <= 0 and oc <= 0)
    # If only one of oa, ob, oc is None, typically because z lie on the line
    # supporting one of the edges and we are using an approximate zero-test, we
    # may still be able to conclude that z is outside the triangle.
    if ((    (oa is not None and oa < 0)
         or  (ob is not None and ob < 0)
         or  (oc is not None and oc < 0))
        and ((oa is not None and oa > 0)
         or  (ob is not None and ob > 0)
         or  (oc is not None and oc > 0))):
        return False
    raise ValueError("unable to decide")
