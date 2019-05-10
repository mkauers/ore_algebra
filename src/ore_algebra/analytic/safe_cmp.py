# -*- coding: utf-8 - vim: tw=80
r"""
Safe comparisons
"""

# Copyright 2015 Marc Mezzarobba
# Copyright 2015 Centre national de la recherche scientifique
# Copyright 2015 Université Pierre et Marie Curie
#
# Distributed under the terms of the GNU General Public License (GPL) either
# version 2, or (at your option) any later version
#
# http://www.gnu.org/licenses/

__all__ = ["safe_lt", "safe_le", "safe_gt", "safe_ge", "safe_eq", "safe_ne"]

import logging

from sage.structure.element import parent

logger = logging.getLogger(__name__)

def _check_parents(a, b):
    # comparison between different parent may be okay if the elements are
    # instances of the same class (e.g., balls with different precisions)
    if parent(a) is not parent(b) and type(a) is not type(b):
        raise TypeError("unsafe comparison", parent(a), parent(b))

def safe_lt(a, b):
    _check_parents(a, b)
    return a < b

def safe_le(a, b):
    _check_parents(a, b)
    return a <= b

def safe_gt(a, b):
    _check_parents(a, b)
    return a > b

def safe_ge(a, b):
    _check_parents(a, b)
    return a >= b

def safe_eq(a, b):
    if parent(a) is not parent(b):
        logger.debug("comparing elements of %s and %s",
                     parent(a), parent(b))
        return False
    else:
        return a == b

def safe_ne(a, b):
    if parent(a) is not parent(b):
        logger.debug("comparing elements of %s and %s",
                     parent(a), parent(b))
        return True
    else:
        return a != b


