r"""
Local solutions
"""

import collections

FundamentalSolution = collections.namedtuple(
    'FundamentalSolution',
    ['valuation', 'log_power', 'value'])

def sort_key_by_asympt(sol):
    return sol.valuation.real(), -sol.log_power, sol.valuation.imag()

