
from sage.all import *
from sage.misc.preparser import preparse_file

exec(preparse_file(open("../ore_algebra.sage").read()))

del allocatemem
