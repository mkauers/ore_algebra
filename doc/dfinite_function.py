
from sage.all import *
from sage.misc.preparser import preparse_file

exec(preparse_file(open("../dfinite_function.sage").read()))

del allocatemem
