
from sage.all import *
from sage.misc.preparser import preparse_file

os.chdir("..")

try:
    exec(preparse_file(open("dfinite_function.sage").read()))
except:
    pass

os.chdir("doc")

del allocatemem
