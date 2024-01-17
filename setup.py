import sys
import warnings

from setuptools import setup
from setuptools import Command
from setuptools import Extension
from Cython.Build import cythonize

try:
    import sage.env
    import sage.version
except ImportError:
    raise ValueError("this package requires SageMath")

class TestCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        if subprocess.call(['sage', '-tp', '--force-lib', 'src/']):
            raise SystemExit("Doctest failures")

def do_cythonize():
    return cythonize(
            [Extension(
                "*",
                ["src/ore_algebra/analytic/*.pyx"],
                extra_compile_args=['-std=c++11'],
            )],
            aliases = sage.env.cython_aliases(),
        )

if list(map(int, sage.version.version.split('.')[:2])) < [10, 2]:
    # Unfortunately, pip does not display this warning by default. But we will
    # warn about this again when the user tries to call one of the affected
    # functions.
    warnings.warn(
        f"Found SageMath version {sage.version.version}. The Cython extensions "
        "in ore_algebra now require SageMath >= 10.2. The ore_algebra package "
        "will be installed with Cython extensions disabled, making numerical "
        "evaluation and related features slow. To use the full version of "
        "ore_algebra, upgrade SageMath or downgrade ore_algebra to git commit "
        "73a430aaf.")
    extensions = []
else:
    try:
        from sage.misc.package_dir import cython_namespace_package_support
        with cython_namespace_package_support():
            extensions = do_cythonize()
    except ImportError:
        extensions = do_cythonize()

setup(
    name = "ore_algebra",
    version = "0.5",
    author = "Manuel Kauers, Maximilian Jaroschek, Fredrik Johansson",
    author_email = "manuel@kauers.de",
    license = "GPL",
    packages = [
        "ore_algebra",
        "ore_algebra.analytic",
        "ore_algebra.analytic.examples",
        "ore_algebra.examples",
    ],
    package_dir = {'': 'src/'},
    ext_modules = extensions,
    include_dirs = sage.env.sage_include_directories(),
    cmdclass = {'test': TestCommand},
    zip_safe=False,
)
