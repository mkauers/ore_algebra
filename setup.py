from distutils.core import setup
from distutils.cmd import Command
from setuptools import Extension
from Cython.Build import cythonize

try:
    import sage.env
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
    ext_modules = cythonize(
        [Extension(
            "*",
            ["src/ore_algebra/analytic/*.pyx"],
            extra_compile_args=['-std=c++11'],
        )],
        aliases = sage.env.cython_aliases(),
    ),
    include_dirs = sage.env.sage_include_directories(),
    cmdclass = {'test': TestCommand},
)
