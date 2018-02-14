
try:
    import sage.all
except ImportError:
    raise ValueError("this package should be installed inside Sage")

from distutils.core import setup
from distutils.cmd import Command

class TestCommand(Command):
    user_options = []

    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        import subprocess

        if subprocess.call(['sage', '-tp', '--long', '--force-lib', 'src/']):
            raise SystemExit("Doctest failures")


setup(
    name = "ore_algebra",
    version = "0.3",
    author = "Manuel Kauers, Maximilian Jaroschek, Fredrik Johansson",
    author_email = "manuel@kauers.de",
    license = "GPL",
    packages = ["ore_algebra", "ore_algebra.analytic", "ore_algebra.analytic.examples"],
    package_dir = {'': 'src/'},
    cmdclass = {'test': TestCommand}
    )
