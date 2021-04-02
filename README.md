Ore algebra
============

https://github.com/mkauers/ore_algebra/

Description
-----------

A Sage implementation of Ore algebras, Ore polynomials, and differentially
finite functions

Main features for the most common algebras include basic arithmetic and actions;
gcrd and lclm; D-finite closure properties; creative telescoping; natural
transformations between related algebras; guessing; desingularization; solvers
for polynomials, rational functions and (generalized) power series. Univariate
differential operators also support the numerical computation of analytic
solutions with rigorous error bounds and related features.

License
-------

Distributed under the terms of the GNU General Public License (GPL, see the
COPYING file), either version 2 or (at your option) any later version

- http://www.gnu.org/licenses/

Requirements
------------

Sage 8.7 or later is recommended. Some features should work with older versions.

Installation
------------

### With Sage built from source or binaries from sagemath.org

To download and install the latest development version on a system where Sage
was built from source or installed from official packages, run

    sage -pip install git+https://github.com/mkauers/ore_algebra.git

or

    sage -pip install --user git+https://github.com/mkauers/ore_algebra.git

The optional `--user` flag causes the package to be installed in your `.sage`
directory instead of the Sage installation tree.

Alternatively, run (square brackets indicate optional flags)

    sage -pip install [--user] [--editable] .

from the root of a local git checkout. The `--editable` flag causes the
"installed" version to point to your local checkout, making it easier,
if you edit the code, to run the modified version. See the pip documentation
for more installation options.

Microsoft Windows users should run the above commands in a "SageMath shell", see

- https://wiki.sagemath.org/SageWindows

Apple macOS users may need additional steps before they are able to add external
packages to their Sage installations. See

- https://github.com/3-manifolds/fix_mac_sage/releases
- https://ask.sagemath.org/question/51130

for more information.

### With Sage installed from operating system packages

If your copy of Sage comes from operating system packages (e.g., Debian, Ubuntu,
Gentoo, or Arch Linux packages), try replacing `sage -pip` by `pip` or `pip3`
above.

You may need development packages that are not automatically installed as
dependencies of the main SageMath package, e.g., a C/C++ compiler, the Cython
compiler, and header files for Linbox, Pari, and Singular.

For example, Debian and Ubuntu users need to install the `liblinbox-dev`,
`libpari-dev` and `libsingular-dev` packages (and possibly others) in addition
to the `sagemath package`. The command `apt build-dep sagemath` can be used to
install all necessary development packages (and more).

Arch Linux users will need at least the `cython` and `python-pkgconfig`
packages.

### Using ore_algebra without installation

To use ore_algebra directly from a git checkout (without installation), run

    sage -python setup.py build_ext --inplace

from the checkout, and add the `src/` directory to your Python `sys.path`.

ore_algebra contains compiled (Cython) modules which are automatically built as
part of the installation procedure. Installation will fail if they cannot be
built. Only some specific features depend on these modules, though, and the core
features should work even if Cython modules are unavailable.

Documentation
-------------

The documentation generated from the doc strings is available online at

- http://www.algebra.uni-linz.ac.at/people/mkauers/ore_algebra

Testing
-------

To run the test suite, install the package and run the command

    sage -tp --long --force-lib src/

at the root of the git checkout.

Contact
-------

Manuel Kauers <manuel@kauers.de>
