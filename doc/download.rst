Download
========

Installation requires a working C compiler installed on the system.
It looks through the PATH environment variable to try to find the
CPlex and ILog Concert Technology libraries.

The easiest way to install PyCPX is through the python setuptools
``easy_install`` utility by typing::

    easy_install pycpx

You can also install treedict by directly downloading a source tarball
from the python cheeseshop at http://pypi.python.org/pypi/pycpx/. 

For developers, a `git`_ source repository is available on `github`_.
You can clone that repository with the following command::

    git clone git://github.com/hoytak/pycpx.git

This repository does not include the generated cython_ source; you
need to have cython_ installed and pass ``--cython`` to setup.py to
compile it.

Bug reports, questions, or comments are very welcome and can be
submitted at http://github.com/hoytak/pycpx/issues.  Feature additions
and further development are also quite welcome!  Please email me at
hoytak@gmail.com if you have any questions.

Changelog
---------

Version 0.03 fixes a critical bug in array slicing.

Version 0.02 fixes several small bugs and gives vast speed
improvements for model creation in many models. 


.. _ILog Concert Technology: http://www-01.ibm.com/software/integration/optimization/cplex-optimizer/interfaces/#concert_technology
.. _github: http://github.com/
.. _git: http://git-scm.com/
.. _cython: http://www.cython.org/
