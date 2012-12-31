==================
PyCPX Introduction
==================

PyCPX is a python wrapper for the CPlex optimization suite that
focuses on ease of use and seamless integration with numpy_.  It
allows one to naturally specify linear and quadratic problems over
real, boolean, and integer variables.  PyCPX allows one to naturally
express such programs using numpy and natural python constructs.

It requires IBM's `ILog Concert Technology`_ Suite, part of the CPlex
Optimization Suite, which is available for free to eligible academic
researchers under IBM's Academic Initiative program.

PyCPX Distinctives
==================

- Seamless integration with Numpy arrays.

- Easy and natural syntax using CPlex Concert technology.

- Written in cython_/C++ for speed.

- The :ref:`API` is well documented and covered by unit tests.

- Licensed under the LGPL open source license.

Short Example
=============

A brief example to wet the appetite::

  >>> import numpy as np
  >>> from PyCPX import CPlexModel
  >>>
  >>> A = np.array([[1,0,0], [1,1,0], [1,1,1]])
  >>> b = np.array([1,2,3])
  >>> 
  >>> m = CPlexModel()
  >>> 
  >>> x = m.new(3)
  >>> t = m.new()
  >>> 
  >>> m.constrain( abs((A*x - b)) <= t)
  >>> m.minimize(t)
  0.0
  >>> m[x]
  array([ 1.,  1.,  1.])

A more detailed example::

  >>> from PyCPX import CPlexModel
  >>> from numpy import array, arange
  >>> 
  >>> A = 2*arange(1,10).reshape( (3, 3) )
  >>> m = CPlexModel()
  >>> 
  >>> X = m.new( (3, 3), vtype = int)
  >>> u = m.new( 3, vtype = int)
  >>> s = m.new(vtype = int)
  >>> 
  >>> m.constrain(s <= A.T * X <= 10*s)
  >>> m.constrain(1 <= X.sum(axis = 1) <= u)
  >>> 
  >>> m.minimize(u.sum())
  3.0
  >>> m[X]
  matrix([[-2.,  3.,  0.],
	  [ 0.,  0.,  1.],
	  [ 1.,  0.,  0.]])
  >>> m[u]
  array([ 1.,  1.,  1.])
  >>> m[s]
  2.0
  >>> m[X[0,0]]
  -2.0


.. toctree::
    :maxdepth: 2
    :hidden:
    
    self
    api
    download
    license

.. _cython: http://www.cython.org/
.. _numpy: http://numpy.scipy.org/
.. _ILog Concert Technology: http://www-01.ibm.com/software/integration/optimization/cplex-optimizer/interfaces/#concert_technology
