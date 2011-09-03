.. _API:

===
API
===

.. currentmodule:: pycpx

Models are instances of the class :class:`CPlexModel`.  Variables are
created using :meth:`CPlexModel.new`.  Variables, combinations of
variables, and mathematical formulas on variables are called
:ref:`Expressions`.  Constraints are interactions of expressions and
added using :meth:`CPlexModel.constrain`. The model is typically
solved using :meth:`CPlexModel.minimize` or
:meth:`CPlexModel.maximize`.  Finally, values of expressions are then
retrieved with :meth:`CPlexModel.value` or simply ``[]`` notation.

Model Creation
==============

.. method:: CPlexModel.__init__(self, verbosity = 2)

  Creates a new empty model.

  The verbosity level may be passed as a special parameter; see
  :meth:`setVerbosity` for a description of the possible values.

  When there is a problem instantiating a model or starting CPlex, an
  exception is raised.  Sometimes, additional error messages or
  warnings may be printed, and passing this parameter here allows for
  more debugging information in this

.. automethod:: CPlexModel.setVerbosity(self, verbosity)

Variables
=========

.. automethod:: CPlexModel.new(self, size = s_scalar, vtype = 'real', lb = None, ub = None, name = None)
	
.. _expressions:

Expressions
===========

An Expression object in PyCPX is any block of variables or an
expression involving such variables formed by allowable algebraic
manipulations.  An example is ``A*x + y``, where ``A`` is a numpy_
matrix and ``x`` and ``y`` are variables.  Constraints 

Internally, expressions are always represented by 2d arrays; vectors
are either column vectors or row vectors and scalars are 1x1 arrays.
As such, they can either behave as arrays (in which products are done
elementwise) or as matrices, the default (in which products represent
matrix products).  They default to matrix mode; the :meth:`A` and
:meth:`M` properties switch between them.  

Supported operations include ``+``, ``*``, ``-``, ``/``.  Comparison
operators, such as ``<=``, ``>=``, or ``==``, create constraints to be
passed to :meth:`CPlexModel.constrain`

.. automethod:: _CPlexExpression.dot(self, expression)

.. automethod:: _CPlexExpression.mult(self, expression)

.. automethod:: _CPlexExpression.T(self)

.. automethod:: _CPlexExpression.transpose(self)

.. automethod:: _CPlexExpression.A(self)

.. automethod:: _CPlexExpression.M(self)

.. automethod:: _CPlexExpression.shape(self)

.. automethod:: _CPlexExpression.size(self)

Reduction Operations
--------------------

.. automethod:: _CPlexExpression.sum(self)

.. automethod:: _CPlexExpression.mean(self)

.. automethod:: _CPlexExpression.max(self)

.. automethod:: _CPlexExpression.min(self)

.. automethod:: _CPlexExpression.abs(self)

.. automethod:: _CPlexExpression.copy(self)

Constriants
===========

.. automethod:: CPlexModel.constrain(self, *constraints)

.. automethod:: CPlexModel.removeConstraint(self, *constraints)

Optimizing the Model
====================

.. automethod:: CPlexModel.solve(self, objective, maximize = None, minimize = None, recycle_variables = False, recycle_basis = True, starting_dict = {}, basis_file = None, algorithm = "auto")

.. automethod:: CPlexModel.maximize(self, objective, **options)

.. automethod:: CPlexModel.minimize(self, objective, **options)

.. automethod:: CPlexModel.getSolverTime(self)

.. automethod:: CPlexModel.getNIterations(self)

.. automethod:: CPlexModel.saveBasis(self, filename)

Variable Retrieval
==================

.. automethod:: CPlexModel.value(self, var_block_or_expression)

Model Information
=================

.. automethod:: CPlexModel.asString(self)

Exceptions
==========

.. autoclass:: CPlexException

.. autoclass:: CPlexInitError

.. autoclass:: CPlexNoSolution

.. _numpy: http://numpy.scipy.org/
