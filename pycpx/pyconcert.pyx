from numpy cimport ndarray as ar, \
    int_t, uint_t, int32_t, uint32_t, int64_t, uint64_t, float_t

cimport cython

from numpy import int_, int32,uint32,int64, uint64, float32, float64,\
    uint, empty, ones, zeros, uint, arange, isscalar, amax, amin, \
    ndarray, array, asarray, isfinite, argsort, matrix, nan, inf, float_

import numpy.random as rn
import tempfile
import os

cdef object issparse

try:
    import scipy.sparse
    issparse = scipy.sparse.issparse
except ImportError:
    issparse = lambda x: False

################################################################################
# The following defines the external definitions from the cplex
# concert wrapper

cdef extern from "cplex_interface.hpp":

    # The Ilo stuff we actually do need.  Most of it is handled using the templated wrappers
    cdef double IloInfinity

    cdef cppclass ostream:
        pass
    
    cdef ostream cout, cerr
    
    cdef cppclass IloEnv:
        IloEnv()
        void setOut(ostream&)
        void setWarning(ostream&)
        void setError(ostream&)
        ostream getNullStream()

    cdef cppclass IloObjective:
        pass

    # Expressions
    cdef cppclass IloExpr:
        IloObjective asObjective()

    cdef cppclass IloExprArray:
        IloExprArray(IloEnv env, long n)
        IloExpr& operator[](long i)

    cdef enum NumType "IloNumVar::Type":
        Float "IloNumVar::Float"
        Int   "IloNumVar::Int"
        Bool  "IloNumVar::Bool"

    cdef cppclass IloNumArray:
        IloNumArray(IloEnv env, long)
        double& operator[](long)

    cdef cppclass IloNumVar(IloExpr):
        void setName(char *)

    # We'll work with the type constraint to implement integer or boolean constraints
    cdef cppclass IloNumVarArray:
        IloNumVarArray(IloEnv env, IloNumArray lb, IloNumArray ub, NumType type)
        IloNumVar& operator[](long i)

    # Now parts of the array functions that we use for working with
    # the structures holding the data

    int MATRIX_MODE, ARRAY_MODE, DIAG_MODE, CONSTRAINT_MODE
    int OP_B_ADD, OP_B_MULTIPLY, OP_B_SUBTRACT, OP_B_DIVIDE
    int OP_B_MATRIXMULTIPLY, OP_B_ARRAYMULTIPLY
    int OP_B_EQUAL, OP_B_NOTEQ, OP_B_LT, OP_B_LTEQ, OP_B_GT, OP_B_GTEQ
    int OP_SIMPLE_FLAG, OP_SIMPLE_MASK

    int OP_U_NO_TRANSLATE, OP_U_NEGATIVE, OP_U_ABS
    int OP_R_SUM, OP_R_MAX, OP_R_MIN

    int MODEL_UNBOUNDED, MODEL_INFEASABLE, MODEL_UNBOUNDED_OR_INFEASABLE

    cdef cppclass MetaData:
        MetaData()
        MetaData(int, long, long)
        MetaData(int, long, long, long, long)
        int mode()
        long shape(int)
        long size()
        long stride(int)
        long offset()
        MetaData transposed()
        void printMetaData "print" ()
        bint matrix_multiplication_applies(MetaData md_right)

    MetaData newMetadata(int op_type, MetaData md1, MetaData md2, bint* okay)

    cdef cppclass Slice:
        Slice()
        Slice(long start, long stop, long step)

    cdef cppclass SliceFull:
        Slicefull()
        SliceFull(long size)

    cdef cppclass SliceSingle:
        SliceSingle()
        SliceSingle(long index)

    cdef cppclass ExpressionArray:
        ExpressionArray(IloEnv, MetaData)
        ExpressionArray(IloEnv, IloNumVarArray*, MetaData)
        ExpressionArray(ExpressionArray, MetaData)
        void set(long, long, IloNumVar)
        IloNumVar get(long, long)

        ExpressionArray* newFromGeneralSlice(Slice, Slice)
        ExpressionArray* newFromSlice(Slice, SliceFull)
        ExpressionArray* newFromSlice(SliceFull, Slice)
        ExpressionArray* newFromSlice(SliceFull, SliceFull)
        ExpressionArray* newFromSlice(Slice, SliceSingle)
        ExpressionArray* newFromSlice(SliceSingle, Slice)
        ExpressionArray* newFromSlice(SliceSingle, SliceSingle)
        ExpressionArray* newFromSlice(SliceFull, SliceSingle)
        ExpressionArray* newFromSlice(SliceSingle, SliceFull)
        ExpressionArray* newTransposed()
        ExpressionArray* newCopy()
        ExpressionArray* newAsArray()
        ExpressionArray* newAsMatrix()

        void setVariables(IloNumVarArray*)
        
        MetaData md()

    cdef cppclass ConstraintArray:
        ConstraintArray(IloEnv, MetaData)
        MetaData md()
        
    cdef cppclass NumericalArray:
        NumericalArray(IloEnv, double*, MetaData)
        MetaData md()
        
    cdef cppclass Scalar: 
        Scalar(IloEnv, double, MetaData)
        Scalar(IloEnv, double)
        MetaData md()

    ExpressionArray* newFromUnaryOp(ExpressionArray, int)
    ExpressionArray* newFromReduction(ExpressionArray, int op_type, int axis)

    void binary_op(int op, ConstraintArray&, NumericalArray, ExpressionArray)
    void binary_op(int op, ConstraintArray&, ExpressionArray, NumericalArray)
    void binary_op(int op, ConstraintArray&, Scalar, ExpressionArray)
    void binary_op(int op, ConstraintArray&, ExpressionArray, Scalar)
    void binary_op(int op, ConstraintArray&, ExpressionArray, ExpressionArray)

    void binary_op(int op, ExpressionArray&, NumericalArray, ExpressionArray)
    void binary_op(int op, ExpressionArray&, ExpressionArray, NumericalArray)
    void binary_op(int op, ExpressionArray&, Scalar, ExpressionArray)
    void binary_op(int op, ExpressionArray&, ExpressionArray, Scalar)
    void binary_op(int op, ExpressionArray&, ExpressionArray, ExpressionArray)
    
    void binary_op(int op, ConstraintArray&, ConstraintArray, ConstraintArray)

    cdef struct Status "CPlexModelInterface::Status":
        int error_code 
        char* message

    cdef cppclass string:
        string()
        char* c_str()

    cdef cppclass IntParam "IloCplex::IntParam":
        pass

    cdef IntParam RootAlg "IloCplex::RootAlg"

    cdef int CPX_ALG_NONE, CPX_ALG_AUTOMATIC, CPX_ALG_PRIMAL, CPX_ALG_DUAL, CPX_ALG_BARRIER,
    cdef int CPX_ALG_SIFTING, CPX_ALG_CONCURRENT, CPX_ALG_NET
        
    cdef cppclass CPlexModelInterface:
        CPlexModelInterface(IloEnv)
        Status addVariables(ExpressionArray)
        Status addConstraint(ConstraintArray)
        Status removeConstraint(ConstraintArray)
        Status setObjective(ExpressionArray, bint)
        Status solve()
        Status solve(double*)
        Status setParameter(IntParam, int value)
        Status getValues(NumericalArray&, ExpressionArray)
        Status setStartingValues(ExpressionArray&, NumericalArray&)
        Status readBasis(char *filename)
        Status writeBasis(char *filename)
        bint solved()
        string asString()
        double getObjectiveValue()
        long getNIterations()

    cdef Status newCPlexModelInterface(CPlexModelInterface**, IloEnv)

# Set up the environment
cdef IloEnv env = IloEnv() 

#Check if this is valid or not; if not, raise an import error.
# TODO....

cdef str s_scalar = "scalar"

################################################################################
# Try to reproduce the C-Plex type hierarchy.  This should be fun :-/

# Seems to be how the concert data model works:
#    Variable, Num*Variable, Num*Expr, Num + expr, etc. ---> expr
#    Expr <= num, Expr <= Expr, etc. ---> constraint
#    

cdef class CPlexModel
cdef class CPlexExpression
cdef class NumericalArrayWrapper

class CPlexException(Exception):
    """
    The base class for any exception raised due to problems with
    initializing the CPlex engine or creating, populating, or
    optimizing a model.

    Two more specific exception classes are :class:`CPlexInitError` or
    :class:`CPlexNoSolution`.  Standard Python exception classes are
    raised where appropriate.
    """

    pass

class CPlexInitError(CPlexException):
    """
    Raised if there is an error initializing the CPlex engine or a
    CPlex Model.
    """

    pass

class CPlexNoSolution(CPlexException):
    """
    Exception raised if the model is unbounded or infeasible.
    """

    pass

        
################################################################################
# The variable model;

DEF FLOAT_TYPE = 0
DEF INT_TYPE   = 1
DEF BOOL_TYPE  = 2

################################################################################
# Fast creation of these functions

cdef extern from "py_new_wrapper.h":
    CPlexExpression createBlankCPlexExpression "PY_NEW" (object t)
    NumericalArrayWrapper createBlankNumericalArrayWrapper "PY_NEW" (object t)

# This function should be the only way to instantiate a cplex expression

cdef inline CPlexExpression newCPEFromExisting(CPlexModel model, ExpressionArray* data):
    
    cdef CPlexExpression expr = createBlankCPlexExpression(CPlexExpression)
    expr.model              = model
    expr.is_simple          = False
    expr.data               = data
    expr.original_size      = None
    expr.key                = None
    expr.__array_priority__ = 20.1
    
    return expr

cdef inline CPlexExpression newCPE(CPlexModel model, MetaData md):
    return newCPEFromExisting(model, new ExpressionArray(env, md))

cdef inline CPlexExpression newCPEAsView(CPlexExpression cpx, MetaData md):
    return newCPEFromExisting(cpx.model, new ExpressionArray(cpx.data[0], cpx.data.md()))

cdef inline CPlexExpression newCPEwithVariables(CPlexModel model, MetaData md, IloNumVarArray* v):
    cdef CPlexExpression expr = newCPEFromExisting(model, new ExpressionArray(env, v, md))
    expr.is_simple  = True
    return  expr
    
cdef inline CPlexExpression newCPEFromCPEWithSameProperties(CPlexExpression src, ExpressionArray* data):
    cdef CPlexExpression expr = newCPEFromExisting(src.model, data)
    expr.is_simple = src.is_simple
    return expr
    

################################################################################
# For when we need a good numerical wrapper

cdef class NumericalArrayWrapper(object):
    cdef NumericalArray *data
    cdef ar X

    def __init__(self):
        raise Exception("NumericalArrayWrapper not meant to be instantiated directly.")

    def __dealloc__(self):
        del self.data

cdef inline NumericalArrayWrapper newNAW(NumericalArray* data, ar X):

    cdef NumericalArrayWrapper naw = createBlankNumericalArrayWrapper(NumericalArrayWrapper)
    
    naw.data        = data
    naw.X           = X

    return naw

cdef inline MetaData metadataFromNDArray(ar X, bint is_matrix):
    
    X = asarray(X, dtype=float_)

    cdef long itemsize = X.itemsize

    # See if we need to do an upcast
    return MetaData(MATRIX_MODE if is_matrix else ARRAY_MODE,
                    X.shape[0], 1 if X.ndim == 1 else X.shape[1],
                    (<long>X.strides[0])/itemsize,
                    1 if X.ndim == 1 else (<long>X.strides[1])/itemsize)


cdef NumericalArrayWrapper newCoercedNumericalArray(Xo, MetaData md):
    # Attempts to return a NumericalArray, checking it against MetaData md 

    if type(Xo) is list:
        Xo = asarray(Xo)
    elif isscalar(Xo):
        Xo = asarray([Xo])
    elif issparse(Xo):
        Xo = Xo.todense()        
    else:
        if type(Xo) is not ndarray:
            raise TypeError("Unable to understand numerical array value.")

    cdef ar X = asarray(Xo, dtype=float_)

    cdef MetaData Xmd = metadataFromNDArray(X, type(Xo) is matrix)

    # Now raise an issue if things are not compatible
    if not ( (md.shape(0) == Xmd.shape(0) and md.shape(1) == Xmd.shape(1))
             or Xmd.shape(0) == Xmd.shape(1) == 1):

        # One way out
        if X.ndim == 1 and md.shape(0) == Xmd.shape(1) and md.shape(1) == Xmd.shape(0):
            Xmd = Xmd.transposed()
        else:
            raise IndexError("Incompatible array indices (%d, %d), needs (%d, %d)."
                             % (Xmd.shape(0), Xmd.shape(1), md.shape(0), md.shape(1)))
    
    cdef NumericalArray *Xna = new NumericalArray(env, (<double*>(X.data)), Xmd)

    return newNAW(Xna, X)

################################################################################
# Operations

cdef dict _op_type_strings = {
    OP_B_ADD      : "+",
    OP_B_SUBTRACT : "-",
    OP_B_MULTIPLY : "*",
    OP_B_DIVIDE   : "/",
    OP_B_EQUAL    : "==",
    OP_B_NOTEQ    : "!=",
    OP_B_LTEQ     : "<=",
    OP_B_LT       : "<",
    OP_B_GT       : ">",
    OP_B_GTEQ     : ">=" }

cdef str opTypeStrings(int op_code):
    return _op_type_strings[op_code & OP_SIMPLE_MASK]

cdef CPlexExpression newEmptyExpression(
    int op_type, CPlexModel model, MetaData md1, MetaData md2):

    cdef bint okay = False
    cdef MetaData md_dest = newMetadata(op_type, md1, md2, &okay)

    if not okay:
        if ((op_type & OP_SIMPLE_MASK) == OP_B_MULTIPLY and md1.matrix_multiplication_applies(md2)
            or (op_type & OP_SIMPLE_MASK) == OP_B_MATRIXMULTIPLY):
            
            raise ValueError("Indexing error in dot product: Left shape = (%d, %d), right shape = (%d, %d)."
                             % (md1.shape(0), md1.shape(1), md2.shape(0), md2.shape(1)))
        else:
            raise ValueError("Indexing error in '%s': Left shape = (%d, %d), right shape = (%d, %d)."
                             % (opTypeStrings(op_type),
                                md1.shape(0), md1.shape(1), md2.shape(0), md2.shape(1)))

    return newCPE(model, md_dest)

################################################################################
# Now classes for expression interaction, constraint arrays, etc.

cdef inline CPlexExpression expression_op_expression(
    int op_type, CPlexExpression expr1, CPlexExpression expr2):

    if expr1.model is not expr2.model:
        raise ValueError("Cannot combine expressions from two different models.")

    cdef CPlexExpression dest = newEmptyExpression(op_type, expr1.model, expr1.data.md(), expr2.data.md())

    binary_op(op_type, dest.data[0], expr1.data[0], expr2.data[0])

    return dest


cdef inline CPlexExpression expression_op_array(
    int op_type, CPlexExpression expr, Xo, bint reverse):

    cdef ar X = Xo

    if X.ndim >= 3:
        raise ValueError("Cannot work with arrays/matrices of dimension >= 3.")

    X = asarray(X, dtype=float_)

    # See if we need to do an upcast
    cdef MetaData Xmd = metadataFromNDArray(X, type(Xo) is matrix)
    cdef MetaData Xmdt
    cdef NumericalArray *Xna = new NumericalArray(env, (<double*>(X.data)), Xmd)

    cdef CPlexExpression dest

    # First see if we can make it a "simple" type
    cdef bint matrix_multiplication = False
    cdef bint is_simple 

    try:
        if reverse:
            try:
                dest = newEmptyExpression(op_type, expr.model, Xmd, expr.data.md())
                matrix_multiplication = Xmd.matrix_multiplication_applies(expr.data.md())
            except ValueError, ve:
                if X.ndim == 1:
                    try:
                        Xmdt = Xmd.transposed()
                        dest = newEmptyExpression(op_type, expr.model, Xmdt, expr.data.md())
                        matrix_multiplication = Xmdt.matrix_multiplication_applies(expr.data.md())
                    except ValueError:
                        raise ve
                else:
                    raise
                
            is_simple = expr.is_simple or not matrix_multiplication
            binary_op(op_type | (OP_SIMPLE_FLAG if is_simple else 0), dest.data[0], Xna[0], expr.data[0])
                        
        else:
            try:
                dest = newEmptyExpression(op_type, expr.model, expr.data.md(), Xmd)
                matrix_multiplication = expr.data.md().matrix_multiplication_applies(Xmd)
            except ValueError, ve:
                if X.ndim == 1:
                    try:
                        Xmdt = Xmd.transposed()
                        dest = newEmptyExpression(op_type, expr.model, expr.data.md(), Xmdt)
                        matrix_multiplication = expr.data.md().matrix_multiplication_applies(Xmdt)
                    except ValueError:
                        raise ve
                else:
                    raise

            is_simple = expr.is_simple or not matrix_multiplication
            binary_op(op_type | (OP_SIMPLE_FLAG if is_simple else 0), dest.data[0], expr.data[0], Xna[0])
            
    finally:
        del Xna

    # Need to determine when the simple flag can be propegated
    dest.is_simple = (expr.is_simple and not matrix_multiplication)
    
    return dest

cdef inline CPlexExpression expression_op_scalar(
    int op_type, CPlexExpression expr, double v, bint reverse):
    
    cdef Scalar *sc = new Scalar(env, v)
    cdef CPlexExpression dest

    try:
        if reverse:
            dest = newEmptyExpression(op_type, expr.model, sc.md(), expr.data.md())
            binary_op(op_type | OP_SIMPLE_FLAG, dest.data[0], sc[0], expr.data[0])

        else:
            dest = newEmptyExpression(op_type, expr.model, expr.data.md(), sc.md())
            binary_op(op_type | OP_SIMPLE_FLAG, dest.data[0], expr.data[0], sc[0])
        
    finally:
        del sc
        
    dest.is_simple = expr.is_simple

    return dest

############################################################
# The main control function for expressions
    
cdef CPlexExpression expr_var_op_var(int op_type, a1, a2):

    cdef CPlexExpression expr1 = None, expr2 = None

    if type(a1) is CPlexExpression:
        expr1 = a1

    if type(a2) is CPlexExpression:
        expr2 = a2

    # Got two expressions?
    if expr1 is not None and expr2 is not None:
        return expression_op_expression(op_type, expr1, expr2)
    
    elif expr1 is not None:
        if isinstance(a2, ndarray):
            return expression_op_array(op_type, expr1, a2, False)
        elif isscalar(a2):
            return expression_op_scalar(op_type, expr1, a2, False)
        else:
            raise TypeError("Unknown type: %s" % repr(type(a2))) 

    elif expr2 is not None:
        if isinstance(a1, ndarray):
            return expression_op_array(op_type, expr2, a1, True)
        elif isscalar(a1):
            return expression_op_scalar(op_type, expr2, a1, True)
        else:
            raise TypeError("Unknown type: %s" % repr(type(a1)))

    else:
        assert False        

##################################################
# This is the visible class of any expressions.

# This HAS to be a subclass of matrix (which is a subclass of ndarray)
# so that __radd__ and such operations override the left-oriented ones
# of ndarray.  This is partly because ndarray raises a typeerror
# instead of a NotImplemented error when the types don't match, so
# this isn't added correctly.


cdef inline setSliceParts(Slice* s, t, long md_size):
    cdef slice s0 = None
    cdef long ti
    
    if type(t) is slice:
        sl_is_slice = True
        s0 = t
    elif type(t) is Ellipsis:
        sl_is_slice = True
        s0 = slice(None,None,None)
    elif type(t) is int or type(t) is long:
        sl_is_slice = False
        ti = <long>t
        if ti > md_size:
            raise IndexError("Invalid index (%d >= %d) " % (ti, md_size))
        
        s0 = slice(ti,ti+1,1)
    else:
        raise TypeError("Index %s not understood." % t)

    cdef long shape, size, step

    shape, size, step = s0.indices(md_size)

    if shape == size:
        raise IndexError("Invalid index or range.")

    s[0] = Slice(shape, size, step)
    return sl_is_slice


cdef class CPlexExpression(object):

    cdef CPlexModel model
    cdef bint is_simple
    cdef ExpressionArray *data
    cdef object original_size
    cdef str key
    cdef readonly object __array_priority__

    def __init__(self):
        raise Exception("CPlexExpression not meant to be instantiated directly.")

    def __add__(self, v):
        return expr_var_op_var(OP_B_ADD, self, v)

    def __radd__(self, v):
        return expr_var_op_var(OP_B_ADD, v, self)
    
    def __sub__(self, v):
        return expr_var_op_var(OP_B_SUBTRACT, self, v)

    def __rsub__(self, v):
        return expr_var_op_var(OP_B_SUBTRACT, v, self)

    def __mul__(self, v):
        return expr_var_op_var(OP_B_MULTIPLY, self, v)

    def __rmul__(self, v):
        return expr_var_op_var(OP_B_MULTIPLY, v, self)

    def __div__(self, v):
        return expr_var_op_var(OP_B_DIVIDE, self, v)

    def __rdiv__(self, v):
        return expr_var_op_var(OP_B_DIVIDE, v, self)

    def dot(self, v):
        """
        Performs the matrix dot product with `v`, regardless of
        whether this expression or `v` is array or matrix type. 
        """
        return expr_var_op_var(OP_B_MATRIXMULTIPLY, v, self)

    def mult(self, v):
        """
        Performs coefficient-wise mutliply operation with `v`,
        regardless of whether this expression or `v` is array or
        matrix type.  This expresion and `v` must both have the same
        dimensions, unless one is a scalar (1x1 matrix/array), in
        which case it is repeated to match the shape of the other.
        """
        return expr_var_op_var(OP_B_ARRAYMULTIPLY, v, self)

    ##################################################
    # Generation of constraints

    def __richcmp__(a1, a2, int op):
        # <	0
        # ==	2
        # >	4
        # <=	1
        # !=	3
        # >=	5
        if op == 0:
            return cstr_var_op_var(OP_B_LT, a1, a2)
        elif op == 1:
            return cstr_var_op_var(OP_B_LTEQ, a1, a2)
        elif op == 2:
            return cstr_var_op_var(OP_B_EQUAL, a1, a2)
        elif op == 3:
            return cstr_var_op_var(OP_B_NOTEQ, a1, a2)
        elif op == 4:
            return cstr_var_op_var(OP_B_GT, a1, a2)
        elif op == 5:
            return cstr_var_op_var(OP_B_GTEQ, a1, a2)
        else:
            assert False

    # More special methods coming
    @property
    def T(self):
        """
        Returns the transpose of the current expression matrix.  This
        is the property form of :meth:`transpose()`.
        """

        return self.transpose()

    cpdef transpose(self):
        """
        Returns the transpose of the current expression matrix.
        """
        
        return newCPEFromCPEWithSameProperties(self, self.data.newTransposed())

    @property
    def A(self):
        """
        Returns the expression object as an array, causing it to
        interact with other objects using array semantics.  By
        default, expression objects are 2-d matrices (possibly n by 1
        column vectors or a 1 x n row vectors).

        Note that if at least one of the operands of the product
        operator, ``*``, is a matrix, matrix multiplication is
        performed.  Element-wise array operation is performed only if
        the two operands are arrays.  This behavior is identical to
        numpy, with the exception that 1x1 blocks are treated as
        scalars.
        """
        
        return newCPEFromCPEWithSameProperties(self, self.data.newAsArray())

    @property
    def M(self):
        """
        Returns the expression object as a matrix, causing it to
        interact with other objects using matrix semantics.  By
        default, expression objects are 2-d matrices (possibly n by 1
        column vectors or a 1 x n row vectors).

        Note that if at least one of the operands of the product
        operator, ``*``, is a matrix, matrix multiplication is
        performed.  Element-wise array operation is performed only if
        the two operands are arrays.  This behavior is identical to
        numpy, with the exception that 1x1 blocks are treated as
        scalars.
        """
        return newCPEFromCPEWithSameProperties(self, self.data.newAsMatrix())

    def __neg__(self):
        return newCPEFromCPEWithSameProperties(self, newFromUnaryOp(self.data[0], OP_U_NEGATIVE))

    @property
    def shape(self):
        """
        Returns the shape of the expression as a 2-tuple.
        """
        return (self.data.md().shape(0), self.data.md().shape(1))

    @property
    def size(self):
        """
        Returns the total number of elements in the current expression.
        """
        
        return self.data.md().size()

    def __hash__(self):

        # This is a little tricky.  We need to make variables
        # available as keys in a dictionary, but we must not allow
        # them to go on to test equality if it isn't valid, as
        # constraints always evaluate to True to allow chaining.
        # However, since all variable objects are unique by their key
        # string, we're in luck, as long as we keep key strings
        # identical as the same objects throughout the module.  To do
        # this, we run it through a dictionary in the model this
        # variable originated from, along with the metadata info
        # (which would handle all the slicing stuff).
        
        if self.key is None:
            raise NotImplemented

        return self.model._getKeyStringId(self.key, self.data.md())

    def __getitem__(self, key):

        cdef tuple t
        cdef Slice s0, s1
        cdef CPlexExpression new_cpx
        cdef bint sl0_is_slice = True, sl1_is_slice = True
        
        if type(key) is tuple:
            t = <tuple>key

            if len(t) == 1:
                return self.__getitem__(t[0])
            if len(t) != 2:
                raise IndexError("Expression arrays only 2 dimensional.")

            sl0_is_slice = setSliceParts(&s0, t[0], self.data.md().shape(0))
            sl1_is_slice = setSliceParts(&s1, t[1], self.data.md().shape(1))

            new_cpx = newCPEFromExisting(
                self.model, self.data.newFromGeneralSlice(s0, s1))

        else:
            sl0_is_slice = setSliceParts(&s0, key, self.data.md().shape(0))

            new_cpx = newCPEFromExisting(
                self.model, self.data.newFromSlice(s0, SliceFull(self.data.md().shape(1))))

        cdef long shape_0 = new_cpx.data.md().shape(0)
        cdef long shape_1 = new_cpx.data.md().shape(1)

        # Now that we have set new_cpx, we need to figure out the new size
        if self.original_size is not None:
            size = self.original_size
            
            if size == s_scalar:
                assert shape_0 == 1 and shape_1 == 1
                new_size = s_scalar
                
            elif (type(size) is tuple and len(<tuple>size) == 2):

                if sl0_is_slice and not sl1_is_slice:
                    assert shape_1 == 1
                    new_size = shape_0
                elif sl0_is_slice and not sl1_is_slice:
                    assert shape_0 == 1
                    new_size = shape_1
                elif not sl0_is_slice and not sl1_is_slice:
                    assert shape_0 == shape_1 == 1
                    new_size = s_scalar
                else:
                    new_size = (shape_0, shape_1)

            elif (type(size) is tuple and len(<tuple>size) == 1) or isscalar(size):
                assert shape_1 == 1

                if sl0_is_slice:
                    new_size = shape_0
                else:
                    new_size = s_scalar
                    
            else:
                assert False

            new_cpx.original_size = new_size

        new_cpx.key = self.key

        return new_cpx


    cpdef CPlexExpression sum(self, axis = None):
        """
        Returns an expression representing the sum of the current
        expression.  If `axis` is None (default), it is the sum of
        every element in the expression; otherwise, the sum is
        performed along the particular axis (0 or 1).  If ``X`` has
        shape ``(m, n)``, then ``X.sum(0)`` has shape ``(1,n)``.

        The sum can be used in constraints, the objective, or in
        retriving values.  For example, the following are all valid::

          m.constrain(X.sum(axis = 1) <= 10)

          m.minimize(X.sum())

          print m[X.sum(axis = 0)]

        """
        return newCPEFromExisting(self.model, newFromReduction(
            self.data[0],
            OP_R_SUM | (OP_SIMPLE_FLAG if self.is_simple else 0),
            -1 if axis is None else axis))

    def mean(self, axis = None):
        """
        Returns an expression representing the mean of the current
        expression.  If `axis` is None (default), it is the mean of
        every element in the expression; otherwise, the mean is
        performed along the particular axis (0 or 1).  If ``X`` has
        shape ``(m, n)``, then ``X.mean(0)`` has shape ``(1,n)``.

        The mean can be used in constraints, the objective, or in
        retriving values.  For example, the following are all valid::

          m.constrain(X.mean(axis = 1) <= 10)

          m.minimize(X.mean())

          print m[X.mean(axis = 0)]

        """

        cdef CPlexExpression sum_res = self.sum(axis)
        
        if axis == 0:
            return sum_res / self.data.md().shape(0)
        elif axis == 1:
            return sum_res / self.data.md().shape(1)
        else:
            return sum_res / self.data.md().size()

    def max(self, axis = None):
        """
        Returns an expression representing the maximum value of the
        current expression.  If `axis` is None (default), it is the
        maximum of every element in the expression; otherwise, it is
        the maximum value along the particular axis (0 or 1).  If
        ``X`` has shape ``(m, n)``, then ``X.max(0)`` has shape
        ``(1,n)``.

        `max()` can be used in constraints, the objective, or in
        retriving values.  For example, the following are all valid::

          m.constrain(X.max(axis = 1) <= 10)

          m.minimize(X.max())

          print m[X.max(axis = 0)]

        """
        return newCPEFromExisting(self.model, newFromReduction(
            self.data[0],
            OP_R_MAX | (OP_SIMPLE_FLAG if self.is_simple else 0),
            -1 if axis is None else axis))

    def min(self, axis = None):
        """
        Returns an expression representing the minimum value of the
        current expression.  If `axis` is None (default), it is the
        minimum of every element in the expression; otherwise, it is
        the minimum value along the particular axis (0 or 1).  If
        ``X`` has shape ``(m, n)``, then ``X.min(0)`` has shape
        ``(1,n)``.

        `min()` can be used in constraints, the objective, or in
        retriving values.  For example, the following are all valid::

          m.constrain(X.min(axis = 1) <= 10)

          m.minimize(X.min())

          print m[X.min(axis = 0)]

        """
        return newCPEFromExisting(self.model, newFromReduction(
            self.data[0],
            OP_R_MIN | (OP_SIMPLE_FLAG if self.is_simple else 0),
            -1 if axis is None else axis))

    def __abs__(self):
        return newCPEFromCPEWithSameProperties(self, newFromUnaryOp(self.data[0], OP_U_ABS))

    def abs(self):
        """
        Returns an expression representing the absolute value,
        elementwise, of the current expression.  The returned value
        has the same shape and properties as the current expression.

        Can also be called simply using the ``abs()`` builtin function.
        """
        
        return self.__abs__()

    def copy(self):
        """
        Returns a copy of the current expression.
        """
        
        return newCPEFromCPEWithSameProperties(self, newFromUnaryOp(self.data[0], OP_U_NO_TRANSLATE))

    def __len__(self):
        return self.data.md().size()

    def __len__(self):
        return self.size()

    def __pos__(self):
        return self

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.copy()

    def __call__(self, *args):
        if args:
            return self.model[self[args]]
        else:
            return self.model[self]


################################################################################
# 

# Set up a dictionary mapping the variable type to an easy thing 
cdef dict _vartype_map = {
    float     : FLOAT_TYPE,
    "float"   : FLOAT_TYPE,
    "real"    : FLOAT_TYPE,
    "double"  : FLOAT_TYPE,
    "f"       : FLOAT_TYPE,
    "d"       : FLOAT_TYPE,
    # Integer types
    int       : INT_TYPE,
    long      : INT_TYPE,
    "int"     : INT_TYPE,
    "integer" : INT_TYPE,
    "i"       : INT_TYPE,

    # Boolean types
    bool      : BOOL_TYPE,
    "bool"    : BOOL_TYPE,
    "binary"  : BOOL_TYPE,
    "boolean" : BOOL_TYPE,
    "b"       : BOOL_TYPE}
           

cdef inline ar toDoubleArray_1d(a, str name, long required_size):
    cdef ar a_1
    cdef size_t i

    if issparse(a):
        a = a.todense()

    try:
        a_1 = asarray(a, dtype = float_)
    except ValueError:
        raise TypeError("`%s` must be convertable to 1d numpy array of length %d."
                        % (name, required_size))
    except TypeError:
        raise TypeError("`%s` must be convertable to 1d numpy array of length %d."
                        % (name, required_size))

    if a_1.size == 0:
        return None

    if a_1.ndim != 1:
        
        if a_1.ndim == 2 and (a_1.shape[0] == 1 or a_1.shape[1] == 1):
            a_1 = a_1.ravel()        
        else:
            raise TypeError("`%s` must be convertable to 1d numpy array of length %d (shape = (%s) )."
                            % (name, required_size,
                               ', '.join(["%d" % a_1.shape[i] for i in range(a_1.ndim)])))

    if a_1.shape[0] != required_size:
        raise TypeError("`%s` must be convertable to 1d numpy array of length %d (length = %d)."
                        % (name, required_size, a_1.shape[0]))

    return a_1

cdef CPlexExpression newVariableBlock(CPlexModel model, size, var_type, lower_bound,
                                      upper_bound, str name, str key):

    ########################################
    # First set the variable type

    cdef NumType cpx_var_type

    cdef long n, d_0, d_1

    if size == s_scalar:
        d_0 = d_1 = n = 1
        var_mode = "scalar"
        
    elif isscalar(size):
        d_0 = n = size
        d_1 = 1
        var_mode = "array"
        
    elif type(size) is tuple and len(<tuple>size) == 1:
        d_0 = n = (<tuple>size)[0]
        d_1 = 1
        var_mode = "array"
        
    elif type(size) is tuple and len(<tuple>size) == 2:
        d_0 = (<tuple>size)[0]
        d_1 = (<tuple>size)[1]
        n = d_0 * d_1
        var_mode = "matrix"

    else:
        raise ValueError("Size '%s' not understood." % str(size))

    cdef int var_type_n
    cdef long i
    cdef double d
    cdef ar[double] dv

    try:
        var_type_n = _vartype_map[var_type.lower() if type(var_type) is str else var_type]
    except KeyError:
        raise ValueError("Variable mode %s not recognized." % repr(var_type))

    if var_type_n == FLOAT_TYPE:
        cpx_var_type = Float
    elif var_type_n == INT_TYPE:
        cpx_var_type = Int
    elif var_type_n == BOOL_TYPE:
        cpx_var_type = Bool
    else:
        assert False

    ########################################
    # Now set the lower bounds

    cdef IloNumArray *lb = new IloNumArray(env, n)
    cdef ar[int_t, mode="c"] finite_elements = None

    if lower_bound is None:
        for 0 <= i < n:
            lb[0][i] = -IloInfinity

    elif isscalar(lower_bound):
        d = lower_bound
        for 0 <= i < n:
            lb[0][i] = d

    else:
        dv_r = toDoubleArray_1d(lower_bound, "lower_bound", n)

        if dv_r is None:
            for 0 <= i < n:
                lb[0][i] = -IloInfinity
        else:
            dv = dv_r

            finite_elements = empty(n, int_)
            isfinite(dv_r, finite_elements)

            for 0 <= i < n:
                lb[0][i] = dv[i] if finite_elements[i] else -IloInfinity

    ########################################
    # Now set the upper bounds

    cdef IloNumArray *ub = new IloNumArray(env, n)

    if upper_bound is None:
        for 0 <= i < n:
            ub[0][i] = IloInfinity

    elif isscalar(upper_bound):
        d = upper_bound
        for 0 <= i < n:
            ub[0][i] = d

    else:
        dv_r = toDoubleArray_1d(upper_bound, "upper_bound", n)

        if dv_r is None:
            for 0 <= i < n:
                ub[0][i] = IloInfinity
        else:
            dv = dv_r

            if finite_elements is None:
                finite_elements = empty(n, int_)
                
            isfinite(dv_r, finite_elements)

            for 0 <= i < n:
                ub[0][i] = dv[i] if finite_elements[i] else IloInfinity


    ########################################
    # Now get the variables
    cdef IloNumVarArray* v = new IloNumVarArray(env, lb[0], ub[0], cpx_var_type)

    del lb
    del ub

    ############################################################
    # Set names if applicable

    if name is not None:

        if n == 1:
            s = bytes(name)
            v[0][0].setName(s)

        elif var_mode == "array":
            for 0 <= i < d_0:
                for 0 <= j < d_1:
                    s = bytes("%s[%d]" % (name, i*d_1 + j))
                    v[0][i*d_1 + j].setName(s)
        else:
            for 0 <= i < d_0:
                for 0 <= j < d_1:
                    s = bytes("%s[%d,%d]" % (name, i,j))
                    v[0][i*d_1 + j].setName(s)

    ############################################################
    # Now initialize the base class with an expression consisting
    # of the new variable set.

    cdef CPlexExpression cpx = newCPEwithVariables(model, MetaData(MATRIX_MODE, d_0, d_1), v)

    for 0 <= i < d_0:
        for 0 <= j < d_1:
            cpx.data.set(i, j, v[0][i*d_1 + j])

    cpx.original_size = size
    cpx.key = key

    return cpx

def concatenate(list expression_list, int axis = 0):
    """
    Concatenates arrays along a particular axis.
    """

    cdef CPlexExpression cpe

    if len(expression_list) == 0:
        raise ValueError("List of expressions is empty!")

    if not (axis == 0 or axis == 1):
        raise ValueError("Axis must be 0 or 1.")

    cdef int same_axis = 1 - axis
    cdef CPlexExpression cpe_0 = (<CPlexExpression?>(expression_list[0]))
    cdef long same_axis_size = cpe_0.data.md().shape(same_axis)
    cdef CPlexModel model = cpe_0.model
    cdef int mode = ARRAY_MODE

    cdef list breaks = [None]*(len(expression_list) + 1)
    
    cdef size_t i, pos, pos_start = 0, pos_end = 0

    for i, cpe in enumerate(expression_list):
        
        if cpe.data.md().shape(same_axis) != same_axis_size:
            raise ValueError("All dimensions must be same, except along concatenation axis.")

        if cpe.model is not model:
            raise ValueError("Cannot combine variables from two different models.")

        if cpe.data.md().mode() == MATRIX_MODE:
            mode = MATRIX_MODE

        pos_start = pos_end
        pos_end += cpe.data.md().shape(axis)

        breaks[i] = (pos_start, pos_end)

    cdef size_t cat_axis_size = pos_end

    cdef MetaData md = MetaData(mode,
                                cat_axis_size if axis == 0 else same_axis_size,
                                same_axis_size if axis == 0 else cat_axis_size)

    cdef CPlexExpression dest = newCPE(model, md)

    cdef long j, k
    
    for i, cpe in enumerate(expression_list):
        pos_start, pos_end = breaks[i]

        if axis == 0:
            for j in range(0, pos_end - pos_start):
                for k in range(0, same_axis_size):
                    dest.data.set(pos_start + j, k, cpe.data.get(j,k))
        else:
            for j in range(0, same_axis_size):
                for k in range(0, pos_end - pos_start):
                    dest.data.set(j, pos_start + k, cpe.data.get(j,k))

    return dest


################################################################################
# Constraint creation functions

cdef class CPlexConstraint(object):
    cdef CPlexModel model
    cdef ConstraintArray *data
    cdef size_t id_left, id_right
    cdef CPlexConstraint hooked_constraint

    def __init__(self):
        raise Exception("Class CPlexConstraint not meant to be instantiated directly.")

    def __dealloc__(self):
        del self.data

    def __nonzero__(self):
        self.model.hook_id_1         = self.id_right
        self.model.hook_id_2         = self.id_left
        self.model.hooked_constraint = self

        return True

############################################################
# Creation / initialization

cdef extern from "py_new_wrapper.h":
    CPlexConstraint createBlankCPlexConstraint "PY_NEW" (object t)

cdef inline CPlexConstraint newCPC(CPlexModel model, MetaData md, left, right):
    
    cdef CPlexConstraint c = createBlankCPlexConstraint(CPlexConstraint)
    
    c.model               = model
    c.data                = new ConstraintArray(env, md)
    c.id_right            = id(right)
    c.id_left             = id(left)

    # Now see if we're pulling in an attached constraint
    if (model.hook_id_1 == c.id_left or model.hook_id_1 == c.id_right
        or model.hook_id_2 == c.id_left or model.hook_id_2 == c.id_right):
        
        c.hooked_constraint = model.hooked_constraint
        
    else:
        c.hooked_constraint = None
        
    model.hook_id_1 = id(None)
    model.hook_id_2 = id(None)
    model.hooked_constraint = None

    return c

cdef CPlexConstraint newEmptyConstraint(
    int op_type, CPlexModel model, left, MetaData md1, right, MetaData md2):

    cdef bint okay = False
    cdef MetaData md_dest = newMetadata(op_type, md1, md2, &okay)

    if not okay:
        raise ValueError("Indexing error for '%s' constraint: Left shape = (%d, %d), right shape = (%d, %d)."
                         % (opTypeStrings(op_type),
                            md1.shape(0), md1.shape(1), md2.shape(0), md2.shape(1)))

    return newCPC(model, md_dest, left, right)


################################################################################
# Operators

cdef CPlexConstraint cstr_expression_op_expression(
    int op_type, CPlexExpression expr1, CPlexExpression expr2):
    
    if expr1.model is not expr2.model:
        raise ValueError("Cannot combine expressions from two different models.")

    cdef CPlexConstraint dest = newEmptyConstraint(
        op_type, expr1.model, expr1, expr1.data.md(), expr2, expr2.data.md())

    binary_op(op_type, dest.data[0], expr1.data[0], expr2.data[0])

    return dest

cdef CPlexConstraint cstr_expression_op_array(
    int op_type, CPlexExpression expr, Xo, bint reverse):

    cdef ar X = Xo

    if X.ndim >= 3:
        raise ValueError("Cannot work with arrays/matrices of dimension >= 3.")

    X = asarray(X, dtype=float_)

    cdef long itemsize = X.itemsize

    # See if we need to do an upcast
    cdef MetaData Xmd = MetaData(CONSTRAINT_MODE,
                                 X.shape[0], 1 if X.ndim == 1 else X.shape[1],
                                 (<long>X.strides[0])/itemsize,
                                 1 if X.ndim == 1 else (<long>X.strides[1])/itemsize)

    cdef NumericalArray *Xna = new NumericalArray(env, (<double*>(X.data)), Xmd)

    cdef CPlexConstraint dest

    try:

        if reverse:
            try:
                dest = newEmptyConstraint(op_type, expr.model, Xo, Xmd, expr, expr.data.md())
            except ValueError, ve:
                if X.ndim == 1:
                    try:
                        dest = newEmptyConstraint(
                            op_type, expr.model, Xo, Xmd.transposed(), expr, expr.data.md())
                        
                    except ValueError:
                        raise ve
                else:
                    raise

            binary_op(op_type | OP_SIMPLE_FLAG, dest.data[0], Xna[0], expr.data[0])

        else:
            try:
                dest = newEmptyConstraint(
                    op_type, expr.model, expr, expr.data.md(), Xo, Xmd)
                
            except ValueError, ve:
                if X.ndim == 1:
                    try:
                        dest = newEmptyConstraint(
                            op_type, expr.model, expr, expr.data.md(), Xo, Xmd.transposed())
                        
                    except ValueError:
                        raise ve
                else:
                    raise

            binary_op(op_type | OP_SIMPLE_FLAG, dest.data[0], expr.data[0], Xna[0])
    finally:
        del Xna
        
    return dest

cdef CPlexConstraint cstr_expression_op_scalar(
    int op_type, CPlexExpression expr, v, bint reverse):

    cdef Scalar *sc = new Scalar(env, <double?>v)
    cdef CPlexConstraint dest

    try:
        if reverse:
            dest = newEmptyConstraint(op_type, expr.model, v, sc.md(), expr, expr.data.md())
            binary_op(op_type | OP_SIMPLE_FLAG, dest.data[0], sc[0], expr.data[0])

        else:
            dest = newEmptyConstraint(op_type, expr.model, expr, expr.data.md(), v, sc.md())
            binary_op(op_type | OP_SIMPLE_FLAG, dest.data[0], expr.data[0], sc[0])
    finally:
        del sc

    return dest

##################################################
# The main constraint operator class

cdef CPlexConstraint cstr_var_op_var(op_type, a1, a2):

    cdef CPlexExpression expr1 = None, expr2 = None

    if type(a1) is CPlexExpression:
        expr1 = <CPlexExpression>a1

    if type(a2) is CPlexExpression:
        expr2 = <CPlexExpression>a2

    if expr1 is not None and expr2 is not None:
        return cstr_expression_op_expression(op_type, expr1, expr2)

    elif expr1 is not None:
        if isinstance(a2, ndarray):
            return cstr_expression_op_array(op_type, expr1, a2, False)
        elif isscalar(a2):
            return cstr_expression_op_scalar(op_type, expr1, a2, False)
        elif issparse(a2):
            return cstr_expression_op_array(op_type, expr1, a2.todense(), False)
        else:
            raise TypeError("Iteraction with type %s not supported yet." % type(a2))
    elif expr2 is not None:
        if isinstance(a1, ndarray):
            return cstr_expression_op_array(op_type, expr2, a1, True)
        elif isscalar(a1):
            return cstr_expression_op_scalar(op_type, expr2, a1, True)
        elif issparse(a1):
            return cstr_expression_op_array(op_type, expr2, a1.todense(), True)
        else:
            raise TypeError("Iteraction with type %s not supported yet." % type(a1))
    else:
        assert False

################################################################################
# Now the model

cdef dict model_lookup = {
    "auto"       : CPX_ALG_AUTOMATIC,
    "automatic"  : CPX_ALG_AUTOMATIC,
    "primal"     : CPX_ALG_PRIMAL,
    "dual"       : CPX_ALG_DUAL,
    "barrier"    : CPX_ALG_BARRIER,
    "sifting"    : CPX_ALG_SIFTING,
    "concurrent" : CPX_ALG_CONCURRENT,
    "net"        : CPX_ALG_NET,
    "netflow"    : CPX_ALG_NET }
           
cdef class CPlexModel(object):

    # THis is for the constraint stuff
    cdef size_t hook_id_1, hook_id_2
    cdef CPlexConstraint hooked_constraint
    cdef CPlexModelInterface *model
    cdef int verbosity
    cdef size_t rv_number
    cdef dict key_strings
    cdef list variables
    cdef double last_op_time 

    def __cinit__(self, int verbosity = 2):
        """
        Creates a new empty model.

        The verbosity level may be passed as a special parameter; see
        :meth:`setVerbosity` for a description of the possible values.  

        When there is a problem instantiating a model or starting
        CPlex, an exception is raised.  Sometimes, additional error
        messages or warnings may be printed, passing this parameter
        here allows for more debugging information.
        """

        self.rv_number = 0
        self.key_strings = {}
        self.variables = []
        self.last_op_time = 0

        self.model = NULL

        self.setVerbosity(verbosity)
        self._checkVerbosity()

        cdef Status model_status = newCPlexModelInterface(&self.model, env)

        if model_status.error_code != 0:
            raise CPlexInitError("Error initializing new cplex model: %s" % str(model_status.message))

    def __dealloc__(self):
        if self.model != NULL:
            del self.model

    cpdef setVerbosity(self, int verbosity):
        """
        Sets the verbosity level of the solver.  The verbosity level
        may be 0,1,2 (default), or 3. These indicate:

          verbosity = 0: no output, ever.
          verbosity = 1: errors are printed.
          verbosity = 2: warnings and errors are printed.
          verbosity = 3: warnings, errors, and progress reports are printed.

        Note that problems such as the model being infeasible or
        unbounded are handled with exceptions; see class documentation
        for more information..
        """
        
        if verbosity not in [0,1,2,3]:
            raise ValueError("Verbosity must be 0, 1, 2, or 3.")

        self.verbosity = verbosity

    cdef _checkOkay(self):
        if self.model == NULL:
            raise RuntimeError("CPlex model not properly initialized!")

    def new(self, size = s_scalar, vtype = 'real', lb = None, ub = None, str name = None):
        """
        Creates a new variable or set of variables for use in the
        model.
        
        **Variable Size**
        
        By default, a single scalar variable is returned.  A vector of
        variables is created by specifying ``size = n``, or a
        matrix-block of variables is created by specifying ``size =
        (n,m)``, where ``n`` and ``m`` are variable sizes.  
        
        Both 1d and 2d variable blocks behave like a matrix-like
        structure of size ``(n,m)``, with ``m`` being 1 for 1d blocks.
        If `size` is given as a single number ``n``, a column matrix
        of size ``(n,1)`` is created.  If size is given as a 2-tuple,
        then a matrix expression of that shape is created.  Thus the
        expression::

          x = m.new(5)
          m.constrain(A*x <= b)

        performs a matrix multiply to evaluate ``A*x``.  This example
        works if the number of rows in ``A`` is the same as the size
        of ``b``, and ``A`` has 5 columns to match the 5 rows of
        ``x``.
        
        When the values of a variable block are requested after
        solving the model, variables created using the default scalar
        value is returned as a single number, 1d variable blocks are
        returned as 1d vectors, and 2d variable blocks are returned as
        2d arrays.  

        **Variable Types**

        Variable types can be specified using ``vtype = <type>``.
        Available types are are reals, integers, and boolean (0 or 1).
        Understood parameters that can be passed as the `vtype`
        parameter are:

          Reals:    float, 'float', 'real', 'double', 'f', or 'd'.

          Integers: int, long, 'int', 'integer', 'i'

          Boolean:  bool, 'bool', 'binary', 'boolean', 'b'

        Example 1::

          >>> m = pycpx.CPlexModel()
          >>> x = m.new(3, vtype=bool)
          >>> m.constrain(x[0] + 2*x[1] + 4*x[2] <= 4.5)
          >>> m.maximize(x.sum())
          2.0
          >>> m[x]
          array([ 1.,  1.,  0.])

        Example 2::

          >>> m = pycpx.CPlexModel()
          >>> x = m.new(3, vtype=int, lb = 0)
          >>> m.constrain(x[0] + 2*x[1] + 4*x[2] <= 4.5)
          >>> m.maximize(x.sum())
          4.0
          >>> m[x]
          array([ 4.,  0.,  0.])
        

        **Bounds**

        Upper and lower bounds may be specified using `ub` or `lb`.
        These bounds may be None (unbounded, default), a scalar value
        (which bounds all variables in the block uniformly), a list of
        values the same length as the variable block being requested
        (1d only), or a numpy array of values the same size as the
        variable block being requested (1d or 2d).  Individual None
        values in the list or NaNs in the array indicate that
        particular variable block is unbounded.

        Example 1::

          >>> import pycpx
          >>> m = pycpx.CPlexModel()
          >>> x = m.new(3, ub = [2,3,4])
          >>> m.maximize(x.sum())
          9.0
          >>> m[x]
          array([ 2.,  3.,  4.])

        Example 2::

          >>> m = pycpx.CPlexModel()
          >>> x = m.new(3, ub = [2,None,4])
          >>> z = m.new(ub = 10)
          >>> m.constrain(x <= z)
          >>> m.maximize(x.sum())
          16.0
          >>> m[x]
          array([  2.,  10.,   4.])


        **Variable Names**

        Optionally, a name for the variable block or group can be
        passed as, e.g. ``name = 'x'``.  This mostly helps with
        printing and debugging, as this name is printed when the model
        or specific constraints are printed.

        Example 1::

          >>> m = pycpx.CPlexModel()
          >>> x = m.new(3, name = 'x')
          >>> m.constrain(0 <= x[0] + 2*x[1] + 3*x[2] <= 4)
          >>> print m
          x[0][-inf..inf] 
            x[1][-inf..inf] 
            x[2][-inf..inf] 
            0 <= x[0]  + 2 * x[1]  + 3 * x[2]  
            x[0]  + 2 * x[1]  + 3 * x[2]  <= 4 
  
          >>> m.minimize(x.sum().abs())
          0.0
          >>> print m
          minimize abs(x[0]  + x[1]  + x[2] ) such that
            x[0][-inf..inf] 
            x[1][-inf..inf] 
            x[2][-inf..inf] 
            0 <= x[0]  + 2 * x[1]  + 3 * x[2]  
            x[0]  + 2 * x[1]  + 3 * x[2]  <= 4 

        If name is None or not given, then variables are named in the
        format ``_<varnum>``, where `varnum` is the variable number in
        order of request.  Thus the first set of variables is ``_1``,
        the second set ``_2``, etc.
        
        """

        self._checkOkay()
        
        self.rv_number += 1
        key = "%s-%s" % (id(self), self.rv_number)

        if name is None:
            name = "_%d" % self.rv_number
        
        cdef CPlexExpression new_var = newVariableBlock(self, size, vtype, lb, ub, name, key)
        self.variables.append(new_var)
        s = self.model.addVariables(new_var.data[0])        
        return new_var

    cdef _checkVerbosity(self):
        if self.verbosity == 0:
            env.setError(env.getNullStream())
            env.setWarning(env.getNullStream())
            env.setOut(env.getNullStream())
        elif self.verbosity == 1:
            env.setError(cout)
            env.setWarning(env.getNullStream())
            env.setOut(env.getNullStream())
        elif self.verbosity == 2:
            env.setError(cout)
            env.setWarning(cout)
            env.setOut(env.getNullStream())
        elif self.verbosity == 3:
            env.setError(cerr)
            env.setWarning(cerr)
            env.setOut(cout)
        else:
            assert False

    cdef _getKeyStringId(self, str key, MetaData md):
        # A mapping to keep things unique

        cdef str query_key = ("%s-%d-%d-%d-%d-%d"
                              % (key,
                                 md.shape(0), md.shape(1),
                                 md.stride(0), md.stride(1),
                                 md.offset()))
        
        # Map it through a dictionary so all the id's are unique and equal
        if query_key in self.key_strings:
            return id(self.key_strings[query_key])
        else:
            self.key_strings[query_key] = query_key
            return id(query_key)

    cdef _checkConstraints(self, tuple constraints):

        cdef size_t i
        for i, c in enumerate(constraints):
            if type(c) is list:
                self._checkConstraints(tuple(c))
                
            elif type(c) is tuple:
                self._checkConstraints(<tuple>c)
            
            elif type(c) is not CPlexConstraint:
                raise TypeError("Expected constraint in argument %d, got %s."
                                % (i + 1, repr(type(c))))

            elif (<CPlexConstraint>c).model is not self:
                raise CPlexException("Constraint %d not from this model." % (i + 1))

    cdef _addConstraint(self, CPlexConstraint c):
        cdef Status s
        s = self.model.addConstraint(c.data[0])
        if s.error_code != 0:
            raise CPlexException("Error adding constraint: %s" % s.message)

    cdef _addConstraints(self, tuple constraints):

        cdef CPlexConstraint c, c2

        for ce in constraints:
            if type(ce) is tuple:
                self._addConstraints(<tuple>ce)
            elif type(ce) is list:
                self._addConstraints(tuple(ce))
            elif ce is True:
                # to handle corner case of (x == x), which gets
                # compared by id.
                
                continue
            else:
                c2 = c = ce

                while c2.hooked_constraint is not None:
                    c2 = c2.hooked_constraint
                    self._addConstraint(c2)

                self._addConstraint(c)

    def constrain(self, *constraints):
        """
        Add a constraint or set of constraints to the model.  

        Constraints are created by relating two expressions with an
        inequality or equality.  Expressions can be numerical arrays,
        variable blocks, or combinations of these.  For example, if
        ``A`` is a numerical matrix, ``x`` is a column vector of
        variables, and ``b`` is a numerical vector, then::

          m.constrain(A*x <= b)

        constrains ``A*x <= b``, provided the dimensions properly
        match up.

        This function accepts lists of constraints, or multiple
        constraints as arguments.  


        **Expressions**

        Expressions can be variable blocks, numerical expressions, or
        any allowable combination thereof.

        A variable block is a matrix of variables to optimize
        (possibly consisting of a single variable).  These are created
        using :meth:`new`.  A 1x1 variable block is treated as a
        scalar.  A numerical expression can be a numpy array or matrix
        or a single scalar value.  These may be combined in any
        allowable way, e.g. ``3*x - 4*y + 5`` is perfectly fine; if
        ``x`` is 1x1 and ``y`` is 10x1, the resulting expression will
        be 10x1, with ``x`` expanded to match the proper dimensions of
        ``y``.

        Expressions can also be treated as regular python variables.
        For example::

          my_expr = A*x + B*y - 5
          m.constrain(my_expr <= 5)

        is perfectly valid.


        **Constraint Behavior**

        Constraints are formed by combining two expressions with one
        of the common python operators: ``==``, ``!=``, ``<``, ``<=``,
        ``>``, or ``>=``.  An example is ``A*x <= b``, described
        above.

        As with other operators, scalar values (possibly 1x1
        expressions) are expanded out to match an interaction
        expression.  For example, ``A*x <= 5`` is valid regardless of
        the dimensions of ``A`` and ``x``, provided ``A*x`` makes
        sense.
        
        It is also possible to chain constraints together with
        multiple inequalities based off common expressions. For
        example::

          m.constrain(-t <= A*x <= t)

        constrains ``A*x`` to be in the interval ``[-t, t]``.  

        Finally, constraint expressions can themselves be held using
        python variables.  For example::

          my_cstr1 = (A*x <= b)
          my_cstr2 = (-t <= A*y <= t)
 
          m.constrain(my_cstr1)
          m.constrain(my_cstr2)

        is perfectly valid.  This can be useful if these constraints
        need to be removed from the model later on using
        :meth:`removeConstraint`.  
        """

        self._checkOkay()

        # First check types
        self._checkConstraints(constraints)
        self._addConstraints(constraints)

    # Removing constraints if need be

    cdef _removeConstraint(self, CPlexConstraint c):
        cdef Status s
        s = self.model.removeConstraint(c.data[0])
        if s.error_code != 0:
            raise CPlexException("Error removing constraint: %s" % s.message)
 
    cdef _removeConstraints(self, tuple constraints):

        cdef CPlexConstraint c, c2

        for ce in constraints:
            if type(ce) is list:
                self._addConstraints(tuple(ce))
            if type(ce) is tuple:
                self._addConstraints(<tuple>ce)
            elif ce is True:
                # to handle corner case of (x == x), which gets
                # compared by id.
                
                continue
            else:
                c2 = c = ce

                while c2.hooked_constraint is not None:
                    c2 = c2.hooked_constraint
                    self._removeConstraint(c2)

                self._removeConstraint(c)

    def removeConstraint(self, *constraints):
        """
        Removes one or more constraints associated with the model.
        The constraints must have been added previously by
        :meth:`constrain()`.

        Example 1::

          >>> from pycpx import CPlexModel
          >>> m = CPlexModel()
          >>> x = m.new()
          >>> y = m.new()
          >>> m.constrain(0 <= x <= y <= 5)
          >>> c = (x <= 2)
          >>> m.constrain(c)
          >>> m.maximize(x + y)
          7.0
          >>> m.removeConstraint(c)
          >>> m.maximize(x + y)
          10.0

        """

        self._checkOkay()

        self._checkConstraints(constraints)
        self._removeConstraints(constraints)

        
    cpdef solve(self, objective, maximize = None, minimize = None,
              bint recycle_variables = False, bint recycle_basis = True,
              dict starting_dict = {}, str basis_file = None,
              algorithm = "auto"):
        """
        Solves the current model trying to maximize (default) or
        minimize `objective` subject to the constraints given by
        :meth:`constrain()`.  `objective` can be any expression (as
        described in the documentation for :class:`CPlexModel`).  The
        function returns the value of the objective after
        optimization.

        Typically, this function is called using one of the alias
        functions, :meth:`minimize` or :meth:`maximize`, to set the
        sense of the optimization.  Alternatively, one can pass any of
        the following as keyword arguments:

          - ``maximize = True``: Sets sense to find maximum.

          - ``minimize = True``: Sets sense to find minimum,

          - ``maximize = False``: Sets sense to find minimum.
          
          - ``minimize = False``: Sets sense to find maximum.

        **Available options**

        starting_dict:

          Specify starting points for variables given in a dictionary.

          For optimization problems having integer values, this may
          give a speedup.  Note, however, that for non-integer linear
          programs, constructing a solver state from a starting point
          is usually as time consuming as solving it in the first
          place, thus it is rare to get any speedup.

        recycle_variables:

          Can be True or False (default).  If True, the model has
          already been solved at least once, and no constraints have
          been added or removed, then the variable values from the
          previous run are used to form the starting point for this
          current run.  Note, however, that CPlex takes a usually
          non-trivial amount of time to construct a basis from a given
          starting point, so this usually doesn't help that much
          except in combinatorial problems.

        recycle_basis:
        
          Can be True or False (default).  If True, then the basis
          from the last run of the model is used to instantiate this
          run.  If the basis is saved from before using
          :meth:`saveBasis`, then one should use basis_file instead.

        basis_file:

          Specify a file to load a basis from.  This file should be
          from a previous call to :meth:`saveBasis`.

        algorithm:

          Specify which algorithm to use.  Available options are auto
          (default), primal, dual, barrier, sifting, concurrent, or
          netflow.  See CPlex doumentation for the specifics..

        Example 1::

          >>> from pycpx import CPlexModel
          >>>
          >>> m = CPlexModel()
          >>> x = m.new(lb = 0, ub = 5)
          >>> m.maximize(2*x)
          10.0
          >>> m[x]
          5.0

        Example 2::

          >>> import numpy as np
          >>> from pycpx import CPlexModel
          >>>
          >>> A = np.array([[1,0,0], [1,1,0], [1,1,1]])
          >>> b = np.array([1,2,3])
          >>> 
          >>> m = CPlexModel()
          >>> x = m.new(3, lb = 0)
          >>> m.constrain(A*x <= b)
          >>> 
          >>> m.maximize(3*x[0] + 2*x[1] + x[2])
          6.0
          >>> m[x]
          array([ 1.,  1.,  1.])

        Example 3::

          >>> import numpy as np
          >>> from pycpx import CPlexModel
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

        Example 4::

          >>> m = CPlexModel()
          >>> 
          >>> x = m.new(lb = 0, vtype=int, name = 'x')
          >>> y = m.new(lb = 0, vtype=int, name = 'y')
          >>> 
          >>> m.constrain(5*x - y <= 12)
          >>> m.constrain(3*x + 2*y <= 8)
          >>> 
          >>> m.maximize(2*x + y)
          5.0
          >>> m[x]
          2.0
          >>> m[y]
          1.0

        """

        self._checkOkay()

        cdef Status s
        cdef CPlexExpression var

        ################################################################################
        # Set local parameters

        cdef CPlexExpression obj

        if not type(objective) is CPlexExpression:
            raise TypeError("Objective must be an expression.")

        obj = objective
        
        cdef bint _maximize = True, _minimize = False

        if maximize is not None and minimize is None:
            _maximize = maximize
            _minimize = not _maximize
            
        elif maximize is None and minimize is not None:
            _maximize = not minimize
            _minimize = not _maximize
            
        elif maximize is not None and minimize is not None:
            _maximize = maximize
            _minimize = minimize

            if _maximize == _minimize:
                raise ValueError("Cannot both maximize and minimize the problem at the same time.")

        ################################################################################
        # Get any model parameters that we need from the previous model

        cdef list recycle_variable_values = None

        if recycle_variables and self.model.solved():
            recycle_variable_values = [self.value(v) for v in self.variables]

        cdef str tmp_basis_file_name = None
        
        if recycle_basis and self.model.solved():
            tmp_basis_file, tmp_basis_file_name = tempfile.mkstemp(suffix='bas', prefix='tmp_cplex')
            
            b = bytes(tmp_basis_file_name)
            self.model.writeBasis(b)

        try:

            ################################################################################
            # Now see if we're maximizing or minimizing

            s = self.model.setObjective(obj.data[0], _maximize)

            if s.error_code != 0:
                raise CPlexException("Error setting objective: %s" % s.message)

            ################################################################################
            # Now do all the stuff we were going to do, but now it's after
            # the objective is set, so these things stay put

            try:
                self.model.setParameter(RootAlg, model_lookup[algorithm.lower()])
            except KeyError:
                raise ValueError("Algorithm '%s' not recognized, can be auto, primal, dual, barrier, sifting, concurrent, or netflow.")

            if tmp_basis_file_name is not None:
                b = bytes(tmp_basis_file_name)
                self.model.readBasis(b)

            if basis_file is not None:
                b = bytes(basis_file)
                self.model.readBasis(b)

            if recycle_variable_values is not None:

                for var, val in zip(self.variables, recycle_variable_values):
                    s = self.model.setStartingValues(
                        var.data[0], newCoercedNumericalArray(val, var.data.md()).data[0])
                    if s.error_code != 0:
                        raise CPlexException("Error setting starting values: %s" % str(s.message))

            if starting_dict:
                for var, X in starting_dict.iteritems():
                    s = self.model.setStartingValues(
                        var.data[0], newCoercedNumericalArray(X, var.data.md()).data[0])
                    if s.error_code != 0:
                        raise CPlexException("Error setting starting values: %s" % str(s.message))

            ###############################################################################
            # Now solve it!
            s = self.model.solve(&self.last_op_time)

            if s.error_code != 0:
                if s.error_code in [MODEL_UNBOUNDED, MODEL_INFEASABLE,
                                    MODEL_UNBOUNDED_OR_INFEASABLE]:
                    
                    raise CPlexNoSolution(str(s.message))
                
                else:
                    raise CPlexException(str(s.message))
                
            return self.model.getObjectiveValue()
        
        finally:
            if tmp_basis_file_name is not None:
                os.remove(tmp_basis_file_name)

    def saveBasis(self, str filename):
        """
        Writes a basis for the current solution to a file.  This may
        be used to reinstate a previous state of the solver at a later
        time.
        """

        self._checkOkay()

        b = bytes(filename)
        
        cdef Status s = self.model.writeBasis(b)
        
        if s.error_code != 0:
            raise CPlexException(str(s.message))

    def maximize(self, objective, **options):
        """
        Solves the model by maximizing `objective`. This function
        accepts the same options as :meth:`solve`, with the exception
        of `maximize` or `mininmize`.
        """

        return self.solve(objective, maximize = True, **options)

    def minimize(self, objective, **options):
        """
        Solves the model by minimizing `objective`.  This function
        accepts the same options as :meth:`solve`, with the exception
        of `maximize` or `mininmize`.
        """

        return self.solve(objective, maximize = False, **options)

    def getSolverTime(self):
        """
        Returns the time (in seconds, as a float) of the previous call
        to solve, as measured by CPlex.  Returns 0 if :meth:`solve`,
        :meth:`minimize`, or :meth:`maximize` have not been called.
        """

        return self.last_op_time

    def getNIterations(self):
        """
        Returns the number of iterations made during the previous call
        to solve, as measured by CPlex.  Returns 0 if :meth:`solve`,
        :meth:`minimize`, or :meth:`maximize` have not been called.
        """

        self._checkOkay()
        
        return self.model.getNIterations()
        

    cpdef value(self, var_block_or_expression):
        """
        Returns a scalar, numpy array, or matrix filled by the values
        of the variable block or expression.  Calling ``m.value(x)``
        is the same as calling ``m[x]`` or ``x()`` to retrieve the
        result of a variable.  This function can only be called after
        the model has been solved at least once.

        In all other places, variables are represented using 2d
        matrices, with a 1x1 matrix representing a scalar and column
        vectors representing vectors.  However, a flag is preserved so
        that returned values are in the same form as originally
        requested.  Thus 'scalar' variable values are returned as
        scalars, vector values are returned as a 1d numpy array, and
        2d variable blocks are returned as 2d matrices.

        Example::

          >>> from pycpx import CPlexModel
          >>> from numpy import array, arange
          >>> 
          >>> A = 2*arange(1,10).reshape( (3, 3) )
          >>> m = CPlexModel()
          >>> 
          >>> X = m.new( (3, 3), vtype = int)
          >>> u = m.new( 3, vtype = int)
          >>> s = m.new(vtype = int)
          >>> 
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
          >>> m[X[:,1]]
          array([ 3.,  0.,  0.])
          >>> m[X[2,:]]
          matrix([[ 1.,  0.,  0.]])
          >>> m[10*s - A.T * X]
          matrix([[ 10.,  14.,  12.],
                  [ 12.,   8.,  10.],
                  [ 14.,   2.,   8.]])
        
        """

        self._checkOkay()
        
        if type(var_block_or_expression) is not CPlexExpression:
            raise TypeError("Can only retrieve variables or expressions.")

        # if not self.model.solved():
        #     raise CPlexException("Can only retrieve variables after model is solved.")

        cdef Status s

        cdef CPlexExpression v = (<CPlexExpression>var_block_or_expression)

        if v.model is not self:
            raise ValueError("Can only retrieve variables from the model in which they were created.")
        
        # print "var_block size = ", (v.data.md().shape(0), v.data.md().shape(1))

        M = matrix(empty( (v.data.md().shape(0), v.data.md().shape(1)) ) )

        # print "M.shape = ", M.shape

        cdef ar[double,ndim=2, mode = "c"] X = M
        
        cdef NumericalArray *na = new NumericalArray(env, (<double*>(X.data)),
                MetaData(v.data.md().mode(), v.data.md().shape(0), v.data.md().shape(1)))

        try:
            s = self.model.getValues(na[0], v.data[0])

            if s.error_code != 0:
                raise CPlexException("Error while retrieving variables: %s" % s.message)

            size = v.original_size

            if size == s_scalar:
                assert X.shape[0] == X.shape[1] == 1
                return M[0,0]

            elif (type(size) is tuple and len(<tuple>size) == 2):
                return M

            elif (type(size) is tuple and len(<tuple>size) == 1):
                assert X.shape[0] == size[0]
                return asarray(M).ravel()

            elif isscalar(size):
                assert X.shape[0] == size and X.shape[1] == 1
                return asarray(M).ravel()

            else:
                return M
            
        finally:
            del na

    cpdef asString(self):
        """
        Returns a string representation of the model.  If the
        variables are named, they are printed with those names,
        otherwise ``_##`` is used (e.g. ``_1, _2, _3, ...``.

        Example 1:: 
        
          >>> m = CPlexModel()
          >>> 
          >>> x = m.new(lb = 0, vtype=int, name = 'x')
          >>> y = m.new(lb = 0, vtype=int, name = 'y')
          >>> 
          >>> m.constrain(5*x - y <= 12)
          >>> m.constrain(3*x + 2*y <= 8)
          >>> 
          >>> m.maximize(2*x + y)
          5.0
          >>> m
          maximize 2 * x  + y  such that
            x[0..9007199254740991] 
            y[0..9007199254740991] 
            5 * x  + -1 * y  <= 12 
            3 * x  + 2 * y  <= 8 

          >>> m[x]
          2.0
          >>> m[y]
          1.0

        Example 2::

          >>> import numpy as np
          >>> from pycpx import CPlexModel
          >>> 
          >>> A = np.array([[1,0,0], [1,1,0], [1,1,1]])
          >>> b = np.array([1,2,3])
          >>> 
          >>> m = CPlexModel()
          >>> x = m.new(3, lb = 0, ub = 5, name = 'x')
          >>> t = m.new(name = 't')
          >>> 
          >>> m.constrain( abs((A*x - b)) <= t)
          >>> 
          >>> m.minimize(t)
          0.0
          >>> m
          minimize t  such that
            x[0][0..5] 
            x[1][0..5] 
            x[2][0..5] 
            t[-inf..inf] 
            abs(x[0]  + -1 ) <= t[-inf..inf] 
            abs(x[0]  + x[1]  + -2 ) <= t[-inf..inf] 
            abs(x[0]  + x[1]  + x[2]  + -3 ) <= t[-inf..inf] 
         
        """


        self._checkOkay()
        
        return self.model.asString().c_str()

    def __repr__(self):
        return self.asString()

    def __getitem__(self, var_block):
        return self.value(var_block)
