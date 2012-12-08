#ifndef _CONTAINERS_HPP_
#define _CONTAINERS_HPP_

// Sorta provides intermediate wrapper functions for many of the
// operations; it's just easier to do this with templating here than
// in cython.

#include <ilconcert/iloexpression.h>
#include <ilconcert/iloenv.h>

#include "debug.h"
#include "optimizations.h"
#include "simple_shared_ptr.h"
#include "constants.h"

using namespace std;


#define MATRIX_MODE     0
#define ARRAY_MODE      1
#define DIAG_MODE       2
#define CONSTRAINT_MODE 3

#define SLICE_SINGLE  0
#define SLICE_FULL    1
#define SLICE_GENERAL 2
#define S2_IDX(S0, S1) (3*(S0) + (S1))

class MetaData {
public:
    MetaData()
	: _mode(0), 
	  _offset(0),
	  _shape(make_pair<long,long>(0,0)), 
	  _stride(make_pair<long,long>(0,0))
	{
	}

    MetaData(int mode, long shape_0, long shape_1)
	: _mode(mode), 
	  _offset(0),
	  _shape(make_pair(shape_0, shape_1)), 
	  _stride(shape_0*shape_1 == 1 
		  ? make_pair<long, long>(0,0)
		  : make_pair<long, long>(shape_1, 1))
	{
	}
    
    MetaData(int mode, long shape_0, long shape_1, long stride_0, long stride_1)
	: _mode(mode), 
	  _offset(0),
	  _shape(make_pair(shape_0, shape_1)), 
	  _stride((shape_0 == 1 && shape_1 == 1)
		  ? make_pair<long,long>(0,0)
		  : make_pair(stride_0, stride_1))
	{
	}

    MetaData(int mode, 
	     long offset,
	     long shape_0, long shape_1, 
	     long stride_0, long stride_1)
	: _mode(mode), 
	  _offset(offset),
	  _shape(make_pair(shape_0, shape_1)), 
	  _stride(make_pair(stride_0, stride_1))
	{
	    if(DEBUG_MODE && shape_0 == 1 && shape_1 == 1)
	    {
		assert_equal(stride_0, 0);
		assert_equal(stride_1, 0);
	    }
	}

    template <typename Slice0, typename Slice1>
    MetaData(const MetaData& md, const Slice0& s0, const Slice1& s1)
	: _mode(md._mode), 
	  _offset(md._offset + s0.start()*md.stride(0) + (s1.start()*md.stride(1))),
	  _shape(make_pair(s0.size(), s1.size())),
	  _stride(_shape.first * _shape.second == 1
		  ? make_pair<long, long>(0,0)
		  : make_pair(md._stride.first*s0.step(), md._stride.second*s1.step()))
	{
	}

    MetaData(const MetaData& md, bool preserve_striding = true)
	: _mode(md._mode)
	, _offset(preserve_striding ? md._offset : 0)
	, _shape(md._shape)
	, _stride(preserve_striding ? md._stride : 
		  (make_pair<long, long>(md._shape.first == 1 ? 0 : md._shape.second,
					 md._shape.second == 1 ? 0 : 1)))
	{
	}

    void print() const
	{
	    cout << "Metadata: \n" 
		 << " mode = " << mode() << ")\n" 
		 << " shape = (" << shape(0) << "," << shape(1) << ")\n"
		 << " offset = (" << offset() << "\n"
		 << " stride= (" << stride(0) << "," << stride(1) << ")" << endl;
	}

    inline int mode() const
	{
	    return _mode;
	}

    inline void setMode(int new_mode)
	{
	    _mode = new_mode;
	}

    inline const pair<long, long>& shape() const 
	{ 
	    return _shape; 
	}

    inline long shape(int i) const
	{
	    switch(i) {
	    case 0:
		return _shape.first;
	    case 1:
		return _shape.second;
	    default:
		return 0;
	    }
	}
    
    inline long offset() const
	{ 
	    return _offset; 
	}
				   
    inline const pair<long, long>& stride() const 
	{ 
	    return _stride; 
	}

    inline long stride(int i) const 
	{ 
	    if(DEBUG_MODE && shape(0) == 1 && shape(1) == 1)
	    {
		assert_equal(_stride.first, 0);
		assert_equal(_stride.second, 0);
	    }

	    switch(i) {
	    case 0:
		return _stride.first;
	    case 1:
		return _stride.second;
	    default:
		return 0;
	    }
	}

    inline long size() const
	{
	    return _shape.first * _shape.second;
	}

    MetaData transposed() const
	{
	    return MetaData(_mode, _shape.second, _shape.first, _stride.second, _stride.first);
	}

    bool matrix_multiplication_applies(const MetaData& md_right) const
	{
	    return ((mode() == MATRIX_MODE || md_right.mode() == MATRIX_MODE)
		    && !(shape(0) == 1 && shape(1) == 1)
		    && !(md_right.shape(0) == 1 && md_right.shape(1) == 1));
	}

    private:
    int _mode;
    long _offset;
    pair<long, long> _shape;
    pair<long, long> _stride;
};


////////////////////////////////////////////////////////////////////////////////

class Slice {
public:
    Slice() : _start(0), _stop(0), _step(1) {}
    
    Slice(long start, long stop, long step)
	: _start(start), _stop(stop), _step(step)
	{
	}

    long size() const
	{
	    assert_equal( (_stop - _start) % _step, 0);
	    return (_stop - _start) / _step;
	}

    long step() const  { return _step;}
    long start() const { return _start;}
    long stop() const  { return _stop;}

private:
    long _start, _stop, _step;
};

class SliceFull{
public:
    SliceFull() : _size(0) {}
    SliceFull(long size) : _size(size) {}

    long size() const  { return _size; }
    long step() const  { return 1; }
    long start() const { return 0;}
    long stop() const  { return _size;}
private:
    long _size;
};

class SliceSingle{
public:
    SliceSingle() : _index(0) {}

    SliceSingle(long index) 
	: _index(index) 
	{}

    inline long size() const  { return 1; }
    inline long step() const  { return 1; }
    inline long start() const { return _index;}
    inline long stop() const  { return _index+1;}

private:
    long _index;
};

template<typename ParentType, typename T, int scalar_type> class ComponentBase {
public:
    typedef T Value;

  ComponentBase(IloEnv _env, const MetaData& md, bool preserve_striding)
    : _md(md, preserve_striding), env(_env)
  {
  }

    inline IloEnv getEnv() const 
	{
	    return env;
	}

    inline bool preferReversedTraverse() const 
	{
	    return scalar_type ? false : _md.stride(0) < _md.stride(1);
	}

    inline void setMode(int _mode)		{ _md.setMode(_mode); }

    inline const MetaData& md() const		{ return _md; }

    long offset() const				{ return scalar_type ? 0 : _md.offset(); }

    const pair<long, long>& shape() const	{ return _md.shape(); }

    long shape(int dim) const		  
	{ return scalar_type ? ((dim == 1 || dim == 0) ? 1 : 0) : _md.shape(dim); }

    const pair<long, long>& stride() const	{ return _md.stride(); }

    long stride(int i) const			{ return scalar_type ? 0 : _md.stride(i); }

    long size() const				{ return scalar_type ? 1 : shape(0) * shape(1); }

    inline ParentType* newFromGeneralSlice(const Slice& s0, const Slice& s1) const
	{
	    int type0 = SLICE_GENERAL;
	    int type1 = SLICE_GENERAL;

	    if(s0.stop() - s0.start() == s0.step())
		type0 = SLICE_SINGLE;
	    else if(s0.step() == 1 && s0.stop() - s0.start() == _md.shape(0))
		type0 = SLICE_FULL;

	    if(s1.stop() - s1.start() == s1.step())
		type1 = SLICE_SINGLE;
	    else if(s1.step() == 1 && s1.stop() - s1.start() == _md.shape(1))
		type1 = SLICE_FULL;
	    
	    switch(S2_IDX(type0, type1)) {

	    case S2_IDX(SLICE_SINGLE, SLICE_SINGLE):
		return newFromSlice(SliceSingle(s0.start()), SliceSingle(s1.start()));
		
	    case S2_IDX(SLICE_SINGLE, SLICE_FULL):
		return newFromSlice(SliceSingle(s0.start()), SliceFull(shape(1)));
		
	    case S2_IDX(SLICE_SINGLE, SLICE_GENERAL):
		return newFromSlice(SliceSingle(s0.start()), s1);
	    
	    case S2_IDX(SLICE_FULL, SLICE_SINGLE):
		return newFromSlice(SliceFull(shape(0)), SliceSingle(s1.start()));
		
	    case S2_IDX(SLICE_FULL, SLICE_FULL):
		return newFromSlice(SliceFull(shape(0)), SliceFull(shape(1)));
		
	    case S2_IDX(SLICE_FULL, SLICE_GENERAL):
		return newFromSlice(SliceFull(shape(0)), s1);
		
	    case S2_IDX(SLICE_GENERAL, SLICE_SINGLE):
		return newFromSlice(s0, SliceSingle(s1.start()));
		
	    case S2_IDX(SLICE_GENERAL, SLICE_FULL):
		return newFromSlice(s0, SliceFull(shape(1)));

	    default:
            case S2_IDX(SLICE_GENERAL, SLICE_GENERAL):
		return newFromSlice(s0, s1);
	    }
        }

    template <typename Slice0, typename Slice1>
    inline ParentType* newFromSlice(const Slice0& s0, const Slice1& s1) const
	{
	    // cout << "Creating new from ";
	    // md().print();
	    // cout << "with slice 0: start=" << s0.start() << ", step=" << s0.step() << ", stop=" <<s0.stop() << endl;
	    // cout << "with slice 1: start=" << s1.start() << ", step=" << s1.step() << ", stop=" <<s1.stop() << endl;

	    ParentType * dest = new ParentType(parent(), s0, s1);

	    // cout << "New metdata = ";
	    // dest->md().print();
	    return dest;
	}

    inline ParentType* newTransposed() const
	{
	    return new ParentType(parent(), _md.transposed());
	}

    inline ParentType* newCopy()
	{
	    return new ParentType(parent(), _md);
	}

    inline ParentType* newAsArray()
	{
	    MetaData new_md(_md);
	    new_md.setMode(ARRAY_MODE);
	    
	    return new ParentType(parent(), new_md);
	}

    inline ParentType* newAsMatrix()
	{
	    MetaData new_md(_md);
	    new_md.setMode(MATRIX_MODE);
	    
	    return new ParentType(parent(), new_md);
	}


    // inline ParentType* newDiag()
    // 	{
    // 	    if( (shape(0) == 1 && shape(1) != 1) )
    // 	    {
		

    // 	    }
    // 	    else if ( (shape(1) == 1 && shape(0) != 1) )
    // 	    {
		
    // 	    }
    // 	    else if ( (shape(1) == shape(0)) )
    // 	    {
		
    // 	    }
    // 	    else
		
			  
    // 	}

    inline long getIndex(long i, long j) const
	{
	    if(DEBUG_MODE && stride(0) != 0 && stride(1) != 0)
	    {
		assert_leq(i, shape(0));
		assert_leq(j, shape(1));
	    }

	    if(DEBUG_MODE && shape(0) == 1 && shape(1) == 1)
	    {
		assert_equal(stride(0), 0);
		assert_equal(stride(1), 0);
	    }

	    // cerr << "offset = " << offset()
	    // 	 << "stride(0) = " << stride(0)
	    // 	 << "stride(1) = " << stride(1) 
	    // 	 << "shape(0) = " << shape(0)
	    // 	 << "shape(1) = " << shape(1)
	    // 	 << "i = " << i
	    // 	 << "j = " << j
	    // 	 << endl;

	    return offset() + stride(0)*i  + stride(1)*j;
	}

protected:
    inline const ParentType& parent() const 
	{
	    return *(static_cast<const ParentType*>(this));
	}

    MetaData _md;
    IloEnv env;
};

class ExpressionArray : public ComponentBase<ExpressionArray, IloNumExpr, 0> {
public:  
    typedef ComponentBase<ExpressionArray, IloNumExpr, 0> Base;
  
    ExpressionArray(IloEnv env, const MetaData& md)
      : Base(env, md, false), data_ptr(new IloExprArray(env, shape(0) * shape(1))),
	  aux_var_ptr(NULL)
	{
	}

    ExpressionArray(IloEnv env, IloNumVarArray * v, const MetaData& md)
      : Base(env, md, false), data_ptr(new IloExprArray(env, shape(0) * shape(1))),
	  aux_var_ptr(v)
	{
	    assert_equal(v->getSize(), shape(0)*shape(1));

	    for(long i = 0; i < shape(0) * shape(1); ++i)
		(*data_ptr)[i] = (*v)[i];
	}
    
    ExpressionArray(const ExpressionArray& ea, const MetaData& md)
      : Base(ea.env, md, true), data_ptr(ea.data_ptr), aux_var_ptr(NULL)
	{
	}

    template<typename Slice0, typename Slice1>
    ExpressionArray(const ExpressionArray& ea, const Slice0& s0, const Slice1& s1)
      : Base(ea.env, MetaData(ea.md(), s0, s1), true), data_ptr(ea.data_ptr),
	  aux_var_ptr(NULL)
	{
	}

private:
    SharedPointer<IloExprArray> data_ptr;

    // This allows us to work with an auxilary variable 
    SharedPointer<IloNumVarArray> aux_var_ptr;

public:

    inline Value& operator()(long i, long j)
	{ 
	    long idx = getIndex(i,j);
	    assert_lt(idx,data_ptr->getSize());
	    return (*data_ptr)[idx];
	}

    inline const Value& operator()(long i, long j) const
	{ 
	    long idx = getIndex(i,j);
	    assert_lt(idx,data_ptr->getSize());
	    return (*data_ptr)[idx];
	}

    inline void set(long i, long j, const Value& v) 
	{
	    (*this)(i,j) = v;
	}

    inline Value& get(long i, long j)
	{
	  return (*this)(i,j);
	}

    inline const Value& get(long i, long j) const
	{
	  return (*this)(i,j);
	}

    ////////////////////////////////////////////////////////////////////////////////
    // Methods specific to this case

    inline const IloExprArray& expression() const { return *data_ptr; }

    inline bool hasVar() const { return aux_var_ptr != NULL; }

    inline const IloNumVarArray& variables() const 
	{
	    assert(hasVar());

	    return (*aux_var_ptr);
	}

    inline bool isComplete() const
	{
	    return md().offset() == 0 && data_ptr->getSize() == size();
	}
};

class ConstraintArray : public ComponentBase<ConstraintArray, IloConstraint, 0> {
public:
    typedef IloConstraint Value;
    typedef ComponentBase<ConstraintArray, IloConstraint, 0> Base;

    ConstraintArray(IloEnv env, const MetaData& _md)
      : Base(env, _md, false), data_ptr(new IloConstraintArray(env, shape(0)*shape(1)) )
	{
	}
    
private:
    SharedPointer<IloConstraintArray> data_ptr;

public:

    inline Value& operator()(long i, long j) 
	{ 
	    return (*data_ptr)[getIndex(i,j)];
	}

    inline const Value& operator()(long i, long j) const
	{ 
	    return (*data_ptr)[getIndex(i,j)];
	}

    inline void set(long i, long j, const Value& v) 
	{
	    (*this)(i,j) = v;
	}

    inline const IloConstraintArray& constraint() const { return *data_ptr; }

};

// These come from other operations

class NumericalArray : public ComponentBase<NumericalArray, double, 0> {

public:
    typedef ComponentBase<NumericalArray, double, 0> Base;
    typedef Base::Value Value;

    NumericalArray(IloEnv env, double* _data, const MetaData& _md)
      : Base(env, _md, true), data(_data)
	{
	}

private:
    double* const data;

public:
    double& operator()(long i, long j)
	{
	    return *(data + getIndex(i,j));
	}

    const double& operator()(long i, long j) const
	{
	    return *(data + getIndex(i,j));
	}

    inline void set(long i, long j, const Value& v) 
	{
	    (*this)(i,j) = v;
	}
};

struct Scalar : public ComponentBase<Scalar, double, 1> {
public:
    typedef ComponentBase<Scalar, double, 1> Base;
    typedef double Value;

    Scalar(IloEnv env, const double& _value)
      : Base(env, MetaData(ARRAY_MODE, 1, 1, 0, 0), true), value(_value)
	{
	}

    Scalar(IloEnv env, const double& _value, const MetaData&)
      : Base(env, MetaData(ARRAY_MODE, 1, 1, 0, 0), true), value(_value)
	{
	}

private:
    double value;

public:

    inline long offset() const
	{
	    return 0;
	}

    inline long stride(int) const
	{
	    return 0;
	}

    inline long shape(int dim) const
	{
	    return (dim == 1 || dim == 0) ? 1 : 0;
	}

    inline const double& operator()(long, long) const
	{
	    assert_equal(shape(0), 1);
	    assert_equal(shape(1), 1);
	    assert_equal(stride(0), 0);
	    assert_equal(stride(1), 0);
	    assert_equal(offset(), 0);

	    return value;
	}
};

////////////////////////////////////////////////////////////////////////////////
// Now need a generic way to combine the basic structures.  This
// populates the metadata md based on the 

inline int newMode(int op_type, int m1, int m2)
{
    return ( (m1 == MATRIX_MODE || m2 == MATRIX_MODE) ? MATRIX_MODE : ARRAY_MODE );
}

template <typename Slice0, typename Slice1>
MetaData newMetaData(const MetaData& md, const Slice0& s0, const Slice1& s1)
{
    return MetaData(md.mode(), s0.size(), s1.size(), s1.size(), 1);
}

MetaData newMetadata(int op_type, const MetaData& md1, const MetaData& md2, int* okay)
{
    // Eventually will have to consider other modes here, like
    // diaganol, sparse, etc.

    int mode = newMode(op_type, md1.mode(), md2.mode());

    op_type = op_type & OP_SIMPLE_MASK;

    if( (op_type == OP_B_MULTIPLY && md1.matrix_multiplication_applies(md2))
	|| op_type == OP_B_MATRIXMULTIPLY)
    {
	// Okay, need to do a matrix comparison here
	if(md1.shape(1) == md2.shape(0))
	{
	    *okay = true;
	    return MetaData(mode, md1.shape(0), md2.shape(1));
	}
	else
	{
	    *okay = false;
	    return MetaData();
	}
    }
    else if( (md1.shape(0) == md2.shape(0)) && (md1.shape(1) == md2.shape(1)) )
    {
	*okay = true;
	
	if( (md1.stride(0) == md2.stride(0)) && (md1.stride(1) == md2.stride(1)) )
	    return MetaData(mode, md1.shape(0), md1.shape(1), md1.stride(0), md1.stride(1));

	else
	    return MetaData(mode, md1.shape(0), md1.shape(1));
    }
    else if( (md1.shape(0) == 1 && md1.shape(1) == 1) )
    {
	assert_equal(md1.stride(0), 0);
	assert_equal(md1.stride(1), 0);

	*okay = true;
	
        return MetaData(mode, md2.shape(0), md2.shape(1));
    }
    else if( (md2.shape(0) == 1 && md2.shape(1) == 1) )
    {
	assert_leq(md2.stride(0), 0);
	assert_leq(md2.stride(1), 0);


	*okay = true;

	return MetaData(mode, md1.shape(0), md1.shape(1));
    }
    else
    {
	*okay = false;
	return MetaData();
    }
}

#endif
