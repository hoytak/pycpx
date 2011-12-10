#ifndef _OPERATORS_HPP_
#define _OPERATORS_HPP_

// Sorta provides intermediate wrapper functions for many of the
// operations; it's just easier to do this with templating here than
// in cython.

#include <ilconcert/iloexpression.h>
#include <ilconcert/iloalg.h>
#include <ilconcert/iloenv.h>

#include "debug.h"
#include "optimizations.h"
#include "containers.hpp"
#include "constants.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Unary operators

template <int OpType, typename D, typename S> struct UOp {};

template <typename D, typename S> struct UOp<OP_U_NO_TRANSLATE, D, S> {
    inline D operator()(const S& s) const { return s; }
};

template <typename D, typename S> struct UOp<OP_U_ABS, D, S> {
    inline D operator()(const S& s) const { return IloAbs(s); }
};

template <typename D, typename S> struct UOp<OP_U_NEGATIVE, D, S> {
    inline D operator()(const S& s) const { return -s; }
};

template <typename DA, typename SA, typename UnaryFunction>
void unary_op(DA& dest, const SA& src, const UnaryFunction& op)
{
    assert_equal(dest.shape(0), src.shape(0));
    assert_equal(dest.shape(1), src.shape(1));

    if(unlikely(dest.preferReversedTraverse()))
    {
	for(long j = 0; j < dest.shape(1); ++j)
	    for(long i = 0; i < dest.shape(0); ++i)
		dest(i,j) = op(src(i,j));
    }
    else
    {
	for(long i = 0; i < dest.shape(0); ++i)
	    for(long j = 0; j < dest.shape(1); ++j)
		dest(i,j) = op(src(i,j));
    }
}

template <typename DA, typename SA, typename Slice0, typename Slice1, typename UnaryFunction>
void unary_op(DA& dest, const SA& src, const Slice0& src_sl0, const Slice1& src_sl1, const UnaryFunction& op)
{
    assert_equal(dest.shape(0), src_sl0.size());
    assert_equal(dest.shape(1), src_sl1.size());

    if(unlikely(dest.preferReversedTraverse()))
    {
	for(long j = 0, sj = src_sl1.start(); j < dest.shape(1); ++j, sj += src_sl1.step())
	    for(long i = 0, si = src_sl0.start(); i < dest.shape(0); ++i, si += src_sl0.step())
		dest(i,j) = op(src(si,sj));
    }
    else
    {
	for(long i = 0, si = src_sl0.start(); i < dest.shape(0); ++i, si += src_sl0.step())
	    for(long j = 0, sj = src_sl1.start(); j < dest.shape(1); ++j, sj += src_sl1.step())
		dest(i,j) = op(src(si,sj));
    }
}

ExpressionArray* newFromUnaryOp(const ExpressionArray& src, int op_type) 
{
    typedef ExpressionArray::Value Value;

    ExpressionArray *dest = new ExpressionArray(src.getEnv(), src.md());

    switch(op_type) {

    case OP_U_NO_TRANSLATE:
	unary_op(*dest, src, UOp<OP_U_NO_TRANSLATE, Value, Value>());
	return dest;
    case OP_U_ABS:
	unary_op(*dest, src, UOp<OP_U_ABS, Value, Value>());
	return dest;
    case OP_U_NEGATIVE:
	unary_op(*dest, src, UOp<OP_U_NEGATIVE, Value, Value>());
	return dest;
    default:
	assert(false);
	return dest;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Reduction operators

template <int OpType, typename T> struct ROp {};

template <typename T> struct ROp<OP_R_SUM, T> {
    inline void operator()(T& dest, const T& src) const { dest += src; }
};

template <typename T> struct ROp<OP_R_MAX, T> {
    inline void operator()(T& dest, const T& src) const { dest = max(dest, src); }
};

template <> struct ROp<OP_R_MAX, IloNumExpr> {
    inline void operator()(IloNumExpr& dest, const IloNumExpr& src) const { dest = IloMax(dest, src); }
};

template <typename T> struct ROp<OP_R_MIN, T> {
    inline void operator()(T& dest, const T& src) const { dest = min(dest, src); }
};

template <> struct ROp<OP_R_MIN, IloNumExpr> {
    inline void operator()(IloNumExpr& dest, const IloNumExpr& src) const { dest = IloMin(dest, src); }
};

template <typename D, typename SA, typename Slice0, typename Slice1, typename ReductionOp>
void reduction_op(D& dest, const SA& src, const Slice0& src_sl0, const Slice1& src_sl1, 
		  const ReductionOp& op, bool is_simple)
{
    dest = src(src_sl0.start(), src_sl1.start());

    dest.getEnv().setNormalizer(is_simple ? IloFalse : IloTrue);

    if(unlikely(src.preferReversedTraverse()))
    {
	for(long i = src_sl0.start() + src_sl0.step(); i != src_sl0.stop(); i += src_sl0.step())
	    op(dest, src(i,src_sl1.start()));

	for(long j = src_sl1.start() + src_sl1.step(); j != src_sl1.stop(); j += src_sl1.step())
	    for(long i = src_sl0.start(); i != src_sl0.stop(); i += src_sl0.step())
		op(dest, src(i,j));
    }
    else
    {
	for(long j = src_sl1.start() + src_sl1.step(); j != src_sl1.stop(); j += src_sl1.step())
	    op(dest, src(src_sl0.start(),j));

	for(long i = src_sl0.start() + src_sl0.step(); i != src_sl0.stop(); i += src_sl0.step())
	    for(long j = src_sl1.start(); j != src_sl1.stop(); j += src_sl1.step())
		op(dest, src(i,j));
    }

    dest.getEnv().setNormalizer(IloTrue);
}

template <typename ReductionOp>
ExpressionArray* newFromReduction(const ExpressionArray& src, 
				  int axis, const ReductionOp& op, bool is_simple)
{
    ExpressionArray* dest_ptr;

    dest_ptr = new ExpressionArray(src.getEnv(), MetaData(
	        src.md().mode(), (axis == 1) ? src.shape(0) : 1, (axis == 0) ? src.shape(1) : 1));

    ExpressionArray& dest = *dest_ptr;

    switch(axis){
    case 0:
	for(long i = 0; i < src.shape(1); ++i)
	    reduction_op(dest(0,i), src, SliceFull(src.shape(0)), SliceSingle(i), op, is_simple);
	break;
    case 1:
	for(long i = 0; i < src.shape(0); ++i)
	    reduction_op(dest(i,0), src, SliceSingle(i), SliceFull(src.shape(1)), op, is_simple);
	break;
    default:
	reduction_op(dest(0,0), src, SliceFull(src.shape(0)), SliceFull(src.shape(1)), op, is_simple);
	break;
    }
    
    return dest_ptr;
}


ExpressionArray* newFromReduction(const ExpressionArray& src, int op_type, int axis)
{
    typedef ExpressionArray::Value Value;

    bool is_simple = !!(op_type & OP_SIMPLE_FLAG);
    
    switch(op_type & OP_SIMPLE_MASK){
    case OP_R_SUM: return newFromReduction(src, axis, ROp<OP_R_SUM, Value>(), is_simple);
    case OP_R_MAX: return newFromReduction(src, axis, ROp<OP_R_MAX, Value>(), is_simple);
    case OP_R_MIN: return newFromReduction(src, axis, ROp<OP_R_MIN, Value>(), is_simple);
    default: 
	assert(false);
	return NULL;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Binary operators

template <int OpType, typename D, typename S1, typename S2> struct Op {};

template <typename D, typename S1, typename S2> struct Op<OP_B_ADD, D, S1, S2> {
    inline D operator()(const S1& s1, const S2& s2) const { return s1 + s2; }
};

template <typename D, typename S1, typename S2> struct Op<OP_B_MULTIPLY, D, S1, S2>  {
    inline D operator()(const S1& s1, const S2& s2) const { return s1 * s2; }
};

template <typename D, typename S1, typename S2> struct Op<OP_B_SUBTRACT, D, S1, S2>  {
    inline D operator()(const S1& s1, const S2& s2) const { return s1 - s2; }
};

template <typename D, typename S1, typename S2> struct Op<OP_B_DIVIDE, D, S1, S2>  {
    inline D operator()(const S1& s1, const S2& s2) const { return s1 / s2; }
};

template <typename D, typename S1, typename S2> struct Op<OP_B_EQUAL, D, S1, S2>  {
    inline D operator()(const S1& s1, const S2& s2) const { return s1 == s2; }
};

template <typename D, typename S1, typename S2> struct Op<OP_B_NOTEQ, D, S1, S2>  {
    inline D operator()(const S1& s1, const S2& s2) const { return s1 != s2; }
};

template <typename D, typename S1, typename S2> struct Op<OP_B_LTEQ, D, S1, S2>  {
    inline D operator()(const S1& s1, const S2& s2) const { return s1 <= s2; }
};

template <typename D, typename S1, typename S2> struct Op<OP_B_LT, D, S1, S2>  {
    inline D operator()(const S1& s1, const S2& s2) const { return s1 < s2; }
};

template <typename D, typename S1, typename S2> struct Op<OP_B_GTEQ, D, S1, S2>  {
    inline D operator()(const S1& s1, const S2& s2) const { return s1 >= s2; }
};

template <typename D, typename S1, typename S2> struct Op<OP_B_GT, D, S1, S2>  {
    inline D operator()(const S1& s1, const S2& s2) const { return s1 > s2; }
};

template <typename DA, typename SA1, typename SA2, typename BinaryFunction>
void binary_op(DA& dest, const SA1& src1, const SA2& src2, const BinaryFunction& op, bool is_simple)
{
    // this is what it means to be simple
    dest.getEnv().setNormalizer(is_simple ? IloFalse : IloTrue);

    if(unlikely(dest.preferReversedTraverse()))
    {
	for(long j = 0; j < dest.shape(1); ++j)
	    for(long i = 0; i < dest.shape(0); ++i)
		dest(i,j) = op(src1(i,j), src2(i,j));
    }
    else
    {
	for(long i = 0; i < dest.shape(0); ++i)
	    for(long j = 0; j < dest.shape(1); ++j)
		dest(i,j) = op(src1(i,j), src2(i,j));
    }

    dest.getEnv().setNormalizer(IloTrue);
}

template <typename DA, typename SA1, typename SA2>
void matrix_multiply(DA& dest, const SA1& src1, const SA2& src2, bool is_simple)
{
    const long n_left = dest.shape(0);
    const long n_inner = src1.shape(1);
    const long n_right = dest.shape(1);

    assert_equal(dest.shape(0), src1.shape(0));
    assert_equal(src1.shape(1), src2.shape(0));
    assert_equal(dest.shape(1), src2.shape(1));

    IloEnv env = dest.getEnv();
	    
    env.setNormalizer(is_simple ? IloFalse : IloTrue);

    for(long left = 0; left < n_left; ++left)
    {
	for(long right = 0; right < n_right; ++right)
	{
	    dest(left, right) = src1(left, 0) * src2(0, right);

	    for(long inner = 1; inner < n_inner; ++inner)
		dest(left, right) += src1(left, inner) * src2(inner, right);
	}
    }

    dest.getEnv().setNormalizer(IloTrue);
}


////////////////////////////////////////////////////////////////////////////////
// A generic operator interface

// This function allows for easy wrapping with the cython functions 
template <typename SA1, typename SA2>
void binary_op(const int op_type, ExpressionArray& dest, const SA1& src1, const SA2& src2)
{
    typedef ExpressionArray::Value  DAValue;
    typedef typename SA1::Value SA1Value;
    typedef typename SA2::Value SA2Value;

    bool is_simple = !!(op_type & OP_SIMPLE_FLAG);

    switch(op_type & OP_SIMPLE_MASK) {

    case OP_B_ADD:     
	binary_op(dest, src1, src2, Op<OP_B_ADD, DAValue, SA1Value, SA2Value>(), is_simple);
	return;

    case OP_B_MULTIPLY:
	if(src1.md().matrix_multiplication_applies(src2.md()))
	    matrix_multiply(dest, src1, src2, is_simple);
	else
	    binary_op(dest, src1, src2, Op<OP_B_MULTIPLY, DAValue, SA1Value, SA2Value>(), is_simple);
	return;

    case OP_B_MATRIXMULTIPLY:
	matrix_multiply(dest, src1, src2, is_simple);
	return;
	
    case OP_B_ARRAYMULTIPLY:
	binary_op(dest, src1, src2, Op<OP_B_MULTIPLY, DAValue, SA1Value, SA2Value>(), is_simple);
	return;

    case OP_B_SUBTRACT:     
	binary_op(dest, src1, src2, Op<OP_B_SUBTRACT, DAValue, SA1Value, SA2Value>(), is_simple);
	return;

    case OP_B_DIVIDE:
	binary_op(dest, src1, src2, Op<OP_B_DIVIDE, DAValue, SA1Value, SA2Value>(), is_simple);
	return;

    default: 
	assert(false);
    }
}

// This function allows for easy wrapping with the cython functions 
template <typename SA1, typename SA2>
void binary_op(const int op_type, ConstraintArray& dest, const SA1& src1, const SA2& src2)
{
    typedef ConstraintArray::Value  DAValue;
    typedef typename SA1::Value SA1Value;
    typedef typename SA2::Value SA2Value;

    bool is_simple = !!(op_type & OP_SIMPLE_FLAG);

    switch(op_type & OP_SIMPLE_MASK) {

    case OP_B_EQUAL:
	binary_op(dest, src1, src2, Op<OP_B_EQUAL, DAValue, SA1Value, SA2Value>(), is_simple);
	return;
		
    case OP_B_NOTEQ:
	binary_op(dest, src1, src2, Op<OP_B_NOTEQ, DAValue, SA1Value, SA2Value>(), is_simple);
	return;
		
    case OP_B_LT:
	binary_op(dest, src1, src2, Op<OP_B_LT, DAValue, SA1Value, SA2Value>(), is_simple);
	return;

    case OP_B_LTEQ:
	binary_op(dest, src1, src2, Op<OP_B_LTEQ, DAValue, SA1Value, SA2Value>(), is_simple);
	return;

    case OP_B_GT:
	binary_op(dest, src1, src2, Op<OP_B_GT, DAValue, SA1Value, SA2Value>(), is_simple);
	return;

    case OP_B_GTEQ:
	binary_op(dest, src1, src2, Op<OP_B_GTEQ, DAValue, SA1Value, SA2Value>(), is_simple);
	return;

    default: 
	assert(false);
    }
}



#endif
