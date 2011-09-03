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

using namespace std;

//////////////////////////////////////////////////
// Define the basic operations

#define OP_B_ADD	        1
#define OP_B_MULTIPLY	        2
#define OP_B_MATRIXMULTIPLY	3
#define OP_B_ARRAYMULTIPLY	4
#define OP_B_SUBTRACT		5
#define OP_B_DIVIDE		6
#define OP_B_EQUAL		7
#define OP_B_NOTEQ		8
#define OP_B_LT			9
#define OP_B_LTEQ		10 
#define OP_B_GT			11
#define OP_B_GTEQ		12

#define OP_U_NO_TRANSLATE	1
#define OP_U_ABS		2
#define OP_U_NEGATIVE		3

#define OP_R_SUM		1
#define OP_R_PROD		2
#define OP_R_MAX		3
#define OP_R_MIN		4

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
void reduction_op(D& dest, const SA& src, const Slice0& src_sl0, const Slice1& src_sl1, const ReductionOp& op)
{
    dest = src(src_sl0.start(), src_sl1.start());

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
void binary_op(DA& dest, const SA1& src1, const SA2& src2, const BinaryFunction& op)
{
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
}

template <typename DA, typename SA1, typename SA2>
void matrix_multiply(DA& dest, const SA1& src1, const SA2& src2)
{
    const long n_left = dest.shape(0);
    const long n_inner = src1.shape(1);
    const long n_right = dest.shape(1);

    assert_equal(dest.shape(0), src1.shape(0));
    assert_equal(dest.shape(1), src2.shape(1));
	    
    for(long left = 0; left < n_left; ++left)
    {
	for(long right = 0; right < n_right; ++right)
	{
	    dest(left, right) = src1(left, 0) * src2(0, right);
	    
	    for(long inner = 1; inner < n_inner; ++inner)
		dest(left, right) += src1(left, inner) * src2(inner, right);
	}
    }
}

#endif
