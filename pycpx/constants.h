#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

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

#define OP_SIMPLE_FLAG        64
#define OP_SIMPLE_MASK        (OP_SIMPLE_FLAG - 1)

#endif /* _CONSTANTS_H_ */
