

#ifndef HK_DEBUG_H
#define HK_DEBUG_H

#ifdef assert
#undef assert
#endif

#ifndef NDEBUG

#warning ">>>>>>>>>>>>>>>>>>>>>> Debug On <<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

#define DEBUG_MODE true

#define DBHERE					\
    do{									\
	cout << __FILE__ << ":"						\
	     << __FUNCTION__ << ":"					\
	     << __LINE__ << ": "					\
	     << "HERE" << endl;						\
    }while(0)
    

#define db_printval(x)							\
    do{									\
	cout << __FILE__ << ":"						\
	     << __FUNCTION__ << ":"					\
	     << __LINE__ << ": "					\
	     << #x << " = " << (x) << endl;				\
    }while(0)

#define assert(x)							\
    do{									\
	if(!(x))							\
	{								\
	    cout << "ASSERTION FAILED: "				\
		 << __FILE__ << ":"					\
		 << __FUNCTION__ << ":"					\
		 << __LINE__ << ": "					\
		 << #x << endl;						\
	    abort();							\
	}								\
    }while(0)

#define assert_equal(x, y)						\
    do{									\
	if( (x) != (y) )						\
	{								\
	    cout << "ASSERT == FAILED: "				\
		 << __FILE__ << ":"					\
		 << __FUNCTION__ << ":"					\
		 << __LINE__ << ": "					\
		 << "(" << #x << ") " << (x)				\
		 << " != "						\
		 << (y) << " (" << #y << ") " << endl;			\
	    abort();							\
	}								\
    }while(0)

#define assert_almost_equal(x,y)					\
    do{									\
	double xv = (x);						\
	double yv = (y);						\
	if( abs(yv - xv) > max(xv, yv)*1e-8 )				\
	{								\
	    cout << "ASSERT == FAILED: "				\
		 << __FILE__ << ":"					\
		 << __FUNCTION__ << ":"					\
		 << __LINE__ << ": "					\
		 << "(" << #x << ") " << (x)				\
		 << " != "						\
		 << (y) << " (" << #y << ") " << endl;			\
	    abort();							\
	}								\
    }while(0)

#define assert_leq(x, y)						\
    do{									\
	if(!( (x) <= (y) ))						\
	{								\
	    cout << "ASSERT <= FAILED: "				\
		 << __FILE__ << ":"					\
		 << __FUNCTION__ << ":"					\
		 << __LINE__ << ": "					\
		 << "(" << #x << ") " << (x)				\
		 << " > "						\
		 << (y) << " (" << #y << ") " << endl;			\
	    abort();							\
	}								\
    }while(0)

#define assert_geq(x, y)						\
    do{									\
	if(!( (x) >= (y) ))						\
	{								\
	    cout << "ASSERT >= FAILED: "				\
		 << __FILE__ << ":"					\
		 << __FUNCTION__ << ":"					\
		 << __LINE__ << ": "					\
		 << "(" << #x << ") " << (x)				\
		 << " < "						\
		 << (y) << " (" << #y << ") " << endl;			\
	    abort();							\
	}								\
    }while(0)

#define assert_lt(x, y)							\
    do{									\
	if(!( (x) < (y) ))						\
	{								\
	    cout << "ASSERT < FAILED: "					\
		 << __FILE__ << ":"					\
		 << __FUNCTION__ << ":"					\
		 << __LINE__ << ": "					\
		 << "(" << #x << ") " << (x)				\
		 << " >= "						\
		 << (y) << " (" << #y << ") " << endl;			\
	    abort();							\
	}								\
    }while(0)

#define assert_gt(x, y)							\
    do{									\
	if(!( (x) >= (y) ))						\
	{								\
	    cout << "ASSERT > FAILED: "					\
		 << __FILE__ << ":"					\
		 << __FUNCTION__ << ":"					\
		 << __LINE__ << ": "					\
		 << "(" << #x << ") " << (x)				\
		 << " >= "						\
		 << (y) << " (" << #y << ") " << endl;			\
	    abort();							\
	}								\
    }while(0)

#define assert_notequal(x, y)						\
    do{									\
	if( (x) == (y) )						\
	{								\
	    cout << "ASSERT != FAILED: "				\
		 << __FILE__ << ":"					\
		 << __FUNCTION__ << ":"					\
		 << __LINE__ << ": "					\
		 << "(" << #x << ") " << (x)				\
		 << " == "						\
		 << (y) << " (" << #y << ") " << endl;			\
	    abort();							\
	}								\
    }while(0)

#else

#define DEBUG_MODE false

#define DBHERE
#define db_printval(x)
#define assert(x) 
#define assert_equal(x, y)
#define assert_notequal(x, y)
#define assert_almost_equal(x,y)		
#define assert_geq(x,y)
#define assert_gt(x,y)
#define assert_leq(x,y)
#define assert_lt(x,y)

#warning ">>>>>>>>>>>>>> Debug Off <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

#endif
#endif
