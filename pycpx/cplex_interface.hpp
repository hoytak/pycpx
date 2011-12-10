#ifndef _ARRAY_FUNCTIONS_H_
#define _ARRAY_FUNCTIONS_H_

// Sorta provides intermediate wrapper functions for many of the
// operations; it's just easier to do this with templating here than
// in cython.

#include <ilconcert/iloalg.h>
#include <ilconcert/iloenv.h>
#include <ilconcert/ilosolution.h>
#include <ilconcert/ilomodel.h>
#include <ilcplex/ilocplexi.h>
#include <sstream>

#include "debug.h"
#include "optimizations.h"
#include "simple_shared_ptr.h"
#include "containers.hpp"
#include "operators.hpp"

using namespace std;

#define MODEL_UNBOUNDED 2
#define MODEL_INFEASABLE 3
#define MODEL_UNBOUNDED_OR_INFEASABLE 4

////////////////////////////////////////////////////////////////////////////////
// Now the same for the model

class CPlexModelInterface {
public:
  struct Status {
    Status(const char* _message, int _error_code = 1)
      : error_code(_error_code), message(_message)
    {
    }

    Status()
      : error_code(0), message("Okay.")
    {
    }
		
    int error_code;
    const char* message;
  };
  
  CPlexModelInterface(IloEnv _env) 
    : env(_env), model(_env), solver(_env), current_objective(NULL), model_extracted(false), model_solved(false)
  {
  }

  Status addVariables(const ExpressionArray& expr)
  {
    try{
      model.add(expr.variables());
    } catch(IloException& e) {
      return Status(e.getMessage());
    }

    return Status();
  }

  Status addConstraint(const ConstraintArray& cstr)
  {
    model_solved = false;

    try{
      model.add(cstr.constraint());
    } catch(IloException& e) {
      return Status(e.getMessage());
    }

    return Status();
  }

  Status setObjective(const ExpressionArray& expr, bool maximize)
  {
    model_solved = false;
	    
    try {
      if(current_objective != NULL) {
	model.remove(*current_objective);
	delete current_objective;
	current_objective = NULL;
      }
    } catch(IloException& e) {
      return Status(e.getMessage());
    }

    if(expr.shape(0) != 1 || expr.shape(1) != 1)
      return Status("Objective must be a scalar (or 1x1 matrix) expression.");

    try {
		
      if(maximize)
	current_objective = new IloObjective(IloMaximize(env, expr(0,0)));
      else
	current_objective = new IloObjective(IloMinimize(env, expr(0,0)));
		
      model.add(*current_objective);
    }
    catch(IloException& e) {
      return Status(e.getMessage());
    }

    return Status();
  }
    
  Status removeConstraint(const ConstraintArray& csr)
  {
    model_solved = false;

    try {
      model.remove(csr.constraint());
    } catch(IloException& e) {
      return Status(e.getMessage());
    }

    return Status();
  }

  Status setStartingValues(const ExpressionArray& expr, const NumericalArray& numr)
  {
    try {
      if(!model_extracted)
	extractModel();

      if(!expr.hasVar())
	return Status("Only variables may be set, not expressions.");

      IloNumArray X(env, expr.size());

      if(expr.isComplete()) 
	{
	  for(long i = 0; i < expr.shape(0); ++i)
	    for(long j = 0; j < expr.shape(1); ++j)
	      X[expr.getIndex(i,j)] = numr(i,j);

	  solver.getImpl()->setVectors(X, 0, expr.variables(), 0, 0, 0);
	} 
      else
	{
	  IloNumVarArray xpr(env, expr.size());

	  for(long i = 0; i < expr.shape(0); ++i) {
	    for(long j = 0; j < expr.shape(1); ++j)
	      {
		const IloNumVarArray& vars = expr.variables();
			    
		X[i*expr.shape(1) + j] = numr(i,j);
		xpr[i*expr.shape(1) + j] = vars[expr.getIndex(i,j)];
	      }
	  }
	  solver.getImpl()->setVectors(X, 0, xpr, 0, 0, 0);
	}
    }
    catch(IloException& e){
      return Status(e.getMessage());
    }

    return Status();
  }
    
  template <typename Param, typename V>
  Status setParameter(const Param& p, V value)
  {
    try{
      solver.setParam(p, value);
    } catch(IloException& e){
      return Status(e.getMessage());
    }

    return Status();
  }

  Status solve(IloNum * elapsed_time = NULL)
  {

    try{
      if(!model_extracted)
	extractModel();
    
      if(elapsed_time != NULL)
	*elapsed_time = -solver.getImpl()->getCplexTime();

      if(! solver.solve() )
	{
	  *elapsed_time = 0;

	  IloAlgorithm::Status status = solver.getStatus();
		    
	  switch(status) {
	  case IloAlgorithm::Unbounded:
	    return Status("Model unbounded.", MODEL_UNBOUNDED);
	  case IloAlgorithm::Infeasible:
	    return Status("Model infeasible.", MODEL_INFEASABLE);
	  case IloAlgorithm::InfeasibleOrUnbounded:
	    return Status("Model unbounded or infeasable.", MODEL_UNBOUNDED_OR_INFEASABLE);
	  default:
	    return Status("Unknown error occured while solving model.");
	  }
	}

      if(elapsed_time != NULL)
	*elapsed_time += solver.getImpl()->getCplexTime();
    }
    catch(IloException& e){
      return Status(e.getMessage());
    }
	    
    model_solved = true;

    return Status();
  }

  Status readBasis(const char* filename)
  {
    try{
      solver.readBasis(filename);
    }
    catch(IloException& e){
      return Status(e.getMessage());
    }
    return Status();
  }

  Status writeBasis(const char* filename)
  {
    try{
      solver.writeBasis(filename);
    }
    catch(IloException& e){
      return Status(e.getMessage());
    }
    return Status();
  }

  double getObjectiveValue()
  {
    if(!model_solved)
      return 0;
	    
    return solver.getObjValue();
  }

  Status getValues(NumericalArray& dest, const ExpressionArray& expr)
  {
    assert_equal(dest.shape(0), expr.shape(0));
    assert_equal(dest.shape(1), expr.shape(1));
	    
    // if(!model_solved)
    // 	return Status("Cannot get value; model not in a solved state.");
		    
    try{
      for(long i = 0; i < dest.shape(0); ++i)
	for(long j = 0; j < dest.shape(1); ++j)
	  dest(i,j) = solver.getValue(expr(i,j));
    }
    catch(IloException& e){
      return Status(e.getMessage());
    }
	    
    return Status();
  }

  long getNIterations() const
  {
    if(solved())
      return solver.getNiterations();
    else
      return 0;
  }

  bool solved() const { return model_solved; }
	
  string asString() const
  {
    ostringstream constraints;
    ostringstream objective;

    for(IloModel::Iterator it(model); it.ok(); ++it){
      if( (*it).isObjective())
	{
	  IloObjective obj = (*it).asObjective();
		    
	  objective << ( (obj.getSense() == IloObjective::Maximize) 
			 ? "maximize " : "minimize ")
		    << obj.getExpr() << " such that\n  ";
	}
      else
	constraints << (*it) << "\n  ";
    }
	    
    objective << constraints.str();
    return objective.str();
  }

private:

  void extractModel() 
  {
    env.setNormalizer(IloFalse);
    solver.extract(model);
    model_extracted = true;
    env.setNormalizer(IloTrue);
  }

  IloEnv env;
  IloModel model;
  IloCplex solver;
  IloObjective* current_objective;
  bool model_extracted;
  bool model_solved;
};

inline CPlexModelInterface::Status newCPlexModelInterface(CPlexModelInterface **cpx, IloEnv env)
{

  try {
    *cpx = new CPlexModelInterface(env);
    return CPlexModelInterface::Status();
  } catch(IloException& e) {
    return CPlexModelInterface::Status(e.getMessage());
  }
}

#endif /* _ARRAY_FUNCTIONS_H_ */



