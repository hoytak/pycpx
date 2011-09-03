import unittest, cPickle, collections
from copy import deepcopy, copy
from pycpx import *
from numpy import *
from numpy import array as ar
import numpy.random as rn 

def getWithDimensionCheck(m, var, size):
    
    v = m[var]

    if size is None:
        assert isscalar(v)
    elif isscalar(size):
        assert isinstance(v, ndarray)
        assert v.ndim == 1
        assert v.size == size
    else:
        assert isinstance(v, ndarray)
        assert v.shape == size

    return v
    
        
