import numpy as np

"""
IndexToAssignment(I, D)
  Enumerates all the assignments with the leftmost column being the one 
  that varies the most (least significant) and the rightmost varying the least. 
  By sectioning it off, this algorithm determines the corrresponding assignment 
  index.

Input:
  Index I
  D - a vector containing the cardinality of each variable
Output:
  return the corresponding assignment indices 
"""
def IndexToAssignment (I, D):
    # Change I from a row vector to a column vector
    I_tile = np.tile((I-1)[:,np.newaxis], (1, D.__len__()))
    C_tile = np.tile(np.concatenate(([1], np.cumprod(D[:-1])), axis = 0),
                     (I.__len__(), 1))
    return np.mod(I_tile/C_tile, np.tile(D, (I.__len__(), 1))) + 1

"""
AssignmentToIndex(A,D)

Input: 
  Assignment A
  D - a vector containing the cardinality of each variable
Output:
  return the corresponding index
"""
def AssignmentToIndex (A, D):
    cprod = np.concatenate(([1], D[:-1]), axis = 0)
#    print A
#    print D
#    print cprod
    cprod = np.cumprod(cprod)
#    print cprod
    return ((A - 1).dot(cprod.T)) + 1

"""
FindIndices(A, varbs)

Input:
  Array A containing variables (all elements must be unique)
  varbs - the value aliases that you want to find in the array A (all elements must be unique)
Output:
  return the vector of indices in the same order as varbs
"""
def FindIndices (A, varbs):
    mapIndices = np.array([])
    for i, a in enumerate(A):
        if varbs.__contains__(a):
            mapIndices = np.append(mapIndices, i)
    #Have to convert to integer so we can use this to map an array
    mapIndices = np.asarray(mapIndices, dtype = int)
    return mapIndices
