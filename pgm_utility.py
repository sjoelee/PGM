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
def IndexToAssignment (I, card):
    # Make sure the elements are integers
    # We'll typecast them. Warning: may lose information
    I = I.astype(int)
    card = card.astype(int)

    # Make sure I is a column vector and card is a row vector
    I = np.reshape(I, (-1,1))
    card = np.reshape(card, (1,-1))

    I_tile = np.tile(I, (1, card.shape[1]))
    C_tile = np.tile(np.concatenate(([1], np.cumprod(card[0][:-1])), axis = 0),
                     (I.shape[1], 1))

    return np.mod(I_tile/C_tile, np.tile(card, (I.shape[1], 1))) + 1

"""
AssignmentToIndex(A,card)

Input: 
  Assignment A
  card - a vector containing the cardinality of each variable
Output:
  return the corresponding index
"""
def AssignmentToIndex (A, card):
    cprod = np.concatenate(([1], card[:-1]), axis = 0)
    cprod = np.cumprod(cprod)
#    if (A.shape[0] > 1):
#        cprod = np.reshape(cprod, (1,-1)) 
#    else:
#        cprod = np.reshape(cprod, (-1, 1))
#    A = np.reshape(A, (1, -1))

    # Reduce to a 1-D array instead of a 2-D vector
    return ((A - 1).dot(cprod.reshape((-1,1)))).reshape(-1)

"""
FindIndices(A, varbs)
This is a copy of someone's implementation of ismember in stackoverflow.com

Input:
  Array A containing variables (all elements must be unique)
  varbs - the value aliases that you want to find in the array A (all elements must be unique)
Output:
  return the vector of indices in the same order as varbs
"""
def FindIndices (A, varbs):
    tf = np.array([variable in A for variable in varbs])
    u = np.unique(varbs[tf])
    mapIndices = np.array([(np.where(A == variable))[0][-1] if t else 0 for variable, t in zip(varbs,tf)])

    return tf, mapIndices

