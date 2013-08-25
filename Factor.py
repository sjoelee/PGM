from pgm_utility import *
import numpy as np

class Factor:
    """PGM Factor class"""
    def __init__ (self, variables, card, vals):
        self.varbs = variables
        self.card = card
        self.vals = vals

    def marginalize (self, variables):
        """Update the factor by marginalizing out the variables"""
        # Find the ones that are not in variables but in self.varbs
        varbs = np.setdiff1d(self.varbs, variables)
        tf, mapIndices = FindIndices(self.varbs, varbs)
        card = self.card[mapIndices]
        vals  = np.zeros((1, np.prod(card)))
        vals = vals.reshape(-1)

        # Get the assignments
        indices = np.arange(np.prod(self.card))
        assignments = IndexToAssignment(indices, self.card)
        # Subtract by 1 so we can access valid entries
        indx = AssignmentToIndex(assignments[:,mapIndices], card)
        for i, val in enumerate(self.vals):
            vals[indx[i]] = vals[indx[i]] + self.vals[i]

        # Assign to new factor
        self.varbs = varbs
        self.card = card
        self.vals = vals

    def observeEvidence (self, evidence):
        """ Modify a set of factors given some evidence.
        
        "evidence" is an N-by-2 matrix, where each row consists of a var/value pair. 
        variables are in the first column, values are in the second.
        """
        if evidence.shape == (2,):
            var, val = evidence
            tf, var_index = FindIndices(self.varbs, np.array([var]))
            if (tf == True):
                assignments = IndexToAssignment(np.arange(np.prod(self.card)), self.card)
                for a in assignments:
                    if (a[var_index] != val):
                        self.vals[AssignmentToIndex(a, self.card)] = 0
        else:
            for var, val in evidence:
                # Find where var is in factor.varbs
                # Make sure val valid (less than the card of the var)
                tf, var_index = FindIndices(self.varbs, np.array([var]))
                if (tf == True):
                    assignments = IndexToAssignment(np.arange(np.prod(self.card)), self.card)
                    for a in assignments:
                        if (a[var_index] != val):
                            self.vals[AssignmentToIndex(a, self.card)] = 0

    def printFactor (self):
        print "variables:   ", self.varbs
        print "cardinality: ", self.card
        print "values:      \n", self.vals


# NOTE: any reshape(-1) is to make a 2D vector into a 1D vector
# i.e. a 1x2 vector in 2D to a 1D vector of length 2.
def product (f1, f2):
    varbs = np.union1d(f1.varbs, f2.varbs)
    tf1, f1_to_prod_indices = FindIndices(varbs, f1.varbs)
    tf2, f2_to_prod_indices = FindIndices(varbs, f2.varbs)

    card = np.zeros((1, varbs.__len__())).reshape(-1)
    card[f1_to_prod_indices] = f1.card
    card[f2_to_prod_indices] = f2.card
    card = card.astype(int)

    vals = np.zeros((1, np.prod(card))).reshape(-1)
    index_array = np.arange(np.prod(card))
    assignments = IndexToAssignment(index_array.T, card)

    for i, assignment in enumerate(assignments):
        f1_idx_prod_assign = AssignmentToIndex(assignment[:,f1_to_prod_indices], f1.card)
        f2_idx_prod_assign = AssignmentToIndex(assignment[:,f2_to_prod_indices], f2.card)

        vals[i] = f1.vals[f1_idx_prod_assign] * f2.vals[f2_idx_prod_assign]

    prod = Factor(varbs, card, vals)
    return prod

def joint (factors):
    """ Given a list of factors, compute their joint distribution
    factors - a N x 1 array of factors, where N is the number of distinct factors
    
    return a Factor object that characterizes this distribution
    """
    jointFactor = factors[0]
    for factor in factors[1:]:
        jointFactor = product(jointFactor, factor)

    return jointFactor

def computeMarginal (varbs, factors, evidence):
    """Create a factor that has a distribution taken from the factors input and 
    takes the evidence into account.
    """
    resultFactor = joint(factors)
    resultFactor.observeEvidence(evidence)
    resultFactor.marginalize(np.setdiff1d(resultFactor.varbs, varbs))

    # Normalize values
    norm_sum = np.sum(resultFactor.vals)
    resultFactor.vals = resultFactor.vals/norm_sum

    return resultFactor    

def copy_factor (factor):
    factor_copy = Factor(factor.varbs, factor.card, factor.vals)
    return factor_copy
