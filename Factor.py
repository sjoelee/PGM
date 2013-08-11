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
        # Find the ones that are not in varaibles but in self.varbs
        varbs = np.setdiff1d(self.varbs, variables)
        tf, mapIndices = FindIndices(self.varbs, varbs)
        card = self.card[mapIndices]
        vals  = np.zeros((np.prod(card),1))

        # Get the assignments
        assignments = IndexToAssignment(np.arange(np.prod(self.card)), self.card)
        # Subtract by 1 so we can access valid entries
        indx = AssignmentToIndex(assignments[:,mapIndices], card)
        for i, val in enumerate(self.vals):
            vals[indx[i]] = vals[indx[i]] + self.vals[i]

        # Assign to new factor
        self.varbs = varbs
        self.card = card
        self.vals = vals

    def printFactor (self):
        print "variables:   ", self.varbs
        print "cardinality: ", self.card
        print "values:      \n", self.vals


def product (f1, f2):
    varbs = np.union1d(f1.varbs, f2.varbs)
    tf1, f1_to_prod_indices = FindIndices(varbs, f1.varbs)
    tf2, f2_to_prod_indices = FindIndices(varbs, f2.varbs)

    card = np.zeros((varbs.__len__(), 1))
    card[f1_to_prod_indices] = f1.card[:, np.newaxis]
    card[f2_to_prod_indices] = f2.card[:, np.newaxis]
    card = card.astype(int)

    vals = np.zeros((np.prod(card), 1))
    index_array = np.arange(np.prod(card))+1
#    print "Index Array: ", index_array
#    print "Card: ", card

    assignments = IndexToAssignment(index_array.T, card)
#    print "For loop: "
#    print f1_to_prod_indices
#    print f2_to_prod_indices
#    print assignments
#
    for i, assignment in enumerate(assignments):
#        print "Product Assignment: ", assignment
        f1_idx_prod_assign = AssignmentToIndex(assignment[:,f1_to_prod_indices], f1.card)
        f2_idx_prod_assign = AssignmentToIndex(assignment[:,f2_to_prod_indices], f2.card)

#        print "Assignment: ", assignment[:,f1_to_prod_indices]
#        print "Assignment: ", assignment[:,f2_to_prod_indices]
#        print "ATOI: ", f1_idx_prod_assign
#        print "ATOI: ", f2_idx_prod_assign
#        print f1.vals[f1_idx_prod_assign]
#        print f2.vals[f2_idx_prod_assign]
        vals[i] = f1.vals[f1_idx_prod_assign] * f2.vals[f2_idx_prod_assign]

    prod = Factor(varbs, card, vals)
    return prod

def copy_factor (factor):
    factor_copy = Factor(factor.varbs, factor.card, factor.vals)
    return factor_copy
