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
        mapIndices = FindIndices(self.varbs, varbs)
        card = self.card[mapIndices]
        vals  = np.zeros((np.prod(card),1))

        # Get the assignments
        assignments = IndexToAssignment(np.arange(np.prod(self.card))+1, self.card)
        # Subtract by 1 so we can access valid entries
        indx = AssignmentToIndex(assignments[:,mapIndices], card) - 1
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


