import Factor as pgmf
import pgm_utility as util
import numpy as np
import unittest2 as unittest

"""
Note: Always ALWAYS test each function thoroughly before using it. What is the 
expected behavior? What would happen if you did not give it the proper input?
Do you get the proper error? Will the user know what happened? Will you know??

Do your due diligence so that you don't pay heavily in the end.
"""

class TestPGMUtilityFunctions (unittest.TestCase):
    def setUp (self):
        self.factorA = pgmf.Factor(np.array([1]), np.array([2]), np.array([0.11, 0.89]))
        self.factorB = pgmf.Factor(np.array([2,1]), np.array([2,2]), np.array([0.59, 0.41, 0.22, 0.78]))
        self.factorC = pgmf.Factor(np.array([3,2]), np.array([2,2]), np.array([0.39, 0.61, 0.06, 0.94]))

    def test_AssignmentToIndex1 (self):
        # Test #1
        idx = util.AssignmentToIndex(np.array([1,1,1]), np.array([2,2,2]))
        np.testing.assert_equal(idx, 0)

    def test_AssignmentToIndex2 (self):
        # Test #2
        idx = util.AssignmentToIndex(np.array([2,2,2]), np.array([2,2,2]))
        np.testing.assert_equal(idx, 7)

#    def test_AssignmentToIndex3 (self):
#        # Test #3
#        idx = util.AssignmentToIndex(np.array([[2,1],[2,3]]), np.array([3,4]))
#        np.testing.assert_equal(idx, np.array([5,11]))

    def test_IndexToAssignment1 (self):
        # Test #1
        assignment = util.IndexToAssignment(np.array([0]), np.array([2,2,2]))
        np.testing.assert_equal(assignment, np.array([[1,1,1]]))

    def test_IndexToAssignment2 (self):
        # Test #2
        assignment = util.IndexToAssignment(np.array([7]), np.array([2,2,2]))
        np.testing.assert_equal(assignment, np.array([[2,2,2]]))

    def test_IndexToAssignment3 (self):
        # Test #2
        assignment = util.IndexToAssignment(np.array([5,11]), np.array([3,4]))
        np.testing.assert_equal(assignment, np.array([[3,2],[3,4]]))

    def test_marginalize1 (self):
        # Test #1
        expectedFactor = pgmf.Factor(np.array([1]), np.array([2]), np.array([1,1]))
        self.factorB.marginalize(np.array([2]))
#        np.testing.assert_equal(self.factorB, expectedFactor)
        np.testing.assert_array_equal(self.factorB.varbs, expectedFactor.varbs)
        np.testing.assert_array_equal(self.factorB.card, expectedFactor.card)
        np.testing.assert_array_equal(self.factorB.vals, expectedFactor.vals)

    def test_marginalize2 (self):
        # Test #2
        expectedFactor = pgmf.Factor(np.array([3]), np.array([2]), np.array([0.45, 0.55]))
        self.factorC.marginalize(np.array([2]))
        np.testing.assert_equal(self.factorC, expectedFactor)

        # Reset factorC
        self.factorC = pgmf.Factor(np.array([3,2]), np.array([2,2]), np.array([0.39, 0.61, 0.06, 0.94])) 
#                                     
#        # Test #3
#        expectedFactor = pgmf.Factor(np.array([3]), np.array([2]), np.array([1,1]))
#        self.factorC.marginalize(np.array([3]))
#        np.testing.assert_equal(self.factorC, expectedFactor)
#
#        self.factorC = pgmf.Factor(np.array([3,2]), np.array([2,2]), np.array([0.39, 0.61, 0.06, 0.94])) 
#
    def test_FindIndices1 (self):
        # Test #1 - test to see if all variables are located correctly in the array
        A = np.array([5,4,3,2,1])
        B = np.array([5,4,3,2,1])
        tf, C = util.FindIndices(A, B)

        np.testing.assert_array_equal(tf, [True,True,True,True,True])
        np.testing.assert_array_equal(C, [0,1,2,3,4])

    def test_FindIndices2 (self):
        # Test #2 
        A = np.array([5,4,3,2,1])
        B = np.array([0])
        tf, C = util.FindIndices(A, B)
        np.testing.assert_array_equal(tf, [False])
        np.testing.assert_array_equal(C, [0])

    def test_FindIndices3 (self):
        # Test #3
        A = np.array([5,4,3,2,1])
        B = np.array([1,3,5,9])
        tf, C = util.FindIndices(A, B)
        np.testing.assert_array_equal(tf, [True,True,True,False])
        np.testing.assert_array_equal(C, [4,2,0,0])

def main():
#    unittest.test_FindIndices();
    unittest.main()

if __name__ == '__main__':
    main()

# Checklist:
# Test FindIndices
# Test IndexToAssignment
# Test AssignmentToIndex
# Test marginalize method in Class Factor
# Test product method in Class Factor
