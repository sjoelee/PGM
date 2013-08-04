import Factor as pgmf
import numpy as np


if __name__ == '__main__':

    A = pgmf.Factor(np.array([2,1]), np.array([2,2]), np.array([0.59, 0.41, 0.22, 0.78]))
    A.marginalize(np.array([2]))
    A.printFactor()
