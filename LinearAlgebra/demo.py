# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:34:49 2018

@author: jiase
"""

from MatrixLinearAlgebra import Matrix

# define matrix
matA = Matrix([[1, 2],
               [3, 4],
               [5, 6]])

matB = Matrix([[-1, 1, -1],
               [1, -1, 1]])

matC = Matrix([[0.5, 1],
               [1.5, 2],
               [2.5, 3]])

# display elements value
print("matA and matB:")
matA.display()
matB.display()

# transposition
print("transpose A:")
matA.T().display()

# matrix multiplication
print("matA * matB:")
matResult = matA.matMul(matB)
matResult.display()

# matrix addition
print("matA + matB:")
matResult = matA.add(matC)
matResult.display()

# matrix numerical multiplication
print("matA * k:")
matResult = matA.numMul(0.5)
matResult.display()

# matrix dot multiplication
print("matA .* matB:")
matResult = matA.dotMul(matC)
matResult.display()

# matrix dot division
print("matA ./ matB:")
matResult = matA.dotDiv(matC)
matResult.display()