# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:47:07 2018

@author: jiasen
"""
from __future__ import division

class Matrix:
    
    def __init__(self, data):
        
        # self-checking
        dim_row_vector = self.__listLen(data[0])
        for rw in range(self.__listLen(data)):
            assert(self.__listLen(data[rw]) == dim_row_vector)
            
        self.num_row = self.__listLen(data)
        self.num_col = self.__listLen(data[0])
        self.data = data
        
    def __listLen(self, listA):
        if type(listA) is list:
            return len(listA)
        else:
            return 1
            
    def display(self):
        
        # show elements
        print(self.num_row, "x", self.num_col, " matrix:")
        
        for rw in range(len(self.data)):
            print(self.data[rw])
            
        print()
    
    def subMat(self, list_rw, list_cl):
        
        # extract sub matrix given rows and cols
        return self.data
            
    def matMul(self, matB):
        
        # matrix multiplication
        assert(type(matB) == type(self))
        assert(self.num_col == matB.num_row)
        
        matMulResult = Matrix([[0 for col in range(matB.num_col)]\
                                    for row in range(self.num_row)])
                     
        for rw in range(self.num_row):
            for cl in range(matB.num_col):
                for idx in range(self.num_col):
                    matMulResult.data[rw][cl] +=\
                        self.data[rw][idx] * matB.data[idx][cl]
        
        return matMulResult
    
    def T(self):
        
        # matrix transposition
        matTransResult = Matrix([[0 for col in range(self.num_row)]\
                                      for row in range(self.num_col)])

        for rw in range(self.num_row):
            for cl in range(self.num_col):
                matTransResult.data[cl][rw] = self.data[rw][cl]

        return matTransResult
        
    def add(self, matB):
        
        # matrix addition
        assert(type(matB) == type(self))
        assert(self.num_col == matB.num_col)
        assert(self.num_row == matB.num_row)

        matAddResult = Matrix([[0 for col in range(self.num_col)]\
                                    for row in range(self.num_row)])
                     
        for rw in range(self.num_row):
            for cl in range(self.num_col):
                matAddResult.data[rw][cl] =\
                    self.data[rw][cl] + matB.data[rw][cl]
    
        return matAddResult
    
    def sub(self, matB):
        
        # matrix subtraction
        assert(type(matB) == type(self))
        assert(self.num_col == matB.num_col)
        assert(self.num_row == matB.num_row)

        matSubResult = Matrix([[0 for col in range(self.num_col)]\
                                    for row in range(self.num_row)])
                     
        for rw in range(self.num_row):
            for cl in range(self.num_col):
                matSubResult.data[rw][cl] =\
                    self.data[rw][cl] - matB.data[rw][cl]
    
        return matSubResult
    
    def numMul(self, k):
        
        # numerical multiplication
        numMulResult = Matrix([[0 for col in range(self.num_col)]\
                                    for row in range(self.num_row)])
                     
        for rw in range(self.num_row):
            for cl in range(self.num_col):
                numMulResult.data[rw][cl] = self.data[rw][cl] * k

        return numMulResult

    def numAdd(self, k):
        
        # numerical addition
        numAddlResult = Matrix([[0 for col in range(self.num_col)]\
                                    for row in range(self.num_row)])
                     
        for rw in range(self.num_row):
            for cl in range(self.num_col):
                numAddlResult.data[rw][cl] = self.data[rw][cl] + k

        return numAddlResult

    def dotMul(self, matB):
        
        # dot
        assert(type(matB) == type(self))
        assert(self.num_col == matB.num_col)
        assert(self.num_row == matB.num_row)

        dotMulResult = Matrix([[0 for col in range(self.num_col)]\
                                    for row in range(self.num_row)])
                     
        for rw in range(self.num_row):
            for cl in range(self.num_col):
                dotMulResult.data[rw][cl] =\
                    self.data[rw][cl] * matB.data[rw][cl]
    
        return dotMulResult

    def dotDiv(self, matB):
        
        # dot
        assert(type(matB) == type(self))
        assert(self.num_col == matB.num_col)
        assert(self.num_row == matB.num_row)

        dotDivResult = Matrix([[0 for col in range(self.num_col)]\
                                    for row in range(self.num_row)])
                     
        for rw in range(self.num_row):
            for cl in range(self.num_col):
                assert(matB.data[rw][cl] != 0)
                dotDivResult.data[rw][cl] =\
                    self.data[rw][cl] / matB.data[rw][cl]
    
        return dotDivResult
