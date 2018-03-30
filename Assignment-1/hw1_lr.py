from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        
    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        y = []
        X = []
        k = self.nb_features
        
        for value in values:
            y.append([value])
            
        if k == 1:
            for feature in features:
                X.append([1,feature[0]])
            X_transpose = numpy.array(X).transpose()   
            a = numpy.matmul(X_transpose, X)
            ainv = numpy.linalg.inv(a)    
        elif k > 1:
            for i in range(len(features)//(k+1)):
                line = [1]
                for j in range(k + 1):
                    line.append(features[i])
                    i += len(features)//(k+1)
                X.append(line)
                i -= k * len(features)//(k+1)
            global dimension
            dimension = int(len(X)/len(values))
            if dimension!=1:
                X_new = []
                i = 0
                while i < len(X):
                    line = [1]
                    for j in range(dimension):
                        for k in range(1,len(X[i])):
                            line.append(X[i][k])
                        i += 1
                    X_new.append(line)
                X = X_new
            lambd = 1
            X_transpose = numpy.array(X).transpose()
            a = numpy.matmul(X_transpose, X)
            '''
            identity_matrix = numpy.identity(len(a))
            revise = numpy.multiply(lambd,identity_matrix)
            a_revise = numpy.add(a, revise)
            ainv = numpy.linalg.inv(a_revise)
            '''
            ainv = numpy.linalg.pinv(a)
        b = numpy.matmul(ainv,X_transpose)
        global theta
        theta = numpy.matmul(b,y)
           
    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        X = []
        k = self.nb_features
        X_new = []
        if k == 1:
            for feature in features:
                X.append([1,feature[0]])
        elif k > 1:
            for i in range(len(features)//(k+1)):
                line = [1]
                for j in range(k + 1):
                    line.append(features[i])
                    i += len(features)//(k+1)
                X.append(line)
                i -= 2 * len(features)//(k+1)
            if dimension !=1:
                i = 0
                while i < len(X):
                    line = [1]
                    for j in range(dimension):
                        for k in range(1,len(X[i])):
                            line.append(X[i][k])
                        i += 1
                    X_new.append(line)
                X = X_new
        values_predict = numpy.matmul(X,theta)
        res = []
        for i in values_predict:
            res.append(i[0])
        return res

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        y = []
        X = []
        k = self.nb_features
        
        for value in values:
            y.append([value])
            
        if k == 1:
            for feature in features:
                X.append([1,feature[0]])
            X_transpose = numpy.array(X).transpose()   
            a = numpy.matmul(X_transpose, X)
            ainv = numpy.linalg.inv(a)    
        elif k > 1:
            for i in range(len(features)//(k+1)):
                line = [1]
                for j in range(k + 1):
                    line.append(features[i])
                    i += len(features)//(k+1)
                X.append(line)
                i -= k * len(features)//(k+1)
            global dimension
            dimension = int(len(X)/len(values))
            if dimension!=1:
                X_new = []
                i = 0
                while i < len(X):
                    line = [1]
                    for j in range(dimension):
                        for k in range(1,len(X[i])):
                            line.append(X[i][k])
                        i += 1
                    X_new.append(line)
                X = X_new
            X_transpose = numpy.array(X).transpose()
            a = numpy.matmul(X_transpose, X)
            identity_matrix = numpy.identity(len(a))
            revise = numpy.multiply(self.alpha,identity_matrix)
            a_revise = numpy.add(a, revise)
            ainv = numpy.linalg.inv(a_revise)
        b = numpy.matmul(ainv,X_transpose)
        global theta
        theta = numpy.matmul(b,y)
    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        X = []
        k = self.nb_features
        X_new = []
        if k == 1:
            for feature in features:
                X.append([1,feature[0]])
        elif k > 1:
            for i in range(len(features)//(k+1)):
                line = [1]
                for j in range(k + 1):
                    line.append(features[i])
                    i += len(features)//(k+1)
                X.append(line)
                i -= 2 * len(features)//(k+1)
            if dimension !=1:
                i = 0
                while i < len(X):
                    line = [1]
                    for j in range(dimension):
                        for k in range(1,len(X[i])):
                            line.append(X[i][k])
                        i += 1
                    X_new.append(line)
                X = X_new
        values_predict = numpy.matmul(X,theta)
        res = []
        for i in values_predict:
            res.append(i[0])
        return res
                        
    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
