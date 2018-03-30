from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        for iter in range(self.max_iteration):
            i_random = np.arange(len(features))
            np.random.shuffle(i_random)
            update_times = 0
            for i in i_random:
                sum = 0
                for j in range(len(features[i])):
                    sum += features[i][j] * self.w[j]
                    if sum > self.margin:
                        predict_label = 1
                    else:
                        predict_label = -1  
                if predict_label != labels[i]:
                    divisor = 0
                    for j in range(len(features[i])):
                        divisor += features[i][j] ** 2
                    d = labels[i]/np.sqrt(divisor)
                    for j in range(len(features[i])):
                        self.w[j] += features[i][j] * d
                    update_times += 1
                else:
                    continue
            if update_times != 0:
                continue
            else:
                return True   
        return False
                    
                    
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        res = []
        for i in range(len(features)):
            sum = 0
            for j in range(len(features[i])):
                sum += self.w[j]*features[i][j]
            if sum > self.margin:
                res.append(1)
            else:
                res.append(-1)
        return res

    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
    