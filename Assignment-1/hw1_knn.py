from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################
from collections import Counter
from operator import itemgetter
class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        global training_set
        training_set = []
        for i in range(len(features)):
            training_set.append([features[i],labels[i]])
            
    def predict(self, features: List[List[float]]) -> List[int]:
        ans=[]
        res = []
        for i in range(len(features)):
            res = []
            for j in range(len(training_set)):
                distance = self.distance_function(features[i], training_set[j][0])
                res.append([training_set[j],distance])
            sorted_res = sorted(res,key=itemgetter(1))
            sorted_training_instance = [x[0] for x in sorted_res]
            nearest_k = sorted_training_instance[:self.k]
            classes = [neighbor[1] for neighbor in nearest_k]
            count = Counter(classes)
            ans.append(count.most_common()[0][0])
        return ans

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
