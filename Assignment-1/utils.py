from typing import List

import numpy as np
import math

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    sum_square = 0
    for i in range(len(y_true)):
        sum_square += (y_true[i] - y_pred[i]) ** 2
    mean_square_error = sum_square / len(y_true)
    return mean_square_error


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    a = b = c = precision = recall = 0
    for i in range(len(real_labels)):
        if real_labels[i] == 1 and predicted_labels[i] == 1:
            a += 1
        elif real_labels[i] == 1 and predicted_labels[i] == 0:
            b += 1
        elif real_labels[i] == 0 and predicted_labels[i] == 1:
            c += 1
    try:
        precision = a/(a + c)
        recall = a / (a + b)
        f_score = 2 * (precision * recall)/(precision + recall)
    except ZeroDivisionError:
        f_score = 0
    return f_score

def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
    if k == 1:
        return features
    else:
        x_polynomial = []
        for fact in range(k+1):
            for i in range(len(features)):
                for j in range(len(features[i])):
                    a = round((features[i][j]**fact)/math.factorial(fact),6)
                    x_polynomial.append(a)
        return x_polynomial


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    sum = 0
    for i in range(len(point1)):
        sum += (point1[i]-point2[i]) ** 2
    return math.sqrt(sum)
    '''
    points = zip(point1, point2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))
    '''
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    sum = 0
    for i in range(len(point1)):
        sum += point1[i] * point2[i]
    return sum

def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    sum = 0
    for i in range(len(point1)):
        sum += (point1[i]-point2[i]) ** 2
    ed = math.sqrt(sum)
    return math.exp(-0.5 * ed)

def normalize(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[3, 4], [1, -1], [0, 0]],
    the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
    """
    res = []
    for i in range(len(features)):
        sum = 0
        line = []
        for j in range(len(features[i])):
            sum += features[i][j] ** 2
        if sum == 0:
            res.append(features[i])
        else:
            divisor = math.sqrt(sum)
            for j in range(len(features[i])):
                line.append(features[i][j]/divisor)
            res.append(line)
    return res

def min_max_scale(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[2, -1], [-1, 5], [0, 0]],
    the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
    """
    res = []
    for i in range(len(features)):
        max_value = max(features[i])
        min_value = min(features[i])
        line = []
        for j in range(len(features[i])):
            new_value = (features[i][j] - min_value)/(max_value-min_value)
            line.append(new_value)
        res.append(line)
    return res
