# code check
import numpy as np
import os
import json
from typing import List

from classifier import Classifier

class DecisionStump(Classifier):
	def __init__(self, s:int, b:float, d:int):
		self.clf_name = "Decision_stump"
		self.s = s
		self.b = b
		self.d = d

	def train(self, features: List[List[float]], labels: List[int]):
		pass
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			if feature[self.d] > self.b:
				y_pred.append(self.s)
			else:
				y_pred.append(-self.s)
		return y_pred

# decision_stump.py
def check_decision_stump():
	np.random.seed(8)
	test_sample = np.random.rand(5,5)

	try:
		import decision_stump
		test_decision_stump = decision_stump.DecisionStump(1, 0.5, 3)
		predictions = test_decision_stump.predict(test_sample.tolist())

	except:
		return 0
	if np.squeeze(predictions).tolist() == [1,1,1,-1,-1]:
		score = 0.5
	else:
		score = 0
	return round(score, 1)

# boosting.py
def check_boosting():
	np.random.seed(5)

	test_classifiers = set()
	s_set = {1, -1}
	b_set = {1, 3, 5, 7, 9, 11}
	for s in s_set:
		for b in b_set:
			test_classifiers.add(DecisionStump(s,b,0))
	test_features = [[2], [4], [6], [8], [10]]
	test_labels = [1, 1, -1, -1, 1]


	try:
		import boosting
	except: 
		return 0

	try:
		test_ada1 = boosting.AdaBoost(test_classifiers, 1)
		test_ada2 = boosting.AdaBoost(test_classifiers, 3)
		test_ada1.train(test_features, test_labels)
		test_ada2.train(test_features, test_labels)
		predictions_ada1 = test_ada1.predict(test_features)
		predictions_ada2 = test_ada2.predict(test_features)

		if predictions_ada1 == [1,1,-1,-1,-1] and predictions_ada2 == [1,1,-1,-1,1]:
			score_ada = 0.5
		else:
			score_ada = 0
	except:
		score_ada = 0

	try:
		test_logit1 = boosting.LogitBoost(test_classifiers, 1)
		test_logit2 = boosting.LogitBoost(test_classifiers, 3)
		test_logit1.train(test_features, test_labels)
		test_logit2.train(test_features, test_labels)
		predictions_logit1 = test_logit1.predict(test_features)
		predictions_logit2 = test_logit2.predict(test_features)
		if predictions_logit1 == [1,1,-1,-1,-1] and predictions_logit2 == [1,1,-1,-1,1]:
			score_logit = 0.5
		else:
			score_logit = 0
	except:
		score_logit = 0

	return round(score_ada + score_logit, 1)

def check_decision_tree():
	try:
		results = json.load(open('decision_tree.json', 'r'))
		if results['test_accu'] >= 0.8 and  results['train_accu'] >= 0.8:
			score_results = 0.5
		else:
			score_results = 0
	except:
		return 0

	test_features = [[0, 0], [0, 0], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]]
	test_labels = [0, 0, 0, 1, 1, 1, 0, 0, 1]

	try:
		import decision_tree
		test_tree = decision_tree.DecisionTree()
		test_tree.train(test_features, test_labels)
		predictions = test_tree.predict(test_features)
		if predictions == [0, 0, 0, 0, 1, 1, 0, 0, 0]:
			score_tree = 0.5
		else:
			score_tree = 0
	except:
		return 0

	return round(score_results + score_tree, 1)

def check_pegasos():

    try:
        os.system("python3 pegasos.py")  # run students' code

        with open('pegasos.json') as data_file:  # load results
            result = json.load(data_file)

        corr = 0.0
        for key, value in result[0].items():
            if value > 0.7:
                corr += 1
        score = 1.5 * corr / 6. # final score

        return score

    except:

        return 0.0


if __name__ == "__main__":

	score_decision_stump = check_decision_stump()
	score_boosting = check_boosting()
	score_decision_tree = check_decision_tree()
	
	score_pegasos = check_pegasos()
	
	score = [score_pegasos] + [score_decision_stump + score_boosting] + [score_decision_tree]
	
	with open('output_hw3.csv', 'w') as f:
		for i in range(len(score)):
			f.write(str(score[i]) + '\n')