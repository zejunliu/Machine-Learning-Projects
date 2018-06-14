import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
    # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO: implement "predict"
		########################################################
		labels = []
		for i in range(len(features)):
			sum = 0
			for j in range(len(self.clfs_picked)):
				if features[i][self.clfs_picked[j].d] > self.clfs_picked[j].b:
					sum += self.clfs_picked[j].s * self.betas[j]
				else:
					sum -= self.clfs_picked[j].s * self.betas[j]
			if sum > 0:
				labels.append(1)
			else:
				labels.append(-1)
		return labels

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		w = [[0 for n in range(N)] for m in range(self.T +1)] ###row weight of feature, col iteration
		w[0] = [1/N for n in range(N)] ###initial weight of each feature is 1/N

		for t in range(self.T): ### find T classifiers(T iterations)
			min_w_sum = 2147483647
			best_d = 0
			best_b = 0.0
			best_s = 0

			for clf in self.clfs:### find the best classifier
				w_sum = 0 ### sum of weight of n features whose label is not equal to  classifer estimate

				for j in range(len(features)): ### compute the error rate of each classifier
					a = 0
					if features[j][clf.d] > clf.b:
						a = clf.s
					else:
						a = -clf.s
					if a != labels[j]:
						w_sum += w[t][j]
				if w_sum < min_w_sum:
					min_w_sum = w_sum
					best_d = clf.d
					best_b = clf.b
					best_s = clf.s

			self.clfs_picked.append(DecisionStump(best_s,best_b,best_d))
			error = min_w_sum
			beta = 0.5 * np.log((1-error)/error)
			self.betas.append(beta)
			
			for i in range(len(features)): #upadate weight of N features
				b =0
				if features[i][best_d] > best_b:
					b = best_s
				else:
					b = -best_s
				if b == labels[i]:
					w[t+1][i] = w[t][i] * np.exp(-beta)
				else:
					w[t+1][i] = w[t][i] * np.exp(beta)
			
			sum = 0 #normalize weight
			for i in range(len(w[0])):
				sum += w[t+1][i]
			for i in range(len(w[0])):
				w[t+1][i] /= sum

	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)

class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		pi = [[0 for i in range(N)] for j in range(self.T + 1)]### pi_0 is 1/2, N*(T +1) matrix
		pi[0] = [0.5 for i in range(N)]
		z = [[0 for i in range(N)] for j in range(self.T)] ### z is N * T
		w = [[0 for i in range(N)] for j in range(self.T)] ### w is N * T, w_i[0] = 1/N, i =1,2,...N
		w[0] = [0 for i in range(N)]
		f = [[0 for i in range(N)] for j in range(self.T + 1)] ### F(x) = 0
		for t in range(self.T):### T iterations
			for n in range(len(features)): ### update w and z
				z[t][n]=((labels[n]+1)/2 - pi[t][n])/(pi[t][n]*(1 - pi[t][n]))
				w[t][n] = pi[t][n] * (1 - pi[t][n])
			min_w_sum = 2147483647
			best_d = 0
			best_b = 0.0
			best_s = 0
			for clf in self.clfs: ### find the best classifier
				w_sum = 0 ### compute the sum of weight of each classifier
				for n in range(len(features)):
					h_xn = 0
					if features[n][clf.d] > clf.b:
						h_xn = clf.s
					else:
						h_xn = -clf.s
					w_sum += w[t][n] * np.square(z[t][n] - h_xn)
				if w_sum < min_w_sum: ### find the best classifer h_t
					min_w_sum = w_sum
					best_d = clf.d
					best_b = clf.b
					best_s = clf.s			
			self.clfs_picked.append(DecisionStump(best_s,best_b,best_d)) ### add the best classifier 
			self.betas.append(1) ### should we use 0.5 or 1 as beta?
			for n in range(len(features)):
				h_t_x = 0
				if features[n][best_d] > best_b:
					h_t_x = best_s
				else:
					h_t_x = -best_s
				f[t+1][n] = f[t][n] + 0.5 * h_t_x

			for n in range(len(features)): ### the last step use f to update pi
				pi[t+1][n] = 1/(1+ np.exp(-2 * f[t+1][n])) 
					
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
