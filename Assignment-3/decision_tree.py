import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			res =[0 for i in range(len(branches[0]))]
			for i in range(len(branches[0])): ###list of sum of each branches
				for j in range(len(branches)):
					res[i] += branches[j][i]

			avg_entropy = 0
			for i in range(len(branches[0])):
				entropy = 0
				for j in range(len(branches)):
					val = branches[j][i]/res[i]
					if val == 0 or val == 1:
						entropy += 0
					else:
						entropy += val * np.log2(val)
				avg_entropy += res[i]/np.sum(res) * (-entropy)
			return avg_entropy

		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			dim = []
			unique_dim = []
			min_score = 2147483647
			for i in range(len(self.features)):
				dim.append(self.features[i][idx_dim])
			unique_dim = np.unique(dim) ###compute unique values 

			branches =[[0 for i in range(len(unique_dim))] for j in range(self.num_cls)]
			for i in range(len(unique_dim)):###computer the branches
				for j in range(len(self.features)):
					if self.features[j][idx_dim] == unique_dim[i]:
						branches[self.labels[j]][i] += 1 ###update the branches(what are the labels like?)
			score = conditional_entropy(branches)
			if score < min_score:###find the best feature to split
				min_score = score
				self.dim_split = idx_dim
		

		############################################################
		# TODO: split the node, add child nodes
		############################################################
		s =[]
		for i in range(len(self.features)):
			s.append(self.features[i][self.dim_split]) #features to be splitted
		
		unique_s = np.unique(s)
		self.feature_uniq_split = unique_s.tolist()

		for i in range(len(unique_s)):###compute the branches
			'''
			new_branches =[[0 for i in range(1)] for j in range(self.num_cls)]
			for i in range(1):###computer the branches
				for j in range(len(self.features)):
					if self.features[j][0] == unique_s[i]:
						new_branches[self.labels[j]][0] += 1 ###update the branches(what are the labels like?)
			score = conditional_entropy(new_branches)
			'''

			new_labels = []
			new_features = []
			for x in range(len(self.features)):
				if self.features[x][self.dim_split] == unique_s[i]:
					l =[]
					for y in range(len(self.features[0])):
						if y!=self.dim_split: ###remove the dim_split feature
							l.append(self.features[x][y])
					new_features.append(l)
					new_labels.append(self.labels[x])	
			node = TreeNode(new_features,new_labels,np.max(new_labels)+1)
			if len(node.features[0]) == 0:
				node.splittable = False
			self.children.append(node)
	
		
		# split the child nodes
		for child in self.children:
			'''
			if len(child.features[0]) == 0:
				child.splittable = False
			'''
			if child.splittable:
				child.split()
		return
		
	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])

			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



