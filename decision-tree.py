#!/usr/bin/python
import argparse
import numpy as np
import re
import math
import sys
from time import time

def binary_entropy(a, b):
	p_a = 0 if (a + b) == 0 else a * 1.0 / (a + b)
	p_b = 0 if (a + b) == 0 else b * 1.0 / (a + b)
	entropy = 0.0
	if abs(p_a - 0.0) > 1e-9:
		entropy += -p_a * math.log(p_a, 2)
	if abs(p_b - 0.0) > 1e-9:
		entropy += -p_b * math.log(p_b, 2)
	return entropy

def split_candidates_for_numeric(col):
	unique_col = np.unique(col)
	return [(unique_col[i-1] + unique_col[i]) / 2.0 for i in range(1, len(unique_col))]

def information_gain_for_numeric(col, output, split):
	left = [output[idx] for idx in range(0, len(col)) if col[idx] <= split]
	right = [output[idx] for idx in range(0, len(col)) if col[idx] > split]
	lnpos = sum(left)
	lnneg = len(left) - lnpos
	rnpos = sum(right)
	rnneg = len(right) - rnpos
	npos = sum(output)
	nneg = len(col) - npos
	entropy_cur = binary_entropy(npos, nneg)
	entropy_next = len(left) * 1.0 / len(col) * binary_entropy(lnpos, lnneg) + len(right) * 1.0 / len(col) * binary_entropy(rnpos, rnneg)
	return entropy_cur - entropy_next

def	information_gain(T, a, attrs, output):
	values = []
	i = attrs.index(a)
	col = T[:,i]
	if a.attr_type == 'real' or a.attr_type == 'integer' or a.attr_type == 'numeric':
		max_info_gain = -float("inf")
		candidates = split_candidates_for_numeric(col)
		for candidate in candidates:
			info_gain = information_gain_for_numeric(col, output, candidate)
			max_info_gain = max(info_gain, max_info_gain)
		return max_info_gain
	if a.attr_type == 'nominal':
		npos = sum(output)
		nneg = len(col) - npos
		entropy_cur = binary_entropy(npos, nneg)
		entropy_next = 0.0
		for v in range(0, len(a.values)):
			nv = sum([1 for idx in range(0, len(col)) if col[idx] == v])
			npos_a_v = sum([1 for idx in range(0, len(col)) if output[idx] == 1 and col[idx] == v])
			nneg_a_v = nv - npos_a_v
			entropy_next += nv * 1.0 / len(col) * binary_entropy(npos_a_v, nneg_a_v)
		return entropy_cur - entropy_next

def choose_attr(T, attrs, output):
	argmax_a = None
	max_info_gain = -float("inf")
	for a in attrs:
		info_gain = information_gain(T, a, attrs, output)
		if info_gain > max_info_gain:
			max_info_gain = info_gain
			argmax_a = a
	# incase of negative information gain, return None
	return None if max_info_gain < 0.0 else argmax_a

def learn(dt, node, dataset, m):
	nneg, npos = split(dataset)
	# stop splitting if there is a pure split or number of instances is less than m
	if len(dataset.data) < m or npos == 0 or nneg == 0:
		return
	a = choose_attr(dataset.data, dataset.attrs, dataset.output)
	if a is None:
		return
	else:
		attr_index = dataset.attrs.index(a)
		attr_values = dataset.data[:,attr_index]
		if a.attr_type == 'nominal':
			for i in range(0, len(a.values)):
				v = a.values[i]
				npos = sum([1 for x, y in zip(attr_values, dataset.output) if x == i and y == 1])
				nneg = sum([1 for x, y in zip(attr_values, dataset.output) if x == i and y == 0])
				to_idx = dt.create_node((nneg, npos))
				dt.add_edge(Label(a.attr_name, a.attr_type, v, lambda x, y: x == y, '='), node.idx, to_idx)
				learn(dt, dt.get_node(to_idx), filter_by_attr_value(dataset, a, v, lambda x, y: x == y), m)
		elif a.attr_type == 'real' or a.attr_type == 'integer' or a.attr_type == 'numeric':
			max_info_gain = -float("inf")
			argmax_v = None
			candidates = split_candidates_for_numeric(attr_values)
			for candidate in candidates:
				info_gain = information_gain_for_numeric(attr_values, dataset.output, candidate)
				if info_gain > max_info_gain:
					max_info_gain = info_gain
					argmax_v = candidate
			v = argmax_v
			lnpos = sum([1 for x, y in zip(attr_values, dataset.output) if x <= v and y == 1])
			lnneg = sum([1 for x, y in zip(attr_values, dataset.output) if x <= v and y == 0])
			rnpos = sum([1 for x, y in zip(attr_values, dataset.output) if x > v and y == 1])
			rnneg = sum([1 for x, y in zip(attr_values, dataset.output) if x > v and y == 0])
			left_idx = dt.create_node((lnneg, lnpos))
			right_idx = dt.create_node((rnneg, rnpos))
			dt.add_edge(Label(a.attr_name, a.attr_type, v, lambda x, y: x <= y, '<='), node.idx, left_idx)
			dt.add_edge(Label(a.attr_name, a.attr_type, v, lambda x, y: x > y, '>'), node.idx, right_idx)
			learn(dt, dt.get_node(left_idx), filter_by_attr_value(dataset, a, v, lambda x, y: x <= y, True), m)
			learn(dt, dt.get_node(right_idx), filter_by_attr_value(dataset, a, v, lambda x, y: x > y, True), m)

def filter_by_attr_value(dataset, attr, value, filter, retain_attr = False):
	# filter rows
	j = dataset.attrs.index(attr)
	data = []
	if not retain_attr:
		attrs = dataset.attrs[:j] + dataset.attrs[j+1:]
	else:
		attrs = dataset.attrs
	output = []
	for i in range(0, len(dataset.data)):
		value_to_check = dataset.data[i,j]
		if attr.attr_type == 'nominal':
			value_to_check = attr.values[int(value_to_check)]
		if filter(value_to_check, value):
			if not retain_attr:
				data.append(dataset.data[i,:j].tolist() + dataset.data[i,j+1:].tolist())
			else:
				data.append(dataset.data[i,:].tolist())
			output.append(dataset.output[i])
	return Dataset(dataset.name, attrs, np.array(data), np.array(output), dataset.output_attr)

def learn_decision_tree(dataset, m):
	dt = DecisionTree()
	nneg, npos = split(dataset)
	node_idx = dt.create_node((nneg, npos))
	node = dt.get_node(node_idx)
	learn(dt, node, dataset, m)
	return dt

def split(dataset):
	return (len(dataset.output) - sum(dataset.output), sum(dataset.output))

def classify(dt, dataset):
	attrs = dataset.attrs
	output_attr = dataset.output_attr
	output_class_labels = [output_attr.values[int(x)] for x in dataset.output]
	visualize(dt, attrs, output_attr, dt.get_node(0), 0, None)

	predictions = []
	print("<Predictions for the Test Set Instances>")
	for i in range(0, len(dataset.data)):
		instance = dataset.data[i, :].tolist()
		node = dt.get_node(0)
		out_edges = dt.get_out_edges(node.idx)
		best_class_so_far = None
		# iterate until there are no edges to traverse ie. until you reach a leaf node
		while out_edges:
			# picking the majority class at each node
			if node.split[0] > node.split[1]:
				best_class_so_far = output_attr.values[0]
			elif node.split[1] > node.split[0]:
				best_class_so_far = output_attr.values[1]

			for edge in out_edges:
				label = edge.label
				attr_index = [j for j in range(0, len(attrs)) if attrs[j].attr_name == label.attr_name][0]
				attr_value = instance[attr_index]
				if label.attr_type == 'nominal':
					# convert to actual value
					attr_value = int(attr_value)
					possible_values = attrs[attr_index].values
					attr_value = possible_values[attr_value]
				if label.comparator(attr_value, label.value):
					node = dt.get_node(edge.toNode)
					out_edges = dt.get_out_edges(node.idx)
					break
		# picking the majority class at each node
		if node.split[0] > node.split[1]:
			best_class_so_far = output_attr.values[0]
		elif node.split[1] > node.split[0]:
			best_class_so_far = output_attr.values[1]
		predictions.append(best_class_so_far)
		print("{0}: Actual: {1} Predicted: {2}".format(i+1, output_class_labels[i], best_class_so_far))
	no_correct_preds = sum([1 for x, y in zip(predictions, output_class_labels) if x == y])
	print("Number of correctly classified: {0} Total number of test instances: {1}".format(no_correct_preds, len(predictions)))

'''
Do a DFS traversal of the decision tree to print it out, indent is a function of level number
'''
def visualize(dt, attrs, output_attr, node, level, parent_class):
	out_edges = dt.get_out_edges(node.idx)
	for edge in out_edges:
		to = dt.get_node(edge.toNode)
		label = edge.label
		line = '|\t' * level + str(edge) + ' ' + str(to)
		nneg, npos = to.split
		pred_class = parent_class
		if nneg > npos:
			pred_class = output_attr.values[0]
		elif npos > nneg:
			pred_class = output_attr.values[1]
		# to is a leaf node, print class label too
		if not dt.get_out_edges(to.idx):
			nneg, npos = to.split

			line += ': ' + pred_class
		print(line)
		# go down the tree
		visualize(dt, attrs, output_attr, to, level + 1, pred_class)

class Dataset:
	def __init__(self, name, attrs, data, output, output_attr):
		self.name = name
		self.attrs = attrs
		self.data = data
		self.output = output
		self.output_attr = output_attr

class DecisionTree:
	def __init__(self, nodes = [], edges = {}):
		self.nodes = nodes
		self.edges = edges

	def create_node(self, split):
		cur_id = len(self.nodes)
		self.nodes.append(Node(len(self.nodes), split))
		return cur_id

	def get_node(self, idx):
		return self.nodes[idx]

	def add_edge(self, label, fromNode, toNode):
		edge = Edge(label, fromNode, toNode)
		if fromNode not in self.edges.keys():
			self.edges[fromNode] = [edge]
		else:
			self.edges[fromNode].append(edge)

	def get_out_edges(self, fromNode):
		return self.edges[fromNode] if fromNode in self.edges.keys() else []


class Node:
	def __init__(self, idx, split):
		self.idx = idx
		self.split = split

	def __str__(self):
		return "[{0} {1}]".format(self.split[0], self.split[1])

class Edge:
	def __init__(self, label, fromNode, toNode):
		self.label = label
		self.fromNode = fromNode
		self.toNode = toNode

	def __str__(self):
		return str(self.label)

class Label:
	def __init__(self, attr_name, attr_type, value, comparator, operator):
		self.attr_name = attr_name
		self.attr_type = attr_type
		self.value = value
		self.comparator = comparator
		self.operator = operator

	def __str__(self):
		if self.attr_type == 'nominal':
			return "{0} {1} {2}".format(self.attr_name, self.operator, self.value)
		else:
			return "%s %s %.6f" % (self.attr_name, self.operator, self.value)

class Attribute:
	def __init__(self, attr_name, attr_type, values):
		self.attr_name = attr_name
		self.attr_type = attr_type
		self.values = values

	def __str__(self):
		return "attr_name = {0}, attr_type = {1}, values = {2}".format(self.attr_name, self.attr_type, self.values)

# read ARFF file
def readDataSet(path):
	relation = ''
	attr_dict = {}
	attrs = []
	matrix = []
	output_vector = []
	output_attr = None
	parsing_data = False
	with open(path) as f:
		for line in f.readlines():
			# remove newlines, trailing and leading whitespaces
			l = line.strip()
			# skipping comment lines
			if l.startswith('%'):
				continue
			elif l.lower().startswith('@relation'):
				relation = re.split('[\t ]+', l, maxsplit=1)[1]
			elif l.lower().startswith('@attribute'):
				# assumes that attribute names don't contain spaces, that would make life too hard
				tokens = re.split('[\t ]+', l, maxsplit=2)
				attr_name = tokens[1].strip("\'")
				attr_type = tokens[2]
				values = []
				if attr_type.startswith('{') and attr_type.endswith('}'):
					attr_type = attr_type.strip('{}')
					values = [attr_value.strip() for attr_value in attr_type.split(',')]
					attr_type = 'nominal'
				if attr_name == 'class':
					output_attr = Attribute('class', 'nominal', values)
				else:
					attr = Attribute(attr_name, attr_type.lower(), values)
					attrs.append(attr)
			elif l.lower().startswith('@data'):
				parsing_data = True
			else:
				if parsing_data:
					tokens = l.split(',')
					feature_vector = tokens[:-1]
					pred_col = tokens[-1]
					conv_feature_vector = []
					for v, a in zip(feature_vector, attrs):
						col_value = v
						if a.attr_type == 'nominal':
							col_value = a.values.index(v)
						elif a.attr_type == 'real' or a.attr_type == 'numeric':
							col_value = float(v)
						elif a.attr_type == 'integer':
							col_value = int(v)
						conv_feature_vector.append(col_value)
					matrix.append(conv_feature_vector)
					output_vector.append(output_attr.values.index(pred_col))
		data = np.array(matrix)
		output = np.array(output_vector)
		return Dataset(relation, attrs, data, output, output_attr)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'A script that can be run from command line to train and classify with decision trees')
	parser.add_argument('train_data', help = 'Relative path to the file containing training data')
	parser.add_argument('test_data', help = 'Relative path to the file containing test data')
	parser.add_argument('m', type = int, default = 1, help = 'A parameter that limits the depth of the decision tree')
	args = parser.parse_args()
	train_dataset = readDataSet(args.train_data)
	test_dataset = readDataSet(args.test_data)
	dt = learn_decision_tree(train_dataset, args.m)
	classify(dt, test_dataset)
