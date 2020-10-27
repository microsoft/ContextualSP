import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tree:
    def __init__(self, value):
        # value = [1] tensor, the value is: output_token_idx
        # value of tree root should be [word_to_idx('<sos>')]
        self.value = value
        self.children = dict()


class Trie:
	def __init__(self, node = None, sos_token = None):
		if node:
			self.root = node
		elif sos_token:
			self.root = Tree(sos_token)

	def add_path(self, sparql):
		sparql_list = sparql.split()
		node = self.root
		for v in sparql_list:
			# a1, r, a2 = sub_sparql.split()
			# for v in sub_sparql.split():
			if v not in node.children:
				node.children[v] = Tree(v)
			node = node.children[v]

	def get_label(self, sparql):
		## 目的是根据树获得target的标签
		target_list = []
		sparql_list = sparql.split()
		node = self.root
		for token in sparql_list:
			target_list.append(' '.join(list(node.children.keys())))
			node = node.children[token]
		return target_list

	def print_tree(self):
		def dfs(node):
			if len(node.children) > 0:
				print(f"value:{node.value}, children:{node.children.keys()}")
				for k, v in node.children.items():
					dfs(v)
		dfs(self.root)

	def get_path(self):
		ans = []
		path = []
		def dfs(node, ans, path):
			if len(node.children) > 0:
				# print(f"value:{node.value}, children:{node.children.keys()}")
				for k, v in node.children.items():
					dfs(v, ans, path+[k])
			else:
				ans.append(path)
				path = []
		dfs(self.root, ans, path)

		return ans


def collate_fn(samples):

	source_samples, target_samples, label_samples, duplicate_len = [], [], [], []
	for sample in samples:
		source_samples += sample[0]
		target_samples += sample[1]
		label_samples += sample[2]
		duplicate_len.append(sample[3])


	return source_samples, target_samples, label_samples, duplicate_len

