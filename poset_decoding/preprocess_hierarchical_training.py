# coding=utf-8
import numpy as np
from collections import defaultdict
import re
from nltk.corpus import stopwords
from enum import Enum
from itertools import permutations
import re
import json
import random
from collections import OrderedDict
import pickle

# words = stopwords.words('english')

from collections import defaultdict
from functools import reduce

# from utils import *

class Sample:
	def __init__(self, query, sparql, tag):
		self.query = query
		self.sparql = sparql
		self.tag = tag
	def string():
		return '\t'.join([self.query, self.sparql, self.tag])

	
class Helper:
	def __init__(self):
		self.stop_words = ["Did", "and", ",", "'s", "M0", "M1", "M2", "M3", "M4", "M5","M6", "whose", "Whose", \
							"What", "did", "Was", "was", "Which", "Were", "were", "that", "M", "a"]
		self.stop_words += stopwords.words('english')
		self.dict = defaultdict(list)
		data = open("./data/phrase_table")
		for i in data:
			i = eval(i.strip())[0]
			self.dict[i[0]].append(i[1])
		

	def count_var(self, lf):
		value = {'?x0':2.5, '?x1':2, '?x2':1.5, '?x3':1, '?x4':0.5, '?x5':0}
		a1, r, a2 = lf.split()
		cn1, cn2, cn3, cn4 = 0, 0, 0, 0
		if a1.startswith("?x"):
			cn1 = 5
			cn1 += value[a1]
		if a2.startswith("?x"):
			cn2 = 3
			cn2 += value[a2]
		if r == 'a':
			cn3 = -1

		return cn1 + cn2 + cn3


	def term_extract(self, query, type):
		## 0520
		##改了新版的兼容 识别Did M

		terms = []
		entities = []
		if query.startswith("Did M") or query.startswith("Was M") or query.startswith("Were M") or query.startswith("Was a"):
			if type in ['mcd2', 'mcd3']:
				nl_pattern = query.split()[0] +" " + query.split()[1]
				terms.append((nl_pattern, [f'?x0#is#{query.split()[1]}'], (0, 1)))
			else:
				nl_pattern = query.split()[0] +" M"
				terms.append((nl_pattern, ['?x0#is#M'], (0, 1)))

		query = query.split()
		idx = 0

		####三元组
		while idx < len(query):
			
			if re.match(r'M[0-9]', query[idx]):
				entities.append(( query[idx:idx+1],query[idx:idx+1] ,(idx, idx)))
				idx += 1
			
			
			elif idx +1 <= len(query) and ' '.join(query[idx:idx+1]) in self.dict:
				terms.append((' '.join(query[idx:idx+1]), self.dict.get(' '.join(query[idx:idx+1])), (idx, idx)))
				idx += 1
			else:
				idx  +=1
		## 二元组
		idx = 0
		while idx < len(query) - 3:
			if idx +3 <= len(query) and ' '.join(query[idx:idx+3]) in self.dict:
				terms.append((' '.join(query[idx:idx+3]), self.dict.get(' '.join(query[idx:idx+3])),(idx, idx+2)))
			idx += 1
		idx = 0
		while idx < len(query) - 2:
			if idx +2 <= len(query) and' '.join(query[idx:idx+2]) in self.dict:
				terms.append(( ' '.join(query[idx:idx+2]), self.dict.get(' '.join(query[idx:idx+2])), (idx, idx+1)))
			idx += 1

		terms = sorted(terms, key = lambda s:s[2][0])
		# print(query, entities, terms)
		return entities, terms

		pass

	def fill_skeleton(self, query, skeleton, split):

		
		## 通过query + 对齐的双语词典align的结果得到候选的candidate triples
		## 候选的cndidate_triples通过给定的skeleton来过滤
		## 就是?x a M, ?x nationality 以及 gender的都做区分
		## v3的版本是把原始的M P ?x 换成了?x版本 无M开头的sparql
	
		def preprocess_sparql(query):

			tokens = []
			for token in query:
			# Replace 'ns:' prefixes.
				if token.startswith('ns:'):
					token = token[3:]
				# Replace mid prefixes.
				if token.startswith('m.'):
					token = 'm_' + token[2:]
				tokens.append(token)

			return ' '.join(tokens)

		def check_valid(skeleton_list, skeleton_pattern):
			skeleton_pattern = re.sub(r'\?x[0-9]', "?x", skeleton_pattern)
			# skeleton = re.sub(r'\?x[0-9]', "?x", skeleton)
			for skeleton in skeleton_list:
				if re.sub(r'\?x[0-9]', "?x", skeleton) not in skeleton_pattern:
					return False
			return True

		def transform_term_to_pattern(term):


			term_split = []
			for i in term.split():
				term_split +=  i.split("|||")

			skeleton_list = []
			term_list = []
			for i in term_split:
				if i.startswith("FILTER"):
					continue
				i = preprocess_sparql(i.split("#"))
				a1, r, a2 = i.split()

				if a1.startswith("?x") and a2.startswith("?x"):
					## ?x P ?x
					skeleton_list.append(f"{a1} P {a2}")
				elif a1.startswith("?x") and a2.startswith("M"):
					## ?x P M
					skeleton_list.append(f"{a1} P M")
				elif a1.startswith("?x") and r == "a":
					## ?x a M => ?x P M
					skeleton_list.append(f"{a1} a M")
				else:
					skeleton_list.append(f"{a1} V S")

				term_list.append(i)

			skeleton_str = []

			return skeleton_list, ' . '.join(term_list)



		entities, terms = self.term_extract(query, split)
		

		candidate_terms = defaultdict(set)
		for term in terms:
			for sub_term in term[1]:
				sub_pattern , sub_term = transform_term_to_pattern(sub_term)
				if check_valid(sub_pattern, skeleton):
					candidate_terms[" ".join(sub_pattern)].add(sub_term)

		
		candidate_triplets = defaultdict(list)
		# print("candidate_term:", candidate_terms)
		for candidate_skeleton, candidate_terms in candidate_terms.items():
			# a1, r, a2 = candidate_term.split("#")
			for candidate_term in candidate_terms:
				candidate_term = candidate_term.replace("#", " ")
				
				if candidate_term.count("M") == 1:
					if candidate_term.startswith("?x0 is M") and split in ['mcd2', 'mcd3']:
						candidate_triplets[candidate_skeleton] += [candidate_term]
					else:
						candidate_triplets[candidate_skeleton] += [''.join(candidate_term.replace("M", entity[0][0])) for entity in entities]
				elif candidate_term.count("M") == 2:
					candidate_term = list(candidate_term)
					index_m = candidate_term.index('M')
					candidate_term[index_m] = 'W'
					index_m = candidate_term.index('M')
					candidate_term[index_m] = 'Y'
					candidate_term = ''.join(candidate_term)
					for i in permutations(entities, 2):

						a1, a2 = i[0][0][0], i[1][0][0]
						# print(a1, a2, candidate_term)
						candidate_term_ =  candidate_term.replace("W", a1)
						candidate_term_ =  candidate_term_.replace("Y", a2)
						candidate_triplets[candidate_skeleton].append(candidate_term_)
				else:
					candidate_triplets[candidate_skeleton].append(candidate_term)

		return candidate_triplets


	
	def abstract_sparql_to_sketch(self,sparql):
		## 20200519
		## 把 M P ?x -> ?x0 is M ?x0 P M
		##首先把M开头的三元组排在最前面 

		sparql = sparql.replace("SELECT count(*) WHERE { ", " ")
		sparql = sparql.replace("SELECT DISTINCT ?x0 WHERE { ", " ")
		sparql = sparql.replace("M", "?M")

		sparql_list = [i.replace("?M", "M") for i in sorted(sparql.strip().split(" . "))]
		Mflag = True if sparql_list[0].startswith("M") else False
		FILTER_list, OTHER_list = [], []

		skeleton_list = []
		for item in sparql_list:
			if item.startswith("FILTER"):
				FILTER_list.append(item)
				continue
			OTHER_list.append(item)

			a1, r, a2 = item.strip().split()
			# print(a1, r, a2)

			if a1.startswith("?x") and a2.startswith("?x"):
				skeleton_list.append(f"{a1} P {a2}")
			elif a1.startswith("?x") and a2.startswith("M"):
				skeleton_list.append(f"{a1} P M")
			elif a2.startswith("?x") and a1.startswith("M"):
				skeleton_list.append("?xx P M")
				skeleton_list.append(f"?xx P {a2}")
			elif a1.startswith("M") and a2.startswith("M"):
				##这里其实是?x0 is M . ?x0 P M
				##合并一下就是?x0 P M
				skeleton_list.append(f"?xx P M")
			elif a1.startswith("?x") and r == "a":
				skeleton_list.append(f"{a1} a M")
			elif a1.startswith("M") and r == "a":
				skeleton_list.append("?xx P M")
				skeleton_list.append(f"?xx a M")
			elif re.match(r'M[0-9]', a1):
				skeleton_list.append("?xx P M")
				skeleton_list.append(f"?xx V S")
			else:
				skeleton_list.append(f"{a1} V S")
			

		skeleton_set = list(set(skeleton_list))
		skeleton_set.sort()
		skeleton_str = []
		OTHER_list.sort()
		FILTER_list.sort()
		sparql = ' . '.join(OTHER_list+FILTER_list)
		
		if Mflag:
			for token in " . ".join(skeleton_set).split():
				if token.startswith("?x") and token !='?xx':
					token = token[:2]+str(int(token[-1])+1)
				skeleton_str.append(token)
			skeleton_str = " ".join(skeleton_str).replace("?xx", "?x0").split(" . ")
			skeleton_str.sort()
			return sparql, " . ".join(skeleton_str)
		else:

			return sparql, " . ".join(skeleton_set)


	def generate_traversal_path(self, sparql):
	
		def trans_tuple_str(tuple_list):
			
			t_all = tuple_list[0]
			for i in range(1, len(tuple_list)):
				if isinstance(tuple_list[i], tuple):
					t_all += tuple_list[i]
				else:
					return False
		
			return ' '.join(t_all)
	
		results, triples, FILTER_triples = [], [], []
		
		for clf in sparql.split(" . "):
			if clf.startswith("FILTER"):
				continue

			elif len(clf.split()) != 3:
				continue
			a1, r, a2 = clf.split()
			var_cnt = self.count_var(clf)
			triples.append((a1, r, a2, var_cnt))

		split_dict = defaultdict(list)
		sorted_triples = sorted(triples, key=lambda k: k[-1])
	
		##划分方法
		for triple in sorted_triples:

			if isinstance(triple, tuple) and len(triple) == 4:
				arg1, rel, arg2, _ = triple
				triple = (arg1, rel, arg2)
				## 对于两个变量的三元组
				## 把他们尽可能的插入之前已有的三元组中
				## [?x0 ?x1] [?x1, ?x2] [?x2, ?x3]
				if arg1.startswith('?x') and arg2.startswith('?x'):
					##对于链式 的特定修正！！！
					## 每次需要更新他们匹配的组
					arg_max = arg1 if arg1 > arg2 else arg2
					arg_min = arg2 if arg1 > arg2 else arg1
					if len(split_dict[arg_max]) > 0:
						for cur_list in split_dict[arg_max]:
							cur_list_ = cur_list[:]
							cur_list_.insert(0, triple)
							split_dict[arg_min].append(cur_list_)
							
					else:
						split_dict[arg_max].append([triple])
				## 如果只有一个变量
				## 看能不能为之前添加的做补充
				## 形如(?x, r, M)为之前（M, r, ?x)的做补充
				elif  arg1.startswith('?x') and not arg1.startswith("?x0"):
					flag = True
					for t in split_dict[arg1]:
						if t[0][0] != arg1:
							t.append(triple)
							flag = False
					if flag:
						split_dict[arg1].append([triple])
					# print("h:",split_dict)
				##都没有 为该变量的第一个三元组关系
				elif (arg1.startswith("M") and arg2.startswith("?x") and not arg2.startswith("?x0")):
					flag =True
					for t in split_dict[arg2]:
							
							t.append(triple)
							flag = False
					if flag:
						split_dict[arg2].append([triple])
				else:
					variable = arg2 if arg2.startswith("?x") else arg1
					split_dict[variable].append([triple])
			else:
				split_dict[triple] = [triple]
	
		final_split = []


		for v in split_dict.values():
			for vv in v:
				vv_len = len(vv)
				xidx, xflag = 0, False
				for idx in range(vv_len):
					vv[idx] = ' '.join(vv[idx])
					if not xflag and vv[idx].startswith("?x"):
						xidx, xflag = idx, True

				vv = ' . '.join(vv[:xidx] + sorted(list(set(vv[xidx:vv_len]))) + vv[vv_len:])
				if not (vv.startswith('?x') and int(vv[2])> 0):
					##去掉不合法的?x
					final_split.append(vv)

				

		return final_split

	def distribute_triples_to_skeleton(self, skeleton_groups, candidate_triplets):
		fn = lambda x, code=',': reduce(lambda x, y: [str(i)+code+str(j) for i in x for j in y], x)
		ans = []

		def replace_variable(pattern, candidates):
			
			a1, _, a2 = pattern.split()
			modify_candidates = []
			for idx, candidate in enumerate(candidates):
				a1_c, r_c, a2_c = candidate.split()

				a1_c = a1 if a1_c == "?x" else a1_c
				a2_c = a2 if a2_c == "?x" else a2_c
				modify_candidates.append(' '.join([a1_c, r_c, a2_c]))
			# print("modify:", modify_candidates)
			return modify_candidates



		for skeleton_group in skeleton_groups:
			skeleton_group = skeleton_group.split(" . ")
			if len(skeleton_group) == 1:
				if skeleton_group[0] in candidate_triplets:
					ans += candidate_triplets.get(skeleton_group[0])
				temp_candidates = candidate_triplets.get(re.sub(r'\?x[0-9]', '?x', skeleton_group[0]), [])
				ans += replace_variable(skeleton_group[0], temp_candidates)
				
					
			else:
				triples_groups = [replace_variable(skeleton_item, candidate_triplets.get(re.sub(r'\?x[0-9]', '?x', skeleton_item), [])) for skeleton_item in skeleton_group]
				ans += fn(triples_groups, ' . ')
		return ans

	def generate_samples(self, query, sparql, triples, type):
		
		pos_ans, neg_ans = [], []

		valid_cnt = len([i for i in sparql.split(" . ") if not i.startswith("FILTER")])
		coverage_sparql = set()
		for triple_group in triples:
			flag = True
			for triple in triple_group.split(" . "):
				# print(triple)
				if triple not in sparql:
					flag = False
					continue
				else:
					coverage_sparql.add(triple)
			# print(flag, triple_group)

			if flag:
				pos_ans.append(Sample(query, triple_group, flag).__dict__)
				# print(triple_group)
			else:
				neg_ans.append(Sample(query, triple_group, flag).__dict__)


		coverage = True if  len(coverage_sparql) == valid_cnt else False

			
	
		if type == "train":
			return coverage, pos_ans + random.sample(neg_ans, min(len(neg_ans), len(pos_ans)))
		else:	
			return coverage, pos_ans+neg_ans

	def mask(self, query, sparqls):
	## return (orignal query, sparql), (masked query,sparql)

		entities = re.findall(r"M[0-9]",query)
		mask_query, mask_sparqls = query, []
		if len(entities) <=1:
			return (query, sparqls), (query, sparqls, dict())
		else:
			stack_tokens = []
			entity_tokens = []
			stack_state = False
			mask_mapping = dict()
			token_mapping = dict()
			query_tokens = query.split()
			for idx in range(len(query_tokens)):

				token = query_tokens[idx]
				if token.startswith("M") and (idx + 1 == len(query_tokens) or (idx+1 < len(query_tokens) and query_tokens[idx+1]!="'")):
					stack_tokens.append(token)
					entity_tokens.append(token)
					stack_state = True
				elif stack_state and (token == "," or token == "and"):
					stack_tokens.append(token)
				else:

					if len(entity_tokens) > 1:
						if stack_tokens[-1] == 'and' or stack_tokens[-1] == ',':
							stack_tokens = stack_tokens[:-1]
						mask_mapping[' '.join(stack_tokens)] = entity_tokens[0]
						token_mapping[entity_tokens[0]] = entity_tokens[1:]
					stack_tokens, stack_state, entity_tokens = [], False, []
			if len(entity_tokens) > 1:
				if stack_tokens[-1] == 'and' or stack_tokens[-1] == ',':
					stack_tokens = stack_tokens[:-1]
				mask_mapping[' '.join(stack_tokens)] = entity_tokens[0]
				token_mapping[entity_tokens[0]] = entity_tokens[1:]

			for key, v in mask_mapping.items():
				mask_query = mask_query.replace(key, v)

			if len(mask_mapping) == 0:
				return (query, sparqls), (query, sparqls, dict())

			
			for sparql_info in sparqls:
				flag = True
				for key, v in token_mapping.items():
					for vv in v:
						if vv in re.findall(r"M[0-9]",sparql_info[0]) :
							flag = False
				if flag:	
					mask_sparqls.append(sparql_info)
			assert len(mask_sparqls) <= len(sparqls), print("mask mapping", mask_mapping,token_mapping,  "\n",(query, sparqls),"\n", (mask_query, mask_sparqls, token_mapping))
			return (query, sparqls), (mask_query, mask_sparqls, token_mapping)
	

if __name__ == '__main__':
	helper = Helper()
	for split in ["mcd1", "mcd2", "mcd3"]:
		word_dict  =[word.strip() for word in open(f"./data/{split}/vocab.cfq.tokens").readlines()]
		src_vocab, sketch_vocab = set(), set()
		for type in ['train', 'dev', 'test']:
			src_data = open(f'./data/{split}/{type}/{type}_encode.txt')
			tgt_data = open(f'./data/{split}/{type}/{type}_decode.txt')
			sketch_list, tgt_list, poset_sketch_list, data_samples = [], [], [], []
			mapping_classification_data = defaultdict(list)

			for src, trg in zip(src_data, tgt_data):
				src, trg= src.strip(), trg.strip()
				trg = re.findall(r'[{](.*?)[}]', trg)[0].strip()

				## abstract sparql to sketch
				target, abstract_sketch = helper.abstract_sparql_to_sketch(trg)
				sketch_vocab = sketch_vocab | set(abstract_sketch.split())
				src_vocab = src_vocab | set(src.split())
				poset_abstract_sketch = helper.generate_traversal_path(abstract_sketch)
				sketch_list.append(abstract_sketch)
				tgt_list.append(target)
				poset_sketch_list.append(' ### '.join(poset_abstract_sketch))

				#### primitive prediction
				candidate_triplets = helper.fill_skeleton(src, abstract_sketch, split)
				final_triplets = helper.distribute_triples_to_skeleton(poset_abstract_sketch, candidate_triplets)
				_, samples = helper.generate_samples(src, trg, final_triplets, type)
				data_samples += samples

			src_dict = [word for word in word_dict if word in src_vocab ]
			sketch_dict = list(sketch_vocab)
			if type == 'train':
				open(f"./data/{split}/vocab.cfq.tokens.src", "w").write('\n'.join(src_dict))
				open(f"./data/{split}/vocab.cfq.tokens.sketch", "w").write('\n'.join(sketch_dict))	

			open(f"./data/{split}/{type}/{type}_sketch.txt", "w").write('\n'.join(poset_sketch_list))
			open(f"./data/{split}/{type}/{type}_target.txt", "w").write('\n'.join(tgt_list))
			json.dump(data_samples, open(f"./data/{split}/{type}/{type}_classification.json", "w"))
			mask_sample_info = [f"sentence1\tsentence2\tgold_label"]
			mask_full_info = [f"ori_sentence1\tsentence1\tsentence2\tgold_label\tmapping_entities"]
			for idx, item in enumerate(data_samples):
				query, sparql, tag = item.get('query'), item.get('sparql'), item.get('tag')
				mapping_classification_data[query].append((sparql, tag))
			for key, v in mapping_classification_data.items():	
				query_info,  mask_info = helper.mask(key, v)

				for vv in mask_info[1]:
					mask_sample_info.append(f"{mask_info[0]}\t{vv[0]}\t{vv[1]}")
					mask_full_info.append(f"{key}\t{mask_info[0]}\t{vv[0]}\t{vv[1]}\t{mask_info[-1]}")

			open(f"./data/{split}/{type}/{type}_mask_classification.csv", "w").write("\n".join(mask_sample_info))
			open(f"./data/{split}/{type}/{type}_mask_mapping.csv", "w").write("\n".join(mask_full_info))
