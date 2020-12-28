

import numpy as np
from collections import defaultdict
import re
from nltk.corpus import stopwords
from enum import Enum
from itertools import permutations
import re
import json
import random
# words = stopwords.words('english')

from collections import defaultdict
from functools import reduce

class ResType(Enum):
	ENTITY = 1,
	RELATION = 2

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

	def load_phrase_table(self):
		self.dict = defaultdict(list)
		data = open("./coor_file-0520.filter_wto_no")
		for i in data:
			i = eval(i.strip())[0]
			self.dict[i[0]].append(i[1])
			# maxlen = max(maxlen, len(i[0].split()))
			# if len(i[0].split()) > 3:
			# 	print(i)
		# print(self.dict)
		
		# self.indexTree = Trie()
		# for key in self.dict:
		# 	self.indexTree.add(key)

	def count_var(self, lf):
		value = {'?x0':2.5, '?x1':2, '?x2':1.5, '?x3':1, '?x4':0.5, '?x5':0}
		a1, r, a2 = lf.split()
		cn1, cn2, cn3 = 0, 0, 0
		if a1.startswith("?x"):
			cn1 = 5
			cn1 += value[a1]
		if a2.startswith("?x"):
			cn2 = 3
			cn2 += value[a2]
		if r == 'a':
			cn3 = -1

		return cn1 + cn2 + cn3


	def add_index(self, file):
		data = np.array(open(file).readlines()).reshape(-1, 3)
		res = []
		for item in data:
			_, src, trg = item
			# print(_, src, trg)
			src_new = []
			for idx, token  in enumerate(src.split(), 1):
				src_new.append(f'{token} ({idx})')
			res.append('\n'.join([_, ' '.join(src_new), trg]))
		return "\n".join(res)


	def update_output_format(self, version):

		for type in ["formula2query", "query2formula"]:
			file = f"/home/v-yinguo/DKI/GIZA_alignment/{version}/{type}/{type}.A3.final"
			open(f'{file}.update', "w").write(self.add_index(file))

	def statistic_coor(self, q2f, f2q):
		## type = 1是query2formula的映射
		## type = 0是formula2query的映射

		

		def split_up(sent):
			 return [token for token in sent.split() if not token.startswith("(")]

		def split_bottom(sent):
			tokens = re.split(r' *\({[0-9 ]*}\) *', sent)[1:-1]
			return tokens
		
		def split_alignment(sent):
			alignment = [re.findall(r'[0-9].', i) for i in re.findall(r'\({[0-9 ]*}\)', sent)]
			##把开始的NULL去掉
			# print(alignment)
			alignment = [[int(ii) for ii in i] if len(i) else [0] for i in alignment[1:]]
			## 将alignment 处理成连续的span
			alignment_combine = []
			# print(alignment)
			for sub_align in alignment:
				sub_align_span = []
				start, end = 0, 0
				for idx in range(len(sub_align) - 1):
					if sub_align[idx + 1] == sub_align[idx] + 1:
						end += 1
					else:
						sub_align_span.append((sub_align[start], sub_align[end]))
						start, end = idx + 1, idx + 1
				sub_align_span.append((sub_align[start], sub_align[end]))
				alignment_combine.append(sub_align_span)
			return alignment_combine



		coor_dict = defaultdict(int)
		for type, file in enumerate([f2q, q2f]):
			# print(f"file name:{file}")
			for index, line in enumerate(open(file),1):
				line = line.strip()
				# print(line)
				if (not type and index % 5 == 3) or (type and index % 5 == 4) :
					query = line
				elif (not type and index % 5 == 4) or (type and index % 5 == 3) :
					formula = line
				elif index %5 == 0:
					query_tokens = split_bottom(query) if type else split_up(query)
					formula_tokens= split_up(formula) if type else split_bottom(formula)
					alignment = split_alignment(query) if type else split_alignment(formula)
					# print(f"len formula:{len(formula_tokens)}, len(query):{len(query_tokens)},len alignment:{len(alignment)}")
					# print(f"query:{query_tokens}\nformula:{formula_tokens}\nalignment:{alignment}")
					assert(len(formula_tokens) == len(alignment) or len(query_tokens) == len(alignment))
					
					if not type:
						for spans, formula_token in zip(alignment, formula_tokens):
							for sub_span in spans:
								s_pos, e_pos = sub_span[0], sub_span[1]
								query_token = ' '.join(query_tokens[s_pos-1:e_pos])
								if s_pos == e_pos:
									if s_pos and not re.match(r'M[0-9]', query_token) and query_token not in self.stop_words:
										# print(f"1-1 matching:{s_pos}-{e_pos} {query_token}-{formula_token}")
										coor_dict[(query_token, formula_token)] += 1
								else:
									# print(f"multi matching:{s_pos}-{e_pos} {query_token}-{formula_token}")
									coor_dict[(query_token, formula_token)] += 1
					else:
						# print(f"\n\nori query:{query}\nori formula:{formula}")
						# print(f"query:{query_tokens}\nformula:{formula_tokens}\nalignment:{alignment}")
						for spans, query_token in zip(alignment, query_tokens):
							if query_token in self.stop_words or re.match(r'M[0-9]', query_token):
								continue



							for sub_span in spans:
								s_pos, e_pos = sub_span[0], sub_span[1]
								formula_token = ' '.join(formula_tokens[s_pos-1:e_pos])
								if s_pos == e_pos:
									if s_pos:
										# print(f"1-1 matching:{s_pos}-{e_pos} {query_token}-{formula_token}")
										coor_dict[(query_token, formula_token)] += 1
								else:
									# print(f"multi matching:{s_pos}-{e_pos} {query_token}-{formula_token}")
									coor_dict[(query_token, formula_token)] += 1
						pass




		# print(coor_dict)
		sorted_coor_dict = sorted(coor_dict.items(), key=lambda s:s[1], reverse = True)
		# print(sorted_coor_dict)
		return sorted_coor_dict

	def filter_str(self, line):
		qf, cnt = eval(line.strip())
		key, v = qf


		
		if key in self.stop_words:
			return
		if v.count("|||") > 1:
			return
		key_entities = re.findall(r'M[0-9]', key)
		v_entities = re.findall(r'M[0-9]', v)
		key_entities.sort()
		v_entities.sort()

		if len(key_entities) == 0:
			v = re.sub(r'M[0-9]', 'M', v)
			# coor_dict[(key, v)] += cnt
		elif len(v_entities) == 0:
			key = re.sub(r'M[0-9]', 'M', key)
		elif len(key_entities) == len(v_entities):
			key = re.sub(r'M[0-9]', 'M', key)
			v = re.sub(r'M[0-9]', 'M', v)
		else:
			return

		while(len(key)):
			if key.split()[0] in self.stop_words and '#is#M' not in v:
				key = ' '.join(key.split()[1:])

			elif key.split()[-1] in self.stop_words and '#is#M' not in v:
				key = ' '.join(key.split()[:-1])
			else:
				break

		if len(key) == 0:
			return

		if len(key.split()) == 1 and len(v.split(" ")) > 2:
			return
		if len(key.split()) == 1 and (("FILTER" not in v and len(v.split("|||")) > 1) or ("FILTER" in v and len(v.split("|||")) > 2)):
			return

		if len(key.split()) > 1 and len(set(key.split()) - set(self.stop_words)) == 0 and not '#is#M' in v:
			return
		if v.startswith("FILTER"):
			return

		v = list(set(v.split()))
		if len(v) > 1:
			return 
		v.sort()
		v = ' '.join(v)
		v = re.sub(r'\?x[0-9]', '?x', v)
		return (key, v), cnt

	def filter_result(self, src1):

		coor_dict = defaultdict(int)

		## src1是 debug_opt
		for file in [open(src1)]:
			for line in file:
				
				# print("after key:", key)
				result = self.filter_str(line)
				if result:
					coor_dict[result[0]] += result[1]


				# elif len(key_entities) == len(v_entities):
				# 	key = re.sub(r'M[0-9]', 'M', key)
				# 	v = re.sub(r'M[0-9]', 'M', v)
				# 	coor_dict[(key, v)] += cnt
			
					# coor_dict[(key, v)] += cnt


				# coor_dict[(key, v)] += cnt
		sorted_coor_dict = sorted(coor_dict.items(), key=lambda s:s[1], reverse = True)
		return sorted_coor_dict

	def filter_result_pred(self, src1):

		coor_dict = defaultdict(int)

		

		## src1是 debug_opt
		for file in [open(src1)]:
			for line in file:
				
				# print("after key:", key)
				result = self.filter_str(line)
				if result:
					key, v = result[0]
					# v = "?x#ns:people.person.nationality#ns:m.0f8l9c"
					a1, r, a2 = v.split('#')
					
					if re.match(r'\?x[0-9]*|M[0-9]*', a1) and re.match(r'\?x[0-9]*|M[0-9]*', a2) and r!='is':
						# print("1111111")
						v = r[3:] if r.startswith('ns:') else r
					else:
						# print("222222222")
						r = r[3:] if r.startswith('ns:') else r
						a2 = a2[3:] if a2.startswith('ns:') else a2
						a2 = 'm_'+a2[2:] if a2.startswith('m.') else a2

						v = f"{r} {a2}"
					# print(key, v)
					coor_dict[(key, v)] += result[1]


				# elif len(key_entities) == len(v_entities):
				# 	key = re.sub(r'M[0-9]', 'M', key)
				# 	v = re.sub(r'M[0-9]', 'M', v)
				# 	coor_dict[(key, v)] += cnt
			
					# coor_dict[(key, v)] += cnt


				# coor_dict[(key, v)] += cnt
		sorted_coor_dict = sorted(coor_dict.items(), key=lambda s:s[1], reverse = True)
		return sorted_coor_dict

	def term_extract(self, query):

		terms = []
		entities = []
		query = query.split()
		idx = 0

		####三元组
		while idx < len(query):
			
			if re.match(r'M[0-9]', query[idx]):
				entities.append(( query[idx:idx+1],query[idx:idx+1] ,(idx, idx)))
				idx += 1
			# elif idx +3 <= len(query) and ' '.join(query[idx:idx+3]) in self.dict:
			# 	terms.append((' '.join(query[idx:idx+3]), self.dict.get(' '.join(query[idx:idx+3])),(idx, idx+2)))
			# 	idx += 3
			
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
		return entities, terms


		pass

	def term_extract_v2(self, query, type):
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
			# print("terms:", terms)


		query = query.split()
		idx = 0

		####三元组
		while idx < len(query):
			
			if re.match(r'M[0-9]', query[idx]):
				entities.append(( query[idx:idx+1],query[idx:idx+1] ,(idx, idx)))
				idx += 1
			# elif idx +3 <= len(query) and ' '.join(query[idx:idx+3]) in self.dict:
			# 	terms.append((' '.join(query[idx:idx+3]), self.dict.get(' '.join(query[idx:idx+3])),(idx, idx+2)))
			# 	idx += 3
			
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

	def fill_skeleton(self, query, skeleton):
		## fill skeleton 是之前的细粒度版本
		## 就是?x a M, ?x nationality 以及 gender的都做区分
		## v2的版本把他们都做成?x P M

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

		def transform_term_to_pattern(term):
			# print("term here:", term)

			term_split = []
			for i in term.split():
				term_split +=  i.split("|||")

			# print("term split:", term_split)

			skeleton_list = []
			term_list = []
			for i in term_split:
				if i.startswith("FILTER"):
					continue
				# print(i)
				i = preprocess_sparql(i.split("#"))
				a1, r, a2 = i.split()

				if a1.startswith("?x") and a2.startswith("?x"):
					skeleton_list.append(f"{a1} P {a2}")
				elif a1.startswith("?x") and a2.startswith("M"):
					skeleton_list.append(f"{a1} P M")
				elif a2.startswith("?x") and a1.startswith("M"):
					skeleton_list.append(f"M P {a2}")
				elif a1.startswith("M") and a2.startswith("M"):
					skeleton_list.append(f"M P M")
				elif r == "a":
					skeleton_list.append(f"{a1} a M")
				else:
					skeleton_list.append(f"{a1} V S")

				term_list.append(i)

			return skeleton_list, ' . '.join(term_list)


		entities, terms = self.term_extract(query)
		
		# print(f"\nquery:{query}\nskeleton:{skeleton}\nsparql:{sparql}")
		

		candidate_terms = defaultdict(set)
		for term in terms:
			for sub_term in term[1]:
				sub_pattern , sub_term = transform_term_to_pattern(sub_term)
				# print(sub_pattern)
				if " ".join(sub_pattern) in skeleton:
					candidate_terms[" ".join(sub_pattern)].add(sub_term)

		
		candidate_triplets = defaultdict(list)
		# print("candidate_term:", candidate_terms)
		for candidate_skeleton, candidate_terms in candidate_terms.items():
			# a1, r, a2 = candidate_term.split("#")
			for candidate_term in candidate_terms:
				candidate_term = candidate_term.replace("#", " ")
				
				if candidate_term.count("M") == 1:
					
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






		# print(entities)
		# for i in terms:
		# 	print(i)
		# print(terms)

		# print("candidate_terms:", candidate_terms)
		# print("candidate_triplets:", candidate_triplets)

		return candidate_triplets
			
		# print(terms)
		# for term in terms:
		# 	for candidate_term in term:
		# 		if candiidate_term
		# 		for pattern in skeleton:

		# 	pass
	def fill_skeleton_v2(self, query, skeleton):
		## 通过query + 对齐的双语词典align的结果得到候选的candidate triples
		## 候选的cndidate_triples通过给定的skeleton来过滤
		## 就是?x a M, ?x nationality 以及 gender的都做区分
		## v2的版本把他们都做成?x P M

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

		def transform_term_to_pattern(term):


			term_split = []
			for i in term.split():
				term_split +=  i.split("|||")

			# print("term split:", term_split)

			skeleton_list = []
			term_list = []
			for i in term_split:
				if i.startswith("FILTER"):
					continue
				# print(i)
				i = preprocess_sparql(i.split("#"))
				a1, r, a2 = i.split()

				if a1.startswith("?x") and a2.startswith("?x"):
					## ?x P ?x
					skeleton_list.append(f"{a1} P {a2}")
				elif a1.startswith("?x") and a2.startswith("M"):
					## ?x P M
					skeleton_list.append(f"{a1} P M")
				elif a2.startswith("?x") and a1.startswith("M"):
					## M P ?x
					skeleton_list.append(f"M P {a2}")
				elif a1.startswith("M") and a2.startswith("M"):
					## M P M
					skeleton_list.append(f"M P M")
				elif a1.startswith("?x") and r == "a":
					## ?x a M => ?x P M
					skeleton_list.append(f"{a1} P M")
				else:
					## ?x nationality/gender => ?x P M
					skeleton_list.append(f"{a1} P M")

				term_list.append(i)

			return skeleton_list, ' . '.join(term_list)



		entities, terms, Mflag = self.term_extract(query)
		
		# print(f"\nquery:{query}\nskeleton:{skeleton}\nsparql:{sparql}")
		

		candidate_terms = defaultdict(set)
		for term in terms:
			for sub_term in term[1]:
				# print("sub_term:", sub_term)
				sub_pattern , sub_term = transform_term_to_pattern(sub_term)
				# print("sub_pattern:", sub_pattern)
				if " ".join(sub_pattern) in skeleton:
					candidate_terms[" ".join(sub_pattern)].add(sub_term)

		
		candidate_triplets = defaultdict(list)
		# print("candidate_term:", candidate_terms)
		for candidate_skeleton, candidate_terms in candidate_terms.items():
			# a1, r, a2 = candidate_term.split("#")
			for candidate_term in candidate_terms:
				candidate_term = candidate_term.replace("#", " ")
				
				if candidate_term.count("M") == 1:
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

		# print(entities)
		# for i in terms:
		# 	print(i)
		# print(terms)

		# print("candidate_terms:", candidate_terms)
		# print("candidate_triplets:", candidate_triplets)

		return candidate_triplets

	def fill_skeleton_v3(self, query, skeleton, split):

		

		## 通过query + 对齐的双语词典align的结果得到候选的candidate triples
		## 候选的cndidate_triples通过给定的skeleton来过滤
		## 就是?x a M, ?x nationality 以及 gender的都做区分
		## v3的版本是把原始的M P ?x 换成了?x版本 无M开头的sparql
		# Mflag = False
		# for triple in skeleton.split(" . "):
		# 	if triple.strip().startswith("M"):
		# 		Mflag 


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



		entities, terms = self.term_extract_v2(query, split)
		# print("terms:", terms)
		
		# print(f"\nquery:{query}\nskeleton:{skeleton}\nsparql:{sparql}")
		

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

		# print(entities)
		# for i in terms:
		# 	print(i)
		# print(terms)

		# print("candidate_terms:", candidate_terms)
		# print("candidate_triplets:", candidate_triplets)

		return candidate_triplets

	def modify_skeleton(self,sparql):
		sparql = sparql.replace("SELECT count(*) WHERE { ", " ")
		sparql = sparql.replace("SELECT DISTINCT ?x0 WHERE { ", " ")
		sparql_list = sparql.strip().split(" . ")

		skeleton_list = []
		# print(sparql)
		for item in sparql_list:
			if item.startswith("FILTER"):
				continue

			a1, r, a2 = item.strip().split()
			# print(a1, r, a2)

			if a1.startswith("?x") and a2.startswith("?x"):
				skeleton_list.append(f"{a1} P {a2}")
			elif a1.startswith("?x") and a2.startswith("M"):
				skeleton_list.append(f"{a1} P M")
			elif a2.startswith("?x") and a1.startswith("M"):
				skeleton_list.append(f"M P {a2}")
			elif a1.startswith("M") and a2.startswith("M"):
				skeleton_list.append(f"M P M")
			elif a1.startswith("?x") and r == "a":
				skeleton_list.append(f"{a1} a M")
			elif re.match(r'M[0-9]', a1):
				skeleton_list.append(f"M V S")
			else:
				skeleton_list.append(f"{a1} V S")

			skeleton_set = list(set(skeleton_list))
			skeleton_set.sort(key=skeleton_list.index)

		return sparql, " . ".join(skeleton_set)


	def clear_sparql(self, sparql):

		return re.findall(r'[{](.*?)[}]', sparql.replace('\n', ' '))[0]

	def clear_skeleton(self, skeleton):
		# print(skeleton)
		# print(re.findall(r'[{](.*?)[}]', skeleton.strip())[0])
		skeleton = [i for i in re.findall(r'[{](.*?)[}]', skeleton)[0].strip().split(" . ") if not i.startswith("FILTER")]


		for idx, item in enumerate(skeleton):
			
			a1, r, a2 = item.split()
			if re.match(r'M[0-9]', a1):
				a1 = 'M'
			if re.match(r'M[0-9]', a2):
				a2 = 'M'
			if re.match(r'P[0-9]', r):
				r = 'P'
			skeleton[idx] = " ".join([a1, r, a2])

		return " . ".join(skeleton)

	def split_skeleton(self, skeleton, flag):
		# print("skeleton:", skeleton)
		if isinstance(skeleton, list):
			skeleton = " . ".join(skeleton)
		sparql_groups_part= self.split_sub_skeleton(skeleton, flag, 0)
		sparql_groups = []

		for i in sparql_groups_part:
			xidx = 0
			isplit = i.split(" . ")
			for kkidx, kk in enumerate(isplit):
				if kk.startswith('?x'):
					xidx =kkidx 
					##记录当前分割开的sparql是用那个中间变量！
					variable_idx = int(kk[2])
					break
			
			if xidx > 0:						
				sub_sparql_groups_p1 = self.split_sub_skeleton(' . '.join(isplit[xidx:]), False, variable_idx)
				sub_sparql_groups_p2 = self.split_sub_skeleton(' . '.join(isplit[:xidx]), False, variable_idx)
				
				for i in sub_sparql_groups_p2:
					for j in sub_sparql_groups_p1:
						sparql_groups.append(i+' . ' + j)				
			else:
				
				sparql_groups.append(i)

		return sparql_groups


	def split_sub_skeleton(self, sparql, Mflag, count):
	
		# split_lf_results = []
		# split_lf_combine_results = []

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
		# print("sorted triples:", sparql, "\n", sorted_triples)
	
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
							if Mflag and len(split_dict[arg_min]) > 0:
								for j in split_dict[arg_min]:
									j+=cur_list_
							else:
								split_dict[arg_min].append(cur_list_)
							
					else:
						split_dict[arg_max].append([triple])
				## 如果只有一个变量
				## 看能不能为之前添加的做补充
				## 形如(?x, r, M)为之前（M, r, ?x)的做补充
				elif (not Mflag and arg1.startswith('?x') and not arg1.startswith("?x0")) \
					or (Mflag and arg1.startswith('?x')):
					flag = True
					for t in split_dict[arg1]:
					
						if t[0][0] != arg1:
							t.append(triple)
							flag = False
							

					if flag:
						split_dict[arg1].append([triple])
					# print("h:",split_dict)
				##都没有 为该变量的第一个三元组关系
				elif (not Mflag and arg1.startswith("M") and arg2.startswith("?x") and not arg2.startswith("?x0")) \
					or (Mflag and arg1.startswith("M") and arg2.startswith("?x")):
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
					# if SPOUSE_PRED in vv[idx] or SIBLING_PRED in vv[idx]:
					# 	# print("vvidx:", vv[idx])
					# 	a1, r, a2 = vv[idx]
					# 	vv.append(f"FILTER ( {a1} != {a2} )")
					vv[idx] = ' '.join(vv[idx])
					if not xflag and vv[idx].startswith("?x"):
						xidx, xflag = idx, True

				vv = ' . '.join(vv[:xidx] + sorted(list(set(vv[xidx:vv_len]))) + vv[vv_len:])
				# print(vv,vv[2], count, vv.startswith('?x'), vv[2]!=count)
				if not (vv.startswith('?x') and int(vv[2])!=count):
					##去掉不合法的?x
					final_split.append(vv)

				

		return final_split

	def distribute_triples_to_skeleton(self, skeleton_groups, candidate_triplets):
		print("skeleton_groups:", skeleton_groups)
		print("candidate_triplets:", candidate_triplets)
		fn = lambda x, code=',': reduce(lambda x, y: [str(i)+code+str(j) for i in x for j in y], x)
		ans = []

		def replace_variable(pattern, candidates):
			# print("pattern:", pattern)
			# print("candidates:", candidates)
			
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
				# print("11",re.sub(r'\?x[0-9]', '?x', skeleton_group[1]))
				# print("22",candidate_triplets.get(re.sub(r'\?x[0-9]', '?x', skeleton_group[1])))
				triples_groups = [replace_variable(skeleton_item, candidate_triplets.get(re.sub(r'\?x[0-9]', '?x', skeleton_item), [])) for skeleton_item in skeleton_group]
				# print(triples_groups)
				ans += fn(triples_groups, ' . ')
		return ans

	def generate_samples(self, query, sparql, triples, type):
		# triples = triples.split(" . ")
		pos_ans = []
		neg_ans = []

		valid_cnt = len([i for i in sparql.split(" . ") if not i.startswith("FILTER")])

		# print(f"query:{query}\nsparql:{sparql}")
		
		coverage_sparql = set()
		for triple_group in triples:
			flag = True
			for triple in triple_group.split(" . "):
				# print(triple)

				if triple not in sparql.split(" . "):
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

		# coverage_sparql = " . ".join(coverage_sparql).split(" . ")
		coverage = True if  len(coverage_sparql) == valid_cnt else False

		# if not coverage:
		# 	print(coverage_sparql)
		# 	print(sparql)
		# 	print(f"query:{query}\nsparql:{sparql}")
		# 	print(pos_ans, triples)

		if type == "train":
			# print(len(pos_ans), len(neg_ans))
			return coverage, pos_ans + random.sample(neg_ans, min(len(neg_ans), len(pos_ans)))
		else:	
			return coverage, pos_ans + neg_ans


	def generate_samples_v2(self, query, trie_sparql, triples, type):
		##目的是为了统计primitive prediction中 的P/R/F1
		
		pos_ans = []
		neg_ans = []

		# valid_cnt = len([i for i in sparql.split(" . ") if not i.startswith("FILTER")])
		valid_paths = [path.strip() for path in trie_sparql.split("###")]

		# print(f"query:{query}\nsparql:{sparql}")
		# print("triples:", triples)
		
		coverage_sparql = set()

		for triple_group in triples:
			if triple_group in valid_paths:
				pos_ans.append(Sample(query, triple_group, True).__dict__)
				# print(triple_group)
			else:
				# print("triple_group:", triple_group)
				# print("valid_paths:", valid_paths)
				neg_ans.append(Sample(query, triple_group, False).__dict__)

		p = float(len(set(triples) & set(valid_paths))) / float(len(triples))
		r = float(len(set(triples) & set(valid_paths))) / float(len(valid_paths))
		# f = 2 * p*r / (p+r)
		# print(p, r)


		if type == "train":
			# print(len(pos_ans), len(neg_ans))
			return (p, r), pos_ans + random.sample(neg_ans, min(len(neg_ans), len(pos_ans)))
		else:	
			return (p, r), len(pos_ans), len(neg_ans), pos_ans + neg_ans


if __name__ == '__main__':
	helper = Helper()

	#####一些GIZA++的指令
	#参加博客：http://codepothunter.github.io/2016/07/11/How-to-use-GIZA-for-alignment/


	############################################生成.update文件############################################
	# helper.update_output_format("v4-0520")

	############################################生成统计共现文件#############################################

	# qf_file统计结果在debug_opt中
	# fq_file的统计结果在debug_opt2中 
	# qf_file = f"/home/v-yinguo/DKI/GIZA_alignment/v4-0520/query2formula/query2formula.A3.final.update"
	# fq_file = f"/home/v-yinguo/DKI/GIZA_alignment/v4-0520/formula2query/formula2query.A3.final.update"
	# res_file = open(f"/home/v-yinguo/DKI/GIZA_alignment/coor_file-0520","w")
	# sorted_res = helper.statistic_coor(qf_file, fq_file)
	# for k in sorted_res:
	# 	# print("here")
	# 	res_file.write(str(k) + '\n')

	# 	print(k)

	############################################只过滤共献词表3################################################

	# helper = Helper() 
	# src = "/home/v-yinguo/DKI/GIZA_alignment/coor_file-0520"
	# sorted_res = helper.filter_result(src)
	# f = open("./coor_file-0520.filter_wto_no", "w")
	# for k in sorted_res:
	# 	# print("k")
	# 	if k[-1] > 200:
	# 		f.write(str(k) + '\n')

	###########################################只保留词和predicate的对应################################################
	##生成phrase_table.pred文件

	# helper = Helper() 
	# src = "/home/v-yinguo/DKI/GIZA_alignment/coor_file-0520"
	# sorted_res = helper.filter_result_pred(src)
	# f = open("./coor_file-0520.filter.pred", "w")
	# for k in sorted_res:
	# 	# print("k")
	# 	if k[-1] > 30:
	# 		f.write(str(k) + '\n')


	############################################过滤共现词表################################################

	# src1, src2 = "/home/v-yinguo/DKI/GIZA_alignment/debug_opt", "/home/v-yinguo/DKI/GIZA_alignment/debug_opt"
	# sorted_res = helper.filter_result(src1, src2)
	# f = open("./debug_opt4", "w")
	# for k in sorted_res:
	# 	# print("here")
	# 	if k[-1] > 50:
	# 		f.write(str(k) + '\n')
	# f.close()







