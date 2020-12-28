import json
from collections import defaultdict
import re

FILTER_PRED = ["people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses",
"people.person.sibling_s/ns:people.sibling_relationship.sibling|ns:fictional_universe.fictional_character.siblings/ns:fictional_universe.sibling_relationship_of_fictional_characters.siblings"]

split = "mcd1"
dir = f"./data/{split}/"
test_file = f"{dir}/test/test_mask_predict_classification.csv"
test_sparql = open(f"{dir}/test/test_target.txt")
test_query = open(f"{dir}/test/test_encode.txt")
test_mask_file = f"{dir}/test/test_mask_predict_mapping.csv"
test_predict = json.loads(open(f"./output/esim-mask-{split}-predict").readlines()[0])
test_predict_prob = json.loads(open(f"./output/esim-mask-{split}-predict.prob").readlines()[0])

test_data = [i.strip().split('\t') for i in open(test_file).readlines()[1:]]
test_mask_data = [i.strip().split('\t') for i in open(test_mask_file).readlines()[1:]]

token_pred_dict_file = open("./data/phrase_table.pred").readlines()
token_pred_dict = defaultdict(list)
for i in token_pred_dict_file:
	i = eval(i.strip())[0]
	token_pred_dict[i[0]].append(i[1])


assert len(test_data) == len(test_mask_data) == len(test_predict) == len(test_predict_prob),print(len(test_data),len(test_mask_data),len(test_predict))
golden_dict = {i.strip():j.strip() for i, j in zip(test_query, test_sparql)}

pred_dict = defaultdict(list)
pos_error = 0
false_error = 0
right = 0
pred_1 = 0
pos_cnt = 0

false_errors, predicate_set, token_state = defaultdict(list), [], []
cur_query = ''
for idx, info in enumerate(zip(test_data, test_predict, test_predict_prob)):
	item, pred_v, pred_prob = info
	if item[0] != cur_query:
		if len(predicate_set):

			for token, preds in predicate_set.items():
				if token_state[token] == 0 and len(preds):
					max_idx, max_score, max_item = -1, float('-inf'), ''
					for pred in preds:
						if len(false_errors[pred]):
							false_errors[pred] = sorted(false_errors[pred], key = lambda k:k[2], reverse = True)
							if false_errors[pred][0][2] > max_score:
								max_pred = pred
								max_idx, max_item , max_score = false_errors[pred][0]
					if max_idx != -1:
						false_errors[max_pred].pop(0)
						simplified_query = max_item[0].strip()
						original_query = test_mask_data[max_idx][0]
						mask_entities = eval(test_mask_data[max_idx][-1])
						# print("mask mask_entities:", mask_entities)
						assert isinstance(mask_entities, dict)
						for key, v in mask_entities.items():
							if key in max_item[1]:
								for vv in v:
									pred_dict[original_query] += max_item[1].strip().replace(key, vv).split(" . ")

						pred_dict[original_query] += max_item[1].strip().split(" . ")

		predicate_set = {token:token_pred_dict[token] for token in item[0].split()}
		predicate2token = defaultdict(list)
		for k,v in predicate_set.items():
			for vv in v:
				predicate2token[vv].append(k)

		token_state = {token:0 for token in item[0].split()}
		cur_query = item[0]
		false_errors = defaultdict(list)
		# print("="*50)

	if pred_v == 1 and item[-1].strip() == 'True':
		# if item[1].strip().startswith("M"):
		# print("pred False", item)
		if len(item[1].split(" . ")) == 1:
			a1, r, a2 = item[1].split()
			if re.match(r'\?x[0-9]*|M[0-9]*', a1) and re.match(r'\?x[0-9]*|M[0-9]*', a2) and r!='is':
				v= r
			else:
				v = f"{r} {a2}"
			false_errors[v].append((idx, item, pred_prob[0]))

		false_error += 1
		pass
	elif int(pred_v) == 0 and item[-1].strip() == 'False':
		pos_error += 1

		pass

	else:
		right += 1

	if int(pred_v) == 0:

		for triple in item[1].split(" . "):
			a1, r, a2 = triple.split()
			if re.match(r'\?x[0-9]*|M[0-9]*', a1) and re.match(r'\?x[0-9]*|M[0-9]*', a2) and r!='is':
				token_key= r
			else:
				token_key = f"{r} {a2}"

		for token in predicate2token[token_key]:
			token_state[token] = 1
			if 'executive' not in token_key: 
				break

		simplified_query = item[0].strip()
		original_query = test_mask_data[idx][0]
		mask_entities = eval(test_mask_data[idx][-1])
		assert isinstance(mask_entities, dict)
		for key, v in mask_entities.items():
			if key in item[1]:
				for vv in v:
					pred_dict[original_query] += item[1].strip().replace(key, vv).split(" . ")

		pred_dict[original_query] += item[1].strip().split(" . ")

		pred_1 += 1
	if item[-1].strip() == 'True':
		pos_cnt += 1
	

# print(right / len(test_data))
# print(pos_error / (len(test_data) - pos_cnt))
# print(false_error / pos_cnt)
# print(false_error, pos_error)

# print("pred_dict:",len(pred_dict))
# print("golden_dict:",len(golden_dict))

cnt = 0
for item, val in golden_dict.items():

	sub_sparqls = list(set(pred_dict.get(item, [])))
	sub_sparqls.sort()
	filter_sparqls = []
	for sub_sparql in sub_sparqls:
		a1, r, a2 = sub_sparql.split()
		if r in FILTER_PRED:
			filter_sparqls.append(f"FILTER ( {a1} != {a2} )")
	filter_sparqls.sort()

	if " . ".join(sub_sparqls+filter_sparqls) == val:
		cnt += 1
	else:

		if len(set(sub_sparqls + filter_sparqls) - set(val.split(' . '))) == 0 and len(set(val.split(' . ')) - set(sub_sparqls + filter_sparqls)) == 0:
			print(f"\n\nquery:{item}\n\nsparqls-golden:{set(sub_sparqls + filter_sparqls) - set(val.split(' . '))}\n\ngolden-sparqls:{set(val.split(' . ')) - set(sub_sparqls + filter_sparqls)}\n\nsub-sparql:{sub_sparqls + filter_sparqls}\n\ngolden:{val}")
	

print("accuracy:", cnt / len(golden_dict))















