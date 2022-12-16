#%%
import sys
sys.path.append("..")
from typing import List
from utils.data_types import *

@dataclass
class AlignmentExample:
    question: str
    schema: List[str]
    sql: str
    identify_results: List[str]
    gold_align_lables: List[AlignmentLabel]
    pred_align_labels: List[AlignmentLabel]

    def get_wrong_tokens_count(self):
        count = 0
        for gold_align, pred_align in zip(self.gold_align_lables, self.pred_align_labels):
            if not gold_align == pred_align:
                count += 1
        return count
    
    def to_string(self):
        out_string = ''
        out_string += "Q: {}\n".format(self.question)
        out_string += "\n".join(self.schema) + '\n'
        out_string += "Gold SQL: {}\n".format(self.sql)
        if len(self.identify_results) > 0:
            out_string += "\n".join(self.identify_results) + '\n'
        out_string += "Gold Align: {}\n".format(" ".join([str(x) for x in self.gold_align_lables]))
        out_string += "Pred Align: {}\n".format(" ".join([str(x) for x in self.pred_align_labels]))
        out_string += "Alignment Wrong: {}\n".format(self.get_wrong_tokens_count())
        out_string += '\n'
        return out_string

def parse_align_labels_from_line(align_str: str) -> List[AlignmentLabel]:
    items = align_str.split(' ')
    align_labels = []
    for i, item in enumerate(items):
        ss = item.split('/')
        if item == '/':
            token = Token(index=i, token=item, lemma=item, pieces=item)
        else:
            token = Token(index=i, token=ss[0], lemma=ss[0].lower(), pieces=[ss[0].lower()])
        if len(ss) == 1 or item == '/':
            align_labels.append(AlignmentLabel(token=token, align_type=SQLTokenType.null, align_value=None, confidence=1.0))
            continue

        assert len(ss) == 3, "{}\t{}".format(item, align_str)
        align_type = SQLTokenType.column
        if len(ss[1].split('.')) == 1:
            align_type = SQLTokenType.table
        score = float(ss[2])
        align_labels.append(AlignmentLabel(token=token, align_type=align_type, align_value=ss[1], confidence=score))
    return align_labels

def load_saved_alignment_results(path: str):
    align_examples = []
    with open(path, 'r', encoding='utf-8') as fr:
        example: AlignmentExample = None
        for line in fr:
            line = line[:-1]
            if line.startswith('Q: '):
                if example is not None:
                    if len(example.gold_align_lables) != len(example.pred_align_labels):
                        print(example.question)
                    align_examples.append(example)
                
                example = AlignmentExample(
                    question=line.replace('Q: ', ""),
                    schema=[],
                    identify_results=[],
                    sql=None,
                    gold_align_lables=None,
                    pred_align_labels=None
                )
                continue

            if line.startswith("db id: "):
                example.schema.append(line.strip())

            if line.startswith("T ") or line.startswith("C ") or line.startswith("V ") or line.startswith("Alignment: "):
                example.identify_results.append(line.strip())
            
            if line.startswith("Gold SQL: "):
                example.sql = line.replace("Gold SQL: ", "")
            
            if line.startswith("Gold Align: "):
                example.gold_align_lables = parse_align_labels_from_line(line.replace("Gold Align: ", ""))
            
            if line.startswith("Pred Align: "):
                example.pred_align_labels = parse_align_labels_from_line(line.replace("Pred Align: ", ""))

    if example is not None:
        align_examples.append(example)
    print("load {} alignment examples from {} over".format(len(align_examples), path))
    return align_examples

#%%
path = r'../pt/spider_alignment_old_202012261608/spider_alignment/SpiderAlignmentModel_202012261610/SpiderAlignmentModel.step_51000.acc_0.634dev.threshold0.30.tbl_0.837.col_0.830.val_0.917.results.txt'
align_examples = load_saved_alignment_results(path)
sorted_align_examples = sorted(align_examples, key=lambda x: x.get_wrong_tokens_count(), reverse=True)
saved_path = path.replace(".txt", '.sorted.txt')
saved_align_lines = [x.to_string() for x in sorted_align_examples]
open(saved_path, 'w', encoding='utf-8').writelines(saved_align_lines)
print('sort and save alignment results over')

# %%
