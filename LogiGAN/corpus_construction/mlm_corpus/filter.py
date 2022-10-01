import json
import re
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start_index', type=int)
parser.add_argument('--end_index', type=int)
parser.add_argument('--indicator_type')
args = parser.parse_args()

with open(f"./{args.indicator_type}.jsonl", "r") as f :
    lines = f.readlines()
start_index = max(args.start_index, 0)
end_index = None if args.end_index == -1 or args.end_index >= len(lines) else args.end_index
indicator_type = args.indicator_type
stopwords = []
with open("../data/Indicators/stopwords.txt", "r") as f:
    for l in f.readlines():
        l = l.strip().lower()
        if f == "": continue
        stopwords.append(l)
    
rexp = re.compile("|".join([r"(\b{}\b)".format(ind) for ind in stopwords]))



with open(f"./filter_{indicator_type}/{indicator_type}_{start_index}_{end_index}.jsonl", "w") as f:
    for l in tqdm(lines[start_index:end_index]):
        try:
            dic = json.loads(l)
            stripped_output = re.sub(rexp, "", dic["output"])
            stripped_output = re.sub(r"\W+", " ", stripped_output).strip().split()
            if len(stripped_output) <= 3 or len(stripped_output) >= 15: continue
            dic["output"] = dic["output"].replace("_", "")
            json.dump(dic, f)
            f.write("\n")
        except: continue
