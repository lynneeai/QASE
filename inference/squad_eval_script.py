import argparse
import json
from evaluate import load


parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", required=True)
parser.add_argument("--gold_file", default="../datasets/SQuAD/test_readable.json")
args = parser.parse_args()

# load pred and gold
pred_objs = json.load(open(args.pred_file, "r"))
gold_objs = json.load(open(args.gold_file, "r"))
gold_objs = [
    {"id": item["id"], "answers": item["answers"]}
    for item in gold_objs
]

# evaluate
print("Evaluating inference...")
squad_metric = load("squad")
results = squad_metric.compute(predictions=pred_objs, references=gold_objs)
print(results)

print("Done!")