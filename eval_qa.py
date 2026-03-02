import json
import os
import re
import string
import argparse

gold_data_path = ""
pred_data_path = ""

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold, a_pred):
  return normalize_answer(a_gold) in normalize_answer(a_pred)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_data_path", type=str, required=True)
    parser.add_argument("--pred_data_path", type=str, required=True)
    args = parser.parse_args()  

    gold_data_path = args.gold_data_path
    pred_data_path = args.pred_data_path

    gold_answers = []
    with open(gold_data_path, "r") as f:
        for line in f:
            proc_line = line.strip()
            if proc_line:
                data = json.loads(proc_line)
                if data["assigned_process"] == 0:
                    gold_answers.append(data["gold_answers"])
                    
    acc_num = 0
    with open(pred_data_path, "r") as f:
        for line in f:
            proc_line = line.strip()
            if proc_line:
                data = json.loads(proc_line)
                pred = data["string"][0].strip().split('\n')[0].strip()
                gold_ans = gold_answers[data["input_index"]]
                if compute_exact(gold_ans, pred):
                    acc_num += 1
    print(acc_num / len(gold_answers))