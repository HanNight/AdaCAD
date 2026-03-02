import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import sys
import re
import nltk

import numpy as np

from alignscore import AlignScore
scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0', ckpt_path=sys.argv[3], evaluation_mode='nli_sp')

scores = []

# data = json.load(open(sys.argv[1]))
print("-----------------")
print(sys.argv[2])

data = []
with open(sys.argv[2], 'r') as f:
    for line in f:
        proc_line = line.strip()
        if proc_line:
            x = json.loads(proc_line)
            data.append(x)

print(len(data))

factual_data = json.load(open("tofueval/topics/factual_explanations.json"))
topic_type_list = [x["topic_type"] for x in factual_data]

documents = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        proc_line = line.strip()
        if proc_line:
            x = json.loads(proc_line)
            if x["assigned_process"] == 0:
                documents.append(x["article"])

print(len(documents))
assert len(documents) == len(data)

# documents = [p["document"] for p in data]
summaries = []
for p in data:
    ss = []
    # pred = p["pred"]
    if "coiecd" in sys.argv[2]:
        pred = p["coiecd_answer"]
    else:
        pred = p["string"][0]
    for s in pred.strip().split("\n"):
        ss += nltk.sent_tokenize(s)
    summaries.append(ss)
    p["pred_sentence"] = ss
print(len(documents), len(summaries))

documents_full, summaries_full = [], []
for doc, summ in zip(documents, summaries):
    if summ is not None:
        for s in summ:
            documents_full.append(doc)
            summaries_full.append(s)

label = scorer.score(contexts=documents_full, claims=summaries_full)


# combine scores
scores_all = []
for i, summ in enumerate(summaries):
    scores = []
    if summ is not None:
        for s in summ:
            if s is not None:
                scores.append(label.pop(0))
    scores_all.append(scores)
# scores = [np.nanmean(s) for s in scores_all]
# print(np.nanmean(scores)*100, np.nanmean(scores[:150])*100, np.nanmean(scores[150:])*100)

scores = []
scores_binary = []
scores_main = []
scores_main_binary = []
scores_marginal = []
scores_marginal_binary = []
for i, (dat, score) in enumerate(zip(data, scores_all)):
    dat["alignscore"] = score

    if sum(score) == 0:
        scores.append(0)
        if topic_type_list[i] == "main":
            scores_main.append(0)
        else:
            scores_marginal.append(0)
    else:
        scores.append(np.mean(score))
        if topic_type_list[i] == "main":
            scores_main.append(np.mean(score))
        else:
            scores_marginal.append(np.mean(score))

    align_scores = [0 if x < 0.5 else 1 for x in score]
    if sum(align_scores) == 0:
        scores_binary.append(0)
        if topic_type_list[i] == "main":
            scores_main_binary.append(0)
        else:
            scores_marginal_binary.append(0)
    else:
        scores_binary.append(np.mean(align_scores))
        if topic_type_list[i] == "main":
            scores_main_binary.append(np.mean(align_scores))
        else:
            scores_marginal_binary.append(np.mean(align_scores))

print(sys.argv[2])
print("alignscore main: ", np.mean(scores_main))
print("alignscore_binary main: ", np.mean(scores_main_binary))
print("alignscore marginal: ", np.mean(scores_marginal))
print("alignscore_binary marginal: ", np.mean(scores_marginal_binary))
print("alignscore: ", np.mean(scores))
print("alignscore_binary: ", np.mean(scores_binary))



# save
# if len(sys.argv) == 3:
#     print("saving to ", sys.argv[2])
#     with open(sys.argv[2], 'w') as f:
#         json.dump(data, f, indent=4)