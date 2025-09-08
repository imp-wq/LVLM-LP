#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Task 1 pipeline (VizWiz example) with W&B logging.
Steps match the Jupyter notebook:
1) Load first-token logits & labels via repo's read_data()
2) (Optional) Baseline from model's open-ended outputs (if JSON exists)
3) Rule baseline from response text (if available)
4) Specific-token probability baseline (token id = 853 by default)
5) Linear probing with LogisticRegression
"""

import os, json, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import wandb
from utils.func import read_data, softmax
# --- utils ---

def eval_and_log(name, y_true, scores, proba=True, step=None):
    if proba:
        y_pred = (scores >= 0.5).astype(int)
    else:
        y_pred = np.asarray(scores)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, scores) if proba else float("nan")
    except Exception:
        auroc = float("nan")
    print(f"[{name}] acc={acc:.4f}, f1={f1:.4f}, auroc={auroc:.4f}")
    wandb.log({f"{name}/acc": acc, f"{name}/f1": f1, f"{name}/auroc": auroc}, step=step)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="yepluovozz-the-australian-national-university")
    ap.add_argument("--dataset", default="VizWiz")
    ap.add_argument("--model-name", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--prompt", default="oe")
    ap.add_argument("--token-idx", type=int, default=0)
    ap.add_argument("--token-id", type=int, default=853)
    ap.add_argument("--project", default="LVLM-LP-Task1")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    wandb.init(
        project=args.project, name=args.run_name, config=vars(args))

    # 1) load features
    _, x_train, y_train = read_data(args.model_name, args.dataset, "train", args.prompt, args.token_idx, False)
    val_data, x_val, y_val = read_data(args.model_name, args.dataset, "val", args.prompt, args.token_idx, True)

    # 2) baseline: json labeled outputs
    if args.prompt == "oe":
        path = f"./output/{args.model_name}/{args.dataset}_val_oe_labeled.json"
        if os.path.exists(path):
            val_labeled = json.load(open(path))
            pred = [0 if ins["is_answer"] == "no" else 1 for ins in val_labeled]
            eval_and_log("baseline/json", [ins["label"] for ins in val_data], pred, proba=False, step=1)

    # 3) baseline: rule from response
    if "response" in val_data[0]:
        y_true = [ins["label"] for ins in val_data]
        pred = [0 if "unanswerable" in ins["response"].lower() else 1 for ins in val_data]
        eval_and_log("baseline/rule", y_true, pred, proba=False, step=2)

    # 4) baseline: specific token probability
    probs = softmax(x_val)
    score = 1 - probs[:, args.token_id] if args.token_id < probs.shape[1] else 1 - probs[:, -1]
    eval_and_log("baseline/token", y_val, score, proba=True, step=3)

    # 5) linear probe
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    val_score = clf.predict_proba(x_val)[:, 1]
    eval_and_log("probe/logreg", y_val, val_score, proba=True, step=4)

    wandb.finish()

if __name__ == "__main__":
    main()
