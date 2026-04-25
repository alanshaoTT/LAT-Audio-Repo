#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Temporal Audio Grounding (TAG) predictions.

This script computes:
- mean IoU (mIoU)
- Recall@IoU thresholds

Input Format:
-------------
Each line in JSONL must contain:
- prediction field (e.g. "response")
- reference field (e.g. "labels")

Format example:
{
  "response": "[00:10-00:20]",
  "labels": "[00:12-00:22]"
}
"""

import argparse
import json
import re
import numpy as np


# ================= 工具函数 =================

def parse_time_span(text):
    """解析 [MM:SS-MM:SS] 格式"""
    if not isinstance(text, str):
        return None

    text = text.replace("[", "").replace("]", "").strip()

    match = re.search(r'(\d+):(\d+)\s*-\s*(\d+):(\d+)', text)
    if not match:
        return None

    m1, s1, m2, s2 = map(int, match.groups())
    return m1 * 60 + s1, m2 * 60 + s2


def compute_iou(pred, gt):
    if not pred or not gt:
        return 0.0

    s1, e1 = pred
    s2, e2 = gt

    if s1 >= e1 or s2 >= e2:
        return 0.0

    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter

    return inter / union if union > 0 else 0.0


# ================= Evaluator =================

class TAGEvaluator:
    def __init__(self, thresholds=(0.3, 0.5, 0.7)):
        self.thresholds = thresholds

    def evaluate(self, input_file, pred_field="response", ref_field="labels"):
        iou_list = []
        total = 0
        valid = 0

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                try:
                    item = json.loads(line)
                except:
                    continue

                pred_span = parse_time_span(item.get(pred_field, ""))
                gt_span = parse_time_span(item.get(ref_field, ""))

                if not pred_span or not gt_span:
                    continue

                iou = compute_iou(pred_span, gt_span)
                iou_list.append(iou)
                valid += 1

        if not iou_list:
            return {
                "total": total,
                "valid": 0,
                "mIoU": 0.0,
                "recall": {}
            }

        iou_arr = np.array(iou_list)

        results = {
            "total": total,
            "valid": valid,
            "mIoU": float(iou_arr.mean()),
            "recall": {}
        }

        for t in self.thresholds:
            results["recall"][f"R@{t}"] = float(np.mean(iou_arr >= t))

        return results


# ================= CLI =================

def parse_thresholds(s):
    return [float(x) for x in s.split(",") if x]


def main():
    parser = argparse.ArgumentParser(description="TAG Evaluator (mIoU + Recall)")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--pred-field", default="response")
    parser.add_argument("--ref-field", default="labels")
    parser.add_argument("--thresholds", default="0.3,0.5,0.7")

    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)

    evaluator = TAGEvaluator(thresholds=thresholds)

    results = evaluator.evaluate(
        args.input,
        pred_field=args.pred_field,
        ref_field=args.ref_field
    )

    print("\n" + "="*40)
    print(f"Total samples : {results['total']}")
    print(f"Valid samples : {results['valid']}")
    print(f"mIoU          : {results['mIoU']:.4f}")
    for k, v in results["recall"].items():
        print(f"{k:<13}: {v:.4f}")
    print("="*40)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
