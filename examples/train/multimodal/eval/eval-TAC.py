#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import torch
from tqdm import tqdm
from aac_metrics.classes.fense import FENSE


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return text.strip()


class TACEvaluator:
    def __init__(self, device="cuda", sbert_model=None):
        self.device = device
        print(f"Loading FENSE (device={device})...")
        self.fense = FENSE(device=device, sbert_model=sbert_model)

    def evaluate(self, input_file, pred_field="response", ref_field="labels"):
        candidates = []
        references = []

        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Processing"):
                try:
                    item = json.loads(line)
                except:
                    continue

                pred = clean_text(item.get(pred_field, ""))
                ref = clean_text(item.get(ref_field, ""))

                if not pred or not ref:
                    continue

                candidates.append(pred)
                references.append([ref])  # FENSE格式

        if not candidates:
            return 0.0, 0

        _, scores = self.fense(candidates, references)
        fense_score = scores["fense"].mean().item()

        return fense_score, len(candidates)


def main():
    parser = argparse.ArgumentParser(description="Simple TAC FENSE Evaluator")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--pred-field", default="response")
    parser.add_argument("--ref-field", default="labels")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sbert-model", required=True)

    args = parser.parse_args()

    evaluator = TACEvaluator(
        device=args.device,
        sbert_model=args.sbert_model
    )

    score, count = evaluator.evaluate(
        args.input,
        pred_field=args.pred_field,
        ref_field=args.ref_field
    )

    print("\n" + "="*40)
    print(f"Samples: {count}")
    print(f"FENSE: {score:.4f}")
    print("="*40)

    if args.output:
        result = {
            "input_file": args.input,
            "samples": count,
            "fense": score
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
