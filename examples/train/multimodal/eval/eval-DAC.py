#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Dense Audio Captioning (DAC) predictions with FENSE under IoU thresholds.

The input file should be a JSONL file. Each line contains at least two fields:
- prediction field, e.g. "response"
- reference field, e.g. "labels"

Both fields can be either a JSON list or a stringified JSON list with items like:
[
  {"start": "00:00", "end": "00:10", "caption": "..."},
  {"start": "00:10", "end": "00:20", "caption": "..."}
]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from aac_metrics.classes.fense import FENSE

Segment = Dict[str, Any]


def setup_logger(verbose: bool = False) -> None:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def parse_time_to_seconds(time_value: Any) -> float:
    """Convert HH:MM:SS, MM:SS, or seconds to float seconds."""
    if time_value is None:
        return 0.0

    text = str(time_value).strip()
    if not text:
        return 0.0

    try:
        parts = text.split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def temporal_iou(seg1: Segment, seg2: Segment) -> float:
    """Calculate temporal IoU between two segments."""
    required = {"start", "end"}
    if not required.issubset(seg1) or not required.issubset(seg2):
        return 0.0

    s1, e1 = parse_time_to_seconds(seg1["start"]), parse_time_to_seconds(seg1["end"])
    s2, e2 = parse_time_to_seconds(seg2["start"]), parse_time_to_seconds(seg2["end"])

    if e1 <= s1 or e2 <= s2:
        return 0.0

    intersection = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - intersection
    return intersection / union if union > 0 else 0.0


def _validate_segments(data: Any) -> List[Segment]:
    """Keep valid DAC segments only."""
    if not isinstance(data, list):
        return []

    valid_segments: List[Segment] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if all(key in item for key in ("start", "end", "caption")):
            valid_segments.append(
                {
                    "start": str(item["start"]).strip(),
                    "end": str(item["end"]).strip(),
                    "caption": str(item["caption"]).strip(),
                }
            )
    return valid_segments


def robust_json_load(value: Any) -> List[Segment]:
    """
    Robustly parse a DAC segment list.

    This first tries normal json.loads. If the model output is malformed,
    it falls back to a conservative regex extraction for {start, end, caption} blocks.
    """
    if isinstance(value, list):
        return _validate_segments(value)

    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []

    try:
        return _validate_segments(json.loads(text))
    except json.JSONDecodeError:
        pass

    repaired: List[Segment] = []
    blocks = re.findall(r"\{[^{}]+\}", text, flags=re.DOTALL)
    for block in blocks:
        item: Segment = {}
        success = True
        for key in ("start", "end", "caption"):
            # Captures quoted values and simple unquoted values.
            match = re.search(
                rf'"{key}"\s*:\s*"?([^",}}]+)"?\s*[,}}]',
                block,
                flags=re.DOTALL,
            )
            if not match:
                success = False
                break
            item[key] = match.group(1).strip().replace('"', "”")
        if success:
            repaired.append(item)
    return _validate_segments(repaired)


class DACJsonlEvaluator:
    """Evaluator for DAC JSONL results."""

    def __init__(
        self,
        device: str = "cuda",
        sbert_model: Optional[str] = None,
        thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    ) -> None:
        self.device = device
        self.thresholds = list(thresholds)

        logging.info("Loading FENSE metric on device=%s", device)
        if sbert_model:
            self.fense_metric = FENSE(device=device, sbert_model=sbert_model)
        else:
            self.fense_metric = FENSE(device=device)

    def evaluate_file(
        self,
        input_jsonl: Path,
        pred_field: str = "response",
        ref_field: str = "labels",
    ) -> Dict[str, Any]:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Input file not found: {input_jsonl}")

        global_sums = {str(t): 0.0 for t in self.thresholds}
        total_lines = 0
        valid_samples = 0
        skipped_samples = 0

        with input_jsonl.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc="Evaluating"), start=1):
                total_lines += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    skipped_samples += 1
                    logging.debug("Skip line %d: invalid JSON line", line_idx)
                    continue

                preds = robust_json_load(item.get(pred_field, []))
                refs = robust_json_load(item.get(ref_field, []))

                if not preds or not refs:
                    skipped_samples += 1
                    logging.debug("Skip line %d: empty predictions or references", line_idx)
                    continue

                iou_matrix = np.zeros((len(refs), len(preds)), dtype=np.float32)
                for i, ref in enumerate(refs):
                    for j, pred in enumerate(preds):
                        iou_matrix[i, j] = temporal_iou(ref, pred)

                for threshold in self.thresholds:
                    candidates: List[str] = []
                    references: List[List[str]] = []
                    ref_indices: List[int] = []

                    for i in range(len(refs)):
                        for j in range(len(preds)):
                            if iou_matrix[i, j] >= threshold:
                                candidates.append(preds[j]["caption"])
                                references.append([refs[i]["caption"]])
                                ref_indices.append(i)

                    ref_best_scores = {i: 0.0 for i in range(len(refs))}
                    if candidates:
                        try:
                            _, sent_scores = self.fense_metric(candidates, references)
                            fense_scores = sent_scores["fense"].detach().cpu().tolist()
                            for ref_idx, score in zip(ref_indices, fense_scores):
                                ref_best_scores[ref_idx] = max(ref_best_scores[ref_idx], float(score))
                        except Exception as exc:  # noqa: BLE001
                            logging.warning("FENSE failed at line %d: %s", line_idx, exc)

                    global_sums[str(threshold)] += sum(ref_best_scores.values()) / len(refs)

                valid_samples += 1

        metrics = {
            f"FENSE@IoU={threshold}": (
                global_sums[str(threshold)] / valid_samples if valid_samples > 0 else 0.0
            )
            for threshold in self.thresholds
        }

        return {
            "input_file": str(input_jsonl),
            "total_lines": total_lines,
            "valid_samples": valid_samples,
            "skipped_samples": skipped_samples,
            "metrics": metrics,
        }


def parse_thresholds(value: str) -> List[float]:
    try:
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds should be comma-separated floats, e.g. 0.3,0.5,0.7") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DAC JSONL predictions with FENSE under temporal IoU thresholds."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to the input JSONL result file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to save evaluation results as JSON.",
    )
    parser.add_argument(
        "--pred-field",
        default="response",
        help="Field name for model predictions. Default: response.",
    )
    parser.add_argument(
        "--ref-field",
        default="labels",
        help="Field name for ground-truth references. Default: labels.",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=[0.3, 0.5, 0.7],
        help="Comma-separated temporal IoU thresholds. Default: 0.3,0.5,0.7.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for FENSE, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--sbert-model",
        default=None,
        help="Optional local path or Hugging Face name of the SBERT model used by FENSE.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logger(args.verbose)

    evaluator = DACJsonlEvaluator(
        device=args.device,
        sbert_model=args.sbert_model,
        thresholds=args.thresholds,
    )
    results = evaluator.evaluate_file(
        input_jsonl=args.input,
        pred_field=args.pred_field,
        ref_field=args.ref_field,
    )

    print("\n" + "=" * 50)
    print(f"Input file      : {results['input_file']}")
    print(f"Valid samples   : {results['valid_samples']} / {results['total_lines']}")
    print(f"Skipped samples : {results['skipped_samples']}")
    print("-" * 50)
    for name, value in results["metrics"].items():
        print(f"{name}: {value:.4f}")
    print("=" * 50)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
