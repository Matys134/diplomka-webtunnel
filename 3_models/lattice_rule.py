#!/usr/bin/env python3
"""The zero-parameter detector: the Tor cell lattice rule.

    A flow is WebTunnel  <=>  at least THRESHOLD of its upstream TLS records satisfy
                              (L - 44) mod 514 == 0   and   L >= 558

That is two integer operations per record. No model, no training, no features, no GPU.

Why this is the right baseline for the thesis (docs/04-v2-audit.md section 4.4):
  * it is DERIVED, not fitted:  44 = 5 (TLS header) + 22 (WS/HTTPT framing)
                                   + 1 (TLS 1.3 inner content type) + 16 (AEAD tag)
                                514 = the Tor cell (tor-spec.txt section 3, link proto v4+)
  * it is falsifiable by anyone with a packet capture;
  * it answers the committee's hardest question -- "your AUC is 1.0, what single feature does
    that?" -- with an arithmetic identity instead of a feature-importance plot;
  * it makes the censor-cost argument in the cascade chapter for free: a censor needs no
    machine learning at all, which is a much stronger claim about WebTunnel's detectability
    than "our XGBoost got 100%".

Reported with Clopper-Pearson intervals and an explicit resolution floor, because a TPR/FPR
quoted without them is exactly what audit finding F-13 objected to.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.contracts import LATTICE_BASE, LATTICE_OFFSET, TOR_CELL_BYTES, on_tor_lattice  # noqa: E402

DEFAULT_THRESHOLD = 0.20


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact binomial interval. k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


class LatticeRule:
    """The whole model. Two integers and a threshold."""

    name = "LatticeRule"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, direction: int = 1,
                 post_handshake: bool = True):
        self.threshold = threshold
        self.direction = direction
        self.post_handshake = post_handshake

    # -- the two-instruction core ----------------------------------------
    @staticmethod
    def is_cell_record(length: int) -> bool:
        return length >= LATTICE_BASE and (length - LATTICE_OFFSET) % TOR_CELL_BYTES == 0

    def score_records(self, records: Sequence[Tuple[float, int, int]], hs_end_idx: int = 0) -> float:
        src = records[hs_end_idx:] if self.post_handshake else records
        lens = [l for (_t, d, l) in src if d == self.direction]
        if not lens:
            return 0.0
        return sum(1 for l in lens if self.is_cell_record(l)) / len(lens)

    def score_flows(self, flows) -> np.ndarray:
        return np.asarray([self.score_records(f["records"] if isinstance(f, dict) else f.records,
                                              f["hs_end_idx"] if isinstance(f, dict) else f.hs_end_idx)
                           for f in flows], dtype=float)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores >= self.threshold).astype(int)

    # -- evaluation -------------------------------------------------------
    def evaluate(self, scores: np.ndarray, y: np.ndarray) -> Dict[str, object]:
        y = np.asarray(y).astype(int)
        pred = self.predict(scores)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tpr_lo, tpr_hi = clopper_pearson(tp, n_pos)
        fpr_lo, fpr_hi = clopper_pearson(fp, n_neg)
        return {
            "model": self.name,
            "rule": f"(L - {LATTICE_OFFSET}) mod {TOR_CELL_BYTES} == 0 and L >= {LATTICE_BASE}",
            "threshold": self.threshold,
            "adversary": "static",
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "fpr_resolution_floor": 1.0 / max(1, n_neg),
            "tp": tp, "fp": fp,
            "tpr": tp / max(1, n_pos), "tpr_ci95": [tpr_lo, tpr_hi],
            "fpr": fp / max(1, n_neg), "fpr_ci95": [fpr_lo, fpr_hi],
            "fpr_upper_bound_95": fpr_hi,
            "roc_auc": float(roc_auc_score(y, scores)) if len(set(y)) > 1 else float("nan"),
            "average_precision": float(average_precision_score(y, scores)) if len(set(y)) > 1 else float("nan"),
        }


def sweep(scores: np.ndarray, y: np.ndarray, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)) -> List[dict]:
    return [LatticeRule(threshold=t).evaluate(scores, y) for t in thresholds]


def main():
    ap = argparse.ArgumentParser(description="Tor cell lattice rule -- zero-parameter detector")
    ap.add_argument("--flows", default="data/processed/flow_records.jsonl")
    ap.add_argument("--dataset", default="data/processed/tabular_dataset.npz",
                    help="used only to restrict the evaluation to the held-out test split")
    ap.add_argument("--split", default="test", choices=["test", "all"])
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--post-handshake", action="store_true", default=True)
    ap.add_argument("--json", default="4_evaluation/lattice_rule_results.json")
    a = ap.parse_args()

    flows = [json.loads(l) for l in open(a.flows, encoding="utf-8") if l.strip()]
    if a.split == "test" and os.path.exists(a.dataset):
        d = np.load(a.dataset, allow_pickle=True)
        if "capture_ids_test" in d.files:
            keep = set(str(c) for c in d["capture_ids_test"])
            flows = [f for f in flows if f["capture_id"] in keep]

    rule = LatticeRule(threshold=a.threshold, post_handshake=a.post_handshake)
    scores = rule.score_flows(flows)
    y = np.asarray([1 if f["label"] == "webtunnel" else 0 for f in flows])

    print("=" * 78)
    print("  Tor cell lattice rule -- no machine learning")
    print(f"  L = {LATTICE_OFFSET} + {TOR_CELL_BYTES}k   ->   "
          f"{LATTICE_BASE}, {LATTICE_BASE+TOR_CELL_BYTES}, {LATTICE_BASE+2*TOR_CELL_BYTES}, ...")
    print(f"  split={a.split}   flows={len(flows)}   positives={int(y.sum())}   "
          f"negatives={int((y==0).sum())}")
    print("=" * 78)

    print("\n  lattice fraction of upstream records, by class:")
    labels = np.asarray([f["label"] for f in flows])
    for c in sorted(set(labels)):
        m = labels == c
        print(f"    {c:<24} mean={scores[m].mean():.4f}  median={np.median(scores[m]):.4f}  n={int(m.sum())}")

    print(f"\n  threshold sweep (Clopper-Pearson 95% intervals, resolution floor "
          f"{1.0/max(1,int((y==0).sum())):.2e}):")
    print(f"  {'thr':>5}{'TPR':>9}{'TPR 95% CI':>22}{'FPR':>10}{'FPR 95% upper':>16}{'ROC-AUC':>10}")
    rows = sweep(scores, y)
    for r in rows:
        print(f"  {r['threshold']:5.2f}{r['tpr']:9.4f}"
              f"   [{r['tpr_ci95'][0]:.4f}, {r['tpr_ci95'][1]:.4f}]"
              f"{r['fpr']:10.5f}{r['fpr_upper_bound_95']:16.2e}{r['roc_auc']:10.4f}")

    chosen = rule.evaluate(scores, y)
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump({"chosen": chosen, "sweep": rows}, f, indent=2)
    print(f"\n  operating point thr={a.threshold}: TPR {chosen['tpr']:.4f} "
          f"[{chosen['tpr_ci95'][0]:.4f}, {chosen['tpr_ci95'][1]:.4f}], "
          f"{chosen['fp']} false positives in {chosen['n_negatives']} negatives "
          f"(95% upper bound {chosen['fpr_upper_bound_95']:.2e})")
    print(f"  wrote {a.json}")


if __name__ == "__main__":
    main()
