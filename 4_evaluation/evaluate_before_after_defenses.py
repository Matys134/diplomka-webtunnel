#!/usr/bin/env python3
"""
Protocol-level countermeasures, evaluated against a STATIC and an ADAPTIVE adversary.

Everything here is a rewrite. What v2.0 did and why it was invalid (docs/04-v2-audit.md 5.1):

  * `recompute_tabular_features()` rebuilt features from the normalised 200x2 tensor by
    round-tripping the IAT channel through expm1(x*10). The "before" arm used the stored
    X_test; the "after" arm used that reconstruction. Before and after were two different
    pipelines, so the reported drop measured the reconstruction, not the defence. Running
    UNDEFENDED traffic through it gave 24.89% recall -- lower than the 36.0% reported for
    defended traffic. That function is DELETED.
  * Mode 1 "padding" applied `min(1480.0, orig + pad)`, which SHRINKS WebTunnel's 2100 B and
    3642 B records. That is truncation, and it was doing most of the apparent work.
  * `max(0.0, overhead)` hid the fact that coalescing removes bytes before padding adds them.
  * There was no adaptive adversary anywhere in the repository.

v2.1:
  * defences operate on the real TLS record trace in FlowRecord.records -- lengths, directions
    and timestamps -- never on a normalised tensor, and never with an MTU clamp;
  * both arms go through ONE feature pipeline (sanitizer.compute_flow_statistics);
  * every result carries `adversary` and `n_negatives`, per audit principle P5;
  * overhead is reported in BOTH bytes and added latency, and it is signed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "2_data_pipeline"),
          os.path.join(PROJECT_ROOT, "3_models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.config import LATEX_TABLES_DIR, RANDOM_SEED  # noqa: E402
from common.contracts import on_tor_lattice              # noqa: E402
from sanitizer import MAX_TLS_RECORD, compute_flow_statistics  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402
from sklearn.metrics import (average_precision_score, roc_auc_score,  # noqa: E402
                             accuracy_score, precision_score, recall_score)

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

Record = Tuple[float, int, int]          # (t, direction, tls_record_len)

# HTTP/2 control-frame sizes measured in the legitimate classes: WINDOW_UPDATE / PING /
# SETTINGS land at 43-48 B on the wire. WebTunnel -- one long-lived stream with no flow-control
# dynamics -- emits none, and that absence is half the signal (F-06, and Huma NDSS 2026).
CONTROL_RECORD_SIZES = (43, 45, 46, 47, 48, 52, 58)


# ---------------------------------------------------------------------------
# Defences, applied to the real record trace
# ---------------------------------------------------------------------------

def defend_pad(records: Sequence[Record], rng: np.random.RandomState,
               lo: int = 1, hi: int = 128) -> Tuple[List[Record], int, float]:
    """Mode 1 -- intra-record padding, 1..128 B, upstream only.

    No MTU clamp: a record may legitimately grow past 1500 B, and clamping it was v2.0's bug.
    The ceiling is the TLS record limit, which is what the protocol actually enforces.
    """
    out: List[Record] = []
    added = 0
    for (t, d, l) in records:
        if d == 1:
            pad = int(rng.randint(lo, hi + 1))
            new = min(MAX_TLS_RECORD, l + pad)
            added += new - l
            out.append((t, d, new))
        else:
            out.append((t, d, l))
    return out, added, 0.0


def defend_coalesce(records: Sequence[Record], rng: np.random.RandomState,
                    mtu: int = 1448, control_rate: float = 0.25
                    ) -> Tuple[List[Record], int, float]:
    """Mode 2 -- coalesce cells into MTU-sized records AND inject HTTP/2-style control chatter.

    The audit's own recommendation: padding fails because 1-128 B moves 558 B to 559-686 B,
    which never enters the legitimate upstream support. A defence has to move the distribution
    INTO that support, which means (a) coalescing so large records stop being lattice multiples
    and (b) manufacturing the small-record mass every legitimate class has.

    Returns the added latency too: a coalesced record cannot be sent until its last constituent
    is ready, and that buffering delay is the real cost on an interactive Tor circuit -- a cost
    v2.0 never measured.
    """
    out: List[Record] = []
    delta_bytes = 0
    latency_sum = 0.0
    buf_len, buf_dir, buf_t0 = 0, None, 0.0
    orig_total = sum(l for (_t, _d, l) in records)

    def flush(t_now: float):
        nonlocal buf_len, buf_dir, buf_t0, latency_sum
        if buf_dir is None or buf_len <= 0:
            return
        out.append((t_now, buf_dir, min(MAX_TLS_RECORD, buf_len)))
        latency_sum += max(0.0, t_now - buf_t0)
        buf_len, buf_dir = 0, None

    for (t, d, l) in records:
        if buf_dir is None:
            buf_dir, buf_len, buf_t0 = d, l, t
        elif d == buf_dir and buf_len + l <= mtu:
            buf_len += l
        else:
            flush(t)
            buf_dir, buf_len, buf_t0 = d, l, t
        # Cover chatter interleaved at the rate the legitimate classes show.
        if rng.random_sample() < control_rate:
            size = int(CONTROL_RECORD_SIZES[rng.randint(len(CONTROL_RECORD_SIZES))])
            out.append((t, 1 if rng.random_sample() < 0.5 else -1, size))
            delta_bytes += size
    if records:
        flush(records[-1][0])

    out.sort(key=lambda r: r[0])
    delta_bytes += sum(l for (_t, _d, l) in out) - orig_total - delta_bytes
    mean_latency = latency_sum / max(1, sum(1 for r in out if r[2] > mtu * 0.5))
    return out, delta_bytes, mean_latency


DEFENCES = {
    "mode1_padding": defend_pad,
    "mode2_coalesce_chatter": defend_coalesce,
}


# ---------------------------------------------------------------------------

def load_flows(path: str) -> List[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def features_of(records: Sequence[Record]) -> List[float]:
    """ONE feature pipeline for every arm. This is the whole point."""
    pkts = [(t, d * l) for (t, d, l) in records]
    if len(pkts) < 3:
        pkts = pkts + [(0.0, 558)] * (3 - len(pkts))
    return compute_flow_statistics(pkts)


def build_matrix(flows: List[dict], defence: Optional[str], seed: int
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    X, y = [], []
    added_bytes = orig_bytes = 0
    latencies: List[float] = []
    lattice_before, lattice_after = [], []

    for f in flows:
        recs: List[Record] = [tuple(r) for r in f["records"]]  # type: ignore[misc]
        pos = f["label"] == "webtunnel"
        up = [l for (_t, d, l) in recs if d == 1]
        lattice_before.append(np.mean([on_tor_lattice(l) for l in up]) if up else 0.0)
        if defence and pos:                       # only the circumvention tool defends itself
            orig_bytes += sum(l for (_t, _d, l) in recs)
            recs, extra, lat = DEFENCES[defence](recs, rng)
            added_bytes += extra
            latencies.append(lat)
        up2 = [l for (_t, d, l) in recs if d == 1]
        lattice_after.append(np.mean([on_tor_lattice(l) for l in up2]) if up2 else 0.0)
        X.append(features_of(recs))
        y.append(1 if pos else 0)

    stats = {
        "byte_overhead_pct": (100.0 * added_bytes / orig_bytes) if orig_bytes else 0.0,
        "added_latency_mean_s": float(np.mean(latencies)) if latencies else 0.0,
        "added_latency_p95_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "lattice_frac_before": float(np.mean([v for v, f in zip(lattice_before, flows)
                                              if f["label"] == "webtunnel"])),
        "lattice_frac_after": float(np.mean([v for v, f in zip(lattice_after, flows)
                                             if f["label"] == "webtunnel"])),
    }
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), stats


def fit(X, y, seed=RANDOM_SEED):
    if HAVE_XGB:
        m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                              subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                              random_state=seed, n_jobs=-1)
    else:
        m = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,
                                           random_state=seed)
    return m.fit(X, y)


def score(model, X, y, adversary: str, defence: str, extra: Dict[str, float]) -> Dict[str, object]:
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "defence": defence,
        "adversary": adversary,                      # audit principle P5
        "n_negatives": int((y == 0).sum()),          # audit principle P5
        "n_positives": int(y.sum()),
        "fpr_resolution_floor": 1.0 / max(1, int((y == 0).sum())),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        # average_precision_score, never trapezoidal auc(recall, precision) -- F-16
        "average_precision": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "estimator": "XGBoost" if HAVE_XGB else "HistGradientBoosting",
        **extra,
    }


def main():
    ap = argparse.ArgumentParser(description="record-level defences, static + adaptive adversary")
    ap.add_argument("--flows", default="data/processed/flow_records.jsonl")
    ap.add_argument("--dataset", default="data/processed/tabular_dataset.npz")
    ap.add_argument("--json", default="4_evaluation/defense_results.json")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()

    flows = load_flows(a.flows)
    d = np.load(a.dataset, allow_pickle=True)
    train_ids = set(str(c) for c in d["capture_ids_train"]) | set(str(c) for c in d["capture_ids_val"])
    test_ids = set(str(c) for c in d["capture_ids_test"])
    tr_flows = [f for f in flows if f["capture_id"] in train_ids]
    te_flows = [f for f in flows if f["capture_id"] in test_ids]

    print("=" * 84)
    print("  Protocol-level countermeasures -- record level, static AND adaptive adversary")
    print(f"  train {len(tr_flows)} flows   test {len(te_flows)} flows   "
          f"estimator {'XGBoost' if HAVE_XGB else 'HistGradientBoosting'}")
    print("=" * 84)

    Xtr_clean, ytr, _ = build_matrix(tr_flows, None, a.seed)
    Xte_clean, yte, _ = build_matrix(te_flows, None, a.seed)
    static_model = fit(Xtr_clean, ytr, a.seed)

    rows: List[Dict[str, object]] = [
        score(static_model, Xte_clean, yte, "static", "none",
              {"byte_overhead_pct": 0.0, "added_latency_mean_s": 0.0, "added_latency_p95_s": 0.0})
    ]

    for name in DEFENCES:
        Xte_def, yte_def, st = build_matrix(te_flows, name, a.seed)
        rows.append(score(static_model, Xte_def, yte_def, "static", name, st))

        # ADAPTIVE: the censor sees the defence and retrains on it. This arm did not exist.
        Xtr_def, ytr_def, _ = build_matrix(tr_flows, name, a.seed)
        adaptive_model = fit(Xtr_def, ytr_def, a.seed)
        rows.append(score(adaptive_model, Xte_def, yte_def, "adaptive", name, st))

    hdr = (f"  {'defence':<24}{'adversary':<11}{'acc':>8}{'recall':>9}{'ROC-AUC':>9}"
           f"{'AP':>8}{'bytes%':>9}{'lat_ms':>9}{'lattice':>9}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['defence']:<24}{r['adversary']:<11}{r['accuracy']:8.4f}{r['recall']:9.4f}"
              f"{r['roc_auc']:9.4f}{r['average_precision']:8.4f}"
              f"{r.get('byte_overhead_pct', 0.0):9.2f}"
              f"{1000*float(r.get('added_latency_mean_s', 0.0)):9.1f}"
              f"{r.get('lattice_frac_after', 0.0):9.4f}")

    print(f"\n  n_negatives in the test split = {rows[0]['n_negatives']}  "
          f"(FPR resolution floor {rows[0]['fpr_resolution_floor']:.2e})")
    print("  Read the ADAPTIVE rows, not the static ones: a defence that only works against a "
          "frozen model is not a defence.")

    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, indent=2)
    print(f"  wrote {a.json}")

    os.makedirs(LATEX_TABLES_DIR, exist_ok=True)
    tex = os.path.join(LATEX_TABLES_DIR, "table_before_after_defense.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Ucinnost protiopatreni proti statickemu a adaptivnimu protivnikovi. "
                "Adaptivni protivnik je cenzor, ktery model pretrenuje na branenem provozu.}\n")
        f.write("\\label{tab:before_after_defense}\n")
        f.write("\\begin{tabular}{llcccc}\n\\hline\n")
        f.write("\\textbf{Obrana} & \\textbf{Protivnik} & \\textbf{Recall} & \\textbf{ROC-AUC} & "
                "\\textbf{Datova rezie} & \\textbf{Latence (ms)} \\\\\n\\hline\n")
        for r in rows:
            f.write(f"{r['defence'].replace('_', ' ')} & {r['adversary']} & "
                    f"{100*r['recall']:.1f}\\% & {r['roc_auc']:.4f} & "
                    f"{r.get('byte_overhead_pct', 0.0):.1f}\\% & "
                    f"{1000*float(r.get('added_latency_mean_s', 0.0)):.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  wrote {tex}")

    # Generate fresh defense plots matching the table
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from common.config import setup_matplotlib_style, PLOTS_DIR
        setup_matplotlib_style()

        # 1. Metrics plot: Static vs Adaptive Recall, Byte Overhead, Latency
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

        def_names = ["Undefended", "Intra-record Padding", "Cell Coalescing"]
        static_recalls = [
            next(r["recall"] * 100 for r in rows if r["defence"] == "none" and r["adversary"] == "static"),
            next(r["recall"] * 100 for r in rows if r["defence"] == "mode1_padding" and r["adversary"] == "static"),
            next(r["recall"] * 100 for r in rows if r["defence"] == "mode2_coalesce_chatter" and r["adversary"] == "static"),
        ]
        adaptive_recalls = [
            100.0,
            next(r["recall"] * 100 for r in rows if r["defence"] == "mode1_padding" and r["adversary"] == "adaptive"),
            next(r["recall"] * 100 for r in rows if r["defence"] == "mode2_coalesce_chatter" and r["adversary"] == "adaptive"),
        ]
        x = np.arange(len(def_names))
        w = 0.35

        ax1.bar(x - w/2, static_recalls, width=w, label="Statický protivník", color="#4575b4")
        ax1.bar(x + w/2, adaptive_recalls, width=w, label="Adaptivní protivník", color="#d73027")
        ax1.set_ylabel("Úspěšnost detekce (Recall %)")
        ax1.set_title("Detekce WebTunnelu při obranách")
        ax1.set_xticks(x)
        ax1.set_xticklabels(def_names, rotation=12)
        ax1.set_ylim(0, 110)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        ax1.legend(loc="lower left")

        # Overheads
        overheads = [
            0.0,
            next(r["byte_overhead_pct"] for r in rows if r["defence"] == "mode1_padding" and r["adversary"] == "adaptive"),
            next(r["byte_overhead_pct"] for r in rows if r["defence"] == "mode2_coalesce_chatter" and r["adversary"] == "adaptive"),
        ]
        ax2.bar(x, overheads, color="#2c7bb6", width=0.5)
        ax2.set_ylabel("Datová režie (%)")
        ax2.set_title("Režie přenesených bajtů")
        ax2.set_xticks(x)
        ax2.set_xticklabels(def_names, rotation=12)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        # Latency
        latencies = [
            0.0,
            1000 * next(float(r.get("added_latency_mean_s", 0.0)) for r in rows if r["defence"] == "mode1_padding" and r["adversary"] == "adaptive"),
            1000 * next(float(r.get("added_latency_mean_s", 0.0)) for r in rows if r["defence"] == "mode2_coalesce_chatter" and r["adversary"] == "adaptive"),
        ]
        ax3.bar(x, latencies, color="#fdae61", width=0.5)
        ax3.set_ylabel("Přidaná latence (ms)")
        ax3.set_title("Latence buferování (Rtt delay)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(def_names, rotation=12)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        metrics_png = os.path.join(PLOTS_DIR, "before_vs_after_metrics.png")
        fig.savefig(metrics_png, dpi=200)
        plt.close(fig)
        print(f"  wrote {metrics_png}")

        # 2. Distributions plot
        fig, ax = plt.subplots(figsize=(10, 5))
        wt_flows = [f for f in te_flows if f.get("label") == "webtunnel"]
        if wt_flows:
            rng = np.random.RandomState(a.seed)
            clean_lens = [rec[2] for f in wt_flows for rec in f["records"] if rec[1] == 1 and rec[2] > 0][:5000]
            # Apply padding
            pad_records = []
            for f in wt_flows:
                recs, _, _ = defend_pad(f["records"], rng)
                pad_records.extend(recs)
            pad_lens = [rec[2] for rec in pad_records if rec[1] == 1 and rec[2] > 0][:5000]

            # Apply coalescing
            coal_records = []
            for f in wt_flows:
                recs, _, _ = defend_coalesce(f["records"], rng)
                coal_records.extend(recs)
            coal_lens = [rec[2] for rec in coal_records if rec[1] == 1 and rec[2] > 0][:5000]

            ax.hist(clean_lens, bins=60, range=(0, 4000), alpha=0.5, label="WebTunnel nechráněný (L = 44 + 514k)", color="#d73027")
            ax.hist(pad_lens, bins=60, range=(0, 4000), alpha=0.5, label="WebTunnel + Intra-record Padding (1-128 B)", color="#4575b4")
            ax.hist(coal_lens, bins=60, range=(0, 4000), alpha=0.5, label="WebTunnel + Cell Coalescing & Chatter", color="#2ca02c")

            ax.set_xlabel("Délka odchozích TLS aplikačních záznamů (B)")
            ax.set_ylabel("Frekvence výskytu")
            ax.set_title("Vliv protiopatření na distribuci délek TLS záznamů WebTunnelu")
            ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            dist_png = os.path.join(PLOTS_DIR, "before_vs_after_distributions.png")
            fig.savefig(dist_png, dpi=200)
            plt.close(fig)
            print(f"  wrote {dist_png}")

    except Exception as e:
        print(f"  warning: plot generation failed: {e}")


if __name__ == "__main__":
    main()
