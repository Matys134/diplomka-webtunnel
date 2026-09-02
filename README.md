# WebTunnel detectability — practical part

Master's thesis, Faculty of Science, University of South Bohemia.
**Bc. Matěj Kouba** · supervisor Ing. Petr Břehovský.

> *Analýza odolnosti protokolu WebTunnel v prostředí strukturálně podobného legitimního provozu*

---

## Status: v2.2 — Pilot Campaign Completed, Audited, and Cleared for Thesis Text

This repository contains the complete experimental framework, testbed, data pipeline, and evaluation suite for the Master's thesis.

| Document | Description |
| --- | --- |
| `docs/01-audit-findings.md` | First audit — 18 findings (F-01 … F-18) |
| `docs/02-rebuild-plan.md` | The 16-week refactor plan and methodology |
| `docs/03-evidence.md` | Raw forensic measurements from initial audit |
| `docs/04-v2-audit.md` | Second audit — evaluation of v2.0 rebuild |
| `docs/05-final-review.md` | Pre-pilot verification of testbed fixes |
| `docs/06-pilot-v22-review.md` | v2.2 pilot campaign audit (2,016 captures → 1,873 flows, 310 positive sockets) |
| `docs/07-signoff.md` | Sign-off verification of models, cross-validation, and LaTeX tables |
| `docs/08-order-shuffle-audit.md` | Order-shuffle control audit and the 4-separator leakage taxonomy (§5.3) |

**Current State (v2.2 Corpus):**
- **Provenance & Splits:** 1,873 valid flows across 310 independent positive sockets. G5 PASS, G6 PASS (0 5-tuple mismatches, 100% authoritative accounting, 0 cross-split duplicates).
- **Core Finding (Tor Cell Lattice):** WebTunnel upstream records sit on $L = 44 + 514k$ in 92.65% of cases vs. ≤0.19% for legitimate traffic. A zero-parameter deterministic rule achieves TPR 1.0000 with 0 FP across 233 test negatives and 1,563 CV negatives.
- **Leakage Taxonomy (§5.3):** Clear deconstruction into S1 (Tor cell protocol invariant), S2 (ClientHello configuration artifact), S3 (WebSocket Upgrade transport property), and S4 (session volume).
- **Countermeasures (§5.4):** Evaluated against static and adaptive adversary with measured 120.8 ms buffering latency cost.
- **Base-Rate Fallacy (§5.5):** Analytically presented with the 4.29×10⁻³ resolution floor.

---

## The one result the thesis rests on

WebTunnel's TLS record lengths lie on an exact arithmetic lattice:

```
L = 44 + 514·k        44 = 5 (TLS record header)
                         + 22 (WebSocket / HTTPT framing)
                         + 1  (TLS 1.3 inner content type)
                         + 16 (AEAD tag)
                     514 = one Tor cell (tor-spec.txt §3, link protocol v4+)

k = 1     2     3     4     5     6     7
  558  1072  1586  2100  2614  3128  3642      ← all seven observed on the wire
```

74.1 % of WebTunnel's upstream records sit on this lattice; every legitimate class is at
0.0–0.3 %. A detector with **no machine learning at all** —

```python
def is_webtunnel(upstream_record_lengths, threshold=0.20):
    on = sum(1 for L in upstream_record_lengths if L >= 558 and (L - 44) % 514 == 0)
    return on / len(upstream_record_lengths) >= threshold
```

— is implemented in `3_models/lattice_rule.py` and reported with Clopper–Pearson intervals
and an explicit FPR resolution floor. It is the reference method the assignment asks for, and
it is a stronger claim about WebTunnel's detectability than any classifier score.

---

## Repository layout

```
├── CLAUDE.md                  operating context
├── docs/                      the two audits, the plan, the evidence
├── 1_testbed/
│   ├── docker-compose.yml     4 containers on one bridge network
│   ├── client/
│   │   ├── generator/main.go  ONE Go binary for every class: uTLS HelloChrome_Auto,
│   │   │                      one ALPN list, budget-driven sessions, 3 behaviours
│   │   ├── stop_tor.sh        full daemon teardown  ← fresh bridge socket per sample
│   │   ├── start_tor.sh       bootstrap + `ss` snapshot of the REAL bridge 5-tuple
│   │   ├── offload_off.sh     ethtool + verification (exits non-zero if offload is still on)
│   │   └── probe_utls_support.sh   does this webtunnel build accept a uTLS argument?
│   ├── capture/collect_scaled_dataset.py   the collector; writes ground truth
│   └── router/netem_profiles.sh            netem on ingress (ifb) AND egress, with a rate ceiling
├── 2_data_pipeline/
│   ├── sanitizer.py           TCP stream reassembly → TLS records → FlowRecord (no MTU clamp)
│   ├── build_dataset.py       socket-disjoint splits, per-class attrition accounting
│   └── repair_legacy_manifests.py   DIAGNOSTIC ONLY, see the file header
├── common/contracts.py        CaptureManifest / FlowRecord / the lattice / split assertions
├── checks/                    the six build gates + test_gates.py (self-test, 15/15)
├── 3_models/                  XGBoost · 1D-CNN · Flow-Transformer · lattice_rule.py
├── 4_evaluation/              defences (static + adaptive), empirical LLR, DET, cascade, XAI
└── run_full_benchmark.py      orchestrator — the gate phase is BLOCKING
```

---

## Quickstart

```bash
python3 -m venv venv && venv/bin/pip install -U pip && venv/bin/pip install -r requirements.txt

# 0. regenerate the CA and server key (not in the repo)
cd 1_testbed/webtunnel_server/certs && ./make_certs.sh && cd -

# 1. bring the testbed up
docker compose -f 1_testbed/docker-compose.yml build client
docker compose -f 1_testbed/docker-compose.yml up -d

# 2. does this webtunnel build accept a uTLS imitation argument?  (decides G1's fate)
docker compose -f 1_testbed/docker-compose.yml exec client /usr/local/bin/probe_utls_support.sh

# 3. 2,016-capture pilot — one fresh TCP connection per sample
venv/bin/python3 1_testbed/capture/collect_scaled_dataset.py --pilot

# 4. build + gates.  A failing gate BLOCKS the build.
venv/bin/python3 2_data_pipeline/build_dataset.py
venv/bin/python3 checks/run_gates.py --dataset data/processed/tabular_dataset.npz

# 5. only when all six are green:
venv/bin/python3 run_full_benchmark.py --skip-capture
```

`checks/test_gates.py` proves each gate discriminates in both directions — it passes on clean
fixtures and fails on a synthetic reproduction of the exact historical defect. Run it any time
you change a gate:

```bash
venv/bin/python3 checks/test_gates.py     # 15/15
```

---

## Hard rules

These exist because each maps to an audit finding. Do not relax one without saying so out loud.

- **A sample is one TCP connection**, opened fresh, with its SYN inside the capture window.
- **The collector writes ground truth; the parser never infers it.** A capture whose 5-tuple is
  unknown or ambiguous is dropped with a reason, not guessed at.
- **`socket_id` is the grouping key** for splits and cross-validation — not `sample_id`, and not
  `conn_id` alone (which mixes in the SYN timestamp and is therefore unique per capture).
- **No dataset reaches a model before the gates pass.**
- **Strong features are justified, not deleted** — with arithmetic, in `checks/expected_invariants.py`.
- **Every experiment declares `adversary ∈ {static, adaptive}` and `n_negatives`.**
- **Never plot below 1/n_negatives.** Clopper–Pearson everywhere; label projections as projections.
- **PR-AUC via `average_precision_score`**, never trapezoidal `auc(recall, precision)`.
