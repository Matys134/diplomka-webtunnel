# CLAUDE.md — operating context for this repository

You are working on a **Master's thesis project** at the Faculty of Science, University of South
Bohemia (PřF JU). Author: **Bc. Matěj Kouba**. Supervisor: Ing. Petr Břehovský.

**Thesis topic:** *Analýza odolnosti protokolu WebTunnel v prostředí strukturálně podobného
legitimního provozu* — "Resilience analysis of the WebTunnel protocol in an environment of
structurally similar legitimate traffic."

**Working language:** the thesis text is Czech; code, comments and docs in this repo are English.
Talk to Matěj in whichever language he uses.

---

## 1. Read this first: the project failed an audit and is being rebuilt

In **September 2026** the practical part was subjected to an adversarial peer review
(a simulated USENIX/NDSS-level review). It found the headline results to be **invalid** —
not because of the models, but because of how the data was captured.

**Everything in `project/` is the pre-audit `v1` state.** Do not treat its numbers,
its README or its LaTeX tables as trustworthy. They are kept because the thesis will cite
`v1` as the "what went wrong" methodological chapter.

Two documents define the current work:

| File | What it is |
| --- | --- |
| `docs/01-audit-findings.md` | 18 findings, each re-derived from the raw PCAPs. Read in full before touching anything. |
| `docs/02-rebuild-plan.md` | The 16-week refactor plan that is now being executed. |
| `docs/03-evidence.md` | Raw measured numbers behind every claim in the audit. Cite these, don't re-derive from memory. |

Rendered HTML versions of the first two are in `docs/*.html` (open in a browser).

---

## 2. The three findings that matter most

If you remember nothing else from the audit, remember these — they explain almost everything else.

1. **The positive class is ~13 TCP connections, not 1,500 flows.**
   Tor keeps a single long-lived TLS connection to its bridge. The v1 collector started
   `tcpdump`, fired one request through the *already open* tunnel, and stopped. So 500
   consecutive "sessions" are 500 windows carved out of the same socket. Train and test
   share client port `54502` (broadband), `51040` (lte), `34636`/`50860` (lossy).

2. **One feature reaches 100.00 % accuracy, AUC 1.0000.**
   `up_len_p50` is exactly **558 B** at the 1st, 50th *and* 99th percentile of every
   WebTunnel capture, against 40.5–81 B for every negative class. Zero overlap.
   Also: 0 % of WebTunnel captures contain a TLS handshake vs 69–100 % of negatives,
   so `down_len_min > 79` (the 80-byte ChangeCipherSpec record) alone scores 98.6 %.

3. **The countermeasure chapter measures its own bug.**
   Running *undefended* traffic through `recompute_tabular_features()` already drops recall
   to 24.89 %, below the 36.0 % reported "after defence". And retraining the censor on
   defended traffic gives **100 % recall, AUC 1.0000** for both defence modes —
   the defences provide zero protection against an adaptive adversary.

Plus one that changes the thesis's framing: **the WebTunnel client is not using uTLS.**
Its ClientHello is 267 B, 19 ciphers, no ALPN, no GREASE — a stock Go `crypto/tls`
fingerprint, not a Chrome imitation. Because there is no ALPN, nginx cannot negotiate h2,
so WebTunnel here runs HTTP/1.1 while three negative classes run HTTP/2. The README's
"native HTTP/2 framing" claim is wrong.

---

## 3. The one real scientific result to protect

**558 B on the wire = one Tor cell.** The arithmetic:

```
558 = 5 (TLS record header)
    + 536 (plaintext: 514-byte Tor cell + 22 bytes WebSocket/HTTPT framing)
    + 1   (TLS 1.3 inner content type)
    + 16  (AEAD tag)
```

Confirmed by the 2× multiple at 1072 B (= 2 × 536 coalesced). **81.4 %** of all upstream
WebTunnel records are exactly 558 B. A two-instruction rule — *"≥50 % of upstream records
are exactly 558 B"* — detects WebTunnel with no ML at all.

This survives handshake removal, single-flow demultiplexing and undoing segmentation
offload. It is a genuine protocol invariant and it is the thesis's contribution.
**Do not "fix" it away.** The v2 job is to prove it is protocol and not artefact.

---

## 4. Current state and what to do next

The project has been through **two** adversarial audits. `docs/04-v2-audit.md` is the second
one, plus the remediation log for it. The code is now at **v2.1** on branch `v2-remediation`.

All P0 items from the second audit are implemented:
fresh bridge socket per sample, real 5-tuple recording, ALPN parity, no negative payload
ceiling, verified offload disabling, true TCP reassembly, correct TLS 1.3 handshake cutoff,
six falsifiable gates (`checks/test_gates.py` — 15/15), gates wired in as a blocking phase,
record-level defences with an adaptive adversary, empirical LLR aggregation, and the
zero-parameter lattice rule.

**Nothing is validated yet.** The corpus on disk was captured by the v2.0 collector, and all six
gates correctly refuse it. Next actions, in order:

1. `docker compose -f 1_testbed/docker-compose.yml build client` — the Go generator has never
   been compiled; that is the one thing the remediation could not check.
2. `docker compose exec client /usr/local/bin/probe_utls_support.sh` — decide the uTLS question
   BEFORE capturing. If the PT accepts `utls-imitate`, set `WEBTUNNEL_EXTRA_ARGS` in
   docker-compose; if it does not, that is a reportable finding and G1 relaxes to
   post-handshake parity.
3. `venv/bin/python3 1_testbed/capture/collect_scaled_dataset.py --pilot` (2,016 captures).
4. `venv/bin/python3 2_data_pipeline/build_dataset.py && venv/bin/python3 checks/run_gates.py \
   --dataset data/processed/tabular_dataset.npz`
5. Iterate the testbed until all six are green. **Only then** the 80-hour campaign.

Expect two or three pilot iterations. G4 (budget parity) and G3 (early-vs-late drift) are the
likely holdouts.

## 5. Hard rules for this project

These exist because each one maps to a finding. Do not relax them without saying so out loud.

- **A sample is one TCP connection.** Never a capture window, never "everything on eth0".
- **`conn_id` is the only legal grouping key** for train/val/test splits and for
  `StratifiedGroupKFold`. Never group by sample index.
- **The collector writes ground truth; the parser never infers it.** Client identity,
  target 5-tuple, class, profile, seed and git commit come from the sidecar manifest.
  A flow whose 5-tuple does not match its manifest is discarded, not analysed.
- **No dataset reaches a model before the gates pass.** A failing gate blocks the build.
- **Strong features are justified, not deleted.** Any single feature above 0.90 stump AUC
  must be registered in `project/checks/expected_invariants.py` with a protocol-level
  derivation, or the testbed changes.
- **Every experiment declares `adversary ∈ {static, adaptive}` and `n_negatives`.**
  Defence results without an adaptive-adversary arm are not reportable.
- **Never plot below the resolution floor.** With *n* negatives the smallest measurable
  FPR is 1/*n*. Report Clopper–Pearson intervals; label projections as projections.
- **PR-AUC via `average_precision_score`**, never trapezoidal `auc(recall, precision)`.

---

## 6. Scope discipline (agreed with the author)

Target is a **solid, safely defensible thesis** — not a publication. Explicitly out of scope:

- Pre-training a Transformer (correctly ruled out in the roadmap already)
- Inventing a novel defence — evaluate known ones honestly instead
- Chasing FPR = 10⁻⁵ or 10⁻⁶ by measurement (600+ h of capture); measure 10⁻⁴, project the rest
- Adding a fourth ML model — add the trivial 558 B rule instead
- Rewriting in a new language/framework; the only new non-Python code is the Go traffic client

---

## 7. Environment notes

- **Hardware:** the author's home machine — AMD Ryzen 9800X3D, NVIDIA RTX 5070 Ti (CUDA),
  Docker + Docker Compose. Capture and training both run there.
- **`project/requirements.txt` in v1 pins versions that do not exist**
  (`xgboost>=3.4.1`, `scikit-learn>=1.9.0`, …), so `pip install -r` fails.
  Use `requirements-working.txt` at the repo root, then re-freeze once the env is built.
- **TLS keys are not in this repo.** `project/1_testbed/webtunnel_server/certs/` ships
  only the certificates; run `make_certs.sh` in that directory to regenerate the CA and
  server key before `docker compose up`.
- **PCAPs are not in this repo** (4.1 GB, 9,000 files). They live on the author's machine at
  `…/diplomka/webtunnel_pcaps_9000/raw_pcap/`. The processed feature matrices
  (`project/data/processed/*.npz`) *are* included, so the gates and the defence re-check
  run without them.
- The audit scripts in `audit/` take a `--pcap-dir` argument; point it at the raw PCAPs
  to reproduce any number in `docs/03-evidence.md`.

---

## 8. Key literature (already verified as real and correctly cited)

- **Wails, Sullivan, Sherr & Jansen**, *On Precisely Detecting Censorship Circumvention in
  Real-World Networks*, NDSS 2024. 60 M flows / 600 k destinations. Cite as **Wails et al.**,
  not "Jansen et al." The host-based aggregation idea comes from here.
- **Kamali & Barradas**, *Huma: Censorship Circumvention via Web Protocol Tunneling with
  Deferred Traffic Replacement*, NDSS 2026. Independently names the same WebTunnel weakness:
  it does not preserve normal traffic patterns with the overt site and lacks traffic shaping.
- **Zhou et al.**, *TrafficFormer*, IEEE S&P 2025. No public pretrained checkpoint exists —
  the roadmap's decision to build a lightweight sequence Transformer instead is correct.
- **Frolov & Wustrow**, *HTTPT: A Probe-Resistant Proxy*, FOCI 2020. WebTunnel's ancestor.
- **Sirinam et al.**, *Deep Fingerprinting*, CCS 2018 — the 1D-CNN lineage.
- **Lotfollahi et al.**, *Deep Packet*, Soft Computing 2020.
- **Axelsson**, *The Base-Rate Fallacy and the Difficulty of Intrusion Detection*, TISSEC 2000.

---

## 9. Repository layout

```
├── CLAUDE.md                 ← you are here
├── README.md                 ← human-facing orientation and quickstart
├── requirements.txt          ← pinned to versions that actually resolve
├── docs/
│   ├── 01-audit-findings.md  ← first audit, F-01..F-18
│   ├── 02-rebuild-plan.md    ← the 16-week plan
│   ├── 03-evidence.md        ← raw measured numbers
│   ├── 04-v2-audit.md        ← SECOND audit + the v2.1 remediation log
│   └── *.html                ← rendered reports
├── audit/                    ← re-runnable verification scripts (need the PCAPs)
├── common/                   ← config.py, contracts.py (CaptureManifest, FlowRecord, lattice)
├── checks/                   ← the six gates + test_gates.py (gate self-test)
├── 1_testbed/                ← docker-compose, Go generator, tor lifecycle scripts, netem
├── 2_data_pipeline/          ← sanitizer.py (reassembly), build_dataset.py
├── 3_models/                 ← XGBoost, 1D-CNN, Transformer, lattice_rule.py
├── 4_evaluation/             ← defences, base rate, DET, cascade, XAI, LaTeX export
├── 0_thesis_text/tables/     ← generated LaTeX
└── reference/                ← assignment protocol, roadmaps
```

Note: there is no `project/` directory any more — the layout was flattened. Anything in the
older docs that refers to `project/...` means the repository root.
