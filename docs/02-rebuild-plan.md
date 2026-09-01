# Rebuild plan — 16 weeks, three gates

**Constraints agreed with the author:** 4+ months to defence · full testbed access, overnight
runs possible · target is a **solid, safely defensible thesis**, not a publication.

**Verdict: refactor, don't restart.** The costly assets — the master orchestrator, the model
implementations, the evaluation and XAI modules, the LaTeX/figure export, the Docker topology —
are sound and would take two months to rebuild for no gain. Roughly 3,000 of ~5,400 lines survive
untouched.

But it isn't a patch either. **The unit of analysis is wrong.** Today a sample is "every packet on
`eth0` during a five-second window". It has to become "one TCP connection, with known identity".
That change propagates into split logic, CV groups, host aggregation, defence simulation and the
cascade — so layers 1–2 get rewritten outright and layers 3–4 get surgical edits behind a new,
stable data contract.

Roughly **55 % keep, 30 % rewrite, 15 % delete.** The long pole is not code — it is the capture
campaign, which is mostly unattended and must not start until the harness can fail it.

---

## 1. The single decision everything follows from

`sanitizer.py` currently *infers* what it is looking at: which host is the client (from the first
packet), which packets belong to the flow (all of them), where the handshake ends (first `0x17`
byte). Every one of those inferences is wrong somewhere in the corpus, and the pipeline has no way
to notice.

The collector, by contrast, **knows** everything — class, server, port, netem profile, behaviour
script, generator seed, git commit. It just never writes it down. So write it down, one sidecar
JSON per capture, and the parser stops guessing.

See `project/common/contracts.py` for the implementation. Two rules make it self-enforcing:

- **A flow whose 5-tuple does not match `target_5tuple` is discarded, not analysed.** That alone
  kills cross-class contamination, mDNS noise and the direction-inversion bug in one line.
- **`conn_id` is the only legal grouping key.** Splits, `StratifiedGroupKFold` and the host
  aggregator all take it, so the F-11 misalignment becomes structurally impossible.

---

## 2. Keep · rewrite · delete

### Keep as-is
- `run_full_benchmark.py` — the orchestrator is good; only the phase list changes
- `architectures.py` — plus a key-padding mask on the Transformer
- `train_xgboost.py`, `train_1d_cnn.py`, `train_transformer.py`
- `explain_models.py` — reframed from "interpretation" to **leakage diagnosis**
- `export_latex_tables.py`, `common/config.py` styling, the plots module
- Docker Compose topology, nginx reverse proxy, decoy site, cert CA
- The netem *parameters* (not how they are applied)

### Rewrite
- `collect_scaled_dataset.py` → connection-per-sample, sidecar manifests, interleaved order
- `traffic_generator.py` → one Go client, one TLS stack, budget-driven sessions
- `sanitizer.py` → 5-tuple demux, TCP desegmentation, TLS-record extraction
- `build_dataset.py` → splits keyed on `conn_id`, four split axes
- `cross_validate.py` → fix group alignment, inner-fold model selection
- `evaluate_before_after_defenses.py` → record-level defences + adaptive adversary
- `netem_profiles.sh` → ingress via `ifb` + rate ceiling

### Delete
- The `fpr**M` "Bayesian aggregation" curve — replaced by a real LLR sum
- The `0x17` cutoff heuristic — replaced by `hs_end_idx` from the flow builder
- `recompute_tabular_features()` — an artefact generator with no legitimate use
- The DET curve's `np.maximum(fpr, 1e-4)` floor — plot to 1/N and stop
- Every accuracy-first headline in the README

The rewritten modules are ~1,200 lines of Python plus ~350 lines of Go: two to three weeks of
coding. The schedule is dominated by capture time, not typing.

---

## 3. Six design principles for v2

**P1 · Parity by construction, not by correction.**
One Go binary generates *all* classes — HTTP/1.1 WebSocket, HTTP/2 fetch, HTTP/3 QUIC, and
SOCKS5-through-Tor — using `refraction-networking/utls` with a single pinned `HelloChrome_Auto`
profile. Configure the WebTunnel client with the same imitation. ClientHellos are then
byte-identical across classes by construction, and gate G1 asserts it. This replaces the
roadmap's Mode A / Mode B choice with "both" and is the highest-leverage change in the plan:
~350 lines of Go that permanently removes the TLS-stack criticism.

**P2 · Session budgets drawn from a shared distribution.**
Before each capture, draw a triple — target duration, target upstream bytes, target downstream
bytes — from one common distribution and pass it into whichever generator runs. Every class must
hit the same budget. This makes `total_pkts`, `total_bytes`, `duration` and `burst_count`
non-informative *by design*, the only clean way to kill the volumetric leak that currently gives
100 % accuracy from four features. It also forces the negatives to be genuinely hard: a WebSocket
session that must move 80 KB in 4 s looks structurally much more like a tunnel than one that
sends six chat messages.

**P3 · The dataset is guilty until proven innocent.**
Write the validation harness *before* the testbed work and wire it into the build so it can fail.
A dataset that does not pass the gates never reaches a model. Reported deliberately in the thesis
this becomes a methodological contribution rather than an embarrassment.

**P4 · Strong features are not banned — they are justified.**
The tripwire must not push you into engineering away the 558 B invariant; that invariant is the
real finding. The gate is: any single feature above ~0.90 stump AUC must appear in
`project/checks/expected_invariants.py` with a written protocol-level derivation and a citation.
`up_len_p50 = 558 = 5 + (514 + 22) + 1 + 16` passes. `down_len_min = 80` (a ChangeCipherSpec
record) does not, and must be eliminated. The registry becomes a table in the thesis.

**P5 · Every experiment declares its adversary and its resolution floor.**
Two fields in every results JSON: `adversary ∈ {static, adaptive}` and `n_negatives`. The defence
chapter reports both adversaries side by side. No plot may extend below 1/`n_negatives`, and no
FPR is reported without a Clopper–Pearson interval.

**P6 · Determinism in the lab, realism in a validation set.**
Today the bridge relays into the **live** Tor network and three of the five WebTunnel targets are
public sites (`check.torproject.org`, Wikipedia, DuckDuckGo — the other two resolve through an
exit node and silently fail), while the negatives hit a local mock server. Content, RTT and
reproducibility therefore differ by class, in the positive class's favour. Fix with a private Tor
network (`chutney`: authority + 3 relays + your bridge, all in Compose) for the controlled corpus,
plus a smaller live-network set — same generator, same budgets — purely to show the findings
transfer.

---

## 4. The capture campaign

### Size it from the claim, not from a round number

Rule of three: zero false positives in *n* negatives gives a 95 % upper bound of 3/*n*.

| Claim | Clean negative flows | Capture time @ ~500/h | Verdict |
| --- | ---: | ---: | --- |
| FPR ≤ 10⁻³ | 3,000 | 6 h | Trivial — you already have this scale |
| FPR ≤ 10⁻⁴ | 30,000 | 60 h | **The right target.** 5–7 overnight runs |
| FPR ≤ 7.5 × 10⁻⁵ | 40,000 | 80 h | Comfortable margin; lets you say "below 10⁻⁴" |
| FPR ≤ 10⁻⁵ | 300,000 | 600 h | Out of reach — report as analytical projection |

**Target: 40,000 negative flows and 8,000 positive flows.** That buys a measured "FPR below 10⁻⁴
at 95 % confidence" and leaves 10⁻⁵/10⁻⁶ as clearly labelled projections. Run four to six client
containers in parallel against separate destinations; the bottleneck is `docker compose exec`
round-trips, not the network.

### The design matrix

Randomise order across the whole matrix — never block by class again — and write one row of
`manifest.jsonl` per capture.

| Factor | Levels | Values | Why it must vary |
| --- | ---: | --- | --- |
| Class | 7 | webtunnel + 5 TCP negatives + QUIC/HTTP-3 | QUIC closes the assignment gap (F-18) |
| Behaviour | 3 | interactive browse / bulk / interactive shell | Applied to *every* class, positive included |
| Network profile | 3 | broadband / LTE / lossy, ingress + egress + rate | Domain-shift axis; roadmap's own promise |
| Destination | 8 + 3 | 8 legit vhosts (own certs), 3 bridge instances | Makes destination-split possible (§5.2) |
| Session budget | cont. | duration × bytes from a shared distribution | Removes the volumetric leak by construction |
| Capture epoch | 2 | campaign A (weeks 6–7), campaign B (week 8) | Gives a genuine temporal split for free |

> **Run a 2,000-flow pilot first, and be willing to throw it away.** A pilot costs four hours and
> is the difference between discovering a design flaw in week 5 and discovering it in month 8.
> Run the full gate suite on it. If a gate fails, fix the testbed and re-pilot. Do not start the
> 80-hour campaign until every gate is green on pilot data.

---

## 5. The build gates

`project/checks/` is the first thing you write and the last thing you'd think to write.

| Gate | Assertion | Closes |
| --- | --- | --- |
| **G1 · Stack parity** | ClientHello record length and extension list identical across classes; JA4 hash constant. | F-02, F-03 |
| **G2 · Leakage tripwire** | No single-feature depth-1 stump above 0.90 AUC unless registered in `EXPECTED_INVARIANTS` with a protocol derivation. | F-09 |
| **G3 · Null controls** | Label-shuffle AUC ∈ [0.45, 0.55]; same-generator/different-label AUC ∈ [0.45, 0.55]. | F-09 |
| **G4 · Budget parity** | Two-sample KS on duration, `total_bytes`, `total_pkts` between positive and each negative class: *p* > 0.01. | F-06 |
| **G5 · Split integrity** | No `conn_id` in two splits or two CV folds; group-vector alignment asserted against the feature matrix. | F-01, F-11 |
| **G6 · Provenance** | Every flow's 5-tuple matches its manifest; every capture has a manifest; drop reasons logged per class. | F-04, F-08 |

**How G2 should feel in practice.** On the first pilot it will fire, and it should — that is the
harness working. The response is not to delete the feature but to ask why it separates. If the
answer is a protocol invariant with an arithmetic derivation, register it and move on; that is
your result. If the answer is "because my negatives are 5-second scripts and my positive is a
saturated tunnel", the testbed has to change. The gate's job is to force that question every build.

---

## 6. Schedule

Writing starts in **week 5**, not week 13. The theoretical chapters depend on none of this and
are the cheapest insurance available.

### W1 — Freeze v1 and build the harness (~30 h)
Tag the repository as `v1-audit` — you will cite it in the thesis, so it must stay reproducible.
Then write `contracts.py` and all six gates in `checks/`, with unit tests, against the *existing*
dataset so you can watch them fail correctly.
**Output:** a harness that red-flags the current corpus, plus the tripwire table that becomes a
thesis figure. *(Already done — both are in this repo.)*

### W2–3 — Testbed v2, part one: parity and isolation (~60 h)
- The Go traffic client: uTLS `HelloChrome_Auto`, four transport modes, budget-driven sessions,
  deterministic seed
- Configure the WebTunnel client with matching uTLS imitation; verify against upstream whether
  the bridge line needs an explicit option
- Private Tor network via `chutney` in Compose; 3 bridge instances; 8 legit vhosts with distinct
  certs

**Output:** two flows from different classes whose ClientHellos are byte-identical. Screenshot it.

### W4 — Testbed v2, part two: capture correctness (~30 h)
- Collector rewrite: one connection per sample, BPF-filtered `tcpdump`, sidecar manifest,
  randomised interleaving, parallel clients
- `ethtool -K` offload off on the capture interface; verify no captured packet exceeds MSS
- netem via `ifb` for ingress plus an `htb` rate ceiling per profile; generate
  `PROFILE_DISPLAY_NAMES` from the applied parameters
- QUIC/HTTP-3 endpoint and class

**Output:** a collector that refuses to write a capture lacking a fresh SYN on an unseen port.

### W5 — Pilot capture + flow builder (~25 h)
2,000 flows across the full design matrix. Write `flow_builder.py` (5-tuple demux, TCP
desegmentation, TLS record extraction, `hs_end_idx`) and run every gate. Expect two or three
iterations. **Start writing the theory chapters this week.**

> ### GATE A — go / no-go
> All six gates green on pilot data, and every surviving tripwire feature has a written protocol
> derivation. If not: fix the testbed and re-pilot. Do not proceed on hope.

### W6–8 — Full capture campaign (~80 h wall, ~15 h hands-on)
Two epochs — A in weeks 6–7, B in week 8 — giving a genuine temporal split. Target 40,000
negative and 8,000 positive flows, plus a ~3,000-flow live-Tor validation set. Run gates nightly
on the accumulated corpus so drift is caught in hours.
**Output:** corpus, manifest, hash list, attrition table with per-class drop reasons.

### W9 — Dataset build and split protocol (~30 h)
`build_dataset.py` v2 emitting four split axes from one corpus: *unseen connection*, *unseen
destination*, *unseen profile*, *unseen epoch* — all keyed on `conn_id`. Fix `cross_validate.py`:
carry groups through the same permutation as features, group by connection, select epochs on an
inner fold.

> ### GATE B — results freeze point
> Everything downstream is now cheap and rerunnable. If the corpus is wrong, this is the last
> moment where fixing it costs weeks instead of a term.

### W10–11 — Models and the generalisation matrix (~50 h)
Retrain the three existing models plus a fourth "model" that is the trivial rule *≥50 % of
upstream records are exactly 558 B*. Produce the generalisation matrix: four split axes × four
models, each cell reporting TPR at fixed FPR with Clopper–Pearson bounds, average precision, and
effective *n*.

### W12 — Defences, cascade, base rate (~35 h)
- Defences applied at TLS-record level on raw traces; static *and* adaptive adversary; bandwidth
  *and* latency overhead
- Cascade rebuilt on measured end-to-end cost — flow table, reassembly, feature computation,
  then inference
- Host-based LLR sum over real per-flow scores, grouped by destination, swept over *M*, using the
  three bridge instances

### W13–16 — Writing, figures, defence rehearsal (~90 h)
Chapters 3–6, regenerate every table from one clean run, diff prose against emitted numbers line
by line. Rehearse the five questions out loud, with slides ready for the tripwire table and the
adaptive-adversary result.

> ### GATE C — no new experiments
> From week 13, results are frozen. A late experiment that contradicts a written chapter is how
> theses miss deadlines.

---

## 7. The thesis narrative gets stronger

The current narrative is "we built a classifier and it gets 99 %". That is unfalsifiable and a
committee knows it. The v2 narrative:

- **We built a measurement apparatus and validated it adversarially.** Here is the gate suite,
  here is what it rejected, here is the corpus it accepted. Include a short section on the v1
  corpus and how the harness detected the leak — that is scientific maturity, and it converts the
  strongest attack on the thesis into a chapter.
- **WebTunnel carries one hard invariant.** 558 B = 5 (TLS header) + 514 (Tor cell) + 22
  (WebSocket framing) + 1 (inner content type) + 16 (AEAD tag), with the 2× multiple at 1072 B.
  It survives handshake removal, connection independence, destination change and network profile.
  A two-instruction rule exploits it — no ML needed.
- **Detection is cheap at the flow level and still hard at the network level.** A measured FPR
  bound below 10⁻⁴ against 40,000 negatives, and the base-rate analysis showing what that does
  and does not buy at ISP scale.
- **Padding is not a defence; distribution relocation is.** Static and adaptive results side by
  side, with the mechanism: 1–128 B moves 558 B to 559–686 B, never entering the legitimate
  upstream support of 40–81 B. A defence must move the distribution *into* that support — MTU-sized
  coalescing plus injected HTTP/2 control chatter (`WINDOW_UPDATE`, `PING`, `SETTINGS`) to
  reproduce the small-record mass every legitimate class has and WebTunnel entirely lacks.
  Independently corroborated by Huma (NDSS 2026).

None of that requires novelty. It requires the claims to match the evidence.

---

## 8. Scope discipline — do **not** do these

- **Do not pre-train a Transformer.** The roadmap correctly ruled this out. The lightweight
  sequence model already satisfies the assignment's "např. 1D-CNN, TrafficFormer".
- **Do not invent a novel defence.** Evaluating two known defence families honestly against an
  adaptive adversary is worth more than a new one evaluated badly.
- **Do not chase FPR = 10⁻⁵ or 10⁻⁶ by measurement.** 600+ h of capture for a number you can
  present as a projection.
- **Do not add a fourth ML model.** Add the trivial 558 B rule instead — an afternoon's work and a
  stronger scientific anchor than any additional network.
- **Do not rewrite in a new language or framework.** The only new non-Python code is the Go
  traffic client, and only because uTLS parity requires it.

---

## 9. Risk register

| Risk | Mitigation | Decide by |
| --- | --- | --- |
| uTLS parity harder than expected — the client may not expose a uTLS option | Fall back to strict handshake removal (`hs_end_idx`) + G1 relaxed to "post-handshake only"; report the stock-Go-fingerprint discovery as a finding in its own right | End of W2 |
| `chutney` adds a week you don't have | It is optional. Keeping live Tor is acceptable if negatives are routed through matched per-destination netem delay so RTT is not class-informative — state the reproducibility limitation | End of W3 |
| Gate A fails repeatedly (most likely G4, budget parity) | Hard stop after three pilot iterations; narrow the budget distribution rather than widening the gate, and document the narrowed range as a scope limitation | W5 |
| Capture throughput below 500 flows/h | Parallel client containers; 30,000 negatives (FPR ≤ 10⁻⁴) is still a full-strength claim. Size the campaign from the *measured* pilot rate, not this document's estimate | W5 |
| Everything slips by three weeks | Theory chapters written from W5 and depend on nothing. W12 and W16 carry slack; cut the cascade chapter before you cut a gate | ongoing |

---

## 10. What to do first

The plan hinges on one counter-intuitive thing: **write the tests before you fix anything.**

1. `git tag v1-audit && git push --tags` — freeze the state the audit refers to.
2. Read `project/common/contracts.py`. *(Already written.)*
3. Run `project/checks/tripwire.py` on the existing `tabular_dataset.npz` and read the output.
   That table is the first figure of the new chapter 4.
4. Run `project/checks/null_controls.py`. Watch it pass on shuffled labels and fail informatively
   on the real ones.
5. **Then, and only then,** open `collect_scaled_dataset.py`.

By the end of that you have a harness that objectively describes what is wrong with the current
corpus — which makes every subsequent decision easy, including the ones this plan gets wrong.
