# Final review — v2.1 pilot, 1,843 flows

**Date:** 1 September 2026
**Scope:** the v2.1 pilot corpus and everything built on it — 2,016 captures, 1,843 flows,
six gates, four models, the defence simulation, the LLR aggregation and six LaTeX tables.
**Method:** every claim re-derived independently. Gates re-executed; the 5-tuples, ClientHellos,
JA4 fingerprints and TLS record lattices re-parsed from `flow_records.jsonl`; the classifiers
re-trained under feature ablations; the defences re-applied to the raw record traces.

> ## Verdict: 7/10. The science is now real, and three of the four hardest findings are closed.
> ## It is **not yet defensible**, because four of six gates fail and one of them makes a
> ## one-line JA4 lookup a perfect classifier on this corpus.
>
> The distance from here to a defensible thesis is roughly **two days of testbed work and one
> re-pilot** — smaller than any previous gap. None of the remaining problems are in the analysis
> layer; all four are in the last few lines of the generator and the server configuration.

---

## 1. What is now genuinely fixed — verified, not accepted

These were the blocking findings of the first two audits. They are closed.

| Finding | v2.0 | **v2.1 (measured)** |
| --- | --- | --- |
| **F-01** independent positive connections | 8.4 effective sockets; one carried 33.2 % of positives and spanned all three splits | **308 sockets, effective *n* = 308.0, largest carries 0.3 %**, none spans a split |
| **F-02** captures containing a handshake | 56.6 % of WebTunnel | **1,841 / 1,843 flows (99.9 %)**; client SYN present in **1,843 / 1,843** |
| **F-04 / V-02** provenance | 5-tuple hardcoded `172.20.0.3:0`; 81.8 % of positive captures merged ≥2 sockets | **G6 PASS** — 1,843 flows, 0 missing manifests, **0 five-tuple mismatches**, 100 % capture accounting, structured per-class drop reasons |
| **F-05** segmentation offload | 72.7 % of `video_streaming` payload packets > 1500 B | `offloads_disabled: true`, `mss: 1460` on **all 2,051** manifest rows |
| **F-08** class/profile blocked by wall-clock | fully blocked | matrix shuffled across class *and* profile; one continuous 3.25 h campaign, no gaps > 5 min |
| **V-01** vacuous gates | 4 of 6 could not fail | all six falsifiable; `checks/test_gates.py` **15/15**; the freshness check fired **139 times** (`socket_reused_from`), so it is doing real work |

**The single biggest result: F-01 is genuinely dead.** Effective *n* went from 8.4 to 308.0.
Every connection-level confidence interval in the thesis is now meaningful. That was the finding
that made the whole practical part indefensible, and it is gone.

### The lattice invariant, confirmed at full strength

Measured on post-handshake application records only:

| class | upstream on lattice | downstream on lattice |
| --- | ---: | ---: |
| **webtunnel** | **91.95 %** | **70.01 %** |
| websocket_chat | 0.12 % | 0.18 % |
| websocket_ticker | 0.12 % | 0.00 % |
| direct_web_browsing | 0.06 % | 0.01 % |
| web_assets | 0.02 % | 0.01 % |
| video_streaming | 0.00 % | 0.00 % |

The complete ladder `L = 44 + 514k` is present with real mass, k = 1 … 7:

```
558 ×12,721   1072 ×2,418   1586 ×1,048   2100 ×983   2614 ×129   3128 ×…   3642 ×540
```

The two lattice features **alone** give accuracy 1.0000 / ROC-AUC 1.0000 on the socket-disjoint
test split. `lattice_rule.py` reports TPR 1.0000 [0.9229, 1.0000] at **0 false positives in 227
negatives** (Clopper–Pearson 95 % upper bound 1.61 × 10⁻²), and it does so **after the handshake
is stripped**. This is a real protocol result and it is the thesis.

---

## 2. Gate reality check

The summary supplied with this run reported G5 and G6 as PASS and described G1/G2 without a
verdict. Re-executed:

```
G1  FAIL   3 distinct JA4 fingerprints; 2 distinct ALPN offers; webtunnel ClientHello {267} disjoint
G2  FAIL   down_len_min AUC 1.0000 (known artefact) + 4 unregistered features
G3  FAIL   webtunnel early-vs-late AUC 0.6469 [0.5726, 0.7212]
G4  FAIL   paired budget parity, worst p = 3.7e-48, worst median ratio 13.87x
G5  PASS   308 independent positive sockets, all splits disjoint, groups aligned
G6  PASS   100 % authoritative provenance, 0 mismatches, full accounting
```

**Four of six gates block the build.** By the project's own hard rule — *"no dataset reaches a
model before the gates pass"* — the model results, the CV table, the cascade table and the class
breakdown should not exist yet. They were produced anyway. That is the process failure to fix
first, because it is the one a committee can see from the outside.

---

## 3. Four blocking issues

### B-1 · **C** · JA4 alone is a perfect classifier, and it is now in 100 % of samples

```
webtunnel   t13d190900_9dc949149365_e7c285222651     ClientHello 267 B, 9 ciphers, 9 exts, no ALPN
negatives   t13d1514h2_8daaf6152771_02713d6af862     ClientHello 506/538/570/602 B  (HTTP/2 classes)
            t13d1514h1_8daaf6152771_02713d6af862     ClientHello 503/535/567/599 B  (WebSocket classes)
```

Zero overlap. A single threshold on `clienthello_len` at 385 B gives **TPR 1.0000, FPR 0.0000**.
`8daaf6152771` is the published Chrome JA4_b; `9dc949149365` is stock Go `crypto/tls`.

There is a bitter irony here: **fixing F-01 made F-03 worse.** In v2.0 only 56 % of WebTunnel
captures contained a handshake, so the fingerprint leak was partial. Now every capture opens a
fresh connection, so the leak is in **100 %** of samples. No accuracy number computed on the full
flow can be attributed to traffic analysis while this holds — a censor would run one JA4 lookup
and stop.

The remedy is already scripted (`probe_utls_support.sh`) and was not run. Two acceptable outcomes,
either of which is defensible; silence is not:
- the PT accepts a uTLS argument → set `WEBTUNNEL_EXTRA_ARGS`, re-pilot, G1 goes green;
- it does not → **report the stock-Go fingerprint as a first-class finding** (it is one — WebTunnel
  is JA3/JA4-detectable out of the box), and make the handshake-stripped analysis the *primary*
  result rather than an ablation.

### B-2 · **C** · ALPN parity was reverted, and the manifest now misreports it

`main.go` was edited after the remediation commit:

```go
runWebSocket:  dialUTLS(ctx, addr, sni, []string{"http/1.1"})     // V-07, re-introduced
runHTTP2:      dialUTLS(ctx, addr, sni, []string{"h2","http/1.1"})
```

That splits the negatives into two JA4s (`h1` vs `h2`) and shifts their ClientHello by exactly
3 bytes — the transport is readable from the handshake again.

Worse, `newResult()` still reports `ALPNOffered: ALPNParity` unconditionally, so **all 2,051
manifests record `("h2","http/1.1")` while the wire shows `("http/1.1",)` for both WebSocket
classes.** The manifest is now factually wrong on a field it claims as ground truth. G1 caught it
only because it reads the ClientHello out of the PCAP rather than trusting the sidecar — which is
the design working, but the provenance regression has to be repaired regardless.

### B-3 · **C** · WebTunnel fetches the live public internet; every negative fetches a local mock

```go
var webtunnelTargets = []string{
    "https://duckduckgo.com", "https://check.torproject.org",
    "https://en.wikipedia.org/wiki/Main_Page",
}
```

The default `--target-url` triggers the substitution, so this is what ran for all 342 WebTunnel
captures. Consequences, measured:

| | webtunnel | negatives |
| --- | ---: | ---: |
| median duration | **5.18 s** | 2.42 – 2.64 s |
| median `bytes_up` | **55,566** | 9,336 – 22,002 |
| G4 median ratio | — | up to **13.87 ×** |

Content, RTT, response size and session length are all confounded with the class by *destination*,
not by protocol. The session budget cannot be honoured because `duckduckgo.com` does not return
B_down bytes on request — which is exactly why G4 fails at *p* ≈ 10⁻⁴⁸. This is rebuild-plan
principle **P6** violated, and it is a first-order confound: a reviewer will ask whether the models
learned "Tor cell quantization" or "this flow talked to the real web".

(Also worth a moment's thought: the generator POSTs random bytes to DuckDuckGo and Wikipedia
several hundred times per campaign. Point it at the local decoy for the controlled corpus.)

### B-4 · **M** · A server-side TLS stack fingerprint — the mirror image of F-03

```
down_len_min    webtunnel: min = p50 = max = 6      (all 308 flows)
                negatives: 24 – 58                  (no flow at 6)
                -> single-feature stump AUC 1.0000
```

Six bytes is the TLS 1.3 middlebox-compatibility ChangeCipherSpec. The bridge front-end is
**nginx**; the legitimate servers are **Python/OpenSSL**. They differ in whether they emit it.
So after fixing the client-stack asymmetry, the corpus still carries a *server*-stack asymmetry
that separates the classes perfectly on its own. This is a new finding, and the fix is cheap:
terminate the negatives' TLS with the same nginx build that fronts the bridge.

---

## 4. The forensic answer: the classifiers are not using the invariant

This is the most important measurement in the review. Applying each defence to the raw record
traces and scoring the same test split with both detectors:

| condition | lattice rule TPR | ML (static adversary) recall |
| --- | ---: | ---: |
| undefended | 1.0000 | 1.0000 |
| Mode 1 — intra-record padding | **0.0000** | **1.0000** |
| Mode 2 — coalescing + control chatter | **1.0000** | **0.0000** |

A perfect dissociation. Padding annihilates the lattice — the rule's recall goes to zero — and the
classifier does not notice. Coalescing leaves the lattice largely intact (fraction 0.9195 → 0.5365)
and breaks the frozen classifier completely.

**The two detectors are measuring different things.** The lattice rule measures the protocol. The
ML models ride the JA4, handshake and volumetric leaks (B-1 … B-4), which is why they survive a
defence that destroys the invariant and collapse under one that does not.

This has a direct consequence for the write-up: **`lattice_rule.py` is the primary detector of
this thesis, and the neural models are the comparison, not the headline.** Once B-1 … B-4 are
fixed, the models should converge on the lattice too — and demonstrating that convergence is a
far better result than three more rows of 100 %.

The rest of the ablation supports the same reading. On the socket-disjoint test split every
strong surviving feature has its stump threshold at **555 – 557.5 B**, i.e. it is the lattice
re-expressed as a percentile, not an independent signal:

```
lattice features only (2)                                    acc 1.0000  AUC 1.0000
drop lattice + percentile echoes + artefacts + volumetric    acc 0.9963  AUC 0.9975  (20 feats)
  top residual: down_len_p25 AUC 0.9912 (thr 557.5) · len_p25 0.9714 (557.5) · up_len_p25 0.9559 (557)
```

---

## 5. Non-blocking, but a committee will ask

- **The cascade has no second tier.** `n_l2_escalated = 0`, `pct_l2_escalated = 0.0 %`. A two-tier
  architecture whose L2 never runs cannot support an economic argument, and `table_cascaded_pipeline.tex`
  presents the degenerate case as a result. Either report the escalation rate as *zero on a saturated
  task and therefore uninformative*, or drop the chapter (the rebuild plan already said to cut the
  cascade before cutting a gate).
- **Post-handshake ≥ full flow.** 1D-CNN 99.64 % → 100.00 %; XGBoost 100 % → 100 %. Removing
  information improves accuracy, which means the difference is noise at n = 273. This cannot support
  "detection is independent of TLS metadata"; the lattice derivation supports that, this table does not.
- **Resolution floor.** 46 positives / 227 negatives in the test split → smallest measurable FPR
  4.41 × 10⁻³. The corpus supports "FPR below 1.6 × 10⁻² at 95 % confidence" and nothing more.
  The DET curve and the base-rate table handle this correctly now; the prose must match them.
- **G3 temporal drift in the positive class.** `webtunnel` early-vs-late AUC 0.6469 [0.5726, 0.7212]
  over a 3.25 h campaign — consistent with live-internet targets and changing Tor circuits.
  Fixing B-3 should resolve it; re-check after the re-pilot.
- **`table_class_breakdown.tex` still says "Tor over HTTP/2 WSS".** WebTunnel offers no ALPN
  (`ja4_a` ends `00`), so it runs HTTP/1.1 Upgrade. F-18, still open, and easy to spot.
- **`manifest.jsonl` holds 2,051 rows for 2,016 captures.** 35 captures were retried; the sidecars
  agree with the last attempt and 8 retries turned a failure into a success, so the data is sound —
  but the corpus is 2,016 captures / 1,843 flows, not "2,016 entries", and the thesis must say so.
- **`± 0.0 %` from five folds** is not a confidence interval. Report Clopper–Pearson at the
  connection level; with 308 independent positive sockets you can finally afford to.

---

## 6. The fix list — small, and all of it upstream

Ordered. None of this touches the analysis layer.

1. **Revert the ALPN edit** in `runWebSocket` (one line), and make `newResult()` report
   `uConn.ConnectionState()`'s *actual* offer instead of the constant. *(30 min)*
2. **Run `probe_utls_support.sh`** and take one of the two documented outcomes. Decide before
   capturing. *(1 h)*
3. **Point WebTunnel at a local target through the bridge** — the decoy site, or a vhost on
   `legitimate-servers` reached through the tunnel — so content, RTT and budget match the negatives.
   Keep the live-internet runs as a separate, clearly labelled ~300-flow transfer set. *(2 h)*
4. **Terminate the negatives' TLS with the same nginx** that fronts the bridge, killing
   `down_len_min`. *(1 h)*
5. **Re-pilot (2,016 captures ≈ 3.5 h) and re-run the gates.** Expect G1, G2, G4 to go green and
   G3 to follow once B-3 is gone. *(4 h)*
6. **Register the lattice percentile echoes** (`up_len_p25`, `down_len_p25`, `len_p25`, `up_len_p50`)
   in `expected_invariants.py` with the note that their thresholds sit at 555–557.5 B, i.e. they are
   the same invariant — or, cleaner, exclude raw percentile features and keep the two explicit
   lattice features. *(30 min)*
7. Only then re-run `run_full_benchmark.py`, and let the blocking gate phase do its job.

---

## 7. The narrative for chapter 5 (Results)

The current draft narrates "we built three classifiers and they all got 100 %". That is the
weakest possible framing of genuinely good work, and it invites the one question the corpus cannot
yet answer. Restructure around the invariant and the adversary.

### §5.1 — The measurement apparatus, and what it rejected

Open with the harness, not the accuracy. Present the six gates, the fact that
`checks/test_gates.py` proves each one discriminates in both directions, and the table of what
they rejected: the v1 corpus, the v2.0 corpus, and — honestly — the first v2.1 pilot on G1–G4.
Finish with the accepted corpus: 2,016 captures, 1,843 flows, **308 independent positive
connections**, 100 % authoritative provenance, per-class attrition with reasons.

*Why first:* it converts the project's history from an embarrassment into a method contribution,
and it earns the reader's trust before a single number is quoted. It is also the assignment's
*"předzpracování dat pro zamezení učení modelu z irelevantních znaků"* requirement, discharged
visibly.

### §5.2 — The invariant: `L = 44 + 514k`

The centrepiece. Derive it from first principles — 514 B Tor cell (tor-spec §3), 22 B
WebSocket/HTTPT framing, 5 B TLS record header, 1 B inner content type, 16 B AEAD tag — then show
the measured ladder with counts (558 ×12,721 … 3642 ×540, k = 1…7), then the class table
(91.95 % vs 0.00–0.12 %). One figure: the upstream record-length histogram per class, log-y, with
the lattice positions marked.

State the result as a *detector*, not a feature: **TPR 1.0000 [0.9229, 1.0000] at 0 false positives
in 227 negatives, post-handshake, with no machine learning**. Give the Clopper–Pearson bound and
the resolution floor in the same sentence — volunteering the limit is what makes the number
credible.

### §5.3 — The learned models as a *comparison*, not a headline

Report XGBoost / 1D-CNN / Flow-Transformer against the lattice rule as the reference method the
assignment asks for. The interesting statement is not that they reach 100 % but that **they do not
beat two integer operations**, and that on a corpus with residual stack asymmetries they are
demonstrably learning something else. Include the ablation table (lattice only → 1.0000; drop
lattice + echoes + artefacts + volume → 0.9963) and the threshold column showing every residual
feature sitting at 555–557.5 B.

### §5.4 — Countermeasures and the adaptive adversary — the strongest chapter

Lead with the dissociation table from §4 above. It is the most interesting empirical result in the
whole thesis:

| | lattice rule | frozen classifier |
| --- | ---: | ---: |
| padding | recall 0.00 | recall 1.00 |
| coalescing + chatter | recall 1.00 | recall 0.00 |

Then the adaptive arm: retrain the censor on defended traffic and both defences collapse to
recall 1.0000. Then the mechanism, which the data now supports precisely — padding moves 558 B to
559–686 B, still nowhere near the legitimate upstream support; coalescing halves the lattice
fraction (0.9195 → 0.5365) but does not leave it, so a retrained model re-finds it. Then the cost
that nobody had measured: **76.4 ms of added buffering latency** on an interactive circuit for
0.95 % of bandwidth.

Close with the design requirement that follows: a defence must move the distribution *into* the
legitimate support — MTU-sized coalescing **plus** injected HTTP/2 control chatter to manufacture
the small-record mass every legitimate class has and WebTunnel entirely lacks — and note the
independent corroboration in Huma (NDSS 2026).

### §5.5 — Base rate, and what the corpus can and cannot say

Per-flow operating point with its interval, the host-level LLR sweep, and then the honest
statement: 227 test negatives means a measured floor of 4.41 × 10⁻³, so the 10⁻⁴ and 10⁻⁵ columns
are **analytical projections** and are labelled as such in the table. Say what measuring 10⁻⁴
would cost (30,000 clean negatives, rule of three) and why it was out of scope. A committee
rewards a stated limit far more than a projected number presented as a measurement.

### §5.6 — Threats to validity

Write this section yourself before anyone writes it for you: single bridge and single legitimate
vhost (no destination split), no QUIC class, one capture epoch, live-vs-local destination
asymmetry if B-3 is not fixed, and the residual TLS stack differences. Each one already has a
number attached from the gates — use them.

---

## 8. What to say if asked "is it ready?"

Not today. After the six fixes in §6 and one re-pilot, yes — and it will be stronger than most
work at this level, because it will rest on a derived protocol invariant, an adversarially
validated corpus, and a countermeasure analysis with a genuine adaptive arm. The gap is two days
of testbed work, not two months of science.

---

# Appendix — v2.2 upstream fixes (implemented)

Branch `v2.2-upstream-fixes`. Four of the six items in §6 are done; the two that remain are the
uTLS decision (§6.2) and the re-pilot (§6.5), both of which need the testbed hardware.

| § | Fix | What changed | Expected gate effect |
| --- | --- | --- | --- |
| 6.1 | **B-2 ALPN parity + truthful reporting** | `dialUTLS` no longer takes an ALPN parameter — every class offers `ALPNParity` and the function *returns the list it actually sent*. `newResult(local, remote, offered, picked, …)` records that list and `ConnectionState().NegotiatedProtocol`, never the constant. The collector stopped substituting `ALPN_PARITY` for a missing measurement, and `CaptureManifest.alpn_offered` now defaults to `()` instead of the parity tuple. | G1: negatives collapse from two JA4s to one |
| 6.3 | **B-3 local target** | The `webtunnelTargets` public list is deleted. `--target-url` has no default and an unset target is a hard error in both the generator and the collector. The bridge hosts a v3 onion service (`HiddenServiceDir /var/lib/tor/onion_decoy`) pointing at the legitimate front end's plain-HTTP port, and the collector discovers the address and passes it in. `runWebTunnel` now issues the *same* request mix as `direct_web_browsing` via a shared `budgetedRequest()`, against the same handlers. | G4: paired budgets become satisfiable; G3 positive-class drift should subside |
| 6.4 | **B-4 one server stack** | `common_conf/tls_common.conf` holds the TLS parameters and is included by *both* nginx front ends. The FastAPI app moved to plain HTTP on 8000 behind a new `legitimate-servers` nginx container (8443 = h2, 8444 = http/1.1, 8080 = plain for the onion), using the same image and the same certificate as the bridge. | G2: `down_len_min` — the last unregistered offender — should disappear |
| 6.6 | **G2 percentile echoes** | `up_len_p25`, `up_len_p50`, `down_len_p25`, `len_p25` registered with the derivation tying each to the k=1 rung. Their measured thresholds (557.5, 557, 557, 554.8 B) are the evidence. `checks/tripwire.py` also runs standalone again. | G2: 5 unregistered offenders → 1 |

**Verified on the existing corpus:** gate self-test still 15/15; the tripwire now reports exactly
one unregistered feature, `down_len_min`, which is precisely what the B-4 change removes. That is
the falsifiable prediction for the re-pilot.

### Why an onion service rather than `http://172.20.0.10/`

A private address cannot be fetched through Tor SOCKS: the circuit exits to the public internet,
exit policies reject RFC 1918, and no exit has a route into the lab network. An onion service
keeps the traffic inside Tor end to end and terminates it on the same nginx and the same FastAPI
handlers the negative classes use — which is stronger parity than a plain local fetch would give,
because the positive and negative sessions now differ *only* in the WebTunnel transport.

Cost: the client re-fetches the onion descriptor after each `stop_tor`/`start_tor` cycle, so
per-sample bootstrap is slower than in v2.1. Size the campaign from the measured pilot rate.

### Still open

- **B-1 remains the blocking leak.** WebTunnel's ClientHello is still 267 B stock Go, so JA4 alone
  is a perfect classifier. `probe_utls_support.sh` has not been run. Until it is, G1 cannot pass
  and no full-flow accuracy number is attributable to traffic analysis.
- The Go generator has not been compiled — no toolchain in the review environment. `go build` is
  the first thing the runner should do.
