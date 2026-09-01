"""Registry of features that ARE allowed to separate the classes strongly.

The leakage tripwire (G2) does not ban strong features -- it forces you to explain them.
Any feature whose single-feature stump AUC exceeds TRIPWIRE_AUC_LIMIT must appear here with a
protocol-level derivation, or the testbed has to change.

v2.1 PRUNE (audit V-01 / section 4).
------------------------------------
v2.0 turned G2 green by adding six features to this registry:
    up_len_max, up_len_std, up_len_p90, up_len_mean, len_p90, down_len_p90
None of them had an arithmetic derivation.

  * `up_len_max` separated at AUC 1.0000 with threshold 951 B because three literal constants
    in main.go capped every negative class's upstream records at 830 B.  That is a fact about
    the generator, not about Tor.  Fixed in the generator (P0.5); removed from this registry.
  * `len_p90` and `down_len_p90` were exactly 1500.0 for video_streaming and web_assets --
    the `min(pkt_len, 1500)` clamp firing on TSO super-packets.  That is a capture artefact.
    Fixed by real TCP reassembly (P0.7); removed from this registry.
  * `up_len_std`, `up_len_p90` and `up_len_mean` are downstream consequences of the same two
    artefacts plus the genuine lattice.  They are not independent derivations, so they do not
    get their own entries: the lattice below is the claim, and it is the only claim.

Exactly ONE invariant is registered, and it is the thesis's contribution.
"""

from common.contracts import LATTICE_BASE, LATTICE_OFFSET, TOR_CELL_BYTES

TRIPWIRE_AUC_LIMIT = 0.90

#: feature name -> written, arithmetic justification
EXPECTED_INVARIANTS = {
    "up_lattice_frac": (
        "Tor cell quantization lattice. A WebTunnel upstream TLS record carrying k Tor cells has "
        f"length L = {LATTICE_OFFSET} + {TOR_CELL_BYTES}k, where "
        f"{LATTICE_OFFSET} = 5 (TLS record header) + 22 (WebSocket/HTTPT framing) "
        "+ 1 (TLS 1.3 inner content type) + 16 (AEAD tag), and 514 B is the fixed Tor cell. "
        f"k=1 -> {LATTICE_BASE} B, k=2 -> 1072 B, k=3 -> 1586 B, k=4 -> 2100 B, k=6 -> 3128 B, "
        "k=7 -> 3642 B -- all seven observed on the wire. This feature is the fraction of a "
        "flow's upstream records that satisfy (L - 44) mod 514 == 0. It is a protocol invariant "
        "of Tor's fixed-size cell scheduling, not a laboratory artefact: it survives handshake "
        "removal, TCP reassembly, single-socket demultiplexing, offload removal and every "
        "network profile. It is also directly exploitable without machine learning -- see "
        "3_models/lattice_rule.py. Derivation: tor-spec.txt section 3 (CELL_LEN = 514 for "
        "link protocol v4+), RFC 8446 section 5.2 (inner content type + AEAD expansion), "
        "RFC 6455 section 5.2 (frame header). See docs/04-v2-audit.md section 4.4."
    ),
    "down_lattice_frac": (
        "The same Tor cell lattice, downstream. The bridge relays cells back to the client "
        "under identical framing, so downstream records satisfy the same L = 44 + 514k relation "
        "(3642 B = k7 and 3128 B = k6 are both common). It is registered separately from the "
        "upstream feature only because the two directions have different coalescing behaviour: "
        "downstream bursts coalesce more cells per record, so the fraction is lower "
        "(57.5% vs 74.1% measured). Same derivation, same citation."
    ),

    # ------------------------------------------------------------ percentile echoes
    # docs/05-final-review.md section 6 item 6.
    #
    # These are NOT independent signals.  Every one of them is the lattice re-expressed as an
    # order statistic, and the proof is in their fitted thresholds: the leakage tripwire places
    # every single one at 555 - 557.5 B, i.e. astride the k=1 rung
    #
    #     L(k=1) = 44 + 514*1 = 558 B
    #
    # A depth-1 stump on a percentile can only ask "is this percentile above or below 558?", so
    # once >=50% of a WebTunnel flow's upstream records sit exactly on 558 B (measured: 91.95%),
    # the median and the lower quartile ARE 558 by construction, while every legitimate class --
    # whose upstream records are HTTP/2 control frames and JSON payloads, 0.00-0.12% on the
    # lattice -- sits far below it.  They are registered here rather than deleted because they
    # are legitimate consequences of the invariant; they are registered SEPARATELY from
    # up_lattice_frac so the thesis can state plainly that they add no information beyond it.
    #
    # If any of these ever fires with a threshold that is NOT within a few bytes of a lattice
    # rung, that is a different phenomenon and this justification does not cover it.
    "up_len_p25": (
        "Percentile echo of the Tor cell lattice. With 91.95% of WebTunnel upstream records at "
        "exactly 558 B, the 25th percentile of the upstream record length is 558 B for virtually "
        "every positive flow; the legitimate classes sit at 40-350 B. Measured stump threshold "
        "557 B -- one byte below the k=1 rung, which is what a split on a point mass looks like. "
        "Adds no information beyond up_lattice_frac."
    ),
    "up_len_p50": (
        "Percentile echo of the Tor cell lattice: the upstream median. This is the feature the "
        "v1 audit identified as the original 558 B result (F-09). Measured stump threshold "
        "555 B. It is the weakest of the echoes (AUC 0.9141) precisely because the hard negatives "
        "were designed to overlap the cell band -- 39% of them have up_len_p50 in [500, 620] B -- "
        "so the median alone no longer separates cleanly and the lattice test does. Adds no "
        "information beyond up_lattice_frac."
    ),
    "down_len_p25": (
        "Percentile echo of the Tor cell lattice, downstream. The bridge relays cells back under "
        "identical framing, so 70.01% of downstream records are on the lattice and the lower "
        "quartile lands on the k=1 rung. Measured stump threshold 557.5 B. Adds no information "
        "beyond down_lattice_frac."
    ),
    "len_p25": (
        "Percentile echo of the Tor cell lattice over both directions combined -- the union of "
        "the two cases above. Measured stump threshold 557.5 B. Adds no information beyond "
        "up_lattice_frac and down_lattice_frac."
    ),
}


#: Features that are KNOWN laboratory artefacts. If any of these fires, the corpus is broken.
#: They are listed so the gate names the failure precisely instead of just printing a number.
KNOWN_ARTEFACTS = {
    "up_len_max": (
        "The negative generators' upstream payload ceiling. v2.0 capped negative upstream "
        "payloads at 400+rand(400) / 350+rand(400) B, so no negative could emit a TLS record "
        "above 830 B while WebTunnel coalesces to 1072 B and beyond. If this fires, the "
        "generator's payload distribution does not reach the MSS -- see main.go payloadSize()."
    ),
    "len_p90": (
        "The min(pkt_len, 1500) clamp on TSO super-packets. If this reads exactly 1500 for any "
        "class, segmentation offload is still on and/or TCP reassembly is not running (F-05, V-04)."
    ),
    "down_len_p90": "Same clamp as len_p90, downstream. See F-05 / V-04.",
    "down_len_min": (
        "The 6-byte TLS 1.3 ChangeCipherSpec record (or the 80-byte TLS 1.2 one). Present in "
        "flows that contain a handshake and absent from mid-connection windows. If this fires, "
        "the collector is not opening a fresh connection per sample (F-02)."
    ),
    "total_bytes": "Session-budget parity failed -- see gate G4 (F-06).",
    "total_pkts": "Session-budget parity failed -- see gate G4 (F-06).",
    "ratio_up_bytes": "Session-budget parity failed -- see gate G4 (F-06).",
    "ratio_up_pkts": "Session-budget parity failed -- see gate G4 (F-06).",
    "iat_max": "Capture-window duration is acting as a class label -- see gate G4 (F-06).",
    "clienthello_len": "TLS stack parity failed -- see gate G1 (F-03).",
}
