"""
v2 data contract for the WebTunnel detectability testbed.

WHY THIS EXISTS
---------------
The v1 pipeline let `sanitizer.py` *infer* what it was looking at: which host was the client
(from the first packet in the file), which packets belonged to the flow (all of them), and where
the TLS handshake ended (the first 0x17 byte). Each of those inferences is wrong somewhere in the
v1 corpus, and nothing in the pipeline could notice. See docs/01-audit-findings.md F-01..F-04.

The collector, by contrast, KNOWS all of it. So it writes it down, and the parser stops guessing.

THREE RULES THAT MAKE THIS SELF-ENFORCING
-----------------------------------------
1. A flow whose 5-tuple does not match `CaptureManifest.target_5tuple` EXACTLY is DISCARDED,
   not analysed.  `target_5tuple` must name a concrete client port -- a zero port is a
   contract violation, not a wildcard (v2 audit V-02).
2. `FlowRecord.conn_id` is the ONLY legal grouping key for train/val/test splits and for
   StratifiedGroupKFold. Never group by sample index (that is what produced F-11).
3. Provenance is graded. Only `provenance == "collector"` is authoritative ground truth.
   Anything reconstructed after the fact is marked and fails gate G6.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

CLASSES: Tuple[str, ...] = (
    "webtunnel",
    "direct_web_browsing",
    "websocket_ticker",
    "websocket_chat",
    "video_streaming",
    "web_assets",
    "quic_http3",          # reserved -- closes the assignment's explicit QUIC requirement
)

#: Client behaviour, applied to EVERY class including the positive one.
BEHAVIOURS: Tuple[str, ...] = ("browse", "bulk", "interactive")

PROFILES: Tuple[str, ...] = ("broadband", "lte", "lossy")

#: The single TLS fingerprint every class must present. Enforced by gate G1.
CLIENT_STACK: str = "utls-HelloChrome_Auto"

#: The ALPN list every class must offer.  A class-dependent ALPN list changes the
#: ClientHello length by exactly the size of the extension and leaks the transport (V-07).
ALPN_PARITY: Tuple[str, ...] = ("h2", "http/1.1")

#: Only this value means "written by the collector at capture time".
PROVENANCE_AUTHORITATIVE: str = "collector"

FiveTuple = Tuple[str, int, str, int, str]  # (src_ip, src_port, dst_ip, dst_port, "tcp"|"udp")


# ---------------------------------------------------------------------------
# The Tor cell lattice -- the project's one genuine protocol invariant
# ---------------------------------------------------------------------------
#
#   L = TLS_HDR + (CELL * k + FRAMING) + INNER_TYPE + AEAD_TAG
#     = 5       + (514  * k + 22     ) + 1          + 16
#     = 44 + 514 * k
#
# k = 1 -> 558, k = 2 -> 1072, k = 3 -> 1586, k = 4 -> 2100, ... all observed on the wire.
TOR_CELL_BYTES: int = 514
LATTICE_OFFSET: int = 44          # 5 (TLS header) + 22 (WS/HTTPT framing) + 1 (inner type) + 16 (AEAD)
LATTICE_BASE: int = LATTICE_OFFSET + TOR_CELL_BYTES   # 558


def on_tor_lattice(record_len: int) -> bool:
    """True iff `record_len` is an exact k-cell WebTunnel TLS record (k >= 1)."""
    return record_len >= LATTICE_BASE and (record_len - LATTICE_OFFSET) % TOR_CELL_BYTES == 0


def lattice_k(record_len: int) -> int:
    """Number of Tor cells in the record, or 0 if it is not on the lattice."""
    return (record_len - LATTICE_OFFSET) // TOR_CELL_BYTES if on_tor_lattice(record_len) else 0


# ---------------------------------------------------------------------------
# Emitted by the collector, one per capture, alongside the .pcap
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaptureManifest:
    """Ground truth for a single capture. Written by the collector; never inferred."""

    capture_id: str
    pcap_path: str

    # --- what was generated -------------------------------------------------
    label: str                      # one of CLASSES
    behaviour: str                  # one of BEHAVIOURS -- and actually applied by the generator
    profile: str                    # one of PROFILES
    dest_id: str                    # "vhost-03" | "bridge-02" -- enables destination-split
    client_stack: str = CLIENT_STACK

    # --- the flow the analysis is allowed to look at ------------------------
    target_5tuple: Optional[FiveTuple] = None
    client_ip: str = ""             # authoritative; NEVER derive direction from packet order

    # --- session budget (drawn from a distribution SHARED by all classes) ----
    #: In v2 the budget is drawn ONCE per (sample_id, profile) and handed to EVERY class,
    #: so gate G4 becomes a paired test rather than a two-sample test (v2 audit P0.9/G4).
    budget_id: str = ""
    target_duration_s: float = 0.0
    target_bytes_up: int = 0
    target_bytes_down: int = 0

    # --- provenance ---------------------------------------------------------
    provenance: str = PROVENANCE_AUTHORITATIVE
    alpn_offered: Tuple[str, ...] = ALPN_PARITY
    mss: int = 0                    # negotiated MSS on the capture interface; 0 = unknown
    offloads_disabled: bool = False
    t_start: float = 0.0
    t_end: float = 0.0
    generator_seed: int = 0
    netem_params: Dict[str, str] = field(default_factory=dict)
    git_commit: str = ""
    epoch: str = "A"                # capture campaign epoch -> free temporal split
    ok: bool = True                 # did the generator complete the session?
    drop_reason: str = ""           # why the capture is unusable, if it is
    notes: str = ""

    @property
    def is_positive(self) -> bool:
        return self.label == "webtunnel"

    @property
    def is_authoritative(self) -> bool:
        return self.provenance == PROVENANCE_AUTHORITATIVE

    def validate(self) -> Optional[str]:
        """Return a drop reason, or None when the manifest is usable ground truth."""
        if not self.ok:
            return self.drop_reason or "generator_failed"
        if self.label not in CLASSES:
            return f"unknown_label:{self.label}"
        if self.behaviour not in BEHAVIOURS:
            return f"unknown_behaviour:{self.behaviour}"
        if not self.target_5tuple:
            return "no_target_5tuple"
        src_ip, src_port, dst_ip, dst_port, proto = self.target_5tuple
        if not src_ip or not dst_ip:
            return "incomplete_5tuple"
        # A zero client port is what v1/v2 used as a wildcard. It is now illegal.
        if int(src_port) == 0:
            return "client_port_zero"
        if int(dst_port) == 0:
            return "server_port_zero"
        if proto not in ("tcp", "udp"):
            return f"bad_proto:{proto}"
        if not self.client_ip:
            return "no_client_ip"
        if self.client_ip != src_ip:
            return "client_ip_disagrees_with_5tuple"
        return None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(text: str) -> "CaptureManifest":
        d = json.loads(text)
        if d.get("target_5tuple") is not None:
            d["target_5tuple"] = tuple(d["target_5tuple"])
        if d.get("alpn_offered") is not None:
            d["alpn_offered"] = tuple(d["alpn_offered"])
        # A manifest written before v2.1 has no `provenance` key.  Do NOT let it inherit the
        # authoritative default -- gate G6 must be able to tell reconstructed ground truth from
        # recorded ground truth (audit V-02/V-03).
        if "provenance" not in d:
            d["provenance"] = "legacy-pre-v2.1"
        known = CaptureManifest.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        d = {k: v for k, v in d.items() if k in known}
        return CaptureManifest(**d)


# ---------------------------------------------------------------------------
# Produced by the flow builder, consumed by everything above it
# ---------------------------------------------------------------------------

@dataclass
class FlowRecord:
    """One TCP (or QUIC) connection. THE atom of the v2 dataset.

    `records` holds TLS-record-level observations after **TCP stream reassembly**, so it is
    independent of segmentation offload on the capture host (v1 failure F-05, v2 failure V-04).
    Lengths are true TLS record lengths; nothing is ever clamped to the MTU.
    """

    flow_uid: str                   # sha256(capture_id | 5-tuple | syn_ts)[:16]
    conn_id: str                    # identity of the underlying CONNECTION -- THE grouping key
    socket_id: str                  # client ip:port -- collides across captures of one socket
    capture_id: str

    label: str
    behaviour: str
    profile: str
    dest_id: str
    client_stack: str
    epoch: str

    # (t_relative_seconds, direction:+1 client->server / -1 server->client, tls_record_len)
    records: List[Tuple[float, int, int]] = field(default_factory=list)
    #: parallel to `records`: the TLS ContentType byte of each record (20/21/22/23)
    content_types: List[int] = field(default_factory=list)

    #: index into `records` of the first true application record -- the client's first
    #: application_data record AFTER its Finished (not the first 0x17 byte, see F-12)
    hs_end_idx: int = 0
    saw_client_syn: bool = False
    saw_full_handshake: bool = False

    # denormalised for convenience / gate checks
    n_up: int = 0
    n_down: int = 0
    bytes_up: int = 0
    bytes_down: int = 0
    duration: float = 0.0

    #: ClientHello observables. Gate G1 asserts these are identical across classes.
    clienthello_len: Optional[int] = None
    ja4: Optional[str] = None
    tls_extensions: Optional[List[int]] = None
    alpn_offered: Optional[List[str]] = None

    #: budget the collector asked this session to hit -- gate G4 pairs on budget_id
    budget_id: str = ""
    target_duration_s: float = 0.0
    target_bytes_up: int = 0
    target_bytes_down: int = 0

    provenance: str = PROVENANCE_AUTHORITATIVE
    git_commit: str = ""
    generator_seed: int = 0
    #: wall-clock start of the capture, from the manifest. Needed by gate G3's temporal control:
    #: without it there is no way to ask "can the model tell early captures from late ones?",
    #: which is the only same-generator control that can actually fail.
    t_start: float = 0.0

    @property
    def is_positive(self) -> bool:
        return self.label == "webtunnel"

    @property
    def app_records(self) -> List[Tuple[float, int, int]]:
        """Post-handshake records only. Use this for any 'handshake-free' experiment."""
        return self.records[self.hs_end_idx:]

    def lattice_fraction(self, direction: int = 1, post_handshake: bool = True) -> float:
        """Fraction of records in `direction` that sit on the 44 + 514k Tor cell lattice."""
        src = self.app_records if post_handshake else self.records
        lens = [l for (_t, d, l) in src if d == direction]
        if not lens:
            return 0.0
        return sum(1 for l in lens if on_tor_lattice(l)) / len(lens)

    @staticmethod
    def make_uid(capture_id: str, five_tuple: FiveTuple, syn_ts: float) -> str:
        raw = f"{capture_id}|{'|'.join(map(str, five_tuple))}|{syn_ts:.6f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def make_socket_id(five_tuple: FiveTuple) -> str:
        """Socket identity WITHOUT any timestamp.

        Two captures of the SAME long-lived socket collide here on purpose -- that is exactly
        the v1/v2 failure (F-01) the split logic has to be able to see. `conn_id` alone cannot
        see it, because it mixes in the SYN timestamp.
        """
        src_ip, src_port, dst_ip, dst_port, proto = five_tuple
        return f"{src_ip}:{src_port}->{dst_ip}:{dst_port}/{proto}"

    @staticmethod
    def make_conn_id(five_tuple: FiveTuple, syn_ts: float) -> str:
        """Connection identity: socket + the SYN that opened it."""
        raw = f"{FlowRecord.make_socket_id(five_tuple)}|{syn_ts:.3f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Split protocol
# ---------------------------------------------------------------------------

#: The four generalisation axes the v2 evaluation reports. All are keyed on conn_id.
SPLIT_AXES: Tuple[str, ...] = (
    "unseen_connection",   # baseline: no conn_id AND no socket_id shared between folds
    "unseen_destination",  # test dest_ids never seen in training  (roadmap 5.2)
    "unseen_profile",      # train broadband, test lte + lossy
    "unseen_epoch",        # train campaign A, test campaign B     (genuine temporal split)
)


def assert_split_disjoint(splits: Dict[str, Sequence[str]], key_name: str = "conn_id") -> None:
    """Raise if any key appears in more than one split.

    Call this with conn_ids AND with socket_ids -- conn_id disjointness is necessary but not
    sufficient, because two windows of one long-lived socket carry different conn_ids.
    """
    seen: Dict[str, str] = {}
    for split_name, keys in splits.items():
        for k in keys:
            if k in seen and seen[k] != split_name:
                raise AssertionError(
                    f"{key_name} {k!r} appears in both {seen[k]!r} and {split_name!r}. "
                    "This is exactly the v1 leak (F-01): the same TCP connection in train and test."
                )
            seen[k] = split_name


def assert_groups_aligned(rows: Sequence, groups: Sequence, feature_matrix_name: str = "X") -> None:
    """Raise unless the group vector is element-wise aligned with the feature matrix.

    v1 built `X = concat(X_train, X_val, X_test)` (a permutation of file order) but passed
    `groups = sample_ids_all` (file order). Only 6.85% of positions matched -- and the lengths
    were equal, so a length-only check (what v2 shipped, audit V-01) passes on the bug it was
    written to catch. `rows` must therefore be the per-row key vector, not a row count.
    """
    if len(groups) != len(rows):
        raise AssertionError(
            f"group vector has {len(groups)} entries but {feature_matrix_name} has {len(rows)} rows. "
            "Carry groups through the SAME permutation as the features."
        )
    mismatched = [i for i, (a, b) in enumerate(zip(rows, groups)) if a != b]
    if mismatched:
        raise AssertionError(
            f"group vector disagrees with {feature_matrix_name} at {len(mismatched)}/{len(rows)} "
            f"positions (first at index {mismatched[0]}). "
            "Carry groups through the SAME permutation as the features."
        )
