"""
v2.1 flow builder.

What changed, and why (docs/04-v2-audit.md):

P0.7  REAL TCP desegmentation.  v2.0 parsed TLS records out of individual TCP segments and, when
      a segment did not start on a record boundary, fell back to `min(pkt_len, 1500)`.  With
      segmentation offload still on, that clamp made `down_len_p90` exactly 1500.0 for the bulk
      classes -- a capture artefact that v2.0 then registered as a Tor protocol invariant.
      There is now a per-direction byte-stream reassembler; records are parsed out of the
      reassembled stream; there is NO clamp and no fallback.  A byte that cannot be attributed
      to a complete TLS record is dropped and counted, never guessed at.

P0.8  REAL handshake cutoff.  In TLS 1.3 the client's Finished travels in the FIRST client
      application_data (0x17) record.  `hs_end_idx` is therefore the index of the SECOND client
      0x17 record.  v2.0 used `if d == 1 and idx > 3` -- a fixed index that ignored content type.

P0.2  STRICT demultiplexing.  The manifest's 5-tuple must name a concrete client port and it is
      matched exactly, in both directions.  The `sport == 0` wildcard (which merged >=2 sockets
      into one "flow" in 81.8% of v2.0 WebTunnel captures) and the hardcoded `172.20.0.30` /
      `startswith("172.20.0.3")` direction test are both gone.  Direction comes from the
      manifest's client_ip and nothing else.

G1    ClientHello is parsed properly: record length, cipher list, extension list, ALPN and a
      JA4 fingerprint, so stack parity is an assertion instead of a wish.  (v2.0 never populated
      `ja4`, so G1's JA4 clause was `len(empty_set) <= 1` -- always true.  It also had an
      operator-precedence bug in the length computation, V-08.)
"""

from __future__ import annotations

import bisect
import hashlib
import os
import socket
from typing import Any, Dict, List, Optional, Tuple

import dpkt
import numpy as np
from scipy import stats

from common.contracts import CaptureManifest, FiveTuple, FlowRecord, on_tor_lattice, lattice_k

# TLS ContentTypes
CT_CCS, CT_ALERT, CT_HANDSHAKE, CT_APPDATA = 20, 21, 22, 23
VALID_CT = (CT_CCS, CT_ALERT, CT_HANDSHAKE, CT_APPDATA)
VALID_VER = (0x0301, 0x0302, 0x0303, 0x0304)
MAX_TLS_RECORD = 16384 + 256          # RFC 8446 ceiling plus expansion allowance

FEATURE_NAMES = [
    # 0-9: all record lengths
    "len_min", "len_max", "len_mean", "len_std", "len_skew",
    "len_p10", "len_p25", "len_p50", "len_p75", "len_p90",
    # 10-18: upstream
    "up_len_min", "up_len_max", "up_len_mean", "up_len_std",
    "up_len_p10", "up_len_p25", "up_len_p50", "up_len_p75", "up_len_p90",
    # 19-27: downstream
    "down_len_min", "down_len_max", "down_len_mean", "down_len_std",
    "down_len_p10", "down_len_p25", "down_len_p50", "down_len_p75", "down_len_p90",
    # 28-36: inter-arrival times
    "iat_min", "iat_max", "iat_mean", "iat_std",
    "iat_p10", "iat_p25", "iat_p50", "iat_p75", "iat_p90",
    # 37-43: burst dynamics
    "burst_count", "burst_len_mean_pkts", "burst_len_std_pkts",
    "burst_len_mean_bytes", "burst_len_std_bytes", "burst_dur_mean", "burst_dur_std",
    # 44-47: ratios and totals
    "ratio_up_pkts", "ratio_up_bytes", "total_pkts", "total_bytes",
    # 48-49: the Tor cell lattice -- the one registered invariant (see checks/expected_invariants)
    "up_lattice_frac", "down_lattice_frac",
]

# RFC 8701 GREASE values are 0x0a0a, 0x1a1a, ... 0xfafa -- step 0x1010.  Getting this wrong
# leaves the per-connection-random GREASE values in the cipher/extension lists, which makes
# every JA4 unique and gate G1 permanently unsatisfiable.
GREASE = frozenset(range(0x0a0a, 0xfafa + 1, 0x1010))


# ---------------------------------------------------------------------------
# TCP stream reassembly (P0.7)
# ---------------------------------------------------------------------------

class DirectionalStream:
    """Reassembles one direction of a TCP connection into a contiguous byte stream.

    Keeps a sparse map from stream offset -> capture timestamp so every reassembled TLS record
    can be stamped with the arrival time of its LAST byte, which is when the record is actually
    complete on the wire.
    """

    def __init__(self) -> None:
        self.isn: Optional[int] = None
        self.next_off = 0
        self.buf = bytearray()
        self.pending: Dict[int, bytes] = {}
        self._mark_ends: List[int] = []      # sorted stream offsets (exclusive end of a segment)
        self._mark_ts: List[float] = []
        self.retransmits = 0
        self.gaps = 0

    def set_isn(self, seq: int) -> None:
        if self.isn is None:
            self.isn = seq

    def _offset(self, seq: int) -> int:
        return (seq - self.isn) & 0xFFFFFFFF

    def add(self, seq: int, ts: float, data: bytes) -> None:
        if not data:
            return
        self.set_isn(seq)
        off = self._offset(seq)
        if off > 0x7FFFFFFF:                 # sequence number before the ISN -> ignore
            return
        end = off + len(data)
        if end <= self.next_off:             # pure retransmission
            self.retransmits += 1
            return
        if off < self.next_off:              # partial overlap -> keep the new bytes only
            data = data[self.next_off - off:]
            off = self.next_off
        self.pending[off] = data if off not in self.pending else max(self.pending[off], data, key=len)
        self._drain(ts)

    def _drain(self, ts: float) -> None:
        while self.next_off in self.pending:
            chunk = self.pending.pop(self.next_off)
            self.buf.extend(chunk)
            self.next_off += len(chunk)
            self._mark_ends.append(self.next_off)
            self._mark_ts.append(ts)

    def finalize(self) -> None:
        """Flush what can still be flushed; count anything left as a gap."""
        self.gaps = len(self.pending)
        self.pending.clear()

    def ts_at(self, offset: int) -> float:
        i = bisect.bisect_left(self._mark_ends, offset + 1)
        if i >= len(self._mark_ts):
            i = len(self._mark_ts) - 1
        return self._mark_ts[i] if i >= 0 else 0.0

    def records(self) -> List[Tuple[float, int, int]]:
        """(timestamp, content_type, record_length_including_header) for every COMPLETE record.

        No clamping, no fallback: a trailing partial record is simply not emitted.
        """
        out: List[Tuple[float, int, int]] = []
        b, n, i = self.buf, len(self.buf), 0
        while i + 5 <= n:
            ctype = b[i]
            ver = (b[i + 1] << 8) | b[i + 2]
            if ctype not in VALID_CT or ver not in VALID_VER:
                break                        # desynchronised -- stop rather than invent records
            rlen = (b[i + 3] << 8) | b[i + 4]
            if rlen == 0 or rlen > MAX_TLS_RECORD:
                break
            end = i + 5 + rlen
            if end > n:
                break                        # incomplete tail
            out.append((self.ts_at(end - 1), ctype, rlen + 5))
            i = end
        self.trailing_bytes = n - i
        return out


# ---------------------------------------------------------------------------
# ClientHello / JA4 (G1)
# ---------------------------------------------------------------------------

def _u16(b: bytes, i: int) -> int:
    return (b[i] << 8) | b[i + 1]


def parse_client_hello(stream: bytes) -> Dict[str, Any]:
    """Parse the ClientHello out of the reassembled client stream.

    Returns {} when there is none.  Computes a JA4 fingerprint following the published JA4
    construction (ja4_a _ ja4_b _ ja4_c) so gate G1 can assert one fingerprint across classes.
    """
    if len(stream) < 9 or stream[0] != CT_HANDSHAKE:
        return {}
    rec_len = _u16(stream, 3) + 5
    body = stream[5:rec_len]
    if len(body) < 4 or body[0] != 0x01:
        return {}

    i = 4 + 2 + 32                            # msg header + legacy_version + random
    if i >= len(body):
        return {}
    sid_len = body[i]; i += 1 + sid_len
    if i + 2 > len(body):
        return {}
    cs_len = _u16(body, i); i += 2
    ciphers = [_u16(body, i + k) for k in range(0, cs_len, 2)]
    i += cs_len
    if i >= len(body):
        return {}
    comp_len = body[i]; i += 1 + comp_len

    exts: List[int] = []
    alpn: List[str] = []
    sig_algs: List[int] = []
    sni_present = False
    tls_version = _u16(stream, 1)

    if i + 2 <= len(body):
        ext_total = _u16(body, i); i += 2
        stop = min(len(body), i + ext_total)
        while i + 4 <= stop:
            eid = _u16(body, i); elen = _u16(body, i + 2); i += 4
            edata = body[i:i + elen]; i += elen
            exts.append(eid)
            if eid == 0x0000:
                sni_present = True
            elif eid == 0x0010 and len(edata) >= 2:      # ALPN
                j, lim = 2, min(len(edata), 2 + _u16(edata, 0))
                while j < lim:
                    ln = edata[j]; j += 1
                    alpn.append(edata[j:j + ln].decode("ascii", "replace")); j += ln
            elif eid == 0x002b and len(edata) >= 1:      # supported_versions
                vs = [_u16(edata, 1 + k) for k in range(0, edata[0], 2)]
                vs = [v for v in vs if v not in GREASE]
                if vs:
                    tls_version = max(vs)
            elif eid == 0x000d and len(edata) >= 2:      # signature_algorithms
                sig_algs = [_u16(edata, 2 + k) for k in range(0, _u16(edata, 0), 2)]

    real_ciphers = sorted(c for c in ciphers if c not in GREASE)
    real_exts = sorted(e for e in exts if e not in GREASE)
    ja4_exts = [e for e in real_exts if e not in (0x0000, 0x0010)]

    ver_code = {0x0304: "13", 0x0303: "12", 0x0302: "11", 0x0301: "10"}.get(tls_version, "00")
    first_alpn = alpn[0] if alpn else "00"
    alpn_code = (first_alpn[0] + first_alpn[-1]) if len(first_alpn) >= 2 else "00"
    ja4_a = f"t{ver_code}{'d' if sni_present else 'i'}{len(real_ciphers):02d}{len(ja4_exts):02d}{alpn_code}"
    ja4_b = hashlib.sha256(",".join(f"{c:04x}" for c in real_ciphers).encode()).hexdigest()[:12]
    ja4_c = hashlib.sha256(
        (",".join(f"{e:04x}" for e in ja4_exts) + "_" +
         ",".join(f"{s:04x}" for s in sig_algs)).encode()).hexdigest()[:12]

    return {
        "clienthello_len": rec_len,
        "ja4": f"{ja4_a}_{ja4_b}_{ja4_c}",
        "tls_extensions": real_exts,
        "alpn_offered": alpn,
        "n_ciphers": len(real_ciphers),
    }


# ---------------------------------------------------------------------------
# Flow extraction
# ---------------------------------------------------------------------------

def _ip_str(ip_obj, raw: bytes) -> str:
    return (socket.inet_ntoa(raw) if isinstance(ip_obj, dpkt.ip.IP)
            else socket.inet_ntop(socket.AF_INET6, raw))


def extract_flow_from_pcap(pcap_path: str,
                           manifest: Optional[CaptureManifest] = None,
                           post_handshake_only: bool = False
                           ) -> Tuple[Optional[FlowRecord], str]:
    """Build one FlowRecord from one capture. Returns (flow, drop_reason).

    A manifest is REQUIRED.  Without authoritative ground truth there is nothing to demultiplex
    against, and inferring it is what produced F-04.
    """
    if manifest is None:
        return None, "no_manifest"
    bad = manifest.validate()
    if bad:
        return None, bad
    if not os.path.exists(pcap_path):
        return None, "pcap_missing"

    t_src, t_sport, t_dst, t_dport, t_proto = manifest.target_5tuple  # type: ignore[misc]
    t_sport, t_dport = int(t_sport), int(t_dport)
    client_ip = manifest.client_ip

    up, down = DirectionalStream(), DirectionalStream()
    first_ts: Optional[float] = None
    syn_ts: Optional[float] = None
    saw_client_syn = False
    n_matched = 0

    try:
        fh = open(pcap_path, "rb")
    except OSError:
        return None, "pcap_unreadable"
    with fh:
        try:
            pcap = dpkt.pcap.Reader(fh)
        except Exception:
            fh.seek(0)
            try:
                pcap = dpkt.pcapng.Reader(fh)
            except Exception:
                return None, "pcap_unparseable"

        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            ip = eth.data
            if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                continue
            tcp = ip.data
            if not isinstance(tcp, dpkt.tcp.TCP):
                continue

            src = _ip_str(ip, ip.src)
            dst = _ip_str(ip, ip.dst)

            # Strict, exact, bidirectional 5-tuple match.  No wildcards (P0.2).
            if src == t_src and tcp.sport == t_sport and dst == t_dst and tcp.dport == t_dport:
                direction = 1
            elif src == t_dst and tcp.sport == t_dport and dst == t_src and tcp.dport == t_sport:
                direction = -1
            else:
                continue

            n_matched += 1
            if first_ts is None:
                first_ts = ts

            is_syn = bool(tcp.flags & dpkt.tcp.TH_SYN)
            is_ack = bool(tcp.flags & dpkt.tcp.TH_ACK)
            if is_syn and not is_ack and direction == 1:
                saw_client_syn = True
                if syn_ts is None:
                    syn_ts = ts
                up.set_isn((tcp.seq + 1) & 0xFFFFFFFF)
            elif is_syn and is_ack and direction == -1:
                down.set_isn((tcp.seq + 1) & 0xFFFFFFFF)

            payload = bytes(tcp.data)
            if payload:
                (up if direction == 1 else down).add(tcp.seq, ts, payload)

    if n_matched == 0:
        return None, "no_packets_for_target_5tuple"

    # Direction sanity: the manifest says who the client is; assert it agrees with the tuple.
    if client_ip != t_src:
        return None, "client_ip_disagrees_with_5tuple"

    up.finalize(); down.finalize()
    up_recs = up.records()
    down_recs = down.records()
    if not up_recs and not down_recs:
        return None, "no_tls_records"

    merged = sorted([(t, 1, ct, ln) for (t, ct, ln) in up_recs] +
                    [(t, -1, ct, ln) for (t, ct, ln) in down_recs], key=lambda r: r[0])
    if len(merged) < 3:
        return None, "too_few_records"

    base_ts = syn_ts if syn_ts is not None else merged[0][0]
    records = [(t - base_ts, d, ln) for (t, d, ct, ln) in merged]
    ctypes = [ct for (_t, _d, ct, _l) in merged]

    # P0.8 -- TLS 1.3: the client's Finished is its FIRST application_data record, so the first
    # true application record is its SECOND.  Anything before that index is handshake.
    hs_end_idx = 0
    saw_full_handshake = False
    client_appdata_seen = 0
    for idx, (_t, d, ct, _l) in enumerate(merged):
        if d == 1 and ct == CT_APPDATA:
            client_appdata_seen += 1
            if client_appdata_seen == 2:
                hs_end_idx = idx
                saw_full_handshake = True
                break
    if not saw_full_handshake:
        # No complete client handshake in this capture: keep everything and say so, rather than
        # silently cutting at a made-up index (which is what v2.0 did).
        hs_end_idx = 0

    ch = parse_client_hello(bytes(up.buf))

    ft: FiveTuple = (t_src, t_sport, t_dst, t_dport, t_proto)
    if post_handshake_only:
        records = records[hs_end_idx:]
        ctypes = ctypes[hs_end_idx:]
        if len(records) < 3:
            return None, "too_few_records_post_handshake"
        hs_end_idx = 0

    n_up = sum(1 for _t, d, _l in records if d == 1)
    n_down = len(records) - n_up
    b_up = sum(l for _t, d, l in records if d == 1)
    b_down = sum(l for _t, d, l in records if d == -1)

    flow = FlowRecord(
        flow_uid=FlowRecord.make_uid(manifest.capture_id, ft, base_ts),
        conn_id=FlowRecord.make_conn_id(ft, base_ts),
        socket_id=FlowRecord.make_socket_id(ft),
        capture_id=manifest.capture_id,
        label=manifest.label,
        behaviour=manifest.behaviour,
        profile=manifest.profile,
        dest_id=manifest.dest_id,
        client_stack=manifest.client_stack,
        epoch=manifest.epoch,
        records=records,
        content_types=ctypes,
        hs_end_idx=hs_end_idx,
        saw_client_syn=saw_client_syn,
        saw_full_handshake=saw_full_handshake,
        n_up=n_up, n_down=n_down, bytes_up=b_up, bytes_down=b_down,
        duration=(records[-1][0] - records[0][0]) if len(records) > 1 else 0.0,
        clienthello_len=ch.get("clienthello_len"),
        ja4=ch.get("ja4"),
        tls_extensions=ch.get("tls_extensions"),
        alpn_offered=ch.get("alpn_offered"),
        budget_id=manifest.budget_id,
        target_duration_s=manifest.target_duration_s,
        target_bytes_up=manifest.target_bytes_up,
        target_bytes_down=manifest.target_bytes_down,
        provenance=manifest.provenance,
        t_start=manifest.t_start,
        git_commit=manifest.git_commit,
        generator_seed=manifest.generator_seed,
    )
    return flow, ""


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _stats10(arr: List[float]) -> List[float]:
    if not arr:
        return [0.0] * 10
    a = np.asarray(arr, dtype=np.float64)
    return [float(a.min()), float(a.max()), float(a.mean()), float(a.std()),
            float(stats.skew(a)) if len(a) > 2 and a.std() > 1e-6 else 0.0,
            *[float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90)]]


def _stats9(arr: List[float]) -> List[float]:
    if not arr:
        return [0.0] * 9
    a = np.asarray(arr, dtype=np.float64)
    return [float(a.min()), float(a.max()), float(a.mean()), float(a.std()),
            *[float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90)]]


def compute_flow_statistics(packets: List[Tuple[float, int]]) -> List[float]:
    """50 flow features from a list of (timestamp, signed TLS record length)."""
    times = [p[0] for p in packets]
    lengths = [abs(p[1]) for p in packets]
    up_l = [abs(p[1]) for p in packets if p[1] > 0]
    down_l = [abs(p[1]) for p in packets if p[1] < 0]

    iats = [times[i] - times[i - 1] for i in range(1, len(times))] or [0.0]

    bursts, cur_dir, cur_pkts, cur_bytes, t0 = [], None, 0, 0, 0.0
    for t, sl in packets:
        d = 1 if sl > 0 else -1
        if cur_dir is None:
            cur_dir, cur_pkts, cur_bytes, t0 = d, 1, abs(sl), t
        elif d == cur_dir:
            cur_pkts += 1
            cur_bytes += abs(sl)
        else:
            bursts.append((cur_pkts, cur_bytes, t - t0))
            cur_dir, cur_pkts, cur_bytes, t0 = d, 1, abs(sl), t
    if cur_dir is not None:
        bursts.append((cur_pkts, cur_bytes, times[-1] - t0))

    bp = [b[0] for b in bursts]; bb = [b[1] for b in bursts]; bd = [b[2] for b in bursts]
    burst_stats = [float(len(bursts)),
                   float(np.mean(bp)) if bp else 0.0, float(np.std(bp)) if bp else 0.0,
                   float(np.mean(bb)) if bb else 0.0, float(np.std(bb)) if bb else 0.0,
                   float(np.mean(bd)) if bd else 0.0, float(np.std(bd)) if bd else 0.0]

    tot_pkts = len(packets)
    tot_bytes = sum(lengths)
    totals = [float(len(up_l) / max(1, tot_pkts)), float(sum(up_l) / max(1.0, tot_bytes)),
              float(tot_pkts), float(tot_bytes)]

    lattice = [
        float(np.mean([on_tor_lattice(l) for l in up_l])) if up_l else 0.0,
        float(np.mean([on_tor_lattice(l) for l in down_l])) if down_l else 0.0,
    ]

    return _stats10(lengths) + _stats9(up_l) + _stats9(down_l) + _stats9(iats) + burst_stats + totals + lattice


def normalize_sequence_tensor(packets: List[Tuple[float, int]], max_seq_len: int = 200) -> np.ndarray:
    """[max_seq_len, 2] tensor of (signed length / 16640, log1p(IAT)/5).

    The divisor is the TLS record ceiling, not the MTU: WebTunnel's coalesced records legitimately
    exceed 1500 B and v2.0's /1500 normalisation saturated them.  The IAT channel is scaled by 5
    rather than 10 so it spans a comparable dynamic range to the size channel (F-16).
    """
    seq = np.zeros((max_seq_len, 2), dtype=np.float32)
    prev_t = 0.0
    for i, (t, sl) in enumerate(packets[:max_seq_len]):
        dt = max(0.0, t - prev_t)
        prev_t = t
        seq[i, 0] = float(np.clip(sl / float(MAX_TLS_RECORD), -1.0, 1.0))
        seq[i, 1] = float(min(1.0, np.log1p(dt) / 5.0))
    return seq


build_sequence_tensor = normalize_sequence_tensor


def extract_raw_packets_from_pcap(pcap_path: str,
                                  manifest: Optional[CaptureManifest] = None,
                                  post_handshake_only: bool = False) -> List[Tuple[float, int]]:
    """(t, signed_len) helper. A manifest is now MANDATORY -- passing None used to silently
    disable demultiplexing, which is how evaluate_cross_profile.py and evaluate_post_handshake.py
    ended up running on contaminated data (audit section 4.6)."""
    if manifest is None:
        raise ValueError(
            "extract_raw_packets_from_pcap requires a CaptureManifest. "
            "Load the .manifest.json sidecar; do not analyse a capture without ground truth."
        )
    flow, _reason = extract_flow_from_pcap(pcap_path, manifest=manifest,
                                           post_handshake_only=post_handshake_only)
    if flow is None:
        return []
    return [(t, d * l) for (t, d, l) in flow.records]


def load_manifest_for(pcap_path: str) -> Optional[CaptureManifest]:
    side = os.path.splitext(pcap_path)[0] + ".manifest.json"
    if not os.path.exists(side):
        return None
    try:
        with open(side, "r", encoding="utf-8") as f:
            return CaptureManifest.from_json(f.read())
    except Exception:
        return None
