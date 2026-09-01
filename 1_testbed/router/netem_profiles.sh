#!/bin/bash
# F-07 -- network emulation applied in BOTH directions with a real rate ceiling.
#
# v1/v2.0 ran `tc qdisc ... dev eth0 root` only, which shapes EGRESS.  Downstream loss -- the
# direction carrying video and asset traffic -- was never emulated, and there was no rate limit,
# so "4G/LTE" had unlimited bandwidth and no bufferbloat.  Ingress is now redirected through an
# ifb device so the same netem applies to inbound traffic, and htb enforces a rate ceiling.
#
# The applied parameters are ALSO the ones common/config.py prints, generated from
# `netem_params()` in the collector -- no more hand-written display strings that drift.
set -u
IFACE="${CAPTURE_IFACE:-eth0}"
IFB="ifb0"
PROFILE="${1:-reset}"

apply() {
    local delay="$1" jitter="$2" dist="$3" lossspec="$4" rate="$5" extra="${6:-}"

    tc qdisc del dev "$IFACE" root        2>/dev/null || true
    tc qdisc del dev "$IFACE" ingress     2>/dev/null || true
    tc qdisc del dev "$IFB"   root        2>/dev/null || true

    modprobe ifb numifbs=1 2>/dev/null || true
    ip link add "$IFB" type ifb 2>/dev/null || true
    ip link set dev "$IFB" up 2>/dev/null || true

    # ---- egress: rate ceiling, then netem ----
    tc qdisc add dev "$IFACE" root handle 1: htb default 10
    tc class add dev "$IFACE" parent 1: classid 1:10 htb rate "$rate" ceil "$rate" burst 32k
    # shellcheck disable=SC2086
    tc qdisc add dev "$IFACE" parent 1:10 handle 11: netem \
        delay "$delay" "$jitter" distribution "$dist" $lossspec $extra

    # ---- ingress: mirror everything to ifb0 and shape it identically ----
    tc qdisc add dev "$IFACE" handle ffff: ingress
    tc filter add dev "$IFACE" parent ffff: protocol all u32 match u32 0 0 \
        action mirred egress redirect dev "$IFB"
    tc qdisc add dev "$IFB" root handle 1: htb default 10
    tc class add dev "$IFB" parent 1: classid 1:10 htb rate "$rate" ceil "$rate" burst 32k
    # shellcheck disable=SC2086
    tc qdisc add dev "$IFB" parent 1:10 handle 11: netem \
        delay "$delay" "$jitter" distribution "$dist" $lossspec $extra
}

case "$PROFILE" in
    broadband)
        echo "[tc-netem] broadband: 20ms +/- 4ms normal, 0.05% loss, 200mbit, ingress+egress"
        apply 20ms 4ms normal "loss random 0.05%" 200mbit "duplicate 0.02%"
        ;;
    lte)
        echo "[tc-netem] lte: 45ms +/- 15ms paretonormal, 0.2% loss, 40mbit, ingress+egress"
        apply 45ms 15ms paretonormal "loss random 0.2%" 40mbit "reorder 0.5% 25%"
        ;;
    lossy)
        echo "[tc-netem] lossy: 90ms +/- 25ms paretonormal, Gilbert-Elliot burst loss, 8mbit"
        apply 90ms 25ms paretonormal "loss state 0.02 0.30 0.01 0.10" 8mbit
        ;;
    reset|none)
        echo "[tc-netem] resetting"
        tc qdisc del dev "$IFACE" root    2>/dev/null || true
        tc qdisc del dev "$IFACE" ingress 2>/dev/null || true
        tc qdisc del dev "$IFB"   root    2>/dev/null || true
        ip link set dev "$IFB" down       2>/dev/null || true
        ;;
    status)
        tc qdisc show dev "$IFACE"; tc qdisc show dev "$IFB" 2>/dev/null
        exit 0
        ;;
    *)
        echo "Usage: $0 {broadband|lte|lossy|reset|status}"; exit 1 ;;
esac

echo "[tc-netem] egress:"; tc qdisc show dev "$IFACE"
echo "[tc-netem] ingress(ifb):"; tc qdisc show dev "$IFB" 2>/dev/null || true
