#!/bin/bash
IFACE="eth0"
PROFILE="${1:-reset}"

case "$PROFILE" in
    broadband)
        echo "[tc-netem] Applying Broadband Profile (Fiber/VDSL: 20ms +/- 4ms, 0.05% loss)..."
        tc qdisc replace dev ${IFACE} root handle 1: netem delay 20ms 4ms distribution normal loss random 0.05% duplicate 0.02%
        ;;
    lte)
        echo "[tc-netem] Applying 4G/LTE Profile (45ms +/- 15ms paretonormal, 0.2% loss, reordering)..."
        tc qdisc replace dev ${IFACE} root handle 1: netem delay 45ms 15ms distribution paretonormal loss random 0.2% reorder 0.5% 25%
        ;;
    lossy)
        echo "[tc-netem] Applying Lossy/Congested Profile (Gilbert-Elliot burst loss)..."
        tc qdisc replace dev ${IFACE} root handle 1: netem delay 90ms 25ms distribution paretonormal loss state 0.02 0.30 0.01 0.10
        ;;
    reset|none)
        echo "[tc-netem] Resetting network emulation rules on ${IFACE}..."
        tc qdisc del dev ${IFACE} root 2>/dev/null || true
        ;;
    status)
        tc qdisc show dev ${IFACE}
        ;;
    *)
        echo "Usage: $0 {broadband|lte|lossy|reset|status}"
        exit 1
        ;;
esac

echo "[tc-netem] Current qdisc status:"
tc qdisc show dev ${IFACE}
