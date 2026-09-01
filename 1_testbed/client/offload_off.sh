#!/bin/bash
# P0.6 -- turn segmentation offload off on the capture interface and PROVE it.
# Exit 0 only when tso/gso/gro all read "off".
IFACE="${CAPTURE_IFACE:-eth0}"
ethtool -K "$IFACE" tso off gso off gro off lro off >/dev/null 2>&1 || true
bad=""
for f in tcp-segmentation-offload generic-segmentation-offload generic-receive-offload; do
    state=$(ethtool -k "$IFACE" 2>/dev/null | grep -E "^${f}:" | awk '{print $2}')
    [ "$state" = "off" ] || bad="${bad} ${f}=${state:-unknown}"
done
mss=$(cat /sys/class/net/"$IFACE"/mtu 2>/dev/null)
mss=$(( ${mss:-1500} - 40 ))
if [ -n "$bad" ]; then
    echo "{\"offloads_disabled\": false, \"mss\": ${mss}, \"still_on\":\"${bad# }\"}"
    exit 1
fi
echo "{\"offloads_disabled\": true, \"mss\": ${mss}}"
