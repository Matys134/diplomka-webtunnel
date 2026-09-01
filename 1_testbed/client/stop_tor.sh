#!/bin/bash
# P0.1 -- tear the Tor client down completely so the next capture opens a NEW bridge socket.
# `pkill -HUP tor` (what v2.0 did) only reloads the config; it never closes an OR connection,
# which is why one client port appeared in 234 of 336 WebTunnel captures.
BRIDGE_IP="${BRIDGE_IP:-172.20.0.10}"
BRIDGE_PORT="${BRIDGE_PORT:-443}"

pkill -x tor              2>/dev/null || true
pkill -f webtunnel-client 2>/dev/null || true

for _ in $(seq 1 40); do
    n=$(ss -Htn state established "dst ${BRIDGE_IP}" "dport = :${BRIDGE_PORT}" 2>/dev/null | wc -l)
    p=$(pgrep -x tor | wc -l)
    if [ "$n" -eq 0 ] && [ "$p" -eq 0 ]; then
        echo '{"stopped": true}'
        exit 0
    fi
    sleep 0.25
done
pkill -9 -x tor 2>/dev/null || true
pkill -9 -f webtunnel-client 2>/dev/null || true
sleep 0.5
n=$(ss -Htn state established "dst ${BRIDGE_IP}" "dport = :${BRIDGE_PORT}" 2>/dev/null | wc -l)
echo "{\"stopped\": $([ "$n" -eq 0 ] && echo true || echo false), \"lingering\": ${n}}"
