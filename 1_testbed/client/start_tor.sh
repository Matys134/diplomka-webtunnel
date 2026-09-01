#!/bin/bash
# P0.1 / P0.2 -- start Tor, wait for bootstrap, then report the REAL bridge 5-tuple.
#
# Ordering matters and is enforced by the collector:
#     stop_tor.sh  ->  tcpdump starts  ->  start_tor.sh  ->  generator  ->  tcpdump stops
# so the client SYN that opens the bridge connection lands INSIDE the capture window.
#
# Exactly one connection to the bridge must exist after bootstrap.  Anything else is
# ambiguous provenance and the capture is refused rather than guessed at (V-02).
BRIDGE_IP="${BRIDGE_IP:-172.20.0.10}"
BRIDGE_PORT="${BRIDGE_PORT:-443}"
TIMEOUT="${TOR_BOOTSTRAP_TIMEOUT:-90}"
LOG=/var/lib/tor/notice.log

: > "$LOG" 2>/dev/null || true
chown debian-tor:debian-tor "$LOG" 2>/dev/null || true

su -s /bin/bash debian-tor -c "tor -f /etc/tor/torrc" >/dev/null 2>&1 &

deadline=$(( $(date +%s) + TIMEOUT ))
booted=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    if grep -q "Bootstrapped 100%" "$LOG" 2>/dev/null; then booted=1; break; fi
    if grep -qE "Bootstrapped .*(FAILED|failed)" "$LOG" 2>/dev/null; then break; fi
    sleep 0.25
done

if [ "$booted" -ne 1 ]; then
    echo '{"ok": false, "drop_reason": "tor_bootstrap_timeout"}'
    exit 1
fi

# Give the PT a moment to settle, then snapshot the socket.
for _ in $(seq 1 20); do
    mapfile -t rows < <(ss -Htn state established "dst ${BRIDGE_IP}" "dport = :${BRIDGE_PORT}" 2>/dev/null \
                        | awk '{print $3}' | grep -E '^[0-9.]+:[0-9]+$')
    [ "${#rows[@]}" -ge 1 ] && break
    sleep 0.25
done

if [ "${#rows[@]}" -eq 0 ]; then
    echo '{"ok": false, "drop_reason": "no_bridge_socket"}'
    exit 1
fi
if [ "${#rows[@]}" -gt 1 ]; then
    printf '{"ok": false, "drop_reason": "ambiguous_bridge_socket", "n_sockets": %d}\n' "${#rows[@]}"
    exit 1
fi

client_ip="${rows[0]%:*}"
client_port="${rows[0]##*:}"
printf '{"ok": true, "client_ip": "%s", "client_port": %s, "server_ip": "%s", "server_port": %s}\n' \
       "$client_ip" "$client_port" "$BRIDGE_IP" "$BRIDGE_PORT"
