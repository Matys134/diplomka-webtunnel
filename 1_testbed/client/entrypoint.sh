#!/bin/bash
set -e

BRIDGE_ADDR="${BRIDGE_ADDR:-172.20.0.10:443}"
BRIDGE_URL="${BRIDGE_URL:-https://webtunnel.local/secret-path-wt-research-2026}"
# P0.3: uTLS imitation for the pluggable transport itself.  Empty by default because an
# argument the PT does not understand makes Tor fail to bootstrap -- run
# ./probe_utls_support.sh first, then set WEBTUNNEL_EXTRA_ARGS=" utls-imitate=hellochrome_auto"
# (note the leading space) in docker-compose.  The collector records the ClientHello that
# actually appears on the wire, so the manifest never claims parity it did not get.
BRIDGE_EXTRA="${WEBTUNNEL_EXTRA_ARGS:-}"

sed -e "s|__BRIDGE_ADDR__|${BRIDGE_ADDR}|" \
    -e "s|__BRIDGE_URL__|${BRIDGE_URL}|" \
    -e "s|__BRIDGE_EXTRA__|${BRIDGE_EXTRA}|" \
    /etc/tor/torrc.tmpl > /etc/tor/torrc

if [ -f /usr/local/share/ca-certificates/ca.crt ]; then
    cp /usr/local/share/ca-certificates/ca.crt /etc/ssl/certs/webtunnel-ca.pem
    update-ca-certificates --fresh > /dev/null 2>&1 || true
fi

echo "172.20.0.10 webtunnel.local" >> /etc/hosts 2>/dev/null || true
echo "172.20.0.10 decoy.local"     >> /etc/hosts 2>/dev/null || true

mkdir -p /var/lib/tor
chown -R debian-tor:debian-tor /var/lib/tor
chmod 700 /var/lib/tor

# P0.6: disable segmentation offload on the capture interface and record the result.
# v2.0 ran this with `2>/dev/null || true` and never checked, so 72.7% of video_streaming
# payload packets still exceeded 1500 B (F-05 unresolved).
/usr/local/bin/offload_off.sh || echo "[Client] WARNING: offload disable failed"

# The collector drives tor explicitly (stop_tor.sh / start_tor.sh) so that every WebTunnel
# capture gets a genuinely fresh bridge socket.  Do NOT start tor here.
echo "[Client] Ready. Tor is managed per-capture by the collector."

if [ "$#" -gt 0 ]; then
    exec "$@"
else
    tail -f /dev/null
fi
