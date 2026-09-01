#!/bin/bash
set -e

mkdir -p /var/lib/tor
chown -R debian-tor:debian-tor /var/lib/tor
chmod 700 /var/lib/tor

echo "[WebTunnel Bridge] Starting Tor daemon with WebTunnel plugin..."
su -s /bin/bash debian-tor -c "tor -f /etc/tor/torrc" &
TOR_PID=$!

# FIX B-3: publish the onion address where the collector can read it. The first boot generates
# the key; every later boot reuses it because /var/lib/tor is a named volume.
for _ in $(seq 1 120); do
    if [ -s /var/lib/tor/onion_decoy/hostname ]; then
        echo "[WebTunnel Bridge] decoy onion service: $(cat /var/lib/tor/onion_decoy/hostname)"
        break
    fi
    sleep 1
done

wait $TOR_PID
