#!/bin/bash
set -e

mkdir -p /var/lib/tor
chown -R debian-tor:debian-tor /var/lib/tor
chmod 700 /var/lib/tor

echo "[WebTunnel Bridge] Starting Tor daemon with WebTunnel plugin..."
exec su -s /bin/bash debian-tor -c "tor -f /etc/tor/torrc"
