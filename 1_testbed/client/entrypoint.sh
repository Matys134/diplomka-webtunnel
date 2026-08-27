#!/bin/bash
set -e

# Update trusted CA certificates
if [ -f /usr/local/share/ca-certificates/ca.crt ]; then
    cp /usr/local/share/ca-certificates/ca.crt /etc/ssl/certs/webtunnel-ca.pem
    update-ca-certificates --fresh > /dev/null 2>&1 || true
fi

# Ensure host resolution
echo "172.20.0.10 webtunnel.local" >> /etc/hosts 2>/dev/null || true

mkdir -p /var/lib/tor
chown -R debian-tor:debian-tor /var/lib/tor
chmod 700 /var/lib/tor

echo "[Client] Starting Tor client with WebTunnel transport..."
su -s /bin/bash debian-tor -c "tor -f /etc/tor/torrc" &
TOR_PID=$!

echo "[Client] Tor client PID: $TOR_PID. Waiting for SOCKS5 proxy on 127.0.0.1:9050..."
for i in $(seq 1 30); do
    if nc -z 127.0.0.1 9050 2>/dev/null; then
        echo "[Client] SOCKS5 proxy is ready!"
        break
    fi
    sleep 1
done

if [ "$#" -gt 0 ]; then
    exec "$@"
else
    # Keep container running
    wait $TOR_PID
fi
