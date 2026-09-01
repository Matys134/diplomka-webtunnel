#!/bin/bash
# P0.3 -- does THIS build of the webtunnel client accept a uTLS imitation argument?
# Run once, before the campaign.  Prints the SOCKS args the PT parses.
BIN=/usr/local/bin/webtunnel-client
echo "== strings ${BIN} | grep -i utls =="
strings "$BIN" 2>/dev/null | grep -iE '^utls|utls-imitate|hellochrome|HelloChrome' | sort -u
echo "== strings ${BIN} | grep -i 'unsupported|unknown' arg handling =="
strings "$BIN" 2>/dev/null | grep -iE 'unknown (SOCKS )?arg|unsupported' | sort -u
echo
echo "If 'utls-imitate' (or 'utls') appears above, set in docker-compose.yml:"
echo '    WEBTUNNEL_EXTRA_ARGS: " utls-imitate=hellochrome_auto"'
echo "then re-run a single capture and confirm gate G1 sees one ClientHello length."
