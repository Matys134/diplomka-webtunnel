FROM golang:1.23-bookworm AS builder
WORKDIR /src
RUN git clone --depth 1 https://gitlab.torproject.org/tpo/anti-censorship/pluggable-transports/webtunnel.git . && \
    cd main/server && CGO_ENABLED=0 go build -ldflags "-s -w" -o /webtunnel-server && \
    cd ../client && CGO_ENABLED=0 go build -ldflags "-s -w" -o /webtunnel-client

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tor \
    ca-certificates \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /webtunnel-server /usr/local/bin/webtunnel-server
COPY --from=builder /webtunnel-client /usr/local/bin/webtunnel-client
COPY torrc /etc/tor/torrc
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /usr/local/bin/webtunnel-server /usr/local/bin/webtunnel-client

EXPOSE 15000 9001
ENTRYPOINT ["/entrypoint.sh"]
