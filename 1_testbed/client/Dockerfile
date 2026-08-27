FROM golang:1.23-bookworm AS builder
WORKDIR /src
RUN git clone --depth 1 https://gitlab.torproject.org/tpo/anti-censorship/pluggable-transports/webtunnel.git . && \
    sed -i 's/|| ip.IsPrivate()//g' main/client/main.go && \
    sed -i 's/|| ip.IsLoopback()//g' main/client/main.go && \
    cd main/client && CGO_ENABLED=0 go build -ldflags "-s -w" -o /webtunnel-client

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tor \
    ca-certificates \
    curl \
    netcat-traditional \
    procps \
    tcpdump \
    iproute2 \
    python3 \
    python3-pip \
    python3-requests \
    python3-socks \
    python3-websockets \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /webtunnel-client /usr/local/bin/webtunnel-client
COPY torrc /etc/tor/torrc
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /usr/local/bin/webtunnel-client

EXPOSE 9050
ENTRYPOINT ["/entrypoint.sh"]
