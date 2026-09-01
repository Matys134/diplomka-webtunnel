FROM golang:1.23-bookworm AS builder
WORKDIR /src
RUN git clone --depth 1 https://gitlab.torproject.org/tpo/anti-censorship/pluggable-transports/webtunnel.git . && \
    sed -i 's/|| ip.IsPrivate()//g' main/client/main.go && \
    sed -i 's/|| ip.IsLoopback()//g' main/client/main.go && \
    cd main/client && CGO_ENABLED=0 go build -ldflags "-s -w" -o /webtunnel-client

WORKDIR /src/generator
COPY generator/go.mod ./
COPY generator/main.go ./
RUN go mod tidy && CGO_ENABLED=0 go build -ldflags "-s -w" -o /traffic-generator .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tor \
    ca-certificates \
    curl \
    netcat-traditional \
    procps \
    tcpdump \
    iproute2 \
    ethtool \
    python3 \
    python3-pip \
    python3-requests \
    python3-socks \
    python3-websockets \
    python3-httpx \
    python3-h2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /webtunnel-client /usr/local/bin/webtunnel-client
COPY --from=builder /traffic-generator /usr/local/bin/traffic-generator
COPY torrc /etc/tor/torrc
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /usr/local/bin/webtunnel-client /usr/local/bin/traffic-generator

EXPOSE 9050
ENTRYPOINT ["/entrypoint.sh"]
