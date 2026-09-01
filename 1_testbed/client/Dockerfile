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
    psmisc \
    tcpdump \
    iproute2 \
    ethtool \
    binutils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /webtunnel-client   /usr/local/bin/webtunnel-client
COPY --from=builder /traffic-generator  /usr/local/bin/traffic-generator
COPY torrc.tmpl             /etc/tor/torrc.tmpl
COPY entrypoint.sh          /entrypoint.sh
COPY offload_off.sh         /usr/local/bin/offload_off.sh
COPY stop_tor.sh            /usr/local/bin/stop_tor.sh
COPY start_tor.sh           /usr/local/bin/start_tor.sh
COPY probe_utls_support.sh  /usr/local/bin/probe_utls_support.sh
RUN chmod +x /entrypoint.sh \
             /usr/local/bin/webtunnel-client \
             /usr/local/bin/traffic-generator \
             /usr/local/bin/offload_off.sh \
             /usr/local/bin/stop_tor.sh \
             /usr/local/bin/start_tor.sh \
             /usr/local/bin/probe_utls_support.sh

EXPOSE 9050
ENTRYPOINT ["/entrypoint.sh"]
