// WebTunnel testbed traffic generator (v2.1).
//
// One binary generates EVERY class, over ONE TLS stack (uTLS HelloChrome_Auto) offering ONE
// ALPN list, so the ClientHello is byte-identical across classes by construction (gate G1).
//
// Changes forced by the second-pass audit (docs/04-v2-audit.md):
//   P0.4  ALPN parity  -- every class offers {"h2","http/1.1"}; the SERVER chooses.  v2.0 offered
//         {"http/1.1"} for the WebSocket classes, which shifted the ClientHello by 3 bytes and
//         leaked the transport (V-07).
//   P0.5  No upstream record ceiling -- v2.0 capped negative upstream payloads at 400+rand(400)
//         and 350+rand(400), so no negative could emit a TLS record above 830 B while WebTunnel
//         coalesces to 1072 B and beyond.  `up_len_max > 951` was therefore a perfect classifier
//         and it was measuring main.go, not Tor.  Payloads now draw from a heavy-tailed mixture
//         that spans 100 B to 16 KB for every class.
//   P0.2  No fabricated 5-tuple -- the WebTunnel path runs through Tor's SOCKS proxy, so this
//         process cannot see the bridge socket.  It now reports NOTHING rather than the
//         hardcoded 172.20.0.3 / port 0 of v2.0 (V-02).  The collector snapshots the real socket.
//   V-05  --behaviour is a real flag that changes cadence and payload mix for every class,
//         including the positive one.  v2.0 recorded a behaviour label that had no effect.
//
// v2.2 fixes (docs/05-final-review.md section 6):
//   B-2   ALPN parity restored -- runWebSocket had been edited back to offering only
//         {"http/1.1"}, which split the negatives into two JA4 fingerprints and shifted their
//         ClientHello by 3 bytes.  And newResult() reported the ALPNParity CONSTANT rather than
//         what was actually offered, so all 2,051 manifests claimed an ALPN the wire contradicted.
//         Both the offer and the negotiated protocol now come from the connection itself.
//   B-3   No public targets.  The WebTunnel class fetched duckduckgo.com / check.torproject.org /
//         en.wikipedia.org while every negative fetched the local mock, confounding content, RTT
//         and volume with the class by destination (duration 5.18 s vs 2.45 s, bytes_up 55,566 vs
//         9-22k, gate G4 at p ~ 1e-48).  The target is now a mandatory local onion address and
//         there is NO silent fallback -- an unset target is a hard error, because the silent
//         fallback is exactly what produced the defect.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	mrand "math/rand"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/websocket"
	utls "github.com/refraction-networking/utls"
	"golang.org/x/net/http2"
	"golang.org/x/net/proxy"
)

// ALPNParity is the single ALPN list every class offers.  Keep in sync with
// common/contracts.py :: ALPN_PARITY -- gate G1 asserts they agree.
var ALPNParity = []string{"h2", "http/1.1"}

type GeneratorResult struct {
	ClientIP    string  `json:"client_ip"`
	ClientPort  int     `json:"client_port"`
	ServerIP    string  `json:"server_ip"`
	ServerPort  int     `json:"server_port"`
	Protocol    string  `json:"protocol"`
	ClientStack string  `json:"client_stack"`
	ALPNOffered []string `json:"alpn_offered"`
	ALPNPicked  string  `json:"alpn_picked"`
	Behaviour   string  `json:"behaviour"`
	BytesUp     int     `json:"bytes_up"`
	BytesDown   int     `json:"bytes_down"`
	DurationS   float64 `json:"duration_s"`
	// TupleKnown is false for the WebTunnel class: this process dials a SOCKS proxy, so the
	// bridge socket belongs to the Tor daemon and must be recorded by the collector.
	TupleKnown bool   `json:"tuple_known"`
	OK         bool   `json:"ok"`
	Error      string `json:"error,omitempty"`
}

var userAgents = []string{
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
	"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}

// ---------------------------------------------------------------------------
// Behaviour profiles -- applied to EVERY class, positive included (V-05)
// ---------------------------------------------------------------------------

type behaviour struct {
	name       string
	thinkMinMs int
	thinkMaxMs int
	// payload size mixture weights: small (100-400), mid (400-1400), large (1400-16384)
	wSmall, wMid, wLarge float64
}

var behaviours = map[string]behaviour{
	"interactive": {"interactive", 20, 120, 0.70, 0.26, 0.04},
	"browse":      {"browse", 80, 400, 0.35, 0.45, 0.20},
	"bulk":        {"bulk", 5, 40, 0.10, 0.30, 0.60},
}

func (b behaviour) think() time.Duration {
	span := b.thinkMaxMs - b.thinkMinMs
	if span <= 0 {
		span = 1
	}
	return time.Duration(b.thinkMinMs+mrand.Intn(span)) * time.Millisecond
}

// payloadSize draws an upstream payload size from a heavy-tailed mixture that spans the full
// range up to and beyond the MSS.  This is the fix for P0.5: no class may have a hard ceiling
// below WebTunnel's multi-cell records, or that ceiling becomes the classifier.
func (b behaviour) payloadSize() int {
	u := mrand.Float64()
	switch {
	case u < b.wSmall:
		return 100 + mrand.Intn(300) // 100 .. 399
	case u < b.wSmall+b.wMid:
		return 400 + mrand.Intn(1000) // 400 .. 1399
	default:
		// Log-uniform over [1400, 16384] so the tail is genuinely heavy and reaches the
		// 16 KB TLS record ceiling, well past WebTunnel's largest coalesced record.
		lo, hi := math.Log(1400), math.Log(16384)
		return int(math.Exp(lo + mrand.Float64()*(hi-lo)))
	}
}

func randomBody(n int) []byte {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return bytes.Repeat([]byte("A"), n)
	}
	return b
}

// approxRequestHeaderBytes is a deliberate, documented estimate of the HPACK-compressed header
// block size.  It is reported as app-layer accounting only; the flow builder measures the wire.
const approxRequestHeaderBytes = 120

// ---------------------------------------------------------------------------
// TLS
// ---------------------------------------------------------------------------

// dialUTLS establishes a TLS connection using uTLS with the HelloChrome_Auto fingerprint and
// the parity ALPN list.
//
// B-2: ALPN is deliberately NOT a parameter.  Every class offers exactly ALPNParity, so the
// ClientHello is byte-identical across classes and gate G1 is satisfiable.  Which protocol the
// SERVER then selects is a per-endpoint decision (nginx advertises h2 on 8443 and http/1.1 on
// 8444) and is not visible in the client's fingerprint, which is what a censor matches on.
// The offered list is returned so the caller reports what was actually sent, never a constant.
func dialUTLS(ctx context.Context, targetAddr, serverName string) (*utls.UConn, *net.TCPAddr, *net.TCPAddr, []string, error) {
	alpnProtocols := append([]string(nil), ALPNParity...)
	dialer := net.Dialer{
		Timeout: 15 * time.Second,
	}
	rawConn, err := dialer.DialContext(ctx, "tcp", targetAddr)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("dial failed: %w", err)
	}

	localTCP, _ := rawConn.LocalAddr().(*net.TCPAddr)
	remoteTCP, _ := rawConn.RemoteAddr().(*net.TCPAddr)

	config := &utls.Config{
		ServerName:         serverName,
		InsecureSkipVerify: true,
		NextProtos:         alpnProtocols,
	}

	uConn := utls.UClient(rawConn, config, utls.HelloCustom)
	spec, err := utls.UTLSIdToSpec(utls.HelloChrome_Auto)
	if err != nil {
		uConn = utls.UClient(rawConn, config, utls.HelloChrome_Auto)
	} else {
		for i, ext := range spec.Extensions {
			if _, ok := ext.(*utls.ALPNExtension); ok {
				spec.Extensions[i] = &utls.ALPNExtension{AlpnProtocols: alpnProtocols}
			}
		}
		if err := uConn.ApplyPreset(&spec); err != nil {
			uConn = utls.UClient(rawConn, config, utls.HelloChrome_Auto)
		}
	}

	if err := uConn.HandshakeContext(ctx); err != nil {
		rawConn.Close()
		return nil, nil, nil, nil, fmt.Errorf("utls handshake failed: %w", err)
	}

	return uConn, localTCP, remoteTCP, alpnProtocols, nil
}

// newResult records what the connection ACTUALLY did.
//
// B-2: `offered` is the list dialUTLS put into the ClientHello and `picked` comes from
// uConn.ConnectionState().NegotiatedProtocol.  Neither is the ALPNParity constant.  v2.1
// reported the constant unconditionally, so every manifest claimed ("h2","http/1.1") while the
// WebSocket classes had actually offered ("http/1.1",) -- the sidecar contradicted the wire on
// a field it declares as ground truth.
func newResult(local, remote *net.TCPAddr, offered []string, picked, beh string) *GeneratorResult {
	r := &GeneratorResult{
		Protocol:    "tcp",
		ClientStack: "utls-HelloChrome_Auto",
		ALPNOffered: offered,
		ALPNPicked:  picked,
		Behaviour:   beh,
		TupleKnown:  true,
	}
	if local != nil {
		r.ClientIP = local.IP.String()
		r.ClientPort = local.Port
	}
	if remote != nil {
		r.ServerIP = remote.IP.String()
		r.ServerPort = remote.Port
	}
	return r
}

// budgetedRequest builds the next request of a session so that BOTH byte budgets converge.
//
// The same function is used by direct_web_browsing (over TLS, to nginx) and by the WebTunnel
// class (over the tunnel, to the onion service that fronts the SAME nginx).  Making the two
// classes issue the same application-layer session is what turns gate G4 from a p ~ 1e-48
// failure into a matched comparison: after this change the only difference between them is the
// WebTunnel transport, which is the research question.
func budgetedRequest(ctx context.Context, base string, b behaviour,
	bytesUp, targetUp, bytesDown, targetDown int) (*http.Request, int) {

	if bytesUp < targetUp {
		n := b.payloadSize()
		req, _ := http.NewRequestWithContext(ctx, "POST", base+"/api/v1/telemetry",
			bytes.NewReader(randomBody(n)))
		req.Header.Set("Content-Type", "application/octet-stream")
		return req, n
	}
	// Ask for exactly the bytes still missing from the downstream budget, capped at the
	// server's per-response limit so a single response cannot overshoot the whole session.
	want := targetDown - bytesDown
	if want < 1024 {
		want = 1024
	}
	if want > 262144 {
		want = 262144
	}
	req, _ := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/api/v1/blob?bytes=%d", base, want), nil)
	return req, 0
}

// ---------------------------------------------------------------------------
// HTTP/2 classes
// ---------------------------------------------------------------------------

func runHTTP2(ctx context.Context, serverAddr, sni string, b behaviour, targetDur time.Duration,
	targetUp, targetDown int, mode string) (*GeneratorResult, error) {

	start := time.Now()
	uConn, local, remote, offered, err := dialUTLS(ctx, serverAddr, sni)
	if err != nil {
		return nil, err
	}
	defer uConn.Close()

	res := newResult(local, remote, offered, uConn.ConnectionState().NegotiatedProtocol, b.name)

	tr := &http2.Transport{AllowHTTP: false}
	clientConn, err := tr.NewClientConn(uConn)
	if err != nil {
		return nil, fmt.Errorf("http2 client conn failed: %w", err)
	}

	bytesUp, bytesDown, seg := 0, 0, 0
	for time.Since(start) < targetDur && (bytesUp < targetUp || bytesDown < targetDown) {
		var req *http.Request
		var reqBody int

		switch {
		case mode == "video_streaming":
			url := fmt.Sprintf("https://%s/video/segment_%d.m4s", sni, seg)
			req, _ = http.NewRequestWithContext(ctx, "GET", url, nil)
		case mode == "web_assets":
			url := fmt.Sprintf("https://%s/web/assets/chunk_%d.bin", sni, seg)
			req, _ = http.NewRequestWithContext(ctx, "GET", url, nil)
		default: // direct_web_browsing -- the matched control for the WebTunnel class
			req, reqBody = budgetedRequest(ctx, "https://"+sni, b,
				bytesUp, targetUp, bytesDown, targetDown)
		}

		// The GET-dominated classes still emit real upstream bodies sometimes -- a telemetry
		// beacon, an analytics POST, a resumable-upload chunk.  That is what removes the negative
		// upstream record ceiling for them too (P0.5).
		//
		// direct_web_browsing is excluded: budgetedRequest is already steering it onto the same
		// byte budget as the WebTunnel class, and overriding its request here would break the
		// convergence that gate G4 tests.
		if mode != "direct_web_browsing" && reqBody == 0 && mrand.Float64() < 0.35 {
			reqBody = b.payloadSize()
			req, _ = http.NewRequestWithContext(ctx, "POST",
				fmt.Sprintf("https://%s/api/v1/telemetry", sni), bytes.NewReader(randomBody(reqBody)))
			req.Header.Set("Content-Type", "application/octet-stream")
		}

		req.Header.Set("User-Agent", userAgents[mrand.Intn(len(userAgents))])

		resp, err := clientConn.RoundTrip(req)
		if err == nil {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			bytesDown += len(body)
			bytesUp += reqBody + approxRequestHeaderBytes
		}
		seg++
		time.Sleep(b.think())
	}

	res.BytesUp, res.BytesDown, res.DurationS = bytesUp, bytesDown, time.Since(start).Seconds()
	res.OK = true
	return res, nil
}

// ---------------------------------------------------------------------------
// WebSocket classes
// ---------------------------------------------------------------------------

func runWebSocket(ctx context.Context, serverAddr, sni, path string, b behaviour,
	targetDur time.Duration, targetUp, targetDown int) (*GeneratorResult, error) {

	start := time.Now()
	var local, remote *net.TCPAddr
	var alpn string
	var offered []string

	dialer := websocket.Dialer{
		NetDialTLSContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			// B-2: the SAME ClientHello as every other class -- HelloChrome_Auto offering
			// {h2, http/1.1}.  The server picks http/1.1 because this endpoint listens on the
			// nginx port that does not advertise h2; the client's offer is unchanged, and the
			// client's offer is what a censor fingerprints.
			uConn, l, r, off, err := dialUTLS(ctx, addr, sni)
			if err != nil {
				return nil, err
			}
			local, remote, offered = l, r, off
			alpn = uConn.ConnectionState().NegotiatedProtocol
			return uConn, nil
		},
		HandshakeTimeout: 20 * time.Second,
	}

	wsConn, resp, err := dialer.DialContext(ctx, fmt.Sprintf("wss://%s%s", serverAddr, path), nil)
	if err != nil {
		if resp != nil {
			return nil, fmt.Errorf("ws dial error (status %d): %w", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("ws dial error: %w", err)
	}
	defer wsConn.Close()

	res := newResult(local, remote, offered, alpn, b.name)

	bytesUp, bytesDown := 0, 0
	wsConn.SetReadDeadline(time.Now().Add(targetDur + 10*time.Second))

	for time.Since(start) < targetDur && (bytesUp < targetUp || bytesDown < targetDown) {
		// Full-range message sizes (P0.5).  A chat client uploading an image or a ticker
		// pushing a full order-book snapshot both produce multi-KB frames.
		n := b.payloadSize()
		if err := wsConn.WriteMessage(websocket.BinaryMessage, randomBody(n)); err == nil {
			bytesUp += n
		} else {
			break
		}
		if _, msg, err := wsConn.ReadMessage(); err == nil {
			bytesDown += len(msg)
		} else {
			break
		}
		time.Sleep(b.think())
	}

	res.BytesUp, res.BytesDown, res.DurationS = bytesUp, bytesDown, time.Since(start).Seconds()
	res.OK = true
	return res, nil
}

// ---------------------------------------------------------------------------
// WebTunnel (through the Tor SOCKS proxy)
// ---------------------------------------------------------------------------

func runWebTunnel(ctx context.Context, socksProxy, targetURL string, b behaviour,
	targetDur time.Duration, targetUp, targetDown int) (*GeneratorResult, error) {

	// B-3: the target is MANDATORY and must be the local decoy onion service.  There is no
	// fallback list.  v2.1 silently substituted duckduckgo.com / check.torproject.org /
	// en.wikipedia.org whenever the flag was unset or still held its placeholder default, so the
	// positive class fetched the live web while every negative fetched the local mock -- and
	// nothing in the pipeline noticed.  An unset target is now a hard error.
	if targetURL == "" {
		return nil, fmt.Errorf("no webtunnel target configured: pass --target-url with the decoy " +
			"onion address (docker compose exec tor-bridge cat /var/lib/tor/onion_decoy/hostname)")
	}
	base := strings.TrimSuffix(targetURL, "/")

	start := time.Now()
	dialer, err := proxy.SOCKS5("tcp", socksProxy, nil, proxy.Direct)
	if err != nil {
		return nil, fmt.Errorf("socks5 init error: %w", err)
	}

	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				return dialer.Dial(network, addr)
			},
		},
		Timeout: targetDur + 20*time.Second,
	}

	bytesUp, bytesDown, nErr := 0, 0, 0
	for time.Since(start) < targetDur && (bytesUp < targetUp || bytesDown < targetDown) {
		// Identical request mix to direct_web_browsing, against the same handlers on the same
		// nginx front end -- reached through the tunnel instead of directly.
		req, body := budgetedRequest(ctx, base, b, bytesUp, targetUp, bytesDown, targetDown)
		req.Header.Set("User-Agent", userAgents[mrand.Intn(len(userAgents))])

		resp, err := client.Do(req)
		if err != nil {
			nErr++
			if nErr >= 3 {
				break
			}
			time.Sleep(b.think())
			continue
		}
		rb, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		bytesDown += len(rb)
		bytesUp += body + approxRequestHeaderBytes
		time.Sleep(b.think())
	}

	// P0.2: this process dialled 127.0.0.1:9050.  The bridge socket belongs to the Tor daemon
	// and is invisible from here.  Reporting a guessed 5-tuple is what produced V-02, so we
	// report none and let the collector snapshot `ss` inside the container.
	//
	// ClientStack is "webtunnel-pt" and ALPNOffered is nil because the TLS on the wire is the
	// pluggable transport's, not this process's.  Until probe_utls_support.sh confirms a uTLS
	// imitation argument, that stack is stock Go crypto/tls and gate G1 will keep failing --
	// see docs/05-final-review.md B-1.
	res := &GeneratorResult{
		Protocol:    "tcp",
		ClientStack: "webtunnel-pt",
		ALPNOffered: nil,
		Behaviour:   b.name,
		BytesUp:     bytesUp,
		BytesDown:   bytesDown,
		DurationS:   time.Since(start).Seconds(),
		TupleKnown:  false,
		OK:          bytesDown > 0,
	}
	if !res.OK {
		res.Error = "no bytes received through the tunnel"
	}
	return res, nil
}

// ---------------------------------------------------------------------------

func main() {
	mode := flag.String("mode", "direct_web_browsing", "traffic class")
	server := flag.String("server", "legitimate-servers:8443", "server host:port")
	sni := flag.String("sni", "legitimate-servers", "TLS SNI / HTTP authority")
	socks := flag.String("socks", "127.0.0.1:9050", "Tor SOCKS5 proxy")
	target := flag.String("target-url", "", "REQUIRED for --mode webtunnel: base URL of the local "+
		"decoy onion service reached through Tor (e.g. http://<56-char>.onion). No default and no "+
		"public fallback -- see B-3 in docs/05-final-review.md.")
	behav := flag.String("behaviour", "browse", "browse | bulk | interactive")
	duration := flag.Float64("target-duration", 3.0, "target session duration (s)")
	bytesUp := flag.Int("target-bytes-up", 40000, "target upstream bytes")
	bytesDown := flag.Int("target-bytes-down", 100000, "target downstream bytes")
	seed := flag.Int64("seed", time.Now().UnixNano(), "random seed")
	flag.Parse()

	mrand.Seed(*seed)

	b, ok := behaviours[*behav]
	if !ok {
		out, _ := json.Marshal(&GeneratorResult{OK: false, Error: "unknown behaviour: " + *behav})
		fmt.Println(string(out))
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(),
		time.Duration(math.Max(*duration*2.5, 20.0))*time.Second)
	defer cancel()

	dur := time.Duration(*duration * float64(time.Second))
	var res *GeneratorResult
	var err error

	switch *mode {
	case "direct_web_browsing", "video_streaming", "web_assets":
		res, err = runHTTP2(ctx, *server, *sni, b, dur, *bytesUp, *bytesDown, *mode)
	case "websocket_ticker":
		res, err = runWebSocket(ctx, *server, *sni, "/ws/ticker", b, dur, *bytesUp, *bytesDown)
	case "websocket_chat":
		res, err = runWebSocket(ctx, *server, *sni, "/ws/chat", b, dur, *bytesUp, *bytesDown)
	case "webtunnel":
		res, err = runWebTunnel(ctx, *socks, *target, b, dur, *bytesUp, *bytesDown)
	default:
		err = fmt.Errorf("unknown mode: %s", *mode)
	}

	if err != nil {
		res = &GeneratorResult{
			ClientStack: "utls-HelloChrome_Auto",
			ALPNOffered: nil, // B-2: nothing was offered -- the connection never completed
			Behaviour:   *behav,
			OK:          false,
			Error:       err.Error(),
		}
	}

	out, _ := json.Marshal(res)
	fmt.Println(string(out))
}
