package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	mrand "math/rand"
	"net"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	utls "github.com/refraction-networking/utls"
	"golang.org/x/net/http2"
	"golang.org/x/net/proxy"
)

type GeneratorResult struct {
	ClientIP     string  `json:"client_ip"`
	ClientPort   int     `json:"client_port"`
	ServerIP     string  `json:"server_ip"`
	ServerPort   int     `json:"server_port"`
	Protocol     string  `json:"protocol"`
	ClientStack  string  `json:"client_stack"`
	BytesUp      int     `json:"bytes_up"`
	BytesDown    int     `json:"bytes_down"`
	DurationS    float64 `json:"duration_s"`
	Error        string  `json:"error,omitempty"`
}

var userAgents = []string{
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
	"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}

// dialUTLS establishes a TLS connection using uTLS with HelloChrome_Auto fingerprint
func dialUTLS(ctx context.Context, targetAddr, serverName string, alpn []string) (*utls.UConn, *net.TCPAddr, *net.TCPAddr, error) {
	var dialer net.Dialer
	rawConn, err := dialer.DialContext(ctx, "tcp", targetAddr)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("dial failed: %w", err)
	}

	localTCP := rawConn.LocalAddr().(*net.TCPAddr)
	remoteTCP := rawConn.RemoteAddr().(*net.TCPAddr)

	config := &utls.Config{
		ServerName:         serverName,
		InsecureSkipVerify: true,
		NextProtos:         alpn,
	}

	uConn := utls.UClient(rawConn, config, utls.HelloCustom)
	spec, err := utls.UTLSIdToSpec(utls.HelloChrome_Auto)
	if err != nil {
		uConn = utls.UClient(rawConn, config, utls.HelloChrome_Auto)
	} else {
		for i, ext := range spec.Extensions {
			if _, ok := ext.(*utls.ALPNExtension); ok {
				spec.Extensions[i] = &utls.ALPNExtension{AlpnProtocols: alpn}
			}
		}
		if err := uConn.ApplyPreset(&spec); err != nil {
			uConn = utls.UClient(rawConn, config, utls.HelloChrome_Auto)
		}
	}

	if err := uConn.HandshakeContext(ctx); err != nil {
		rawConn.Close()
		return nil, nil, nil, fmt.Errorf("utls handshake failed: %w", err)
	}

	return uConn, localTCP, remoteTCP, nil
}

func runDirectBrowsing(ctx context.Context, serverAddr string, targetDur time.Duration, targetUp, targetDown int) (*GeneratorResult, error) {
	start := time.Now()
	uConn, localAddr, remoteAddr, err := dialUTLS(ctx, serverAddr, "legitimate-servers", []string{"h2", "http/1.1"})
	if err != nil {
		return nil, err
	}
	defer uConn.Close()

	res := &GeneratorResult{
		ClientIP:    localAddr.IP.String(),
		ClientPort:  localAddr.Port,
		ServerIP:    remoteAddr.IP.String(),
		ServerPort:  remoteAddr.Port,
		Protocol:    "tcp",
		ClientStack: "utls-HelloChrome_Auto",
	}

	tr := &http2.Transport{}
	clientConn, err := tr.NewClientConn(uConn)
	if err != nil {
		return nil, fmt.Errorf("http2 client conn failed: %w", err)
	}

	bytesUp := 0
	bytesDown := 0

	for time.Since(start) < targetDur && (bytesUp < targetUp || bytesDown < targetDown) {
		// Alternate between GET feed and POST telemetry/graphql
		if bytesUp < targetUp {
			chunkSize := 400 + mrand.Intn(400)
			payload := bytes.Repeat([]byte("A"), chunkSize)
			req, _ := http.NewRequestWithContext(ctx, "POST", "https://legitimate-servers/api/v1/telemetry", bytes.NewReader(payload))
			req.Header.Set("User-Agent", userAgents[mrand.Intn(len(userAgents))])
			req.Header.Set("Content-Type", "application/json")

			resp, err := clientConn.RoundTrip(req)
			if err == nil {
				body, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				bytesUp += chunkSize
				bytesDown += len(body)
			}
		} else {
			req, _ := http.NewRequestWithContext(ctx, "GET", "https://legitimate-servers/api/v1/feed", nil)
			req.Header.Set("User-Agent", userAgents[mrand.Intn(len(userAgents))])
			resp, err := clientConn.RoundTrip(req)
			if err == nil {
				body, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				bytesDown += len(body)
			}
		}
		time.Sleep(time.Duration(50+mrand.Intn(150)) * time.Millisecond)
	}

	res.BytesUp = bytesUp
	res.BytesDown = bytesDown
	res.DurationS = time.Since(start).Seconds()
	return res, nil
}

func runWebSocket(ctx context.Context, serverAddr, path string, targetDur time.Duration, targetUp, targetDown int) (*GeneratorResult, error) {
	start := time.Now()
	var localAddr *net.TCPAddr
	var remoteAddr *net.TCPAddr

	dialer := websocket.Dialer{
		NetDialTLSContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			uConn, loc, rem, err := dialUTLS(ctx, addr, "legitimate-servers", []string{"http/1.1"})
			if err != nil {
				return nil, err
			}
			localAddr = loc
			remoteAddr = rem
			return uConn, nil
		},
		HandshakeTimeout: 5 * time.Second,
	}

	wsConn, resp, err := dialer.DialContext(ctx, "wss://legitimate-servers:8443"+path, nil)
	if err != nil {
		if resp != nil {
			return nil, fmt.Errorf("ws dial error (status %d): %w", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("ws dial error: %w", err)
	}
	defer wsConn.Close()

	res := &GeneratorResult{
		ClientIP:    localAddr.IP.String(),
		ClientPort:  localAddr.Port,
		ServerIP:    remoteAddr.IP.String(),
		ServerPort:  remoteAddr.Port,
		Protocol:    "tcp",
		ClientStack: "utls-HelloChrome_Auto",
	}

	bytesUp := 0
	bytesDown := 0
	wsConn.SetReadDeadline(time.Now().Add(targetDur + 5*time.Second))

	for time.Since(start) < targetDur && (bytesUp < targetUp || bytesDown < targetDown) {
		msgLen := 350 + mrand.Intn(400)
		payload := bytes.Repeat([]byte("M"), msgLen)
		if err := wsConn.WriteMessage(websocket.TextMessage, payload); err == nil {
			bytesUp += len(payload)
		}

		_, msg, err := wsConn.ReadMessage()
		if err == nil {
			bytesDown += len(msg)
		}

		time.Sleep(time.Duration(40+mrand.Intn(100)) * time.Millisecond)
	}

	res.BytesUp = bytesUp
	res.BytesDown = bytesDown
	res.DurationS = time.Since(start).Seconds()
	return res, nil
}

func runVideoStreaming(ctx context.Context, serverAddr string, targetDur time.Duration, targetUp, targetDown int) (*GeneratorResult, error) {
	start := time.Now()
	uConn, localAddr, remoteAddr, err := dialUTLS(ctx, serverAddr, "legitimate-servers", []string{"h2", "http/1.1"})
	if err != nil {
		return nil, err
	}
	defer uConn.Close()

	res := &GeneratorResult{
		ClientIP:    localAddr.IP.String(),
		ClientPort:  localAddr.Port,
		ServerIP:    remoteAddr.IP.String(),
		ServerPort:  remoteAddr.Port,
		Protocol:    "tcp",
		ClientStack: "utls-HelloChrome_Auto",
	}

	tr := &http2.Transport{}
	clientConn, err := tr.NewClientConn(uConn)
	if err != nil {
		return nil, err
	}

	bytesUp := 0
	bytesDown := 0
	segID := 0

	for time.Since(start) < targetDur && bytesDown < targetDown {
		urlStr := fmt.Sprintf("https://legitimate-servers/video/segment_%d.m4s", segID)
		req, _ := http.NewRequestWithContext(ctx, "GET", urlStr, nil)
		req.Header.Set("User-Agent", userAgents[mrand.Intn(len(userAgents))])

		resp, err := clientConn.RoundTrip(req)
		if err == nil {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			bytesDown += len(body)
			bytesUp += 200 // HTTP/2 headers frame size
		}
		segID++
		time.Sleep(time.Duration(200+mrand.Intn(400)) * time.Millisecond)
	}

	res.BytesUp = bytesUp
	res.BytesDown = bytesDown
	res.DurationS = time.Since(start).Seconds()
	return res, nil
}

func runWebAssets(ctx context.Context, serverAddr string, targetDur time.Duration, targetUp, targetDown int) (*GeneratorResult, error) {
	start := time.Now()
	uConn, localAddr, remoteAddr, err := dialUTLS(ctx, serverAddr, "legitimate-servers", []string{"h2", "http/1.1"})
	if err != nil {
		return nil, err
	}
	defer uConn.Close()

	res := &GeneratorResult{
		ClientIP:    localAddr.IP.String(),
		ClientPort:  localAddr.Port,
		ServerIP:    remoteAddr.IP.String(),
		ServerPort:  remoteAddr.Port,
		Protocol:    "tcp",
		ClientStack: "utls-HelloChrome_Auto",
	}

	tr := &http2.Transport{}
	clientConn, err := tr.NewClientConn(uConn)
	if err != nil {
		return nil, err
	}

	bytesUp := 0
	bytesDown := 0
	assetID := 0

	for time.Since(start) < targetDur && bytesDown < targetDown {
		urlStr := fmt.Sprintf("https://legitimate-servers/web/assets/chunk_%d.bin", assetID)
		req, _ := http.NewRequestWithContext(ctx, "GET", urlStr, nil)
		req.Header.Set("User-Agent", userAgents[mrand.Intn(len(userAgents))])

		resp, err := clientConn.RoundTrip(req)
		if err == nil {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			bytesDown += len(body)
			bytesUp += 180
		}
		assetID++
		time.Sleep(time.Duration(20+mrand.Intn(80)) * time.Millisecond)
	}

	res.BytesUp = bytesUp
	res.BytesDown = bytesDown
	res.DurationS = time.Since(start).Seconds()
	return res, nil
}

func runWebTunnel(ctx context.Context, socksProxy, targetURL string, targetDur time.Duration, targetUp, targetDown int) (*GeneratorResult, error) {
	start := time.Now()

	// SOCKS5 Dialing to ensure fresh connection per sample
	dialer, err := proxy.SOCKS5("tcp", socksProxy, nil, proxy.Direct)
	if err != nil {
		return nil, fmt.Errorf("socks5 init error: %w", err)
	}

	httpTransport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return dialer.Dial(network, addr)
		},
	}
	client := &http.Client{
		Transport: httpTransport,
		Timeout:   targetDur + 10*time.Second,
	}

	bytesUp := 0
	bytesDown := 0

	for time.Since(start) < targetDur && (bytesUp < targetUp || bytesDown < targetDown) {
		req, _ := http.NewRequestWithContext(ctx, "GET", targetURL, nil)
		req.Header.Set("User-Agent", userAgents[mrand.Intn(len(userAgents))])

		resp, err := client.Do(req)
		if err == nil {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			bytesDown += len(body)
			bytesUp += 300
		}
		time.Sleep(time.Duration(100+mrand.Intn(300)) * time.Millisecond)
	}

	res := &GeneratorResult{
		ClientIP:    "172.20.0.3",
		ClientPort:  0, // Will be matched by sanitizer via bridge target IP
		ServerIP:    "172.20.0.10",
		ServerPort:  443,
		Protocol:    "tcp",
		ClientStack: "utls-HelloChrome_Auto",
		BytesUp:     bytesUp,
		BytesDown:   bytesDown,
		DurationS:   time.Since(start).Seconds(),
	}
	return res, nil
}

func main() {
	mode := flag.String("mode", "direct_web_browsing", "Traffic class")
	server := flag.String("server", "legitimate-servers:8443", "Server host:port")
	socks := flag.String("socks", "127.0.0.1:9050", "Tor SOCKS5 proxy")
	duration := flag.Float64("target-duration", 3.0, "Target session duration in seconds")
	bytesUp := flag.Int("target-bytes-up", 40000, "Target upstream bytes")
	bytesDown := flag.Int("target-bytes-down", 100000, "Target downstream bytes")
	seed := flag.Int64("seed", time.Now().UnixNano(), "Random seed")
	flag.Parse()

	mrand.Seed(*seed)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(math.Max(*duration*2.0, 15.0))*time.Second)
	defer cancel()

	dur := time.Duration(*duration * float64(time.Second))
	var res *GeneratorResult
	var err error

	switch *mode {
	case "direct_web_browsing":
		res, err = runDirectBrowsing(ctx, *server, dur, *bytesUp, *bytesDown)
	case "websocket_ticker":
		res, err = runWebSocket(ctx, *server, "/ws/ticker", dur, *bytesUp, *bytesDown)
	case "websocket_chat":
		res, err = runWebSocket(ctx, *server, "/ws/chat", dur, *bytesUp, *bytesDown)
	case "video_streaming":
		res, err = runVideoStreaming(ctx, *server, dur, *bytesUp, *bytesDown)
	case "web_assets":
		res, err = runWebAssets(ctx, *server, dur, *bytesUp, *bytesDown)
	case "webtunnel":
		res, err = runWebTunnel(ctx, *socks, "https://check.torproject.org", dur, *bytesUp, *bytesDown)
	default:
		err = fmt.Errorf("unknown mode: %s", *mode)
	}

	if err != nil {
		res = &GeneratorResult{
			ClientStack: "utls-HelloChrome_Auto",
			Error:       err.Error(),
		}
	}

	out, _ := json.Marshal(res)
	fmt.Println(string(out))
}
