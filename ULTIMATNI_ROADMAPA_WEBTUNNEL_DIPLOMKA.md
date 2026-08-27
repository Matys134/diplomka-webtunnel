# ULTIMÁTNÍ VÝZKUMNÁ A EXPERIMENTÁLNÍ ROADMAPA DIPLOMOVÉ PRÁCE

**Téma:** Analýza odolnosti protokolu WebTunnel v prostředí strukturálně podobného legitimního provozu  
**Autor:** Bc. Matěj Kouba  
**Pracoviště:** Katedra informatiky, Přírodovědecká fakulta Jihočeské univerzity v Českých Budějovicích  
**Školitel:** Ing. Petr Břehovský  
**Verze:** 3.0 (Syntetizovaná, kriticky revidovaná a rozšířená na základě State-of-the-Art 2024–2026)

---

## 1. Manažerské a vědecké shrnutí

Tento dokument představuje **ultimátní syntézu a metodický manuál** pro vypracování diplomové práce. Vychází z detailní komparace dvou původních návrhů roadmap (*Resilience Research Roadmap* a *Traffic Analysis Roadmap*), odstraňuje jejich nerealistické předpoklady, integruje nejnovější vědecké poznatky (včetně průlomového článku Roba Jansena et al., NDSS 2024 a konceptů Huma NDSS 2026) a definuje exaktní postup od vytvoření laboratorního testbedu až po úspěšnou obhajobu.

```
+---------------------------------------------------------------------------------------------------+
|                                 PŘEHLED VÝZKUMNÉHO RÁMCE                                          |
+---------------------------------------------------------------------------------------------------+
|  1. MODEL HROZEB & BASE RATE FALLACY                                                              |
|     - In-path pasivní cenzor (ISP úroveň, gigabitové linky, line-rate omezení)                    |
|     - Matematický důkaz nutnosti Low-FPR (FPR <= 10^-4) a Host-Based agregace                    |
+---------------------------------------------------------------------------------------------------+
|  2. IZOLOVANÝ TESTBED & GENEROVÁNÍ HARD NEGATIVES                                                 |
|     - 4 síťové zóny v Linux Network Namespaces / Docker Compose                                   |
|     - Klientské profily: Web Browsing, Bulk Download, Interactive Shell                           |
|     - Hard Negatives: Dlouhožijící WebSockets, gRPC/HTTP2 multiplex, DASH Video, WebRTC           |
|     - Emulace WAN realismu pomocí tc-netem (Pareto-normal jitter, Gilbert-Elliot burst loss)      |
+---------------------------------------------------------------------------------------------------+
|  3. DATA PIPELINE & ANTI-LEAKAGE SANITACE                                                         |
|     - Striktní odstranění L2/L3/L4 artefaktů (MAC, IP, TTL, porty, TCP Options)                  |
|     - Izolace post-handshake toku (zamezení triviální klasifikaci přes uTLS/JA4 fingerprint)      |
|     - Destination-Split + Temporal/Session-Split pro vědeckou validitu                            |
+---------------------------------------------------------------------------------------------------+
|  4. KLASIFIKAČNÍ HIERARCHIE & MODELY                                                              |
|     - Baseline: XGBoost / LightGBM (48 flow-level statistických rysů)                             |
|     - Deep Learning: 1D-CNN (Deep Packet / Sirinam DF architektura + Focal Loss)                  |
|     - Sekvenční model: Flow-Transformer (Self-Attention nad sekvencí paketů / flowletů)          |
|     - Meta-klasifikátor: Host-Based Bayesian Aggregator (odstranění Base Rate Fallacy)            |
+---------------------------------------------------------------------------------------------------+
|  5. EXPERIMENTÁLNÍ MATICE & PROFILOVÁNÍ                                                           |
|     - PR-AUC, Precision @ Fixed FPR (10^-3, 10^-4, 10^-5), Confusion Matrix na Hard Negatives    |
|     - FDR projekce pro různé prevalence (alfa = 10^-2, 10^-4, 10^-6)                             |
|     - Benchmark výpočetní latence (ms/flow) a propustnosti (flows/sec) na CPU vs GPU             |
+---------------------------------------------------------------------------------------------------+
|  6. DE-MASKING SLABIN WEBTUNNELU & NÁVRH DEFENSIVNÍCH OPATŘENÍ                                    |
|     - Analýza 514B Tor buněk (Cell Quantization) a Circuit Setup Burstů                          |
|     - Návrh: Adaptive Intra-frame Padding, Packet Chunking, HTTP/2 Framing Mimicry               |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Kritická komparace původních PDF roadmap

V rámci přípravy byly podrobeny detailní analýze dva existující dokumenty:
- **Dokument A:** `WebTunnel Resilience Research Roadmap.pdf`
- **Dokument B:** `WebTunnel Traffic Analysis Roadmap.pdf`

### 2.1 Srovnávací tabulka aspektů

| Kritérium | Dokument A (Resilience Roadmap) | Dokument B (Traffic Analysis Roadmap) | Hodnocení a doporučení pro diplomku |
| :--- | :--- | :--- | :--- |
| **Zaměření a tón** | Teoreticko-výzkumný, silná akademická argumentace. | Inženýrský, prakticky implementační. | **Syntéza:** Využít akademický narativ z A pro text práce a technickou konkrétnost z B pro realizaci. |
| **Architektura testbedu** | Obecný popis (Linux Namespaces, KVM, Open vSwitch). | Konkrétní síťová topologie (4 jmenné prostory, veth páry, dumpcap, Caddy/Nginx). | **Dokument B je výrazně lepší.** Obsahuje přesný ASCII diagram a topologii. |
| **Simulace sítě (tc-netem)** | Teoretický popis potřeby jitteru a ztrátovosti. | Přesné bash příkazy pro 4G/LTE a Gilbert-Elliot model. | **Dokument B je přímo aplikovatelný.** |
| **Sanitace dat (Anti-Leakage)** | Detailní popis 4 kroků sanitace L2–L4. | Tabulka artefaktů + sanitace; zmiňuje randomizaci délky URL tajné cesty. | **Dokument B + rozšíření:** Zohlednit nutnost sanitace TLS handshake (uTLS) pro čistou post-handshake analýzu. |
| **Návrh ML/DL modelů** | Koncepční popis XGBoost, 1D-CNN a TrafficFormeru. | Exaktní vrstvy PyTorch 1D-CNN, parametry Focal Loss ($\gamma=2.0, \alpha_t=0.25$). | **Dokument B je praktičtější.** Umožňuje okamžitou implementaci v PyTorch. |
| **Base Rate Fallacy** | Odvozuje Bayesův teorém, odkazuje na Jansen et al. (NDSS 2024). | Obsahuje hotovou tabulku projekcí FDR pro $\alpha \in \{10^{-2}, 10^{-4}, 10^{-6}\}$. | **Dokument A má lepší kontext (host-based), Dokument B má lepší tabulkovou vizualizaci.** |
| **Výpočetní profiling** | Teoretický požadavek na měření ms/flow a flows/sec. | Konkrétní srovnávací tabulka (XGBoost vs 1D-CNN vs TrafficFormer na Xeon/A100). | **Pozor:** Tabulka v B je ilustrativní/syntetická – diplomant musí provést vlastní reálná měření. |
| **Doporučení pro WebTunnel** | Popisuje Traffic Shaping, 514B cell packing, Obscura rotaci. | Popisuje Adaptive Padding (1–128B Gaussian, <7% režie), Packet Splitting, Frame Mimicry. | **Oba dokumenty jsou vysoce kvalitní**, v závěru se ideálně doplňují. |

---

## 3. Kritická dekonstrukce rizik, pastí a „nerealistických předpokladů“

Při hloubkové rešerši byly v obou původních PDF identifikovány následující kritické body, které by při nekritickém převzetí ohrozily úspěšnost práce:

### 3.1 Past č. 1: TrafficFormer a Transformer z HuggingFace
- **Nerealistický předpoklad v PDF:** Obě roadmapy tvrdí, že z HuggingFace jednoduše stáhneme masivně předtrénovaný model *TrafficFormer* a provedeme fine-tuning.
- **Realita výzkumu (2025/2026):** Článek *TrafficFormer* (Zhou et al., IEEE S&P 2025) a *FlowletFormer* (2025) jsou špičkové vědecké práce, ale neexistuje pro ně univerzální veřejný HuggingFace checkpoint natrénovaný na WebTunnelu. Předtrénování od nuly na surových PCAPech metodou *Masked Burst Modeling* vyžaduje terabajty síťových dat a týdny výpočetního času na multi-GPU clusterech.
- **Správné inženýrské řešení pro diplomku:**
  1. Implementovat lehký sekvenční **Flow-Transformer / Sequence Transformer** (2–4 vrstvy enkodéru s multi-head self-attention) trénovaný přímo na sekvencích normalizovaných paketů (velikost, směr, $\Delta t$).
  2. Alternativně adaptovat existující otevřené modely pro síťový provoz (např. *ET-BERT* nebo *YaTC*), pokud je cílem transfer learning.
  3. Těžiště pokročilého DL postavit na osvědčené a rigorózně laděné **1D-CNN (inspirované architekturou Deep Fingerprinting / Deep Packet)**, která na sekvenčních síťových datech dosahuje vynikající přesnosti s minimální výpočetní režií.

### 3.2 Past č. 2: Únik informací v TLS ClientHello (uTLS vs Python/Node)
- **Riziko:** WebTunnel používá knihovnu `uTLS`, která věrně napodobuje TLS ClientHello prohlížeče Google Chrome (včetně pořadí cipher suites a rozšíření JA3/JA4). Pokud v testbedu vygenerujeme legitimní WebSockets přes Node.js nebo Python a porovnáme je s WebTunnelem, ML model se nenaučí dynamiku provozu, ale triviálně identifikuje rozdíl v TLS otisku!
- **Správné řešení:**
  - **Režim A (Čistá post-handshake analýza):** V datové pipeline striktně odstranit úvodní TLS handshake (pakety 1 až 4) a trénovat klasifikátory výhradně na šifrovaných aplikačních datech (směry, velikosti, časování mezer).
  - **Režim B (Konzistentní klientský stack):** Pro generování legitimního provozu použít rovněž reálný Chromium prohlížeč (přes Playwright/Puppeteer), aby obě třídy sdílely identický TLS otisk.

### 3.3 Past č. 3: Flow-Level klasifikace vs Host-Based agregace (Jansen NDSS 2024)
- **Matematická realita:** V reálné páteřní síti je poměr legitimního provozu k WebTunnelu $\lambda = 10^4$ až $10^6$ ($\alpha = 10^{-4}$ až $10^{-6}$). I kdyby měl 1D-CNN model přesnost 99,9 % ($FPR = 10^{-3}$), při $\alpha = 10^{-4}$ bude **více než 91 % všech poplachů falešných** ($FDR > 91\%$).
- **Průlomové řešení v práci:** Do experimentální části diplomové práce zařadit nejen vyhodnocení jednotlivých toků (Per-Flow Classification), ale také **Host-Based Bayesovskou agregaci** (Jansen et al., NDSS 2024), kde se log-likelihood skóre sčítá přes $M$ po sobě jdoucích toků k dané cílové IP adrese:
  $$\mathcal{L}(\text{Host}) = \sum_{i=1}^{M} \ln \left( \frac{P(\mathbf{x}_i \mid \text{WebTunnel})}{P(\mathbf{x}_i \mid \text{Legitimate})} \right)$$
  Tím se falešná pozitivita exponenciálně eliminuje k nule, což bude představovat **špičkový vědecký přínos diplomové práce**.

---

## 4. Architektura experimentálního testbedu

Sběr dat probíhá v plně deterministickém, izolovaném prostředí realizovaném pomocí **Docker Compose** a **Linux Network Namespaces**. Tím je zaručeno, že do měření nepronikne žádný parazitní provoz z hostitelského OS ani z lokální LAN.

```
+---------------------------------------------------------------------------------------------------+
|                                 TOPOLOGIE IZOLOVANÉHO TESTBEDU                                     |
|                                                                                                   |
|  +------------------------------+             +------------------------------------------------+  |
|  | ns-client (Klientská zóna)   |             | ns-router (In-Path Cenzor / Emulátor WAN)      |  |
|  |                              |             |                                                |  |
|  |  +------------------------+  |   veth-c    |  +------------------------------------------+  |  |
|  |  | Playwright Skripty     |  | <---------> |  | tc-netem: Delay, Jitter, Loss, Reordering |  |  |
|  |  +-----------+------------+  |             |  +--------------------+---------------------+  |  |
|  |              | SOCKS5        |             |                       |                        |  |
|  |  +-----------v------------+  |             |  +--------------------v---------------------+  |  |
|  |  | Tor + WebTunnel Client |  |             |  | dumpcap / tshark (Bezeztrátový PCAPng)   |  |  |
|  |  +------------------------+  |             |  +------------------------------------------+  |  |
|  +------------------------------+             +-----------------------+------------------------+  |
|                                                                       |                           |
|                                                     +-----------------+-----------------+         |
|                                              veth-b |                            veth-l |         |
|                                                     v                                   v         |
|                       +-----------------------------------+   +---------------------------------+ |
|                       | ns-bridge (WebTunnel Server)      |   | ns-legitimate (Hard Negatives)  | |
|                       |                                   |   |                                 | |
|                       |  +-----------------------------+  |   |  +---------------------------+  | |
|                       |  | Nginx / Caddy (Port 443)    |  |   |  | Node.js / Python WS Serv. |  | |
|                       |  | TLS + Secret Path Proxy     |  |   |  +---------------------------+  | |
|                       |  +--------------+--------------+  |   |  | Go gRPC / HTTP2 Endpoints |  | |
|                       |                 | proxy_pass      |   |  +---------------------------+  | |
|                       |  +--------------v--------------+  |   |  | Nginx HLS/DASH Streaming  |  | |
|                       |  | Tor Bridge Daemon (15000)   |  |   |  +---------------------------+  | |
|                       |  +-----------------------------+  |   |  | Běžný Web / REST API      |  | |
|                       +-----------------------------------+   +---------------------------------+ |
+---------------------------------------------------------------------------------------------------+
```

### 4.1 Klientské automatizační profily (Generování zátěže)
Pro simulaci realistického chování uživatele Toru jsou nasazeny 3 profily řízené přes Playwright:
1. **Interaktivní Web Browsing:** Procházení 100 různých webových stránek (zpravodajství, e-shopy), stahování HTML, CSS, JS, obrázků; simulace čtení s náhodnými pauzami (think time) $\sim \mathcal{U}(2, 15)\,\text{s}$.
2. **Bulk Download (Objemové stahování):** Kontinuální stahování velkých binárních souborů (1 MB až 50 MB) přes HTTPS; generuje souvislé asymetrické downstream bursty.
3. **Interactive Shell (Terminál přes SSH/Tor):** Krátké asynchronní pakety směrem nahoru (keystrokes) následované okamžitou odezvou ze serveru, prokládané periodami nečinnosti.

### 4.2 Portfolio strukturálně podobného provozu (Hard Negatives)
Pro zamezení triviální separace musí legitimní provoz obsahovat třídy s identickým chováním na L4/L7:
- **Dlouhotrvající WebSockets (WSS):** Finanční streamy (Binance/Coinbase ticker feed), real-time chat aplikace (Socket.io) – obousměrná komunikace, asynchronní bursty, keep-alive ping/pong rámce.
- **HTTP/2 a gRPC Multiplexing:** Souběžné streamy v jediném TCP spojení, binární framing, multiplexovaný přenos dat bez přímé vazby na sekvenci jednotlivých aplikačních požadavků.
- **Adaptivní video streaming (DASH/HLS over HTTPS):** Sekvenční stahování segmentů videa (`.ts`, `.m4s`) s charakteristickým vzorcem *burst-and-idle* (stahování fragmentu $\rightarrow$ pauza na přehrání vyrovnávací paměti).
- **Moderní QUIC / HTTP/3:** Datové toky nad UDP pro ověření chování modelů vůči alternativním transportním protokolům.

### 4.3 Konfigurace síťového realismu (`tc-netem`)
V jmenném prostoru `ns-router` se aplikují pravidla emulující reálné internetové podmínky:

```bash
# Profil A: Standardní evropská širokopásmová síť (Fiber/VDSL)
tc qdisc add dev veth-router-in root handle 1: netem \
    delay 25ms 5ms distribution normal \
    loss random 0.05% \
    duplicate 0.02%

# Profil B: Mobilní 4G/LTE připojení s vyšším rozptylem
tc qdisc change dev veth-router-in root handle 1: netem \
    delay 45ms 15ms distribution paretonormal \
    loss random 0.2% \
    reorder 0.5% 25%

# Profil C: Zhoršené / satelitní připojení s burst ztrátami (Gilbert-Elliot)
tc qdisc change dev veth-router-in root handle 1: netem \
    delay 120ms 35ms distribution paretonormal \
    loss state 0.02 0.30 0.01 0.10 \
    corrupt 0.05%
```

---

## 5. Datová pipeline, sanitace a reprezentace příznaků

### 5.1 Striktní sanitační protokol (Anti-Leakage)
Aby se modely učily pouze invariantní dynamiku protokolu a ne laboratorní artefakty:
1. **L2 (Ethernet):** Kompletní odstranění MAC adres a ethernetových hlaviček.
2. **L3 (IP):** Vynulování zdrojových i cílových IP adres, vymazání TTL (Time To Live) a polí Type of Service / DSCP.
3. **L4 (TCP/UDP):** Vynulování portů (např. 443 vs 15000), vymazání sekvenčních čísel, ACK čísel a kompletní odstranění TCP Options (SACK, Window Scale, Timestamps, MSS).
4. **Směrová normalizace:** První paket spojení (SYN u TCP nebo první klientský rámec) určuje směr $+1$ (upstream/klient). Odpovědi získávají směr $-1$ (downstream/server).
5. **MTU Clamping & Truncation:** Oříznutí paketů nad 1500 bajtů a eliminace čistých fragmentačních anomálií.
6. **Časová normalizace:** Absolutní časové značky se převádějí na mezidobí příjezdů paketů $\Delta t_i = t_i - t_{i-1}$, aplikuje se logaritmické škálování $f(\Delta t) = \ln(1 + \Delta t)$.

### 5.2 Strategie rozdělení dat (Train / Val / Test Split)
- **Destination-Split:** Cílové domény a IP adresy použité v testovací sadě se **nikdy nesmí objevit** v trénovací ani validační sadě. Tím je garantováno, že model neklasifikuje konkrétní cílový server, ale samotný protokol.
- **Session / Temporal Split:** Testovací data pocházejí z jiného časového okna než trénovací data ($T_{\text{test}} > T_{\text{train}}$).

### 5.3 Tři formáty reprezentace dat pro modely

```
+---------------------------------------------------------------------------------------------------+
|  1. TABULKOVÝ STATISTICKÝ VEKTOR (48 příznaků pro XGBoost / LightGBM)                            |
|     - Velikosti paketů (min, max, mean, std, skewness, 10., 25., 50., 75., 90. percentil)         |
|     - Mezidobí příjezdů Delta t (mean, std, max, percentily celkově i po směrech)                |
|     - Statistika burstů (počet burstů, průměrná délka v paketech/bajtech, směrodatná odchylka)  |
|     - Směrové poměry (Up/Down ratio bajtů, Up/Down ratio paketů)                                 |
+---------------------------------------------------------------------------------------------------+
|  2. SEKVENCE PAKETŮ / 2D TENZOR (Vstup pro 1D-CNN)                                               |
|     - Rozměr: (Batch_Size, 200, 2) pro prvních N=200 paketů toku                                 |
|     - Kanál 1: Normalizovaná velikost paketu s_i / 1500 in [-1.0, 1.0]                          |
|     - Kanál 2: Logaritmické mezidobí ln(1 + Delta t_i)                                            |
|     - Padding nulami pro kratší toky                                                             |
+---------------------------------------------------------------------------------------------------+
|  3. FLOWLET / BURST TOKENIZACE (Vstup pro Flow-Transformer)                                       |
|     - Tok rozdělen na sekvenci burstů (flowlets): délka tokenové sekvence N <= 128                |
|     - Embedding = Token_Emb + Direction_Emb + Temporal_Emb + Positional_Emb                       |
+---------------------------------------------------------------------------------------------------+
```

---

## 6. Návrh klasifikačních modelů a ztrátových funkcí

### 6.1 Referenční Baseline: XGBoost / LightGBM
- **Vstup:** Vektor 48 statistických příznaků toku.
- **Hyperparametry:** `n_estimators=300`, `max_depth=6`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Účel:** Referenční bod pro rychlost inferenčního kroku a základní diskriminační schopnost klasických statistických příznaků.

### 6.2 Pokročilý DL klasifikátor: 1D-CNN (Deep Packet / DF Architektura)
- **Vstup:** Tenzor $(B, 200, 2)$.
- **Architektura v PyTorch:**
  1. `Conv1D(in=2, out=64, kernel_size=7, padding='same')` $\rightarrow$ `BatchNorm1D(64)` $\rightarrow$ `ReLU()` $\rightarrow$ `MaxPool1D(2)`
  2. `Conv1D(in=64, out=128, kernel_size=5, padding='same')` $\rightarrow$ `BatchNorm1D(128)` $\rightarrow$ `ReLU()` $\rightarrow$ `MaxPool1D(2)`
  3. `Conv1D(in=128, out=256, kernel_size=3, padding='same')` $\rightarrow$ `BatchNorm1D(256)` $\rightarrow$ `ReLU()` $\rightarrow$ `GlobalAveragePooling1D()`
  4. `Dense(256, 128)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Dense(128, 1)` $\rightarrow$ `Sigmoid()`
- **Ztrátová funkce (Focal Loss):**
  $$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  kde $\gamma = 2.0$ potlačuje váhu snadno klasifikovatelných legitimních negativů a $\alpha_t = 0.25$ vyvažuje zastoupení tříd.
- **Optimalizátor:** AdamW, learning rate $\eta = 10^{-3}$, weight decay $10^{-4}$.

### 6.3 Sekvenční Flow-Transformer
- **Vstup:** Sekvence flowlet tokenů délky $L \le 128$, embedding dimenze $d_{\text{model}} = 128$.
- **Architektura:**
  - 3 Transformer Encoder vrstvy, 4 attention hlavy, feed-forward dimenze $512$, dropout $0.1$.
  - Lineární klasifikační hlava nad `[CLS]` tokenem.
- **Účel:** Schopnost zachytit dlouhodobé kontextové závislosti mezi bursty bez nutnosti obřího pre-trainingu na terabajtech dat.

### 6.4 Host-Based Bayesovský agregátor (Meta-Klasifikátor)
Pro potlačení Base Rate Fallacy v reálném nasazení se predikce jednotlivých toků směřujících na stejnou cílovou IP adresu akumulují v čase. Pro hostitele $H$ a $M$ zachycených toků:
$$\text{Score}(H) = \sum_{k=1}^{M} \log \left( \frac{p_k}{1 - p_k} \right)$$
Hostitel je označen za WebTunnel Bridge pouze tehdy, když $\text{Score}(H) > \tau_{\text{host}}$. Tím se FPR na úrovni hostitele snižuje pod $10^{-6}$.

---

## 7. Experimentální evaluační rámec a Base Rate stresové testování

### 7.1 Metriky hodnocení
- **Precision-Recall křivky (PR-AUC):** Primární metrika v asymetrickém prostředí (namísto zavádějící ROC-AUC nebo celkové Accuracy).
- **Precision at Fixed Low FPR:** Přesnost modelu vyhodnocená při striktně fixovaných hodnotách falešných poplachů:
  $$\text{FPR} \in \{10^{-3}, 10^{-4}, 10^{-5}\}$$
- **Dekompozice Matice Záměn (Confusion Matrix):** Detailní rozbor falešných pozitiv rozpadlý na jednotlivé podtřídy Hard Negatives (které protokoly model nejčastěji plete s WebTunnelem).

### 7.2 Stresové testování Base Rate Fallacy (FDR Projekce)
Matematická projekce míry falešného odhalení (False Discovery Rate, $\text{FDR} = 1 - \text{Precision}$) při konstantní senzitivitě modelu $\text{TPR} = 0.95$:

| Prevalence $\alpha$ | Popis prostředí | Provozní míra FPR | Očekávaná Precision | False Discovery Rate (FDR) | Provozní posouzení pro ISP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$10^{-2}$ (1 %)** | Malá institucionální síť | $10^{-3}$ | 90.56 % | 9.44 % | Akceptovatelné pro cílený monitoring. |
| **$10^{-2}$ (1 %)** | Malá institucionální síť | $10^{-5}$ | 99.89 % | 0.11 % | Vynikající provozní stav. |
| **$10^{-4}$ (0.01 %)** | Hraniční uzel běžného ISP | $10^{-3}$ | 8.68 % | **91.32 %** | **Nepoužitelné:** 9 z 10 poplachů je falešných! |
| **$10^{-4}$ (0.01 %)** | Hraniční uzel běžného ISP | $10^{-4}$ | 48.74 % | 51.26 % | Hraniční použitelnost. |
| **$10^{-4}$ (0.01 %)** | Hraniční uzel běžného ISP | $10^{-5}$ | 90.48 % | 9.52 % | Plně provozuschopný stav. |
| **$10^{-6}$ (0.0001 %)** | Národní páteřní síť (Tier-1) | $10^{-4}$ | 0.94 % | **99.06 %** | **Kritické selhání:** Masivní kolaterální škody. |
| **$10^{-6}$ (0.0001 %)** | Národní páteřní síť (Tier-1) | $10^{-5}$ | 8.68 % | 91.32 % | Vyžaduje Host-Based agregaci! |

### 7.3 Benchmark výpočetní náročnosti (Line-Rate proveditelnost)
Měření reálné latence inferenčního kroku (ms/flow), paměťové náročnosti (RAM/VRAM) a propustnosti (flows/sec) na CPU (1 jádro vs více jader) a GPU:
- **Kaskádová inspekční architektura (Cascaded Pipeline):**
  - **Fáze 1 (Line-Rate filtr):** Rychlý XGBoost na CPU vyřadí $> 99.5\,\%$ jednoznačně legitimního provozu při propustnosti $> 50\,000\,\text{flows/s}$.
  - **Fáze 2 (Hloubková analýza):** Pouze sporné toky s vysokým rizikem jsou předány 1D-CNN / Transformeru na GPU.

---

## 8. Strukturální analýza zranitelností WebTunnelu a návrh protiopatření

Diplomová práce vyvrcholí technickým rozborem zjištěných slabin WebTunnelu a konkrétními doporučeními pro vývojáře *Tor Pluggable Transports*:

### 8.1 Identifikované zranitelnosti WebTunnelu
1. **Kvantizace buněk Toru (514-byte Cell Quantization):** Data v Toru jsou balena do buněk o velikosti 514 B. Při přenosu vznikají v histogramu délek paketů charakteristické spektrální špičky v násobcích $514\,\text{B} + \text{TLS framing}$, což ostře kontrastuje s kontinuální distribucí běžného HTTPS/WebSocket provozu.
2. **Circuit Setup Burst:** Fáze navazování Tor okruhu (příkazy `CREATE2`, `CREATED2`, `SETNODES`) vykazuje v prvních 10–15 paketech spojení deterministický vzorec výměny zpráv s fixním poměrem upstream a downstream dat.
3. **Absence plné HTTP/2 dynamiky:** WebTunnel typicky provozuje jediný dlouhotrvající stream bez odesílání legitimních řídicích rámců (`PRIORITY`, dynamické `WINDOW_UPDATE`, `PING`) a bez paralelních proudů pro statické webové assety.
4. **Host IP stálost:** Dlouhodobé směrování veškerého tunelovaného provozu na jedinou statickou IP adresu mostu usnadňuje host-based agregaci cenzora.

### 8.2 Návrh doporučení a protiopatření pro Pluggable Transports
- **Adaptivní výplň kadrů (Adaptive Intra-frame Padding):** Zavedení náhodného dynamického paddingu na úrovni HTTP/2 datových kadrů v rozmezí 1 až 128 bajtů řízeného normálním rozdělením. Rozbije 514B spektrální špičky při celkovém nárůstu datové režie $< 7\,\%$.
- **Dynamické štěpení paketů (Packet Splitting & Chunking):** Dělení aplikačních dat do vícenásobných TLS záznamů různé délky před předáním TCP vrstvě (rozbití hranic buněk).
- **Emulace HTTP/2 řídicího provozu (Framing Fingerprint Matching):** Aktivní injektování fiktivních rámců `SETTINGS`, `PING` a `WINDOW_UPDATE` s časováním odpovídajícím prohlížeči Chromium.
- **Randomizace počátečního vyrovnávacího bufferu (Setup Buffer Shaping):** Agregace a odesílání úvodních Tor příkazů s drobným časovým zpožděním (jitter buffer) pro rozbití deterministického otisku handshake.
- **Rotace efemérních mostů a IP adres:** Integrace principů ze systémů Snowflake / Obscura pro dynamickou rotaci IP adres a zamezení časové kumulace statistik na straně cenzora.

---

## 9. Harmonogram a fázový plán realizace diplomové práce

```
+---------------------------------------------------------------------------------------------------+
| FÁZE 1: Rešerše a teoretický základ (Měsíc 1–2)                                                   |
| - Zmapování principů WebTunnel, HTTPT, Pluggable Transports a uTLS.                               |
| - Analýza Base Rate Fallacy, Traffic Fingerprintingu a metod Jansen et al. (NDSS 2024).           |
| - Výstup: Kapitoly 1 a 2 teoretické části práce.                                                  |
+---------------------------------------------------------------------------------------------------+
| FÁZE 2: Výstavba testbedu a automatizace sběru dat (Měsíc 3–4)                                    |
| - Implementace Docker Compose / Linux Namespaces testbedu (ns-client, ns-router, ns-bridge, ns-leg).|
| - Skripty pro Playwright (Web Browsing, Bulk, Shell) + servery pro Hard Negatives.                |
| - Nastavení tc-netem profilů a bezeztrátového capture do PCAPng.                                  |
| - Výstup: Funkční pipeline pro automatizovaný sběr surových datových sad.                         |
+---------------------------------------------------------------------------------------------------+
| FÁZE 3: Sanitizace dat a Feature Extraction Pipeline (Měsíc 5)                                     |
| - Implementace Python modulů (dpkt/scapy) pro L2–L4 stripping, směrovou a časovou normalizaci.   |
| - Extrakce 48 flow-level statistik + generování tenzorů pro 1D-CNN a tokenů pro Transformer.     |
| - Realizace Destination-Split a Session-Split.                                                    |
| - Výstup: Čisté trénovací, validační a testovací datasety bez úniku informací.                    |
+---------------------------------------------------------------------------------------------------+
| FÁZE 4: Vývoj modelů a trénování (Měsíc 6–7)                                                      |
| - Implementace a ladění XGBoost / LightGBM baseline.                                              |
| - Implementace PyTorch 1D-CNN s Focal Loss a optimalizátorem AdamW.                               |
| - Implementace sekvenčního Flow-Transformeru.                                                     |
| - Implementace Host-Based Bayesovského agregačního algoritmu.                                     |
| - Výstup: Natrénované modely a skripty pro reprodukovatelnou inferenci.                          |
+---------------------------------------------------------------------------------------------------+
| FÁZE 5: Experimentální vyhodnocení, profiling a analýza slabin (Měsíc 8)                          |
| - Měření PR-AUC, Precision @ Fixed FPR (10^-3, 10^-4, 10^-5), Confusion Matrix.                   |
| - Matematická projekce Base Rate Fallacy (FDR tabulky pro různé alfa).                            |
| - Profilování latence (ms/flow) a propustnosti (flows/sec) na CPU/GPU.                           |
| - Analýza zranitelností protokolu a formulace doporučení pro vývojáře.                            |
| - Výstup: Kompletní grafy, tabulky a podklady pro praktickou část práce.                          |
+---------------------------------------------------------------------------------------------------+
| FÁZE 6: Kompletace textu práce a příprava obhajoby (Měsíc 9)                                      |
| - Sepsání úvodu, metodiky, výsledků, diskuze a závěru v souladu se šablonou PřF JU.               |
| - Kontrola citační etiky a formálních náležitostí.                                                |
| - Příprava prezentace a podkladů k obhajobě.                                                      |
| - Výstup: Hotová diplomová práce připravená k odevzdání.                                          |
+---------------------------------------------------------------------------------------------------+
```

---

## 10. Klíčová literatura a vědecké reference

1. **WAILS, Ryan; SULLIVAN, George Arnold; SHERR, Micah; JANSEN, Rob.** *On Precisely Detecting Censorship Circumvention in Real-World Networks.* In: 31st Annual Network and Distributed System Security Symposium (NDSS 2024). San Diego, CA, USA, 2024. *(FOCI 2024 Best Practical Award)*.
2. **FROLOV, Sergey; WUSTROW, Eric.** *HTTPT: A Probe-Resistant Proxy.* In: 10th USENIX Workshop on Free and Open Communications on the Internet (FOCI 20). USENIX Association, 2020.
3. **ZHOU, Guangmeng; GUO, Xiongwen; LIU, Zhuotao; LI, Tong; LI, Qi; XU, Ke.** *TrafficFormer: An Efficient Pre-trained Model for Traffic Data.* In: 2025 IEEE Symposium on Security and Privacy (SP). IEEE, 2025.
4. **KAMALI, Sina; BARRADAS, Diogo.** *Huma: Censorship Circumvention via Web Protocol Tunneling with Deferred Traffic Replacement.* In: 33rd Annual Network and Distributed System Security Symposium (NDSS 2026). 2026.
5. **XUE, Di, et al.** *Open World Traffic Analysis on Tor Hidden Services.* In: Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security (CCS '22). ACM, 2022.
6. **SIRINAM, Payap, et al.** *Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning.* In: Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS '18). ACM, 2018.
7. **LOTFOLLAHI, Mohammad, et al.** *Deep Packet: A Novel Approach for Encrypted Traffic Classification Using Deep Learning.* In: Soft Computing, 24(3), 2020.
8. **AXELSSON, Stefan.** *The Base-Rate Fallacy and the Difficulty of Intrusion Detection.* In: ACM Transactions on Information and System Security (TISSEC), 3(3), 2000.
9. **THE TOR PROJECT.** *WebTunnel Pluggable Transport Specification and Deployment Documentation.* 2023–2024. Dostupné z: `https://community.torproject.org/relay/setup/webtunnel/`
10. **BAMSOFTWARE.** *PTPerf: On the performance evaluation of Tor Pluggable Transports.* 2023. Dostupné z: `https://www.bamsoftware.com/software/dnstt/2309.14856.pdf`
