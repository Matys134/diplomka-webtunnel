# Oponentský audit: Realita nasazení stavového DPI cenzora na páteři

Jako oponent musím říci: **Váš skeptický instinkt je naprosto správný.** 

Výsledek simulace (0 zablokovaných nevinných, 100 % odhalených WebTunnelů, 625 KB RAM) je působivý inženýrský mezikrok, ale v prezentované podobě představuje **učebnicový laboratorní ideál**. V reálné páteřní síti (Tier-1/Tier-2 tranzit nebo velký mobilní operátor) by tento cenzor v podobě, v jaké je nasimulován, narazil na tvrdé fyzikální a síťové limity.

Pojďme podrobit jednotlivé předpoklady nekompromisnímu rozboru:

---

## 1. Carrier-Grade NAT (CGNAT) a sdílené IP adresy

Simulátor tiše předpokládá ideální svět: **1 klientská IP adresa = 1 fyzický uživatel**.  
V moderním internetu (zejména v mobilních 4G/5G sítích a na optice s IPv4 nedostatkem) se však používá **CGNAT (RFC 6598)**. Za jedinou veřejnou IP adresou běžně koexistují stovky až tisíce koncových zařízení.

Tento fakt staví Leaky-Bucket cenzora před neřešitelné dilema dvou protichůdných selhání:

### A. Naředění podezření (Dilution Effect / False Negatives)
Váš algoritmus má v sobě dekrement skóre:
```python
# simulate_censor_deployment.py: řádek 258
scores[h] = max(0.0, scores[h] - 1.2)
```
Pokud za jednou CGNAT IP adresou sedí 1 uživatel WebTunnelu (vygeneruje 5 podezřelých spojení za 5 minut, tj. $+22,5$ bodu) a současně 2 000 běžných uživatelů brouzdajících po webu, YouTube a sociálních sítích (vygenerují 5 000 běžných toků za minutu):
- Proud legitimních toků okamžitě stlačí akumulované skóre dané IP adresy na **0.0**.
- Podezření se v záplavě negativních důkazů doslova rozpustí.
- **WebTunnel uživatel se za CGNATem stane pro tento filtr zcela neviditelným.**

### B. Masivní kolaterální škoda (Collateral Catastrophe / DoS)
Pokud by cenzor dekrementaci odstranil a spoléhal pouze na exponenciální časový rozpad ($\tau_{1/2} = 15$ min):
- Jakmile skóre překročí $\tau_{block} = 9,5$, cenzor uvalí 1hodinový blackhole na celou veřejnou IP adresu.
- **Výsledek:** Kvůli jednomu disidentovi či novináři používajícímu Tor odřízne cenzor od internetu celou univerzitní kolej, nemocnici nebo 5 000 mobilních předplatitelů. V civilizované síti to znamená okamžitý kolaps zákaznické podpory; v autoritářském státě to poškodí bankovní a státní infrastrukturu.

> **Oponentský závěr k bodu 1:** Pokud cenzor agreguje podle *Source IP*, CGNAT buď detekci zcela paralyzuje, nebo způsobí nepřípustné kolaterální škody. Řešením by bylo sledovat kombinaci `(Client_IP, Client_Port)` nebo provádět klientskou DPI demultiplexaci, což však exponenciálně zvyšuje paměťové nároky.

---

## 2. Skutečná paměťová náročnost: Stavová tabulka vs. TCP Reassembly

Tvrzení, že *„stavový cenzor vyžaduje pouhých 625 KB RAM pro 10 000 hostů“*, je technická iluze. Započítáváte pouze **řídicí rovinu (Control Plane)**, ale zcela ignorujete **datovou rovinu (Data Plane)**.

### A. Paměťová past TCP Stream Reassembly na 100Gbps lince
Aby cenzor vůbec mohl spočítat `up_lattice_frac` (tedy zjistit, zda má TLS záznam 558 B), **musí nejprve provést TCP reassembly**. Nemůže věřit jednotlivým IP paketům (fragmentace, segmentace MTU, out-of-order doručení).
- Na páteřním spoji 100 Gbps protéká cca **15 milionů paketů za sekundu**.
- Běžný počet souběžně otevřených TCP spojení se pohybuje mezi **500 000 až 2 000 000 aktivních toků**.
- Pokud pro reassembly out-of-order paketů a posuvných oken alokujete pro každý tok konzervativních **16 KB až 64 KB vyrovnávací paměti**, potřebuje DPI box:
  $$\text{RAM}_{\text{reassembly}} = 1\,000\,000 \text{ toků} \times 32\text{ KB} \approx \mathbf{32\text{ GB ultra-rychlé paměti (QDR/DDR5)}}\text{!}$$
- Těch 625 KB je pouze drobný hash-map čítačů na samém konci řetězce. Skutečné úzké hrdlo je v udržení TCP streamů při linkové rychlosti.

### B. Zranitelnost vůči State-Exhaustion DoS útoku
Stavový middlebox s Leaky-Bucket tabulkou je primárním terčem pro útoky na vyčerpání stavu:
1. **SYN Flood / IP Churn:** Útočník začne generovat toky s náhodnými podvrženými zdrojovými IP adresami. Pokud cenzor alokuje záznam v hashovací tabulce pro každou novou IP, tabulka o 10 000 položkách okamžitě přeteče (hash collisions, memory thrashing).
2. **Indukce falešných obvinění (Censorship Weaponization):**  
   Útočník vezme IP adresy legitimních vládních webů, bank či DNS serverů a pošle z nich směrem ven podvržené pakety o délce 558 B. Cenzor po 3 paketech tyto klíčové servery na 1 hodinu zablokuje. Cenzor se tak stává zbraní v rukou útočníka pro Denial-of-Service.

---

## 3. Asymetrické směrování na páteři (Asymmetric Routing)

V Tier-1 a Tier-2 sítích je asymetrický routing běžným jevem: pakety od klienta k serveru (upload) jdou přes jednoho tranzitního operátora (např. Telia/Arelion), zatímco odpovědi (download) se vracejí přes jiného (např. Cogent či Lumen) vlivem BGP hot-potato routingu.

- **Dopad na detekci:**  
  Naštěstí pro cenzora leží mřížka buněk Toru primárně v odchozím směru ($L_{\text{up}} = 44 + 514k$). Cenzor sledující klientský uplink teoreticky nepotřebuje vidět odpovědi ze serveru, aby zaznamenal 558B záznam.
- **Kde asymetrie láme vaz:**  
  Bez zpětného toku (downloadu) cenzor nevidí **SYN-ACK ani klientské potvrzení (ACK)**. Nemůže tedy ověřit, zda TCP handshake skutečně proběhl a zda spojení existuje, nebo zda jde o podvržený (spoofovaný) paket. Pokud by cenzor penalizoval IP adresy na základě jednosměrného toku bez potvrzeného 3-way handshaku, je zranitelný vůči triviálnímu spoofingu zmíněnému v bodě 2B.

---

## 4. Adaptivní únik ze strany WebTunnelu (Adversarial Evasion)

Pokud provozovatel WebTunnelu ví, že cenzor nasadil tento stavový Leaky-Bucket mechanismus, má k dispozici několik triviálních protiopatření:

```
                    ┌─────────────────────────────────────────────────────────┐
                    │               TAKTIKY ÚNIKU WEBTUNNELU                  │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         ▼                                       ▼                                       ▼
   [1. LOW & SLOW]                        [2. BRIDGE HOPPING]                     [3. CELL PADDING]
   Spacing > 15 minut                     Rotace IP adres bridge                  Zrušení mřížky L=44+514k
   ──────────────────                     ──────────────────────                  ────────────────────────
   Exponenciální rozpad                   Agregace na cíli selže;                 up_lattice_frac = 0 %
   vynuluje skóre dříve,                  na klientovi naráží na                  Detektor nezaznamená
   než dosáhne tau_block.                 CGNAT kolaterál.                        vůbec žádný HIT.
```

1. **Režim „Low & Slow“ (Časové rozptýlení):**  
   Poločas rozpadu filtru je $\tau_{1/2} = 15$ minut. Pokud WebTunnel drží jediné dlouhožijící TCP spojení (místo otevírání nových), nebo mezi novými spojeními počká 20 minut, skóre v Leaky-Bucketu exponenciálně vyprchá k nule. Cenzor nikdy nenasbírá potřebných 9,5 bodu.
2. **Rotace cílů (Bridge Hopping):**  
   Pokud by cenzor agregoval podle *Destination IP* (jak navrhují Wails et al. v NDSS 2024 pro ochranu před CGNATem), WebTunnelu stačí rotovat IP adresy bridge (např. přes CDN, domain fronting nebo pool bridge serverů). Žádná jednotlivá IP nenasbírá dostatečný počet zásahů.
3. **Zabudovaný padding (Fundamentální smrt filtru):**  
   Jak ukazuje váš vlastní experiment v tabulce `table_before_after_defense.tex`: jakmile WebTunnel zavede intra-record padding (1–128 B), podíl `up_lattice_frac` klesne z 92,65 % na **0,0 %**.  
   V tom okamžiku v simulátoru platí `hit = False` pro 100 % toků. **Celý stavový stroj, Leaky-Bucket i časový rozpad jsou okamžitě vyřazeny ze hry**, protože do akumulátoru nepřiteče ani jediný bod podezření.

---

## 5. Jak tento výsledek poctivě orámovat v textu diplomové práce

Tento simulátor je vynikající inženýrská práce, ale v textu 5. kapitoly **nesmí být prezentován triumfalisticky** jako „konečné řešení cenzury“. Pokud to napíšete stylem: *„Navrhli jsme filtr za 625 KB, který bezchybně zničí WebTunnel“*, oponent z praxe vás roznese na kopytech právě na CGNATu a paměti TCP reassembly.

### Správný vědecký narativ:
Prezentujte simulaci jako **studii teoretických mezí a odhalení asymetrie mezi útočníkem a obráncem**:

1. **Uveďte simulátor jako demonstraci toho, CO BY CENZOR MUSET UDĚLAT:**  
   > *„Simulace dokazuje, že naivní per-flow inspekce na páteři selhává kvůli Base-Rate Fallacy (FDR > 70 %). Cenzor je matematicky nucen přejít k časové stavové agregaci (např. Leaky-Bucket model).“*
2. **Kvantifikujte ideální laboratorní podmínky:**  
   > *„V modelovém prostředí s dedikovanými IP adresami postačí akumulace 2–3 po sobě jdoucích relací během 15minutového poločasu rozpadu k bezchybné identifikaci s nulovým kolaterálním poškozením (MTTD ~ 3 minuty).“*
3. **Okamžitě a dobrovolně pojmenujte limity (tím komisi vyrazíte trumfy z rukou):**  
   > *„Tento ideální výsledek však naráží na realitu provozu ISP:  
   > a) **CGNAT:** Sdílení IP adres tisíci uživateli buď podezření naředí běžným provozem, nebo vystaví operátora nepřijatelnému riziku odpojení nevinných uživatelů.  
   > b) **Skutečná paměťová cena:** Těchto 625 KB reprezentuje pouze řídicí stav; skutečné náklady leží v gigabajtech paměti nutné pro TCP stream reassembly při rychlostech 100 Gbps.  
   > c) **Křehkost vůči adaptaci:** Účinnost celého stavového aparátu stojí a padá s přítomností mřížky $558\text{ B}$. Jakmile protokol zavede kryptografický padding, stavový filtr ztratí jakýkoliv vstupní signál.“*

### Shrnutí oponenta:
Implementace stavového Leaky-Bucket simulátoru v `simulate_censor_deployment.py` je **výborná a posouvá diplomku na úroveň publikovatelného výzkumu**.  
Její hodnota však nespočívá v tom, že „vyřešila cenzuru“, ale v tom, že **přesně zmapovala podmínky a limity, za kterých je stavová cenzura na páteři teoreticky možná a kde v reálném světě selhává**. Pokud ji takto v práci prodáte, práce bude mít obrovský vědecký kredit.