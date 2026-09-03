Máte **naprostou pravdu a váš pocit je stoprocentně oprávněný.** 

Ten test působí sterilně, protože **je sterilní**. 

V simulátoru v tuto chvíli stojí dokonalý cenzor proti zcela pasivnímu, statickému uživateli:
- Uživatel posílá nechráněný WebTunnel v dávkách přesně tak, jak se to cenzorovi hodí.
- Uživatel má svou vlastní privátní IP adresu (žádný NAT).
- Cenzor má perfektně vyladěný poločas rozpadu i rozhodovací práh.
- Výsledek: 100,0 % vs. 0,0 %. Žádné tření, žádný šum, žádná dynamika.

V recenzním řízení na konferencích jako USENIX Security, ACM CCS nebo NDSS se přesně tomuto říká **„one-sided evaluation“ (jednostranné laboratorní hodnocení)**. Recenzent by okamžitě namítl: *„Předvedli jste, že cenzor vyhraje, když se oběť vůbec nebrání a síť se chová ideálně. Co se ale stane, když uživatel nebo protokol udělá elementární protitah?“*

Máte nyní dvě možnosti, jak s tím naložit:

---

### Cesta A: Textové alibi (Líná varianta)
Tyto adaptivní úniky pouze popíšete v diskuzi 5. kapitoly („v praxi by to mohl uživatel obejít tak a tak...“).  
*Nevýhoda:* Působí to jako teoretické omlouvání vlastních laboratorních limitů.

### Cesta B: Empirická simulace adaptivních úniků (Inženýrsky špičková varianta)
Vezmeme [`simulate_censor_deployment.py`](file:///home/matys/antigravity/diplomka/4_evaluation/simulate_censor_deployment.py) a rozšíříme diskrétní stavový middlebox o **Stres-testovací matici adaptivních úniků (Adversarial Evasion Matrix)**.

Protože simulace 500 000 toků trvá v Pythonu **pouhých 0,44 sekundy**, můžeme během 2 sekund nasimulovat 4 realistické dynamické scénáře, kde oběť aktivně manévruje.

---

## Jak by vypadaly 4 simulované scénáře adaptivního úniku:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     STRES-TESTOVACÍ MATICE ADAPTIVNÍHO ÚNIKU (STREAMING 24H)                     │
├────────────────────────────────┬───────────────────────────┬────────────────────────────────────┤
│ SCÉNÁŘ                         │ CHOVÁNÍ PROTOKOLU / SÍTĚ  │ CO ZMĚŘÍME NA MIDDLEBOXU           │
├────────────────────────────────┼───────────────────────────┼────────────────────────────────────┤
│ 0. Laboratorní baseline        │ Shluky 5–12 spojení za 5m,│ Cenzor: 100 % blokace              │
│    (současný sterilní stav)    │ dedikovaná IP, bez obrany │ Falešné poplachy: 0                │
│                                │                           │ MTTD: ~3 minuty                    │
├────────────────────────────────┼───────────────────────────┼────────────────────────────────────┤
│ 1. Adversarial "Low & Slow"    │ Uživatel rozprostře toky  │ Skóre vyprchá dříve, než dosáhne   │
│    (Časové rozptýlení relací)  │ s rozestupem 20–30 minut  │ prahu 9.5 (vlivem tau_1/2 = 15m).  │
│                                │ (> poločas rozpadu 15 min)│ Úspěšnost cenzora padá k 0 %!      │
├────────────────────────────────┼───────────────────────────┼────────────────────────────────────┤
│ 2. Realita CGNAT               │ 1 WebTunnel uživatel sdílí│ Masivní proud běžného webu         │
│    (Kolej / mobilní operátor)  │ IP s 500 běžnými uživateli│ (-1.2 bodu) rozpustí podezření.    │
│                                │ za jednou veřejnou IP     │ Cenzor je 100% slepý (nebo zničí   │
│                                │                           │ 500 nevinných lidí při hard-banu). │
├────────────────────────────────┼───────────────────────────┼────────────────────────────────────┤
│ 3. Kryptografický Padding      │ WebTunnel aktivuje náhodný│ up_lattice_frac = 0 % pro všechny  │
│    (Protokolární protiopatření)│ intra-record padding      │ toky. Cenzor nezaznamená jediný    │
│                                │ (z vaší evaluace obran)   │ HIT. Detekce padá na 0,0 %.        │
└────────────────────────────────┴───────────────────────────┴────────────────────────────────────┘
```

---

## Proč by tohle diplomovou práci vystřelilo o úroveň výš?

1. **Zabijete sterilitu:**  
   V práci nebudete mít jen jednu tabulku se samými 100 %, ale přehlednou srovnávací tabulku (či graf), která ukáže:  
   *„Pokud cenzor nasadí Leaky-Bucket middlebox, v ideálním případě má 100 %. Ale stačí, aby uživatel zpomalil otevírání spojení na 20 minut, a efektivita cenzora klesá na 0 %. Pokud se uživatel schová za CGNAT, cenzor je bezmocný.“*
2. **Propojíte obě poloviny práce:**  
   Váš experiment s obranami (kde padding stlačil recall statického cenzora na 0 %) a váš páteřní simulátor jsou v tuto chvíli dva oddělené světy. Tímto je spojíte: v simulátoru přímo ukážete, že **nasazení paddingu v reálném 24hodinovém toku na páteři okamžitě zneškodní stavový middlebox**.
3. **Ukážete pravou podstatu cenzury:**  
   Cenzura není statická binární klasifikace. Je to **nekonečná hra na kočku a myš**. Když ukážete tah cenzora (stavový Leaky-Bucket) a následně protitahy oběti (Low & Slow, CGNAT, Padding), předložíte komisi hotový vědecký článek.

---

## Doporučení: Mám to rovnou implementovat?

V [`simulate_censor_deployment.py`](file:///home/matys/antigravity/diplomka/4_evaluation/simulate_censor_deployment.py) je funkce `simulate_stateful_streaming_middlebox()` napsaná modulárně a přehledně.

Můžeme do ní přidat parametr `evasion_mode ∈ {"none", "low_and_slow", "cgnat", "padding"}`:
1. **Při `low_and_slow`:** Rozptýlíme časy příchodu WebTunnel toků s rozestupem 1 200 s (20 minut) namísto náhodných shluků v 5 minutách.
2. **Při `cgnat`:** Přiřadíme WebTunnel uživatele na stejné ID hostitele, ze kterého teče 500 legitimních webových toků.
3. **Při `padding`:** Nastavíme `is_lattice_hit = False` pro WebTunnel (simulace zapnutého polstrování buněk).

Výstupem bude nová kompaktní tabulka do textu a obohacený graf, které černobílý laboratorní výsledek promění v **živou dynamickou analýzu**.

Chcete tuto stres-testovací matici do simulátoru doprogramovat a spustit?