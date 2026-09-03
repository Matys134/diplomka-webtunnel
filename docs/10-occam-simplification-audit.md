# Technický audit překombinovanosti a redundance (Occamova břitva)

**Role auditora:** Seniorní výzkumník síťové bezpečnosti, pragmatický oponent a technický auditor ML systémů.  
**Předmět prověrky:** Architektura projektu, smysluplnost jednotlivých modulů a experimentů, eliminace akademického balastu a zhodnocení, zda projekt netrpí syndromem *„kanónu na vrabce“*.

---

## 1. Katedrála kolem jednoduché pravdy?

Stručně a otevřeně: **Ano, kolem elementárního fyzikálního faktu byla postavena velkolepá katedrála.** 

Jádro celého vědeckého problému lze shrnout do jediné věty:  
> *Tor kvantizuje data do 514bajtových buněk a WebTunnel je bez polstrování posílá přes WebSocket/TLS, takže 92,65 % jeho odchozích TLS záznamů leží na diskrétní mřížce $L = 44 + 514k$, zatímco běžný web na ní neleží téměř nikdy.*

Kolem této fundamentální pravdy v repozitáři vyrostlo:
- 50 tabulkových rysů v XGBoostu,
- 1D-CNN s Focal Loss,
- Flow-Transformer s multi-head attention a `[CLS]` tokenem,
- SHAP hodnoty a gradientové saliency mapy,
- kontrola permutace pořadí paketů (*order-shuffle*),
- ablace TLS handshaku,
- doménová generalizace napříč profily (LTE, Lossy WAN),
- 2-vrstvá kaskáda L1/L2 s měřením latence na mikrosekundy,
- Bayesovská agregace a simulace na 1M spojeních,
- a 6 validačních bran G1–G6 s 15 jednotkovými testy.

### Působí to jako tříbení, nebo jako bezradný slepenec?
Pravda leží uprostřed a závisí na tom, **jak je tento aparát orámován**:

1. **Kde je aparát obhajitelný (a nutný):**  
   Zadávací protokol diplomové práce explicitně **nařizoval**: *„implementace referenční metody detekce a návrh pokročilého DL klasifikátoru (např. 1D-CNN, TrafficFormer)“*. Pokud by student přinesl pouze dvouřádkový skript `if (L - 44) % 514 == 0: return True`, akademická komise by práci hodila na hlavu pro nesplnění zadání („Kde máte ty slibované neuronové sítě a Transformer?“). Existence modelů a testbedu je tedy **povinnou daní zadání**.
2. **Kde je aparát silný:**  
   V síťovém výzkumu je zvykem, že když autor ohlásí 100% přesnost z hluboké neuronové sítě, recenzent to okamžitě smete jako „overfitting na laboratorní šum“. To, že autor má v ruce validační brány (G5 socket-disjoint, G6 provenienci, G3 label shuffle), ořezání handshaku a mřížkové pravidlo, slouží jako **neprůstřelné brnění proti nařčení z laboratorního podvodu**.
3. **Kde už Occamova břitva pláče (over-engineering):**  
   Celá řada experimentů v `4_evaluation/` produkuje stále totéž: **tabulky plné 100,00 % a grafy s prázdnými chybovými oblastmi**. V okamžiku, kdy 5 různých experimentů dokazuje stejnou věc z pěti různých úhlů pohledu a všechny končí na 100,00 %, začíná to působit jako křečovitá snaha vygenerovat co nejvíce stránek grafů bez nového informačního přínosu.

---

## 2. Konkrétní redundance v kódu a evaluaci

Podrobme jednotlivé komponenty nekompromisní inženýrské kritice:

### 2.1 Kaskádový filtr (`evaluate_cascaded_pipeline.py`) — Fata morgána
- **Koncept:** Naivní myšlenka kaskády (vycházející např. z Anderson & McGrew) říká: *„Levný L1 filtr na CPU odbaví 95 % jednoznačného provozu, sporné toky pošleme na drahé GPU do hluboké neuronky.“*
- **Realita v repozitáři:** Podívejme se na [`table_cascaded_pipeline.tex`](file:///home/matys/antigravity/diplomka/0_thesis_text/tables/table_cascaded_pipeline.tex):  
  - L1 (XGBoost): **100.0 % toků**  
  - L2 (1D-CNN na GPU): **0.0 % toků**
- **Verdikt:** **Slepá ulička.** Kaskáda, jejíž druhý stupeň se v praxi nespustí ani pro jeden jediný paket, není kaskádou. Nemá žádný ekonomický ani inženýrský význam provozovat GPU inferenci pro 0 toků. Experiment pouze simuloval latenci GPU na datech, která tam za reálného provozu nikdy nedorazí.

### 2.2 Souběh 1D-CNN a Transformeru (`architectures.py`) — Formální nutnost, inženýrská duplicita
- **Realita:**  
  - Vstup: oba modely dostávají identický tenzor `[batch, 200, 2]` (délky a IAT).
  - Výkon: oba dosahují v 5-fold CV přesně $100,0 \pm 0,0\,\%$, $PR\text{-}AUC = 1,000$.
  - Náročnost: 1D-CNN zvládne **592 345 toků/s** (latence 1,7 µs), Flow-Transformer zvládne **140 365 toků/s** (latence 7,1 µs).
- **Verdikt:** Transformer nepřináší proti 1D-CNN vůbec nic kromě 4× vyšší výpočetní režie a 10× většího počtu parametrů.  
  *Zadání jej však jmenovitě vyžaduje.* Správný pragmatický přístup proto není Transformer smazat, ale **otočit jeho interpretaci v textu**: Transformer v projektu neslouží jako „hrdina“, ale jako důkaz, že pro detekci kvantovaného provozu je masivní sekvenční pozornost (*self-attention*) naprosto zbytečný a neekonomický kanón na vrabce.

### 2.3 SHAP a Gradient Saliency (`explain_models.py`) — Pseudosmysluplné cvičení
- **Realita:** Skript počítá Tree SHAP pro XGBoost a gradientové mapy pro 1D-CNN/Transformer na datech, kde třídy odděluje triviální mřížka a odchylka v ClientHello.
- **Verdikt:** **Balast.** Pokud máme exaktní uzavřenou formuli $L = 44 + 514k$ odvozenou ze specifikace Toru, nepotřebujeme heuristické aproximační metody typu SHAP, aby nám sdělily, že nejdůležitější rys je `up_lattice_frac` (43,2 % váhy) nebo `up_len_p10`. Vypadá to jako povinná položka ze šablony „moderní diplomka o ML“, ale vědeckou hodnotu práce to nijak nezvyšuje.

### 2.4 Tabulky se samými 100,00 % — Degenerované ablace
1. [`table_cross_profile_generalization.tex`](file:///home/matys/antigravity/diplomka/0_thesis_text/tables/table_cross_profile_generalization.tex):
   - Broadband: 100,0 %, LTE: 100,0 %, Lossy: 100,0 %.  
   - *Důvod:* Ztrátovost ani jitter nemění délky reassemblovaných TLS záznamů. Tabulka neříká nic nového.
2. [`table_handshake_comparison.tex`](file:///home/matys/antigravity/diplomka/0_thesis_text/tables/table_handshake_comparison.tex):
   - S handshakem: 100,0 %, Bez handshaku: 100,0 %.  
   - *Důvod:* Po ořezání handshaku zbývá mřížka buněk Toru (S1) i 164B Upgrade (S3). Opět monotónní řada stovek.
3. [`table_order_shuffle.tex`](file:///home/matys/antigravity/diplomka/0_thesis_text/tables/table_order_shuffle.tex):
   - Seřazeno: 100,0 %, Promícháno: 100,0 %.  
   - *Důvod:* Všechny signály v datech jsou multisetové invarianty.

---

## 3. Co lze bezpečně vyhodit / zjednodušit (Inventura repozitáře)

Rozdělme kódovou základnu do tří zřetelných kategorií:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                    KÓDOVÁ ZÁKLADNA                      │
                  └────────────────────────────┬────────────────────────────┘
                                               │
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         ▼                                     ▼                                     ▼
  [SKUPINA A: PILÍŘE]                 [SKUPINA B: KONTROLY]                 [SKUPINA C: BALAST]
  Bez nich práce padá                 Důkaz metodické čistoty               Zralé na ořezání
  ─────────────────────               ─────────────────────                 ───────────────────
  • Docker testbed (Go)               • test_gates.py (G5, G6)              • explain_models.py (SHAP)
  • sanitizer.py (TCP stream)         • evaluate_post_handshake.py          • evaluate_cascaded_pipeline.py
  • lattice_rule.py (mřížka)          • evaluate_order_shuffle.py           • evaluate_cross_profile.py
  • train_xgboost + 1D-CNN            • evaluate_det_curve.py (Low-FPR)     • duplicitní base-rate skripty
  • simulate_censor_deployment.py     • checks/expected_invariants.py
  • evaluate_before_after_defenses.py
```

### Skupina A: Absolutní pilíře (Ponechat v plné parádě)
Bez těchto komponent by práce ztratila vědeckou váhu nebo nesplnila zadání:
1. **Docker testbed a autoritativní pipeline (`1_testbed/`, `2_data_pipeline/sanitizer.py`, `build_dataset.py`):** Základ experimentální reprodukovatelnosti. TCP reassembly je klíčový inženýrský příspěvek.
2. **Deterministické pravidlo mřížky (`3_models/lattice_rule.py`):** **Srdce celé práce.** Důkaz, že k odhalení WebTunnelu netřeba GPU ani trénovací váhy, ale 2 instrukce.
3. **Srovnání modelů (`train_xgboost.py`, `train_1d_cnn.py`, `train_transformer.py`, `cross_validate.py`):** Plní formální zadání diplomky a demonstruje výpočetní propustnost (od 0,8 µs po 7,1 µs).
4. **Simulace páteřního cenzora na 1M spojení (`4_evaluation/simulate_censor_deployment.py`):** Jediný experiment, který usvědčuje laboratorních 100 % z naivity a vysvětluje Axelssonův klam základní míry (97,7 % nevinných obětí při $M=1$ vs. 0 při $M \ge 3$).
5. **Evaluace obran s adaptivním protivníkem (`4_evaluation/evaluate_before_after_defenses.py`):** Zásadní kapitola o paddingu (pád statika na 0 %) a coalescingu (cena 120 ms latence).

### Skupina B: Metodické kontroly (Ponechat v repozitáři, ale netlačit do popředí)
Tyto skripty jsou skvělé jako „pojistka při obhajobě“, pokud se komise začne vyptávat na artefakty:
- [`checks/test_gates.py`](file:///home/matys/antigravity/diplomka/checks/test_gates.py) a [`checks/split_integrity.py`](file:///home/matys/antigravity/diplomka/checks/split_integrity.py): Důkaz, že split je socket-disjoint a data neunikají.
- [`evaluate_post_handshake.py`](file:///home/matys/antigravity/diplomka/4_evaluation/evaluate_post_handshake.py): Rychlý důkaz, že modely nestojí jen na ClientHello otisku.
- [`evaluate_order_shuffle.py`](file:///home/matys/antigravity/diplomka/4_evaluation/evaluate_order_shuffle.py): Důkaz, že sítě se neučí fixní n-gramový prefix.
- [`evaluate_det_curve.py`](file:///home/matys/antigravity/diplomka/4_evaluation/evaluate_det_curve.py): Formální vykreslení cenzurní křivky s vyznačením meze $1/n$.

### Skupina C: Zbytný balast a kandidáti na redukci
Tyto věci práci nerozšiřují, pouze ji ředí:
1. **[`3_models/explain_models.py`](file:///home/matys/antigravity/diplomka/3_models/explain_models.py) (SHAP a Saliency mapy):** Doporučuji úplně vyřadit z hlavní argumentace. V repozitáři může zůstat, ale v prezentaci/obhajobě o něm netřeba ztratit slovo.
2. **[`4_evaluation/evaluate_cascaded_pipeline.py`](file:///home/matys/antigravity/diplomka/4_evaluation/evaluate_cascaded_pipeline.py) (Kaskáda):** Když L2 odbaví 0,0 % toků, nelze tvrdit, že jsme ověřili kaskádu. Buď to otevřeně přiznat jako slepou uličku saturovaného úkolu, nebo tento experiment nezmiňovat jako funkční architekturu.
3. **[`4_evaluation/evaluate_cross_profile.py`](file:///home/matys/antigravity/diplomka/4_evaluation/evaluate_cross_profile.py) (NetEm profily):** Výsledky jsou identické s broadband profilem. Stačí jedna věta v textu, není nutné držet samostatnou tabulku generující samé 100%.
4. **Duplicita base-rate skriptů:** V repozitáři existují dva skripty: `evaluate_base_rate_fallacy.py` (původní analytický) a `simulate_censor_deployment.py` (nová páteřní simulace na 1M toků). Druhý jmenovaný je výrazně názornější, komplexnější a zahrnuje v sobě jak base-rate sweep, tak agregaci i životní cyklus obran. První je fakticky jeho podmnožinou.

---

## 4. Závěrečné pragmatické doporučení: Jak projekt očistit

Pokud má projekt působit jako sebevědomý, elegantní inženýrský výzkum s jasným tahem na branku (a nikoliv jako přehlídka všech funkcí scikit-learnu a PyTorchu), doporučuji následující očištění:

### 1. Sjednotit příběh kolem jednoho silného poselství
Příběh projektu nesmí znít: *„Zkusili jsme XGBoost, CNN a Transformer a všechno mělo 100 %.“*  
Příběh musí znít:  
> **„Prokázali jsme, že WebTunnel má v současné implementaci zásadní strukturální slabinu: emituje nepolstrované buňky Toru na mřížce $L = 44 + 514k$. K jeho spolehlivé detekci není zapotřebí žádné hluboké učení ani GPU — postačí deterministické pravidlo o dvou celočíselných operacích na CPU (propustnost > 1,2M toků/s). Složité neuronové sítě (1D-CNN, Transformer) pouze znovu objevují tentýž invariant za cenu obrovské výpočetní neefektivity.“**

Tato formulace okamžitě obrací zdánlivou nevýhodu (100% saturaci všech modelů) v obrovskou přednost a originální vědecký přínos práce.

### 2. Konkrétní kroky k redukci kódu a artefaktů:
1. **Opravit syntaktický bug v exportu tabulky:** V [`evaluate_before_after_defenses.py`](file:///home/matys/antigravity/diplomka/4_evaluation/evaluate_before_after_defenses.py) doplnit uzavírací tagy `\hline \end{tabular} \end{table}` (jak bylo nalezeno v prvním auditu).
2. **Zakotvit hierarchii skriptů:**
   - Sloučit nebo jasně oddělit roli `simulate_censor_deployment.py` (hlavní páteřní simulace pro prezentaci a text) a `evaluate_base_rate_fallacy.py` (interní statistický mezikrok).
   - Skripty `explain_models.py` a `evaluate_cascaded_pipeline.py` nechat v repozitáři pouze jako archivní/doplňkové, ale netvořit kolem nich klíčové kapitoly.
3. **Zachovat validační brány jako standard kvality:**
   Brány v `checks/` jsou naopak skvělé. Nemazat! Je to přesně ta věc, která odlišuje poctivého výzkumníka od amatéra, co bezhlavě natrénuje náhodná data.

### Shrnutí:
Projekt **není nefunkční monstrum**, ale je v něm několik vrstev historického nánosu z doby, kdy se hledalo, proč modely saturují. Pokud z reflektorů stáhneš zbytné věci (SHAP, nefunkční L2 kaskádu, redundantní profily) a do záře reflektorů postavíš **analýzu mřížky buněk Toru + páteřní simulaci Base Rate Fallacy + vyčíslení ceny obran (120 ms)**, projekt bude působit jako krystalicky čistá, špičková inženýrská práce, která obhájí své závěry před jakoukoliv komisí.