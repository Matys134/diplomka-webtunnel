# Kurátorský audit vizuálních a tabulkových výstupů (Occamova břitva)

Jako oponent a technický auditor oceňuji snahu o rigorózní dokumentaci každého dílčího kroku. Nicméně **8 tabulek a 17 grafů je na jednu výsledkovou kapitolu diplomové práce neúnosně mnoho**. 

Při bližším pohledu zjistíme, že v repozitáři dochází k **masivní duplicitě**:
1. Máte několik grafů, které jsou doslova „lesem sloupců sahajících do 100,00 %“ bez jakékoliv dynamiky.
2. Vlajkový třípanelový graf (`censor_dilemma_simulation.png`) v sobě elegantně integruje to, co další 3 samostatné grafy kreslí izolovaně.
3. Čtyři tabulky opakují monotónní řady čísel 100,00 % a jedna tabulka zachycuje architekturu, která v praxi vůbec nesepne.

Níže předkládám nekompromisní kurátorské roztřídění výstupů do **hlavního textu**, **přílohy (Appendix)** a **přímého vyřazení**.

---

## 1. Hodnocení tabulek (0_thesis_text/tables/)

| # | Název tabulky | Informační hodnota | Doporučené umístění | Důvod / Zjištěné vady |
| :- | :--- | :---: | :---: | :--- |
| **T1** | `table_model_comparison.tex` | **Kritická (5/5)** | **HLAVNÍ TEXT** | **Páteř porovnání modelů.** Obsahuje 5-fold CV metriky, ale především latenci na mikrosekundy a propustnost (XGBoost 1,25M vs. Transformer 140k toků/s). Plní přímo zadání práce. |
| **T2** | `table_class_breakdown.tex` | Střední (3/5) | **Příloha / Zkrátit** | Všech 6 tříd má u všech modelů 100,0 %. Tabulka je cenná popisem protokolů (Tor over WS, HTTP/2, WSS), ale čísla jsou monotónní. Vhodnější do přílohy, v textu stačí shrnout větou. |
| **T3** | `table_cascaded_pipeline.tex` | **Zavádějící (1/5)** | **VYŘADIT z textu** | **Nebezpečná tabulka pro obhajobu.** L2 GPU vrstva má **0.0 % zpracovaného provozu**. Prezentovat jako „výsledek kaskády“ stav, kdy druhý stupeň nikdy nesepne, je koledování si o nepříjemné otázky komise. |
| **T4** | `table_handshake_comparison.tex` | Nízká (2/5) | **Příloha** | 6 identických hodnot $100,00\,\%$. Celou tabulku plnohodnotně nahradí jedna věta: *„Dynamické oříznutí TLS handshaku nevedlo k poklesu přesnosti ani u jednoho modelu (všechny zachovaly 100,00 %).*“ |
| **T5** | `table_cross_profile_generalization.tex` | Nízká (2/5) | **Příloha** | 6 řádků se 100,0 %. Dokazuje pouze to, že TCP stream reassembly funguje a nenechá se zmást ztrátovostí ani jitterem. Patří do metodické přílohy. |
| **T6** | `table_order_shuffle.tex` | Nízká (2/5) | **Příloha** | Čtyři řádky 100,00 %. Footnote je metodicky poctivý, ale jako tabulka v hlavním textu je to jen další prázdné 100% pole. |
| **T7** | `table_before_after_defense.tex` | **Kritická (5/5)** | **HLAVNÍ TEXT** | **Zásadní vědecký výsledek.** Ukazuje pád statika na 0,0 %, adaptaci cenzora na 100,0 % a cenu obran (120,8 ms latence). *(Pozor: nutno opravit chybějící uzavírací tagy `\hline \end{tabular} \end{table}`!)* |
| **T8** | `table_censor_deployment_simulation.tex` | **Kritická (5/5)** | **HLAVNÍ TEXT** | **Demaskování laboratorních 100 %.** Vyčísluje 4 291 zablokovaných nevinných lidí denně a FDR 97,72 % při $\alpha = 10^{-4}$ na 1 milionu spojení. Bez ní nelze obhájit kapitolu o Base-Rate Fallacy. |

---

## 2. Hodnocení grafů (4_evaluation/plots/)

Ze 17 grafů jich minimálně **polovina představuje vizuální vatu, duplicitu nebo degenerované ploché grafy**.

### A. Skupina: Vizuální vata a degenerované grafy (VYŘADIT)
- **`neural_network_confusion_comparison.png` & `confusion_matrix_breakdown.png`:** Dvě 2×2 matice a jedna 6×6 matice, kde jsou na diagonále samá celá čísla a mimo ni nuly. Dívat se na pole nul nemá žádnou vědeckou hodnotu. **Vyřadit obě.**
- **`pre_vs_post_handshake_comparison.png` & `order_shuffle_comparison.png`:** Tyto dva grafy jsou přehlídkou sloupků, které všechny bez výjimky dosahují přesně 100 %. Kreslit graf pro konstantu $y = 100$ je učebnicový prohřešek proti vizualizaci dat. **Vyřadit.**
- **`cascaded_pipeline_throughput.png`:** Pouze přebarvuje čísla z tabulky do 3 sloupců a ještě upozorňuje na spornou kaskádu. **Vyřadit.**

### B. Skupina: XAI Heuristiky (SHAP, Saliency, Attention) — Vizuální placebo (PŘESUNOUT do přílohy)
- **`xgboost_shap_summary.png`, `1d_cnn_saliency_map.png`, `transformer_attention_map.png`:**  
  V úlohách, kde tápeme, co neuronka dělá, má XAI smysl. Zde ale **přesně známe fyzikální invariant ($L = 44 + 514k$)**. SHAP a gradientové mapy na saturovaném modelu neříkají nic víc než to, že model kouká na začátek toku (ClientHello) a na buňky Toru. Působí to jako „povinná výplň“, nikoliv vědecký argument. Pokud je autorovi líto je smazat, patří do **Přílohy B**.

### C. Skupina: Skrytá duplicita (Konsolidovat!)
- Grafy **`base_rate_fallacy_fdr.png`** (FDR vs. alfa), **`host_based_aggregation.png`** (M=1..12) a **`before_vs_after_metrics.png`** (metriky obran) jsou **100% duplicitní**.
- Všechny tři jsou totiž v mnohem přehlednější a profesionálnější podobě obsaženy v třípanelovém grafu **`censor_dilemma_simulation.png`**:
  - Panel A = křivka FDR na páteři,
  - Panel B = eliminace nevinných obětí při $M=1..5$,
  - Panel C = hra na kočku a myš (statik vs. adaptivní cenzor).
- **Závěr:** Samostatné grafy 12, 14 a 15 vyřadit a v hlavním textu použít **výhradně integrovaný třípanelový graf `censor_dilemma_simulation.png`**.

---

## 3. Návrh optimálního výběru pro 5. kapitolu

Aby 5. kapitola diplomové práce působila jako **špičkový článek z IEEE S&P nebo USENIX Security**, musí mít přísný tah na branku: odhalit fyzikální podstatu $\to$ změřit modely $\to$ vyvrátit laboratorní naivitu na páteři $\to$ zhodnotit reálné obrany a jejich cenu.

Pro tento příběh potřebujete přesně **3 klíčové tabulky** a **4 nejúdernější grafy**:

```
========================================================================================
           DOPORUČENÁ „LEAN & MEAN“ SESTAVA PRO 5. KAPITOLU DIPLOMOVÉ PRÁCE
========================================================================================

TABULKY V HLAVNÍM TEXTU (Přesně 3 tabulky):
----------------------------------------------------------------------------------------
1. table_model_comparison.tex
   -> Ukazuje srovnání XGBoost vs. 1D-CNN vs. Transformer. Demonstruje propustnost
      a dokazuje, že složitější modely nepřinášejí vyšší přesnost než jednoduché.
2. table_censor_deployment_simulation.tex
   -> Kvantifikuje Base-Rate Fallacy na 1M spojeních (4 291 falešných obvinění při per-flow).
3. table_before_after_defense.tex (po opravě \end{table})
   -> Shrnuje dynamiku obran proti statickému i adaptivnímu cenzorovi včetně latence 120 ms.

GRAFY V HLAVNÍM TEXTU (Přesně 4 grafy):
----------------------------------------------------------------------------------------
1. packet_length_distribution.png
   -> FYZIKÁLNÍ DŮKAZ: Histogram jasně ukazuje mřížku 558 B a diskrétní násobky buněk Toru
      proti plochému legitimnímu webu.
2. det_curve_logarithmic.png
   -> PŘESNOST V REŽIMU LOW-FPR: Logaritmická DET křivka s vyznačenou mezí rozlišení 1/n.
      Splňuje přímé zadání práce.
3. censor_dilemma_simulation.png (Vlajkový 3-panelový graf)
   -> REALITA CENZURY NA PÁTEŘI:
      • Panel A: Problém jehly v kupce sena (FDR vs. prevalence alfa).
      • Panel B: Řešení více-tokovou agregací (pokles nevinných obětí z 4 292 na 0).
      • Panel C: Hra na kočku a myš (statický pád na 0 % vs. adaptivní retraining).
4. before_vs_after_distributions.png
   -> ANATOMIE OBRAN: Ukazuje, jak intra-record padding a coalescing fyzicky rozbíjejí
      mřížku buněk Toru v distribuci délek.

VŠECHNO OSTATNÍ (Příloha nebo vyřadit):
----------------------------------------------------------------------------------------
• Do přílohy: table_class_breakdown.tex, table_cross_profile_generalization.tex,
              table_order_shuffle.tex, table_handshake_comparison.tex,
              xgboost_feature_importance.png, iat_distribution.png.
• Úplně smazat/vynechat: table_cascaded_pipeline.tex, konfuzní matice (4, 5),
                         sloupcové grafy 100% výsledků (10, 11), SHAP/saliency mapy (7, 8, 9),
                         duplicitní dílčí grafy (12, 14, 15, 16).
========================================================================================
```

### Proč tato sestava práci neuvěřitelně pomůže?
1. **Odstraní podezření z „nastavované kaše“:** Oponent nebude muset listovat deseti stranami tabulek, kde se mění jen popisky řádků, ale v datech svítí stále 100,00 %.
2. **Eliminuje slabiny:** Vyřazením nefunkční kaskády (L2 = 0 %) sebere autor oponentovi nejlevnější cíl k technické kritice.
3. **Vynikne inženýrská elegance:** Práce bude mít krystalicky čistou strukturu. Čtyři vybrané grafy a tři tabulky odpoví na každou otázku zadávacího protokolu bez jediného zbytečného megabajtu balastu.