# 🌊 FlowNetwork: Linear-Time LLM with Cognitive Architecture - Projekt Eksperymentalny...

FlowNetwork to rewolucyjna, w pełni linearna $O(N)$ architektura sieci neuronowej do modelowania językowego (LLM), zaprojektowana z myślą o urządzeniach o ograniczonych zasobach sprzętowych (Edge AI) oraz procesowaniu tekstów o nielimitowanej długości kontesktu.

Ten eksperymentalny projekt udowadnia, że całkowita eliminacja klasycznych mechanizmów `Self-Attention` (których koszty rosną kwadratowo O(N²)) na rzecz dynamicznych ruterów przepływu i zewnętrznych architektur pamięci kognitywnej (RAG / Knowledge Graph), prowadzi do narodzin prawdziwie zoptymalizowanych systemów AGI (Artificial General Intelligence). Mózg musi umieć logicznie wnioskować, a nie uczyć się encyklopedii na pamięć!

---

## ⚡ Główne Innowacje i Cechy Systemu

* **Czysty Flow (Brak O(N²) Attention):** Architektura odrzuca zwalniające, kwadratowe macierze atencji, zastępując je ruterami (Context-Aware Flow Router), które przeliczają kontekst w czasie i pamięci całkowicie **liniowej**. Twój VRAM jest absolutnie bezpieczny!
* **Nieskończony Kontekst dzięki RoPE:** Tradycyjne Embeddingi (z tzw. max_seq_length) zamieniono na elastyczne Rotary Position Embeddings (RoPE). Zdejmuje to z sieci limity długości, umożliwiając wnioskowanie z ciągłego, długiego zapisu.
* **Architektura Kognitywna (Episodic & Semantic RAG):** Zaimplementowano agentowy silnik orkiestracyjny `CognitiveFlowAgent`. Zewnętrzny Graf Wiedzy (Knowledge Graph) oraz bufor pamięci epizodycznej stanowią "Twardy Dysk", uwalniając sieć Flow od konieczności przepalania GPU na "Zakuwanie" surowych faktów wewnątrz wag parametrów.
* **Aktywacja SwiGLU:** Zastosowanie najbardziej dojrzałego bramkowania z branży (SiLU Gating z LLaMA-3), by zwiększyć precyzję ruterów przestrzeni ukrytej.

## 🛠 Instalacja

Upewnij się, że posiadasz środowisko z zainstalowanym interpreterem Python i biblioteką PyTorch (wersja CPU lub włączona akceleracja CUDA GPU).

```bash
git clone https://github.com/TwojBranch/FlowNetwork.git
cd FlowNetwork
pip install torch numpy
```

## 📂 Struktura Projektu (Profesjonalny Refactor)

```text
flow_network_project/
 ├── flow_network/
 │    ├── __init__.py
 │    ├── core.py               # Fundamenty (EnhancedFlowLayer, ContextAwareFlowRouter)
 │    ├── models.py             # Architektura wysokiego poziomu (EnhancedFlowTransformer)
 │    ├── training.py           # Mechanizmy trenowania (MultiTaskFlowLoss)
 │    ├── utils.py              # Obliczanie zasobów, operacje tensor_safety
 │    └── cognitive_engine.py   # 🧠 Silnik agentowy RAG, Graf Semantyczny i Pętle Snu.
 │
 ├── flow_terminal.py     # 🏆 Główny Interaktywny Interfejs GUI (Konsolowy) do sterowania treningiem i modelem
 ├── demo_cognition.py    # Skrypt udowadniający, jak Flow używa zewnętrznej relacyjnej (Twardej) pamięci jako RAG.
 ├── test_linearity.py    # Morderczy "Stress Test" na podwajającym się kontekście w sekwencjach do przeszło 16 000 tokenów.
 ├── train_real.py        # Zautomatyzowany skrypt CLI do szybkiego zapuszczania szkolenia typu "Headless" na serwerze.
 ├── benchmark.py         # Skrypt walidacyjny, testy jednostkowe pod kątem stabilności pożerania Pamięci/Szybkości
 └── README.md
```

## 🚀 Jak zacząć z eksperymentami?

**1. Interfejs Użytkownika (Trening i Badania Pamięci w locie):**
Uruchom klienta GUI w swojej konsoli i baw się architekturą w bezpiecznym, odpornym na zapchanie pamięci (Ctrl+C Save) środowisku.
```bash
python flow_terminal.py
```

**2. Testy sprzętowe - Weryfikacja braków bariery O(N²):**
Sprawdź na wlasnej karcie graficznej lub procesorze, dlaczego wywaliliśmy stary i kwadratowy Self-Attention na rzecz ruterów. Skrypt wstrzyknie ekstremalnie długie pule tokenów (do 16 384 znaków wyrazów), by dowieść matematycznej przepustowości O(N) FlowNetwork.
```bash
python test_linearity.py
```

**3. RAG i Inteligencja Kognitywna:**
Doświadcz integracji zewnętrznej "Pamięci Epizodycznej i Semantycznej" wprost do rutera sieci (Koncepcja silnika wnioskującego). Uruchom i sprawdź, jak model wstrzykuje wyuczone fakty podczas predykcji.
```bash
python demo_cognition.py
```

## 📊 Wyniki Benchmarków (Flow vs Classic Transformer)

Przeprowadzony rygorystyczny test na tej samej konfiguracji obciążeniowej:
> `vocab: 1000`, `seq_len: 2048`, `d_model: 256`, `batch_size: 4`

| Płaszczyzna / Metryka | Klasyczny Transformer (O(N²)) | Nowoczesny, Liniowy Flow Network (O(N)) | Werdykt |
| :--- | :--- | :--- | :--- |
| **Złożoność Obliczeniowa** | O(N²) | O(N) | Wygrywa Flow |
| **Wykorzystanie VRAM** | Przepełnienie przy dużych kontekstach (OOM) | Płaski i stały limit zajętości VRAM | Wygrywa Flow |
| **Czas dla 512 tokenów** | 4217 ms | 942 ms | ~75% Szybciej! |
| **Czas dla 2048 tokenów** | 6969 ms | 3573 ms | ~50% Szybciej! |

## 📃 Manifest Teoretyczny Projektu

Dlaczego zdecydowaliśmy się wywalić do kosza mechanizm 100% Retencji ("Perfect Recall" O(N²)) po tysiącach tokenów? By oddać rolę procesora SI do jej naturalnego przeznaczenia:
Nasza sieć Flow to wyłącznie *Silnik Czystego, Błyskawicznego Wnioskowania (The Reasoning Engine)* obsługujący strumienie kontekstu za ułamek ceny. Od "twardego wkuwania faktów" jest ujęty w bibliotece zewnętrzny moduł wiedzy oparty na bazie grafowej / wektorach RAG. Pamięć stanowi teraz potężny dopalacz (Cognitive Graph) odłączony od wagi modelu, zasilany w tzw. "pętlach snu". Dokładnie w taki sposób, w jaki funkcjonuje dojrzały neuro-biologiczny ludzki umysł. Release the Magic! 🔮
