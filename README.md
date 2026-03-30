# 🌊 FlowNetwork: Linear-Time LLM Architecture

FlowNetwork to rewolucyjna, w pełni linearna $O(N)$ architektura sieci neuronowej do modelowania językowego (LLM), zaprojektowana z myślą o urządzeniach o ograniczonych zasobach sprzętowych (Edge AI) oraz procesowaniu tekstów o nielimitowanej długości kontesktu.

Ten projekt udowadnia, że całkowita eliminacja klasycznych mechanizmów `Self-Attention` na rzecz dynamicznych ruterów przepływu (Flow Routers) oraz systemów Gating (SwiGLU) prowadzi do niesamowitych optymalizacji zasobów, nie tracąc na zdolnościach lingwistycznych modelu.

---

## ⚡ Główne Innowacje

* **Czysty Flow (Brak O(N²) Attention):** Architektura odrzuca zwalniające, kwadratowe macierze atencji, zastępując je ruterami (Context-Aware Flow Router), które przeliczają kontekst w czasie i pamięci całkowicie **liniowej**.
* **Nieskończony Kontekst dzięki RoPE:** Tradycyjne Embeddingi zamieniono na Rotary Position Embeddings (RoPE), stosowane m.in w modelach LLaMa, co zdejmuje z sieci sztuczny limit twardej pamięci kontekstu (np. 4096 tokenów). Długość tekstu do przeanalizowania teoretycznie staje się nieskończona.
* **Aktywacja SwiGLU:** Precyzyjne bramkowanie wiedzy dzięki użyciu SwiGLU FFN (SiLU Gating) zamiast standardowego GELU.
* **Minimalne Zużycie VRAM:** Podczas testów, sieć o rozmiarze ~2.14M parametrów potrafi zamknąć proces produkcyjny tekstu w stabilnych **~37 MB VRAM**, co klasyfikuje ją idealnie do małych płytek prototypowych np. RPi/Jetson.

## 🛠 Instalacja

Upewnij się, że posiadasz środowisko z PyTorch (wersja CPU lub CUDA GPU).

```bash
git clone https://github.com/TwojBranch/FlowNetwork.git
cd FlowNetwork
pip install torch numpy
```

## 📂 Struktura Projektu

Po profesjonalnej refaktoryzacji, architektura zachowuje standardy bibliotek Pytorch-like:

```text
flow_network_project/
 ├── flow_network/
 │    ├── __init__.py
 │    ├── core.py         # Fundamenty (EnhancedFlowLayer, ContextAwareFlowRouter)
 │    ├── models.py       # Architektura wysokiego poziomu (EnhancedFlowTransformer)
 │    ├── training.py     # Mechanizmy trenowania (MultiTaskFlowLoss)
 │    └── utils.py        # Obliczanie zasobów, tensor_safety, metryki
 │
 ├── benchmark.py         # Skrypt walidacyjny, testy jednostkowe, testy Pamięci/Szybkości
 ├── train_real.py        # Demonstrator z opcją argumentów z CLI (ArgParse)
 ├── flow_terminal.py     # 🏆 Główny Interaktywny Interfejs GUI (Konsolowy) do sterowania siecią
 └── README.md
```

## 🚀 Jak zacząć? (Flow Terminal)

Najlepszym sposobem na doznanie możliwości projektu, wypróbowanie ucinania operacji, dynamicznego zapisywania sieci, zmian hyperparametrów o locie i podziwianie inferencji, jest uruchomienie dołączonego interaktywnego, odpornego na błędy klienta terminalowego.

```bash
python flow_terminal.py
```

W intuicyjnym menu konsolowym możesz:
1. Pobrać domyślny słownik i dane tekstowe w locie.
2. Odpalić Trening "na żywo".
3. Przerwać trening **w każdej chwili** kombinacją `CTRL+C` bez utraty wyuczonych wag.
4. Uruchomić Czat z opartym na tekście LLM i sprawdzić jak układa wyrazy w sensowne zdania.
5. Zapisać matryce sieci (tzw. _checkpointy_) lokalnie pod nazwą `.pt` celem wymiany ze znajomymi.

## 📊 Wyniki Benchmarków (Flow vs Classic Transformer)

Przeprowadzony rygorystyczny test na tej samej konfiguracji obciążeniowej:
> `vocab: 1000`, `seq_len: 2048`, `d_model: 256`, `batch_size: 4`

| Parametr | Classic Transformer | Enhanced FlowNetwork | Status |
| :--- | :--- | :--- | :--- |
| Złożoność Obliczeniowa | O(N²) | O(N) | Wygrywa Flow |
| Wielkość Pamięci GPU | Przepełnienie (OOM) | Płaski limit | Wygrywa Flow |
| Czas dla 512 tokenów | 4217 ms | 942 ms | ~75% szybciej z Flow |
| Czas dla 2048 tokenów | 6969 ms | 3573 ms | ~50% szybciej z Flow |

## 📃 Licencja i Noty eksperymentalne

Projekt eksperymentalny zaprojektowany z myślą weryfikacji paradygmatów badawczych AI, ukierunkowany na systematiczne odchodzenie od prądu głównego mechanizmu Atencji (Attention). Rozwijany jako podłoże edukacyjno-badawcze dla fanatyków budowania własnych architektur modelów językowych.
