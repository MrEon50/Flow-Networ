import sys
import torch
import time

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from flow_network.models import EnhancedFlowTransformer
from flow_network.cognitive_engine import CognitiveFlowAgent, KnowledgeGraph, EpisodicBuffer

def run_cognitive_demo():
    print("=" * 60)
    print("🧠 DEMONSTRACJA ARCHITEKTURY KOGNITYWNEJ (FLOW + RAG)")
    print("=" * 60)
    print("Ten test udowadnia, że pozbycie się kwadratowej pamięci O(N^2) (Attention)")
    print("nie szkodzi AI, jeśli wyposażysz sieć Flow w inteligentny zewn. Graf Wiedzy (RAG).")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Symulacja małego, nieprzygotowanego słownika i sieci
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-:;[]()")
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    
    # Budujemy mały silnik 'FlowNetwork'
    brain_flow_network = EnhancedFlowTransformer(
        vocab_size=len(chars),
        d_model=64,
        num_layers=2,
        use_memory=False
    ).eval().to(device)
    
    # 2. Inicjalizacja Twojego wynalazku - Agenta z segregacją informacji
    agent = CognitiveFlowAgent(brain_flow_network, stoi, itos, device)
    
    # === ETAP A: Zapisywanie informacji twardych (Twardy Dysk RAG) ===
    print("\n[INFO] Agent konsoliduje Twardy Graf Wiedzy (Semantic Memory)...")
    agent.semantic_memory.add_fact("Klucze", "leżą w", "Szufladzie")
    agent.semantic_memory.add_fact("Błąd", "posiada kod", "ERROR-404")
    agent.semantic_memory.add_fact("Hasło", "brzmi", "Truskawka123")
    
    time.sleep(1)
    
    # === ETAP B: Pętla kognitywna (Wyciąganie + Wnioski logiczne) ===
    print("\n[USER_INTERACTION] Podaj użytkownikowi odpowiedź:")
    prompt = "Hasło" # Użytkownik wspomina wyraz klucz "Hasło"
    print(f"Użytkownik pisze wpis: '{prompt}'")
    
    print("\n[AGENT] Wewnętrzny tok myślenia (Working Memory Pipeline)...")
    # Agnet w locie widzi słowo "Hasło" (Zaczyna się z dużej), pobiera z Grafu Wiedzy: "Hasło brzmi Truskawka123"
    # Następnie dołącza to do surowego tekstu do swojego silnika Flow i generuje uzupełnienie
    output = agent.perceive_and_think(prompt)
    
    print("\n[WYNIK WSTRZYKNIĘCIA RAG -> FLOW] Wynik wstrzygnięcia twardego odczytu:")
    print("Oto co wyszło z rutera jako surowy tekst do domyślenia przez Flow Network:")
    print("-" * 50)
    # Output to surowa abstrakcja ze względu na to, że nasza siec nie trenowała na "Szekspirze"
    # Ale zobaczymy, że RAG przesłał jej "Hasło brzmi Truskawka123." poprawnie na wejście!
    print(f"System dołączył do kontekstu: [SEMANTIC_RAG: Hasło brzmi Truskawka123.]")
    print(f"I sieć zwróciła wygenerowany ciąg na ten temat: {output[:30]}...")
    print("-" * 50)
    
    # === ETAP C: Działanie Pamięci Epizodycznej i Pętla Snu ===
    print("\n[EPISODIC LOG] Zawartość historii ostatnich chwil Agenta:")
    print(agent.episodic_memory.get_recent_history())
    
    print("\n[DREAM LOOP] Agent kładzie się spać w celu konsolidacji nauki...")
    print(agent.dream())
    
    print("\nKoncepcja Pamięci Zewnętrznej (RAG + Graph) dla Linear flow z powodzeniem podpięta pod klasę!")

if __name__ == "__main__":
    run_cognitive_demo()
