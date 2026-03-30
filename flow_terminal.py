import os
import sys
import torch
import torch.nn.functional as F
import time
import urllib.request
import threading

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from flow_network.models import EnhancedFlowTransformer

# --- GLOBAL STANY ---
MODEL = None
OPTIMIZER = None
STOI = {}
ITOS = {}
VOCAB_SIZE = 0
DATA_TENSOR = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ustawienia Domyślne
SETTINGS = {
    'data_path': 'tinyshakespeare.txt',
    'batch_size': 32,
    'seq_len': 128,
    'eval_interval': 100,
    'learning_rate': 1e-3,
    'd_model': 128,
    'layers': 4,
    'heads': 8,
    'patterns': 4
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def initialize_data():
    global STOI, ITOS, VOCAB_SIZE, DATA_TENSOR
    
    path = SETTINGS['data_path']
    if not os.path.exists(path):
        if path == 'tinyshakespeare.txt':
            print("📥 Wczytuję TinyShakespeare...")
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            urllib.request.urlretrieve(url, path)
        else:
            print(f"❌ Plik {path} nie istnieje!")
            return False

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    VOCAB_SIZE = len(chars)
    STOI = {ch: i for i, ch in enumerate(chars)}
    ITOS = {i: ch for i, ch in enumerate(chars)}
    
    encoded = [STOI[c] for c in text]
    DATA_TENSOR = torch.tensor(encoded, dtype=torch.long)
    print(f"📚 Załadowano: {len(text):,} znaków. Unikalne znaki (słownik): {VOCAB_SIZE}")
    return True

def build_model():
    global MODEL, OPTIMIZER
    if VOCAB_SIZE == 0:
        print("⚠️ Najpierw załaduj dane, aby zbudować słownik!")
        return

    print("🧠 Budowa lub reset architektury Flow...")
    MODEL = EnhancedFlowTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=SETTINGS['d_model'],
        num_layers=SETTINGS['layers'],
        max_seq_len=0, 
        num_heads=SETTINGS['heads'],
        num_patterns=SETTINGS['patterns'],
        dropout=0.1,
        use_memory=True
    ).to(DEVICE)
    
    OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), lr=SETTINGS['learning_rate'])
    total_params = sum(p.numel() for p in MODEL.parameters())
    print(f"📦 Zbudowano Sieć: rozpiętość {total_params / 1e6:.2f} M parametrów gotowa na {DEVICE}!")

def get_batch(split):
    n = int(0.9 * len(DATA_TENSOR))
    train_data = DATA_TENSOR[:n]
    val_data = DATA_TENSOR[n:]
    data = train_data if split == 'train' else val_data
    
    ix = torch.randint(len(data) - SETTINGS['seq_len'], (SETTINGS['batch_size'],))
    x = torch.stack([data[i:i+SETTINGS['seq_len']] for i in ix])
    y = torch.stack([data[i+1:i+SETTINGS['seq_len']+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss():
    out = {}
    MODEL.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(split)
            logits, _ = MODEL(X)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    MODEL.train()
    return out

@torch.no_grad()
def generate_text(prompt="\n", max_new_tokens=200):
    if MODEL is None or VOCAB_SIZE == 0:
        return "⚠️ Model nie został zbudowany."
        
    MODEL.eval()
    # Kodowanie ze sprawdzaniem znaków nieznanych
    encoded_prompt = [STOI.get(c, 0) for c in prompt]
    idx = torch.tensor(encoded_prompt, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -512:] 
        logits, _ = MODEL(idx_cond)
        logits = logits[:, -1, :] 
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)
    
    out = ''.join([ITOS.get(i, '?') for i in idx[0].tolist()])
    MODEL.train()
    return out

def run_training():
    if MODEL is None:
        if not initialize_data(): return
        build_model()
        
    try:
        iters = int(input("Ile iteracji willst wykonać? (np. 1000): "))
    except ValueError:
        return

    print(f"\n⚡ Rozpoczynanie treningu na {iters} iteracji!")
    print("ZATRZYMAJ W DOWOLNYM MOMENCIE WCISKAJĄC [CTRL+C] (zapiszemy postęp!)")
    time.sleep(2)
    start_time = time.time()
    
    try:
        for iter_num in range(iters):
            if iter_num % SETTINGS['eval_interval'] == 0 or iter_num == iters - 1:
                losses = estimate_loss()
                vram_usage = f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB" if DEVICE == 'cuda' else "N/A"
                elapsed = time.time() - start_time
                throughput = (iter_num * SETTINGS['batch_size'] * SETTINGS['seq_len']) / elapsed if elapsed > 0 else 0
                
                print(f"[{iter_num:4d}/{iters}] Strata TR: {losses['train']:.4f} | VAL: {losses['val']:.4f} | VRAM: {vram_usage} | Speed: {throughput:.0f} tok/s")
                
                if iter_num > 0 and iter_num % (SETTINGS['eval_interval'] * 2) == 0:
                    print("\n--- PRÓBKA (DREAM) ---")
                    print(generate_text(max_new_tokens=100))
                    print("----------------------\n")

            xb, yb = get_batch('train')
            logits, _ = MODEL(xb)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

            OPTIMIZER.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), max_norm=1.0)
            OPTIMIZER.step()
            
        print("\n🎉 Trening zakończony!")
    except KeyboardInterrupt:
        print("\n\n🛑 PRZERWANO TRENING (CTRL+C)!")
        print("Osiągnięty postęp (wagi) został zablokowany i zapamiętany w pamięci sieci.")
    
    input("\nWciśnij Enter, by wrócić do menu...")

def chat_interface():
    if MODEL is None:
        print("⚠️ Zbuduj wpierw i wytrenuj model (Opcja 1)!")
        input("\nWciśnij Enter, by wrócić...")
        return
        
    clear_screen()
    print("═" * 50)
    print(" 🌊 TERMINAL CHAT (Interfejs Inferencji)")
    print(" Podaj początek zdania, a sieć Flow dopisze resztę.")
    print(" Wpisz 'exit' aby powrócić do menu.")
    print("═" * 50)
    
    while True:
        prompt = input("\n👤 Ty: ")
        if prompt.lower() == 'exit':
            break
            
        if not prompt: prompt = "\n"
        print("🤖 Flow:", end=" ")
        result = generate_text(prompt=prompt, max_new_tokens=300)
        # Obetnij prompt z wyniku, żeby pokazać tylko nowo wygenerowany tekst (opcjonalnie)
        print(result[len(prompt):]) 
        
def save_model():
    if MODEL is None:
        print("⚠️ Brak modelu w pamięci do zapisania.")
        return
    path = input("Podaj nazwę pliku (np. base_model.pt): ")
    if not path.endswith('.pt'): path += '.pt'
    
    checkpoint = {
        'model_state_dict': MODEL.state_dict(),
        'optimizer_state_dict': OPTIMIZER.state_dict(),
        'vocab_size': VOCAB_SIZE,
        'stoi': STOI,
        'itos': ITOS,
    }
    torch.save(checkpoint, path)
    print(f"✅ Model + słownik zapisano poprawnie jako {path}!")
    input("\nWciśnij Enter...")

def load_model():
    global MODEL, OPTIMIZER, VOCAB_SIZE, STOI, ITOS
    path = input("Podaj nazwę pliku (np. base_model.pt): ")
    if not os.path.exists(path):
        print("❌ Plik nie istnieje.")
        input("\nWciśnij Enter...")
        return
        
    print(f"📥 Ładowanie {path}...")
    checkpoint = torch.load(path, map_location=DEVICE)
    VOCAB_SIZE = checkpoint['vocab_size']
    STOI = checkpoint['stoi']
    ITOS = checkpoint['itos']
    
    build_model()
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    OPTIMIZER.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✅ Model i wagi wczytano do pamięci pomyślnie!")
    input("\nWciśnij Enter...")

def settings_menu():
    while True:
        clear_screen()
        print("⚙️ USTAWIENIA ARCHITEKTURY")
        for i, (k, v) in enumerate(SETTINGS.items()):
            print(f"{i+1}. {k}: {v}")
        print("0. Wróć do menu")
        
        c = input("Wybierz opcję do edycji: ")
        if c == '0': break
        try:
            idx = int(c) - 1
            key = list(SETTINGS.keys())[idx]
            new_val = input(f"Nowa wartość dla {key} (obecnie {SETTINGS[key]}): ")
            
            if type(SETTINGS[key]) == int:
                SETTINGS[key] = int(new_val)
            elif type(SETTINGS[key]) == float:
                SETTINGS[key] = float(new_val)
            else:
                SETTINGS[key] = new_val
                
            # Jeśli zmieniono architekturę, trzeba usunąć model
            if key in ['d_model', 'layers', 'heads', 'patterns']:
                global MODEL
                MODEL = None
                print("⚠️ Usunięto model z pamięci RAM ze względu na zmianę struktury.")
        except:
            pass

def main():
    while True:
        clear_screen()
        print("╔════════════════════════════════════════════════════════╗")
        print("║ 🌊 FLOW NETWORK TERMINAL - Nanoskalowalne AI (Local)   ║")
        print("╚════════════════════════════════════════════════════════╝")
        state_model = "Aktywny w RAM" if MODEL is not None else "Brak"
        print(f" 🖥️  Urządzenie: {DEVICE.upper()} |  📦 Sieć: {state_model}\n")
        
        print("  [1] ⚡ Trenuj Sieć (Auto-kontynuacja)")
        print("  [2] 💬 Czat (Testuj inferencję)")
        print("  [3] 💾 Zapisz stan sieci do pliku")
        print("  [4] 📂 Wczytaj stan sieci z pliku")
        print("  [5] ⚙️  Ustawienia/Hiperparametry")
        print("  [0] 🚪 Wyjście")
        
        choice = input("\n🤖 Wybierz polecenie: ")
        
        if choice == '1': run_training()
        elif choice == '2': chat_interface()
        elif choice == '3': save_model()
        elif choice == '4': load_model()
        elif choice == '5': settings_menu()
        elif choice == '0': break

if __name__ == '__main__':
    main()
