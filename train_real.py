import argparse
import os
import urllib.request
import torch
import torch.nn.functional as F
import time
import math
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
from flow_network.models import EnhancedFlowTransformer
from flow_network.utils import safe_tensor_to_int

parser = argparse.ArgumentParser(description='Trening Sieci FlowNetwork')
parser.add_argument('--data', type=str, default='tinyshakespeare.txt', help='Ścieżka do pliku w formacie tekstowym (.txt)')
parser.add_argument('--iters', type=int, default=2000, help='Liczba całkowitych iteracji (epok) podczas treningu')
parser.add_argument('--batch_size', type=int, default=32, help='Rozmiar pakietu podczas treningu')
parser.add_argument('--seq_len', type=int, default=128, help='Długość okna sekwencji na wejściu (kontekst)')
parser.add_argument('--eval_interval', type=int, default=200, help='Co ile iteracji sprawdzać i raportować stratę?')
parser.add_argument('--lr', type=float, default=1e-3, help='Współczynnik uczenia (Learning Rate)')

# Parametry Architektury
parser.add_argument('--d_model', type=int, default=128, help='Rozmiar wymiarów sieci przestrzennej')
parser.add_argument('--layers', type=int, default=4, help='Liczba warstw typu FlowLayer')
parser.add_argument('--heads', type=int, default=8, help='Liczba przepływów strumieni (głów)')
parser.add_argument('--patterns', type=int, default=4, help='Liczba matrycowych wzorców routingu we Flow')

args = parser.parse_args()

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_PATH = args.data
BATCH_SIZE = args.batch_size
SEQ_LEN = args.seq_len
MAX_ITERS = args.iters
EVAL_INTERVAL = args.eval_interval
LEARNING_RATE = args.lr
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

D_MODEL = args.d_model
NUM_LAYERS = args.layers
NUM_HEADS = args.heads
NUM_PATTERNS = args.patterns

print(f"🚀 Uruchamiam system na urządzeniu: {DEVICE}")

# 1. Pobieranie zbioru danych (Tiny Shakespeare)
if not os.path.exists(DATA_PATH):
    print("📥 Pobieranie zbioru danych (Szekspir)...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("✅ Pobrano.")

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"📚 Długość zbioru danych: {len(text):,} znaków")

# 2. Tokenizator Poziomu Znaków (Char-level Tokenizer)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"🔤 Rozmiar słownika (unikalne znaki): {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# 3. Przygotowanie Tensorów
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # Train/Val split
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([data_source[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([data_source[i+1:i+SEQ_LEN+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# 4. Inicjalizacja naszej Architektury
print("\n🧠 Inicjalizacja Enhanced Flow Transformer...")
model = EnhancedFlowTransformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    max_seq_len=0, # O(N) brak wymuszonego limitu
    num_heads=NUM_HEADS,
    num_patterns=NUM_PATTERNS,
    dropout=0.1,
    use_memory=True
)
model = model.to(DEVICE)

# Liczenie parametrów
total_params = sum(p.numel() for p in model.parameters())
print(f"📦 Zbudowano Sieć: rozpiętość {total_params / 1e6:.2f} M parametrów")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 5. Silnik Ewaluacji Loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(split)
            logits, _ = model(X)
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets = Y.view(B*T)
            loss = F.cross_entropy(logits_reshaped, targets)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# 6. Generacja Inferencyjna - testowanie co sieć wymyśliła
@torch.no_grad()
def generate_sample(prompt="\n", max_new_tokens=100):
    model.eval()
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        # Ze względu na nieskończony Flow - możemy podać całą dotychczasową sekwencję 
        # By przyspieszyć używamy maksymalnie ostatnie 512 znaków
        idx_cond = idx[:, -512:]
        logits, _ = model(idx_cond)
        # Bierzemy logity tylko z ostatniego tokenu z przewidywanej przyszłości
        logits = logits[:, -1, :] 
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)
    
    generated_text = decode(idx[0].tolist())
    model.train()
    return generated_text

# 7. Główna Pętla Trenująca z Monitoringiem
print("\n⚡ Rozpoczynam trening (zobaczysz tutaj sprzętowe cuda liniowego VRAMu)...")
start_time = time.time()

# Wyniki początkowe - "Belkot przed treningiem"
print(f"\\n--- PRÓBKA (PRZED TRENINGIEM) ---")
print(generate_sample(max_new_tokens=50))
print("---------------------------------\\n")

for iter_num in range(MAX_ITERS):
    
    if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
        losses = estimate_loss()
        
        # MONITORING SPRZĘTOWY
        vram_usage = "N/A"
        if DEVICE == 'cuda':
            vram_usage = f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
            
        elapsed = time.time() - start_time
        tokens_processed = iter_num * BATCH_SIZE * SEQ_LEN
        throughput = tokens_processed / elapsed if elapsed > 0 else 0
        
        print(f"It {iter_num:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | "
              f"VRAM: {vram_usage} | Speed: {throughput:.0f} tok/s")
        
        # Podglądamy co sieć generuje i jak mózg syntetyzuje angielski :D
        if iter_num > 0 and iter_num % (EVAL_INTERVAL * 2) == 0:
            print(f"\\n--- PROGRES W GENEROWANIU SZEKSPIRA (iter {iter_num}) ---")
            print(generate_sample(max_new_tokens=150))
            print("----------------------------------------------------------\\n")

    # Pobieranie batcha
    xb, yb = get_batch('train')

    # Forward
    logits, _ = model(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = yb.view(B*T)
    loss = F.cross_entropy(logits, targets)

    # Backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Clipping zabezpieczający sieć (żeby nie uciekła w nieskończoność matematyczną)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

print("\n🎉 Trening zakończony pomyślnie!")
print(f"--- FINAŁOWA PRÓBKA ---")
print(generate_sample(max_new_tokens=300))
print("-----------------------")
