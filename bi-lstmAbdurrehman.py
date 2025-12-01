# =========================
# Install + Imports
# =========================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from collections import Counter
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 1️⃣ LOAD DATA (Colab Compatible)
# =========================
# If running in Colab and file not found, allow upload
if not os.path.exists("jokes_dataset.jsonl"):
    from google.colab import files
    print("Upload jokes_dataset.jsonl")
    uploaded = files.upload()

data_file = "jokes_dataset.jsonl"

inputs = []
outputs = []

with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        w1 = entry["word1"].strip()
        w2 = entry["word2"].strip()
        joke = entry["joke"].lower().strip()

        inputs.append(f"{w1} {w2}")
        outputs.append(joke)

print("Dataset loaded:", len(inputs), "samples")

# =========================
# 2️⃣ VOCAB BUILD
# =========================
all_words = " ".join(outputs).split()
word_counts = Counter(all_words)

vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + [w for w, _ in word_counts.most_common(5000)]
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

def tokenize(text, max_len=20):
    tokens = [word2idx.get(w, word2idx["<UNK>"]) for w in text.split()]
    tokens = [word2idx["<SOS>"]] + tokens + [word2idx["<EOS>"]]

    if len(tokens) < max_len:
        tokens += [word2idx["<PAD>"]] * (max_len - len(tokens))
    return tokens[:max_len]

input_seqs = [tokenize(i, max_len=5) for i in inputs]
output_seqs = [tokenize(o, max_len=30) for o in outputs]

# =========================
# 3️⃣ DATASET
# =========================
class JokeDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.outputs = torch.tensor(outputs, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

dataset = JokeDataset(input_seqs, output_seqs)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================
# 4️⃣ BI-LSTM SEQ2SEQ MODEL (Fixed)
# =========================
class BiLSTMSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Decoder takes 2*hidden_dim because encoder is bidirectional
        self.decoder = nn.LSTM(embed_dim, hidden_dim * 2, batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, src, trg):
        embedded_src = self.embedding(src)
        enc_out, (h, c) = self.encoder(embedded_src)

        # concatenate forward & backward LSTM hidden states
        h_cat = torch.cat([h[0:1], h[1:2]], dim=2)
        c_cat = torch.cat([c[0:1], c[1:2]], dim=2)

        embedded_trg = self.embedding(trg)
        dec_out, _ = self.decoder(embedded_trg, (h_cat, c_cat))

        return self.fc(dec_out)

vocab_size = len(vocab)
model = BiLSTMSeq2Seq(vocab_size).to(device)

# =========================
# 5️⃣ TRAINING
# =========================
criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5   # Reduce if training slow

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for src, trg in tqdm(loader):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        loss = criterion(output.reshape(-1, vocab_size), trg[:, 1:].reshape(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

# =========================
# 6️⃣ INFERENCE / JOKE GENERATION (No Errors)
# =========================
def generate_joke(word1, word2, max_len=30):
    model.eval()

    src = torch.tensor(tokenize(f"{word1} {word2}", 5), dtype=torch.long).unsqueeze(0).to(device)

    enc_out, (h, c) = model.encoder(model.embedding(src))
    hidden = (torch.cat([h[0:1], h[1:2]], dim=2),
              torch.cat([c[0:1], c[1:2]], dim=2))

    trg = torch.tensor([[word2idx["<SOS>"]]], dtype=torch.long).to(device)
    result = []

    for _ in range(max_len):
        embedded = model.embedding(trg[:, -1:])  # last token only
        dec_out, hidden = model.decoder(embedded, hidden)
        out = model.fc(dec_out[:, -1, :])
        pred = out.argmax(1).item()

        if pred == word2idx["<EOS>"]:
            break

        result.append(idx2word.get(pred, "<UNK>"))

        trg = torch.cat([trg, torch.tensor([[pred]], dtype=torch.long).to(device)], dim=1)

    return " ".join(result)

# Test
print(generate_joke("cat", "coffee"))
