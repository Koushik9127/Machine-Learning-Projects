import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class ReviewDS(Dataset):
    def __init__(self, texts, labels, vocab=None, tokenizer=None, max_len=100):
        self.tokenizer = tokenizer or get_tokenizer("basic_english")
        self.vocab = vocab or build_vocab_from_iterator(
            [self.tokenizer(t) for t in texts], specials=["<pad>"])
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.max_len = max_len
        self.texts = texts
        self.labels = labels

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])[: self.max_len]
        ids = self.vocab(tokens)
        pad = [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return torch.tensor(ids + pad), torch.tensor(self.labels[idx])

df = pd.read_csv("../data/reviews.csv")
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
tok = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator([tok(t) for t in train_df["review_text"]], specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])

train_ds = ReviewDS(train_df["review_text"].tolist(), train_df["label"].tolist(), vocab, tok)
test_ds  = ReviewDS(test_df["review_text"].tolist(),  test_df["label"].tolist(),  vocab, tok)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=64)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, output_dim=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        emb = self.emb(x)
        _, (h, _) = self.lstm(emb)
        return self.fc(h[-1])

model = LSTMClassifier(len(vocab)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        opt.step()
    print(f"epoch {epoch+1} done")

torch.save(model.state_dict(), "../models/lstm.pt")

