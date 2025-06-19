import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class CancerClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train(model, train_loader, epochs=10):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            preds = model(X_batch)
            loss = loss_fn(preds.view(-1), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
