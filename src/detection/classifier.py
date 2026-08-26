"""Two-layer node classifier from Proof Section IV-D3.

The classifier is trained with cross-entropy on benign and malicious node
embeddings produced by the frozen ATHENA encoder.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ._classifier_base import BaseClassify


class TwoLayerMLP(nn.Module):
    """Input -> hidden -> ReLU -> dropout -> two-class logits."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2),  # [benign, malicious]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPConfig:
    hidden_dim: int = 128
    dropout: float = 0.3
    lr: float = 1e-3
    num_epochs: int = 3
    batch_size: int = 64
    seed: int = 42


class MLPClassify(BaseClassify):
    """Train and apply the Proof Section IV-D3 binary MLP head."""

    def __init__(self, cfg: Optional[MLPConfig] = None, gid: Optional[str] = None, **kwargs):
        super().__init__(gid=gid)
        self.cfg = cfg or MLPConfig()
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim: Optional[int] = None
        self.model: Optional[TwoLayerMLP] = None

    def _build_model(self):
        if self.input_dim is None:
            raise ValueError("input_dim set, first use train()")
        return TwoLayerMLP(
            input_dim=self.input_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)

    def train(self, benign_embeddings: np.ndarray,
              malicious_embeddings: np.ndarray = None,
              malicious_labels: np.ndarray = None,
              **kwargs) -> "MLPClassify":
        """Train the binary MLP classifier.

        Args:
            benign_embeddings: ``(N_b, D)`` embeddings labelled benign.
            malicious_embeddings: ``(N_m, D)`` embeddings from attack graphs.
            malicious_labels: Optional binary labels for the attack-graph
                embeddings. If omitted, every row is labelled malicious.
        """
        X_b = np.asarray(benign_embeddings, dtype=np.float32)
        if X_b.ndim != 2 or X_b.shape[0] == 0:
            raise ValueError("MLP training requires at least one benign embedding")
        self.input_dim = X_b.shape[1]

        # Combine benign labels with the supplied attack-graph labels.
        labels_b = np.zeros(X_b.shape[0], dtype=np.int64)

        if malicious_embeddings is not None:
            X_m = np.asarray(malicious_embeddings, dtype=np.float32)
            if X_m.ndim != 2 or X_m.shape[0] == 0:
                raise ValueError("MLP training requires at least one malicious embedding")
            if X_m.shape[1] != X_b.shape[1]:
                raise ValueError("benign and malicious embedding dimensions differ")
            if malicious_labels is not None:
                labels_m = np.asarray(malicious_labels, dtype=np.int64)
            else:
                labels_m = np.ones(X_m.shape[0], dtype=np.int64)

            X_all = np.concatenate([X_b, X_m], axis=0)
            y_all = np.concatenate([labels_b, labels_m])
        else:
            raise ValueError("MLP training requires malicious embeddings")

        if set(np.unique(y_all).tolist()) != {0, 1}:
            raise ValueError("MLP training requires both benign and malicious labels")

        self.model = self._build_model()
        self._train_loop(X_all, labels=y_all)
        return self

    def _train_loop(self, embeddings, labels=None, **kwargs):
        X = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(self.device)

        dataset = TensorDataset(X, y)
        generator = torch.Generator().manual_seed(int(self.cfg.seed))
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True, generator=generator,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        n_pos = int((y == 1).sum().item())
        n_neg = int((y == 0).sum().item())
        if n_pos > 0 and n_neg > 0:
            weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=self.device)
        else:
            weight = None
        criterion = nn.CrossEntropyLoss(weight=weight)

        self.model.train()
        for epoch in range(self.cfg.num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                logits = self.model(xb)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc = correct / total if total > 0 else 0
                avg_loss = total_loss / total if total > 0 else 0
                print(f"[MLP] epoch {epoch+1}/{self.cfg.num_epochs} "
                      f"loss={avg_loss:.4f} acc={acc:.4f}")

        return {}

    def predict(self, embeddings: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Predict a binary label for each node embedding.

        Returns:
            (pred_labels, details):
            - pred_labels: (N,) 0=benign 1=malicious
            - details: {idx: {"prob": float, "logits": array}}
        """
        if self.model is None:
            raise RuntimeError("MLP model is not initialized; train it or load an explicit run checkpoint")

        X = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        pred_labels = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()

        details = {}
        for i in range(len(pred_labels)):
            if pred_labels[i] == 1:
                details[i] = {
                    "position": int(i),
                    "prob_malicious": float(probs_np[i, 1]),
                    "logits": logits[i].cpu().numpy(),
                }

        n_mal = int(pred_labels.sum())
        print(f"[MLP] predictions={len(pred_labels)} malicious={n_mal}")

        return pred_labels, details
