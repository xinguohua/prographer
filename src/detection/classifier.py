"""
MLP classifier - paper Section IV-B.4

two MLP + cross-entropy loss, in's snapshot embeddinguptrain2classifier. 
traintimeneedsbenignandmalicioussnapshot's embeddingandlabel. 

usage:
    classify = MLPClassify(gid="bench")
    classify.train(benign_embeddings, malicious_embeddings, mal_labels)
    pred_labels, details = classify.predict(test_embeddings)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ._classifier_base import BaseClassify


class TwoLayerMLP(nn.Module):
    """two MLP: input -> hidden -> ReLU -> dropout -> output(2)"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2),  # 2class: [benign, malicious]
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
    model_save_path: str = "mlp_classifier.pth"
    meta_save_path: str = "mlp_meta.pkl"


class MLPClassify(BaseClassify):
    """
    paper's two MLP classifier (has, cross-entropy loss) . 

    and TopK 's : 
    - TopK without: notneedslabel, onlydegreetake Top-K
    - MLP has: needsbenign+maliciousembeddingandlabel, train2classifier
    """

    def __init__(self, cfg: Optional[MLPConfig] = None, gid: Optional[str] = None, **kwargs):
        super().__init__(gid=gid)
        self.cfg = cfg or MLPConfig()
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        if gid:
            self.cfg.model_save_path = self.with_gid_suffix(self.cfg.model_save_path)
            self.cfg.meta_save_path = self.with_gid_suffix(self.cfg.meta_save_path)

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
        """
        train MLP classifier. 

        Args:
            benign_embeddings: benignsnapshot embedding (N_b, D), labelallis 0
            malicious_embeddings: malicioussnapshot embedding (N_m, D)
            malicious_labels: malicioussnapshotTruelabel (N_m,), 1=malicious, 0=benign
                ifnotthenallismalicious (label=1) 
        """
        X_b = np.asarray(benign_embeddings, dtype=np.float32)
        self.input_dim = X_b.shape[1]

        # buildtrain: benign(label=0) + maliciousinterval(label=0or1)
        labels_b = np.zeros(X_b.shape[0], dtype=np.int64)

        if malicious_embeddings is not None:
            X_m = np.asarray(malicious_embeddings, dtype=np.float32)
            if malicious_labels is not None:
                labels_m = np.asarray(malicious_labels, dtype=np.int64)
            else:
                labels_m = np.ones(X_m.shape[0], dtype=np.int64)

            X_all = np.concatenate([X_b, X_m], axis=0)
            y_all = np.concatenate([labels_b, labels_m])
        else:
            print("[MLP] warning: withoutmalicious sample, classifiercancanwithouthastrain")
            X_all = X_b
            y_all = labels_b

        self.model = self._build_model()
        self._train_loop(X_all, labels=y_all)
        self._save()
        return self

    def _train_loop(self, embeddings, labels=None, **kwargs):
        X = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(self.device)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

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
        """
        snapshotlabel. 

        Returns:
            (pred_labels, details):
            - pred_labels: (N,) 0=benign 1=malicious
            - details: {idx: {"prob": float, "logits": array}}
        """
        if self.model is None:
            self.load()

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
        print(f"[MLP] : {len(pred_labels)} snapshot, {n_mal} malicious")

        return pred_labels, details

    def _save(self):
        """savemoduloandmetadata"""
        try:
            torch.save(self.model.state_dict(), self.cfg.model_save_path)
            print(f"[MLP] modulosaved: {self.cfg.model_save_path}")
        except Exception as e:
            print(f"[MLP] savemodulofailed: {e}")

        try:
            meta = {
                "input_dim": self.input_dim,
                "config": self.cfg.__dict__,
            }
            with open(self.cfg.meta_save_path, 'wb') as f:
                pickle.dump(meta, f)
        except Exception as e:
            print(f"[MLP] savemetadatafailed: {e}")

    def load(self):
        """loadsaved's modulo"""
        try:
            with open(self.cfg.meta_save_path, 'rb') as f:
                meta = pickle.load(f)
            self.input_dim = meta["input_dim"]
            self.model = self._build_model()
            self.model.load_state_dict(
                torch.load(self.cfg.model_save_path, map_location=self.device)
            )
            self.model.eval()
            print(f"[MLP] moduloalreadyload: {self.cfg.model_save_path}")
        except Exception as e:
            print(f"[MLP] loadfailed: {e}")
