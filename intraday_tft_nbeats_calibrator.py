"""
intraday_tft_nbeats_calibrator.py

One-file starter: PyTorch implementation of a combined Temporal-Fusion-Transformer-like
encoder + N-BEATS forecasting head and a simple Platt-style calibrator for classification.
Designed for intraday bar inputs (e.g., 1m bars). This is intended as a clear, robust,
readable starting point — not a production-optimized library.

This file has been cleaned to avoid runtime issues:
- Dataset returns torch tensors (not raw numpy) for reliable DataLoader behavior
- Safer shape handling in N-BEATS and fusion
- Predict & sizing handles 1D/2D inputs robustly and avoids division-by-zero
- Calibration saving/loading handled consistently

Usage (quick):
  python intraday_tft_nbeats_calibrator.py

Requirements: torch, numpy, pandas
"""

import os
import math
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------- Utilities & config -----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CONFIG = {
    "seq_len": 64,            # lookback bars (e.g., 64 1-minute bars)
    "pred_horizon": 5,        # predict 5 minutes ahead (aggregate)
    "batch_size": 256,
    "lr": 3e-4,
    "epochs": 6,
    "hidden_size": 128,
    "nbeats_blocks": 3,
    "nbeats_hidden": 128,
    "dropout": 0.1,
    "reg_loss_weight": 1.0,
    "cls_loss_weight": 1.0,
    "calibration_lr": 1e-2,
    "target_portfolio_vol": 0.01,  # 1% daily (example)
}

# ----------------------------- Data pipeline (stub) -----------------------------

class IntradayDataset(Dataset):
    """Expect input arrays or DataFrame and produce sliding windows.
    Produces torch.float32 tensors for both seq and target.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        assert X.shape[0] == y.shape[0], "X and y must have same length"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        # number of starting indices where a full sequence + one-step target exist
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len
        seq = self.X[start:end]         # (seq_len, F)
        target = self.y[end]            # next-step (or precomputed horizon)
        # return tensors (Batched by DataLoader)
        return torch.from_numpy(seq).float(), torch.tensor(target, dtype=torch.float32)


def generate_synthetic_intraday(T: int = 20000, F: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Create toy intraday feature matrix and target returns.
    - features: price returns, volume, imbalance, time-of-day sin/cos
    - target: future return aggregated over pred horizon
    """
    t = np.arange(T)
    # base price random walk with intraday seasonality
    drift = 0.0
    noise = np.random.normal(scale=0.0005, size=T)
    price = np.cumsum(drift + noise)

    # features
    returns = np.concatenate([[0.0], np.diff(price)])
    vol = 1 + 0.5 * (np.sin(2 * np.pi * (t % 390) / 390) + np.random.randn(T) * 0.1)
    imbalance = np.random.randn(T) * 0.1
    tod = (t % 390) / 390.0
    tod_sin = np.sin(2 * np.pi * tod)
    tod_cos = np.cos(2 * np.pi * tod)
    extra = np.random.randn(T, max(0, F - 5)) * 0.01

    X = np.stack([returns, np.abs(returns), vol, imbalance, tod_sin, tod_cos], axis=1)
    if extra.shape[1] > 0:
        X = np.concatenate([X, extra], axis=1)

    # target: future `pred_horizon`-step return with small signal depending on imbalance
    horizon = CONFIG['pred_horizon']
    fut_price = price[horizon:] - price[:-horizon]
    fut_price = np.concatenate([fut_price, np.zeros(horizon)])
    y = fut_price.astype(np.float32)
    # add weak relation: positive imbalance -> slightly positive future return
    y += 0.0005 * np.tanh(imbalance)

    return X, y

# ----------------------------- Model building blocks -----------------------------

class VariableSelection(nn.Module):
    """Simple variable selection: learn scalar gates per feature and apply softmax.
    Helps model focus on important channels (TFT-style lightweight).
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # x: (B, T, F)
        scores = self.gate(x.mean(dim=1))        # (B, F)
        weights = torch.softmax(scores, dim=-1)  # (B, F)
        return x * weights.unsqueeze(1)


class SimpleAttention(nn.Module):
    """Lightweight multi-head attention for temporal context.
    Uses batch_first multihead attention (PyTorch >=1.8 expected).
    """
    def __init__(self, dim, n_heads=4):
        super().__init__()
        # ensure embed_dim divisible by n_heads
        assert dim % n_heads == 0, "embed dim must be divisible by n_heads"
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.attn(x, x, x)
        return out


class TFTLite(nn.Module):
    """A compact TFT-like encoder: variable selection -> LSTM encoder -> attention -> pooled
    Produces a context embedding for downstream heads.
    """
    def __init__(self, input_dim, hidden_dim, lstm_layers=1, attn_heads=4, dropout=0.1):
        super().__init__()
        self.varsel = VariableSelection(input_dim, max(16, input_dim))
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # we use bidirectional LSTM: hidden_dim//2 per direction -> output dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.attn = SimpleAttention(hidden_dim, n_heads=attn_heads)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, F)
        x = self.varsel(x)                      # (B, T, F)
        x = self.input_projection(x)            # (B, T, D)
        x, _ = self.lstm(x)                     # (B, T, D)
        x = self.attn(x)                        # (B, T, D)
        x = self.dropout(x)
        # temporal pooling -> (B, D)
        x = x.permute(0, 2, 1)                  # (B, D, T)
        x = self.pool(x).squeeze(-1)
        return x


# N-BEATS simple implementation
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, theta_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.theta = nn.Linear(hidden_dim, theta_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, input_dim)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.theta(x)


class NBeats(nn.Module):
    def __init__(self, input_dim, blocks=3, hidden_dim=128, theta_dim=1):
        super().__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(input_dim, hidden_dim, theta_dim) for _ in range(blocks)])

    def forward(self, x):
        # Expect global pooled features (B, input_dim)
        outs = []
        for b in self.blocks:
            theta = b(x)  # (B, theta_dim)
            outs.append(theta)
        out = torch.sum(torch.stack(outs, dim=0), dim=0)  # (B, theta_dim)
        if out.size(-1) == 1:
            return out.squeeze(-1)  # (B,)
        return out


# Optional micro head (small conv net) for short-term microstructure
class MicroHead(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.conv(x)        # (B, hidden, 1)
        return x.squeeze(-1)


# Fusion head: combine embeddings and output regression + classification
class FusionHead(nn.Module):
    def __init__(self, emb_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.reg = nn.Linear(hidden, 1)
        self.cls = nn.Linear(hidden, 1)  # output logit for positive return

    def forward(self, emb):
        h = self.mlp(emb)
        pred_ret = self.reg(h).squeeze(-1)
        logit = self.cls(h).squeeze(-1)
        prob = torch.sigmoid(logit)
        return pred_ret, prob, logit


# ----------------------------- End-to-end model -----------------------------
class IntradayModel(nn.Module):
    def __init__(self, input_dim, seq_len, cfg):
        super().__init__()
        hid = cfg['hidden_size']
        self.tft = TFTLite(input_dim, hid, lstm_layers=1, attn_heads=4, dropout=cfg['dropout'])
        self.micro = MicroHead(input_dim, seq_len, hidden_dim=hid//2)
        # combine pooled features (tft_emb + micro_emb)
        combined_dim = hid + (hid//2)
        # N-Beats takes pooled historical features too (we'll pass pooled vector)
        self.nbeats = NBeats(combined_dim, blocks=cfg['nbeats_blocks'], hidden_dim=cfg['nbeats_hidden'], theta_dim=1)
        self.fusion = FusionHead(emb_dim=combined_dim + 1, hidden=hid, dropout=cfg['dropout'])

    def forward(self, x):
        # x: (B, T, F)
        tft_emb = self.tft(x)          # (B, hid)
        micro_emb = self.micro(x)      # (B, hid//2)
        pooled = torch.cat([tft_emb, micro_emb], dim=-1)  # (B, combined_dim)
        nbeats_out = self.nbeats(pooled)                  # (B,) or (B,)
        if nbeats_out.dim() == 1:
            nbeats_out = nbeats_out.unsqueeze(-1)
        fusion_input = torch.cat([pooled, nbeats_out], dim=-1)
        pred_ret, prob, logit = self.fusion(fusion_input)
        return pred_ret, prob, logit


# ----------------------------- Calibrator -----------------------------
class PlattCalibrator(nn.Module):
    """A simple Platt-style calibrator: learns scale and bias on logits to better fit probabilities.
    Trained on holdout val set's logits -> labels.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, logit):
        return torch.sigmoid(self.a * logit + self.b)

    def fit_on_val(self, logits: np.ndarray, labels: np.ndarray, lr=1e-2, steps=200):
        self.train()
        logits_t = torch.tensor(logits, dtype=torch.float32).to(DEVICE)
        labels_t = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        bce = nn.BCELoss()
        for _ in range(steps):
            opt.zero_grad()
            prob = self.forward(logits_t)
            loss = bce(prob, labels_t)
            loss.backward()
            opt.step()
        self.eval()


# ----------------------------- Training loop & helpers -----------------------------

def train_loop(model, train_loader, val_loader, cfg, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    best_val_loss = 1e9
    calib = PlattCalibrator().to(device)

    for epoch in range(cfg['epochs']):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # xb: (B, T, F), yb: (B,)
            ret_target = yb  # regression target
            cls_target = (yb > 0).float()

            pred_ret, prob, logit = model(xb)
            loss_reg = mse(pred_ret, ret_target)
            loss_cls = bce(prob, cls_target)
            loss = cfg['reg_loss_weight'] * loss_reg + cfg['cls_loss_weight'] * loss_cls

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                ret_target = yb
                cls_target = (yb > 0).float()
                pred_ret, prob, logit = model(xb)
                loss_reg = mse(pred_ret, ret_target)
                loss_cls = bce(prob, cls_target)
                loss = cfg['reg_loss_weight'] * loss_reg + cfg['cls_loss_weight'] * loss_cls
                val_loss += loss.item() * xb.size(0)
                logits_list.append(logit.detach().cpu().numpy())
                labels_list.append(cls_target.detach().cpu().numpy())
        val_loss /= len(val_loader.dataset)
        if len(logits_list) > 0:
            logits_all = np.concatenate(logits_list)
            labels_all = np.concatenate(labels_list)
        else:
            logits_all = np.array([])
            labels_all = np.array([])

        # Fit calibrator on validation logits if any
        if logits_all.size > 0:
            calib.fit_on_val(logits_all, labels_all, lr=cfg['calibration_lr'], steps=200)

        print(f"Epoch {epoch+1}/{cfg['epochs']}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  time={time.time()-t0:.1f}s")

        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state': model.state_dict(), 'calib_state': calib.state_dict()}, 'best_model.pt')

    return model, calib


# ----------------------------- Inference + sizing -----------------------------

def predict_and_size(model, calib: PlattCalibrator, x: np.ndarray, cfg, device):
    """Given a single sample or batch (numpy), return expected return, calibrated prob, and target size.
    Basic volatility-targeted sizing: size = (target_portfolio_vol / predicted_vol) * sign * prob
    Here predicted_vol is approximated by recent realized vol from features — this function assumes
    the first feature is returns and computes rolling volatility.
    """
    model.eval()
    calib.eval()
    with torch.no_grad():
        arr = np.asarray(x, dtype=np.float32)
        # arr expected shape: (T, F) or (B, T, F)
        if arr.ndim == 2:
            xb = torch.from_numpy(arr).unsqueeze(0).to(device)
            batch_mode = False
        elif arr.ndim == 3:
            xb = torch.from_numpy(arr).to(device)
            batch_mode = True
        else:
            raise ValueError("Input x must be shape (T,F) or (B,T,F)")

        pred_ret, prob, logit = model(xb)
        prob_cal = calib(logit)

        # estimate vol from input returns (feature 0): use latest seq for each batch element
        if not batch_mode:
            recent_rets = arr[-cfg['seq_len']:, 0]
            est_vol = float(np.std(recent_rets))
            est_vol = est_vol * math.sqrt(252 * 390 / 60) if est_vol > 0 else 1e-6
            sign = float(torch.sign(pred_ret).cpu().numpy())
            prob_val = float(prob_cal.cpu().numpy())
            expected_ret = float(pred_ret.cpu().numpy())
            size = (cfg['target_portfolio_vol'] / (est_vol + 1e-12)) * prob_val * sign
            return expected_ret, prob_val, size
        else:
            B = xb.size(0)
            sizes = []
            expected = []
            probs = []
            for i in range(B):
                recent_rets = arr[i, -cfg['seq_len']:, 0]
                est_vol = float(np.std(recent_rets))
                est_vol = est_vol * math.sqrt(252 * 390 / 60) if est_vol > 0 else 1e-6
                sign = float(torch.sign(pred_ret[i]).cpu().numpy())
                prob_val = float(prob_cal[i].cpu().numpy())
                expected_ret = float(pred_ret[i].cpu().numpy())
                size = (cfg['target_portfolio_vol'] / (est_vol + 1e-12)) * prob_val * sign
                sizes.append(size)
                expected.append(expected_ret)
                probs.append(prob_val)
            return np.array(expected), np.array(probs), np.array(sizes)


# ----------------------------- Main (example) -----------------------------

def main():
    print("Building synthetic dataset...")
    X, y = generate_synthetic_intraday(T=15000, F=8)
    seq_len = CONFIG['seq_len']

    # train/val split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_ds = IntradayDataset(X_train, y_train, seq_len)
    val_ds = IntradayDataset(X_val, y_val, seq_len)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    model = IntradayModel(input_dim=X.shape[1], seq_len=seq_len, cfg=CONFIG)
    model, calib = train_loop(model, train_loader, val_loader, CONFIG, DEVICE)

    print("Loading best checkpoint and running a quick inference check...")
    if os.path.exists('best_model.pt'):
        ckpt = torch.load('best_model.pt', map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        try:
            calib.load_state_dict(ckpt['calib_state'])
        except Exception:
            # if calib not present or mismatched, keep newly trained calibrator
            pass

    # Take last sequence from val set
    sample_seq = X_val[-seq_len:]
    expected_ret, prob, size = predict_and_size(model, calib, sample_seq, CONFIG, DEVICE)
    if isinstance(expected_ret, (list, np.ndarray)):
        print(f"Expected ret[0]={expected_ret[0]:.6f}  prob_positive[0]={prob[0]:.4f}  raw_size[0]={size[0]:.6f}")
    else:
        print(f"Expected ret={expected_ret:.6f}  prob_positive={prob:.4f}  raw_size={size:.6f}")

    print("Done. Model & calibrator saved as best_model.pt if training completed")


if __name__ == '__main__':
    main()
