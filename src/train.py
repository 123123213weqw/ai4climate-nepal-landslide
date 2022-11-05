"""Train models with spatial block cross-validation and save the final fit.

Usage:
  python -m src.train --model all --device auto --folds 5 --epochs 200
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from .config import DATA_DIR, FEATURES, RANDOM_SEED, RUNS_DIR
from .models import HybridCNN, count_params, make_gbm, make_mlp, sklearn_models

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def pick_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available on this machine")
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_mlp_fold(Xtr, ytr, device, epochs, batch_size=1024, patience=25):
    """Train on the given fold data; returns (state_dict, scaler mean, scaler std)."""
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = (Xtr - mean) / std
    n_val = max(24, int(0.1 * len(Xn)))
    idx = torch.randperm(len(Xn))
    va, tr = idx[:n_val], idx[n_val:]
    X = torch.as_tensor(Xn, dtype=torch.float32, device=device)
    y = torch.as_tensor(ytr, dtype=torch.float32, device=device)

    model = make_mlp(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))].to(device)
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(X[b]).squeeze(-1)
                loss = lossf(logits, y[b])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            vl = lossf(model(X[va]).squeeze(-1), y[va]).item()
        if vl < best_val - 1e-4:
            best_val, best_state, bad = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


def load_patch_data():
    z = np.load(DATA_DIR / "patches.npz", allow_pickle=True)
    return (
        z["patches"].astype(np.float32),
        z["scalars"].astype(np.float32),
        z["label"].astype(np.float32),
        z["block_id"].astype(str),
    )


def train_cnn_fold(P, S, y, device, epochs, batch_size=128, patience=30, width=32):
    """Train the hybrid CNN; returns (model, per-channel stats, scalar stats)."""
    ch_mean = P.mean(axis=(0, 2, 3), keepdims=True)  # (1,8,1,1)
    ch_std = P.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    sc_mean, sc_std = S.mean(0), S.std(0) + 1e-6
    Pn = (P - ch_mean) / ch_std
    Sn = (S - sc_mean) / sc_std
    n_val = max(24, int(0.1 * len(Pn)))
    idx = torch.randperm(len(Pn))
    va, tr = idx[:n_val], idx[n_val:]
    Pt = torch.as_tensor(Pn, device=device)
    St = torch.as_tensor(Sn, device=device)
    yt = torch.as_tensor(y, device=device)

    model = HybridCNN(in_ch=P.shape[1], scalar_dim=S.shape[1], width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))].to(device)
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = lossf(model(Pt[b], St[b]), yt[b])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            vl = lossf(model(Pt[va.to(device)], St[va.to(device)]), yt[va.to(device)]).item()
        if vl < best_val - 1e-4:
            best_val, best_state, bad = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, ch_mean, ch_std, sc_mean, sc_std


def predict_cnn(model, P, S, ch_mean, ch_std, sc_mean, sc_std, device, batch=2048) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(P), batch):
            p = torch.as_tensor(((P[i : i + batch] - ch_mean) / ch_std), dtype=torch.float32, device=device)
            s = torch.as_tensor(((S[i : i + batch] - sc_mean) / sc_std), dtype=torch.float32, device=device)
            out.append(torch.sigmoid(model(p, s)).float().cpu().numpy())
    return np.concatenate(out)


def predict_mlp(model, X, mean, std, device, batch=65536) -> np.ndarray:
    model.eval()
    Xn = (X - mean) / std
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            xb = torch.as_tensor(Xn[i : i + batch], dtype=torch.float32, device=device)
            out.append(torch.sigmoid(model(xb).squeeze(-1)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", "logreg", "gbm", "mlp", "cnn"])
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--split", default="spatial", choices=["spatial", "random"],
                    help="random = classic (leaky) CV, for the leakage experiment")
    ap.add_argument("--zero-scalars", action="store_true",
                    help="ablation: zero the CNN scalar head inputs (terrain patches only)")
    args = ap.parse_args()

    device = pick_device(args.device)
    suffix = ("_noscalars" if args.zero_scalars else "") + ("_randomsplit" if args.split == "random" else "")
    save_final = suffix == ""
    patch_data = None
    names = ["logreg", "gbm", "mlp", "cnn"] if args.model == "all" else [args.model]
    if "cnn" in names:
        if (DATA_DIR / "patches.npz").exists():
            patch_data = load_patch_data()
            print(f"patches: {patch_data[0].shape} | climate: {patch_data[1].shape}")
        else:
            print("cnn skipped (data/patches.npz missing — run: python -m src.data build-patches)")
            names = [n for n in names if n != "cnn"]

    df = pd.read_parquet(DATA_DIR / "dataset.parquet")
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y_all = df["label"].to_numpy()
    groups = df["block_id"].to_numpy()
    print(f"dataset: {len(df)} rows | features: {len(FEATURES)} | device: {device}")
    print(f"blocks: {df['block_id'].nunique()} | positives: {int(y_all.sum())}")

    splitter = (
        KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_SEED)
        if args.split == "random"
        else GroupKFold(n_splits=args.folds)
    )
    results = {}
    for name in names:
        if name == "cnn":
            P, S, y_p, groups_p = patch_data
            if args.zero_scalars:
                S = np.zeros_like(S)
            folds = list(splitter.split(P, y_p, groups_p))
        else:
            folds = list(splitter.split(X, y_all, groups))
        aucs, aps = [], []
        for k, (tr_i, te_i) in enumerate(folds):
            if name == "mlp":
                model, mean, std = train_mlp_fold(X[tr_i], y_all[tr_i], device, args.epochs)
                p = predict_mlp(model, X[te_i], mean, std, device)
            elif name == "cnn":
                model, cm, cs, sm, ss = train_cnn_fold(P[tr_i], S[tr_i], y_p[tr_i], device, args.epochs)
                p = predict_cnn(model, P[te_i], S[te_i], cm, cs, sm, ss, device)
            else:
                model = sklearn_models()["logreg"] if name == "logreg" else make_gbm()
                model.fit(X[tr_i], y_all[tr_i])
                p = model.predict_proba(X[te_i])[:, 1]
            yy = y_p[te_i] if name == "cnn" else y_all[te_i]
            aucs.append(roc_auc_score(yy, p))
            aps.append(average_precision_score(yy, p))
            print(f"  [{name}] fold {k + 1}/{args.folds}: AUC={aucs[-1]:.3f} AP={aps[-1]:.3f}")
        results[name + suffix] = {
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps)),
            "folds": {"auc": [round(a, 3) for a in aucs], "ap": [round(a, 3) for a in aps]},
        }
        print(f"[{name + suffix}] AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} | AP {np.mean(aps):.3f}")

    # Final fit on all data, saved for mapping (full configuration only).
    for name in names:
        if not save_final:
            break
        if name == "mlp":
            model, mean, std = train_mlp_fold(X, y_all, device, args.epochs)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mean": mean, "std": std, "features": FEATURES,
                    "hidden": [256, 128, 64],
                    "params": count_params(model),
                },
                RUNS_DIR / "final_mlp.pt",
            )
        elif name == "cnn":
            P, S, y_p, _ = patch_data
            model, cm, cs, sm, ss = train_cnn_fold(P, S, y_p, device, args.epochs)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "ch_mean": cm, "ch_std": cs, "sc_mean": sm, "sc_std": ss,
                    "in_ch": int(P.shape[1]), "scalar_dim": int(S.shape[1]),
                    "patch_size": int(P.shape[-1]), "width": 32,
                    "params": count_params(model),
                },
                RUNS_DIR / "final_cnn.pt",
            )
        else:
            model = sklearn_models()["logreg"] if name == "logreg" else make_gbm()
            model.fit(X, y_all)
            joblib.dump(model, RUNS_DIR / f"final_{name}.joblib")

    RUNS_DIR.mkdir(exist_ok=True)
    metrics_path = RUNS_DIR / "metrics.json"
    merged = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    merged.update({"device": str(device), "cv": {**merged.get("cv", {}), **results}})
    metrics_path.write_text(json.dumps(merged, indent=2))
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
