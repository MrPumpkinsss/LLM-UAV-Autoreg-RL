from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ensure_dir


class LayerOneHotMLP(nn.Module):
    def __init__(self, num_boundaries: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_boundaries + 1
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, layer_idx: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(layer_idx, num_classes=self.num_boundaries).float()
        x = torch.cat([onehot, residual[:, None] * self.input_scale], dim=1)
        gamma = F.softplus(self.net(x).squeeze(1))
        return residual * gamma


def _extract_dataset(profile_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    curve = json.loads((profile_dir / "layer_ppl_curve.json").read_text(encoding="utf-8"))
    summary = json.loads((profile_dir / "layer_ppl_summary.json").read_text(encoding="utf-8"))
    ppl_ref = float(summary["clean_ppl"])
    xs_layer = []
    xs_residual = []
    ys = []
    for row in curve:
        layer = int(row["layer"])
        for point in row["drop_curve"]:
            drop = float(point["drop_rate"])
            if drop <= 0:
                continue
            # Calibration corrupts hidden states directly. This drop value is
            # already the residual loss probability seen by the LLM.
            residual = drop
            ppl = float(point["ppl_mean"])
            xs_layer.append(layer)
            xs_residual.append(residual)
            ys.append(math.log(max(ppl, 1e-12) / max(ppl_ref, 1e-12)))
    if not xs_layer:
        raise ValueError(f"no positive-drop calibration rows found in {profile_dir}")
    return (
        np.asarray(xs_layer, dtype=np.int64),
        np.asarray(xs_residual, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        ppl_ref,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return {
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "rmse_log_ratio": float(np.sqrt(np.mean(residual**2))),
        "mae_log_ratio": float(np.mean(np.abs(residual))),
    }


def _export_npz(model: LayerOneHotMLP, path: Path, num_boundaries: int, input_scale: float) -> None:
    linear_layers = [module for module in model.net if isinstance(module, nn.Linear)]
    payload: dict[str, np.ndarray] = {
        "type": np.asarray("layer_onehot_mlp_v1"),
        "num_boundaries": np.asarray(num_boundaries, dtype=np.int64),
        "input_scale": np.asarray(input_scale, dtype=np.float32),
        "max_calibrated_residual": np.asarray(1.0 / max(input_scale, 1e-9), dtype=np.float32),
        "hidden_layers": np.asarray(len(linear_layers) - 1, dtype=np.int64),
    }
    for idx, layer in enumerate(linear_layers[:-1]):
        payload[f"w{idx}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        payload[f"b{idx}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    payload["w_out"] = linear_layers[-1].weight.detach().cpu().numpy().astype(np.float32)
    payload["b_out"] = linear_layers[-1].bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an optional MLP PPL surrogate from layer calibration curves.")
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    profile_dir = Path(args.profile_dir)
    out_path = Path(args.out) if args.out else profile_dir / "ppl_surrogate_mlp.npz"
    ensure_dir(out_path.parent)

    layer_np, residual_np, y_np, ppl_ref = _extract_dataset(profile_dir)
    num_boundaries = int(layer_np.max()) + 1
    input_scale = 1.0 / max(float(np.max(residual_np)), 1e-9)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    layer = torch.as_tensor(layer_np, dtype=torch.long, device=device)
    residual = torch.as_tensor(residual_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)

    model = LayerOneHotMLP(num_boundaries, int(args.hidden_dim), int(args.depth)).to(device)
    model.num_boundaries = num_boundaries
    model.input_scale = input_scale
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, int(args.epochs) + 1):
        pred = model(layer, residual)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 1000 == 0 or epoch == int(args.epochs):
            print(f"epoch={epoch} loss={value:.8f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred_np = model(layer, residual).detach().cpu().numpy().astype(np.float64)
    metric = _metrics(y_np.astype(np.float64), pred_np)
    metric.update(
        {
            "profile_dir": profile_dir.as_posix(),
            "out": out_path.as_posix(),
            "ppl_ref": float(ppl_ref),
            "rows": int(y_np.shape[0]),
            "num_boundaries": int(num_boundaries),
            "input_scale": float(input_scale),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "epochs": int(args.epochs),
            "device": str(device),
        }
    )
    _export_npz(model, out_path, num_boundaries, input_scale)
    (out_path.with_suffix(".json")).write_text(json.dumps(metric, indent=2), encoding="utf-8")
    print(json.dumps(metric, indent=2), flush=True)


if __name__ == "__main__":
    main()
