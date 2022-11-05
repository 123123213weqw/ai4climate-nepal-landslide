"""Models: logistic regression (classic LSM baseline), GBM, and a CUDA-ready MLP."""
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def sklearn_models() -> dict:
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, C=1.0)),
            ]
        ),
        "gbm": None,  # imported lazily to keep startup light
    }


def make_gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=None, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0,
    )


def make_mlp(in_dim: int, hidden=(256, 128, 64), dropout: float = 0.15) -> nn.Module:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class HybridCNN(nn.Module):
    """Terrain-patch encoder (CNN) fused with climate scalars in the head.

    Input: patch [B, 8, H, W] (terrain rasters window) and scalars [B, S]
    (annual/monsoon precipitation, seasonality). Output: logit [B].
    """

    def __init__(self, in_ch: int = 8, scalar_dim: int = 3, width: int = 32, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.GELU(),
        )
        self.stage1 = ResBlock(width)
        self.down = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.GELU(),
        )
        self.stage2 = ResBlock(width * 2)
        self.head = nn.Sequential(
            nn.Linear(width * 2 + scalar_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, patch, scalars):
        z = self.stage2(self.down(self.stage1(self.stem(patch))))
        z = z.mean(dim=(2, 3))  # global average pool
        return self.head(torch.cat([z, scalars], dim=1)).squeeze(-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
