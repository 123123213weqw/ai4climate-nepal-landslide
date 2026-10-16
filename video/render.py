"""Render the 3-minute AI4Climate Nepal landslide video, frame by frame.

12 scenes at 1920x1080/30fps. Scene durations are locked to the narration WAVs.
Static bases (background, kicker, titles) are cached per scene; animated layers
are drawn per frame with PIL. Matplotlib assets (map bases, channel tiles,
metric charts) are pre-rendered once into assets/.
"""
import json
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VID = ROOT / "video"
FRAMES = VID / "frames"
ASSETS = VID / "assets"
AUDIO = VID / "audio"
W, H, FPS = 1920, 1080, 30

# ---------- palette ----------
BG0 = (13, 17, 23)
BG1 = (21, 28, 40)
TXT = (232, 237, 244)
SUB = (154, 167, 181)
AMBER = (245, 168, 60)
CYAN = (88, 183, 216)
GREEN = (123, 196, 127)
RED = (224, 96, 85)
CARD = (24, 32, 46)
CARD_HI = (32, 43, 62)
GRID = (38, 48, 66)

FOOT = "github.com/123123213weqw/ai4climate-nepal-landslide  ·  open data · MIT"

FONT_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"
_fcache = {}


def F(size, bold=False):
    key = (size, bold)
    if key not in _fcache:
        _fcache[key] = ImageFont.truetype(FONT_PATH, size, index=1 if bold else 0)
    return _fcache[key]


def txt(d, xy, s, size, color=TXT, bold=False, anchor="la", tracking=0, alpha=1.0):
    """Draw text with optional letter tracking and alpha (anchor-aware)."""
    if alpha <= 0.01:
        return
    c = tuple(int(a * alpha) for a in color)
    f = F(size, bold)
    if tracking == 0:
        d.text(xy, s, font=f, fill=c, anchor=anchor)
        return
    widths = [d.textlength(ch, font=f) for ch in s]
    total = sum(widths) + tracking * max(0, len(s) - 1)
    x, y = xy
    if anchor[0] == "m":
        x -= total / 2
    elif anchor[0] == "r":
        x -= total
    for ch, w in zip(s, widths):
        d.text((x, y), ch, font=f, fill=c, anchor="l" + anchor[1])
        x += w + tracking


def tw(d, s, size, bold=False, tracking=0):
    f = F(size, bold)
    return d.textlength(s, font=f) + tracking * max(0, len(s) - 1)


def ease(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def seg(t, start, dur):
    return ease((t - start) / dur) if dur > 0 else (1.0 if t >= start else 0.0)


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def count_up(d, center, value, t, suffix="", prefix="", size=150, color=TXT, bold=True):
    v = value * ease(t)
    s = f"{prefix}{v:.3f}" if isinstance(value, float) else f"{prefix}{int(round(v))}{suffix}"
    txt(d, center, s, size, color, bold, anchor="mm")


# ---------- shared background ----------
def make_bg():
    img = Image.new("RGB", (W, H), BG0)
    top = Image.new("RGB", (W, H), BG1)
    mask = Image.linear_gradient("L").resize((W, H)).rotate(180)
    img = Image.composite(top, img, mask.point(lambda p: int(p * 0.55)))
    d = ImageDraw.Draw(img, "RGBA")
    for gx in range(0, W + 1, 96):
        d.line([(gx, 0), (gx, H)], fill=GRID + (36,), width=1)
    for gy in range(0, H + 1, 96):
        d.line([(0, gy), (W, gy)], fill=GRID + (36,), width=1)
    return img


BG = None


def chrome(kicker, idx, total, progress):
    """Base image with kicker, scene counter, footer, progress bar."""
    img = BG.copy()
    d = ImageDraw.Draw(img, "RGBA")
    txt(d, (120, 62), kicker.upper(), 26, AMBER, bold=True, tracking=6)
    txt(d, (W - 120, 62), f"{idx:02d} / {total:02d}", 26, SUB, anchor="ra", tracking=2)
    d.rectangle([120, 108, W - 120, 114], fill=(38, 48, 66))
    d.rectangle([120, 108, 120 + (W - 240) * progress, 114], fill=AMBER)
    txt(d, (120, H - 66), FOOT, 22, SUB, tracking=1)
    return img


def title_block(d, title, sub=None, t=1.0, y=150):
    a1 = seg(t, 0.0, 0.5)
    txt(d, (120, y), title, 64, TXT, bold=True, alpha=a1)
    if sub:
        a2 = seg(t, 0.25, 0.5)
        txt(d, (120, y + 84), sub, 30, SUB, alpha=a2)


def fade_edge(img, t, dur, scene_len):
    """Fade in at scene start, fade out at scene end."""
    a = min(1.0, t / 0.3, max(0.0, (scene_len - t) / 0.3))
    if a >= 1.0:
        return img
    black = Image.new("RGB", img.size, BG0)
    return Image.blend(black, img, max(0.0, a))


# ---------- data ----------
z = np.load(ROOT / "data" / "patches.npz", allow_pickle=True)
LON, LAT, LAB = z["longitude"], z["latitude"], z["label"]
SCAL = z["scalars"]
PATCHES = z["patches"].astype(np.float32)
POS = LAB == 1

DS = json.loads((ROOT / "runs" / "data_summary.json").read_text())
MV = json.loads((ROOT / "runs" / "metrics_v100.json").read_text())["cv"]
TRIG = DS["top_triggers"]

# fold coloring (illustrative grouping by 0.5-degree block)
def block_fold(lon, lat):
    return int((math.floor(lon / 0.5) * 7 + math.floor(lat / 0.5) * 13) % 5)


FOLDCOL = [(245, 168, 60), (88, 183, 216), (123, 196, 127), (224, 96, 85), (196, 148, 233)]

MAP_BOX = (150, 300, 950, 980)  # x0,y0,x1,y1 for map panels
BBOX = (80.0, 26.3, 88.2, 30.5)


def lonlat_to_px(lon, lat, box):
    x0, y0, x1, y1 = box
    return (x0 + (lon - BBOX[0]) / (BBOX[2] - BBOX[0]) * (x1 - x0),
            y1 - (lat - BBOX[1]) / (BBOX[3] - BBOX[1]) * (y1 - y0))


# ---------- matplotlib assets ----------
def build_assets():
    ASSETS.mkdir(exist_ok=True)
    # event footprint hull for the map base (monotone chain, no scipy)
    pts = sorted(map(tuple, np.column_stack([LON[POS], LAT[POS]])))

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) -
                                     (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    hull = np.array(lower[:-1] + upper[:-1])
    np.save(ASSETS / "hull.npy", hull)

    # pick an interesting positive patch (max relief-9x9 spread)
    ch = PATCHES[POS][:, 6]
    pick = int(np.argmax(ch.std(axis=(1, 2))))
    patch = PATCHES[np.where(POS)[0][pick]]
    np.save(ASSETS / "patch.npy", patch)

    names = ["Elevation", "Slope", "Northness", "Eastness", "Curvature", "Roughness", "Relief 9×9", "TWI proxy"]
    cols = ["terrain", "magma", "RdBu", "RdBu", "PuOr", "viridis", "inferno", "BrBG"]
    for i in range(8):
        fig, ax = plt.subplots(figsize=(3, 3), dpi=140)
        ax.set_facecolor("#10151e")
        fig.patch.set_facecolor("#10151e")
        v = patch[i]
        ax.imshow(v, cmap=cols[i])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2a3446")
        fig.tight_layout(pad=0.4)
        fig.savefig(ASSETS / f"chan{i}.png", facecolor="#10151e")
        plt.close(fig)
    (ASSETS / "names.json").write_text(json.dumps(names))


# ---------- scene draw functions ----------
def draw_points(d, box, mask, color, r=5, t=1.0, stagger=0.0, outline=None):
    n = int(mask.sum())
    idx = np.where(mask)[0]
    order = np.argsort(LON[idx] * 3 + LAT[idx])
    for k, i in enumerate(idx[order]):
        a = seg(t, stagger * (k / max(1, n - 1)), max(0.05, 1 - stagger))
        if a <= 0.02:
            continue
        x, y = lonlat_to_px(LON[i], LAT[i], box)
        c = tuple(int(v * a) for v in color)
        if outline:
            d.ellipse([x - r - 1.5, y - r - 1.5, x + r + 1.5, y + r + 1.5], fill=tuple(int(v * a) for v in outline))
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)


def map_axes(d, box, label):
    d.rectangle(box, outline=(60, 74, 96), width=2)
    for lon in range(81, 89, 2):
        x, _ = lonlat_to_px(lon, 0, box)
        d.line([(x, box[1]), (x, box[3])], fill=(46, 58, 78), width=1)
    for lat in (27, 28, 29, 30):
        _, y = lonlat_to_px(0, lat, box)
        d.line([(box[0], y), (box[2], y)], fill=(46, 58, 78), width=1)
    txt(d, (box[2] - 12, box[1] + 12), label, 22, SUB, anchor="ra")


def hull_outline(d, box, t=1.0, color=CYAN, width=3, label=None):
    hull = np.load(ASSETS / "hull.npy")
    a = ease(t)
    pts = [lonlat_to_px(p[0], p[1], box) for p in hull] + [lonlat_to_px(hull[0][0], hull[0][1], box)]
    n = max(2, int(len(pts) * a))
    c = tuple(int(v * min(1, t * 2)) for v in color)
    d.line(pts[:n], fill=c, width=width, joint="curve")
    if label and a > 0.95:
        txt(d, (box[0] + 14, box[1] + 48), label, 22, CYAN)


# ---- scenes ----
def sc_title(img, d, t, L):
    # faint event scatter backdrop (dimmed so titles stay readable)
    dim = lambda col: tuple(int(v * 0.35) for v in col)
    draw_points(d, (140, 330, W - 140, 950), POS, dim(AMBER), r=3, t=seg(t, 0.3, 2.5))
    draw_points(d, (140, 330, W - 140, 950), ~POS, dim(CYAN), r=3, t=seg(t, 0.8, 2.5))
    a = seg(t, 0.2, 0.8)
    txt(d, (W / 2, 460), "THE MOUNTAIN THAT", 92, TXT, bold=True, anchor="mm", alpha=a, tracking=2)
    txt(d, (W / 2, 580), "REMEMBERS THE RAIN", 92, AMBER, bold=True, anchor="mm", alpha=seg(t, 0.5, 0.8), tracking=2)
    txt(d, (W / 2, 700), "Open-data landslide susceptibility mapping for the Nepal Himalaya",
        34, SUB, anchor="mm", alpha=seg(t, 0.9, 0.7))
    b = seg(t, 1.4, 0.6)
    bx = (W / 2 - 330, 790, W / 2 + 330, 862)
    rrect(d, bx, 36, outline=tuple(int(v * b) for v in AMBER), width=2)
    txt(d, (W / 2, 826), "AI4Climate Global Youth Challenge · Track 2", 28, TXT, bold=True,
        anchor="mm", alpha=b)


def sc_problem(img, d, t, L):
    title_block(d, "The problem", "Monsoon rain on broken Himalayan terrain", t)
    rows = [
        ("84%", "of cataloged Nepal landslides are rainfall-triggered", AMBER),
        ("479", "events in NASA's Global Landslide Catalog", CYAN),
        ("0", "susceptibility maps at municipal resolution, for most offices", RED),
    ]
    for i, (big, small, col) in enumerate(rows):
        a = seg(t, 0.3 + i * 0.35, 0.5)
        y = 330 + i * 170
        if a > 0.02:
            d.rounded_rectangle((120, y, 1130, y + 140), radius=22,
                                fill=tuple(int(v * a) for v in CARD), outline=tuple(int(v * a) for v in GRID), width=2)
            v = int(84 * a) if big == "84%" else (int(479 * a) if big == "479" else 0)
            txt(d, (170, y + 70), f"{v}", 74, col, bold=True, anchor="lm")
            txt(d, (420, y + 70), small, 30, TXT, anchor="lm", alpha=a)
    # right: donut rainfall-triggered
    cx, cy, R, r = 1500, 560, 170, 110
    frac = 0.84 * ease(min(1, t / 1.8))
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(38, 48, 66))
    d.pieslice([cx - R, cy - R, cx + R, cy + R], -90, -90 + 360 * frac, fill=AMBER)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BG1)
    txt(d, (cx, cy - 14), f"{int(84 * ease(min(1, t / 1.8)))}%", 64, AMBER, bold=True, anchor="mm")
    txt(d, (cx, cy + 44), "rainfall-triggered", 24, SUB, anchor="mm")
    txt(d, (cx, cy + R + 44), "NASA GLC · Nepal · 479 events", 24, SUB, anchor="mm")


def sc_inventory(img, d, t, L):
    title_block(d, "The inventory", "NASA Global Landslide Catalog + uniform pseudo-absences", t)
    box = MAP_BOX
    map_axes(d, box, "Nepal · event footprint (84–88.2°E, 26.3–30.5°N)")
    hull_outline(d, box, seg(t, 0.2, 1.0), label="convex hull of events")
    draw_points(d, box, POS, AMBER, r=7, t=seg(t, 0.5, 2.0), stagger=0.6)
    draw_points(d, box, ~POS, CYAN, r=6, t=seg(t, 1.6, 2.0), stagger=0.5)
    # legend
    ly = 330
    for col, s in ((AMBER, "479 landslide events"), (CYAN, "479 pseudo-absences  (≥ 2 km from events)")):
        d.ellipse([1030, ly, 1054, ly + 24], fill=col)
        txt(d, (1070, ly + 12), s, 26, TXT, anchor="lm")
        ly += 52
    # trigger bars
    txt(d, (1030, ly + 30), "Trigger types (top 6)", 26, SUB, bold=True)
    top = [("downpour", 221), ("rain", 96), ("continuous rain", 59), ("unknown", 55), ("monsoon", 28), ("earthquake", 9)]
    by = ly + 80
    for i, (name, v) in enumerate(top):
        a = seg(t, 2.0 + i * 0.15, 0.5)
        full = 340
        wpx = full * (v / 221) * a
        d.rectangle([1290, by, 1290 + full, by + 26], fill=(30, 39, 55))
        d.rectangle([1290, by, 1290 + wpx, by + 26], fill=AMBER if i < 5 else SUB)
        txt(d, (1280, by + 13), name, 22, SUB, anchor="rm")
        txt(d, (1290 + full + 14, by + 13), str(v), 22, TXT, anchor="lm")
        by += 44


def sc_terrain(img, d, t, L):
    title_block(d, "Reading terrain like an image", "Terrarium tiles (SRTM, ~34 m/px) → 8 channels → 33×33 patches", t)
    names = json.loads((ASSETS / "names.json").read_text())
    tiles = [Image.open(ASSETS / f"chan{i}.png").convert("RGB").resize((240, 240), Image.LANCZOS) for i in range(8)]
    gx, gy = 120, 330
    for i in range(8):
        a = seg(t, 0.3 + i * 0.18, 0.45)
        if a <= 0.02:
            continue
        x = gx + (i % 4) * 264
        y = gy + (i // 4) * 300
        tile = tiles[i] if a >= 0.99 else Image.blend(Image.new("RGB", (240, 240), BG1), tiles[i], a)
        img.paste(tile, (x, y))
        d2 = ImageDraw.Draw(img, "RGBA")
        d2.rectangle([x, y, x + 240, y + 240], outline=tuple(int(v * a) for v in GRID), width=2)
        txt(d2, (x + 120, y + 256), names[i], 22, SUB, anchor="ma", alpha=a)
    # right panel: patch window schematic
    px, py, S = 1330, 330, 460
    a = seg(t, 1.9, 0.6)
    rrect(d, (px - 30, py - 30, px + S + 230, py + S + 30), 24, fill=tuple(int(v * a) for v in CARD))
    zoom = tiles[0].resize((S, S), Image.NEAREST)
    if a >= 0.99:
        img.paste(zoom, (px, py))
    else:
        img.paste(Image.blend(Image.new("RGB", (S, S), CARD), zoom, a), (px, py))
    d2 = ImageDraw.Draw(img, "RGBA")
    d2.rectangle([px, py, px + S, py + S], outline=tuple(int(v * a) for v in AMBER), width=3)
    # animated scanning frame 33x33
    ph = seg(t, 2.4, 1.6)
    cell = S / 33
    ix = int(10 + 13 * (0.5 - 0.5 * math.cos(ph * math.pi * 2)))
    iy = int(10 + 13 * (0.5 - 0.5 * math.sin(ph * math.pi * 2)))
    d2.rectangle([px + ix * cell, py + iy * cell, px + (ix + 9) * cell, py + (iy + 9) * cell],
                 outline=CYAN, width=3)
    txt(d2, (px + S / 2, py + S + 66), "33 × 33 px  ≈  1.1 km of context", 30, TXT, bold=True, anchor="ma", alpha=a)
    txt(d2, (px + S / 2, py + S + 112), "per site · float16 · 958 sites", 24, SUB, anchor="ma", alpha=a)


def sc_climate(img, d, t, L):
    title_block(d, "The climate link", "NASA POWER precipitation climatology (keyless, open)", t)
    # left: monthly schematic
    mx, my, mw, mh = 120, 360, 900, 540
    rrect(d, (mx - 30, my - 40, mx + mw + 30, my + mh + 80), 24, fill=CARD)
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    # schematic monsoon-shaped annual cycle (illustrative of site normals)
    vals = np.array([0.15, 0.2, 0.35, 0.5, 0.9, 2.4, 3.6, 3.4, 2.5, 0.9, 0.25, 0.15])
    vals = vals / vals.max()
    bw = mw / 12 - 18
    for i, v in enumerate(vals):
        a = seg(t, 0.3 + i * 0.06, 0.4)
        hh = mh * 0.82 * v * a
        x = mx + 10 + i * (mw / 12)
        mon = 5 <= i <= 8
        col = AMBER if mon else (70, 88, 116)
        d.rectangle([x, my + mh - hh, x + bw, my + mh], fill=tuple(int(c * (1 if mon else a)) for c in col))
        txt(d, (x + bw / 2, my + mh + 18), months[i], 22, SUB, anchor="ma")
    a2 = seg(t, 1.6, 0.5)
    d.rectangle([mx + 10 + 5 * (mw / 12) - 6, my - 10, mx + 10 + 9 * (mw / 12), my + mh],
                outline=tuple(int(v * a2) for v in AMBER), width=3)
    txt(d, (mx + 10 + 5 * (mw / 12) + 4, my - 4), "Jun–Sep monsoon", 24, AMBER, alpha=a2)
    txt(d, (mx + mw / 2, my + mh + 60), "monthly rainfall shape (schematic of site normals)", 22, SUB, anchor="ma", alpha=seg(t, 0.4, 0.6))
    # right: stats
    stats = [
        ("~1,100–1,500 mm", "annual precipitation at typical sites"),
        ("≈ 80%", "arrives June–September"),
        ("3 covariates", "annual · monsoon total · seasonality"),
    ]
    for i, (big, small) in enumerate(stats):
        a = seg(t, 0.8 + i * 0.3, 0.5)
        y = 360 + i * 180
        txt(d, (1160, y), big, 58, CYAN if i == 1 else TXT, bold=True, alpha=a)
        txt(d, (1160, y + 76), small, 27, SUB, alpha=a)


def sc_cvv(img, d, t, L):
    title_block(d, "The honest yardstick", "Spatial block cross-validation vs random splits", t)
    box = (140, 330, 940, 950)
    map_axes(d, box, "0.5° blocks → 5 folds (illustrative)")
    # blocks colored by fold
    a0 = seg(t, 0.3, 0.8)
    for lon in np.arange(80.0, 88.0, 0.5):
        for lat in np.arange(26.5, 30.5, 0.5):
            f = block_fold(lon + 0.25, lat + 0.25)
            x0, y1 = lonlat_to_px(lon, lat, box)
            x1, y0 = lonlat_to_px(lon + 0.5, lat + 0.5, box)
            col = FOLDCOL[f]
            al = 0.30 + 0.25 * ((math.sin(lon * 12.9 + f) + 1) / 2)
            d.rectangle([x0, y0, x1, y1], fill=tuple(int(v * al * a0) for v in col))
    draw_points(d, box, POS, (240, 240, 240), r=4, t=seg(t, 1.0, 1.2))
    txt(d, (box[0] + 14, box[1] + 10), "whole blocks held out together", 22, SUB, alpha=seg(t, 1.6, 0.5))
    # right: comparison card
    rx, ry = 1020, 350
    a = seg(t, 1.8, 0.6)
    rrect(d, (rx, ry, rx + 780, ry + 560), 24, fill=tuple(int(v * a) for v in CARD))
    rows = [
        ("Random CV", "tests a model on its neighbor's slope", "0.901 ± 0.047", RED, "optimistic, unstable"),
        ("Spatial CV", "whole 0.5° blocks held out", "0.893 ± 0.023", GREEN, "honest, stable"),
    ]
    for i, (h, s, v, col, tag) in enumerate(rows):
        y = ry + 60 + i * 250
        txt(d, (rx + 50, y), h, 44, col, bold=True, alpha=a)
        txt(d, (rx + 50, y + 62), s, 26, SUB, alpha=a)
        txt(d, (rx + 730, y + 10), v, 44, TXT, bold=True, anchor="ra", alpha=a)
        txt(d, (rx + 730, y + 70), f"AUC · {tag}", 24, col, anchor="ra", alpha=a)
    txt(d, (rx + 50, ry + 500), "fold variance grows ×6 under random splits (CNN: ±0.048 vs ±0.008)",
        24, AMBER, alpha=a)


def sc_models(img, d, t, L):
    title_block(d, "Four contenders", "same data · same 5-fold spatial CV · same seed", t)
    cards = [
        ("Logistic regression", "the classic statistical LSM baseline", "11 scalar factors", CYAN),
        ("Gradient boosting", "HistGradientBoosting", "11 scalar factors", CYAN),
        ("MLP", "256-128-64, dropout", "11 scalar factors", CYAN),
        ("HybridCNN", "ResNet-style patch encoder + context fusion", "8×33×33 patches + 6 scalars", AMBER),
    ]
    for i, (h, s, inp, col) in enumerate(cards):
        a = seg(t, 0.3 + i * 0.3, 0.5)
        x = 120 + (i % 2) * 880
        y = 330 + (i // 2) * 300
        hot = i == 3
        fill = CARD_HI if hot else CARD
        d.rounded_rectangle((x, y, x + 840, y + 260), radius=24,
                            fill=tuple(int(v * a) for v in fill),
                            outline=tuple(int(v * a) for v in (col if hot else GRID)), width=3 if hot else 2)
        txt(d, (x + 44, y + 40), h, 42, col if hot else TXT, bold=True, alpha=a)
        txt(d, (x + 44, y + 104), s, 26, SUB, alpha=a)
        txt(d, (x + 44, y + 176), inp, 26, TXT, alpha=a)
        if hot:
            # mini architecture glyph: patch -> blocks -> fuse -> head
            gy = y + 118
            xs = x + 500
            boxes = [(xs, 60, AMBER, "33²"), (xs + 80, 44, AMBER, ""), (xs + 152, 34, AMBER, ""),
                     (xs + 216, 28, CYAN, "⊕"), (xs + 272, 22, GREEN, "")]
            for j, (bx, bw2, c, lab) in enumerate(boxes):
                aj = seg(t, 0.9 + j * 0.12, 0.35)
                d.rounded_rectangle((bx, gy - bw2 / 2, bx + 30, gy + bw2 / 2), radius=6,
                                    fill=tuple(int(v * aj) for v in c))
                if j < 4:
                    d.line([(bx + 30, gy), (bx + 44, gy)], fill=(120, 134, 152), width=2)
            txt(d, (xs + 306, gy + 52), "terrain → encode → fuse ⊕ → head", 20, SUB, alpha=a)


def sc_results(img, d, t, L):
    title_block(d, "Results", "5-fold spatial block CV · 479 events vs 479 absences", t)
    rows = [
        ("Logistic regression", 0.852, 0.041, CYAN, MV["logreg"]),
        ("MLP", 0.875, 0.024, CYAN, MV["mlp"]),
        ("Gradient boosting", 0.893, 0.023, GREEN, MV["gbm"]),
        ("HybridCNN", 0.893, 0.013, AMBER, MV["cnn"]),
    ]
    x0, x1 = 560, 1660
    lo, hi = 0.80, 0.92
    for i, (name, auc, std, col, m) in enumerate(rows):
        a = seg(t, 0.3 + i * 0.3, 0.5)
        y = 360 + i * 140
        txt(d, (120, y + 34), name, 34, TXT, anchor="lm", alpha=a)
        grow = ease(min(1, max(0, (t - (0.3 + i * 0.3)) / 0.9)))
        frac = (auc - lo) / (hi - lo)
        bw = (x1 - x0) * frac * grow
        d.rounded_rectangle((x0, y, x0 + bw, y + 68), radius=14, fill=col)
        # error whisker
        if grow > 0.98:
            ex0 = x0 + (x1 - x0) * (auc - std - lo) / (hi - lo)
            ex1 = x0 + (x1 - x0) * (auc + std - lo) / (hi - lo)
            ey = y + 34
            d.line([(ex0, ey), (ex1, ey)], fill=TXT, width=3)
            d.line([(ex0, ey - 10), (ex0, ey + 10)], fill=TXT, width=3)
            d.line([(ex1, ey - 10), (ex1, ey + 10)], fill=TXT, width=3)
        txt(d, (x0 + bw + 24, y + 34), f"{auc:.3f}", 40, TXT, bold=True, anchor="lm", alpha=grow)
    a2 = seg(t, 2.2, 0.6)
    txt(d, (120, 950), "AUC-ROC (Tesla V100 rerun; original run: CNN 0.898 ± 0.008) — whiskers: ±1 fold σ",
        26, SUB, alpha=a2)
    chips = [("tie at the top", AMBER), ("CNN: most fold-stable", CYAN), ("GBM: best AP 0.855", GREEN)]
    cxp = 120
    for label, col in chips:
        a3 = seg(t, 2.6, 0.5)
        wch = tw(d, label, 25) + 56
        rrect(d, (cxp, 860, cxp + wch, 916), 28, outline=tuple(int(v * a3) for v in col), width=2)
        txt(d, (cxp + wch / 2, 888), label, 25, col, anchor="mm", alpha=a3)
        cxp += wch + 24


def sc_ablation(img, d, t, L):
    title_block(d, "Is bigger better?  No.", "New ablation study · Tesla V100 · same spatial CV", t)
    groups = [
        ("Width (33 px patches)", [("w 32", 0.893, AMBER), ("w 64", 0.877, SUB), ("w 128", 0.882, SUB)]),
        ("Patch size (width 32)", [("17 px", 0.883, SUB), ("33 px", 0.893, AMBER), ("65 px", 0.894, SUB)]),
        ("Context scalars", [("patches only", 0.886, SUB), ("+ climate/fault/river", 0.893, AMBER)]),
    ]
    lo, hi = 0.865, 0.900
    for g, (gname, bars) in enumerate(groups):
        a = seg(t, 0.3 + g * 0.5, 0.5)
        x = 120 + g * 570
        rrect(d, (x, 330, x + 550, 800), 24, fill=tuple(int(v * a) for v in CARD))
        txt(d, (x + 30, 360), gname, 28, TXT, bold=True, alpha=a)
        for i, (lab, v, col) in enumerate(bars):
            ag = seg(t, 0.5 + g * 0.5 + i * 0.15, 0.5)
            y = 430 + i * 110
            frac = (v - lo) / (hi - lo)
            bw = 340 * frac * ag
            d.rectangle([x + 30, y, x + 30 + 340, y + 46], fill=(30, 39, 55))
            d.rectangle([x + 30, y, x + 30 + bw, y + 46], fill=tuple(int(c * (1 if col == AMBER else 0.9)) for c in col))
            txt(d, (x + 30, y - 8), lab, 24, SUB, alpha=a)
            if ag > 0.9:
                txt(d, (x + 396, y + 23), f"{v:.3f}", 32, TXT, bold=True, anchor="lm")
    av = seg(t, 2.4, 0.6)
    rrect(d, (120, 840, 1800, 940), 24, fill=tuple(int(v * av) for v in (60, 42, 16)))
    txt(d, (W / 2, 890), "Context helps.  Capacity doesn't.  —  the bottleneck is 958 sites, not the network",
        36, AMBER, bold=True, anchor="mm", alpha=av)


def sc_map(img, d, t, L):
    # Ken Burns over the real susceptibility map (src 1430x1040: zoom >= 1920/1430 = 1.343)
    src = Image.open(ROOT / "runs" / "susceptibility_map_cnn_default.png").convert("RGB")
    p = ease(min(1, t / max(0.8, L - 0.8)))
    zoom = 1.35 + 0.20 * p
    cx = 0.5 + 0.05 * math.sin(p * math.pi)
    cy = 0.44 + 0.04 * math.sin(p * math.pi * 0.8)
    sw, sh = W / zoom, H / zoom
    sx = max(0.0, min(src.width - sw, (src.width - sw) * cx))
    sy = max(0.0, min(src.height - sh, (src.height - sh) * cy))
    frame = src.crop((int(sx), int(sy), int(sx + sw), int(sy + sh))).resize((W, H), Image.LANCZOS)
    img.paste(frame, (0, 0))
    # gradient scrim bottom for lower third
    scrim = Image.new("L", (1, 320))
    for i in range(320):
        scrim.putpixel((0, i), int(238 * (i / 320) ** 1.3))
    scrim = scrim.resize((W, 320))
    black = Image.new("RGB", (W, 320), BG0)
    img.paste(black, (0, H - 320), scrim)
    dd = ImageDraw.Draw(img, "RGBA")
    a = seg(t, 0.4, 0.6)
    txt(dd, (120, H - 216), "The map: 1.5 million susceptibility scores", 46, (255, 255, 255), bold=True, alpha=a)
    txt(dd, (120, H - 146), "Gorkha–Trishuli–Langtang corridor · high-risk channels follow the drainage network · events overlaid",
        29, (205, 214, 226), alpha=a)
    a2 = seg(t, 1.2, 0.6)
    rrect(dd, (W - 580, H - 226, W - 120, H - 146), 20, fill=(13, 17, 23, 235))
    txt(dd, (W - 350, H - 186), "runs/susceptibility.npz → GIS", 25, CYAN, anchor="mm", alpha=a2)


def sc_limits(img, d, t, L):
    title_block(d, "What this map is not", "stated plainly, because deployment decisions follow", t)
    items = [
        ("Reports ≠ inventory", "GLC counts reported events; remote slopes are undercounted", "bias toward roads & towns"),
        ("Absences are assumptions", "sampled 'safe' points 2 km from events — probably safe, not certainly", "1:1 pseudo-absences"),
        ("Climate at ~55 km", "POWER normals miss valley-scale rain shadows", "CHIRPS/WorldClim next"),
        ("Rankings, not probabilities", "0.8 means 'higher than 0.5', not an 80% chance", "calibration on roadmap"),
    ]
    for i, (h, s, tag) in enumerate(items):
        a = seg(t, 0.3 + i * 0.35, 0.5)
        x = 120 + (i % 2) * 900
        y = 330 + (i // 2) * 300
        d.rounded_rectangle((x, y, x + 860, y + 260), radius=24,
                            fill=tuple(int(v * a) for v in CARD), outline=tuple(int(v * a) for v in GRID), width=2)
        txt(d, (x + 44, y + 36), h, 36, TXT, bold=True, alpha=a)
        txt(d, (x + 44, y + 106), s, 26, SUB, alpha=a)
        rrect(d, (x + 44, y + 178, x + 44 + tw(d, tag, 24) + 44, y + 226), 16,
              outline=tuple(int(v * a) for v in RED), width=2)
        txt(d, (x + 66, y + 202), tag, 24, RED, anchor="lm", alpha=a)


def sc_roadmap(img, d, t, L):
    title_block(d, "The road ahead is data, not hardware", "where the next AUC point lives", t)
    items = [
        ("Extend the inventory", "COOLR live feed · ICIMOD · post-2016 monsoons"),
        ("Sharper rainfall", "CHIRPS ~5 km replaces ~55 km POWER normals"),
        ("Dynamic hazard", "antecedent 7/30-day rain → next-72-hour advisories"),
        ("Municipal pilot", "maps in Gandaki offices, scored on next monsoon"),
    ]
    for i, (h, s) in enumerate(items):
        a = seg(t, 0.3 + i * 0.3, 0.5)
        y = 330 + i * 150
        d.rounded_rectangle((120, y, 1080, y + 126), radius=20, fill=tuple(int(v * a) for v in CARD))
        d.rectangle((120, y, 152, y + 126), fill=tuple(int(v * a) for v in CYAN))
        txt(d, (190, y + 30), h, 34, TXT, bold=True, alpha=a)
        txt(d, (190, y + 80), s, 25, SUB, alpha=a)
    a2 = seg(t, 1.8, 0.7)
    txt(d, (1170, 430), "Everything open:", 34, TXT, bold=True, alpha=a2)
    for i, s in enumerate(["code (MIT) · data · paper (6-page preprint)",
                           "one command: bash scripts/run_all.sh",
                           "V100 ablations in runs/metrics_v100.json"]):
        txt(d, (1170, 500 + i * 54), s, 26, SUB, alpha=a2)
    a3 = seg(t, 2.5, 0.8)
    rrect(d, (1140, 700, 1800, 920), 24, fill=tuple(int(v * a3) for v in (60, 42, 16)))
    txt(d, (1470, 770), "The mountain remembers the rain.", 34, TXT, bold=True, anchor="mm", alpha=a3)
    txt(d, (1470, 840), "Now the map does too.", 34, AMBER, bold=True, anchor="mm", alpha=a3)
    txt(d, (1470, 890), "Yue Wang · Chenfeng Zhang — AI4Climate Track 2, 2026", 22, SUB, anchor="mm", alpha=a3)


SCENES = [
    ("ai4climate · track 2 · nepal", sc_title),
    ("the problem", sc_problem),
    ("the inventory", sc_inventory),
    ("terrain as images", sc_terrain),
    ("the climate link", sc_climate),
    ("honest evaluation", sc_cvv),
    ("the models", sc_models),
    ("results", sc_results),
    ("ablations · tesla v100", sc_ablation),
    ("the map", sc_map),
    ("limitations", sc_limits),
    ("roadmap · close", sc_roadmap),
]


def main():
    global BG
    FRAMES.mkdir(exist_ok=True)
    build_assets()
    BG = make_bg()

    # scene timings from narration
    durs = {}
    for f in sorted((AUDIO).glob("seg*.wav")):
        d = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)]))
        durs[f.stem] = d
    HEAD, TAIL = 0.25, 0.55
    lengths = [max(8.0, durs[f"seg{i+1:02d}"] + HEAD + TAIL) for i in range(len(SCENES))]
    starts = np.cumsum([0.0] + lengths[:-1])
    total = sum(lengths)
    print("scene lengths:", [round(x, 2) for x in lengths], "| total", round(total, 1), "s")
    (VID / "timing.json").write_text(json.dumps(
        {"lengths": lengths, "starts": starts.tolist(), "total": total}, indent=1))

    n_total = int(round(total * FPS))
    fi = 0
    for si, (kick, fn) in enumerate(SCENES):
        L = lengths[si]
        nf = int(round(L * FPS))
        for k in range(nf):
            tt = k / FPS
            base = chrome(kick, si + 1, len(SCENES), (starts[si] + tt) / total)
            if fn is not sc_map:
                fn(base, ImageDraw.Draw(base, "RGBA"), tt, L)
                base = fade_edge(base, tt, 0.3, L)
            else:
                tmp = base.copy()
                fn(tmp, ImageDraw.Draw(tmp, "RGBA"), tt, L)
                base = fade_edge(tmp, tt, 0.3, L)
            base.save(FRAMES / f"f_{fi:05d}.png", compress_level=1)
            fi += 1
        print(f"scene {si+1} done ({nf} frames)")
    print("total frames:", fi)


if __name__ == "__main__":
    main()
