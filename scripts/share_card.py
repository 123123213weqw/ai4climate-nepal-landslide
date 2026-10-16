"""Generate a social share card for the submission (runs/share_card.png)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.config import RUNS_DIR

CARD = RUNS_DIR / "share_card.png"
MAP = RUNS_DIR / "susceptibility_map_cnn_default.png"

fig = plt.figure(figsize=(10.5, 14), dpi=140)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# background
ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#0b1526"))
ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color="#10233d"))

# headline block
ax.text(0.06, 0.955, "AI4Climate 2026 · Track 02 Nepal", color="#7fd4a8",
        fontsize=17, fontweight="bold", va="center")
ax.text(0.06, 0.912, "Rainfall-driven landslide susceptibility, learned end-to-end from open data",
        color="#ffffff", fontsize=15, va="center")

# map image
img = Image.open(MAP)
ax_img = fig.add_axes([0.05, 0.33, 0.9, 0.50])
ax_img.imshow(img)
ax_img.axis("off")
for s in ax_img.spines.values():
    s.set_edgecolor("#2b4a6f")
    s.set_linewidth(2)

# stats row
stats = [
    ("0.898", "AUC-ROC, spatial\nblock CV (best of 4)"),
    ("84%", "of 479 catalog events\nare rainfall-triggered"),
    ("200 m", "susceptibility map,\n376k scored windows"),
]
for i, (big, small) in enumerate(stats):
    x = 0.06 + i * 0.313
    ax.add_patch(plt.Rectangle((x, 0.175), 0.29, 0.115, color="#10233d"))
    ax.text(x + 0.145, 0.248, big, color="#ffd166", fontsize=30,
            fontweight="bold", ha="center", va="center")
    ax.text(x + 0.145, 0.203, small, color="#c9d8ea", fontsize=10.5,
            ha="center", va="center")

# takeaway + links
ax.text(0.5, 0.135, "Residual CNN over 8-channel terrain patches + climate/geo fusion — no API keys, one command to reproduce",
        color="#ffffff", fontsize=12.5, ha="center", va="center")
ax.text(0.5, 0.088, "github.com/123123213weqw/ai4climate-nepal-landslide",
        color="#7fd4a8", fontsize=13, fontweight="bold", ha="center", va="center")
ax.text(0.5, 0.052, "deepseek.club/topic/5205", color="#8fb3d9",
        fontsize=12, ha="center", va="center")
ax.text(0.5, 0.018, "Yue Wang · Chenfeng Zhang — AI4Climate Global Youth Climate Solutions Challenge",
        color="#6d87a8", fontsize=9.5, ha="center", va="center")

fig.savefig(CARD, facecolor="#0b1526")
print(f"saved {CARD}")
