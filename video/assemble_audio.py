"""Assemble the narration track aligned to scene frame counts.

Per scene: 250 ms head silence + narration + silence padded to the exact
frame-quantized scene length. Then concat all 12 into narration.wav.
"""
import json
import subprocess
from pathlib import Path

VID = Path(__file__).resolve().parent
FPS = 30
HEAD_MS = 250

timing = json.loads((VID / "timing.json").read_text())
lengths = timing["lengths"]

pads = []
for i, L in enumerate(lengths):
    nf = round(L * FPS)
    exact = nf / FPS
    out = VID / "audio" / f"pad{i+1:02d}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(VID / "audio" / f"seg{i+1:02d}.wav"),
         "-af", f"adelay={HEAD_MS}|{HEAD_MS},apad",
         "-t", f"{exact:.6f}", "-ar", "44100", "-ac", "2", str(out)],
        check=True)
    pads.append(out)
    print(f"scene {i+1:02d}: {exact:.3f}s")

lst = VID / "audio" / "list.txt"
lst.write_text("\n".join(f"file '{p.name}'" for p in pads))
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
     "-i", str(lst), "-c", "copy", str(VID / "audio" / "narration.wav")], check=True)
d = subprocess.check_output(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "csv=p=0", str(VID / "audio" / "narration.wav")])
print("narration.wav:", float(d), "s | frames available:",
      len(list((VID / "frames").glob("f_*.png"))), "=",
      round(len(list((VID / 'frames').glob('f_*.png'))) / FPS, 3), "s")
