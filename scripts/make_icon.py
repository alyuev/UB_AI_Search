"""Draw the program icon (rounded blue tile, two text lines = bilingual, magnifier) into app/static/icon.ico.

Pure numpy + zlib, no Pillow. Run: .venv\\Scripts\\python scripts\\make_icon.py
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "static" / "icon.ico"
SIZES = [16, 24, 32, 48, 64, 128, 256]


def render(n: int) -> np.ndarray:
    ss = 4  # supersampling
    m = n * ss
    y, x = (np.mgrid[0:m, 0:m] + 0.5) / m  # 0..1
    img = np.zeros((m, m, 4), float)

    def paint(mask, rgb, a=1.0):
        for c in range(3):
            img[..., c] = np.where(mask, rgb[c] * a + img[..., c] * (1 - a), img[..., c])
        img[..., 3] = np.where(mask, np.maximum(img[..., 3], a), img[..., 3])

    # rounded tile with vertical gradient
    r, pad = 0.2, 0.03
    cx = np.clip(x, pad + r, 1 - pad - r)
    cy = np.clip(y, pad + r, 1 - pad - r)
    tile = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    top, bot = np.array([0.20, 0.45, 0.85]), np.array([0.10, 0.25, 0.60])
    for c in range(3):
        img[..., c] = np.where(tile, top[c] + (bot[c] - top[c]) * y, 0)
    img[..., 3] = tile.astype(float)

    # two "text" lines (two languages)
    for ly, w in ((0.30, 0.56), (0.44, 0.40)):
        line = (np.abs(y - ly) < 0.045) & (x > 0.16) & (x < 0.16 + w)
        paint(line & tile, (1, 1, 1), 0.9)

    # magnifier: ring + handle
    mx, my, rr = 0.58, 0.62, 0.17
    d = np.sqrt((x - mx) ** 2 + (y - my) ** 2)
    paint(d < rr, (0.10, 0.25, 0.60), 1.0)          # glass background hides lines under it
    paint((d >= rr - 0.055) & (d < rr), (1, 0.82, 0.25))
    hx = np.clip((x - mx + y - my) / 2, rr * 0.70, 0.38)  # along the 45° diagonal
    hd = np.sqrt((x - (mx + hx)) ** 2 + (y - (my + hx)) ** 2)
    paint((hd < 0.055) & tile, (1, 0.82, 0.25))

    img = img.reshape(n, ss, n, ss, 4).mean(axis=(1, 3))
    rgb = np.where(img[..., 3:] > 0, img[..., :3], 0)
    return (np.concatenate([rgb, img[..., 3:]], axis=2) * 255 + 0.5).clip(0, 255).astype(np.uint8)


def png(a: np.ndarray) -> bytes:
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[i].tobytes() for i in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    images = [png(render(s)) for s in SIZES]
    head = struct.pack("<HHH", 0, 1, len(images))
    off = 6 + 16 * len(images)
    entries = b""
    for s, data in zip(SIZES, images):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), off)
        off += len(data)
    OUT.write_bytes(head + entries + b"".join(images))
    (OUT.with_suffix(".png")).write_bytes(images[-1])
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
