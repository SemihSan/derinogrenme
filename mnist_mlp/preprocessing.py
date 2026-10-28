from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def prepare_digit(path: Path, normalization, polarity="auto"):
    """Tek rakamı siyah fonda açık renkli, merkezlenmiş 28x28 görüntüye çevirir."""
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        gray = background.convert("L")
    pixels = np.asarray(gray)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if polarity == "dark-on-light" or (polarity == "auto" and np.median(border) > 127):
        gray = ImageOps.invert(gray)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    pixels = np.asarray(gray).copy()
    pixels[pixels < 32] = 0
    ys, xs = np.where(pixels > 0)
    if len(xs) == 0:
        raise ValueError(f"Rakam bulunamadı (boş/tek renk görüntü): {path}")
    cropped = Image.fromarray(pixels).crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = 20 / max(cropped.size)
    resized = cropped.resize(tuple(max(1, round(v * scale)) for v in cropped.size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 255
    yy, xx = np.indices(array.shape)
    mass = array.sum()
    dx = round(13.5 - float((xx * array).sum() / mass))
    dy = round(13.5 - float((yy * array).sum() / mass))
    shifted = Image.new("L", (28, 28), 0)
    shifted.paste(canvas, (dx, dy))
    array = np.asarray(shifted, dtype=np.float32).copy() / 255
    tensor = torch.from_numpy((array - normalization["mean"]) / normalization["std"])
    return tensor.unsqueeze(0), shifted
