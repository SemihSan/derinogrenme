"""Bir fotoğrafı kayıtlı bölgelerden rakamlara ayırır; rakam üretmez veya yeniden çizmez."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser(description="Kaydedilmiş koordinatlardan gerçek fotoğrafı rakamlara ayırır")
    parser.add_argument("--regions", type=Path, default=Path("handwritten/crops.json"))
    parser.add_argument("--output", type=Path, default=Path("handwritten/digits"))
    parser.add_argument("--preview", type=Path, default=Path("handwritten/crop_preview.png"))
    args = parser.parse_args()
    spec = json.loads(args.regions.read_text(encoding="utf-8"))
    source = Path(spec["source"])
    with Image.open(source) as image:
        original = ImageOps.exif_transpose(image).convert("RGB")
    regions = spec["regions"]
    for region in regions:
        left, top, right, bottom = region
        if not (0 <= left < right <= original.width and 0 <= top < bottom <= original.height):
            parser.error(f"Görsel dışına taşan veya boş bölge: {region}")
    args.output.mkdir(parents=True, exist_ok=True)
    destinations = [args.output / f"{i:02d}.png" for i in range(len(regions))]
    if any(path.exists() for path in destinations):
        parser.error("Kırpılmış görseller mevcut. Yeni bir --output klasörü seçin.")
    scale = min(4, 1600 / max(original.size))
    preview = original.resize((round(original.width * scale), round(original.height * scale)))
    drawing = ImageDraw.Draw(preview)
    for index, (region, destination) in enumerate(zip(regions, destinations)):
        original.crop(tuple(region)).save(destination)
        left, top, right, bottom = [round(coordinate * scale) for coordinate in region]
        color = "red" if index % 2 == 0 else "blue"
        drawing.rectangle((left, top, right - 1, bottom - 1), outline=color, width=2)
        drawing.text((left + 2, top - 15), f"{index:02d}", fill=color)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.preview)
    print(f"{len(regions)} rakam kırpıldı. Ön izleme: {args.preview}")


if __name__ == "__main__":
    main()
