import argparse
import csv
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

from mnist_mlp.preprocessing import prepare_digit
from mnist_mlp.reporting import write_json
from mnist_mlp.training import load_model


def main():
    parser = argparse.ArgumentParser(description="Öğrenciye ait ayrı rakam görsellerini sırasıyla tanır.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("handwritten/digits"))
    parser.add_argument("--labels", type=Path, default=Path("handwritten/labels.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results") / f"handwritten_{uuid4().hex[:8]}",
                        help="Yeni sonuç klasörü; belirtilmezse benzersiz bir klasör oluşturulur.")
    parser.add_argument("--polarity", choices=["auto", "dark-on-light", "light-on-dark"], default="auto")
    args = parser.parse_args()
    paths = sorted(p for p in args.input_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not paths:
        parser.error(f"{args.input_dir} içinde rakam görseli yok. Her rakamı 00.png, 01.png ... olarak kaydedin.")
    labels = {}
    if args.labels.exists():
        with args.labels.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if row["file"] in labels or row["label"] not in set("0123456789"):
                    parser.error("CSV etiketleri benzersiz dosya adı ve 0–9 arasında tek rakam içermeli.")
                labels[row["file"]] = int(row["label"])
    filenames = {p.name for p in paths}
    if labels and set(labels) != filenames:
        parser.error("labels.csv her giriş görselini tam bir kez içermeli; fazladan/kayıp satır var.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Çıktı klasörü dolu. Yeni bir --output yolu seçin.")
    model, saved = load_model(args.checkpoint, torch.device("cpu"))
    torch.set_num_threads(4)
    prepared = []
    for path in paths:
        try:
            prepared.append(prepare_digit(path, saved["normalization"], args.polarity))
        except ValueError as error:
            parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    processed = args.output / "processed"
    processed.mkdir()
    rows = []
    figure, axes = plt.subplots(2, len(paths), figsize=(max(5, len(paths) * 1.35), 4), squeeze=False)
    with torch.inference_mode():
        probabilities = model(torch.stack([p[0] for p in prepared])).softmax(dim=1)
    for index, (path, (_, image), probability) in enumerate(zip(paths, prepared, probabilities)):
        predicted = int(probability.argmax())
        confidence = float(probability.max())
        image.save(processed / f"{path.stem}.png")
        row = {"file": path.name, "prediction": predicted, "softmax_score": confidence, "label": labels.get(path.name)}
        rows.append(row)
        with Image.open(path) as original:
            axes[0, index].imshow(ImageOps.exif_transpose(original).convert("L"), cmap="gray")
        axes[0, index].set_title(path.name, fontsize=8)
        axes[1, index].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[1, index].set_title(f"Pred: {predicted} ({confidence:.1%})\nTrue: {labels.get(path.name, '?')}", fontsize=8)
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.suptitle("Original / MNIST-style input")
    figure.tight_layout()
    figure.savefig(args.output / "predictions.png", dpi=150)
    plt.close(figure)
    predicted_number = "".join(str(r["prediction"]) for r in rows)
    expected = "".join(str(labels[p.name]) for p in paths) if labels else None
    accuracy = sum(r["prediction"] == r["label"] for r in rows) / len(rows) if labels else None
    write_json(args.output / "predictions.json", {"checkpoint": str(args.checkpoint), "rows": rows,
               "predicted_number": predicted_number, "expected_number": expected,
               "digit_accuracy": accuracy, "exact_match": predicted_number == expected if labels else None})
    lines = ["# Kendi el yazısı değerlendirmesi", "", f"- Model: `{args.checkpoint}`", f"- Görsel sayısı: {len(paths)}",
             f"- Tahmin edilen numara: `{predicted_number}`", f"- Gerçek numara: `{expected}`" if expected else "- Gerçek etiketler girilmedi; accuracy hesaplanmadı."]
    if labels:
        lines.extend([f"- Rakam accuracy: {accuracy:.2%}", f"- Numaranın tamamı doğru mu: {'Evet' if predicted_number == expected else 'Hayır'}"])
    lines.extend(["", "| Dosya | Gerçek | Tahmin | Softmax skoru |", "|---|---:|---:|---:|"])
    lines.extend(f"| {r['file']} | {r['label'] if r['label'] is not None else '?'} | {r['prediction']} | {r['softmax_score']:.2%} |" for r in rows)
    lines.extend(["", "![Orijinal ve ön işlenmiş görüntüler](predictions.png)", "",
                  "Softmax skoru kalibre edilmiş doğruluk garantisi değildir. Az sayıdaki kişisel rakam sonucu, "
                  "MNIST'in 10.000 test örneğiyle doğrudan eşdeğer bir başarı ölçümü değildir. "
                  "Yanlış tahminlerde orijinal ve işlenmiş görüntüleri karşılaştırın: kırpma, gölge, "
                  "çizgi kalınlığı, rakam şekli ve merkezleme etkili olabilir. "
                  "Bu örnekleri kullanarak yapılan ek ayarlar sonrası aynı görseller bağımsız test sayılmaz.", ""])
    (args.output / "handwritten_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Tahmin: {predicted_number}")
    if accuracy is not None:
        print(f"Rakam accuracy: {accuracy:.2%}; numara tam eşleşme: {predicted_number == expected}")
    print(f"Rapor: {args.output / 'handwritten_evaluation.md'}")


if __name__ == "__main__":
    main()
