import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(output, runs, metadata, selected, handwriting_dir=None):
    summary_rows = []
    for run in runs:
        name = run["config"]["name"]
        history = run["history"]
        with (output / f"{name}_history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, metric in zip(axes, ["loss", "accuracy"]):
            for prefix, label in [("train", "Train"), ("validation", "Validation")]:
                axis.plot([h["epoch"] for h in history], [h[f"{prefix}_{metric}"] for h in history], label=label)
            axis.axvline(run["best_epoch"], color="gray", linestyle=":", label="Selected epoch")
            axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"{name}: {metric}")
            axis.legend()
            axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output / f"{name}_curves.png", dpi=150)
        plt.close(figure)
        best = run["best"]
        summary_rows.append({"experiment": name, "parameters": run["parameters"],
                             "best_epoch": run["best_epoch"], **{k: v for k, v in best.items() if k != "epoch"},
                             "test_accuracy": run["test"]["accuracy"], "training_seconds": run["training_seconds"]})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(output / "results.json", {"metadata": metadata, "selected": selected, "runs": runs})
    lines = ["# MNIST MLP — Deney Raporu", "", "Bu dosya gerçek eğitim çıktılarından otomatik oluşturulmuştur.", "",
             "## Yöntem", "", f"- Veri ayrımı: {metadata['train_count']} train / {metadata['validation_count']} validation / {metadata['test_count']} test.",
             f"- Seed: {metadata['seed']}; cihaz: {metadata['device']}; işlemci: {metadata['processor']}.",
             f"- Train üzerinden hesaplanan normalizasyon: mean={metadata['normalization']['mean']:.6f}, std={metadata['normalization']['std']:.6f}.",
             "- Stratified ayrım indeksleri split_indices.npz dosyasındadır. Validation ve test normalizasyon istatistiklerine katılmaz.",
             "- Tüm ağlar sıfırdan eğitildi: Flatten → Linear → ReLU → Linear → ReLU → Linear(10). CNN kullanılmadı.",
             "- CrossEntropyLoss ham logits alır. Accuracy doğru tahmin sayısı / örnek sayısıdır; loss olasılık hatasını da yansıtır.",
             "- Her deneyde en düşük validation loss görülen epoch kaydedildi. Deney seçimi de yalnızca bu loss ile yapıldı.",
             "- Bütün yapılandırmalar ve seçilen model test değerlendirmesinden önce sabitlendi (selection.json). Test sonuçları ile ayar yapılmadı.",
             "- Train/validation metrikleri epoch sonunda aynı ağırlıklarla ölçülür. Tablodaki değerler seçilen epoch'a aittir; son epoch ayrıca results.json içinde bulunur.",
             "- Süre eğitim döngüsü, her epoch train/validation ölçümü ve checkpoint yazımını içerir; indirme, son test ve grafik çizimi dahil değildir.", "", "## Hiperparametreler", "",
             "| Deney | Hidden katmanlar | Optimizer | LR | Batch | Epoch |", "|---|---|---|---|---|---|"]
    for run in runs:
        c = run["config"]
        lines.append(f"| {c['name']} | {c['hidden_sizes']} | {c['optimizer']} | {c['learning_rate']} | {c['batch_size']} | {c['epochs']} |")
    lines.extend(["", "## Ölçülen sonuçlar", "", "| Deney | Parametre | Epoch | Train loss | Val loss | Train acc | Val acc | Test acc | Süre (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary_rows:
        lines.append(f"| {row['experiment']} | {row['parameters']} | {row['best_epoch']} | {row['train_loss']:.4f} | {row['validation_loss']:.4f} | {row['train_accuracy']:.2%} | {row['validation_accuracy']:.2%} | {row['test_accuracy']:.2%} | {row['training_seconds']:.1f} |")
    winner = next(r for r in runs if r["config"]["name"] == selected)
    lines.extend(["", "## Genel değerlendirme", "", f"Validation loss ölçütüne göre seçilen model **{selected}**, epoch **{winner['best_epoch']}** oldu. "
                  f"Train–validation accuracy farkı {(winner['best']['train_accuracy'] - winner['best']['validation_accuracy']) * 100:.2f} yüzde puan; "
                  f"ayrılmış test kümesindeki accuracy {winner['test']['accuracy']:.2%}.", "",
                  "wide deneyi yalnızca katman genişliklerini; low_lr deneyi yalnızca öğrenme oranını baseline'a göre değiştirir. "
                  "Daha fazla parametre daha yüksek kapasite ve hesaplama maliyeti getirir. Daha düşük LR aynı epoch bütçesinde daha yavaş yakınsayabilir. "
                  "Yorum tablodaki ölçümlerle birlikte yapılmalıdır; tek seed ile yapılan deney istatistiksel üstünlük kanıtı değildir. "
                  "Süreler tek koşunun duvar saati ölçümüdür; süreç ısınması ve sistem yükü süreyi etkiler. "
                  "Bu çalışma mimarileri hız açısından sıralayan kontrollü, tekrarlanmış bir benchmark değildir. "
                  "Train loss düşerken validation loss artıyorsa overfitting işaretidir; seçilen checkpoint son epoch olmak zorunda değildir.", "",
                  "## Grafikler", ""])
    baseline = next((r for r in runs if r["config"]["name"] == "baseline"), None)
    if baseline:
        comparisons = []
        for run in runs:
            if run is baseline:
                continue
            difference = (run["best"]["validation_accuracy"] - baseline["best"]["validation_accuracy"]) * 100
            loss_difference = run["best"]["validation_loss"] - baseline["best"]["validation_loss"]
            comparisons.extend([f"{run['config']['name']} için baseline'a göre validation accuracy farkı "
                                f"{difference:+.2f} yüzde puan, validation loss farkı {loss_difference:+.4f}. "
                                "Bu sonuç aynı epoch bütçesi ve tek seed koşullarına aittir.", ""])
        position = lines.index("## Grafikler")
        lines[position:position] = comparisons
    for run in runs:
        name = run["config"]["name"]
        lines.extend([f"![{name} loss ve accuracy]({name}_curves.png)", ""])
    lines.extend(["![Seçilen model confusion matrix](confusion_matrix.png)", ""])
    if handwriting_dir is not None:
        personal = json.loads((handwriting_dir / "predictions.json").read_text(encoding="utf-8"))
        expected_checkpoint = output / "checkpoints" / f"{selected}.pt"
        if expected_checkpoint.resolve() != Path(personal["checkpoint"]).resolve():
            raise ValueError("Kişisel değerlendirme seçili modelin checkpoint'i ile eşleşmiyor.")
        relative = Path(os.path.relpath(handwriting_dir, output)).as_posix()
        correct = sum(row["prediction"] == row["label"] for row in personal["rows"])
        count = len(personal["rows"])
        lines.extend(["## Kendi el yazısı değerlendirmesi", "",
                      f"Öğrencinin sağladığı fotoğraf {count} ayrı rakam görseline kırpıldı. Gerçek numara kullanıcı tarafından **{personal['expected_number']}** olarak doğrulandı. "
                      f"Seçilen {selected} modeli yeniden eğitilmeden kullanıldı. Tahmin **{personal['predicted_number']}**; "
                      f"rakam accuracy **{correct}/{count} = {personal['digit_accuracy']:.2%}**; tam eşleşme: {'evet' if personal['exact_match'] else 'hayır'}.", "",
                      f"MNIST test accuracy {winner['test']['accuracy']:.2%} iken bu tek fotoğraftaki başarı {personal['digit_accuracy']:.2%} oldu. "
                      "Bu ölçümler farklı dağılım ve örnek büyüklüklerine aittir; "
                      "bu kadar az rakam genel gerçek dünya başarısını tahmin etmek için yeterli değildir. "
                      "Işık, yüzey dokusu, kontrast, yazı biçimi, kırpma ve ölçekleme sonucu etkileyebilir. "
                      "Bu sonuç hem kırpma/ön işleme hem sınıflandırma aşamalarını içeren tüm akışın performansıdır. "
                      "Bu fotoğrafa özgü gözlemler ve önceki çekimle karşılaştırma aşağıdaki değerlendirme dosyasındadır.", "",
                      f"[Rakam bazında sonuçlar ve değerlendirme]({relative}/handwritten_evaluation.md)", "",
                      f"[Fotoğrafa ilişkin gözlemler]({relative}/observations.md)", "",
                      f"![Gerçek rakamlar ve modele verilen girdiler]({relative}/predictions.png)", ""])
    else:
        lines.extend(["## Kendi el yazısı — tamamlanması gereken bölüm", "",
                  "Öğrenciye ait gerçek görüntüler henüz eklenmedi; kişisel el yazısı başarısı ölçülmedi. "
                  "MNIST test accuracy, öğrenci numarası üzerinde aynı başarıyı garanti etmez. "
                  "Kamera ışığı, çizgi kalınlığı, eğim ve rakamın konumu distribution shift yaratır.", "",
                  "`predict.py` çalıştırılınca `handwritten_evaluation.md` oluşur. "
                  "Teslimden önce bu dosyayı, gerçek görselleri ve ön işleme karşılaştırmasını repo'ya ekleyin. "
                  "Örnek sayısı az olduğundan bu ölçüm genel el yazısı başarısı olarak yorumlanmamalıdır.", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def confusion_plot(output, targets, predictions):
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (targets, predictions), 1)
    np.savetxt(output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    for i in range(10):
        for j in range(10):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    axis.set(xticks=range(10), yticks=range(10), xlabel="Predicted", ylabel="True", title="Selected MLP - MNIST test")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(figure)
