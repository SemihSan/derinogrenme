"""Örnek: python train.py --output results/reproduction --device cpu"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from mnist_mlp.data import test_data, training_data
from mnist_mlp.reporting import confusion_plot, write_json, write_report
from mnist_mlp.training import evaluate, fit, load_model


def main():
    parser = argparse.ArgumentParser(description="CNN kullanmadan MNIST MLP deneyleri")
    parser.add_argument("--config", type=Path, default=Path("experiments.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path,
                        default=Path("results") / f"train_{uuid4().hex[:8]}",
                        help="Yeni sonuç klasörü; belirtilmezse benzersiz bir klasör oluşturulur.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Çıktı klasörü dolu. Yeni bir --output yolu seçin; eski sonuçlar korunur.")
    configs = json.loads(args.config.read_text(encoding="utf-8"))
    if not configs or len({c["name"] for c in configs}) != len(configs):
        parser.error("Deney adları benzersiz ve liste boş olmamalı.")
    for config in configs:
        if not config["name"].replace("_", "").isalnum():
            parser.error("Deney adlarında yalnızca harf, rakam ve alt çizgi kullanın.")
        if config["epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
            parser.error("Epoch, batch size ve learning rate pozitif olmalı.")
        if config["optimizer"] not in ("Adam", "SGD"):
            parser.error("Optimizer Adam veya SGD olmalı.")
    if args.threads < 1:
        parser.error("threads pozitif olmalı.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA kullanılamıyor. --device cpu kullanın.")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    train, validation, train_indices, val_indices, normalization = training_data(args.data_dir, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "split_indices.npz", train=train_indices.numpy(), validation=val_indices.numpy())
    write_json(args.output / "experiments.json", configs)
    runs = [fit(config, train, validation, normalization, args.seed, device, args.output) for config in configs]
    selected = min(runs, key=lambda run: run["best"]["validation_loss"])["config"]["name"]
    # Test sonuçlarını görmeden önce model/epoch seçimini diske sabitle.
    write_json(args.output / "selection.json", {"selected": selected, "criterion": "minimum validation loss",
               "epochs": {r["config"]["name"]: r["best_epoch"] for r in runs}, "test_evaluated_at_selection": False})
    test = test_data(args.data_dir, normalization)
    test_loader = DataLoader(test, batch_size=1024)
    for run in runs:
        name = run["config"]["name"]
        model, _ = load_model(args.output / "checkpoints" / f"{name}.pt", device)
        metrics = evaluate(model, test_loader, device, collect=name == selected)
        if name == selected:
            confusion_plot(args.output, metrics.pop("targets"), metrics.pop("predictions"))
        run["test"] = metrics
        print(f"Final test {name}: {metrics['accuracy']:.2%}", flush=True)
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                "device": str(device), "threads": args.threads, "processor": platform.processor(),
                "python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "train_count": len(train),
                "validation_count": len(validation), "test_count": len(test), "normalization": normalization}
    write_report(args.output, runs, metadata, selected)
    print(f"Seçilen model: {selected}. Rapor: {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
