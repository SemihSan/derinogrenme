import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import MLP


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


@torch.inference_mode()
def evaluate(model, loader, device, collect=False):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, count = 0.0, 0, 0
    predictions, targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predicted = logits.argmax(1)
        total_loss += criterion(logits, labels).item()
        correct += (predicted == labels).sum().item()
        count += len(labels)
        if collect:
            predictions.extend(predicted.cpu().tolist())
            targets.extend(labels.cpu().tolist())
    if not count:
        raise ValueError("Boş veri kümesi değerlendirilemez.")
    result = {"loss": total_loss / count, "accuracy": correct / count, "count": count}
    if collect:
        result.update(predictions=predictions, targets=targets)
    return result


def fit(config, train, validation, normalization, seed, device, output):
    seed_everything(seed)
    model = MLP(config["hidden_sizes"]).to(device)
    optimizer_class = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}[config["optimizer"]]
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=generator)
    train_eval = DataLoader(train, batch_size=1024)
    val_loader = DataLoader(validation, batch_size=1024)
    criterion = nn.CrossEntropyLoss()
    checkpoint = output / "checkpoints" / f"{config['name']}.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    best_loss, best_epoch, history = float("inf"), 0, []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        # Her iki metrik de aynı epoch sonu ağırlıklarıyla ve eval modunda ölçülür.
        train_metrics = evaluate(model, train_eval, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"],
               "train_accuracy": train_metrics["accuracy"],
               "validation_loss": val_metrics["loss"],
               "validation_accuracy": val_metrics["accuracy"]}
        history.append(row)
        if val_metrics["loss"] < best_loss:
            best_loss, best_epoch = val_metrics["loss"], epoch
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "normalization": normalization, "seed": seed,
                        "best_epoch": best_epoch}, checkpoint)
        print(f"{config['name']:8s} {epoch:02d}/{config['epochs']} "
              f"train_loss={row['train_loss']:.4f} val_loss={row['validation_loss']:.4f} "
              f"train_acc={row['train_accuracy']:.4%} val_acc={row['validation_accuracy']:.4%}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"config": config, "best_epoch": best_epoch,
            "best": history[best_epoch - 1], "last": history[-1], "history": history,
            "training_seconds": time.perf_counter() - start,
            "parameters": sum(p.numel() for p in model.parameters())}


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = MLP(saved["config"]["hidden_sizes"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved
