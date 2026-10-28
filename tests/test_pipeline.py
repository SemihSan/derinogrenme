import tempfile
import unittest
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_mlp.data import stratified_split
from mnist_mlp.model import MLP
from mnist_mlp.preprocessing import prepare_digit
from mnist_mlp.training import evaluate


class PipelineTests(unittest.TestCase):
    def test_split_disjoint_complete_stratified_and_reproducible(self):
        labels = torch.arange(10).repeat_interleave(60)
        train, val = stratified_split(labels, seed=42)
        self.assertEqual(len(set(train.tolist()) & set(val.tolist())), 0)
        self.assertEqual(sorted(torch.cat([train, val]).tolist()), list(range(600)))
        self.assertTrue(torch.equal(val, stratified_split(labels, seed=42)[1]))
        self.assertEqual(torch.bincount(labels[val]).tolist(), [10] * 10)

    def test_mlp_has_no_convolution_and_can_learn_step(self):
        torch.manual_seed(42)
        model = MLP()
        self.assertFalse(any(isinstance(m, nn.modules.conv._ConvNd) for m in model.modules()))
        images, labels = torch.randn(8, 1, 28, 28), torch.arange(8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = nn.CrossEntropyLoss()(model(images), labels)
        optimizer.zero_grad()
        before.backward()
        optimizer.step()
        after = nn.CrossEntropyLoss()(model(images), labels)
        self.assertEqual(model(images).shape, (8, 10))
        self.assertLess(after.item(), before.item())

    def test_eval_uses_sample_weighted_loss(self):
        # Son batch bir örnek: batch ortalamalarını ortalamak yanlış sonuç verir.
        logits = torch.tensor([[3., 0.], [3., 0.], [0., 3.]])
        labels = torch.tensor([0, 0, 0])
        metrics = evaluate(nn.Identity(), DataLoader(TensorDataset(logits, labels), batch_size=2), torch.device("cpu"))
        self.assertAlmostEqual(metrics["loss"], nn.CrossEntropyLoss()(logits, labels).item(), places=6)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)

    def test_handwriting_polarity_shape_center_and_normalization(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-") as directory:
            image = Image.new("L", (90, 100), 255)
            ImageDraw.Draw(image).line((30, 20, 30, 80), fill=0, width=8)
            path, inverse = Path(directory) / "light.png", Path(directory) / "dark.png"
            image.save(path)
            ImageOps.invert(image).save(inverse)
            norm = {"mean": 0.13, "std": 0.31}
            tensor, processed = prepare_digit(path, norm)
            other, _ = prepare_digit(inverse, norm)
            self.assertEqual(tensor.shape, (1, 28, 28))
            self.assertTrue(torch.equal(tensor, other))
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertAlmostEqual(tensor[0, 0, 0].item(), -0.13 / 0.31, places=6)
            array = np.asarray(processed)
            yy, xx = np.indices(array.shape)
            self.assertLess(abs((xx * array).sum() / array.sum() - 13.5), 0.6)
            self.assertLess(abs((yy * array).sum() / array.sum() - 13.5), 0.6)
            Image.new("L", (20, 20), 255).save(path)
            with self.assertRaises(ValueError):
                prepare_digit(path, norm)

    def test_prediction_cli_preserves_leading_zero_and_reports_real_labels(self):
        # Yapay girdiler yalnızca yazılım testidir; öğrenci el yazısı sonucu değildir.
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-") as directory:
            root = Path(directory)
            inputs = root / "digits"
            inputs.mkdir()
            image = Image.new("L", (80, 80), 255)
            ImageDraw.Draw(image).line((40, 15, 40, 65), fill=0, width=7)
            for name in ("00.png", "01.png"):
                image.save(inputs / name)
            labels = root / "labels.csv"
            with labels.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows([["file", "label"], ["00.png", "0"], ["01.png", "1"]])
            model = MLP()
            for parameter in model.parameters():
                nn.init.zeros_(parameter)
            checkpoint = root / "model.pt"
            torch.save({"state_dict": model.state_dict(), "config": {"hidden_sizes": [128, 64]},
                        "normalization": {"mean": 0.13, "std": 0.31}}, checkpoint)
            result = subprocess.run([sys.executable, "predict.py", "--checkpoint", str(checkpoint),
                                     "--input-dir", str(inputs), "--labels", str(labels),
                                     "--output", str(root / "out")], capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            report = json.loads((root / "out" / "predictions.json").read_text(encoding="utf-8"))
            self.assertEqual(report["expected_number"], "01")
            self.assertEqual(report["predicted_number"], "00")
            self.assertEqual(report["digit_accuracy"], 0.5)
            self.assertFalse(report["exact_match"])
            self.assertTrue((root / "out" / "predictions.png").exists())


if __name__ == "__main__":
    unittest.main()
