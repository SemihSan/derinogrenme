"""Yalnızca teslim dosyalarını kontrol edip ZIP paketler; üçüncü taraf bağımlılık gerekmez."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
TOP_FILES = [
    ".gitignore", ".gitattributes", "README.md", "TESLIM.md", "requirements.txt", "requirements-lock.txt",
    "experiments.json", "train.py", "predict.py", "prepare_handwritten.py", "package_submission.py",
]
FOLDERS = ["mnist_mlp", "tests", "docs", "handwritten", "results/main", "results/handwritten", "results/handwritten_v2"]


def submission_files():
    paths = [ROOT / name for name in TOP_FILES]
    for folder in FOLDERS:
        paths.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    missing = [path.relative_to(ROOT).as_posix() for path in paths if not path.is_file()]
    required = [
        "results/main/REPORT.md", "results/main/results.json", "results/main/split_indices.npz",
        "results/main/checkpoints/baseline.pt", "results/main/checkpoints/wide.pt", "results/main/checkpoints/low_lr.pt",
        "handwritten/original/student_number.png", "handwritten/original/student_number_v2.jpeg",
        "results/handwritten/predictions.json", "results/handwritten_v2/predictions.json",
    ]
    missing.extend(name for name in required if not (ROOT / name).is_file())
    if missing:
        raise ValueError("Eksik teslim dosyaları: " + ", ".join(sorted(set(missing))))
    for folder in ["handwritten/digits", "handwritten/v2/digits"]:
        if len(list((ROOT / folder).glob("*.png"))) != 11:
            raise ValueError(f"{folder} içinde 11 gerçek rakam görseli bulunmalı.")
    return sorted(set(paths))


def check_markdown_links(paths):
    included = {path.resolve() for path in paths}
    broken = []
    for path in paths:
        if path.suffix != ".md":
            continue
        content = re.sub(r"```[\s\S]*?```", "", path.read_text(encoding="utf-8"))
        for raw_target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            target = raw_target.strip().strip("<>")
            if re.match(r"^[a-zA-Z][\w+.-]*:", target) or target.startswith("#"):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            if resolved not in included:
                broken.append(f"{path.relative_to(ROOT).as_posix()} → {target}")
    if broken:
        raise ValueError("Paket dışında/eksik Markdown hedefleri:\n" + "\n".join(broken))


def main():
    parser = argparse.ArgumentParser(description="MNIST ödevinin temiz teslim paketini hazırlar")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/mnist-mlp-submission.zip")
    parser.add_argument("--check-only", action="store_true", help="Dosya ve bağlantıları kontrol eder; ZIP üretmez.")
    args = parser.parse_args()
    try:
        paths = submission_files()
        check_markdown_links(paths)
    except ValueError as error:
        parser.error(str(error))
    print(f"Teslim dosyaları ve Markdown bağlantıları doğrulandı: {len(paths)} dosya.")
    if args.check_only:
        return
    if args.output.exists():
        parser.error("ZIP zaten mevcut. Yeni bir --output yolu kullanın.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with ZipFile(args.output, mode="x", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            archive.writestr("mnist-mlp/" + relative, content)
            records.append({"path": relative, "bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest()})
        manifest = {"format_version": 1, "project": "MNIST MLP", "files": records}
        archive.writestr("mnist-mlp/MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    # Arşivi yeniden okuyup sıkıştırma bütünlüğünü ve her dosyanın hash'ini doğrula.
    with ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP bütünlük doğrulaması başarısız.")
        for record in records:
            content = archive.read("mnist-mlp/" + record["path"])
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError(f"Hash eşleşmedi: {record['path']}")
    print(f"Doğrulanmış paket: {args.output} ({args.output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
