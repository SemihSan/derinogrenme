# Teslim durumu

**Öğrenci:** Semih, 24120205084 — İstanbul Medeniyet Üniversitesi, Bilgisayar Mühendisliği.

## Ödev koşullarının karşılığı

| Koşul | Projedeki karşılığı |
|---|---|
| CNN kullanılmaması | `mnist_mlp/model.py`: yalnızca Flatten, Linear ve ReLU |
| En az iki hidden layer | Baseline 128→64; geniş model 256→128 |
| ReLU | İki hidden layer sonrasında ReLU |
| Uygun loss / optimizer | CrossEntropyLoss / Adam |
| Sıfırdan eğitim | Rastgele başlangıç, her deney 10 epoch; hazır dış model yok |
| Train / validation / test | 49.999 / 10.001 / 10.000; ayrım indeksleri `results/main/split_indices.npz` |
| Testin geliştirmede kullanılmaması | Seçim minimum validation loss ile testten önce `selection.json` içine kaydedildi |
| Train / validation loss ve accuracy | `results/main/REPORT.md`, epoch CSV'leri ve grafikler |
| Test accuracy ve eğitim süresi | Raporun sonuç tablosu ve `comparison.csv` |
| Hiperparametrelerin açıklanması | `experiments.json`, rapor ve README |
| Öğrencinin gerçek el yazısı | `handwritten/original/` altında kullanıcının sağladığı iki fotoğraf |
| 28×28, grayscale, renk ve normalizasyon | `mnist_mlp/preprocessing.py`; `processed/` ve `predictions.png` çıktıları |
| Gerçek el yazısı sonucu | İlk çekim 3/11, ikinci çekim 8/11; her iki ölçüm de raporlandı |
| Çalıştırılabilir kod | README kurulum komutları, kayıtlı checkpoint'ler ve `tests/` |

## Repo'da bulunacak dosyalar

- `README.md`, `TESLIM.md`, `requirements.txt`, `requirements-lock.txt`, `experiments.json`, `.gitignore`, `.gitattributes`.
- `train.py`, `predict.py`, `prepare_handwritten.py`, `package_submission.py` ve `mnist_mlp/`.
- `handwritten/`: orijinaller, koordinatlar, etiketler, kırpılmış rakamlar ve ön izlemeler.
- `results/main/`: ölçülmüş eğitim raporu, CSV/JSON dosyaları, grafikler, ayrım indeksleri ve üç checkpoint.
- `results/handwritten/`, `results/handwritten_v2/`: kişisel ölçümler, 28×28 girdiler ve yorumlar.
- `tests/` ve [modeli açıklama notları](docs/SAVUNMA_NOTLARI.md).

Sanal ortam, önbellek, ham MNIST indirmesi ve ders PDF'leri teslim dosyaları arasında değildir. `package_submission.py` bu sınırlarla ZIP hazırlar; arşiv içindeki `MANIFEST.json` her dosyanın boyutunu ve SHA-256 özetini içerir.

## GitHub'a yükleme

1. ZIP'i çıkart. İçindeki `mnist-mlp` klasöründe `README.md` ile Python dosyalarının birlikte olduğunu kontrol et.
2. Bu proje için hedef repo [SemihSan/derinogrenme](https://github.com/SemihSan/derinogrenme). Yeni bir kopya yükleyeceksen kendi repo adresini kullan.
3. Bu klasörün içindeki dosyaları repo'nun köküne yükle. `.venv` veya tüm ders klasörünü yükleme.
4. Repo ana sayfasında README'yi, rapor grafiklerini ve gerçek fotoğrafları açarak kontrol et.
5. Hocaya repo bağlantısını gönder. Özel repo kullanılıyorsa öğretim elemanının erişimi olmalı.

Komut satırıyla yüklemeyi tercih edersen, **ZIP'ten çıkardığın temiz proje klasöründe**:

```powershell
git init
git add .
git commit -m "Add MNIST MLP experiments and handwriting evaluation"
git branch -M main
git remote add origin https://github.com/SemihSan/derinogrenme.git
git push -u origin main
```

Komutlar boş bir repo'ya ilk yükleme içindir; zaten klonlanmış repoda yeniden `git init` veya `git remote add` çalıştırılmaz. Repo adresi ve `main` dalı, bu projenin hedefini gösterir.

## Hızlı doğrulama

README'deki bağımlılık kurulumundan sonra:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe predict.py --checkpoint results/main/checkpoints/wide.pt --input-dir handwritten/v2/digits --labels handwritten/v2/labels.csv
```

İkinci komut yeni bir sonuç klasörü oluşturur. Kaydedilmiş geniş model ile beklenen tahmin `24129245984`, rakam accuracy %72,73'tür. Sıfırdan yeniden eğitim için README'deki komutu kullan; ortam farklılıkları eğitim sonuçlarını etkileyebilir.
