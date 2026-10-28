# MNIST ile MLP tabanlı el yazısı rakam tanıma

PyTorch ile sıfırdan eğitilen, **CNN içermeyen** MLP projesi. En az iki hidden layer, ReLU, CrossEntropyLoss ve Adam kullanır. Eğitim, karşılaştırma ve kişisel el yazısı değerlendirmesi ayrı komutlardır.

Öğrenci: **Semih — 24120205084**, İstanbul Medeniyet Üniversitesi, Bilgisayar Mühendisliği.

Proje deposu: [SemihSan/derinogrenme](https://github.com/SemihSan/derinogrenme).

## Sonuçlara hızlı bakış

| Deney | Validation loss | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Temel MLP | 0.0812 | %97,70 | %97,56 |
| Geniş MLP | **0.0702** | **%98,15** | **%97,95** |
| Düşük öğrenme oranı | 0.0831 | %97,44 | %97,44 |

Gerçek ayrım **49.999 train / 10.001 validation / 10.000 test**. En düşük validation loss ile geniş MLP'nin 9. epoch'u seçildi; test sonuçları daha sonra ölçüldü. Kişisel fotoğraflarda aynı model ilk çekimde **3/11**, ikinci çekimde **8/11** doğru rakam tanıdı.

[Tam deney raporu](results/main/REPORT.md) · [İki el yazısı fotoğrafının karşılaştırması](results/handwritten_v2/observations.md) · [Teslim dosyaları](TESLIM.md) · [Modeli açıklama notları](docs/SAVUNMA_NOTLARI.md)

## Kurulum (Python 3.12, Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU kurulumu ödev için yeterlidir. Ortam oluşturma `ensurepip` hatası verirse ve `uv` kuruluysa:

```powershell
python -m venv .venv --without-pip
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Bu projede gerçekten kullanılan sürümler `requirements-lock.txt` dosyasındadır. Aynı sürümleri yeni ortamda kurmak için:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Linux'ta `.venv/Scripts/python.exe` yerine `.venv/bin/python` kullan.

## Eğit ve raporla

```powershell
.\.venv\Scripts\python.exe train.py --output results/reproduction
```

İlk çalıştırma MNIST'i internetten indirir. Eğitim verisinin 60.000 örneği sınıf oranları korunarak yaklaşık 50.000 train / 10.000 validation olarak ayrılır; yuvarlamadan dolayı tam sayılar raporda belirtilir. Resmî 10.000 test örneği bağımsız tutulur. Normalizasyon mean/std değerleri yalnızca train üzerinden hesaplanır. İndirme sırasında test dosyalarının da diske gelmesi test verisinin eğitimde kullanıldığı anlamına gelmez.

`results/main/` repo ile gelen ölçülmüş referans koşudur; yeniden eğitim `results/reproduction/` altında kendi raporunu ve checkpoint'lerini oluşturur. Çıktı yolu verilmezse `train.py` benzersiz bir yeni klasör seçer. Aynı komutu tekrar çalıştıracaksan yeni `--output` seç veya bu argümanı çıkar. Yeni eğitimdeki seçili modeli o koşunun `selection.json` dosyasından kontrol et.

`experiments.json` üç önceden belirlenmiş deney içerir:

| Deney | Mimari | Öğrenme oranı | Amaç |
|---|---|---|---|
| baseline | 784 → 128 → 64 → 10 | 0.001 | Temel MLP |
| wide | 784 → 256 → 128 → 10 | 0.001 | Yalnızca kapasite etkisi |
| low_lr | 784 → 128 → 64 → 10 | 0.0003 | Yalnızca öğrenme oranı etkisi |

Tüm deneyler Adam, batch size 128, 10 epoch ve seed 42 kullanır. Her modelin en iyi epoch'u ve deneyler arasındaki seçim **minimum validation loss** ile yapılır. Test sonuçları seçimden sonra tek bir son değerlendirme aşamasında çıkarılır; test accuracy'ye bakarak model seçilmez veya yeniden ayar yapılmaz. Seed karşılaştırmayı kolaylaştırır; tek koşu kesin istatistiksel sonuç sağlamaz. Donanım/sürüm değişiminde aynı seed bit düzeyinde aynı sonucu garanti etmez.

Yeni deney için farklı `--output` kullan; mevcut sonuçların üzerine yazılmaz. Aynı test kümesine tekrar tekrar bakarak hiperparametre ayarlamak test sızıntısı yaratır. Ek geliştirmede yalnızca train/validation sonuçlarını kullan; son test değerlerini bağımsız değerlendirme olarak yorumlamak için protokolü koru.

## Üretilen çıktılar

- [Deney raporu](results/main/REPORT.md): yöntem, hiperparametreler, gerçek metrikler ve genel değerlendirme.
- `results/main/*_curves.png`: her deneyin train/validation loss ve accuracy grafikleri.
- `results/main/comparison.csv`, `results/main/results.json`: ölçümler ve ortam bilgisi.
- `results/main/split_indices.npz`: ayrık train/validation indeksleri.
- `results/main/selection.json`: testten önce sabitlenmiş model seçimi.
- `results/main/checkpoints/*.pt`: bu projede sıfırdan eğitilen modellerin en iyi validation loss ağırlıkları.
- `results/main/confusion_matrix.png`: seçilen modelin test hata dağılımı.

Üç checkpoint toplam yaklaşık 1,8 MB olduğu için teslim paketine dahil edilmiştir. Böylece aşağıdaki el yazısı komutu yeniden eğitim yapmadan çalışır. Bu ağırlıklar bu projenin eğitim koşusunda üretildi; başka bir çalışmadan hazır model alınmadı. Eğitim komutuyla sıfırdan yeniden üretilebilir. MNIST indirmesi repo'ya eklenmez.

## Kendi el yazını değerlendir

Kullanıcının iki gerçek fotoğrafı ve her biri için 11 kırpılmış rakam projeye eklendi. Doğrulanmış numara `24120205084`. Aynı wide modeli ve aynı ön işleme ile ilk fotoğrafta **3/11 = %27,27**, ikinci fotoğrafta **8/11 = %72,73** doğruluk elde edildi. İkinci tahmin `24129245984`; üç sıfır yanlış sınıflandırıldı. [İlk sonuç](results/handwritten/handwritten_evaluation.md), [ikinci sonuç](results/handwritten_v2/handwritten_evaluation.md) ve [iki çekimin karşılaştırması](results/handwritten_v2/observations.md) mevcut. Modeller kişisel fotoğraflara göre yeniden eğitilmedi. Mevcut çıktı klasörleri dolu olduğundan tekrar çalıştırmada yeni `--output` kullan.

İkinci fotoğrafın verileriyle yeniden değerlendirme:

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoint results/main/checkpoints/wide.pt --input-dir handwritten/v2/digits --labels handwritten/v2/labels.csv --output results/handwritten_v2_repeat
```

[Görsel hazırlama talimatlarını](handwritten/README.md) uygula. Orijinal numara fotoğrafını `handwritten/original/`, tek tek kırpılmış rakamları `handwritten/digits/` içine koy; gerçek etiketleri `handwritten/labels.csv` içine yaz. Görselleri öğrencinin kendisi oluşturmalı.

`selection.json` içindeki seçilen adı kullan (örnek `wide`):

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoint results/main/checkpoints/wide.pt
```

Çıktılar: orijinal/işlenmiş görsel karşılaştırması, rakam tahminleri, rakam accuracy, numaranın tam eşleşmesi ve seçtiğin çıktı klasöründeki `handwritten_evaluation.md` raporu. `--output` verilmezse yeni bir `results/handwritten_<kimlik>/` klasörü açılır; kayıtlı iki kişisel sonuç korunur. Numara metin olarak tutulur; baştaki sıfırlar kaybolmaz. Softmax skoru yalnızca model skorudur, doğruluk garantisi değildir. Hazır örnek veya yapay görsel öğrencinin gerçek el yazısı yerine kullanılmaz.

## Kod yapısı ve mantığı

```text
train.py                  Eğitim, validation seçimi ve son test
predict.py                Öğrencinin tek rakam görsellerini değerlendirme
prepare_handwritten.py    Gerçek fotoğrafı kayıtlı koordinatlarla kırpma
experiments.json          Önceden sabitlenmiş hiperparametreler
mnist_mlp/model.py        Linear/ReLU katmanlarından MLP
mnist_mlp/data.py         Stratified ayrım ve train istatistikleri
mnist_mlp/training.py     Eğitim, örnek ağırlıklı metrikler, checkpoint
mnist_mlp/preprocessing.py Gri ton, renk, boyut, merkezleme, normalizasyon
mnist_mlp/reporting.py    Grafik, CSV, JSON ve rapor
handwritten/              Öğrenciye ait gerçek fotoğraf ve etiketler
tests/test_pipeline.py    Veri sızıntısı/ölçüm/ön işleme kontrolleri
```

28×28 görüntü 784 özellikli vektöre açılır. Bir hidden layer `ReLU(Wx+b)` hesaplar; son katman 10 ham logits üretir. CrossEntropyLoss sınıf indeksleri ve logits ile çalışır; eğitim öncesi ayrıca softmax uygulanmaz. Geri yayılım gradyanları, Adam da ağırlık güncellemelerini hesaplar. Accuracy aynıyken loss farklı olabilir: modelin doğru/yanlış sınıflara verdiği olasılıkların dağılımı önemlidir. MLP piksellerin uzamsal komşuluğunu CNN gibi doğrudan modellemez; bu ödevde özellikle MLP kullanılır.

## Kontroller ve sorun giderme

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kontroller: ayrımın ayrık/tam/tekrarlanabilir olması; konvolüsyon bulunmaması; ağırlıkların bir optimizasyon adımında öğrenmesi; son eksik batch'in loss hesabı; iki renk düzeninin aynı girdiye dönüşmesi; boş görüntü reddi ve merkezleme.

- `ModuleNotFoundError`: aynı `.venv` Python'u ile kurulum ve çalıştırma yap.
- MNIST indirme hatası: bağlantıyı/proxy ayarını kontrol et; veri indikten sonra yeniden indirilmez.
- `Çıktı klasörü dolu`: yeni `--output results/yeni_deney` kullan.
- CUDA yok: varsayılan CPU ile çalıştır. GPU için donanıma uygun PyTorch kurulumu gerekir.
- Kişisel accuracy düşük: `predictions.png` içindeki 28×28 girdileri incele. Aydınlatma, şekil ve konum farkları distribution shift yaratabilir. Kendi görsellerini training verisine ekleyip aynı görüntüleri bağımsız test diye raporlama.

## Teslim

Kod, `requirements*.txt`, deney yapılandırması, `results/main/` rapor/grafik ve checkpoint'leri, iki gerçek fotoğraf, kırpılmış rakamlar, etiketler ve kişisel değerlendirme raporları hazırdır. `.venv`, önbellekler, MNIST indirmesi, ders PDF'leri ve `dist/` `.gitignore` ile dışarıda tutulur. GitHub'a yüklenecek temiz paketi hazırlamak için:

```powershell
python package_submission.py
```

Komut yalnızca Python standart kütüphanesini kullanır; gerekli dosyaları ve Markdown bağlantılarını denetler, temiz ZIP ve SHA-256 manifesti oluşturur. Mevcut ZIP'in üzerine yazmaz. GitHub'a ZIP'in içindeki proje klasörünün **dosyalarını** yükle; ZIP dosyasını tek başına repo'ya koyma. Sonra repo bağlantısını teslim et. Ayrıntılar [TESLIM.md](TESLIM.md) içindedir.

## Kaynaklar

- [PyTorch eğitim akışı](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Torchvision MNIST veri kümesi](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
