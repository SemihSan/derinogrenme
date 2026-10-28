# Kendi el yazını ekle

## Projedeki mevcut gerçek fotoğraf

Kullanıcının gönderdiği fotoğraf `original/student_number.png` olarak korundu. Doğru numara kullanıcı tarafından `24120205084` olarak doğrulandı; `digits/00.png`–`digits/10.png` ve `labels.csv` hazır. Kırpma sınırları `crops.json`, görsel kontrolü `crop_preview.png` içindedir. Mevcut modelin kişisel sonuçları `results/handwritten/` klasöründedir: 3/11 (%27,27), tüm numara eşleşmedi.

Mevcut veriyi yeniden değerlendirmek için sonuçları koruyan yeni bir çıktı yolu kullan:

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoint results/main/checkpoints/wide.pt --output results/handwritten_repeat
```

Orijinalden aynı kırpımları yeniden üretmek için:

```powershell
.\.venv\Scripts\python.exe prepare_handwritten.py --output handwritten/digits_repeat --preview handwritten/crop_preview_repeat.png
```

## İkinci gerçek fotoğraf

İkinci gerçek fotoğraf da eklendi: `original/student_number_v2.jpeg`. İlgili kırpma bölgeleri, ön izleme, 11 rakam ve etiketler `v2/` içinde; sonuçlar `results/handwritten_v2/` içindedir. Aynı modelin ikinci çekimdeki başarısı **8/11 = %72,73** oldu; üç sıfır yanlış tanındı. İlk fotoğraf ve sonuçları korunur.

İkinci fotoğrafın kırpımlarını yeniden üretmek için:

```powershell
.\.venv\Scripts\python.exe prepare_handwritten.py --regions handwritten/v2/crops.json --output handwritten/v2/digits_repeat --preview handwritten/v2/crop_preview_repeat.png
```

## Yeni fotoğraf hazırlama

1. Okul numaranı beyaz, çizgisiz kâğıda koyu kalemle yaz. Rakamlar arasında boşluk bırak; rakamlar birleşmesin.
2. Aydınlık ortamda, kamerayı kâğıda paralel tutarak çek. Numaranın tamamının fotoğrafını `original/` içine ekle.
3. Her rakamı ayrı kırp. Görselde tek rakam ve etrafında bir miktar boşluk olsun; kenarlara değmesin. `digits/00.png`, `digits/01.png`, … diye soldan sağa sıralı kaydet. `10.png` öncesinde sıralama bozulmasın diye iki hane kullan.
4. `labels.csv` dosyasına gerçek rakamları sırayla yaz. Örnek biçim (kendi numaranla değiştir):

```csv
file,label
00.png,2
01.png,0
02.png,3
```

Bu sayılar yalnızca CSV biçimini gösterir; gerçek numaran olduğu varsayılmaz. Tekrarlanan rakamlar da ayrı dosya olmalı. Dosya isimlerinde gerçek rakamı belirtmek zorunlu değil; model dosya ismine bakmaz.

5. Eğitimde seçilen modeli `results/main/selection.json` dosyasında bul. Örneğin seçilen model `wide` ise:

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoint results/main/checkpoints/wide.pt
```

Ön işleme EXIF yönünü düzeltir, grayscale yapar, kenar parlaklığına göre gerekirse renkleri ters çevirir, kontrastı ayarlar, boş kenarları kırpar, en uzun kenarı 20 piksele getirip 28×28 siyah tuvalde merkezler. Train istatistikleriyle normalize eder. Aşırı gölge ve lekeleri otomatik temizlemez; `predictions.png` üzerinden kontrol et. Otomatik renk seçimi yanlışsa `--polarity dark-on-light` veya `--polarity light-on-dark` kullan.

Repo teslimi için orijinal fotoğrafı, kırpılmış rakamları ve `results/handwritten/` çıktılarını ekle. Okul numaran repo'yu gören kişiler tarafından görülebilir; hocanın kabul ettiği repo görünürlüğünü seç.
