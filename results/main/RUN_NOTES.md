# Çalıştırma ve doğrulama notları

28 Ekim 2026 tarihinde CPU üzerinde `python train.py --output results/main` çalıştırıldı. Üç modelin her biri sıfırdan 10 epoch eğitildi. Test sonuçları görülmeden minimum validation loss ile `wide`, epoch 9 seçildi. Test sonuçlarından sonra mimari veya hiperparametre değişikliği yapılmadı.

Eğitimle aynı zaman aralığında yazılım kontrolleri ve ortam tanılama komutları da çalıştı. Özellikle ilk deneyin süre ölçümü bu yükten ve ilk kullanım maliyetlerinden etkilenmiş olabilir. Tablodaki süreler bu koşunun gerçek ölçümleridir; modeller arasında kontrollü hız karşılaştırması sayılmaz. Rapor açıklamaları ve biçimi sonradan kaydedilmiş sonuçlardan yeniden üretildi; model tekrar eğitilmedi veya test üzerinde yeniden değerlendirilmedi.

## Doğrulama

- `python -m unittest discover -s tests -v`: 5 test geçti.
- Train/validation indeksleri ayrık, toplam 60.000 örneği tam olarak kapsıyor.
- Her deneyde 10 epoch kaydı ve 10.000 örnek üzerinden son test sonucu bulunuyor.
- Confusion matrix toplamı 10.000; rapordaki seçili deney minimum validation loss ile uyumlu.
- Loss/accuracy grafikleri ve confusion matrix görsel olarak kontrol edildi.
- El yazısı komutu geçici sentetik girdilerle uçtan uca doğrulandı. Bu dosyalar yalnızca test girdileriydi; öğrencinin el yazısı olarak sunulmadı ve teslim çıktısına eklenmedi.

## Gerçek el yazısı değerlendirmesi

Kullanıcının sağladığı gerçek fotoğraf projeye eklendi. Kullanıcı numarayı `24120205084` olarak doğruladı; 11 rakam kırpıldı. Mevcut wide checkpoint'i ile yapılan kişisel testte 3/11 (%27,27) doğruluk elde edildi. Ayrıntılar `../handwritten/handwritten_evaluation.md` ve `../handwritten/observations.md` içindedir. Kişisel fotoğraf sonucu görüldükten sonra model, kırpımlar veya ön işleme performansı yükseltmek amacıyla değiştirilmedi.

## Kalan teslim adımı

İkinci bir gerçek fotoğraf daha sağlandı. 11 rakam yeni fotoğrafa göre görsel incelemeyle kırpıldı ve aynı checkpoint/ön işleme ile değerlendirildi: **8/11 (%72,73)**, tahmin `24129245984`. Üç sıfır yanlış sınıflandırıldı. İlk değerlendirme korunuyor; karşılaştırma `../handwritten_v2/observations.md` içindedir. İkinci sonuca bakılarak model veya ön işleme değiştirilmedi. Ana rapor artık ikinci değerlendirmeyi ve iki çekimin karşılaştırmasını bağlantılar. Kırpma ön izleme betiğinde yalnızca büyük fotoğrafların ön izleme boyutu sınırlandı; modele verilen rakamları değiştiren bir kod değişikliği yapılmadı.

Kod, MNIST raporları, iki gerçek fotoğraf, toplam 22 rakam görseli, doğrulanmış etiketler ve iki kişisel değerlendirme teslim dosyaları arasındadır. Hedef repo [SemihSan/derinogrenme](https://github.com/SemihSan/derinogrenme).

## Son teslim kontrolü

Güncel kodla 5 test tekrar geçti. Train ve predict komutlarının varsayılan çıktısı artık benzersiz yeni bir klasör kullanır; kayıtlı raporlarla çakışmaz. Bu değişiklik eğitim algoritmasını veya kaydedilmiş ölçümleri değiştirmez. Üç küçük checkpoint de teslim dosyaları arasına alındı. `package_submission.py` gerekli dosyaları, yerel Markdown bağlantılarını ve oluşturulan ZIP'teki tüm SHA-256 özetlerini denetler. Kurulum, yeniden eğitim, kişisel değerlendirme ve modeli açıklama notları README, TESLIM ve docs klasöründedir.
