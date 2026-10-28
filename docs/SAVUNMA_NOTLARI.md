# Modeli ve deneyleri açıklama notları

Bu notlar kodun yerine geçmez; her cevabı ilgili fonksiyonu açıp açıklayarak çalış.

## 1. Görüntü modele nasıl giriyor?

MNIST görüntüsü 1×28×28 tensördür. Piksel değerleri önce 255'e bölünür; sonra yalnızca train örneklerinden hesaplanan mean=0.130728 ve std=0.308191 ile `(x - mean) / std` uygulanır. Flatten, görüntüyü 784 özellikli vektöre çevirir. MLP komşu pikselleri konvolüsyonla işlemez; her Linear katman girişinin tamamına bağlıdır.

## 2. MLP'nin yaptığı hesap nedir?

Baseline için `h1 = ReLU(W1*x + b1)`, `h2 = ReLU(W2*h1 + b2)`, `logits = W3*h2 + b3`. Boyutlar 784→128→64→10. ReLU negatif girdileri sıfırlar, pozitifleri geçirir. Aktivasyon kullanılmazsa ardışık Linear katmanlar tek bir doğrusal dönüşüme indirgenebilir; doğrusal olmayan karar sınırlarını öğrenme kapasitesi sınırlanır.

Parametre sayısı ağırlıklar ve bias'ların toplamıdır: `(784×128 + 128) + (128×64 + 64) + (64×10 + 10) = 109.386`. Geniş modelde 235.146 parametre vardır. İkisinde de iki hidden layer bulunur.

## 3. Neden CrossEntropyLoss ve neden çıkışta softmax yok?

Bir örnek için kayıp `-log(p_doğru_sınıf)` biçimindedir. PyTorch CrossEntropyLoss ham logits üzerinden log-softmax işlemini kendi içinde kararlı biçimde hesaplar. Etiket 0–9 arasında bir sınıf indeksidir. Eğitimde önce softmax uygulamak gerekmez. Tahminde en yüksek logits'in sınıfı seçilir; skor göstermek için softmax uygulanabilir.

Accuracy doğru tahmin sayısı / toplam örnek sayısıdır. İki model aynı accuracy'ye sahip olsa da doğru sınıfa verdikleri olasılıklar farklı olduğu için loss farklı olabilir. Bu nedenle en iyi epoch'u seçerken validation loss kullanıldı.

## 4. Bir eğitim adımında ne oluyor?

`optimizer.zero_grad()` önceki gradyanları temizler. Model ileri geçişle logits üretir. Loss hesaplanır, `loss.backward()` gradyanları çıkarır ve `optimizer.step()` Adam ile parametreleri günceller. Bir epoch tüm train örneklerinin bir kez işlenmesidir. Batch size 128, tek seferde 128 örneğin gradyan hesabına katılmasıdır; son batch daha küçük olabilir.

Train ve validation metrikleri epoch sonunda aynı ağırlıklarla `eval()` modunda ölçülür. `inference_mode()` değerlendirmede gradyan takibini kapatır. Loss toplamı örnek sayısına bölünür; küçük son batch'e diğer batch'lerle aynı ağırlığı vermek yanlış olur.

## 5. Train, validation ve test neden ayrı?

Train ağırlıkları öğrenmek, validation yapılandırma ve epoch seçmek, test ise seçim tamamlandıktan sonra sonucu ölçmek için kullanılır. Resmî MNIST train dosyasındaki 60.000 örnek sınıf başına oran korunarak 49.999 train ve 10.001 validation'a ayrıldı. Sınıf başına yuvarlama nedeniyle tam 50.000/10.000 olmadı. Resmî 10.000 test örneği bu ayrımın dışındadır.

Validation veya test mean/std hesabına katılmadı. Her deney aynı indekslerle çalıştı. Seed 42, karşılaştırmayı tekrarlanabilir kılar; farklı donanım ve sürümlerde bit düzeyinde eşit sonuç garantisi değildir.

## 6. Hangi deney neyi değiştirdi?

Baseline: hidden katmanlar [128,64], LR 0.001. Wide yalnızca katmanları [256,128] yaptı. Low_lr yalnızca öğrenme oranını 0.0003 yaptı. Adam, batch 128, epoch bütçesi 10 ve veri ayrımı ortaktır. Böylece iki değişiklik ayrı ayrı incelenir.

Wide minimum validation loss 0.0702 ile seçildi; baseline 0.0812, low_lr 0.0831 elde etti. Wide'ın seçilen 9. epoch'unda train accuracy %99,68, validation accuracy %98,15, son test accuracy %97,95'tir. Test sonuçlarıyla daha sonra ayar yapılmadı. Her modelin en iyi checkpoint'i kullanıldığı için son epoch ile tabloda seçilen epoch farklı olabilir.

## 7. Grafiklerde overfitting nasıl yorumlanır?

Train loss azalırken validation loss artıyorsa eğitim verisine uyum genellemeyi artırmıyor olabilir. Örneğin wide'ın 10. epoch validation loss'u 9. epoch'tan yüksektir; seçilen checkpoint bu yüzden 9'dur. Tek bir dalgalanma veya tek koşu kesin genelleme iddiası sağlamaz. Bu deneyde 10 epoch tamamlandı; eğitim erken durdurulmadı, en iyi epoch kaydedildi.

## 8. Kendi fotoğraflarına ne uygulandı?

Fotoğraflar kullanıcı tarafından sağlandı; numara `24120205084` olarak doğrulandı. Her biri kayıtlı koordinatlarla 11 ayrı rakama kırpıldı. Grayscale, renk ters çevirme, kontrast ayarı, eşikle boş kenar bulma, oranı koruyarak en uzun kenarı 20 piksele indirme, 28×28 tuvalde kütle merkezine göre hizalama ve train istatistikleriyle normalizasyon uygulandı.

MNIST'te olduğu gibi zemin koyu, rakam açık renklidir. Orijinaller, kırpımlar ve işlenmiş girdiler ayrı korunur. Hiçbir rakam yeniden çizilmedi. Model aynı wide checkpoint'idir; bu fotoğraflarla eğitilmedi.

## 9. MNIST'te %97,95 iken kendi yazında neden %72,73?

MNIST ve fotoğraflar farklı veri dağılımlarına sahiptir. Işık, yüzey dokusu, yazı şekli, kalınlık, eğim, kırpma ve merkezleme değişir. İlk çekimde 3/11, ikinci çekimde 8/11 doğru tahmin edildi. İkinci fotoğraftaki üç sıfır sırasıyla 9, 4 ve 9 tahmin edildi. Bu, tüm akışın başarısıdır; hataların yalnızca modelden kaynaklandığı kanıtlanmış değildir.

İki fotoğrafta birden fazla koşul değiştiği için iyileşmeyi yalnızca netliğe bağlayamayız. 11 rakamlık ölçümle genel gerçek dünya accuracy'si de tahmin edilemez. Softmax skoru yüksekken yanlış tahmin olabilir; örneğin ikinci fotoğrafta bir sıfır yaklaşık %99 skorla 9 sınıfına atandı.

## 10. Nasıl geliştirilebilir?

Önce çok sayıda farklı kişisel örnek toplanır; geliştirme ve bağımsız kişisel test ayrılır. Train/validation üzerinde arka plan temizleme, merkezleme ve augmentation etkileri kontrollü denenebilir. Daha fazla seed ile deney tekrarı belirsizliği ölçmeye yardım eder. Bu öneriler mevcut raporun sonuçlarını değiştirmek için uygulanmadı. Aynı test fotoğraflarına tekrar tekrar bakarak ayar yapıp sonucu bağımsız test diye sunmak doğru olmaz.

Eğitim süreleri bu koşudaki duvar saati ölçümleridir. Eşzamanlı yazılım kontrolleri ve ilk kullanım maliyetleri nedeniyle geniş modelin tablodaki daha kısa süresinden "geniş model daha hızlıdır" sonucu çıkarılamaz.
