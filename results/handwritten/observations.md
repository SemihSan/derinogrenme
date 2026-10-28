# Bu fotoğrafa ilişkin gözlemler

Kullanıcı 28 Ekim 2026 tarihinde gerçek el yazısı fotoğrafını sağladı ve doğru numarayı **24120205084** olarak doğruladı. Görsel 357×96 piksel boyutunda; yazı bulanık ve çizgili kâğıt üzerindedir.

Orijinal dosya `handwritten/original/student_number.png` altında korunur. Fotoğraf yeniden çizilmedi, yapay rakam üretilmedi. `handwritten/crops.json` koordinatlarıyla 11 ayrı rakam kırpıldı. Bölgeler model tahminleri görülmeden görsel incelemeyle seçildi; `handwritten/crop_preview.png` ayırma sınırlarını gösterir. Bazı rakamlar yakın olduğu için dikdörtgen kırpımlarda komşu çizgi parçaları kalmış olabilir.

Önceden validation loss ile seçilmiş **wide, epoch 9** checkpoint'i kullanıldı. Modelin ağırlıkları, hiperparametreleri ve MNIST sonuçları değiştirilmedi. Mevcut genel ön işleme akışı uygulandı: grayscale, renk ters çevirme, kontrast ayarı, sınır kırpma, oranı koruyarak boyutlandırma, 28×28 merkezleme ve train mean/std ile normalizasyon.

| Ölçüm | Sonuç |
|---|---|
| Kullanıcının doğruladığı gerçek numara | `24120205084` |
| Model tahmini | `24119141416` |
| Doğru rakamlar | İlk üç rakam: `2`, `4`, `1` |
| Rakam accuracy | 3 / 11 = **%27,27** |
| Numaranın tamamı doğru mu? | Hayır |
| Aynı modelin MNIST test accuracy'si | %97,95 |

İşlenmiş görüntülerde özellikle üst/alt kâğıt çizgileri ve yer yer komşu rakam parçaları görülüyor. Bunlar merkezleme ve ölçekleme hesabına da katılabiliyor. Bazı rakamların yazılış biçimi MNIST örneklerinden farklı. Düşük performansı yalnızca MLP'ye veya yalnızca fotoğraf kalitesine bağlamak bu deneyden çıkarılabilecek bir sonuç değildir; ölçüm kırpma, ön işleme ve model aşamalarının ortak sonucudur.

Örneğin dördüncü rakam gerçekte 2 olmasına rağmen model yaklaşık %99 softmax skoru ile 1 tahmin etti. Bu, yüksek model skorunun doğru tahmin garantisi olmadığını gösterir. Ayrıntılı değerler `predictions.json` içindedir.

Bu fotoğraftan görülen hatalara göre yeni ön işleme geliştirilirse aynı fotoğraftaki sonuç bir geliştirme sonucu olarak ayrıca raporlanmalı; bağımsız başarı için farklı bir yeni fotoğraf kullanılmalıdır. Daha net, çizgisiz kâğıtta ve rakamları aralıklı yazılmış yeni bir fotoğraf, ikinci bir değerlendirme için daha uygun olur.
