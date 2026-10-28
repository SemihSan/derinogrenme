# İkinci fotoğraf ve iki çekimin karşılaştırması

Kullanıcı ikinci fotoğrafı 28 Ekim 2026 tarihinde sağladı. Orijinali `handwritten/original/student_number_v2.jpeg` altında değiştirilmeden korunur. Görsel 1200×1600 pikseldir; daha net yazı, çizgisiz fakat dokulu bir yüzey ve fotoğrafta yazı dışında klavye içerir. Model yalnızca yazıdan kırpılan rakamları gördü.

Gerçek numara, kullanıcının önceki mesajında doğruladığı **24120205084** numarasıdır. İkinci fotoğrafın yazısı da bu numarayla uyumludur. `handwritten/v2/crops.json` bölgeleri bu fotoğrafın tahminleri görülmeden görsel incelemeyle belirlendi; `handwritten/v2/crop_preview.png` ayırma sınırlarını gösterir. Kırpımlar `handwritten/v2/digits/` içindedir. Rakamlar yeniden çizilmedi veya yapay olarak üretilmedi.

## Sonuç

| Ölçüm | İlk fotoğraf | İkinci fotoğraf |
|---|---|---|
| Gerçek numara | `24120205084` | `24120205084` |
| Tahmin | `24119141416` | `24129245984` |
| Doğru rakam | 3/11 | **8/11** |
| Rakam accuracy | %27,27 | **%72,73** |
| Numaranın tamamı doğru mu? | Hayır | Hayır |
| Model | wide, epoch 9 | Aynı checkpoint |
| Ön işleme | Mevcut genel akış | Aynı akış |

İkinci çekimde 5 ek rakam doğru tanındı; fark **45,45 yüzde puan**. İlk fotoğrafın verileri ve sonuçları `results/handwritten/` altında korunmuştur. [İlk fotoğraf raporu](../handwritten/handwritten_evaluation.md).

İkinci çekimde yalnızca üç adet **0** yanlış tahmin edildi:

| Rakamın sırası (1'den başlar) | Dosya | Gerçek | Tahmin | Softmax skoru |
|---|---|---:|---:|---:|
| 5 | 04.png | 0 | 9 | %97,93 |
| 7 | 06.png | 0 | 4 | %88,74 |
| 9 | 08.png | 0 | 9 | %99,10 |

Bu örnekler yüksek softmax skorunun doğruluk garantisi olmadığını gösterir. 8 rakamı ise yaklaşık %32,55 skorla doğru sınıflandırıldı. Üç sıfırın da yanlış çıkması bu küçük örnek grubundaki bir gözlemdir; modelin tüm sıfırlarda başarısız olduğu anlamına gelmez.

## Yorum ve sınırlar

MNIST test accuracy %97,95 olmasına rağmen ikinci gerçek fotoğrafta %72,73 elde edildi. Ön işlenmiş girdilerde yüzey dokusu ve bazı komşu çizgi parçaları kalıyor. El yazısının şekli, eğimi, stroke kalınlığı ve merkezleme de MNIST dağılımından farklı olabilir. Bu ölçüm tüm kırpma–ön işleme–sınıflandırma akışını değerlendirir; hatayı tek bir bileşene kesin olarak bağlamaz.

İkinci fotoğraf daha net olsa da yazı biçimi, boyut, yüzey ve kırpma bölgeleri de farklıdır. Bu nedenle 45,45 puanlık farkı yalnızca fotoğraf netliğine bağlayan kontrollü bir deney yapılmış değildir. Tek kişiden iki numara fotoğrafı genel gerçek dünya performansını tahmin etmek için yeterli değildir.

Her iki kişisel değerlendirmede de önceden minimum validation loss ile seçilen aynı wide checkpoint kullanıldı. Model ağırlıkları, hiperparametreler ve `mnist_mlp/preprocessing.py` değiştirilmedi. İkinci fotoğrafın sonucu görüldükten sonra kırpma koordinatları ayarlanmadı veya başka modeller denenmedi. Bu gözlemlere göre yeni bir ön işleme geliştirilirse, aynı görsellerdeki sonuç geliştirme sonucu olarak ayrıca raporlanmalı; bağımsız değerlendirme için yeni veri ayrılmalıdır.

![İkinci fotoğraf: gerçek ve ön işlenmiş rakamlar](predictions.png)
