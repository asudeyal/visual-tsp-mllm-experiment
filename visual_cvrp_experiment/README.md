# Visual CVRP Capacity Experiment

Bu klasör, kapasite kısıtlı araç rotalama problemlerinin
(CVRP) görsel girdiler üzerinden çok modlu dil modellerine
çözdürülmesini araştıran deneysel çalışma alanıdır.

Çalışma, mevcut `dynamic_tsp_experiment` kodundan ayrı
tutulmaktadır. İlk aşamada critic-scorer yapısı kullanılmadan
tek bir model çağrısı, deterministik çözüm doğrulaması ve
sonuç raporlaması üzerinden ilerlenmektedir.

## İlk deney düzeni

İlk problem aşağıdaki yapıya sahiptir:

- 1 depo ve 9 müşteri olmak üzere toplam 10 düğüm
- müşteri talepleri: 1, 2 ve 3
- toplam talep: 18
- araç kapasitesi: 6
- araç sayısı alt sınırı: 3
- sabit araç sayısı: 3
- Öklid mesafesi

Başlangıçta kapasite ve talepler nümerik olarak
gösterilecektir. Sonraki aşamalarda müşteri talebini ifade
etmek için daire büyüklüğü, renk, tekrar eden semboller ve
birleşik görsel kodlama yöntemleri karşılaştırılacaktır.

## Mevcut bileşenler

- `src/problem.py`: CVRP düğüm ve problem veri modeli
- `src/validation.py`: deterministik rota ve çözüm doğrulaması
- `tests/`: problem modeli ve doğrulama testleri

Doğrulayıcı şu kuralları denetler:

- her rotanın depoda başlayıp depoda bitmesi
- rota içinde ara depo bulunmaması
- bütün müşterilerin tam bir kez ziyaret edilmesi
- bilinmeyen düğüm bulunmaması
- araç kapasitesinin aşılmaması
- sabit filo sınırının aşılmaması
- toplam rota mesafesinin hesaplanması

## Testler

Testleri çalıştırmak için:

```powershell
cd visual_cvrp_experiment

..\dynamic_tsp_experiment\.venv\Scripts\python.exe `
    -m pytest `
    -q
```

Henüz model API entegrasyonu veya görsel üretim bileşeni
eklenmemiştir.
