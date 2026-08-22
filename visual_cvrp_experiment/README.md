# Visual CVRP Capacity Experiment

Bu klasör, kapasite kısıtlı araç rotalama problemlerinin (CVRP) görsel
girdiler üzerinden çok modlu dil modelleriyle çözülmesini araştıran aktif
deney alanıdır.

TSP sistemi `dynamic_tsp_experiment/` altında ayrı tutulur. CVRP tarafında
amaç, müşteri taleplerinin farklı görsel kodlamalarının modelin feasibility
ve rota kalitesi üzerindeki etkisini karşılaştırmaktır.

## Problem

Güncel sabit örnek `capacity_demo_10`:

- 1 depo + 9 müşteri = 10 düğüm
- müşteri talepleri: 1, 2 ve 3
- toplam talep: 18
- araç kapasitesi `Q = 6`
- sabit araç sayısı `K = 3`
- Öklid mesafesi

Exact solver bu örnek için kanıtlanmış optimum çözümü üretir. Exact sonuç
modele verilmez; değerlendirme ve erken durdurma için Python tarafında tutulur.

## Görsel talep kodlamaları

`src/rendering.py` içindeki `DemandEncoding` şu yöntemleri destekler:

- `numeric`: talep değeri doğrudan sayı ile gösterilir
- `bar_length`: talep, kapasiteye göre ölçeklenmiş çubuk uzunluğu ile gösterilir
- `color_intensity`: talep, renk yoğunluğu ile gösterilir
- `size`: talep, müşteri işaretinin boyutu ile gösterilir

Güncel refinement run'ında aktif karşılaştırma:

- `bar_length`
- `color_intensity`
- `size`

Yeni bir dördüncü görsel kodlama eklenecektir.

## Deney akışları

### 1. Tek çağrılık / zero-shot

`run_experiment.py` bir kodlama için:

1. problemi ve exact baseline'ı hazırlar,
2. problem görselini üretir,
3. Gemini'ye tek çağrı yapar,
4. cevabı parse eder,
5. çözümü deterministik doğrular,
6. mesafe ve optimum gap'i kaydeder.

Analiz:

```powershell
python run_analysis.py --run-id <RUN_ID>
```

Bu akış tarihsel zero-shot karşılaştırmalarını yeniden üretilebilir tutmak için
korunmaktadır.

### 2. Geri bildirimli refinement

Güncel ana deney akışı `run_refinement.py` ile çalışır.

Her kodlama için:

1. **İterasyon 1:** sıfırdan / zero-shot çözüm
2. çözümün deterministik doğrulanması
3. **İterasyon 2+**: önceki çözüm + doğrulama geri bildirimi ile yeni çözüm
4. feasibility sağlandıktan sonra mesafeyi düşürme
5. gap `0` bulunursa ilgili yöntemi erken durdurma

Geri bildirim; geçerlilik, rota yükleri, kapasite aşımı, eksik/tekrar müşteri,
bilinmeyen düğüm ve filo sınırı gibi deterministik bilgilerden oluşur. Çözüm
geçerliyse toplam rota mesafesi de refinement geri bildirimine eklenir.

## Güncel run

Aktif refinement run:

```text
capacity_demo_10_refinement_01
```

Bu run içinde mevcut yöntemlerin sonuçları korunur. Yeni bir kodlama eklenirken
tamamlanmış yöntemler yeniden API'ye gönderilmemelidir.

Yeni kodlama eklendikten sonra örnek devam komutu:

```powershell
python run_refinement.py `
    --run-id capacity_demo_10_refinement_01 `
    --encodings bar_length color_intensity size <YENI_KODLAMA> `
    --max-refinement-iterations 7 `
    --resume `
    --extend-encodings
```

`--resume`, kayıtlı iterasyonları yeniden çağırmaz. Gap `0` sonucu bulunan yöntem
otomatik atlanır.

Yeni ve bağımsız bir deney tasarımı için mevcut run'ı genişletmek yerine yeni
bir `--run-id` kullanılmalıdır.

## Refinement analizi

Mevcut sonuçları yeniden API çağrısı yapmadan analiz etmek için:

```powershell
python run_refinement_analysis.py `
    --run-id capacity_demo_10_refinement_01
```

Analiz raporu:

- yöntem sonuç özeti
- iterasyon gelişimi
- `Mesafe`
- `Gap %`
- `GBest` (o ana kadarki en iyi **geçerli** çözüm)
- API süresi
- token sayısı
- exact optimum rotaları
- her yöntemin rota geçmişi
- iterasyon rota görselleri

üretir.

GBest yalnız geçerli çözümlerden güncellenir. Geçersiz bir çözüm daha kısa
mesafeye sahip olsa bile GBest'i değiştirmez.

## Çıktı yapısı

Refinement çıktıları genel olarak:

```text
output/
└── runs/
    └── capacity_demo_10_refinement_01/
        ├── inputs/
        ├── baseline/
        ├── refinement_manifest.json
        ├── providers/
        │   └── gemini/
        │       └── <model>/
        │           └── <encoding>/
        │               ├── initial_prompt.txt
        │               ├── iteration_01/
        │               ├── iteration_02/
        │               └── refinement_results.json
        └── analysis/
            ├── terminal_analysis.txt
            └── images/
```

Her iterasyon kendi prompt ve sonuç JSON'unu korur. Başarısız API denemeleri
resume sırasında `request_failure_XX.json` olarak arşivlenebilir.

## Temel bileşenler

- `src/problem.py`: CVRP problem ve düğüm veri modeli
- `src/instances.py`: `capacity_demo_10`
- `src/exact_solver.py`: deterministik exact CVRP baseline
- `src/rendering.py`: problem/rota görselleri ve demand encoding'leri
- `src/model_contract.py`: model prompt ve cevap sözleşmesi
- `src/gemini_client.py`: Gemini vision istemcisi
- `src/validation.py`: deterministik rota doğrulaması
- `run_experiment.py`: tek çağrılık deney
- `run_analysis.py`: tek çağrılık analiz
- `run_refinement.py`: geri bildirimli refinement
- `run_refinement_analysis.py`: refinement analizi
- `tests/`: bütün bileşenlerin testleri

## Doğrulama kuralları

Doğrulayıcı:

- her rotanın depoda başlayıp bitmesini
- rota içinde ara depo bulunmamasını
- tüm müşterilerin tam bir kez ziyaret edilmesini
- bilinmeyen düğüm bulunmamasını
- araç kapasitesinin aşılmamasını
- filo sınırının aşılmamasını
- toplam rota mesafesini

kontrol eder.

## Kurulum

Windows PowerShell:

```powershell
cd visual_cvrp_experiment

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pytest -q
```

Başka bir mevcut sanal ortam da kullanılabilir; önemli olan
`requirements.txt` bağımlılıklarının kurulu olmasıdır.

Gemini ile gerçek API çağrısı öncesi:

```powershell
$env:GEMINI_API_KEY="..."
```

Anahtar yalnız ortam değişkeninde tutulmalıdır.

## Birlikte geliştirme

Arkadaşla aynı çalışma üzerinde devam ederken önerilen akış:

```powershell
git switch research/visual-cvrp-capacity
git pull origin research/visual-cvrp-capacity
```

Yeni değişiklikten önce:

```powershell
git status
```

Değişiklik tamamlanınca:

```powershell
python -m pytest -q
git status
git diff
```

Yeni görsel kodlama eklenirken en az şu alanlar birlikte kontrol edilmelidir:

1. `DemandEncoding`
2. problem rendering'i
3. model prompt sözleşmesi gerekiyorsa ilgili açıklama
4. refinement tarafından desteklenen encoding listesi
5. testler
6. analiz/etiketleme desteği
7. mevcut run'ı genişletirken `--resume --extend-encodings`

Mevcut deney sonuçları elle değiştirilmemeli veya aynı iterasyonlar gereksiz yere
yeniden üretilmemelidir.

## Güvenlik

- API anahtarı kaynak koda yazılmaz.
- Gerçek anahtarlar JSON, prompt arşivi, ekran görüntüsü veya commit içinde
  paylaşılmaz.
- `.venv`, Python önbellekleri ve pytest cache Git dışında tutulur.
