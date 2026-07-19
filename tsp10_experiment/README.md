# 10 Noktalı Görsel TSP Deneyi

Bu klasör, Elhenawy ve arkadaşlarının görsel TSP/mTSP çalışmasındaki yöntemin
en küçük kapsamda anlaşılması, yeniden uygulanması ve doğrulanması amacıyla
hazırlanmıştır.

Çalışma, depo dahil 10 düğümlü ve tek satıcılı Gezgin Satıcı Problemi (TSP)
ile sınırlandırılmıştır. Özgün makalede GPT-4o kullanılırken bu deneyde
`gemini-2.5-flash` kullanılmıştır. Bu nedenle çalışma birebir replikasyon
değil, yöntemin Gemini tabanlı bir uyarlamasıdır.

## Deneyin amacı

Bu deneyde aşağıdaki sorular araştırılmıştır:

1. Gemini yalnızca nokta görselini inceleyerek geçerli bir TSP rotası
   oluşturabilir mi?
2. Gemini’nin zero-shot çözümü OR-Tools ve kesin optimum ile ne kadar
   örtüşür?
3. Critic iterasyonları rota kalitesini düzenli olarak iyileştirir mi?
4. Birden fazla critic adayı arasından seçim yapan görsel scorer, düşük
   kaliteli rotaları ayırt edebilir mi?
5. Modelin ürettiği rotalar TSP kısıtlarına uyuyor mu?

## Deney kapsamı

- Depo dahil 10 düğüm
- Tek satıcı
- Klasik ve simetrik TSP
- Sabit rastgelelik değeri: `seed=42`
- 5 × 5 düzlemde tekdüze nokta üretimi
- Öklid mesafesi
- OR-Tools referans çözümü
- 9! olasılığı tarayan kesin brute-force çözümü
- Gemini zero-shot
- Gemini Multi-Agent 2
- Gemini Multi-Agent 1
- Rota geçerlilik, mesafe ve optimum gap kontrolü

Gemini’ye sayısal koordinatlar veya mesafe matrisi verilmez. Model yalnızca
problem ve rota görsellerini inceler. Python tarafından hesaplanan mesafe ve
gap değerleri yalnızca deney sonrasında değerlendirme amacıyla kullanılır.

## Dosya yapısı

```text
tsp10_experiment/
├── src/
│   ├── tsp_core.py
│   └── llm_routes.py
├── tests/
├── output/
├── run_baseline.py
├── run_gemini_zero_shot.py
├── run_gemini_multi_agent1.py
├── run_gemini_multi_agent2.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Temel dosyalar

| Dosya | Görevi |
|---|---|
| `src/tsp_core.py` | Nokta üretimi, rota doğrulama, mesafe hesabı, OR-Tools, kesin çözüm ve görselleştirme |
| `src/llm_routes.py` | Gemini istemleri, API çağrıları ve model cevaplarının ayrıştırılması |
| `run_baseline.py` | Problem üretimi, OR-Tools ve kesin optimum hesabı |
| `run_gemini_zero_shot.py` | Tek görsel ve tek Gemini çağrısıyla zero-shot rota üretimi |
| `run_gemini_multi_agent2.py` | Initializer ve tek critic rolüyle iteratif deney |
| `run_gemini_multi_agent1.py` | Yedi critic adayı ve görsel scorer içeren deney |
| `tests/` | Rota, ayrıştırıcı, checkpoint ve deney bileşenlerinin birim testleri |
| `output/` | Görseller, checkpoint dosyaları ve deney sonuçları |

## Kurulum

Python 3.10 veya daha yeni bir sürüm kullanılabilir. Bu çalışma Python 3.13
ile doğrulanmıştır.

### Windows PowerShell

Proje klasörüne geçin:

```powershell
cd "D:\visual-tsp-mllm-experiment\visual-tsp-mllm-experiment\tsp10_experiment"
```

Sanal ortam oluşturun:

```powershell
py -3.13 -m venv .venv
```

PowerShell çalıştırma politikası sanal ortam aktivasyonunu engellerse yalnızca
mevcut terminal için izin verin:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Sanal ortamı etkinleştirin:

```powershell
.\.venv\Scripts\Activate.ps1
```

Paketleri yükleyin:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Terminal satırının başında `(.venv)` görülmesi sanal ortamın etkin olduğunu
gösterir.

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1. Baseline deneyi

Baseline deneyi, Gemini sonuçlarının karşılaştırılacağı problem örneğini ve
referans çözümleri üretir.

Geliştirme için kısa OR-Tools süresi:

```powershell
python run_baseline.py --seed 42 --ortools-time-limit 2
```

Makalede belirtilen 120 saniyelik ayar:

```powershell
python run_baseline.py --seed 42 --ortools-time-limit 120
```

Üretilen temel dosyalar:

| Dosya | İçerik |
|---|---|
| `output/runs/default/baseline/images/points.png` | Gemini’ye gönderilecek 10 düğümlü problem görseli |
| `output/runs/default/baseline/images/or_tools_route.png` | OR-Tools rotası |
| `output/runs/default/baseline/images/exact_route.png` | Kesin optimum rota |
| `output/runs/default/baseline/baseline_results.json` | Koordinatlar, rotalar, mesafeler ve gap değerleri |

### Ayrı çalıştırma klasörü kullanma

Önceki sonuçların üzerine yazılmaması için dört yöntemde de aynı `--run-id`
değeri kullanılabilir:

```powershell
python run_baseline.py --seed 42 --ortools-time-limit 2 --run-id seed42_timing_run_01
python run_gemini_zero_shot.py --run-id seed42_timing_run_01
python run_gemini_multi_agent2.py --iterations 10 --run-id seed42_timing_run_01
python run_gemini_multi_agent1.py --iterations 10 --candidate-count 7 --run-id seed42_timing_run_01
```

Bu komutlar dosyaları şu yapıda saklar:

```text
output/runs/seed42_timing_run_01/
├── baseline/
│   └── images/
├── zero_shot/
│   └── images/
├── multi_agent1/
│   └── images/
│       ├── iteration_01/
│       └── ...
└── multi_agent2/
    └── images/
```

`--run-id` verilmezse sonuçlar düzenli biçimde `output/runs/default/` altında
saklanır. Checkpoint dosyaları geçici dosyalardır ve Git tarafından izlenmez.

Eski düz `output/` dosyalarını ve yöntem klasörlerindeki PNG dosyalarını yeni
yapıya taşımadan önce işlem planı görüntülenebilir:

```powershell
python organize_output.py --dry-run
```

Plan kontrol edildikten sonra taşıma ve JSON yol güncellemeleri uygulanır:

```powershell
python organize_output.py --apply
```

Bu işlem API çağrısı yapmaz ve deneyleri yeniden çalıştırmaz. Kök dizindeki
eski çıktılar varsayılan olarak `output/runs/seed42_initial_run/` altına
taşınır.

Sonuç JSON'larında API çağrı süresi, yerel ayrıştırma, doğrulama, çizim,
token kullanımı ve başarısız çağrı bilgileri ilgili yöntem ve iterasyonun
altında saklanır. `api_call_wall_seconds`, ağ ve SDK süresini de içeren uçtan
uca çağrı süresidir; yalnızca modelin sunucu içi çıkarım süresi değildir.

### Kesin optimum neden hesaplanıyor?

OR-Tools, kullanılan SAVINGS ve GUIDED_LOCAL_SEARCH ayarlarıyla sezgisel bir
çözüm üretir ve teorik optimum garantisi vermez.

Bu deneyde depo sabit tutulduğunda ziyaret edilecek dokuz düğüm kalır:

```text
9! = 362.880
```

Bu büyüklük brute-force yöntemle taranabildiği için gerçek optimum
hesaplanabilir. Böylece OR-Tools ve Gemini sonuçları yalnızca birbirleriyle
değil, kesin optimumla da karşılaştırılabilir.

## 2. Gemini API anahtarı

Gemini API anahtarı Google AI Studio üzerinden alınır. Anahtar kaynak kod
içerisine yazılmaz ve yalnızca ortam değişkeninden okunur.

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="GEMINI_API_ANAHTARINIZ"
```

Anahtarın tanımlandığını, değerini ekrana yazdırmadan kontrol etmek için:

```powershell
if ($env:GEMINI_API_KEY) {
    "API anahtarı tanımlı"
} else {
    "API anahtarı eksik"
}
```

### macOS/Linux

```bash
export GEMINI_API_KEY="GEMINI_API_ANAHTARINIZ"
```

Yeni terminal açıldığında ortam değişkeninin yeniden tanımlanması gerekebilir.

## 3. Gemini zero-shot deneyi

Zero-shot deneyinden önce baseline çalıştırılmış ve
`output/runs/default/baseline/images/points.png`
oluşturulmuş olmalıdır.

```powershell
python run_gemini_zero_shot.py
```

Bu deneyde:

- Model olarak `gemini-2.5-flash` kullanılır.
- Sıcaklık `0.0` olarak ayarlanır.
- Gemini’ye yalnızca problem görseli gönderilir.
- Koordinatlar ve mesafe matrisi gönderilmez.
- Model tek çağrıda bir TSP rotası üretir.
- Ham model cevabı sonuç JSON dosyasında saklanır.
- Rota Python tarafında ayrıştırılır ve doğrulanır.

Sonuç dosyası:

```text
output/runs/default/zero_shot/gemini_zero_shot_results.json
```

## 4. Gemini Multi-Agent 2

Multi-Agent 2 iki temel rol kullanır:

1. **Initializer:** Zero-shot rotasını başlangıç çözümü olarak üretir.
2. **Critic:** Mevcut rota görselini inceleyerek yeni bir rota önerir.

Critic sıcaklığı `0.7` olarak ayarlanmıştır. Daha yüksek sıcaklık, farklı
rotaların üretilmesini kolaylaştırırken optimum rotadan uzaklaşma riskini de
artırır.

### Tek iterasyonluk doğrulama

```powershell
python run_gemini_multi_agent2.py --iterations 1
```

### Üç iterasyonluk ara deney

```powershell
python run_gemini_multi_agent2.py --iterations 3
```

### On iterasyonluk tam deney

```powershell
python run_gemini_multi_agent2.py `
    --iterations 10
```

Deney sürelerini yapay olarak büyütmemek için kod çağrılar arasında bekleme
uygulamaz. Kota hataları süre ve hata türüyle sonuç JSON'una yazılır.

### Checkpoint’ten devam etme

Her başarılı critic iterasyonu aşağıdaki checkpoint dosyasına kaydedilir:

```text
output/runs/default/multi_agent2/gemini_multi_agent2_checkpoint.json
```

Kota veya ağ hatası nedeniyle deney yarıda kalırsa önceki çağrıları
tekrarlamadan devam edilebilir:

```powershell
python run_gemini_multi_agent2.py `
    --iterations 10 `
    --resume
```

Final sonuç dosyası:

```text
output/runs/default/multi_agent2/gemini_multi_agent2_results.json
```

## 5. Gemini Multi-Agent 1

Multi-Agent 1 üç rol kullanır:

1. **Initializer:** Zero-shot rotası başlangıç çözümü olarak alınır.
2. **Critic:** Tek Gemini çağrısında yedi farklı rota adayı üretir.
3. **Scorer:** Yedi adayın rota görsellerini birlikte değerlendirerek en iyi
   adayı seçer.

Bir iterasyonun işlem sırası şöyledir:

1. Mevcut çözümün rota görseli critic’e gönderilir.
2. Critic sıcaklık `0.7` ile yedi rota adayı üretir.
3. Her aday ayrı bir görsel olarak çizilir.
4. Yedi aday görseli scorer’a birlikte gönderilir.
5. Scorer sıcaklık `0.0` ile görselleri puanlar.
6. Scorer’ın seçtiği rota bir sonraki iterasyona aktarılır.
7. Python, model seçiminden sonra rotaları gerçek mesafe ve gap bakımından
   değerlendirir.

Scorer’a aşağıdaki bilgiler gönderilmez:

- Düğüm koordinatları
- Mesafe matrisi
- Python tarafından hesaplanan rota mesafeleri
- Kesin optimum rota
- Optimum gap değerleri

Böylece scorer seçimi yalnızca görsel akıl yürütmeye dayanır.

### API kullanmadan doğrulama

```powershell
python run_gemini_multi_agent1.py `
    --iterations 1 `
    --validate-only
```

Bu komut:

- API anahtarı istemez,
- Gemini çağrısı yapmaz,
- kota kullanmaz,
- gerekli girdileri ve parametreleri kontrol eder,
- tahmini API isteği sayısını gösterir.

### Tek iterasyonluk başlangıç deneyi

```powershell
python run_gemini_multi_agent1.py `
    --iterations 1 `
    --candidate-count 7
```

### On iterasyonluk tam deney

```powershell
python run_gemini_multi_agent1.py `
    --iterations 10 `
    --candidate-count 7
```

Bir tam Multi-Agent 1 iterasyonu iki Gemini isteği kullanır:

1. Bir critic isteği
2. Bir scorer isteği

On tam iterasyon için toplam 20 başarılı Gemini isteği gerekir. Günlük ücretsiz
kota başka çağrılarla paylaşılabileceği için deney birden fazla güne
yayılabilir.

### Checkpoint’ten devam etme

Critic çağrısı tamamlandığında aday rotalar ve aday görselleri hemen
checkpoint’e kaydedilir:

```text
output/runs/default/multi_agent1/gemini_multi_agent1_checkpoint.json
```

Scorer aşaması kota veya ağ hatası nedeniyle tamamlanamazsa critic adayları
kaybolmaz. `--resume` kullanıldığında adaylar yeniden üretilmeden yalnızca
eksik scorer aşamasından devam edilir:

```powershell
python run_gemini_multi_agent1.py `
    --iterations 10 `
    --candidate-count 7 `
    --resume
```

Final sonuç dosyası:

```text
output/runs/default/multi_agent1/gemini_multi_agent1_results.json
```

### Sonuç JSON alanları

Multi-Agent 1 sonuç dosyasında aşağıdaki alanlar birbirinden ayrılır:

- `final_solution`: Son tamamlanan scorer seçimi
- `best_valid_solution`: Initializer ve scorer seçimleri arasındaki en kısa
  geçerli sistem çözümü
- `best_critic_candidate_oracle`: Python mesafesine göre bütün critic
  adayları arasındaki en iyi geçerli aday

`best_critic_candidate_oracle` yalnızca deney sonrasında analiz amacıyla
hesaplanır. Bu bilgi scorer’a gösterilmez.

Her iterasyonda ayrıca şunlar saklanır:

- Ham critic cevabı
- Ayrıştırılan critic adayları
- Adayların geçerlilik bilgileri
- Aday rota görselleri
- Ham scorer cevabı
- Scorer puanları
- Seçilen aday kimliği
- Seçilen rotanın gerçek mesafesi
- Kesin optimum gap
- Seçim regret değeri

## 6. Rota değerlendirmesi

Bir rotanın geçerli TSP rotası sayılması için:

1. Depoda başlaması gerekir.
2. Depoda bitmesi gerekir.
3. 1–9 numaralı düğümlerin tamamını içermesi gerekir.
4. Her düğümü tam bir kez ziyaret etmesi gerekir.
5. Bilinmeyen bir düğüm içermemesi gerekir.

Model rotası geçerli olduktan sonra toplam Öklid mesafesi hesaplanır.

Optimum gap aşağıdaki formülle bulunur:

```text
gap (%) = ((model mesafesi - kesin optimum) / kesin optimum) × 100
```

Gap değerinin `%0` olması, model rotasının kesin optimumla aynı uzunlukta
olduğunu gösterir.

## 7. Tamamlanan deney sonuçları

Bütün deneyler `seed=42` ile üretilen aynı 10 düğümlü problem üzerinde
çalıştırılmıştır.

### Baseline ve zero-shot

| Yöntem | Rota mesafesi | Gap | Geçerli |
|---|---:|---:|---:|
| Kesin brute-force | `13.226401287596719` | `%0` | Evet |
| Google OR-Tools | `13.226401287596719` | `%0` | Evet |
| Gemini zero-shot | `13.226401287596719` | `%0` | Evet |

Kesin optimum ve OR-Tools rotası:

```text
0-8-7-4-2-5-6-3-1-9-0
```

Gemini zero-shot rotası:

```text
0-9-1-3-6-5-2-4-7-8-0
```

Zero-shot rota, kesin optimum turun ters yöndeki eşdeğeridir.

### Multi-Agent 2 sonuçları

| İterasyon | Mesafe | Optimum gap | Geçerli | Optimum |
|---:|---:|---:|---:|---:|
| 1 | `13.226401` | `%0` | Evet | Evet |
| 2 | `13.226401` | `%0` | Evet | Evet |
| 3 | `16.313341` | `%23.3392` | Evet | Hayır |
| 4 | `18.132450` | `%37.0928` | Evet | Hayır |
| 5 | `16.313341` | `%23.3392` | Evet | Hayır |
| 6 | `16.313341` | `%23.3392` | Evet | Hayır |
| 7 | `13.226401` | `%0` | Evet | Evet |
| 8 | `13.226401` | `%0` | Evet | Evet |
| 9 | `13.226401` | `%0` | Evet | Evet |
| 10 | `13.226401` | `%0` | Evet | Evet |

Toplu Multi-Agent 2 metrikleri:

| Metrik | Sonuç |
|---|---:|
| Tamamlanan iterasyon | 10/10 |
| Geçerli rota | 10/10 (`%100`) |
| Optimum rota | 6/10 (`%60`) |
| Ortalama mesafe | `14.643088118983604` |
| Ortalama optimum gap | `%10.7111` |
| En kötü optimum gap | `%37.0928` |
| Son iterasyon | Optimum |

Bütün rotalar geçerli olmasına rağmen rota kalitesi iterasyonlar boyunca
monoton ilerlememiştir. Critic bazı aşamalarda optimum çözümü kötüleştirmiş,
daha sonraki iterasyonlarda yeniden optimum çözüme dönmüştür.

### Multi-Agent 1 sonuçları

| Metrik | Sonuç |
|---|---:|
| Tamamlanan iterasyon | 10/10 |
| Her iterasyondaki critic adayı | 7 |
| Toplam critic adayı | 70 |
| Geçerli critic adayı | 70/70 (`%100`) |
| Optimum critic adayı | 68/70 (`%97.14`) |
| Optimum olmayan critic adayı | 2/70 |
| Scorer tarafından seçilen optimum çözüm | 10/10 (`%100`) |
| Seçim regret değeri | Her iterasyonda `%0` |

Optimum olmayan critic adayları:

| İterasyon | Aday | Rota | Mesafe | Gap |
|---:|---:|---|---:|---:|
| 6 | 2 | `0-4-7-8-9-1-3-6-5-2-0` | `18.132449638797556` | `%37.0928` |
| 10 | 4 | Sonuç JSON’unda kayıtlı | `13.76743989765542` | `%4.0906` |

Scorer her iki iterasyonda da optimum olmayan adayı seçmemiştir. On
iterasyonun tamamında seçilen sistem rotası kesin optimum olmuştur.

Bu sonuç, scorer rolünün critic tarafından üretilen düşük kaliteli adayları
görsel olarak ayırt edip sistem çıktısından uzak tutabildiğini göstermektedir.

## 8. Testler

Bütün testleri çalıştırmak için:

```powershell
python -m pytest -q
```

Güncel sonuç:

```text
25 passed, 3 warnings
```

Üç uyarı OR-Tools’un kullandığı SWIG bileşenlerinden gelen
`DeprecationWarning` mesajlarıdır:

```text
SwigPyPacked has no __module__ attribute
SwigPyObject has no __module__ attribute
swigvarlink has no __module__ attribute
```

Bu uyarılar testlerin başarısını veya deney sonuçlarını etkilemez.

## 9. Hata dayanıklılığı

Deney kodunda aşağıdaki hata dayanıklılığı özellikleri bulunmaktadır:

- Her başarılı iterasyondan sonra checkpoint kaydı
- Critic ve scorer aşamalarının ayrı kaydedilmesi
- `--resume` ile kaldığı yerden devam etme
- Başarılı API cevaplarının ham biçimde saklanması
- Eksik veya hatalı cevaplarda açıklayıcı hata mesajları
- Birden fazla scorer cevap biçimini destekleyen ayrıştırıcı
- Scorer çıktısı kesildiğinde daha yüksek çıktı token sınırı
- Scorer için sınırlandırılmış düşünme bütçesi
- Bütün skorlar mevcutsa en yüksek skoru seçebilen yedek ayrıştırma
- Kota hatasında tamamlanan iterasyonların korunması
- Başsız Matplotlib arka ucu sayesinde Tkinter gerektirmeden görsel üretimi

## 10. Sık karşılaşılan sorunlar

### `429 RESOURCE_EXHAUSTED`

Gemini ücretsiz API kotası aşılmıştır.

Hata mesajında aşağıdaki kota türlerinden biri görülebilir:

- Dakika başına istek kotası
- Günlük istek kotası

Dakikalık sınır için hata mesajındaki bekleme süresinden sonra tekrar
denenebilir. Günlük sınır için kota yenilenene kadar beklenmelidir.

Checkpoint bulunan deneylerde devam komutu kullanılmalıdır:

```powershell
python run_gemini_multi_agent1.py `
    --iterations 10 `
    --candidate-count 7 `
    --resume
```

Yeni API anahtarı oluşturarak ücretsiz kotayı aşmaya çalışmak yerine mevcut
projenin kotasının yenilenmesi beklenmelidir.

### `ScorerParseError`

Scorer cevabı beklenen biçimde ayrıştırılamamıştır. Ham cevap sonuç veya
checkpoint dosyasında saklanır.

`--resume` kullanıldığında critic adayları yeniden oluşturulmadan scorer
çağrısı tekrarlanır.

### `ModuleNotFoundError`

Sanal ortam etkin olmayabilir veya paketler yanlış Python kurulumuna yüklenmiş
olabilir.

Kontrol edin:

```powershell
where.exe python
python --version
python -m pip --version
```

Gerekirse sanal ortamı yeniden etkinleştirin:

```powershell
.\.venv\Scripts\Activate.ps1
```

### `TclError: Can't find a usable init.tcl`

Matplotlib etkileşimli Tk arka ucunu kullanmaya çalışıyor olabilir. Proje
başsız `Agg` arka ucunu kullandığı için güncel kodda Tkinter gerekmez.

### `KeyboardInterrupt`

İşlem `Ctrl+C` ile durdurulmuştur. OR-Tools’un ilk yüklenmesi bazı Windows
kurulumlarında kısa süre bekletebilir.

OR-Tools’u ayrı test etmek için:

```powershell
python -c "import ortools; print('OR-Tools hazır:', ortools.__version__)"
```

## 11. Güvenlik

- Gemini API anahtarı yalnızca `GEMINI_API_KEY` ortam değişkeninden okunur.
- API anahtarı kaynak koda yazılmaz.
- API anahtarı sonuç JSON dosyalarına kaydedilmez.
- `.env` ve `.venv` Git’e dahil edilmez.
- Checkpoint dosyaları ve geçici görseller Git’e dahil edilmez.
- API anahtarı terminal çıktısı veya ekran görüntüsüyle paylaşılmaz.

Sonuç dosyalarında hassas bilgi bulunmadığını kontrol etmek için:

```powershell
Get-ChildItem .\output -Recurse -Filter *.json | Select-String `
    -Pattern "AIza|GEMINI_API_KEY|api[_-]?key|secret" `
    -CaseSensitive:$false
```

Komutun herhangi bir eşleşme döndürmemesi beklenir.

## 12. GitHub’da saklanan sonuçlar

Yeniden üretilebilirlik amacıyla aşağıdaki tamamlanmış sonuç dosyalarının
GitHub’da tutulması önerilir:

```text
output/runs/<run-id>/zero_shot/gemini_zero_shot_results.json
output/runs/<run-id>/multi_agent2/gemini_multi_agent2_results.json
output/runs/<run-id>/multi_agent1/gemini_multi_agent1_results.json
```

Checkpoint dosyaları, geçici görseller ve sanal ortam GitHub’a gönderilmez.

## 13. Sınırlılıklar

Bu deney yalnızca:

- tek bir `seed`,
- tek bir 10 düğümlü problem,
- tek Gemini modeli,
- tek görsel düzen,
- tek sıcaklık yapılandırması

üzerinde gerçekleştirilmiştir.

Bu nedenle sonuçlar, Gemini’nin bütün TSP problemlerinde aynı performansı
göstereceği anlamına gelmez.

Daha güvenilir sonuçlar için çalışma:

- farklı seed değerleri,
- daha fazla problem örneği,
- farklı düğüm sayıları,
- farklı sıcaklık değerleri,
- farklı Gemini modelleri,
- API süresi ve token kullanımı,
- istatistiksel karşılaştırmalar

ile genişletilmelidir.

Tek satıcılı TSP doğrulaması tamamlandıktan sonra makaledeki çok satıcılı mTSP
senaryolarına geçilebilir.

## Kaynak

Elhenawy, M. ve diğerleri (2024). *Visual Reasoning and Multi-Agent Approach
in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP
Combinatorial Challenges*. *Machine Learning and Knowledge Extraction, 6*,
1894–1920.

- Makale: https://doi.org/10.3390/make6030093
- Özgün GitHub deposu:
  https://github.com/ahmed-abdulhuy/Solving-TSP-and-mTSP-Combinatorial-Challenges-using-Visual-Reasoning-and-Multi-Agent-Approach-MLLMs-
