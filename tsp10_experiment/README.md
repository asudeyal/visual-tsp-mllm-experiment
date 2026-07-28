# 10 Noktalı Görsel TSP Deneyi

Bu klasör, görsel TSP yönteminin ilk ve sabit kapsamlı uygulamasını korur.
Deney; `seed=42` ile üretilen, depo dahil 10 düğümlü tek satıcılı bir problem
üzerinde OR-Tools, kesin brute-force, Gemini zero-shot, Multi-Agent 1 ve
Multi-Agent 2 yöntemlerini karşılaştırır.

Özgün makalede GPT-4o, bu deneyde `gemini-2.5-flash` kullanılmıştır. Çalışma
birebir replikasyon değil, Gemini tabanlı yöntem uyarlamasıdır.

> Bu klasör tarihsel deneyin yeniden üretilebilir kaydıdır. Yeni düğüm
> sayıları, TSPLIB problemleri veya farklı sağlayıcılar için
> [`../dynamic_tsp_experiment/`](../dynamic_tsp_experiment/) kullanılmalıdır.

## Deney koşulları

- Depo dahil 10 düğüm
- Tek satıcı, simetrik TSP
- `seed=42`
- 5 × 5 düzlemde tekdüze nokta üretimi
- Öklid mesafesi
- OR-Tools referansı
- Depo sabitken `9! = 362.880` turu tarayan kesin brute-force
- Gemini zero-shot
- Gemini Multi-Agent 1 ve Multi-Agent 2

Gemini'ye koordinatlar, mesafe matrisi, hesaplanmış rota uzunlukları veya gap
değerleri gönderilmez. Model yalnız problem ve rota görsellerini görür.

## Dosya yapısı

```text
tsp10_experiment/
├── src/
│   ├── experiment_metrics.py
│   ├── llm_routes.py
│   ├── output_paths.py
│   └── tsp_core.py
├── tests/
├── output/
│   └── runs/<run-id>/
├── organize_output.py
├── run_baseline.py
├── run_gemini_zero_shot.py
├── run_gemini_multi_agent1.py
├── run_gemini_multi_agent2.py
└── requirements.txt
```

## Kurulum

### Windows PowerShell

```powershell
cd "D:\Projects\visual-tsp-mllm-experiment\tsp10_experiment"

py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Yeni terminal açıldığında:

```powershell
cd "D:\Projects\visual-tsp-mllm-experiment\tsp10_experiment"
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
cd tsp10_experiment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

## Çalışma sırası

Dört yöntemde de aynı `--run-id` değerini kullanın:

```powershell
$runId = "seed42_run_01"
```

### 1. Baseline

```powershell
python run_baseline.py `
    --seed 42 `
    --ortools-time-limit 2 `
    --run-id $runId
```

Makalede belirtilen uzun OR-Tools süresini denemek için:

```powershell
python run_baseline.py `
    --seed 42 `
    --ortools-time-limit 120 `
    --run-id $runId
```

Baseline şunları üretir:

```text
output/runs/<run-id>/baseline/
├── images/
│   ├── points.png
│   ├── or_tools_route.png
│   └── exact_route.png
├── baseline_results.json
└── baseline_summary.json
```

Kesin çözüm, 10 düğümlü bu küçük problemde gerçek optimumu belirlemek için
kullanılır. OR-Tools sezgisel olduğundan tek başına optimum garantisi vermez.

### 2. Gemini API anahtarı

```powershell
$env:GEMINI_API_KEY="GEMINI_API_ANAHTARINIZ"
```

Anahtarı ekrana yazdırmadan kontrol edin:

```powershell
if ($env:GEMINI_API_KEY) {
    "Gemini API anahtarı tanımlı."
} else {
    "Gemini API anahtarı eksik."
}
```

### 3. Zero-shot

```powershell
python run_gemini_zero_shot.py `
    --run-id $runId
```

Zero-shot:

- `gemini-2.5-flash` ve sıcaklık `0.0` kullanır,
- modele yalnız `points.png` gönderir,
- tek API çağrısında rota ister,
- ham cevabı, rotayı, doğrulamayı, mesafeyi, gap'i, süreyi ve tokenları
  kaydeder.

Sonuç yolu:

```text
output/runs/<run-id>/zero_shot/
├── images/gemini_zero_shot_route.png
├── gemini_zero_shot_results.json
└── gemini_zero_shot_summary.json
```

### 4. Multi-Agent 2

Tek critic iterasyonu:

```powershell
python run_gemini_multi_agent2.py `
    --iterations 1 `
    --run-id $runId
```

On iterasyon:

```powershell
python run_gemini_multi_agent2.py `
    --iterations 10 `
    --run-id $runId
```

Kota veya ağ hatasından sonra:

```powershell
python run_gemini_multi_agent2.py `
    --iterations 10 `
    --run-id $runId `
    --resume
```

Multi-Agent 2, zero-shot rotasını initializer olarak alır. Critic sıcaklık
`0.7` ile mevcut rota görselinden yeni rota üretir. Yapay bekleme kullanılmaz.
Her başarılı iterasyon checkpoint'e yazılır.

```text
output/runs/<run-id>/multi_agent2/
├── images/
├── gemini_multi_agent2_checkpoint.json
├── gemini_multi_agent2_results.json
└── gemini_multi_agent2_summary.json
```

### 5. Multi-Agent 1

Önce API çağrısı yapmadan girdileri doğrulayın:

```powershell
python run_gemini_multi_agent1.py `
    --iterations 1 `
    --candidate-count 7 `
    --run-id $runId `
    --validate-only
```

Tek tam iterasyon:

```powershell
python run_gemini_multi_agent1.py `
    --iterations 1 `
    --candidate-count 7 `
    --run-id $runId
```

On iterasyon:

```powershell
python run_gemini_multi_agent1.py `
    --iterations 10 `
    --candidate-count 7 `
    --run-id $runId
```

Kota veya ağ hatasından sonra:

```powershell
python run_gemini_multi_agent1.py `
    --iterations 10 `
    --candidate-count 7 `
    --run-id $runId `
    --resume
```

Her iterasyonda:

1. Critic tek çağrıda yedi rota adayı üretir.
2. Adaylar ayrı görsellere çizilir.
3. Görsel scorer adayları aynı çağrıda değerlendirir.
4. Seçilen rota sonraki iterasyona aktarılır.
5. Python seçimden sonra geçerlilik, mesafe, gap ve regret hesaplar.

Scorer'a koordinatlar, mesafe matrisi, Python mesafeleri, optimum rota veya
gap değerleri verilmez. Seçim yalnız görsellere dayanır.

Checkpoint scorer aşamasından önce de yazılır. Scorer çağrısı yarıda kalırsa
`--resume`, critic adaylarını yeniden üretmeden kayıtlı adaylarla devam eder.

```text
output/runs/<run-id>/multi_agent1/
├── images/iteration_*/
├── gemini_multi_agent1_checkpoint.json
├── gemini_multi_agent1_results.json
└── gemini_multi_agent1_summary.json
```

## Kısa özetleri API kullanmadan yeniden üretme

Uzun sonuç JSON'larından kısa özet çıkarmak için:

```powershell
python run_baseline.py `
    --run-id $runId `
    --summary-only

python run_gemini_zero_shot.py `
    --run-id $runId `
    --summary-only

python run_gemini_multi_agent2.py `
    --run-id $runId `
    --summary-only

python run_gemini_multi_agent1.py `
    --run-id $runId `
    --summary-only
```

Bu komutlar problemi veya API deneylerini yeniden çalıştırmaz.

## Sonuç alanları

- `validation.is_valid`: Rota TSP kısıtlarına uyuyor mu?
- `distance`: Toplam Öklid mesafesi.
- `gap_to_exact_percent`: Kesin optimuma göre yüzde fark.
- `final_solution`: Son iterasyonun seçimi.
- `best_valid_solution`: Initializer ve sistem seçimleri içindeki en kısa
  geçerli rota.
- `best_critic_candidate_oracle`: Yalnız sonradan analiz için bütün critic
  adayları arasındaki en kısa geçerli rota.
- `selection_regret_percent`: Scorer seçimi ile iterasyondaki en kısa geçerli
  aday arasındaki yüzde kayıp.
- `timing`: Hazırlama, API, parsing, doğrulama, çizim ve toplam süreler.

Gap:

```text
gap (%) = 100 × (bulunan mesafe - kesin optimum) / kesin optimum
```

## Tamamlanan tarihsel deney

Bütün sonuçlar aynı `seed=42` problemine aittir.

| Yöntem | Sonuç |
|---|---|
| Kesin brute-force | Mesafe `13.226401287596719`, optimum |
| OR-Tools | Mesafe `13.226401287596719`, optimum |
| Gemini zero-shot | Geçerli, mesafe `13.226401287596719`, gap `%0` |
| Gemini Multi-Agent 2 | 10/10 geçerli, 6/10 optimum, ortalama gap `%10.7111` |
| Gemini Multi-Agent 1 | 70/70 critic adayı geçerli, 68/70 optimum, 10/10 scorer seçimi optimum |

Zero-shot rota, kesin optimum turun ters yöndeki eşdeğeridir. Multi-Agent 2
bütün iterasyonlarda geçerli rota üretmiş, ancak kalite monoton artmamıştır:
critic optimum rotayı bazı iterasyonlarda kötüleştirip daha sonra yeniden
optimuma dönmüştür.

Multi-Agent 1'de iki critic adayı optimum değildir. Görsel scorer bu adayları
seçmemiş ve 10 iterasyonun tamamında optimum çözümü korumuştur. Bu sonuçlar
yalnız tek problem örneği için geçerlidir ve genel performans kanıtı değildir.

## Eski çıktı düzenini taşıma

Önce yalnız planı görüntüleyin:

```powershell
python organize_output.py --dry-run
```

Plan doğruysa:

```powershell
python organize_output.py --apply
```

Bu işlem API çağrısı yapmaz. Eski düz dosyaları varsayılan olarak
`output/runs/seed42_initial_run/` altına taşır ve JSON içindeki yolları
günceller.

## Testler

```powershell
python -m pytest -q
```

OR-Tools SWIG bileşenlerinden gelen `DeprecationWarning` mesajları test
başarısızlığı değildir. README içinde sabit bir “passed” sayısı tutulmaz;
güncel test sayısı komut çıktısından görülmelidir.

## Sık karşılaşılan durumlar

| Durum | Çözüm |
|---|---|
| `GEMINI_API_KEY` eksik | Anahtarı açık terminal oturumunda ortam değişkeni olarak tanımlayın |
| `429 RESOURCE_EXHAUSTED` | Kota yenilendiğinde aynı komutu `--resume` ile çalıştırın |
| Scorer cevabı ayrıştırılamadı | Ham cevap checkpoint'te korunur; `--resume` ile yalnız scorer yeniden denenir |
| `ModuleNotFoundError` | Sanal ortamı etkinleştirip requirements dosyasını kurun |
| `TclError` | Güncel kod başsız Matplotlib `Agg` arka ucunu kullanır; Tkinter gerekmez |

## Güvenlik

- Anahtar yalnız `GEMINI_API_KEY` ortam değişkeninden okunur.
- `.env`, `.venv`, önbellek ve checkpoint dosyaları Git dışında tutulur.
- API anahtarı kaynak koda veya sonuç JSON'larına yazılmaz.
- Gerçek anahtarı commit, terminal çıktısı veya ekran görüntüsünde paylaşmayın.

## Sınırlılıklar

Bu tarihsel deney:

- tek seed,
- tek 10 düğümlü problem,
- tek Gemini modeli,
- tek görsel düzen,
- sabit sıcaklık ayarları

ile sınırlıdır. Genellenebilir karşılaştırmalar için güncel dinamik sistemde
farklı düğüm sayıları, seed'ler, TSPLIB problemleri, sağlayıcılar ve modeller
kullanılmalıdır.

## Kaynak

Elhenawy, M. ve diğerleri (2024). *Visual Reasoning and Multi-Agent Approach
in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP
Combinatorial Challenges*. *Machine Learning and Knowledge Extraction, 6*,
1894–1920. https://doi.org/10.3390/make6030093

Özgün kod deposu:
https://github.com/ahmed-abdulhuy/Solving-TSP-and-mTSP-Combinatorial-Challenges-using-Visual-Reasoning-and-Multi-Agent-Approach-MLLMs-
