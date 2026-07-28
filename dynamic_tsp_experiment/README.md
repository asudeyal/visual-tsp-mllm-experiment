# Dinamik Görsel TSP Deneyi

Bu klasör, rastgele üretilen veya TSPLIB'den yüklenen tek satıcılı TSP
problemlerini aynı deney akışıyla farklı vision modellerinde karşılaştırır.

| Yöntem | Görevi |
|---|---|
| OR-Tools | Referans rota üretir |
| Zero-shot | Problem görselinden tek çağrıda rota üretir |
| Multi-Agent 1 | Critic adayları üretir, görsel scorer seçim yapar |
| Multi-Agent 2 | Critic mevcut rotayı iteratif olarak düzenler |

Gemini, Groq ve OpenRouter desteklenir. Modele koordinat, mesafe matrisi,
hesaplanmış rota uzunluğu veya gap verilmez. Model yalnız problem ve rota
görsellerini görür; sayısal değerlendirme Python tarafından sonradan yapılır.

## Klasör yapısı

```text
dynamic_tsp_experiment/
├── data/                       # TSPLIB problem ve tur dosyaları
├── output/runs/<run-id>/
│   ├── run_manifest.json
│   ├── inputs/
│   ├── baseline/
│   ├── providers/
│   │   └── <provider>/<model>/
│   │       ├── zero_shot/
│   │       ├── multi_agent1/
│   │       └── multi_agent2/
│   └── analysis/
├── src/
├── tests/
├── migrate_output_layout.py
├── run_analysis.py
├── run_baseline.py
├── run_zero_shot.py
├── run_multi_agent1.py
└── run_multi_agent2.py
```

Yeni deneylerde yalnız ortak `run_zero_shot.py`, `run_multi_agent1.py` ve
`run_multi_agent2.py` dosyalarını kullanın.

## Kurulum

Windows PowerShell:

```powershell
cd "D:\Projects\visual-tsp-mllm-experiment\dynamic_tsp_experiment"

py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Yeni terminal açıldığında:

```powershell
cd "D:\Projects\visual-tsp-mllm-experiment\dynamic_tsp_experiment"
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
cd dynamic_tsp_experiment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Tüm seçenekleri görmek için:

```powershell
python run_baseline.py --help
python run_zero_shot.py --help
python run_multi_agent1.py --help
python run_multi_agent2.py --help
python run_analysis.py --help
```

## Zorunlu çalışma sırası

Her deney aynı `--run-id` altında şu sırayla yürütülür:

1. Baseline
2. Gerçek zero-shot çağrısı
3. Multi-Agent 1 ve/veya Multi-Agent 2
4. Analiz

Multi-Agent yöntemleri başlangıç rotasını aynı `run-id`, `provider` ve
`model` için kaydedilmiş zero-shot sonucundan alır.

> `run_zero_shot.py --validate-only` yalnız girdiyi ve promptu doğrular;
> sonuç JSON'u üretmez. Yeni bir run üzerinde Multi-Agent çalıştırmadan önce
> gerçek zero-shot deneyi tamamlanmalıdır.

## 1. Baseline

### Rastgele problem

```powershell
python run_baseline.py `
    --num-nodes 25 `
    --seed 42 `
    --depot-id 0 `
    --ortools-time-limit 30 `
    --run-id random25_run_01
```

`--depot-id` verilmezse depo `0` olur. Aynı düğüm sayısı ve seed aynı
koordinatları üretir. OR-Tools sonucu referanstır, fakat teorik optimum
olduğu garanti edilmez.

### TSPLIB problemi

```powershell
python run_baseline.py `
    --tsplib-file .\data\eil51.tsp `
    --optimal-tour-file .\data\eil51.opt.tour `
    --ortools-time-limit 120 `
    --run-id eil51_run_01
```

`--instance`, `--tsplib-file`; `--optimal-tour`,
`--optimal-tour-file` seçeneğinin eşidir. Koordinat tabanlı
`EDGE_WEIGHT_TYPE: EUC_2D` desteklenir.

Optimum tur verilirse TSPLIB optimumu referans olur. Verilmezse aynı isimli
`*.opt.tour` aranır; o da yoksa OR-Tools sezgisel referansı kullanılır.

Baseline çıktıları:

```text
output/runs/<run-id>/run_manifest.json
output/runs/<run-id>/inputs/
output/runs/<run-id>/baseline/baseline_results.json
output/runs/<run-id>/baseline/images/
```

Farklı çıktı kökü için baseline ve sonraki bütün komutlara aynı seçenek
eklenir:

```powershell
--output-dir "D:\TSP_outputs"
```

## 2. API anahtarları ve varsayılan modeller

```powershell
$env:GEMINI_API_KEY="GEMINI_API_ANAHTARINIZ"
$env:GROQ_API_KEY="GROQ_API_ANAHTARINIZ"
$env:OPENROUTER_API_KEY="OPENROUTER_API_ANAHTARINIZ"
```

| Sağlayıcı | `--provider` | Varsayılan model | Ortam değişkeni |
|---|---|---|---|
| Gemini | `gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| Groq | `groq` | `qwen/qwen3.6-27b` | `GROQ_API_KEY` |
| OpenRouter | `openrouter` | Yok; `--model` zorunlu | `OPENROUTER_API_KEY` |

Gemini ve Groq için `--model` atlanabilir. Deney kaydının açık olması için
model adını komutta yazmak önerilir.

Anahtarı ekrana yazdırmadan kontrol etmek için:

```powershell
if ($env:GEMINI_API_KEY) {
    "Gemini API anahtarı tanımlı."
} else {
    "Gemini API anahtarı eksik."
}
```

## 3. Zero-shot

Önce API kullanmadan doğrulama:

```powershell
python run_zero_shot.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --run-id random25_run_01 `
    --validate-only
```

Bu komut kota kullanmaz ve sonuç JSON'u üretmez.

Gerçek çağrılar:

```powershell
# Gemini
python run_zero_shot.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --run-id random25_run_01

# Groq
python run_zero_shot.py `
    --provider groq `
    --model qwen/qwen3.6-27b `
    --run-id random25_run_01

# OpenRouter
python run_zero_shot.py `
    --provider openrouter `
    --model nemotron-3-nano-omni `
    --run-id random25_run_01
```

Sonuç:

```text
output/runs/<run-id>/providers/<provider>/<model>/zero_shot/
├── images/
└── zero_shot_results.json
```

## 4. Multi-Agent 2

Critic, mevcut rota görselini inceler ve her iterasyonda yeni rota önerir.

```powershell
# API kullanmadan doğrulama
python run_multi_agent2.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --run-id random25_run_01 `
    --validate-only

# Gerçek deney
python run_multi_agent2.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --run-id random25_run_01

# Kota veya ağ hatasından sonra devam
python run_multi_agent2.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --run-id random25_run_01 `
    --resume
```

Başka sağlayıcı için yalnız `--provider` ve `--model` değiştirilir.

```text
output/runs/<run-id>/providers/<provider>/<model>/multi_agent2/
├── images/
├── multi_agent2_checkpoint.json
└── multi_agent2_results.json
```

Her iterasyonun geçerliliği, mesafesi, gap'i, API ve toplam süresi ile token
kullanımı kaydedilir.

## 5. Multi-Agent 1

Her iterasyonda critic adaylar üretir. Python geçersiz adayları scorer'dan
önce eler. Birden fazla geçerli aday kalırsa görsel scorer seçim yapar; tek
geçerli aday otomatik seçilir; hiç geçerli aday yoksa önceki rota korunur.

```powershell
# API kullanmadan doğrulama
python run_multi_agent1.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --candidate-count 7 `
    --candidate-strategy auto `
    --run-id random25_run_01 `
    --validate-only

# Gerçek deney
python run_multi_agent1.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --candidate-count 7 `
    --candidate-strategy auto `
    --run-id random25_run_01

# Kota veya ağ hatasından sonra devam
python run_multi_agent1.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --candidate-count 7 `
    --candidate-strategy auto `
    --run-id random25_run_01 `
    --resume
```

Sağlayıcı örnekleri:

```powershell
# Groq
python run_multi_agent1.py `
    --provider groq `
    --model qwen/qwen3.6-27b `
    --iterations 10 `
    --candidate-count 2 `
    --candidate-strategy auto `
    --run-id random25_run_01

# OpenRouter
python run_multi_agent1.py `
    --provider openrouter `
    --model nemotron-3-nano-omni `
    --iterations 1 `
    --candidate-count 7 `
    --candidate-strategy independent_calls `
    --run-id random25_run_01
```

| Strateji | Davranış |
|---|---|
| `auto` | Sağlayıcının güvenli varsayılanını seçer |
| `native_multiple_choices` | Tek HTTP çağrısında birden fazla aday ister |
| `independent_calls` | Her aday için ayrı API çağrısı yapar |

Gemini en fazla 7, Groq en fazla 5 scorer görseliyle yapılandırılmıştır.
`independent_calls` istek sayısını, süreyi ve kota kullanımını artırır.

```text
output/runs/<run-id>/providers/<provider>/<model>/multi_agent1/
├── images/
├── multi_agent1_checkpoint.json
└── multi_agent1_results.json
```

`--resume` kullanırken `run-id`, provider, model ve aday ayarlarını mevcut
checkpoint ile uyumlu tutun.

## 6. Birleşik analiz

```powershell
python run_analysis.py `
    --run-id random25_run_01
```

Bu komut API çağrısı yapmaz. Bütün provider/model/yöntem sonuçlarını ve
Multi-Agent iterasyonlarını terminalde karşılaştırır; şu dosyayı günceller:

```text
output/runs/<run-id>/analysis/experiment_analysis_summary.json
```

Eksik yöntem varken de rapor üretilebilir. Bütün keşfedilmiş yöntemlerin
tamamlanmasını zorunlu tutmak için:

```powershell
python run_analysis.py `
    --run-id random25_run_01 `
    --require-complete
```

Rapor; geçerlilik, mesafe, gap, süre, token, scorer regret ve kayıtlı hataları
içerir.

## 7. Baştan sona kısa örnek

```powershell
$runId = "random25_run_02"
$provider = "gemini"
$model = "gemini-2.5-flash"

python run_baseline.py `
    --num-nodes 25 `
    --seed 42 `
    --ortools-time-limit 30 `
    --run-id $runId

python run_zero_shot.py `
    --provider $provider `
    --model $model `
    --run-id $runId

python run_multi_agent2.py `
    --provider $provider `
    --model $model `
    --iterations 10 `
    --run-id $runId

python run_multi_agent1.py `
    --provider $provider `
    --model $model `
    --iterations 10 `
    --candidate-count 7 `
    --candidate-strategy auto `
    --run-id $runId

python run_analysis.py `
    --run-id $runId
```

## 8. Eski çıktı düzenini taşıma

Bu bölüm yalnız eski kayıtlar içindir. Yeni deneyler doğrudan güncel klasör
yapısına yazılır.

```powershell
# Dry-run: hiçbir dosya değiştirmez
python migrate_output_layout.py `
    --run-id random25_run_01

# Planı uygula
python migrate_output_layout.py `
    --run-id random25_run_01 `
    --apply

# Son uygulanan taşımayı geri al
python migrate_output_layout.py `
    --run-id random25_run_01 `
    --undo
```

Taşımadan sonra:

```powershell
python run_analysis.py --run-id random25_run_01
python -m pytest -q
```

## Temel sonuç alanları

- `validation.is_valid`: Rota bütün düğümleri tam bir kez ziyaret ediyor mu?
- `distance`: Problem metriğine göre rota uzunluğu.
- `gap_to_reference_percent`: Referansa göre yüzde fark.
- `final_solution`: Son iterasyonun çıktısı.
- `best_valid_solution`: Sistem tarafından seçilmiş en iyi geçerli çıktı.
- `best_critic_candidate_oracle`: Yalnız analiz için en kısa geçerli critic
  adayı.
- `selection_regret_percent`: Scorer seçimi ile en kısa geçerli aday arasındaki
  yüzde kayıp.
- `timing`: Hazırlama, API, parsing, doğrulama, çizim ve toplam süreler.
- `run_summary`: Toplam çağrı, süre ve token bilgileri.

```text
gap (%) = 100 × (bulunan mesafe - referans mesafe) / referans mesafe
```

Geçersiz rotalarda gap `null` tutulur. OR-Tools sezgisel referansına göre
`%0 gap`, kanıtlanmış optimum anlamına gelmez.

## Sık karşılaşılan durumlar

| Durum | Çözüm |
|---|---|
| Manifest bulunamadı | Önce aynı `run-id` ve `output-dir` ile baseline çalıştırın |
| Zero-shot initializer bulunamadı | Aynı `run-id`, provider ve model ile gerçek zero-shot çalıştırın |
| `429` | Kota yenilendiğinde checkpoint üzerinden `--resume` kullanın |
| `503` veya `504` | Geçici sağlayıcı yoğunluğu; daha sonra `--resume` kullanın |
| Geçersiz rota | Gap ve en iyi geçerli çözüm hesabına alınmaz |
| `ModuleNotFoundError` | Sanal ortamı etkinleştirip requirements dosyasını kurun |

Kod yapay bekleme uygulamaz. Tamamlanan iterasyonlar ve hatalar JSON'a
kaydedilir; yeniden deneme kullanıcı tarafından başlatılır.

## Testler

```powershell
python -m pytest -q
```

Testler problem yüklemeyi, manifesti, provider adaptörlerini, parser'ları,
checkpoint güvenliğini, Multi-Agent filtrelerini, migration ve analizi kapsar.
OR-Tools SWIG `DeprecationWarning` mesajları test başarısızlığı değildir.

## Güvenlik

- Anahtarlar yalnız ortam değişkenlerinden okunur.
- `.env`, `.venv`, önbellek ve checkpoint dosyaları Git dışında tutulur.
- Anahtarlar sonuç JSON'larına yazılmaz.
- Gerçek anahtarları kodda, commit içinde veya ekran görüntüsünde paylaşmayın.
