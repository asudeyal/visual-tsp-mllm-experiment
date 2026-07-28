# Dinamik Görsel TSP Deneyi

Bu klasör; rastgele üretilen veya TSPLIB'den yüklenen tek satıcılı TSP
problemlerini aynı deney akışıyla Gemini, Groq ve OpenRouter vision modelleri
üzerinde karşılaştırır.

Modele koordinat, mesafe matrisi, hesaplanmış rota uzunluğu veya gap verilmez.
Model yalnız problem/rota görsellerini görür. Geçerlilik, mesafe ve gap
hesapları deney sonrasında Python tarafından yapılır.

## Klasör yapısı

```text
dynamic_tsp_experiment/
├── data/                       # TSPLIB problem ve optimum tur dosyaları
├── output/
│   └── runs/
│       └── <run-id>/
│           ├── run_manifest.json
│           ├── inputs/
│           ├── baseline/
│           ├── providers/
│           │   └── <provider>/<model>/
│           │       ├── zero_shot/
│           │       ├── multi_agent1/
│           │       └── multi_agent2/
│           └── analysis/
├── src/                        # Problem, model, analiz ve provider kodları
├── tests/
├── migrate_output_layout.py
├── run_analysis.py
├── run_baseline.py
├── run_zero_shot.py
├── run_multi_agent1.py
└── run_multi_agent2.py
```

Yeni deneylerde yalnız ortak `run_zero_shot.py`, `run_multi_agent1.py` ve
`run_multi_agent2.py` dosyaları kullanılır. Sağlayıcı `--provider` ile,
model ise gerektiğinde `--model` ile seçilir.

## Terminali hazırlama

Yeni PowerShell terminalinde:

```powershell
cd "D:\Projects\visual-tsp-mllm-experiment\dynamic_tsp_experiment"

Set-ExecutionPolicy `
    -Scope Process `
    -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
```

İlk kurulum:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## 1. Problem ve baseline oluşturma

### Rastgele problem

```powershell
python run_baseline.py `
    --num-nodes 25 `
    --seed 42 `
    --ortools-time-limit 30 `
    --run-id random25_run_01
```

Aynı düğüm sayısı ve seed aynı koordinatları üretir. OR-Tools sezgisel çözümü
referans alınır; bu referansın teorik optimum olduğu garanti edilmez.

### TSPLIB problemi

```powershell
python run_baseline.py `
    --tsplib-file .\data\eil51.tsp `
    --optimal-tour-file .\data\eil51.opt.tour `
    --ortools-time-limit 120 `
    --run-id eil51_run_02
```

Koordinat tabanlı `EDGE_WEIGHT_TYPE: EUC_2D` desteklenir. Optimum tur dosyası
verilirse kanıtlanmış TSPLIB optimumu referans olur; verilmezse OR-Tools
çözümü sezgisel referans olarak kullanılır.

## 2. API anahtarları

Yalnız kullanılacak sağlayıcının anahtarını açık terminal oturumunda tanımlayın:

```powershell
$env:GEMINI_API_KEY="GEMINI_API_ANAHTARINIZ"
$env:GROQ_API_KEY="GROQ_API_ANAHTARINIZ"
$env:OPENROUTER_API_KEY="OPENROUTER_API_ANAHTARINIZ"
```

Anahtar değiştirmek için aynı ortam değişkenine yeni değer atanır. Anahtarlar
kaynak koda, sonuç JSON'larına veya checkpoint dosyalarına yazılmaz.

Tanımlı olup olmadığını anahtarı ekrana basmadan kontrol etmek için:

```powershell
if ($env:GROQ_API_KEY) {
    "Groq API anahtarı tanımlı."
} else {
    "Groq API anahtarı eksik."
}
```

## 3. Zero-shot

Gemini:

```powershell
python run_zero_shot.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --run-id random25_run_01
```

Groq:

```powershell
python run_zero_shot.py `
    --provider groq `
    --model qwen/qwen3.6-27b `
    --run-id random25_run_01
```

OpenRouter:

```powershell
python run_zero_shot.py `
    --provider openrouter `
    --model nemotron-3-nano-omni `
    --run-id random25_run_01
```

API çağrısı yapmadan girişleri kontrol etmek için komuta `--validate-only`
eklenebilir.

## 4. Multi-Agent 2

Bir critic mevcut rota görselini inceler ve her iterasyonda yeni rota önerir.

```powershell
python run_multi_agent2.py `
    --provider groq `
    --model qwen/qwen3.6-27b `
    --iterations 10 `
    --run-id random25_run_01
```

Kota veya ağ hatası sonrasında:

```powershell
python run_multi_agent2.py `
    --provider groq `
    --model qwen/qwen3.6-27b `
    --iterations 10 `
    --run-id random25_run_01 `
    --resume
```

Her iterasyonun geçerliliği, mesafesi, gap'i, API süresi, toplam süresi ve
token kullanımı sonuç JSON'unda tutulur. `best_valid_solution`, geçersiz ama
sayısal olarak kısa rotaları en iyi sonuç olarak kabul etmez.

## 5. Multi-Agent 1

Her iterasyonda critic birden fazla aday üretir. Python geçersiz adayları
scorer'a göndermeden eler. Birden fazla geçerli aday kalırsa görsel scorer
seçim yapar; tek geçerli aday varsa otomatik seçilir; hiç geçerli aday yoksa
önceki geçerli rota korunur.

Gemini:

```powershell
python run_multi_agent1.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --candidate-count 7 `
    --candidate-strategy auto `
    --run-id random25_run_01
```

Groq:

```powershell
python run_multi_agent1.py `
    --provider groq `
    --model qwen/qwen3.6-27b `
    --iterations 10 `
    --candidate-count 2 `
    --candidate-strategy auto `
    --run-id random25_run_01
```

OpenRouter:

```powershell
python run_multi_agent1.py `
    --provider openrouter `
    --model nemotron-3-nano-omni `
    --iterations 1 `
    --candidate-count 7 `
    --candidate-strategy independent_calls `
    --run-id random25_run_01
```

Kota veya ağ hatası sonrasında aynı komut `--resume` ile çalıştırılır.
Groq scorer çağrısında en fazla beş görsel desteklediği için Groq
`candidate-count` değeri `5` değerini aşamaz.

`--candidate-strategy auto`, sağlayıcının desteklediği uygun yöntemi seçer:

- `native_multiple_choices`: tek HTTP çağrısından birden fazla aday,
- `independent_calls`: her aday için ayrı HTTP çağrısı.

## 6. Birleşik analiz

```powershell
python run_analysis.py `
    --run-id random25_run_01
```

Bu komut API çağrısı yapmaz. Kayıtlı JSON dosyalarından:

- bütün provider/model/yöntem sonuçlarını,
- Multi-Agent 1 ve 2'nin bütün iterasyonlarını,
- geçerlilik, mesafe, gap, timing ve token ölçümlerini,
- Multi-Agent 1 scorer regret değerlerini,
- toparlanmış ve çözülmemiş API hatalarını

terminalde gösterir ve şu dosyayı günceller:

```text
output/runs/<run-id>/analysis/experiment_analysis_summary.json
```

Bütün yöntemlerin tamamlanmış olmasını zorunlu tutmak için:

```powershell
python run_analysis.py `
    --run-id random25_run_01 `
    --require-complete
```

## 7. Tarihsel çıktı düzenini taşıma

Eski koşularda Gemini sonuçları doğrudan `zero_shot/`, `multi_agent1/`,
`multi_agent2/`; OpenRouter sonuçları ise `model_comparisons/` altında
bulunabilir. Bunları birleşik `providers/` düzenine taşımadan önce:

```powershell
python migrate_output_layout.py `
    --run-id random25_run_01
```

Bu varsayılan olarak dry-run yapar ve hiçbir dosyayı değiştirmez. Plan doğruysa:

```powershell
python migrate_output_layout.py `
    --run-id random25_run_01 `
    --apply
```

Ardından doğrulama:

```powershell
python run_analysis.py `
    --run-id random25_run_01

python -m pytest -q
```

Migration dosyaları taşırken JSON içindeki göreli görsel/sonuç yollarını da
günceller. İşlemi geri almak gerekirse:

```powershell
python migrate_output_layout.py `
    --run-id random25_run_01 `
    --undo

python run_analysis.py `
    --run-id random25_run_01
```

Her koşu ayrı ayrı taşınır. `eil51_run_01` gibi eski ve uyumsuz şema kullanan
pilot kayıtlar tarihsel arşiv olarak olduğu yerde bırakılabilir.

## Sonuç JSON'larındaki temel alanlar

- `validation.is_valid`: rota bütün düğümleri tam bir kez ziyaret ediyor mu?
- `distance`: kullanılan problem metriğine göre rota uzunluğu.
- `gap_to_reference_percent`: referansa göre yüzde fark.
- `final_solution`: son iterasyonun sistem çıktısı.
- `best_valid_solution`: sistem tarafından seçilmiş en iyi geçerli çıktı.
- `best_critic_candidate_oracle`: yalnız sonradan analiz için, bütün critic
  adayları arasındaki sayısal olarak en iyi geçerli rota.
- `selection_regret_percent`: scorer'ın seçimi ile iterasyondaki en kısa
  geçerli aday arasındaki yüzde kayıp.
- `timing`: hazırlama, API, parsing, doğrulama, çizim ve toplam süreler.
- `run_summary`: toplam çağrı, süre ve token bilgileri.

## Gap

```text
gap (%) = 100 × (bulunan mesafe - referans mesafe) / referans mesafe
```

Geçersiz rotalarda düşük mesafe yanıltıcı olabileceğinden gap `null` tutulur.
OR-Tools sezgisel referans kullanıldığında `%0 gap`, kanıtlanmış optimum
anlamına gelmez.

## Testler

```powershell
python -m pytest -q
```

Testler problem üretimini/yüklemeyi, TSPLIB doğrulamasını, manifest
fingerprint'ini, provider adaptörlerini, parserları, checkpoint güvenliğini,
Multi-Agent filtrelerini, migration işlemini, terminal raporunu ve birleşik
analizi kapsar.
