# Dinamik Görsel TSP Deneyi

Bu klasör, rastgele üretilen veya TSPLIB'den yüklenen tek satıcılı TSP
problemlerini aynı deney akışıyla farklı vision modellerinde karşılaştırır.
Koordinat tabanlı TSPLIB `EUC_2D` ve `GEO` mesafe türleri desteklenir.

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
`EDGE_WEIGHT_TYPE: EUC_2D` ve `EDGE_WEIGHT_TYPE: GEO` desteklenir.

Küçük bir `GEO` doğrulama örneği:

```powershell
python run_baseline.py `
    --tsplib-file .\data\ulysses16.tsp `
    --optimal-tour-file .\data\ulysses16.opt.tour `
    --ortools-time-limit 30 `
    --run-id ulysses16_run_01
```

`GEO` problemlerinde rota mesafesi TSPLIB'in küresel mesafe kuralıyla
hesaplanır. Görseller ise modelin inceleyebilmesi için düzleme yansıtılır;
bu nedenle görsel olarak kısa görünen rota ile gerçek `GEO` mesafe sıralaması
her zaman aynı olmayabilir.

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

### Ortak çalışma zamanı ve istek kontrolü seçenekleri

Baseline ve bütün model runner'ları yerel kaynak kullanımını varsayılan olarak
ölçer. Model runner'ları ayrıca kontrollü istek aralığı ile geçici API hataları
için sınırlı otomatik retry uygular.

| Seçenek | Varsayılan | Anlamı |
|---|---:|---|
| `--profile-resources` | açık | Yerel CPU, RAM ve varsa NVIDIA/NVML GPU kullanımını örnekler |
| `--no-profile-resources` | - | Kaynak profilini kapatır |
| `--resource-sample-interval-seconds` | `0.5` | İki kaynak örneği arasındaki süre |
| `--request-interval-seconds` | `0` | Ardışık API istek başlangıçları arasındaki bilinçli minimum süre |
| `--max-retries` | `2` | `429`, `503` ve `504` sonrasında yapılabilecek ek deneme sayısı |
| `--retry-base-delay-seconds` | `2` | Sağlayıcı süre vermediyse ilk backoff süresi |
| `--retry-maximum-delay-seconds` | `60` | Tek retry için azami backoff süresi |
| `--early-stop-gap-percent` | `1.0` | Uygun Multi-Agent deneyinde Sistem GBest bu gap'e ulaştığında durur |
| `--disable-early-stop` | - | İstenen iterasyon sayısına kadar erken durdurmayı kapatır |

Örnek ortak ayarlar:

```powershell
--resource-sample-interval-seconds 0.5 `
--request-interval-seconds 2 `
--max-retries 2 `
--retry-base-delay-seconds 2 `
--retry-maximum-delay-seconds 60
```

Erken durdurma yalnız Gemini veya Groq kullanılırken ve referans kanıtlanmış
optimumsa uygulanır. Karar, sistemin gerçekten seçip benimsediği
`system_gbest` üzerinden verilir; analiz için kaydedilen fakat scorer tarafından
seçilmeyen `observed_candidate_gbest` erken durdurmayı tetiklemez.

Süre alanları birbirinden ayrılır:

- `active_wall_seconds`: Kontrollü bekleme ve retry backoff çıkarıldıktan sonra
  kalan yerel işlem + API duvar süresi.
- `deliberate_delay_seconds`: `--request-interval-seconds` nedeniyle yapılan
  bilinçli bekleme.
- `rate_limit_backoff_seconds`: Geçici API hatası sonrasındaki retry beklemesi.
- `total_wall_seconds`: Aktif süre ile bütün kontrollü beklemelerin toplamı.

API aktif süresi; internet aktarımı, sağlayıcı kuyruğu ve uzak model çıkarımını
birlikte içerir. İstemci bunları kesin olarak birbirinden ayıramaz. CPU/RAM/GPU
tablosu yalnız deneyi çalıştıran bilgisayarı gösterir; uzak model sunucusunun
kaynak kullanımını göstermez. `Profil=evet`, ölçümün açık olduğunu; `Örnek`,
belirlenen aralıkta kaç kaynak örneği toplandığını ifade eder.

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
kullanımı kaydedilir. İterasyonun ürettiği rota, sistemin o ana kadarki en iyi
seçilmiş rotası (`system_gbest`) ve erken durdurma durumu ayrı tutulur.

## 5. Multi-Agent 1

Her iterasyonda critic adaylar üretir. Python geçersiz adayları scorer'dan
önce eler. Birden fazla geçerli aday kalırsa görsel scorer seçim yapar; tek
geçerli aday otomatik seçilir; hiç geçerli aday yoksa önceki rota korunur.

Scorer yalnız geçerli adayların numaralandırılmış rota görsellerini ve görsel
değerlendirme promptunu alır. Aday rotaları, koordinatlar, mesafe matrisi,
Python'ın hesapladığı mesafeler ve gap değerleri scorer'a gönderilmez. Bu nedenle
scorer puanı gerçek TSP mesafesi değil, modelin görsel tercih puanıdır.

Her iterasyonda şu değerler ayrıca kaydedilir:

- `iteration_best_distance`: O iterasyondaki en kısa geçerli critic adayı.
- `system_gbest_distance`: Scorer seçimleri arasından sistemin benimsediği en
  iyi rota.
- `observed_candidate_gbest_distance`: Scorer seçmese bile şimdiye kadar
  üretilmiş en kısa geçerli critic adayı.
- `selection_regret_percent`: Seçilen adayın iterasyonun en iyi adayına göre
  yüzde kaybı.

Özellikle `GEO` problemlerinde düzleme yansıtılmış rota görünümü ile gerçek
küresel mesafe sıralaması ayrışabilir. Critic optimuma yakın bir aday üretse
bile görsel scorer başka bir adayı seçebilir. Bu davranış sonuçlarda kaydedilen
bir deney bulgusudur; `system_gbest` ile `observed_candidate_gbest` bu nedenle
birbirine karıştırılmamalıdır.

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

Rapor; geçerlilik, mesafe, gap, iterasyon en iyisi, Sistem GBest, token,
scorer regret, aktif/kontrollü/backoff süreleri, yerel CPU/RAM/GPU profili ve
kayıtlı hataları içerir.

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

## 8. Deney yürütücüsüne teslim akışı

Kod geliştirmesi tamamlandıktan ve ilgili özellik dalı `main` ile
birleştirildikten sonra yeni TSPLIB deneyi ayrı bir sonuç dalında yürütülür.
Deney yürütücüsü kaynak kodu değiştirmeden komutları çalıştırır, checkpoint ve
sonuçları kontrol eder, birleşik analizi üretir.

### A. Temiz başlangıç

```powershell
cd "D:\Projects\visual-tsp-mllm-experiment"

git switch main
git pull --ff-only origin main
git status -sb

git switch -c experiment/PROBLEM_ADI-gemini-results

cd .\dynamic_tsp_experiment
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

`PROBLEM_ADI` yerine seçilen TSPLIB problem adı yazılır. Deney dalı, kod dalından
ayrı tutulur.

### B. Problem ve baseline

```powershell
$runId = "PROBLEM_ADI_gemini_run_01"
$instance = ".\data\PROBLEM_ADI.tsp"
$optimalTour = ".\data\PROBLEM_ADI.opt.tour"

python run_baseline.py `
    --tsplib-file $instance `
    --optimal-tour-file $optimalTour `
    --ortools-time-limit 120 `
    --run-id $runId `
    --resource-sample-interval-seconds 0.5
```

Terminalde problem adı, düğüm sayısı, `EDGE_WEIGHT_TYPE`, referans türü ve
optimum mesafe kontrol edilir. Bilinen optimum dosyası kullanıldıysa referansın
`is_proven_optimal=true` olması beklenir.

### C. Gemini deneyleri

Önce kota kullanmadan doğrulama, sonra gerçek zero-shot çalıştırılır:

```powershell
python run_zero_shot.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --run-id $runId `
    --validate-only

python run_zero_shot.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --run-id $runId `
    --resource-sample-interval-seconds 0.5 `
    --request-interval-seconds 2 `
    --max-retries 2 `
    --retry-base-delay-seconds 2 `
    --retry-maximum-delay-seconds 60
```

Ardından iki Multi-Agent yöntemi çalıştırılır:

```powershell
python run_multi_agent2.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --run-id $runId `
    --early-stop-gap-percent 1.0 `
    --resource-sample-interval-seconds 0.5 `
    --request-interval-seconds 2 `
    --max-retries 2 `
    --retry-base-delay-seconds 2 `
    --retry-maximum-delay-seconds 60

python run_multi_agent1.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --candidate-count 7 `
    --candidate-strategy auto `
    --run-id $runId `
    --early-stop-gap-percent 1.0 `
    --resource-sample-interval-seconds 0.5 `
    --request-interval-seconds 2 `
    --max-retries 2 `
    --retry-base-delay-seconds 2 `
    --retry-maximum-delay-seconds 60
```

Geçici hata veya kota kesintisinde aynı komuta `--resume` eklenir. Bilinen
optimuma göre gap `%1` veya altına indiğinde erken durdurma istenmiyorsa
`--disable-early-stop` eklenir.

### D. Analiz ve teslim kontrolü

```powershell
python run_analysis.py `
    --run-id $runId `
    --require-complete

python -m pytest -q
git status --short
git diff --check
```

Analizde en az şu noktalar kontrol edilir:

- Zero-shot, Multi-Agent 1 ve Multi-Agent 2 sonuçlarının geçerliliği.
- Her iterasyonun token, aktif süre, bilinçli bekleme ve backoff değerleri.
- İterasyon en iyisi, Sistem GBest ve gözlenen critic GBest ayrımı.
- Multi-Agent 1 scorer doğru seçim oranı ve selection regret.
- Erken durdurma veya normal bitiş nedeni.
- Yerel CPU/RAM profili ve destekleniyorsa GPU ölçümü.
- Kayıtlı API, parser ve geçersiz rota hataları.

Sonuç dalına yalnız seçilen run'ın yeniden üretilebilir sonuçları ve güncel
analiz JSON'u eklenir. API anahtarı, `.env`, `.venv`, önbellek ve yeni
checkpoint dosyaları commit edilmez. Commit öncesinde staged dosyalar ayrıca
kontrol edilir:

```powershell
cd "D:\Projects\visual-tsp-mllm-experiment"

git add -- `
    "dynamic_tsp_experiment/output/runs/$runId"

git diff --cached --name-only |
Select-String -Pattern `
    "\.env|checkpoint|__pycache__|\.venv|secret|credential"

git diff --cached --check
git diff --cached --stat
```

Kontrol çıktısında istenmeyen dosya yoksa sonuçlar Türkçe ve açıklayıcı tek bir
commit ile gönderilir; ardından dal `main` hedefli pull request olarak açılır.

## 9. Eski çıktı düzenini taşıma

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
- `iteration_best_distance`: İlgili iterasyondaki en kısa geçerli aday.
- `system_gbest_distance`: Sistem tarafından seçilip benimsenen çözümler
  arasındaki en iyi mesafe.
- `observed_candidate_gbest_distance`: Seçilmemiş adaylar dahil gözlenen en iyi
  critic mesafesi.
- `selection_regret_percent`: Scorer seçimi ile en kısa geçerli aday arasındaki
  yüzde kayıp.
- `timing`: Hazırlama, API, parsing, doğrulama, çizim ve toplam süreler.
- `observability.resources`: Yerel kaynak profili ve örnek sayısı.
- `observability.request_control`: API denemeleri, retry ve bekleme türleri.
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
| `429` | Runner sınırlı otomatik retry uygular; günlük kota bitmişse yenilendiğinde `--resume` kullanın |
| `503` veya `504` | Runner sınırlı otomatik retry uygular; devam ederse daha sonra `--resume` kullanın |
| Geçersiz rota | Gap ve en iyi geçerli çözüm hesabına alınmaz |
| `ModuleNotFoundError` | Sanal ortamı etkinleştirip requirements dosyasını kurun |

Kod, yalnız `--request-interval-seconds` sıfırdan büyükse bilinçli istek aralığı
uygular. Geçici API hatalarında retry ve backoff süreleri ayrıca kaydedilir.
Retry sınırı aşılırsa tamamlanan iterasyonlar korunur ve deney daha sonra
`--resume` ile sürdürülebilir.

## Testler

```powershell
python -m pytest -q
```

Testler problem yüklemeyi, `EUC_2D`/`GEO` mesafe kurallarını, manifesti,
provider adaptörlerini, parser'ları, retry/bekleme ölçümlerini, kaynak profilini,
GBest ve erken durdurmayı, checkpoint güvenliğini, Multi-Agent filtrelerini,
migration ve analizi kapsar.
OR-Tools SWIG `DeprecationWarning` mesajları test başarısızlığı değildir.

## Güvenlik

- Anahtarlar yalnız ortam değişkenlerinden okunur.
- `.env`, `.venv`, önbellek ve checkpoint dosyaları Git dışında tutulur.
- Anahtarlar sonuç JSON'larına yazılmaz.
- Gerçek anahtarları kodda, commit içinde veya ekran görüntüsünde paylaşmayın.
