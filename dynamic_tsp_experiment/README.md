# Dinamik Görsel TSP Deneyi

Bu klasör, görsel TSP çözüm yaklaşımını farklı düğüm sayıları ve farklı
TSPLIB `EUC_2D` problemleri üzerinde aynı deney akışıyla çalıştırır. Sistem
Gemini 2.5 Flash kullanır; modele koordinatlar veya mesafe matrisi değil,
yalnızca problem ve rota görselleri gönderilir.

> Özgün makalede GPT-4o kullanılmıştır. Bu çalışma yöntemin Gemini'ye
> uyarlanmış deneysel bir uygulamasıdır; birebir model replikasyonu değildir.

## Terminali hazırlama

Repo daha önce kurulmuşsa yeni bir PowerShell terminalinde:

```powershell
cd "D:\visual-tsp-mllm-experiment\visual-tsp-mllm-experiment\dynamic_tsp_experiment"

Set-ExecutionPolicy `
    -Scope Process `
    -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
```

İlk kurulumda sanal ortam ve bağımlılıklar:

```powershell
cd "D:\visual-tsp-mllm-experiment\visual-tsp-mllm-experiment\dynamic_tsp_experiment"

py -3.13 -m venv .venv

Set-ExecutionPolicy `
    -Scope Process `
    -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## Desteklenen girdiler

İki giriş biçimi vardır:

1. Rastgele problem:

   ```powershell
   python run_baseline.py `
       --num-nodes 10 `
       --seed 42 `
       --run-id random10_run_01
   ```

2. TSPLIB problemi:

   ```powershell
   python run_baseline.py `
       --tsplib-file .\data\eil51.tsp `
       --optimal-tour-file .\data\eil51.opt.tour `
       --run-id eil51_run_02
   ```

TSPLIB tarafında şu anda koordinat tabanlı `EDGE_WEIGHT_TYPE: EUC_2D`
desteklenir. Başka ağırlık türleri sessizce yanlış hesaplanmaz; açık hata ile
reddedilir.

`--optimal-tour-file` verilirse o tur kanıtlanmış referans optimum olarak
kullanılır. Verilmezse OR-Tools çözümü karşılaştırma referansı olur ve
`is_proven_optimal=false` olarak etiketlenir. Böyle bir durumda `%0 gap`,
kanıtlanmış optimum anlamına gelmez; yalnızca OR-Tools referansıyla eşitlik
anlamına gelir.

## Deney akışı

Her koşu için tek bir `run-id` seçilir ve dört adım aynı kimlikle çalıştırılır:

```powershell
python run_baseline.py `
    --tsplib-file .\data\eil51.tsp `
    --optimal-tour-file .\data\eil51.opt.tour `
    --ortools-time-limit 120 `
    --run-id eil51_run_02

python run_gemini_zero_shot.py `
    --run-id eil51_run_02

python run_gemini_multi_agent2.py `
    --iterations 10 `
    --run-id eil51_run_02

python run_gemini_multi_agent1.py `
    --iterations 10 `
    --candidate-count 7 `
    --run-id eil51_run_02

python run_analysis.py `
    --run-id eil51_run_02 `
    --require-complete
```

Gemini adımlarından önce API anahtarı yalnız açık PowerShell oturumunda
tanımlanır:

```powershell
$env:GEMINI_API_KEY="YENI_GEMINI_API_ANAHTARINIZ"
```

Başka bir Gemini anahtarı kullanmak için aynı değişkene yeni değer atamak
yeterlidir. Anahtar kaynak kodda, sonuç JSON'larında veya checkpoint'lerde
tutulmaz.

Anahtarı ekrana yazdırmadan tanımlı olup olmadığını kontrol etmek için:

```powershell
if ($env:GEMINI_API_KEY) {
    "Gemini API anahtarı tanımlı."
} else {
    "Gemini API anahtarı eksik."
}
```

## API kullanmadan doğrulama

Baseline tamamlandıktan sonra üç Gemini yönteminin girdileri kota harcamadan
kontrol edilebilir:

```powershell
python run_gemini_zero_shot.py `
    --run-id eil51_run_02 `
    --validate-only

python run_gemini_multi_agent2.py `
    --iterations 10 `
    --run-id eil51_run_02 `
    --validate-only

python run_gemini_multi_agent1.py `
    --iterations 10 `
    --candidate-count 7 `
    --run-id eil51_run_02 `
    --validate-only
```

## Yöntemler

### Baseline

Problem yüklenir, görseli çizilir ve OR-Tools ile bir rota üretilir. TSPLIB
optimal turu varsa ayrıca doğrulanır. Problem tanımı ve SHA-256 fingerprint'i
`run_manifest.json` dosyasına yazılır. Sonraki yöntemler problemi yeniden
yorumlamak yerine bu manifesti kullanır.

### Zero-shot

Gemini, yalnız düğüm görselinden tek çağrıda bir TSP turu üretir. Sıcaklık
`0.0` değerindedir. Rota Python tarafında ayrıştırılır, geçerliliği kontrol
edilir ve gerçek mesafesi hesaplanır.

### Multi-Agent 2

Zero-shot rota initializer olarak alınır. Her critic iterasyonu önceki rota
görselini inceleyerek sıcaklık `0.7` ile yeni bir rota önerir. Her iterasyon
hemen checkpoint'e kaydedilir.

Kota, ağ veya servis hatası sonrası devam etmek için:

```powershell
python run_gemini_multi_agent2.py `
    --iterations 10 `
    --run-id eil51_run_02 `
    --resume
```

### Multi-Agent 1

Her iterasyonda critic tek Gemini çağrısıyla en fazla yedi aday üretir.
Python adayların yalnız TSP geçerliliğini kontrol eder:

- Birden fazla geçerli aday varsa görsel scorer yalnız bu adayları görür.
- Tek geçerli aday varsa ek scorer API çağrısı olmadan seçilir.
- Geçerli aday yoksa önceki geçerli rota korunur.
- Mesafe ve gap scorer'a gönderilmez; bunlar yalnız değerlendirme içindir.

Critic tamamlandıktan sonra adaylar checkpoint'e yazılır. Scorer aşamasında
hata olursa `--resume`, critic isteğini tekrar etmeden kayıtlı adaylarla devam
eder:

```powershell
python run_gemini_multi_agent1.py `
    --iterations 10 `
    --candidate-count 7 `
    --run-id eil51_run_02 `
    --resume
```

## Zaman ölçümü

Yapay `delay` veya `sleep` kullanılmaz. Süreler gerçek işlem süreleridir.
Ayrıntılı sonuç JSON'larında aşağıdaki aşamalar ayrı tutulur:

- Gemini API çağrı süresi,
- yanıt ayrıştırma,
- rota geçerlilik ve metrik hesabı,
- görsel oluşturma,
- checkpoint yazma,
- iterasyon toplam işlem süresi.

Multi-Agent 1'de critic ve scorer süreleri ayrıca ayrılır. Başarısız API veya
ayrıştırma denemeleri de hata kaydında aşama ve süre bilgisiyle tutulur.

## Tek analiz raporu

Her yöntemin ayrıntılı `*_results.json` dosyası deney kanıtı olarak korunur.
Kısa ve okunabilir karşılaştırma için yalnız bir analiz dosyası üretilir:

```text
output/runs/<run-id>/analysis/experiment_analysis_summary.json
```

Bu dosya:

- dört yöntemin durumunu ve temsilî çözümünü,
- Multi-Agent 1 ve 2'nin her iterasyonunu,
- her iterasyonun geçerlilik, mesafe, gap, süre ve token özetini,
- geçerli çözümlerin mesafeye göre sıralamasını

içerir. Ham model cevaplarını, promptları, koordinatları, bütün rotaları ve
görsel listelerini tekrar etmez. Analiz API çağrısı yapmaz ve LLM tarafından
yorum yazdırmaz; değerleri sonuç JSON'larından doğrudan hesaplar.

Kısmi deney için de rapor alınabilir:

```powershell
python run_analysis.py --run-id eil51_run_02
```

Tüm yöntemlerin tamamlanmasını zorunlu kılmak için:

```powershell
python run_analysis.py `
    --run-id eil51_run_02 `
    --require-complete
```

Komut terminalde önce yöntem özetini, ardından Multi-Agent 1 ve 2'nin bütün
iterasyonlarını satır satır gösterir.

### Analizi daha sonra tekrar görüntüleme

Analiz komutu istenen her zaman yeniden çalıştırılabilir:

```powershell
python run_analysis.py `
    --run-id eil51_run_02
```

Bu komut:

- Gemini API çağrısı yapmaz,
- API anahtarı veya internet bağlantısı gerektirmez,
- mevcut ayrıntılı sonuç JSON'larını yeniden okur,
- terminalde yöntem özetini ve iterasyonları tekrar gösterir,
- aynı `experiment_analysis_summary.json` dosyasını günceller.

Sonuçlar değişmediyse hesaplanan metrikler de değişmez; yalnız
`generated_at_utc` alanı komutun yeniden çalıştırıldığı zamanı gösterir.
Multi-Agent 1 henüz bitmediyse yöntem durumu `partial` olur ve yalnız
tamamlanan iterasyonlar gösterilir. Bütün yöntemlerin tamamlandığını zorunlu
kılmak için `--require-complete` kullanılır.

### Analiz JSON'unu PowerShell'de inceleme

```powershell
$a = Get-Content `
    .\output\runs\eil51_run_02\analysis\experiment_analysis_summary.json `
    -Raw |
ConvertFrom-Json
```

Tamamlanma durumları ve genel karşılaştırma:

```powershell
$a.completion | Format-List

$a.comparison.method_ranking_by_valid_distance |
Format-Table

$a.comparison.best_valid_mllm_solution |
Format-List
```

Multi-Agent 2 iterasyonları:

```powershell
$a.methods.multi_agent_2.iterations |
Select-Object `
    iteration,
    is_valid,
    distance,
    gap_to_reference_percent,
    total_token_count |
Format-Table
```

Multi-Agent 1 iterasyonları:

```powershell
$a.methods.multi_agent_1.iterations |
Select-Object `
    iteration,
    returned_candidate_count,
    valid_candidate_count,
    selection_mode,
    selected_candidate_id,
    selected_distance,
    selected_gap_to_reference_percent,
    selection_regret_percent |
Format-Table
```

## Çıktı yapısı

```text
output/
└── runs/
    └── <run-id>/
        ├── run_manifest.json
        ├── inputs/
        ├── baseline/
        │   ├── baseline_results.json
        │   └── images/
        ├── zero_shot/
        │   ├── zero_shot_results.json
        │   └── images/
        ├── multi_agent1/
        │   ├── multi_agent1_results.json
        │   ├── multi_agent1_checkpoint.json
        │   └── images/
        ├── multi_agent2/
        │   ├── multi_agent2_results.json
        │   ├── multi_agent2_checkpoint.json
        │   └── images/
        └── analysis/
            └── experiment_analysis_summary.json
```

Checkpoint dosyaları `.gitignore` kapsamındadır. Sonuçlar ve görseller,
istenirse deney kanıtı olarak GitHub'a eklenebilir.

`output/runs/eil51_run_01` ilk pilot çalışmanın tarihsel çıktısıdır ve eski
şemayı korur. Yeni analiz aracı yalnız yeni `schema_version: 2.0` koşularını
karşılaştırır. Final deney için yeni bir kimlik, örneğin `eil51_run_02`,
kullanılmalıdır.

## Testler

```powershell
python -m pytest -q
```

Testler dinamik problem yüklemeyi, TSPLIB doğrulamayı, manifest
fingerprint'ini, Gemini ayrıştırıcılarını, checkpoint güvenliğini,
Multi-Agent 1 geçerlilik filtresini ve analiz raporunu kapsar.

## Gap

Gap, bulunan geçerli rotanın referans mesafeden yüzde olarak ne kadar uzun
olduğunu gösterir:

```text
gap (%) = 100 × (bulunan mesafe - referans mesafe) / referans mesafe
```

Geçersiz rotalarda mesafe tek başına yanıltıcı olabileceğinden
`gap_to_reference_percent` değeri `null` tutulur.
