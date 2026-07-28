# Dinamik Görsel TSP Deneyi

Bu klasör, görsel TSP çözüm yaklaşımını farklı düğüm sayıları ve farklı
TSPLIB `EUC_2D` problemleri üzerinde aynı deney akışıyla çalıştırır. Ortak
runner'lar Gemini, OpenRouter ve Groq vision modellerini destekler; modele
koordinatlar veya mesafe matrisi değil, yalnızca problem ve rota görselleri
gönderilir.

> Özgün makalede GPT-4o kullanılmıştır. Bu çalışma yöntemin farklı vision
> modellere uyarlanmış deneysel bir uygulamasıdır; birebir model replikasyonu
> değildir.

## Terminali hazırlama

Repo daha önce kurulmuşsa yeni bir PowerShell terminalinde:

```powershell
cd "D:\Projects\visual-tsp-mllm-experiment\dynamic_tsp_experiment"

Set-ExecutionPolicy `
    -Scope Process `
    -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
```

İlk kurulumda sanal ortam ve bağımlılıklar:

```powershell
cd "D:\Projects\visual-tsp-mllm-experiment\dynamic_tsp_experiment"

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

## Ortak sağlayıcı runner'ları

Yeni deneylerde önerilen arayüz üç sağlayıcı için aynıdır:

```text
run_zero_shot.py
run_multi_agent1.py
run_multi_agent2.py
```

Yalnız `--provider` ve gerektiğinde `--model` değişir. Eski
`run_gemini_*` ve `run_openrouter_*` komutları geçmiş deneyleri yeniden
üretebilmek için korunur; yeni ortak komutlar bunların sonuçlarının üzerine
yazmaz.

### API anahtarları

Gerekli anahtarlardan yalnız kullanılacak sağlayıcıya ait olanı tanımlayın:

```powershell
$env:GEMINI_API_KEY="YENI_GEMINI_API_ANAHTARINIZ"
$env:OPENROUTER_API_KEY="YENI_OPENROUTER_API_ANAHTARINIZ"
$env:GROQ_API_KEY="YENI_GROQ_API_ANAHTARINIZ"
```

Aynı sağlayıcıda başka anahtar kullanmak için ilgili ortam değişkenine yeni
değeri atamak yeterlidir. Anahtarlar çıktı JSON'larına yazılmaz.

### Zero-shot

Gemini:

```powershell
python run_zero_shot.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --run-id random25_run_01
```

OpenRouter:

```powershell
python run_zero_shot.py `
    --provider openrouter `
    --model nemotron-3-nano-omni `
    --run-id random25_run_01
```

Groq:

```powershell
python run_zero_shot.py `
    --provider groq `
    --model qwen/qwen3.6-27b `
    --run-id random25_run_01
```

Groq'a gönderilen görseller, sağlayıcının TPM sınırlarını aşmamak için yalnız
HTTP isteği hazırlanırken bellekte küçültülür. Tek görselli zero-shot ve
critic çağrılarında en uzun kenar `768`, çok görselli scorer çağrılarında
`384` piksel kullanılır. `output/` altındaki özgün problem ve rota görselleri
değiştirilmez. Genel Groq rota/critic çağrılarının azami çıktı sınırı `4096`,
scorer çağrısının sınırı `2048` tokendır.

`qwen/qwen3.6-27b`, varsayılan açık düşünme çıktısının rota satırından önce
token sınırını tüketmesini önlemek için `reasoning_effort=none` ve en fazla
`1024` çıktı tokenıyla çağrılır. Bu modele özgü ayar sonuç JSON'undaki
`model.inference_settings` alanına; kullanılan yükleme boyutları, byte
sayıları, reasoning ayarı ve etkin çıktı sınırı ise ilgili API çağrı kaydına
yazılır. Parser başarısız olsa bile ham model cevabı sonuç veya hata kaydında
korunur.

Her komuta `--validate-only` eklenerek manifest, model, görsel ve prompt API
çağrısı yapılmadan doğrulanabilir.

### Multi-Agent 2

```powershell
python run_multi_agent2.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --run-id random25_run_01
```

OpenRouter veya Groq için yalnız sağlayıcı/model değiştirilir:

```powershell
python run_multi_agent2.py `
    --provider groq `
    --model qwen/qwen3.6-27b `
    --iterations 10 `
    --run-id random25_run_01
```

Yarım kalan aynı sağlayıcı/model deneyi `--resume` ile kendi checkpoint'inden
devam eder.

### Multi-Agent 1

`--candidate-strategy auto` güvenli sağlayıcı varsayılanını seçer:

- Gemini: tek istekte native çoklu aday,
- OpenRouter: bağımsız critic istekleri,
- Groq: bağımsız critic istekleri.

Gemini/OpenRouter örneği:

```powershell
python run_multi_agent1.py `
    --provider openrouter `
    --model nemotron-3-nano-omni `
    --iterations 10 `
    --candidate-count 7 `
    --candidate-strategy auto `
    --run-id random25_run_01
```

Groq Chat Completions `n=1` kullandığı ve bir scorer isteğinde en fazla beş
görsel kabul ettiği için Groq deneyinde aday sayısı en fazla `5` olmalıdır:

```powershell
python run_multi_agent1.py `
    --provider groq `
    --model qwen/qwen3.6-27b `
    --iterations 10 `
    --candidate-count 5 `
    --candidate-strategy auto `
    --run-id random25_run_01
```

Ortak komutlarda zero-shot initializer aynı `--provider` ve `--model` ile
önceden oluşturulmuş olmalıdır. Uyumlu fingerprint ve model adı taşıyan eski
Gemini/OpenRouter zero-shot sonucu varsa yeniden API çağrısı yapmadan
initializer olarak kullanılabilir. Problem veya model eşleşmiyorsa açık hata
verilir. Böylece farklı modellerin initializer, checkpoint, görsel ve
sonuçları birbirine karışmaz.

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
- bulunan bütün sağlayıcı/model/yöntem kombinasyonlarını,
- Multi-Agent 1 ve 2'nin her iterasyonunu,
- her iterasyonun geçerlilik, mesafe, gap, süre ve token özetini,
- Multi-Agent 1 scorer'ının en kısa geçerli adayı seçme oranını,
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

Komut terminalde kutulu tablolar kullanır. Önce bütün sağlayıcı/model/yöntem
sonuçlarını, ardından her model için Multi-Agent 1 ve 2 iterasyonlarını ve
varsa kaydedilmiş hataları ayrı tablolar halinde gösterir.

## OpenRouter vision model taraması

Mevcut baseline görseli, Gemini sonuçlarının üzerine yazmadan dört sabit
OpenRouter vision modeliyle zero-shot olarak karşılaştırılabilir:

- `google/gemma-4-26b-a4b-it:free`
- `google/gemma-4-31b-it:free`
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`
- `nvidia/nemotron-nano-12b-v2-vl:free`

Groq/Qwen bu taramaya dahil değildir; ayrı sağlayıcı deneyi olarak
çalıştırılmalıdır. Rastgele model seçen `openrouter/free` de tekrarlanabilir
bir karşılaştırma sağlamadığı için kullanılmaz.

Önce OpenRouter anahtarı yalnız açık PowerShell oturumunda tanımlanır:

```powershell
$env:OPENROUTER_API_KEY="YENI_OPENROUTER_API_ANAHTARINIZ"
```

Anahtarı ekrana yazdırmadan kontrol etmek için:

```powershell
if ($env:OPENROUTER_API_KEY) {
    "OpenRouter API anahtarı tanımlı."
} else {
    "OpenRouter API anahtarı eksik."
}
```

Dört modelin manifest, prompt ve görsel girdisini API kullanmadan doğrulama:

```powershell
python run_openrouter_zero_shot.py `
    --run-id random25_run_01 `
    --all-models `
    --validate-only
```

Önce tek model çalıştırılması önerilir:

```powershell
python run_openrouter_zero_shot.py `
    --run-id random25_run_01 `
    --model gemma-4-26b-a4b-it
```

Başarılı kontrolden sonra dört model sırayla çalıştırılabilir:

```powershell
python run_openrouter_zero_shot.py `
    --run-id random25_run_01 `
    --all-models
```

Mevcut model sonuçları varsayılan olarak atlanır. Belirli bir modeli bilinçli
olarak yeniden çağırmak için `--overwrite` kullanılır:

```powershell
python run_openrouter_zero_shot.py `
    --run-id random25_run_01 `
    --model nemotron-3-nano-omni `
    --overwrite
```

API çağrısı yapmadan mevcut sonuçlardan karşılaştırma JSON'unu yeniden
oluşturmak ve tabloyu terminalde göstermek için:

```powershell
python run_openrouter_zero_shot.py `
    --run-id random25_run_01 `
    --summary-only
```

Her modelin ayrıntılı sonucu şu yapıda saklanır:

```text
output/runs/<run-id>/model_comparisons/openrouter/<model-alias>/
├── zero_shot_results.json
└── images/
    └── route.png
```

Kompakt model karşılaştırması:

```text
output/runs/<run-id>/model_comparisons/openrouter/
└── openrouter_model_comparison.json
```

Bütün modeller aynı baseline görselini, aynı promptu, `temperature=0.0` ve
`reasoning_effort=none` ayarlarını kullanır. Koordinatlar, mesafe matrisi ve
referans rota modellere gönderilmez. Format veya rota hatası da başarısız
deney sonucu olarak saklanır; geçersiz rota kısa görünse bile sıralamaya
alınmaz. Karşılaştırma ayrıca yanıtın istenen çıktı biçimine uyup uymadığını
ve rotanın yalnızca artan düğüm kimliklerinden oluşup oluşmadığını raporlar.
En iyi geçerli mesafe birden fazla modelde aynıysa tek bir kazanan ilan etmek
yerine eşitlik açıkça gösterilir.

### OpenRouter Multi-Agent 2

Her model kendi OpenRouter zero-shot rotasını initializer olarak kullanır.
Aynı model, `temperature=0.7` ile critic rolünde rotayı iteratif olarak
yeniler. Gemini Multi-Agent 2 ile aynı prompt, doğrulama, timing, checkpoint
ve `--resume` politikası uygulanır.

Önce API kullanmadan doğrulama:

```powershell
python run_openrouter_multi_agent2.py `
    --run-id random25_run_01 `
    --model nemotron-3-nano-omni `
    --iterations 1 `
    --validate-only
```

Ardından tek gerçek iterasyon:

```powershell
python run_openrouter_multi_agent2.py `
    --run-id random25_run_01 `
    --model nemotron-3-nano-omni `
    --iterations 1
```

Kontrolden sonra aynı model 10 iterasyona tamamlanabilir:

```powershell
python run_openrouter_multi_agent2.py `
    --run-id random25_run_01 `
    --model nemotron-3-nano-omni `
    --iterations 10 `
    --resume
```

Sonuçlar modelin kendi klasöründeki `multi_agent2/` altında saklanır. Geçersiz
bir zero-shot initializer engellenmez; critic'in kısıt hatasını düzeltebilme
başarısı da deneyin bir parçası olarak ölçülür.

### OpenRouter Multi-Agent 1

Her model kendi zero-shot rotasını initializer olarak kullanır. Aynı model
`temperature=0.7` ile yedi bağımsız critic çağrısında yedi rota adayı üretir.
Bu strateji, bazı OpenRouter sağlayıcılarının `n=7` değerini yok sayarak tek
çıktı döndürmesine karşı gerçek aday sayısını garanti eder. Python geçersiz
adayları eler; aynı model kalan rota görsellerini
`temperature=0.0` ile scorer rolünde değerlendirir. Koordinat, mesafe, gap ve
hangi adayın sayısal olarak kısa olduğu scorer'a gönderilmez.

Önce API kullanmadan doğrulama:

```powershell
python run_openrouter_multi_agent1.py `
    --run-id random25_run_01 `
    --model nemotron-3-nano-omni `
    --iterations 1 `
    --candidate-count 7 `
    --candidate-strategy independent_calls `
    --validate-only
```

Ardından tek gerçek iterasyon:

```powershell
python run_openrouter_multi_agent1.py `
    --run-id random25_run_01 `
    --model nemotron-3-nano-omni `
    --iterations 1 `
    --candidate-count 7 `
    --candidate-strategy independent_calls
```

Kontrolden sonra:

```powershell
python run_openrouter_multi_agent1.py `
    --run-id random25_run_01 `
    --model nemotron-3-nano-omni `
    --iterations 10 `
    --candidate-count 7 `
    --candidate-strategy independent_calls `
    --resume
```

Varsayılan `independent_calls` stratejisinde bir iterasyon en fazla yedi
critic ve bir scorer olmak üzere sekiz HTTP isteği kullanır. Deneysel
karşılaştırma amacıyla `native_multiple_choices` seçeneği de vardır; bu
seçenek `n=7` gönderir fakat sağlayıcı daha az çıktı döndürebilir. Scorer
aşamasında kota veya sağlayıcı hatası oluşursa critic adayları checkpoint'te
korunur ve `--resume` yalnız kalan scorer aşamasından devam eder. Sonuçlar
model klasöründeki `multi_agent1/` altında saklanır.

### Analizi daha sonra tekrar görüntüleme

Analiz komutu istenen her zaman yeniden çalıştırılabilir:

```powershell
python run_analysis.py `
    --run-id eil51_run_02
```

Bu komut:

- hiçbir model API'sine çağrı yapmaz,
- API anahtarı veya internet bağlantısı gerektirmez,
- mevcut ayrıntılı sonuç JSON'larını yeniden okur,
- terminalde sağlayıcı/model özetini ve iterasyon tablolarını tekrar gösterir,
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
        ├── providers/
        │   ├── gemini/
        │   │   └── <model>/
        │   ├── openrouter/
        │   │   └── <model>/
        │   └── groq/
        │       └── <model>/
        │           ├── zero_shot/
        │           ├── multi_agent1/
        │           └── multi_agent2/
        └── analysis/
            └── experiment_analysis_summary.json
```

Kök seviyedeki `zero_shot/`, `multi_agent1/` ve `multi_agent2/` klasörleri
eski Gemini runner'larının tarihsel düzenidir. Yeni ortak runner'lar
`providers/<provider>/<model>/` altında çalışır. Analiz aracı hem bu tarihsel
düzeni, hem eski `model_comparisons/openrouter/` düzenini, hem de yeni ortak
provider düzenini aynı raporda okuyabilir.

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
fingerprint'ini, sağlayıcı adaptörlerini, ayrıştırıcıları, checkpoint
güvenliğini, Multi-Agent 1 geçerlilik filtresini, terminal tablosunu ve
birleşik analiz raporunu kapsar.

## Gap

Gap, bulunan geçerli rotanın referans mesafeden yüzde olarak ne kadar uzun
olduğunu gösterir:

```text
gap (%) = 100 × (bulunan mesafe - referans mesafe) / referans mesafe
```

Geçersiz rotalarda mesafe tek başına yanıltıcı olabileceğinden
`gap_to_reference_percent` değeri `null` tutulur.
