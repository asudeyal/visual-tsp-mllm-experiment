# Visual TSP with Multimodal LLMs

Bu depo, tek satıcılı Gezgin Satıcı Problemi'nin (TSP) yalnız görsel girdiler
kullanan multimodal dil modelleriyle çözümünü inceler. OR-Tools referansı;
zero-shot, Multi-Agent 1 ve Multi-Agent 2 yöntemleriyle karşılaştırılır.

Çalışma, Elhenawy ve arkadaşlarının görsel akıl yürütme ve çok ajanlı MLLM
yaklaşımını temel alır. Özgün çalışmada GPT-4o kullanılmıştır; bu depoda yöntem
Gemini, Groq ve OpenRouter üzerinden erişilen vision modellerine uyarlanmıştır.
Bu nedenle çalışma birebir replikasyon değil, yöntem uyarlaması ve deneysel
doğrulamadır.

## Depo yapısı

| Klasör | Amaç |
|---|---|
| [`dynamic_tsp_experiment/`](dynamic_tsp_experiment/) | Yeni deneyler için güncel sistem; rastgele `N` düğümlü veya TSPLIB `EUC_2D`/`GEO` problemleri destekler |
| [`tsp10_experiment/`](tsp10_experiment/) | `seed=42` ile oluşturulan sabit 10 düğümlü ilk Gemini deneyi ve tarihsel sonuçları |
| [`upstream_reference/`](upstream_reference/) | Özgün depodan alınan notebook, Multi-Agent çıktıları ve açıklama PDF'i |

Yeni çalışmalar için yalnız `dynamic_tsp_experiment/` kullanılmalıdır.
`tsp10_experiment/`, ilk uygulamanın yeniden üretilebilir tarihsel kaydıdır.

## Güncel sistemin özellikleri

- `--num-nodes N --seed S` ile rastgele TSP üretimi
- TSPLIB `EUC_2D` ve `GEO` problemleri ile isteğe bağlı bilinen optimum tur desteği
- OR-Tools baseline ve problem manifesti
- Gemini, Groq ve OpenRouter sağlayıcıları için ortak çalıştırma dosyaları
- Zero-shot, Multi-Agent 1 ve Multi-Agent 2 yöntemleri
- Her API çağrısı ve iterasyon için süre, token, geçerlilik, mesafe ve gap kaydı
- Sistem GBest, iterasyonun en iyi adayı, scorer regret ve erken durdurma takibi
- Kontrollü istek aralığı ile rate-limit backoff sürelerinin ayrı kaydı
- Yerel CPU/RAM ve destekleniyorsa NVIDIA GPU kaynak profili
- `429`, `503` ve `504` hataları için ölçümlü ve sınırlı otomatik retry
- Kota veya ağ hatalarından checkpoint ile devam etme
- Bütün sağlayıcı/model/yöntem sonuçlarını birleştiren tek analiz raporu

Modele düğüm koordinatları, mesafe matrisi, hesaplanmış rota uzunlukları veya
gap değerleri gönderilmez. Model yalnız problem ve rota görsellerini görür.
Sayısal değerlendirme deneyden sonra Python tarafından yapılır.

Görsel scorer, geçerli critic adaylarını yalnız rota görsellerinden karşılaştırır.
Bu seçim özellikle `GEO` problemlerinde gerçek küresel mesafe sıralamasıyla her
zaman örtüşmeyebilir. Sistem bu farkı `selection_regret_percent` ve gözlenen en
iyi critic adayı alanlarıyla ayrıca kaydeder.

## Hızlı başlangıç

Windows PowerShell:

```powershell
cd .\dynamic_tsp_experiment

py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pytest -q
```

25 düğümlü rastgele bir problem oluşturun:

```powershell
python run_baseline.py `
    --num-nodes 25 `
    --seed 42 `
    --ortools-time-limit 30 `
    --run-id random25_run_01
```

Kullanacağınız sağlayıcının API anahtarını yalnız açık terminal oturumunda
tanımlayın:

```powershell
$env:GEMINI_API_KEY="..."
$env:GROQ_API_KEY="..."
$env:OPENROUTER_API_KEY="..."
```

Gemini ile tam deney sırası:

```powershell
python run_zero_shot.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --run-id random25_run_01

python run_multi_agent2.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --run-id random25_run_01

python run_multi_agent1.py `
    --provider gemini `
    --model gemini-2.5-flash `
    --iterations 10 `
    --candidate-count 7 `
    --candidate-strategy auto `
    --run-id random25_run_01

python run_analysis.py `
    --run-id random25_run_01
```

Multi-Agent yöntemleri aynı `run-id`, `provider` ve `model` ile oluşturulmuş
gerçek bir zero-shot sonucunu başlangıç rotası olarak kullanır.

Ayrıntılı komutlar, TSPLIB örnekleri, checkpoint kullanımı ve çıktı yapısı için
[`dynamic_tsp_experiment/README.md`](dynamic_tsp_experiment/README.md)
dosyasına bakın.

## Güvenlik

- Anahtarlar yalnız `GEMINI_API_KEY`, `GROQ_API_KEY` ve
  `OPENROUTER_API_KEY` ortam değişkenlerinden okunur.
- API anahtarları kaynak koda, sonuç JSON'larına veya checkpoint dosyalarına
  yazılmaz.
- `.env`, `.venv`, önbellek ve checkpoint dosyaları Git dışında tutulur.
- Gerçek API anahtarlarını terminal çıktısında, ekran görüntüsünde veya
  commit içinde paylaşmayın.

## Kaynak

Elhenawy, M. ve diğerleri (2024). *Visual Reasoning and Multi-Agent Approach
in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP
Combinatorial Challenges*. *Machine Learning and Knowledge Extraction, 6*,
1894–1920. https://doi.org/10.3390/make6030093

Özgün kod deposu:
https://github.com/ahmed-abdulhuy/Solving-TSP-and-mTSP-Combinatorial-Challenges-using-Visual-Reasoning-and-Multi-Agent-Approach-MLLMs-
