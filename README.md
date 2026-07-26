# Visual TSP with Gemini MLLM

Bu depo, görsel akıl yürütme ve çok ajanlı MLLM yaklaşımıyla tek satıcılı
Gezgin Satıcı Problemi (TSP) çözümünü inceleyen deneyleri içerir.

Çalışma, Elhenawy ve arkadaşlarının *Visual Reasoning and Multi-Agent
Approach in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP
Combinatorial Challenges* makalesindeki yöntemi temel alır. Özgün çalışmada
GPT-4o kullanılmıştır; bu depodaki deneyler Gemini 2.5 Flash uyarlamasıdır.

## Klasörler

| Klasör | Amaç |
|---|---|
| `tsp10_experiment/` | `seed=42` ile oluşturulan sabit 10 düğümlü ilk deney; kesin brute-force, OR-Tools, Zero-shot, Multi-Agent 1 ve Multi-Agent 2 sonuçlarını korur |
| `dynamic_tsp_experiment/` | Rastgele `N` düğümlü veya TSPLIB `EUC_2D` girdili güncel ve genellenebilir deney sistemi |
| [`upstream_reference/`](upstream_reference/) | Fork alınan özgün depodaki Multi-Agent klasörleri, notebooklar ve açıklama PDF'i |

Yeni deneyler için
[`dynamic_tsp_experiment/README.md`](dynamic_tsp_experiment/README.md)
dosyasındaki akış kullanılmalıdır. İlk 10 düğümlü çalışmanın ayrıntıları
[`tsp10_experiment/README.md`](tsp10_experiment/README.md) dosyasındadır.
Makalenin özgün uygulama dosyaları karşılaştırma ve kaynak geçmişini korumak
amacıyla `upstream_reference/` klasöründe değiştirilmeden saklanır.

## Güncel dinamik sistem

Dinamik sistem:

- `--num-nodes N --seed S` ile rastgele TSP üretir,
- TSPLIB `EUC_2D` dosyalarını ve isteğe bağlı optimal tur dosyasını okur,
- problem manifesti ve SHA-256 fingerprint'i oluşturur,
- OR-Tools, Gemini Zero-shot, Multi-Agent 1 ve Multi-Agent 2'yi aynı problem
  üzerinde çalıştırır,
- her iterasyonun API, ayrıştırma, değerlendirme, çizim, checkpoint ve toplam
  süresini kaydeder,
- yapay bekleme kullanmaz,
- uzun yöntem sonuçlarından tek bir
  `experiment_analysis_summary.json` karşılaştırma raporu üretir.

Gemini'ye sayısal koordinatlar veya mesafe matrisi gönderilmez; model yalnız
görselleri inceler. Python tarafındaki mesafe ve gap hesapları yalnız deney
değerlendirmesinde kullanılır.

## Güvenlik

- Gemini anahtarı yalnız `GEMINI_API_KEY` ortam değişkeninden okunur.
- `.env`, sanal ortam ve checkpoint dosyaları Git dışında tutulur.
- API anahtarı kaynak koda veya sonuç dosyalarına yazılmaz.

## Kaynak

Elhenawy, M. ve diğerleri (2024). *Visual Reasoning and Multi-Agent Approach
in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP
Combinatorial Challenges*. Machine Learning and Knowledge Extraction, 6,
1894–1920. https://doi.org/10.3390/make6030093

Özgün kod:
https://github.com/ahmed-abdulhuy/Solving-TSP-and-mTSP-Combinatorial-Challenges-using-Visual-Reasoning-and-Multi-Agent-Approach-MLLMs-
