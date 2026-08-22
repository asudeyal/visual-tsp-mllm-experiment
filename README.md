# Visual TSP & CVRP with Multimodal LLMs

Bu depo, kombinatoryal rotalama problemlerinin çok modlu dil modelleriyle (MLLM)
görsel girdiler üzerinden çözülmesini inceleyen deneysel çalışmaları içerir.

Çalışmanın ilk bölümü TSP üzerinedir. Güncel araştırma kolunda buna ek olarak
kapasite kısıtlı araç rotalama problemi (CVRP) için görsel talep kodlamaları,
deterministik doğrulama ve geri bildirimli iteratif iyileştirme deneyleri
geliştirilmektedir.

TSP tarafı, Elhenawy ve arkadaşlarının görsel akıl yürütme ve çok ajanlı MLLM
yaklaşımını temel alır. Özgün çalışmada GPT-4o kullanılmıştır; bu depoda yöntem
farklı vision modellerine uyarlanmış ve ayrıca CVRP yönünde genişletilmiştir.
Bu nedenle çalışma birebir replikasyon değil, yöntem uyarlaması ve deneysel
genişletmedir.

## Depo yapısı

| Klasör | Amaç |
|---|---|
| [`dynamic_tsp_experiment/`](dynamic_tsp_experiment/) | Güncel TSP deney sistemi; rastgele problemler ve TSPLIB örnekleri, zero-shot, Multi-Agent 1 ve Multi-Agent 2 |
| [`tsp10_experiment/`](tsp10_experiment/) | Sabit 10 düğümlü ilk TSP/Gemini deneyinin tarihsel ve yeniden üretilebilir kaydı |
| [`visual_cvrp_experiment/`](visual_cvrp_experiment/) | Aktif CVRP araştırma alanı; kapasite/talep görselleştirmeleri, refinement ve analiz |
| [`upstream_reference/`](upstream_reference/) | Özgün çalışmadan alınan referans notebook, çıktılar ve belgeler |

TSP çalışmaları için `dynamic_tsp_experiment/`, CVRP çalışmaları için
`visual_cvrp_experiment/` kullanılmalıdır.

## CVRP çalışmasının güncel durumu

CVRP tarafında sabit 10 düğümlü `capacity_demo_10` problemi kullanılmaktadır:

- 1 depo + 9 müşteri
- toplam talep: 18
- araç kapasitesi: 6
- sabit araç sayısı: 3
- Öklid mesafesi
- deterministik exact baseline

Mevcut görsel talep kodlamaları:

- `numeric`
- `bar_length`
- `color_intensity`
- `size`

Güncel refinement deneyinde `bar_length`, `color_intensity` ve `size`
karşılaştırılmaktadır. Yeni kodlamalar aynı deney altyapısına eklenebilir.

Refinement akışı:

1. İlk iterasyon sıfırdan/zero-shot çözüm üretir.
2. Çözüm Python tarafında deterministik olarak doğrulanır.
3. Sonraki iterasyona önceki çözüm ve doğrulama geri bildirimi verilir.
4. Önce feasibility düzeltilir, ardından toplam rota mesafesi azaltılmaya çalışılır.
5. Kanıtlanmış exact baseline ile gap `0` elde edilirse o yöntem erken durur.

Exact optimum, optimum gap veya gizli çözüm modele verilmez. Exact baseline deney
sonuçlarının değerlendirilmesi ve erken durdurma kontrolü için kullanılır.

Ayrıntılı kullanım için
[`visual_cvrp_experiment/README.md`](visual_cvrp_experiment/README.md)
dosyasına bakın.

## CVRP hızlı başlangıç

Aktif çalışma kolu:

```powershell
git switch research/visual-cvrp-capacity
git pull origin research/visual-cvrp-capacity

cd visual_cvrp_experiment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Gemini anahtarı yalnız terminal oturumunda tanımlanmalıdır:

```powershell
$env:GEMINI_API_KEY="..."
```

Mevcut refinement sonucunu yeniden API çağrısı yapmadan analiz etmek için:

```powershell
python run_refinement_analysis.py `
    --run-id capacity_demo_10_refinement_01
```

## CVRP deney çalıştırıcıları

| Script | Rol |
|---|---|
| `run_experiment.py` | Tek çağrılık / zero-shot CVRP deneyi |
| `run_analysis.py` | Tek çağrılık sonuçların analizi |
| `run_refinement.py` | Zero-shot başlangıç + geri bildirimli iteratif refinement |
| `run_refinement_analysis.py` | Refinement sonuçlarının toplu analizi |

`run_experiment.py` ve `run_analysis.py` tarihsel zero-shot karşılaştırmalarını
yeniden üretilebilir tutmak için korunmaktadır. Yeni CVRP yöntemleri için esas
akış `run_refinement.py` + `run_refinement_analysis.py`'dır.

## Birlikte çalışma notları

- Mevcut deney çıktılarının üzerine yazmadan önce run ID'yi ve branch'i kontrol edin.
- Yeni bir görsel kodlama eklerken `DemandEncoding`, rendering, testler ve
  refinement desteğini birlikte güncelleyin.
- Mevcut `capacity_demo_10_refinement_01` run'ı genişletilecekse tamamlanmış
  yöntemler yeniden çağrılmamalıdır; `--resume` kullanılmalıdır.
- Yeni bir kodlama mevcut run'a eklenecekse `--resume --extend-encodings`
  kullanılmalıdır.
- Yeni ve bağımsız bir deney tasarımı için yeni bir `run-id` kullanın.
- Commit atmadan önce `python -m pytest -q` çalıştırın.
- API anahtarlarını kaynak koda, sonuç JSON'larına veya commitlere yazmayın.

## TSP sistemi

`dynamic_tsp_experiment/` tarafında:

- rastgele ve TSPLIB TSP problemleri
- OR-Tools baseline
- zero-shot, Multi-Agent 1 ve Multi-Agent 2
- Gemini, Groq ve OpenRouter sağlayıcıları
- süre, token, geçerlilik, mesafe, gap ve GBest takibi
- checkpoint/resume ve hata sonrası kontrollü devam
- birleşik analiz raporları

bulunmaktadır. Ayrıntılar için
[`dynamic_tsp_experiment/README.md`](dynamic_tsp_experiment/README.md)
dosyasına bakın.

## Güvenlik

- API anahtarları yalnız ortam değişkenlerinden okunmalıdır.
- `.env`, `.venv`, önbellek ve yerel checkpoint dosyaları Git dışında tutulmalıdır.
- Gerçek API anahtarlarını terminal çıktısında, ekran görüntüsünde veya commit
  içinde paylaşmayın.

## Kaynak

Elhenawy, M. ve diğerleri (2024). *Visual Reasoning and Multi-Agent Approach
in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP
Combinatorial Challenges*. *Machine Learning and Knowledge Extraction, 6*,
1894–1920. https://doi.org/10.3390/make6030093

Özgün kod deposu:
https://github.com/ahmed-abdulhuy/Solving-TSP-and-mTSP-Combinatorial-Challenges-using-Visual-Reasoning-and-Multi-Agent-Approach-MLLMs-
