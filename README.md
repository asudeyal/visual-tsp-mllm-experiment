# Visual TSP with Gemini MLLM

Bu depo, Elhenawy ve arkadaşlarının *Visual Reasoning and Multi-Agent Approach
in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP
Combinatorial Challenges* çalışmasındaki görsel rota oluşturma yönteminin
10 düğümlü, tek satıcılı TSP kapsamında yeniden uygulanmasını içerir.

> **Önemli:** Özgün makalede GPT-4o kullanılmıştır. Bu projede yöntem Gemini
> 2.5 Flash modeline uyarlanmıştır. Bu nedenle çalışma birebir replikasyon
> değil, yöntem uyarlaması ve deneysel doğrulamadır.

## Çalışmanın kapsamı

- Depo dahil 10 düğüm
- Tek satıcı, klasik TSP
- Noktalar 5 x 5 düzlemde uniform dağılımla üretilir
- Aynı problem üzerinde kesin çözüm, OR-Tools ve Gemini karşılaştırılır
- Gemini'ye koordinatlar değil, yalnızca düğümlerin yer aldığı görsel verilir
- Zero-shot, Multi-Agent 2 ve ilerleyen aşamada Multi-Agent 1 incelenir

## Karşılaştırılan yöntemler

| Yöntem | Açıklama |
|---|---|
| Kesin çözüm | 10 düğüm için 9! olası ziyaret sırasını kontrol eder |
| Google OR-Tools | SAVINGS ve GUIDED_LOCAL_SEARCH ile referans rota üretir |
| Gemini zero-shot | Nokta görselinden tek çağrıda rota üretir |
| Gemini Multi-Agent 2 | Başlatıcı ve eleştirmen ajanlarla rotayı iteratif olarak düzenler |
| Gemini Multi-Agent 1 | Başlatıcı, yedi eleştirmen adayı ve görsel puanlayıcı; kod hazır, API deneyi bekliyor |

## Mevcut durum

| Aşama | Durum |
|---|---|
| Veri üretimi ve görselleştirme | Tamamlandı |
| OR-Tools ve kesin optimum | Tamamlandı |
| Gemini zero-shot | Tamamlandı |
| Gemini Multi-Agent 2 | 8/10 iterasyon tamamlandı; günlük API kotası sonrası checkpoint'ten devam edecek |
| Gemini Multi-Agent 1 | Kod ve çevrimdışı testler tamamlandı; gerçek API deneyi henüz çalıştırılmadı |

### İlk problem örneğinin ön sonuçları

- Kesin optimum mesafe: `13.226401`
- OR-Tools mesafesi: `13.226401`
- Gemini zero-shot mesafesi: `13.226401`
- Zero-shot optimum gap: `%0`
- Multi-Agent 2'nin ilk 8 iterasyonunda geçerli rota oranı: `%100`
- Optimum rota oranı: `%50`
- En kötü gözlenen optimum gap: `%37.0928`

Başlatıcı ajan optimum rotayı doğrudan bulmuştur. Eleştirmen ajan bazı
iterasyonlarda optimum rotayı korurken bazı iterasyonlarda yapısal olarak
geçerli fakat daha uzun rotalar üretmiş, ardından yeniden optimum çözüme
dönebilmiştir. Bu bulgu, iteratif öz-iyileştirmenin her adımda monoton bir
kalite artışı sağlamadığını göstermektedir.

## Proje yapısı

```text
.
├── MTSP_GPT.ipynb
├── MTSP_GPT_with scorer.ipynb
├── Multi_agent1/
├── Multi-agent2/
├── Read me.pdf
└── tsp10_experiment/
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
    └── README.md
```

Ayrıntılı kurulum, komutlar ve deney açıklamaları için
[`tsp10_experiment/README.md`](tsp10_experiment/README.md) dosyasına bakın.

## Hızlı başlangıç - Windows PowerShell

```powershell
cd tsp10_experiment
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_baseline.py --seed 42 --ortools-time-limit 2
python -m pytest -q
```

Gemini API anahtarı kaynak koda yazılmaz:

```powershell
$env:GEMINI_API_KEY="GEMINI_API_ANAHTARINIZ"
python run_gemini_zero_shot.py
```

Kota nedeniyle yarıda kalan Multi-Agent 2 deneyi checkpoint'ten sürdürülür:

```powershell
python run_gemini_multi_agent2.py --iterations 10 --delay-seconds 13 --resume
```

Multi-Agent 1 planı kota kullanmadan doğrulanabilir:

```powershell
python run_gemini_multi_agent1.py --iterations 1 --validate-only
```

Gerçek ilk Multi-Agent 1 iterasyonu, makaledeki gibi tek critic çağrısında
yedi aday ve ardından bir görsel scorer çağrısı kullanır:

```powershell
python run_gemini_multi_agent1.py --iterations 1 --candidate-count 7 --delay-seconds 13
```

## Güvenlik

- API anahtarı yalnızca `GEMINI_API_KEY` ortam değişkeninden okunur.
- `.env` ve sanal ortam `.gitignore` ile dışlanır.
- API anahtarı, `.venv` ve geçici deney çıktıları GitHub'a gönderilmez.

## Kaynak

Elhenawy, M. ve diğerleri (2024). *Visual Reasoning and Multi-Agent Approach
in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP
Combinatorial Challenges*. Machine Learning and Knowledge Extraction, 6,
1894-1920. https://doi.org/10.3390/make6030093

Özgün kod deposu:
https://github.com/ahmed-abdulhuy/Solving-TSP-and-mTSP-Combinatorial-Challenges-using-Visual-Reasoning-and-Multi-Agent-Approach-MLLMs-
