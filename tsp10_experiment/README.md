# 10 Noktalı Görsel TSP Tekrar Deneyi

Bu klasör, Elhenawy ve arkadaşlarının görsel TSP/mTSP çalışmasını önce en
küçük kapsamda anlamak ve doğrulamak için hazırlanmıştır.

İlk aşamanın kapsamı:

- depo dahil 10 nokta,
- tek satıcı (TSP),
- aynı noktalar için OR-Tools çözümü,
- 10 nokta küçük olduğu için brute-force kesin optimum,
- rota geçerlilik ve gerçek Öklid mesafesi kontrolleri.

Zero-shot adımı ayrıca çalıştırılır; API anahtarı kaynak koda yazılmaz.

## Kurulum

Python 3.10 veya daha yeni bir sürümle:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell aktivasyonu:

```powershell
.venv\Scripts\Activate.ps1
```

## İlk deney

Geliştirme için kısa OR-Tools süresiyle:

```bash
python run_baseline.py --seed 42 --ortools-time-limit 2
```

Makalede belirtilen 120 saniyelik ayarla:

```bash
python run_baseline.py --seed 42 --ortools-time-limit 120
```

Üretilen dosyalar `output/` klasörüne yazılır:

- `points.png`: Daha sonra MLLM'ye verilecek nokta görseli
- `or_tools_route.png`: OR-Tools rotası
- `exact_route.png`: Kesin optimum rota
- `baseline_results.json`: Koordinatlar, rotalar, mesafeler ve gap

## Testler

```bash
pytest -q
```

## Zero-shot deneyi

Önce baseline çalışmış ve `output/points.png` oluşmuş olmalıdır. OpenAI API
anahtarını güvenli bir ortam değişkeni olarak ayarlayın.

Windows PowerShell, yalnızca açık terminal oturumu için:

```powershell
$env:OPENAI_API_KEY="API_ANAHTARINIZ"
python run_zero_shot.py
```

macOS/Linux:

```bash
export OPENAI_API_KEY="API_ANAHTARINIZ"
python run_zero_shot.py
```

Kod, makaleyle karşılaştırılabilirlik için `gpt-4o`, sıcaklık `0.0` ve Chat
Completions kullanır. Ham model cevabı da sonuç JSON'una kaydedilir.

## Neden kesin optimum da var?

Makaledeki OR-Tools ayarları sezgiseldir ve teorik optimum garantisi vermez.
Bu çalışmada yalnızca 10 nokta bulunduğu için 9! = 362.880 olası ziyaret
sırasını kontrol edebiliriz. Böylece OR-Tools ve ileride MLLM ile bulunan
rotaların gerçek optimumdan ne kadar uzak olduğunu da ölçebiliriz.
