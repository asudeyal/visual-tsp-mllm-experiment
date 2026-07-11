# 10 Noktalı Görsel TSP Tekrar Deneyi

Bu klasör, Elhenawy ve arkadaşlarının görsel TSP/mTSP çalışmasını önce en
küçük kapsamda anlamak ve doğrulamak için hazırlanmıştır.

İlk aşamanın kapsamı:

- depo dahil 10 nokta,
- tek satıcı (TSP),
- aynı noktalar için OR-Tools çözümü,
- 10 nokta küçük olduğu için brute-force kesin optimum,
- rota geçerlilik ve gerçek Öklid mesafesi kontrolleri.

Zero-shot adımı Gemini API ile ayrıca çalıştırılır; API anahtarı kaynak koda
yazılmaz. Bu çalışma, makaledeki GPT-4o yönteminin Gemini uyarlamasıdır.

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

## Gemini zero-shot deneyi

Önce baseline çalışmış ve `output/points.png` oluşmuş olmalıdır. Gemini API
anahtarını Google AI Studio'dan alın ve güvenli bir ortam değişkeni olarak
ayarlayın.

Windows PowerShell, yalnızca açık terminal oturumu için:

```powershell
$env:GEMINI_API_KEY="GEMINI_API_ANAHTARINIZ"
python run_gemini_zero_shot.py
```

macOS/Linux:

```bash
export GEMINI_API_KEY="GEMINI_API_ANAHTARINIZ"
python run_gemini_zero_shot.py
```

Kod `gemini-2.5-flash` ve sıcaklık `0.0` kullanır. Modele koordinatlar değil,
yalnızca nokta görseli gönderilir. Ham model cevabı da sonuç JSON'una
kaydedilir.

## Gemini Multi-Agent 2

Zero-shot sonucu hazırlandıktan sonra ilk olarak tek eleştirmen iterasyonu:

```powershell
python run_gemini_multi_agent2.py --iterations 1
```

Eleştirmen mevcut rota görselini inceler ve sıcaklık `0.7` ile yeni bir rota
önerir. Kod son iterasyon ile bütün iterasyonlar içinde bulunan en iyi geçerli
rotayı ayrı ayrı saklar. İlk çağrı doğrulandıktan sonra sırasıyla 3 ve 10
iterasyon denenebilir:

```powershell
python run_gemini_multi_agent2.py --iterations 3
python run_gemini_multi_agent2.py --iterations 10 --delay-seconds 13
```

Ücretsiz `gemini-2.5-flash` katmanında dakika başına istek sınırı bulunduğu
için final deneyinde çağrılar arasına 13 saniye konur. Her başarılı iterasyon
`gemini_multi_agent2_checkpoint.json` dosyasına hemen kaydedilir; olası kota
veya ağ hatalarında önceki sonuçlar kaybolmaz.

Kota nedeniyle deney yarıda kalırsa, kota yenilendikten sonra önceki çağrıları
tekrarlamadan checkpoint'ten devam edilir:

```powershell
python run_gemini_multi_agent2.py --iterations 10 --delay-seconds 13 --resume
```

## Gemini Multi-Agent 1

Multi-Agent 1, makaledeki üç rolü kullanır:

1. Daha önce üretilen zero-shot rota initializer çözümü olarak alınır.
2. Critic, sıcaklık `0.7` ile tek Gemini çağrısında yedi farklı rota adayı
   üretir (`candidate_count=7`).
3. Yedi aday ayrı ayrı çizilir.
4. Scorer, yedi görseli aynı Gemini çağrısında inceler ve sıcaklık `0.0` ile
   en iyi gördüğü adayı seçer.
5. Seçilen rota bir sonraki iterasyonun critic girdisi olur.

Bu yöntem yalnız görsel akıl yürütme koşulunu korur. Scorer'a koordinatlar,
mesafe matrisi veya Python tarafından hesaplanan rota uzunlukları gönderilmez.
Python tarafındaki gerçek mesafe ve gap hesapları yalnız deney
değerlendirmesinde kullanılır.

Kodun girdilerini ve planını **Gemini API çağrısı yapmadan** doğrulamak için:

```powershell
python run_gemini_multi_agent1.py --iterations 1 --validate-only
```

Bu komut API anahtarı istemez ve kota kullanmaz. Gerçek API deneyi için önce
tek iterasyon çalıştırılmalıdır:

```powershell
python run_gemini_multi_agent1.py --iterations 1 --candidate-count 7 --delay-seconds 13
```

Bir tam Multi-Agent 1 iterasyonu iki Gemini isteği kullanır: bir critic isteği
tek seferde yedi aday üretir, bir scorer isteği yedi görseli birlikte puanlar.
Gemini bazı çağrılarda istenenden az aday döndürürse kod mevcut adaylarla devam
eder ve `returned_candidate_count` alanına gerçek sayıyı yazar. Makaleyle tam
uyumlu bir final iterasyonu kabul etmek için bu alanın `7` olduğu kontrol
edilmelidir.
İlk iterasyon doğrulandıktan sonra final hedefi 10 iterasyondur:

```powershell
python run_gemini_multi_agent1.py --iterations 10 --candidate-count 7 --delay-seconds 13 --resume
```

Her critic çağrısından hemen sonra aday rotalar ve görseller
`gemini_multi_agent1_checkpoint.json` dosyasına kaydedilir. Kota scorer
aşamasında dolarsa `--resume`, yedi critic adayını yeniden üretmeden yalnız
scorer çağrısından devam eder. Sonuçlar
`gemini_multi_agent1_results.json` dosyasına yazılır.

Sonuç JSON'unda üç farklı kayıt özellikle ayrılır:

- `final_solution`: son scorer seçimi,
- `best_valid_solution`: initializer ve scorer seçimleri içindeki en kısa
  geçerli sistem çıktısı,
- `best_critic_candidate_oracle`: Python mesafesine göre bütün critic adayları
  arasındaki en iyi geçerli aday. Bu alan scorer performansını sonradan analiz
  etmek içindir; scorer'a gösterilmez.

## Neden kesin optimum da var?

Makaledeki OR-Tools ayarları sezgiseldir ve teorik optimum garantisi vermez.
Bu çalışmada yalnızca 10 nokta bulunduğu için 9! = 362.880 olası ziyaret
sırasını kontrol edebiliriz. Böylece OR-Tools ve ileride MLLM ile bulunan
rotaların gerçek optimumdan ne kadar uzak olduğunu da ölçebiliriz.
