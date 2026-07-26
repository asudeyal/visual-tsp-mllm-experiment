# TSPLIB eil51 Görsel TSP Deneyi

Bu klasör, mevcut `tsp10_experiment` çalışmasına dokunmadan TSPLIB'nin
`eil51` problemini ayrı bir deney olarak çalıştırır. Problem 51 düğüm ve tek
satıcı içerir; düğüm `1` depodur.

Bu bir Gemini uyarlamasıdır. Model yalnız etiketli görselleri görür;
koordinatlar, mesafe matrisi ve Python tarafından hesaplanan rota uzunlukları
modele gönderilmez.

## Neden ayrı klasör?

Tüm kaynak kod, veri ve çıktılar `eil51_experiment/` altında tutulur. Böylece
10 noktalı mevcut deney değişmez. Denemeden vazgeçilirse klasörün tamamı
silinebilir:

```powershell
cd ..
Remove-Item -Recurse -Force .\eil51_experiment
```

Klasör Git'e eklendiyse silme işlemi `git rm -r eil51_experiment` ile yapılır.
Saklanmak istenen sonuçlar varsa önce `output/` klasörü kopyalanmalıdır.

## Yöntemler ve doğru referans

| Yöntem | Açıklama |
|---|---|
| TSPLIB bilinen optimum | Paketlenmiş optimum tur; mesafe `426` |
| OR-Tools | SAVINGS + GUIDED_LOCAL_SEARCH ile sezgisel çözüm |
| Gemini zero-shot | Problem görselinden tek çağrıda rota |
| Gemini Multi-Agent 2 | Her iterasyonda bir critic rota revizyonu |
| Gemini Multi-Agent 1 | Her iterasyonda 7 critic adayı ve bir görsel scorer |

### Multi-Agent 1 geçerlilik filtresi

51 düğümlü yoğun görsellerde görsel scorer bir düğümün eksik veya tekrarlı
olduğunu güvenilir biçimde sayamayabilir. Bu nedenle scorer öncesinde yalnız
TSP kısıtlarını kontrol eden deterministik bir geçerlilik filtresi uygulanır.
Python tarafından hesaplanan mesafe veya gap modele verilmez.

- Scorer yalnız geçerli critic adaylarını görür.
- Tek geçerli aday varsa API çağrısı yapılmadan o aday seçilir.
- Geçerli aday yoksa önceki rota korunur.
- Filtre ve seçim biçimi sonuç JSON'unda `scorer_policy`, `selection_mode`,
  `eligible_candidate_ids` ve `excluded_invalid_candidate_ids` alanlarında
  açıkça kaydedilir.

Bu sürüm, makaledeki saf görsel scorer'ın **feasibility-filtered** bir
uyarlamasıdır. Akademik raporda bu farklılık belirtilmelidir.

Eil51 için brute-force uygulanmaz. Depo sabitken `50!` ziyaret sırası vardır;
bu sayı pratik hesaplama için aşırı büyüktür. Bu nedenle gerçek karşılaştırma
referansı TSPLIB'nin bilinen optimum mesafesi `426`dır.

Mesafeler normal ondalıklı Öklid mesafesiyle değil TSPLIB `EUC_2D` kuralıyla,
her kenar en yakın tam sayıya yuvarlanarak hesaplanır. Gap:

```text
gap (%) = 100 × (bulunan mesafe - 426) / 426
```

Gap yalnız geçerli TSP rotaları için hesaplanır. Geçersiz bir rotanın çizilen
kenar uzunluğu tanı amacıyla saklanabilir, fakat gap alanı `null` olur ve bu
rota optimumla karşılaştırılmaz.

## Dosya yapısı

```text
eil51_experiment/
├── data/
│   ├── eil51.tsp
│   └── eil51.opt.tour
├── src/
│   ├── core.py
│   ├── gemini.py
│   ├── metrics.py
│   └── summaries.py
├── tests/
├── output/runs/<run-id>/
│   ├── baseline/
│   ├── zero_shot/
│   ├── multi_agent1/
│   └── multi_agent2/
├── run_baseline.py
├── run_gemini_zero_shot.py
├── run_gemini_multi_agent1.py
└── run_gemini_multi_agent2.py
```

Her yöntemin klasöründe ayrıntılı `*_results.json`, okunması kolay
`*_summary.json` ve `images/` bulunur. Checkpoint dosyaları Git tarafından
yok sayılır; sonuç JSON'ları ve görseller istenirse commit edilebilir.

## Kurulum — Windows PowerShell

Repo kökünde, `tsp10_experiment` ile aynı seviyede:

```powershell
cd .\eil51_experiment
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

API anahtarı yalnız açık terminal oturumu için ayarlanır ve kaynak koda
yazılmaz:

```powershell
$env:GEMINI_API_KEY="GEMINI_API_ANAHTARINIZ"
```

## Önerilen çalışma sırası

Aynı deney grubundaki bütün komutlarda aynı `--run-id` kullanılmalıdır:

```powershell
python run_baseline.py --run-id eil51_run_01 --ortools-time-limit 30
python run_gemini_zero_shot.py --run-id eil51_run_01
python run_gemini_multi_agent2.py --iterations 1 --run-id eil51_run_01
python run_gemini_multi_agent1.py --iterations 1 --candidate-count 7 --run-id eil51_run_01
```

İlk çağrılar ve çıktılar kontrol edildikten sonra 10 iterasyona tamamlama:

```powershell
python run_gemini_multi_agent2.py --iterations 10 --run-id eil51_run_01 --resume
python run_gemini_multi_agent1.py --iterations 10 --candidate-count 7 --run-id eil51_run_01 --resume
```

Kodda yapay `delay` veya `sleep` yoktur. Ücretsiz API kotası dolarsa deney
hata kaydını ve checkpoint'i yazar. Kota yenilendiğinde aynı komut `--resume`
ile tekrar çalıştırılır. Multi-Agent 1 scorer aşamasında kesilirse üretilmiş
critic adayları tekrar çağrılmadan kullanılır.

Multi-Agent 1 girdilerini kota kullanmadan kontrol etmek için:

```powershell
python run_gemini_multi_agent1.py --iterations 1 --candidate-count 7 --run-id eil51_run_01 --validate-only
```

## Yalnız özet JSON'u yeniden üretme

Bu komutlar mevcut uzun JSON'lardan API çağrısı yapmadan özet çıkarır:

```powershell
python run_baseline.py --run-id eil51_run_01 --summary-only
python run_gemini_zero_shot.py --run-id eil51_run_01 --summary-only
python run_gemini_multi_agent2.py --run-id eil51_run_01 --summary-only
python run_gemini_multi_agent1.py --run-id eil51_run_01 --summary-only
```

## Zaman ve kullanım kayıtları

JSON dosyalarında yöntem/iterasyon türü, API başlangıç-bitiş zamanı, API duvar
süresi, görsel sayısı ve boyutu, token kullanımı, ayrıştırma, doğrulama,
çizim süreleri ve hata bilgileri tutulur. Multi-Agent 1'de critic ve scorer;
Multi-Agent 2'de her critic iterasyonu ayrı kaydedilir. Yapay bekleme süresi
olmadığı `artificial_delay_enabled: false` alanıyla açıkça belirtilir.

## Notlar

- 51 etiketli görsel 10 noktalı örnekten çok daha zordur; Gemini geçersiz veya
  optimumdan uzak rota üretebilir. Bu da deneyin ölçmek istediği sonuçtur.
- Aynı `run-id` ile `--resume` kullanmadan yeniden başlatmak önceki yöntem
  sonuçlarını üzerine yazabilir. Yeni tekrar için `eil51_run_02` gibi yeni bir
  kimlik kullanın.
- API anahtarı, `.env`, sanal ortam ve checkpoint'ler commit edilmez.

## Veri kaynağı

`eil51.tsp` ve `eil51.opt.tour`, TSPLIB eil51 örneğinin yerel kopyalarıdır.
Bilinen optimum rota uzunluğu `426`dır.
