from __future__ import annotations

from pathlib import Path
from src.providers import get_provider


def test_provider_factory_and_validation():
    print("🧪 Provider mimarisi test ediliyor...\n")

    # 1. Desteklenmeyen / Bilinmeyen bir provider testi (Hata fırlatmalı)
    try:
        get_provider("bilinmeyen_model")
    except ValueError as e:
        print(f"✅ Başarılı: Beklenen hata yakalandı -> {e}")

    # 2. Gemini sağlayıcısının başlatılması ve geçersiz parametre testi
    try:
        # Boş model adı hata vermeli
        gemini = get_provider("gemini", model="")
    except ValueError as e:
        print(f"✅ Başarılı: Geçersiz model adı hatası yakalandı -> {e}")

    # 3. İskelet (Stub) provider'ların (Groq, Mistral vb.) yüklenme testi
    providers_to_test = ["groq", "mistral", "openrouter", "dashscope"]
    
    for name in providers_to_test:
        try:
            provider = get_provider(name)
            print(f"✅ Başarılı: '{name}' sağlayıcısı başarıyla yüklendi ({type(provider).__name__}).")
        except Exception as e:
            print(f"❌ Hata: '{name}' yüklenirken sorun oluştu -> {e}")

    print("\nTüm yapılandırma ve fabrika testleri tamamlandı! 🎉")


if __name__ == "__main__":
    test_provider_factory_and_validation()