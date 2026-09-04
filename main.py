"""
Konu bazlı RAG servisi. AITestModel şemasına uygun /ask endpoint'i.

Akış:
  1. Soru + konuId gelir
  2. Qdrant'ta ilgili koleksiyonda (sınıfa göre) konu_id filtresiyle arama
  3. Skor eşiği üstündeki chunk'lar toplanır
  4. Sonuç H100'deki vLLM'e gönderilir, cevap üretilir
  5. Cevap + kaynak bilgisi döner

Çalıştırma: uvicorn main:app --host 0.0.0.0 --port 5005
"""

import logging
import re
from typing import Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kanka_rag")

def temizle_think_blogu(metin: str) -> str:
    """Qwen3'un urettigi <think>...</think> blogunu temizler."""
    temiz = re.sub(r"<think>.*?</think>\s*", "", metin, flags=re.DOTALL)
    return temiz.strip()


def temizle_uydurma_kaynak(metin: str) -> str:
    """Modelin kendiliginden ekledigi (Video: ...), (Sayfa: ...) gibi
    uydurma kaynak referanslarini temizler. Gercek kaynak notu ayrica
    kod tarafindan eklenir."""
    desenler = [
        r"\(Video[:\s][^)]*\)",
        r"\(Sayfa[:\s][^)]*\)",
        r"\(Kaynak[:\s][^)]*\)",
        r"\([Dd]akika[:\s][^)]*\)",
    ]
    temiz = metin
    for desen in desenler:
        temiz = re.sub(desen, "", temiz)
    temiz = re.sub(r"\s+", " ", temiz).strip()
    return temiz

# ---------- ayarlar ----------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBED_MODEL = "BAAI/bge-m3"
SKOR_ESIGI = 0.45
MAX_CHUNK = 4  # vLLM max-model-len 2048 olduğu için düşük tutuyoruz

VLLM_URL = "http://10.106.250.94:8000/v1/chat/completions"
VLLM_MODEL = "ogretmen"

# ---------- başlangıçta bir kez yükle ----------

app = FastAPI()
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
logger.info("Embedding modeli yükleniyor: %s", EMBED_MODEL)
embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
logger.info("Hazır.")


class AskRequest(BaseModel):
    model_config = {"populate_by_name": True}

    Sinif: str = Field(alias="sinif")
    Ders: str = Field(alias="ders")
    KonuId: str = Field(alias="konuId")
    Question: str = Field(alias="question")
    K: str = Field(default="3", alias="k")
    UserId: Optional[str] = Field(default=None, alias="userId")


class AskResponse(BaseModel):
    Response: Optional[str] = None


def koleksiyon_adi(sinif: str) -> str:
    # Şimdilik tüm pilot veri dok_YKS koleksiyonunda; Sinif ne gelirse gelsin oraya yönlendiriliyor.
    return "dok_YKS"


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    ad = koleksiyon_adi(req.Sinif)
    logger.info("Soru: %s | konu_id: %s | koleksiyon: %s", req.Question, req.KonuId, ad)

    soru_vektoru = embed_model.encode([req.Question], normalize_embeddings=True)[0]

    try:
        limit = max(int(req.K) * 3, 15)
    except ValueError:
        limit = 15

    try:
        sonuclar = qdrant.query_points(
            collection_name=ad,
            query=soru_vektoru.tolist(),
            query_filter=Filter(
                must=[FieldCondition(key="konu_id", match=MatchValue(value=req.KonuId.upper()))]
            ),
            limit=limit,
        ).points
    except Exception as e:
        logger.error("Qdrant hatası: %s", e)
        return AskResponse(Response=f"[Qdrant hatası: {e}]")

    ilgili = [s for s in sonuclar if s.score >= SKOR_ESIGI][:MAX_CHUNK]
    logger.info("Bulunan chunk sayısı: %d (eşik üstü: %d)", len(sonuclar), len(ilgili))

    if not sonuclar:
        # konu_id filtresine hiç uyan nokta yok — sistemde ne var, onu söyleyelim.
        try:
            mevcut, _ = qdrant.scroll(collection_name=ad, limit=50, with_payload=["baslik"])
            basliklar = sorted({p.payload.get("baslik", "") for p in mevcut if p.payload.get("baslik")})
        except Exception:
            basliklar = []

        if basliklar:
            liste = ", ".join(basliklar)
            mesaj = f"Bu sayfada bu konuda içeriğim yok. Şu an şu konularda sorularını cevaplayabilirim: {liste}."
        else:
            mesaj = "Bu sayfada henüz bir içerik bulunmuyor."

        return AskResponse(Response=mesaj)

    if not ilgili:
        return AskResponse(Response="Bu konuda elimde yeterli bilgi bulunmuyor.")

    baglam_parcalari = []
    kaynaklar = []
    for s in ilgili:
        p = s.payload
        baglam_parcalari.append(p["metin"])
        kaynak_bilgi = {"baslik": p["baslik"], "kaynak_tur": p["kaynak_tur"], "skor": round(s.score, 3)}
        if p["kaynak_tur"] == "pdf":
            kaynak_bilgi["sayfalar"] = p.get("sayfalar", [])
        else:
            kaynak_bilgi["baslangic_sn"] = p.get("baslangic_sn")
        kaynaklar.append(kaynak_bilgi)

    baglam = "\n\n---\n\n".join(baglam_parcalari)

    sistem_mesaji = (
        "Sen bir öğretmen asistanısın. Sana verilen ders içeriğine dayanarak "
        "öğrencinin sorusunu açık ve anlaşılır şekilde cevapla. "
        "Sadece verilen içerikteki bilgileri kullan, ekleme yapma. "
        "Cevabında zaman damgası, video saniyesi, sayfa numarası gibi "
        "kaynak referansları EKLEME — bunlar ayrıca ve doğru şekilde "
        "sistem tarafından sağlanacaktır. Sadece konu içeriğini anlat."
    )
    kullanici_mesaji = f"Ders içeriği:\n\n{baglam}\n\nÖğrenci sorusu: {req.Question}"

    try:
        yanit = requests.post(
            VLLM_URL,
            json={
                "model": VLLM_MODEL,
                "messages": [
                    {"role": "system", "content": sistem_mesaji},
                    {"role": "user", "content": kullanici_mesaji},
                ],
                "temperature": 0.3,
                "max_tokens": 400,
            },
            timeout=30,
        )
        yanit.raise_for_status()
        cevap_metni = yanit.json()["choices"][0]["message"]["content"]
        cevap_metni = temizle_think_blogu(cevap_metni)
        cevap_metni = temizle_uydurma_kaynak(cevap_metni)
    except Exception as e:
        logger.error("vLLM hatası: %s", e)
        cevap_metni = f"[vLLM hatası: {e}]"
        return AskResponse(Response=cevap_metni)

    if kaynaklar:
        en_iyi = kaynaklar[0]
        if en_iyi["kaynak_tur"] == "pdf" and en_iyi.get("sayfalar"):
            kaynak_notu = f"(Kaynak: {en_iyi['baslik']}, sayfa {en_iyi['sayfalar'][0]})"
        elif en_iyi.get("baslangic_sn") is not None:
            dk = int(en_iyi["baslangic_sn"] // 60)
            sn = int(en_iyi["baslangic_sn"] % 60)
            kaynak_notu = f"(Kaynak: {en_iyi['baslik']}, {dk}:{sn:02d})"
        else:
            kaynak_notu = f"(Kaynak: {en_iyi['baslik']})"
        cevap_metni = f"{cevap_metni}\n\n{kaynak_notu}"

    return AskResponse(Response=cevap_metni)


@app.get("/health")
def health():
    return {"status": "ok"}
