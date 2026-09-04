# -*- coding: utf-8 -*-
# server_V3.py

import re as _re
import unicodedata
from typing import Dict, List, Tuple, Optional
import time
import traceback
import json
from datetime import datetime

import numpy as np
import torch
from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import JSONResponse
from sklearn.cluster import AgglomerativeClustering
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langdetect import detect, LangDetectException

# ================== AYARLAR ==================
QDRANT_URL = "http://localhost:6333"
EMB_MODEL = "ytu-ce-cosmos/turkish-e5-large"

SCORE_GAP_THRESHOLD = 0.04
INTENT_MIN_THRESH = 0.65
CLUSTER_THRESHOLD = 0.1

TOP_K_PER_KEY = 10
MAX_RETURN = 5
MIN_DOC_SCORE = 0.58

FRAGMENT_WEIGHT = 0.65
SENTENCE_WEIGHT = 0.35

# ---- Sınıf -> Koleksiyon(lar) ----
SINIF_COLLECTIONS = {
    "5Sinif": ["5Sinif"],
    "6Sinif": ["5Sinif", "6Sinif"],
    "9Sinif": ["9Sinif"],
    "10Sinif": ["9Sinif", "10Sinif"],
    "YKS": ["YKS"],
    "LGS": ["LGS"],
}

# ---- 12 intent ----
INTEREST = [
    "Video", "Konu Özeti", "Hazır Bulunuşluk Testi", "Değerlendirme Testi",
    "Soru Çöz", "Oynayarak Öğren", "Sesli Özet", "Tarama Testi",
    "Çıkmış Sorular", "Etkinliklerle Öğren", "Bağlam Temelli Sorular", "3B Model",
]

EXACT_META_KEY: Dict[str, str] = {
    "Video": "video",
    "Konu Özeti": "konu_özeti",
    "Hazır Bulunuşluk Testi": "hazırbulunuşluk",
    "Değerlendirme Testi": "değerlendirme_testi",
    "Soru Çöz": "soru_çöz",
    "Oynayarak Öğren": "oynayarak_öğren",
    "Sesli Özet": "sesli_özet",
    "Tarama Testi": "tarama_testi",
    "Çıkmış Sorular": "çıkmış_sorular",
    "Etkinliklerle Öğren": "etkinliklerle_öğren",
    "Bağlam Temelli Sorular": "bağlam_temelli_sorular",
    "3B Model": "3b_model",
}

REVERSE_META_KEY = {v: k for k, v in EXACT_META_KEY.items()}


def keys_for_label(label: str) -> List[str]:
    return [EXACT_META_KEY[label]]


# ================== Regex kalıpları ==================
INTENT_REGEX: Dict[str, List[str]] = {
    "Tarama Testi": [
        r"\btarama\s*test(?:i|leri)?\b",
        r"\bgenel\s*tarama\b",
        r"\bünite\s*tarama(?:s[ıi])?\b",
        r"\byoklama\s*test(?:i|leri)?\b",
        r"\bgenel\s*değerlendirme\s*tarama\b",
        r"\btarama\s*sorular[ıi]\b",
    ],
    "Video": [
        r"\bvideo(?:su|yu|yü|ya|da|de|dan|den|lar[ıi])?\b",
        r"\bvideolu\b", r"\bgörüntülü\b",
        r"\b(?:konu|ders)\s*video(?:su)?\b",
        r"\bvideo\s*(?:izle(?:mek)?|aç(?:mak)?|oynat(?:mak)?|göster(?:mek)?)\b",
        r"\b(video|vidyo)(?:su|yu|yü|ya|da|de|dan|den|lar[ıi])?\b",
        r"\bvideolu\b", r"\bgörüntülü\b",
        r"\b(?:konu|ders)\s*(?:video|vidyo)(?:su)?\b",
        r"\b(video|vidyo)\s*(?:izle(?:mek)?|aç(?:mak)?|oynat(?:mak)?|göster(?:mek)?|istiyorum)\b",
    ],
    "Konu Özeti": [
        r"^(?!.*\b(sesli|audio|video|izle|dinle|aç|oynat|göster)\b).*?\b(konu|ünite|ders)\s*özet(?:i|ini|ler|leri)?\b",
        r"^(?!.*\b(sesli|audio|video|izle|dinle|aç|oynat|göster)\b).*?\b(konu|ünite|ders).{0,20}\bözet\s*(?:ver|çıkar|yap|geç)\b",
        r"\b(?<!sesli\s|audio\s)(?:konu|ünite|ders)\s*özet(?:i|ini|ler|leri)?\b",
        r"\b(?<!sesli\s|audio\s)özet\s*(?:istiyorum|ver|çıkar|yap|geç)?\b",
    ],
    "Hazır Bulunuşluk Testi": [
        r"\bhazır\s*bulunuşluk\b", r"\bhazir\s*bulunusluk\b",
        r"\bön\s*test\b", r"\bbaşlangıç\s*test(?:i|leri)?\b",
        r"\bseviye\s*belirleme\b", r"\bdüzey\s*belirleme\b",
        r"\bdiagnostik(?:\s*test)?\b",
        r"\bhazır\s*bulunuşluk\s*test(?:i|leri)?\b", r"\bhazir\s*bulunusluk\s*test(?:i|leri)?\b",
    ],
    "Değerlendirme Testi": [
        r"\bdeğerlendirme\s*test(?:i|leri)?\b", r"\bdegerlendirme\s*test(?:i|leri)?\b",
        r"\bünite\s*sonu\s*test(?:i|leri)?\b", r"\bson\s*test\b",
        r"\bkapanış\s*test(?:i|leri)?\b",
    ],
    "Soru Çöz": [
        r"\bsoru\s*çöz(?:üm|mek|ümü)?\b", r"\bçözümlü\s*sorular?\b",
        r"\btest\s*çöz(?:mek)?\b", r"\bdeneme\s*çöz(?:mek)?\b",
        r"\bproblem\s*çöz(?:mek)?\b", r"\balıştırma(?:lar)?\b", r"\begzersiz(?:ler)?\b",
    ],
    "Oynayarak Öğren": [
        r"\boynayarak\s*öğren\b", r"\b(eğlenerek|zevkli)\s*öğren(?:me)?\b",
        r"\boyun\s*tabanlı\b", r"\binteraktif\b", r"\betkileşim(?:li)?\b",
        r"\boynayarak\s*öğren(?:me|meyi|meye)?\b",
        r"\boyun(?:la)?\s*öğren(?:me)?\b",
        r"\beğitici\s*oyun\b",
        r"\binteraktif\s*etkinlik\b",
        r"\betkileşimli\s*etkinlik\b"
    ],
    "Sesli Özet": [
        r"\bsesli\s*özet\b", r"\bsesli\s*anlat(?:ım|im)\b",
        r"\bözet\s*dinle\b", r"\bkonuyu\s*dinle\b",
        r"\bsesli\s*özet(?:i|ini|ler|leri)?\b",
        r"\bözet(?:i|ini)?\s*dinle\b",
        r"\bkonu(?:yu)?\s*dinle\b",
    ],
    "Çıkmış Sorular": [
        r"\b(çıkm(?:ış|is)|cikmis)\s*soru(?:lar|ler)?\b",
        r"\b(ösym|yks|tyt|ayt|lgs)\s*(çıkm(?:ış|is)|geçmiş\s*yıl(?:lar)?(?:ın)?)\s*soru(?:lar|ler)?\b",
        r"\b(geçmiş\s*yıl(?:lar)?(?:ın)?|eski\s*sınav)\s*soru(?:lar|ler)?\b",
    ],
    "Etkinliklerle Öğren": [
        r"\betkinlik(?:lerle|li)?\s*öğren\b", r"\betkinlik(?:ler)?\b",
        r"\bçalışma\s*(kağıdı|sayfas[ıi])\b", r"\bdeney(?:ler)?\b", r"\bproje(?:ler)?\b",
        r"\betkinlik(?:lerle|li)?\s*öğren(?:me|meyi|meye)?\b",
        r"\betkinlik(?:ler|leri|lerle|li|liğini|liği|liğe)?\b",
    ],
    "Bağlam Temelli Sorular": [
        r"^(?!.*\b(çıkm(?:ış|is)|cikmis|geçmiş\s*yıl(?:lar)?(?:ın)?|ösym|yks|tyt|ayt|lgs)\b).*?\bbağlam(?:\s*temelli)?\b.*\b(soru|sorular|metin|paragraf|yorum|grafik|tablo)\b",
        r"^(?!.*\b(çıkm(?:ış|is)|cikmis|geçmiş\s*yıl(?:lar)?(?:ın)?|ösym|yks|tyt|ayt|lgs)\b).*?\bbağlamsal\b.*\b(soru|metin|paragraf|yorum)\b",
        r"^(?!.*\b(çıkm(?:ış|is)|cikmis|geçmiş\s*yıl(?:lar)?(?:ın)?|ösym|yks|tyt|ayt|lgs)\b).*?\bmetne\s*dayalı\s*sorular?\b",
        r"^(?!.*\b(çıkm(?:ış|is)|cikmis|geçmiş\s*yıl(?:lar)?(?:ın)?|ösym|yks|tyt|ayt|lgs)\b).*?\bgrafik\s*yorum(?:lama|u)\b",
        r"\bbağlam(?:\s*temelli)?\s*soru(?:lar[ıi]?)?\b",
        r"\bbağlamsal\s*soru(?:lar[ıi]?)?\b",
        r"\bmetne\s*dayalı\s*soru(?:lar[ıi]?)?\b",
    ],
    "3B Model": [
        r"\b(?:3b|3d|3\s*boyutlu|üç\s*boyutlu)\s*(model|görsel|nesne|simülasyon|animasyon)\b",
        r"\b3d\s*model(?:leme)?\b",
    ],
}

BLACKLIST = [
    "merhaba", "mrb", "mrb.", "mrbk", "mrbler",
    "selam", "selamlar", "slm", "slm.", "slmlar",
    "selamün aleyküm", "selamun aleykum", "sea", "sa", "s.a",
    "hello", "hi", "hey", "heyy", "heyyy",
    "alo", "yo", "sup", "whats up", "what’s up",
    "good morning", "good evening", "good night",
    "günaydın", "tünaydın", "iyi akşamlar", "iyi geceler",
    "nasılsın", "nasilsin", "nbr", "naber", "ne haber",
    "nabıyon", "nabıyorsun", "napıyorsun", "napıyosun", "napiyon",
    "how are you", "how r u", "wassup", "sup bro", "what’s up bro",
    "iyiyim", "ben iyiyim", "çok iyiyim", "idare eder", "fena değil",
    "iyi sen", "sen nasılsın", "iyi valla", "şükür",
    "i’m fine", "fine thanks", "good", "not bad",
    "teşekkürler", "tesekkurler", "sağol", "sagol", "eyvallah", "çok sağ ol",
    "thank you", "thanks", "thx", "ty",
    "bye", "goodbye", "see you", "see ya", "take care",
    "görüşürüz", "gorusuruz", "hoşça kal", "kendine iyi bak", "bay bay",
    "ok", "okey", "oki", "tamam", "aynen", "aynn",
    "lol", "haha", "hahaha", "ahah", "xd", "xD", "😂", "🙂", "👍",
    "hmm", "hımm", "hım",
]


# ================== Yardımcılar ==================
def normalize_tr_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("İ", "I").replace("I", "ı")
    s = s.lower()
    return _re.sub(r"\s+", " ", s).strip()


def is_blacklisted(q: str) -> bool:
    qn = normalize_tr_text(q)
    for t in BLACKLIST:
        tn = normalize_tr_text(t)
        if any(ch.isalnum() for ch in tn):
            if _re.search(r'(?<!\w)' + _re.escape(tn) + r'(?!\w)', qn):
                return True
        else:
            if tn in qn:
                return True
    return False


def combine_ders_query(ders_adi: str, text: str) -> str:
    ders_adi_norm = normalize_tr_text(ders_adi) if ders_adi else ""
    text_norm = normalize_tr_text(text)
    if not ders_adi_norm:
        return text.strip()
    if text_norm.startswith(ders_adi_norm + " "):
        return text.strip()
    return f"{ders_adi.strip()} {text.strip()}"


# ================== Embeddings & etiket vektörleri ==================
embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

LABEL_ALIASES: Dict[str, List[str]] = {
    "Video": ["video", "ders videosu", "videolu içerik", "videoyu aç", "video istiyorum"],
    "Konu Özeti": ["konu özeti", "konunun özeti", "özet çıkarımı", "özet istiyorum", "özet ver"],
    "Hazır Bulunuşluk Testi": ["hazır bulunuşluk", "ön test", "diagnostik test", "başlangıç testi", "hazırbulunuşluk"],
    "Değerlendirme Testi": ["değerlendirme testi", "ünite değerlendirmesi", "kapanış testi"],
    "Soru Çöz": ["soru çöz", "soru çözümü", "çözümlü sorular", "alıştırma çöz"],
    "Oynayarak Öğren": ["oynayarak öğren", "oyunla öğrenme", "oyun tabanlı etkinlik", "eğitici oyun"],
    "Sesli Özet": ["sesli özet", "dinlemeli özet", "audio özet", "sesli özet dinle"],
    "Tarama Testi": ["tarama testi", "genel tarama", "tarama soruları", "ünite taraması"],
    "Çıkmış Sorular": ["çıkmış sorular", "ÖSYM soruları", "geçmiş yıllar soruları", "eski sınav soruları"],
    "Etkinliklerle Öğren": ["etkinliklerle öğren", "etkinlikli öğrenme", "etkinlik çalışması"],
    "Bağlam Temelli Sorular": ["bağlam temelli sorular", "bağlamsal sorular", "metne dayalı sorular"],
    "3B Model": ["3B model", "3D model", "3 boyutlu model", "3 boyutlu görselleştirme"],
}

_label_vecs: Dict[str, np.ndarray] = {}
for label, aliases in LABEL_ALIASES.items():
    vecs = [np.array(embeddings.embed_query("query: " + normalize_tr_text(a)), dtype=np.float32) for a in aliases]
    _label_vecs[label] = np.stack(vecs, axis=0).mean(axis=0)


def embed_query(text: str) -> np.ndarray:
    return np.array(embeddings.embed_query("query: " + normalize_tr_text(text)), dtype=np.float32)


def extract_fragments(query: str, intent: str) -> List[str]:
    q = normalize_tr_text(query)
    pats = INTENT_REGEX.get(intent, [])
    frags: List[str] = []
    for p in pats:
        for m in _re.finditer(p, q, _re.IGNORECASE | _re.UNICODE):
            frag = m.group(0).strip()
            if frag and frag not in frags:
                frags.append(frag)
    return frags


def score_intents(query: str) -> Dict[str, float]:
    qv = embed_query(query)
    scores: Dict[str, float] = {}
    for lab, lab_vec in _label_vecs.items():
        sent_sim = float(np.dot(qv, lab_vec))
        frags = extract_fragments(query, lab)
        if frags:
            best = -1.0
            for f in frags:
                fv = embed_query(f)
                sim = float(np.dot(fv, lab_vec))
                if sim > best:
                    best = sim
            final = FRAGMENT_WEIGHT * best + SENTENCE_WEIGHT * sent_sim
        else:
            final = sent_sim
        scores[lab] = final
    return scores


def cluster_scores(scores: Dict[str, float]) -> List[Tuple[int, List[Tuple[str, float]]]]:
    if not scores:
        return []

    filt = {k: v for k, v in scores.items() if v >= INTENT_MIN_THRESH}
    if not filt:
        return []

    labels_sorted = sorted(filt.items(), key=lambda kv: kv[1], reverse=True)

    if len(labels_sorted) > 1:
        top, second = labels_sorted[0][1], labels_sorted[1][1]
        gap = top - second
        if gap < SCORE_GAP_THRESHOLD:
            return [(0, [(labels_sorted[0][0], float(top)), (labels_sorted[1][0], float(second))])]

    if len(labels_sorted) == 1:
        lab, sc = labels_sorted[0]
        return [(0, [(lab, float(sc))])]

    labels = [lab for lab, _ in labels_sorted]
    vals = np.array([filt[k] for k in labels], dtype=np.float32).reshape(-1, 1)

    clu = AgglomerativeClustering(n_clusters=None, distance_threshold=CLUSTER_THRESHOLD, linkage="ward").fit(vals)
    clusters: Dict[int, List[Tuple[str, float]]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(clu.labels_[idx]), []).append((lab, float(vals[idx, 0])))

    for k in clusters:
        clusters[k].sort(key=lambda x: x[1], reverse=True)

    ranked = sorted(clusters.items(), key=lambda kv: max(s for _, s in kv[1]), reverse=True)
    return [(new_id, items) for new_id, (_, items) in enumerate(ranked)]


def get_store(client: QdrantClient, collection_name: str) -> QdrantVectorStore:
    return QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)


def search_in_collections(text: str, class_label: str, ders_adi: Optional[str] = None) -> Tuple[List[dict], List[str]]:
    scores = score_intents(text)
    clusters = cluster_scores(scores)
    top_labels = [lab for lab, _ in clusters[0][1]] if clusters else []

    cols = SINIF_COLLECTIONS.get(class_label, [])
    if not cols:
        raise ValueError(f"'{class_label}' için koleksiyon eşleşmesi yok.")

    client = QdrantClient(url=QDRANT_URL)
    augmented = combine_ders_query(ders_adi or "", text)
    qtext_url = "query: " + (text if str(text).strip() else augmented)
    qtext_intent = "query: " + augmented

    results: List[dict] = []

    for col in cols:
        try:
            client.get_collection(col)
        except Exception:
            print(f"[Uyarı] Koleksiyon yok atlandı: {col}")
            continue

        vs = get_store(client, col)

        # URL-first
        try:
            base_hits = vs.similarity_search_with_score(qtext_url, k=TOP_K_PER_KEY)
            for doc, score in base_hits:
                meta = doc.metadata or {}
                url = meta.get("url")
                if url:
                    true_flags = [k for k, v in meta.items() if isinstance(v, bool) and v]
                    url_payload = meta.get("text_payload")
                    text_content = (url_payload or (doc.page_content or "")).strip()
                    results.append({
                        "collection": col,
                        "score": float(score),
                        "intent": None,
                        "intent_key": None,
                        "konu_id": meta.get("konu_id"),
                        "true_flags": true_flags,
                        "available_intents": [REVERSE_META_KEY.get(f) for f in true_flags if REVERSE_META_KEY.get(f)],
                        "text": text_content,
                        "url": url,
                        "url_payload": url_payload,
                    })
        except Exception as _e:
            print(f"[Uyarı] URL-first filtresiz arama hatası ({col}): {_e}")

        # Intent filtreli / Filtresiz
        if top_labels:
            for lab in top_labels:
                for key in keys_for_label(lab):
                    flt = qm.Filter(must=[qm.FieldCondition(
                        key=f"metadata.{key}", match=qm.MatchValue(value=True)
                    )])
                    for doc, score in vs.similarity_search_with_score(qtext_intent, k=TOP_K_PER_KEY, filter=flt):
                        meta = doc.metadata or {}
                        true_flags = [k for k, v in meta.items() if isinstance(v, bool) and v]
                        url = meta.get("url")
                        url_payload = meta.get("text_payload")
                        text_content = (url_payload or (doc.page_content or "")).strip()
                        results.append({
                            "collection": col,
                            "score": float(score),
                            "intent": lab,
                            "intent_key": key,
                            "konu_id": meta.get("konu_id"),
                            "true_flags": true_flags,
                            "available_intents": [REVERSE_META_KEY.get(f) for f in true_flags if
                                                  REVERSE_META_KEY.get(f)],
                            "text": text_content,
                            "url": url,
                            "url_payload": url_payload,
                        })
        else:
            for doc, score in vs.similarity_search_with_score(qtext_intent, k=TOP_K_PER_KEY):
                meta = doc.metadata or {}
                true_flags = [k for k, v in meta.items() if isinstance(v, bool) and v]
                url = meta.get("url")
                url_payload = meta.get("text_payload")
                text_content = (url_payload or (doc.page_content or "")).strip()
                results.append({
                    "collection": col,
                    "score": float(score),
                    "intent": None,
                    "intent_key": None,
                    "konu_id": meta.get("konu_id"),
                    "true_flags": true_flags,
                    "available_intents": [REVERSE_META_KEY.get(f) for f in true_flags if REVERSE_META_KEY.get(f)],
                    "text": text_content,
                    "url": url,
                    "url_payload": url_payload,
                })

    def _group_key(r: dict) -> str:
        kid = (r.get("konu_id") or "").strip()
        if kid:
            return f"k::{kid}"
        url = (r.get("url") or "").strip()
        if url:
            return f"u::{url}"
        txt = normalize_tr_text(r.get("text", ""))[:80]
        col = r.get("collection", "")
        return f"t::{col}::{txt}"

    grouped_results: Dict[str, dict] = {}
    for r in results:
        gk = _group_key(r)
        prev = grouped_results.get(gk)

        if prev is None:
            grouped_results[gk] = {
                **r,
                "intent": [r["intent"]] if r.get("intent") else [],
                "intent_key": [r["intent_key"]] if r.get("intent_key") else [],
            }
        else:
            if r.get("score", 0.0) > prev.get("score", 0.0):
                prev["score"] = r["score"]
                if r.get("url") and not prev.get("url"):
                    prev["url"] = r["url"]
                if r.get("konu_id") and not prev.get("konu_id"):
                    prev["konu_id"] = r["konu_id"]
                if r.get("text") and not prev.get("text"):
                    prev["text"] = r["text"]
                if r.get("url_payload") and not prev.get("url_payload"):
                    prev["url_payload"] = r["url_payload"]

            if r.get("intent") and r["intent"] not in prev["intent"]:
                prev["intent"].append(r["intent"])
                prev["intent_key"].append(r.get("intent_key"))

            if r.get("true_flags"):
                prev_flags = set(prev.get("true_flags", []))
                prev["true_flags"] = list(prev_flags.union(set(r["true_flags"])))

            if r.get("available_intents"):
                prev_av = set(prev.get("available_intents", []))
                prev["available_intents"] = list(prev_av.union(set(r["available_intents"])))

    final_results = [r for r in grouped_results.values() if r.get("score", 0.0) >= MIN_DOC_SCORE]
    final_results.sort(key=lambda r: r["score"], reverse=True)
    return final_results[:MAX_RETURN], top_labels


# ================== FastAPI Başlangıcı & Middleware ==================
app = FastAPI(title="Kanka RAG Semantic Search API")


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    response = await call_next(request)

    if request.url.path == "/search":
        try:
            params = request.query_params
            ders_adi = (params.get('ders_adi') or params.get('text') or params.get('title') or '').strip()
            ui_class = (params.get('class') or '').strip()
            query = (params.get('query') or '').strip() or ders_adi

            body_chunks = [chunk async for chunk in response.body_iterator]
            full_body = b"".join(body_chunks)
            # Response iterator'ını yeniden üretmek gerekir
            response = Response(
                content=full_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

            try:
                data = json.loads(full_body.decode("utf-8"))
                results = data.get("results", []) if isinstance(data, dict) else []
            except Exception:
                results = []

            items = []
            for r in results or []:
                text = (r.get("text") or "").replace("\n", " ").strip()
                kid = (r.get("konu_id") or "").strip()
                url = (r.get("url") or "").strip()
                if text or url:
                    items.append({"text": text if text else url, "konu_id": kid, "url": url})

            url_str = next((it["url"] for it in items if it["url"]), "")
            konu_id_str = next((it["konu_id"] for it in items if it["konu_id"]), "")

            sep = "#" * 129
            with open("log.txt", "a", encoding="utf-8") as f:
                f.write(sep + "\n")
                f.write(f"ders={ders_adi} \n")
                f.write(f"class={ui_class}\n")
                f.write(f"query={query}\n")
                f.write(f"url={url_str}\n")
                f.write(f"konu_id={konu_id_str}\n")
                f.write("yanit=\n")
                if items:
                    for i, it in enumerate(items, 1):
                        suffix = f" | konu_id={it['konu_id']}" if it["konu_id"] else ""
                        f.write(f"{i}. {it['text']}{suffix}\n")
                f.write(sep + "\n\n\n")
        except Exception as e:
            print(f"[LOG][ERROR] {e}")

    return response


@app.get("/")
def health_check():
    return {"status": "Server çalışıyor", "timestamp": time.time()}


@app.get("/search")
def similarity(
        query: Optional[str] = Query(None),
        class_: Optional[str] = Query(None, alias="class"),
        ders_adi: Optional[str] = Query(None),
        text: Optional[str] = Query(None),
        title: Optional[str] = Query(None),
):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === YENİ İSTEK ===")
    start_time = time.time()
    try:
        q = (query or '').strip()
        ui_class = (class_ or '').strip()
        d_adi = (ders_adi or text or title or '').strip()

        if not q and d_adi:
            q = d_adi

        if not q or not ui_class:
            raise HTTPException(status_code=400, detail="Both 'query' and 'class' parameters are required.")

        # Sınıf eşleştirmeleri
        class_mapping = {
            "5": "5Sinif", "6": "6Sinif", "9": "9Sinif",
            "10": "10Sinif", "YKS": "YKS", "LGS": "LGS",
            "5Sinif": "5Sinif", "6Sinif": "6Sinif", "9Sinif": "9Sinif", "10Sinif": "10Sinif"
        }
        if ui_class in class_mapping:
            ui_class = class_mapping[ui_class]
        else:
            raise HTTPException(status_code=400, detail=f"Invalid 'class' parameter: {ui_class}")

        if is_blacklisted(q):
            print("[INFO] Kara liste tetiklendi; boş yanıt döndürüldü.")
            return {
                "results": [{
                    "rank": 1,
                    "collection": ui_class,
                    "score": 0.0,
                    "user_intent": [],
                    "current_intent": [],
                    "current_intent_key": [],
                    "konu_id": "",
                    "true_flags": [],
                    "available_ordered_list": [],
                    "text": ""
                }]
            }

        try:
            detected_lang = detect(q)
            print("Tespit edilen dil:", detected_lang)
            if detected_lang not in ['tr', 'en']:
                print(f"[UYARI] Desteklenmeyen dil tespit edildi: '{detected_lang}'. Sorgu reddedildi.")
                return {
                    "results": [{
                        "rank": 1, "collection": ui_class, "score": 0.0,
                        "user_intent": [], "current_intent": [], "current_intent_key": [],
                        "konu_id": "", "true_flags": [],
                        "available_ordered_list": [],
                        "text": ""
                    }]
                }
        except LangDetectException:
            print("[UYARI] Anlamsız sorgu, dil tespit edilemedi. Sorgu reddedildi.")
            return {
                "results": [{
                    "rank": 1, "collection": ui_class, "score": 0.0,
                    "user_intent": [], "current_intent": [], "current_intent_key": [],
                    "konu_id": "", "true_flags": [],
                    "available_ordered_list": [],
                    "text": ""
                }]
            }

        print(f"[DEBUG] Query: '{q}' | Ders: '{d_adi}' | Class: '{ui_class}'")

        hits, user_intents = search_in_collections(q, ui_class, d_adi)

        if not hits:
            print("[DEBUG] Sonuç bulunamadı.")
            total_time = time.time() - start_time
            print(f"[DEBUG] ✓ İstek tamamlandı: {total_time:.2f} saniye")
            return {
                "results": [{
                    "rank": 1, "collection": None, "score": 0.0,
                    "user_intent": user_intents, "current_intent": [], "current_intent_key": [],
                    "konu_id": "", "true_flags": [],
                    "available_ordered_list": [],
                    "text": ""
                }]
            }

        response_results = []
        for i, r in enumerate(hits, 1):
            current_intents = r.get('intent', [])
            available_intents = r.get('available_intents', [])
            prioritized = [intent for intent in current_intents if intent in available_intents]
            remaining = [intent for intent in available_intents if intent not in prioritized]
            ordered_list = prioritized + remaining

            url = r.get("url")
            konu_id = "" if (url and str(url).strip()) else r.get("konu_id", "")

            response_results.append({
                "rank": i,
                "collection": r['collection'],
                "score": r['score'],
                "user_intent": user_intents,
                "current_intent": r['intent'],
                "current_intent_key": r['intent_key'],
                "konu_id": konu_id,
                "true_flags": r.get('true_flags', []),
                "available_ordered_list": ordered_list,
                "text": r.get("text", ""),
                "url": url,
                "url_payload": r.get("url_payload", "")
            })

        total_time = time.time() - start_time
        print(f"[DEBUG] ✓ İstek tamamlandı: {total_time:.2f} saniye")
        return {"results": response_results}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Hata oluştu: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server hatası: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_V3:app", host="0.0.0.0", port=5000, reload=False)