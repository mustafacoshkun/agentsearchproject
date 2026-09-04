# -*- coding: utf-8 -*-
"""
FastAPI ile Tam Asenkron Eğitim İçeriği Arama Sistemi
Gerçek async/await ile paralel koleksiyon araması
"""

import re
import time
import asyncio
import hashlib
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache
from pathlib import Path

import numpy as np
import torch
import aiohttp
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import AgglomerativeClustering
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


# ================== CONFIGURATION ==================
class Config:
    QDRANT_URL = "http://localhost:6333"
    EMB_MODEL = "ytu-ce-cosmos/turkish-e5-large"

    # Connection pooling
    QDRANT_TIMEOUT = 30
    QDRANT_PREFER_GRPC = False
    MAX_WORKERS = 4

    # Cache settings
    EMBEDDING_CACHE_SIZE = 1000  # Query embedding cache
    RESULT_CACHE_SIZE = 500  # Search result cache
    RESULT_CACHE_TTL = 300  # 5 dakika TTL (saniye)

    # Logging
    LOG_FILE = "log.txt"  # Log dosyası
    LOG_SEPARATOR = "#" * 129  # Log ayırıcı

    # 🆕 LLM settings — H100/vLLM (KANKA, ana model), Ollama/CPU (yedek plan)
    VLLM_URL = "http://10.106.250.94:8000/v1/chat/completions"
    VLLM_MODEL = "./local_model"
    VLLM_TIMEOUT = 8  # H100 bazen kapalı olabiliyor; hızlı vazgeçip Ollama'ya düş
    VLLM_SYSTEM_PROMPT = (
        "Sen MEBİ platformunun yapay zekâlı eğitim asistanı KANKA'sın. "
        "Amacın öğrencilere okul derslerinde, LGS ve YKS hazırlıklarında samimi ve insancıl bir dille "
        "rehberlik etmektir. Asla argo, küfür kullanmazsın. Öğrenci kışkırtıcı bir şey yazarsa tartışmaya "
        "girmez, onu sakinleştirip nazikçe ders çalışmaya yönlendirirsin. Sana bulunan içerikler ayrıca "
        "ekranda gösterileceği için, cevabında bu içerikleri tekrar listeleme; sadece kısa, sıcak ve doğal "
        "bir yönlendirme cümlesi kur."
    )

    OLLAMA_URL = "http://localhost:11434/api/chat"
    OLLAMA_MODEL = "egitim-gemma"

    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 150  # num_predict
    LLM_TIMEOUT = 60  # saniye (Ollama/CPU için — daha yavaş olabilir)
    LLM_CONTEXT_TOP_N = 3  # Kaç arama sonucu bağlam olarak verilecek

    # Scoring thresholds
    SCORE_GAP = 0.04
    INTENT_MIN = 0.65
    CLUSTER_THRESH = 0.1

    # Search limits
    TOP_K = 10
    MAX_RESULTS = 5
    MIN_SCORE = 0.58

    # Hybrid weights
    FRAGMENT_WEIGHT = 0.65
    SENTENCE_WEIGHT = 0.35

    # Class to collections mapping
    CLASS_COLLECTIONS = {
        "5Sinif": ["5Sinif"],
        "6Sinif": ["5Sinif", "6Sinif"],
        "9Sinif": ["9Sinif"],
        "10Sinif": ["9Sinif", "10Sinif"],
        "YKS": ["YKS"],
        "LGS": ["LGS"],
    }


# ================== INTENT DEFINITIONS ==================
class IntentConfig:
    INTENTS = [
        "Video", "Konu Özeti", "Hazır Bulunuşluk Testi",
        "Değerlendirme Testi", "Soru Çöz", "Oynayarak Öğren",
        "Sesli Özet", "Tarama Testi", "Çıkmış Sorular",
        "Etkinliklerle Öğren", "Bağlam Temelli Sorular", "3B Model"
    ]

    META_KEYS = {
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

    ALIASES = {
        "Video": ["video", "ders videosu", "videolu içerik"],
        "Konu Özeti": ["konu özeti", "konunun özeti", "özet"],
        "Hazır Bulunuşluk Testi": ["hazır bulunuşluk", "ön test"],
        "Değerlendirme Testi": ["değerlendirme testi", "ünite testi"],
        "Soru Çöz": ["soru çöz", "soru çözümü", "çözümlü sorular"],
        "Oynayarak Öğren": ["oynayarak öğren", "eğitici oyun"],
        "Sesli Özet": ["sesli özet", "audio özet"],
        "Tarama Testi": ["tarama testi", "genel tarama"],
        "Çıkmış Sorular": ["çıkmış sorular", "ÖSYM soruları"],
        "Etkinliklerle Öğren": ["etkinliklerle öğren", "etkinlik"],
        "Bağlam Temelli Sorular": ["bağlam temelli sorular"],
        "3B Model": ["3B model", "3D model"],
    }

    REGEX = {
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
            r"\bvideolu\b",
            r"\bgörüntülü\b",
            r"\b(?:konu|ders)\s*video(?:su)?\b",
            r"\bvideo\s*(?:izle(?:mek)?|aç(?:mak)?|oynat(?:mak)?|göster(?:mek)?)\b",
            r"\b(video|vidyo)(?:su|yu|yü|ya|da|de|dan|den|lar[ıi])?\b",
            r"\b(?:konu|ders)\s*(?:video|vidyo)(?:su)?\b",
            r"\b(video|vidyo)\s*(?:izle(?:mek)?|aç(?:mak)?|oynat(?:mak)?|göster(?:mek)?|istiyorum)\b",
        ],
        "Konu Özeti": [
            r"^(?!.*\b(sesli|audio|video|izle|dinle|aç|oynat|göster)\b).*?\b(konu|ünite|ders)\s*özet(?:i|ini|ler|leri)?\b",
            r"^(?!.*\b(sesli|audio|video|izle|dinle|aç|oynat|göster)\b).*?\b(konu|ünite|ders).{0,20}\bözet\s*(?:ver|çıkar|yap|geç)\b",
            r"\b(?<!sesli\s|audio\s)(?:konu|ünite|ders)\s*özet(?:i|ini|ler|leri)?\b",
            r"\bözet\s*(?:istiyorum|ver|çıkar|yap|geç)?\b",
            r"\b(?<!sesli\s|audio\s)özet\s*(?:istiyorum|ver|çıkar|yap|geç)?\b",
        ],
        "Hazır Bulunuşluk Testi": [
            r"\bhazır\s*bulunuşluk\b",
            r"\bhazir\s*bulunusluk\b",
            r"\bön\s*test\b",
            r"\bbaşlangıç\s*test(?:i|leri)?\b",
            r"\bseviye\s*belirleme\b",
            r"\bdüzey\s*belirleme\b",
            r"\bdiagnostik(?:\s*test)?\b",
            r"\bhazır\s*bulunuşluk\s*test(?:i|leri)?\b",
            r"\bhazir\s*bulunusluk\s*test(?:i|leri)?\b",
        ],
        "Değerlendirme Testi": [
            r"\bdeğerlendirme\s*test(?:i|leri)?\b",
            r"\bdegerlendirme\s*test(?:i|leri)?\b",
            r"\bünite\s*sonu\s*test(?:i|leri)?\b",
            r"\bson\s*test\b",
            r"\bkapanış\s*test(?:i|leri)?\b",
        ],
        "Soru Çöz": [
            r"\bsoru\s*çöz(?:üm|mek|ümü)?\b",
            r"\bçözümlü\s*sorular?\b",
            r"\btest\s*çöz(?:mek)?\b",
            r"\bdeneme\s*çöz(?:mek)?\b",
            r"\bproblem\s*çöz(?:mek)?\b",
            r"\balıştırma(?:lar)?\b",
            r"\begzersiz(?:ler)?\b",
        ],
        "Oynayarak Öğren": [
            r"\boynayarak\s*öğren\b",
            r"\b(eğlenerek|zevkli)\s*öğren(?:me)?\b",
            r"\boyun\s*tabanlı\b",
            r"\binteraktif\b",
            r"\betkileşim(?:li)?\b",
            r"\boynayarak\s*öğren(?:me|meyi|meye)?\b",
            r"\boyun(?:la)?\s*öğren(?:me)?\b",
            r"\beğitici\s*oyun\b",
            r"\binteraktif\s*etkinlik\b",
            r"\betkileşimli\s*etkinlik\b",
        ],
        "Sesli Özet": [
            r"\bsesli\s*özet\b",
            r"\bsesli\s*anlat(?:ım|im)\b",
            r"\bözet\s*dinle\b",
            r"\bkonuyu\s*dinle\b",
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
            r"\betkinlik(?:lerle|li)?\s*öğren\b",
            r"\betkinlik(?:ler)?\b",
            r"\bçalışma\s*(kağıdı|sayfas[ıi])\b",
            r"\bdeney(?:ler)?\b",
            r"\bproje(?:ler)?\b",
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
            r"\bgrafik\s*yorum(?:lama|u)\b",
        ],
        "3B Model": [
            r"\b(?:3b|3d|3\s*boyutlu|üç\s*boyutlu)\s*(model|görsel|nesne|simülasyon|animasyon)\b",
            r"\b3d\s*model(?:leme)?\b",
        ],
    }


# ================== LOGGER ==================
class SearchLogger:
    """Asenkron loglama için utility sınıfı"""

    def __init__(self, log_file: str = Config.LOG_FILE):
        self.log_file = Path(log_file)
        self.executor = ThreadPoolExecutor(max_workers=1)  # Tek thread log için

        # Log dosyası yoksa oluştur
        if not self.log_file.exists():
            self.log_file.touch()
            print(f"[LOGGER] ✅ Log file created: {self.log_file.absolute()}")
        else:
            print(f"[LOGGER] ✅ Using existing log file: {self.log_file.absolute()}")

    def _write_log_sync(self, log_entry: dict):
        """Senkron log yazma (thread pool'da çalışır)"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(Config.LOG_SEPARATOR + "\n")
                f.write(f"timestamp={log_entry['timestamp']}\n")
                f.write(f"response_time={log_entry['response_time']}\n")
                f.write(f"cache_hit={log_entry['cache_hit']}\n")
                f.write(f"ders={log_entry['ders_adi']} \n")  # Sonda boşluk
                f.write(f"class={log_entry['class']}\n")
                f.write(f"query={log_entry['query']}\n")
                f.write(f"modelanswer={log_entry.get('modelanswer', '')}\n")
                f.write(f"url={log_entry['url']}\n")
                f.write(f"konu_id={log_entry['konu_id']}\n")
                f.write("yanit=\n")

                # Tüm yanıtları yaz
                for i, item in enumerate(log_entry['results'], 1):
                    text = item.get('text', '').replace('\n', ' ').strip()
                    kid = item.get('konu_id', '').strip()
                    url = item.get('url', '').strip()

                    # Konu_id ve URL varsa ekle
                    suffix = ""
                    if kid:
                        suffix += f" | konu_id={kid}"
                    if url:
                        suffix += f" | url={url}"

                    f.write(f"{i}. {text}{suffix}\n")

                f.write(Config.LOG_SEPARATOR + "\n\n\n")
        except Exception as e:
            print(f"[LOGGER] ❌ Error writing log: {e}")

    async def log_search(self, query: str, class_label: str, ders_adi: str,
                         results: List[dict], response_time: float, cache_hit: bool, modelanswer: str = ""):
        """Asenkron log yazma"""
        # İlk URL ve konu_id'yi al
        url_str = ""
        konu_id_str = ""
        if results:
            url_str = results[0].get('url', '')
            konu_id_str = results[0].get('konu_id', '')

        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'response_time': f"{response_time:.3f}s",
            'cache_hit': 'yes' if cache_hit else 'no',
            'ders_adi': ders_adi,
            'class': class_label,
            'query': query,
            'url': url_str,
            'konu_id': konu_id_str,
            'results': results,
            'modelanswer': modelanswer
        }

        # Thread pool'da asenkron yaz
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._write_log_sync, log_entry)

    def shutdown(self):
        """Executor'ı kapat"""
        self.executor.shutdown(wait=True)


BLACKLIST = [
    # Selamlaşma
    "merhaba", "mrb", "mrb.", "mrbk", "mrbler",
    "selam", "selamlar", "slm", "slm.", "slmlar",
    "selamün aleyküm", "selamun aleykum", "sea", "sa", "s.a",
    "hello", "hi", "hey", "heyy", "heyyy",
    "alo", "yo", "sup", "whats up", "what's up",
    "good morning", "good evening", "good night",
    "günaydın", "tünaydın", "iyi akşamlar", "iyi geceler",

    # Hal hatır sorma / small talk
    "nasılsın", "nasilsin", "nbr", "naber", "ne haber",
    "nabıyon", "nabıyorsun", "napıyorsun", "napıyosun", "napiyon",
    "how are you", "how r u", "wassup", "sup bro", "what's up bro",

    # Kısa cevaplar / reaksiyonlar
    "iyiyim", "ben iyiyim", "çok iyiyim", "idare eder", "fena değil",
    "iyi sen", "sen nasılsın", "iyi valla", "şükür",
    "i'm fine", "fine thanks", "good", "not bad",

    # Teşekkür & veda
    "teşekkürler", "tesekkurler", "sağol", "sagol", "eyvallah", "çok sağ ol",
    "thank you", "thanks", "thx", "ty",
    "bye", "goodbye", "see you", "see ya", "take care",
    "görüşürüz", "gorusuruz", "hoşça kal", "kendine iyi bak", "bay bay",

    # Gündelik kısa ifadeler / tepkiler
    "ok", "okey", "oki", "tamam", "aynen", "aynn",
    "lol", "haha", "hahaha", "ahah", "xd", "xD", "😂", "🙂", "👍",
    "hmm", "hımm", "hım",
]


# ================== TEXT UTILITIES ==================
class TextUtils:
    # Türkçe + İngilizce Latin alfabesi ve yaygın noktalama/özel karakterler.
    # langdetect gibi istatistiksel bir sınıflandırıcı yerine kullanılıyor çünkü kısa/argo
    # metinlerde ("naber", "kanka" gibi) langdetect sık sık yanlış dil tahmini yapıyordu.
    _ALLOWED_CHARS_PATTERN = re.compile(
        r'^[a-zA-ZçÇğĞıİöÖşŞüÜ0-9\s.,!?;:\'"%/\\\-_+()&@#*=\[\]{}<>]*$'
    )

    @staticmethod
    def is_turkish_or_english_charset(text: str) -> bool:
        """Sorgu sadece Türkçe/İngilizce Latin karakter setinden mi oluşuyor?
        Kiril, Arapça, Çince, Yunanca vb. farklı bir alfabe varsa False döner."""
        return bool(TextUtils._ALLOWED_CHARS_PATTERN.match(text))

    @staticmethod
    def normalize(text: str) -> str:
        text = text.replace("İ", "I").replace("I", "ı").lower()
        return re.sub(r"\s+", " ", text).strip()

    # Selamlaşma/hitap kelimeleri — bunlar tek başına "gerçek içerik" sayılmaz,
    # sohbet tespitinde göz ardı edilir (örn. "merhaba kanka naber" yine saf sohbet sayılır).
    _SMALLTALK_STOPWORDS = {"kanka", "abi", "abla", "hocam", "dostum", "canım", "ya", "be"}

    @staticmethod
    def is_blacklisted(query: str) -> bool:
        """Sorgunun TAMAMI (hitap kelimeleri hariç) saf sohbet/selamlaşma ifadelerinden mi oluşuyor?
        Karışık sorgular ("merhaba, kesirler konusunu çalışmak istiyorum" gibi) burada YAKALANMAZ —
        böyle sorgular normal arama akışına girer; gerçek bir eşleşme yoksa LLM zaten
        "hangi konuyla ilgiliydi?" tarzı doğal bir yönlendirme yapacak şekilde eğitildi."""
        normalized = TextUtils.normalize(query)
        if not normalized:
            return False

        blacklist_normalized = {TextUtils.normalize(term) for term in BLACKLIST}

        # Sorgunun tamamı tek bir blacklist ifadesiyle birebir eşleşiyorsa (örn. "sen nasılsın")
        if normalized in blacklist_normalized:
            return True

        # Aksi halde: hitap kelimeleri hariç HER kelime, blacklist'teki tek kelimelik
        # ifadelerden biri olmalı — yoksa sorguda gerçek içerik talebi var demektir.
        single_word_blacklist = {b for b in blacklist_normalized if ' ' not in b}
        words = [w for w in re.split(r'\s+', normalized) if w]
        meaningful_words = [w for w in words if w not in TextUtils._SMALLTALK_STOPWORDS]

        if not meaningful_words:
            return True  # sadece hitap kelimelerinden oluşuyor (örn. "kanka")

        return all(w in single_word_blacklist for w in meaningful_words)

    @staticmethod
    def strip_smalltalk(query: str) -> str:
        """Cosine similarity/embedding araması için sorgudan selamlaşma/sohbet ve hitap
        kelimelerini temizler (örn. "merhaba, kesirler konusu" → "kesirler konusu").
        Embedding tabanlı arama, LLM'in aksine gürültü kelimelerine karşı çok daha
        hassastır — kısa sorgularda "merhaba" gibi kelimeler asıl konunun vektörünü
        kaydırabilir. Temizlenmiş metin boş kalırsa orijinal sorgu döner (arama tamamen
        boşa düşmesin diye)."""
        blacklist_normalized = {TextUtils.normalize(term) for term in BLACKLIST}
        single_word_blacklist = {b for b in blacklist_normalized if ' ' not in b}
        skip_words = single_word_blacklist | TextUtils._SMALLTALK_STOPWORDS

        tokens = re.split(r'(\s+)', query)  # boşlukları koru, orijinal formatı bozma
        cleaned_tokens = []
        for tok in tokens:
            if tok.strip() == "":
                cleaned_tokens.append(tok)
                continue
            core = re.sub(r'^[^\wçÇğĞıİöÖşŞüÜ]+|[^\wçÇğĞıİöÖşŞüÜ]+$', '', tok)
            if TextUtils.normalize(core) in skip_words:
                continue
            cleaned_tokens.append(tok)

        cleaned = re.sub(r"\s+", " ", "".join(cleaned_tokens)).strip()
        return cleaned if cleaned else query.strip()

    @staticmethod
    def combine_ders_query(ders_adi: str, text: str) -> str:
        if not ders_adi:
            return text.strip()

        ders_norm = TextUtils.normalize(ders_adi)
        text_norm = TextUtils.normalize(text)

        if text_norm.startswith(ders_norm):
            return text.strip()
        return f"{ders_adi.strip()} {text.strip()}"


# ================== EMBEDDING ENGINE ==================
class EmbeddingEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMB_MODEL,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self._label_vectors = self._precompute_label_vectors()

        # 🆕 Query embedding cache (LRU)
        self._embedding_cache = {}
        print(f"[CACHE] Embedding cache initialized (max: {Config.EMBEDDING_CACHE_SIZE})")

    def _precompute_label_vectors(self) -> Dict[str, np.ndarray]:
        vectors = {}
        for label, aliases in IntentConfig.ALIASES.items():
            vecs = [self.embed_uncached(alias) for alias in aliases]
            vectors[label] = np.stack(vecs).mean(axis=0)
        return vectors

    def embed_uncached(self, text: str) -> np.ndarray:
        """Embedding without cache (for label precomputation)."""
        normalized = TextUtils.normalize(text)
        return np.array(
            self.embeddings.embed_query(f"query: {normalized}"),
            dtype=np.float32
        )

    def embed(self, text: str) -> np.ndarray:
        """🆕 Cached embedding - aynı sorgu için cache kullanır."""
        normalized = TextUtils.normalize(text)
        cache_key = normalized

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Cache miss - hesapla
        vector = np.array(
            self.embeddings.embed_query(f"query: {normalized}"),
            dtype=np.float32
        )

        # LRU mantığı - max size aşarsa en eskiyi sil
        if len(self._embedding_cache) >= Config.EMBEDDING_CACHE_SIZE:
            # İlk eklenen key'i sil (basit FIFO)
            first_key = next(iter(self._embedding_cache))
            del self._embedding_cache[first_key]

        self._embedding_cache[cache_key] = vector
        return vector

    def get_label_vector(self, label: str) -> np.ndarray:
        return self._label_vectors[label]

    def get_cache_stats(self) -> dict:
        """🆕 Cache istatistikleri."""
        return {
            "embedding_cache_size": len(self._embedding_cache),
            "embedding_cache_max": Config.EMBEDDING_CACHE_SIZE,
            "label_vectors_cached": len(self._label_vectors)
        }


# ================== INTENT DETECTOR ==================
class IntentDetector:
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine

    def extract_fragments(self, query: str, intent: str) -> List[str]:
        normalized = TextUtils.normalize(query)
        patterns = IntentConfig.REGEX.get(intent, [])

        fragments = []
        for pattern in patterns:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                fragment = match.group(0).strip()
                if fragment and fragment not in fragments:
                    fragments.append(fragment)
        return fragments

    def score_intent(self, query: str, intent: str) -> float:
        query_vec = self.embedding_engine.embed(query)
        label_vec = self.embedding_engine.get_label_vector(intent)

        sentence_sim = float(np.dot(query_vec, label_vec))

        fragments = self.extract_fragments(query, intent)
        if not fragments:
            return sentence_sim

        fragment_sims = [
            float(np.dot(self.embedding_engine.embed(frag), label_vec))
            for frag in fragments
        ]
        best_fragment_sim = max(fragment_sims)

        return (Config.FRAGMENT_WEIGHT * best_fragment_sim +
                Config.SENTENCE_WEIGHT * sentence_sim)

    def detect_intents(self, query: str) -> List[str]:
        scores = {
            intent: self.score_intent(query, intent)
            for intent in IntentConfig.INTENTS
        }

        filtered = {k: v for k, v in scores.items() if v >= Config.INTENT_MIN}
        if not filtered:
            return []

        sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_items) > 1:
            gap = sorted_items[0][1] - sorted_items[1][1]
            if gap < Config.SCORE_GAP:
                return [sorted_items[0][0], sorted_items[1][0]]

        if len(sorted_items) == 1:
            return [sorted_items[0][0]]

        labels = [item[0] for item in sorted_items]
        values = np.array([filtered[k] for k in labels]).reshape(-1, 1)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=Config.CLUSTER_THRESH,
            linkage="ward"
        )
        clustering.fit(values)

        clusters = {}
        for idx, label in enumerate(labels):
            cluster_id = int(clustering.labels_[idx])
            clusters.setdefault(cluster_id, []).append((label, float(values[idx, 0])))

        top_cluster = max(clusters.items(), key=lambda x: max(s for _, s in x[1]))
        return [label for label, _ in top_cluster[1]]


# ================== ASYNC SEARCH ENGINE ==================
class AsyncSearchEngine:
    def __init__(self, intent_detector: IntentDetector):
        self.intent_detector = intent_detector

        # Qdrant client - TEK SEFERLIK OLUŞTURULUYOR ✅
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            timeout=Config.QDRANT_TIMEOUT,
            prefer_grpc=Config.QDRANT_PREFER_GRPC
        )

        # Caches
        self._store_cache = {}  # VectorStore cache
        self._valid_collections = set()  # Collection validation cache

        # 🆕 Result cache (TTL Cache - 5 dakika)
        self._result_cache = TTLCache(
            maxsize=Config.RESULT_CACHE_SIZE,
            ttl=Config.RESULT_CACHE_TTL
        )
        self._cache_hits = 0
        self._cache_misses = 0

        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

        print(f"[INIT] ✅ Qdrant Client ID: {id(self.client)} (singleton)")
        print(f"[CACHE] Result cache: {Config.RESULT_CACHE_SIZE} entries, TTL={Config.RESULT_CACHE_TTL}s")

    def _get_store(self, collection: str) -> QdrantVectorStore:
        """✅ Cached VectorStore getter."""
        if collection not in self._store_cache:
            self._store_cache[collection] = QdrantVectorStore(
                client=self.client,
                collection_name=collection,
                embedding=self.intent_detector.embedding_engine.embeddings
            )
            print(f"[CACHE] VectorStore '{collection}' cached")
        return self._store_cache[collection]

    def _make_cache_key(self, query: str, class_label: str, ders_adi: Optional[str]) -> str:
        """🆕 Create cache key for search results."""
        key_str = f"{class_label}:{query}:{ders_adi or ''}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_cache_stats(self) -> dict:
        """🆕 Cache istatistikleri."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "result_cache_size": len(self._result_cache),
            "result_cache_max": Config.RESULT_CACHE_SIZE,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "store_cache_size": len(self._store_cache),
            "valid_collections": len(self._valid_collections),
            **self.intent_detector.embedding_engine.get_cache_stats()
        }

    def _search_collection_sync(self, collection: str, query: str,
                                intent_filter: Optional[str] = None) -> List[dict]:
        """Synchronous collection search (runs in thread pool)."""
        if collection not in self._valid_collections:
            try:
                self.client.get_collection(collection)
                self._valid_collections.add(collection)
            except Exception:
                return []

        store = self._get_store(collection)
        search_text = f"query: {query}"

        filter_obj = None
        if intent_filter:
            meta_key = IntentConfig.META_KEYS.get(intent_filter)
            if meta_key:
                filter_obj = qm.Filter(must=[
                    qm.FieldCondition(
                        key=f"metadata.{meta_key}",
                        match=qm.MatchValue(value=True)
                    )
                ])

        try:
            hits = store.similarity_search_with_score(
                search_text,
                k=Config.TOP_K,
                filter=filter_obj
            )

            results = []
            for doc, score in hits:
                meta = doc.metadata or {}
                true_flags = [k for k, v in meta.items() if isinstance(v, bool) and v]

                results.append({
                    "collection": collection,
                    "score": float(score),
                    "intent": intent_filter,
                    "konu_id": meta.get("konu_id", ""),
                    "true_flags": true_flags,
                    "text": (meta.get("text_payload") or doc.page_content or "").strip(),
                    "url": meta.get("url", ""),
                })
            return results
        except Exception as e:
            print(f"[Warning] Search error in {collection}: {e}")
            return []

    async def _search_collection_async(self, collection: str, query: str,
                                       intent_filter: Optional[str] = None) -> List[dict]:
        """Async wrapper - runs sync search in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._search_collection_sync,
            collection,
            query,
            intent_filter
        )

    def _group_results(self, results: List[dict]) -> List[dict]:
        """Group and deduplicate results."""

        def group_key(r):
            if r.get("konu_id"):
                return f"k::{r['konu_id']}"
            if r.get("url"):
                return f"u::{r['url']}"
            text = TextUtils.normalize(r.get("text", ""))[:80]
            return f"t::{r['collection']}::{text}"

        grouped = {}
        for r in results:
            key = group_key(r)

            if key not in grouped:
                grouped[key] = {**r, "intent": [r["intent"]] if r.get("intent") else []}
            else:
                prev = grouped[key]
                if r["score"] > prev["score"]:
                    prev.update({
                        "score": r["score"],
                        "url": r.get("url") or prev.get("url"),
                        "konu_id": r.get("konu_id") or prev.get("konu_id"),
                    })
                if r.get("intent") and r["intent"] not in prev["intent"]:
                    prev["intent"].append(r["intent"])

        final = [r for r in grouped.values() if r["score"] >= Config.MIN_SCORE]
        final.sort(key=lambda x: x["score"], reverse=True)
        return final[:Config.MAX_RESULTS]

    async def search_async(self, query: str, class_label: str,
                           ders_adi: Optional[str] = None) -> Tuple[List[dict], List[str]]:
        """
        🚀 GERÇEK ASYNC ARAMA + 🆕 RESULT CACHE
        """
        start_time = time.time()

        # 🆕 Cache kontrolü
        cache_key = self._make_cache_key(query, class_label, ders_adi)

        if cache_key in self._result_cache:
            self._cache_hits += 1
            cached_data = self._result_cache[cache_key]
            cache_time = time.time() - start_time
            print(f"[CACHE] ✅ HIT! Returned in {cache_time:.3f}s (saved ~{cached_data['original_time']:.2f}s)")
            return cached_data['results'], cached_data['intents']

        self._cache_misses += 1
        print(f"[CACHE] ❌ MISS - Executing search...")

        # Intent detection (CPU-bound, run in executor)
        loop = asyncio.get_event_loop()
        intents = await loop.run_in_executor(
            self.executor,
            self.intent_detector.detect_intents,
            query
        )

        intent_time = time.time() - start_time
        print(f"[ASYNC] Intent detection: {intent_time:.3f}s → {intents}")

        collections = Config.CLASS_COLLECTIONS.get(class_label, [])
        if not collections:
            raise ValueError(f"No collections for class: {class_label}")

        augmented_query = TextUtils.combine_ders_query(ders_adi or "", query)

        # Prepare async tasks - HEPSİ AYNI ANDA ÇALIŞACAK! 🚀
        tasks = []
        for collection in collections:
            # URL-first search
            tasks.append(
                self._search_collection_async(collection, query or augmented_query, None)
            )

            # Intent-filtered searches
            if intents:
                for intent in intents:
                    tasks.append(
                        self._search_collection_async(collection, augmented_query, intent)
                    )

        print(f"[ASYNC] 🚀 Launching {len(tasks)} concurrent tasks across {len(collections)} collections")
        search_start = time.time()

        # Execute ALL searches concurrently
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        search_time = time.time() - search_start
        print(f"[ASYNC] ✅ All searches completed in {search_time:.3f}s")

        all_results = []
        for result in results_lists:
            if isinstance(result, Exception):
                print(f"[Warning] Task error: {result}")
            elif isinstance(result, list):
                all_results.extend(result)

        # Group results (CPU-bound, run in executor)
        grouped_results = await loop.run_in_executor(
            self.executor,
            self._group_results,
            all_results
        )

        total_time = time.time() - start_time
        print(f"[ASYNC] 📊 Total request time: {total_time:.3f}s")

        # 🆕 Cache'e kaydet
        self._result_cache[cache_key] = {
            'results': grouped_results,
            'intents': intents,
            'original_time': total_time,
            'cached_at': time.time()
        }
        print(f"[CACHE] 💾 Cached result for {Config.RESULT_CACHE_TTL}s")

        return grouped_results, intents

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# ================== LLM ENGINE (H100/vLLM birincil, Ollama/CPU yedek) ==================
class LLMEngine:
    """
    İki motorlu LLM katmanı:
      1) ANA PLAN: H100 üzerindeki vLLM sunucusu (KANKA, hızlı GPU inference).
      2) B PLANI: H100 kapalıysa / erişilemezse / zaman aşımına uğrarsa, aynı istek
         yerel CPU'daki Ollama (egitim-gemma) motoruna otomatik düşer.

    vLLM çağrısı OpenAI-uyumlu /v1/chat/completions formatını kullanır ve system
    promptu HER İSTEKTE açıkça gönderir (model eğitilirken system prompt gömülmedi).
    Ollama çağrısı ise sadece user mesajı gönderir; system prompt Modelfile içinde
    SYSTEM olarak zaten gömülüdür.

    Girdi formatı (her iki motor için de aynı):
        Soru: <kullanıcı sorgusu>

        Bulunan İçerikler:
        - <içerik 1>
        - <içerik 2>

    Bağlam bulunamazsa sadece "Soru: ..." gönderilir.
    """

    def __init__(self):
        self.vllm_url = Config.VLLM_URL
        self.vllm_model = Config.VLLM_MODEL
        self.ollama_url = Config.OLLAMA_URL
        self.ollama_model = Config.OLLAMA_MODEL
        self._session: Optional[aiohttp.ClientSession] = None
        print(f"[LLM] ✅ Hibrit motor hazır → ana=vLLM/H100 ({self.vllm_url}), "
              f"yedek=Ollama/CPU ({self.ollama_model})")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy singleton aiohttp session (genel LLM_TIMEOUT ile; vLLM çağrısı kendi kısa timeout'unu kullanır)."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=Config.LLM_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    @staticmethod
    def build_prompt(query: str, results: List[dict]) -> str:
        """Sorgu + arama sonuçlarını modelin eğitildiği formata çevirir."""
        prompt = f"Soru: {query.strip()}"

        contexts = []
        for r in results[:Config.LLM_CONTEXT_TOP_N]:
            text = (r.get("text") or "").replace("\n", " ").strip()
            if text and text not in contexts:
                contexts.append(text)

        if contexts:
            prompt += "\n\nBulunan İçerikler:\n"
            prompt += "\n".join(f"- {c}" for c in contexts)

        return prompt

    async def _call_vllm(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """H100/vLLM'e OpenAI-uyumlu istek atar. Başarısız olursa exception fırlatır (fallback tetiklenir)."""
        payload = {
            "model": self.vllm_model,
            "messages": [
                {"role": "system", "content": Config.VLLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": Config.LLM_MAX_TOKENS,
            "temperature": Config.LLM_TEMPERATURE,
        }
        timeout = aiohttp.ClientTimeout(total=Config.VLLM_TIMEOUT)
        async with session.post(self.vllm_url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                raise RuntimeError(f"vLLM HTTP {resp.status}: {err_text[:200]}")
            data = await resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()

    async def _call_ollama(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """Yedek plan: yerel CPU'daki Ollama motoruna istek atar."""
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": Config.LLM_TEMPERATURE,
                "num_predict": Config.LLM_MAX_TOKENS,
                "num_thread": 8,
            },
        }
        async with session.post(self.ollama_url, json=payload) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                raise RuntimeError(f"Ollama HTTP {resp.status}: {err_text[:200]}")
            data = await resp.json()
            return (data.get("message", {}) or {}).get("content", "").strip()

    async def generate(self, query: str, results: List[dict]) -> dict:
        """
        Önce H100/vLLM'i dener; erişilemezse/timeout olursa otomatik olarak
        Ollama/CPU'ya düşer. Dönen dict: {"answer", "prompt", "llm_time", "error", "engine"}
        """
        prompt = self.build_prompt(query, results)
        start = time.time()
        session = await self._get_session()

        # 1) ANA PLAN: H100 / vLLM
        try:
            answer = await self._call_vllm(prompt, session)
            llm_time = time.time() - start
            print(f"[LLM] ✅ vLLM/H100 yanıtı {llm_time:.3f}s ({len(answer)} karakter)")
            return {"answer": answer, "prompt": prompt, "llm_time": llm_time,
                    "error": None, "engine": "vllm"}
        except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError) as e:
            print(f"[LLM] ⚠️ vLLM/H100'e erişilemedi ({type(e).__name__}: {e}) → Ollama/CPU'ya düşülüyor")

        # 2) B PLANI: Ollama / CPU
        try:
            answer = await self._call_ollama(prompt, session)
            llm_time = time.time() - start
            print(f"[LLM] ✅ Ollama/CPU yanıtı {llm_time:.3f}s ({len(answer)} karakter)")
            return {"answer": answer, "prompt": prompt, "llm_time": llm_time,
                    "error": None, "engine": "ollama"}
        except asyncio.TimeoutError:
            print(f"[LLM] ❌ Ollama/CPU timeout ({Config.LLM_TIMEOUT}s)")
            return {"answer": "", "prompt": prompt, "llm_time": time.time() - start,
                    "error": "timeout", "engine": None}
        except Exception as e:
            print(f"[LLM] ❌ {type(e).__name__}: {e}")
            return {"answer": "", "prompt": prompt, "llm_time": time.time() - start,
                    "error": str(e), "engine": None}

    async def shutdown(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ================== FASTAPI APP ==================
app = FastAPI(
    title="Async Education Search API",
    description="Tam asenkron eğitim içeriği arama sistemi",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances - TEK SEFER OLUŞTURULUR ✅
embedding_engine = None
intent_detector = None
search_engine = None
logger = None  # 🆕 Logger instance
llm_engine = None  # 🆕 Hibrit LLM instance (vLLM/H100 → Ollama/CPU)


@app.on_event("startup")
async def startup_event():
    """Server başlangıcında tek seferlik init"""
    global embedding_engine, intent_detector, search_engine, logger, llm_engine

    print("=" * 60)
    print("🚀 FastAPI Async Server Starting...")
    print("=" * 60)

    print("[INIT] Initializing logger...")
    logger = SearchLogger()

    print("[INIT] Initializing LLM engine (vLLM/H100 → Ollama/CPU fallback)...")
    llm_engine = LLMEngine()

    print("[INIT] Loading embedding model...")
    embedding_engine = EmbeddingEngine()

    print("[INIT] Initializing intent detector...")
    intent_detector = IntentDetector(embedding_engine)

    print("[INIT] Connecting to Qdrant...")
    search_engine = AsyncSearchEngine(intent_detector)

    print("=" * 60)
    print(f"✅ Server ready!")
    print(f"   Mode: ASYNC (True Concurrency)")
    print(f"   Workers: {Config.MAX_WORKERS}")
    print(f"   Qdrant: {Config.QDRANT_URL}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   Log File: {Path(Config.LOG_FILE).absolute()}")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    print("\n🛑 Server shutting down...")
    if search_engine and hasattr(search_engine, 'executor'):
        search_engine.executor.shutdown(wait=True)
    if logger:
        logger.shutdown()
    if llm_engine:
        await llm_engine.shutdown()
    print("✅ Cleanup completed")


@app.get("/")
async def health_check():
    """Health check + cache statistics"""
    cache_stats = search_engine.get_cache_stats() if search_engine else {}

    return {
        "status": "running",
        "mode": "async",
        "qdrant_client_id": id(search_engine.client) if search_engine else None,
        "timestamp": time.time(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cache_stats": cache_stats  # 🆕 Cache istatistikleri
    }


@app.get("/cache/clear")
async def clear_cache():
    """🆕 Cache temizleme endpoint'i"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    # Result cache temizle
    result_cache_size = len(search_engine._result_cache)
    search_engine._result_cache.clear()

    # Embedding cache temizle
    embedding_cache_size = len(search_engine.intent_detector.embedding_engine._embedding_cache)
    search_engine.intent_detector.embedding_engine._embedding_cache.clear()

    # İstatistikleri sıfırla
    search_engine._cache_hits = 0
    search_engine._cache_misses = 0

    return {
        "status": "cleared",
        "result_cache_cleared": result_cache_size,
        "embedding_cache_cleared": embedding_cache_size,
        "timestamp": time.time()
    }


@app.get("/logs/latest")
async def get_latest_logs(lines: int = Query(10, description="Son kaç log entry gösterilsin")):
    """🆕 Son logları görüntüleme"""
    try:
        log_file = Path(Config.LOG_FILE)
        if not log_file.exists():
            return {"error": "Log file not found", "logs": []}

        # Dosyayı oku
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Separator'lara göre böl
        entries = content.split(Config.LOG_SEPARATOR)
        # Boş olanları filtrele ve son N tanesini al
        entries = [e.strip() for e in entries if e.strip()][-lines:]

        return {
            "total_entries": len(entries),
            "logs": entries,
            "log_file": str(log_file.absolute())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")


@app.get("/logs/stats")
async def get_log_stats():
    """🆕 Log istatistikleri"""
    try:
        log_file = Path(Config.LOG_FILE)
        if not log_file.exists():
            return {"error": "Log file not found"}

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # İstatistikler
        total_entries = content.count(Config.LOG_SEPARATOR) // 2  # Her entry 2 separator
        cache_hits = content.count("cache_hit=yes")
        cache_misses = content.count("cache_hit=no")

        file_size_kb = log_file.stat().st_size / 1024

        return {
            "total_requests": total_entries,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate": f"{(cache_hits / total_entries * 100):.1f}%" if total_entries > 0 else "0%",
            "file_size_kb": f"{file_size_kb:.2f}",
            "log_file": str(log_file.absolute())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log stats: {str(e)}")


@app.get("/search")
async def search_endpoint(
        query: str = Query(..., description="Arama sorgusu"),
        class_param: str = Query(..., alias="class", description="Sınıf seviyesi (5, 6, 9, 10, YKS, LGS)"),
        ders_adi: Optional[str] = Query(None, description="Ders adı")
):
    """
    🔍 Asenkron arama endpoint'i

    Örnek: /search?query=fotosintez+videosu&class=6&ders_adi=Fen+Bilimleri
    """
    request_start = time.time()
    cache_hit = False

    try:
        # Query fallback
        if not query and ders_adi:
            query = ders_adi

        if not query:
            raise HTTPException(status_code=400, detail="'query' parameter required")

        # Normalize class
        class_map = {
            "5": "5Sinif", "6": "6Sinif", "9": "9Sinif",
            "10": "10Sinif", "YKS": "YKS", "LGS": "LGS"
        }
        ui_class = class_map.get(class_param, class_param)

        if ui_class not in Config.CLASS_COLLECTIONS:
            raise HTTPException(status_code=400, detail=f"Invalid class: {class_param}")

        print(f"\n{'=' * 60}")
        print(f"📥 NEW REQUEST")
        print(f"   Query: '{query}'")
        print(f"   Class: {ui_class}")
        print(f"   Ders: '{ders_adi or 'N/A'}'")
        print(f"{'=' * 60}")

        # Smalltalk / selamlaşma kontrolü (güvenlik değil, verimlilik amaçlı: saf sohbet
        # sorgularında Qdrant araması gereksiz — direkt LLM'e yönlendiriyoruz, o da bu
        # kategori için özel olarak eğitildiği için doğal ve çeşitli bir cevap üretiyor).
        loop = asyncio.get_event_loop()
        is_smalltalk = await loop.run_in_executor(
            None,
            TextUtils.is_blacklisted,
            query
        )

        if is_smalltalk:
            print("[INFO] 💬 Saf sohbet/selamlaşma sorgusu → arama atlanıp doğrudan LLM'e yönlendiriliyor")

            llm_result = await llm_engine.generate(query, [])

            response_time = time.time() - request_start

            await logger.log_search(
                query, ui_class, ders_adi or '',
                [], response_time, cache_hit, llm_result["answer"]
            )

            return {"modelanswer": llm_result["answer"], "subjectlist": []}

        # Language / alphabet check
        # Türkçe ve İngilizce'nin ikisi de Latin alfabesiyle yazılır. langdetect gibi istatistiksel
        # bir sınıflandırıcı kısa/argo metinlerde ("merhaba kanka naber nasılsın" gibi) sık sık
        # yanlış bir dil tahmin edip geçerli sorguları reddediyordu. Onun yerine, sorgunun sadece
        # Türkçe/İngilizce Latin karakter setinden oluşup oluşmadığına bakan basit ve güvenilir bir
        # kontrol kullanıyoruz — farklı bir alfabe (Kiril, Arapça, Çince, Yunanca vb.) varsa reddedilir.
        query_stripped = query.strip()
        if query_stripped and not TextUtils.is_turkish_or_english_charset(query_stripped):
            print(f"[SECURITY] ⚠️ Desteklenmeyen alfabe/karakter seti: '{query_stripped[:50]}'")

            response_time = time.time() - request_start

            # 🆕 Log empty result
            await logger.log_search(
                query, ui_class, ders_adi or '',
                [], response_time, cache_hit, ""
            )

            return {
                "modelanswer": "Ben seni çalışmak istediğin sayfaya yönlendirmek için varım. Lütfen çalışmak istediğin konuyu yaz.",
                "subjectlist": []}

        # 🚀 ASYNC SEARCH - Gerçek async paralel arama!
        # Cosine similarity/embedding araması gürültü kelimelerine (selamlaşma, hitap) LLM'den
        # çok daha hassastır — "merhaba, kesirler konusu" gibi karışık sorgularda "merhaba" asıl
        # konunun vektörünü kaydırabilir. Arama SADECE temizlenmiş metinle yapılır; LLM'e giden ve
        # loglanan sorgu ise orijinal haliyle kalır (LLM doğal dili zaten anlıyor).
        search_query = TextUtils.strip_smalltalk(query)
        if search_query != query:
            print(f"[SEARCH] 🧹 Temizlenmiş arama sorgusu: '{search_query}'")

        cache_key = search_engine._make_cache_key(search_query, ui_class, ders_adi)
        cache_hit = cache_key in search_engine._result_cache

        results, user_intents = await search_engine.search_async(
            search_query, ui_class, ders_adi
        )

        if not results:
            print("[RESULT] ℹ️ No results found → LLM without context")

            # Bağlam yok → model sadece "Soru: ..." alır
            llm_result = await llm_engine.generate(query, [])

            response_time = time.time() - request_start

            # Log empty result (Boş arama loglanırken artık empty_result dizisi yerine boş liste verilebilir)
            await logger.log_search(
                query, ui_class, ders_adi or '',
                [], response_time, cache_hit, llm_result["answer"]
            )

            # Sadeleştirilmiş JSON dönüşü (Boş sonuç)
            return {
                "modelanswer": llm_result["answer"],
                "subjectlist": []
            }

        # LLM: Sorgu + getirilen içerikler birleştirilip modele verilir
        llm_result = await llm_engine.generate(query, results)

        # İstenilen formatta subjectlist oluşturulması
        subjectlist = [
            {
                "konu_id": r.get("konu_id", ""),
                "konutext": r.get("text", "")
            }
            for r in results
        ]

        total_time = time.time() - request_start
        print(f"[RESULT] ✅ Returned {len(subjectlist)} results + LLM answer in {total_time:.3f}s")
        print(f"{'=' * 60}\n")

        # Asenkron loglama orijinal results ile devam etmeli ki loglar eksik kalmasın
        asyncio.create_task(
            logger.log_search(
                query, ui_class, ders_adi or '',
                results, total_time, cache_hit, llm_result["answer"]
            )
        )

        # Sadeleştirilmiş JSON dönüşü (Başarılı sonuç)
        return {
            "modelanswer": llm_result["answer"],
            "subjectlist": subjectlist
        }

    except Exception as e:
        error_time = time.time() - request_start
        print(f"[ERROR] ❌ {type(e).__name__}: {str(e)} (after {error_time:.3f}s)")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ================== RUN SERVER ==================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🎯 Starting Async Search Server with Cache")
    print("=" * 60)
    print(f"📦 Embedding Cache: {Config.EMBEDDING_CACHE_SIZE} queries")
    print(f"📦 Result Cache: {Config.RESULT_CACHE_SIZE} results (TTL: {Config.RESULT_CACHE_TTL}s)")
    print("=" * 60)

    # Production configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5002,
        log_level="info",
        access_log=True,
    )

    # Development: uvicorn server_V3:app --reload --port 5002