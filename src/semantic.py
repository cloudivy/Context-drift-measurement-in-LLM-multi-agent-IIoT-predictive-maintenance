"""
semantic.py — embedding-based semantic similarity for SFS-4 and IGS-3.

Primary : sentence-transformers (all-MiniLM-L6-v2). Captures meaning; correct
          paraphrases score high even with no shared words. No calibration factor.
Fallback: TF-IDF bigram cosine (offline). Lexical only; ×4.5 calibration applied.

get_backend() reports which is active so the paper states it accurately.
IMPORTANT: only describe SFS-4/IGS-3 as 'embedding-based' if get_backend()=='embedding'.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from config import EMBED_MODEL

_MODEL = None
_BACKEND = None


def _load():
    global _MODEL, _BACKEND
    if _MODEL is not None or _BACKEND == "tfidf":
        return
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBED_MODEL)
        _BACKEND = "embedding"
    except Exception:
        _BACKEND = "tfidf"


def get_backend() -> str:
    if _BACKEND is None:
        _load()
    return _BACKEND


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-9 or nb < 1e-9 else float(np.dot(a, b) / (na * nb))


def _tfidf(a, b):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not a.strip() or not b.strip():
        return 0.0
    try:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        m = v.fit_transform([a, b])
        return float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except Exception:
        return 0.0


def similarity(text_a: str, text_b: str) -> float:
    """Return semantic similarity in [0,1]."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    if get_backend() == "embedding":
        emb = _MODEL.encode([text_a, text_b], convert_to_numpy=True, show_progress_bar=False)
        return round(max(0.0, min(1.0, _cos(emb[0], emb[1]))), 4)
    return round(min(1.0, _tfidf(text_a, text_b) * 4.5), 4)


if __name__ == "__main__":
    print("backend:", get_backend())
    kb = "s3 HPC outlet temperature normal range 1585 to 1592 R NASA TM-2007-215026"
    print("grounded   :", similarity(kb, "s3=1604 exceeds KB max 1592 per NASA TM-2007-215026"))
    print("paraphrased :", similarity(kb, "compressor outlet temperature is above its certified limit"))
    print("vague      :", similarity(kb, "the engine seems to be running warm"))
