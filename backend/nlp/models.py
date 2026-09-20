"""NLP model loading and caching module."""

import spacy
from sentence_transformers import SentenceTransformer

_nlp_model = None
_sentence_model = None


def get_nlp_model():
    """
    Load and cache the spaCy English NLP model.
    Uses en_core_web_sm for fast processing.
    """
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            print("[DL] Downloading spaCy model 'en_core_web_sm'...")
            spacy.cli.download("en_core_web_sm")
            _nlp_model = spacy.load("en_core_web_sm")
        print("[OK] spaCy model loaded")
    return _nlp_model


def get_sentence_model():
    """
    Load and cache the Sentence Transformer model.
    Uses all-MiniLM-L6-v2 for fast semantic similarity.
    """
    global _sentence_model
    if _sentence_model is None:
        print("[DL] Loading Sentence Transformer model...")
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[OK] Sentence Transformer model loaded")
    return _sentence_model


def preload_models():
    """Pre-load all NLP models at startup."""
    get_nlp_model()
    get_sentence_model()
    print("[OK] All NLP models ready")
