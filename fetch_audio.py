from __future__ import annotations
from difflib import SequenceMatcher
from config import STORAGE_PATH
from pydub import AudioSegment
import pysrt
import uuid
import os

# from pydub.utils import which

# AudioSegment.converter = which(
#     os.path.join("G:", "ffmpeg-8.0-full_build", "bin", "ffmpeg")
# )
# AudioSegment.ffprobe = which(
#     os.path.join("G:", "ffmpeg-8.0-full_build", "bin", "ffprobe")
# )


# ---- Load SRT file ----
def _load_srt_words(srt_file):
    subs = pysrt.open(srt_file)
    words = []
    for sub in subs:
        start = sub.start.ordinal  # in ms
        end = sub.end.ordinal
        text = sub.text.strip()
        words.append({"word": text, "start": start, "end": end})
    return words


# ---- Find sentence range in words ----
def find_sentence_range(sentence, words):
    sentence_words = sentence.strip().split()
    text_words = [w["word"] for w in words]

    # join into one text for matching
    full_text = " ".join(text_words)
    match = SequenceMatcher(
        None, full_text.lower(), sentence.lower()
    ).find_longest_match(0, len(full_text), 0, len(sentence))

    if match.size == 0:
        return None, None

    # Approximate match: locate sentence words in sequence
    for i in range(len(words) - len(sentence_words) + 1):
        window = [w["word"].lower() for w in words[i : i + len(sentence_words)]]
        if window == [sw.lower() for sw in sentence_words]:
            start_time = words[i]["start"]
            end_time = words[i + len(sentence_words) - 1]["end"]
            return start_time, end_time

    return None, None


# ---- Extract audio segment ----
# def extract_audio(audio_file, start_ms, end_ms, output_file):
#     audio = AudioSegment.from_file(audio_file)
#     segment = audio[start_ms:end_ms]
#     segment.export(output_file, format="mp3")
#     # print(f"Saved: {output_file}")


def extract_audio(audio_file, start_ms, end_ms, output_dir="storage"):
    audio = AudioSegment.from_file(audio_file)
    segment = audio[start_ms:end_ms]

    os.makedirs(output_dir, exist_ok=True)
    unique_name = f"sentence_{uuid.uuid4().hex}.mp3"
    output_file = os.path.join(output_dir, unique_name)
    # print(f"Saved: {output_file}")

    segment.export(output_file, format="mp3")
    return output_file


_srt_file = "096_words.srt"
_audio_file = "096.mp3"
srt_file = os.path.join(STORAGE_PATH, _srt_file)
audio_file = os.path.join(STORAGE_PATH, _audio_file)


def fetch_audio(sentence) -> tuple[str, int]:
    words = _load_srt_words(srt_file)
    start, end = find_sentence_range(sentence, words)
    if start is not None:
        output_file = extract_audio(audio_file, start, end, output_dir=STORAGE_PATH)
        return output_file, end-start
    else:
        return None, None


# Quran Text Validator & Locator (Embeddings + Alignment)
# -------------------------------------------------------
# This module:
# 1) Loads a word-by-word SRT (Quran) and builds searchable text chunks.
# 2) Uses sentence embeddings (NLP) to find where a user's text likely belongs.
# 3) Aligns user words vs reference words to detect exact mistakes (missing/extra/wrong).
# 4) Returns: status (Correct/Not correct), best matching reference text, and mistake details.
# 5) (Optional) Can extract the matched audio segment if you pass an audio file path.
#
# Notes:
# - Install deps: pip install "sentence-transformers>=3.0" rapidfuzz pysrt pydub
# - For Arabic normalization, we strip harakat/diacritics, normalize alef/yaa/ta marbuta, remove tatweel.
# - If embeddings are unavailable, we fallback to a RapidFuzz search.

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import re
import pysrt
import numpy as np

# Optional imports (use if installed)
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from rapidfuzz import fuzz, process

    _HAS_RF = True
except Exception:
    _HAS_RF = False

try:
    from pydub import AudioSegment

    _HAS_PYDUB = True
except Exception:
    _HAS_PYDUB = False

# -------------------------
# Arabic text normalization
# -------------------------
_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]"
)
_TATWEEL = "\u0640"

ALEF_FORMS = {
    "\u0622": "\u0627",  # ALEF WITH MADDA ABOVE → ALEF
    "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE → ALEF
    "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW → ALEF
    "\u0671": "\u0627",  # ALEF WASLA → ALEF
}

YA_HAMZA_MAP = {
    "\u0649": "\u064a",  # ALEF MAKSURA → YEH
}

TA_MARBUTA_MAP = {
    "\u0629": "\u0647",  # TEH MARBUTA → HEH (common normalization for matching)
}

_PUNCT = re.compile(
    r"[\u0600-\u0605\u061B\u061F\u066A-\u066D،؛؟.,:;!()\[\]{}\-\—\–\_\"'«»`~]+"
)
_SPACES = re.compile(r"\s+")


def normalize_ar(text: str) -> str:
    """Normalize Arabic text for robust matching.
    - remove diacritics (harakat)
    - remove tatweel
    - normalize Alef/Ya/Ta marbuta
    - strip punctuation
    - squeeze spaces, trim
    """
    if not text:
        return ""
    t = text
    t = _ARABIC_DIACRITICS.sub("", t)
    t = t.replace(_TATWEEL, "")
    for k, v in ALEF_FORMS.items():
        t = t.replace(k, v)
    for k, v in YA_HAMZA_MAP.items():
        t = t.replace(k, v)
    for k, v in TA_MARBUTA_MAP.items():
        t = t.replace(k, v)
    t = _PUNCT.sub(" ", t)
    t = _SPACES.sub(" ", t).strip()
    return t


# -------------------------
# SRT parsing
# -------------------------
@dataclass
class WordToken:
    word: str  # raw word (as in SRT)
    word_norm: str  # normalized
    start_ms: int  # start time in ms
    end_ms: int  # end time in ms


def load_srt_words(srt_path: str) -> List[WordToken]:
    subs = pysrt.open(srt_path)
    words: List[WordToken] = []
    for sub in subs:
        w = sub.text.strip()
        words.append(
            WordToken(
                word=w,
                word_norm=normalize_ar(w),
                start_ms=sub.start.ordinal,
                end_ms=sub.end.ordinal,
            )
        )
    return words


# -------------------------
# Chunking & Embeddings
# -------------------------
@dataclass
class TextChunk:
    text: str
    text_norm: str
    start_idx: int  # word index inclusive
    end_idx: int  # word index exclusive
    vector: Optional[np.ndarray] = None


def build_chunks(
    words: List[WordToken], chunk_size: int = 25, stride: int = 12
) -> List[TextChunk]:
    """Build overlapping chunks to search globally when ayah boundaries are unknown.
    Returns [TextChunk]. Each chunk is words[start:end].
    """
    chunks: List[TextChunk] = []
    N = len(words)
    i = 0
    while i < N:
        j = min(i + chunk_size, N)
        text = " ".join(w.word for w in words[i:j])
        text_norm = normalize_ar(text)
        chunks.append(TextChunk(text=text, text_norm=text_norm, start_idx=i, end_idx=j))
        if j == N:
            break
        i += stride
    return chunks


class EmbeddingIndex:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        if _HAS_ST:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "SentenceTransformer is not installed. Install sentence-transformers or use RapidFuzz fallback."
            )
        emb = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


# -------------------------
# Alignment (word-level mistakes)
# -------------------------
from difflib import SequenceMatcher


@dataclass
class Mistake:
    position: int  # 1-based position within the reference window
    type: str  # 'equal', 'substitution', 'insertion', 'deletion'
    user_word: Optional[str]
    correct_word: Optional[str]


def align_words(user_tokens: List[str], ref_tokens: List[str]) -> List[Mistake]:
    """Align two token lists and emit edit operations.
    Uses difflib opcodes as a lightweight alignment.
    """
    sm = SequenceMatcher(a=user_tokens, b=ref_tokens)
    mistakes: List[Mistake] = []
    # Track ref position for reporting (1-based)
    ref_pos = 1
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                mistakes.append(
                    Mistake(
                        position=ref_pos,
                        type="equal",
                        user_word=ref_tokens[j1 + k],
                        correct_word=ref_tokens[j1 + k],
                    )
                )
                ref_pos += 1
        elif tag == "replace":
            # substitution(s)
            # pair up min length, then treat extras as ins/del
            u_seg = user_tokens[i1:i2]
            r_seg = ref_tokens[j1:j2]
            m = min(len(u_seg), len(r_seg))
            for k in range(m):
                mistakes.append(
                    Mistake(
                        position=ref_pos,
                        type="substitution",
                        user_word=u_seg[k],
                        correct_word=r_seg[k],
                    )
                )
                ref_pos += 1
            # if user has extra words beyond r_seg -> insertion (no ref advance)
            for k in range(m, len(u_seg)):
                mistakes.append(
                    Mistake(
                        position=ref_pos,
                        type="insertion",
                        user_word=u_seg[k],
                        correct_word=None,
                    )
                )
            # if ref has extra words beyond u_seg -> deletion (advance ref)
            for k in range(m, len(r_seg)):
                mistakes.append(
                    Mistake(
                        position=ref_pos,
                        type="deletion",
                        user_word=None,
                        correct_word=r_seg[k],
                    )
                )
                ref_pos += 1
        elif tag == "delete":
            # ref has words missed by user
            for k in range(j1, j2):
                mistakes.append(
                    Mistake(
                        position=ref_pos,
                        type="deletion",
                        user_word=None,
                        correct_word=ref_tokens[k],
                    )
                )
                ref_pos += 1
        elif tag == "insert":
            # user added words not in ref window (attach to current ref_pos)
            for k in range(i1, i2):
                mistakes.append(
                    Mistake(
                        position=ref_pos,
                        type="insertion",
                        user_word=user_tokens[k],
                        correct_word=None,
                    )
                )
        else:
            pass
    return mistakes


# -------------------------
# Search over Quran SRT (sura now, whole Quran later)
# -------------------------
@dataclass
class MatchResult:
    status: str  # 'Correct' | 'Not correct'
    best_match_text: str  # reference text window
    mistakes: List[Mistake]  # detailed alignment
    start_word_index: int  # start index in the SRT words list
    end_word_index: int  # end index (exclusive)
    similarity: float  # similarity score (0..1)
    start_ms: Optional[int]  # audio start (if available)
    end_ms: Optional[int]  # audio end (if available)
    audio_file: Optional[str]


def _similarity_fallback(user_norm: str, chunk_norms: List[str]) -> np.ndarray:
    """Fallback similarity using RapidFuzz or simple token overlap if ST not available."""
    sims = []
    if _HAS_RF:
        for c in chunk_norms:
            # token_set_ratio ~ robust to word order
            sims.append(fuzz.token_set_ratio(user_norm, c) / 100.0)
    else:
        # very basic Jaccard on tokens
        u = set(user_norm.split())
        for c in chunk_norms:
            v = set(c.split())
            sims.append(len(u & v) / max(1, len(u | v)))
    return np.asarray(sims, dtype=np.float32)


def find_best_chunks(
    user_text: str,
    chunks: List[TextChunk],
    embedder: Optional[EmbeddingIndex],
    top_k: int = 5,
) -> List[int]:
    user_norm = normalize_ar(user_text)
    if embedder is not None and _HAS_ST:
        user_vec = embedder.encode([user_norm])  # (1, d)
        chunk_vecs = []
        for ch in chunks:
            if ch.vector is None:
                ch.vector = embedder.encode([ch.text_norm])[0]
            chunk_vecs.append(ch.vector)
        chunk_mat = np.vstack(chunk_vecs)
        # cosine similarity via dot since normalized
        sims = (user_vec @ chunk_mat.T).flatten()
    else:
        sims = _similarity_fallback(user_norm, [c.text_norm for c in chunks])

    # pick top_k indices
    top_idx = np.argsort(-sims)[:top_k].tolist()
    return top_idx


def validate_user_text(
    user_text: str,
    srt_path: str,
    audio_path: Optional[str] = None,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size: int = 25,
    stride: int = 12,
    top_k: int = 5,
    exact_threshold: float = 0.99,
) -> MatchResult:
    """Main entry point.
    - Searches globally to locate user_text within the SRT content.
    - Aligns tokens to report mistakes.
    - If audio_path provided (and pydub installed), returns the ms range of the best match.
    """
    words = load_srt_words(srt_path)
    chunks = build_chunks(words, chunk_size=chunk_size, stride=stride)

    # Build embedder if available
    embedder = EmbeddingIndex(model_name) if _HAS_ST else None

    # Find candidate chunks
    candidates = find_best_chunks(user_text, chunks, embedder, top_k=top_k)

    # Tokenize
    user_tokens_raw = user_text.strip().split()
    user_tokens_norm = normalize_ar(user_text).split()

    # Score and select the best by alignment-based similarity
    best = None
    best_score = -1.0
    best_alignment: List[Mistake] = []

    for idx in candidates:
        ch = chunks[idx]
        ref_tokens_raw = ch.text.split()
        ref_tokens_norm = ch.text_norm.split()
        # Align (on normalized tokens for robustness), but store raw for display
        alignment = align_words(user_tokens_norm, ref_tokens_norm)
        # Compute a simple similarity: fraction of 'equal' vs total excluding insertions
        equal = sum(1 for m in alignment if m.type == "equal")
        subs = sum(1 for m in alignment if m.type == "substitution")
        dels = sum(1 for m in alignment if m.type == "deletion")
        # ignore insertions in denominator to not over-penalize extras
        denom = max(1, equal + subs + dels)
        score = equal / denom
        if score > best_score:
            best_score = score
            best = ch
            # Rebuild alignment but with *raw* tokens for reporting while keeping types/positions
            alignment_raw = align_words(user_text.strip().split(), ch.text.split())
            best_alignment = alignment_raw

    assert best is not None

    status = (
        "Correct"
        if best_score >= exact_threshold
        and len([m for m in best_alignment if m.type != "equal"]) == 0
        else "Not correct"
    )

    # Compute ms range from word indices
    start_ms = words[best.start_idx].start_ms if words else None
    end_ms = words[best.end_idx - 1].end_ms if words else None

    return MatchResult(
        status=status,
        best_match_text=best.text,
        mistakes=best_alignment,
        start_word_index=best.start_idx,
        end_word_index=best.end_idx,
        similarity=float(best_score),
        start_ms=start_ms,
        end_ms=end_ms,
        audio_file=audio_path,
    )


# -------------------------
# Optional: extract matched audio
# -------------------------


def extract_matched_audio(result: MatchResult, output_path: str) -> Optional[str]:
    if not _HAS_PYDUB:
        raise RuntimeError("pydub is not installed. pip install pydub")
    if not result.audio_file or result.start_ms is None or result.end_ms is None:
        return None
    audio = AudioSegment.from_file(result.audio_file)
    segment = audio[result.start_ms : result.end_ms]
    segment.export(output_path, format="mp3")
    return output_path


"""
=============================================================
"""

# Adjust these paths to your files
SRT_PATH = os.path.join(STORAGE_PATH, "096_words.srt")
AUDIO_PATH = os.path.join(STORAGE_PATH, "096.mp3")


def fetch_wrong_audio(sentence_):

    result = validate_user_text(
        user_text=sentence_,
        srt_path=SRT_PATH,
        audio_path=AUDIO_PATH,
        top_k=8,
        chunk_size=20,
        stride=10,
    )
    # Print mistakes in a readable way
    wrong_sentence = ""
    for m in result.mistakes:
        if m.type != "equal":
            print(
                f"pos={m.position:02d}\t{m.type:12s}\tuser={m.user_word!r}\tcorrect={m.correct_word!r}"
            )
            wrong_sentence = m.correct_word
    print("wrong_sentence", wrong_sentence)
    fetch_audio(wrong_sentence)


# # Example user inputs (typos, missing words, etc.)
# user_text = "اقرأ بسم ربك الذذي خلق"  # intentionally wrong (missing ل)
# fetch_wrong_audio(user_text)
