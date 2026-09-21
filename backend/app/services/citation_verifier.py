import re
from typing import Dict, List, Tuple
from app.models.schemas import Citation

def verify_citation(quote: str, chunk_text: str) -> Tuple[str, float]:
    """
    Verifies that the cited quote exists in the source chunk.
    Returns (status, match_confidence) where status is 'verified', 'partial_match', or 'unverified'.
    """
    if not quote or not chunk_text:
        return "unverified", 0.0

    norm_quote = normalize_text(quote)
    norm_chunk = normalize_text(chunk_text)

    # 1. Exact substring match
    if norm_quote in norm_chunk:
        return "verified", 1.0

    # 2. Check if a major substring of the quote exists
    quote_words = norm_quote.split()
    if len(quote_words) >= 4:
        # Check 4-word shingles
        shingles = [" ".join(quote_words[i:i+4]) for i in range(len(quote_words) - 3)]
        matched_shingles = sum(1 for s in shingles if s in norm_chunk)
        ratio = matched_shingles / len(shingles) if shingles else 0.0

        if ratio >= 0.70:
            return "verified", round(ratio, 2)
        elif ratio >= 0.35:
            return "partial_match", round(ratio, 2)

    # 3. Bag of words Jaccard overlap
    chunk_words_set = set(norm_chunk.split())
    quote_words_set = set(quote_words)
    overlap = len(quote_words_set.intersection(chunk_words_set))
    jaccard = overlap / len(quote_words_set) if quote_words_set else 0.0

    if jaccard >= 0.80:
        return "verified", round(jaccard, 2)
    elif jaccard >= 0.50:
        return "partial_match", round(jaccard, 2)

    return "unverified", round(jaccard, 2)

def normalize_text(text: str) -> str:
    # Lowercase, remove quotation marks, normalize spaces
    cleaned = text.lower()
    cleaned = re.sub(r'["\'`“”‘’]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def find_best_excerpt_in_chunk(query: str, chunk_text: str, max_words: int = 160) -> str:
    """Finds the most relevant excerpt or paragraph in the chunk to serve as a verified citation."""
    clean_chunk = chunk_text.strip()
    words = clean_chunk.split()
    if len(words) <= max_words:
        return clean_chunk

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_chunk) if s.strip()]
    query_tokens = set(normalize_text(query).split())

    # Score sentences by token overlap
    scored_sents = []
    for idx, sent in enumerate(sentences):
        sent_tokens = set(normalize_text(sent).split())
        score = len(query_tokens.intersection(sent_tokens))
        scored_sents.append((idx, sent, score))

    scored_sents.sort(key=lambda x: x[2], reverse=True)
    if not scored_sents:
        return " ".join(words[:max_words]) + "..."

    # Take the best sentence and adjacent sentences up to max_words
    best_idx = scored_sents[0][0]
    selected_indices = {best_idx}
    if len(sentences) > best_idx + 1:
        selected_indices.add(best_idx + 1)
    if best_idx > 0 and len(" ".join([sentences[i] for i in selected_indices]).split()) < max_words:
        selected_indices.add(best_idx - 1)

    combined = " ".join([sentences[i] for i in sorted(selected_indices)])
    c_words = combined.split()
    if len(c_words) > max_words:
        return " ".join(c_words[:max_words]) + "..."
    return combined
