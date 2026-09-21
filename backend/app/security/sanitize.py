import re
from typing import Tuple, Optional

INJECTION_PATTERNS = [
    (r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", "Attempt to override system instructions"),
    (r"(?i)\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\b", "Attempt to disregard instructions"),
    (r"(?i)\byou\s+are\s+now\s+(?:in\s+)?(?:dan|developer|god|unrestricted)\s+mode\b", "Jailbreak persona prompt"),
    (r"(?i)\b(?:reveal|show|print|dump|repeat)\s+(?:your\s+)?(?:system\s+prompt|secret|api\s*key|credentials|instructions)\b", "System prompt / secret exfiltration attempt"),
    (r"(?i)<\s*(?:system|instruction|prompt)\s*>", "Context boundary injection tags"),
    (r"(?i)\[\s*(?:inst|instruction|system)\s*\]", "Instruction delimiter injection"),
]

def check_prompt_injection(text: str) -> Tuple[bool, Optional[str]]:
    """Checks whether the input query contains known prompt injection signatures."""
    if not text:
        return False, None

    for pattern, reason in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True, reason

    return False, None

def sanitize_query_text(query: str) -> str:
    """Sanitizes query text by removing adversarial delimiter tags and excessive control sequences."""
    if not query:
        return ""

    cleaned = re.sub(r'<\s*/?\s*(?:system|instruction|admin|prompt)[^>]*>', ' ', query, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[/?(?:INST|INSTRUCTION|SYSTEM)\]', ' ', cleaned, flags=re.IGNORECASE)
    # Collapse multiple whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def sanitize_document_context(chunk_text: str) -> str:
    """
    Sanitizes untrusted document text before it is embedded in the LLM prompt context.
    Neutralizes boundary break attempts and indirect prompt injection delimiters.
    """
    if not chunk_text:
        return ""

    # Neutralize control tags that mimic system/instruction boundaries
    cleaned = re.sub(r'<\s*/?\s*(?:system|instruction|admin|prompt)[^>]*>', '[filtered_tag]', chunk_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[/?(?:INST|INSTRUCTION|SYSTEM)\]', '[filtered_delimiter]', cleaned, flags=re.IGNORECASE)
    # Escape triple dashes or block separators if on their own line
    cleaned = re.sub(r'^\s*---\s*$', '- - -', cleaned, flags=re.MULTILINE)
    # Escape backticks if used to break code blocks
    cleaned = re.sub(r'```', '` ` `', cleaned)
    return cleaned
