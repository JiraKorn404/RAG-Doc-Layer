"""Cutting text into sentences, for the strategies that decide boundaries between them."""

import re

# After sentence punctuation (and any closing quote or bracket) when a capital, digit or
# heading follows; or at a blank line. English only.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9#])|\n\s*\n\s*")
# A shorter piece ("1.", "e.g.", a one-word heading) stays with what follows it.
_MIN_CHARS = 15


def split_sentences(text: str) -> list[str]:
    """The sentences of `text`, each with its trailing whitespace, so that joining them gives
    the text back unchanged. Empty for text that is only whitespace."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if len(text[start : match.end()].strip()) >= _MIN_CHARS:
            sentences.append(text[start : match.end()])
            start = match.end()
    rest = text[start:]
    if sentences and len(rest.strip()) < _MIN_CHARS:
        sentences[-1] += rest
    elif rest.strip():
        sentences.append(rest)
    return sentences
