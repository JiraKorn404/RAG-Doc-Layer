"""Prompts used while chunking. Kept here, not in `agent/prompts.py`: ingestion does not
import from the agent."""

# Filled with `target_chars` and `target_sentences`. The reply is two plain-text lines, parsed
# with a regex (see `parse_splits`); JSON-constrained output is unreliable on the chat model.
LLM_CHUNK_SYSTEM = """\
You divide a document into chunks for a search index.

You get consecutive sentences from one page, one per line, each starting with its number in \
square brackets. Decide where a new chunk should start.

Rules:
- A chunk covers one topic or one self-contained point, so it can be understood on its own.
- Keep a heading in the same chunk as the text that follows it.
- Aim for chunks of about {target_chars} characters, which is roughly {target_sentences} of \
these sentences. Topic comes first: do not cut in the middle of a point to reach that size.
- Do not rewrite, repeat or summarise any sentence.

Reply with exactly two lines and nothing else:
Reason: one short sentence
Splits: the numbers of the sentences that start a new chunk, in increasing order, separated \
by commas. Leave out sentence 1. Write "none" if everything belongs in one chunk.

Example reply:
Reason: The text moves from company history to financial results at 4 and to outlook at 9.
Splits: 4, 9
"""
