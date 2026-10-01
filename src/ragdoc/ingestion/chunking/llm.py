"""LLM-based chunking: the chat model picks the sentences at which a new chunk starts.

The model only returns sentence numbers. The chunks are cut from the original text, so the
model cannot change or invent content.
"""

import re
from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ragdoc.ingestion.chunking import prompts
from ragdoc.ingestion.chunking.base import BaseChunker, ChunkCheckpoint
from ragdoc.ingestion.chunking.sentences import split_sentences
from ragdoc.llm.chat import text_of

PROGRESS_MESSAGE = "Chunking: sentences read by the chat model"
# A sentence is shown to the model up to this length: enough to tell its topic, and it bounds
# the prompt when the text has a very long run without sentence punctuation.
SHOWN_CHARS = 500

_SPLITS_LINE = re.compile(r"(?im)^[\s*_>#-]*splits[\s*_]*:(.*)$")


def parse_splits(reply: str, n_sentences: int) -> list[int] | None:
    """The 0-based positions at which a new chunk starts, from a `Splits: 4, 9` reply.

    Numbers that are out of range or not increasing are dropped. Returns None when the reply
    has no usable `Splits:` line (neither numbers nor "none"); the last such line wins.
    """
    lines = _SPLITS_LINE.findall(reply)
    if not lines:
        return None
    numbers = [int(number) for number in re.findall(r"\d+", lines[-1])]
    if not numbers and "none" not in lines[-1].lower():
        return None
    splits: list[int] = []
    for number in numbers:
        if 2 <= number <= n_sentences and (not splits or number - 1 > splits[-1]):
            splits.append(number - 1)
    return splits


def shown(sentence: str) -> str:
    """A sentence as the model sees it: on one line, cut if very long."""
    line = " ".join(sentence.split())
    return line if len(line) <= SHOWN_CHARS else line[:SHOWN_CHARS] + " ..."


class LLMChunker(BaseChunker):
    def __init__(
        self,
        *,
        model: BaseChatModel,
        target_chunk_chars: int,
        max_chunk_chars: int,
        window_chars: int,
        table_max_chars: int,
    ):
        super().__init__(table_max_chars=table_max_chars)
        self._model = model
        self._target_chars = target_chunk_chars
        self._max_chars = max_chunk_chars
        self._window_chars = window_chars

    def split_texts(
        self, texts: Sequence[str], checkpoint: ChunkCheckpoint | None
    ) -> list[list[str]]:
        sentences = [split_sentences(text) for text in texts]
        total = sum(len(units) for units in sentences)
        done = 0

        result: list[list[str]] = []
        for text, units in zip(texts, sentences, strict=True):
            if len(text) <= self._target_chars or len(units) < 2:
                pieces = self._limit([text], self._max_chars)  # nothing to decide: no model call
            else:

                def report(position: int, done: int = done) -> None:
                    if checkpoint is not None:
                        checkpoint(PROGRESS_MESSAGE, done + position, total)

                pieces = self._split(units, report)
            done += len(units)
            result.append(pieces)
        return result

    def _split(self, units: list[str], report: Callable[[int], None]) -> list[str]:
        """Cut one text, a window of sentences at a time."""
        pieces: list[str] = []
        start = 0
        while start < len(units):
            end = self._window_end(units, start)
            window = units[start:end]
            if len(window) < 2:
                pieces.extend(self._limit_sentences([window], self._max_chars))
                start = end
                continue

            report(start)
            splits = self._ask(window)
            if splits is None:
                # No usable reply: this window is cut by size instead, between sentences.
                self.n_fallbacks += 1
                pieces.extend(self._limit_sentences([window], self._target_chars))
                start = end
                continue

            bounds = [0, *splits, len(window)]
            segments = [window[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]
            if end < len(units) and splits:
                # The window ended mid-text, so its last chunk may continue: the sentences
                # after the last boundary open the next window instead.
                segments.pop()
                start += splits[-1]
            else:
                start = end
            pieces.extend(self._limit_sentences(segments, self._max_chars))
        return pieces

    def _window_end(self, units: list[str], start: int) -> int:
        """The end of the window starting at `start`: as many sentences as fit, at least one."""
        size = 0
        end = start
        while end < len(units):
            size += len(shown(units[end])) + 1
            if size > self._window_chars and end > start:
                break
            end += 1
        return end

    def _ask(self, window: list[str]) -> list[int] | None:
        lines = [shown(sentence) for sentence in window]
        average = max(sum(len(line) for line in lines) / len(lines), 1)
        system = prompts.LLM_CHUNK_SYSTEM.format(
            target_chars=self._target_chars,
            target_sentences=max(round(self._target_chars / average), 1),
        )
        numbered = "\n".join(f"[{number}] {line}" for number, line in enumerate(lines, start=1))
        reply = self._model.invoke([SystemMessage(system), HumanMessage(numbered)])
        return parse_splits(text_of(reply), len(window))
