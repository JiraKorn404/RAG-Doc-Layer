"""Ask a question from the command line and watch the agent work.

uv run python scripts/chat.py "What was the operating profit in 2023?"
uv run python scripts/chat.py "And the year after?" --conversation <id>
"""

import argparse
import logging
import sys

from ragdoc.agent.context import source_label
from ragdoc.console import use_utf8_output
from ragdoc.schemas import ChatEventType, ChatResult
from ragdoc.services.chat_service import ChatService


def print_sources(result: ChatResult) -> None:
    if result.retrieval is None or result.retrieval.is_empty:
        print("\nSources: none")
        return
    print("\nSources (* = cited in the answer):")
    for number, chunk in enumerate(result.retrieval.chunks, start=1):
        mark = "*" if chunk.id in result.used_chunk_ids else " "
        print(f" {mark} {source_label(number, chunk)}  similarity {chunk.similarity_score:.4f}")


def main() -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--conversation", help="continue this conversation id")
    parser.add_argument("--debug", action="store_true", help="print the traceback on failure")
    args = parser.parse_args()
    if not args.debug:
        # The service logs the full traceback; here the one-line error message is enough.
        logging.disable(logging.CRITICAL)

    service = ChatService()
    conversation_id = args.conversation or str(service.create_conversation().id)
    print(f"Conversation: {conversation_id}\n")

    section = ""  # which stream is on the current line: "", "reasoning" or "answer"

    def start_section(name: str, heading: str) -> None:
        nonlocal section
        if section != name:
            print(("\n" if section else "") + heading)
            section = name

    for event in service.stream(conversation_id, args.question):
        if event.type is ChatEventType.NODE_START:
            if section:
                print()
                section = ""
            print(f"[{event.node}] ...", flush=True)
        elif event.type is ChatEventType.TRACE:
            if section:
                print()
                section = ""
            trace = event.trace
            print(f"[{trace.node}] {trace.summary}  ({trace.duration_ms:.0f} ms)", flush=True)
        elif event.type is ChatEventType.REASONING:
            start_section("reasoning", "--- thinking ---")
            print(event.text, end="", flush=True)
        elif event.type is ChatEventType.TOKEN:
            start_section("answer", "--- answer ---")
            print(event.text, end="", flush=True)
        elif event.type is ChatEventType.ERROR:
            print(f"\nError: {event.text}")
            return 1
        elif event.type is ChatEventType.FINAL:
            result = event.result
            print_sources(result)
            metrics = result.metrics
            print(
                f"\nTotal {metrics['total_ms'] / 1000:.1f} s  "
                f"(retrieve {metrics['retrieve_ms']:.0f} ms, grade {metrics['grade_ms']:.0f} ms, "
                f"generate {metrics['generate_ms']:.0f} ms)  "
                f"tokens in/out {metrics['prompt_tokens']}/{metrics['completion_tokens']}  "
                f"retries {metrics['n_retries']}"
            )
            print(f"Saved as message {result.message_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
