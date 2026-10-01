"""Console helpers for the command-line scripts."""

import sys


def use_utf8_output() -> None:
    """Make stdout/stderr UTF-8, replacing what cannot be shown.

    Windows consoles often use a legacy code page, where printing model output (emoji, symbols)
    or document text would otherwise raise UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
