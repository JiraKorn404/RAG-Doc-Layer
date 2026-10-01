"""Small text helpers shared by the components."""


def safe_markdown(text: str) -> str:
    """Model or document text for `st.markdown`, with `$` escaped.

    Streamlit renders `$...$` as LaTeX, which garbles any text that mentions money.
    """
    return text.replace("$", "\\$")


def seconds(milliseconds: float) -> str:
    """A duration for display: `850 ms` or `12.3 s`."""
    if milliseconds < 1000:
        return f"{milliseconds:.0f} ms"
    return f"{milliseconds / 1000:.1f} s"


def duration(total_seconds: float) -> str:
    """A longer duration for display: `48 s` or `1 min 12 s`."""
    total = round(total_seconds)
    if total < 60:
        return f"{total} s"
    return f"{total // 60} min {total % 60:02d} s"
