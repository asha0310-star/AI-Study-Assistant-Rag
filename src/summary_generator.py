def summarize_text(text: str, max_length: int = 256) -> str:
    """Try to use transformers pipeline for summarization; fall back to a simple heuristic."""
    if not text:
        return ""
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        out = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return out[0]["summary_text"]
    except Exception:
        # naive fallback: return first ~max_length characters (end at sentence)
        s = text.replace("\n", " ")
        if len(s) <= max_length:
            return s
        # try to end at sentence boundary
        idx = s.rfind('.', 0, max_length)
        if idx != -1 and idx > 20:
            return s[: idx + 1]
        return s[:max_length] + '...'
