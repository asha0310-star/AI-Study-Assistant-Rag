from typing import List, Dict


def generate_quiz_from_text(text: str, n: int = 5) -> List[Dict[str, str]]:
    """Generate simple question/answer pairs from text by using leading sentences as answers.

    This is a lightweight helper — replace with an LLM-based generator for better quizzes.
    """
    if not text:
        return []
    s = text.replace("\n", " ")
    sents = [seg.strip() for seg in s.split('. ') if seg.strip()]
    qa = []
    for i, sent in enumerate(sents[:n]):
        question = f"In one sentence, summarize: '{sent[:80]}...'?"
        answer = sent if sent.endswith('.') else sent + '.'
        qa.append({"question": question, "answer": answer})
    return qa
