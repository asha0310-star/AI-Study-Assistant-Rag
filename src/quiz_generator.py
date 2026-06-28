from typing import List, Dict
import random
import re
import os

import google.generativeai as genai


QUIZ_MODEL = "gemini-2.5-flash"


SUPPORTED_TYPES = ["MCQ", "Short Answer", "True/False"]


def _split_into_sentences(text: str) -> List[str]:
    s = text.replace("\n", " ")
    # naive sentence split
    parts = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", s) if seg.strip()]
    return parts


def _pick_source_sentences(chunks: List[Dict], n: int) -> List[str]:
    # Flatten chunk texts and pick n sentences at random (or first n if small)
    all_sents = []
    for chunk in chunks:
        all_sents.extend(_split_into_sentences(chunk.get("text", "")))

    if not all_sents:
        return []

    if len(all_sents) <= n:
        return all_sents

    return random.sample(all_sents, n)


def _make_mcq_from_sentence(sentence: str, other_sentences: List[str]) -> Dict[str, str]:
    # Choose a word to blank (longest reasonable word)
    words = [w for w in re.findall(r"\w+", sentence) if len(w) > 3]
    if not words:
        # fallback: use entire sentence as answer (convert to short answer)
        return {"type": "Short Answer", "question": f"Summarize: {sentence}", "answer": sentence, "explanation": sentence}

    correct = max(words, key=len)
    question = sentence.replace(correct, "______", 1)

    # Create distractors from other sentences' words
    pool = []
    for s in other_sentences:
        pool.extend([w for w in re.findall(r"\w+", s) if len(w) > 3 and w.lower() != correct.lower()])

    distractors = list(dict.fromkeys([w for w in pool]))  # unique
    random.shuffle(distractors)
    options = [correct]
    for w in distractors[:3]:
        options.append(w)

    # Fill with simple placeholders if not enough distractors
    while len(options) < 4:
        options.append("(none)")

    random.shuffle(options)

    return {
        "type": "MCQ",
        "question": question,
        "options": options,
        "answer": correct,
        "explanation": f"Original sentence: {sentence}",
    }


def _make_true_false_from_sentence(sentence: str, other_sentences: List[str]) -> Dict[str, str]:
    # Randomly decide truth. If false, swap a noun with one from other sentences.
    is_true = random.choice([True, False])
    if is_true or not other_sentences:
        statement = sentence
        answer = "True"
        explanation = f"Based on source: {sentence}"
    else:
        # try to replace a noun-like word
        nouns = [w for w in re.findall(r"\w+", sentence) if w[0].isalpha() and len(w) > 3]
        replacement_pool = []
        for s in other_sentences:
            replacement_pool.extend([w for w in re.findall(r"\w+", s) if len(w) > 3])
        if nouns and replacement_pool:
            orig = random.choice(nouns)
            repl = random.choice(replacement_pool)
            statement = sentence.replace(orig, repl, 1)
            answer = "False"
            explanation = f"The original statement from sources was: {sentence}"
        else:
            # fallback: mark true
            statement = sentence
            answer = "True"
            explanation = f"Based on source: {sentence}"

    return {"type": "True/False", "question": statement, "answer": answer, "explanation": explanation}


def _make_short_answer_from_sentence(sentence: str) -> Dict[str, str]:
    # Use the sentence as the answer prompt; keep it simple
    question = f"In one sentence, summarize: '{sentence[:100]}...'"
    answer = sentence if sentence.endswith('.') else sentence + '.'
    return {"type": "Short Answer", "question": question, "answer": answer, "explanation": answer}


def generate_quiz_from_chunks(chunks: List[Dict], n: int = 5, qtype: str = "MCQ") -> List[Dict]:
    """Generate a quiz from processed chunks.

    - `chunks` is the list returned by `chunk_pages` and stored in session state.
    - `qtype` can be one of: `MCQ`, `Short Answer`, `True/False`.
    Returns a list of question dicts containing the question, answer, explanation, and for MCQ the options.
    """
    if not chunks:
        return []

    qtype = qtype if qtype in SUPPORTED_TYPES else "MCQ"
    sentences = _pick_source_sentences(chunks, max(n * 3, 20))
    if not sentences:
        return []

    selected = sentences[:n] if len(sentences) <= n else random.sample(sentences, n)

    quiz = []
    for sent in selected:
        other = [s for s in sentences if s != sent]
        if qtype == "MCQ":
            quiz.append(_make_mcq_from_sentence(sent, other))
        elif qtype == "True/False":
            quiz.append(_make_true_false_from_sentence(sent, other))
        else:
            quiz.append(_make_short_answer_from_sentence(sent))

    return quiz


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file to use Gemini for quizzes.")
    return api_key


def _call_gemini_for_question(sentence: str, qtype: str) -> Dict:
    """Ask Gemini to produce a single quiz item as JSON. Returns a dict or raises.

    This is optional and used only when `use_gemini=True`.
    """
    prompt = f"""
You are a helpful assistant that creates short quiz questions from a single source sentence.
Create one question of type: {qtype} based ONLY on the following sentence.

Sentence:
{sentence}

Return output as JSON with these keys:
- type: one of 'MCQ', 'Short Answer', or 'True/False'
- question: the question text
- options: an array of strings (only for MCQ)
- answer: the correct answer (string)
- explanation: a one-sentence explanation that references the sentence

Output JSON only. Do not include any additional commentary.
"""

    genai.configure(api_key=get_api_key())
    model = genai.GenerativeModel(QUIZ_MODEL)
    response = model.generate_content(prompt)
    text = (response.text or "").strip()

    # Try to parse JSON safely
    try:
        import json

        parsed = json.loads(text)
        return parsed
    except Exception:
        # If parsing fails, raise to allow caller to fallback to heuristics
        raise RuntimeError("Gemini did not return valid JSON for quiz item")


def generate_quiz_from_chunks(chunks: List[Dict], n: int = 5, qtype: str = "MCQ", use_gemini: bool = False) -> List[Dict]:
    """Generate a quiz from processed chunks.

    By default this uses a simple, deterministic heuristic. Set `use_gemini=True`
    to ask Gemini to produce higher-quality questions, options and explanations.
    """
    if not chunks:
        return []

    qtype = qtype if qtype in SUPPORTED_TYPES else "MCQ"
    sentences = _pick_source_sentences(chunks, max(n * 3, 20))
    if not sentences:
        return []

    selected = sentences[:n] if len(sentences) <= n else random.sample(sentences, n)

    quiz = []
    for sent in selected:
        other = [s for s in sentences if s != sent]
        if use_gemini:
            try:
                item = _call_gemini_for_question(sent, qtype)
                # Normalise keys and ensure required fields exist
                item_type = item.get("type", qtype)
                question = item.get("question", sent)
                answer = item.get("answer", "")
                explanation = item.get("explanation", "")
                options = item.get("options") if item_type == "MCQ" else None

                entry = {"type": item_type, "question": question, "answer": answer, "explanation": explanation}
                if options:
                    entry["options"] = options

                quiz.append(entry)
                continue
            except Exception:
                # fallback to heuristics silently
                pass

        # fallback heuristics (no Gemini or Gemini failed)
        if qtype == "MCQ":
            quiz.append(_make_mcq_from_sentence(sent, other))
        elif qtype == "True/False":
            quiz.append(_make_true_false_from_sentence(sent, other))
        else:
            quiz.append(_make_short_answer_from_sentence(sent))

    return quiz


if __name__ == "__main__":
    # tiny smoke test
    sample = [{"text": "Photosynthesis is the process by which plants create energy from sunlight. It involves chlorophyll."}]
    print(generate_quiz_from_chunks(sample, n=3, qtype="MCQ"))
