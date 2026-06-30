# AI Study Assistant (RAG)

A Streamlit app that turns your PDF study notes into a searchable, grounded
study tool. It extracts text from uploaded PDFs, chunks it, embeds the chunks
into a local ChromaDB vector store, and uses Google Gemini to answer questions,
generate summaries, and build quizzes — **using only the content of your
documents**.

## Features

- Upload one or more PDFs and extract text page by page
- Chunk text with overlap and store embeddings in a local ChromaDB
- Ask questions and get answers grounded only in the retrieved chunks, with
  page-level source references
- Generate summaries in three styles: beginner, exam-focused, bullet-point
- Generate quizzes (MCQ, Short Answer, True/False), optionally improved by Gemini

## Project structure

```
app.py                     Streamlit UI and app flow
src/pdf_loader.py          Extract text from uploaded PDFs
src/chunker.py             Split pages into overlapping chunks
src/vector_store.py        Embeddings + ChromaDB storage and search
src/rag_pipeline.py        Build context, call Gemini, format sources
src/summary_generator.py   Gemini-powered summaries
src/quiz_generator.py      Heuristic + Gemini quiz generation
```

## Setup

Requires Python 3.10+.

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd AI-Study-Assistant-Rag

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# then edit .env and paste your Gemini API key
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Your `.env` should look like:

```
GEMINI_API_KEY=your_real_key_here
CHROMA_DB_DIR=./chroma_db
```

## Running

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

Workflow in the app: upload PDFs → **Process PDFs** → then summarise, quiz,
or ask questions.

## Security

- **Never commit your `.env` file.** It is listed in `.gitignore`. If a real
  key was ever committed, revoke it immediately in Google AI Studio, generate a
  new one, and remove it from git history (e.g. with `git filter-repo`).
- The `chroma_db/` folder is local app data and is also git-ignored. Don't
  commit it.

## Known limitations

- **Single document set at a time.** Processing new PDFs clears the existing
  ChromaDB collection, so the app holds only the most recently processed
  documents. The local `chroma_db/` folder is shared, so multiple tabs or users
  pointing at the same folder will overwrite each other.
- **Embedding model name.** `src/vector_store.py` uses
  `models/gemini-embedding-001`. If your account exposes a different name, the
  app will tell you how to list available models and update the constant.
- **Quiz quality.** The default heuristic quiz generator (fill-in-the-blank
  with random-word distractors) is basic. Enable the "Use Gemini" option for
  higher-quality questions and explanations.
- **SDK deprecation.** This project uses `google-generativeai`, which Google is
  phasing out in favor of `google-genai`. A future migration will be needed.

## How it works

1. **Extract** — `pypdf` reads each page's text, tagged with file name and page
   number.
2. **Chunk** — `RecursiveCharacterTextSplitter` splits pages into ~1000-char
   chunks with 200-char overlap, preserving metadata.
3. **Embed & store** — each chunk is embedded with Gemini and stored in
   ChromaDB.
4. **Retrieve** — a question is embedded and the top-k nearest chunks are
   pulled from ChromaDB.
5. **Generate** — retrieved chunks become the context for Gemini, which is
   instructed to answer only from that context and otherwise say it couldn't
   find the answer.