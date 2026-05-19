# AI Video Meeting Assistant

AI Video Meeting Assistant turns a YouTube video, local video, or audio file into a meeting intelligence workspace. It transcribes the source, generates a professional summary, extracts action items, decisions, and open questions, then lets users ask transcript-grounded questions with RAG.

## Features

- YouTube or local audio/video ingestion
- Audio conversion and chunking with FFmpeg and pydub
- English transcription with local Whisper
- Hinglish speech-to-text translation with Sarvam AI
- Meeting title and summary generation with Mistral
- Action item, decision, and open-question extraction
- Transcript-grounded Q&A using LangChain, Chroma, and HuggingFace embeddings
- Streamlit interface plus a CLI pipeline
- Request validation for media paths, FFmpeg, language choices, and API keys
- Retry handling for external calls and cleanup of generated audio artifacts

## Architecture

```text
Input video/audio
  -> audio download/conversion
  -> audio chunking
  -> transcription
  -> title, summary, and extraction chains
  -> transcript chunking and embeddings
  -> Chroma vector store
  -> RAG chat over transcript context
```

## Project Structure

```text
.
├── app.py                  # Streamlit UI
├── main.py                 # CLI entry point
├── test.py                 # Manual smoke-test script
├── core/
│   ├── extractor.py        # Action item, decision, question extraction
│   ├── pipeline.py         # Validated, fault-tolerant pipeline runner
│   ├── rag_engine.py       # RAG chain construction and Q&A
│   ├── summarizer.py       # Summary and title generation
│   ├── transcriber.py      # Whisper and Sarvam transcription
│   └── vector_store.py     # Chroma vector store helpers
└── utils/
    ├── audio_processor.py  # YouTube download, conversion, and chunking
    ├── retrying.py         # Retry helpers for external calls
    └── validation.py       # Input and environment validation
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install Python dependencies.

```bash
pip install -r requirements.txt
```

3. Install FFmpeg and make sure it is available on your `PATH`.

```bash
ffmpeg -version
```

4. Create a `.env` file from the example.

```bash
copy .env.example .env
```

5. Fill in your API keys in `.env`.

## Run the Streamlit App

```bash
streamlit run app.py
```

Then enter a YouTube URL or a local file path, select the language, and run the analysis.

## Run the CLI Pipeline

```bash
python main.py
```

## Run the Smoke Test

```bash
python test.py
```

You can also provide `SMOKE_TEST_SOURCE` and `SMOKE_TEST_LANGUAGE` in your environment to avoid the prompt.

## Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `MISTRAL_API_KEY` | Yes | Powers summary, extraction, title generation, and RAG answers |
| `SARVAM_API_KEY` | Only for Hinglish | Enables Sarvam speech-to-text translation |
| `SARVAM_STT_MODEL` | No | Sarvam model name, defaults to `saaras:v2.5` |
| `WHISPER_MODEL` | No | Whisper model name, defaults to `small` |

## Current Limitations

- RAG indexing currently uses a shared local Chroma collection.
- There is no automated evaluation suite yet.
- Long-running jobs execute inside the Streamlit request flow.
- Transcript chunks do not yet include timestamps or speaker labels.

These are the next production-hardening targets.

## Roadmap

- Session-isolated vector stores
- Safer rendering and stronger input validation
- Transcript, embedding, and model caching
- Reranked retrieval with citations
- RAG and extraction evaluation suite
- Structured logging and latency metrics
- Exportable meeting reports
