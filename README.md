# Enterprise Documentation Q&A Assistant

## Stack
LangChain document loaders/splitter, ChromaDB, all-MiniLM-L6-v2, BM25, Reciprocal Rank Fusion, LangGraph, Groq and Streamlit.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.env
```
## Run
```powershell
streamlit run app.py
```
Upload PDF, HTML, Markdown or TXT documents from the sidebar. Enter metadata and click Ingest and index.

## Workflow
- LangChain `RecursiveCharacterTextSplitter` chunks the documents.
- Each chunk carries document ID/title, source, page number, section, document type, version, effective date, upload date, status, chunk ID and start index.
- `all-MiniLM-L6-v2` generates local embeddings; no embedding API key is needed.
- ChromaDB stores embeddings, content and metadata persistently.
- BM25 keyword search and ChromaDB dense search are combined using RRF.
- LangGraph checks ambiguity before retrieval, then validates retrieved evidence and possible conflicts before generating an answer.

## Important
- Python 3.10/3.11 recommended.
- First run downloads model weights and needs internet.
- Scanned/image-only PDFs need OCR; not included.
- The 850-character chunk size is a starting point, not a universal optimum. MiniLM's maximum sequence length is 256 wordpiece tokens; test chunking on your corpus.
- The upload metadata form applies to all files in one batch. Upload different document versions separately.
- LLM ambiguity/evidence/conflict checks are not guaranteed. Build labelled evaluation cases for ambiguous, answerable, unanswerable and conflicting questions.
- No numeric confidence probability is fabricated. Statuses are AMBIGUOUS, NOT_FOUND, ANSWERED or CONFLICT.
- Production hardening should include authentication, access controls, document deletion/version lifecycle, OCR, robust section extraction and automated evaluation.
