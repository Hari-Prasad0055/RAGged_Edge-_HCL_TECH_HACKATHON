from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
from typing import TypedDict

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

load_dotenv()

BASE = Path(__file__).parent
DATA = BASE / "data"
CHROMA = BASE / "chroma_db"
STORE = BASE / "storage"

for p in (DATA, CHROMA, STORE):
    p.mkdir(exist_ok=True)

BM25_FILE = STORE / "bm25.pkl"
RECORDS_FILE = STORE / "records.pkl"

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
COLLECTION = "enterprise_docs"

TOP_DENSE = 8
TOP_BM25 = 8
TOP_FINAL = 6
RRF_K = 60


def client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing. Configure .env.")
    return Groq(api_key=key)


def ask_json(system, prompt):
    res = client().chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    raw = res.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise RuntimeError("LLM did not return valid JSON: " + raw)
        return json.loads(m.group())


def clean(s):
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def heading(text):
    for line in text.splitlines():
        x = line.strip().lstrip("#").strip()
        if 3 <= len(x) <= 140:
            return x
    return "General"


def load_file(path, base_meta):
    ext = path.suffix.lower()
    docs = []

    if ext == ".pdf":
        for n, page in enumerate(PdfReader(str(path)).pages, 1):
            t = clean(page.extract_text() or "")
            if t:
                docs.append(Document(page_content=t, metadata={**base_meta, "page_number": n}))
    elif ext in (".txt", ".md"):
        t = clean(path.read_text(encoding="utf-8", errors="ignore"))
        if t:
            docs = [Document(page_content=t, metadata={**base_meta, "page_number": 1})]
    elif ext in (".html", ".htm"):
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else base_meta["document_title"]
        t = clean(soup.get_text("\n", strip=True))
        if t:
            docs = [
                Document(
                    page_content=t,
                    metadata={**base_meta, "document_title": title, "page_number": 1},
                )
            ]
    else:
        raise ValueError(f"Unsupported file: {ext}")

    for d in docs:
        # Best-effort heading metadata; complex layouts need format-specific parsing.
        heads = re.findall(r"(?m)^\s{0,3}#{1,6}\s+(.+)$", d.page_content)
        d.metadata["section"] = heads[-1][:180] if heads else heading(d.page_content)

    return docs


def embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def vectorstore():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embedder(),
        persist_directory=str(CHROMA),
    )


def ingest(path, doc_type="General", version="1.0", effective_date="unknown", status="active"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {
        "document_id": digest[:16],
        "document_title": path.stem.replace("_", " "),
        "source": path.name,
        "document_type": doc_type,
        "version": version,
        "effective_date": effective_date or "unknown",
        "uploaded_date": datetime.now(timezone.utc).date().isoformat(),
        "status": status,
    }
    pages = load_file(path, meta)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=140,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    ids = []
    for i, d in enumerate(chunks):
        d.metadata["chunk_id"] = i
        d.metadata["start_index"] = int(d.metadata.get("start_index", 0))
        ids.append(f"{digest[:16]}_{i}")

    if chunks:
        vectorstore().add_documents(chunks, ids=ids)

    rebuild_bm25()
    return len(chunks)


def tokenize(s):
    return re.findall(r"[A-Za-z0-9_+#.-]+", s.lower())


def records_from_chroma():
    data = vectorstore().get(include=["documents", "metadatas"])
    return [
        {"text": t, "metadata": m or {}}
        for t, m in zip(data.get("documents") or [], data.get("metadatas") or [])
    ]


def rebuild_bm25():
    records = records_from_chroma()
    if not records:
        for p in (BM25_FILE, RECORDS_FILE):
            if p.exists():
                p.unlink()
        return

    bm = BM25Okapi([tokenize(x["text"]) for x in records])
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm, f)
    with open(RECORDS_FILE, "wb") as f:
        pickle.dump(records, f)


def retrieve(query):
    if not RECORDS_FILE.exists():
        rebuild_bm25()
    if not RECORDS_FILE.exists():
        return []

    with open(RECORDS_FILE, "rb") as f:
        recs = pickle.load(f)
    with open(BM25_FILE, "rb") as f:
        bm = pickle.load(f)

    dense = vectorstore().similarity_search(query, k=min(TOP_DENSE, len(recs)))

    # Build identity map for matching dense hits to BM25 corpus records.
    lookup = {}
    for i, r in enumerate(recs):
        m = r["metadata"]
        lookup[(r["text"], m.get("document_id"), m.get("page_number"), m.get("chunk_id"))] = i

    fused = {}
    for rank, d in enumerate(dense, 1):
        m = d.metadata
        idx = lookup.get(
            (d.page_content, m.get("document_id"), m.get("page_number"), m.get("chunk_id"))
        )
        if idx is not None:
            fused[idx] = fused.get(idx, 0) + 1 / (RRF_K + rank)

    sparse = np.argsort(bm.get_scores(tokenize(query)))[::-1][:TOP_BM25]
    for rank, idx in enumerate(sparse, 1):
        fused[int(idx)] = fused.get(int(idx), 0) + 1 / (RRF_K + rank)

    return [
        {
            "text": recs[i]["text"],
            "metadata": recs[i]["metadata"],
            "rrf_score": round(float(score), 6),
        }
        for i, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:TOP_FINAL]
    ]


class State(TypedDict, total=False):
    question: str
    history: str
    ambiguity: dict
    retrieved: list
    validation: dict
    answer: dict
    status: str


AMBIGUITY_SYSTEM_PROMPT = """You are a gatekeeper that decides whether a user question can be sent to a document search system as-is.

Your ONLY job is to judge the clarity of the question. You are NOT judging whether the answer exists, whether it is in the knowledge base, or whether the question is on-topic. You have not seen any documents, so never reason about what the documents might or might not contain.

DEFAULT TO CLEAR. Only mark is_ambiguous=true when BOTH conditions hold:
  1. The question cannot be turned into a useful search query as written, AND
  2. The reason is one of these:
     a. A pronoun/reference ("it", "that policy", "the second one") that cannot be resolved from the conversation history.
     b. A required entity is missing (e.g. "What is the price?" with no product mentioned anywhere in the history).
     c. The question has two or more materially different interpretations that would lead to different searches, and history does not disambiguate.

Do NOT mark as ambiguous when:
  - The question is specific but may have no answer in the documents.
  - The question is broad or general but still has one clear meaning ("Tell me about the leave policy").
  - The question is about an obscure, unusual, or possibly nonexistent topic.
  - The question is short, informal, has typos, or is grammatically imperfect but the intent is understandable.
  - You merely wish for more detail. Optional detail is not missing information.
  - The question is off-topic or something you can't answer. That's not your concern here.

Always resolve references using the conversation history first. If history resolves the reference, the question is CLEAR.

Examples:
  Q: "What is the notice period for resignation?" -> is_ambiguous=false
  Q: "Who won the 2031 cricket world cup?" -> is_ambiguous=false (clear, even if unanswerable)
  Q: "What is its cost?" with empty history -> is_ambiguous=true (missing entity)
  Q: "What is its cost?" with history discussing "Plan A" -> is_ambiguous=false
  Q: "Tell me about the policy" with history listing 3 different policies and no focus -> is_ambiguous=true
  Q: "leave policy?" -> is_ambiguous=false

Return ONLY valid JSON:
{"is_ambiguous": bool,
 "reason": str,
 "missing_information": [str],
 "clarification_question": str}

If is_ambiguous is false: missing_information must be [] and clarification_question must be "".
If is_ambiguous is true: ask ONE short, specific clarification question."""


def ambiguity_node(s):
    result = ask_json(
        AMBIGUITY_SYSTEM_PROMPT,
        f"Conversation history:\n{s.get('history') or '(none)'}\n\nQuestion:\n{s['question']}",
    )

    # Safety net: only trust the flag if the model gave a real clarification
    is_ambiguous = bool(
        result.get("is_ambiguous")
        and result.get("missing_information")
        and result.get("clarification_question", "").strip()
    )

    return {
        "ambiguity": result,
        "status": "AMBIGUOUS" if is_ambiguous else "CLEAR",
    }


def retrieval_node(s):
    return {"retrieved": retrieve(s["question"])}


def validation_node(s):
    docs = s.get("retrieved", [])
    if not docs:
        return {
            "validation": {
                "evidence_sufficient": False,
                "has_conflict": False,
                "reason": "No passages retrieved.",
                "conflicts": [],
            },
            "status": "NOT_FOUND",
        }

    evidence = []
    for i, d in enumerate(docs, 1):
        m = d["metadata"]
        evidence.append(
            {
                "id": i,
                "source": m.get("source"),
                "section": m.get("section"),
                "page": m.get("page_number"),
                "version": m.get("version"),
                "effective_date": m.get("effective_date"),
                "status": m.get("status"),
                "text": d["text"],
            }
        )

    result = ask_json(
        """Assess whether passages directly answer the question and whether they truly conflict.
Return JSON: {"evidence_sufficient":bool,"reason":str,"has_conflict":bool,"conflicts":[{"topic":str,"evidence_ids":[int],"description":str,"resolution":str}]}.
Only call claims contradictory if they concern the same scope and incompatible facts. Different regions, products, employee groups or effective periods may explain differences. Do not assume latest upload is authoritative. Use effective date/version/status only when metadata supports precedence. Keyword overlap alone is insufficient.""",
        f"Question:\n{s['question']}\nEvidence:\n{json.dumps(evidence, ensure_ascii=False)}",
    )
    return {
        "validation": result,
        "status": "READY" if result.get("evidence_sufficient") else "NOT_FOUND",
    }


def answer_node(s):
    evidence = []
    for i, d in enumerate(s["retrieved"], 1):
        m = d["metadata"]
        evidence.append(
            {
                "citation_id": i,
                "source": m.get("source"),
                "title": m.get("document_title"),
                "section": m.get("section"),
                "page": m.get("page_number"),
                "version": m.get("version"),
                "effective_date": m.get("effective_date"),
                "excerpt": d["text"],
            }
        )

    result = ask_json(
        """Answer only from evidence. Do not invent. Make unresolved conflicts visible. Every material claim needs a citation.
Return JSON: {"answer":str,"status":"ANSWERED" or "CONFLICT","citations":[{"citation_id":int,"claim_supported":str}],"limitations":[str]}.""",
        f"Question:\n{s['question']}\nValidation:\n{json.dumps(s.get('validation', {}), ensure_ascii=False)}\nEvidence:\n{json.dumps(evidence, ensure_ascii=False)}",
    )
    result["evidence"] = evidence
    return {"answer": result, "status": result.get("status", "ANSWERED")}


def clarify_node(s):
    a = s.get("ambiguity", {})
    return {
        "answer": {
            "answer": a.get("clarification_question") or "Could you clarify your question?",
            "status": "AMBIGUOUS",
            "citations": [],
            "limitations": [a.get("reason", "More context is required.")],
            "missing_information": a.get("missing_information", []),
        }
    }


def not_found_node(s):
    return {
        "answer": {
            "answer": "I could not find sufficient supporting evidence in the supplied documents to answer reliably.",
            "status": "NOT_FOUND",
            "citations": [],
            "limitations": [s.get("validation", {}).get("reason", "Insufficient evidence.")],
        }
    }


def route_ambiguity(s):
    return "clarify" if s.get("status") == "AMBIGUOUS" else "retrieve"


def route_evidence(s):
    return "answer" if s.get("status") == "READY" else "not_found"


def build_graph():
    g = StateGraph(State)
    nodes = [
        ("ambiguity", ambiguity_node),
        ("clarify", clarify_node),
        ("retrieve", retrieval_node),
        ("validate", validation_node),
        ("answer", answer_node),
        ("not_found", not_found_node),
    ]
    for name, fn in nodes:
        g.add_node(name, fn)

    g.add_edge(START, "ambiguity")
    g.add_conditional_edges("ambiguity", route_ambiguity, {"clarify": "clarify", "retrieve": "retrieve"})
    g.add_edge("clarify", END)
    g.add_edge("retrieve", "validate")
    g.add_conditional_edges("validate", route_evidence, {"answer": "answer", "not_found": "not_found"})
    g.add_edge("answer", END)
    g.add_edge("not_found", END)
    return g.compile()


def run_question(question, history=""):
    out = build_graph().invoke({"question": question, "history": history})
    return {
        "status": out.get("status", out.get("answer", {}).get("status", "UNKNOWN")),
        "answer": out.get("answer", {}),
        "retrieved": out.get("retrieved", []),
        "ambiguity": out.get("ambiguity", {}),
        "validation": out.get("validation", {}),
    }


def indexed_documents():
    data = vectorstore().get(include=["metadatas"])
    unique = {}
    for m in data.get("metadatas") or []:
        if m:
            unique[m.get("document_id", m.get("source", "unknown"))] = m
    return list(unique.values())
