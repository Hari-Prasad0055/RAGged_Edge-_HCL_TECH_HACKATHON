from datetime import date
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from rag_engine import (
    DATA,
    MODEL,
    indexed_documents,
    ingest,
    rebuild_bm25,
    run_question,
)

load_dotenv()

st.set_page_config(
    page_title="Enterprise Documentation Q&A",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Enterprise Documentation Q&A Assistant")
st.caption("LangChain · ChromaDB · MiniLM · BM25 · RRF · LangGraph · Groq")

with st.sidebar:
    st.header("Document ingestion")
    uploads = st.file_uploader(
        "Upload PDF, HTML, MD or TXT",
        type=["pdf", "html", "htm", "md", "txt"],
        accept_multiple_files=True,
    )
    dtype = st.selectbox(
        "Document type",
        ["General", "Policy", "Product Manual", "FAQ", "Release Notes", "Support Article"],
    )
    version = st.text_input("Version", "1.0")
    eff = st.date_input("Effective date", date.today())
    status = st.selectbox("Status", ["active", "superseded", "draft", "archived"])

    if st.button("Upload ", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Upload at least one document.")
        else:
            total = 0
            for f in uploads:
                p = DATA / Path(f.name).name
                p.write_bytes(f.getbuffer())
                try:
                    total += ingest(p, dtype, version, eff.isoformat(), status)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            st.success(f"Indexed {total} chunks.")


ask_tab, docs_tab= st.tabs(["Ask questions", "Documents"])

with ask_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = ""

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("details"):
                with st.expander("Diagnostics"):
                    st.json(msg["details"])

    q = st.chat_input("Ask about the uploaded documentation...")
    if q:
        st.session_state.messages.append({"role": "user", "text": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Checking ambiguity and evidence..."):
                    result = run_question(q, st.session_state.history)

                obj = result["answer"]
                state = obj.get("status", result["status"])
                st.markdown(f"**Status: `{state}`**")
                st.write(obj.get("answer", "No response generated."))

                citations = obj.get("citations", [])
                evidence = obj.get("evidence", [])
                for c in citations:
                    item = next(
                        (x for x in evidence if x["citation_id"] == c.get("citation_id")),
                        None,
                    )
                    if item:
                        st.markdown(
                            f"**[{item['citation_id']}] {item.get('title')}** — "
                            f"{item.get('source')}, page {item.get('page')}, section: {item.get('section')}"
                        )
                        st.caption(c.get("claim_supported", ""))
                        with st.expander(f"View cited excerpt [{item['citation_id']}]"):
                            st.write(item.get("excerpt", ""))

                if obj.get("limitations"):
                    st.info(" ".join(obj["limitations"]))

                details = {
                    "ambiguity": result["ambiguity"],
                    "validation": result["validation"],
                    "retrieved_count": len(result["retrieved"]),
                }
                with st.expander("Diagnostics"):
                    st.json(details)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "text": f"**{state}**\n\n{obj.get('answer', '')}",
                        "details": details,
                    }
                )
                st.session_state.history = (
                    st.session_state.history
                    + f"\nUser: {q}\nAssistant: {obj.get('answer', '')}\n"
                )[-5000:]
            except Exception as e:
                st.error(f"Error: {e}")

with docs_tab:
    docs = indexed_documents()
    if docs:
        st.dataframe(
            [
                {
                    "Title": m.get("document_title"),
                    "Source": m.get("source"),
                    "Type": m.get("document_type"),
                    "Version": m.get("version"),
                    "Effective date": m.get("effective_date"),
                    "Status": m.get("status"),
                }
                for m in docs
            ],
            use_container_width=True,
        )
    else:
        st.info("No indexed documents. Upload documents in the sidebar.")


