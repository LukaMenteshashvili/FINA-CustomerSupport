"""FastAPI service that exposes the FINA chatbot for Framer embeds.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

The service reuses the same Gemini + Chroma RAG logic as the Streamlit app but
presents a simple JSON API that front-ends (like Framer custom code widgets) can
call.
"""
import os
import json
import time
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import networkx as nx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------- CONFIG ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

genai.configure(api_key=GOOGLE_API_KEY)

PERSIST_DIR = "./chroma_data"
os.makedirs(PERSIST_DIR, exist_ok=True)

DEFAULT_PATH_CHUNKS = "fina_company_profile_chunks.jsonl"
DEFAULT_PATH_TRIPLES = "fina_knowledge_graph.jsonl"
DEFAULT_PATH_FAQ = "fina_faq_prompts.jsonl"
DEFAULT_PATH_SCHEMA = "fina_schema_readme.json"

EMBED_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"

COLL_CHUNKS = "fina_chunks"
COLL_FAQ = "fina_faq"

# ---------- TYPES ----------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2)
    sections_filter: List[str] = Field(default_factory=list, description="Optional list of section names to filter retrieval")
    include_chunks: bool = Field(default=False, description="If true, return the retrieved chunk metadata")


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks: Optional[List[Dict[str, Any]]]
    latency_ms: float


# ---------- HELPERS ----------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def embed_texts(texts: List[str]) -> List[List[float]]:
    vecs: List[List[float]] = []
    for t in texts:
        r = genai.embed_content(model=EMBED_MODEL, content=t)
        vecs.append(r["embedding"])
    return vecs


def to_primitive_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple, set)):
            out[f"{k}_csv"] = "|".join(str(x) for x in v)
            out[f"{k}_count"] = len(v)
        elif isinstance(v, dict):
            out[f"{k}_json"] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def get_chroma():
    return chromadb.Client(Settings(persist_directory=PERSIST_DIR, anonymized_telemetry=False))


def load_graph(path_triples: str) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    if not os.path.exists(path_triples):
        return G
    for t in read_jsonl(path_triples):
        head, rel, tail = t["head"], t["relation"], t["tail"]
        G.add_edge(head, tail, relation=rel, prov=t.get("provenance", {}).get("chunk_id"))
    return G


def graph_expand(G: nx.MultiDiGraph, entity: str, max_edges: int = 20) -> List[str]:
    facts: List[str] = []
    if entity in G:
        for _, tail, d in G.out_edges(entity, data=True):
            facts.append(f"{entity} —{d.get('relation')}→ {tail} (prov: {d.get('prov')})")
            if len(facts) >= max_edges:
                break
    return facts


def _first_entity_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    ecsv = meta.get("entities_csv")
    if not ecsv:
        return None
    parts = [p.strip() for p in ecsv.split("|") if p.strip()]
    return parts[0] if parts else None


def build_where_from_sections(sections: List[str]) -> Optional[Dict[str, Any]]:
    if not sections:
        return None
    return {"$or": [{"section": {"$eq": s}} for s in sections]}


# ---------- INDEXING ----------
def reset_collections(chroma_client, names: List[str]):
    for name in names:
        try:
            chroma_client.delete_collection(name)
        except Exception:
            # ignore missing collections
            pass


def index_chunks(chroma_client, path_chunks: str, batch: int = 64):
    recs = read_jsonl(path_chunks)
    coll = chroma_client.get_or_create_collection(COLL_CHUNKS)
    for i in range(0, len(recs), batch):
        batch_recs = recs[i : i + batch]
        ids = [f"{r['doc_id']}::{r['chunk_id']}" for r in batch_recs]
        docs = [r["text"] for r in batch_recs]
        metas = [
            to_primitive_meta(
                {
                    "doc_id": r.get("doc_id"),
                    "chunk_id": r.get("chunk_id"),
                    "section": r.get("section"),
                    "title": r.get("title"),
                    "entities": r.get("entities", []),
                    "keywords": r.get("keywords", []),
                }
            )
            for r in batch_recs
        ]
        embs = embed_texts(docs)
        coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(recs)


def index_faq(chroma_client, path_faq: str, batch: int = 64):
    faqs = read_jsonl(path_faq)
    coll = chroma_client.get_or_create_collection(COLL_FAQ)
    for i in range(0, len(faqs), batch):
        b = faqs[i : i + batch]
        ids = [f"{q['doc_id']}::faq::{i+j}" for j, q in enumerate(b)]
        docs = [q["question"] for q in b]
        metas = [
            to_primitive_meta({"answer": q["answer"], "source_chunk_ids": q.get("source_chunk_ids", [])})
            for q in b
        ]
        embs = embed_texts(docs)
        coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(faqs)


# ---------- RETRIEVAL ----------
def search_chunks(chroma_client, query: str, k: int = 8, where: Optional[Dict[str, Any]] = None):
    coll = chroma_client.get_collection(COLL_CHUNKS)
    qemb = embed_texts([query])
    kwargs: Dict[str, Any] = {"query_embeddings": qemb, "n_results": k, "include": ["metadatas", "documents", "distances"]}
    if isinstance(where, dict) and len(where) > 0:
        kwargs["where"] = where
    res = coll.query(**kwargs)
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append(
            {
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                "score": res["distances"][0][i] if "distances" in res else None,
            }
        )
    return hits


def compose_context(chroma_client, query: str, G: Optional[nx.MultiDiGraph], sections_filter: List[str], k: int = 12, max_sections: int = 4):
    where = build_where_from_sections(sections_filter)
    top_hits = search_chunks(chroma_client, query, k=k, where=where)
    seen, picked = set(), []
    for h in top_hits:
        sec = h["meta"].get("section")
        if sec not in seen:
            picked.append(h)
            seen.add(sec)
        if len(picked) >= max_sections:
            break

    graph_facts = []
    if G and picked:
        first_ent = _first_entity_from_meta(picked[0]["meta"])
        if first_ent:
            graph_facts = graph_expand(G, first_ent, max_edges=12)

    ctx_blocks = []
    for h in picked:
        ctx_blocks.append(f"[CHUNK {h['meta']['chunk_id']} | {h['meta']['title']}] {h['text']}")
    if graph_facts:
        ctx_blocks.append("[GRAPH FACTS]\n" + "\n".join(f"- {f}" for f in graph_facts))

    return {"context_text": "\n\n".join(ctx_blocks), "used_chunks": picked, "graph_facts": graph_facts}


def answer_with_gemini(chroma_client, query: str, schema_path: str, G: Optional[nx.MultiDiGraph], sections_filter: List[str]):
    schema_note = ""
    if schema_path and os.path.exists(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_note = f.read()[:800]
        except Exception:
            pass

    pack = compose_context(chroma_client, query, G=G, sections_filter=sections_filter, k=12, max_sections=4)
    context_text, used = pack["context_text"], pack["used_chunks"]

    sys = "You are a factual assistant for FINA LLC. Use ONLY the provided context. Cite sources as [CHUNK id]. If information is missing, say so plainly."
    prompt = (
        f"{sys}\n\n"
        f"SCHEMA HINT (optional):\n{schema_note}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"USER QUESTION: {query}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer concisely and specifically.\n"
        "- Use bullet points for lists.\n"
        "- End with a 'Sources:' line listing CHUNK ids used.\n"
    )

    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(prompt)
    srcs = sorted({h["meta"].get("chunk_id") for h in used if h.get("meta")})
    return resp.text.strip(), used, srcs


# ---------- APP ----------
app = FastAPI(title="FINA Chatbot API")


@app.on_event("startup")
async def _startup():
    global chroma_client, G
    chroma_client = get_chroma()
    G = load_graph(DEFAULT_PATH_TRIPLES)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    try:
        answer, used, srcs = answer_with_gemini(
            chroma_client=chroma_client,
            query=req.question.strip(),
            schema_path=DEFAULT_PATH_SCHEMA,
            G=G,
            sections_filter=req.sections_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start) * 1000
    payload = {
        "answer": answer,
        "sources": srcs,
        "chunks": used if req.include_chunks else None,
        "latency_ms": round(latency_ms, 2),
    }
    return payload


@app.post("/reindex")
async def reindex():
    reset_collections(chroma_client, [COLL_CHUNKS, COLL_FAQ])
    n_chunks = index_chunks(chroma_client, DEFAULT_PATH_CHUNKS)
    n_faq = index_faq(chroma_client, DEFAULT_PATH_FAQ)
    return {"chunks_indexed": n_chunks, "faq_indexed": n_faq}
