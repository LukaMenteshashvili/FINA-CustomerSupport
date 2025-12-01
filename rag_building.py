import os
import json
import shutil
from typing import List, Dict, Any, Optional, Tuple

import chromadb
import networkx as nx
from chromadb.config import Settings
import google.generativeai as genai

# ----------------------------
# CONFIG
# ----------------------------
PATH_CHUNKS = "fina_company_profile_chunks.jsonl"
PATH_TRIPLES = "fina_knowledge_graph.jsonl"
PATH_FAQ = "fina_faq_prompts.jsonl"
PATH_SCHEMA = "fina_schema_readme.json"
# Additional chunk sources (same schema as PATH_CHUNKS).
EXTRA_CHUNK_PATHS = [
    "fina_website_chunks.jsonl",
]

EMBED_MODEL = "text-embedding-004"  # 768-dim
CHAT_MODEL = "gemini-2.5-flash"

COLL_CHUNKS = "fina_chunks"
COLL_FAQ = "fina_faq"

# Reset strategy
RESET_COLLECTIONS_FIRST = False
NUKE_PERSIST_DIR = False
PERSIST_DIR = "./chroma_data"

# ----------------------------
# GEMINI & CHROMA SETUP
# ----------------------------
_GENAI_CONFIGURED = False


def _ensure_genai_config():
    """Configure Gemini client lazily to avoid import-time failures."""
    global _GENAI_CONFIGURED
    if _GENAI_CONFIGURED:
        return
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError('GOOGLE_API_KEY not set. In PowerShell: setx GOOGLE_API_KEY "YOUR_KEY" and restart terminal.')
    genai.configure(api_key=key)
    _GENAI_CONFIGURED = True


if NUKE_PERSIST_DIR and os.path.isdir(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

chroma = chromadb.PersistentClient(
    path=PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False),
)

# ----------------------------
# UTILITIES
# ----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed with Gemini (768-dim)."""
    _ensure_genai_config()
    vecs = []
    for t in texts:
        r = genai.embed_content(model=EMBED_MODEL, content=t)
        vecs.append(r["embedding"])
    return vecs


def to_primitive_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    """Make metadata Chroma-safe (no lists/dicts)."""
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


def reset_collections(names: List[str]):
    for name in names:
        try:
            chroma.delete_collection(name)
            print(f"[reset] Deleted collection '{name}'.")
        except Exception:
            pass


# ----------------------------
# 1) INDEX CHUNKS
# ----------------------------
def _collect_chunk_records(main_path: str, extra_paths: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load chunk records from provided paths and return records with per-file counts."""
    recs: List[Dict[str, Any]] = []
    counts: List[str] = []
    paths = [main_path] + list(extra_paths)
    for p in paths:
        if os.path.exists(p):
            parsed = read_jsonl(p)
            recs.extend(parsed)
            counts.append(f"{os.path.basename(p)}={len(parsed)}")
        else:
            print(f"[chunks] Skip missing path: {p}")
    return recs, counts


def index_chunks(
    main_path: str = PATH_CHUNKS,
    extra_paths: List[str] = EXTRA_CHUNK_PATHS,
    collection_name: str = COLL_CHUNKS,
    batch: int = 64,
):
    recs, path_counts = _collect_chunk_records(main_path, extra_paths)

    if not recs:
        print("[chunks] No records found in any chunk files, nothing to index.")
        return chroma.get_or_create_collection(collection_name)

    coll = chroma.get_or_create_collection(collection_name)

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

    print(f"[chunks] Indexed {len(recs)} records into '{collection_name}'. Sources: {', '.join(path_counts)}")
    return coll


# ----------------------------
# 2) KNOWLEDGE GRAPH
# ----------------------------
def load_graph(path=PATH_TRIPLES) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    if not os.path.exists(path):
        print(f"[graph] File not found, skipping load: {path}")
        return G
    for t in read_jsonl(path):
        head, rel, tail = t["head"], t["relation"], t["tail"]
        G.add_edge(head, tail, relation=rel, prov=t.get("provenance", {}).get("chunk_id"))
    print(f"[graph] Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def graph_expand(G: nx.MultiDiGraph, entity: str, max_edges: int = 30) -> List[str]:
    facts = []
    if entity in G:
        for _, tail, d in G.out_edges(entity, data=True):
            facts.append(f"{entity} --{d.get('relation')}--> {tail} (prov: {d.get('prov')})")
            if len(facts) >= max_edges:
                break
    return facts


# ----------------------------
# 3) INDEX FAQ
# ----------------------------
def index_faq(path=PATH_FAQ, collection_name=COLL_FAQ, batch=50):
    faqs = read_jsonl(path)
    if not faqs:
        print(f"[faq] No FAQ records found in {path}.")
        return chroma.get_or_create_collection(collection_name)

    coll = chroma.get_or_create_collection(collection_name)
    for i in range(0, len(faqs), batch):
        b = faqs[i : i + batch]
        ids = [f"{q['doc_id']}::faq::{i+j}" for j, q in enumerate(b)]
        docs = [q["question"] for q in b]
        metas = [
            to_primitive_meta(
                {
                    "answer": q["answer"],
                    "source_chunk_ids": q.get("source_chunk_ids", []),
                }
            )
            for q in b
        ]
        embs = embed_texts(docs)
        coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(f"[faq] Indexed {len(faqs)} QAs into '{collection_name}'.")
    return chroma.get_or_create_collection(collection_name)


def faq_search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    faq_coll = chroma.get_or_create_collection(COLL_FAQ)
    if faq_coll.count() == 0:
        return []
    qemb = embed_texts([query])
    res = faq_coll.query(query_embeddings=qemb, n_results=k, include=["metadatas", "documents", "distances"])
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append(
            {
                "id": res["ids"][0][i],
                "question": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                "score": res["distances"][0][i] if "distances" in res else None,
            }
        )
    return hits


# ----------------------------
# 4) RETRIEVAL + CONTEXT
# ----------------------------
def search_chunks(query: str, k: int = 8, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    coll = chroma.get_or_create_collection(COLL_CHUNKS)
    if coll.count() == 0:
        return []
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


def _first_entity_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    ecsv = meta.get("entities_csv")
    if not ecsv:
        return None
    parts = [p.strip() for p in ecsv.split("|") if p.strip()]
    return parts[0] if parts else None


def compose_context(query: str, G: Optional[nx.MultiDiGraph] = None, max_sections: int = 4) -> Dict[str, Any]:
    top_hits = search_chunks(query, k=12)
    seen_sections, picked = set(), []
    for h in top_hits:
        sec = h["meta"].get("section")
        if sec not in seen_sections:
            picked.append(h)
            seen_sections.add(sec)
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

    return {"context_text": "\n\n".join(ctx_blocks), "graph_facts": graph_facts}


# ----------------------------
# 5) GEMINI ANSWER
# ----------------------------
def answer_with_gemini(query: str, schema_path: str = PATH_SCHEMA, G: Optional[nx.MultiDiGraph]=None) -> str:
    schema_note = ""
    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_note = f.read()[:800]
        except Exception:
            pass

    pack = compose_context(query, G=G, max_sections=4)
    context_text = pack["context_text"]

    # 1) System style and role
    sys = (
        "You are FINA AI, a helpful and friendly assistant for FINA LLC. "
        "You speak to website visitors in clear, natural language. "
        "You only use the information provided in the context as your source of truth. "
        "If something is not covered in the context, say that you do not have that information instead of guessing."
    )

    # 2) Prompt that controls tone and formatting
    prompt = (
        f"{sys}\n\n"
        f"SCHEMA HINT (optional):\n{schema_note}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"USER QUESTION: {query}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer in a warm, natural tone, as if you are chatting with a person on the website.\n"
        "- Use full sentences and short paragraphs. Use bullet points only when it really helps clarity.\n"
        "- Do not include any [CHUNK ...] or chunk ids in the main answer text.\n"
        "- Do not include any 'Sources' section or any chunk ids (for example company_overview:01, web:...:chunk_02) in your answer.\n"
        "- If the context does not fully answer the question, say what you can from the context and be transparent about what is missing.\n"
    )

    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(prompt)

    answer_text = resp.text.strip()

    return answer_text



# ----------------------------
# 6) COLLECTION READY CHECK
# ----------------------------
def ensure_collections_ready(reset: bool = False) -> Dict[str, int]:
    """
    Make sure the required Chroma collections exist and have data.
    Re-index if counts do not match the JSONL sources or if reset is requested.
    """
    if reset:
        reset_collections([COLL_CHUNKS, COLL_FAQ])

    # Expected counts from JSONL files
    chunk_recs, chunk_counts = _collect_chunk_records(PATH_CHUNKS, EXTRA_CHUNK_PATHS)
    expected_chunks = len(chunk_recs)
    faq_recs = read_jsonl(PATH_FAQ) if os.path.exists(PATH_FAQ) else []
    expected_faq = len(faq_recs)

    # Chunks
    chunks = chroma.get_or_create_collection(COLL_CHUNKS)
    if reset or chunks.count() != expected_chunks:
        if not reset:
            print(f"[ensure] Reindexing chunks (current={chunks.count()}, expected={expected_chunks}). Sources: {', '.join(chunk_counts)}")
        reset_collections([COLL_CHUNKS])
        index_chunks()
        chunks = chroma.get_or_create_collection(COLL_CHUNKS)

    # FAQ
    faq = chroma.get_or_create_collection(COLL_FAQ)
    if reset or faq.count() != expected_faq:
        if not reset:
            print(f"[ensure] Reindexing FAQ (current={faq.count()}, expected={expected_faq}).")
        reset_collections([COLL_FAQ])
        index_faq()
        faq = chroma.get_or_create_collection(COLL_FAQ)

    return {"chunks": chunks.count(), "faq": faq.count()}


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    if RESET_COLLECTIONS_FIRST:
        reset_collections([COLL_CHUNKS, COLL_FAQ])

    index_chunks()
    G = load_graph()
    index_faq()

    print("\n=== TEST 1: Product definition ===")
    print(answer_with_gemini("What is FINA IRP and who is it for?", G=G))

    print("\n=== TEST 2: Countries implemented ===")
    print(answer_with_gemini("List countries where FINA IRP is implemented.", G=G))

    print("\n=== TEST 3: Services overview ===")
    print(answer_with_gemini("Summarize FINA's core services in 4 bullets.", G=G))
