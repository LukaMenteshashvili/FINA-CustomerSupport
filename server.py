from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from rag_building import (
    answer_with_gemini,
    load_graph,
    index_chunks,
    index_faq,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


G = None


@app.on_event("startup")
def startup_event():
    global G
    print("[server] Indexing chunks and FAQ (safe every startup)...")
    try:
        index_chunks()   # creates or updates 'fina_chunks'
        index_faq()      # creates or updates 'fina_faq'
    except Exception as e:
        print("[server] Error during indexing:", e)

    print("[server] Loading knowledge graph...")
    G = load_graph()
    print("[server] Startup complete.")


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        text = req.message.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty message")

        if G is None:
            raise HTTPException(status_code=500, detail="Graph not initialized")

        answer = answer_with_gemini(text, G=G)
        return {"reply": answer}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
