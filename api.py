"""FastAPI backend + statik arayüz."""
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import database
import diet_rules
import recipe_parser
import recommender
import search_engine

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
MAX_QUESTION = 300
MAX_BAG = 500
MAX_FILES = 20

app = FastAPI(title="Tarif Defteri Asistanı", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_conn = None
_rag = None


def get_conn():
    global _conn
    if _conn is None:
        _conn = database.get_connection()
        database.seed_if_empty(_conn)
    return _conn


def _rag_index():
    global _rag
    if _rag is None:
        from embeddings import get_embedding_client
        from main import load_knowledge_items

        client = get_embedding_client()
        items = load_knowledge_items()
        docs = [item["content"] for item in items]
        sources = [item["source"] for item in items]
        embs = [item.embedding for item in client.generate_embeddings(docs).data]
        _rag = (client, docs, embs, sources)
    return _rag


class RecipeOut(BaseModel):
    id: int
    title: str
    source_file: str
    ingredients: list[str]
    diet: str
    full_text: str | None = None
    overlap_percentage: int | None = None
    missing_ingredients: list[str] | None = None


class SearchOut(BaseModel):
    question: str
    answer: str
    recipes: list[RecipeOut]


class MatchIn(BaseModel):
    ingredients: str = Field(..., min_length=1, max_length=MAX_BAG)


def to_recipe(recipe, include_text=False):
    data = {
        "id": recipe["id"],
        "title": recipe["title"],
        "source_file": recipe["source_file"],
        "ingredients": recipe["ingredients"],
        "diet": diet_rules.diet_label(recipe),
    }
    if include_text:
        data["full_text"] = recipe["full_text"]
    if "overlap_percentage" in recipe:
        data["overlap_percentage"] = int(recipe["overlap_percentage"])
        data["missing_ingredients"] = recipe.get("missing_ingredients", [])
    return data


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/recipes", response_model=list[RecipeOut])
def list_recipes():
    recipes = database.get_all_recipes(get_conn())
    return [to_recipe(recipe, include_text=True) for recipe in recipes]


@app.get("/api/search", response_model=SearchOut)
def search(q: str = Query("", max_length=MAX_QUESTION)):
    question = q.strip()[:MAX_QUESTION]
    if not question:
        return {"question": question, "answer": search_engine.build_answer(question, []), "recipes": []}
    from main import resolve_answer

    client, docs, embs, sources = _rag_index()
    answer, hit_sources = resolve_answer(
        question,
        client,
        embs,
        docs=docs,
        sources=sources,
        conn=get_conn(),
    )
    by_file = {recipe["source_file"]: recipe for recipe in database.get_all_recipes(get_conn())}
    found = [by_file[name] for name in hit_sources if name in by_file]
    if not found:
        found = search_engine.search_recipes(get_conn(), question, limit=8)
    return {
        "question": question,
        "answer": answer,
        "recipes": [to_recipe(recipe, include_text=True) for recipe in found],
    }


@app.post("/api/match", response_model=list[RecipeOut])
def match(payload: MatchIn):
    bag = payload.ingredients.strip()[:MAX_BAG]
    matches = recommender.match_by_ingredients(get_conn(), bag, top_n=5)
    return [to_recipe(recipe) for recipe in matches]


@app.post("/api/recipes/upload")
async def upload(files: list[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"En fazla {MAX_FILES} dosya.")
    saved, errors = [], []
    conn = get_conn()
    for uploaded in files:
        try:
            raw = await uploaded.read()
            source = recipe_parser.safe_filename(uploaded.filename)
            title, text, ingredients = recipe_parser.parse_recipe_bytes(source, raw)
            database.upsert_recipe(conn, source, title, text, ingredients)
            saved.append(source)
        except ValueError as exc:
            errors.append({"file": uploaded.filename, "error": str(exc)})
    return {"saved": saved, "errors": errors}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
