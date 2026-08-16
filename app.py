"""Streamlit arayüzü."""
import html

import streamlit as st

import database
import diet_rules
import recipe_parser
import recommender
import search_engine
from knowledge_store import KB_FOLDER

MAX_UPLOAD_FILES = 20
MAX_CHAT = 40
MAX_QUESTION_LEN = 300
MAX_BAG_LEN = 500

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Nunito:wght@400;600;700&display=swap');
:root, [data-theme="dark"], [data-theme="light"] {
  --secondary-background-color: #F8E9A6 !important;
  --background-color: #FBF3E7 !important;
  --text-color: #6F4A26 !important;
}
.stApp, .stAppViewContainer, .main, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
  background:#FBF3E7 !important; color:#6F4A26 !important; font-family:'Nunito',sans-serif;
}
h1,h2,h3 { font-family:'Fraunces',serif !important; color:#B5541E !important; }
[data-testid="stSidebar"], [data-testid="stSidebarContent"], section[data-testid="stSidebar"] > div {
  background:#F3E3CC !important;
}
header[data-testid="stHeader"], .stAppHeader, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
div[data-testid="stHeader"] {
  background:#F8E9A6 !important; background-color:#F8E9A6 !important; color:#6F4A26 !important;
  border:none !important;
}
header[data-testid="stHeader"] * { color:#6F4A26 !important; }
[data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div,
section[data-testid="stFileUploader"] div {
  background:#F8E9A6 !important; background-color:#F8E9A6 !important;
}
[data-testid="stFileUploaderDropzone"] {
  border:1.5px dashed #D4B86A !important; border-radius:14px !important;
}
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzoneInstructions"] * {
  color:#6F4A26 !important; background:transparent !important;
}
[data-testid="stFileUploaderDropzone"] span[data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzone"] [class*="material-symbols"] { display:none !important; }
[data-testid="stFileUploaderDropzone"] button {
  background:#F3E3CC !important; color:#6F4A26 !important; border:1px solid #D4B86A !important;
}
.stTabs [data-baseweb="tab"] {
  background:#F3E3CC; color:#8A5A2B !important; border-radius:14px 14px 0 0; font-weight:600;
}
.stTabs [aria-selected="true"] { background:#D9724C !important; color:#FFF8EF !important; }
div.stButton > button { background:#D9724C; color:#FFF8EF; border:none; border-radius:14px; font-weight:600; }
.stTextInput input, .stTextArea textarea,
[data-baseweb="input"], [data-baseweb="base-input"] {
  background:#FFF8EF !important; color:#6F4A26 !important; border-color:#E8D4B0 !important;
}
.soru,.cevap,.card,.side-item {
  border-radius:14px; padding:14px 16px; margin-bottom:10px; background:#FFF8EF; color:#6F4A26;
}
.soru { background:#F3E3CC; border-left:5px solid #B5541E; }
.cevap { border-left:5px solid #6F8F5B; }
.badge { display:inline-block; background:#6F8F5B; color:#FFF8EF; border-radius:10px; padding:2px 10px; font-weight:700; }
.miss { display:inline-block; background:#F4C9BC; color:#8A3A24; border-radius:10px; padding:2px 8px; margin:2px; font-size:.8em; }
.side-item { font-size:.9em; }
</style>
"""

st.set_page_config(page_title="Tarif Defteri Asistanı", page_icon="🍲", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def boot_rag():
    from main import load_models, prepare_index

    _emb_model, embedding_client, _chat_model, chat_client = load_models()
    docs, embeddings, sources = prepare_index(embedding_client)
    return embedding_client, chat_client, docs, embeddings, sources


rag = None
rag_error = ""
try:
    rag = boot_rag()
except Exception as exc:
    rag = None
    rag_error = str(exc)[:200]

if "conn" not in st.session_state:
    st.session_state.conn = database.get_connection()
    database.seed_if_empty(st.session_state.conn)
if "chat" not in st.session_state:
    st.session_state.chat = []

conn = st.session_state.conn

with st.sidebar:
    st.markdown("### Tarif Defteri Asistanı")
    st.caption("Soru sekmesi Foundry Local RAG ile tariflerde arar.")
    if rag:
        st.caption("RAG hazır.")
    else:
        st.warning("Foundry Local yok; kural tabanlı tarif araması kullanılıyor.")
        if rag_error:
            st.caption(rag_error)
    files = st.file_uploader("Tarif dosyaları yükle (.txt)", type=["txt"], accept_multiple_files=True)
    if files and st.button("Yükle ve İndeksle"):
        if len(files) > MAX_UPLOAD_FILES:
            st.error(f"En fazla {MAX_UPLOAD_FILES} dosya yükleyebilirsin.")
        else:
            ok, errors = 0, []
            for uploaded in files:
                try:
                    source = recipe_parser.safe_filename(uploaded.name)
                    title, text, ingredients = recipe_parser.parse_recipe_bytes(
                        source, uploaded.getvalue()
                    )
                    database.upsert_recipe(conn, source, title, text, ingredients)
                    (KB_FOLDER).mkdir(parents=True, exist_ok=True)
                    (KB_FOLDER / source).write_text(text, encoding="utf-8")
                    ok += 1
                except ValueError as exc:
                    errors.append(f"{uploaded.name}: {exc}")
            if ok:
                boot_rag.clear()
                st.success(f"{ok} tarif kaydedildi. RAG indeksi yenilenecek.")
            for message in errors:
                st.error(message)
    st.markdown("---")
    st.markdown("#### İndekslenmiş Tarifler")
    recipes = database.get_all_recipes(conn)
    st.markdown(f"<p style='color:#6F4A26'>{len(recipes)} tarif kayıtlı</p>", unsafe_allow_html=True)
    for recipe in recipes:
        title = html.escape(str(recipe["title"]))
        label = html.escape(diet_rules.diet_label(recipe))
        st.markdown(
            f"<div class='side-item'>🍽️ {title} · {label} · {len(recipe['ingredients'])} malzeme</div>",
            unsafe_allow_html=True,
        )

st.markdown("# 🍲 Tarif Defteri Asistanı")
st.markdown(
    "<p style='color:#6F4A26'>Tarif defteri. Soru sorunca Foundry Local RAG tariflerde arar, "
    "sonra yerel model cevaplar.</p>",
    unsafe_allow_html=True,
)
tab_q, tab_m, tab_g = st.tabs(["Soru Sor", "Malzemelerimle Ne Yapabilirim", "Tarif Galerisi"])

with tab_q:
    question = st.text_input("Tariflerinle ilgili bir şey sor", placeholder="Menemen nasıl yapılır?")
    if st.button("Sor", key="ask"):
        q = question.strip()[:MAX_QUESTION_LEN]
        sources = []
        if not q:
            answer = "Boş soru gönderildi."
        elif rag:
            from main import resolve_answer

            embedding_client, chat_client, docs, embeddings, rag_sources = rag
            answer, sources = resolve_answer(
                q,
                embedding_client,
                embeddings,
                docs=docs,
                sources=rag_sources,
                chat_client=chat_client,
                conn=conn,
            )
        else:
            found = search_engine.search_recipes(conn, q, limit=8)
            answer = search_engine.build_answer(q, found)
            sources = [recipe["source_file"] for recipe in found]
        st.session_state.chat.append({"q": q or "(boş)", "a": answer, "sources": sources})
        st.session_state.chat = st.session_state.chat[-MAX_CHAT:]
    for entry in reversed(st.session_state.chat):
        q_html = html.escape(entry["q"])
        st.markdown(f"<div class='soru'><b>Soru:</b> {q_html}</div>", unsafe_allow_html=True)
        body = html.escape(entry["a"]).replace("\n", "<br>")
        src = " ".join(
            f"<span class='miss'>dosya: {html.escape(str(source))}</span>"
            for source in entry["sources"]
        )
        st.markdown(f"<div class='cevap'><b>Cevap:</b><br>{body}<br>{src}</div>", unsafe_allow_html=True)

with tab_m:
    bag = st.text_input("Elindeki malzemeleri virgülle ayırarak yaz", placeholder="yumurta, un, süt")
    if st.button("Ne Yapabilirim?", key="match") and bag.strip():
        matches = recommender.match_by_ingredients(conn, bag[:MAX_BAG_LEN])
        if not matches:
            st.info("Eşleşen tarif yok.")
        else:
            cols = st.columns(2)
            for index, recipe in enumerate(matches):
                with cols[index % 2]:
                    title = html.escape(str(recipe["title"]))
                    miss = "".join(
                        f"<span class='miss'>eksik: {html.escape(str(name))}</span>"
                        for name in recipe["missing_ingredients"]
                    )
                    st.markdown(
                        f"<div class='card'><div class='badge'>%{int(recipe['overlap_percentage'])} uyum</div>"
                        f"<b>{title}</b><div>{miss}</div></div>",
                        unsafe_allow_html=True,
                    )

with tab_g:
    gallery = database.get_all_recipes(conn)
    if not gallery:
        st.info("Henüz tarif yok.")
    else:
        cols = st.columns(3)
        for index, recipe in enumerate(gallery):
            with cols[index % 3]:
                title = html.escape(str(recipe["title"]))
                preview = html.escape(", ".join(str(x) for x in recipe["ingredients"][:5]))
                st.markdown(
                    f"<div class='card'><b>{title}</b><div style='color:#8A6A45'>{preview}</div></div>",
                    unsafe_allow_html=True,
                )
                with st.expander("Detay göster"):
                    st.text(str(recipe["full_text"]))
