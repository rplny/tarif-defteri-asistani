"""Foundry Local RAG Streamlit arayüzü, Tarif Defteri görünümü."""
import html

import streamlit as st

from main import (
    build_messages,
    get_top_chunks,
    load_models,
    prepare_index,
    stream_answer,
)
from retrieval import format_context

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
.miss { display:inline-block; background:#F4C9BC; color:#8A3A24; border-radius:10px; padding:2px 8px; margin:2px; font-size:.8em; }
.side-item { font-size:.9em; }
</style>
"""


@st.cache_resource
def boot():
    embedding_model, embedding_client, chat_model, chat_client = load_models()
    docs, embeddings, sources = prepare_index(embedding_client)
    return embedding_model, embedding_client, chat_model, chat_client, docs, embeddings, sources


st.set_page_config(page_title="Tarif Defteri Asistanı", page_icon="🍲", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

try:
    _, embedding_client, _, chat_client, docs, embeddings, sources = boot()
except Exception as exc:
    st.error("Foundry Local yüklenemedi. `python main.py` ile modelleri bir kez indir.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.markdown("### Tarif Defteri Asistanı")
    st.caption("Cevaplar defterindeki tariflerden. Foundry Local; internet yok.")
    names = sorted({name for name in sources if name})
    st.markdown(
        f"<p style='color:#6F4A26'>{len(names)} belge · {len(docs)} parça</p>",
        unsafe_allow_html=True,
    )
    for name in names:
        st.markdown(
            f"<div class='side-item'>📄 {html.escape(name)}</div>",
            unsafe_allow_html=True,
        )

st.markdown("# 🍲 Tarif Defteri Asistanı")
st.markdown(
    "<p style='color:#6F4A26'>Tarif sor. Önce defterde arar, sonra yerel model cevaplar.</p>",
    unsafe_allow_html=True,
)

question = st.text_input("Bir şey sor", placeholder="Menemen nasıl yapılır?")
if st.button("Sor"):
    q = question.strip()
    if not q:
        st.markdown(
            "<div class='cevap'><b>Cevap:</b><br>Boş soru gönderildi.</div>",
            unsafe_allow_html=True,
        )
    else:
        hits = get_top_chunks(q, embedding_client, docs, embeddings, sources=sources, top_k=3)
        context = format_context(hits)
        st.markdown(f"<div class='soru'><b>Soru:</b> {html.escape(q)}</div>", unsafe_allow_html=True)
        badges = " ".join(
            f"<span class='miss'>{html.escape(str(hit['source'] or 'kaynak yok'))} · {hit['score']:.2f}</span>"
            for hit in hits
        )
        with st.expander("Bulunan parçalar", expanded=True):
            if not hits:
                st.write("Uygun parça yok.")
            else:
                st.markdown(badges, unsafe_allow_html=True)
                for hit in hits:
                    preview = html.escape(hit["content"])
                    st.markdown(f"<div class='card'>{preview}</div>", unsafe_allow_html=True)
        if not context.strip():
            st.markdown(
                "<div class='cevap'><b>Cevap:</b><br>Bu bilgi context'te yok.</div>",
                unsafe_allow_html=True,
            )
        else:
            placeholder = st.empty()
            collected = []

            def writer(*args, **kwargs):
                text = args[0] if args else ""
                if kwargs.get("end") == "" and text not in {"Cevap: "}:
                    collected.append(text)
                    body = html.escape("".join(collected)).replace("\n", "<br>")
                    placeholder.markdown(
                        f"<div class='cevap'><b>Cevap:</b><br>{body}</div>",
                        unsafe_allow_html=True,
                    )

            stream_answer(chat_client, build_messages(q, context), writer=writer)
            if collected:
                body = html.escape("".join(collected)).replace("\n", "<br>")
                placeholder.markdown(
                    f"<div class='cevap'><b>Cevap:</b><br>{body}</div>",
                    unsafe_allow_html=True,
                )
