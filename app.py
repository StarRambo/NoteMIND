import os, streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

st.set_page_config(page_title="NoteMind", layout="wide")
st.title("🧠 NoteMind – Streamlit Cloud")

# ―― 1. API key (pulled from Streamlit Cloud Secret) ――――――――――――――――
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY – add it in Streamlit > Secrets!")
    st.stop()

# --- imports ---
from llama_index.embeddings import HuggingFaceEmbedding    # ✅ works with 0.11
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbedding(model_name=HF_MODEL)    # local, no API key
llm         = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

svc_ctx = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# ―― 2. Upload & index docs ――――――――――――――――――――――――――――――――――――――――
os.makedirs("data", exist_ok=True)
files = st.file_uploader("📄 Upload PDFs", type="pdf", accept_multiple_files=True)

if files:
    for f in files:
        with open(f"data/{f.name}", "wb") as out:
            out.write(f.read())

    with st.spinner("Indexing…"):
        docs   = SimpleDirectoryReader("data").load_data()
        index  = VectorStoreIndex.from_documents(docs, service_context=svc_ctx)
        engine = index.as_query_engine()

    st.success("Indexed! Ask away ↓")

    q = st.text_input("❓ Question")
    if q:
        st.write(engine.query(q).response)
