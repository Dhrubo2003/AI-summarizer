import streamlit as st
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from graphviz import Digraph
import tempfile

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="AI Research Paper Summarizer",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------

st.title("📄 AI Research Paper Summarizer")
st.write("Upload a research paper to generate a **structured summary and mind map**.")

st.divider()

# -----------------------------
# Models (load once)
# -----------------------------

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    return embed_model, summarizer


embed_model, summarizer = load_models()

# -----------------------------
# PDF Text Extraction
# -----------------------------

def extract_text(uploaded_file):

    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text


# -----------------------------
# Section Detection
# -----------------------------

def detect_sections(text):

    sections = text.split("\n\n")

    return sections


# -----------------------------
# Chunking
# -----------------------------

def chunk_text(text, size=500):

    words = text.split()
    chunks = []

    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)

    return chunks


# -----------------------------
# Retrieve Important Chunks
# -----------------------------

def retrieve_chunks(chunks):

    embeddings = embed_model.encode(chunks)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))

    query = embed_model.encode(["main research contribution"])

    D, I = index.search(np.array(query), 10)

    important_chunks = [chunks[i] for i in I[0]]

    return important_chunks


# -----------------------------
# Hierarchical Summarization
# -----------------------------

def summarize_chunks(chunks):

    summaries = []

    for chunk in chunks:

        summary = summarizer(
            chunk,
            max_length=120,
            min_length=40,
            do_sample=False
        )[0]["summary_text"]

        summaries.append(summary)

    return summaries


def final_summary(chunk_summaries):

    combined = " ".join(chunk_summaries)

    summary = summarizer(
        combined,
        max_length=200,
        min_length=80,
        do_sample=False
    )[0]["summary_text"]

    return summary


# -----------------------------
# Mind Map Generation
# -----------------------------

def generate_mindmap(summary):

    dot = Digraph()

    dot.node("Paper")

    nodes = [
        "Problem",
        "Method",
        "Results",
        "Conclusion"
    ]

    for n in nodes:
        dot.node(n)
        dot.edge("Paper", n)

    tmp = tempfile.NamedTemporaryFile(delete=False)

    path = dot.render(tmp.name, format="png")

    return path


# -----------------------------
# Upload Section
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Research Paper (PDF)",
    type="pdf"
)

# -----------------------------
# Processing
# -----------------------------

if uploaded_file is not None:

    if st.button("Generate Summary & Mind Map"):

        with st.spinner("Processing research paper..."):

            # Extract
            text = extract_text(uploaded_file)

            # Section detection
            sections = detect_sections(text)

            # Chunking
            chunks = chunk_text(text)

            # Retrieval
            important_chunks = retrieve_chunks(chunks)

            # Summaries
            chunk_summaries = summarize_chunks(important_chunks)

            summary = final_summary(chunk_summaries)

            # Mindmap
            mindmap_path = generate_mindmap(summary)

        st.success("Analysis Complete!")

        st.subheader("📑 Final Summary")

        st.write(summary)

        st.subheader("🧠 Mind Map")

        st.image(mindmap_path)

        st.download_button(
            "Download Summary",
            summary,
            file_name="summary.txt"
        )

        with open(mindmap_path, "rb") as f:
            st.download_button(
                "Download Mind Map",
                f,
                file_name="mindmap.png"
            )
