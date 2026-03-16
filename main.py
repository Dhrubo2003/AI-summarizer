import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from graphviz import Digraph

uploaded_file = st.file_uploader("Upload Research Paper", type="pdf")

def extract_text(pdf):
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def detect_sections(text):
    sections = text.split("\n\n")
    return sections


def chunk_text(text, size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)

    return chunks



model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

query_embedding = model.encode(["main research idea"])
D, I = index.search(query_embedding, 10)



summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)



dot = Digraph()

dot.node("Paper")
dot.node("Problem")
dot.node("Method")
dot.node("Results")

dot.edge("Paper","Problem")
dot.edge("Paper","Method")
dot.edge("Paper","Results")

dot.render("mindmap", format="jpg")


st.download_button(
    "Download Summary",
    summary,
    "summary.txt"
)



