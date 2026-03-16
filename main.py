```python
import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from graphviz import Digraph

st.title("AI Research Paper Summarizer")

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Upload PDF
uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")


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
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks


def summarize_chunks(chunks):
    summaries = []
    for chunk in chunks[:10]:
        result = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
        summaries.append(result[0]['summary_text'])
    return summaries


def create_mindmap():
    dot = Digraph()
    dot.node("Paper")
    dot.node("Problem")
    dot.node("Method")
    dot.node("Results")
    dot.node("Conclusion")

    dot.edge("Paper", "Problem")
    dot.edge("Paper", "Method")
    dot.edge("Paper", "Results")
    dot.edge("Paper", "Conclusion")

    dot.render("mindmap", format="jpg", cleanup=True)


if uploaded_file is not None:

    if st.button("Generate Summary"):

        # Extract text
        text = extract_text(uploaded_file)

        # Section detection
        sections = detect_sections(text)

        # Chunking
        chunks = chunk_text(text)

        # Embeddings (FIX: now chunks exists before this line)
        embeddings = embedding_model.encode(chunks)

        # FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Retrieve important chunks
        query_embedding = embedding_model.encode(["main research contribution"])
        D, I = index.search(query_embedding, 5)

        important_chunks = [chunks[i] for i in I[0]]

        # Summarization
        summaries = summarize_chunks(important_chunks)

        final_summary = "\n\n".join(summaries)

        # Create mind map
        create_mindmap()

        st.subheader("Final Summary")
        st.write(final_summary)

        # Download summary
        st.download_button(
            "Download Summary",
            final_summary,
            file_name="summary.txt"
        )

        # Download mindmap
        with open("mindmap.jpg", "rb") as file:
            st.download_button(
                "Download Mind Map",
                file,
                file_name="mindmap.jpg"
            )
```
