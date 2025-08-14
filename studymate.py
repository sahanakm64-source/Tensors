
import streamlit as st
import PyPDF2
import io
import re
import nltk
from typing import List
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Ensure NLTK resources are ready
nltk.download("punkt")
nltk.download("stopwords")


class StudyMate:
    def __init__(self):
        # Always initialize vectorizer
        self.vectorizer = TfidfVectorizer()
        self.chunk_vectors = None
        self.chunks = []

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF."""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

    def preprocess_text(self, text: str) -> str:
        """Lowercase, remove stopwords, stem words."""
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        stop_words = set(stopwords.words("english"))
        ps = PorterStemmer()
        tokens = [ps.stem(w) for w in text.split() if w not in stop_words]
        return " ".join(tokens)

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks of ~chunk_size chars."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def train_vectorizer(self, text_chunks: List[str]):
        """Fit TF-IDF on text chunks."""
        self.chunks = text_chunks
        self.chunk_vectors = self.vectorizer.fit_transform(text_chunks)

    def answer_question(self, question: str, top_k: int = 3) -> List[str]:
        """Find top_k most relevant chunks."""
        if self.chunk_vectors is None:
            return ["❌ Vectorizer not trained yet."]
        question_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, self.chunk_vectors).flatten()
        ranked_indices = similarities.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in ranked_indices]


def main():
    st.title("📚 StudyMate - AI PDF Q&A System")

    # 🔹 SAFEGUARD: Recreate object if class definition changed
    if (
        "studymate" not in st.session_state
        or not hasattr(st.session_state.studymate, "vectorizer")
    ):
        st.session_state.studymate = StudyMate()

    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_pdf:
        pdf_bytes = uploaded_pdf.read()
        with st.spinner("Extracting text from PDF..."):
            text = st.session_state.studymate.extract_text_from_pdf(pdf_bytes)

        if not text.strip():
            st.error("No text could be extracted. The PDF might be scanned or encrypted.")
            return

        st.success("✅ PDF text extracted!")

        processed_text = st.session_state.studymate.preprocess_text(text)
        chunks = st.session_state.studymate.chunk_text(processed_text)

        st.session_state.studymate.train_vectorizer(chunks)
        st.success("📊 Vectorizer trained with document chunks!")

        question = st.text_input("Ask a question from the PDF:")
        if question:
            answers = st.session_state.studymate.answer_question(question)
            st.subheader("Top Answers:")
            for idx, ans in enumerate(answers, 1):
                st.write(f"**{idx}.** {ans}")


if __name__ == "__main__":
    main()
