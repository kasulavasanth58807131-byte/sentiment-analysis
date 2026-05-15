import streamlit as st
import pickle
import re
import string
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ── Load model & vectorizer ───────────────────────────────────
@st.cache_resource
def load_model():
    model      = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()
stop_words = set(stopwords.words('english'))

# ── Preprocessing (same as notebook) ─────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(w for w in text.split() if w not in stop_words)
    return text

# ── UI ────────────────────────────────────────────────────────
st.title("🎬 IMDB Sentiment Analyzer")
st.markdown("Predict whether a movie review is **Positive** or **Negative** using an SVM model trained on 50,000 IMDB reviews.")
st.markdown("---")

review = st.text_area(
    "Paste your movie review here:",
    height=180,
    placeholder="e.g. This movie was absolutely fantastic! The acting was superb..."
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(review)
        vec     = vectorizer.transform([cleaned])
        pred    = model.predict(vec)[0]

        if pred == 1:
            st.success("✅ **POSITIVE** — This review expresses a positive sentiment.")
        else:
            st.error("❌ **NEGATIVE** — This review expresses a negative sentiment.")

        with st.expander("See cleaned text"):
            st.write(cleaned)

st.markdown("---")
st.caption("Model: LinearSVC | Vectorizer: TF-IDF (unigrams + bigrams) | Dataset: IMDB 50K")
