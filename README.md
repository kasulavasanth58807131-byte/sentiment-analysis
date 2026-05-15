# 🎬 IMDB Sentiment Analysis

A machine learning project that classifies IMDB movie reviews as **Positive** or **Negative** using NLP techniques.  
Trained on the **50,000 IMDB Movie Reviews** dataset.

---

## 📊 Models & Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~90%     |
| Naive Bayes         | ~87%     |
| **LinearSVC (SVM)** | **~90%** |

Best model: **LinearSVC** — saved as `model.pkl`

---

## 🗂️ Project Structure

```
imdb-sentiment/
│
├── app.py                          # Streamlit web app
├── sentment_analysis__imdb_.ipynb  # Full analysis notebook
├── requirements.txt                # Dependencies
├── .gitignore                      # Files to exclude
├── README.md                       # This file
│
├── model.pkl                       # Trained SVM model      (generate via notebook)
└── vectorizer.pkl                  # TF-IDF vectorizer      (generate via notebook)
```

> ⚠️ `model.pkl` and `vectorizer.pkl` are **not included** in the repo (too large).  
> Run the notebook to generate them, then place them in the root folder.

---

## ⚙️ Pipeline

1. **Load** — IMDB Dataset CSV (50K reviews)
2. **EDA** — Class distribution, review length analysis, word clouds
3. **Preprocess** — Lowercase, remove HTML/URLs/punctuation/stopwords
4. **Vectorize** — TF-IDF with unigrams + bigrams (`max_features=50000`)
5. **Train** — Logistic Regression, Naive Bayes, LinearSVC
6. **Evaluate** — Accuracy, Classification Report, Confusion Matrix
7. **Save** — Best model serialized with `pickle`

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/imdb-sentiment.git
cd imdb-sentiment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate model files (run notebook first)
#    Place model.pkl and vectorizer.pkl in this folder

# 4. Launch the app
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file** to `app.py`
5. Click **Deploy**

> ⚠️ Upload `model.pkl` and `vectorizer.pkl` to the repo before deploying  
> (use Git LFS if files exceed 100MB)

---

## 📦 Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — Kaggle

---

## 🛠️ Tech Stack

- **Python** · Pandas · NumPy
- **Scikit-learn** — TF-IDF, LinearSVC, Logistic Regression, Naive Bayes
- **NLTK** — Stopwords removal
- **Streamlit** — Web app
- **Matplotlib / Seaborn / WordCloud** — Visualization

---

## 👤 Author

**Vasanth** — Data Science Intern  
B.Tech CSE | Bharat Institute of Engineering and Technology
