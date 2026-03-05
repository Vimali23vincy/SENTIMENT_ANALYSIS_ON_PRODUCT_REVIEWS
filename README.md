# 💬 Sentiment Pro: Advanced Product Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?style=for-the-badge&logo=streamlit)
![Machine Learning](https://img.shields.io/badge/ML-Sentiment--Analysis-vibrantgreen?style=for-the-badge)

**Sentiment Pro** is a high-performance, interactive sentiment analysis dashboard powered by Machine Learning. It provides real-time emotional intelligence for product reviews using advanced NLP techniques and a stunning visual interface.

---

## 🚀 Live Demo
Experience the live application here:  
👉 **[Sentiment Pro on Streamlit Cloud](https://sentimentanalysisonappuctreviews-vincy0409.streamlit.app/)**

---

## ✨ Features

- **🧠 Neural Intelligence:** Powered by a calibrated Linear Support Vector Machine (SVM) for superior text classification.
- **⚡ Advanced Preprocessing:** Custom text cleaning engine handling negations, bigrams, and TF-IDF vectorization.
- **📊 Real-time Insights:** Instant sentiment prediction (Positive, Negative, Neutral) with confidence scores.
- **📤 Bulk Analytics:** Upload CSV files to analyze thousands of reviews in seconds.
- **🎨 Premium UI:** Modern "Soft Aurora" theme with glassmorphism effects and interactive Plotly visualizations.
- **☁️ Global Stream Simulator:** Simulate a live feedback stream to see the model in action.

---

## 🛠️ Tech Stack

- **Core:** Python 3.10+
- **Machine Learning:** Scikit-learn (SVM, Pipeline)
- **NLP:** NLTK (Stopwords removal, negation handling)
- **Internal Data:** Pandas, NumPy
- **Dashboard:** Streamlit
- **Visuals:** Plotly Express, WordCloud, Matplotlib

---

## 📂 Project Structure

```text
SENTIMENT_ANALYSIS_ON_PRODUCT_REVIEWS/
│
├── models/
│   └── sentiment_pipeline.pkl    # Trained SVM Model Pipeline
│
├── Datasets_Merged.csv           # Unified high-quality training dataset
├── app.py                        # Main Streamlit Application (Premium UI)
├── retrain.py                    # Model retraining script (Optimized SVM)
├── optimize_dataset.py           # Dataset augmentation & boosting script
├── requirements.txt              # Project dependencies
└── README.md                     # Enhanced documentation
```

---

## ⚡ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Vimali23vincy/SENTIMENT_ANALYSIS_ON_PRODUCT_REVIEWS.git
cd SENTIMENT_ANALYSIS_ON_PRODUCT_REVIEWS
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

---

## 📖 How it Works
The model uses a **Support Vector Machine (SVM)** with calibrated probabilities. Unlike simpler models, it's trained to understand common English negations (e.g., "not good" is correctly identified as negative). The project uses **TF-IDF Vectorization** with bigrams (2-word phrases) to capture more context than single words.

---

## 🙌 Author
**Vimali Vincy M**  
- **LinkedIn:** [vimalivincy-m](https://www.linkedin.com/in/vimalivincy-m)
- **GitHub:** [@Vimali23vincy](https://github.com/Vimali23vincy)
- **Email:** vincymicheal123@gmail.com

---
*Built with ❤️ for Data-Driven Decision Making.*
