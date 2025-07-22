# 💬 Sentiment Analysis on Product Reviews (Text & Speech Input)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)
![License](https://img.shields.io/badge/license-MIT-green)

An **interactive sentiment analysis dashboard** using **Machine Learning** to predict **Positive, Negative, or** sentiment on product reviews, supporting **text input** and **speech input**.

---

## 🚀 Features

✅ Predict sentiment using **typed text** or **speech input** via microphone/audio upload.  
✅ Displays **confidence scores** and **emoji feedback**.  
✅ **Clean Streamlit dashboard** with custom gradient styling.  
✅ Suitable for **placements, labs, and portfolio projects**.  
✅ Easy to extend with advanced ML features.

---

## 🛠️ Tech Stack

- Python (pandas, numpy, scikit-learn, nltk, joblib)
- Streamlit (dashboard)
- SpeechRecognition (speech-to-text)
- Machine Learning (Naive Bayes / Logistic Regression)

---

## 📂 Project Structure

```
sentiment_analysis/
│
├── models/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── app.py
├── sentiment_training.ipynb
├── requirements.txt
└── README.md
```

---

## ⚡ Setup Instructions

1️⃣ **Clone the repository:**

```bash
git clone https://github.com/yourusername/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard
```

2️⃣ **Create and activate a virtual environment:**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

3️⃣ **Install required dependencies:**

```bash
pip install -r requirements.txt
```

4️⃣ **Run the Streamlit dashboard:**

```bash
streamlit run app.py
```

---

## 🎤 Speech Input Notes

- Allow **microphone access** on your system for speech input.
- Speak clearly in a quiet environment.
- Alternatively, use **audio file upload** for stable speech-to-text analysis.

---

## 🧠 Model Training

- The model is trained on a labeled product review dataset.
- You can retrain or improve using `sentiment_training.ipynb`.

---

## 🙌 Acknowledgements

- [Streamlit](https://streamlit.io)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
- [Scikit-learn](https://scikit-learn.org)
- [NLTK](https://www.nltk.org)

---

## 🚀 Made with ❤️ using Streamlit for your Data Science Portfolio.

---

## ✨ Contact

For queries or collaboration:

- Email:vincymicheal123@gmail.com
- LinkedIn: [https://www.linkedin.com/in/vimalivincy-m]
- GitHub: [https://github.com/Vimali23vincy]

---
