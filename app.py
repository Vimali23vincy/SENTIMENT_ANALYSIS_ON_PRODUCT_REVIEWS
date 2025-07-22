import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk
import speech_recognition as sr

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load your trained model and vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Page config for a clean dashboard feel
st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="💬", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #a8edea, #fed6e3);
        padding: 20px;
        border-radius: 15px;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #6c63ff;
        font-size: 16px;
    }
    .stButton>button {
        border-radius: 10px;
        background: linear-gradient(to right, #6c63ff, #5de0e6);
        color: white;
        font-size: 18px;
        height: 3em;
        width: 100%;
    }
    .title {
        text-align: center;
        font-size: 42px;
        color: #6c63ff;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 24px;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<div class='title'>💬 Sentiment Analysis Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict sentiment from product reviews with ML ⚡</div>", unsafe_allow_html=True)

# Layout container
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)

    # --- TEXT INPUT ---
    st.subheader("📝 Type Your Review")
    user_input = st.text_area("Enter your product review below 👇", height=150)

    if st.button("🔍 Predict Sentiment from Text"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review before predicting.")
        else:
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0]
            prediction_proba = model.predict_proba(vectorized_input)[0]

            st.markdown("<h4>✨ Prediction Result:</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px;'>🎯 The predicted sentiment is: <b>{prediction.upper()}</b></p>", unsafe_allow_html=True)

            st.markdown("<h4>📊 Confidence Scores:</h4>", unsafe_allow_html=True)
            for label, prob in zip(model.classes_, prediction_proba):
                st.write(f"**{label.capitalize()}**: {prob*100:.2f}%")

            emoji_map = {'positive': '😄', 'negative': '😡', 'neutral': '😐'}
            st.markdown(f"<h1 style='text-align: center;'>{emoji_map[prediction]}</h1>", unsafe_allow_html=True)

    st.markdown("---")

    # --- SPEECH INPUT ---
    st.subheader("🎤 Or Speak Your Review")

    if st.button("🎙️ Record and Predict Sentiment from Speech"):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        st.info("🎙️ Listening... Please speak clearly into your microphone.")

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            speech_text = recognizer.recognize_google(audio)
            st.success(f"📝 Transcribed Text: {speech_text}")

            cleaned_input = clean_text(speech_text)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0]
            prediction_proba = model.predict_proba(vectorized_input)[0]

            st.markdown("<h4>✨ Prediction Result:</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px;'>🎯 The predicted sentiment is: <b>{prediction.upper()}</b></p>", unsafe_allow_html=True)

            st.markdown("<h4>📊 Confidence Scores:</h4>", unsafe_allow_html=True)
            for label, prob in zip(model.classes_, prediction_proba):
                st.write(f"**{label.capitalize()}**: {prob*100:.2f}%")

            emoji_map = {'positive': '😄', 'negative': '😡', 'neutral': '😐'}
            st.markdown(f"<h1 style='text-align: center;'>{emoji_map[prediction]}</h1>", unsafe_allow_html=True)

        except sr.UnknownValueError:
            st.error("❌ Could not understand audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"❌ Could not request results from Google Speech Recognition service; {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr>
<center>
🚀 Built with Streamlit for your Data Science portfolio and lab projects.
</center>
""", unsafe_allow_html=True)

import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    r.adjust_for_ambient_noise(source)
    print("Listening...")
    audio = r.listen(source)

try:
    text = r.recognize_google(audio)
    print(text)
except Exception as e:
    print(e)
