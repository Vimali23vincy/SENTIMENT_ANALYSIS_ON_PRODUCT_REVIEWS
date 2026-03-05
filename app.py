import streamlit as st
import base64
import joblib
import string
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import io

# Initialize NLTK
nltk.download('stopwords', quiet=True)
NEGATIONS = {"no", "not", "nor", "neither", "never", "none", "nt", "isnt", "wasnt", "arent", "werent", "dont", "doesnt", "didnt", "hasnt", "havent", "hadnt", "shouldnt", "wouldnt", "couldnt", "mightnt", "mustnt"}
stop_words = set(stopwords.words('english')) - NEGATIONS

# Load your trained model and vectorizer
@st.cache_resource
def load_assets():
    try:
        # Load the whole pipeline
        pipeline = joblib.load('models/sentiment_pipeline.pkl')
        return pipeline
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

pipeline = load_assets()

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Handle "n't" before removing punctuation
    text = text.replace("n't", " nt")
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Page configuration
st.set_page_config(
    page_title="Sentiment Pro - Product Analytics",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Soft Aurora Theme - Light & Vibrant */
    .stApp {
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar glassmorphism - Modern Light */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Premium Logic Card 디자인 - Light Mode Glass */
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        padding: 15px 10px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.7);
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    
    .metric-card h3 {
        font-size: 0.9rem !important;
        margin-bottom: 5px !important;
    }
    
    .metric-card h2 {
        font-size: 1.6rem !important;
        margin: 0 !important;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.1);
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Header Styling */
    .main-title {
        font-weight: 800;
        color: #1a2a6c;
        text-shadow: 1px 1px 15px rgba(255,255,255,0.5);
        font-size: 4rem;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: -2px;
    }

    .accuracy-badge {
        display: block;
        width: fit-content;
        margin: 0 auto 15px;
        background: #1a2a6c;
        color: white;
        padding: 6px 18px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .sub-title {
        color: #4a5568;
        text-align: center;
        font-size: 1.4rem;
        margin-bottom: 50px;
        font-weight: 400;
        font-style: italic;
        background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
    }
    
    .hero-container {
        padding: 60px 20px;
        text-align: center;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 30px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 40px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.05);
    }
    
    /* Result Box Design */
    .res-box {
        padding: 40px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 25px;
        background: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.06);
        border: 1px solid #edf2f7;
    }

    /* Interactive Elements Styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #2d3748 !important;
        border-radius: 18px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 20px !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02) !important;
    }

    .stTextArea textarea:focus {
        border-color: #a1c4fd !important;
        box-shadow: 0 4px 20px rgba(161, 196, 253, 0.2) !important;
    }

    /* Premium Button - Indigo Blue Gradient */
    .stButton>button {
        width: 100%;
        border-radius: 18px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 1.25rem;
        height: 4.2rem;
        border: none;
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.25);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
        filter: brightness(1.1);
    }

    /* Sidebar text color fix */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2, 
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #1a2a6c !important;
    }
    
    .metric-card h3 { color: #4a5568; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
def get_base64(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

logo_b64 = get_base64('sentiment_logo.png')
if logo_b64:
    st.sidebar.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_b64}" width="200" style="border-radius: 20px; filter: drop-shadow(0 10px 20px rgba(0,0,0,0.15));">
        </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("<h1 style='text-align: center; font-size: 6rem; filter: drop-shadow(0 0 15px rgba(0,0,0,0.1));'>🧠</h1>", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='text-align: center; color: #1a2a6c; font-weight: 800; letter-spacing: 1px;'>SENTIMENT PRO</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("COMMAND CENTER", ["Single Review", "Bulk Analytics", "Social Stream Simulator"])

# Helper function to predict
def predict_sentiment(text):
    if pipeline is None:
        return "neutral", [0.33, 0.33, 0.34]
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    prob = pipeline.predict_proba([cleaned])[0]
    return pred, prob

# --- MAIN PAGE CONTENT ---
if app_mode == "Single Review":
    st.markdown("""
        <div class="hero-container">
            <h1 class='main-title'>Sentiment Pro</h1>
            <p class='sub-title'>"Unlocking the Power of Human Emotion through Advanced Neural Intelligence"</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("### 📝 Input Feedback")
        user_input = st.text_area("", height=250, placeholder="Paste customer review here...", help="Enter any text to see if it's positive, negative or neutral.")
        
        analyze_btn = st.button("Generate Insights")
        
    with col2:
        if user_input.strip() and analyze_btn:
            prediction, probabilities = predict_sentiment(user_input)
            
            st.markdown("### 📊 Prediction Result")
            # Visual Sentiment Indicator
            emoji_map = {'positive': '🌟 Positive', 'negative': '⚠️ Negative', 'neutral': '⚖️ Neutral'}
            color_map = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f1c40f'}
            bg_map = {'positive': '#ebfaf0', 'negative': '#fdedec', 'neutral': '#fef9e7'}
            
            st.markdown(f"""
                <div class="res-box" style="background-color: {bg_map[prediction]}; border: 1px solid {color_map[prediction]}55;">
                    <h1 style="color: {color_map[prediction]}; font-size: 3rem; margin: 0;">{emoji_map[prediction]}</h1>
                    <p style="color: #636e72; font-size: 1.2rem; margin-top: 10px;">Reliability: <b>{max(probabilities)*100:.1f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence Breakdown Chart
            fig = px.bar(
                x=[c.capitalize() for c in pipeline.classes_],
                y=probabilities,
                labels={'x': 'Sentiment', 'y': 'Confidence'},
                color=[c for c in pipeline.classes_],
                color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f1c40f'},
                template="plotly_dark"
            )
            fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10, l=10, r=10),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        elif analyze_btn:
            st.info("👆 Please input text first to run analysis.")
        else:
            st.markdown("### 📊 Analysis Console")
            st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.4); padding: 30px; border-radius: 20px; border: 1px dashed rgba(26, 42, 108, 0.3); text-align: center;">
                    <h2 style="color: #1a2a6c; margin-bottom: 10px;">Ready to Decode?</h2>
                    <p style="color: #4a5568; font-size: 1.1rem;">"Understanding the heart of your customer, one word at a time."</p>
                    <div style="margin-top: 20px; font-size: 0.9rem; color: #718096; text-transform: uppercase; letter-spacing: 2px;">
                        Waiting for Input • AI Ready
                    </div>
                </div>
            """, unsafe_allow_html=True)

elif app_mode == "Bulk Analytics":
    st.markdown("""
        <div class="hero-container">
            <h1 class='main-title'>Bulk Analysis Center</h1>
            <p class='sub-title'>"Mass-Scale Emotional Intelligence for Data-Driven Decisions"</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'review' in df.columns:
            with st.spinner('🚀 Engines accelerating... Analyzing in high-speed batch mode...'):
                # 1. Clean the entire column at once (Vectorized/Batch Cleaning)
                df['cleaned_review'] = df['review'].fillna("").apply(clean_text)
                
                # 2. Batch Prediction (MUCH faster than row-by-row)
                cleaned_list = df['cleaned_review'].tolist()
                
                # Predict sentiments for the whole batch
                df['sentiment'] = pipeline.predict(cleaned_list)
                
                # Predict probabilities for the whole batch
                probs = pipeline.predict_proba(cleaned_list)
                df['confidence'] = probs.max(axis=1)
                
                # Metrics Row
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3, m4, m5 = st.columns(5)
                
                with m1:
                    st.markdown(f"""<div class="metric-card"><h3>Total</h3><h2 style="color:#2575fc">{len(df)}</h2></div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-card"><h3>Positive</h3><h2 style="color:#2ecc71">{len(df[df['sentiment']=='positive'])}</h2></div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="metric-card"><h3>Neutral</h3><h2 style="color:#f1c40f">{len(df[df['sentiment']=='neutral'])}</h2></div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""<div class="metric-card"><h3>Negative</h3><h2 style="color:#e74c3c">{len(df[df['sentiment']=='negative'])}</h2></div>""", unsafe_allow_html=True)
                with m5:
                    st.markdown(f"""<div class="metric-card"><h3>Avg Conf.</h3><h2 style="color:#6a11cb">{df['confidence'].mean()*100:.1f}%</h2></div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

                # Visualizations
                col_left, col_right = st.columns([1, 1.2], gap="large")
                
                with col_left:
                    st.markdown("### 🍩 Sentiment Mix")
                    fig_pie = px.pie(df, names='sentiment', hole=0.6, 
                                   color='sentiment',
                                   color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f1c40f'})
                    fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20),
                                          paper_bgcolor='rgba(0,0,0,0)', 
                                          plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_right:
                    st.markdown("### ☁️ Keyword Universe")
                    sentiment_filter = st.radio("Focus on:", ["Positive", "Neutral", "Negative"], horizontal=True)
                    
                    text_data = " ".join(df[df['sentiment'] == sentiment_filter.lower()]['cleaned_review'])
                    if text_data:
                        wc = WordCloud(width=800, height=500, background_color=None, mode="RGBA", colormap='magma').generate(text_data)
                        fig_wc, ax = plt.subplots(facecolor='none')
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_wc)
                    else:
                        st.info(f"No {sentiment_filter} feedback detected.")
                
                # Data Table
                st.markdown("### 📋 Analytical Ledger")
                st.dataframe(df[['review', 'sentiment', 'confidence']].style.background_gradient(subset=['confidence'], cmap='Greens'), use_container_width=True)
                
                # Download
                st.markdown("<br>", unsafe_allow_html=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📩 Download Analyzed Dataset", csv, "sentiment_pro_report.csv", "text/csv")
        else:
            st.error("CSV must contain a column named 'review'.")

elif app_mode == "Social Stream Simulator":
    st.markdown("""
        <div class="hero-container">
            <h1 class='main-title'>Global Stream Simulator</h1>
            <p class='sub-title'>"Real-time Pulse Monitoring of the Digital Collective Consciousness"</p>
        </div>
    """, unsafe_allow_html=True)
    
    import random
    
    samples = [
        "Incredible quality, absolutely loving it!",
        "The battery life is really disappointing.",
        "It's okay I guess, nothing special.",
        "Worst experience ever. Do not buy!",
        "Stunning design and very fast delivery.",
        "Broken after two days of use.",
        "I'm neutral on this product.",
        "Five stars! Exceeded my expectations.",
        "Customer service was rude and unhelpful.",
        "Pretty good value for the money."
    ]
    
    if st.button("🚀 Generate Random Feed Stream"):
        st.markdown("### 🕒 Live Feedback Stream")
        stream_cols = st.columns(2)
        
        test_batch = random.sample(samples, 6)
        
        for i, text in enumerate(test_batch):
            pred, prob = predict_sentiment(text)
            color = "#2ecc71" if pred == 'positive' else "#e74c3c" if pred == 'negative' else "#f1c40f"
            bg = "#ebfaf0" if pred == 'positive' else "#fdedec" if pred == 'negative' else "#fef9e7"
            
            with stream_cols[i % 2]:
                st.markdown(f"""
                    <div style="background:{bg}; padding:15px; border-radius:15px; border-left:5px solid {color}; margin-bottom:10px;">
                        <p style="margin-bottom:5px;"><i>"{text}"</i></p>
                        <b>Sentiment: <span style="color:{color}">{pred.upper()}</span></b> ({max(prob)*100:.1f}%)
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); padding: 40px; border-radius: 25px; border: 1px solid rgba(118, 75, 162, 0.2); text-align: center; margin-top: 30px;">
                <h2 style="color: #4a5568; font-weight: 700;">Experience Real-time Intelligence</h2>
                <p style="color: #718096; font-size: 1.2rem; max-width: 600px; margin: 15px auto;">
                    "Witness how Sentiment Pro processes live data streams to uncover hidden patterns and shifts in public opinion instantly."
                </p>
                <div style="display: flex; justify-content: center; gap: 20px; margin-top: 25px;">
                    <span style="background: white; padding: 8px 15px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; color: #667eea; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">LIVE FEED</span>
                    <span style="background: white; padding: 8px 15px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; color: #764ba2; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">NEURAL ENGINE</span>
                    <span style="background: white; padding: 8px 15px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; color: #2ecc71; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">FAST SYNC</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")

