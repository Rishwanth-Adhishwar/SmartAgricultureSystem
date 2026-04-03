"""
Smart Agriculture System
AI-Powered Farming Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from PIL import Image
import requests

st.set_page_config(
    page_title="Smart Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
GROWTH_CSV = os.path.join(DATASET_DIR, "growth_data.csv")

os.makedirs(UPLOAD_DIR, exist_ok=True)

from utils.fertigation import get_combined_recommendation, CROP_WATER_NEEDS
from utils.visualization import (
    plot_growth_trend,
    plot_market_prices,
    plot_market_comparison_bar,
    plot_confidence_pie,
    plot_seasonal_heatmap
)

import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .st-emotion-cache-16idsys p {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Elegant Title and Header styling */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    h2, h3, h4 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* Sidebar - Soft styling inheriting theme */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(46, 125, 50, 0.2) !important;
    }
    [data-testid="stSidebarNav"] span {
        font-weight: 500;
    }

    /* Hide anchor links */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a, .st-emotion-cache-10y5sf6 {
        display: none !important;
    }
    
    /* Clean Dynamic Cards - Adapts to Light/Dark Mode Native */
    div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stExpander"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(46, 125, 50, 0.15) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border: 1px solid rgba(46, 125, 50, 0.4) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important;
    }

    /* Professional Buttons - Solid Green works beautifully in both modes */
    .stButton>button {
        border-radius: 8px !important;
        background: #2e7d32 !important; 
        color: #FFFFFF !important; 
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.2) !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton>button p, .stButton>button span {
        color: #FFFFFF !important; /* Always keep button text crisp white */
    }
    .stButton>button:hover {
        background: #1b5e20 !important; 
        box-shadow: 0 6px 15px rgba(46, 125, 50, 0.3) !important;
        color: #FFFFFF !important;
    }
    
    /* Input Fields */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        border: 1px solid rgba(46, 125, 50, 0.3) !important;
    }
    
    /* Clean Metrics - Inherits theme but adds green accent and fixes truncation */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color) !important;
        border-left: 5px solid #4CAF50 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        border-top: 1px solid rgba(46, 125, 50, 0.1);
        border-right: 1px solid rgba(46, 125, 50, 0.1);
        border-bottom: 1px solid rgba(46, 125, 50, 0.1);
        display: block !important;
        word-wrap: break-word !important;
        white-space: normal !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] div, div[data-testid="stMetricLabel"] p {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        white-space: normal !important; 
        overflow: visible !important;
        text-overflow: clip !important;
    }
    
    /* Upload section */
    section[data-testid="stFileUploadDropzone"] {
        border-radius: 12px !important;
        border: 2px dashed #4CAF50 !important;
        background-color: var(--secondary-background-color) !important;
        padding: 3rem !important;
    }
    section[data-testid="stFileUploadDropzone"] p, section[data-testid="stFileUploadDropzone"] span {
        font-weight: 600 !important;
    }
    
    /* Dataframe overflow fix */
    [data-testid="stDataFrame"] {
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    /* Proper text wrapping, no overflow */
    .stMarkdown, .stText, [data-testid="stText"] {
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: normal !important;
    }

    /* Beautiful Slide-in Animation dynamically applied to all cards and blocks */
    @keyframes slideUpIn {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stExpander"], div[data-testid="stMetric"], section[data-testid="stFileUploadDropzone"] {
        animation: slideUpIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Tighter Sidebar Metric labels to ensure no truncation */
    [data-testid="stSidebar"] div[data-testid="stMetricLabel"] label {
        font-size: 0.75rem !important;
        letter-spacing: 0px !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }

    /* Spacing between sections */
    hr {
        border-color: rgba(46, 125, 50, 0.2) !important;
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }

    /* Custom Sleek Weather Widget */
    .weather-widget {
        background-color: var(--secondary-background-color) !important;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid rgba(46, 125, 50, 0.2);
        margin-top: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        animation: slideUpIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .weather-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: #2E7D32;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .weather-temp {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.2rem;
    }
    .weather-hum {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-color);
        opacity: 0.7;
    }
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=20.5937&longitude=78.9629&current=temperature_2m,relative_humidity_2m,weather_code&timezone=auto"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            d = r.json()
            return d["current"]["temperature_2m"], d["current"]["relative_humidity_2m"], d["current"]["weather_code"]
    except:
        pass
    return None, None, None


@st.cache_resource(show_spinner="Loading crop models...")
def load_crop_models():
    try:
        with open(os.path.join(MODEL_DIR, "crop_model.pkl"), "rb") as f:
            crop_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "model_metrics.pkl"), "rb") as f:
            metrics = pickle.load(f)
        return crop_model, encoder, metrics
    except:
        return None, None, None


@st.cache_resource(show_spinner="Loading disease models...")
def load_disease_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "disease_model.h5"), compile=False)
        with open(os.path.join(MODEL_DIR, "disease_labels.json"), "r") as f:
            labels = json.load(f)
        return model, labels
    except:
        return None, None


def get_growth_data():
    try:
        if not os.path.exists(GROWTH_CSV):
            df = pd.DataFrame(columns=["day", "height_cm", "notes"])
            df.to_csv(GROWTH_CSV, index=False)
        return pd.read_csv(GROWTH_CSV)
    except:
        return pd.DataFrame(columns=["day", "height_cm", "notes"])


def save_growth_data(df):
    df.to_csv(GROWTH_CSV, index=False)


def predict_crop(model, encoder, n, p, k, temp, humidity, ph, rainfall):
    features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    pred = model.predict(features)
    probs = model.predict_proba(features)[0]
    crop = encoder.inverse_transform(pred)[0]
    return crop, max(probs) * 100, probs, encoder


def detect_disease(model, labels, image):
    try:
        img = image.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr, verbose=0)[0]
        idx = preds.argsort()[::-1][:5]
        return [labels.get(str(i), f"Class_{i}") for i in idx], [float(preds[i]) * 100 for i in idx]
    except:
        return None, None


def format_disease_name(name):
    return name.replace("___", " - ").replace("_", " ").title()


def get_advice(disease):
    advices = {
        "powdery_mildew": "Improve air circulation, apply neem oil or sulfur spray.",
        "blight": "Remove infected plants, apply copper fungicide.",
        "rust": "Remove infected parts, apply sulfur fungicide.",
        "spot": "Remove affected leaves, avoid overhead watering.",
        "virus": "Control insect vectors, remove infected plants.",
        "rot": "Improve drainage, reduce watering.",
        "wilt": "Improve soil drainage, rotate crops."
    }
    d = disease.lower()
    for k, v in advices.items():
        if k in d:
            return v
    return "Consult local agricultural officer."


def get_gemini_response(message):
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        models = genai.list_models()
        model_name = None
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.replace("models/", "")
                break
        
        if not model_name:
            return "No available model found."
        
        model = genai.GenerativeModel(model_name)
        prompt = f"You are a farming advisor. Answer clearly in simple sentences.\n\nQuestion: {message}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"I apologize, I encountered an error: {str(e)}. Please try again."


FARMING_KNOWLEDGE = {
    "plant rice": "The best time to plant rice is during the monsoon season (June-July in India). Rice requires standing water and warm temperatures (20-35°C) for optimal growth.",
    
    "best time to plant rice": "The best time to plant rice is during the monsoon season (June-July in India). Rice requires standing water and warm temperatures (20-35°C) for optimal growth.",
    
    "when to plant rice": "The best time to plant rice is during the monsoon season (June-July in India). Rice requires standing water and warm temperatures (20-35°C) for optimal growth.",
    
    "pest control": "Natural pest control methods include: 1) Neem oil spray - effective against aphids and mites. 2) Companion planting - marigolds repel many pests. 3) Biological control - ladybugs eat aphids. 4) Crop rotation - prevents pest buildup. 5) Hand-picking large pests.",
    
    "how to control pests": "Natural pest control methods include: 1) Neem oil spray - effective against aphids and mites. 2) Companion planting - marigolds repel many pests. 3) Biological control - ladybugs eat aphids. 4) Crop rotation - prevents pest buildup. 5) Hand-picking large pests.",
    
    "control pests naturally": "Natural pest control methods include: 1) Neem oil spray - effective against aphids and mites. 2) Companion planting - marigolds repel many pests. 3) Biological control - ladybugs eat aphids. 4) Crop rotation - prevents pest buildup. 5) Hand-picking large pests.",
    
    "tomato fertilizer": "For tomatoes, use NPK 10-10-10 or 20-20-20 during growth. At flowering, switch to 10-20-20 for more phosphorus. Add calcium to prevent blossom end rot. Apply every 2 weeks.",
    
    "best fertilizer for tomatoes": "For tomatoes, use NPK 10-10-10 or 20-20-20 during growth. At flowering, switch to 10-20-20 for more phosphorus. Add calcium to prevent blossom end rot. Apply every 2 weeks.",
    
    "nitrogen deficiency": "Signs of nitrogen deficiency: 1) Yellowing of older leaves first (bottom up). 2) Stunted growth. 3) Small leaves. 4) Poor yield. Solution: Apply urea or organic compost.",
    
    "signs of nitrogen deficiency": "Signs of nitrogen deficiency: 1) Yellowing of older leaves first (bottom up). 2) Stunted growth. 3) Small leaves. 4) Poor yield. Solution: Apply urea or organic compost.",
    
    "water tomato": "Tomatoes need 1-2 inches of water per week. Water deeply but less frequently. Avoid wetting leaves to prevent disease. Mulch to retain moisture.",
    
    "irrigation": "Modern irrigation methods: 1) Drip irrigation - saves 30-50% water. 2) Sprinkler systems - good for large fields. 3) Furrow irrigation - traditional but less efficient. 4) Schedule irrigation based on crop needs and soil moisture.",
    
    "crop rotation": "Crop rotation benefits: 1) Prevents soil depletion. 2) Reduces pest/disease buildup. 3) Improves soil structure. 4) Increases yield. Example: Rice -> Wheat -> Legumes -> Cotton.",
    
    "organic farming": "Organic farming principles: 1) Use compost and manure instead of chemical fertilizers. 2) Practice crop rotation. 3) Use biological pest control. 4) Avoid genetically modified crops. 5) Maintain soil health through cover cropping.",
    
    "soil testing": "Soil testing steps: 1) Collect soil from 6-8 spots in field. 2) Mix well and take a sample. 3) Send to agricultural lab or use home kit. 4) Test for pH, N, P, K levels. 5) Adjust based on results.",
    
    "compost": "Making compost: 1) Layer green materials (grass, leaves) and brown materials (straw, cardboard). 2) Keep moist but not wet. 3) Turn every 2-3 weeks. 4) Ready in 2-6 months. 5) Use as natural fertilizer.",
    
    "drought": "Drought management: 1) Install drip irrigation. 2) Use mulch to retain moisture. 3) Plant drought-resistant varieties. 4) Practice rainwater harvesting. 5) Apply wetting agents to soil.",
    
    "fertilizer": "Common fertilizers: 1) Urea - provides nitrogen. 2) DAP - provides phosphorus. 3) MOP - provides potassium. 4) NPK复合肥 - balanced nutrition. 5) Apply based on soil test results.",
    
    "wheat": "Wheat cultivation: Sow in November-December (rabi season). Requires cool climate (15-20°C). Harvest in April-May. Apply NPK before sowing. Irrigate 4-5 times during growth period.",
    
    "cotton": "Cotton cultivation: Sow in June-July (kharif). Requires warm climate (25-35°C). Harvest in October-December. Needs well-drained soil. Apply heavy NPK at boll formation.",
    
    "maize": "Maize cultivation: Sow in June-July or February-March. Requires warm weather (20-30°C). Harvest in 60-90 days. Needs fertile, well-drained soil. Apply NPK at planting and knee height.",
    
    "vegetable": "General vegetable tips: 1) Plant in well-drained soil. 2) Water morning hours. 3) Use mulch to control weeds. 4) Harvest regularly for continuous production. 5) Watch for pest signs.",
    
    "irrigate": "General irrigation tips: 1) Water early morning or evening. 2) Avoid midday watering. 3) Deep watering less often is better than shallow frequent watering. 4) Check soil moisture before irrigating. 5) Use drip irrigation for efficiency.",
    
    "yellow leaves": "Yellow leaves causes: 1) Nitrogen deficiency - apply urea. 2) Overwatering - reduce irrigation. 3) Poor drainage - improve soil. 4) Pest/disease - inspect and treat. 5) Natural aging - normal for older leaves.",
    
    "diseased": "Common plant diseases: 1) Powdery mildew - white powder on leaves, treat with neem oil. 2) Black spot - dark spots, remove affected leaves. 3) Rust - orange spots, apply sulfur. 4) Blight - wilting, remove infected plants.",
    
    "yield": "Improve crop yield: 1) Use quality seeds. 2) Test soil and apply right fertilizers. 3) Practice proper spacing. 4) Control pests regularly. 5) Use drip irrigation. 6) Apply mulch for moisture retention.",
}


def get_farming_response(message):
    message_lower = message.lower()
    
    for key, response in FARMING_KNOWLEDGE.items():
        if key in message_lower:
            return response
    
    general_responses = {
        "hello": "Hello! I'm your farming assistant. Ask me about crop cultivation, pest control, fertilizer, irrigation, or any farming topic.",
        "hi": "Hello! I'm your farming assistant. Ask me about crop cultivation, pest control, fertilizer, irrigation, or any farming topic.",
        "help": "I can help you with: 1) Crop planting times. 2) Pest control methods. 3) Fertilizer recommendations. 4) Irrigation techniques. 5) Soil management. Just ask your question!",
    }
    
    for key, response in general_responses.items():
        if key in message_lower:
            return response
    
    return "I'm a farming assistant. I can help with crop cultivation, pest control, fertilizer recommendations, irrigation tips, and more. Please ask a specific farming question!"


def render_sidebar():
    st.sidebar.title("🌱 Smart Agri")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", [
        "Home",
        "Crop Prediction",
        "Disease Detection",
        "Growth Monitoring",
        "Market Analysis",
        "Fertigation System",
        "AI Chatbot"
    ])
    
    temp, humidity, code = get_weather()
    if temp:
        icons = {0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️", 51: "🌧️", 61: "🌧️", 95: "⛈️"}
        icon = icons.get(code, "🌤️")
        
        weather_html = f"""
        <div class="weather-widget">
            <div class="weather-header">Current Weather</div>
            <div class="weather-temp">{icon} {temp}°C</div>
            <div class="weather-hum">💧 Humidity: {humidity}%</div>
        </div>
        """
        st.sidebar.markdown(weather_html, unsafe_allow_html=True)
    
    return page


def render_home():
    st.title("🌱 Smart Agriculture System")
    st.caption("AI-Powered Farming Assistant")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.subheader("🌾 Crop Prediction")
            st.caption("ML-based crop recommendation based on soil and climate conditions")
    with col2:
        with st.container(border=True):
            st.subheader("🔬 Disease Detection")
            st.caption("CNN-powered leaf disease analysis using deep learning")
    with col3:
        with st.container(border=True):
            st.subheader("📈 Growth Monitoring")
            st.caption("Track and visualize crop growth over time")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        with st.container(border=True):
            st.subheader("💹 Market Analysis")
            st.caption("Analyze crop price trends and seasonal patterns")
    with col5:
        with st.container(border=True):
            st.subheader("💧 Fertigation System")
            st.caption("Smart irrigation and fertilizer recommendations")
    with col6:
        with st.container(border=True):
            st.subheader("🤖 AI Chatbot")
            st.caption("Get expert farming advice powered by AI")
    
    st.markdown("---")
    st.subheader("📊 System Status")
    
    # Check status without loading heavy ML/TF models to avoid massive startup latency
    crop_ready = os.path.exists(os.path.join(MODEL_DIR, "crop_model.pkl"))
    dis_ready = os.path.exists(os.path.join(MODEL_DIR, "disease_model.h5"))
    df = get_growth_data()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Crop Model", "Ready" if crop_ready else "Not Found", "Random Forest" if crop_ready else None)
    with c2:
        st.metric("Disease Model", "Ready" if dis_ready else "Not Found", "CNN Available" if dis_ready else None)
    with c3:
        st.metric("Growth Records", len(df), "entries" if len(df) != 1 else "entry")


def render_crop_prediction():
    st.title("🌾 Crop Prediction")
    st.caption("Get AI-powered crop recommendations based on soil and climate data")
    
    crop_m, enc, met = load_crop_models()
    if not crop_m:
        st.error("Crop model not found. Please run the training script first.")
        with st.expander("How to train the model"):
            st.code("python training/train_crop_model.py")
        return
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Soil Parameters")
            n = st.number_input("Nitrogen (N)", 0, 200, 80, help="Soil nitrogen content (kg/ha)")
            p = st.number_input("Phosphorus (P)", 0, 200, 45, help="Soil phosphorus content (kg/ha)")
            k = st.number_input("Potassium (K)", 0, 200, 40, help="Soil potassium content (kg/ha)")
        
        with col2:
            st.subheader("Climate Parameters")
            temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, help="Average temperature")
            humidity = st.slider("Humidity (%)", 0, 100, 70, help="Relative humidity")
            ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1, help="Soil acidity level")
            rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0, help="Annual rainfall")
    
    if st.button("Predict Recommended Crop", type="primary"):
        crop, conf, probs, encoder = predict_crop(crop_m, enc, n, p, k, temp, humidity, ph, rain)
        
        st.markdown("---")
        
        with st.container(border=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"**Recommended Crop:** {crop.upper()}")
            with col2:
                st.metric("Confidence", f"{conf:.1f}%")
        
        st.subheader("📋 Crop Details")
        crop_info = get_crop_details(crop)
        if crop_info:
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Growth Duration", crop_info.get("duration", "N/A"))
                with col2:
                    st.metric("Water Need", crop_info.get("water", "N/A"))
                with col3:
                    st.metric("Best Season", crop_info.get("season", "N/A"))
                
                st.markdown(f"**Description:** {crop_info.get('description', '')}")
                st.markdown(f"**Soil Requirements:** {crop_info.get('soil', '')}")
                st.markdown(f"**Climate:** {crop_info.get('climate', '')}")
        
        st.subheader("🏆 Top 5 Recommendations")
        top_idx = probs.argsort()[::-1][:5]
        top_crops = encoder.inverse_transform(top_idx)
        top_probs = [probs[i] * 100 for i in top_idx]
        
        for i, (c, pr) in enumerate(zip(top_crops, top_probs)):
            bar_width = int(pr / 2)
            bar = "█" * bar_width + "░" * (50 - bar_width)
            st.write(f"**{i+1}. {c.title()}** |{bar}| {pr:.1f}%")
        
        fig = plot_confidence_pie(list(top_crops), top_probs)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Probability Details"):
            df = pd.DataFrame({
                "Rank": range(1, 6),
                "Crop": [c.title() for c in top_crops],
                "Probability": [f"{p:.1f}%" for p in top_probs]
            })
            st.dataframe(df, use_container_width=True, hide_index=True)


def get_crop_details(crop):
    details = {
        "rice": {
            "duration": "120-150 days",
            "water": "High (flooded)",
            "season": "Kharif (Jun-Oct)",
            "description": "Rice is a staple food crop grown in flooded fields. It's a major source of carbohydrates for billions of people worldwide.",
            "soil": "Clayey soil with good water retention. pH 5.5-7.0",
            "climate": "Warm (20-35°C), high humidity, heavy rainfall"
        },
        "wheat": {
            "duration": "120-150 days",
            "water": "Medium",
            "season": "Rabi (Nov-Apr)",
            "description": "Wheat is the second most important staple food crop. It's rich in carbohydrates and provides essential proteins.",
            "soil": "Loamy soil, well-drained. pH 6.0-7.5",
            "climate": "Cool (15-25°C), moderate rainfall"
        },
        "maize": {
            "duration": "60-90 days",
            "water": "Medium-High",
            "season": "Kharif (Jun-Oct)",
            "description": "Maize (corn) is a versatile crop used for food, feed, and industrial purposes. High in carbohydrates and vitamins.",
            "soil": "Well-drained loamy soil. pH 5.8-7.0",
            "climate": "Warm (20-30°C), adequate rainfall"
        },
        "cotton": {
            "duration": "150-180 days",
            "water": "Low-Medium",
            "season": "Kharif (Jun-Dec)",
            "description": "Cotton is a major cash crop grown for its fiber. It's used in textile industry worldwide.",
            "soil": "Black cotton soil. pH 5.5-8.0",
            "climate": "Warm (20-30°C), dry climate preferred"
        },
        "sugarcane": {
            "duration": "12-18 months",
            "water": "High",
            "season": "Planting: Feb-Mar",
            "description": "Sugarcane is grown for sugar production and jaggery. It's also used for biofuel and animal feed.",
            "soil": "Deep, fertile soil. pH 6.0-7.5",
            "climate": "Tropical (20-35°C), high humidity"
        },
        "potato": {
            "duration": "90-120 days",
            "water": "Medium",
            "season": "Rabi (Oct-Mar)",
            "description": "Potato is a major vegetable crop and staple food. It's rich in carbohydrates and vitamins.",
            "soil": "Sandy loam, loose. pH 5.0-6.5",
            "climate": "Cool (15-25°C), moderate rainfall"
        },
        "tomato": {
            "duration": "60-90 days",
            "water": "Medium",
            "season": "All seasons",
            "description": "Tomato is a popular vegetable rich in lycopene and vitamins. Used in cooking and salads.",
            "soil": "Well-drained, fertile. pH 6.0-6.8",
            "climate": "Warm (20-30°C), frost-free"
        },
    }
    return details.get(crop.lower(), {
        "duration": "Varies",
        "water": "Medium",
        "season": "Varies",
        "description": f"{crop.title()} is a crop with specific requirements.",
        "soil": "Well-drained fertile soil",
        "climate": "Moderate temperature and rainfall"
    })


def render_disease_detection():
    st.title("🔬 Disease Detection")
    st.caption("Upload a leaf image for AI-powered disease analysis")
    
    dis_m, dis_l = load_disease_model()
    if not dis_m:
        st.error("Disease model not found. Please run the training script first.")
        with st.expander("How to train the model"):
            st.code("python training/train_disease_model.py")
        return
    
    img = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png", "bmp"])
    
    if img:
        try:
            image = Image.open(img).convert("RGB")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    labels, confs = detect_disease(dis_m, dis_l, image)
                
                if labels:
                    name = format_disease_name(labels[0])
                    
                    st.markdown("---")
                    
                    if "healthy" in labels[0].lower():
                        with st.container(border=True):
                            st.success(f"✅ **Healthy Plant**")
                            st.metric("Confidence", f"{confs[0]:.1f}%")
                            st.info("This plant appears to be healthy. Continue with regular care.")
                        
                        with st.expander("🌱 General Care Tips"):
                            st.write("""
                            - **Watering**: Water at base, avoid wetting leaves
                            - **Fertilizing**: Apply balanced NPK monthly
                            - **Pruning**: Remove dead leaves regularly
                            - **Monitoring**: Check weekly for any changes
                            - **Sunlight**: Ensure adequate light exposure
                            """)
                    else:
                        with st.container(border=True):
                            st.warning(f"⚠️ **Disease Detected:** {name}")
                            st.metric("Confidence", f"{confs[0]:.1f}%")
                        
                        disease_info = get_disease_info(labels[0])
                        
                        with st.container(border=True):
                            st.subheader("📋 Disease Information")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Severity", disease_info.get("severity", "Unknown"))
                            with col2:
                                st.metric("Spread", disease_info.get("spread", "Unknown"))
                            
                            st.markdown(f"**Description:** {disease_info.get('description', '')}")
                        
                        with st.container(border=True):
                            st.subheader("💊 Treatment & Recommendations")
                            st.markdown(f"**Immediate Action:** {disease_info.get('treatment', '')}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Chemical Treatment:**")
                                st.write(disease_info.get('chemical', ''))
                            with col2:
                                st.markdown("**Organic Treatment:**")
                                st.write(disease_info.get('organic', ''))
                        
                        with st.expander("🔧 Prevention Tips"):
                            st.write(f"**Preventive Measures:** {disease_info.get('prevention', '')}")
                        
                        st.markdown("### ⚠️ Important Notes")
                        st.warning("""
                        - Isolate affected plant from healthy ones
                        - Monitor nearby plants for symptoms
                        - Consider consulting local agricultural expert
                        - Do not eat or consume affected plant parts
                        """)
                    
                    with st.expander("📊 Detailed Predictions"):
                        df = pd.DataFrame({
                            "Rank": range(1, 6),
                            "Condition": [format_disease_name(l) for l in labels[:5]],
                            "Confidence": [f"{c:.1f}%" for c in confs[:5]]
                        })
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=[format_disease_name(l) for l in labels[:5]],
                            values=confs[:5],
                            hole=0.4,
                            marker_colors=['#2E7D32', '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B']
                        )])
                        fig.update_layout(title="Prediction Confidence Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not analyze the image. Please try again.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    with st.expander("📷 Tips for best results"):
        st.write("""
        - Use clear, well-lit images
        - Focus on the affected area
        - Avoid blurry or dark images
        - Supported formats: JPG, PNG, BMP
        - Include both healthy and affected parts if possible
        """)


def get_disease_info(disease):
    info = {
        "powdery_mildew": {
            "severity": "Medium",
            "spread": "Moderate",
            "description": "Powdery mildew is a fungal disease that appears as white powdery coating on leaves, stems, and fruits. It thrives in warm, humid conditions.",
            "treatment": "Remove affected leaves immediately. Improve air circulation. Apply fungicide.",
            "chemical": "Sulfur-based fungicides, neem oil, or potassium bicarbonate (1 tbsp per gallon)",
            "organic": "Mix 1 tsp baking soda + 1 quart water + few drops mild soap. Spray weekly.",
            "prevention": "Ensure good air circulation, avoid overhead watering, plant resistant varieties, maintain proper spacing"
        },
        "blight": {
            "severity": "High",
            "spread": "Fast",
            "description": "Blight is a serious fungal/bacterial disease causing rapid leaf death, brown lesions, and plant collapse. Can destroy entire crops.",
            "treatment": "Remove and destroy infected plants immediately. Apply copper fungicide.",
            "chemical": "Copper-based fungicides (Copper sulfate, Bordeaux mixture)",
            "organic": "Remove infected parts, apply compost tea, use biological controls",
            "prevention": "Use resistant varieties, rotate crops, avoid overhead irrigation, remove plant debris"
        },
        "rust": {
            "severity": "Medium",
            "spread": "Moderate",
            "description": "Rust appears as orange or brown pustules on leaves and stems. Caused by fungal pathogens that spread through spores.",
            "treatment": "Remove infected leaves. Apply sulfur fungicide. Improve air flow.",
            "chemical": "Sulfur fungicides, copper oxide, or mancozeb",
            "organic": "Baking soda spray, neem oil, milk spray (1:9 milk to water)",
            "prevention": "Plant resistant varieties, ensure good air circulation, remove infected debris, avoid wet foliage"
        },
        "spot": {
            "severity": "Low-Medium",
            "spread": "Slow",
            "description": "Leaf spot diseases cause dark or discolored spots on leaves. Usually caused by fungi or bacteria. Often cosmetic but can weaken plant.",
            "treatment": "Remove affected leaves. Avoid overhead watering. Apply appropriate fungicide.",
            "chemical": "Copper fungicides or chlorothalonil",
            "organic": "Neem oil, baking soda spray, remove affected leaves",
            "prevention": "Avoid overhead watering, ensure good air flow, remove plant debris, use clean tools"
        },
        "virus": {
            "severity": "High",
            "spread": "Fast (via vectors)",
            "description": "Plant viruses cause mottling, stunted growth, and deformities. Spread by insects, nematodes, or contaminated tools.",
            "treatment": "Remove infected plants immediately. Control insect vectors.",
            "chemical": "No direct cure - focus on vector control with insecticides",
            "organic": "Remove infected plants, control aphids/whiteflies with insecticidal soap",
            "prevention": "Use virus-free seeds, control insect vectors, sanitize tools, remove weed hosts"
        },
        "rot": {
            "severity": "High",
            "spread": "Fast",
            "description": "Root/stem rot causes wilting, yellowing, and mushy roots. Caused by overwatering or soil-borne fungi.",
            "treatment": "Reduce watering immediately. Improve drainage. Apply fungicide to soil.",
            "chemical": "Metalaxyl, fosetyl-aluminum, or copper fungicides",
            "organic": "Improve drainage, add perlite to soil, apply cinnamon to affected areas",
            "prevention": "Avoid overwatering, ensure proper drainage, use sterile soil, don't overcrowd plants"
        },
        "wilt": {
            "severity": "Medium-High",
            "spread": "Moderate",
            "description": "Wilt diseases cause drooping leaves and stems even with adequate water. Caused by soil-borne fungi or bacteria blocking water flow.",
            "treatment": "Improve soil drainage. Remove severely infected plants. Solarize soil.",
            "chemical": "Soil fumigation (professional only), copper-based fungicides",
            "organic": "Add compost to improve soil health, crop rotation, solarize soil",
            "prevention": "Practice crop rotation, use resistant varieties, ensure good drainage, avoid wounding plants"
        },
        "healthy": {
            "severity": "None",
            "spread": "N/A",
            "description": "The plant appears healthy with no visible signs of disease or stress.",
            "treatment": "Continue regular care routine. Maintain good growing conditions.",
            "chemical": "None needed",
            "organic": "Continue balanced fertilization, regular monitoring",
            "prevention": "Maintain proper watering, adequate nutrients, good air circulation, regular monitoring"
        }
    }
    
    for key, value in info.items():
        if key in disease.lower():
            return value
    
    return {
        "severity": "Unknown",
        "spread": "Unknown",
        "description": "A disease condition was detected. Please consult local agricultural expert for specific diagnosis.",
        "treatment": "Remove affected plant parts. Monitor closely. Consult expert.",
        "chemical": "Consult agricultural expert",
        "organic": "Improve growing conditions, ensure good air flow",
        "prevention": "Regular monitoring, proper plant care, good hygiene"
    }


import plotly.graph_objects as go
import plotly.express as px


def render_growth_monitoring():
    st.title("📈 Growth Monitoring")
    st.caption("Track and visualize your crop growth over time with detailed analytics")
    
    if "growth_init" not in st.session_state:
        st.session_state.growth_init = False
    
    df = get_growth_data()
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "➕ Add Entry", "📋 Settings"])
    
    with tab3:
        st.subheader("🌱 Crop Setup")
        with st.container(border=True):
            crop_name = st.text_input("Crop Name", value="My Crop", help="Enter your crop name")
            planting_date = st.date_input("Planting Date", help="When did you plant?")
        
        st.markdown("---")
        
        if st.button("Start New Crop", type="primary"):
            df = pd.DataFrame(columns=["day", "height_cm", "notes"])
            save_growth_data(df)
            st.session_state.growth_init = True
            st.success(f"New crop '{crop_name}' started! Now add your first entry.")
            st.rerun()
        
        if st.button("Clear All Data", type="secondary"):
            df = pd.DataFrame(columns=["day", "height_cm", "notes"])
            save_growth_data(df)
            st.session_state.growth_init = False
            st.success("All growth data cleared!")
            st.rerun()
    
    with tab1:
        if len(df) > 0:
            df = df.sort_values("day").reset_index(drop=True)
            
            st.subheader("📈 Growth Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Days", len(df), delta=f"+{len(df)} days")
            with col2:
                st.metric("Current Height", f"{df['height_cm'].iloc[-1]:.1f} cm")
            with col3:
                if len(df) > 1:
                    growth_rate = (df['height_cm'].iloc[-1] - df['height_cm'].iloc[0]) / df['day'].iloc[-1]
                    st.metric("Avg Growth Rate", f"{growth_rate:.2f} cm/day")
                else:
                    st.metric("Growth Rate", "N/A")
            with col4:
                max_height = df['height_cm'].max()
                max_day = df.loc[df['height_cm'].idxmax(), 'day']
                st.metric("Max Height", f"{max_height:.1f} cm (Day {int(max_day)})")
            
            fig = plot_growth_trend(df)
            st.plotly_chart(fig, use_container_width=True)
            
            if len(df) > 1:
                st.subheader("📊 Growth Analytics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Weekly Growth Rate**")
                    df['week'] = (df['day'] - 1) // 7 + 1
                    weekly = df.groupby('week')['height_cm'].agg(['first', 'last'])
                    weekly['growth'] = weekly['last'] - weekly['first']
                    
                    for week, row in weekly.iterrows():
                        if row['growth'] > 0:
                            st.write(f"Week {int(week)}: +{row['growth']:.1f} cm")
                        elif row['growth'] < 0:
                            st.write(f"Week {int(week)}: {row['growth']:.1f} cm")
                        else:
                            st.write(f"Week {int(week)}: No change")
                
                with col2:
                    st.markdown("**Growth Trend Analysis**")
                    if len(df) >= 3:
                        recent = df.tail(3)
                        if recent['height_cm'].iloc[-1] > recent['height_cm'].iloc[0]:
                            st.success("📈 Growth is accelerating!")
                        elif recent['height_cm'].iloc[-1] < recent['height_cm'].iloc[0]:
                            st.warning("📉 Growth is declining - check conditions!")
                        else:
                            st.info("➡️ Growth is stable")
            
            st.subheader("📝 Growth Log")
            display_df = df.copy()
            display_df["height_cm"] = display_df["height_cm"].apply(lambda x: f"{x:.1f} cm")
            
            if len(display_df) > 10:
                st.dataframe(display_df.tail(10), use_container_width=True, hide_index=True)
                with st.expander("View All Entries"):
                    st.dataframe(display_df.sort_values("day", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.dataframe(display_df.sort_values("day", ascending=False), use_container_width=True, hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button("📥 Download Data (CSV)", csv, "growth_data.csv", "text/csv")
            with col2:
                if st.button("🗑️ Delete Last Entry"):
                    if len(df) > 0:
                        df = df.iloc[:-1]
                        save_growth_data(df)
                        st.success("Last entry deleted!")
                        st.rerun()
        else:
            st.info("🌱 No growth data yet. Go to the 'Settings' tab to start tracking!")
            
            with st.expander("📖 How to use Growth Monitoring"):
                st.write("""
                **Getting Started:**
                1. Go to the **Settings** tab
                2. Enter your crop name and planting date
                3. Click **Start New Crop**
                4. Go to **Add Entry** tab to record growth data
                
                **Tips:**
                - Add entries regularly (every 1-7 days)
                - Record height in centimeters
                - Add notes about weather, fertilizer, etc.
                - Check Dashboard for growth analysis
                """)
    
    with tab2:
        with st.container(border=True):
            st.subheader("➕ Add New Entry")
            
            df = get_growth_data()
            
            next_day = 1
            if len(df) > 0:
                next_day = int(df['day'].max()) + 1
            
            col1, col2 = st.columns(2)
            with col1:
                day = st.number_input("Day Number", min_value=1, max_value=365, value=next_day, help="Day since planting")
            with col2:
                height = st.number_input("Height (cm)", min_value=0.0, max_value=500.0, value=10.0, help="Plant height in centimeters")
            
            notes = st.text_area("Notes (optional)", placeholder="e.g., Applied fertilizer, heavy rain, temperature... ")
            
            st.markdown("**Quick Tags:**")
            col1, col2, col3, col4 = st.columns(4)
            tag = ""
            if col1.button("🌧️ Rain"):
                tag = "Rain"
            if col2.button("☀️ Sunny"):
                tag = "Sunny"
            if col3.button("💧 Irrigated"):
                tag = "Irrigated"
            if col4.button("🧪 Fertilized"):
                tag = "Fertilized"
            
            if tag:
                notes = f"{notes} [{tag}]" if notes else f"[{tag}]"
                st.rerun()
            
            if st.button("💾 Save Entry", type="primary"):
                new = pd.DataFrame({"day": [day], "height_cm": [height], "notes": [notes]})
                df = pd.concat([df, new], ignore_index=True)
                save_growth_data(df)
                st.success(f"✅ Entry saved: Day {day}, Height {height} cm")
                st.rerun()
        
        if len(df) > 0:
            st.markdown("---")
            st.subheader("📋 Recent Entries")
            recent_df = df.tail(5).copy()
            recent_df["height_cm"] = recent_df["height_cm"].apply(lambda x: f"{x:.1f} cm")
            st.dataframe(recent_df.sort_values("day", ascending=False), use_container_width=True, hide_index=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data():
    try:
        return pd.read_csv(os.path.join(DATASET_DIR, "market_prices.csv"))
    except FileNotFoundError:
        return None

def render_market_analysis():
    st.title("💹 Market Analysis")
    st.caption("Analyze crop price trends and seasonal patterns")
    
    mkt = load_market_data()
    if mkt is None:
        st.error("Market data file not found.")
        return
        
    crops = [c for c in mkt.columns if c != "month"]
    
    selected = st.multiselect("Select crops to compare", crops, default=["rice", "wheat", "cotton"])
    
    if selected:
        st.markdown("---")
        
        st.subheader("📊 Price Overview")
        overview_data = []
        for crop in selected:
            if crop in mkt.columns:
                overview_data.append({
                    "Crop": crop.title(),
                    "Min Price": f"₹{mkt[crop].min():.0f}",
                    "Max Price": f"₹{mkt[crop].max():.0f}",
                    "Avg Price": f"₹{mkt[crop].mean():.0f}",
                    "Current": f"₹{mkt[crop].iloc[-1]:.0f}",
                    "Trend": "📈" if mkt[crop].iloc[-1] > mkt[crop].iloc[0] else "📉"
                })
        
        overview_df = pd.DataFrame(overview_data)
        st.dataframe(overview_df, use_container_width=True, hide_index=True)
        
        t1, t2, t3, t4 = st.tabs(["📈 Price Trends", "📊 Comparison", "🗓️ Seasonal Heatmap", "📋 Detailed Analysis"])
        
        with t1:
            fig = plot_market_prices(mkt, selected)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📖 How to read this chart"):
                st.write("""
                - This chart shows price variation over 12 months
                - Higher prices indicate scarcity or high demand
                - Seasonal patterns help plan when to sell
                - Compare multiple crops to find best marketing timing
                """)
        
        with t2:
            fig = plot_market_comparison_bar(mkt, selected)
            st.plotly_chart(fig, use_container_width=True)
            
            for crop in selected:
                if crop in mkt.columns:
                    avg = mkt[crop].mean()
                    st.metric(f"{crop.title()} Average", f"₹{avg:.0f}")
        
        with t3:
            crop = st.selectbox("Select crop for seasonal analysis", selected)
            fig = plot_seasonal_heatmap(mkt, crop)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📖 Quarterly Analysis"):
                quarters = ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"]
                q_data = []
                for q_idx in range(4):
                    start = q_idx * 3
                    end = min(start + 3, 12)
                    q_prices = mkt[crop].iloc[start:end]
                    q_data.append({
                        "Quarter": quarters[q_idx],
                        "Average": f"₹{q_prices.mean():.0f}",
                        "Min": f"₹{q_prices.min():.0f}",
                        "Max": f"₹{q_prices.max():.0f}"
                    })
                st.dataframe(pd.DataFrame(q_data), use_container_width=True, hide_index=True)
        
        with t4:
            st.subheader(f"📋 Detailed Analysis: {selected[0].title()}")
            crop = selected[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Price Statistics**")
                stats = {
                    "Minimum": f"₹{mkt[crop].min():.0f}",
                    "Maximum": f"₹{mkt[crop].max():.0f}",
                    "Mean": f"₹{mkt[crop].mean():.0f}",
                    "Median": f"₹{mkt[crop].median():.0f}",
                    "Std Dev": f"₹{mkt[crop].std():.0f}"
                }
                for k, v in stats.items():
                    st.write(f"**{k}:** {v}")
            
            with col2:
                best_month = mkt.loc[mkt[crop].idxmax(), "month"]
                worst_month = mkt.loc[mkt[crop].idxmin(), "month"]
                st.markdown("**Best Time to Sell**")
                st.success(f"🎯 Best Month: {best_month} (₹{mkt[crop].max():.0f})")
                st.warning(f"⚠️ Lowest Month: {worst_month} (₹{mkt[crop].min():.0f})")
            
            st.markdown("### 💡 Selling Recommendations")
            
            if mkt[crop].iloc[-1] > mkt[crop].mean():
                st.success("✅ Prices are currently ABOVE average - Good time to sell!")
            else:
                st.info("📉 Prices are currently BELOW average - Consider waiting for better prices")
            
            best_season = mkt.loc[mkt[crop].idxmax(), "month"]
            st.write(f"**Best selling season:** {best_season}")
            
            with st.expander("📊 Monthly Breakdown"):
                monthly_df = pd.DataFrame({
                    "Month": mkt["month"],
                    "Price": mkt[crop],
                    "vs Average": [(f"+₹{x-mkt[crop].mean():.0f}" if x > mkt[crop].mean() else f"-₹{mkt[crop].mean()-x:.0f}") for x in mkt[crop]]
                })
                st.dataframe(monthly_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Please select at least one crop to analyze")
        
        with st.expander("📖 How to use Market Analysis"):
            st.write("""
            **This tool helps you:**
            - Compare prices across different crops
            - Identify best months to sell
            - Understand seasonal price patterns
            - Make informed selling decisions
            
            **Tips:**
            - Select multiple crops for comparison
            - Check all tabs for detailed insights
            - Use seasonal heatmap for quarterly planning
            """)


def render_fertigation():
    st.title("💧 Fertigation System")
    st.caption("Smart irrigation and fertilizer recommendations for your crops")
    
    crop = st.selectbox("Select Crop", sorted(CROP_WATER_NEEDS.keys()))
    stage = st.selectbox("Growth Stage", ["Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"])
    moisture = st.slider("Soil Moisture (%)", 0, 100, 35, help="Current soil moisture level")
    
    if st.button("Get Recommendation", type="primary"):
        rec = get_combined_recommendation(crop, moisture, stage.lower())
        irr = rec["irrigation"]
        fert = rec["fertilizer"]
        
        st.markdown("---")
        
        with st.container(border=True):
            st.subheader("💦 Irrigation Plan")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Status", irr["moisture_status"].split(" - ")[0])
            with c2:
                st.metric("Water Required", f"{irr['water_required_mm']} mm/day")
            with c3:
                st.metric("Irrigation Amount", f"{irr['irrigation_amount_mm']} mm")
            
            urgency = irr["urgency"]
            if "IMMEDIATE" in urgency:
                st.error("🚨 URGENT: Irrigate immediately!")
            elif "SOON" in urgency:
                st.warning("Irrigate within 24 hours")
            elif "MONITOR" in urgency:
                st.info("Monitor moisture levels")
            else:
                st.success("No irrigation needed")
            
            st.caption(f"**Recommended Frequency:** {irr['frequency']}")
            
            with st.expander("💧 Irrigation Guide"):
                st.write(f"""
                **Current Status:** {irr['moisture_status']}
                
                **Recommended Actions:**
                - Water early morning (6-8 AM) or evening (4-6 PM)
                - Avoid midday watering to reduce evaporation
                - Use drip irrigation for efficiency
                - Apply mulch to retain moisture
                - Check soil moisture regularly
                """)
            
            st.subheader("📅 Weekly Schedule")
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            sched = []
            for d in days:
                if irr["frequency"] == "Daily":
                    sched.append({"Day": d, "Action": f"Irrigate ({irr['irrigation_amount_mm']}mm)", "Time": "Morning"})
                elif d in ["Mon", "Wed", "Fri", "Sun"]:
                    sched.append({"Day": d, "Action": f"Irrigate ({irr['irrigation_amount_mm']}mm)", "Time": "Morning"})
                else:
                    sched.append({"Day": d, "Action": "Check moisture", "Time": "-"})
            
            sched_df = pd.DataFrame(sched)
            st.dataframe(sched_df, use_container_width=True, hide_index=True)
        
        with st.container(border=True):
            st.subheader("🌱 Fertilizer Plan")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Fertilizer", fert["fertilizer_name"])
            with c2:
                st.metric("Nitrogen (N)", f"{fert['nitrogen_kg_ha']} kg/ha")
            with c3:
                st.metric("Phosphorus (P)", f"{fert['phosphorus_kg_ha']} kg/ha")
            with c4:
                st.metric("Potassium (K)", f"{fert['potassium_kg_ha']} kg/ha")
            
            total = fert["total_npk_kg_ha"]
            if total > 0:
                st.success(f"✅ Apply {fert['fertilizer_name']} at {total} kg/ha NPK")
                st.caption("💡 Best time: Early morning or late evening")
                
                with st.expander("🌿 Fertilizer Application Guide"):
                    st.write(f"""
                    **Recommended Dose:** {total} kg/ha
                    
                    **Application Method:**
                    1. Dissolve in water for fertigation
                    2. Apply during irrigation
                    3. For foliar spray, dilute appropriately
                    
                    **Timing:**
                    - Best: Early morning or late evening
                    - Avoid: Hot midday sun
                    
                    **Tips:**
                    - Wear protective gear when handling
                    - Store in cool, dry place
                    - Follow package instructions
                    - Don't mix with other chemicals unless specified
                    """)
            else:
                st.success("✅ No fertilizer needed at this growth stage")
                st.caption("At maturity stage, focus on harvesting rather than fertilization")
        
        st.markdown("---")
        st.subheader("📋 Summary")
        
        summary = f"""
        **{crop.title()} - {stage.title()} Stage**
        
        💧 **Irrigation:** {irr['irrigation_amount_mm']} mm per session
        📅 **Frequency:** {irr['frequency']}
        ⚠️ **Urgency:** {urgency}
        
        🌱 **Fertilizer:** {fert['fertilizer_name']}
        📊 **NPK Total:** {total} kg/ha
        """
        st.info(summary)


def render_chatbot():
    st.title("🤖 AI Chatbot")
    st.caption("Get expert farming advice powered by AI")
    
    if not GEMINI_API_KEY:
        st.warning("⚠️ Gemini API key not configured. Set GEMINI_API_KEY environment variable to enable AI chat.")
    
    st.markdown("---")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for role, msg in st.session_state.chat_history:
        with st.container():
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**AgriBot:** {msg}")
            st.markdown("---")
    
    st.subheader("Quick Questions")
    q1, q2 = st.columns(2)
    q_selected = None
    if q1.button("When to plant rice?"):
        q_selected = "What is the best time to plant rice?"
    if q2.button("Pest control tips?"):
        q_selected = "How to control pests naturally?"
    
    q3, q4 = st.columns(2)
    if q3.button("Tomato fertilizer?"):
        q_selected = "What is the best fertilizer for tomatoes?"
    if q4.button("N deficiency signs?"):
        q_selected = "What are signs of nitrogen deficiency in plants?"
    
    st.markdown("---")
    st.subheader("Ask a Question")
    
    msg = st.text_input("Your question:", placeholder="Type your farming question here...")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send = st.button("Send", type="primary")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    if (send or q_selected) and (msg or q_selected):
        question = msg if msg else q_selected
        st.session_state.chat_history.append(("user", question))
        
        with st.spinner("Thinking..."):
            response = get_gemini_response(question)
        
        st.session_state.chat_history.append(("bot", response))
        st.rerun()


def main():
    page = render_sidebar()
    
    if page == "Home":
        render_home()
    elif page == "Crop Prediction":
        render_crop_prediction()
    elif page == "Disease Detection":
        render_disease_detection()
    elif page == "Growth Monitoring":
        render_growth_monitoring()
    elif page == "Market Analysis":
        render_market_analysis()
    elif page == "Fertigation System":
        render_fertigation()
    elif page == "AI Chatbot":
        render_chatbot()


if __name__ == "__main__":
    main()
