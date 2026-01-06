import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# -----------------------------
# Background Image Function
# -----------------------------
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🔹 Set your background image file name here
set_background("434135.jpg")

# -----------------------------
# Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("finalized_RFmodel.sav", "rb"))
    scaler = pickle.load(open("scaler_model.sav", "rb"))
    return model, scaler

RF_model, scaler = load_model()

# -----------------------------
# App Title
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center;color:white;'>🍷 Wine Quality Prediction App</h1>
    <p style='text-align:center;color:#f0f0f0;font-size:18px;'>
    Predict the <b>quality of red wine</b> using a <b>Random Forest ML model</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# Sidebar – User Input
# -----------------------------
st.sidebar.header("🔧 Input Wine Features")

fixed_acidity = st.sidebar.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", 0.0, 2.0, 0.70)
citric_acid = st.sidebar.number_input("Citric Acid", 0.0, 1.0, 0.00)
residual_sugar = st.sidebar.number_input("Residual Sugar", 0.0, 20.0, 2.0)
chlorides = st.sidebar.number_input("Chlorides", 0.0, 1.0, 0.08)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", 0.0, 300.0, 46.0)
density = st.sidebar.number_input("Density", 0.9900, 1.0100, 0.9968, format="%.4f")
pH = st.sidebar.number_input("pH", 2.0, 4.5, 3.31)
sulphates = st.sidebar.number_input("Sulphates", 0.0, 2.0, 0.66)
alcohol = st.sidebar.number_input("Alcohol", 5.0, 15.0, 10.5)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free sulfur dioxide': [free_sulfur_dioxide],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

# -----------------------------
# Log Transform (Same as Training)
# -----------------------------
input_data["residual sugar"] = np.log1p(input_data["residual sugar"])
input_data["chlorides"] = np.log1p(input_data["chlorides"])
input_data["free sulfur dioxide"] = np.log1p(input_data["free sulfur dioxide"])
input_data["total sulfur dioxide"] = np.log1p(input_data["total sulfur dioxide"])
input_data["sulphates"] = np.log1p(input_data["sulphates"])

# -----------------------------
# Prediction Button
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Wine Quality"):
    input_scaled = scaler.transform(input_data)
    prediction = RF_model.predict(input_scaled)

    st.markdown(
        f"""
        <h2 style='text-align:center;color:white;'>
        🍷 Predicted Wine Quality: {int(prediction[0])}
        </h2>
        """,
        unsafe_allow_html=True
    )

    if prediction[0] >= 7:
        st.balloons()
        st.success("✅ High quality wine!")
    elif prediction[0] >= 5:
        st.warning("⚠️ Average quality wine")
    else:
        st.error("❌ Low quality wine")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#dddddd;'>Made with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
