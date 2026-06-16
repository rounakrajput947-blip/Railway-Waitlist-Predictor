import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

st.set_page_config(page_title="Railway Waitlist Predictor", page_icon="🚆")
st.title("🚆 Pro Train Ticket Predictor")

# Load the AI Brain and the Translators
@st.cache_resource
def load_assets():
    model = XGBClassifier()
    model.load_model("model.json")
    
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
        
    return model, encoders

model, encoders = load_assets()

# User Interface: Dynamic Dropdowns based on actual data
col1, col2 = st.columns(2)
with col1:
    wl_pos = st.number_input("Waitlist Position", min_value=1, value=15)
    source = st.selectbox("Source Station", encoders['Source Station'].classes_)
    quota = st.selectbox("Quota", encoders['Quota'].classes_)
    
with col2:
    days_left = st.slider("Days to Departure", 1, 120, 30)
    dest = st.selectbox("Destination Station", encoders['Destination Station'].classes_)
    travel_class = st.selectbox("Class of Travel", encoders['Class of Travel'].classes_)

if st.button("Predict Confirmation Probability"):
    # Translate the user's text inputs into math using our saved encoders
    input_data = pd.DataFrame({
        'Waitlist Position': [wl_pos],
        'Days_to_Departure': [days_left],
        'Quota': [encoders['Quota'].transform([quota])[0]],
        'Class of Travel': [encoders['Class of Travel'].transform([travel_class])[0]],
        'Source Station': [encoders['Source Station'].transform([source])[0]],
        'Destination Station': [encoders['Destination Station'].transform([dest])[0]]
    })
    
    # Predict
    probability = model.predict_proba(input_data)[0][1] * 100
    
    st.subheader(f"Chance of Clearing: {probability:.1f}%")
    if probability > 75:
        st.success("High chance of confirmed seat.")
        st.balloons()
    elif probability > 40:
        st.warning("Coin toss. Monitor closely.")
    else:
        st.error("Low probability. Book a backup!")