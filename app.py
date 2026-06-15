import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# Set up the web page
st.set_page_config(page_title="Railway Waitlist Predictor", page_icon="🚆")
st.title("🚆 Train Ticket Waitlist Predictor")
st.write("Built by Rounak Rajput (IIT Patna)")
st.write("Enter your ticket details below to see your chances of clearing the waitlist!")

# Load the AI Brain
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("model.json")
    return model

model = load_model()

# User Interface: Sliders and Inputs
col1, col2 = st.columns(2)
with col1:
    wl_pos = st.number_input("Waitlist Position (e.g., 15)", min_value=1, max_value=500, value=15)
with col2:
    days_left = st.slider("Days to Departure", min_value=1, max_value=120, value=30)

# The Prediction Engine
if st.button("Predict Confirmation Probability"):
    # Format the data exactly how the model expects it
    input_data = pd.DataFrame({'Waitlist Position': [wl_pos], 'Days_to_Departure': [days_left]})
    
    # Get the probability (returns a percentage)
    probability = model.predict_proba(input_data)[0][1] * 100
    
    st.subheader(f"Chance of Clearing: {probability:.1f}%")
    
    # Show dynamic feedback
    if probability > 75:
        st.success("Looking great! You have a high chance of getting a confirmed seat.")
        st.balloons()
    elif probability > 40:
        st.warning("It's a coin toss. You might want to monitor this closely.")
    else:
        st.error("Low probability. Definitely look into booking a backup ticket!")