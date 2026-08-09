import streamlit as st
import pandas as pd
import joblib
import os

# Configuration
MODEL_FILE = 'kepler_pipeline.joblib'

# --- UI Setup ---
st.set_page_config(page_title="Kepler Exoplanet Detector", page_icon="🪐")
st.title("🪐 Kepler Exoplanet Candidate Analysis")
st.markdown("Enter the astrophysical observation values to analyze the candidate.")

# --- Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

pipeline_data = load_model()

if not pipeline_data:
    st.error(f"Model file '{MODEL_FILE}' not found. Please ensure it exists.")
    st.stop()

loaded_model = pipeline_data['model']
loaded_scaler = pipeline_data['scaler']
feature_names = pipeline_data['features']

column_mapping = {
    "koi_period": "Orbital Period (days)",
    "koi_time0bk": "Transit Epoch (BJD)",
    "koi_impact": "Impact Parameter",
    "koi_duration": "Transit Duration (hours)",
    "koi_depth": "Transit Depth (parts per million)",
    "koi_prad": "Planetary Radius (Earth radii)",
    "koi_teq": "Equilibrium Temperature (Kelvin)",
    "koi_insol": "Insolation Flux (Earth flux)",
    "koi_model_snr": "Transit Signal-to-Noise",
    "koi_tce_plnt_num": "TCE Planet Number",
    "koi_steff": "Stellar Effective Temperature (Kelvin)",
    "koi_slogg": "Stellar Surface Gravity (log10(cm/s^2))",
    "koi_srad": "Stellar Radius (Solar radii)",
    "ra": "Right Ascension (decimal degrees)",
    "dec": "Declination (decimal degrees)",
    "koi_kepmag": "Kepler Band (magnitude)"
}

# --- Input Form ---
user_data = {}
with st.form("input_form"):
    cols = st.columns(2)
    for i, column in enumerate(feature_names):
        readable_name = column_mapping.get(column, column)
        # Using columns to create a better layout
        with cols[i % 2]:
            user_data[column] = st.number_input(f"{readable_name}", value=0.0, step=0.01)
    
    submitted = st.form_submit_button("Analyze Candidate")

# --- Inference ---
if submitted:
    new_df = pd.DataFrame({k: [v] for k, v in user_data.items()})
    
    # Scale and Predict
    new_df_scaled = loaded_scaler.transform(new_df)
    prediction = loaded_model.predict(new_df_scaled)
    probability = loaded_model.predict_proba(new_df_scaled)[0]

    st.markdown("---")
    st.subheader("Analysis Result")
    
    if prediction[0] == 1:
        st.success("POSITIVE - This is a confirmed exoplanet.")
        st.metric("Confidence Level", f"{probability[1] * 100:.2f}%")
    else:
        st.error("NEGATIVE - This is a false positive (stellar anomaly or noise).")
        st.metric("Probability of False Positive", f"{probability[0] * 100:.2f}%")
