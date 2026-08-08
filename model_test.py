import pandas as pd
import joblib
import sys
import os


MODEL_FILE = 'kepler_pipeline.joblib'
if not os.path.exists(MODEL_FILE):
    print(f"[ERROR] Model file '{MODEL_FILE}' not found.")
    print("Please run the training notebook first to generate the model.")
    sys.exit(1)


print("Loading the Kepler AI model...")
pipeline_data = joblib.load(MODEL_FILE)

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

print("-" * 60)
print(" NEW KEPLER EXOPLANET CANDIDATE ANALYSIS SYSTEM")
print("-" * 60)
print("Please enter the astrophysical observation values:\n")

user_data = {}


for column in feature_names:
    readable_name = column_mapping.get(column, column)
    
    while True:
        try:
            value = float(input(f"Enter value for '{readable_name}': "))
            user_data[column] = [value]
            break
        except ValueError:
            print("   [ERROR] Invalid input. Please enter a numerical value.")

print("\nProcessing data and calculating probabilities...")


new_df = pd.DataFrame(user_data)
new_df_scaled = loaded_scaler.transform(new_df)

prediction = loaded_model.predict(new_df_scaled)
probability = loaded_model.predict_proba(new_df_scaled)[0]


print("\n" + "=" * 60)
print(" ANALYSIS RESULT")
print("=" * 60)

if prediction[0] == 1:
    print("RESULT: POSITIVE - This is a confirmed exoplanet.")
    print(f"Confidence Level: {probability[1] * 100:.2f}%")
else:
    print("RESULT: NEGATIVE - This is a false positive (stellar anomaly or noise).")
    print(f"Probability of being a False Positive: {probability[0] * 100:.2f}%")
    
print("=" * 60)