
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import sklearn # To verify scikit-learn version

# --- Configuration for Wide Layout ---
st.set_page_config(layout="wide", page_title="African Sickle Cell Mortality Risk Prediction")

# --- Assumed Universal Scikit-learn Version ---
EXPECTED_SKLEARN_VERSION = "1.6.1"

# --- Model Artifact Paths per Region ---
REGION_ARTIFACTS = {
    "Central Africa": {
        "model": "central_africa_model.joblib",
        "features": "central_africa_feature_cols.joblib",
    },
    "East Africa": {
        "model": "east_africa_model.joblib",
        "features": "east_africa_feature_cols.joblib",
    },
    "North Africa": {
        "model": "north_africa_model.joblib",
        "features": "north_africa_feature_cols.joblib",
    },
    "Southern Africa": {
        "model": "southern_africa_model.joblib",
        "features": "southern_africa_feature_cols.joblib",
    },
    "West Africa": {
        "model": "west_africa_model.joblib",
        "features": "west_africa_feature_cols.joblib",
    },
}

# --- Feature Specifications (Min, Max, Mean for Input Widgets) per Region ---
REGION_FEATURE_SPECS = {
    "Central Africa": {
        'tmed_temperature': {'min': 18.43, 'max': 33.62, 'mean': 25.42, 'label': "Median Temperature (°C)"},
        'max_aod': {'min': 261.00, 'max': 4988.00, 'mean': 1570.00, 'label': "Max Aerosol Optical Depth"},
        'precip_range': {'min': 0.00, 'max': 657.94, 'mean': 180.61, 'label': "Precipitation Range (mm)"},
        'max_aod_lag1': {'min': 261.00, 'max': 4988.00, 'mean': 1564.74, 'label': "Max AOD (previous month)"},
        'max_aod_lag3': {'min': 261.00, 'max': 4988.00, 'mean': 1548.77, 'label': "Max AOD (3 months ago)"},
        'max_aod_roll6': {'min': 283.00, 'max': 3832.33, 'mean': 1539.20, 'label': "Max AOD (6-month rolling avg)"},
    },
    "East Africa": {
        'max_aod': {'min': 103.0, 'max': 5000.00, 'mean': 871.44, 'label': "Max Aerosol Optical Depth"},
        'precip_range': {'min': 0.0, 'max': 929.16, 'mean': 103.56, 'label': "Precipitation Range (mm)"},
        'max_aod_lag1': {'min': 103.0, 'max': 5000.00, 'mean': 866.80, 'label': "Max AOD (previous month)"},
        'max_aod_lag3': {'min': 117.0, 'max': 5000.00, 'mean': 873.96, 'label': "Max AOD (3 months ago)"},
        'min_precipitation_lag1': {'min': 0.0, 'max': 517.37, 'mean': 36.56, 'label': "Min Precipitation (previous month)"},
        'max_aod_roll6': {'min': 172.5, 'max': 3796.50, 'mean': 873.07, 'label': "Max AOD (6-month rolling avg)"},
    },
    "North Africa": {
        'med_precipitation': {'min': 0.00, 'max': 100.66, 'mean': 5.24, 'label': "Median Precipitation (mm)"},
        'min_precipitation': {'min': 0.00, 'max': 48.37, 'mean': 0.66, 'label': "Min Precipitation (mm)"},
        'max_aod': {'min': 178.00, 'max': 5000.00, 'mean': 1769.02, 'label': "Max Aerosol Optical Depth"},
        'precip_range': {'min': 0.00, 'max': 557.87, 'mean': 57.51, 'label': "Precipitation Range (mm)"},
        'aridity_index': {'min': 0.00, 'max': 4.00, 'mean': 0.12, 'label': "Aridity Index"},
        'precip_range_lag1': {'min': 0.00, 'max': 557.87, 'mean': 56.63, 'label': "Precipitation Range (prev month)"},
        'precip_range_lag3': {'min': 0.00, 'max': 344.10, 'mean': 55.21, 'label': "Precipitation Range (3 months ago)"},
        'max_aod_lag1': {'min': 178.00, 'max': 5000.00, 'mean': 1812.27, 'label': "Max AOD (previous month)"},
        'max_aod_lag3': {'min': 178.00, 'max': 5000.00, 'mean': 1805.73, 'label': "Max AOD (3 months ago)"},
        'aridity_index_lag1': {'min': 0.00, 'max': 4.00, 'mean': 0.12, 'label': "Aridity Index (previous month)"},
        'med_precipitation_lag1': {'min': 0.00, 'max': 100.66, 'mean': 5.19, 'label': "Median Precip. (previous month)"},
        'med_precipitation_lag3': {'min': 0.00, 'max': 100.66, 'mean': 5.10, 'label': "Median Precip. (3 months ago)"},
        'min_precipitation_lag1': {'min': 0.00, 'max': 48.37, 'mean': 0.65, 'label': "Min Precip. (previous month)"},
        'min_precipitation_lag3': {'min': 0.00, 'max': 48.37, 'mean': 0.61, 'label': "Min Precip. (3 months ago)"},
        'precip_range_roll6': {'min': 0.00, 'max': 346.38, 'mean': 56.44, 'label': "Precip. Range (6-month roll avg)"},
        'temp_range_roll6': {'min': 11.57, 'max': 28.55, 'mean': 21.30, 'label': "Temp. Range (6-month roll avg)"},
        'max_aod_roll3': {'min': 240.00, 'max': 5000.00, 'mean': 1794.55, 'label': "Max AOD (3-month roll avg)"},
        'max_aod_roll6': {'min': 240.00, 'max': 4196.33, 'mean': 1772.71, 'label': "Max AOD (6-month roll avg)"},
        'med_precipitation_roll6': {'min': 0.00, 'max': 305.30, 'mean': 6.23, 'label': "Median Precip. (6-month roll avg)"},
    },
    "Southern Africa": {
        'min_precipitation': {'min': 0.00, 'max': 388.69, 'mean': 28.57, 'label': "Min Precipitation (mm)"},
        'max_aod': {'min': 45.00, 'max': 5000.00, 'mean': 848.04, 'label': "Max Aerosol Optical Depth"},
        'precip_range': {'min': 0.00, 'max': 445.52, 'mean': 63.02, 'label': "Precipitation Range (mm)"},
        'precip_range_lag1': {'min': 0.00, 'max': 482.28, 'mean': 63.26, 'label': "Precipitation Range (prev month)"},
        'max_aod_lag1': {'min': 45.00, 'max': 5000.00, 'mean': 848.71, 'label': "Max AOD (previous month)"},
        'max_aod_lag3': {'min': 45.00, 'max': 5000.00, 'mean': 848.07, 'label': "Max AOD (3 months ago)"},
        'med_precipitation_lag1': {'min': 0.00, 'max': 388.69, 'mean': 46.16, 'label': "Median Precip. (previous month)"},
        'max_aod_roll6': {'min': 176.17, 'max': 3221.50, 'mean': 854.35, 'label': "Max AOD (6-month rolling avg)"},
    },
    "West Africa": {
        'tmin_temperature': {'min': 5.02, 'max': 26.46, 'mean': 20.04, 'label': "Min Temperature (°C)"},
        'tmax_temperature': {'min': 28.38, 'max': 46.47, 'mean': 35.65, 'label': "Max Temperature (°C)"},
        'min_precipitation': {'min': 0.00, 'max': 528.41, 'mean': 51.63, 'label': "Min Precipitation (mm)"},
        'max_aod': {'min': 266.00, 'max': 5000.00, 'mean': 1351.40, 'label': "Max Aerosol Optical Depth"},
        'precip_range': {'min': 0.00, 'max': 909.98, 'mean': 125.48, 'label': "Precipitation Range (mm)"},
        'max_aod_lag1': {'min': 266.00, 'max': 5000.00, 'mean': 1372.11, 'label': "Max AOD (previous month)"},
        'max_aod_lag3': {'min': 266.00, 'max': 5000.00, 'mean': 1369.19, 'label': "Max AOD (3 months ago)"},
        'min_precipitation_lag1': {'min': 0.00, 'max': 528.41, 'mean': 51.24, 'label': "Min Precipitation (previous month)"},
        'max_aod_roll6': {'min': 173.00, 'max': 4705.00, 'mean': 1367.38, 'label': "Max AOD (6-month rolling avg)"},
        'min_precipitation_roll6': {'min': 0.00, 'max': 369.95, 'mean': 51.02, 'label': "Min Precipitation (6-month rolling avg)"},
    },
}

# --- Consolidate all unique features and their overall min/max/mean for universal widget rendering ---
MASTER_FEATURE_SPECS = {}
for region_specs in REGION_FEATURE_SPECS.values():
    for feature_name, details in region_specs.items():
        if feature_name not in MASTER_FEATURE_SPECS:
            MASTER_FEATURE_SPECS[feature_name] = {
                'min': details['min'],
                'max': details['max'],
                'mean': details['mean'],
                'label': details['label']
            }
        else: # If a feature appears in multiple regions, consolidate ranges and average mean
            current_master = MASTER_FEATURE_SPECS[feature_name]
            MASTER_FEATURE_SPECS[feature_name]['min'] = min(current_master['min'], details['min'])
            MASTER_FEATURE_SPECS[feature_name]['max'] = max(current_master['max'], details['max'])
            MASTER_FEATURE_SPECS[feature_name]['mean'] = (current_master['mean'] + details['mean']) / 2
# Add cyclical month features manually as they are always present but derived
MASTER_FEATURE_SPECS['month_sin'] = {'min': -1.0, 'max': 1.0, 'mean': 0.0, 'label': 'Month Sine'}
MASTER_FEATURE_SPECS['month_cos'] = {'min': -1.0, 'max': 1.0, 'mean': 0.0, 'label': 'Month Cosine'}


# --- Helper function for cyclical month features ---
def compute_month_cyclical_features(month_number):
    month_sin = np.sin(2 * np.pi * month_number / 12)
    month_cos = np.cos(2 * np.pi * month_number / 12)
    return month_sin, month_cos

# --- Load ALL Model Artifacts (Cached for efficiency) ---
@st.cache_resource
def load_all_models_artifacts():
    """
    Loads all pre-trained XGBoost models and their feature columns for all regions.
    Performs a universal scikit-learn version check.
    """
    status_placeholder = st.empty() # Placeholder for loading messages
    all_loaded_models = {}

    # --- Global scikit-learn version check ---
    current_sklearn_version = sklearn.__version__
    if current_sklearn_version != EXPECTED_SKLEARN_VERSION:
        st.error(f"**CRITICAL: Sci-kit learn version mismatch!** "
                   f"Models were saved with v{EXPECTED_SKLEARN_VERSION}, "
                   f"but current environment has v{current_sklearn_version}. "
                   f"This *will* cause `joblib.load` to fail or hang. "
                   f"**Action required:** Install `pip install scikit-learn=={EXPECTED_SKLEARN_VERSION}` "
                   f"and restart the Streamlit app.")
        st.stop() # Halt execution immediately if versions don't match.

    # --- Load models for each region ---
    for region_name, paths in REGION_ARTIFACTS.items():
        status_placeholder.info(f"Loading model artifacts for {region_name}...")
        try:
            model = joblib.load(paths["model"])
            feature_cols = joblib.load(paths["features"])
            explainer = shap.TreeExplainer(model) 
            
            all_loaded_models[region_name] = {
                "model": model,
                "explainer": explainer,
                "feature_cols": feature_cols
            }
        except FileNotFoundError:
            st.error(f"Error: Artifacts for {region_name} not found. Please ensure '{paths['model']}' and '{paths['features']}' are in the app's directory.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load or initialize SHAP for {region_name}. Error: {e}. "
                     f"Ensure files are correct and compatible with `xgboost` and `shap` versions.")
            st.stop()
    
    status_placeholder.success("All regional models loaded successfully!")
    return all_loaded_models

# --- Main Streamlit App ---
st.title("🌡️African Sickle Cell Mortality Risk Prediction")
st.markdown("This application predicts monthly sickle cell mortality risk in various African regions based on climate factors and generates a 'Climate Impact Score' for proactive planning.")

# Load all models once
with st.spinner("Initializing application (loading all regional models and data)..."):
    ALL_MODELS_DATA = load_all_models_artifacts()

st.sidebar.header("Select Region & Input Features")

# Region Selection
selected_region = st.sidebar.selectbox(
    "Select African Region",
    options=list(ALL_MODELS_DATA.keys()),
    key="region_selector" # Unique key for this widget
)

# Get the currently active model, explainer, and feature list
current_model_data = ALL_MODELS_DATA[selected_region]
model = current_model_data["model"]
explainer = current_model_data["explainer"]
active_feature_cols = current_model_data["feature_cols"]


# Month Selection
month_numbers = list(range(1, 13))
month_names = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}
selected_month_number = st.sidebar.selectbox(
    "Month", 
    options=month_numbers, 
    format_func=lambda x: month_names[x],
    key="month_selector" # Unique key for this widget
)

# Derived cyclical features
month_sin, month_cos = compute_month_cyclical_features(selected_month_number)

user_inputs = {}
st.sidebar.markdown("---")
st.sidebar.markdown("**Adjust Climate Features:**")

# Generate ALL possible input widgets, but disable/enable based on region
all_feature_names_sorted = sorted([f for f in MASTER_FEATURE_SPECS.keys() if f not in ['month_sin', 'month_cos']])

for feature_name in all_feature_names_sorted:
    specs = MASTER_FEATURE_SPECS.get(feature_name, {}) # Get specs from master list
    label = specs.get('label', feature_name.replace('_', ' ').title())
    
    min_val = float(specs.get('min', -10000.0)) 
    max_val = float(specs.get('max', 10000.0))
    mean_val = float(specs.get('mean', 0.0))

    # Ensure mean_val is within min/max, especially for new default ranges
    if mean_val < min_val: mean_val = min_val
    if mean_val > max_val: mean_val = max_val

    # Determine step and format based on typical values or feature name
    if 'temperature' in feature_name:
        step = 0.1
        format_str = "%.1f"
    elif 'aod' in feature_name:
        step = 10.0 # A larger step for a large range makes the slider more usable
        format_str = "%.0f"
    elif 'precipitation' in feature_name or 'range' in feature_name or 'aridity_index' in feature_name:
        step = 1.0
        format_str = "%.1f"
    else: # Fallback
        step = 0.1
        format_str = "%.2f"
    
    # Determine if the widget should be disabled
    is_disabled = (feature_name not in active_feature_cols)

    user_inputs[feature_name] = st.sidebar.slider(
        label,
        min_value=min_val,
        max_value=max_val,
        value=mean_val, # Use the mean as the default starting position
        step=step,
        format=format_str,
        disabled=is_disabled, # THIS IS THE KEY CHANGE for disabling
        key=f"slider_{feature_name}" # Unique key for each slider
    )

# Add the cyclical features (always enabled and derived) to the input dictionary
user_inputs['month_sin'] = month_sin
user_inputs['month_cos'] = month_cos

# Create a DataFrame for prediction, containing ONLY the active features in the correct order
# This is crucial so the model receives exactly the feature set it was trained with.
input_data_for_model = {feature: user_inputs[feature] for feature in active_feature_cols}
input_df = pd.DataFrame([input_data_for_model], columns=active_feature_cols)


st.sidebar.markdown("---")
if st.sidebar.button("Predict Mortality & Climate Impact", key="predict_button"): # Unique key for button
    st.subheader(f"Prediction Results for {selected_region}:")
    try:
        # Predict primary target (monthly mortality)
        predicted_mortality = model.predict(input_df)[0]
        predicted_mortality = np.round(np.clip(predicted_mortality, 0, None), 0)

        st.markdown(f"### Predicted Monthly Sickle Cell Mortality: **{int(predicted_mortality)}**")
        st.markdown("---")

        # Calculate SHAP values for the current prediction to get the climate score
        shap_values_obj = explainer(input_df) 

        # The climate_score in your notebook was the sum of raw SHAP values, then clipped
        raw_climate_impact_sum = sum(shap_values_obj.values[0])
        climate_score = np.round(np.clip(raw_climate_impact_sum, 0, None), 0)

        st.markdown(f"### Derived Climate Impact Score: **{int(climate_score)}**")

        st.markdown("---")

        st.subheader("Interpretation:")
        st.markdown(f"""
        *   **Predicted Mortality ({int(predicted_mortality)}):** This is the estimated number of sickle cell deaths for the specified month and climate conditions in **{selected_region}**, as predicted by the model.
        *   **Climate Impact Score ({int(climate_score)}):** This score quantifies the overall *adverse* influence of the inputted climate conditions on the predicted mortality.
            *   A **higher score** indicates that the climate factors are predicted to contribute more significantly to increased mortality risk for this specific prediction.
            *   A **score of 0** suggests that, according to the model, the combined climate conditions are either neutral or predicted to have a beneficial impact (pushing mortality *below* the average baseline) during this period.
        """)

        st.subheader("Feature Contribution (SHAP Plot):")

        # For a single prediction waterfall plot - this is instance specific
        st.write("#### Individual Prediction Waterfall Plot")
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
        # shap_values_obj[0] is the Explanation object for the first (and only) instance in input_df
        shap.plots.waterfall(shap_values_obj[0], max_display=10, show=False)
        st.pyplot(fig_waterfall)
        st.caption(f"Waterfall plot showing how each feature pushes the prediction from the base value to the final predicted mortality for the *specific inputs entered for {selected_region}*.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure all input values are reasonable for the selected region and the model artifacts are loaded correctly.")

st.markdown("---")
st.markdown("### About the Models")
st.markdown(f"""
These models are specifically trained for each selected **African region**. They utilize various climate indicators, including temperature, precipitation, and aerosol optical depth, along with their lagged and rolling average values, to predict monthly sickle cell mortality. The 'Climate Impact Score' is derived using SHAP values to explain the models' predictions, providing insight into the overall influence of climate factors.
""")
