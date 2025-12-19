import streamlit as st
import pandas as pd
import numpy as np
import joblib
from config import MODEL_RF_OPTIMIZED, SCALER, MODEL_COLUMNS

# Page Config
st.set_page_config(page_title="Tunisia House Price Predictor", page_icon="🏠")

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_RF_OPTIMIZED)
        scaler = joblib.load(SCALER)
        model_columns = joblib.load(MODEL_COLUMNS)
        return model, scaler, model_columns
    except FileNotFoundError:
        st.error("Artifacts not found. Please run the training notebook first to generate .pkl files.")
        return None, None, None

model, scaler, model_columns = load_artifacts()

if model is not None:
    # --- UI Layout ---
    st.title("🏠 Tunisia Real Estate Price Predictor")
    st.markdown("Enter the property details below to get an estimated market price.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            superficie = st.number_input("Surface Area (m²)", min_value=20, max_value=1000, value=100, step=10)
            chambres = st.slider("Bedrooms (S+)", 1, 10, 2)
        
        with col2:
            bains = st.slider("Bathrooms", 1, 5, 1)
            
            # Extract city names dynamically from the model columns
            # Columns are named like 'city_Ariana', 'city_Tunis', etc.
            city_cols = [c for c in model_columns if c.startswith('city_')]
            cities = [c.replace('city_', '') for c in city_cols]
            cities.sort()
            
            selected_city = st.selectbox("City", cities)

        submit_btn = st.form_submit_button("Predict Price", type="primary")

    # --- Prediction Logic ---
    if submit_btn:
        # 1. Prepare a dictionary with all model columns set to 0
        input_data = {col: 0 for col in model_columns}
        
        # 2. Scale the numerical inputs
        # The scaler expects a DataFrame with columns: ['superficie', 'chambres', 'bains']
        df_numerical = pd.DataFrame([[superficie, chambres, bains]], 
                                    columns=['superficie', 'chambres', 'bains'])
        
        scaled_values = scaler.transform(df_numerical)
        
        # Update the input dictionary with scaled values
        input_data['superficie'] = scaled_values[0][0]
        input_data['chambres'] = scaled_values[0][1]
        input_data['bains'] = scaled_values[0][2]
        
        # 3. Handle Categorical (One-Hot Encoding)
        # Find the column corresponding to the selected city and set it to 1
        city_col_name = f"city_{selected_city}"
        if city_col_name in input_data:
            input_data[city_col_name] = 1
        
        # 4. Convert to DataFrame (1 row)
        input_df = pd.DataFrame([input_data])
        
        # 5. Predict
        # The model predicts log(price), so we apply expm1 to get the real price
        try:
            log_prediction = model.predict(input_df)[0]
            real_prediction = np.expm1(log_prediction)
            
            st.success(f"### 💰 Estimated Price: {real_prediction:,.0f} TND")
            st.info(f"Prediction based on a {superficie}m² property with {chambres} bedrooms in {selected_city}.")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")