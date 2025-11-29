import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.title("🔋 Battery Health Prediction Dashboard")
st.write("Upload dataset, explore data, and predict battery health using pre-trained models.")

uploaded_file = st.file_uploader("Upload your battery dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean column names same as training
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Preview
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    st.subheader("🔎 Exploratory Data Analysis")
    st.write("**Shape of Data:**", df.shape)
    st.write("**Summary Statistics:**")
    st.write(df.describe())
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    st.write("**Correlation Heatmap:**")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Add time_diff feature
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)

    # Load models
    model_dir = "models"  # assumes models in 'models' folder
    soc_rf = joblib.load(os.path.join(model_dir, "soc_rf.pkl"))
    soh_rf = joblib.load(os.path.join(model_dir, "soh_rf.pkl"))
    features = joblib.load(os.path.join(model_dir, "features.pkl"))

    st.success("✅ Models loaded successfully!")

    st.subheader("📝 Input Features for Prediction")
    with st.form("input_form"):
        input_data = {}
        for feature in features:
            if feature in df.columns:
                input_data[feature] = st.number_input(
                    f"Enter {feature}",
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].mean()),
                    step=0.01
                )
            else:
                input_data[feature] = st.number_input(f"Enter {feature}", 0.0, 100.0, 0.0, step=0.01)
        submitted = st.form_submit_button("Predict Battery Health")

    if submitted:
        input_df = pd.DataFrame([input_data])[features]  # enforce correct order
        st.write("Input Data for Prediction:")
        st.write(input_df)

        try:
            soc_pred = soc_rf.predict(input_df)[0]
            soh_pred = soh_rf.predict(input_df)[0]

            st.success(f"🔋 Predicted SoC (Random Forest): {soc_pred:.2f}")
            st.success(f"⚡ Predicted SoH (Random Forest): {soh_pred:.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
