import streamlit as st
import requests
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
.title-text {
    font-size: 40px; font-weight: 800;
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center;
}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="title-text">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#aaa;">Powered by XGBoost + FastAPI + PostgreSQL</p>', unsafe_allow_html=True)
st.markdown("---")

model = pickle.load(open('model/churn_model.pkl', 'rb'))
encoders = pickle.load(open('model/encoders.pkl', 'rb'))
feature_cols = pickle.load(open('model/feature_cols.pkl', 'rb'))

tab1, tab2 = st.tabs(["🔮 Predict Churn", "📈 SHAP Explainability"])

with tab1:
    st.markdown("### Enter Customer Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    with c2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    with c3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

    if st.button("🔮 Predict Churn", type="primary"):
        payload = {
            "gender": gender, "senior_citizen": senior,
            "partner": partner, "dependents": dependents,
            "tenure": tenure, "phone_service": phone_service,
            "multiple_lines": multiple_lines, "internet_service": internet_service,
            "online_security": online_security, "online_backup": online_backup,
            "device_protection": device_protection, "tech_support": tech_support,
            "streaming_tv": streaming_tv, "streaming_movies": streaming_movies,
            "contract": contract, "paperless_billing": paperless,
            "payment_method": payment, "monthly_charges": monthly,
            "total_charges": total
        }
        res = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = res.json()
        prob = result['churn_probability']
        churn = result['predicted_churn']

        c1, c2 = st.columns(2)
        with c1:
            color = "#f87171" if churn == "Yes" else "#4ade80"
            st.markdown(f"""
            <div style='background:{color}22;border:2px solid {color};border-radius:12px;padding:20px;text-align:center'>
                <h2 style='color:{color}'>{'⚠️ Will Churn' if churn=='Yes' else '✅ Will Stay'}</h2>
                <h1 style='color:{color}'>{prob*100:.1f}%</h1>
                <p style='color:#aaa'>Churn Probability</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            colors = ['#4ade80', '#f87171']
            ax.barh(['Stay', 'Churn'], [1-prob, prob], color=colors)
            ax.set_xlim(0, 1)
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

with tab2:
    st.markdown("### SHAP Feature Importance")
    st.info("Shows which features most influence churn predictions")
    if st.button("📊 Generate SHAP Plot"):
        from sqlalchemy import create_engine
        engine = create_engine('postgresql://postgres:postgres123@localhost:5432/churn_db')
        df = pd.read_sql('SELECT * FROM customers LIMIT 200', engine)
        cat_cols = ['gender','partner','dependents','phone_service','multiple_lines',
                    'internet_service','online_security','online_backup','device_protection',
                    'tech_support','streaming_tv','streaming_movies','contract',
                    'paperless_billing','payment_method','churn']
        for col in cat_cols:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))
        X = df[feature_cols]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig)
        plt.close()
        st.success("✅ Features ranked by importance!")