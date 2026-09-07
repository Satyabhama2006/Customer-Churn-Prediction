import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load Model and Preprocessor
# --------------------------------------------------

model = joblib.load("models/xgb_churn_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")
st.markdown(
    "Predict whether a telecom customer is likely to churn "
    "based on their demographics, services, contract and billing information."
)

st.divider()


# --------------------------------------------------
# Customer Information
# --------------------------------------------------

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox("Partner", ["Yes", "No"])

    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


st.divider()


# --------------------------------------------------
# Contract and Billing
# --------------------------------------------------

st.subheader("💳 Contract & Billing")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col1, col2 = st.columns(2)

with col1:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

with col2:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )


st.divider()


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Apply preprocessing
    input_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0][1]

    st.divider()
    st.subheader("📈 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is NOT likely to CHURN")

    with result_col2:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    st.progress(float(probability))

    # Interpretation
    if prediction == 1:
        st.warning(
            "This customer shows a higher likelihood of leaving the service. "
            "The business may consider targeted retention strategies."
        )
    else:
        st.info(
            "This customer currently shows a lower likelihood of churn "
            "based on the information provided."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project | "
    "Python • Scikit-learn • XGBoost • Streamlit"
)