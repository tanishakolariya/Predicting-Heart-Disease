import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("Heart Disease Prediction System")
st.write("Predict the risk of heart disease using Machine Learning")

st.divider()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("heart_model.joblib")
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()

if error:
    st.error(f"Model loading failed: {error}")
else:
    st.success("Model loaded successfully")

# ---------------- INPUT UI ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")

    age = st.number_input("Age", 1, 120, 25)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])

with col2:
    st.subheader("Medical Results")

    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", value=1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

# ---------------- PREPROCESS ----------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Heart Disease Risk"):

    if model is None:
        st.warning("⚠️ Model not loaded. Please check file.")
    else:
        try:
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                    restecg, thalach, exang, oldpeak,
                                    slope, ca, thal]])

            prediction = model.predict(input_data)

            # Safe probability
            try:
                probability = model.predict_proba(input_data)[0][1]
            except:
                probability = 0.5

            # Result
            if prediction[0] == 1:
                st.error("High Risk of Heart Disease")
            else:
                st.success("Low Risk of Heart Disease")

            st.write(f"### Risk Probability: {probability:.2f}")

            # ---------------- GRAPH ----------------
            fig, ax = plt.subplots()
            labels = ["Low Risk", "High Risk"]
            values = [1 - probability, probability]

            ax.bar(labels, values)
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Result")

            st.pyplot(fig)

        except Exception as e:
            st.error(f" Prediction failed: {e}")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Machine Learning Project | Built with Streamlit 🚀")