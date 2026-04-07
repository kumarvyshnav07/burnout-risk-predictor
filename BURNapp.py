import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ==============================
# LOAD MODEL
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "burnout_model.pkl"))
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# Feature order (must match training)
columns = [
    "study_hours", "sleep_hours", "screen_time", "stress_level",
    "anxiety_level", "depression_level", "academic_pressure",
    "financial_stress", "social_support", "physical_activity",
    "sleep_quality", "attendance", "cgpa"
]

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Burnout AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ==============================
# HEADER
# ==============================
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧠 Burnout Prediction Dashboard</h1>
    <p style='text-align: center;'>AI-powered student burnout analysis</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.header("⚙️ Input Controls")

study = st.sidebar.slider("📚 Study Hours", 0.0, 16.0, 5.0)
sleep = st.sidebar.slider("😴 Sleep Hours", 3.0, 10.0, 7.0)
screen = st.sidebar.slider("📱 Screen Time", 0.0, 12.0, 4.0)

cgpa = st.sidebar.slider("🎓 CGPA", 0.0, 10.0, 7.0)
attendance = st.sidebar.slider("📊 Attendance %", 30.0, 100.0, 75.0)

stress = st.sidebar.slider("😰 Stress Level", 1, 10, 5)
anxiety = st.sidebar.slider("😟 Anxiety", 1.0, 10.0, 5.0)
depression = st.sidebar.slider("😔 Depression", 1.0, 10.0, 5.0)

# ==============================
# BUILD INPUT DATA
# ==============================
input_data = {
    "study_hours": study,
    "sleep_hours": sleep,
    "screen_time": screen,
    "stress_level": stress,
    "anxiety_level": anxiety,
    "depression_level": depression,
    "academic_pressure": stress,
    "financial_stress": 5,
    "social_support": 5,
    "physical_activity": 1,
    "sleep_quality": 7,
    "attendance": attendance,
    "cgpa": cgpa
}

df = pd.DataFrame([input_data])
df = df.reindex(columns=columns)

# ==============================
# LAYOUT
# ==============================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Input Summary")
    st.dataframe(df, use_container_width=True)

# ==============================
# PREDICTION
# ==============================
if st.button("🚀 Predict Burnout", use_container_width=True):

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    confidence = max(prob) * 100

    st.markdown("---")

    # ==============================
    # RESULT CARD
    # ==============================
    if prediction == 0:
        st.success(f"🟢 LOW Burnout ({confidence:.2f}%)")
    elif prediction == 1:
        st.warning(f"🟡 MEDIUM Burnout ({confidence:.2f}%)")
    else:
        st.error(f"🔴 HIGH Burnout ({confidence:.2f}%)")

    # ==============================
    # CHART
    # ==============================
    with col2:
        st.subheader("📈 Prediction Confidence")

        labels = ["Low", "Medium", "High"]
        values = prob

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Probability")
        ax.set_title("Confidence Levels")

        st.pyplot(fig)

    # ==============================
    # SUGGESTIONS
    # ==============================
    st.markdown("---")
    st.subheader("💡 Suggestions")

    if prediction == 2:
        st.error("⚠️ High burnout detected!")
        st.write("- Take regular breaks")
        st.write("- Improve sleep quality")
        st.write("- Reduce workload")
        st.write("- Talk to someone")

    elif prediction == 1:
        st.warning("⚡ Medium burnout risk")
        st.write("- Maintain balance")
        st.write("- Reduce stress gradually")

    else:
        st.success("✅ You are doing well!")
        st.write("- Keep your routine")
        st.write("- Stay consistent")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("⚡ Built with Streamlit | AI Burnout Prediction System")