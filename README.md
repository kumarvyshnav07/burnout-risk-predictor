# 🧠 Burnout Prediction Dashboard

## 📌 Overview

This project is an **AI-powered burnout prediction system** that analyzes student lifestyle and mental health factors to predict burnout levels:

* 🟢 Low
* 🟡 Medium
* 🔴 High

The system uses a Machine Learning model to provide **real-time predictions with confidence scores and suggestions**.

---

## 🚀 Features

* 📊 Predict burnout level instantly
* 🧠 Powered by XGBoost Machine Learning model
* 📈 Displays prediction confidence graph
* 💡 Provides personalized suggestions
* 🎯 Achieves ~96% accuracy with cross-validation
* 🖥️ Interactive dashboard built with Streamlit

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Streamlit

---

## 📊 Model Performance

* Accuracy: **~96%**
* Cross-validation score: **~96%**
* Balanced precision & recall across all classes

---

## 📁 Project Structure

```
AI/
│── BURNapp.py             # Streamlit dashboard app
│── burnout_model.py       # Model training script
│── burnout_model.pkl      # Saved trained model
│── README.md              # Project documentation
```

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install numpy pandas scikit-learn xgboost matplotlib streamlit joblib
```

### 2. Run the app

```
streamlit run BURNapp.py
```

### 3. Open in browser

```
http://localhost:8501/
```

---

## 💡 Key Highlights

* Uses realistic synthetic dataset with noise to avoid overfitting
* Avoids unrealistic 100% accuracy
* Designed with proper ML practices
* Clean and interactive UI

---

## 🔮 Future Improvements

* 🌐 Deploy online (Streamlit Cloud / Render)
* 🧠 Add SHAP explainability
* 🎨 Improve UI with advanced styling
* 📱 Make mobile-friendly version

---

## 👨‍💻 Author

**Kumar Vyshnav**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub 
