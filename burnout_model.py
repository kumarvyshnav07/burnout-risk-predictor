import numpy as np
import pandas as pd
import random
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# =========================================
# STEP 1 — Generate Realistic Dataset
# =========================================
def generate_data(n=50000, seed=42):
    np.random.seed(seed)
    data = []

    for _ in range(n):
        burnout = np.random.choice(["Low", "Medium", "High"], p=[0.34, 0.33, 0.33])

        if burnout == "Low":
            study        = np.random.normal(4.5, 1.5)
            sleep        = np.random.normal(7.0, 1.2)
            screen       = np.random.normal(4, 1.5)
            stress       = np.random.randint(1, 6)
            anxiety      = np.random.uniform(1, 5)
            depression   = np.random.uniform(1, 5)
            academic_p   = np.random.randint(1, 5)
            financial_s  = np.random.randint(1, 5)
            social_sup   = np.random.uniform(5, 10)
            physical_act = np.random.normal(1.2, 0.6)
            sleep_qual   = np.random.randint(6, 10)
            attendance   = np.random.uniform(70, 100)
            cgpa         = np.random.uniform(6.5, 10)

        elif burnout == "Medium":
            study        = np.random.normal(6.5, 1.5)
            sleep        = np.random.normal(6.2, 1.2)
            screen       = np.random.normal(5.5, 1.5)
            stress       = np.random.randint(3, 8)
            anxiety      = np.random.uniform(3, 8)
            depression   = np.random.uniform(3, 8)
            academic_p   = np.random.randint(3, 8)
            financial_s  = np.random.randint(3, 7)
            social_sup   = np.random.uniform(3, 7)
            physical_act = np.random.normal(0.8, 0.5)
            sleep_qual   = np.random.randint(4, 8)
            attendance   = np.random.uniform(60, 90)
            cgpa         = np.random.uniform(5.5, 8.5)

        else:  # High
            study        = np.random.normal(8.5, 1.5)
            sleep        = np.random.normal(5.5, 1.2)
            screen       = np.random.normal(7.5, 1.5)
            stress       = np.random.randint(5, 10)
            anxiety      = np.random.uniform(5, 10)
            depression   = np.random.uniform(5, 10)
            academic_p   = np.random.randint(5, 10)
            financial_s  = np.random.randint(4, 10)
            social_sup   = np.random.uniform(1, 6)
            physical_act = np.random.normal(0.5, 0.4)
            sleep_qual   = np.random.randint(2, 7)
            attendance   = np.random.uniform(40, 85)
            cgpa         = np.random.uniform(4.5, 7.5)

        # Add feature noise
        study += np.random.normal(0, 0.5)
        sleep += np.random.normal(0, 0.5)
        screen += np.random.normal(0, 0.5)

        data.append([
            study, sleep, screen, stress, anxiety, depression,
            academic_p, financial_s, social_sup, physical_act,
            sleep_qual, attendance, cgpa, burnout
        ])

    df = pd.DataFrame(data, columns=[
        "study_hours", "sleep_hours", "screen_time", "stress_level",
        "anxiety_level", "depression_level", "academic_pressure",
        "financial_stress", "social_support", "physical_activity",
        "sleep_quality", "attendance", "cgpa", "burnout"
    ])

    # Add 5% label noise
    flip_idx = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
    df.loc[flip_idx, "burnout"] = np.random.choice(["Low", "Medium", "High"], size=len(flip_idx))

    return df


# =========================================
# STEP 2 — Prepare Data
# =========================================
df = generate_data()

X = df.drop("burnout", axis=1)
y = df["burnout"].map({"Low": 0, "Medium": 1, "High": 2})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# =========================================
# STEP 3 — Train Model
# =========================================
model = XGBClassifier(
    max_depth=3,
    n_estimators=80,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# =========================================
# STEP 4 — Evaluation
# =========================================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("\nCross-validation accuracy:", cv_scores.mean())

# =========================================
# STEP 5 — Feature Importance Plot
# =========================================
plt.figure()
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# =========================================
# STEP 6 — Save Model
# =========================================
joblib.dump(model, "burnout_model.pkl")

print("\nModel saved as burnout_model.pkl ✅")