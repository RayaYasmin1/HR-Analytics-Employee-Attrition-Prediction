import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import shap

df = pd.read_csv("hr_attrition_dataset.csv")
y = (df["Attrition"] == "Yes").astype(int)
feature_cols = [c for c in df.columns if c not in
                ["EmployeeID", "Attrition", "SalaryBand"]]
X = df[feature_cols].copy()

for c in X.select_dtypes(include="object").columns:
    X[c] = LabelEncoder().fit_transform(X[c])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42,
                             class_weight="balanced")
rf.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# shap_values can be a list [class0, class1] or a 3D array depending on version
if isinstance(shap_values, list):
    sv_pos = shap_values[1]
else:
    sv_pos = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

plt.figure()
shap.summary_plot(sv_pos, X_test, show=False, max_display=12)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight", dpi=150)
print("Saved shap_summary.png")
