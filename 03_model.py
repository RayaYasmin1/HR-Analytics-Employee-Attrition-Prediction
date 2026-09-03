import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, RocCurveDisplay)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("hr_attrition_dataset.csv")
y = (df["Attrition"] == "Yes").astype(int)

feature_cols = [c for c in df.columns if c not in
                ["EmployeeID", "Attrition", "SalaryBand"]]
X = df[feature_cols].copy()

cat_cols = X.select_dtypes(include="object").columns.tolist()
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c])
    encoders[c] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42,
                                             class_weight="balanced"),
}

results = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC_AUC": roc_auc_score(y_test, probs),
    })
    fitted[name] = model

results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison_metrics.csv", index=False)

best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
print(f"\nBest model: {best_name}")

best_preds = best_model.predict(X_test)
best_probs = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report ({}):\n".format(best_name),
      classification_report(y_test, best_preds, target_names=["No Attrition", "Attrition"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Attrition", "Attrition"],
            yticklabels=["No Attrition", "Attrition"])
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", bbox_inches="tight")
print("Saved confusion_matrix.png")

# ROC curve for all models
plt.figure(figsize=(6, 5))
for name, model in fitted.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=plt.gca())
plt.title("ROC Curves - Model Comparison")
plt.tight_layout()
plt.savefig("roc_curves.png", bbox_inches="tight")
print("Saved roc_curves.png")

# Feature importance (best tree-based model) or coefficients
plt.figure(figsize=(8, 6))
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
else:
    importances = pd.Series(np.abs(best_model.coef_[0]), index=X.columns)
top_imp = importances.sort_values(ascending=False).head(12)
sns.barplot(x=top_imp.values, y=top_imp.index, color="teal")
plt.title(f"Top Feature Importances - {best_name}")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", bbox_inches="tight")
print("Saved feature_importance.png")

# Save predictions CSV
output = X_test.copy()
output["EmployeeID"] = df.loc[X_test.index, "EmployeeID"].values
output["Actual_Attrition"] = y_test.values
output["Predicted_Attrition"] = best_preds
output["Attrition_Probability"] = np.round(best_probs, 3)
output = output[["EmployeeID", "Actual_Attrition", "Predicted_Attrition", "Attrition_Probability"]]
output.to_csv("attrition_predictions.csv", index=False)
print("Saved attrition_predictions.csv")

import joblib
joblib.dump(best_model, "best_attrition_model.pkl")
print(f"Saved best_attrition_model.pkl ({best_name})")

# Save which model + key numbers for the report
with open("model_summary.txt", "w") as f:
    f.write(f"Best model: {best_name}\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
