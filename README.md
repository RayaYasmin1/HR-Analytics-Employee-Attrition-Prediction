# HR-Analytics-Employee-Attrition-Prediction
# HR Analytics: Predicting Employee Attrition

A data analyst internship project (DataX Labs) that analyzes employee data to
uncover the drivers of voluntary attrition and builds a classification model
to flag employees at risk of leaving.

## Objective

Use HR analytics to understand the main causes of employee resignation and
predict future attrition, so retention efforts can be targeted rather than
reactive.

## Dataset

`hr_attrition_dataset.csv` — 1,470 employees x 26 attributes, generated to
mirror the well-known IBM HR Analytics Attrition dataset schema and
real-world attrition patterns (overtime, pay, satisfaction, tenure,
promotions, etc.). Attrition rate: ~37%.

Key columns: `Age`, `Department`, `JobRole`, `MonthlyIncome`, `SalaryBand`,
`OverTime`, `BusinessTravel`, `YearsAtCompany`, `YearsSinceLastPromotion`,
`JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`, `Attrition`
(target).

## Project Structure

```
├── 01_generate_data.py          # Builds the HR dataset
├── 02_eda.py                    # Exploratory data analysis + charts
├── 03_model.py                  # Trains/evaluates classification models
├── 04_shap_analysis.py          # SHAP explainability on Random Forest
├── 05_build_report.py           # Generates the PDF report
│
├── hr_attrition_dataset.csv     # Source dataset
├── attrition_predictions.csv    # Per-employee predictions + risk scores
├── model_comparison_metrics.csv # Accuracy/precision/recall/F1/ROC-AUC per model
├── best_attrition_model.pkl     # Trained best model (Logistic Regression)
│
├── eda_attrition_drivers.png    # Attrition by dept/salary/overtime/promotion
├── eda_correlation_heatmap.png  # Correlation matrix of numeric features
├── confusion_matrix.png         # Confusion matrix, best model
├── roc_curves.png               # ROC curves, all models compared
├── feature_importance.png       # Top feature importances
├── shap_summary.png             # SHAP summary plot
│
├── HR_Attrition_Project_Report.pdf  # 2-page project report
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap reportlab joblib

python 01_generate_data.py     # creates hr_attrition_dataset.csv
python 02_eda.py               # creates EDA charts
python 03_model.py             # trains models, saves best model + predictions
python 04_shap_analysis.py     # creates shap_summary.png
python 05_build_report.py      # creates HR_Attrition_Project_Report.pdf
```

Each script reads/writes to the working directory, so run them in order from
the same folder.

## Methodology

1. **Data preparation** — checked for missing values, encoded categorical
   features (department, job role, marital status, overtime, business travel).
2. **EDA** — attrition rate by department, salary band, overtime status, and
   years since last promotion; correlation matrix of numeric features.
3. **Modelling** — Logistic Regression, Decision Tree, and Random Forest
   classifiers, trained on a 75/25 stratified split with class balancing.
4. **Evaluation** — accuracy, precision, recall, F1, and ROC-AUC; confusion
   matrix and ROC curves for the best model.
5. **Explainability** — SHAP values on a Random Forest to validate and rank
   the features actually driving predictions.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (best)** | 73.6% | 61.3% | 77.9% | 0.686 | **0.805** |
| Random Forest | 74.2% | 67.5% | 58.1% | 0.625 | 0.785 |
| Decision Tree | 64.7% | 51.8% | 64.0% | 0.572 | 0.675 |

**Top attrition drivers** (EDA + SHAP agree): overtime, low monthly income,
low job satisfaction, long gap since last promotion, low work-life balance.

- Employees working overtime leave **~2x more often** (56% vs 30%)
- Attrition drops from **73% → 16%** as salary band rises from Low → Very High

## Tools Used

Python, Pandas, NumPy, Scikit-learn, Seaborn/Matplotlib, SHAP, ReportLab.
(Power BI was substituted with Python visualizations since it wasn't
available in the build environment — the same charts translate directly
into a Power BI dashboard using the same fields.)

## Recommendations for HR

- Audit and cap chronic overtime, especially in Research & Development
- Review compensation for employees in the "Low" salary band
- Flag employees who haven't been promoted in 4+ years for a career conversation
- Prioritize retention check-ins for employees scoring low on job satisfaction
  and work-life balance surveys

## Author's Note

This project was built as part of the DataX Labs Data Analyst Internship
Project Phase. See `HR_Attrition_Project_Report.pdf` for the full write-up
(Introduction, Abstract, Tools Used, Steps, Conclusion).
