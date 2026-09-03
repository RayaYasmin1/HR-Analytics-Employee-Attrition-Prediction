"""
Generate a realistic synthetic HR dataset for the Employee Attrition project.

The schema and feature relationships are modeled on the well-known IBM HR
Analytics Employee Attrition dataset (departments, job roles, salary bands,
overtime, promotions, satisfaction scores, etc.) and on documented real-world
attrition drivers (low pay, overtime, long commutes, low satisfaction, few
promotions). Attrition is generated probabilistically from these features so
the dataset has genuine, learnable signal for a classification model.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1470  # matches scale of the classic IBM HR dataset

departments = ["Sales", "Research & Development", "Human Resources"]
dept_probs = [0.31, 0.65, 0.04]

job_roles = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": [
        "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Research Director", "Healthcare Representative"
    ],
    "Human Resources": ["Human Resources", "Manager"],
}

education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree",
                     "Other", "Human Resources"]

df = pd.DataFrame()
df["EmployeeID"] = np.arange(1001, 1001 + N)
df["Age"] = rng.integers(18, 61, N)
df["Department"] = rng.choice(departments, N, p=dept_probs)
df["JobRole"] = df["Department"].apply(lambda d: rng.choice(job_roles[d]))
df["EducationField"] = rng.choice(education_fields, N)
df["Gender"] = rng.choice(["Male", "Female"], N, p=[0.6, 0.4])
df["MaritalStatus"] = rng.choice(["Single", "Married", "Divorced"], N, p=[0.32, 0.46, 0.22])

df["DistanceFromHome"] = rng.integers(1, 30, N)
df["MonthlyIncome"] = np.round(rng.gamma(shape=4.0, scale=1500, size=N) + 1500, -1).astype(int)
df["MonthlyIncome"] = df["MonthlyIncome"].clip(1000, 20000)

# Salary band derived from income (used heavily in EDA + interview talking points)
df["SalaryBand"] = pd.cut(df["MonthlyIncome"], bins=[0, 3000, 6000, 10000, 20001],
                           labels=["Low", "Medium", "High", "Very High"])

df["TotalWorkingYears"] = np.clip((df["Age"] - 20) - rng.integers(0, 5, N), 0, 40)
df["YearsAtCompany"] = np.clip(df["TotalWorkingYears"] - rng.integers(0, 6, N), 0, 40)
df["YearsSinceLastPromotion"] = np.clip(
    (rng.integers(0, df["YearsAtCompany"].clip(lower=1) + 1)), 0, 15
)
df["YearsWithCurrManager"] = np.clip(df["YearsAtCompany"] - rng.integers(0, 4, N), 0, 40)

df["JobSatisfaction"] = rng.integers(1, 5, N)          # 1=Low, 4=Very High
df["EnvironmentSatisfaction"] = rng.integers(1, 5, N)
df["WorkLifeBalance"] = rng.integers(1, 5, N)
df["JobInvolvement"] = rng.integers(1, 5, N)
df["PerformanceRating"] = rng.choice([3, 4], N, p=[0.85, 0.15])
df["RelationshipSatisfaction"] = rng.integers(1, 5, N)

df["OverTime"] = rng.choice(["Yes", "No"], N, p=[0.28, 0.72])
df["BusinessTravel"] = rng.choice(
    ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], N, p=[0.1, 0.71, 0.19]
)
df["TrainingTimesLastYear"] = rng.integers(0, 7, N)
df["NumCompaniesWorked"] = rng.integers(0, 10, N)
df["StockOptionLevel"] = rng.choice([0, 1, 2, 3], N, p=[0.4, 0.38, 0.15, 0.07])

# ---- Generate Attrition target from a logistic combination of real drivers ----
z = (
    -1.6
    + 1.15 * (df["OverTime"] == "Yes").astype(int)
    + 0.55 * (df["BusinessTravel"] == "Travel_Frequently").astype(int)
    - 0.35 * (df["JobSatisfaction"] - 2.5)
    - 0.30 * (df["EnvironmentSatisfaction"] - 2.5)
    - 0.30 * (df["WorkLifeBalance"] - 2.5)
    - 0.00028 * (df["MonthlyIncome"] - 6500)
    + 0.05 * (df["DistanceFromHome"] - 9)
    - 0.09 * (df["YearsAtCompany"].clip(upper=10) - 5)
    + 0.10 * (df["YearsSinceLastPromotion"] - 2)
    + 0.35 * (df["MaritalStatus"] == "Single").astype(int)
    - 0.12 * (df["Age"] - 37) / 5
    + 0.20 * (df["NumCompaniesWorked"] - 2.5) / 2
    + rng.normal(0, 0.55, N)
)
prob_attrition = 1 / (1 + np.exp(-z))
df["Attrition"] = (rng.random(N) < prob_attrition).astype(int)
df["Attrition"] = df["Attrition"].map({1: "Yes", 0: "No"})

# Reorder columns sensibly
cols = ["EmployeeID", "Age", "Gender", "MaritalStatus", "Department", "JobRole",
        "EducationField", "MonthlyIncome", "SalaryBand", "DistanceFromHome",
        "BusinessTravel", "OverTime", "TotalWorkingYears", "YearsAtCompany",
        "YearsSinceLastPromotion", "YearsWithCurrManager", "NumCompaniesWorked",
        "TrainingTimesLastYear", "StockOptionLevel", "JobSatisfaction",
        "EnvironmentSatisfaction", "WorkLifeBalance", "JobInvolvement",
        "RelationshipSatisfaction", "PerformanceRating", "Attrition"]
df = df[cols]

df.to_csv("hr_attrition_dataset.csv", index=False)
print("Dataset shape:", df.shape)
print("\nAttrition rate:\n", df["Attrition"].value_counts(normalize=True).round(3))
print("\nSample:\n", df.head())
