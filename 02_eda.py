import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("hr_attrition_dataset.csv")
df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 1. Attrition rate by Department
dept_rate = df.groupby("Department")["AttritionFlag"].mean().sort_values(ascending=False)
sns.barplot(x=dept_rate.values, y=dept_rate.index, ax=axes[0, 0], palette="Reds_r")
axes[0, 0].set_title("Attrition Rate by Department")
axes[0, 0].set_xlabel("Attrition Rate")
for i, v in enumerate(dept_rate.values):
    axes[0, 0].text(v + 0.005, i, f"{v:.0%}", va="center")

# 2. Attrition rate by Salary Band
band_order = ["Low", "Medium", "High", "Very High"]
band_rate = df.groupby("SalaryBand", observed=True)["AttritionFlag"].mean().reindex(band_order)
sns.barplot(x=band_rate.index, y=band_rate.values, ax=axes[0, 1], palette="Blues_r")
axes[0, 1].set_title("Attrition Rate by Salary Band")
axes[0, 1].set_ylabel("Attrition Rate")
for i, v in enumerate(band_rate.values):
    axes[0, 1].text(i, v + 0.005, f"{v:.0%}", ha="center")

# 3. Attrition rate by OverTime
ot_rate = df.groupby("OverTime")["AttritionFlag"].mean()
sns.barplot(x=ot_rate.index, y=ot_rate.values, ax=axes[1, 0], palette="Oranges_r")
axes[1, 0].set_title("Attrition Rate by OverTime Status")
axes[1, 0].set_ylabel("Attrition Rate")
for i, v in enumerate(ot_rate.values):
    axes[1, 0].text(i, v + 0.005, f"{v:.0%}", ha="center")

# 4. Attrition rate by Years Since Last Promotion (binned)
df["PromoBin"] = pd.cut(df["YearsSinceLastPromotion"], bins=[-1, 0, 1, 3, 6, 15],
                         labels=["0", "1", "2-3", "4-6", "7+"])
promo_rate = df.groupby("PromoBin", observed=True)["AttritionFlag"].mean()
sns.barplot(x=promo_rate.index, y=promo_rate.values, ax=axes[1, 1], palette="Purples_r")
axes[1, 1].set_title("Attrition Rate by Years Since Last Promotion")
axes[1, 1].set_ylabel("Attrition Rate")
axes[1, 1].set_xlabel("Years Since Last Promotion")
for i, v in enumerate(promo_rate.values):
    axes[1, 1].text(i, v + 0.005, f"{v:.0%}", ha="center")

plt.tight_layout()
plt.savefig("eda_attrition_drivers.png", bbox_inches="tight")
print("Saved eda_attrition_drivers.png")

# Correlation heatmap of numeric features
plt.figure(figsize=(10, 8))
numeric_cols = ["Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears",
                 "YearsAtCompany", "YearsSinceLastPromotion", "YearsWithCurrManager",
                 "NumCompaniesWorked", "JobSatisfaction", "EnvironmentSatisfaction",
                 "WorkLifeBalance", "JobInvolvement", "AttritionFlag"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix - Numeric HR Features")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", bbox_inches="tight")
print("Saved eda_correlation_heatmap.png")

print("\nOverall attrition rate: {:.1%}".format(df["AttritionFlag"].mean()))
print("\nAttrition by Department:\n", dept_rate)
print("\nAttrition by Salary Band:\n", band_rate)
print("\nAttrition by OverTime:\n", ot_rate)
