
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
file_path = Path(__file__).with_name("sample_eda_sales_dataset.csv")

df = pd.read_csv(file_path)

print("DATASET LOADED SUCCESSFULLY!")
print("\nDataset Shape:", df.shape)

# First five rows
print("\nFirst Five Rows:")
print(df.head())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates().copy()

# Handle missing numeric values
numeric_cols = df.select_dtypes(include="number").columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Handle missing categorical values
categorical_cols = df.select_dtypes(include="str").columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode().iloc[0])

print("\nCleaned Dataset Shape:", df.shape)

# Output folder
output = Path(__file__).with_name("eda_outputs")
output.mkdir(exist_ok=True)

# 1. Sales by Category
category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(output / "sales_by_category.png")
plt.show()

# 2. Sales by Region
region_sales = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(8, 5))
region_sales.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(output / "sales_by_region.png")
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(output / "correlation_heatmap.png")
plt.show()

# 4. Sales Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Sales"], kde=True)
plt.title("Sales Distribution")
plt.tight_layout()
plt.savefig(output / "sales_distribution.png")
plt.show()

# Save cleaned dataset
df.to_csv(output / "cleaned_eda_dataset.csv", index=False)

print("\nEDA PROJECT COMPLETED SUCCESSFULLY!")
print("Check the eda_outputs folder.")
