# Data Cleaning & Visualization Project
# Run this file in VS Code from the folder containing sample_sales_dataset.csv

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "sample_sales_dataset.csv"
OUTPUT_FILE = "cleaned_sales_dataset.csv"
CHART_DIR = "sales_charts"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: '{INPUT_FILE}' was not found.")
        print("Place sample_sales_dataset.csv in the same folder as this Python file.")
        return

    # 1. Load dataset
    df = pd.read_csv(INPUT_FILE)
    print("\n--- ORIGINAL DATASET ---")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    # 2. Standardize text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    for col in ["City", "Category", "Sales_Channel"]:
        if col in df.columns:
            df[col] = df[col].str.title()

    # 3. Convert columns to appropriate types
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    for col in ["Quantity", "Unit_Price", "Sales"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Treat negative sales and non-positive quantities/prices as invalid
    df.loc[df["Sales"] < 0, "Sales"] = np.nan
    df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan
    df.loc[df["Unit_Price"] <= 0, "Unit_Price"] = np.nan

    # 4. Remove duplicate records
    df = df.drop_duplicates().copy()

    # 5. Fill missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # If a date is missing, fill with the most common date
    if df["Order_Date"].isna().any() and df["Order_Date"].notna().any():
        df["Order_Date"] = df["Order_Date"].fillna(df["Order_Date"].mode().iloc[0])

    # Recalculate sales from quantity and unit price for consistency
    df["Sales"] = df["Quantity"] * df["Unit_Price"]

    # 6. Handle numeric outliers using IQR capping
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower_bound, upper_bound)

    # 7. Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print("\n--- CLEANING SUMMARY ---")
    print("Cleaned shape:", df.shape)
    print("Missing values remaining:", int(df.isnull().sum().sum()))
    print("Duplicate rows remaining:", int(df.duplicated().sum()))
    print(f"Cleaned dataset saved as: {OUTPUT_FILE}")

    # 8. Create visualizations
    os.makedirs(CHART_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Sales by category
    category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=category_sales.index, y=category_sales.values)
    plt.title("Total Sales by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Sales")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "sales_by_category.png"), dpi=150)
    plt.show()

    # Sales by city
    city_sales = df.groupby("City")["Sales"].sum().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=city_sales.index, y=city_sales.values)
    plt.title("Total Sales by City")
    plt.xlabel("City")
    plt.ylabel("Sales")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "sales_by_city.png"), dpi=150)
    plt.show()

    # Monthly sales trend
    monthly_sales = df.groupby(df["Order_Date"].dt.to_period("M"))["Sales"].sum()
    plt.figure(figsize=(9, 5))
    monthly_sales.index = monthly_sales.index.astype(str)
    sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker="o")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "monthly_sales_trend.png"), dpi=150)
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(7, 5))
    correlation = df.select_dtypes(include=np.number).corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "correlation_heatmap.png"), dpi=150)
    plt.show()

    print(f"\nCharts saved in the '{CHART_DIR}' folder.")
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()
