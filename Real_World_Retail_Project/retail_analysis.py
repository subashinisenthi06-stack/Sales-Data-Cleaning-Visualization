
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 55)
print("REAL-WORLD RETAIL SALES DATA ANALYSIS")
print("=" * 55)

# ------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------

BASE = Path(__file__).parent
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

FILE_PATH = BASE / "sample_sales_dataset (1).csv"

# ------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------

print("\n1. LOADING DATASET")

if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {FILE_PATH}\n"
        "Keep the CSV file in the same folder as this Python file."
    )

df = pd.read_csv(FILE_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# Check required columns
required_columns = [
    "Order_ID",
    "Order_Date",
    "Customer_Name",
    "City",
    "Category",
    "Product",
    "Quantity",
    "Unit_Price",
    "Sales",
    "Sales_Channel"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# ------------------------------------------------
# 3. DATASET INFORMATION
# ------------------------------------------------

print("\n2. DATASET INFORMATION")
print(df.info())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())

# ------------------------------------------------
# 4. DATA CLEANING
# ------------------------------------------------

print("\n3. DATA CLEANING")

# Remove duplicate records
df = df.drop_duplicates().copy()

# Convert date column
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

# Convert numeric columns
numeric_columns = ["Quantity", "Unit_Price", "Sales"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with invalid dates or numeric values
df = df.dropna(
    subset=["Order_Date", "Quantity", "Unit_Price", "Sales"]
).copy()

# Keep only positive values
df = df[
    (df["Quantity"] > 0) &
    (df["Unit_Price"] > 0) &
    (df["Sales"] > 0)
].copy()

# Fill missing text values
text_columns = [
    "Customer_Name",
    "City",
    "Category",
    "Product",
    "Sales_Channel"
]

for col in text_columns:
    df[col] = df[col].fillna("Unknown")

# Save cleaned dataset
df.to_csv(
    OUTPUT / "cleaned_sales_dataset.csv",
    index=False
)

print("Data cleaning completed!")
print("Cleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ------------------------------------------------
# 5. BASIC SALES ANALYSIS
# ------------------------------------------------

print("\n4. BASIC SALES ANALYSIS")

total_sales = df["Sales"].sum()
total_orders = df["Order_ID"].nunique()
total_quantity = df["Quantity"].sum()
average_sales = df["Sales"].mean()
total_customers = df["Customer_Name"].nunique()

print("Total Sales:", total_sales)
print("Total Orders:", total_orders)
print("Total Quantity Sold:", total_quantity)
print("Average Sales:", round(average_sales, 2))
print("Total Customers:", total_customers)

# ------------------------------------------------
# 6. CATEGORY-WISE SALES
# ------------------------------------------------

print("\n5. CATEGORY-WISE SALES")

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print(category_sales)

category_sales.to_csv(
    OUTPUT / "category_wise_sales.csv"
)

# ------------------------------------------------
# 7. CITY-WISE SALES
# ------------------------------------------------

print("\n6. CITY-WISE SALES")

city_sales = (
    df.groupby("City")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print(city_sales)

city_sales.to_csv(
    OUTPUT / "city_wise_sales.csv"
)

# ------------------------------------------------
# 8. SALES CHANNEL ANALYSIS
# ------------------------------------------------

print("\n7. SALES CHANNEL ANALYSIS")

channel_sales = (
    df.groupby("Sales_Channel")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print(channel_sales)

channel_sales.to_csv(
    OUTPUT / "sales_channel_analysis.csv"
)

# ------------------------------------------------
# 9. TOP SELLING PRODUCTS
# ------------------------------------------------

print("\n8. TOP 10 SELLING PRODUCTS")

top_products = (
    df.groupby("Product")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_products)

top_products.to_csv(
    OUTPUT / "top_selling_products.csv"
)

# ------------------------------------------------
# 10. MONTHLY SALES ANALYSIS
# ------------------------------------------------

print("\n9. MONTHLY SALES ANALYSIS")

monthly_sales = (
    df.set_index("Order_Date")
    .resample("ME")["Sales"]
    .sum()
)

print(monthly_sales)

monthly_sales.to_csv(
    OUTPUT / "monthly_sales.csv"
)

# ------------------------------------------------
# 11. CUSTOMER-WISE SALES
# ------------------------------------------------

print("\n10. CUSTOMER-WISE SALES")

customer_sales = (
    df.groupby("Customer_Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print(customer_sales)

customer_sales.to_csv(
    OUTPUT / "customer_wise_sales.csv"
)

# ------------------------------------------------
# 12. DATA VISUALIZATION
# ------------------------------------------------

print("\n11. GENERATING CHARTS")

sns.set_theme(style="whitegrid")

# Chart 1: Category-wise Sales
plt.figure(figsize=(9, 5))
category_sales.plot(kind="bar")
plt.title("Category-wise Sales")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT / "category_sales.png")
plt.close()

# Chart 2: City-wise Sales
plt.figure(figsize=(9, 5))
city_sales.plot(kind="bar")
plt.title("City-wise Sales")
plt.xlabel("City")
plt.ylabel("Total Sales")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT / "city_sales.png")
plt.close()

# Chart 3: Sales Channel Distribution
plt.figure(figsize=(7, 5))
channel_sales.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Sales Channel Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT / "sales_channel.png")
plt.close()

# Chart 4: Monthly Sales Trend
plt.figure(figsize=(9, 5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(OUTPUT / "monthly_sales.png")
plt.close()

# Chart 5: Top Selling Products
plt.figure(figsize=(9, 5))
top_products.plot(kind="bar")
plt.title("Top 10 Selling Products")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT / "top_products.png")
plt.close()

print("All charts generated successfully!")

# ------------------------------------------------
# 13. FINAL REPORT
# ------------------------------------------------

print("\n12. GENERATING FINAL REPORT")

best_category = category_sales.idxmax()
best_city = city_sales.idxmax()
best_channel = channel_sales.idxmax()
best_product = top_products.idxmax()

report = f"""
=========================================
REAL-WORLD RETAIL SALES ANALYSIS REPORT
=========================================

DATASET SUMMARY

Original Records: 52
Records After Cleaning: {len(df)}
Total Columns: {len(df.columns)}

SALES SUMMARY

Total Sales: {total_sales}
Total Orders: {total_orders}
Total Quantity Sold: {total_quantity}
Average Sales: {average_sales:.2f}
Total Customers: {total_customers}

BUSINESS INSIGHTS

Best Selling Category: {best_category}
Highest Sales City: {best_city}
Best Performing Sales Channel: {best_channel}
Top Selling Product: {best_product}

CATEGORY-WISE SALES

{category_sales.to_string()}

CITY-WISE SALES

{city_sales.to_string()}

MONTHLY SALES

{monthly_sales.to_string()}

=========================================
END OF REPORT
=========================================
"""

with open(
    OUTPUT / "final_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print("\nFinal report saved successfully!")

# ------------------------------------------------
# 14. PROJECT COMPLETION
# ------------------------------------------------

print("\n" + "=" * 55)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 55)

print("Cleaned data, analysis files, charts and report")
print("are saved inside the output folder.")
