"""Exploratory data-quality report for the raw churn data."""
import pandas as pd

from mlkit.data_io import read_customers

RAW = "data/raw/customers.csv"

naive = pd.read_csv(RAW)
print("== Naive pd.read_csv ==")
print("internet_service missing:", int(naive["internet_service"].isna().sum()))
print("internet_service values:", sorted(naive["internet_service"].dropna().unique()))

df = read_customers(RAW)
print("\n== Safe read_customers ==")
print("shape:", df.shape)
print("internet_service values:", sorted(df["internet_service"].unique()))
print("\ndtypes:\n" + df.dtypes.to_string())
print("\nmissing per column:\n" + df.isna().sum().to_string())
print("\nexact duplicate rows:", int(df.duplicated().sum()))
print("blank-string total_charges:", int((df["total_charges"].str.strip() == "").sum()))
print("non-numeric total_charges:", int(pd.to_numeric(df["total_charges"], errors="coerce").isna().sum()))
print("\ncontract value counts:\n" + df["contract"].value_counts().to_string())
print("\nmonthly_charges summary:\n" + df["monthly_charges"].describe().round(2).to_string())
print("monthly_charges > 200:", int((df["monthly_charges"] > 200).sum()))

churn_flag = df["churn"].eq("Yes")
print("\nchurn rate by RAW contract label:\n"
      + churn_flag.groupby(df["contract"]).mean().round(3).to_string())

print("\nchurn rate by internet_service:\n"
      + churn_flag.groupby(df["internet_service"]).mean().sort_values(ascending=False).round(3).to_string())

print("\nmedian tenure by churn:\n"
      + df.groupby("churn")["tenure_months"].median().to_string())
