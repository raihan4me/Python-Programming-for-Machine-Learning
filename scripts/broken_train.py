"""Show why raw DataFrames break scikit-learn estimators."""
from sklearn.linear_model import LogisticRegression

from mlkit.data_io import read_customers

df = read_customers("data/processed/customers_clean.csv")
y = df["churn"]
attempts = {
    "all columns": df.drop(columns=["churn"]),
    "numeric only": df[["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]],
}
for label, X in attempts.items():
    try:
        LogisticRegression(max_iter=1000).fit(X, y)
        print(f"{label} -> OK")
    except ValueError as err:
        print(f"{label} -> ValueError: {str(err).splitlines()[0]}")
