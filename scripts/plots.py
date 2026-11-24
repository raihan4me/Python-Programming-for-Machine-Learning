"""Generate EDA figures on a headless server."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # must be set before importing pyplot
import matplotlib.pyplot as plt

from mlkit.data_io import read_customers

FIG_DIR = Path("reports/figures")
NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen", "churn"]


def save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)                         # free memory: essential in loops/batch jobs
    print(f"saved {path}")


def tenure_hist(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    for value, label in [(0, "Stayed"), (1, "Churned")]:
        ax.hist(df.loc[df["churn"] == value, "tenure_months"], bins=24, alpha=0.6, label=label)
    ax.set(title="Tenure distribution by churn", xlabel="Tenure (months)", ylabel="Customers")
    ax.legend()
    save(fig, "tenure_by_churn.png")


def churn_by_contract(df):
    rates = df.groupby("contract")["churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(rates.index, rates.values, color="steelblue")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
    ax.set(title="Churn rate by contract type", ylabel="Churn rate", ylim=(0, 1))
    save(fig, "churn_by_contract.png")
    return rates


def correlation_heatmap(df):
    corr = df[NUMERIC].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUMERIC)), labels=NUMERIC, rotation=45, ha="right")
    ax.set_yticks(range(len(NUMERIC)), labels=NUMERIC)
    for i in range(len(NUMERIC)):
        for j in range(len(NUMERIC)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation matrix (numeric features and target)")
    save(fig, "correlation_heatmap.png")
    return corr


def monthly_box(df):
    groups = [df.loc[df["churn"] == v, "monthly_charges"].dropna() for v in (0, 1)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(groups, labels=["Stayed", "Churned"])
    ax.set(title="Monthly charges by churn", ylabel="Monthly charges")
    save(fig, "monthly_by_churn.png")
    return df.groupby("churn")["monthly_charges"].median()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = read_customers("data/processed/customers_clean.csv")
    tenure_hist(df)
    print("\nchurn rate by contract:\n" + churn_by_contract(df).round(3).to_string())
    corr = correlation_heatmap(df)
    print("\ncorrelation with churn:\n" + corr["churn"].drop("churn").sort_values().round(3).to_string())
    print(f"\ncorr(tenure, total_charges) = {corr.loc['tenure_months', 'total_charges']:.3f}")
    print("\nmedian monthly_charges by churn:\n" + monthly_box(df).round(2).to_string())


if __name__ == "__main__":
    main()
