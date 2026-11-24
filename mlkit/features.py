"""Feature definitions and the preprocessing pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
CATEGORICAL_FEATURES = ["contract", "internet_service", "payment_method"]
TARGET = "churn"


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric features; impute + one-hot encode categorical features."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def build_pipeline(classifier) -> Pipeline:
    """Full model: preprocessing followed by a classifier."""
    return Pipeline([("prep", build_preprocessor()), ("clf", classifier)])
