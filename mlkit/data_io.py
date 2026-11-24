"""Safe CSV loading for customer data."""
import pandas as pd


def read_customers(path):
    """Read a customer CSV.

    Only empty fields are treated as missing. pandas' default NA list contains
    strings such as "None", "NA" and "null", which are legitimate category values here.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"customer_id": str})
