import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create assessment-required features while preserving raw columns for preprocessing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        result = X.copy()

        dates = pd.to_datetime(result["application_date"], errors="coerce")
        result["application_year"] = dates.dt.year
        result["application_month"] = dates.dt.month

        income = pd.to_numeric(result["income"], errors="coerce")
        loan = pd.to_numeric(result["loan_amount"], errors="coerce")
        result["income_loan_ratio"] = income / loan.replace(0, np.nan)

        result["log_income"] = np.log1p(income.clip(lower=0))

        result = result.drop(columns=["application_date"])
        return result
