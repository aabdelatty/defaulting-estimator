import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.encoding import OneHotEncoder

class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn pre-processing transformers,
    like the SimpleImputer() or OrdinalEncoder(), to allow
    the use of the transformer on a selected group of variables.
    """

    def __init__(self, variables=None, transformer=None):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        The `fit` method allows scikit-learn transformers to
        learn the required parameters from the training data set.
        """

        self.transformer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""
        X = X.copy()
        temp_df = self.transformer.transform(X[self.variables])
        if type(temp_df) == pd.DataFrame:
            X = X.drop(self.variables , axis=1)
            X = pd.concat([X, temp_df], axis=1)
            X = X.loc[:, ~X.columns.duplicated()]
        else:
            X[self.variables] = temp_df
        return X



class AddNewFeatures(BaseEstimator, TransformerMixin):
    """
        add new variable sm_capital_paid_account_0_24m, num_of_paid_inv_0_12m, and status_max_active_0_24
        1- num_of_paid_inv_0_12m = num_active_inv / num_active_div_by_paid_inv_0_12m
        2- sm_capital_paid_account_0_24m = capital_paid_account_0_12m + m_capital_paid_account_12_24m
        3- status_max_active_0_24 is the max(account_worst_status_0_3m, account_worst_status_12_24m, account_worst_status_3_6m, account_worst_status_6_12m)

    """

    def __init__(self, variables=None):
        self.variables=variables

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        X = X.copy()

        X['sum_capital_paid_account_0_24m'] = X.apply(lambda row : row['sum_capital_paid_account_0_12m'] + row['sum_capital_paid_account_12_24m'], axis=1)
        #X['num_of_paid_inv_0_12m'] = X.apply(lambda row: 0 if (row['num_active_inv'] == 0 | row['num_active_div_by_paid_inv_0_12m'] == 0)  else row['num_active_inv'] / row['num_active_div_by_paid_inv_0_12m'], axis=1)
        X['status_max_active_0_24'] = X.apply(lambda row: max(row['account_worst_status_0_3m'], row['account_worst_status_12_24m'], row['account_worst_status_3_6m'], row['account_worst_status_6_12m']), axis=1)

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        # drop unnecesary / unused features from the data set
        X = X.copy()
        X = X.drop(set(self.variables).intersection(set(X.columns)) , axis=1)

        return X
