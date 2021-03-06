
from sklearn.model_selection import train_test_split

import os, sys, path


from config.core import config
from processing.data_management import load_dataset
import pytest

print(config.app_config.training_data_file)
@pytest.fixture()
def pipeline_inputs():
    # For larger datasets, here we would use a testing sub-sample.
    data = load_dataset(file_name=config.app_config.training_data_file)

    # Divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    return X_train, X_test, y_train, y_test



@pytest.fixture()
def sample_input_data():
    data = load_dataset(file_name=config.app_config.test_data_file)
    data = data.drop(['default'], axis=1)
    return data
