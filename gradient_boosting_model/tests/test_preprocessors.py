import pytest
from config.core import config
from processing import preprocessors as pp

@pytest.mark.preprocessors
def test_drop_unnecessary_features_transformer(sample_input_data):
    # Given
    drop_features = set(config.model_config.drop_features)
    input_feature = set(list(sample_input_data.columns))
    
    
    assert drop_features.issubset(input_feature)

    transformer = pp.DropUnecessaryFeatures(
        variables_to_drop=config.model_config.drop_features,
    )

    # When
    X_transformed = transformer.transform(sample_input_data)
    
    # Then
    assert  len(drop_features.intersection(set(X_transformed.columns))) == 0

@pytest.mark.preprocessors
def test_create_new_features(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    transformer = pp.AddNewFeatures()

    # When
    X_transformed = transformer.transform(X_train)

    # Then
    assert (
        X_transformed.iloc[0]["sum_capital_paid_account_0_24m"]
        == X_train.iloc[0]["sum_capital_paid_account_0_12m"] + X_train.iloc[0]["sum_capital_paid_account_12_24m"]
    )
