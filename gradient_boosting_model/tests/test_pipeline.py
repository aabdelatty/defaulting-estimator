import pytest
import pipeline
from config.core import config
from processing.validation import validate_inputs


@pytest.mark.pipeline
def test_pipeline_predict_takes_validated_input(pipeline_inputs, sample_input_data):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    pipeline.default_pipe.fit(X_train, y_train)

    # When
    validated_inputs, errors = validate_inputs(input_data=sample_input_data)
    predictions = pipeline.default_pipe.predict(
        validated_inputs[config.model_config.features]
    )

    # Then
    assert predictions is not None
    assert errors is None
