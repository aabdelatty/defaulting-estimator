import pytest
from processing.validation import validate_inputs


@pytest.mark.validation
def test_validate_inputs_identifies_type_errors(sample_input_data):
    # Given
    test_inputs = sample_input_data.copy()

    # introduce errors
    test_inputs.at[1, "uuid"] = 5  # we expect a string

    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert errors
    assert errors[1] == {"uuid": ["Not a valid string."]}

@pytest.mark.validation
def test_validate_inputs_identifies_none_errors(sample_input_data):
    # Given
    test_inputs = sample_input_data.copy()

    # introduce errors
    test_inputs.at[1, "account_amount_added_12_24m"] = None  # we expect a string

    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert errors
    assert errors[1] == {'account_amount_added_12_24m': ['Field may not be null.']}
