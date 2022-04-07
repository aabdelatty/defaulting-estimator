import typing as t

import pandas as pd


from gradient_boosting_model.config.core import config
from gradient_boosting_model.processing_utils.data_management import load_pipeline, load_dataset
from gradient_boosting_model.processing_utils.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{config.model_config.version}.pkl"
_default_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict],) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": config.model_config.version, "errors": errors}

    if not errors:
        predictions = _default_pipe.predict(
            X=validated_data[config.model_config.features]
        )
        """
        _logger.info(
            f"Making predictions with model version: {config.model_config.version} "
            f"Predictions: {predictions}"
        )
        """
        results = {"predictions": predictions, "version": config.model_config.version, "errors": errors}

    return results
