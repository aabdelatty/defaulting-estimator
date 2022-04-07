from sklearn.impute  import SimpleImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


from gradient_boosting_model.processing_utils import preprocessors as pp
from gradient_boosting_model.config.core import config


default_pipe = Pipeline(
    [
        (
            "numerical_imputer",
            pp.SklearnTransformerWrapper(
                variables=config.model_config.numerical_vars,
                transformer=SimpleImputer(strategy="most_frequent"),
            ),
        ),
        (
            "add_new_features",
            pp.AddNewFeatures(),
        ),
        (
            "drop_features",
            pp.DropUnecessaryFeatures(
                variables_to_drop=config.model_config.drop_features,
            ),
        ),
        (
            "categorical_encoder",
            pp.SklearnTransformerWrapper(
                variables=config.model_config.categorical_vars,
                transformer=OneHotEncoder(),
            ),
            
        ),
        (
            "gb_model",
            XGBClassifier(
                n_estimators=config.model_config.n_estimators,
                scale_pos_weight=config.model_config.scale_pos_weight,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
