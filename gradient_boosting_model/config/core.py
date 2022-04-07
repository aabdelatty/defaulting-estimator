from pathlib import Path
import os
import typing as t

from pydantic import BaseModel, validator
from strictyaml import load, YAML


# Directories
cwd = Path(__file__).resolve().parent#Path(os.getcwd())
PACKAGE_ROOT = cwd.parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = ROOT / "data"



class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    full_data_file: str
    training_data_file: str
    test_data_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    drop_features: t.Sequence[str]
    target: str
    version: str
    features: t.Sequence[str]
    numerical_vars: t.Sequence[str]
    categorical_vars: t.Sequence[str]
    test_size: float
    random_state: int
    n_estimators: int
    scale_pos_weight: int


    


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    return CONFIG_FILE_PATH


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
