from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    metadata_file_path: Path
    training_image_dir_path: Path
    production_image_dir_path: Path
    image_vectorizer_model_name: str
    image_vectorizer_batch_size: int
    search_engine_index_dir_path: Path
    search_engine_n_similar_cases: int
    io_controller_n_attempts: int
    io_controller_backoff_factor: float
    io_controller_min_normal_sleep_time: float
    io_controller_max_normal_sleep_time: float

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @field_validator("metadata_file_path", "training_image_dir_path",
                     "production_image_dir_path", "search_engine_index_dir_path",
                     mode="after")
    @classmethod
    def resolve_file_path(cls, file_path: Path) -> Path:
        resolved_file_path = file_path.resolve()
        return resolved_file_path


settings = Settings()

