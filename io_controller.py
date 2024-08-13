import random
import time
from pathlib import Path
from urllib.parse import quote

import dask
import httpx
import pandas as pd
from dask import delayed
from pydantic import AnyUrl

from helpers.io_controller_helper import IOControllerHelper
from settings import settings


class IOController:

    @classmethod
    def initialize(cls) -> None:
        cls.__train_image_file_stem_to_paths = {}
        for path in settings.training_image_dir_path.iterdir():
            if path.is_dir():
                continue
            file_stem = path.stem
            cls.__train_image_file_stem_to_paths[file_stem] = path

    @classmethod
    def download_image_file(cls, url: str | AnyUrl, base_path: Path) -> Path:
        if not isinstance(url, str):
            url = str(url)
        file_stem = IOControllerHelper.get_image_file_stem(url=url)
        if file_stem in cls.__train_image_file_stem_to_paths:
            return cls.__train_image_file_stem_to_paths[file_stem]
        n_attempts = settings.io_controller_n_attempts
        backoff_factor = settings.io_controller_backoff_factor
        min_normal_sleep_time = settings.io_controller_min_normal_sleep_time
        max_normal_sleep_time = settings.io_controller_max_normal_sleep_time
        for attempt in range(n_attempts):
            try:
                with httpx.Client() as client:
                    response = client.get(url)
                file_extension = IOControllerHelper.get_image_file_extension(
                    content=response.content
                )
                file_path = base_path / f"{file_stem}.{file_extension}"
                with open(file_path, mode="wb") as file:
                    file.write(response.content)
                normal_sleep_time = random.uniform(min_normal_sleep_time,
                                                   max_normal_sleep_time)
                time.sleep(normal_sleep_time)
                return file_path
            except Exception:
                print(f"Sleep on downloading one file: {url}")
                sleep_time = backoff_factor * (2 ** attempt)
                time.sleep(sleep_time)
        raise Exception("Could not download some files!")

    @classmethod
    def download_train_image_files(cls, train_image_file_urls: list[str]) -> list[Path]:
        download_tasks = [None] * len(train_image_file_urls)
        for index, url in enumerate(train_image_file_urls):
            download_tasks[index] = delayed(cls.download_image_file)(
                url=url, base_path=settings.training_image_dir_path)
        train_image_file_paths = list(
            dask.compute(*download_tasks, scheduler="threads")
        )
        return train_image_file_paths

    @classmethod
    def get_train_image_file_urls(cls) -> list[str]:
        metadata = pd.read_csv(filepath_or_buffer=settings.metadata_file_path)
        metadata.columns=["url"]
        metadata["url"] = metadata["url"].str.replace(
            r'["\[\]]', "", regex=True)
        metadata["url"] = metadata["url"].apply(lambda url: quote(url, safe=":/"))
        urls = metadata["url"].to_list()
        return urls
