import hashlib
from pathlib import Path

import magic


class IOControllerHelper:

    @classmethod
    def get_image_file_path(cls, url: str, content: bytes | str,
                            base_path: Path) -> Path:
        file_stem = cls.__get_image_file_stem(url=url)
        file_extension = cls.__get_image_file_extension(content=content)
        file_name = f"{file_stem}.{file_extension}"
        file_path = base_path / file_name
        return file_path

    @classmethod
    def get_image_file_stem(cls, url: str) -> str:
        file_stem = hashlib.sha256(url.encode(encoding="utf-8")).hexdigest()
        return file_stem

    @classmethod
    def get_image_file_extension(
            cls, content: bytes | str) -> str:
        mime = magic.from_buffer(content, mime=True)
        file_extension = mime.split("/")[-1]
        return file_extension
