from pathlib import Path

from image_vectorizer import ImageVectorizer
from io_controller import IOController
from search_engine import SearchEngine


class ImageSearchManager:

    @classmethod
    def initialize(cls) -> None:
        cls.__image_vectorizer = ImageVectorizer()
        cls.__search_engine = SearchEngine(
            vector_dimension=cls.__image_vectorizer.vector_dimension)
        cls.image_file_urls = cls.__get_image_file_urls()
        cls.__load_search_engine_data()

    @classmethod
    def __get_image_file_urls(cls) -> list[str]:
        image_file_urls = IOController.get_train_image_file_urls()
        return image_file_urls

    @classmethod
    def __load_search_engine_data(cls) -> None:
        if cls.__search_engine.index_file_path.exists():
            cls.__search_engine.load_index()
        else:
            train_image_file_paths = IOController.download_train_image_files(
                train_image_file_urls=cls.image_file_urls
            )
            train_image_vectors = cls.__image_vectorizer.get_image_vectors(
                image_file_paths=train_image_file_paths)
            cls.__search_engine.initialize_index()
            cls.__search_engine.add_vectors(vectors=train_image_vectors)
            cls.__search_engine.save_index()

    @classmethod
    def get_n_most_similar_urls(cls, image_file_path: Path) -> list[str]:
        image_vectors = cls.__image_vectorizer.get_image_vectors(
            image_file_paths=[image_file_path])
        n_most_similar_indexes = cls.__search_engine.get_n_most_similar_indexes(
            query_vector=image_vectors)
        n_most_similar_urls = [None] * len(n_most_similar_indexes)
        for index, similar_index in enumerate(n_most_similar_indexes):
            n_most_similar_urls[index] = cls.image_file_urls[similar_index]
        return n_most_similar_urls
