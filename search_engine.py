from pathlib import Path

import faiss
import numpy as np
from faiss import IndexFlatIP

from settings import settings


class SearchEngine:
    def __init__(self, vector_dimension: int) -> None:
        self.vector_dimension = vector_dimension
        self.index_file_path = self.__get_index_file_path()

    def __get_index_file_path(self) -> Path:
        model_name = settings.image_vectorizer_model_name
        index_dir_path = settings.search_engine_index_dir_path
        index_file_path = index_dir_path / f"{model_name}.index"
        return index_file_path

    def get_n_most_similar_indexes(self, query_vector: np.ndarray) -> list[int]:
        n_similar_cases = settings.search_engine_n_similar_cases
        _, indices = self.index.search(query_vector, n_similar_cases)
        similar_indexes = indices[0].tolist()
        return similar_indexes

    def add_vectors(self, vectors: np.ndarray) -> None:
        self.index.add(vectors)

    def save_index(self) -> None:
        faiss.write_index(self.index, str(self.index_file_path))

    def load_index(self) -> None:
        self.index = faiss.read_index(str(self.index_file_path))

    def initialize_index(self) -> None:
        self.index = IndexFlatIP(self.vector_dimension)
