"""Retrieval service for querying the BM25 index on codebase chunks."""

import json
import pickle
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

from src.indexer import tokenize
from src.models import (Chunk, MinimalSource, MinimalSearchResults,
                        StudentSearchResults)


class Retriever:
    """Service in charge of loading the index and searching snippets."""

    def __init__(self, index_path: str = "data/processed/index.pkl") -> None:
        """Load the persisted BM25 model and chunk metadata.

        Args:
            index_path: Path to the pickled index file.
        """
        # 1. Charger self.bm25 et self.chunks depuis index_path
        # 2. Gérer le cas où le fichier n'existe pas encore
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            self.bm25 = None
            self.chunks = []
            raise FileNotFoundError("Index file not found")
        with open(self.index_path, "rb") as file:
            saved_data = pickle.load(file)
            self.bm25 = saved_data["bm25"]
            self.chunks = saved_data["chunks"]
