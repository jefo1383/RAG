"""Retrieval service for querying the BM25 index on codebase chunks."""

import json
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from src.indexer import tokenize
from src.models import (Chunk, MinimalSource, MinimalSearchResults,
                        StudentSearchResults, RagDataset)


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
            raise FileNotFoundError("Error: Index file not found. "
                                    "Please run 'index' first")
        with open(self.index_path, "rb") as file:
            saved_data = pickle.load(file)
            self.bm25 = saved_data["bm25"]
            self.chunks: list[Chunk] = saved_data["chunks"]

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:
        """Search the top-k most relevant sources for a given query.

        Args:
            query: The natural language search query.
            k: Number of top candidate sources to retrieve.

        Returns:
            List of MinimalSource instances covering the best matches.
        """
        if not query or not query.strip() or k <= 0 or self.bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        if isinstance(self.bm25, BM25Okapi) and isinstance(
                self.chunks[0], Chunk):
            scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:k]
            return [self.chunks[i].source for i in top_indices]
        else:
            return []

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Run search over a question dataset and persist
        StudentSearchResults JSON.

        Args:
            dataset_path: Path to the input questions JSON dataset.
            k: Number of retrieved candidate sources per question.
            save_directory: Directory where the output JSON file
            will be written.
        """
        # 1. Vérification défensive :
        #    - Vérifier que self.bm25 est chargé
        #    - Vérifier que dataset_path existe
        if self.bm25 is None:
            raise FileNotFoundError("Error: Index file not found. "
                                    "Please run 'index' first.")
        if not Path(dataset_path).exists():
            raise FileNotFoundError("Error: dataset file not found. "
                                    "Please verify the path.")
        # 2. Charger le dataset JSON :
        #    - Lire le contenu du fichier
        #    - Charger le json
        #    - Intercepter d'éventuelles erreurs de JSON malformé
        with open(Path(dataset_path)) as file:
            content = json.load(file)
            dataset = RagDataset.model_validate(content)
        # 3. Parcourir les questions avec tqdm:
        #    - Pour chaque question : appeler self.search
        #    - Construire un MinimalSearchResults
        res: list[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching",
                      unit="query"):
            sources = self.search(q.question, k)
            res.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources
                )
            )
        # 4. Construire l'objet final StudentSearchResults
        results = StudentSearchResults(
                        search_results=res,
                        k=k
                    )
        # 5. Sauvegarde du fichier :
        #    - Le chemin de sortie doit conserver le nom d'origine du dataset
        #    - Créer le dossier parent si besoin
        #    - Écrire le JSON formaté
        output_path = Path(save_directory) / Path(dataset_path).name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(results.model_dump_json(indent=2))
        # 6. Message terminal conforme au sujet (Section 6.7.2) :
        print(f"Saved student_search_results to {output_path}")
