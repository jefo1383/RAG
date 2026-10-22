"""Indexing service and tokenization utilities for BM25 retrieval."""

import re
import pickle
from tqdm import tqdm
from src.models import Chunk
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.chunking import chunk_markdown, chunk_python_code


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words for BM25."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class Indexer:
    """Service scanning corpus, chunking, and persisting the BM25 index."""

    def __init__(
            self,
            raw_dir: str = "data/raw",
            processed_dir: str = "data/processed",
            max_chunk_size: int = 2000
    ) -> None:
        """Initialize the Indexer with configurable paths and parameters.

        Args:
            raw_dir: Path to the raw corpus directory.
            processed_dir: Directory where the processed index will be saved.
            max_chunk_size: Maximum allowed character length for any chunk.
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.max_chunk_size = max_chunk_size

    def _discover_files(self) -> list[Path]:
        """Find all eligible Python and Markdown files in the raw directory.

        Returns:
            Sorted list of Path objects for all indexable files.
        """
        # 1. Vérifier si self.raw_dir existe (gestion défensive)
        if not self.raw_dir.exists():
            return []
        # 2. Parcourir récursivement self.raw_dir pour trouver
        # les fichiers .py et .md en excluant les dossiers __pycache__ ou .git
        path_list: list[Path] = [
            p for p in self.raw_dir.rglob("*") if p.is_file() and
            p.suffix in (".py", ".md") and
            "__pycache__" not in p.parts and
            ".git" not in p.parts
        ]
        # 3. Retourner la liste triée
        return sorted(path_list)

    def _chunk_corpus(self, files: list[Path]) -> list[Chunk]:
        """Process all discovered files and return all extracted chunks.

        Args:
            files: List of file paths to ingest.

        Returns:
            List of all Chunk instances extracted across the corpus.
        """
        all_chunks: list[Chunk] = []
        # 1. Parcourir les fichiers avec tqdm
        # 2. Pour chaque fichier :
        #    - Lire son contenu textuel en UTF-8
        #    - Selon son extension (.py ou .md), appeler la bonne fonction
        #      de chunking
        #    - Ajouter les chunks obtenus à all_chunks
        for file_path in tqdm(files, desc="Chunking", unit="file"):
            with open(file_path, encoding="utf-8", errors="ignore") as file:
                content = file.read()
                if file_path.suffix == ".py":
                    all_chunks.extend(
                        chunk_python_code(file_path.as_posix(), content,
                                          self.max_chunk_size)
                    )
                elif file_path.suffix == ".md":
                    all_chunks.extend(
                        chunk_markdown(file_path.as_posix(), content,
                                       self.max_chunk_size)
                    )
        # 3. Retourner all_chunks
        return all_chunks

    def run(self) -> None:
        """Execute the indexing pipeline and persist the BM25 index."""
        files = self._discover_files()
        if not files:
            print("No files found")
            return
        chunks = self._chunk_corpus(files)
        tokenized_corpus = [
            tokenize(chunk.content)
            for chunk in tqdm(chunks, desc="Tokenizing", unit="chunk")
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        with open(self.processed_dir / "index.pkl", "wb") as file:
            pickle.dump(
                {"bm25": bm25, "chunks": chunks},
                file
            )
        print(f"Ingestion complete! Indexed {len(chunks)} chunks "
              f"under {self.processed_dir}/")
