"""Command-line interface for the RAG pipeline."""

import fire
from src.indexer import Indexer


class CLI:
    """Command-line interface for codebase indexing, retrieval, and QA."""

    def index(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        max_chunk_size: int = 2000,
    ) -> None:
        """Ingest raw codebase files and build the persistent index.

        Args:
            raw_dir: Path to the raw codebase directory.
            processed_dir: Directory where the processed index will be saved.
            max_chunk_size: Maximum allowed character length for any chunk.
        """
        indexer = Indexer(raw_dir, processed_dir, max_chunk_size)
        indexer.run()


if __name__ == "__main__":
    fire.Fire(CLI)
