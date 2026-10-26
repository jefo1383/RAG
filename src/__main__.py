"""Command-line interface for the RAG pipeline."""

import fire
from src.indexer import Indexer
from src.retriever import Retriever
from src.evaluation import evaluate_search_results
from src.generator import Generator
from typing import Optional
from pathlib import Path


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

    def search(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed/index.pkl",
    ) -> None:
        """Search and display the top-k source locations for a single query.

        Args:
            query: The natural language search query.
            k: Number of relevant source snippets to return.
            index_path: Path to the serialized index file.
        """
        try:
            retriever = Retriever(index_path)
        except FileNotFoundError as e:
            print(e)
            return
        sources_list = retriever.search(query, k)
        if not sources_list:
            return
        for s in sources_list:
            print(f"{s.file_path} "
                  f"[{s.first_character_index}:{s.last_character_index}]")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_path: str = "data/processed/index.pkl",
    ) -> None:
        """Run batch search on a dataset and save results to a JSON file.

        Args:
            dataset_path: Path to the JSON dataset of questions.
            k: Number of retrieved sources to keep per question.
            save_directory: Destination folder for the result JSON.
            index_path: Path to the serialized index file.
        """
        try:
            retriever = Retriever(index_path)
            retriever.search_dataset(dataset_path, k, save_directory)
        except FileNotFoundError as e:
            print(e)
            return

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate student search results against a ground-truth dataset.

        Args:
            student_search_results_path: Path to the student search
            results JSON.
            dataset_path: Path to the ground-truth AnsweredQuestions JSON.
        """
        # 1. Vérifier que les deux fichiers existent (gestion défensive)
        if (not Path(student_search_results_path).exists() or not
                Path(dataset_path).exists()):
            print(f"Error: {student_search_results_path} or "
                  f"{dataset_path} not found")
            return
        # 2. Appeler evaluate_search_results
        evaluate_search_results(student_search_results_path,
                                dataset_path)

    def answer(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed/index.pkl",
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
    ) -> None:
        """Answer a single query using retrieved codebase context.

        Args:
            query: The natural language question to answer.
            k: Number of relevant source snippets to retrieve.
            index_path: Path to the serialized index file.
            model_name: Hugging Face model identifier.
            device: Computing device ('cuda', 'cpu', etc.).
        """
        # 1. Vérification défensive de query
        if not query or not query.strip():
            print("Please add a query")
            return
        # 2. Instancier Retriever et récupérer les top-k sources
        try:
            retriever = Retriever(index_path)
        except FileNotFoundError as e:
            print(e)
            return
        sources = retriever.search(query, k)
        if not sources:
            print("Please try with another parameters")
            return
        # 3. Instancier Generator et générer la réponse
        gen = Generator(model_name=model_name, device=device)
        ans = gen.generate_answer(query, sources)
        # 4. Afficher la réponse dans le terminal
        print(f"Answer: {ans}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
    ) -> None:
        """Generate answers for a dataset and write results to JSON.

        Args:
            student_search_results_path: Path to student search results JSON.
            save_directory: Destination directory for output JSON.
            model_name: Hugging Face model identifier.
            device: Computing device ('cuda', 'cpu', etc.).
        """
        # 1. Vérifier défensivement que student_search_results_path existe
        if not Path(student_search_results_path).exists():
            print(f"Error: {student_search_results_path} not found")
            return
        # 2. Instancier Generator et appeler generator.answer_dataset
        gen = Generator(model_name=model_name, device=device)
        gen.answer_dataset(student_search_results_path, save_directory)


if __name__ == "__main__":
    fire.Fire(CLI)
