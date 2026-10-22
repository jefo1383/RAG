"""Evaluation metrics and comparison against ground-truth datasets."""

from pathlib import Path
from src.models import (StudentSearchResults, RagDataset, AnsweredQuestion)


def compute_iou(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int
) -> float:
    """Compute Intersection over Union (IoU) between two 1D character spans.

    Args:
        start_a: Start position of interval A.
        end_a: End position of interval A.
        start_b: Start position of interval B.
        end_b: End position of interval B.

    Returns:
        The IoU score as a float between 0.0 and 1.0.
    """
    # 1. Calculer la longueur de l'intersection :
    #    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    # 2. Calculer la longueur de l'union :
    #    union = max(end_a, end_b) - min(start_a, start_b)
    union = max(end_a, end_b) - min(start_a, start_b)
    # 3. Retourner intersection / union (gérer la division par zéro)
    if union != 0:
        return inter / union
    else:
        return 0.0


def evaluate_search_results(
    student_search_results_path: str,
    dataset_path: str,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate student search results against a ground-truth dataset.

    Args:
        student_search_results_path: Path to the generated
        student JSON results.
        dataset_path: Path to the ground-truth AnsweredQuestions JSON dataset.
        k_values: List of k thresholds to compute (default: [1, 3, 5, 10]).

    Returns:
        Dictionary mapping metric names (e.g. 'recall@5') to their score.
    """
    # 1. Charger student results
    results = StudentSearchResults.model_validate_json(
        Path(student_search_results_path).read_text(encoding="utf-8"))
    # 2. Charger le dataset vérité terrain
    dataset = RagDataset.model_validate_json(
        Path(dataset_path).read_text(encoding="utf-8"))
    # 3. Indexer les questions par question_id pour les comparer facilement
    student_by_id = {r.question_id: r for r in results.search_results}
    # 4. Pour chaque valeur de k (ex: 1, 3, 5, 10) :
    #    - Pour chaque question :
    #        - Pour chaque source de référence :
    #            - Vérifier si une source étudiante dans les k premières
    # a le même file_path ET un compute_iou >= 0.05
    #        - Calculer le recall de la question
    #    - Calculer la moyenne du recall@k sur l'ensemble des questions
    if k_values is None:
        k_values = [1, 3, 5, 10]
    scores_by_k: dict[str, float] = {}
    for k in k_values:
        total_recall = 0.0
        for ref_q in dataset.rag_questions:
            if not isinstance(ref_q, AnsweredQuestion):
                continue
            hits = 0
            student_entry = student_by_id.get(ref_q.question_id)
            ref_sources = ref_q.sources
            candidate_sources = (student_entry.retrieved_sources[:k] if
                                 student_entry else [])
            for ref_src in ref_sources:
                for cand_src in candidate_sources:
                    iou = compute_iou(
                        ref_src.first_character_index,
                        ref_src.last_character_index,
                        cand_src.first_character_index,
                        cand_src.last_character_index
                    )
                    if (cand_src.file_path == ref_src.file_path and
                            iou >= 0.05):
                        hits += 1
                        break
            question_recall = (hits / len(ref_sources)) if ref_sources else 1.0
            total_recall += question_recall
        scores_by_k[f"recall@{k}"] = total_recall / len(dataset.rag_questions)
    # 5. Afficher et retourner les résultats
    print("Evaluation Results")
    print("=" * 40)
    for k in k_values:
        score = scores_by_k[f"recall@{k}"]
        print(f"Recall@{k}: {score:.3f}")

    return scores_by_k
