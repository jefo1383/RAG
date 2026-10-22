"""Module for answer generation using retrieved context and causal LLMs."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class Generator:
    """Generate answers grounded in retrieved codebase snippets."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
    ) -> None:
        """Initialize tokenizer and causal language model.

        Args:
            model_name: Hugging Face model identifier.
            device: Computing device ('cuda', 'cpu', etc.).
        """
        # 1. Déterminer le device (cuda si dispo, sinon cpu)

        # 2. Choisir le dtype adapté (float16 pour cuda, float32 pour cpu)

        # 3. Charger le tokenizer via AutoTokenizer

        # 4. Charger le modèle via AutoModelForCausalLM, le placer sur le device et en mode eval