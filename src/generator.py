"""Module for answer generation using retrieved context and causal LLMs."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from pathlib import Path
from src.models import MinimalSource


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
        # 1. Déterminer le device avec repli défensif
        self.model_name = model_name
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. "
                  "Falling back to CPU.")
            self.device = "cpu"
        elif device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 2. Choisir le dtype adapté (float16 pour cuda, float32 pour cpu)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # 3. Charger le tokenizer via AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # 4. Charger le modèle via AutoModelForCausalLM,
        # le placer sur le device et en mode eval
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          dtype=dtype)
        self.model.to(device=self.device)  # type: ignore
        self.model.eval()

    def _extract_source_text(self, source: MinimalSource) -> str:
        """Extract text snippet from disk using character boundaries.

        Args:
            source: MinimalSource containing file path and character span.

        Returns:
            Extracted text snippet, or empty string if file is unreadable.
        """
        # 1. Vérifier défensivement si le fichier existe

        # 2. Lire le contenu du fichier avec encodage utf-8

        # 3. Extraire et retourner la tranche de texte [first:last]
