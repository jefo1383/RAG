"""Module for answer generation using retrieved context and causal LLMs."""

import torch
import json
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from typing import Optional
from pathlib import Path
from src.models import (MinimalSource, StudentSearchResults,
                        StudentSearchResultsAndAnswer, MinimalAnswer)
from tqdm import tqdm


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
        self.tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(self.model_name))
        # 4. Charger le modèle via AutoModelForCausalLM,
        # le placer sur le device et en mode eval
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          dtype=dtype)
        self.model.to(device=self.device)  # type: ignore
        # self.model.eval() equivalence pour mypy strict:
        torch.nn.Module.eval(self.model)

    def _extract_source_text(self, source: MinimalSource) -> str:
        """Extract text snippet from disk using character boundaries.

        Args:
            source: MinimalSource containing file path and character span.

        Returns:
            Extracted text snippet, or empty string if file is unreadable.
        """
        # 1. Vérifier défensivement si le fichier existe
        if not Path(source.file_path).exists():
            print(f"Warning: {source.file_path} not found")
            return ""
        # 2. Lire le contenu du fichier avec encodage utf-8
        try:
            with open(Path(source.file_path), encoding="utf-8",
                      errors="replace") as file:
                content = file.read()
        except OSError as err:
            print(f"Warning: Failed to read {source.file_path}: {err}")
            return ""
        # 3. Extraire et retourner la tranche de texte [first:last]
        return content[
            source.first_character_index:source.last_character_index]

    def _build_prompt(
        self,
        query: str,
        sources: list[MinimalSource],
    ) -> str:
        """Build chat prompt combining system instructions, context, and query.

        Args:
            query: The user question.
            sources: list of retrieved MinimalSource snippets.

        Returns:
            Formatted chat prompt ready for tokenization.
        """
        # 1. Extraire et concaténer les extraits textuels de chaque source
        formated_extr = ""
        for i, s in enumerate(sources, 1):
            extract = self._extract_source_text(s)
            if extract:
                formated_extr += f"- Source {i} ({s.file_path}):\n{extract}\n"
        # 2. Créer le message system (rôle, anti-hallucination, concision)
        system = {
            "role": "system",
            "content": ("Answer the question based solely on the provided "
                        "sources. If the context does not contain the "
                        "information, state in your answer that the "
                        "information is not present.")
        }
        # 3. Créer le message user (contexte + question)
        user = {
            "role": "user",
            "content": f"Context:\n{formated_extr}\nQuestion:\n{query}"
        }
        # 4. Appliquer self.tokenizer.apply_chat_template
        # et retourner le prompt
        prompt = self.tokenizer.apply_chat_template(
            conversation=[system, user],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return str(prompt)

    def generate_answer(
        self,
        query: str,
        sources: list[MinimalSource],
        max_new_tokens: int = 256,
    ) -> str:
        """Generate grounded natural-language answer for query using sources.

        Args:
            query: The user question.
            sources: List of retrieved MinimalSource snippets.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Generated answer string.
        """
        # 1. Gestion défensive de query vide
        if not query or not query.strip():
            print("Warning: the question is empty")
            return ""
        # 2. Construction du prompt via _build_prompt
        prompt = self._build_prompt(query, sources)
        # 3. Tokenisation et envoi sur self.device
        inputs = (
            self.tokenizer(prompt, return_tensors="pt").to(self.device)
        )
        # 4. Génération sous torch.no_grad() avec do_sample=False
        with torch.no_grad():
            outputs: torch.LongTensor = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        # 5. Découpage pour extraire uniquement les nouveaux tokens et décodage
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        res = str(self.tokenizer.decode(
            new_tokens, skip_special_tokens=True))
        return res.strip()

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:
        """Generate answers for a dataset of search results and save JSON.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            save_directory: Destination directory for output JSON.
        """
        # 1. Vérifier défensivement si student_search_results_path existe
        if not Path(student_search_results_path).exists():
            print(f"Error: {student_search_results_path} not found")
            return
        # 2. Charger et valider le JSON
        try:
            with open(Path(student_search_results_path)) as file:
                content = json.load(file)
                student_search = StudentSearchResults.model_validate(
                    content)
        except ValueError as e:
            print(e)
            return
        # 3. Boucler sur search_results avec tqdm et générer les réponses
        answers: list[MinimalAnswer] = []
        print(f"Loaded {len(student_search.search_results)} questions ...")
        for data in tqdm(student_search.search_results, desc="Answering",
                         unit="Answers"):
            answer = self.generate_answer(data.question,
                                          data.retrieved_sources)
            answers.append(MinimalAnswer(
                question_id=data.question_id,
                question=data.question,
                retrieved_sources=data.retrieved_sources,
                answer=answer
            ))
        # 4. Construire l'objet final StudentSearchResultsAndAnswer
        dataset_answers = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=student_search.k
        )
        # 5. Sauvegarder dans save_directory avec le nom du fichier source
        output_path = (Path(save_directory) /
                       Path(student_search_results_path).name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(dataset_answers.model_dump_json(indent=2))
        # 6. Afficher le message terminal conforme au sujet
        print(f"Saved student_search_results_and_answer to {output_path}")
