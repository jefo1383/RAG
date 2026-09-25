"""Chunking strategies for Markdown and Python files."""

import re
from typing import List
from src.models import Chunk, MinimalSource


def chunk_markdown(
    file_path: str,
    content: str,
    max_chunk_size: int = 2000
) -> List[Chunk]:
    """Split a Markdown document into semantic chunks within character limits.

    Args:
        file_path: Relative path to the source file.
        content: Complete text of the Markdown file.
        max_chunk_size: Maximum allowed length in characters for any chunk.

    Returns:
        List of valid Chunk instances covering the document.
    """
    if not content.strip():
        return []
    # 1. Identifier les positions de début de chaque titre (ex: regex sur '^#{1,6} ')
    pattern = r"^#{1,6}\s+"
    header_indices: list[int] = [
        match.start()
        for match in re.finditer(pattern, content, flags=re.MULTILINE)
    ]
    # 2. Construire les intervalles [start, end] entre deux titres consécutifs
    # 3. Ne pas oublier le bloc initial (de 0 au premier titre, s'il existe)
    section_bounds: list[tuple[int, int]] = []
    if not header_indices:
        section_bounds.append((0, len(content)))
    else:
        if header_indices[0] != 0:
            header_indices.insert(0, 0)
        header_indices.append(len(content))
        for i in range(len(header_indices) - 1):
            section_bounds.append((header_indices[i], header_indices[i + 1]))
    # 4. Pour chaque intervalle :
    #    - Si la taille <= max_chunk_size: créer le Chunk
    #    - Si la taille > max_chunk_size: sous-découper (ex: par paragraphes '\n\n')
    # 5. Retourner la liste des chunks
