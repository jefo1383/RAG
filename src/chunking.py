"""Chunking strategies for Markdown and Python files."""

import re
import ast
from typing import List
from src.models import Chunk, MinimalSource


def _split_large_section(
    file_path: str,
    section_text: str,
    start_index: int,
    max_chunk_size: int = 2000
) -> list[Chunk]:
    """Split a large Markdown section into chunks respecting max_chunk_size.

    Args:
        file_path: Relative path to the file.
        section_text: Text content of the large section.
        start_index: Starting index of this section in the original file.
        max_chunk_size: Maximum allowed length for any chunk.

    Returns:
        List of Chunk instances, each strictly <= max_chunk_size.
    """
    # 1. Creation dúne liste díntervalles en fonction
    # des separateurs de paragraphes:
    pattern = r"\n\s*\n"
    split_indices: list[tuple[int, int]] = [
        (match.start(), match.end())
        for match in re.finditer(pattern, section_text, flags=re.MULTILINE)
    ]
    section_bounds: list[tuple[int, int]] = []
    if not split_indices:
        section_bounds.append((0, len(section_text)))
    else:
        if split_indices[0][0] != 0:
            split_indices.insert(0, (0, 0))
        end_indice = len(section_text)
        if split_indices[-1][1] != end_indice:
            split_indices.append((end_indice, end_indice))
        for i in range(len(split_indices) - 1):
            section_bounds.append((split_indices[i][1],
                                   split_indices[i + 1][0]))
    # 2. Sous-decoupage:
    good_sections: list[tuple[int, int]] = []
    for p_start, p_end in section_bounds:
        if p_end - p_start <= max_chunk_size:
            good_sections.append((p_start, p_end))
        else:
            nb_sub = (p_end - p_start) // max_chunk_size
            for _ in range(nb_sub):
                n_end = p_start + max_chunk_size
                good_sections.append((p_start, n_end))
                p_start = n_end
            if p_start < p_end:
                good_sections.append((p_start, p_end))
    # 3. Fusion des petits intervalles + overlap:
    merge_sections: list[tuple[int, int]] = []
    cur_start, cur_end = good_sections[0]
    for p_start, p_end in good_sections[1:]:
        if p_end - cur_start <= max_chunk_size:
            cur_end = p_end
        else:
            merge_sections.append((cur_start, cur_end))
            new_start = max(0, p_start - 100)
            if p_end - new_start > max_chunk_size:
                new_start = p_start
            cur_start, cur_end = new_start, p_end
    merge_sections.append((cur_start, cur_end))
    # 4. Conversion en indices absolus, création et retour des Chunk
    chunks: list[Chunk] = []
    for sub_start, sub_end in merge_sections:
        sub_text = section_text[sub_start:sub_end]
        if not sub_text.strip():
            continue
        chunks.append(
            Chunk(
                content=sub_text,
                source=MinimalSource(
                    file_path=file_path,
                    first_character_index=start_index + sub_start,
                    last_character_index=start_index + sub_end
                )
            )
        )
    return chunks


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
    # 1. Identifier les positions de début de chaque titre
    pattern = r"^#{1,6}\s+"
    header_indices: list[int] = [
        match.start()
        for match in re.finditer(pattern, content, flags=re.MULTILINE)
    ]
    # 2. Construire les intervalles [start, end] entre deux titres consécutifs
    #    Ne pas oublier le bloc initial et final
    section_bounds: list[tuple[int, int]] = []
    if not header_indices:
        section_bounds.append((0, len(content)))
    else:
        if header_indices[0] != 0:
            header_indices.insert(0, 0)
        header_indices.append(len(content))
        for i in range(len(header_indices) - 1):
            section_bounds.append((header_indices[i], header_indices[i + 1]))
    # 3. Pour chaque intervalle :
    #    - Si la taille <= max_chunk_size: créer le Chunk
    #    - Si la taille > max_chunk_size: sous-découper
    chunk_list: list[Chunk] = []
    for section in section_bounds:
        section_text = content[section[0]:section[1]]
        section_len = section[1] - section[0]
        if section_len <= max_chunk_size:
            chunk_list.append(Chunk(content=section_text,
                                    source=MinimalSource(
                                        file_path=file_path,
                                        first_character_index=section[0],
                                        last_character_index=section[1]
                                    )))
        else:
            sub_chunks = _split_large_section(
                file_path=file_path,
                section_text=section_text,
                start_index=section[0],
                max_chunk_size=max_chunk_size
            )
            chunk_list.extend(sub_chunks)
    # 4. Retourner la liste des chunks
    return chunk_list


def _get_node_span(
    node: ast.stmt,
    line_offsets: list[int]
) -> tuple[int, int]:
    """Get character span (start, end) for an AST node including decorators.

    Args:
        node: An AST node with location attributes.
        line_offsets: Precalculated line start offsets.

    Returns:
        Tuple of (start_character_index, last_character_index).
    """
    if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.decorator_list):
        start_line = node.decorator_list[0].lineno
        start_col = node.decorator_list[0].col_offset
    else:
        start_line = node.lineno
        start_col = node.col_offset
    end_line = (node.end_lineno if node.end_lineno is not None else
                node.lineno)
    end_col = (node.end_col_offset if node.end_col_offset is not None else
               node.col_offset)
    start_char = line_offsets[start_line - 1] + start_col
    end_char = line_offsets[end_line - 1] + end_col
    return (start_char, end_char)


def chunk_python_code(
    file_path: str,
    content: str,
    max_chunk_size: int = 2000
) -> list[Chunk]:
    """Split Python source code into semantic chunks using an AST.

    Args:
        file_path: Relative path to the Python file.
        content: Complete source code of the Python file.
        max_chunk_size: Maximum allowed length for any chunk.

    Returns:
        List of valid Chunk instances covering functions, classes, and modules.
    """
    if not content.strip():
        return []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _split_large_section(file_path, content, 0, max_chunk_size)
    line_offsets: list[int] = [0]
    for line in content.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))

    chunks: list[Chunk] = []
    module_nodes: list[ast.stmt] = []

    def flush_module_node() -> None:
        """Create a chunk from accumulated module-level statements."""
        if not module_nodes:
            return
        m_start, _ = _get_node_span(module_nodes[0], line_offsets)
        _, m_end = _get_node_span(module_nodes[-1], line_offsets)
        if m_end - m_start <= max_chunk_size:
            chunks.append(
                Chunk(
                    content=content[m_start:m_end],
                    source=MinimalSource(
                        file_path=file_path,
                        first_character_index=m_start,
                        last_character_index=m_end
                    )
                )
            )
        else:
            chunks.extend(
                _split_large_section(
                    file_path=file_path,
                    section_text=content[m_start:m_end],
                    start_index=m_start,
                    max_chunk_size=max_chunk_size
                )
            )
        module_nodes.clear()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush_module_node()
            # Traiter la fonction (vérifier la taille vs max_chunk_size)
            start_fn, end_fn = _get_node_span(node, line_offsets)
            if end_fn - start_fn <= max_chunk_size:
                chunks.append(
                    Chunk(
                        content=content[start_fn:end_fn],
                        source=MinimalSource(
                            file_path=file_path,
                            first_character_index=start_fn,
                            last_character_index=end_fn
                        )
                    )
                )
            else:
                chunks.extend(
                    _split_large_section(
                        file_path=file_path,
                        section_text=content[start_fn:end_fn],
                        start_index=start_fn,
                        max_chunk_size=max_chunk_size
                    )
                )
        elif isinstance(node, ast.ClassDef):
            flush_module_node()
            # Traiter la classe (petite -> 1 chunk,
            # grande -> méthode par méthode)
            start_class, end_class = _get_node_span(node, line_offsets)
            if end_class - start_class <= max_chunk_size:
                chunks.append(
                    Chunk(
                        content=content[start_class:end_class],
                        source=MinimalSource(
                            file_path=file_path,
                            first_character_index=start_class,
                            last_character_index=end_class
                        )
                    )
                )
            else:
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef,
                                           ast.AsyncFunctionDef)):
                        start_m, end_m = _get_node_span(method, line_offsets)
                        if end_m - start_m <= max_chunk_size:
                            chunks.append(
                                Chunk(
                                    content=content[start_m:end_m],
                                    source=MinimalSource(
                                        file_path=file_path,
                                        first_character_index=start_m,
                                        last_character_index=end_m
                                    )
                                )
                            )
                        else:
                            chunks.extend(
                                _split_large_section(
                                    file_path=file_path,
                                    section_text=content[start_m:end_m],
                                    start_index=start_m,
                                    max_chunk_size=max_chunk_size
                                )
                            )
        else:
            module_nodes.append(node)

    flush_module_node()
    return chunks
