*This activity has been created as part of the 42 curriculum by jfoeller.*

# RAG Against The Machine: Codebase Question-Answering System

## Description

**RAG Against The Machine** is a complete Retrieval-Augmented Generation (RAG) system built to answer technical questions about a complex codebase: the [vLLM](https://github.com/vllm-project/vllm) repository (`vllm-0.10.1`).

Large Language Models often lack specific, up-to-date knowledge about private or rapidly evolving codebases. This project bridges that gap by combining:
1. **Syntax-aware chunking** that respects the structure of both Python source code and Markdown documentation.
2. **Lexical retrieval (BM25)** that finds the most relevant code and documentation snippets in milliseconds.
3. **Local generation** using the lightweight `Qwen/Qwen3-0.6B` model running on GPU (PyTorch FP16) to provide accurate, grounded answers without hallucinations.

The system is fully evaluated using a custom 1D interval Intersection-over-Union (IoU) recall metric and strictly complies with the 42 project requirements.

---

## Instructions

### Prerequisites

- **OS**: Linux (Fedora / Ubuntu).
- **Python**: Version 3.10 or higher.
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (required by the subject).
- **Hardware**: An NVIDIA GPU with CUDA support is recommended for generation, but the system also runs on CPU.

### Installation

Clone the repository and install dependencies using `uv`:

```bash
# Standard installation
make install

# School 42 cluster installation (creates virtualenv and caches in /goinfre)
make install-42
```

### Makefile Rules

The project includes standard Makefile targets defined by the subject:

| Command | Description |
| :--- | :--- |
| `make install` | Install dependencies using `uv sync`. |
| `make install-42` | Set up virtual environment and caches inside `/goinfre` with symlinks. |
| `make run cmd="<command>"` | Execute a CLI command through `uv run python -m src <command>`. |
| `make debug` | Launch the interactive Python debugger (`pdb`). |
| `make clean` | Remove Python cache directories (`__pycache__`, `.mypy_cache`). |
| `make fclean` | Remove virtual environment and all generated temporary caches. |
| `make re` | Clean everything and reinstall from scratch. |
| `make lint` | Run mandatory code quality checks: `flake8` and `mypy` with project flags. |
| `make lint-strict` | Run strict code quality checks: `flake8` and `mypy --strict`. |
| `make doc` | Validate docstrings using `pydocstyle` (Google convention). |

---

## System Architecture

The project is structured as a modular Python package located in the `src/` directory:

```
RAG/
├── src/
│   ├── __init__.py        # Package initialization
│   ├── __main__.py        # CLI entry point using Python Fire
│   ├── models.py          # Pydantic data schemas and validation contracts
│   ├── chunking.py        # Python AST and Markdown segmentation strategies
│   ├── indexer.py         # File discovery, tokenization, and BM25 index creation
│   ├── retriever.py       # Query search and batch dataset retrieval
│   ├── generator.py       # Prompt building and local inference with Qwen3-0.6B
│   └── evaluation.py      # IoU computation and Recall@k evaluation logic
├── data/
│   ├── raw/               # Raw vLLM codebase to index
│   ├── processed/         # Serialized BM25 index (index.pkl)
│   ├── datasets/          # Public evaluation question datasets
│   └── output/            # Search results and generated answer JSON files
├── Makefile               # Project workflow automation
├── pyproject.toml         # Dependencies and tool configurations
└── README.md              # Project documentation
```

### Component Workflow

1. **Ingestion & Chunking (`src/chunking.py`, `src/indexer.py`)**:
   The indexer scans all `.py` and `.md` files in `data/raw/`. It routes each file to either the Python AST chunker or the Markdown chunker. Each chunk stores its raw content, relative file path, and exact start/end character boundaries.
2. **Indexing (`src/indexer.py`)**:
   Chunks are tokenized using an alphanumeric regex tokenizer. A `BM25Okapi` index is built over all chunk tokens and persisted to `data/processed/index.pkl`.
3. **Retrieval (`src/retriever.py`)**:
   Given a user question, the query is tokenized and scored against all indexed chunks using BM25. The top-$k$ highest-scoring chunks are returned as `MinimalSource` objects.
4. **Answer Generation (`src/generator.py`)**:
   Retrieved sources are read from disk to extract the real text snippets. The generator builds a structured prompt using Qwen's ChatML template with `enable_thinking=False`. The local model `Qwen/Qwen3-0.6B` generates a direct, grounded answer.
5. **Evaluation (`src/evaluation.py`)**:
   Measures retrieval quality against reference answers by checking file path matches and character interval IoU ($\ge 0.05$).

---

## Chunking Strategy

Chunking is a critical step in RAG: chunks that are too large dilute lexical match scores, while chunks that are too small lose semantic context. In addition, the subject enforces a **strict maximum length of 2000 characters per chunk**; any chunk exceeding this limit invalidates evaluation results.

We implemented two distinct chunking strategies in `src/chunking.py`:

### 1. Markdown Chunking (`chunk_markdown`)
- **Header Detection**: Uses regular expressions to identify Markdown headers (`#`, `##`, `###`). The document is split into logical sections.
- **Sliding Window for Large Sections**: If a single section exceeds 2000 characters, it is broken down into smaller paragraph blocks with a **100-character overlap** to maintain continuity between adjacent pieces.
- **Greedy Aggregation**: Short consecutive sections under the same hierarchical context are merged together up to the 2000-character boundary to create information-dense chunks.

### 2. Python Code Chunking (`chunk_python_code`)
- **AST Parsing**: Instead of naive line splitting, we parse the source code into an Abstract Syntax Tree using Python's built-in `ast` module.
- **Function & Decorator Spans**: Functions and their decorators (`@property`, `@staticmethod`, etc.) are captured as single coherent units using `ast.get_source_segment`.
- **Large Class Decomposition**: When an entire class fits within 2000 characters, it is kept as a single chunk. If a class exceeds 2000 characters, it is decomposed into its individual methods, each prefixed with the class header to preserve namespace context.
- **Module-Level Statements**: Top-level constants, type aliases, and module statements are aggregated and flushed into chunks before functions or classes appear.

---

## Retrieval Method

We selected **BM25Okapi** as our lexical retrieval algorithm (via the `rank-bm25` library):

1. **Why Lexical Retrieval for Code?**
   Technical questions about codebases frequently use exact names: class names (e.g., `LLMEngine`), command flags (`--tensor-parallel-size`), or file paths. Lexical algorithms excel at exact keyword matching where dense vector embeddings sometimes blur subtle technical differences.
2. **Tokenization**:
   We implemented a fast alphanumeric regex tokenizer (`re.findall(r"[a-zA-Z0-9]+", text.lower())`). It strips punctuation while preserving full identifier names and numbers, ensuring exact keyword overlap.
3. **BM25 Scoring & Ranking**:
   BM25 scores the query against each document by taking into account:
   - Term frequency (how often a term appears in the chunk, with saturation).
   - Inverse document frequency (rare words receive higher weight than common ones).
   - Document length normalization (penalizing overly long chunks).
   Top-$k$ indices are extracted efficiently using `numpy.argsort`.

---

## Performance Analysis

All benchmarks were measured on a laptop running Fedora Linux with an **NVIDIA GeForce RTX 4050 GPU** (6 GB VRAM) and an Intel Core i7 CPU.

### Recall@k Results

We evaluated our retrieval system on the official public reference datasets using the IoU threshold of 0.05:

| Dataset | Metric | 42 Required Threshold | Our Result | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Documentation** (`dataset_docs_public.json`) | Recall@5 | $\ge 80.0\%$ | **82.0%** | **PASSED** |
| **Source Code** (`dataset_code_public.json`) | Recall@5 | $\ge 50.0\%$ | **53.5%** | **PASSED** |

Detailed breakdown across $k$ values:

- **Documentation**: Recall@1 = 57.0% | Recall@3 = 76.0% | **Recall@5 = 82.0%** | Recall@10 = 89.0%
- **Source Code**: Recall@1 = 28.5% | Recall@3 = 45.0% | **Recall@5 = 53.5%** | Recall@10 = 61.5%

### System Throughput & Latency

| Operation | 42 Constraint | Measured Time | Performance |
| :--- | :---: | :---: | :---: |
| **Full Ingestion & Indexing** (1,941 files, 23,801 chunks) | $\le 5$ minutes | **~2.1 seconds** | ~140x faster than limit |
| **Retrieval Throughput** (200 questions across code & docs) | $\le 90$ seconds | **~11.2 seconds** | ~18 questions/second |
| **Answer Generation** (100 questions with Qwen3-0.6B on GPU) | N/A | **~100 seconds** | ~1 questions/second |

---

## Design Decisions

1. **BM25 vs Dense Embeddings**:
   Dense neural embeddings (like sentence-transformers) require slow embedding generation at index time (often taking 10 to 20 minutes) and consume GPU memory. BM25 indexes the entire 1,941 files in just 2.1 seconds on CPU, uses zero VRAM during indexing, and achieves higher accuracy on exact code symbols.
2. **AST Parsing vs Fixed Line Splitting**:
   Splitting code by fixed line count often cuts functions in half, breaking syntax and disconnecting variable definitions from their usage. AST parsing ensures every chunk is syntactically meaningful.
3. **Disabling Thinking Tokens (`enable_thinking=False`)**:
   `Qwen/Qwen3-0.6B` is natively an agentic reasoning model that outputs internal thoughts inside `<think>...</think>`. While useful for math problems, this adds latency and wastes tokens in concise RAG QA. By setting `enable_thinking=False` in `apply_chat_template`, the model generates the final answer directly, cutting inference time by half.
4. **Deterministic Greedy Decoding (`do_sample=False`)**:
   For technical documentation, reproducible answers are essential. Greedy decoding picks the most likely token at each step, preventing random variations and limiting hallucinations.
5. **Pydantic Validation**:
   All data passing through each stage is validated by Pydantic models (`MinimalSource`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`), ensuring complete data integrity.

---

## Challenges Faced

1. **Strict 2000-Character Constraint**:
   - *Problem*: In Python, large classes with many methods easily exceed 2000 characters. If a class chunk is even 2001 characters long, the 42 moulinette rejects the entire output.
   - *Solution*: We implemented dynamic class decomposition: if `len(class_source) > 2000`, the chunker iterates through each method in the class AST body and exports each method as an independent chunk, prefixed by the class signature.
2. **IoU Evaluation and Duplicate Prevention**:
   - *Problem*: When calculating Recall@k, if multiple retrieved chunks belong to the same file and overlap with the same reference source, naive matching could over-count matches.
   - *Solution*: Added a `break` statement once a reference source is matched by any candidate chunk, ensuring each ground-truth source is counted at most once per question.
3. **Hugging Face Typing Stubs with Mypy Strict**:
   - *Problem*: `PreTrainedModel.to` uses `@wraps(torch.nn.Module.to)`, which breaks Mypy's method binding checks in strict mode. Additionally, third-party libraries (`fire`, `rank-bm25`, `tqdm`) lack PEP 561 `py.typed` markers.
   - *Solution*: Delegated device transfer to `torch.nn.Module.to(self.model, self.device)` and added type overrides in `pyproject.toml`, achieving **0 errors under `mypy --strict`**.
4. **Degenerate Inputs & Edge Cases**:
   - *Problem*: The subject requires handling empty queries, negative $k$, missing files, and malformed JSON gracefully without crashing.
   - *Solution*: Designed a comprehensive 25-point edge-case test suite covering all CLI entry points with defensive `try...except` error handling.

---

## Example Usage

Every command is invoked through `uv run python -m src <command> [options]`.

### 1. Index the Codebase

Ingest the raw codebase from `data/raw/` and serialize the BM25 index to `data/processed/`:

```bash
uv run python -m src index --max_chunk_size 2000
```

### 2. Search a Single Query

Retrieve and display the top-$k$ source locations for a question:

```bash
uv run python -m src search "How to configure OpenAI server?" --k 5
```

Output:
```text
data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [458:1976]
data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py [120:1840]
data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [2100:3650]
```

### 3. Batch Search over a Dataset

Run retrieval on a full dataset and export a `StudentSearchResults` JSON:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### 4. Evaluate Search Results (Recall@k)

Compute Recall@1, 3, 5, 10 against a reference ground-truth dataset:

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Output:
```text
Evaluation Results
========================================
Recall@1: 0.570 Recall@3: 0.760 Recall@5: 0.820 Recall@10: 0.890
```

### 5. Answer a Single Query

Retrieve snippets and generate a natural-language answer with `Qwen/Qwen3-0.6B`:

```bash
uv run python -m src answer "How to configure OpenAI server?" --k 3
```

Output:
```text
Answer: To configure the OpenAI server, you need to start the vLLM server with the appropriate flags and set the API key and API base. For example, you can start the server with the command `vllm serve <model_name>` and set the API key and base URL accordingly.
```

### 6. Generate Answers for a Dataset

Produce the final `StudentSearchResultsAndAnswer` JSON file for an entire dataset:

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

---

## Resources & AI Disclosure

### References

- **BM25 Algorithm**: Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Information Retrieval, 3(4), 333-389.
- **Qwen3 Language Model**: Alibaba Cloud Qwen Team. *Qwen Technical Report*.
- **Hugging Face Transformers**: Wolf et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*.
- **Python AST Module**: Official Python documentation for `ast` (Abstract Syntax Tree).
- **Pydantic**: Data validation and settings management using Python type hints.

### AI Usage Disclosure

In compliance with the School 42 guidelines, this project utilized AI assistance (Google DeepMind's Antigravity AI coding assistant) as follows:
- **Pedagogical Guidance**: Used as a mentor to clarify RAG concepts, explain causal language model generation mechanisms, and discuss architectural trade-offs (lexical BM25 vs dense embeddings).
- **Pair Programming & Code Review**: Assisted in structuring prototypes, static typing annotations (Mypy strict compatibility), and Google-style docstrings (PEP 257).
- **Debugging & Type Resolution**: Helped diagnose upstream Hugging Face typing discrepancies with `@wraps` on `PreTrainedModel.to` and suggested delegating calls to `torch.nn.Module`.
- **Edge-Case Test Design**: Assisted in creating the automated 25-point edge-case validation test suite to ensure robust handling of degenerate inputs.

All final code, implementation decisions, and performance benchmarks were reviewed, tested, and validated by the student.

