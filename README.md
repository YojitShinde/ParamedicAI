# ParamedicAI

> A source-grounded remote-emergency assistant for exploring how retrieval-augmented generation can support urgent field guidance.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![RAG](https://img.shields.io/badge/Architecture-RAG-0F766E)
![Local LLM](https://img.shields.io/badge/LLM-Ollama-black)
![Evaluation](https://img.shields.io/badge/Evaluation-Gemini--as--a--judge-4285F4)

## Important Safety Notice

ParamedicAI is an experimental educational project. It is **not a medical device, emergency dispatcher, diagnostic system, or substitute for trained professionals**. Never rely on it instead of contacting your local emergency services, mountain rescue, coast guard, poison center, or a qualified clinician.

The generated answers can be incomplete, outdated, or wrong. Verify all medical content against current local protocols and professional guidance.

## What It Does

ParamedicAI combines:

- WMS guideline PDFs as a local knowledge base
- Hugging Face embeddings for document retrieval
- ChromaDB for persistent vector search
- Ollama for local language-model generation
- Gemini-as-a-judge evaluation for safety and medical-accuracy testing

The CLI supports both one-shot questions and interactive conversations. Answers include the source documents retrieved for each query.

## Project Layout

```text
ParamedicAI/
├── config.py                    # Models, retrieval, paths, and response rules
├── ingest.py                    # PDF ingestion and ChromaDB creation
├── rag_engine.py                # Retrieval and Ollama generation
├── main.py                      # Interactive CLI
├── generate_medical_dataset.py  # Gemini-powered synthetic test-set generator
├── evaluate_with_gemini.py      # Medical-accuracy evaluation harness
├── medical_emergency_dataset.jsonl
├── Documents/                   # Local WMS PDFs; not committed by default
└── requirements.txt
```

## Requirements

- Python 3.11 or newer
- [Ollama](https://ollama.com/)
- A local Ollama model, such as `llama3.1:8b`
- WMS guideline PDFs placed in `Documents/`
- A Gemini API key only for dataset generation or evaluation

## Setup

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install and start Ollama:

```powershell
ollama pull llama3.1:8b
ollama serve
```

The default model is configured in `config.py`. Override it without editing code:

```powershell
$env:OLLAMA_MODEL = "llama3.1:8b"
```

## Add Source Documents

Place the WMS PDFs in `Documents/`. Then build the local vector store:

```powershell
python ingest.py
```

To rebuild it after changing the PDFs:

```powershell
python ingest.py --force
```

The PDFs are intentionally ignored by Git. Confirm that redistribution is permitted before adding them to a public repository.

## Ask the Bot

Single-query mode:

```powershell
python main.py --query "What should I do for someone with severe frostbite?"
```

Interactive mode:

```powershell
python main.py
```

Enter `quit`, `exit`, or `q` to stop.

## Generate 300 Test Cases

The generator uses Gemini to create realistic user scenarios and reference answers grounded in the local PDFs. The generated references are evaluation aids, not authoritative medical advice.

```powershell
$env:GEMINI_API_KEY = "your-api-key"
python generate_medical_dataset.py --count 300
```

Output:

```text
medical_emergency_dataset.jsonl
```

## Evaluate Medical Accuracy

The evaluator runs each scenario through ParamedicAI and asks Gemini to judge the result against the matching WMS source document. It reports medical accuracy, safety, triage, action quality, completeness, grounding, unsupported claims, and critical errors.

Run a small smoke test:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
python evaluate_with_gemini.py --limit 3
```

Run the complete dataset:

```powershell
python evaluate_with_gemini.py `
  --dataset medical_emergency_dataset.jsonl `
  --output medical_accuracy_results.json
```

The evaluator uses source-filtered retrieval for each case, so a scenario generated from one WMS document is judged against that document rather than an unrelated mixture of sources.

## Configuration

Key settings live in `config.py`:

| Setting | Purpose |
| --- | --- |
| `MODEL_NAME` | Ollama generation model; supports `OLLAMA_MODEL` override |
| `EMBEDDING_MODEL_NAME` | Hugging Face embedding model |
| `TOP_K` | Number of source chunks retrieved |
| `CHROMA_DB_DIR` | Persistent vector-store location |
| `SYSTEM_PROMPT` | Safety and grounding response contract |

## Development Notes

Before opening a pull request:

```powershell
python -m py_compile config.py rag_engine.py main.py evaluate_with_gemini.py generate_medical_dataset.py
```

Do not commit API keys, `.env` files, local model data, Chroma databases, or source PDFs unless you have permission to redistribute them.

## License and Sources

This repository contains an educational software implementation. Review the licensing and redistribution terms for every WMS document before publishing source PDFs or extracted text. Add an explicit project license before accepting external contributions.

## Roadmap

- Add automated regression tests for retrieval and response structure
- Add source-aware retrieval and answer evaluation for every query type
- Add clinician-reviewed evaluation labels
- Add structured logging for unsupported claims and safety failures
