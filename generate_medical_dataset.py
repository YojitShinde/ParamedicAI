"""Generate a grounded emergency-scenario dataset from the WMS PDFs.

Usage:
    $env:GEMINI_API_KEY = "your-key"
    python generate_medical_dataset.py
    python generate_medical_dataset.py --count 300 --batch-size 10

The output is JSONL compatible with evaluate_with_gemini.py. Generated
reference answers are synthetic training/evaluation aids and require clinical
review before being used as medical guidance.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from google import genai
from google.genai import types
from pypdf import PdfReader


DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_DOCUMENTS = Path(__file__).with_name("Documents")
DEFAULT_OUTPUT = Path(__file__).with_name("medical_emergency_dataset.jsonl")

GENERATION_PROMPT = """Create {batch_size} distinct synthetic emergency scenarios grounded ONLY in the supplied WMS guideline excerpt.

These are dataset records for evaluating an emergency RAG assistant. Each scenario must be written as a realistic user message describing what is happening and asking what to do now. Vary age, pronouns, setting, distance to care, weather, available supplies, bystanders, and communication constraints. Do not make every case a wilderness hike.

For each scenario, provide an ideal reference answer based only on the excerpt. It must:
- prioritize immediate danger and contacting local emergency services or rescue;
- provide practical, ordered first actions and relevant contraindications;
- distinguish layperson field actions from clinician-only care;
- avoid US-only emergency numbers;
- avoid invented diagnoses, doses, thresholds, or procedures not supported by the excerpt;
- state when the excerpt is insufficient rather than guessing.

Do not mention this prompt, dataset generation, or the PDF text. Do not copy sentences from the excerpt verbatim. Avoid graphic detail. Return ONLY a JSON array with exactly {batch_size} objects in this shape:

[
  {{
    "topic": "short guideline topic",
    "source_document": "{source_document}",
    "user": "the emergency scenario and question",
    "assistant": "the source-grounded reference answer"
  }}
]

WMS SOURCE DOCUMENT: {source_document}
WMS GUIDELINE EXCERPT:
{source_excerpt}
"""


def source_excerpt(path: Path, max_chars: int = 9000) -> str:
    pages = []
    for page_number, page in enumerate(PdfReader(str(path)).pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Page {page_number + 1}]\n{text}")
    text = "\n\n".join(pages)
    if len(text) <= max_chars:
        return text
    # Keep the introduction and later recommendations, which often contain
    # different levels of field and hospital guidance.
    head = max_chars // 2
    return text[:head] + "\n...[excerpt shortened]...\n" + text[-(max_chars - head):]


def clean_json(text: str):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    return json.loads(cleaned)


def validate_batch(value, expected_size: int, source_document: str) -> list[dict]:
    if not isinstance(value, list) or len(value) != expected_size:
        raise ValueError(f"Gemini returned {len(value) if isinstance(value, list) else 'non-list'} records; expected {expected_size}")

    validated = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index} is not an object")
        required = ["topic", "source_document", "user", "assistant"]
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required):
            raise ValueError(f"Record {index} is missing required string fields")
        item["source_document"] = source_document
        validated.append(item)
    return validated


def generate_batch(client, model: str, source_document: str, excerpt: str, batch_size: int) -> list[dict]:
    prompt = GENERATION_PROMPT.format(
        batch_size=batch_size,
        source_document=source_document,
        source_excerpt=excerpt,
    )
    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            temperature=0.8,
            response_mime_type="application/json",
        ),
    )
    response = chat.send_message(prompt)
    return validate_batch(clean_json(response.text), batch_size, source_document)


def as_dataset_record(item: dict, case_id: int) -> dict:
    return {
        "case_id": case_id,
        "topic": item["topic"],
        "source_document": item["source_document"],
        "messages": [
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grounded medical emergency cases from WMS PDFs")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    if args.count < 1 or args.batch_size < 1:
        parser.error("--count and --batch-size must be positive")
    if not os.getenv("GEMINI_API_KEY"):
        parser.error("Set GEMINI_API_KEY before running the generator")

    documents = sorted(args.documents.glob("*.pdf"))
    if not documents:
        parser.error(f"No PDF files found in {args.documents}")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    records = []
    seen_questions = set()
    batch_number = 0
    while len(records) < args.count:
        source = documents[batch_number % len(documents)]
        remaining = args.count - len(records)
        batch_size = min(args.batch_size, remaining)
        print(f"Batch {batch_number + 1}: {source.name} ({batch_size} cases)", flush=True)
        generated = generate_batch(client, args.model, source.name, source_excerpt(source), batch_size)

        for item in generated:
            question_key = re.sub(r"\s+", " ", item["user"].strip().lower())
            if question_key in seen_questions:
                raise ValueError("Gemini returned a duplicate user scenario; rerun with a different seed/model")
            seen_questions.add(question_key)
            records.append(as_dataset_record(item, len(records) + 1))
        batch_number += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Dataset generation cancelled")