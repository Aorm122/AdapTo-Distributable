import base64
import json
import os
import sys
from pathlib import Path

import requests

MODEL = "gemini-2.5-flash-lite"
PDF_MODEL = "gemini-2.5-flash-lite"
OUTPUT_DIR_DEFAULT = "Lessons/lesson_files"

SEED_EXAMPLES = [
    {
        "id": "OOP1",
        "term": "Encapsulation",
        "keyword": "hides",
        "definition": "the pillar that hides data, preventing users from accessing it",
        "simple_terms": "public = access, private = no access",
        "difficulty": 1,
        "related_to": ["pillars"],
    },
    {
        "id": "OOP2",
        "term": "Inheritance",
        "keyword": "classes",
        "definition": "makes use of class hierarchy to access functions and variables native to other classes",
        "simple_terms": "children gets parent behavior",
        "difficulty": 2,
        "related_to": ["pillars"],
    },
]


def parse_cli(argv: list[str]) -> tuple[dict, list[str]]:
    opts = {
        "pdf_path": "",
        "output_dir": OUTPUT_DIR_DEFAULT,
        "env_file": "",
        "runtime_root": "",
    }
    positional: list[str] = []

    idx = 0
    while idx < len(argv):
        token = argv[idx]

        if token == "--pdf":
            if idx + 1 >= len(argv):
                raise ValueError("Missing value after --pdf")
            opts["pdf_path"] = argv[idx + 1]
            idx += 2
            continue

        if token == "--output-dir":
            if idx + 1 >= len(argv):
                raise ValueError("Missing value after --output-dir")
            opts["output_dir"] = argv[idx + 1]
            idx += 2
            continue

        if token == "--env-file":
            if idx + 1 >= len(argv):
                raise ValueError("Missing value after --env-file")
            opts["env_file"] = argv[idx + 1]
            idx += 2
            continue

        if token == "--runtime-root":
            if idx + 1 >= len(argv):
                raise ValueError("Missing value after --runtime-root")
            opts["runtime_root"] = argv[idx + 1]
            idx += 2
            continue

        positional.append(token)
        idx += 1

    return opts, positional


def parse_env_file(path: Path) -> dict:
    values: dict = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]

        values[key] = value

    return values


def resolve_api_key(explicit_env_file: str, runtime_root: str) -> tuple[str, str, list[Path]]:
    candidates: list[Path] = []

    if explicit_env_file:
        candidates.append(Path(explicit_env_file).expanduser())

    runtime_root_path = Path(runtime_root).expanduser() if runtime_root else Path.cwd()
    candidates.append(runtime_root_path / ".env")
    candidates.append(runtime_root_path / "gemini.env")

    project_root_env = Path(__file__).resolve().parent.parent / ".env"
    candidates.append(project_root_env)

    seen = set()
    deduped_candidates = []
    for path in candidates:
        normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_candidates.append(path)

    for env_path in deduped_candidates:
        if not env_path.is_file():
            continue
        try:
            parsed = parse_env_file(env_path)
        except OSError:
            continue

        api_key = str(parsed.get("GEMINI_API_KEY", "")).strip()
        if api_key:
            return api_key, str(env_path), deduped_candidates

    env_api_key = str(os.environ.get("GEMINI_API_KEY", "")).strip()
    if env_api_key:
        return env_api_key, "environment variable", deduped_candidates

    return "", "", deduped_candidates


def parse_count(value: str, default: int = 8) -> int:
    if value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(parsed, 1)


def resolve_output_dir(raw_output_dir: str) -> Path:
    output_dir = Path(raw_output_dir).expanduser()
    if output_dir.is_absolute():
        return output_dir
    return (Path.cwd() / output_dir).resolve()


def call_gemini(topic: str, count: int, api_key: str) -> list:
    prompt = f"""
Create {count} new lesson items about "{topic}" based on this style.
Return ONLY valid JSON, no markdown fences.
Schema:
{{
  "items": [
    {{
      "id": "AI_{topic[:3].upper()}1",
      "term": "string",
      "keyword": "string",
      "definition": "string",
      "simple_terms": "string",
      "examples": ["string", "string"],
      "accepted_terms": ["string", "string"],
      "difficulty": 1,
      "related_to": ["string"],
      "type_of_information": ["definition", "apply"],
      "tof_statement": {{
        "true": "string",
        "false": "string"
      }}
    }}
  ]
}}
Seed style examples:
{json.dumps(SEED_EXAMPLES, indent=2)}
"""

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "response_mime_type": "application/json",
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={api_key}"
    response = requests.post(url, json=body, timeout=120)
    if response.status_code != 200:
        print(f"HTTP {response.status_code} error. Response body:")
        print(response.text)
    response.raise_for_status()

    root = response.json()
    text_part = root["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text_part)["items"]


def call_gemini_for_pdf(pdf_b64: str, count: int, api_key: str) -> dict:
    pdf_prompt = f"""Analyze this PDF document and create {count} lesson items based on its content.
Return ONLY valid JSON, no markdown fences.
Schema:
{{
  "topic": "string (the main topic of the document)",
  "items": [
    {{
      "id": "string",
      "term": "string",
      "keyword": "string",
      "definition": "string",
      "simple_terms": "string",
      "examples": ["string", "string"],
      "accepted_terms": ["string", "string"],
      "difficulty": 1,
      "related_to": ["string"],
      "type_of_information": ["definition", "apply"],
      "tof_statement": {{
        "true": "string",
        "false": "string"
      }}
    }}
  ]
}}"""

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": pdf_b64,
                        }
                    },
                    {"text": pdf_prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "response_mime_type": "application/json",
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{PDF_MODEL}:generateContent?key={api_key}"
    response = requests.post(url, json=body, timeout=120)
    if response.status_code != 200:
        print(f"HTTP {response.status_code} error:")
        print(response.text)
    response.raise_for_status()

    root = response.json()
    text_part = root["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text_part)


def sanitize_id(value: str) -> str:
    return value.replace(" ", "_").replace(":", "").replace("/", "_").lower()


def write_tres(topic: str, items: list, out_path: Path) -> None:
    load_steps = 2 + len(items)
    lines = []
    lines.append(f'[gd_resource type="Resource" script_class="Lesson" load_steps={load_steps} format=3]')
    lines.append("")
    lines.append('[ext_resource type="Script" uid="uid://dadne10e1geqd" path="res://Lessons/lesson_item.gd" id="1_7hrog"]')
    lines.append('[ext_resource type="Script" uid="uid://7kxviye2bk4p" path="res://Lessons/lesson.gd" id="2_bkryn"]')
    lines.append("")

    resource_ids = []
    for i, item in enumerate(items):
        rid = f"Resource_{sanitize_id(item.get('id', f'item{i}'))}"
        resource_ids.append(rid)
        lines.append(f'[sub_resource type="Resource" id="{rid}"]')
        lines.append('script = ExtResource("1_7hrog")')
        lines.append(f'id = "{item.get("id", "")}"')
        lines.append(f'term = "{item.get("term", "")}"')
        lines.append(f'keyword = "{item.get("keyword", "")}"')
        lines.append(f'definition = "{item.get("definition", "")}"')
        lines.append(f'simple_terms = "{item.get("simple_terms", "")}"')
        lines.append(f'examples = {json.dumps(item.get("examples", []))}')
        lines.append(f'accepted_terms = {json.dumps(item.get("accepted_terms", []))}')
        lines.append(f'difficulty = {item.get("difficulty", 1)}')
        lines.append(f'related_to = {json.dumps(item.get("related_to", []))}')

        tof = item.get("tof_statement", {"true": "", "false": ""})
        lines.append("tof_statement = {")
        lines.append(f'"false": "{tof.get("false", "")}", ')
        lines.append(f'"true": "{tof.get("true", "")}"')
        lines.append("}")

        lines.append(f'type_of_information = {json.dumps(item.get("type_of_information", ["definition"]))}')
        lines.append('metadata/_custom_type_script = "uid://dadne10e1geqd"')
        lines.append("")

    sub_refs = ", ".join([f'SubResource("{rid}")' for rid in resource_ids])
    lines.append("[resource]")
    lines.append('script = ExtResource("2_bkryn")')
    lines.append(f'lesson_title = "{topic}"')
    lines.append(f'lesson_items = Array[ExtResource("1_7hrog")]([{sub_refs}])')
    lines.append('metadata/_custom_type_script = "uid://7kxviye2bk4p"')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")


def main() -> int:
    try:
        options, positional = parse_cli(sys.argv[1:])
    except ValueError as exc:
        print(f"Argument error: {exc}")
        print(
            "Usage: generate_lesson.py [topic] [count] [folder] "
            "[--pdf <pdf_path>] [--output-dir <path>] [--env-file <path>] [--runtime-root <path>]"
        )
        return 1

    api_key, key_source, key_candidates = resolve_api_key(options["env_file"], options["runtime_root"])
    if not api_key:
        print("Error: GEMINI_API_KEY is not set.")
        print("Looked for key files in:")
        for candidate in key_candidates:
            print(f" - {candidate}")
        print("You can also pass --env-file <path> or set GEMINI_API_KEY in the environment.")
        return 1

    output_dir = resolve_output_dir(options["output_dir"])
    print(f"Using API key source: {key_source}")
    print(f"Writing generated lessons to: {output_dir}")

    if options["pdf_path"]:
        pdf_path = Path(options["pdf_path"]).expanduser()
        if not pdf_path.is_file():
            print(f"PDF file not found: {pdf_path}")
            return 1

        count = parse_count(positional[0], 8) if len(positional) > 0 else 8
        folder_arg = positional[1] if len(positional) > 1 else ""

        print(f"Generating {count} items from PDF: {pdf_path}")
        pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
        parsed = call_gemini_for_pdf(pdf_b64, count, api_key)

        pdf_name = pdf_path.stem
        folder = folder_arg if folder_arg else pdf_name
        topic = str(parsed.get("topic", pdf_name.replace("_", " ").replace("-", " ").title()))
        items = parsed.get("items", [])

        print(f"Detected topic: {topic}")
        print(f"Got {len(items)} items from Gemini")

        out_path = output_dir / folder / f"{sanitize_id(topic)}.tres"
        write_tres(topic, items, out_path)
        return 0

    topic = positional[0] if len(positional) > 0 else "Object Oriented Programming"
    count = parse_count(positional[1], 8) if len(positional) > 1 else 8
    folder = positional[2] if len(positional) > 2 else topic

    print(f"Generating {count} items for topic: {topic}")
    items = call_gemini(topic, count, api_key)
    print(f"Got {len(items)} items from Gemini")

    out_path = output_dir / folder / f"{sanitize_id(topic)}.tres"
    write_tres(topic, items, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())