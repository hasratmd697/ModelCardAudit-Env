import re
from urllib.parse import quote, urlparse

import requests


SECTION_ALIASES = {
    "model_description": [
        "model description",
        "model details",
        "overview",
        "about",
        "introduction",
    ],
    "intended_use": [
        "intended use",
        "intended uses",
        "use cases",
        "use-case",
        "uses",
    ],
    "out_of_scope_use": [
        "out of scope",
        "out-of-scope",
        "out of scope use",
        "out-of-scope use",
        "misuse",
    ],
    "limitations": [
        "limitations",
        "known limitations",
        "risks and limitations",
    ],
    "bias_analysis": [
        "bias",
        "fairness",
        "bias analysis",
        "safety and bias",
    ],
    "training_data": [
        "training data",
        "dataset",
        "data",
    ],
    "evaluation_metrics": [
        "evaluation",
        "metrics",
        "benchmark",
        "results",
        "quantitative analyses",
        "performance",
    ],
    "ethical_considerations": [
        "ethical considerations",
        "ethics",
        "safety",
        "responsible ai",
    ],
    "environmental_impact": [
        "environmental impact",
        "carbon",
        "energy",
        "compute",
    ],
    "citation": [
        "citation",
        "references",
        "paper",
    ],
    "license": [
        "license",
    ],
}

GENERIC_TITLE_HEADINGS = {
    "load model",
    "usage",
    "how to use",
    "inference",
    "training",
    "evaluation",
    "examples",
    "quickstart",
    "quick start",
    "model card",
    "models overview",
    "model overview",
    "overview",
    "introduction",
    "getting started",
    "table of contents",
    "contents",
}


def normalize_hf_repo_id(repo_ref: str) -> str:
    if not repo_ref or not repo_ref.strip():
        raise ValueError("Hugging Face repo id must look like 'owner/model-name'.")

    raw_ref = repo_ref.strip()

    if raw_ref.startswith("http://") or raw_ref.startswith("https://"):
        parsed = urlparse(raw_ref)
        host = parsed.netloc.lower()
        if host not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("Only huggingface.co URLs are supported for model-card import.")

        path_parts = [part for part in parsed.path.strip("/").split("/") if part]
        if path_parts and path_parts[0] == "models":
            path_parts = path_parts[1:]

        if len(path_parts) < 2:
            raise ValueError("Hugging Face URL must include owner and model name.")

        return f"{path_parts[0]}/{path_parts[1]}"

    normalized = raw_ref
    if normalized.startswith("huggingface.co/"):
        normalized = normalized[len("huggingface.co/") :]
    if normalized.startswith("models/"):
        normalized = normalized[len("models/") :]

    parts = [part for part in normalized.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError("Hugging Face repo id must look like 'owner/model-name'.")

    return f"{parts[0]}/{parts[1]}"


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _extract_front_matter(markdown: str) -> tuple[dict, str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, markdown

    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break

    if end_index is None:
        return {}, markdown

    raw_front_matter = lines[1:end_index]
    body = "\n".join(lines[end_index + 1 :])

    front_matter = {}
    current_key = None

    for raw_line in raw_front_matter:
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if re.match(r"^[A-Za-z0-9_-]+\s*:", line):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            if value:
                front_matter[key] = value
                current_key = key
            else:
                front_matter[key] = []
                current_key = key
            continue

        if line.lstrip().startswith("- ") and current_key:
            existing = front_matter.get(current_key)
            item = line.lstrip()[2:].strip().strip("\"'")
            if isinstance(existing, list):
                existing.append(item)
            else:
                front_matter[current_key] = [item]

    return front_matter, body


def _split_sections(markdown: str) -> tuple[str, dict]:
    sections = {}
    intro_parts = []

    current_heading = None
    current_lines = []

    def flush_current() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            return

        if current_heading:
            sections.setdefault(current_heading, [])
            sections[current_heading].append(text)
        else:
            intro_parts.append(text)

        current_lines = []

    for line in markdown.splitlines():
        heading_match = re.match(r"^(#{1,2})\s+(.+?)\s*$", line)
        if heading_match:
            flush_current()
            current_heading = re.sub(r"\s*#+\s*$", "", heading_match.group(2)).strip()
            continue

        current_lines.append(line)

    flush_current()

    merged_sections = {
        heading: "\n\n".join(parts).strip()
        for heading, parts in sections.items()
        if "\n\n".join(parts).strip()
    }
    intro_text = "\n\n".join(intro_parts).strip()

    return intro_text, merged_sections


def _map_sections(raw_sections: dict, intro_text: str, required_sections: list[str]) -> dict:
    mapped = {}
    used_headings = set()
    normalized_headings = {heading: _normalize_text(heading) for heading in raw_sections}

    for target_section, aliases in SECTION_ALIASES.items():
        normalized_aliases = [_normalize_text(alias) for alias in aliases]
        matches = [
            heading
            for heading, normalized in normalized_headings.items()
            if any(alias in normalized for alias in normalized_aliases)
        ]

        if not matches:
            continue

        content = "\n\n".join(raw_sections[heading] for heading in matches if raw_sections[heading].strip()).strip()
        if not content:
            continue

        mapped[target_section] = content
        used_headings.update(matches)

    if intro_text and "model_description" not in mapped:
        mapped["model_description"] = intro_text

    for heading, content in raw_sections.items():
        if heading in used_headings or not content.strip():
            continue

        fallback_key = _normalize_key(heading)
        if fallback_key and fallback_key not in mapped:
            mapped[fallback_key] = content.strip()

    missing_text = "Section not provided in the Hugging Face model card."
    for section_name in required_sections:
        mapped.setdefault(section_name, missing_text)

    return mapped


def _infer_framework(markdown: str, front_matter: dict) -> str:
    text = f"{front_matter} {markdown}".lower()
    if "pytorch" in text or "torch" in text:
        return "PyTorch"
    if "tensorflow" in text or "tf.keras" in text:
        return "TensorFlow"
    if "jax" in text:
        return "JAX"
    if "onnx" in text:
        return "ONNX"
    if "transformers" in text:
        return "Transformers"
    return "Unknown"


def _infer_model_type(markdown: str, front_matter: dict) -> str:
    pipeline_tag = front_matter.get("pipeline_tag")
    if isinstance(pipeline_tag, str) and pipeline_tag.strip():
        return pipeline_tag.strip()

    tags = front_matter.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            tag_value = str(tag).strip()
            if tag_value:
                return tag_value

    text = markdown.lower()
    if "text-generation" in text or "causal lm" in text:
        return "text-generation"
    if "image-classification" in text:
        return "image-classification"
    if "token-classification" in text:
        return "token-classification"
    return "unknown"


def _extract_model_name(markdown: str, front_matter: dict, repo_id: str) -> str:
    # Always use the repo_id the user entered (already normalized to "owner/model").
    # This guarantees the displayed name matches exactly what was typed,
    # regardless of what headings appear in the README.
    return repo_id


def fetch_hf_readme(repo_id: str, revision: str | None = None, timeout: int = 20) -> tuple[str, str]:
    normalized_repo_id = normalize_hf_repo_id(repo_id)

    normalized_repo = quote(normalized_repo_id, safe="/-_.")
    revisions = []
    if revision and revision.strip():
        revisions.append(revision.strip())
    for fallback in ("main", "master"):
        if fallback not in revisions:
            revisions.append(fallback)

    last_error = "Unknown error"
    for rev in revisions:
        url = f"https://huggingface.co/{normalized_repo}/raw/{quote(rev, safe='-_.')}/README.md"
        try:
            response = requests.get(url, timeout=timeout)
        except requests.RequestException as exc:
            last_error = str(exc)
            continue

        if response.status_code == 200:
            return response.text, rev

        last_error = f"{response.status_code} {response.reason}"

    raise RuntimeError(
        f"Unable to fetch README.md for '{normalized_repo_id}'. Tried revisions: {', '.join(revisions)}. Last error: {last_error}"
    )


def build_hf_model_card(repo_id: str, required_sections: list[str], revision: str | None = None) -> dict:
    normalized_repo_id = normalize_hf_repo_id(repo_id)
    markdown, resolved_revision = fetch_hf_readme(normalized_repo_id, revision=revision)
    front_matter, body = _extract_front_matter(markdown)
    intro_text, raw_sections = _split_sections(body)
    mapped_sections = _map_sections(raw_sections, intro_text, required_sections)

    safe_repo = re.sub(r"[^a-z0-9]+", "_", normalized_repo_id.lower()).strip("_")
    safe_revision = re.sub(r"[^a-z0-9]+", "_", resolved_revision.lower()).strip("_")

    return {
        "id": f"hf_{safe_repo}_{safe_revision}",
        "model_name": _extract_model_name(body, front_matter, normalized_repo_id),
        "model_type": _infer_model_type(body, front_matter),
        "framework": _infer_framework(body, front_matter),
        "sections": mapped_sections,
        "source": {
            "provider": "huggingface",
            "repo_id": normalized_repo_id,
            "revision": resolved_revision,
        },
    }
