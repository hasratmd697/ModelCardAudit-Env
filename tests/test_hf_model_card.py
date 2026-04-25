from env.hf_model_card import build_hf_model_card, fetch_hf_readme, normalize_hf_repo_id


def test_fetch_hf_readme_validates_repo_id():
    try:
        fetch_hf_readme("invalid-repo-id")
        assert False, "Expected ValueError for invalid repo id format"
    except ValueError:
        assert True


def test_normalize_hf_repo_id_accepts_full_url():
    normalized = normalize_hf_repo_id("https://huggingface.co/google/gemma-4-31B-it")
    assert normalized == "google/gemma-4-31B-it"


def test_normalize_hf_repo_id_accepts_models_url_form():
    normalized = normalize_hf_repo_id("https://huggingface.co/models/google/gemma-4-31B-it")
    assert normalized == "google/gemma-4-31B-it"


def test_build_hf_model_card_maps_sections(monkeypatch):
    markdown = """
---
pipeline_tag: text-generation
---
# Gemma Demo

Short intro paragraph.

## Intended Use
Use for experimentation only.

## Evaluation Metrics
| Metric | Value |
|---|---|
| Accuracy | 0.91 |

## Bias Analysis
Needs deeper subgroup analysis.
""".strip()

    def fake_fetch(repo_id, revision=None, timeout=20):
        return markdown, "main"

    monkeypatch.setattr("env.hf_model_card.fetch_hf_readme", fake_fetch)

    card = build_hf_model_card(
        repo_id="google/gemma-demo",
        required_sections=["model_description", "evaluation_metrics", "bias_analysis"],
    )

    assert card["model_name"] == "Gemma Demo"
    assert card["model_type"] == "text-generation"
    assert "model_description" in card["sections"]
    assert "evaluation_metrics" in card["sections"]
    assert "bias_analysis" in card["sections"]


def test_build_hf_model_card_fills_missing_required(monkeypatch):
    markdown = """
# Tiny Card

## Model Description
Basic summary.
""".strip()

    def fake_fetch(repo_id, revision=None, timeout=20):
        return markdown, "main"

    monkeypatch.setattr("env.hf_model_card.fetch_hf_readme", fake_fetch)

    card = build_hf_model_card(
        repo_id="org/tiny-card",
        required_sections=["model_description", "bias_analysis"],
    )

    assert card["sections"]["model_description"]
    assert card["sections"]["bias_analysis"] == "Section not provided in the Hugging Face model card."


def test_build_hf_model_card_ignores_generic_load_model_heading(monkeypatch):
    markdown = """
# Load model

Use this section to load weights.
""".strip()

    def fake_fetch(repo_id, revision=None, timeout=20):
        return markdown, "main"

    monkeypatch.setattr("env.hf_model_card.fetch_hf_readme", fake_fetch)

    card = build_hf_model_card(
        repo_id="google/gemma-4-31B-it",
        required_sections=["model_description"],
    )

    assert card["model_name"] == "gemma-4-31B-it"
