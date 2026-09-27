from __future__ import annotations

from pathlib import Path

import yaml

from wakewordworld.hub.dataset_card import hf_licence_id, licence_family, render_dataset_card
from wakewordworld.manifest.schema import CANARY


def test_licence_helpers() -> None:
    assert hf_licence_id("CC-BY-3.0-DE") == "cc-by-3.0"
    assert hf_licence_id("CC0-1.0") == "cc0-1.0"
    assert hf_licence_id("PUBLIC-DOMAIN") == "cc0-1.0"
    assert licence_family("CC-BY-SA-4.0") == "cc-by-sa"
    assert licence_family("CC-BY-3.0-DE") == "cc-by"
    assert licence_family("CC0-1.0") == "cc0"
    assert licence_family("CDLA-PERMISSIVE-1.0") == "other-open"


def test_card_front_matter(manifest_dir: Path) -> None:
    text = render_dataset_card(manifest_dir, repo_id="org/bench")
    assert text.startswith("---\n")
    fm_text, body = text[4:].split("\n---\n", 1)
    fm = yaml.safe_load(fm_text)
    assert fm["license"] == ["cc-by-3.0", "cc-by-sa-4.0", "cc0-1.0"]
    assert "cc-by-nc-nd-3.0" not in " ".join(fm["license"])
    assert fm["language"] == ["de", "en"]
    names = [c["config_name"] for c in fm["configs"]]
    assert names == ["cc0", "cc-by", "cc-by-sa", "metadata"]
    assert fm["configs"][1]["data_files"][0]["path"] == "data/cc-by/*.parquet"
    assert CANARY in fm["extra_gated_prompt"]
    assert "not use this audio" in fm["extra_gated_prompt"]
    assert "checkbox" in fm["extra_gated_fields"].values()
    assert "| kuechenradio |" in body
    assert "Tier B" in body
    assert "org/bench" in body
