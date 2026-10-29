"""
Regression tests for scripts/prepare_ff_official.py.

Covers:
  Task 5 — validate_only / load_splits correctness:
    - malformed JSON
    - malformed pair entries
    - duplicate pairs within a split
    - cross-split video-ID overlap
    - clean split passes
    - PASS message lists exactly the properties checked and not
      the properties that are NOT checked

No real FF++ data is used. Split files are synthesised in-memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add the scripts/ directory to sys.path so we can import prepare_ff_official
SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_splits(split_dir: Path, data: dict) -> None:
    """Write split JSON files; keys are split names, values are list-of-pairs."""
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs in data.items():
        (split_dir / f"{name}.json").write_text(
            json.dumps(pairs), encoding="utf-8"
        )


def _valid_splits():
    """Return a minimal valid dataset with no overlap."""
    return {
        "train": [["001", "002"], ["003", "004"]],
        "val":   [["005", "006"]],
        "test":  [["007", "008"]],
    }


# ---------------------------------------------------------------------------
# load_splits — error cases
# ---------------------------------------------------------------------------

class TestLoadSplitsMalformed:
    """load_splits must raise ValueError or FileNotFoundError on bad input."""

    def test_missing_split_file_raises(self, tmp_path):
        from prepare_ff_official import load_splits
        # Write only train.json; val.json and test.json are absent
        d = tmp_path / "splits"
        d.mkdir()
        (d / "train.json").write_text(json.dumps([["001", "002"]]), encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            load_splits(d)

    def test_malformed_json_raises(self, tmp_path):
        from prepare_ff_official import load_splits
        d = tmp_path / "splits"
        d.mkdir()
        for split in ("train", "val", "test"):
            (d / f"{split}.json").write_text("{NOT JSON}", encoding="utf-8")
        with pytest.raises(Exception):
            load_splits(d)

    def test_non_list_json_raises(self, tmp_path):
        from prepare_ff_official import load_splits
        d = tmp_path / "splits"
        _write_splits(d, {"train": {"key": "not-a-list"}, "val": [], "test": []})
        with pytest.raises((ValueError, TypeError, AttributeError)):
            load_splits(d)

    def test_pair_with_wrong_length_raises(self, tmp_path):
        """A pair entry with 3 elements → ValueError."""
        from prepare_ff_official import load_splits
        d = tmp_path / "splits"
        _write_splits(d, {
            "train": [["001", "002", "003"]],   # 3 elements — invalid
            "val":   [["004", "005"]],
            "test":  [["006", "007"]],
        })
        with pytest.raises(ValueError):
            load_splits(d)

    def test_non_numeric_id_raises(self, tmp_path):
        """A pair entry with a non-numeric ID → ValueError."""
        from prepare_ff_official import load_splits
        d = tmp_path / "splits"
        _write_splits(d, {
            "train": [["abc", "002"]],
            "val":   [["004", "005"]],
            "test":  [["006", "007"]],
        })
        with pytest.raises(ValueError):
            load_splits(d)

    def test_self_pair_raises(self, tmp_path):
        """A pair where both IDs are equal → ValueError."""
        from prepare_ff_official import load_splits
        d = tmp_path / "splits"
        _write_splits(d, {
            "train": [["001", "001"]],
            "val":   [["004", "005"]],
            "test":  [["006", "007"]],
        })
        with pytest.raises(ValueError):
            load_splits(d)


class TestLoadSplitsDuplicatePairs:
    """Duplicate unordered pair within the same split → ValueError."""

    def test_exact_duplicate_raises(self, tmp_path):
        from prepare_ff_official import load_splits
        d = tmp_path / "splits"
        _write_splits(d, {
            "train": [["001", "002"], ["001", "002"]],   # exact duplicate
            "val":   [["003", "004"]],
            "test":  [["005", "006"]],
        })
        with pytest.raises(ValueError, match="(?i)duplicate"):
            load_splits(d)

    def test_reversed_duplicate_raises(self, tmp_path):
        """["001","002"] and ["002","001"] are the same unordered pair."""
        from prepare_ff_official import load_splits
        d = tmp_path / "splits"
        _write_splits(d, {
            "train": [["001", "002"], ["002", "001"]],
            "val":   [["003", "004"]],
            "test":  [["005", "006"]],
        })
        with pytest.raises(ValueError, match="(?i)duplicate"):
            load_splits(d)


class TestLoadSplitsCrossSplitOverlap:
    """A video ID that appears in two different splits → ValueError."""

    def test_id_overlap_raises(self, tmp_path):
        from prepare_ff_official import load_splits
        d = tmp_path / "splits"
        # ID "002" is in both train and val
        _write_splits(d, {
            "train": [["001", "002"]],
            "val":   [["002", "003"]],   # "002" reused
            "test":  [["004", "005"]],
        })
        with pytest.raises(ValueError, match="(?i)overlap"):
            load_splits(d)

    def test_no_overlap_succeeds(self, tmp_path):
        from prepare_ff_official import load_splits
        d = tmp_path / "splits"
        _write_splits(d, _valid_splits())
        # Should not raise
        ids_by_split, pairs_by_split, id_to_split, pair_to_split = load_splits(d)
        assert len(ids_by_split["train"]) == 4
        assert len(ids_by_split["val"]) == 2
        assert len(ids_by_split["test"]) == 2


# ---------------------------------------------------------------------------
# validate_only — PASS message content
# ---------------------------------------------------------------------------

class TestValidateOnlyMessage:
    """
    The PASS message must list exactly the properties that were checked,
    and must not claim to verify things that were not checked.
    """

    def _capture_validate_output(self, tmp_path, capsys):
        from prepare_ff_official import validate_only
        d = tmp_path / "splits"
        _write_splits(d, _valid_splits())
        validate_only(d)
        return capsys.readouterr().out

    def test_pass_message_mentions_json_parsing(self, tmp_path, capsys):
        output = self._capture_validate_output(tmp_path, capsys)
        assert "json pars" in output.lower(), (
            "PASS message must state that JSON parsing was checked"
        )

    def test_pass_message_mentions_malformed_pairs(self, tmp_path, capsys):
        output = self._capture_validate_output(tmp_path, capsys)
        assert "malformed" in output.lower(), (
            "PASS message must state that malformed pairs were checked"
        )

    def test_pass_message_mentions_duplicate_pairs(self, tmp_path, capsys):
        output = self._capture_validate_output(tmp_path, capsys)
        assert "duplicate" in output.lower(), (
            "PASS message must state that duplicate pairs were checked"
        )

    def test_pass_message_mentions_cross_split_overlap(self, tmp_path, capsys):
        output = self._capture_validate_output(tmp_path, capsys)
        assert "overlap" in output.lower(), (
            "PASS message must state that cross-split ID overlap was checked"
        )

    def test_pass_message_does_not_claim_authenticity(self, tmp_path, capsys):
        output = self._capture_validate_output(tmp_path, capsys)
        # The old message "PASS: official split files are internally consistent."
        # could be read as claiming authenticity.  The new message must
        # explicitly disclaim authenticity checks.
        lower = output.lower()
        assert "authenticity" in lower or "not checked" in lower, (
            "PASS message must disclaim release-authenticity verification"
        )

    def test_pass_message_does_not_claim_full_completeness(self, tmp_path, capsys):
        output = self._capture_validate_output(tmp_path, capsys)
        lower = output.lower()
        assert "completeness" in lower or "not checked" in lower, (
            "PASS message must disclaim full-dataset completeness verification"
        )

    def test_pass_message_does_not_claim_person_identity(self, tmp_path, capsys):
        output = self._capture_validate_output(tmp_path, capsys)
        lower = output.lower()
        assert "person" in lower or "identity" in lower or "not checked" in lower, (
            "PASS message must disclaim person-identity-independence verification"
        )

    def test_pass_message_does_not_claim_near_duplicate_independence(self, tmp_path, capsys):
        output = self._capture_validate_output(tmp_path, capsys)
        lower = output.lower()
        assert "near-duplicate" in lower or "near_duplicate" in lower or "not checked" in lower, (
            "PASS message must disclaim near-duplicate independence verification"
        )
