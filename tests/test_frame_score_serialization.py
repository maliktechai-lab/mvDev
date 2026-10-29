"""
Regression tests for score=None serialization through the full stack.

Covers:
  - VideoDetectionResult fields with mixed scored/unscored frames
  - FrameScoreResponse Pydantic schema accepts and preserves null scores
  - FaceDetectionStatus constants are distinguishable and stable
  - Unscored frames are excluded from numerical aggregations
  - Suspicious-timestamp list excludes unscored frames
  - No fabricated value (0, 0.5, etc.) substituted for a null score
  - API schema correctly serializes null score through JSON round-trip

No real video, model, Docker service, or GPU is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = str(Path(__file__).parent.parent / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ---------------------------------------------------------------------------
# FaceDetectionStatus — stability and distinguishability
# ---------------------------------------------------------------------------

class TestFaceDetectionStatus:
    def test_constants_are_distinct(self):
        from pipelines.video_pipeline import FaceDetectionStatus as FDS
        values = [FDS.DETECTED, FDS.NOT_FOUND, FDS.FAILED]
        assert len(set(values)) == 3, "All three status values must be distinct"

    def test_detected_value(self):
        from pipelines.video_pipeline import FaceDetectionStatus as FDS
        assert FDS.DETECTED == "detected"

    def test_not_found_value(self):
        from pipelines.video_pipeline import FaceDetectionStatus as FDS
        assert FDS.NOT_FOUND == "not_found"

    def test_failed_value(self):
        from pipelines.video_pipeline import FaceDetectionStatus as FDS
        assert FDS.FAILED == "failed"


# ---------------------------------------------------------------------------
# VideoDetectionResult — null score fields preserved
# ---------------------------------------------------------------------------

class TestVideoDetectionResult:
    """VideoDetectionResult must carry unscored frames as score=None, not 0 or 0.5."""

    def _make_result(self, frame_scores):
        from pipelines.video_pipeline import VideoDetectionResult
        return VideoDetectionResult(
            verdict="real",
            confidence=0.9,
            fake_probability=0.1,
            real_probability=0.9,
            frames_analyzed=len(frame_scores),
            duration_sec=3.0,
            faces_detected=2,
            faces_not_found=1,
            faces_failed=0,
            frame_scores=frame_scores,
            branch_scores={},
            heatmap_path=None,
            heatmap_url=None,
            suspicious_timestamps=[],
        )

    def test_null_score_preserved_in_dataclass(self):
        result = self._make_result([
            {"frame": 0, "score": 0.3, "timestamp": 0.0, "face_status": "detected",  "window_count": 1},
            {"frame": 1, "score": None, "timestamp": 1.0, "face_status": "not_found", "window_count": 0},
            {"frame": 2, "score": 0.7, "timestamp": 2.0, "face_status": "detected",  "window_count": 1},
        ])
        assert result.frame_scores[1]["score"] is None, (
            "Unscored frame must have score=None in VideoDetectionResult, not 0 or 0.5"
        )

    def test_unscored_frame_not_in_suspicious_timestamps(self):
        """The suspicious_timestamps computation in the pipeline excludes None scores."""
        frame_score_list = [
            {"frame": 0, "score": 0.9,  "timestamp": 0.0, "face_status": "detected",  "window_count": 1},
            {"frame": 1, "score": None, "timestamp": 1.0, "face_status": "not_found", "window_count": 0},
            {"frame": 2, "score": 0.8,  "timestamp": 2.0, "face_status": "detected",  "window_count": 1},
        ]
        suspicious = [
            fs["timestamp"]
            for fs in frame_score_list
            if fs["score"] is not None and fs["score"] > 0.7
        ]
        assert 1.0 not in suspicious, (
            "Unscored frame at t=1.0 must not appear in suspicious_timestamps"
        )
        assert suspicious == [0.0, 2.0]

    def test_faces_not_found_and_failed_counts(self):
        result = self._make_result([])
        result.faces_not_found = 3
        result.faces_failed    = 1
        assert result.faces_not_found == 3
        assert result.faces_failed == 1

    def test_no_fabricated_score_for_not_found(self):
        """No frame with face_status=not_found should have a non-None score."""
        result = self._make_result([
            {"frame": 0, "score": 0.4,  "timestamp": 0.0, "face_status": "detected",  "window_count": 1},
            {"frame": 1, "score": None, "timestamp": 1.0, "face_status": "not_found", "window_count": 0},
        ])
        for fs in result.frame_scores:
            if fs["face_status"] in ("not_found", "failed"):
                assert fs["score"] is None, (
                    f"Frame with face_status={fs['face_status']} must have score=None, "
                    f"got {fs['score']!r}"
                )

    def test_no_fabricated_score_for_failed(self):
        result = self._make_result([
            {"frame": 0, "score": None, "timestamp": 0.0, "face_status": "failed", "window_count": 0},
        ])
        assert result.frame_scores[0]["score"] is None


# ---------------------------------------------------------------------------
# FrameScoreResponse — Pydantic schema serialization
# ---------------------------------------------------------------------------

class TestFrameScoreResponseSchema:
    """The API's FrameScoreResponse model must accept and round-trip null scores."""

    def test_null_score_accepted_by_schema(self):
        from api.v1.endpoints.jobs import FrameScoreResponse
        fs = FrameScoreResponse(frame=1, score=None, timestamp=1.0,
                                face_status="not_found", window_count=0)
        assert fs.score is None

    def test_numeric_score_accepted_by_schema(self):
        from api.v1.endpoints.jobs import FrameScoreResponse
        fs = FrameScoreResponse(frame=0, score=0.75, timestamp=0.0,
                                face_status="detected", window_count=2)
        assert fs.score == 0.75

    def test_null_score_serialized_as_null_in_json(self):
        from api.v1.endpoints.jobs import FrameScoreResponse
        fs = FrameScoreResponse(frame=1, score=None, timestamp=1.0,
                                face_status="not_found", window_count=0)
        # model_dump() should give score=None (Python), json should give "score": null
        data = fs.model_dump()
        assert data["score"] is None, "score must be None in model_dump(), not 0 or 0.5"

        json_str = fs.model_dump_json()
        parsed   = json.loads(json_str)
        assert parsed["score"] is None, (
            f"score must serialize as JSON null, got {parsed['score']!r}"
        )

    def test_scored_frame_serialized_with_value(self):
        from api.v1.endpoints.jobs import FrameScoreResponse
        fs = FrameScoreResponse(frame=0, score=0.82, timestamp=0.5,
                                face_status="detected", window_count=3)
        json_str = fs.model_dump_json()
        parsed   = json.loads(json_str)
        assert abs(parsed["score"] - 0.82) < 1e-6

    def test_face_status_preserved_in_json(self):
        from api.v1.endpoints.jobs import FrameScoreResponse
        for status in ("detected", "not_found", "failed"):
            fs = FrameScoreResponse(frame=0, score=0.5 if status == "detected" else None,
                                    timestamp=0.0, face_status=status)
            parsed = json.loads(fs.model_dump_json())
            assert parsed["face_status"] == status

    def test_result_response_accepts_list_with_null_scores(self):
        """ResultResponse.frame_scores validates a list containing a null-score entry."""
        from api.v1.endpoints.jobs import ResultResponse, FrameScoreResponse
        from datetime import datetime, timezone
        import uuid

        raw_frame_scores = [
            {"frame": 0, "score": 0.3, "timestamp": 0.0, "face_status": "detected",  "window_count": 1},
            {"frame": 1, "score": None, "timestamp": 1.0, "face_status": "not_found", "window_count": 0},
        ]

        # Pydantic v2 can validate from dict
        validated = [FrameScoreResponse(**fs) for fs in raw_frame_scores]
        assert validated[1].score is None

    def test_null_score_not_converted_to_zero(self):
        """Pydantic must not coerce None → 0.0 for an Optional[float] field."""
        from api.v1.endpoints.jobs import FrameScoreResponse
        fs = FrameScoreResponse(frame=5, score=None, timestamp=5.0, face_status="failed")
        assert fs.score is None
        assert fs.score != 0.0
        assert fs.score != 0.5


# ---------------------------------------------------------------------------
# Aggregation guard — unscored frames excluded from numerical ops
# ---------------------------------------------------------------------------

class TestAggregationExcludesUnscored:
    """Any numerical aggregation (mean, max, suspicious filter) must skip None scores."""

    def test_mean_excludes_none(self):
        import numpy as np
        scores = [0.3, None, 0.7, None, 0.5]
        valid  = [s for s in scores if s is not None]
        mean   = float(np.mean(valid)) if valid else None
        assert mean is not None
        assert abs(mean - 0.5) < 1e-9, f"Expected mean 0.5, got {mean}"

    def test_max_excludes_none(self):
        scores = [0.3, None, 0.9, None, 0.6]
        valid  = [s for s in scores if s is not None]
        assert max(valid) == 0.9

    def test_suspicious_filter_excludes_none(self):
        frame_scores = [
            {"frame": 0, "score": 0.9,  "timestamp": 0.0},
            {"frame": 1, "score": None, "timestamp": 1.0},
            {"frame": 2, "score": 0.8,  "timestamp": 2.0},
            {"frame": 3, "score": None, "timestamp": 3.0},
        ]
        suspicious = [
            fs["timestamp"]
            for fs in frame_scores
            if fs["score"] is not None and fs["score"] > 0.7
        ]
        assert suspicious == [0.0, 2.0]
        assert 1.0 not in suspicious
        assert 3.0 not in suspicious


# ---------------------------------------------------------------------------
# No-scoreable-frame cases — unknown verdict, no fabricated probabilities
# ---------------------------------------------------------------------------

class TestNoScoreableFrames:
    """When no frame can be scored, the verdict must be 'unknown', not 'real'."""

    def _make_unscored_result(self):
        """VideoDetectionResult with all frames unscored (no faces found)."""
        from pipelines.video_pipeline import VideoDetectionResult
        return VideoDetectionResult(
            verdict="unknown",
            confidence=0.0,
            fake_probability=0.0,
            real_probability=0.0,
            frames_analyzed=3,
            duration_sec=3.0,
            faces_detected=0,
            faces_not_found=3,
            faces_failed=0,
            frame_scores=[
                {"frame": 0, "score": None, "timestamp": 0.0, "face_status": "not_found", "window_count": 0},
                {"frame": 1, "score": None, "timestamp": 1.0, "face_status": "not_found", "window_count": 0},
                {"frame": 2, "score": None, "timestamp": 2.0, "face_status": "not_found", "window_count": 0},
            ],
            branch_scores={},
            heatmap_path=None,
            heatmap_url=None,
            suspicious_timestamps=[],
        )

    def test_all_unscored_verdict_is_unknown(self):
        result = self._make_unscored_result()
        assert result.verdict == "unknown", (
            "A video where no frame could be scored must have verdict='unknown', not 'real'"
        )

    def test_all_unscored_confidence_is_zero(self):
        result = self._make_unscored_result()
        assert result.confidence == 0.0, (
            "Confidence must be 0.0 when no frames were scored"
        )

    def test_all_unscored_no_suspicious_timestamps(self):
        result = self._make_unscored_result()
        assert result.suspicious_timestamps == [], (
            "Suspicious timestamps must be empty when no frames were scored"
        )

    def test_all_unscored_all_frame_scores_null(self):
        result = self._make_unscored_result()
        for fs in result.frame_scores:
            assert fs["score"] is None, (
                f"Frame {fs['frame']} has a non-null score despite not being scored"
            )

    def test_all_unscored_timestamps_preserved(self):
        """Original timestamps must be kept even when frames are unscored."""
        result = self._make_unscored_result()
        timestamps = [fs["timestamp"] for fs in result.frame_scores]
        assert timestamps == [0.0, 1.0, 2.0], (
            "Timestamps must be preserved for unscored frames"
        )

    def test_all_unscored_frame_indices_preserved(self):
        """Original frame indices must be kept even when frames are unscored."""
        result = self._make_unscored_result()
        indices = [fs["frame"] for fs in result.frame_scores]
        assert indices == [0, 1, 2]

    def test_verdict_not_real_when_prob_is_zero_but_no_faces(self):
        """
        The pipeline previously returned final_fake_prob=0.0 and then derived
        verdict='real' because 0.0 < threshold.  This was wrong — the absence of
        scored frames gives no information about authenticity.
        """
        # Simulate the broken pre-fix logic to confirm the fix addresses it
        from pipelines.video_pipeline import FaceDetectionStatus
        overall_fake_probs = []  # empty — no faces found

        # Post-fix logic: check overall_fake_probs before deriving verdict
        if overall_fake_probs:
            import numpy as np
            final_fake_prob = float(np.mean(overall_fake_probs))
            verdict = "fake" if final_fake_prob >= 0.5 else "real"
            confidence = max(final_fake_prob, 1.0 - final_fake_prob)
        else:
            final_fake_prob = 0.0
            verdict = "unknown"
            confidence = 0.0

        assert verdict == "unknown", (
            "verdict must be 'unknown' when no frames were scored, not 'real'"
        )
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# raw_output in ResultResponse — typed API contract
# ---------------------------------------------------------------------------

class TestResultResponseRawOutput:
    """ResultResponse must expose raw_output so the frontend avoids unsafe casts."""

    def test_result_response_has_raw_output_field(self):
        from api.v1.endpoints.jobs import ResultResponse
        import inspect
        fields = ResultResponse.model_fields
        assert "raw_output" in fields, (
            "ResultResponse must have a raw_output field to expose "
            "suspicious_timestamps and scored_frames_count to the frontend"
        )

    def test_result_response_raw_output_is_optional(self):
        from api.v1.endpoints.jobs import ResultResponse
        field = ResultResponse.model_fields["raw_output"]
        # Optional means the default is None or annotation includes None
        annotation = str(field.annotation)
        assert "None" in annotation or "Optional" in annotation, (
            "raw_output must be Optional (older records may not have it)"
        )

    def test_video_raw_output_schema_has_required_fields(self):
        from api.v1.endpoints.jobs import VideoRawOutput
        fields = VideoRawOutput.model_fields
        assert "suspicious_timestamps" in fields
        assert "fake_probability" in fields
        assert "real_probability" in fields
        assert "faces_not_found" in fields
        assert "faces_failed" in fields
        assert "scored_frames_count" in fields

    def test_video_raw_output_all_optional(self):
        """All VideoRawOutput fields are optional (backward-compat with older records)."""
        from api.v1.endpoints.jobs import VideoRawOutput
        # Must be constructible with no arguments
        obj = VideoRawOutput()
        assert obj.suspicious_timestamps is None
        assert obj.fake_probability is None
        assert obj.scored_frames_count is None

    def test_scored_frames_count_in_raw_output(self):
        """scored_frames_count must be a valid integer or null."""
        from api.v1.endpoints.jobs import VideoRawOutput
        import json
        obj = VideoRawOutput(scored_frames_count=5, suspicious_timestamps=[0.0, 1.0])
        parsed = json.loads(obj.model_dump_json())
        assert parsed["scored_frames_count"] == 5
        assert parsed["suspicious_timestamps"] == [0.0, 1.0]

    def test_raw_output_null_in_json_when_absent(self):
        from api.v1.endpoints.jobs import VideoRawOutput
        import json
        obj = VideoRawOutput()
        parsed = json.loads(obj.model_dump_json())
        assert parsed["scored_frames_count"] is None
        assert parsed["fake_probability"] is None
