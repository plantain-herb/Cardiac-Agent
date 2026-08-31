from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset

from app.services.exam_identity import extract_exam_identity
from app.services.longitudinal import (
    SKILL_VERSION,
    THRESHOLDS_STATUS,
    compare_exams,
    normalize_metrics,
)
from app.services.session_manager import SessionManager


def _write_dicom(directory: Path, patient_id: str, study_uid: str, study_date: str):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "image.dcm"
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.PatientID = patient_id
    dataset.StudyInstanceUID = study_uid
    dataset.StudyDate = study_date
    dataset.StudyTime = "103000"
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.save_as(path, enforce_file_format=True)
    return path


def test_exam_identity_matches_patient_but_distinguishes_study(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dicom(first, "PATIENT-001", pydicom.uid.generate_uid(), "20250101")
    _write_dicom(second, "PATIENT-001", pydicom.uid.generate_uid(), "20250801")

    first_identity = extract_exam_identity([str(first)])
    second_identity = extract_exam_identity([str(second)])

    assert first_identity["patient_key"] == second_identity["patient_key"]
    assert first_identity["exam_key"] != second_identity["exam_key"]
    assert first_identity["exam_date"] == "2025-01-01"
    assert "PATIENT-001" not in repr(first_identity)


def test_incoherent_multi_patient_upload_is_not_matched(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    study_uid = pydicom.uid.generate_uid()
    _write_dicom(first, "PATIENT-001", study_uid, "20250101")
    _write_dicom(second, "PATIENT-002", study_uid, "20250101")
    assert extract_exam_identity([str(first), str(second)]) is None


def test_session_offers_only_same_patient_different_exam():
    manager = SessionManager()
    manager.sessions["session"] = {"exam_history": []}
    metrics = normalize_metrics({"LV_EF": 60, "LV_EDV": 120})
    first_identity = {
        "patient_key": "patient-a", "exam_key": "study-1",
        "exam_date": "2025-01-01", "acquisition": {},
    }
    same_exam = dict(first_identity)
    second_identity = dict(first_identity, exam_key="study-2", exam_date="2025-08-01")
    other_patient = dict(first_identity, patient_key="patient-b", exam_key="study-3")

    assert manager.register_exam("session", first_identity, metrics, ["mr_sa"]) is None
    assert manager.register_exam("session", same_exam, metrics, ["mr_sa"]) is None
    assert manager.register_exam("session", other_patient, metrics, ["mr_sa"]) is None
    offer = manager.register_exam("session", second_identity, metrics, ["mr_sa"])

    assert offer["enabled"] is True
    assert "Would you like" in offer["message"]
    retry_offer = manager.register_exam(
        "session", second_identity, metrics, ["mr_sa"]
    )
    assert retry_offer == offer
    pair = manager.get_exam_pair(
        "session", offer["prior_exam_id"], offer["current_exam_id"]
    )
    assert pair[0]["exam_key"] == "study-1"
    assert pair[1]["exam_key"] == "study-2"


def test_comparison_preserves_unknown_and_labels_research_thresholds():
    baseline = {
        "exam_id": "one", "exam_date": "2025-01-01",
        "metrics": normalize_metrics({"LV_EF": 60, "LV_EDV": 100}),
        "available_sequences": ["mr_4ch", "mr_sa"],
    }
    followup = {
        "exam_id": "two", "exam_date": "2025-08-01",
        "metrics": normalize_metrics({"LV_EF": 52, "LV_EDV": 104}),
        "available_sequences": ["mr_4ch", "mr_sa"],
    }
    result = compare_exams(baseline, followup)
    rows = {row["metric"]: row for row in result["rows"]}

    assert result["skill_version"] == SKILL_VERSION
    assert result["thresholds_status"] == THRESHOLDS_STATUS
    assert rows["LVEF"]["absolute_delta"] == -8
    assert rows["LVEF"]["research_flag"] == "change_flag"
    assert rows["LVEDV"]["research_flag"] == "within_placeholder_threshold"
    assert rows["LVESV"]["research_flag"] == "unknown"
    assert "not clinically approved" in result["disclaimer"]


def test_comparison_includes_extended_measurements_and_wall_motion_evidence():
    baseline = {
        "exam_id": "one", "exam_date": "2025-01-01",
        "metrics": normalize_metrics({
            "LV_EF": 60, "RV_EF": 52, "LA_LD": 36,
            "LGE_SA_Label3_Mass": 8, "LV_BS_01_mean": 9,
        }),
        "available_sequences": ["mr_4ch", "mr_sa", "mr_lge"],
        "findings": {"wall_motion": {
            "probability": 0.55, "positive": False, "threshold": 0.8,
            "validation_auroc": 0.8182749326145553, "model_version": "wall-v1",
        }},
    }
    followup = {
        "exam_id": "two", "exam_date": "2025-08-01",
        "metrics": normalize_metrics({
            "LV_EF": 55, "RV_EF": 49, "LA_LD": 39,
            "LGE_SA_Label3_Mass": 12, "LV_BS_01_mean": 10,
        }),
        "available_sequences": ["mr_4ch", "mr_sa", "mr_lge"],
        "findings": {"wall_motion": {
            "probability": 0.86, "positive": True, "threshold": 0.8,
            "validation_auroc": 0.8182749326145553, "model_version": "wall-v1",
        }},
    }

    result = compare_exams(baseline, followup)
    rows = {row["metric"]: row for row in result["rows"]}
    assert rows["RVEF"]["absolute_delta"] == -3
    assert rows["LA_LD"]["absolute_delta"] == 3
    assert rows["LGE_MASS"]["absolute_delta"] == 4
    assert rows["LVWT_01"]["absolute_delta"] == 1
    assert rows["RVEF"]["research_flag"] == "descriptive_delta"
    wall = result["finding_rows"][0]
    assert round(wall["probability_delta"], 2) == 0.31
    assert wall["binary_transition"] == "negative → positive"
    assert wall["validation_auroc"] == 0.8182749326145553
