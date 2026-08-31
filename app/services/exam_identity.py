"""Privacy-preserving DICOM examination identity extraction.

Raw patient identifiers never leave this module.  They are converted to keyed,
process-local digests used only to match two uploads in the same portal session.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime
from typing import Dict, Iterable, Optional

import pydicom


_IDENTITY_SECRET = secrets.token_bytes(32)


def _digest(value: str) -> str:
    return hmac.new(
        _IDENTITY_SECRET,
        value.encode("utf-8", errors="ignore"),
        hashlib.sha256,
    ).hexdigest()


def _clean(value) -> str:
    return " ".join(str(value or "").strip().split())


def _first_dicom_header(path: str):
    if not path or path.lower().endswith((".nii", ".nii.gz")):
        return None
    candidates = []
    if os.path.isfile(path):
        candidates.append(path)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for filename in sorted(files):
                candidates.append(os.path.join(root, filename))

    for candidate in candidates:
        try:
            dataset = pydicom.dcmread(
                candidate,
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "PatientID",
                    "StudyInstanceUID",
                    "StudyDate",
                    "StudyTime",
                    "AccessionNumber",
                    "Manufacturer",
                    "ManufacturerModelName",
                    "InstitutionName",
                ],
            )
            if _clean(getattr(dataset, "PatientID", "")):
                return dataset
        except Exception:
            continue
    return None


def _iso_date(value: str) -> Optional[str]:
    digits = "".join(char for char in _clean(value) if char.isdigit())
    if len(digits) < 8:
        return None
    try:
        return datetime.strptime(digits[:8], "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def extract_exam_identity(volume_paths: Iterable[str]) -> Optional[Dict]:
    """Return a de-identified identity only when one coherent DICOM exam exists."""
    records = []
    for path in volume_paths or []:
        dataset = _first_dicom_header(path)
        if dataset is None:
            continue
        patient_id = _clean(getattr(dataset, "PatientID", "")).casefold()
        study_uid = _clean(getattr(dataset, "StudyInstanceUID", ""))
        study_date = _clean(getattr(dataset, "StudyDate", ""))
        study_time = _clean(getattr(dataset, "StudyTime", ""))
        accession = _clean(getattr(dataset, "AccessionNumber", ""))
        fallback_exam = "|".join((accession, study_date, study_time))
        exam_source = study_uid or fallback_exam.strip("|")
        if patient_id and exam_source:
            records.append({
                "patient": patient_id,
                "exam": exam_source,
                "exam_date": _iso_date(study_date),
                "acquisition": {
                    "manufacturer": _clean(getattr(dataset, "Manufacturer", "")) or None,
                    "model": _clean(getattr(dataset, "ManufacturerModelName", "")) or None,
                    "institution": _clean(getattr(dataset, "InstitutionName", "")) or None,
                },
            })

    if not records:
        return None
    patients = {record["patient"] for record in records}
    exams = {record["exam"] for record in records}
    if len(patients) != 1 or len(exams) != 1:
        return None

    record = records[0]
    return {
        "patient_key": _digest(record["patient"]),
        "exam_key": _digest(record["exam"]),
        "exam_date": record["exam_date"],
        "acquisition": record["acquisition"],
        "identity_source": "dicom_header",
    }
