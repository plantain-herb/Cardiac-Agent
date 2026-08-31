"""Focused adapter for the previously developed Commander/CMLA comparator.

The portal is deployed independently from Commander, so this module carries the
same v0.2.0 deterministic delta contract for the six core CMR metrics.
Threshold flags remain explicitly research-only.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, Mapping, Tuple


SKILL_VERSION = "cmla-skills-v0.2.0"
THRESHOLD_VERSION = "cmla-research-thresholds-v0.1"
THRESHOLDS_STATUS = "research_placeholder_not_clinically_approved"
CORE_METRICS = ("LVEF", "LVEDV", "LVESV", "LVSV", "LVCO", "LVMASS")
METRIC_REGISTRY = {
    "LVEF": ("LV_EF", "%", "LV Function", "LV Ejection Fraction"),
    "LVEDV": ("LV_EDV", "mL", "LV Function", "LV End-Diastolic Volume"),
    "LVESV": ("LV_ESV", "mL", "LV Function", "LV End-Systolic Volume"),
    "LVSV": ("LV_SV", "mL", "LV Function", "LV Stroke Volume"),
    "LVCO": ("LV_CO", "L/min", "LV Function", "LV Cardiac Output"),
    "LVMASS": ("LV_Mass", "g", "LV Function", "LV Myocardial Mass"),
    "RVEF": ("RV_EF", "%", "RV Function", "RV Ejection Fraction"),
    "RVEDV": ("RV_EDV", "mL", "RV Function", "RV End-Diastolic Volume"),
    "RVESV": ("RV_ESV", "mL", "RV Function", "RV End-Systolic Volume"),
    "RVSV": ("RV_SV", "mL", "RV Function", "RV Stroke Volume"),
    "RVCO": ("RV_CO", "L/min", "RV Function", "RV Cardiac Output"),
    "LA_LD": ("LA_LD", "mm", "Chamber Dimensions", "LA Long Diameter"),
    "RA_LD": ("RA_LD", "mm", "Chamber Dimensions", "RA Long Diameter"),
    "LV_LD": ("LV_LD", "mm", "Chamber Dimensions", "LV Long Diameter"),
    "RV_LD": ("RV_LD", "mm", "Chamber Dimensions", "RV Long Diameter"),
    "LGE_MASS": (
        "LGE_SA_Label3_Mass", "g", "Tissue Characterization", "LGE Scar Mass"
    ),
    **{
        f"LVWT_{segment:02d}": (
            (
                f"LV_BS_{segment:02d}_mean" if segment <= 6
                else f"LV_IP_{segment:02d}_mean" if segment <= 12
                else f"LV_SP_{segment:02d}_mean" if segment <= 16
                else "LV_TP_17_mean"
            ),
            "mm", "LV Wall Thickness (17 Segments)",
            f"LV Segment {segment:02d} Mean Thickness",
        )
        for segment in range(1, 18)
    },
    "RVWT_01": ("RV_BS_01", "mm", "RV Wall Thickness", "RV Basal Thickness"),
    "RVWT_02": ("RV_IP_02", "mm", "RV Wall Thickness", "RV Mid Thickness"),
    "RVWT_03": ("RV_SP_03", "mm", "RV Wall Thickness", "RV Apical Thickness"),
}
RESEARCH_THRESHOLDS = {
    "LVEF": {"absolute_delta": 5.0, "relative_delta": 0.10},
    "LVEDV": {"absolute_delta": 20.0, "relative_delta": 0.10},
    "LVESV": {"absolute_delta": 15.0, "relative_delta": 0.10},
    "LVSV": {"absolute_delta": 10.0, "relative_delta": 0.10},
    "LVCO": {"absolute_delta": 0.5, "relative_delta": 0.10},
    "LVMASS": {"absolute_delta": 15.0, "relative_delta": 0.10},
}


def _number(value: Any):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_metrics(metrics: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        code: {
            "value": _number(metrics.get(source_key)),
            "unit": unit,
            "category": category,
            "display_name": display_name,
        }
        for code, (source_key, unit, category, display_name) in METRIC_REGISTRY.items()
    }


def normalize_sequences(sequences) -> list[str]:
    aliases = {
        "cine 2ch": "mr_2ch", "cine_2ch": "mr_2ch",
        "cine 4ch": "mr_4ch", "cine_4ch": "mr_4ch",
        "cine sa": "mr_sa", "cine_sa": "mr_sa",
        "lge sa": "mr_lge", "lge_sa": "mr_lge",
    }
    normalized = set()
    for item in sequences or []:
        text = item if isinstance(item, str) else item.get("modality", "")
        key = str(text).strip().lower().replace("-", " ")
        mapped = aliases.get(key)
        if mapped:
            normalized.add(mapped)
    return sorted(normalized)


def _as_date(value: Any):
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _chronological(first: Dict, second: Dict) -> Tuple[Dict, Dict]:
    first_date, second_date = _as_date(first.get("exam_date")), _as_date(second.get("exam_date"))
    if first_date and second_date and second_date < first_date:
        return second, first
    return first, second


def compare_exams(first_exam: Dict, second_exam: Dict) -> Dict:
    """Build a display-safe longitudinal comparison without clinical inference."""
    baseline, followup = _chronological(first_exam, second_exam)
    rows = []
    unknown = []
    flagged = []
    for code, (_, default_unit, default_category, default_name) in METRIC_REGISTRY.items():
        baseline_item = (baseline.get("metrics") or {}).get(code) or {}
        followup_item = (followup.get("metrics") or {}).get(code) or {}
        prior = _number(baseline_item.get("value"))
        current = _number(followup_item.get("value"))
        unit = followup_item.get("unit") or baseline_item.get("unit") or default_unit
        absolute = current - prior if current is not None and prior is not None else None
        relative = absolute / abs(prior) if absolute is not None and prior else None
        limits = RESEARCH_THRESHOLDS.get(code)
        exceeds = bool(limits and absolute is not None and (
            abs(absolute) >= limits["absolute_delta"]
            or (relative is not None and abs(relative) >= limits["relative_delta"])
        ))
        if absolute is None:
            research_flag = "unknown"
            unknown.append(code)
        elif exceeds:
            research_flag = "change_flag"
            flagged.append(code)
        elif limits is None:
            research_flag = "descriptive_delta"
        else:
            research_flag = "within_placeholder_threshold"
        rows.append({
            "metric": code,
            "display_name": followup_item.get("display_name") or baseline_item.get("display_name") or default_name,
            "category": followup_item.get("category") or baseline_item.get("category") or default_category,
            "unit": unit,
            "baseline": prior,
            "followup": current,
            "absolute_delta": absolute,
            "relative_delta": relative,
            "research_flag": research_flag,
        })

    baseline_date, followup_date = _as_date(baseline.get("exam_date")), _as_date(followup.get("exam_date"))
    interval_days = (followup_date - baseline_date).days if baseline_date and followup_date else None
    baseline_sequences = set(baseline.get("available_sequences") or [])
    followup_sequences = set(followup.get("available_sequences") or [])
    union = baseline_sequences | followup_sequences
    overlap = sorted(baseline_sequences & followup_sequences)
    sequence_overlap = len(overlap) / len(union) if union else None
    comparability_reasons = []
    if sequence_overlap is None or sequence_overlap < 0.5:
        comparability_reasons.append("limited_sequence_overlap")
    if interval_days is None:
        comparability_reasons.append("unknown_interval")
    elif interval_days <= 7:
        comparability_reasons.append("same_episode_risk")

    finding_rows = []
    prior_wall = (baseline.get("findings") or {}).get("wall_motion") or {}
    current_wall = (followup.get("findings") or {}).get("wall_motion") or {}
    prior_probability = _number(prior_wall.get("probability"))
    current_probability = _number(current_wall.get("probability"))
    if prior_probability is not None or current_probability is not None:
        probability_delta = (
            current_probability - prior_probability
            if prior_probability is not None and current_probability is not None else None
        )
        transition = "unknown"
        if "positive" in prior_wall and "positive" in current_wall:
            transition = (
                f"{'positive' if prior_wall['positive'] else 'negative'} → "
                f"{'positive' if current_wall['positive'] else 'negative'}"
            )
        finding_rows.append({
            "finding": "wall_motion_abnormality",
            "display_name": "Wall-motion Abnormality Probability",
            "baseline_probability": prior_probability,
            "followup_probability": current_probability,
            "probability_delta": probability_delta,
            "binary_transition": transition,
            "threshold": current_wall.get("threshold") or prior_wall.get("threshold"),
            "model_version": current_wall.get("model_version") or prior_wall.get("model_version"),
            "validation_auroc": current_wall.get("validation_auroc") or prior_wall.get("validation_auroc"),
            "reliability": "research_model_evidence_not_calibrated_burden",
        })

    return {
        "skill_version": SKILL_VERSION,
        "threshold_version": THRESHOLD_VERSION,
        "thresholds_status": THRESHOLDS_STATUS,
        "baseline": {"exam_id": baseline["exam_id"], "exam_date": baseline.get("exam_date")},
        "followup": {"exam_id": followup["exam_id"], "exam_date": followup.get("exam_date")},
        "interval_days": interval_days,
        "comparability": {
            "comparable": not comparability_reasons,
            "reasons": comparability_reasons,
            "sequence_overlap": sequence_overlap,
            "overlapping_sequences": overlap,
        },
        "rows": rows,
        "finding_rows": finding_rows,
        "flagged_metrics": flagged,
        "unknown_metrics": unknown,
        "summary": (
            f"{len(flagged)} metric(s) crossed research-placeholder change flags; "
            f"{len(unknown)} metric(s) were unavailable."
        ),
        "disclaimer": (
            "Research comparison only. Placeholder flags are not clinically approved "
            "thresholds and do not constitute diagnosis or treatment advice."
        ),
    }
