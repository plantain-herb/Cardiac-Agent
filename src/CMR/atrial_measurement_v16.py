"""Compatibility fallback for deployments where the v16 atrial module is absent.

The metric pipeline can still produce segmentation and ventricular metrics; atrial
AP diameters are reported as unavailable instead of preventing the MRG worker from
starting.
"""

def measure_atrial_ap_diameters(mask_path, slice_num):
    return {
        "LA_AP_Diameter_mm": None,
        "RA_AP_Diameter_mm": None,
        "method": "unavailable_missing_atrial_measurement_v16",
        "LA_selected_phase": None,
        "LA_selected_slice_index": None,
        "RA_selected_phase": None,
        "RA_selected_slice_index": None,
        "LA_qc": {"status": "unavailable", "reason": "module_missing"},
        "RA_qc": {"status": "unavailable", "reason": "module_missing"},
    }
