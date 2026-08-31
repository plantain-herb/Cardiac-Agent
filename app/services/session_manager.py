"""会话管理器 - 管理上传文件和中间结果的缓存"""

import os
import re
import shutil
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.config import (
    CACHE_FRAMES_DIR,
    CACHE_IMAGES_DIR,
    CACHE_UPLOADS_DIR,
    VERSION,
)


class SessionManager:
    """会话管理器"""

    def __init__(self):
        self.sessions = {}

    def create_session(self) -> str:
        now = datetime.now()
        datetime_prefix = now.strftime("%Y%m%d_%H%M%S")
        uuid_suffix = str(uuid.uuid4())[:8]
        session_id = f"{datetime_prefix}_{uuid_suffix}"

        session_upload_dir = os.path.join(CACHE_UPLOADS_DIR, session_id)
        session_frames_dir = os.path.join(CACHE_FRAMES_DIR, session_id)
        session_images_dir = os.path.join(CACHE_IMAGES_DIR, session_id)
        os.makedirs(session_upload_dir, exist_ok=True)
        os.makedirs(session_frames_dir, exist_ok=True)
        os.makedirs(session_images_dir, exist_ok=True)

        self.sessions[session_id] = {
            "id": session_id,
            "created_at": time.time(),
            "datetime": now.isoformat(),
            "upload_dir": session_upload_dir,
            "frames_dir": session_frames_dir,
            "images_dir": session_images_dir,
            "files": [],
            "frames": [],
            "images": [],
            "metric_correction_context": None,
            "exam_history": [],
        }

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        session = self.sessions.get(session_id)
        if session or not session_id:
            return session

        # Preserve a browser session across a backend restart.  Only restore a
        # known on-disk cache directory and never accept arbitrary path text.
        if not re.fullmatch(r"\d{8}_\d{6}_[0-9a-fA-F]{8}", session_id):
            return None
        session_upload_dir = os.path.join(CACHE_UPLOADS_DIR, session_id)
        if not os.path.isdir(session_upload_dir):
            return None
        session_frames_dir = os.path.join(CACHE_FRAMES_DIR, session_id)
        session_images_dir = os.path.join(CACHE_IMAGES_DIR, session_id)
        os.makedirs(session_frames_dir, exist_ok=True)
        os.makedirs(session_images_dir, exist_ok=True)
        restored = {
            "id": session_id,
            "created_at": os.path.getmtime(session_upload_dir),
            "datetime": datetime.fromtimestamp(
                os.path.getmtime(session_upload_dir)
            ).isoformat(),
            "upload_dir": session_upload_dir,
            "frames_dir": session_frames_dir,
            "images_dir": session_images_dir,
            "files": [],
            "frames": [],
            "images": [],
            "metric_correction_context": None,
            "exam_history": [],
        }
        self.sessions[session_id] = restored
        return restored

    def add_file(self, session_id: str, file_path: str, original_name: str, modality: str = None):
        if session_id in self.sessions:
            self.sessions[session_id]["files"].append({
                "path": file_path,
                "original_name": original_name,
                "modality": modality,
                "added_at": time.time(),
            })

    def add_frames(self, session_id: str, frame_info: Dict):
        if session_id in self.sessions:
            self.sessions[session_id]["frames"].append(frame_info)

    def add_image(self, session_id: str, image_path: str, original_name: str):
        if session_id in self.sessions:
            self.sessions[session_id]["images"].append({
                "path": image_path,
                "original_name": original_name,
                "added_at": time.time(),
                "url": f"/cache/images/{VERSION}/{session_id}/{os.path.basename(image_path)}",
            })

    def list_sessions(self) -> List[Dict]:
        result = []
        for session_id, info in self.sessions.items():
            result.append({
                "id": session_id,
                "created_at": info["created_at"],
                "file_count": len(info["files"]),
                "frame_count": len(info["frames"]),
            })
        return sorted(result, key=lambda x: x["created_at"], reverse=True)

    def cleanup_session(self, session_id: str):
        if session_id in self.sessions:
            info = self.sessions[session_id]
            if os.path.exists(info["upload_dir"]):
                shutil.rmtree(info["upload_dir"], ignore_errors=True)
            if os.path.exists(info["frames_dir"]):
                shutil.rmtree(info["frames_dir"], ignore_errors=True)
            if "images_dir" in info and os.path.exists(info["images_dir"]):
                shutil.rmtree(info["images_dir"], ignore_errors=True)
            del self.sessions[session_id]

    def cleanup_all(self):
        for session_id in list(self.sessions.keys()):
            self.cleanup_session(session_id)
        for dir_path in [CACHE_UPLOADS_DIR, CACHE_FRAMES_DIR, CACHE_IMAGES_DIR]:
            if os.path.exists(dir_path):
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)

    def get_session_files(self, session_id: str) -> List[Dict]:
        if session_id in self.sessions:
            return self.sessions[session_id]["files"]
        return []

    def get_original_name(self, session_id: str, file_path: str) -> str:
        if session_id in self.sessions:
            for f in self.sessions[session_id]["files"]:
                if f["path"] == file_path:
                    return f.get("original_name", "")
        return ""

    def get_session_frames(self, session_id: str) -> List[Dict]:
        if session_id in self.sessions:
            return self.sessions[session_id]["frames"]
        return []

    def get_session_images(self, session_id: str) -> List[Dict]:
        if session_id in self.sessions:
            return self.sessions[session_id].get("images", [])
        return []

    def set_metric_correction_context(self, session_id: str, context: Dict):
        """保存本会话最近一次可用于人工修正后重算的 mask 契约。"""
        if session_id in self.sessions:
            self.sessions[session_id]["metric_correction_context"] = context

    def get_metric_correction_context(self, session_id: str) -> Optional[Dict]:
        if session_id in self.sessions:
            return self.sessions[session_id].get("metric_correction_context")
        return None

    @staticmethod
    def _longitudinal_offer(prior: Dict, current: Dict) -> Dict:
        """Return a stable, retryable offer for one distinct same-patient pair."""
        return {
            "enabled": True,
            "offer_id": f"{prior['exam_id']}:{current['exam_id']}",
            "endpoint": "/api/longitudinal/compare",
            "prior_exam_id": prior["exam_id"],
            "current_exam_id": current["exam_id"],
            "message": (
                "A previous examination for the same patient was found. "
                "Would you like to perform a longitudinal follow-up comparison?"
            ),
        }

    def register_exam(self, session_id: str, identity: Dict, metrics: Dict,
                      detected_sequences=None, findings: Dict = None) -> Optional[Dict]:
        """Store one de-identified ExamCard and offer a distinct matching prior."""
        session = self.sessions.get(session_id)
        if not session or not identity or not metrics:
            return None
        history = session.setdefault("exam_history", [])
        for existing in history:
            if existing["exam_key"] == identity["exam_key"]:
                existing["metrics"] = metrics
                existing["available_sequences"] = list(detected_sequences or [])
                existing["findings"] = findings or existing.get("findings") or {}
                prior = next(
                    (item for item in reversed(history)
                     if item["patient_key"] == existing["patient_key"]
                     and item["exam_key"] != existing["exam_key"]),
                    None,
                )
                if prior:
                    return self._longitudinal_offer(prior, existing)
                return None

        exam = {
            "exam_id": uuid.uuid4().hex[:12],
            "patient_key": identity["patient_key"],
            "exam_key": identity["exam_key"],
            "exam_date": identity.get("exam_date"),
            "acquisition": identity.get("acquisition") or {},
            "metrics": metrics,
            "available_sequences": list(detected_sequences or []),
            "findings": findings or {},
            "created_at": time.time(),
        }
        prior = next(
            (item for item in reversed(history)
             if item["patient_key"] == exam["patient_key"]
             and item["exam_key"] != exam["exam_key"]),
            None,
        )
        history.append(exam)
        if not prior:
            return None
        return self._longitudinal_offer(prior, exam)

    def get_exam_pair(self, session_id: str, prior_exam_id: str,
                      current_exam_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return None
        by_id = {
            item["exam_id"]: item
            for item in session.get("exam_history", [])
        }
        prior = by_id.get(prior_exam_id)
        current = by_id.get(current_exam_id)
        if (not prior or not current
                or prior["patient_key"] != current["patient_key"]
                or prior["exam_key"] == current["exam_key"]):
            return None
        return prior, current


_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return _session_manager
