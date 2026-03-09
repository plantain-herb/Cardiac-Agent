"""会话管理器 - 管理上传文件和中间结果的缓存"""

import os
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
        }

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)

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


_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return _session_manager
