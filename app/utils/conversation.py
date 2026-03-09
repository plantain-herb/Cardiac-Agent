"""对话记录保存"""

import json
import os
from datetime import datetime
from typing import Dict, List

from app.config import CACHE_CONVERSATIONS_DIR


def save_conversation_json(session_id: str, user_message: str, uploaded_files: List[str],
                           task_type: str, response_data: dict, round_label: str = "auto"):
    """
    将一次对话交互保存为 JSON 文件

    保存路径: cache/conversations/{VERSION}/{session_id}/conv_{timestamp}.json
    """
    try:
        conv_dir = os.path.join(CACHE_CONVERSATIONS_DIR, session_id)
        os.makedirs(conv_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_filename = f"conv_{timestamp}.json"
        conv_path = os.path.join(conv_dir, conv_filename)

        first_response = response_data.get("first_response", None)
        api_name = response_data.get("api_name", None)
        has_two_turns = bool(api_name and api_name != "Agent VQA" and first_response)

        conversation_record = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type or "auto",
            "round_label": round_label,
            "has_two_turns": has_two_turns,
            "user": {
                "message": user_message,
                "uploaded_files": uploaded_files,
            },
            "turn_1": {
                "agent_response": first_response,
                "api_name": api_name,
                "detected_sequences": response_data.get("detected_sequences", []),
            },
            "turn_2": {
                "final_response": response_data.get("response", ""),
                "prediction": response_data.get("prediction", None),
                "metrics": response_data.get("metrics", {}),
                "report_data": response_data.get("report_data", None),
            },
            "download_urls": response_data.get("download_urls", []),
            "frame_urls": response_data.get("frame_urls", []),
            "image_urls": response_data.get("image_urls", []),
            "error": response_data.get("error", None),
        }

        with open(conv_path, "w", encoding="utf-8") as f:
            json.dump(conversation_record, f, ensure_ascii=False, indent=2)

        print(f"对话记录已保存: {conv_path}")
        return conv_path

    except Exception as e:
        print(f"保存对话记录失败: {e}")
        return None
