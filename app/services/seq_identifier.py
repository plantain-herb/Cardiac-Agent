"""并行序列识别器"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from app.config import MAX_WORKERS, SEQUENCE_TO_MODALITY
from app.services.expert_client import ExpertWorkerClient
from app.services.session_manager import get_session_manager
from app.utils.dicom import cleanup_temp_files, extract_frames_from_volume


class ParallelSequenceIdentifier:
    """并行序列识别器"""

    def __init__(self, expert_client: ExpertWorkerClient, max_workers: int = MAX_WORKERS):
        self.expert_client = expert_client
        self.max_workers = max_workers

    def identify_single(self, volume_path: str, api_key: str = None,
                       engine: str = "gpt-4o", base_url: str = None,
                       session_id: str = None) -> Dict:
        """识别单个 volume 的序列类型"""
        if not os.path.exists(volume_path):
            return {"path": volume_path, "error": "文件不存在", "modality": None}

        try:
            temp_images, frame_info = extract_frames_from_volume(
                volume_path, num_frames=3,
                session_id=session_id, save_to_cache=(session_id is not None)
            )

            if session_id:
                session_mgr = get_session_manager()
                session_mgr.add_frames(session_id, frame_info)

            prompt = "Which sequence does this cardiac MRI belong to?"

            result = self.expert_client.call_seq(
                "SeqAnalysis",
                prompt,
                temp_images,
                api_key=api_key,
                engine=engine,
                base_url=base_url,
            )

            if not session_id:
                cleanup_temp_files(temp_images)

            detected_sequences = result.get("detected_sequences", [])
            if detected_sequences:
                seq = detected_sequences[0].lower().strip()
                modality = SEQUENCE_TO_MODALITY.get(seq)

                if modality is None:
                    if "2ch" in seq or "2-chamber" in seq or "two chamber" in seq:
                        modality = "2ch"
                    elif "4ch" in seq or "4-chamber" in seq or "four chamber" in seq:
                        modality = "4ch"
                    elif "lge" in seq:
                        modality = "lge"
                    elif "sa" in seq or "short" in seq:
                        modality = "sa"
                    else:
                        modality = seq.split()[-1] if " " in seq else seq
            else:
                modality = None

            return {
                "path": volume_path,
                "detected_sequences": detected_sequences,
                "modality": modality,
                "error": None,
                "frame_info": frame_info if session_id else None,
            }

        except Exception as e:
            return {"path": volume_path, "error": str(e), "modality": None}

    def identify_parallel(self, volume_paths: List[str], api_key: str = None,
                         engine: str = "gpt-4o", base_url: str = None,
                         session_id: str = None) -> Dict[str, Dict]:
        """并行识别多个 volume 的序列类型"""
        results = {}

        print(f"\n[并行序列识别] 开始识别 {len(volume_paths)} 个文件...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(
                    self.identify_single, path, api_key, engine, base_url, session_id
                ): path for path in volume_paths
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results[path] = result
                    print(f"  ✓ {os.path.basename(path)}: {result.get('modality', 'unknown')}")
                except Exception as e:
                    results[path] = {"path": path, "error": str(e), "modality": None}
                    print(f"  ✗ {os.path.basename(path)}: {e}")

        return results

    def match_modalities(self, identification_results: Dict[str, Dict],
                        required_modalities: List[str]) -> Dict[str, str]:
        """根据识别结果匹配所需的模态"""
        matched = {}

        for modality in required_modalities:
            for path, result in identification_results.items():
                if result.get("modality") == modality and path not in matched.values():
                    matched[modality] = path
                    break

        return matched
