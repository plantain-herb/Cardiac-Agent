"""专家模型 Worker 客户端"""

import os
from typing import Dict, List

from app.config import EXPERT_WORKERS
from app.utils.dicom import encode_image_to_base64, get_slice_num_from_path
from app.utils.http_client import get_http_session


class ExpertWorkerClient:
    """专家模型 Worker 客户端"""

    def __init__(self, workers: Dict[str, str] = None):
        self.workers = workers or EXPERT_WORKERS
        self.session = get_http_session()

    def call_seg(self, worker_name: str, volume_path: str,
                          output_path: str = None, image_output_path: str = None) -> Dict:
        worker_url = self.workers.get(worker_name)
        if not worker_url:
            return {"error": f"未找到Worker: {worker_name}"}

        params = {"image": volume_path}
        if output_path:
            params["output_path"] = output_path
        if image_output_path:
            params["image_output_path"] = image_output_path

        try:
            resp = self.session.post(f"{worker_url}/worker_generate", json=params, timeout=300)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def call_cds(self, worker_name: str, image_2ch: str, image_4ch: str,
                 image_sa: str, **kwargs) -> Dict:
        worker_url = self.workers.get(worker_name)
        if not worker_url:
            return {"error": f"未找到Worker: {worker_name}"}

        params = {
            "image_2ch": image_2ch,
            "image_4ch": image_4ch,
            "image_sa": image_sa,
            "slice_num_2ch": kwargs.get("slice_num_2ch") or get_slice_num_from_path(image_2ch, 1),
            "slice_num_4ch": kwargs.get("slice_num_4ch") or get_slice_num_from_path(image_4ch, 1),
            "slice_num_sa": kwargs.get("slice_num_sa") or get_slice_num_from_path(image_sa, 1),
            "num_crops": 3,
            "skip_preprocess": kwargs.get("skip_preprocess", False),
        }

        for key in ["seg_output_2ch", "seg_output_4ch", "seg_output_sa"]:
            if kwargs.get(key):
                params[key] = kwargs[key]

        try:
            resp = self.session.post(f"{worker_url}/worker_generate", json=params, timeout=600)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def call_nicms(self, worker_name: str, image_4ch: str, image_sa: str,
                   image_lge_sa: str, **kwargs) -> Dict:
        worker_url = self.workers.get(worker_name)
        if not worker_url:
            return {"error": f"未找到Worker: {worker_name}"}

        params = {
            "image_4ch": image_4ch,
            "image_sa": image_sa,
            "image_lge_sa": image_lge_sa,
            "slice_num_4ch": kwargs.get("slice_num_4ch") or get_slice_num_from_path(image_4ch, 1),
            "slice_num_sa": kwargs.get("slice_num_sa") or get_slice_num_from_path(image_sa, 1),
            "slice_num_lge_sa": kwargs.get("slice_num_lge_sa") or get_slice_num_from_path(image_lge_sa, 1),
            "num_crops": 3,
            "skip_preprocess": kwargs.get("skip_preprocess", False),
        }

        try:
            resp = self.session.post(f"{worker_url}/worker_generate", json=params, timeout=600)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def call_metrics(self, worker_name: str, image_4ch: str, image_sa: str, **kwargs) -> Dict:
        """调用 metrics_worker（纯指标计算: 分割 + 心脏功能指标）"""
        worker_url = self.workers.get(worker_name)
        if not worker_url:
            return {"error": f"未找到Worker: {worker_name}"}

        params = {
            "image_4ch": image_4ch,
            "image_sa": image_sa,
            "slice_num_4ch": kwargs.get("slice_num_4ch") or get_slice_num_from_path(image_4ch, 1),
            "slice_num_sa": kwargs.get("slice_num_sa") or get_slice_num_from_path(image_sa, 1),
        }

        try:
            resp = self.session.post(f"{worker_url}/worker_generate", json=params, timeout=600)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def call_mrg(self, worker_name: str, image_4ch: str, image_sa: str,
                 image_2ch: str = None, image_lge_sa: str = None, **kwargs) -> Dict:
        """调用新 MRG Worker（编排: metrics + CDS + NICMS）"""
        worker_url = self.workers.get(worker_name)
        if not worker_url:
            return {"error": f"未找到Worker: {worker_name}"}

        params = {
            "image_4ch": image_4ch,
            "image_sa": image_sa,
            "slice_num_4ch": kwargs.get("slice_num_4ch") or get_slice_num_from_path(image_4ch, 1),
            "slice_num_sa": kwargs.get("slice_num_sa") or get_slice_num_from_path(image_sa, 1),
            "skip_preprocess": kwargs.get("skip_preprocess", False),
        }

        if image_2ch:
            params["image_2ch"] = image_2ch
            params["slice_num_2ch"] = kwargs.get("slice_num_2ch") or get_slice_num_from_path(image_2ch, 1)
        if image_lge_sa:
            params["image_lge_sa"] = image_lge_sa
            params["slice_num_lge_sa"] = kwargs.get("slice_num_lge_sa") or get_slice_num_from_path(image_lge_sa, 1)

        try:
            resp = self.session.post(f"{worker_url}/worker_generate", json=params, timeout=1200)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def call_mir(self, worker_name: str, prompt: str, **kwargs) -> Dict:
        worker_url = self.workers.get(worker_name)
        if not worker_url:
            return {"error": f"未找到Worker: {worker_name}"}

        params = {"prompt": prompt, "engine": kwargs.get("engine", "gpt-4o")}
        if kwargs.get("api_key"):
            params["api_key"] = kwargs["api_key"]
        if kwargs.get("base_url"):
            params["base_url"] = kwargs["base_url"]

        try:
            resp = self.session.post(f"{worker_url}/worker_generate", json=params, timeout=120)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def call_seq(self, worker_name: str, prompt: str, images: List[str], **kwargs) -> Dict:
        worker_url = self.workers.get(worker_name)
        if not worker_url:
            return {"error": f"未找到Worker: {worker_name}"}

        encoded_images = [encode_image_to_base64(img) for img in images if os.path.exists(img)]

        params = {"prompt": prompt, "images": encoded_images, "engine": kwargs.get("engine", "gpt-4o")}
        if kwargs.get("api_key"):
            params["api_key"] = kwargs["api_key"]
        if kwargs.get("base_url"):
            params["base_url"] = kwargs["base_url"]

        try:
            resp = self.session.post(f"{worker_url}/worker_generate", json=params, timeout=120)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
