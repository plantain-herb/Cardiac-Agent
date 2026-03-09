"""
医学报告生成 Worker (Medical Report Generation Worker)

功能: 编排调用 metrics_worker、cds_worker、nicms_worker，生成综合医学报告。

输入模态:
    - cine 4ch (必须)
    - cine sa  (必须)
    - cine 2ch (可选，增强 CDS 分类)
    - lge sa   (可选，当 CDS 判定为非缺血时进一步调用 NICMS)

Pipeline:
    1. 调用 metrics_worker → 心脏功能指标 + 可选 LGE SA mass
    2. 调用 cds_worker    → 疾病筛查 (Normal / Ischemic / Non-ischemic)
    3. 若 CDS 结果为 Non-ischemic 且提供了 LGE SA，调用 nicms_worker → 非缺血性心肌病亚分类
       (未提供 LGE SA 时即使 CDS=Non-ischemic 也跳过 NICMS)

Usage:
    python -m serve.mrg_worker --port 21030 --no-register
"""

import argparse
import os
import sys
import threading
import time
import traceback
import uuid
from typing import Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVE_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    from serve.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
    from serve.utils import build_logger, pretty_print_semaphore
except ImportError:
    WORKER_HEART_BEAT_INTERVAL = 45
    SERVER_ERROR_MSG = "**SERVER ERROR. PLEASE TRY AGAIN.**"

    class ErrorCode:
        INTERNAL_ERROR = 50001
        CUDA_OUT_OF_MEMORY = 50002
        SEGMENTATION_FAILED = 50003
        METRICS_CALCULATION_FAILED = 50004

    import logging

    def build_logger(name, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(filename),
                logging.StreamHandler(),
            ],
        )
        return logging.getLogger(name)

    def pretty_print_semaphore(sem):
        if sem is None:
            return "None"
        return f"Semaphore(value={sem._value})"


worker_id = str(uuid.uuid4())[:6]
logger = build_logger("mrg_worker", os.path.join("workers", "mrg.log"))

global_counter = 0
model_semaphore = None

# 下游 Worker 地址
DOWNSTREAM_WORKERS = {
    "metrics": "http://localhost:21031",
    "cds": "http://localhost:21020",
    "nicms": "http://localhost:21021",
}

CC_CLASSES = {
    0: "Normal",
    1: "Ischemic Cardiomyopathy",
    2: "Non-ischemic Cardiomyopathy",
}

NCC_CLASSES = {
    0: "Hypertrophic Cardiomyopathy",
    1: "Dilated Cardiomyopathy",
    2: "Inflammatory Cardiomyopathy",
    3: "Restrictive Cardiomyopathy",
    4: "Arrhythmogenic Cardiomyopathy",
}


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class MRGOrchestratorWorker:
    """医学报告生成编排 Worker"""

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        no_register: bool,
        model_names: list,
        downstream_workers: dict = None,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        self.downstream = downstream_workers or DOWNSTREAM_WORKERS
        logger.info(f"初始化 MRG Orchestrator Worker, ID: {worker_id}")
        logger.info(f"下游服务: {self.downstream}")

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.daemon = True
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")
        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        try:
            r = requests.post(url, json=data, timeout=10)
            assert r.status_code == 200
        except Exception as e:
            logger.error(f"Failed to register to controller: {e}")

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}. "
            f"worker_id: {self.worker_id}. "
        )
        url = self.controller_addr + "/receive_heart_beat"
        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)
        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        return args.limit_model_concurrency - model_semaphore._value + (
            len(model_semaphore._waiters)
            if model_semaphore._waiters is not None
            else 0
        )

    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    # ---- 下游调用 ----

    def _call_metrics(
        self,
        image_4ch: str,
        image_sa: str,
        image_lge_sa: Optional[str] = None,
        slice_num_4ch: Optional[int] = None,
        slice_num_sa: Optional[int] = None,
    ) -> Dict:
        """调用 metrics_worker 计算心脏功能指标（含可选 LGE SA mass）"""
        url = self.downstream["metrics"]
        params = {
            "image_4ch": image_4ch,
            "image_sa": image_sa,
        }
        if image_lge_sa is not None:
            params["image_lge_sa"] = image_lge_sa
        if slice_num_4ch is not None:
            params["slice_num_4ch"] = slice_num_4ch
        if slice_num_sa is not None:
            params["slice_num_sa"] = slice_num_sa

        logger.info(f"[MRG] 调用 metrics_worker: {url}")
        logger.info(f"  image_lge_sa: {'provided' if image_lge_sa else 'N/A'}")
        try:
            resp = requests.post(
                f"{url}/worker_generate", json=params, timeout=600
            )
            return resp.json()
        except Exception as e:
            logger.error(f"[MRG] metrics_worker 调用失败: {e}")
            return {"error": str(e), "error_code": ErrorCode.INTERNAL_ERROR}

    def _call_cds(
        self,
        image_4ch: str,
        image_sa: str,
        image_2ch: Optional[str] = None,
        slice_num_4ch: int = 1,
        slice_num_sa: int = 1,
        slice_num_2ch: int = 1,
        skip_preprocess: bool = False,
    ) -> Dict:
        """调用 cds_worker 进行疾病筛查"""
        url = self.downstream["cds"]
        params = {
            "image_4ch": image_4ch,
            "image_sa": image_sa,
            "image_2ch": image_2ch,
            "slice_num_4ch": slice_num_4ch,
            "slice_num_sa": slice_num_sa,
            "slice_num_2ch": slice_num_2ch,
            "num_crops": 3,
            "skip_preprocess": skip_preprocess,
        }

        logger.info(f"[MRG] 调用 cds_worker: {url}")
        logger.info(f"  image_2ch: {'provided' if image_2ch else 'None (placeholder)'}")
        try:
            resp = requests.post(
                f"{url}/worker_generate", json=params, timeout=600
            )
            return resp.json()
        except Exception as e:
            logger.error(f"[MRG] cds_worker 调用失败: {e}")
            return {"error": str(e), "error_code": ErrorCode.INTERNAL_ERROR}

    def _call_nicms(
        self,
        image_4ch: str,
        image_sa: str,
        image_lge_sa: Optional[str] = None,
        slice_num_4ch: int = 1,
        slice_num_sa: int = 1,
        slice_num_lge_sa: int = 1,
        skip_preprocess: bool = False,
    ) -> Dict:
        """调用 nicms_worker 进行非缺血性心肌病亚分类"""
        url = self.downstream["nicms"]
        params = {
            "image_4ch": image_4ch,
            "image_sa": image_sa,
            "image_lge_sa": image_lge_sa,
            "slice_num_4ch": slice_num_4ch,
            "slice_num_sa": slice_num_sa,
            "slice_num_lge_sa": slice_num_lge_sa,
            "num_crops": 3,
            "skip_preprocess": skip_preprocess,
        }

        logger.info(f"[MRG] 调用 nicms_worker: {url}")
        try:
            resp = requests.post(
                f"{url}/worker_generate", json=params, timeout=600
            )
            return resp.json()
        except Exception as e:
            logger.error(f"[MRG] nicms_worker 调用失败: {e}")
            return {"error": str(e), "error_code": ErrorCode.INTERNAL_ERROR}

    # ---- 编排处理 ----

    def process(
        self,
        image_4ch: str,
        image_sa: str,
        image_2ch: Optional[str] = None,
        image_lge_sa: Optional[str] = None,
        slice_num_4ch: int = 1,
        slice_num_sa: int = 1,
        slice_num_2ch: int = 1,
        slice_num_lge_sa: int = 1,
        skip_preprocess: bool = False,
    ) -> Dict:
        """
        编排处理: metrics + CDS + NICMS(条件触发)

        Args:
            image_4ch: cine 4CH 图像路径 (必须)
            image_sa:  cine SA  图像路径 (必须)
            image_2ch: cine 2CH 图像路径 (可选，增强 CDS)
            image_lge_sa: LGE SA 图像路径 (可选，用于 NICMS)
            slice_num_*: 各模态切片数
            skip_preprocess: 是否跳过分类预处理
        """
        try:
            logger.info(f"[MRG] ===== 开始编排处理 =====")
            logger.info(f"  image_4ch: {image_4ch}")
            logger.info(f"  image_sa:  {image_sa}")
            logger.info(f"  image_2ch: {image_2ch or '(未提供)'}")
            logger.info(f"  image_lge_sa: {image_lge_sa or '(未提供)'}")

            result = {"error_code": 0}

            # ---- Step 1: 计算心脏指标 (含可选 LGE SA mass) ----
            logger.info(f"\n[MRG Step 1] 调用 metrics_worker ...")
            metrics_result = self._call_metrics(
                image_4ch, image_sa,
                image_lge_sa=image_lge_sa,
                slice_num_4ch=slice_num_4ch,
                slice_num_sa=slice_num_sa,
            )

            if metrics_result.get("error_code", -1) != 0:
                logger.error(f"  metrics_worker 失败: {metrics_result.get('error')}")
                result["metrics_error"] = metrics_result.get("error")
            else:
                result["metrics"] = metrics_result.get("metrics", {})
                result["metrics_4ch_raw"] = metrics_result.get("metrics_4ch_raw", {})
                result["metrics_sa_raw"] = metrics_result.get("metrics_sa_raw", {})
                result["segmentation_4ch"] = metrics_result.get("segmentation_4ch", {})
                result["segmentation_sa"] = metrics_result.get("segmentation_sa", {})
                if metrics_result.get("segmentation_lge_sa"):
                    result["segmentation_lge_sa"] = metrics_result["segmentation_lge_sa"]
                logger.info(f"  ✓ metrics 计算完成, {len(result['metrics'])} 个指标")

            # ---- Step 2: 疾病筛查 (CDS) ----
            logger.info(f"\n[MRG Step 2] 调用 cds_worker ...")
            cds_result = self._call_cds(
                image_4ch,
                image_sa,
                image_2ch=image_2ch,
                slice_num_4ch=slice_num_4ch,
                slice_num_sa=slice_num_sa,
                slice_num_2ch=slice_num_2ch,
                skip_preprocess=skip_preprocess,
            )

            if cds_result.get("error_code", -1) != 0:
                logger.error(f"  cds_worker 失败: {cds_result.get('error')}")
                result["cds_error"] = cds_result.get("error")
                result["cds_result"] = None
            else:
                cds_pred_class = cds_result.get("pred_class", -1)
                cds_class_name = CC_CLASSES.get(cds_pred_class, f"Unknown ({cds_pred_class})")
                result["cds_result"] = {
                    "pred_class": cds_pred_class,
                    "class_name": cds_class_name,
                    "avg_pred": cds_result.get("avg_pred"),
                }
                logger.info(f"  ✓ CDS 分类完成: {cds_class_name} (class={cds_pred_class})")

            # ---- Step 3: 非缺血性亚分类 (NICMS) — CDS=Non-ischemic 且提供了 LGE 时触发 ----
            result["nicms_result"] = None
            cds_ok = result.get("cds_result") is not None
            is_non_ischemic = cds_ok and result["cds_result"]["pred_class"] == 2
            has_lge = image_lge_sa is not None and image_lge_sa != ""

            if is_non_ischemic and has_lge:
                logger.info(f"\n[MRG Step 3] CDS=Non-ischemic + LGE SA provided, 调用 nicms_worker ...")
                nicms_result = self._call_nicms(
                    image_4ch,
                    image_sa,
                    image_lge_sa,
                    slice_num_4ch=slice_num_4ch,
                    slice_num_sa=slice_num_sa,
                    slice_num_lge_sa=slice_num_lge_sa,
                    skip_preprocess=skip_preprocess,
                )

                if nicms_result.get("error_code", -1) != 0:
                    logger.error(f"  nicms_worker 失败: {nicms_result.get('error')}")
                    result["nicms_error"] = nicms_result.get("error")
                else:
                    nicms_pred_class = nicms_result.get("pred_class", -1)
                    nicms_class_name = NCC_CLASSES.get(
                        nicms_pred_class, f"Unknown ({nicms_pred_class})"
                    )
                    result["nicms_result"] = {
                        "pred_class": nicms_pred_class,
                        "class_name": nicms_class_name,
                        "avg_pred": nicms_result.get("avg_pred"),
                    }
                    logger.info(f"  ✓ NICMS 分类完成: {nicms_class_name} (class={nicms_pred_class})")
            elif is_non_ischemic and not has_lge:
                logger.info(f"\n[MRG Step 3] CDS=Non-ischemic 但未提供 LGE SA, 跳过 NICMS")
            else:
                logger.info(f"\n[MRG Step 3] CDS 未判定为 Non-ischemic, 跳过 NICMS")

            logger.info(f"\n[MRG] ===== 编排处理完成 =====")
            return result

        except Exception as e:
            error_msg = f"[MRG] 编排处理失败: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "error_code": ErrorCode.INTERNAL_ERROR,
                "error": f"{SERVER_ERROR_MSG}\n\n({e})",
                "traceback": traceback.format_exc(),
            }


# ============ FastAPI 服务 ============
app = FastAPI()
worker = None


@app.post("/worker_generate")
async def generate(request: Request):
    """处理医学报告生成请求（编排 metrics + CDS + NICMS）"""
    params = await request.json()

    image_4ch = params.get("image_4ch")
    image_sa = params.get("image_sa")
    image_2ch = params.get("image_2ch")
    image_lge_sa = params.get("image_lge_sa")

    logger.info(
        f"收到 MRG 请求: 4CH={image_4ch}, SA={image_sa}, "
        f"2CH={image_2ch or 'N/A'}, LGE_SA={image_lge_sa or 'N/A'}"
    )

    if not image_4ch or not image_sa:
        error_msg = "缺少必需参数: image_4ch 和 image_sa"
        logger.error(error_msg)
        return JSONResponse(
            {"error_code": ErrorCode.INTERNAL_ERROR, "error": error_msg}
        )

    result = worker.process(
        image_4ch=image_4ch,
        image_sa=image_sa,
        image_2ch=image_2ch,
        image_lge_sa=image_lge_sa,
        slice_num_4ch=params.get("slice_num_4ch", 1),
        slice_num_sa=params.get("slice_num_sa", 1),
        slice_num_2ch=params.get("slice_num_2ch", 1),
        slice_num_lge_sa=params.get("slice_num_lge_sa", 1),
        skip_preprocess=params.get("skip_preprocess", False),
    )

    logger.info(f"请求处理完成, error_code={result.get('error_code')}")
    return JSONResponse(result)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.get("/health")
async def health():
    return {"status": "healthy", "model_names": worker.model_names}


@app.get("/model_details")
async def model_details():
    return {
        "model_names": worker.model_names,
        "task": "Medical Report Generation",
        "description": "医学报告生成 (编排: metrics + CDS + NICMS)",
        "input": {
            "required": ["image_4ch (cine 4CH)", "image_sa (cine SA)"],
            "optional": ["image_2ch (cine 2CH)", "image_lge_sa (LGE SA)"],
        },
        "pipeline": [
            "1. metrics_worker → 心脏功能指标 + 可选 LGE SA mass",
            "2. cds_worker → 疾病筛查 (Normal/Ischemic/Non-ischemic)",
            "3. nicms_worker → 非缺血亚分类 (仅当 CDS=Non-ischemic, LGE SA 可选)",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Report Generation Worker (Orchestrator)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21030)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21030")
    parser.add_argument("--controller-address", type=str, default="http://localhost:30000")
    parser.add_argument("--worker-id", type=str, default=f"mrg-worker-{worker_id}")
    parser.add_argument(
        "--model-names",
        default="MedicalReportGeneration",
        type=lambda s: s.split(","),
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")

    parser.add_argument("--metrics-worker-url", type=str, default="http://localhost:21031")
    parser.add_argument("--cds-worker-url", type=str, default="http://localhost:21020")
    parser.add_argument("--nicms-worker-url", type=str, default="http://localhost:21021")

    args = parser.parse_args()

    logger.info(f"启动参数: {args}")

    downstream = {
        "metrics": args.metrics_worker_url,
        "cds": args.cds_worker_url,
        "nicms": args.nicms_worker_url,
    }

    worker = MRGOrchestratorWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=args.worker_id,
        no_register=args.no_register,
        model_names=args.model_names,
        downstream_workers=downstream,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Medical Report Generation Worker (Orchestrator) 启动")
    logger.info(f"{'='*60}")
    logger.info(f"监听地址: {args.host}:{args.port}")
    logger.info(f"Worker ID: {args.worker_id}")
    logger.info(f"Model Names: {args.model_names}")
    logger.info(f"Controller: {args.controller_address}")
    logger.info(f"下游服务:")
    logger.info(f"  - Metrics: {downstream['metrics']}")
    logger.info(f"  - CDS:     {downstream['cds']}")
    logger.info(f"  - NICMS:   {downstream['nicms']}")
    logger.info(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
