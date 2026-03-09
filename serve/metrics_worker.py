"""
心脏指标计算 Worker (Cardiac Metrics Calculation Worker)

功能:
1. 接收 4CH 和 SA 图像（必须），可选接收 LGE SA 图像
2. 调用 Cine 4CH / Cine SA / LGE SA Segmentation workers
3. 使用分割结果计算心脏指标（含可选 LGE SA Label3 Mass）
4. 返回完整的心脏功能和形态指标
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from typing import Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Get the directory paths
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVE_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODEL_SRC_DIR = os.path.join(SRC_DIR, "CMR")

# Add paths for imports
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODEL_SRC_DIR)

# 导入计算函数
from calculate_cardiac_metrics_cine_4ch import calculate_cine_4ch_metrics
from calculate_cardiac_metrics_cine_sa import calculate_cine_sa_metrics
from calculate_cardiac_metrics_lge_sa import calculate_lge_sa_metrics

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
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(filename),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(name)
    
    def pretty_print_semaphore(sem):
        if sem is None:
            return "None"
        return f"Semaphore(value={sem._value})"

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("metrics_worker", os.path.join("workers", "metrics.log"))

global_counter = 0
model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


# 分割 Worker 地址
SEG_WORKERS = {
    "4ch": "http://localhost:21011",      # Cine4CHSegWorker
    "sa": "http://localhost:21012",       # CineSAXSegWorker
    "lge_sa": "http://localhost:21013",   # LgeSAXSegWorker
}


class MetricsWorker:
    """心脏指标计算 Worker（分割+指标计算）"""
    
    def __init__(self, controller_addr: str, worker_addr: str,
                 worker_id: str, no_register: bool,
                 model_names: list):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        self.temp_dir = tempfile.mkdtemp(prefix="metrics_")
        logger.info(f"初始化 Metrics Worker, ID: {worker_id}")
        logger.info(f"临时目录: {self.temp_dir}")
        
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
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
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length(),
                }, timeout=5)
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
            len(model_semaphore._waiters) if model_semaphore._waiters is not None else 0)
    
    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }
    
    def call_segmentation(self, modality: str, image_path: str, output_path: str) -> Dict:
        """调用分割 Worker"""
        worker_url = SEG_WORKERS.get(modality)
        if not worker_url:
            error_msg = f"未找到分割Worker: {modality}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        
        logger.info(f"[Metrics] 调用 {modality.upper()} 分割服务: {worker_url}")
        logger.info(f"  输入图像: {image_path}")
        logger.info(f"  输出路径: {output_path}")
        
        try:
            resp = requests.post(
                f"{worker_url}/worker_generate",
                json={
                    "image": image_path,
                    "output_path": output_path,
                },
                timeout=300
            )
            result = resp.json()
            
            if result.get("error_code") == 0:
                logger.info(f"  ✓ {modality.upper()} 分割成功, 标签: {result.get('unique_labels')}")
            else:
                logger.error(f"  ✗ {modality.upper()} 分割失败: {result.get('error')}")
            
            return result
            
        except requests.exceptions.Timeout:
            error_msg = f"调用{modality.upper()}分割Worker超时"
            logger.error(f"  ✗ {error_msg}")
            return {
                "error": error_msg,
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        except Exception as e:
            error_msg = f"调用分割Worker失败: {e}"
            logger.error(f"  ✗ {error_msg}")
            logger.error(traceback.format_exc())
            return {
                "error": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.SEGMENTATION_FAILED,
            }
    
    def calculate_metrics(self, mask_4ch: str, mask_sa: str,
                         mask_lge_sa: Optional[str] = None,
                         slice_num_4ch: Optional[int] = None,
                         slice_num_sa: Optional[int] = None) -> Dict:
        """计算心脏指标（含可选 LGE SA mass）"""
        logger.info("[Metrics] 开始计算心脏指标...")
        logger.info(f"  4CH mask: {mask_4ch}")
        logger.info(f"  SA mask: {mask_sa}")
        logger.info(f"  LGE SA mask: {mask_lge_sa or '(未提供)'}")
        
        try:
            logger.info("  计算 4CH 指标...")
            metrics_4ch = calculate_cine_4ch_metrics(mask_4ch, slice_num_4ch)
            logger.info(f"  4CH 指标计算完成，获得 {len(metrics_4ch)} 个指标")
            
            logger.info("  计算 SA 指标...")
            metrics_sa = calculate_cine_sa_metrics(mask_sa, slice_num_sa)
            logger.info(f"  SA 指标计算完成，获得 {len(metrics_sa)} 个指标")
            
            metrics_lge = None
            if mask_lge_sa:
                logger.info("  计算 LGE SA 指标...")
                metrics_lge = calculate_lge_sa_metrics(mask_lge_sa)
                if metrics_lge:
                    logger.info(f"  LGE SA 指标计算完成: {metrics_lge}")
                else:
                    logger.warning("  LGE SA 指标计算返回空")
            
            logger.info("  合并指标...")
            merged_metrics = self._merge_metrics(metrics_4ch, metrics_sa, metrics_lge)
            
            logger.info(f"  ✓ 指标计算完成，共 {len(merged_metrics)} 个指标")
            
            if "LV_EF" in merged_metrics and merged_metrics["LV_EF"] is not None:
                logger.info(f"    LV_EF: {merged_metrics['LV_EF']:.2f}%")
            if "RV_EF" in merged_metrics and merged_metrics["RV_EF"] is not None:
                logger.info(f"    RV_EF: {merged_metrics['RV_EF']:.2f}%")
            if "LGE_SA_Label3_Mass" in merged_metrics and merged_metrics["LGE_SA_Label3_Mass"] is not None:
                logger.info(f"    LGE_SA_Label3_Mass: {merged_metrics['LGE_SA_Label3_Mass']:.4f} g")
            
            return {
                "error_code": 0,
                "metrics": merged_metrics,
                "metrics_4ch": metrics_4ch,
                "metrics_sa": metrics_sa,
            }
            
        except Exception as e:
            error_msg = f"指标计算失败: {e}"
            logger.error(f"  ✗ {error_msg}")
            logger.error(traceback.format_exc())
            return {
                "error_code": ErrorCode.METRICS_CALCULATION_FAILED,
                "error": f"{SERVER_ERROR_MSG}\n\n({e})",
                "traceback": traceback.format_exc(),
            }
    
    def _merge_metrics(self, metrics_4ch: Dict, metrics_sa: Dict,
                       metrics_lge: Dict = None) -> Dict:
        """合并 4CH、SA、LGE 的指标"""
        merged = {}
        
        # 1. 主要直径测量
        merged["LA_LD"] = metrics_4ch.get('LA_ED_Long_Diameter')
        merged["RA_LD"] = metrics_4ch.get('RA_ED_Long_Diameter')
        merged["LV_LD"] = metrics_sa.get('LV_ED_Long_Diameter')
        merged["RV_LD"] = metrics_sa.get('RV_ED_Long_Diameter')
        
        # 2. LV 壁厚度（16个节段 + 心尖）
        segments_info = [
            ("LV_BS", 1, "基底前间隔"),
            ("LV_BS", 2, "基底前壁"),
            ("LV_BS", 3, "基底侧壁"),
            ("LV_BS", 4, "基底后壁"),
            ("LV_BS", 5, "基底下壁"),
            ("LV_BS", 6, "基底下间隔"),
            ("LV_IP", 7, "中间前间隔"),
            ("LV_IP", 8, "中间前壁"),
            ("LV_IP", 9, "中间侧壁"),
            ("LV_IP", 10, "中间后壁"),
            ("LV_IP", 11, "中间下壁"),
            ("LV_IP", 12, "中间下间隔"),
            ("LV_SP", 13, "心尖前壁"),
            ("LV_SP", 14, "心尖侧壁"),
            ("LV_SP", 15, "心尖下壁"),
            ("LV_SP", 16, "心尖间隔"),
        ]
        
        for prefix, num, name in segments_info:
            for stat in ["max", "mean", "min"]:
                key = f"{prefix}_{num:02d}_{stat}"
                metric_key = f'ED_Segment_{num:02d}_{name}_Thickness_{stat}'
                merged[key] = metrics_sa.get(metric_key)
        
        # 3. 心尖厚度（第17节段，从4CH获取）
        merged["LV_TP_17_max"] = metrics_4ch.get('ED_LV_Apex_Thickness_max')
        merged["LV_TP_17_mean"] = metrics_4ch.get('ED_LV_Apex_Thickness_mean')
        merged["LV_TP_17_min"] = metrics_4ch.get('ED_LV_Apex_Thickness_min')
        
        # 4. RV 壁厚度（从4CH获取）
        rv_thickness_keys = ['ED_RV_Wall_Thickness_Div_1', 'ED_RV_Wall_Thickness_Div_2', 'ED_RV_Wall_Thickness_Div_3']
        rv_prefixes = ['RV_BS_01', 'RV_IP_02', 'RV_SP_03']
        
        for rv_key, prefix in zip(rv_thickness_keys, rv_prefixes):
            merged[prefix] = metrics_4ch.get(rv_key)
        
        # 5. 心室功能指标（从SA获取）
        lv_function_keys = ['LV_EDV', 'LV_ESV', 'LV_SV', 'LV_EF', 'LV_CO', 'LV_Mass']
        for key in lv_function_keys:
            merged[key] = metrics_sa.get(key)
        
        rv_function_keys = ['RV_EDV', 'RV_ESV', 'RV_SV', 'RV_EF', 'RV_CO']
        for key in rv_function_keys:
            merged[key] = metrics_sa.get(key)
        
        # 6. LGE SA 指标（可选）
        if metrics_lge:
            for key, val in metrics_lge.items():
                merged[key] = val
        
        return merged
    
    def process(self, image_4ch: str, image_sa: str,
                image_lge_sa: str = None,
                output_4ch: str = None, output_sa: str = None,
                slice_num_4ch: int = None, slice_num_sa: int = None) -> Dict:
        """
        处理心脏指标计算请求（分割+指标计算）
        
        Args:
            image_4ch: 4CH 图像路径 (必须)
            image_sa: SA 图像路径 (必须)
            image_lge_sa: LGE SA 图像路径 (可选，用于计算 LGE mass)
            output_4ch: 4CH 分割结果输出路径（可选）
            output_sa: SA 分割结果输出路径（可选）
            slice_num_4ch: 4CH 切片数量（可选）
            slice_num_sa: SA 切片数量（可选）
        """
        try:
            logger.info(f"[Metrics] 开始处理心脏指标计算请求")
            logger.info(f"  4CH 图像: {image_4ch}")
            logger.info(f"  SA 图像: {image_sa}")
            logger.info(f"  LGE SA 图像: {image_lge_sa or '(未提供)'}")
            
            if output_4ch is None:
                output_4ch = os.path.join(self.temp_dir, "4ch_seg.nii.gz")
                logger.info(f"  使用临时4CH输出路径: {output_4ch}")
            if output_sa is None:
                output_sa = os.path.join(self.temp_dir, "sa_seg.nii.gz")
                logger.info(f"  使用临时SA输出路径: {output_sa}")
            
            # 1. 调用分割 Workers (4CH + SA 必须, LGE SA 可选)
            logger.info("\n[Step 1] 执行分割...")
            seg_result_4ch = self.call_segmentation("4ch", image_4ch, output_4ch)
            seg_result_sa = self.call_segmentation("sa", image_sa, output_sa)
            
            if seg_result_4ch.get("error_code") != 0:
                error_msg = f"4CH分割失败: {seg_result_4ch.get('error')}"
                logger.error(error_msg)
                return {
                    "error_code": ErrorCode.SEGMENTATION_FAILED,
                    "error": error_msg,
                }
            
            if seg_result_sa.get("error_code") != 0:
                error_msg = f"SA分割失败: {seg_result_sa.get('error')}"
                logger.error(error_msg)
                return {
                    "error_code": ErrorCode.SEGMENTATION_FAILED,
                    "error": error_msg,
                }
            
            output_lge_sa = None
            seg_result_lge_sa = None
            if image_lge_sa:
                output_lge_sa = os.path.join(self.temp_dir, "lge_sa_seg.nii.gz")
                logger.info(f"  LGE SA 分割输出路径: {output_lge_sa}")
                seg_result_lge_sa = self.call_segmentation("lge_sa", image_lge_sa, output_lge_sa)
                if seg_result_lge_sa.get("error_code") != 0:
                    logger.warning(f"  LGE SA 分割失败 (非致命): {seg_result_lge_sa.get('error')}")
                    output_lge_sa = None
                    seg_result_lge_sa = None
            
            # 2. 计算指标
            logger.info("\n[Step 2] 计算心脏指标...")
            metrics_result = self.calculate_metrics(
                output_4ch, output_sa,
                mask_lge_sa=output_lge_sa,
                slice_num_4ch=slice_num_4ch,
                slice_num_sa=slice_num_sa,
            )
            
            if metrics_result.get("error_code") != 0:
                logger.error("指标计算失败")
                return metrics_result
            
            logger.info("✓ 心脏指标计算完成")
            result = {
                "error_code": 0,
                "segmentation_4ch": {
                    "output_path": output_4ch,
                    "unique_labels": seg_result_4ch.get("unique_labels"),
                },
                "segmentation_sa": {
                    "output_path": output_sa,
                    "unique_labels": seg_result_sa.get("unique_labels"),
                },
                "metrics": metrics_result["metrics"],
                "metrics_4ch_raw": metrics_result.get("metrics_4ch", {}),
                "metrics_sa_raw": metrics_result.get("metrics_sa", {}),
            }
            if seg_result_lge_sa and output_lge_sa:
                result["segmentation_lge_sa"] = {
                    "output_path": output_lge_sa,
                    "unique_labels": seg_result_lge_sa.get("unique_labels"),
                }
            return result
            
        except Exception as e:
            error_msg = f"[Metrics] 处理失败: {e}"
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
    """处理心脏指标计算请求"""
    params = await request.json()
    
    image_4ch = params.get("image_4ch")
    image_sa = params.get("image_sa")
    image_lge_sa = params.get("image_lge_sa")
    output_4ch = params.get("output_4ch")
    output_sa = params.get("output_sa")
    slice_num_4ch = params.get("slice_num_4ch")
    slice_num_sa = params.get("slice_num_sa")
    
    logger.info(f"收到心脏指标计算请求: 4CH={image_4ch}, SA={image_sa}, LGE_SA={image_lge_sa or 'N/A'}")
    
    if not image_4ch or not image_sa:
        error_msg = "缺少必需参数: image_4ch 和 image_sa"
        logger.error(error_msg)
        return JSONResponse({
            "error_code": ErrorCode.INTERNAL_ERROR,
            "error": error_msg,
        })
    
    result = worker.process(
        image_4ch, image_sa,
        image_lge_sa=image_lge_sa,
        output_4ch=output_4ch, output_sa=output_sa,
        slice_num_4ch=slice_num_4ch, slice_num_sa=slice_num_sa,
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
        "task": "Cardiac Metrics Calculation",
        "description": "心脏指标计算模型 (自动分割+指标计算, 含可选 LGE SA mass)",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cardiac Metrics Calculation Worker")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21031)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21031")
    parser.add_argument("--controller-address", type=str, default="http://localhost:30000")
    parser.add_argument("--worker-id", type=str, default=f"metrics-worker-{worker_id}")
    parser.add_argument("--model-names", default="CardiacMetricsCalculation",
                        type=lambda s: s.split(","))
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    
    logger.info(f"启动参数: {args}")
    
    worker = MetricsWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=args.worker_id,
        no_register=args.no_register,
        model_names=args.model_names,
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Cardiac Metrics Calculation Worker 启动")
    logger.info(f"{'='*60}")
    logger.info(f"监听地址: {args.host}:{args.port}")
    logger.info(f"Worker ID: {args.worker_id}")
    logger.info(f"Model Names: {args.model_names}")
    logger.info(f"Controller: {args.controller_address}")
    logger.info(f"依赖分割服务:")
    logger.info(f"  - 4CH Seg:    {SEG_WORKERS['4ch']}")
    logger.info(f"  - SA Seg:     {SEG_WORKERS['sa']}")
    logger.info(f"  - LGE SA Seg: {SEG_WORKERS['lge_sa']}")
    logger.info(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
