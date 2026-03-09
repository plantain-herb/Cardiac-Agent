"""
Cardiac Agent 系统 - 配置模块

包含所有常量、映射表、路径配置、正则模式等。
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

# ============ 服务地址 ============
CONTROLLER_URL = "http://localhost:30000"
AGENT_URL = "http://localhost:40000"
DEMO_DATA_DIR = "./demo_data"

# Expert Worker 地址配置
EXPERT_WORKERS = {
    "Cine2CHSegWorker": "http://localhost:21010",
    "Cine4CHSegWorker": "http://localhost:21011",
    "CineSAXSegWorker": "http://localhost:21012",
    "LgeSAXSegWorker": "http://localhost:21013",
    "CDSWorker": "http://localhost:21020",
    "NICMSWorker": "http://localhost:21021",
    "MRGWorker": "http://localhost:21030",
    "MetricsWorker": "http://localhost:21031",
    "MIRWorker": "http://localhost:21040",
    "SeqWorker": "http://localhost:21050",
}

EXPERT_NAMES = [
    "Cine2CHSegmentation",
    "Cine4CHSegmentation",
    "CineSAXSegmentation",
    "LgeSAXSegmentation",
    "CardiacDiseaseScreening",
    "NonIschemicCardiomyopathySubclassification",
    "MedicalReportGeneration",
    "CardiacMetricsCalculation",
    "MedicalInformationRetrieval",
    "SequenceAnalysis",
]

# API_name 到 Expert Worker 的映射
API_NAME_TO_WORKER = {
    "2CH Cine Segmentation": "Cine2CHSegWorker",
    "4CH Cine Segmentation": "Cine4CHSegWorker",
    "SAX Cine Segmentation": "CineSAXSegWorker",
    "SAX LGE Segmentation": "LgeSAXSegWorker",
    "Cardiac Disease Screening": "CDSWorker",
    "Non-ischemic Cardiomyopathy Subclassification": "NICMSWorker",
    "Medical Report Generation": "MRGWorker",
    "Cardiac Metrics Calculation": "MetricsWorker",
    "Medical Info Retrieval": "MIRWorker",
    "Sequence Analysis": "SeqWorker",
}

# 序列类型到模态的映射（支持多种变体）
SEQUENCE_TO_MODALITY = {
    "cine 2ch": "2ch",
    "cine 4ch": "4ch",
    "cine sa": "sa",
    "lge sa": "lge",
    "2ch": "2ch",
    "4ch": "4ch",
    "sa": "sa",
    "lge": "lge",
    "cine 2-chamber": "2ch",
    "cine 4-chamber": "4ch",
    "cine short-axis": "sa",
    "cine short axis": "sa",
    "lge short-axis": "lge",
    "lge short axis": "lge",
    "2-chamber": "2ch",
    "4-chamber": "4ch",
    "two chamber": "2ch",
    "four chamber": "4ch",
    "short axis": "sa",
    "short-axis": "sa",
}

# 线程池配置
MAX_WORKERS = 4

# 版本号
VERSION = "v1"

# ============ 项目根目录 ============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 权重目录 ============
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
AGENT_MODEL_PATH = os.path.join(WEIGHTS_DIR, "agent")
# AGENT_MODEL_PATH = "./merged_model/test"

# ============ 日志 / PID 目录 ============
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_WORKERS_DIR = os.path.join(LOG_DIR, "workers")
PID_DIR = os.path.join(PROJECT_ROOT, "pids")
FRONTEND_PID_DIR = os.path.join(PROJECT_ROOT, "app", "frontend", ".pids")

for _d in [LOG_DIR, LOG_WORKERS_DIR, PID_DIR, FRONTEND_PID_DIR]:
    os.makedirs(_d, exist_ok=True)

# ============ 模态排序与全名映射 ============
MODALITY_FULL_ORDER = [
    "cine_2ch",
    "cine_4ch",
    "cine_sa",
    "lge_2ch",
    "lge_4ch",
    "lge_sa",
    "tp",
]

SEQUENCE_TO_FULL_MODALITY = {
    "cine 2ch": "cine_2ch",
    "cine 4ch": "cine_4ch",
    "cine sa": "cine_sa",
    "lge 2ch": "lge_2ch",
    "lge 4ch": "lge_4ch",
    "lge sa": "lge_sa",
    "t1 mapping": "tp",
    "t2 mapping": "tp",
    "t1": "tp",
    "t2": "tp",
    "2ch": "cine_2ch",
    "4ch": "cine_4ch",
    "sa": "cine_sa",
    "lge": "lge_sa",
    "cine 2-chamber": "cine_2ch",
    "cine 4-chamber": "cine_4ch",
    "cine short-axis": "cine_sa",
    "cine short axis": "cine_sa",
    "lge short-axis": "lge_sa",
    "lge short axis": "lge_sa",
    "2-chamber": "cine_2ch",
    "4-chamber": "cine_4ch",
    "two chamber": "cine_2ch",
    "four chamber": "cine_4ch",
    "short axis": "cine_sa",
    "short-axis": "cine_sa",
}

SEQ_TOKEN_NORMALIZE = {
    "sa": "sa", "short axis": "sa", "short-axis": "sa", "s.a.": "sa",
    "2ch": "2ch", "2-chamber": "2ch", "two chamber": "2ch", "two-chamber": "2ch",
    "4ch": "4ch", "4-chamber": "4ch", "four chamber": "4ch", "four-chamber": "4ch",
    "t1 mapping": "tp", "t2 mapping": "tp", "t1": "tp", "t2": "tp",
}

SEQ_TEMPLATE_PATTERNS = [
    re.compile(r'(?:this\s+cardiac\s+mri\s+is\s+(?:a|an)\s+)(cine|lge)\s+sequence\s+(.+?)\s+view', re.I),
    re.compile(r"it'?s\s+(?:a|an)\s+(cine|lge)\s+(.+?)\s+sequence", re.I),
    re.compile(r'(cine|(?:late\s+gadolinium\s+enhancement\s+\(lge\))|lge)\s+sequence\s+detected:\s*(.+?)\s+view', re.I),
    re.compile(r'this\s+is\s+(?:a|an)\s+(cine|lge)\s+(.+?)\s+acquisition', re.I),
]

FULL_MODALITY_TO_SHORT = {
    "cine_2ch": "2ch",
    "cine_4ch": "4ch",
    "cine_sa": "sa",
    "lge_2ch": "lge",
    "lge_4ch": "lge",
    "lge_sa": "lge",
    "tp": "tp",
}

# ============ 缓存目录配置 ============
CACHE_BASE_DIR = os.path.join(PROJECT_ROOT, "cache")
CACHE_UPLOADS_DIR = os.path.join(CACHE_BASE_DIR, "uploads", VERSION)
CACHE_FRAMES_DIR = os.path.join(CACHE_BASE_DIR, "frames", VERSION)
CACHE_IMAGES_DIR = os.path.join(CACHE_BASE_DIR, "images", VERSION)
CACHE_RESULTS_DIR = os.path.join(CACHE_BASE_DIR, "results", VERSION)
CACHE_CONVERSATIONS_DIR = os.path.join(CACHE_BASE_DIR, "conversations", VERSION)

for _dir in [CACHE_UPLOADS_DIR, CACHE_FRAMES_DIR, CACHE_IMAGES_DIR, CACHE_RESULTS_DIR, CACHE_CONVERSATIONS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ============ 报告指标定义（与 src/CMR/calculate.py 严格对齐） ============
LV_WALL_SEGMENTS = [
    ("LV_BS", 1, "Basal Anteroseptal", "基底前间隔"),
    ("LV_BS", 2, "Basal Anterior", "基底前壁"),
    ("LV_BS", 3, "Basal Lateral", "基底侧壁"),
    ("LV_BS", 4, "Basal Posterior", "基底后壁"),
    ("LV_BS", 5, "Basal Inferior", "基底下壁"),
    ("LV_BS", 6, "Basal Inferoseptal", "基底下间隔"),
    ("LV_IP", 7, "Mid Anteroseptal", "中间前间隔"),
    ("LV_IP", 8, "Mid Anterior", "中间前壁"),
    ("LV_IP", 9, "Mid Lateral", "中间侧壁"),
    ("LV_IP", 10, "Mid Posterior", "中间后壁"),
    ("LV_IP", 11, "Mid Inferior", "中间下壁"),
    ("LV_IP", 12, "Mid Inferoseptal", "中间下间隔"),
    ("LV_SP", 13, "Apical Anterior", "心尖前壁"),
    ("LV_SP", 14, "Apical Lateral", "心尖侧壁"),
    ("LV_SP", 15, "Apical Inferior", "心尖下壁"),
    ("LV_SP", 16, "Apical Septal", "心尖间隔"),
]

RV_WALL_SEGMENTS = [
    ("RV_BS_01", "RV Basal Segment"),
    ("RV_IP_02", "RV Mid Segment"),
    ("RV_SP_03", "RV Apical Segment"),
]
