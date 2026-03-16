"""Cardiac Agent 系统 - 核心 Agent 逻辑"""

import os
import re
import shutil
import tempfile
import time
from typing import Dict, List, Tuple

from app.config import (
    API_NAME_TO_WORKER,
    CACHE_RESULTS_DIR,
    FULL_MODALITY_TO_SHORT,
    LV_WALL_SEGMENTS,
    MODALITY_FULL_ORDER,
    RV_WALL_SEGMENTS,
    SEQ_TEMPLATE_PATTERNS,
    SEQ_TOKEN_NORMALIZE,
    SEQUENCE_TO_FULL_MODALITY,
    VERSION,
)
from app.services.agent_client import LLaVAAgentClient
from app.services.expert_client import ExpertWorkerClient
from app.services.seq_identifier import ParallelSequenceIdentifier
from app.services.session_manager import get_session_manager
from app.utils.dicom import (
    cleanup_temp_files,
    convert_dcm_to_nifti,
    extract_frames_from_volume,
    get_clean_seg_name,
    load_scans,
    save_segmentation_images,
)
from app.utils.report import generate_cardiac_report_pdf


class HeartMRIAgent:
    """Cardiac Agent 系统"""
    
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
    
    def __init__(self):
        self.agent_client = LLaVAAgentClient()
        self.expert_client = ExpertWorkerClient()
        self.seq_identifier = ParallelSequenceIdentifier(self.expert_client)
    
    def _build_feedback_prompt(self, api_name: str, expert_result: Dict, question: str) -> str:
        """构建第二次Agent调用的prompt"""
        if "error" in expert_result:
            result_str = f"{{'error': '{expert_result['error']}'}}"
        elif api_name == "Medical Report Generation" and "metrics" in expert_result:
            metrics = expert_result.get("metrics", {})
            key_metrics = {
                "LV_EF": metrics.get("LV_EF"),
                "RV_EF": metrics.get("RV_EF"),
                "LV_EDV": metrics.get("LV_EDV"),
                "LV_ESV": metrics.get("LV_ESV"),
                "LV_SV": metrics.get("LV_SV"),
                "LV_Mass": metrics.get("LV_Mass"),
                "RV_EDV": metrics.get("RV_EDV"),
                "RV_ESV": metrics.get("RV_ESV"),
                "RV_SV": metrics.get("RV_SV"),
            }
            key_metrics = {k: v for k, v in key_metrics.items() if v is not None}
            parts = [f"'metrics': {key_metrics}"]

            cds_result = expert_result.get("cds_result")
            if cds_result:
                parts.append(f"'cds_prediction': '{cds_result['class_name']}'")
            nicms_result = expert_result.get("nicms_result")
            if nicms_result:
                parts.append(f"'nicms_prediction': '{nicms_result['class_name']}'")

            result_str = "{" + ", ".join(parts) + "}"
        elif api_name == "Cardiac Disease Screening" and "pred_class" in expert_result:
            # CDS — 将真实预测结果传递给Agent
            pred_class = expert_result.get("pred_class", -1)
            class_name = self.CC_CLASSES.get(pred_class, f"Unknown ({pred_class})")
            result_str = f"{{'prediction': '{class_name}', 'pred_class': {pred_class}}}"
        elif api_name == "Non-ischemic Cardiomyopathy Subclassification" and "pred_class" in expert_result:
            # NICMS — 将真实预测结果传递给Agent
            pred_class = expert_result.get("pred_class", -1)
            class_name = self.NCC_CLASSES.get(pred_class, f"Unknown ({pred_class})")
            result_str = f"{{'prediction': '{class_name}', 'pred_class': {pred_class}}}"
        elif api_name == "Medical Info Retrieval" and "text" in expert_result:
            result_str = expert_result["text"]
        elif "seg_files" in expert_result or "output_path" in expert_result:
            # 分割任务 — 告知分割已完成
            result_str = f"{{'message': 'The {api_name} has completed segmentation successfully.'}}"
        else:
            result_str = f"{{'message': 'The {api_name} has processed the image.'}}"
        return f"{api_name} output: {result_str}\n\nAnswer my first question: {question}"
    
    def _two_turn_call(self, question: str, images: List[str], api_name: str, 
                       expert_result: Dict, first_response: str = None) -> Tuple[str, str]:
        """执行两次Agent调用"""
        feedback_prompt = self._build_feedback_prompt(api_name, expert_result, question)
        
        # 注意：chat_with_history 方法会自动在第一条消息中添加正确数量的 <image> token
        # 构建正确的对话历史：human -> assistant -> human
        messages = [
            {"role": "human", "content": question},
        ]
        
        # 如果有第一次响应，加入对话历史
        if first_response:
            messages.append({"role": "assistant", "content": first_response})
        
        messages.append({"role": "human", "content": feedback_prompt})
        
        print("\n[Agent] 第二次调用（总结结果）...")
        final_response, _ = self.agent_client.chat_with_history(messages, images)
        final_value = self.agent_client._parse_value(final_response)
        
        return final_response, final_value
    
    def _build_report_data(self, metrics: Dict, seg_4ch: Dict, seg_sa: Dict,
                           cds_result: Dict = None, nicms_result: Dict = None) -> Dict:
        """
        构建格式化的报告数据（包含 cine SA + cine 4CH 全部指标，与 src/CMR/calculate.py 对齐）
        可选包含 CDS/NICMS 分类结果和 LGE mass。
        """
        report = {
            "title": "Cardiac Function Evaluation Report",
            "sections": []
        }
        
        def _add_section(name, metric_defs):
            section = {"name": name, "items": []}
            for key, label, unit, normal in metric_defs:
                value = metrics.get(key)
                if value is not None:
                    section["items"].append({
                        "key": key, "name": label,
                        "value": round(value, 2) if isinstance(value, float) else value,
                        "unit": unit, "normal_range": normal,
                        "status": self._get_metric_status(key, value),
                    })
            if section["items"]:
                report["sections"].append(section)
        
        # 0. Classification Results (CDS / NICMS)
        cls_items = []
        if cds_result and cds_result.get("class_name"):
            cls_items.append({
                "key": "cds_classification", "name": "Disease Screening (CDS)",
                "value": cds_result["class_name"], "unit": "",
                "normal_range": "Normal / Ischemic / Non-ischemic",
                "status": "normal" if cds_result.get("pred_class") == 0 else "abnormal",
            })
        if nicms_result and nicms_result.get("class_name"):
            cls_items.append({
                "key": "nicms_classification", "name": "Non-ischemic Subtype (NICMS)",
                "value": nicms_result["class_name"], "unit": "",
                "normal_range": "HCM / DCM / ICM / RCM / ACM",
                "status": "abnormal",
            })
        if cls_items:
            report["sections"].append({"name": "Classification Results", "items": cls_items})
        
        # 1. Chamber Dimensions (4CH: LA/RA, SA: LV/RV)
        _add_section("Chamber Dimensions", [
            ("LA_LD", "LA Long Diameter (4CH)", "mm", "27-40"),
            ("RA_LD", "RA Long Diameter (4CH)", "mm", "29-45"),
            ("LV_LD", "LV Long Diameter (SA)", "mm", "42-58"),
            ("RV_LD", "RV Long Diameter (SA)", "mm", "35-45"),
        ])
        
        # 2. LV Wall Thickness — 17-segment model
        wall_items = []
        for prefix, num, en_name, _ in LV_WALL_SEGMENTS:
            mean_v = metrics.get(f"{prefix}_{num:02d}_mean")
            if mean_v is not None:
                wall_items.append({
                    "key": f"{prefix}_{num:02d}_mean", "name": en_name,
                    "value": round(mean_v, 2), "unit": "mm",
                    "normal_range": "6-12", "status": "normal",
                })
        apex_v = metrics.get("LV_TP_17_mean")
        if apex_v is not None:
            wall_items.append({
                "key": "LV_TP_17_mean", "name": "Apex (4CH)",
                "value": round(apex_v, 2), "unit": "mm",
                "normal_range": "", "status": "normal",
            })
        if wall_items:
            report["sections"].append({"name": "LV Wall Thickness (17-Segment)", "items": wall_items})
        
        # 3. RV Wall Thickness (from 4CH)
        rv_wall_items = []
        for key, name in RV_WALL_SEGMENTS:
            val = metrics.get(key)
            if val is not None:
                rv_wall_items.append({
                    "key": key, "name": name,
                    "value": round(val, 2), "unit": "mm",
                    "normal_range": "", "status": "normal",
                })
        if rv_wall_items:
            report["sections"].append({"name": "RV Wall Thickness (4CH)", "items": rv_wall_items})
        
        # 4. LV Function (from Cine SA)
        _add_section("Left Ventricle (LV) Function", [
            ("LV_EF", "Ejection Fraction", "%", "55-70"),
            ("LV_EDV", "End-Diastolic Volume", "mL", "56-155"),
            ("LV_ESV", "End-Systolic Volume", "mL", "19-58"),
            ("LV_SV", "Stroke Volume", "mL", "55-100"),
            ("LV_CO", "Cardiac Output", "L/min", "4.0-8.0"),
            ("LV_Mass", "Myocardial Mass", "g", "85-190"),
        ])
        
        # 5. RV Function (from Cine SA)
        _add_section("Right Ventricle (RV) Function", [
            ("RV_EF", "Ejection Fraction", "%", "40-65"),
            ("RV_EDV", "End-Diastolic Volume", "mL", "88-227"),
            ("RV_ESV", "End-Systolic Volume", "mL", "35-100"),
            ("RV_SV", "Stroke Volume", "mL", "55-100"),
            ("RV_CO", "Cardiac Output", "L/min", "4.0-8.0"),
        ])
        
        # 6. LGE SA Mass (if available)
        lge_mass = metrics.get("LGE_SA_Label3_Mass")
        if lge_mass is not None:
            report["sections"].append({
                "name": "LGE Analysis",
                "items": [{
                    "key": "LGE_SA_Label3_Mass",
                    "name": "LGE Scar Mass (Label 3)",
                    "value": round(lge_mass, 2) if isinstance(lge_mass, float) else lge_mass,
                    "unit": "g",
                    "normal_range": "",
                    "status": "abnormal" if lge_mass and lge_mass > 0 else "normal",
                }],
            })
        
        return report
    
    def _get_metric_status(self, key: str, value: float) -> str:
        """判断指标状态"""
        if value is None:
            return "unknown"
        
        # EF 正常范围
        if key == "LV_EF":
            if value >= 55 and value <= 70:
                return "normal"
            elif value < 40:
                return "severely_reduced"
            elif value < 55:
                return "mildly_reduced"
            else:
                return "elevated"
        elif key == "RV_EF":
            if value >= 40 and value <= 65:
                return "normal"
            elif value < 30:
                return "severely_reduced"
            elif value < 40:
                return "mildly_reduced"
            else:
                return "elevated"
        
        return "normal"
    
    # ============ Agent驱动的序列识别与智能抽帧 ============
    
    def _agent_seq_identify(self, volume_path: str, session_id: str = None) -> Dict:
        """
        使用Agent直接进行序列识别
        
        流程:
        1. 分块随机抽三帧
        2. 发送给Agent，询问序列类型
        3. 解析Agent响应，返回模态信息
        
        Args:
            volume_path: volume文件路径
            session_id: 会话ID
            
        Returns:
            Dict: {path, modality, full_modality, frames, frame_info, seq_response}
        """
        print(f"\n  [Agent序列识别] {os.path.basename(volume_path)}")
        
        if not os.path.exists(volume_path):
            return {
                "path": volume_path, "modality": None, "full_modality": None,
                "frames": [], "frame_info": None, "error": "文件不存在",
            }
        
        try:
            # Step 1: 分块随机抽三帧
            temp_frames, frame_info = extract_frames_from_volume(
                volume_path, num_frames=3,
                session_id=session_id, save_to_cache=(session_id is not None)
            )
            
            if session_id:
                session_mgr = get_session_manager()
                session_mgr.add_frames(session_id, frame_info)
            
            # Step 2: 调用Agent进行序列识别
            seq_prompt = (
                "Which sequence does this cardiac MRI belong to? "
                "Please identify if it is one of: cine 2ch, cine 4ch, cine sa, "
                "lge 2ch, lge 4ch, lge sa, t1 mapping, or t2 mapping."
            )
            seq_response, seq_action = self.agent_client.chat(seq_prompt, temp_frames)
            
            # Step 3: 解析序列类型（多级 fallback）
            seq_value = self.agent_client._parse_value(seq_response)
            modality = self._parse_sequence_from_response(seq_value)
            
            # Fallback 1: value 解析失败时，尝试从完整 Agent 响应中提取
            if modality == "unknown" and seq_response != seq_value:
                print(f"    value部分解析失败，尝试完整响应...")
                modality = self._parse_sequence_from_response(seq_response)
            
            # Fallback 2: 尝试从 thoughts 部分提取
            if modality == "unknown":
                thoughts_match = re.search(r'"thoughts🤔"(.*?)"actions🚀"', seq_response, re.DOTALL)
                if thoughts_match:
                    print(f"    完整响应解析失败，尝试thoughts部分...")
                    modality = self._parse_sequence_from_response(thoughts_match.group(1))
            
            full_modality = SEQUENCE_TO_FULL_MODALITY.get(modality, modality)
            
            print(f"    Agent识别结果: {modality} → {full_modality}")
            if modality == "unknown":
                print(f"    [WARNING] 序列识别失败! value='{seq_value[:100]}'")
            
            return {
                "path": volume_path,
                "modality": modality,
                "full_modality": full_modality,
                "frames": temp_frames,
                "frame_info": frame_info,
                "seq_response": seq_response,
                "error": None,
            }
            
        except Exception as e:
            print(f"    序列识别失败: {e}")
            return {
                "path": volume_path, "modality": None, "full_modality": None,
                "frames": [], "frame_info": None, "error": str(e),
            }
    
    def _parse_sequence_from_response(self, response: str) -> str:
        """
        从Agent响应中解析序列类型
        
        优先级:
          0. 模板匹配 — Agent 使用固定 ANSWER_TEMPLATES，最可靠
          1. 直接键名子串匹配
          2. 去标点 token 组合匹配
          3. 正则间接匹配（关键词中间夹杂其他词）
          4. 上下文推断
        
        Args:
            response: Agent的value部分响应文本
            
        Returns:
            str: 解析出的序列类型（如 "cine sa", "cine 2ch" 等）
        """
        if not response:
            return "unknown"
        
        response_lower = response.lower().strip().strip('"').strip("'").strip()
        
        # ── 优先级0: 模板匹配（最可靠）──
        for pat in SEQ_TEMPLATE_PATTERNS:
            m = pat.search(response_lower)
            if m:
                seq_type_raw = m.group(1).strip().lower()
                seq_view_raw = m.group(2).strip().lower()
                is_lge = "lge" in seq_type_raw
                prefix = "lge" if is_lge else "cine"
                view = SEQ_TOKEN_NORMALIZE.get(seq_view_raw, seq_view_raw)
                result = f"{prefix} {view}"
                if result in SEQUENCE_TO_FULL_MODALITY:
                    return result
                if view in SEQUENCE_TO_FULL_MODALITY:
                    return view
        
        # ── 优先级1: 直接匹配完整键名 ──
        for key in SEQUENCE_TO_FULL_MODALITY:
            if key in response_lower:
                return key
        
        # ── 优先级2: 关键词组合匹配（去掉标点再做 token 检查）──
        tokens = re.findall(r'[a-z0-9]+', response_lower)
        token_set = set(tokens)
        
        has_lge = "lge" in response_lower or "late gadolinium" in response_lower
        has_cine = "cine" in response_lower or "cinematic" in response_lower
        has_2ch = bool(token_set & {"2ch"}) or "2-chamber" in response_lower or "two chamber" in response_lower or "two-chamber" in response_lower
        has_4ch = bool(token_set & {"4ch"}) or "4-chamber" in response_lower or "four chamber" in response_lower or "four-chamber" in response_lower
        has_sa = bool(token_set & {"sa"}) or "short" in response_lower or "short-axis" in response_lower or "short axis" in response_lower
        has_mapping = "mapping" in response_lower or bool(token_set & {"t1", "t2"})
        
        if has_mapping:
            return "tp"
        elif has_lge:
            if has_2ch:
                return "lge 2ch"
            elif has_4ch:
                return "lge 4ch"
            else:
                return "lge sa"
        elif has_2ch:
            return "cine 2ch"
        elif has_4ch:
            return "cine 4ch"
        elif has_sa:
            return "cine sa"
        
        # ── 优先级3: 正则匹配（关键词中间夹杂其他词）──
        if re.search(r'\bcine\b.*?\b(sa|short[\s-]?axis)\b', response_lower):
            return "cine sa"
        if re.search(r'\bcine\b.*?\b(2ch|2[\s-]?chamber|two[\s-]?chamber)\b', response_lower):
            return "cine 2ch"
        if re.search(r'\bcine\b.*?\b(4ch|4[\s-]?chamber|four[\s-]?chamber)\b', response_lower):
            return "cine 4ch"
        if re.search(r'\blge\b.*?\b(sa|short[\s-]?axis)\b', response_lower):
            return "lge sa"
        
        # ── 优先级4: 上下文推断（只检测到 cine/lge 但没有明确视图）──
        if has_cine:
            if re.search(r'(cross[\s-]?section|axial|transvers)', response_lower):
                return "cine sa"
            if re.search(r'(long[\s-]?axis|longitudinal)', response_lower):
                return "cine 4ch"
        
        # Fallback: 短响应直接返回，长响应返回 unknown
        return response_lower if len(response_lower) < 30 else "unknown"
    
    def _extract_cine_sa_specific_frames(self, volume_path: str, 
                                          session_id: str = None) -> Tuple[List[str], Dict]:
        """
        为cine SA提取特定帧（分层后选取某一块的第2/5/7帧）
        
        对于cine SA数据（N个切片 × M个心动周期帧）:
        - 选取中间层的切片（mid-ventricular slice）
        - 从该切片中提取第2、5、7帧（心动周期的不同时相）
        
        Args:
            volume_path: volume文件路径
            session_id: 会话ID
            
        Returns:
            Tuple[List[str], Dict]: (帧文件路径列表, 帧信息字典)
        """
        print(f"    [cine SA特殊抽帧] {os.path.basename(volume_path)}")
        import SimpleITK as sitk
        
        # 获取slice_num信息
        slice_num = None
        if volume_path.endswith(".nii.gz") or volume_path.endswith(".nii"):
            sitk_img = sitk.ReadImage(volume_path)
            total_frames = sitk.GetArrayFromImage(sitk_img).shape[0]
        else:
            sitk_img, slice_num = load_scans(volume_path)
            total_frames = sitk.GetArrayFromImage(sitk_img).shape[0]
        
        frame_indices = []
        target_phases = [2, 5, 7]  # 目标心动周期帧索引
        
        if slice_num and slice_num > 0 and total_frames > slice_num:
            # 计算切片数量（层数）
            num_slices = total_frames // slice_num
            # 选取中间层切片
            target_slice = num_slices // 2
            
            print(f"      总帧数: {total_frames}, 心动周期帧数: {slice_num}, "
                  f"切片数: {num_slices}, 选取切片: {target_slice}")
            
            # 从目标切片中提取第2/5/7帧
            for phase in target_phases:
                if phase < slice_num:
                    frame_idx = target_slice * slice_num + phase
                    if frame_idx < total_frames:
                        frame_indices.append(frame_idx)
            
            print(f"      提取帧索引: {frame_indices}")
        
        if not frame_indices:
            # Fallback: 直接使用第2/5/7帧（如果存在）
            print(f"      无法按层提取，使用fallback策略")
            for idx in target_phases:
                if idx < total_frames:
                    frame_indices.append(idx)
        
        if not frame_indices:
            # 最终fallback: 至少取第一帧
            frame_indices = [0]
        
        return extract_frames_from_volume(
            volume_path, frame_indices=frame_indices,
            session_id=session_id, save_to_cache=(session_id is not None)
        )
    
    # ============ 统一Agent驱动处理流程 ============
    
    def process_unified_auto(self, question: str, volume_paths: List[str],
                             session_id: str = None, **kwargs) -> Dict:
        """
        统一Agent驱动处理流程
        
        核心流程:
        1. 对每个上传图像 → 分块随机抽三帧 → Agent序列识别
        2. 如果是cine sa → 重新抽取2/5/7帧; 其他模态 → 保持随机三帧
        3. 按模态顺序排列: cine 2ch → cine 4ch → cine sa → lge 2ch → lge 4ch → lge sa → tp
        4. 组合所有代表帧 → 发给Agent → Agent决定API
        5. 调用对应专家模型
        6. Agent总结结果
        
        Args:
            question: 用户问题
            volume_paths: 上传的volume文件路径列表
            session_id: 会话ID
            **kwargs: api_key, engine, base_url等
            
        Returns:
            Dict: 处理结果
        """
        print("\n" + "="*60)
        print("统一Agent驱动处理流程")
        print("="*60)
        
        pipeline_start_time = time.time()
        stage_timings = {}  # 记录各环节耗时
        
        # ============ Step 1: Agent对每个图像进行序列识别 ============
        print("\n[Step 1] Agent对每个图像进行序列识别（分块随机抽三帧）...")
        step1_start = time.time()
        identified_volumes = []
        
        for vol_path in volume_paths:
            seq_result = self._agent_seq_identify(vol_path, session_id)
            identified_volumes.append(seq_result)
        
        # 打印识别汇总
        step1_elapsed = time.time() - step1_start
        stage_timings['seq_identification'] = step1_elapsed
        print(f"\n  序列识别汇总 (耗时: {step1_elapsed:.2f}s):")
        for vol_info in identified_volumes:
            status = f"{vol_info.get('full_modality', 'unknown')}" if not vol_info.get('error') else f"ERROR: {vol_info['error']}"
            print(f"    {os.path.basename(vol_info['path'])}: {status}")
        
        # ============ Step 2: 根据模态智能抽帧 ============
        step2_start = time.time()
        print("\n[Step 2] 根据模态智能抽帧...")
        for vol_info in identified_volumes:
            if vol_info.get("error"):
                continue
            
            if vol_info["full_modality"] == "cine_sa":
                # cine SA: 分层后选取某一块的2/5/7帧
                print(f"  {os.path.basename(vol_info['path'])}: cine_sa → 重新抽取2/5/7帧")
                new_frames, new_frame_info = self._extract_cine_sa_specific_frames(
                    vol_info["path"], session_id
                )
                vol_info["frames"] = new_frames
                vol_info["frame_info"] = new_frame_info
                
                if session_id:
                    session_mgr = get_session_manager()
                    session_mgr.add_frames(session_id, new_frame_info)
            else:
                # 其他模态: 保持序列识别时的随机三帧
                print(f"  {os.path.basename(vol_info['path'])}: {vol_info.get('modality', '?')} → 保持随机三帧")
        
        step2_elapsed = time.time() - step2_start
        stage_timings['smart_frame_extraction'] = step2_elapsed
        print(f"  [耗时: {step2_elapsed:.2f}s]")
        
        # ============ Step 3: 按模态顺序排列 ============
        print("\n[Step 3] 按模态顺序排列...")
        modality_order = {m: i for i, m in enumerate(MODALITY_FULL_ORDER)}
        
        # 过滤掉识别失败的
        valid_volumes = [v for v in identified_volumes if not v.get("error") and v.get("full_modality")]
        valid_volumes.sort(key=lambda v: modality_order.get(v["full_modality"], 99))
        
        print(f"  排序结果:")
        for vol_info in valid_volumes:
            print(f"    [{modality_order.get(vol_info['full_modality'], '?')}] "
                  f"{vol_info['full_modality']}: {os.path.basename(vol_info['path'])}")
        
        # ============ Step 4: 组合帧发给Agent获取API ============
        step4_start = time.time()
        print("\n[Step 4] 组合帧发给Agent获取API...")
        combined_frames = []
        all_frame_info = []
        
        for vol_info in valid_volumes:
            combined_frames.extend(vol_info.get("frames", []))
            if vol_info.get("frame_info"):
                all_frame_info.append(vol_info["frame_info"])
        
        if not combined_frames:
            return {
                "error": "没有可用的图像帧",
                "session_id": session_id,
                "detected_sequences": [v.get("full_modality") for v in identified_volumes],
            }
        
        # 构建包含序列信息的prompt
        seq_desc_parts = []
        for vol_info in valid_volumes:
            mod_name = vol_info["full_modality"].replace("_", " ")
            seq_desc_parts.append(mod_name)
        seq_desc = "The uploaded cardiac MRI images include: " + ", ".join(seq_desc_parts) + ". "
        
        full_prompt = seq_desc + question
        print(f"  组合帧数: {len(combined_frames)}")
        print(f"  序列信息: {', '.join(seq_desc_parts)}")
        print(f"  Prompt: {full_prompt[:200]}...")
        
        first_response, action = self.agent_client.chat(full_prompt, combined_frames)
        
        step4_elapsed = time.time() - step4_start
        stage_timings['agent_api_decision'] = step4_elapsed
        print(f"\n[Agent API决策] (耗时: {step4_elapsed:.2f}s)")
        print(f"  Action: {action}")
        print(f"  Response (前200字符): {first_response[:200]}..." 
              if len(first_response) > 200 else f"  Response: {first_response}")
        
        # ============ Step 5: 解析API并调用专家模型 ============
        step5_start = time.time()
        api_name = None
        is_agent_vqa = False
        
        if action and action.get("no_api"):
            is_agent_vqa = True
            api_name = "Agent VQA"
            print(f"\n[Step 5] Agent返回空actions → VQA模式（直接回答）")
        elif action and "API_name" in action:
            api_name = action.get("API_name")
            if api_name and api_name in API_NAME_TO_WORKER:
                print(f"\n[Step 5] Agent选择API: {api_name}")
            elif api_name is None:
                is_agent_vqa = True
                api_name = "Agent VQA"
                print(f"\n[Step 5] Agent返回API_name为None → VQA模式")
            else:
                print(f"\n[Step 5] Agent返回未知API '{api_name}' → 使用VQA模式")
                is_agent_vqa = True
                api_name = "Agent VQA"
        else:
            is_agent_vqa = True
            api_name = "Agent VQA"
            print(f"\n[Step 5] Agent未返回API → VQA模式")
        
        # 如果是VQA模式，直接返回Agent响应
        if is_agent_vqa:
            final_value = self.agent_client._parse_value(first_response)
            
            # 修正: 如果 Step 1 序列识别失败（unknown），尝试从 Step 4 Agent 响应中重新解析
            for vol_info in valid_volumes:
                if vol_info.get("full_modality") in ("unknown", None):
                    re_parsed = self._parse_sequence_from_response(first_response)
                    if re_parsed != "unknown":
                        new_full = SEQUENCE_TO_FULL_MODALITY.get(re_parsed, re_parsed)
                        print(f"  [VQA修正] 从Agent响应重新识别序列: {vol_info['full_modality']} → {new_full}")
                        vol_info["full_modality"] = new_full
                        vol_info["modality"] = re_parsed
            
            pipeline_total = time.time() - pipeline_start_time
            print(f"\n[Pipeline耗时汇总 (VQA)] 总耗时: {pipeline_total:.2f}s")
            for sn, st in stage_timings.items():
                print(f"  {sn:.<30s} {st:>7.2f}s")
            return {
                "question": question,
                "api_name": api_name,
                "final_answer": final_value,
                "first_response": first_response,
                "frame_info": all_frame_info,
                "detected_sequences": [v["full_modality"] for v in valid_volumes],
                "session_id": session_id,
            }
        
        # 构建模态→路径映射
        modality_to_path = {}
        for vol_info in valid_volumes:
            fm = vol_info["full_modality"]
            sm = FULL_MODALITY_TO_SHORT.get(fm, fm)
            modality_to_path[fm] = vol_info["path"]
            modality_to_path[sm] = vol_info["path"]
        
        print(f"  模态映射: {list(modality_to_path.keys())}")
        
        # 调用对应专家模型
        expert_result = {}
        prediction = None
        metrics = None
        report_data = None
        seg_result_info = {}
        download_urls = []  # 可下载文件列表
        
        worker_name = API_NAME_TO_WORKER.get(api_name)
        
        if api_name in ["2CH Cine Segmentation", "4CH Cine Segmentation", 
                        "SAX Cine Segmentation", "SAX LGE Segmentation"]:
            # ---- 分割任务 ----
            seg_modality_map = {
                "2CH Cine Segmentation": ["cine_2ch", "2ch"],
                "4CH Cine Segmentation": ["cine_4ch", "4ch"],
                "SAX Cine Segmentation": ["cine_sa", "sa"],
                "SAX LGE Segmentation": ["lge_sa", "lge"],
            }
            target_mods = seg_modality_map.get(api_name, [])
            target_path = None
            for tm in target_mods:
                if tm in modality_to_path:
                    target_path = modality_to_path[tm]
                    break
            
            if target_path is None and volume_paths:
                target_path = volume_paths[0]
            
            # 转换为NIfTI
            nifti_path = target_path
            if target_path and not (target_path.endswith(".nii.gz") or target_path.endswith(".nii")):
                if session_id:
                    nifti_output_dir = os.path.join(CACHE_RESULTS_DIR, session_id, "nifti")
                    volume_name = os.path.basename(target_path)
                    nifti_path = convert_dcm_to_nifti(target_path, nifti_output_dir, volume_name)
                else:
                    nifti_output_dir = tempfile.mkdtemp()
                    volume_name = os.path.basename(target_path)
                    nifti_path = convert_dcm_to_nifti(target_path, nifti_output_dir, volume_name)
            
            # 分割输出路径 — 使用原始上传文件名
            seg_output_path = None
            if session_id:
                seg_output_dir = os.path.join(CACHE_RESULTS_DIR, session_id, "segmentation")
                os.makedirs(seg_output_dir, exist_ok=True)
                # 查找原始文件名来命名seg mask
                session_mgr = get_session_manager()
                orig_name = session_mgr.get_original_name(session_id, target_path)
                seg_filename = get_clean_seg_name(orig_name, "_seg")
                seg_output_path = os.path.join(seg_output_dir, seg_filename)
            
            print(f"  分割目标: {os.path.basename(target_path) if target_path else 'None'}")
            print(f"  Worker: {worker_name}")
            print(f"  Seg输出: {seg_filename if session_id else 'N/A'}")
            
            expert_result = self.expert_client.call_seg(
                worker_name, nifti_path, output_path=seg_output_path
            )
            
            # 保存分割可视化
            if (session_id and "error" not in expert_result 
                and seg_output_path and os.path.exists(seg_output_path)):
                # 找到对应的帧索引
                target_frame_info = None
                for vol_info in valid_volumes:
                    if vol_info["path"] == target_path:
                        target_frame_info = vol_info.get("frame_info")
                        break
                extracted_indices = target_frame_info.get("extracted_indices") if target_frame_info else None
                seg_result_info = save_segmentation_images(
                    nifti_path, seg_output_path, session_id,
                    frame_indices=extracted_indices
                )
                
                # 添加下载链接
                if nifti_path and os.path.exists(nifti_path):
                    download_urls.append({
                        "type": "nifti",
                        "label": "Original NIFTI",
                        "filename": os.path.basename(nifti_path),
                        "url": f"/api/download/{session_id}/nifti/{os.path.basename(nifti_path)}",
                    })
                if seg_output_path and os.path.exists(seg_output_path):
                    download_urls.append({
                        "type": "seg_label",
                        "label": "Segmentation Label (NIFTI)",
                        "filename": os.path.basename(seg_output_path),
                        "url": f"/api/download/{session_id}/segmentation/{os.path.basename(seg_output_path)}",
                    })
        
        elif api_name == "Cardiac Disease Screening":
            # ---- CDS: 需要 4ch + sa, 可选 2ch ----
            image_2ch = modality_to_path.get("cine_2ch") or modality_to_path.get("2ch")
            image_4ch = modality_to_path.get("cine_4ch") or modality_to_path.get("4ch")
            image_sa = modality_to_path.get("cine_sa") or modality_to_path.get("sa")
            
            if not image_4ch or not image_sa:
                return {
                    "error": f"CDS需要4ch和sa模态。已识别: {[v['full_modality'] for v in valid_volumes]}",
                    "api_name": api_name,
                    "detected_sequences": [v["full_modality"] for v in valid_volumes],
                    "session_id": session_id,
                    "agent_response": first_response,
                }
            
            if image_2ch is None:
                print(f"  注意: 未提供2ch模态，Worker将使用空图像占位")
            
            # 为分类任务准备分割输出路径（CC Worker内部会执行分割预处理）
            # CDS分割输出 — 使用原始文件名
            cc_seg_kwargs = dict(kwargs)
            if session_id:
                cc_seg_dir = os.path.join(CACHE_RESULTS_DIR, session_id, "segmentation")
                os.makedirs(cc_seg_dir, exist_ok=True)
                _smgr = get_session_manager()
                if image_4ch:
                    _orig = _smgr.get_original_name(session_id, image_4ch)
                    cc_seg_kwargs["seg_output_4ch"] = os.path.join(cc_seg_dir, get_clean_seg_name(_orig or "4ch", "_4ch_seg"))
                if image_sa:
                    _orig = _smgr.get_original_name(session_id, image_sa)
                    cc_seg_kwargs["seg_output_sa"] = os.path.join(cc_seg_dir, get_clean_seg_name(_orig or "sa", "_sa_seg"))
                if image_2ch:
                    _orig = _smgr.get_original_name(session_id, image_2ch)
                    cc_seg_kwargs["seg_output_2ch"] = os.path.join(cc_seg_dir, get_clean_seg_name(_orig or "2ch", "_2ch_seg"))
            
            expert_result = self.expert_client.call_cds(
                worker_name, image_2ch, image_4ch, image_sa, **cc_seg_kwargs
            )
            
            if "error" not in expert_result:
                pred_class = expert_result.get("pred_class", -1)
                prediction = self.CC_CLASSES.get(pred_class, f"Unknown ({pred_class})")
            
            # 收集分类任务的分割标签下载链接
            if session_id:
                for seg_key, seg_label in [
                    ("seg_output_4ch", "4CH Segmentation Label"),
                    ("seg_output_sa", "SA Segmentation Label"),
                    ("seg_output_2ch", "2CH Segmentation Label"),
                ]:
                    seg_path = cc_seg_kwargs.get(seg_key)
                    if seg_path and os.path.exists(seg_path):
                        download_urls.append({
                            "type": "seg_label",
                            "label": seg_label,
                            "filename": os.path.basename(seg_path),
                            "url": f"/api/download/{session_id}/segmentation/{os.path.basename(seg_path)}",
                        })
                # 也提供原始NIFTI下载
                for vol_label, vol_path in [("4CH NIFTI", image_4ch), ("SA NIFTI", image_sa), ("2CH NIFTI", image_2ch)]:
                    if vol_path and os.path.exists(vol_path):
                        # 如果是DICOM路径，先转换
                        nifti_p = vol_path
                        if not (vol_path.endswith(".nii.gz") or vol_path.endswith(".nii")):
                            nifti_out = os.path.join(CACHE_RESULTS_DIR, session_id, "nifti")
                            nifti_p = convert_dcm_to_nifti(vol_path, nifti_out, os.path.basename(vol_path))
                        if nifti_p and os.path.exists(nifti_p):
                            download_urls.append({
                                "type": "nifti",
                                "label": vol_label,
                                "filename": os.path.basename(nifti_p),
                                "url": f"/api/download/{session_id}/nifti/{os.path.basename(nifti_p)}",
                            })
        
        elif api_name == "Non-ischemic Cardiomyopathy Subclassification":
            # ---- NICMS: 需要 sa + lge, 可选 4ch ----
            image_4ch = modality_to_path.get("cine_4ch") or modality_to_path.get("4ch")
            image_sa = modality_to_path.get("cine_sa") or modality_to_path.get("sa")
            image_lge = modality_to_path.get("lge_sa") or modality_to_path.get("lge")
            
            if not image_sa or not image_lge:
                return {
                    "error": f"NICMS需要sa和lge模态。已识别: {[v['full_modality'] for v in valid_volumes]}",
                    "api_name": api_name,
                    "detected_sequences": [v["full_modality"] for v in valid_volumes],
                    "session_id": session_id,
                    "agent_response": first_response,
                }
            
            if image_4ch is None:
                print(f"  注意: 未提供4ch模态，Worker将使用空图像占位")
            
            expert_result = self.expert_client.call_nicms(
                worker_name, image_4ch, image_sa, image_lge, **kwargs
            )
            
            if "error" not in expert_result:
                pred_class = expert_result.get("pred_class", -1)
                prediction = self.NCC_CLASSES.get(pred_class, f"Unknown ({pred_class})")
            
            # NICMS也提供原始NIFTI下载
            if session_id:
                for vol_label, vol_path in [("SA NIFTI", image_sa), ("LGE NIFTI", image_lge), ("4CH NIFTI", image_4ch)]:
                    if vol_path and os.path.exists(vol_path):
                        nifti_p = vol_path
                        if not (vol_path.endswith(".nii.gz") or vol_path.endswith(".nii")):
                            nifti_out = os.path.join(CACHE_RESULTS_DIR, session_id, "nifti")
                            nifti_p = convert_dcm_to_nifti(vol_path, nifti_out, os.path.basename(vol_path))
                        if nifti_p and os.path.exists(nifti_p):
                            download_urls.append({
                                "type": "nifti",
                                "label": vol_label,
                                "filename": os.path.basename(nifti_p),
                                "url": f"/api/download/{session_id}/nifti/{os.path.basename(nifti_p)}",
                            })
        
        elif api_name == "Medical Report Generation":
            # ---- 医学报告生成: 需要 4ch + sa，可选 2ch / lge_sa ----
            image_4ch = modality_to_path.get("cine_4ch") or modality_to_path.get("4ch")
            image_sa = modality_to_path.get("cine_sa") or modality_to_path.get("sa")
            image_2ch = modality_to_path.get("cine_2ch") or modality_to_path.get("2ch")
            image_lge = modality_to_path.get("lge_sa") or modality_to_path.get("lge")
            
            if not image_4ch or not image_sa:
                return {
                    "error": f"报告生成需要4ch和sa模态。已识别: {[v['full_modality'] for v in valid_volumes]}",
                    "api_name": api_name,
                    "detected_sequences": [v["full_modality"] for v in valid_volumes],
                    "session_id": session_id,
                    "agent_response": first_response,
                }
            
            if image_2ch:
                print(f"  可选模态 2ch: {os.path.basename(image_2ch)}")
            if image_lge:
                print(f"  可选模态 lge_sa: {os.path.basename(image_lge)}")
            
            expert_result = self.expert_client.call_mrg(
                worker_name, image_4ch, image_sa,
                image_2ch=image_2ch, image_lge_sa=image_lge,
                **kwargs
            )
            
            metrics = expert_result.get("metrics", {})
            cds_res = expert_result.get("cds_result")
            nicms_res = expert_result.get("nicms_result")
            if metrics:
                report_data = self._build_report_data(
                    metrics,
                    expert_result.get("segmentation_4ch", {}),
                    expert_result.get("segmentation_sa", {}),
                    cds_result=cds_res, nicms_result=nicms_res,
                )
            if cds_res:
                prediction = cds_res.get("class_name")
                print(f"  CDS 分类: {prediction}")
            if nicms_res:
                print(f"  NICMS 亚分类: {nicms_res.get('class_name')}")
            
            # 生成报告（PDF优先，自动回退到TXT）
            if session_id and metrics:
                reports_dir = os.path.join(CACHE_RESULTS_DIR, session_id, "reports")
                os.makedirs(reports_dir, exist_ok=True)
                pdf_path = os.path.join(reports_dir, "cardiac_report.pdf")
                print(f"  尝试生成报告: {pdf_path}")
                print(f"  metrics keys ({len(metrics)}): {list(metrics.keys())[:20]}...")
                try:
                    generated_report = generate_cardiac_report_pdf(
                        metrics=metrics,
                        report_data=report_data,
                        output_path=pdf_path,
                    )
                    if generated_report and os.path.exists(generated_report):
                        report_size = os.path.getsize(generated_report)
                        report_filename = os.path.basename(generated_report)
                        is_pdf = report_filename.endswith('.pdf')
                        print(f"  报告生成成功: {generated_report} ({report_size} bytes, {'PDF' if is_pdf else 'TXT'})")
                        download_urls.append({
                            "type": "report_pdf",
                            "label": f"Cardiac Report ({'PDF' if is_pdf else 'TXT'})",
                            "filename": report_filename,
                            "url": f"/api/download/{session_id}/reports/{report_filename}",
                        })
                    else:
                        print(f"  报告生成返回None或文件不存在")
                except Exception as pdf_err:
                    import traceback
                    print(f"报告生成失败: {pdf_err}")
                    traceback.print_exc()
            elif session_id:
                print(f"  无metrics数据，跳过报告生成")
            
            # 收集MRG产生的分割nii.gz下载链接 — 使用原始文件名
            if session_id:
                _smgr = get_session_manager()
                seg_4ch_info = expert_result.get("segmentation_4ch", {})
                seg_sa_info = expert_result.get("segmentation_sa", {})
                for seg_label, seg_info, vol_path in [
                    ("4CH Seg Label", seg_4ch_info, image_4ch), 
                    ("SA Seg Label", seg_sa_info, image_sa),
                ]:
                    seg_path = seg_info.get("output_path")
                    if seg_path and os.path.exists(seg_path):
                        seg_dest_dir = os.path.join(CACHE_RESULTS_DIR, session_id, "segmentation")
                        os.makedirs(seg_dest_dir, exist_ok=True)
                        orig_name = _smgr.get_original_name(session_id, vol_path) if vol_path else ""
                        seg_dest_name = get_clean_seg_name(orig_name, "_seg") if orig_name else os.path.basename(seg_path)
                        seg_dest = os.path.join(seg_dest_dir, seg_dest_name)
                        if not os.path.exists(seg_dest):
                            shutil.copy2(seg_path, seg_dest)
                        download_urls.append({
                            "type": "seg_label",
                            "label": seg_label,
                            "filename": seg_dest_name,
                            "url": f"/api/download/{session_id}/segmentation/{seg_dest_name}",
                        })
                # 原始NIFTI下载
                nifti_pairs = [("4CH NIFTI", image_4ch), ("SA NIFTI", image_sa)]
                if image_2ch:
                    nifti_pairs.append(("2CH NIFTI", image_2ch))
                if image_lge:
                    nifti_pairs.append(("LGE SA NIFTI", image_lge))
                for vol_label, vol_path in nifti_pairs:
                    if vol_path and os.path.exists(vol_path):
                        nifti_p = vol_path
                        if not (vol_path.endswith(".nii.gz") or vol_path.endswith(".nii")):
                            nifti_out = os.path.join(CACHE_RESULTS_DIR, session_id, "nifti")
                            nifti_p = convert_dcm_to_nifti(vol_path, nifti_out, os.path.basename(vol_path))
                        if nifti_p and os.path.exists(nifti_p):
                            download_urls.append({
                                "type": "nifti",
                                "label": vol_label,
                                "filename": os.path.basename(nifti_p),
                                "url": f"/api/download/{session_id}/nifti/{os.path.basename(nifti_p)}",
                            })
        
        elif api_name == "Medical Info Retrieval":
            # ---- MIR (Medical Info Retrieval) ----
            expert_result = self.expert_client.call_mir(worker_name, question, **kwargs)
        
        else:
            expert_result = {"error": f"未知的API: {api_name}"}
        
        step5_elapsed = time.time() - step5_start
        # 根据实际调用的API类型记录耗时
        if api_name in ["2CH Cine Segmentation", "4CH Cine Segmentation", 
                        "SAX Cine Segmentation", "SAX LGE Segmentation"]:
            stage_timings['segmentation'] = step5_elapsed
        elif api_name == "Cardiac Disease Screening":
            stage_timings['classification_cc'] = step5_elapsed
        elif api_name == "Non-ischemic Cardiomyopathy Subclassification":
            stage_timings['classification_ncc'] = step5_elapsed
        elif api_name == "Medical Report Generation":
            stage_timings['report_generation'] = step5_elapsed
        elif api_name == "Medical Info Retrieval":
            stage_timings['mir'] = step5_elapsed
        else:
            stage_timings['expert_model'] = step5_elapsed
        print(f"  [Expert模型执行完成，耗时: {step5_elapsed:.2f}s]")
        
        # ============ Step 6: 第二次Agent调用（总结结果） ============
        step6_start = time.time()
        print(f"\n[Step 6] 第二次Agent调用（总结结果）...")
        full_response, final_value = self._two_turn_call(
            question, combined_frames, api_name, expert_result, first_response
        )
        step6_elapsed = time.time() - step6_start
        stage_timings['agent_summary'] = step6_elapsed
        print(f"  [Agent总结完成，耗时: {step6_elapsed:.2f}s]")
        
        # 清理临时文件
        if not session_id:
            cleanup_temp_files(combined_frames)
        
        # 打印下载链接汇总
        if download_urls:
            print(f"\n可下载文件 ({len(download_urls)}):")
            for dl in download_urls:
                print(f"  [{dl['type']}] {dl['label']}: {dl['filename']}")
        
        # 打印完整pipeline耗时汇总
        pipeline_total = time.time() - pipeline_start_time
        stage_timings['total'] = pipeline_total
        print(f"\n{'='*60}")
        print(f"[Pipeline耗时汇总] 总耗时: {pipeline_total:.2f}s")
        print(f"{'='*60}")
        for stage_name, stage_time in stage_timings.items():
            if stage_name != 'total':
                pct = (stage_time / pipeline_total * 100) if pipeline_total > 0 else 0
                print(f"  {stage_name:.<30s} {stage_time:>7.2f}s ({pct:>5.1f}%)")
        print(f"  {'total':.<30s} {pipeline_total:>7.2f}s (100.0%)")
        print(f"{'='*60}")
        
        # 构建返回结果
        result = {
            "question": question,
            "api_name": api_name,
            "expert_result": expert_result,
            "final_answer": final_value,
            "first_response": first_response,  # 保留第一轮响应用于前端可视化
            "frame_info": all_frame_info,
            "detected_sequences": [v["full_modality"] for v in valid_volumes],
            "session_id": session_id,
            "download_urls": download_urls,  # 可下载文件列表
        }
        
        if prediction:
            result["prediction"] = prediction
        if metrics:
            result["metrics"] = metrics
        if report_data:
            result["report_data"] = report_data
        if seg_result_info:
            result["seg_result"] = seg_result_info
        
        # MRG 编排结果中的 CDS / NICMS
        if api_name == "Medical Report Generation" and expert_result:
            if expert_result.get("cds_result"):
                result["cds_result"] = expert_result["cds_result"]
                if not prediction:
                    result["prediction"] = expert_result["cds_result"].get("class_name")
            if expert_result.get("nicms_result"):
                result["nicms_result"] = expert_result["nicms_result"]
        
        return result
    
    # ============ 通用处理接口（按模态分发） ============
    def process_request(self, question: str, volume_paths: List[str] = None,
                       task_type: str = "mr", session_id: str = None,
                       image_paths: List[str] = None, **kwargs) -> Dict:
        """
        通用请求处理接口 — 按影像模态分发

        Args:
            question: 用户问题
            volume_paths: 上传的文件路径列表（医学影像）
            task_type: 影像模态 (mr / ct / us / ecg)
            session_id: 会话ID
            image_paths: PNG图像路径列表
            **kwargs: api_key, engine, base_url 等
        """
        modality = (task_type or "mr").lower()
        print(f"\n[process_request] modality={modality}")
        print(f"  问题: {question[:100]}..." if len(question) > 100 else f"  问题: {question}")
        print(f"  会话ID: {session_id}")
        print(f"  volume数量: {len(volume_paths) if volume_paths else 0}")
        print(f"  图像数量: {len(image_paths) if image_paths else 0}")

        dispatch = {
            "mr": self.process_mr,
            "ct": self.process_ct,
            "us": self.process_us,
            "ecg": self.process_ecg,
        }
        handler = dispatch.get(modality, self.process_mr)
        return handler(question, volume_paths=volume_paths, session_id=session_id,
                       image_paths=image_paths, **kwargs)

    # ============ MR 处理入口 ============
    def process_mr(self, question: str, volume_paths: List[str] = None,
                   session_id: str = None, image_paths: List[str] = None,
                   **kwargs) -> Dict:
        """
        Cardiac MR 统一入口

        路由逻辑:
          1. 有 volume → process_unified_auto（Agent 序列识别 → 专家模型）
          2. 仅 PNG 图像 → Agent 直接分析图像并回答
          3. 纯文本 → Agent 决策（MIR / VQA）
        """
        print("\n" + "="*60)
        print("Cardiac MR 处理流程")
        print("="*60)

        # --- 有医学影像 volume ---
        if volume_paths and len(volume_paths) > 0:
            print(f"\n  → 统一Agent驱动流程 ({len(volume_paths)} 个volume)")
            return self.process_unified_auto(question, volume_paths,
                                             session_id=session_id, **kwargs)

        # --- 仅 PNG 图像（无 volume）---
        if image_paths and len(image_paths) > 0:
            print(f"\n  → PNG图像分析模式 ({len(image_paths)} 张)")
            return self._handle_image_chat(question, image_paths,
                                           session_id=session_id, **kwargs)

        # --- 纯文本（无任何文件）---
        print(f"\n  → 纯文本问答，调用Agent判断API...")
        return self._handle_text_only(question, session_id=session_id, **kwargs)

    # ============ CT / US / ECG 占位 ============
    def process_ct(self, question: str, **kwargs) -> Dict:
        """Cardiac CT — 功能开发中"""
        return {
            "question": question,
            "api_name": "CT (Coming Soon)",
            "final_answer": "Cardiac CT analysis is not yet available. Please stay tuned.",
            "session_id": kwargs.get("session_id"),
        }

    def process_us(self, question: str, **kwargs) -> Dict:
        """Cardiac Ultrasound — 功能开发中"""
        return {
            "question": question,
            "api_name": "US (Coming Soon)",
            "final_answer": "Cardiac ultrasound analysis is not yet available. Please stay tuned.",
            "session_id": kwargs.get("session_id"),
        }

    def process_ecg(self, question: str, **kwargs) -> Dict:
        """ECG — 功能开发中"""
        return {
            "question": question,
            "api_name": "ECG (Coming Soon)",
            "final_answer": "ECG analysis is not yet available. Please stay tuned.",
            "session_id": kwargs.get("session_id"),
        }

    # ============ 内部辅助：PNG 图像对话 ============
    def _handle_image_chat(self, question: str, image_paths: List[str],
                           session_id: str = None, **kwargs) -> Dict:
        """Agent 直接分析 PNG/JPG 图像并回答"""
        images_info = []
        valid_images = []

        for img_path in image_paths:
            if os.path.exists(img_path):
                valid_images.append(img_path)
                images_info.append({
                    "path": img_path,
                    "filename": os.path.basename(img_path),
                    "url": (f"/cache/images/{VERSION}/{session_id}/"
                            f"{os.path.basename(img_path)}") if session_id else None,
                })

        if not valid_images:
            return {
                "question": question,
                "api_name": "Agent VQA",
                "error": "没有有效的图像文件",
                "session_id": session_id,
            }

        response, action = self.agent_client.chat(question, valid_images)
        final_value = self.agent_client._parse_value(response)

        return {
            "question": question,
            "api_name": "Agent VQA",
            "final_answer": final_value,
            "first_response": response,
            "images_info": images_info,
            "session_id": session_id,
        }

    # ============ 内部辅助：纯文本问答 ============
    def _handle_text_only(self, question: str, session_id: str = None,
                          **kwargs) -> Dict:
        """无文件上传时，Agent 自行决策（MIR 或 VQA 直答）"""
        first_response, action = self.agent_client.chat(question, None)

        api_name = None
        if action and action.get("API_name"):
            api_name = action["API_name"]
            print(f"  Agent选择API: {api_name}")
        elif action and action.get("no_api"):
            print(f"  Agent返回空actions → VQA模式")
        else:
            print(f"  Agent未返回有效API → VQA模式")

        if api_name and api_name in API_NAME_TO_WORKER:
            worker_type = API_NAME_TO_WORKER.get(api_name)
            if worker_type == "MIRWorker":
                print(f"  → Agent选择MIR")
                worker_name = API_NAME_TO_WORKER.get(api_name)
                expert_result = self.expert_client.call_mir(
                    worker_name, question, **kwargs)
                return {
                    "question": question,
                    "api_name": api_name,
                    "expert_result": expert_result,
                    "final_answer": expert_result.get("text", ""),
                    "first_response": first_response,
                    "session_id": session_id,
                }
            else:
                print(f"  → API '{api_name}' 需要图像文件")
                return {
                    "question": question,
                    "api_name": api_name,
                    "final_answer": (f"The API '{api_name}' requires image files. "
                                     "Please upload relevant cardiac MRI images."),
                    "first_response": first_response,
                    "session_id": session_id,
                }

        final_value = self.agent_client._parse_value(first_response)
        return {
            "question": question,
            "api_name": "Agent VQA",
            "final_answer": final_value,
            "first_response": first_response,
            "session_id": session_id,
        }

