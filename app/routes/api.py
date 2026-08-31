"""FastAPI 路由定义"""

import json
import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    CACHE_BASE_DIR,
    CACHE_CONVERSATIONS_DIR,
    CACHE_FRAMES_DIR,
    CACHE_IMAGES_DIR,
    CACHE_RESULTS_DIR,
    VERSION,
)
from app.services.heart_agent import HeartMRIAgent
from app.services.exam_identity import extract_exam_identity
from app.services.longitudinal import compare_exams, normalize_metrics, normalize_sequences
from app.services.wall_motion import score_wall_motion
from app.services.segmentation_correction import (
    SegmentationCorrectionError,
    validate_corrected_segmentation,
)
from app.services.session_manager import get_session_manager
from app.utils.conversation import save_conversation_json
from app.utils.dicom import extract_zip_file
from app.utils.report import generate_cardiac_report_pdf


def create_api_app():
    """创建FastAPI应用"""
    app = FastAPI(
        title="Cardiac Agent API",
        description="Intelligent Cardiac Imaging Analysis System API",
        version="24.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载缓存目录为静态文件服务（用于前端访问抽帧图片）
    app.mount("/cache", StaticFiles(directory=CACHE_BASE_DIR), name="cache")

    # 创建Agent实例
    agent = HeartMRIAgent()
    session_mgr = get_session_manager()

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "Cardiac Agent API"}

    # ============ 会话管理接口 ============
    @app.post("/api/session/create")
    async def create_session():
        """创建新会话"""
        session_id = session_mgr.create_session()
        return {"session_id": session_id}

    @app.get("/api/session/list")
    async def list_sessions():
        """列出所有会话"""
        return {"sessions": session_mgr.list_sessions()}

    @app.get("/api/session/{session_id}")
    async def get_session(session_id: str):
        """获取会话详情"""
        session = session_mgr.get_session(session_id)
        if not session:
            return JSONResponse({"error": "会话不存在"}, status_code=404)
        return session

    @app.get("/api/session/{session_id}/files")
    async def get_session_files(session_id: str):
        """获取会话的上传文件列表"""
        files = session_mgr.get_session_files(session_id)
        return {"files": files}

    @app.get("/api/session/{session_id}/frames")
    async def get_session_frames(session_id: str):
        """获取会话的抽帧结果"""
        frames = session_mgr.get_session_frames(session_id)
        # 转换路径为可访问的URL
        for frame_info in frames:
            for f in frame_info.get("frame_files", []):
                f["url"] = f"/cache/frames/{VERSION}/{session_id}/{f['filename']}"
        print(f"frames: {frames}")
        return {"frames": frames}

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        """删除指定会话"""
        session_mgr.cleanup_session(session_id)
        return {"message": f"会话 {session_id} 已删除"}

    @app.delete("/api/cache/clear")
    async def clear_cache():
        """清理所有缓存"""
        session_mgr.cleanup_all()
        return {"message": "所有缓存已清理"}

    @app.post("/api/chat")
    async def chat(
        message: str = Form(""),
        model: str = Form("agent"),
        session_id: str = Form(""),
        task_type: str = Form(""),
        files: List[UploadFile] = File([]),
    ):
        """统一的聊天接口 - 支持 zip/nii 文件上传、PNG图像上传和缓存"""

        # 如果没有提供 session_id，创建新会话
        if not session_id:
            session_id = session_mgr.create_session()

        session = session_mgr.get_session(session_id)
        if not session:
            session_id = session_mgr.create_session()
            session = session_mgr.get_session(session_id)

        # 获取会话的上传目录
        upload_dir = session["upload_dir"]
        images_dir = session.get("images_dir", os.path.join(CACHE_IMAGES_DIR, session_id))
        os.makedirs(images_dir, exist_ok=True)

        # 保存上传的文件并分类
        volume_paths = []  # 医学影像文件（zip, nii.gz, nii）
        image_paths = []   # PNG/JPG图像文件

        try:
            for file in files:
                file_id = str(uuid.uuid4())[:8]
                filename_lower = file.filename.lower()
                original_name = file.filename

                # 检查文件类型
                if filename_lower.endswith(".zip"):
                    # 保存 zip 文件到缓存目录
                    zip_path = os.path.join(upload_dir, f"{file_id}_{original_name}")
                    content = await file.read()
                    with open(zip_path, "wb") as f:
                        f.write(content)

                    # 解压 zip 文件
                    extract_dir = os.path.join(upload_dir, f"{file_id}_extracted")
                    os.makedirs(extract_dir, exist_ok=True)
                    dcm_path = extract_zip_file(zip_path, extract_dir)

                    if dcm_path:
                        volume_paths.append(dcm_path)
                        session_mgr.add_file(session_id, dcm_path, original_name)
                    else:
                        print(f"警告: zip 文件 {file.filename} 中未找到有效的医学影像文件")

                elif filename_lower.endswith(".nii.gz"):
                    # 直接保存 nii.gz 文件
                    file_path = os.path.join(upload_dir, f"{file_id}_{original_name}")
                    content = await file.read()
                    with open(file_path, "wb") as f:
                        f.write(content)
                    volume_paths.append(file_path)
                    session_mgr.add_file(session_id, file_path, original_name)

                elif filename_lower.endswith(".nii"):
                    # 直接保存 nii 文件
                    file_path = os.path.join(upload_dir, f"{file_id}_{original_name}")
                    content = await file.read()
                    with open(file_path, "wb") as f:
                        f.write(content)
                    volume_paths.append(file_path)
                    session_mgr.add_file(session_id, file_path, original_name)

                elif filename_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    # 保存 PNG/JPG 图像文件
                    file_path = os.path.join(images_dir, f"{file_id}_{original_name}")
                    content = await file.read()
                    with open(file_path, "wb") as f:
                        f.write(content)
                    image_paths.append(file_path)
                    session_mgr.add_image(session_id, file_path, original_name)
                    print(f"保存PNG图像: {file_path}")

                else:
                    print(f"警告: 不支持的文件类型: {file.filename}")

            # 从环境变量获取API配置
            api_key = os.getenv("API_KEY")
            engine = os.getenv("MODEL", "deepseek-chat")
            base_url = os.getenv("API_BASE_URL")

            # 调用Agent处理（传入session_id用于缓存抽帧结果）
            exam_identity = extract_exam_identity(volume_paths)
            result = agent.process_request(
                question=message,
                volume_paths=volume_paths if volume_paths else None,
                task_type=task_type if task_type else "mr",
                session_id=session_id,
                image_paths=image_paths if image_paths else None,
                api_key=api_key,
                engine=engine,
                base_url=base_url,
            )

            # 构建响应
            response_text = result.get("final_answer", result.get("error", "处理完成"))

            # 获取抽帧结果的URL
            frame_urls = []
            for frame_info in result.get("frame_info", []):
                if frame_info:
                    for f in frame_info.get("frame_files", []):
                        frame_urls.append({
                            "url": f"/cache/frames/{VERSION}/{session_id}/{f['filename']}",
                            "filename": f["filename"],
                            "frame_index": f["frame_index"],
                        })

            # 获取上传的图像URL
            image_urls = []
            for img_info in result.get("images_info", []):
                if img_info:
                    image_urls.append({
                        "url": img_info.get("url", f"/cache/images/{VERSION}/{session_id}/{img_info['filename']}"),
                        "filename": img_info.get("filename"),
                    })

            # 获取分割结果
            seg_result = result.get("seg_result", {})
            seg_image_url = None
            if seg_result.get("seg_image_url"):
                seg_image_url = seg_result["seg_image_url"]

            # 获取医学报告生成的 metrics 和 report_data
            metrics = result.get("metrics", {})
            report_data = result.get("report_data", None)

            # 获取可下载文件列表和Agent第一轮响应
            download_urls = result.get("download_urls", [])
            first_response = result.get("first_response", None)

            # 构建响应字典
            response_dict = {
                "response": response_text,
                "model_used": model,
                "session_id": session_id,
                "api_name": result.get("api_name"),
                "prediction": result.get("prediction"),
                "cds_result": result.get("cds_result"),
                "nicms_result": result.get("nicms_result"),
                "detected_sequences": result.get("detected_sequences"),
                "frame_urls": frame_urls,
                "image_urls": image_urls,
                "cache_dir": f"/cache/frames/{VERSION}/{session_id}",
                "images_cache_dir": f"/cache/images/{VERSION}/{session_id}",
                "seg_result": seg_result,
                "seg_image_url": seg_image_url,
                "metrics": metrics,
                "report_data": report_data,
                "download_urls": download_urls,
                "correction_workflow": result.get("correction_workflow"),
                "first_response": first_response,
            }

            if metrics and exam_identity:
                finding_evidence = {}
                correction_context = session_mgr.get_metric_correction_context(session_id)
                if correction_context:
                    wall_motion = score_wall_motion(
                        correction_context.get("image_sa_path"),
                        correction_context.get("mask_sa_path"),
                    )
                    if wall_motion:
                        finding_evidence["wall_motion"] = wall_motion
                        response_dict["finding_evidence"] = finding_evidence
                longitudinal_offer = session_mgr.register_exam(
                    session_id=session_id,
                    identity=exam_identity,
                    metrics=normalize_metrics(metrics),
                    detected_sequences=normalize_sequences(
                        result.get("detected_sequences")
                    ),
                    findings=finding_evidence,
                )
                if longitudinal_offer:
                    response_dict["longitudinal_offer"] = longitudinal_offer

            # 保存对话记录为JSON
            uploaded_filenames = [f.filename for f in files if f.filename]
            conv_path = save_conversation_json(
                session_id=session_id,
                user_message=message,
                uploaded_files=uploaded_filenames,
                task_type=task_type or "auto",
                response_data=response_dict,
                round_label=task_type if task_type else "auto",
            )
            if conv_path:
                response_dict["conversation_json"] = f"/api/conversation/{session_id}/{os.path.basename(conv_path)}"

            return JSONResponse(response_dict)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_response = {
                "response": f"处理失败: {str(e)}",
                "model_used": model,
                "session_id": session_id,
                "error": str(e),
            }
            # 即使出错也保存对话记录
            uploaded_filenames = [f.filename for f in files if f.filename]
            save_conversation_json(
                session_id=session_id,
                user_message=message,
                uploaded_files=uploaded_filenames,
                task_type=task_type or "auto",
                response_data=error_response,
                round_label="error",
            )
            return JSONResponse(error_response)

    # ============ 同一患者不同检查的纵向随访比较 ============
    @app.post("/api/longitudinal/compare")
    async def longitudinal_compare(
        session_id: str = Form(...),
        prior_exam_id: str = Form(...),
        current_exam_id: str = Form(...),
    ):
        pair = session_mgr.get_exam_pair(
            session_id, prior_exam_id, current_exam_id
        )
        if not pair:
            return JSONResponse(
                {"error": "The requested follow-up examination pair is unavailable."},
                status_code=404,
            )
        comparison = compare_exams(*pair)
        response_dict = {
            "response": "Longitudinal follow-up comparison completed.",
            "api_name": "Longitudinal Follow-up Comparison",
            "session_id": session_id,
            "longitudinal_comparison": comparison,
        }
        conv_path = save_conversation_json(
            session_id=session_id,
            user_message="Run longitudinal follow-up comparison",
            uploaded_files=[],
            task_type="mr",
            response_data=response_dict,
            round_label="longitudinal_followup",
        )
        if conv_path:
            response_dict["conversation_json"] = (
                f"/api/conversation/{session_id}/{os.path.basename(conv_path)}"
            )
        return JSONResponse(response_dict)

    # ============ 人工修正分割后二次计算 ============
    @app.post("/api/metrics/recalculate")
    async def recalculate_metrics(
        session_id: str = Form(...),
        corrected_4ch: Optional[UploadFile] = File(None),
        corrected_sa: Optional[UploadFile] = File(None),
    ):
        """上传一个或两个修正 mask；未上传的模态沿用本轮自动 mask。"""
        session = session_mgr.get_session(session_id)
        context = session_mgr.get_metric_correction_context(session_id)
        if not session or not context:
            return JSONResponse(
                {"error": "No active segmentation correction workflow for this session."},
                status_code=404,
            )
        if corrected_4ch is None and corrected_sa is None:
            return JSONResponse(
                {"error": "Upload at least one corrected 4CH or SA NIfTI mask."},
                status_code=400,
            )

        for key in ("mask_4ch_path", "mask_sa_path"):
            if not os.path.isfile(context.get(key, "")):
                return JSONResponse(
                    {"error": "The automatic segmentation baseline is no longer available."},
                    status_code=410,
                )

        corrections_dir = os.path.join(CACHE_RESULTS_DIR, session_id, "segmentation")
        os.makedirs(corrections_dir, exist_ok=True)
        correction_id = uuid.uuid4().hex[:10]
        created_paths = []
        validation = {}

        async def save_and_validate(upload: UploadFile, modality: str, baseline_path: str) -> str:
            original_name = upload.filename or ""
            lower_name = original_name.lower()
            if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
                raise SegmentationCorrectionError(
                    f"Corrected {modality.upper()} mask must be a .nii or .nii.gz file."
                )
            suffix = ".nii.gz" if lower_name.endswith(".nii.gz") else ".nii"
            output_path = os.path.join(
                corrections_dir, f"corrected_{modality}_{correction_id}{suffix}"
            )
            with open(output_path, "wb") as target:
                shutil.copyfileobj(upload.file, target)
            created_paths.append(output_path)
            if os.path.getsize(output_path) == 0:
                raise SegmentationCorrectionError(
                    f"Corrected {modality.upper()} mask is empty."
                )
            validation[modality] = validate_corrected_segmentation(
                output_path, baseline_path, modality
            )
            return output_path

        try:
            mask_4ch_path = context["mask_4ch_path"]
            mask_sa_path = context["mask_sa_path"]
            if corrected_4ch is not None:
                mask_4ch_path = await save_and_validate(
                    corrected_4ch, "4ch", context["mask_4ch_path"]
                )
            if corrected_sa is not None:
                mask_sa_path = await save_and_validate(
                    corrected_sa, "sa", context["mask_sa_path"]
                )
        except SegmentationCorrectionError as exc:
            for path in created_paths:
                if os.path.isfile(path):
                    os.remove(path)
            return JSONResponse({"error": str(exc)}, status_code=422)
        except Exception as exc:
            for path in created_paths:
                if os.path.isfile(path):
                    os.remove(path)
            return JSONResponse(
                {"error": f"Failed to store corrected segmentation: {exc}"},
                status_code=500,
            )

        result = agent.recalculate_from_corrected_masks(
            context=context,
            mask_4ch_path=mask_4ch_path,
            mask_sa_path=mask_sa_path,
        )
        if result.get("error_code") != 0 or result.get("error"):
            return JSONResponse(
                {
                    "error": result.get("error", "Corrected metric calculation failed."),
                    "session_id": session_id,
                },
                status_code=502,
            )

        new_context = dict(context)
        new_context["mask_4ch_path"] = mask_4ch_path
        new_context["mask_sa_path"] = mask_sa_path
        session_mgr.set_metric_correction_context(session_id, new_context)

        download_urls = []
        for modality, label, path in [
            ("4ch", "Active 4CH Seg Label", mask_4ch_path),
            ("sa", "Active SA Seg Label", mask_sa_path),
        ]:
            download_urls.append({
                "type": "seg_label",
                "label": label,
                "filename": os.path.basename(path),
                "url": f"/api/download/{session_id}/segmentation/{os.path.basename(path)}",
            })
        active_lge_path = new_context.get("mask_lge_sa_path")
        if active_lge_path and os.path.isfile(active_lge_path):
            download_urls.append({
                "type": "seg_label",
                "label": "Retained LGE SA Seg Label",
                "filename": os.path.basename(active_lge_path),
                "url": f"/api/download/{session_id}/segmentation/{os.path.basename(active_lge_path)}",
            })

        metrics = result.get("metrics", {})
        report_data = result.get("report_data")
        if metrics:
            reports_dir = os.path.join(CACHE_RESULTS_DIR, session_id, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            report_path = os.path.join(
                reports_dir, f"cardiac_report_corrected_{correction_id}.pdf"
            )
            try:
                generated_report = generate_cardiac_report_pdf(
                    metrics=metrics,
                    report_data=report_data,
                    output_path=report_path,
                )
                if generated_report and os.path.isfile(generated_report):
                    report_filename = os.path.basename(generated_report)
                    download_urls.append({
                        "type": "report_pdf",
                        "label": "Corrected Cardiac Report",
                        "filename": report_filename,
                        "url": f"/api/download/{session_id}/reports/{report_filename}",
                    })
            except Exception as exc:
                print(f"Corrected report generation failed: {exc}")

        response_dict = {
            "response": (
                "Corrected segmentation accepted. Cardiac metrics were recalculated "
                "without rerunning automatic segmentation. Existing CDS/NICMS results, "
                "if shown, were retained and not rerun."
            ),
            "api_name": "Corrected Segmentation Recalculation",
            "session_id": session_id,
            "metrics": metrics,
            "report_data": report_data,
            "cds_result": result.get("cds_result"),
            "nicms_result": result.get("nicms_result"),
            "download_urls": download_urls,
            "correction_validation": validation,
            "correction_workflow": agent.build_metric_correction_workflow(
                session_id, new_context
            ),
        }
        uploaded_names = [
            upload.filename
            for upload in (corrected_4ch, corrected_sa)
            if upload is not None and upload.filename
        ]
        conv_path = save_conversation_json(
            session_id=session_id,
            user_message="Upload corrected segmentation and recalculate metrics",
            uploaded_files=uploaded_names,
            task_type="mr",
            response_data=response_dict,
            round_label="corrected_segmentation_recalculation",
        )
        if conv_path:
            response_dict["conversation_json"] = (
                f"/api/conversation/{session_id}/{os.path.basename(conv_path)}"
            )
        return JSONResponse(response_dict)

    # ============ 文件下载接口 ============
    @app.get("/api/download/{session_id}/{file_type}/{filename}")
    async def download_file(session_id: str, file_type: str, filename: str):
        """
        通用文件下载接口

        支持下载类型:
        - nifti: 原始NIfTI文件 (results/{session_id}/nifti/)
        - segmentation: 分割标签NIfTI (results/{session_id}/segmentation/)
        - reports: PDF报告 (results/{session_id}/reports/)
        """
        # 安全检查：防止路径遍历
        safe_filename = os.path.basename(filename)
        if safe_filename != filename or ".." in filename:
            return JSONResponse({"error": "Invalid filename"}, status_code=400)

        valid_types = {"nifti", "segmentation", "reports"}
        if file_type not in valid_types:
            return JSONResponse({"error": f"Invalid file type. Must be one of: {valid_types}"}, status_code=400)

        file_path = os.path.join(CACHE_RESULTS_DIR, session_id, file_type, safe_filename)

        if not os.path.exists(file_path):
            return JSONResponse({"error": f"File not found: {safe_filename}"}, status_code=404)

        # 确定MIME类型
        if safe_filename.endswith(".pdf"):
            media_type = "application/pdf"
        elif safe_filename.endswith(".txt"):
            media_type = "text/plain"
        elif safe_filename.endswith(".nii.gz"):
            media_type = "application/gzip"
        elif safe_filename.endswith(".nii"):
            media_type = "application/octet-stream"
        else:
            media_type = "application/octet-stream"

        return FileResponse(
            path=file_path,
            filename=safe_filename,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={safe_filename}"}
        )

    @app.get("/api/download/list/{session_id}")
    async def list_downloadable_files(session_id: str):
        """列出会话的所有可下载文件"""
        results_dir = os.path.join(CACHE_RESULTS_DIR, session_id)
        if not os.path.exists(results_dir):
            return {"files": []}

        files = []
        for file_type in ["nifti", "segmentation", "reports"]:
            type_dir = os.path.join(results_dir, file_type)
            if os.path.exists(type_dir):
                for fname in os.listdir(type_dir):
                    fpath = os.path.join(type_dir, fname)
                    if os.path.isfile(fpath):
                        files.append({
                            "type": file_type,
                            "filename": fname,
                            "size": os.path.getsize(fpath),
                            "url": f"/api/download/{session_id}/{file_type}/{fname}",
                        })

        return {"session_id": session_id, "files": files}

    # ============ 对话记录接口 ============
    @app.get("/api/conversation/{session_id}")
    async def list_conversations(session_id: str):
        """列出会话的所有对话记录JSON文件"""
        conv_dir = os.path.join(CACHE_CONVERSATIONS_DIR, session_id)
        if not os.path.exists(conv_dir):
            return {"session_id": session_id, "conversations": []}

        conversations = []
        for fname in sorted(os.listdir(conv_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(conv_dir, fname)
                conversations.append({
                    "filename": fname,
                    "size": os.path.getsize(fpath),
                    "url": f"/api/conversation/{session_id}/{fname}",
                    "created": os.path.getmtime(fpath),
                })

        return {"session_id": session_id, "conversations": conversations}

    @app.get("/api/conversation/{session_id}/{filename}")
    async def get_conversation(session_id: str, filename: str):
        """获取单个对话记录JSON文件"""
        safe_filename = os.path.basename(filename)
        if safe_filename != filename or ".." in filename:
            return JSONResponse({"error": "Invalid filename"}, status_code=400)

        conv_path = os.path.join(CACHE_CONVERSATIONS_DIR, session_id, safe_filename)
        if not os.path.exists(conv_path):
            return JSONResponse({"error": "Conversation file not found"}, status_code=404)

        with open(conv_path, "r", encoding="utf-8") as f:
            conv_data = json.load(f)

        return JSONResponse(conv_data)

    @app.get("/api/conversation/{session_id}/download/{filename}")
    async def download_conversation(session_id: str, filename: str):
        """下载单个对话记录JSON文件"""
        safe_filename = os.path.basename(filename)
        if safe_filename != filename or ".." in filename:
            return JSONResponse({"error": "Invalid filename"}, status_code=400)

        conv_path = os.path.join(CACHE_CONVERSATIONS_DIR, session_id, safe_filename)
        if not os.path.exists(conv_path):
            return JSONResponse({"error": "Conversation file not found"}, status_code=404)

        return FileResponse(
            path=conv_path,
            filename=safe_filename,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={safe_filename}"}
        )

    @app.get("/api/conversation/{session_id}/download_all")
    async def download_all_conversations(session_id: str):
        """下载会话的所有对话记录（合并为单个JSON文件）"""
        conv_dir = os.path.join(CACHE_CONVERSATIONS_DIR, session_id)
        if not os.path.exists(conv_dir):
            return JSONResponse({"error": "No conversations found"}, status_code=404)

        all_conversations = []
        for fname in sorted(os.listdir(conv_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(conv_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    all_conversations.append(json.load(f))

        if not all_conversations:
            return JSONResponse({"error": "No conversations found"}, status_code=404)

        merged = {
            "session_id": session_id,
            "total_conversations": len(all_conversations),
            "exported_at": datetime.now().isoformat(),
            "conversations": all_conversations,
        }

        merged_filename = f"conversations_{session_id}.json"
        merged_path = os.path.join(conv_dir, merged_filename)
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        return FileResponse(
            path=merged_path,
            filename=merged_filename,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={merged_filename}"}
        )

    return app
