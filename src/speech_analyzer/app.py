"""
語音轉錄評估系統 Flask 應用程式
"""

import logging
from datetime import datetime
from pathlib import Path
from functools import wraps
from flask import Flask, request, jsonify, render_template

# 導入服務函數
from .services.stt import (
    stop_recording,
    start_recording,
    is_recording,
    get_recording_duration,
)
from .services.evaluation import evaluate_single_file

from .config import (
    BASE_DIR,
    UPLOAD_FOLDER,
    FLASK_CONFIG,
    LOGGING_CONFIG,
    STT_MODE,
    LLM_MODE,
)

# pylint: disable=invalid-name
app = Flask(__name__, template_folder="../../templates")
app.config.update(FLASK_CONFIG)

# 日誌設定
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[logging.FileHandler(LOGGING_CONFIG["log_file"]), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# 裝飾器
def api_response(func):
    """統一 API 回應格式"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                data, status_code = result
            else:
                data, status_code = result, 200

            # 確保 data 是可序列化的字典
            if not isinstance(data, dict):
                raise TypeError("API endpoint must return a dictionary.")

            return (
                jsonify(
                    {"success": True, "timestamp": datetime.now().isoformat(), **data}
                ),
                status_code,
            )
        except (ValueError, RuntimeError, TypeError) as e:
            logger.error("%s 失敗: %s", func.__name__, e, exc_info=True)
            return (
                jsonify(
                    {
                        "success": False,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                    }
                ),
                500,
            )

    return wrapper


def validate_audio_file(func):
    """驗證上傳的音檔"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "audio" not in request.files:
            return {"error": "未提供音檔"}, 400
        file = request.files["audio"]
        if not file.filename or "." not in file.filename:
            return {"error": "無效檔案名稱"}, 400
        ext = file.filename.rsplit(".", 1)[1].lower()
        if ext not in [e.strip(".") for e in FLASK_CONFIG["UPLOAD_EXTENSIONS"]]:
            return {"error": "不支援的檔案格式"}, 400
        return func(file, *args, **kwargs)

    return wrapper


def save_uploaded_audio(file) -> Path:
    """儲存上傳的音訊檔案"""
    if not file.filename:
        raise ValueError("檔案名稱為空")

    logger.info("📤 收到音訊檔案: %s", file.filename)
    ext_part = file.filename.rsplit(".", 1)[1].lower()

    safe_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_audio.{ext_part}"
    filepath = Path(BASE_DIR) / UPLOAD_FOLDER / safe_filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    file.save(filepath)
    file_size = filepath.stat().st_size

    if file_size > app.config["MAX_CONTENT_LENGTH"]:
        filepath.unlink()
        raise ValueError(f"檔案過大: {file_size / 1024 / 1024:.1f}MB，超過限制")

    logger.info("✅ 音訊檔案已儲存: %s (大小: %d bytes)", safe_filename, file_size)
    return filepath


# === 評估系統主要端點 ===


@app.route("/evaluation/analyze", methods=["POST"])
@api_response
@validate_audio_file
def analyze_complete(audio_file):
    """完整評估流程：從上傳的音訊檔案"""
    logger.info("從檔案上傳啟動完整評估流程")

    reference_text = request.form.get("reference_text", "").strip()
    if not reference_text:
        return {"error": "請提供標準文本 (reference_text 欄位)"}, 400

    filepath = save_uploaded_audio(audio_file)
    try:
        # 呼叫統一的評估服務
        result = evaluate_single_file(filepath, reference_text)
        logger.info("✅ 檔案上傳評估流程完成")
        return result.to_dict()
    finally:
        # 確保上傳的暫存檔案被刪除
        if filepath.exists():
            filepath.unlink()
            logger.debug("已清理上傳的臨時音訊檔案: %s", filepath)


# === 即時錄音端點 ===


@app.route("/recording/start", methods=["POST"])
@api_response
def start_recording_api():
    """開始即時錄音"""
    if start_recording():
        return {"message": "錄音已開始", "recording": True}
    return {"error": "無法開始錄音，請檢查麥克風設定"}, 400


@app.route("/recording/stop", methods=["POST"])
@api_response
def stop_recording_api():
    """停止錄音並執行【完整評估流程】"""
    logger.info("從錄音啟動完整評估流程")

    # 停止錄音並取得暫存檔案路徑
    audio_file_path_str = stop_recording()
    if not audio_file_path_str:
        return {"error": "沒有錄音資料或錄音失敗"}, 400

    # 從請求中取得標準文本
    data = request.get_json(silent=True) or {}
    reference_text = data.get("reference_text", "").strip()
    if not reference_text:
        return {"error": "缺少標準參考文本"}, 400

    filepath = Path(audio_file_path_str)
    try:
        # 【統一邏輯】直接呼叫與檔案上傳相同的完整評估服務
        result = evaluate_single_file(filepath, reference_text)
        logger.info("✅ 從錄音啟動的完整評估流程完成")
        return result.to_dict()
    finally:
        # 確保錄音的暫存檔案在處理完後被刪除
        if filepath.exists():
            filepath.unlink()
            logger.debug("已清理錄音的臨時音訊檔案: %s", filepath)


@app.route("/recording/status", methods=["GET"])
@api_response
def recording_status():
    """取得錄音狀態"""
    return {"recording": is_recording(), "duration": get_recording_duration()}


# === 系統狀態端點 ===


@app.route("/")
def index():
    """渲染主頁面"""
    return render_template("index.html")


@app.route("/status")
def status():
    """提供應用程式狀態資訊"""
    return jsonify(
        {
            "app_status": "running",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",  # 版本升級
            "services": {"stt": STT_MODE, "llm": LLM_MODE},
        }
    )


@app.route("/health")
def health_check():
    """提供健康檢查端點"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


# === 錯誤處理 ===
@app.errorhandler(413)
def file_too_large(_error):
    """處理檔案過大的錯誤 (413)"""
    max_size_mb = app.config["MAX_CONTENT_LENGTH"] / 1024 / 1024
    return (
        jsonify(
            {
                "success": False,
                "error": f"檔案過大，請上傳小於 {max_size_mb:.0f}MB 的音訊檔案",
            }
        ),
        413,
    )


@app.errorhandler(500)
def internal_error(error):
    """處理內部伺服器錯誤 (500)"""
    logger.error("伺服器內部錯誤: %s", error, exc_info=True)
    return jsonify({"success": False, "error": "內部伺服器錯誤"}), 500


@app.errorhandler(404)
def not_found(_error):
    """處理找不到資源的錯誤 (404)"""
    return jsonify({"success": False, "error": "找不到請求的資源"}), 404
