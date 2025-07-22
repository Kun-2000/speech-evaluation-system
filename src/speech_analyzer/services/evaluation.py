"""
語音轉錄評估系統核心模組
整合 STT、LLM、報告生成功能
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Union
import uuid

from .stt import transcribe_audio
from .llm import compare_text_accuracy
from ..config import EVALUATION_CONFIG, get_evaluation_thresholds

logger = logging.getLogger(__name__)


class EvaluationResult:
    """評估結果數據類"""

    def __init__(self):
        """初始化評估結果物件"""
        self.evaluation_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.audio_file = ""
        self.reference_text = ""
        self.transcription = {}
        self.comparison = {}
        self.evaluation_metrics = {}
        self.processing_time = 0.0
        self.success = False
        self.error_message = None

    def to_dict(self) -> dict:
        """將結果物件轉換為字典格式"""
        return {
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp,
            "audio_file": self.audio_file,
            "reference_text": self.reference_text,
            "transcription": self.transcription,
            "comparison": self.comparison,
            "evaluation_metrics": self.evaluation_metrics,
            "processing_time": self.processing_time,
            "success": self.success,
            "error_message": self.error_message,
        }


class EvaluationService:
    """評估服務核心類"""

    def __init__(self):
        """初始化評估服務"""
        self.config = EVALUATION_CONFIG
        logger.info("✅ 評估服務初始化成功")

    def evaluate_single_file(
        self, audio_file_path: Union[str, Path], reference_text: str
    ) -> EvaluationResult:
        """評估單個音頻檔案"""
        result = EvaluationResult()
        start_time = time.time()

        try:
            result.audio_file = str(audio_file_path)
            result.reference_text = reference_text

            logger.info("🔄 開始評估檔案: %s", Path(audio_file_path).name)

            transcript, confidence = transcribe_audio(audio_file_path)

            result.transcription = {
                "file_path": str(audio_file_path),
                "file_name": Path(audio_file_path).name,
                "transcript": transcript,
                "confidence": confidence,
                "success": True,
                "error": None,
            }

            logger.info(
                "✅ 語音轉錄完成: %s",
                transcript[:30] + "..." if len(transcript) > 30 else transcript,
            )

            comparison = compare_text_accuracy(transcript, reference_text)
            comparison["success"] = True
            result.comparison = comparison

            logger.info(
                "✅ 文字比對完成 - 準確率: %.1f%%", comparison.get("accuracy_score", 0)
            )

            result.evaluation_metrics = self._calculate_evaluation_metrics(
                comparison, confidence
            )

            result.success = True

        except (RuntimeError, ValueError) as e:
            logger.error("❌ 檔案評估失敗: %s", e, exc_info=True)
            result.error_message = str(e)
            result.success = False

            if not result.transcription:
                result.transcription = {
                    "file_path": str(audio_file_path),
                    "file_name": Path(audio_file_path).name,
                    "transcript": None,
                    "confidence": 0.0,
                    "success": False,
                    "error": str(e),
                }

            if not result.comparison:
                result.comparison = {
                    "summary": "評估失敗",
                    "accuracy_score": 0.0,
                    "semantic_similarity": 0.0,
                    "error_analysis": {
                        "substitutions": 0,
                        "deletions": 0,
                        "insertions": 0,
                        "total_errors": 0,
                    },
                    "key_differences": [],
                    "suggestions": ["檢查音頻檔案", "確認網路連線"],
                    "reasoning": f"評估過程發生錯誤: {str(e)}",
                    "success": False,
                    "error": str(e),
                }

            result.evaluation_metrics = {
                "accuracy_score": 0.0,
                "semantic_similarity": 0.0,
                "accuracy_level": "評估失敗",
                "confidence": 0.0,
                "processing_status": "failed",
            }

        finally:
            result.processing_time = round(time.time() - start_time, 2)
            logger.info("📊 檔案評估完成，用時 %.2f 秒", result.processing_time)

        return result

    def _calculate_evaluation_metrics(
        self, comparison: dict, confidence: float
    ) -> dict:
        """計算評估指標"""
        accuracy_score = comparison.get("accuracy_score", 0)
        semantic_similarity = comparison.get("semantic_similarity", 0)

        thresholds = get_evaluation_thresholds()
        if accuracy_score >= thresholds["excellent"]:
            accuracy_level = "優秀"
        elif accuracy_score >= thresholds["good"]:
            accuracy_level = "良好"
        elif accuracy_score >= thresholds["fair"]:
            accuracy_level = "普通"
        else:
            accuracy_level = "需要改進"

        return {
            "accuracy_score": accuracy_score,
            "semantic_similarity": semantic_similarity,
            "accuracy_level": accuracy_level,
            "confidence": confidence,
            "processing_status": "completed",
        }


# pylint: disable=invalid-name
evaluation_service = None
try:
    evaluation_service = EvaluationService()
    logger.info("✅ 評估全域服務初始化成功")

except (RuntimeError, ValueError) as e:
    logger.warning("❌ 評估服務初始化失敗: %s", e)


def evaluate_single_file(
    audio_file_path: Union[str, Path], reference_text: str
) -> EvaluationResult:
    """
    評估單個音頻檔案的轉錄品質
    """
    if not evaluation_service:
        raise RuntimeError("評估服務不可用，請檢查系統配置")

    return evaluation_service.evaluate_single_file(audio_file_path, reference_text)
