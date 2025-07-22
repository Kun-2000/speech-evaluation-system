"""
文字比對分析服務
完全使用 OpenAI GPT API，並透過範例引導 AI 進行更精確的判斷。
"""

import json
import logging
import re

from openai import APIError, OpenAI

from ..config import EVALUATION_CONFIG, OPENAI_LLM_CONFIG

logger = logging.getLogger(__name__)


class TextProcessor:
    """文字預處理工具"""

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        文字正規化處理。
        """
        if not text:
            return ""

        text = str(text).strip()

        if EVALUATION_CONFIG.get("text_normalization", True):
            text = re.sub(r"\s+", " ", text)
            text = text.replace("，", ",").replace("。", ".").replace("？", "?")
            text = text.replace("！", "!").replace("：", ":").replace("；", ";")
            text = text.replace("「", '"').replace("」", '"')
            text = text.replace("『", "'").replace("』", "'")

        if EVALUATION_CONFIG.get("punctuation_ignore", True):
            text = re.sub(r"[.,!?;:()\"'-]", "", text)

        if not EVALUATION_CONFIG.get("case_sensitive", False):
            text = text.lower()

        return text.strip()


class LLMService:
    """封裝 OpenAI GPT 服務"""

    def __init__(self):
        """初始化 LLM 服務客戶端"""
        if not OPENAI_LLM_CONFIG["api_key"]:
            raise RuntimeError("缺少 OPENAI_API_KEY")

        self.client = OpenAI(api_key=OPENAI_LLM_CONFIG["api_key"])
        self.model = OPENAI_LLM_CONFIG["model"]
        self.temperature = OPENAI_LLM_CONFIG["temperature"]
        self.max_tokens = OPENAI_LLM_CONFIG["max_tokens"]
        self.top_p = OPENAI_LLM_CONFIG["top_p"]
        self.text_processor = TextProcessor()

        logger.info("✅ OpenAI GPT 文字比對分析師初始化成功")

    def compare_text_accuracy(self, transcribed_text: str, reference_text: str) -> dict:
        """
        比對轉錄文字與標準文本的準確性。
        如果 AI 分析失敗，將會拋出 RuntimeError。
        """
        if not transcribed_text or not transcribed_text.strip():
            raise ValueError("轉錄文字不能為空")
        if not reference_text or not reference_text.strip():
            raise ValueError("標準文本不能為空")

        try:
            norm_transcribed = self.text_processor.normalize_text(transcribed_text)
            norm_reference = self.text_processor.normalize_text(reference_text)

            prompt = self._build_comparison_prompt(norm_reference, norm_transcribed)
            response_text = self._call_openai_api(prompt)
            result = self._parse_comparison_response(response_text)

            logger.info(
                "✅ 分析完成 - 準確率: %.1f%%",
                result.get("accuracy_score", 0),
            )
            return result

        except (APIError, RuntimeError, ValueError) as e:
            logger.error("❌ 分析因 AI 錯誤而失敗: %s", e)
            raise RuntimeError(f"AI 分析過程發生錯誤: {e}") from e

    def _build_comparison_prompt(
        self, reference_text: str, transcribed_text: str
    ) -> str:
        """
        構建包含範例的 "少樣本提示 (Few-Shot Prompt)"
        """
        # 範例一：完全不匹配
        example1_ref = "今天天氣真好"
        example1_trans = "請投入適量衣物"
        example1_json = json.dumps(
            {
                "summary": "轉錄結果與標準文本完全不相關。",
                "accuracy_score": 0,
                "semantic_similarity": 0,
                "error_analysis": {
                    "substitutions": 0,
                    "deletions": 6,
                    "insertions": 7,
                    "total_errors": 13,
                },
                "key_differences": ["內容完全不同"],
                "suggestions": ["請確認音檔內容是否正確"],
                "reasoning": "轉錄文字與標準文本在主題和內容上沒有任何關聯，無法進行有效比對。錯誤數是基於刪除所有標準文本並插入所有轉錄文本計算得出。",
            },
            ensure_ascii=False,
            indent=2,
        )

        # 範例二：部分匹配
        example2_ref = "我喜歡吃蘋果"
        example2_trans = "我喜歡吃蘋安"
        example2_json = json.dumps(
            {
                "summary": "轉錄結果基本正確，但在'果'字上出現同音異字錯誤。",
                "accuracy_score": 80,
                "semantic_similarity": 85,
                "error_analysis": {
                    "substitutions": 1,
                    "deletions": 0,
                    "insertions": 0,
                    "total_errors": 1,
                },
                "key_differences": ["'果'被錯寫為'安'"],
                "suggestions": ["加強對同音異字的辨識模型"],
                "reasoning": "主體語意正確，但存在一個字的替換錯誤，屬於常見的同音字問題。",
            },
            ensure_ascii=False,
            indent=2,
        )

        return f"""
你是專業的語音識別品質評估分析師。請根據我給的範例，比對【待分析文本】並只回傳 JSON 格式的評估結果。

---
【範例一】
[輸入]
標準文本: {example1_ref}
轉錄文字: {example1_trans}
[輸出JSON]
{example1_json}
---
【範例二】
[輸入]
標準文本: {example2_ref}
轉錄文字: {example2_trans}
[輸出JSON]
{example2_json}
---

現在，請根據以上範例的邏輯和格式，評估以下文本：

【待分析文本】
[輸入]
標準文本: {reference_text}
轉錄文字: {transcribed_text}
[輸出JSON]
"""

    def _call_openai_api(self, prompt: str, retry_count: int = 0) -> str:
        """呼叫 OpenAI API，包含重試機制"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是專業的語音識別品質評估分析師，擅長中文語音轉錄準確性分析。請提供客觀、專業的評估結果。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            return response.choices[0].message.content.strip()
        except APIError as e:
            if retry_count < 2:
                logger.warning(
                    "OpenAI API 呼叫失敗，重試中 (%d/2): %s", retry_count + 1, e
                )
                return self._call_openai_api(prompt, retry_count + 1)
            raise RuntimeError("OpenAI API 呼叫失敗") from e

    def _parse_comparison_response(self, response_text: str) -> dict:
        """解析 LLM 回應，包含清理和結構驗證"""
        try:
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]

            result = json.loads(cleaned_text.strip())

            # 確保回傳的字典結構完整，避免因 AI 遺漏欄位導致錯誤
            required_fields = {
                "summary": "分析完成",
                "accuracy_score": 0.0,
                "semantic_similarity": 0.0,
                "error_analysis": {
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": 0,
                    "total_errors": 0,
                },
                "key_differences": [],
                "suggestions": [],
                "reasoning": "",
            }
            for field, default in required_fields.items():
                result.setdefault(field, default)
            if not isinstance(result["error_analysis"], dict):
                result["error_analysis"] = required_fields["error_analysis"]

            # 驗證分數範圍
            result["accuracy_score"] = max(
                0, min(100, float(result.get("accuracy_score", 0)))
            )
            result["semantic_similarity"] = max(
                0, min(100, float(result.get("semantic_similarity", 0)))
            )

            return result
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("JSON 解析失敗: %s", e)
            raise RuntimeError(f"無法解析LLM的回應: {response_text}") from e


# pylint: disable=invalid-name
llm_service = None
try:
    llm_service = LLMService()
    logger.info("✅ LLM 全域服務初始化成功 (少樣本提示模式)")
except (RuntimeError, ValueError) as e:
    logger.warning("❌ 文字比對分析服務初始化失敗: %s", e)


def compare_text_accuracy(transcribed_text: str, reference_text: str) -> dict:
    """
    公開的服務函式接口，用於比對轉錄文字與標準文本的準確性。
    """
    if not llm_service:
        raise RuntimeError("文字比對分析服務不可用")

    return llm_service.compare_text_accuracy(transcribed_text, reference_text)
