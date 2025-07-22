"""
語音轉錄評估系統配置模組
"""

import os
import secrets
from pathlib import Path

# === 載入環境變數 ===
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

# === 基礎設定 ===
BASE_DIR = Path(__file__).parent.parent.parent

# 建立必要資料夾
for folder in ["data/uploads", "data/temp", "data/logs"]:
    (BASE_DIR / folder).mkdir(exist_ok=True)

# === 服務模式配置 ===
STT_MODE = os.environ.get("STT_MODE", "openai").lower()
LLM_MODE = os.environ.get("LLM_MODE", "openai").lower()

# === OpenAI 配置 ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def get_whisper_language():
    """取得 Whisper 語言設定 (此功能保留，因對 STT 仍有用)"""
    lang_setting = os.environ.get("OPENAI_STT_LANGUAGE", "auto")
    return None if lang_setting.lower() == "auto" else lang_setting


OPENAI_STT_CONFIG = {
    "api_key": OPENAI_API_KEY,
    "model": os.environ.get("OPENAI_STT_MODEL", "whisper-1"),
    "language": get_whisper_language(),
    "response_format": os.environ.get("OPENAI_STT_RESPONSE_FORMAT", "json"),
    "temperature": float(os.environ.get("OPENAI_STT_TEMPERATURE", "0")),
}

OPENAI_LLM_CONFIG = {
    "api_key": OPENAI_API_KEY,
    "model": os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini"),
    "temperature": float(os.environ.get("OPENAI_LLM_TEMPERATURE", "0.3")),
    "max_tokens": int(os.environ.get("OPENAI_LLM_MAX_TOKENS", "800")),
    "top_p": float(os.environ.get("OPENAI_LLM_TOP_P", "0.9")),
}

# === 評估系統配置 ===
EVALUATION_CONFIG = {
    "text_normalization": os.environ.get("TEXT_NORMALIZATION", "true").lower()
    == "true",
    "punctuation_ignore": os.environ.get("PUNCTUATION_IGNORE", "true").lower()
    == "true",
    "case_sensitive": os.environ.get("CASE_SENSITIVE", "false").lower() == "true",
    "min_similarity_threshold": float(os.environ.get("MIN_SIMILARITY_THRESHOLD", "60")),
    "high_accuracy_threshold": float(os.environ.get("HIGH_ACCURACY_THRESHOLD", "90")),
    "detailed_error_analysis": os.environ.get("DETAILED_ERROR_ANALYSIS", "true").lower()
    == "true",
    "include_suggestions": os.environ.get("INCLUDE_SUGGESTIONS", "true").lower()
    == "true",
}


# === Flask 配置 ===
def get_secret_key():
    env_key = os.environ.get("SECRET_KEY")
    if env_key and env_key != "auto-generate":
        return env_key
    secret_file = BASE_DIR / ".secret_key"
    if secret_file.exists():
        return secret_file.read_text().strip()
    new_key = secrets.token_hex(32)
    secret_file.write_text(new_key)
    try:
        secret_file.chmod(0o600)
    except OSError:
        pass
    return new_key


FLASK_CONFIG = {
    "SECRET_KEY": get_secret_key(),
    "MAX_CONTENT_LENGTH": int(os.environ.get("MAX_FILE_SIZE", "100")) * 1024 * 1024,
    "UPLOAD_EXTENSIONS": [
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".ogg",
        ".webm",
        ".flac",
        ".mp4",
        ".avi",
        ".mov",
    ],
    "DEBUG": os.environ.get("FLASK_DEBUG", "false").lower() == "true",
}

# === 日誌配置 ===
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOGGING_CONFIG = {
    "level": LOG_LEVEL,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": BASE_DIR / "data" / "logs" / "app.log",
}

# === 路徑常數 ===
UPLOAD_FOLDER = "data/uploads"
TEMP_FOLDER = "data/temp"


# === 配置驗證 ===
def validate_config():
    """驗證必要的配置是否存在"""
    errors = []
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY 未設定")
    if not (0 <= EVALUATION_CONFIG["min_similarity_threshold"] <= 100):
        errors.append("MIN_SIMILARITY_THRESHOLD 必須在 0-100 之間")
    if not (0 <= EVALUATION_CONFIG["high_accuracy_threshold"] <= 100):
        errors.append("HIGH_ACCURACY_THRESHOLD 必須在 0-100 之間")
    if errors:
        raise ValueError(f"配置錯誤: {', '.join(errors)}")
    return True


def get_evaluation_thresholds():
    """取得評估等級門檻"""
    return {
        "excellent": EVALUATION_CONFIG["high_accuracy_threshold"],
        "good": max(75, EVALUATION_CONFIG["min_similarity_threshold"]),
        "fair": EVALUATION_CONFIG["min_similarity_threshold"],
        "poor": 0,
    }
