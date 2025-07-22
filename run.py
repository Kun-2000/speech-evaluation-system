"""
語音轉錄評估系統啟動點
"""

import logging
import sys
from pathlib import Path

from src.speech_analyzer.app import app
from src.speech_analyzer.config import (
    FLASK_CONFIG,
    STT_MODE,
    LLM_MODE,
    validate_config,
    BASE_DIR,
    UPLOAD_FOLDER,
)

logger = logging.getLogger(__name__)


def cleanup_temp_files():
    """清理臨時檔案"""
    try:
        temp_folder = Path(BASE_DIR) / "data" / "temp"
        upload_folder = Path(BASE_DIR) / UPLOAD_FOLDER

        files_to_clean = list(temp_folder.glob("*")) + list(upload_folder.glob("*"))
        cleanup_count = 0

        for file_path in files_to_clean:
            if file_path.is_file():
                try:
                    file_path.unlink()
                    cleanup_count += 1
                except (IOError, OSError) as e:
                    logger.debug("清理臨時檔案失敗: %s", e)

        if cleanup_count > 0:
            logger.info("🧹 啟動時清理了 %d 個臨時檔案", cleanup_count)

    except (IOError, OSError) as e:
        logger.warning("啟動時清理檔案失敗: %s", e)


def validate_environment():
    """驗證環境配置"""
    try:
        validate_config()
        logger.info("✅ 環境配置驗證通過")
        return True
    except ValueError as e:
        logger.error("❌ 環境配置驗證失敗: %s", e)
        return False


def print_startup_info():
    """顯示簡潔的啟動資訊"""
    logger.info("🚀 台灣語音轉錄評估系統啟動")
    logger.info("📊 STT 模式: %s | LLM 模式: %s", STT_MODE, LLM_MODE)
    logger.info("🌐 伺服器運行於 http://127.0.0.1:5000")

    print("\n" + "=" * 50)
    print("🎤 台灣語音轉錄評估系統已啟動")
    print("✅ Web 介面 -> http://localhost:5000")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        if not validate_environment():
            logger.error("❌ 環境配置不正確，無法啟動應用程式")
            print("\n💡 請檢查 .env 檔案中的 OPENAI_API_KEY 是否設定正確。")
            sys.exit(1)

        cleanup_temp_files()

        print_startup_info()

        app.run(debug=FLASK_CONFIG["DEBUG"], host="0.0.0.0", port=5000)

    except KeyboardInterrupt:
        logger.info("🛑 台灣語音轉錄評估系統已停止")
    except (ImportError, RuntimeError) as e:
        logger.error("❌ 應用程式啟動失敗: %s", e)
        print(f"\n💥 啟動失敗: {e}")
        print("\n🔧 請檢查您的 Python 環境和相依套件是否已正確安裝。")
        sys.exit(1)
