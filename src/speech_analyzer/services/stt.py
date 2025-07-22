"""
語音轉文字服務
使用 OpenAI Whisper API + 即時錄音功能
"""

import logging
import os
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import pyaudio
from openai import APIError, OpenAI

from ..config import OPENAI_STT_CONFIG, STT_MODE

logger = logging.getLogger(__name__)


class AudioRecorder:
    """即時錄音功能"""

    def __init__(self):
        """初始化錄音器參數"""
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000

        temp_audio = pyaudio.PyAudio()
        self.sample_width = temp_audio.get_sample_size(self.sample_format)
        temp_audio.terminate()

        self.is_recording = False
        self.audio_data = []
        self.audio = None
        self.stream = None
        self.record_thread = None

        logger.info("✅ 音頻錄音器初始化成功")

    def start_recording(self) -> bool:
        """開始錄音"""
        if self.is_recording:
            logger.warning("錄音已在進行中")
            return False

        try:
            self.audio = pyaudio.PyAudio()
            if self.audio.get_device_count() == 0:
                raise RuntimeError("未找到音頻設備")

            self.stream = self.audio.open(
                format=self.sample_format,
                channels=self.channels,
                rate=self.sample_rate,
                frames_per_buffer=self.chunk,
                input=True,
            )

            self.audio_data = []
            self.is_recording = True

            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.daemon = True
            self.record_thread.start()

            logger.info("🎤 開始錄音...")
            return True

        except (RuntimeError, IOError) as e:
            logger.error("❌ 開始錄音失敗: %s", e)
            self._cleanup_audio()
            return False

    def stop_recording(self) -> Optional[str]:
        """停止錄音並儲存檔案"""
        if not self.is_recording:
            logger.warning("目前沒有在錄音")
            return None

        try:
            self.is_recording = False
            if self.record_thread and self.record_thread.is_alive():
                self.record_thread.join(timeout=2.0)

            if not self.audio_data:
                logger.warning("沒有錄音資料")
                self._cleanup_audio()
                return None

            temp_file = self._save_audio_to_file()
            logger.info("🛑 錄音停止，檔案儲存: %s", temp_file)

            self._cleanup_audio()
            return temp_file

        except (RuntimeError, IOError, InterruptedError) as e:
            logger.error("❌ 停止錄音失敗: %s", e)
            self._cleanup_audio()
            return None

    def _record_audio(self):
        """錄音執行緒函數"""
        try:
            while self.is_recording and self.stream:
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.audio_data.append(data)
                except IOError as e:
                    logger.error("錄音資料讀取錯誤: %s", e)
                    break

        # pylint: disable=broad-except
        except Exception as e:
            logger.error("錄音執行緒錯誤: %s", e)

    def _save_audio_to_file(self) -> str:
        """將錄音資料儲存為 WAV 檔案"""
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="recording_")
        os.close(temp_fd)

        try:
            with wave.open(temp_path, "wb") as wf:
                # pylint: disable=no-member
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(self.audio_data))
            return temp_path
        except (IOError, wave.Error) as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise RuntimeError(f"儲存音頻檔案失敗: {e}") from e

    def _cleanup_audio(self):
        """清理音頻資源"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            if self.audio:
                self.audio.terminate()
                self.audio = None
        except (IOError, AttributeError) as e:
            logger.debug("清理音頻資源時發生錯誤: %s", e)

    def get_recording_duration(self) -> float:
        """取得目前錄音時長（秒）"""
        if not self.is_recording or not self.audio_data:
            return 0.0
        total_frames = len(self.audio_data) * self.chunk
        return total_frames / self.sample_rate

    def __del__(self):
        """解構函數，確保資源清理"""
        if self.is_recording:
            self.stop_recording()
        self._cleanup_audio()


class OpenAISTTClient:
    """OpenAI Whisper Speech-to-Text 服務"""

    def __init__(self):
        """初始化 OpenAI STT 客戶端"""
        if not OPENAI_STT_CONFIG["api_key"]:
            raise RuntimeError("缺少 OPENAI_API_KEY")

        self.client = OpenAI(api_key=OPENAI_STT_CONFIG["api_key"])
        self.model = OPENAI_STT_CONFIG["model"]
        self.language = OPENAI_STT_CONFIG["language"]
        self.response_format = OPENAI_STT_CONFIG["response_format"]
        self.temperature = OPENAI_STT_CONFIG["temperature"]

        logger.info("✅ OpenAI STT 初始化成功")

    def transcribe_audio(self, audio_file_path: Union[str, Path]) -> Tuple[str, float]:
        """
        使用 OpenAI Whisper 進行語音轉文字

        Returns:
            Tuple[str, float]: (轉錄文字, 信心度)
        """
        if not audio_file_path:
            raise ValueError("音頻檔案路徑不能為空")
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音頻檔案不存在: {audio_path}")

        file_size = audio_path.stat().st_size
        max_size = 25 * 1024 * 1024
        if file_size > max_size:
            raise ValueError(
                f"檔案過大: {file_size / 1024 / 1024:.1f}MB，超過 25MB 限制"
            )
        if file_size < 1024:
            raise ValueError("檔案過小，可能沒有有效的音頻內容")

        logger.info("📁 處理檔案: %s (%.1f KB)", audio_path.name, file_size / 1024)

        try:
            with open(audio_file_path, "rb") as audio_file:
                params = {
                    "model": self.model,
                    "file": audio_file,
                    "response_format": self.response_format,
                    "temperature": self.temperature,
                }
                if self.language is not None:
                    params["language"] = self.language
                response = self.client.audio.transcriptions.create(**params)

            transcript = (
                response.text.strip()
                if hasattr(response, "text")
                else str(response).strip()
            )

            if not transcript:
                raise ValueError("無法識別語音內容，檔案可能損壞或不包含語音")

            logger.info("✅ OpenAI STT 識別成功")
            return transcript, 1.0

        except APIError as e:
            logger.error("❌ OpenAI API 錯誤: %s", e)
            raise RuntimeError(f"OpenAI 服務錯誤: {e.message}") from e
        except (IOError, ValueError) as e:
            self._handle_transcription_error(e, audio_path)

    def _handle_transcription_error(self, error: Exception, audio_path: Path):
        """處理轉錄錯誤"""
        error_msg = str(error)
        if "Unrecognized file format" in error_msg:
            logger.error("❌ 檔案格式錯誤: %s", audio_path.suffix)
            raise ValueError("檔案格式不被支援或檔案損壞") from error
        if "file too large" in error_msg.lower():
            logger.error("❌ 檔案過大: %s", audio_path.name)
            raise ValueError("檔案超過 25MB 限制") from error

        logger.error("❌ 檔案處理失敗: %s", error)
        raise RuntimeError(f"語音識別檔案處理失敗: {error}") from error


class STTService:
    """統一的 STT 服務介面"""

    def __init__(self):
        """初始化 STT 服務"""
        self.mode = STT_MODE
        if self.mode == "openai":
            self.client = OpenAISTTClient()
        else:
            raise ValueError(f"不支援的 STT 模式: {self.mode}")

        self.recorder = AudioRecorder()
        logger.info("✅ STT 服務初始化成功")

    def transcribe_audio(self, audio_file_path: Union[str, Path]) -> Tuple[str, float]:
        """語音轉文字"""
        if not audio_file_path:
            raise ValueError("音頻檔案路徑不能為空")
        try:
            return self.client.transcribe_audio(audio_file_path)
        except (RuntimeError, ValueError) as e:
            logger.error("❌ 語音識別失敗: %s", e)
            raise RuntimeError(f"語音識別失敗: {e}") from e

    def start_recording(self) -> bool:
        """開始錄音"""
        return self.recorder.start_recording()

    def stop_recording(self) -> Optional[str]:
        """停止錄音並回傳音頻檔案路徑"""
        return self.recorder.stop_recording()

    def is_recording(self) -> bool:
        """檢查是否正在錄音"""
        return self.recorder.is_recording

    def get_recording_duration(self) -> float:
        """取得目前錄音時長"""
        return self.recorder.get_recording_duration()


# pylint: disable=invalid-name
stt_service = None
try:
    stt_service = STTService()
    logger.info("✅ STT 全域服務初始化成功")
except (RuntimeError, ValueError) as e:
    logger.warning("❌ STT 服務初始化失敗: %s", e)


def transcribe_audio(audio_file_path: Union[str, Path]) -> Tuple[str, float]:
    """語音轉文字主要函數"""
    if not stt_service:
        raise RuntimeError("STT 服務不可用")
    return stt_service.transcribe_audio(audio_file_path)


def start_recording() -> bool:
    """開始錄音"""
    if not stt_service:
        raise RuntimeError("STT 服務不可用")
    return stt_service.start_recording()


def stop_recording() -> Optional[str]:
    """停止錄音並回傳音頻檔案路徑"""
    if not stt_service:
        raise RuntimeError("STT 服務不可用")
    return stt_service.stop_recording()


def is_recording() -> bool:
    """檢查是否正在錄音"""
    if not stt_service:
        return False
    return stt_service.is_recording()


def get_recording_duration() -> float:
    """取得目前錄音時長"""
    if not stt_service:
        return 0.0
    return stt_service.get_recording_duration()
