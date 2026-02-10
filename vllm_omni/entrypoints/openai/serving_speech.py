import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import Request, UploadFile
from fastapi.responses import Response
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import (
    AudioResponse,
    CreateAudio,
    OpenAICreateSpeechRequest,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

# TTS Configuration (currently supports Qwen3-TTS)
_TTS_MODEL_STAGES: set[str] = {"qwen3_tts"}
_TTS_LANGUAGES: set[str] = {
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
}
_TTS_MAX_INSTRUCTIONS_LENGTH = 500
_TTS_MAX_NEW_TOKENS_MIN = 1
_TTS_MAX_NEW_TOKENS_MAX = 4096


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Only allows alphanumeric characters, underscores, hyphens, and dots.
    Replaces any other characters with underscores.
    """
    # Remove any path components
    filename = os.path.basename(filename)
    # Replace any non-alphanumeric, underscore, hyphen, or dot with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "file"
    # Limit length to prevent potential issues
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized


def _validate_path_within_directory(file_path: Path, directory: Path) -> bool:
    """Validate that file_path is within the specified directory.

    Prevents path traversal attacks by ensuring the resolved path
    is within the target directory.
    """
    try:
        # Resolve both paths to absolute paths
        file_path_resolved = file_path.resolve()
        directory_resolved = directory.resolve()
        # Check if file_path is within directory
        return directory_resolved in file_path_resolved.parents or directory_resolved == file_path_resolved
    except Exception:
        return False


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize uploaded speakers storage
        speech_voice_samples_dir = os.environ.get("SPEECH_VOICE_SAMPLES", "/tmp/voice_samples")
        self.uploaded_speakers_dir = Path(speech_voice_samples_dir)
        self.uploaded_speakers_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.uploaded_speakers_dir / "metadata.json"

        # Load supported speakers
        self.supported_speakers = self._load_supported_speakers()
        # Load uploaded speakers
        self.uploaded_speakers = self._load_uploaded_speakers()
        # Merge supported speakers with uploaded speakers
        self.supported_speakers.update(self.uploaded_speakers.keys())

        logger.info(f"Loaded {len(self.supported_speakers)} supported speakers: {sorted(self.supported_speakers)}")
        logger.info(f"Loaded {len(self.uploaded_speakers)} uploaded speakers")

    def _load_supported_speakers(self) -> set[str]:
        """Load supported speakers (case-insensitive) from the model configuration."""
        try:
            talker_config = self.engine_client.model_config.hf_config.talker_config

            # Check for speakers in either spk_id or speaker_id
            for attr_name in ["spk_id", "speaker_id"]:
                speakers_dict = getattr(talker_config, attr_name, None)
                if speakers_dict and isinstance(speakers_dict, dict):
                    # Normalize to lowercase for case-insensitive matching
                    return {speaker.lower() for speaker in speakers_dict.keys()}

            logger.warning("No speakers found in talker_config (checked spk_id and speaker_id)")
        except Exception as e:
            logger.warning(f"Could not load speakers from model config: {e}")

        return set()

    def _load_uploaded_speakers(self) -> dict[str, dict]:
        """Load uploaded speakers from metadata file."""
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file) as f:
                metadata = json.load(f)
            return metadata.get("uploaded_speakers", {})
        except Exception as e:
            logger.warning(f"Could not load uploaded speakers metadata: {e}")
            return {}

    def _save_uploaded_speakers(self) -> None:
        """Save uploaded speakers to metadata file."""
        try:
            metadata = {"uploaded_speakers": self.uploaded_speakers}
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save uploaded speakers metadata: {e}")

    def _get_uploaded_audio_data(self, voice_name: str) -> str | None:
        """Get base64 encoded audio data for uploaded voice."""
        voice_name_lower = voice_name.lower()
        if voice_name_lower not in self.uploaded_speakers:
            return None

        speaker_info = self.uploaded_speakers[voice_name_lower]
        file_path = Path(speaker_info["file_path"])

        if not file_path.exists():
            logger.warning(f"Audio file not found for voice {voice_name}: {file_path}")
            return None

        try:
            # Read audio file
            with open(file_path, "rb") as f:
                audio_bytes = f.read()

            # Encode to base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Get MIME type from file extension
            mime_type = speaker_info.get("mime_type", "audio/wav")

            # Return as data URL
            return f"data:{mime_type};base64,{audio_b64}"
        except Exception as e:
            logger.error(f"Could not read audio file for voice {voice_name}: {e}")
            return None

    async def upload_voice(self, audio_file: UploadFile, consent: str, name: str) -> dict:
        """Upload a new voice sample."""
        # Validate file size (max 10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        audio_file.file.seek(0, 2)  # Seek to end
        file_size = audio_file.file.tell()
        audio_file.file.seek(0)  # Reset to beginning

        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum limit of 10MB. Got {file_size} bytes.")

        # Detect MIME type from filename if content_type is generic
        mime_type = audio_file.content_type
        if mime_type == "application/octet-stream":
            # Simple MIME type detection based on file extension
            filename_lower = audio_file.filename.lower()
            if filename_lower.endswith(".wav"):
                mime_type = "audio/wav"
            elif filename_lower.endswith((".mp3", ".mpeg")):
                mime_type = "audio/mpeg"
            elif filename_lower.endswith(".flac"):
                mime_type = "audio/flac"
            elif filename_lower.endswith(".ogg"):
                mime_type = "audio/ogg"
            elif filename_lower.endswith(".aac"):
                mime_type = "audio/aac"
            elif filename_lower.endswith(".webm"):
                mime_type = "audio/webm"
            elif filename_lower.endswith(".mp4"):
                mime_type = "audio/mp4"
            else:
                mime_type = "audio/wav"  # Default

        # Validate MIME type
        allowed_mime_types = {
            "audio/mpeg",
            "audio/wav",
            "audio/x-wav",
            "audio/ogg",
            "audio/aac",
            "audio/flac",
            "audio/webm",
            "audio/mp4",
        }

        if mime_type not in allowed_mime_types:
            raise ValueError(f"Unsupported MIME type: {mime_type}. Allowed: {allowed_mime_types}")

        # Normalize voice name
        voice_name_lower = name.lower()

        # Check if voice already exists
        if voice_name_lower in self.uploaded_speakers:
            raise ValueError(f"Voice '{name}' already exists")

        # Sanitize name and consent to prevent path traversal
        sanitized_name = _sanitize_filename(name)
        sanitized_consent = _sanitize_filename(consent)

        # Generate filename with sanitized inputs
        timestamp = int(time.time())
        file_suffix = Path(audio_file.filename).suffix
        file_ext = file_suffix[1:] if file_suffix and len(file_suffix) > 1 else "wav"
        # Sanitize file extension as well
        sanitized_ext = _sanitize_filename(file_ext)
        if not sanitized_ext or sanitized_ext == "file":
            sanitized_ext = "wav"

        filename = f"{sanitized_name}_{sanitized_consent}_{timestamp}.{sanitized_ext}"
        file_path = self.uploaded_speakers_dir / filename

        # Double-check that the path is within the upload directory
        if not _validate_path_within_directory(file_path, self.uploaded_speakers_dir):
            raise ValueError("Invalid file path: potential path traversal attack detected")

        # Save audio file
        try:
            with open(file_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)
        except Exception as e:
            raise ValueError(f"Failed to save audio file: {e}")

        # Update metadata
        self.uploaded_speakers[voice_name_lower] = {
            "name": name,
            "consent": consent,
            "file_path": str(file_path),
            "created_at": timestamp,
            "mime_type": mime_type,
            "original_filename": audio_file.filename,
            "file_size": file_size,
            "cache_status": "pending",  # 初始缓存状态为 pending
            "cache_file": None,  # 初始缓存文件为空
            "cache_generated_at": None,  # 初始缓存生成时间为空
        }

        # Update supported speakers
        self.supported_speakers.add(voice_name_lower)

        # Save metadata
        self._save_uploaded_speakers()

        logger.info(f"Uploaded new voice '{name}' with consent ID '{consent}'")

        # Return voice information without exposing the server file path
        return {
            "name": name,
            "consent": consent,
            "created_at": timestamp,
            "mime_type": mime_type,
            "file_size": file_size,
        }

    def _is_tts_model(self) -> bool:
        """Check if the current model is a supported TTS model."""
        stage_list = getattr(self.engine_client, "stage_list", None)
        if stage_list:
            for stage in stage_list:
                model_stage = getattr(stage, "model_stage", None)
                if model_stage in _TTS_MODEL_STAGES:
                    return True
        return False

    def _validate_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate TTS request parameters. Returns error message or None."""
        task_type = request.task_type or "CustomVoice"

        # Normalize voice to lowercase for case-insensitive matching
        if request.voice is not None:
            request.voice = request.voice.lower()

        # Validate input is not empty
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Validate language
        if request.language is not None and request.language not in _TTS_LANGUAGES:
            return f"Invalid language '{request.language}'. Supported: {', '.join(sorted(_TTS_LANGUAGES))}"

        # Validate speaker for CustomVoice task
        if task_type == "CustomVoice" and request.voice is not None:
            if self.supported_speakers and request.voice not in self.supported_speakers:
                return f"Invalid speaker '{request.voice}'. Supported: {', '.join(sorted(self.supported_speakers))}"

        # Validate Base task requirements
        if task_type == "Base":
            if request.voice is None:
                if request.ref_audio is None:
                    return "Base task requires 'ref_audio' for voice cloning"
                # Validate ref_audio format
                if not (request.ref_audio.startswith(("http://", "https://")) or request.ref_audio.startswith("data:")):
                    return "ref_audio must be a URL (http/https) or base64 data URL (data:...)"
            else:
                # voice is not None
                voice_lower = request.voice.lower()
                if voice_lower in self.uploaded_speakers:
                    pass
                else:
                    # need ref_audio
                    if request.ref_audio is None:
                        return (
                            f"Base task with built-in speaker '{request.voice}' requires 'ref_audio' for voice cloning"
                        )

        # Validate cross-parameter dependencies
        if task_type != "Base":
            if request.ref_text is not None:
                return "'ref_text' is only valid for Base task"
            if request.x_vector_only_mode is not None:
                return "'x_vector_only_mode' is only valid for Base task"

        # Validate VoiceDesign task requirements
        if task_type == "VoiceDesign" and not request.instructions:
            return "VoiceDesign task requires 'instructions' to describe the voice"

        # Validate instructions length
        if request.instructions and len(request.instructions) > _TTS_MAX_INSTRUCTIONS_LENGTH:
            return f"Instructions too long (max {_TTS_MAX_INSTRUCTIONS_LENGTH} characters)"

        # Validate max_new_tokens range
        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    def _build_tts_prompt(self, text: str) -> str:
        """Build TTS prompt from input text."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_tts_params(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build TTS parameters from request.

        Processes each parameter if present, skips if not.
        Values are wrapped in lists as required by the model.
        """
        params: dict[str, Any] = {}

        # Text content (always required)
        params["text"] = [request.input]

        # Task type
        if request.task_type is not None:
            params["task_type"] = [request.task_type]
        else:
            params["task_type"] = ["CustomVoice"]

        # Language
        if request.language is not None:
            params["language"] = [request.language]
        else:
            params["language"] = ["Auto"]

        # Speaker (voice)
        if request.voice is not None:
            params["speaker"] = [request.voice]

            # If voice is an uploaded speaker and no ref_audio provided, auto-set it
            if request.voice.lower() in self.uploaded_speakers and request.ref_audio is None:
                audio_data = self._get_uploaded_audio_data(request.voice)
                if audio_data:
                    params["ref_audio"] = [audio_data]
                    params["x_vector_only_mode"] = [True]
                    logger.info(f"Auto-set ref_audio for uploaded voice: {request.voice}")
                else:
                    raise ValueError(f"Audio file for uploaded voice '{request.voice}' is missing or corrupted")

        elif params["task_type"][0] == "CustomVoice":
            params["speaker"] = ["Vivian"]  # Default for CustomVoice

        # Instructions for style/emotion control
        if request.instructions is not None:
            params["instruct"] = [request.instructions]
        else:
            params["instruct"] = [""]

        # Voice clone parameters (used with Base task)
        if request.ref_audio is not None:
            params["ref_audio"] = [request.ref_audio]
        if request.ref_text is not None:
            params["ref_text"] = [request.ref_text]
        if request.x_vector_only_mode is not None:
            params["x_vector_only_mode"] = [request.x_vector_only_mode]

        # Generation parameters
        if request.max_new_tokens is not None:
            params["max_new_tokens"] = [request.max_new_tokens]
        else:
            params["max_new_tokens"] = [2048]

        return params

    async def create_speech(
        self,
        request: OpenAICreateSpeechRequest,
        raw_request: Request | None = None,
    ):
        """
        Create Speech API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createSpeech
        for the API specification. This API mimics the OpenAI
        Create Speech API.

        For Qwen3-TTS models, additional parameters are supported:
        - task_type: "CustomVoice", "VoiceDesign", or "Base"
        - language: Language code (e.g., "Chinese", "English", "Auto")
        - voice: Speaker name (e.g., "Vivian", "Ryan") for CustomVoice
        - instructions: Voice style/emotion instructions
        - ref_audio: Reference audio for voice cloning (Base task)
        - ref_text: Transcript of reference audio (Base task)
        - x_vector_only_mode: Use speaker embedding only (Base task)

        NOTE: Streaming audio generation is not currently supported.
        """

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = f"speech-{random_uuid()}"

        try:
            if self._is_tts_model():
                # Validate TTS parameters
                validation_error = self._validate_tts_request(request)
                if validation_error:
                    return self.create_error_response(validation_error)

                # Build TTS parameters and prompt
                tts_params = self._build_tts_params(request)
                prompt_text = self._build_tts_prompt(request.input)
                prompt = {
                    "prompt": prompt_text,
                    "additional_information": tts_params,
                }
            else:
                # Fallback for unsupported models
                tts_params = {}
                prompt = {"prompt": request.input}

            logger.info(
                "TTS speech request %s: text=%r, task_type=%s",
                request_id,
                request.input[:50] + "..." if len(request.input) > 50 else request.input,
                tts_params.get("task_type", ["unknown"])[0],
            )

            sampling_params_list = self.engine_client.default_sampling_params_list

            generator = self.engine_client.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["audio"],
            )

            final_output: OmniRequestOutput | None = None
            async for res in generator:
                final_output = res

            if final_output is None:
                return self.create_error_response("No output generated from the model.")

            # Extract audio from output
            # Audio can be in final_output.multimodal_output or final_output.request_output.multimodal_output
            # Support both "audio" and "model_outputs" keys for compatibility with different models
            audio_output = None
            if hasattr(final_output, "multimodal_output") and final_output.multimodal_output:
                audio_output = final_output.multimodal_output
            if not audio_output and hasattr(final_output, "request_output"):
                if final_output.request_output and hasattr(final_output.request_output, "multimodal_output"):
                    audio_output = final_output.request_output.multimodal_output

            # Check for audio data using either "audio" or "model_outputs" key
            audio_key = None
            if audio_output:
                if "audio" in audio_output:
                    audio_key = "audio"
                elif "model_outputs" in audio_output:
                    audio_key = "model_outputs"

            if not audio_output or audio_key is None:
                return self.create_error_response("TTS model did not produce audio output.")

            audio_tensor = audio_output[audio_key]
            sample_rate = audio_output.get("sr", 24000)
            if hasattr(sample_rate, "item"):
                sample_rate = sample_rate.item()

            # Convert tensor to numpy
            if hasattr(audio_tensor, "float"):
                audio_tensor = audio_tensor.float().detach().cpu().numpy()

            # Squeeze batch dimension if present, but preserve channel dimension for stereo
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=int(sample_rate),
                response_format=request.response_format or "wav",
                speed=request.speed or 1.0,
                stream_format=request.stream_format,
                base64_encode=False,
            )

            audio_response: AudioResponse = self.create_audio(audio_obj)
            return Response(content=audio_response.audio_data, media_type=audio_response.media_type)

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)
        except Exception as e:
            logger.exception("Speech generation failed: %s", e)
            return self.create_error_response(f"Speech generation failed: {e}")
