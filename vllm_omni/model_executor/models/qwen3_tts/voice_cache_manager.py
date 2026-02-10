# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """

    ref_code: torch.Tensor | None  # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


class VoiceCacheManager:
    """
    Voice cache manager, responsible for managing custom voice cache functionality.

    Main features:
    1. Load uploaded speaker information from metadata.json
    2. Manage voice clone prompt cache
    3. Update cache status to metadata.json
    """

    def __init__(self, speech_voice_samples_dir: str | None = None):
        """
        Initialize the voice cache manager.

        Args:
            speech_voice_samples_dir: Speech voice samples directory path,
                if None, get from environment variable
        """
        self.speech_voice_samples_dir = speech_voice_samples_dir or os.environ.get(
            "SPEECH_VOICE_SAMPLES", "/tmp/voice_samples"
        )

    def load_uploaded_speakers_from_metadata(self) -> dict[str, Any] | None:
        """
        Load uploaded_speakers from metadata.json.

        Returns:
            Optional[Dict[str, Any]]: Dictionary of uploaded speaker information,
                returns None if file doesn't exist or read fails
        """
        try:
            # Get metadata.json path
            metadata_file = Path(self.speech_voice_samples_dir) / "metadata.json"

            if not metadata_file.exists():
                return None

            with open(metadata_file) as f:
                metadata = json.load(f)

            return metadata.get("uploaded_speakers", {})

        except Exception as e:
            logger.warning(f"Failed to load uploaded speakers from metadata: {e}")
            return None

    def update_metadata_cache_info(self, speaker: str, cache_file_path: Path, status: str = "ready") -> bool:
        """
        Update cache information in metadata.json.

        Args:
            speaker: Speaker name
            cache_file_path: Cache file path
            status: Cache status, default is "ready"

        Returns:
            bool: Whether the update was successful
        """
        try:
            metadata_file = Path(self.speech_voice_samples_dir) / "metadata.json"

            if not metadata_file.exists():
                return False

            with open(metadata_file) as f:
                metadata = json.load(f)

            voice_name_lower = speaker.lower()
            if voice_name_lower not in metadata.get("uploaded_speakers", {}):
                return False

            # Update cache information
            metadata["uploaded_speakers"][voice_name_lower].update(
                {"cache_status": status, "cache_file": str(cache_file_path), "cache_generated_at": time.time()}
            )

            # Save back to file
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Updated cache info in metadata.json for speaker: {speaker}")
            return True

        except Exception as e:
            logger.error(f"Failed to update metadata.json: {e}")
            return False

    def load_cached_voice_prompt(self, speaker: str, device: str | None = None) -> list | None:
        """
        Load cached VoiceClonePromptItem.

        Args:
            speaker: Speaker name
            device: Target device (e.g., "cuda", "cpu"), if None keep as is

        Returns:
            Optional[list]: List of cached VoiceClonePromptItem,
                returns None if cache doesn't exist or loading fails
        """
        try:
            uploaded_speakers = self.load_uploaded_speakers_from_metadata()
            if not uploaded_speakers or speaker.lower() not in uploaded_speakers:
                return None

            speaker_info = uploaded_speakers[speaker.lower()]
            cache_status = speaker_info.get("cache_status", "pending")
            cache_file_path = Path(speaker_info.get("cache_file", "")) if speaker_info.get("cache_file") else None

            # Check cache status and file existence
            if cache_file_path and cache_file_path.exists() and cache_status == "ready":
                logger.info(f"Using cached voice clone prompt for speaker: {speaker}")

                # Load cached VoiceClonePromptItem
                cached_items = torch.load(cache_file_path, map_location="cpu", weights_only=False)

                # Verify cache format
                if isinstance(cached_items, list):
                    # Move tensors to target device if needed
                    if device is not None:
                        device_items = []
                        for item in cached_items:
                            device_item = VoiceClonePromptItem(
                                ref_code=item.ref_code.to(device) if item.ref_code is not None else None,
                                ref_spk_embedding=item.ref_spk_embedding.to(device),
                                x_vector_only_mode=item.x_vector_only_mode,
                                icl_mode=item.icl_mode,
                                ref_text=item.ref_text,
                            )
                            device_items.append(device_item)
                        cached_items = device_items

                    logger.info(f"Cache loaded successfully for speaker: {speaker}")
                    return cached_items
                else:
                    logger.warning(f"Cache file format invalid for speaker: {speaker}")
                    return None
            else:
                return None

        except Exception as e:
            logger.warning(f"Failed to load cache for speaker {speaker}: {e}")
            return None

    def save_voice_cache(self, speaker: str, audio_file_path: Path, prompt_items: list) -> bool:
        """
        Save voice cache.

        Args:
            speaker: Speaker name
            audio_file_path: Audio file path
            prompt_items: List of VoiceClonePromptItem

        Returns:
            bool: Whether save was successful
        """
        try:
            # Generate cache file path
            cache_file_path = audio_file_path.with_suffix(".pt")

            # Ensure all tensors are on CPU
            cpu_prompt_items = []
            for item in prompt_items:
                # Create new VoiceClonePromptItem with all tensors moved to CPU
                cpu_item = VoiceClonePromptItem(
                    ref_code=item.ref_code.to("cpu") if item.ref_code is not None else None,
                    ref_spk_embedding=item.ref_spk_embedding.to("cpu"),
                    x_vector_only_mode=item.x_vector_only_mode,
                    icl_mode=item.icl_mode,
                    ref_text=item.ref_text,
                )
                cpu_prompt_items.append(cpu_item)

            # Save cache
            torch.save(cpu_prompt_items, cache_file_path)

            # Update metadata.json
            success = self.update_metadata_cache_info(speaker, cache_file_path, "ready")

            if success:
                logger.info(f"Cache generated and saved for speaker: {speaker}")
            else:
                logger.error(f"Failed to update metadata for speaker: {speaker}")

            return success

        except Exception as e:
            logger.error(f"Failed to save cache for speaker {speaker}: {e}")
            # Update status to failed
            self.update_metadata_cache_info(speaker, Path(""), "failed")
            return False

    def get_speaker_audio_path(self, speaker: str) -> Path | None:
        """
        Get speaker's audio file path.

        Args:
            speaker: Speaker name

        Returns:
            Optional[Path]: Audio file path, returns None if speaker doesn't exist
        """
        uploaded_speakers = self.load_uploaded_speakers_from_metadata()
        if not uploaded_speakers or speaker.lower() not in uploaded_speakers:
            return None

        speaker_info = uploaded_speakers[speaker.lower()]
        audio_file_path = Path(speaker_info["file_path"])

        if audio_file_path.exists():
            return audio_file_path
        else:
            logger.warning(f"Audio file not found for speaker {speaker}: {audio_file_path}")
            return None
