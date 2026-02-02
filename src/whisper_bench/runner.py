"""Whisper.cpp subprocess execution."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import WhisperConfig
from .dataset import Sample

if TYPE_CHECKING:
    from .system import SampleHardwareMetrics


@dataclass
class TranscriptionResult:
    """Result of a single transcription."""

    clip_id: str
    transcription: str
    inference_time_ms: float
    success: bool
    error: str | None = None
    hardware_metrics: SampleHardwareMetrics | None = None


def run_whisper(
    sample: Sample,
    config: WhisperConfig,
    work_dir: Path,
    monitor_resources: bool = True,
    monitor_interval_ms: int = 100,
) -> TranscriptionResult:
    """Run whisper-cli on a single sample."""
    output_base = work_dir / sample.clip_id.replace(".mp3", "").replace(".wav", "")

    cmd = [
        str(config.whisper_cli),
        "-m", str(config.model_path),
        "-l", config.language,
        "-f", str(sample.audio_path),
        "-oj",  # JSON output
        "-np",  # No prints (progress, etc.)
        "-t", str(config.threads),
        "-bs", str(config.beam_size),
        "-of", str(output_base),
    ]

    if config.no_gpu:
        cmd.append("-ng")

    # Set up resource monitoring if enabled
    monitor = None
    hardware_metrics = None
    if monitor_resources:
        from .system import ResourceMonitor

        monitor = ResourceMonitor(interval_ms=monitor_interval_ms)

    start_time = time.perf_counter()

    try:
        if monitor:
            monitor.start()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if monitor:
            monitor.stop()
            hardware_metrics = monitor.get_metrics()

        if result.returncode != 0:
            return TranscriptionResult(
                clip_id=sample.clip_id,
                transcription="",
                inference_time_ms=elapsed_ms,
                success=False,
                error=f"whisper-cli failed: {result.stderr}",
                hardware_metrics=hardware_metrics,
            )

        # Parse JSON output
        json_path = Path(f"{output_base}.json")
        if not json_path.exists():
            return TranscriptionResult(
                clip_id=sample.clip_id,
                transcription="",
                inference_time_ms=elapsed_ms,
                success=False,
                error=f"JSON output not found: {json_path}",
                hardware_metrics=hardware_metrics,
            )

        with open(json_path) as f:
            output = json.load(f)

        # Extract transcription text from segments
        segments = output.get("transcription", [])
        text_parts = [seg.get("text", "") for seg in segments]
        transcription = " ".join(text_parts).strip()

        # Clean up JSON file
        json_path.unlink(missing_ok=True)

        return TranscriptionResult(
            clip_id=sample.clip_id,
            transcription=transcription,
            inference_time_ms=elapsed_ms,
            success=True,
            hardware_metrics=hardware_metrics,
        )

    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if monitor:
            monitor.stop()
            hardware_metrics = monitor.get_metrics()
        return TranscriptionResult(
            clip_id=sample.clip_id,
            transcription="",
            inference_time_ms=elapsed_ms,
            success=False,
            error="Timeout after 300 seconds",
            hardware_metrics=hardware_metrics,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if monitor:
            monitor.stop()
            hardware_metrics = monitor.get_metrics()
        return TranscriptionResult(
            clip_id=sample.clip_id,
            transcription="",
            inference_time_ms=elapsed_ms,
            success=False,
            error=str(e),
            hardware_metrics=hardware_metrics,
        )
