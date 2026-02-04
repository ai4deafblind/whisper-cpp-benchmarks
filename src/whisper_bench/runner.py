"""Whisper.cpp subprocess execution."""

from __future__ import annotations

import json
import re
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


def parse_whisper_timings(stderr: str) -> tuple[float | None, float | None]:
    """Parse encoder and decode times from whisper-cli stderr output.

    Expected format:
        whisper_print_timings:   encode time =  8582.63 ms / 1430.44 ms per layer
        whisper_print_timings:   decode time =   436.16 ms / 72.69 ms per layer

    Args:
        stderr: Standard error output from whisper-cli

    Returns:
        (encode_time_ms, decode_time_ms) or (None, None) if not found
    """
    encode_time = None
    decode_time = None

    # Pattern: "whisper_print_timings:   encode time = <number> ms"
    encode_pattern = r"whisper_print_timings:\s+encode time\s+=\s+([\d.]+)\s+ms"
    decode_pattern = r"whisper_print_timings:\s+decode time\s+=\s+([\d.]+)\s+ms"

    encode_match = re.search(encode_pattern, stderr)
    decode_match = re.search(decode_pattern, stderr)

    if encode_match:
        encode_time = float(encode_match.group(1))
    if decode_match:
        decode_time = float(decode_match.group(1))

    return encode_time, decode_time


@dataclass
class TranscriptionResult:
    """Result of a single transcription."""

    clip_id: str
    transcription: str
    inference_time_ms: float
    success: bool
    error: str | None = None
    hardware_metrics: SampleHardwareMetrics | None = None
    encode_time_ms: float | None = None
    decode_time_ms: float | None = None


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
        "-f", str(sample.audio_path),
        "-oj",  # JSON output
        # "-np",  # REMOVED - we need stderr for timing output
        "-t", str(config.threads),
        "-bs", str(config.beam_size),
        "-of", str(output_base),
    ]

    # Language handling
    if config.auto_detect_language:
        cmd.extend(["-l", "auto"])
    else:
        cmd.extend(["-l", config.language])

    if config.no_gpu:
        cmd.append("-ng")

    # Non-speech suppression
    if config.suppress_non_speech:
        cmd.append("-sns")
    if config.no_speech_threshold is not None:
        cmd.extend(["-nth", str(config.no_speech_threshold)])

    # VAD options
    if config.vad_enabled:
        cmd.append("--vad")
    if config.vad_model_path is not None:
        cmd.extend(["-vm", str(config.vad_model_path)])
    if config.vad_threshold is not None:
        cmd.extend(["-vt", str(config.vad_threshold)])
    if config.vad_min_speech_duration_ms is not None:
        cmd.extend(["-vspd", str(config.vad_min_speech_duration_ms)])
    if config.vad_min_silence_duration_ms is not None:
        cmd.extend(["-vsd", str(config.vad_min_silence_duration_ms)])
    if config.vad_max_speech_duration_s is not None:
        cmd.extend(["-vmsd", str(config.vad_max_speech_duration_s)])
    if config.vad_speech_pad_ms is not None:
        cmd.extend(["-vp", str(config.vad_speech_pad_ms)])
    if config.vad_samples_overlap is not None:
        cmd.extend(["-vo", str(config.vad_samples_overlap)])

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

        # Parse encoder/decoder times from stderr
        encode_time_ms, decode_time_ms = parse_whisper_timings(result.stderr)

        if result.returncode != 0:
            return TranscriptionResult(
                clip_id=sample.clip_id,
                transcription="",
                inference_time_ms=elapsed_ms,
                success=False,
                error=f"whisper-cli failed: {result.stderr}",
                hardware_metrics=hardware_metrics,
                encode_time_ms=encode_time_ms,
                decode_time_ms=decode_time_ms,
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
                encode_time_ms=encode_time_ms,
                decode_time_ms=decode_time_ms,
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
            encode_time_ms=encode_time_ms,
            decode_time_ms=decode_time_ms,
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
            encode_time_ms=None,
            decode_time_ms=None,
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
            encode_time_ms=None,
            decode_time_ms=None,
        )
