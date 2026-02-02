"""Configuration dataclasses for benchmark runs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os


SamplingStrategy = Literal["stratified", "random", "sequential", "all"]
Split = Literal["test", "dev", "train", "validated"]


@dataclass(frozen=True)
class WhisperConfig:
    """Configuration for Whisper.cpp execution."""

    model_path: Path
    language: str = "id"
    threads: int = field(default_factory=lambda: max(1, os.cpu_count() // 2))
    beam_size: int = 5
    no_gpu: bool = False
    whisper_cli: Path = Path("/home/rizal/whisper.cpp/build/bin/whisper-cli")

    # Auto-detect language
    auto_detect_language: bool = False

    # Non-speech suppression
    suppress_non_speech: bool = False
    no_speech_threshold: float | None = None  # whisper.cpp default: 0.60

    # VAD options
    vad_enabled: bool = False
    vad_model_path: Path | None = None
    vad_threshold: float | None = None  # default: 0.50
    vad_min_speech_duration_ms: int | None = None  # default: 250
    vad_min_silence_duration_ms: int | None = None  # default: 100
    vad_max_speech_duration_s: float | None = None  # default: unlimited
    vad_speech_pad_ms: int | None = None  # default: 30
    vad_samples_overlap: float | None = None  # default: 0.10

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.whisper_cli.exists():
            raise FileNotFoundError(f"whisper-cli not found: {self.whisper_cli}")

        # VAD validation
        if self.vad_enabled and self.vad_model_path is None:
            raise ValueError("vad_model_path is required when vad_enabled=True")
        if self.vad_model_path is not None and not self.vad_model_path.exists():
            raise FileNotFoundError(f"VAD model not found: {self.vad_model_path}")

        # Threshold validation
        if self.no_speech_threshold is not None:
            if not 0.0 <= self.no_speech_threshold <= 1.0:
                raise ValueError("no_speech_threshold must be between 0.0 and 1.0")
        if self.vad_threshold is not None:
            if not 0.0 <= self.vad_threshold <= 1.0:
                raise ValueError("vad_threshold must be between 0.0 and 1.0")
        if self.vad_samples_overlap is not None:
            if not 0.0 <= self.vad_samples_overlap <= 1.0:
                raise ValueError("vad_samples_overlap must be between 0.0 and 1.0")


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for Common Voice dataset loading."""

    dataset_path: Path
    split: Split = "test"
    sample_size: int = 100
    sampling_strategy: SamplingStrategy = "stratified"
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        tsv_path = self.dataset_path / f"{self.split}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"Split file not found: {tsv_path}")

    @property
    def tsv_path(self) -> Path:
        return self.dataset_path / f"{self.split}.tsv"

    @property
    def clips_dir(self) -> Path:
        return self.dataset_path / "clips"

    @property
    def durations_path(self) -> Path:
        return self.dataset_path / "clip_durations.tsv"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Combined configuration for a benchmark run."""

    whisper: WhisperConfig
    dataset: DatasetConfig
    output_dir: Path = Path("benchmarks")
    run_name: str | None = None

    def get_run_dir(self) -> Path:
        """Get the output directory for this run."""
        if self.run_name:
            return self.output_dir / self.run_name
        # Generate name from model and dataset
        model_name = self.whisper.model_path.stem
        dataset_name = self.dataset.dataset_path.name
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{model_name}_{dataset_name}_{timestamp}"
