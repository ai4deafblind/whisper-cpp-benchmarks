"""JSON/JSONL report generation."""

import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import BenchmarkConfig
from .metrics import AggregateMetrics, SampleMetrics


def write_sample_result(
    path: Path,
    metrics: SampleMetrics,
    inference_time_ms: float,
    duration_ms: float | None,
) -> None:
    """Append a single sample result to JSONL file."""
    record = {
        "clip_id": metrics.clip_id,
        "ground_truth": metrics.ground_truth,
        "transcription": metrics.transcription,
        "ground_truth_normalized": metrics.ground_truth_normalized,
        "transcription_normalized": metrics.transcription_normalized,
        "wer": metrics.wer,
        "cer": metrics.cer,
        "word_count": metrics.word_count,
        "char_count": metrics.char_count,
        "substitutions": metrics.substitutions,
        "deletions": metrics.deletions,
        "insertions": metrics.insertions,
        "inference_time_ms": inference_time_ms,
        "duration_ms": duration_ms,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_failed_result(
    path: Path,
    clip_id: str,
    error: str,
    inference_time_ms: float,
) -> None:
    """Append a failed sample to JSONL file."""
    record = {
        "clip_id": clip_id,
        "error": error,
        "inference_time_ms": inference_time_ms,
        "success": False,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    config: BenchmarkConfig,
    aggregate: AggregateMetrics,
) -> None:
    """Write summary JSON with aggregate metrics and config."""
    summary = {
        "config": {
            "model_path": str(config.whisper.model_path),
            "language": config.whisper.language,
            "threads": config.whisper.threads,
            "beam_size": config.whisper.beam_size,
            "no_gpu": config.whisper.no_gpu,
            "dataset_path": str(config.dataset.dataset_path),
            "split": config.dataset.split,
            "sample_size": config.dataset.sample_size,
            "sampling_strategy": config.dataset.sampling_strategy,
            "seed": config.dataset.seed,
        },
        "metrics": asdict(aggregate),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_summary(aggregate: AggregateMetrics, console: Console) -> None:
    """Print rich formatted summary to console."""
    # Main metrics table
    table = Table(title="Benchmark Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Corpus WER", f"{aggregate.corpus_wer:.2%}")
    table.add_row("Corpus CER", f"{aggregate.corpus_cer:.2%}")
    table.add_row("", "")
    table.add_row("Mean WER", f"{aggregate.mean_wer:.2%}")
    table.add_row("Mean CER", f"{aggregate.mean_cer:.2%}")
    table.add_row("Median WER", f"{aggregate.median_wer:.2%}")
    table.add_row("Median CER", f"{aggregate.median_cer:.2%}")
    table.add_row("P90 WER", f"{aggregate.p90_wer:.2%}")
    table.add_row("P95 WER", f"{aggregate.p95_wer:.2%}")
    table.add_row("", "")
    table.add_row("Successful Samples", f"{aggregate.successful_samples}")
    table.add_row("Failed Samples", f"{aggregate.failed_samples}")
    table.add_row("Total Words", f"{aggregate.total_words:,}")

    if aggregate.total_duration_ms > 0:
        table.add_row(
            "Total Audio Duration",
            f"{aggregate.total_duration_ms / 1000:.1f}s"
        )
        table.add_row(
            "Total Inference Time",
            f"{aggregate.total_inference_ms / 1000:.1f}s"
        )
        table.add_row("Real-time Factor", f"{aggregate.real_time_factor:.2f}x")

    console.print()
    console.print(table)
    console.print()
