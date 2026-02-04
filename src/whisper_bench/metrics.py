"""WER/CER calculation using jiwer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jiwer

from .normalize import normalize_text

if TYPE_CHECKING:
    from .system import AggregateHardwareMetrics


@dataclass
class SampleMetrics:
    """Metrics for a single sample."""

    clip_id: str
    ground_truth: str
    transcription: str
    ground_truth_normalized: str
    transcription_normalized: str
    wer: float
    cer: float
    word_count: int
    char_count: int
    substitutions: int
    deletions: int
    insertions: int


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all samples."""

    corpus_wer: float
    corpus_cer: float
    mean_wer: float
    mean_cer: float
    median_wer: float
    median_cer: float
    p90_wer: float
    p90_cer: float
    p95_wer: float
    p95_cer: float
    total_samples: int
    successful_samples: int
    failed_samples: int
    total_words: int
    total_chars: int
    total_duration_ms: float
    total_inference_ms: float
    real_time_factor: float
    hardware: AggregateHardwareMetrics | None = None
    encode_time_ms_mean: float | None = None
    encode_time_ms_median: float | None = None
    decode_time_ms_mean: float | None = None
    decode_time_ms_median: float | None = None


def compute_sample_metrics(
    clip_id: str,
    ground_truth: str,
    transcription: str,
) -> SampleMetrics:
    """Compute WER/CER for a single sample."""
    gt_norm = normalize_text(ground_truth)
    tr_norm = normalize_text(transcription)

    # Handle empty strings
    if not gt_norm:
        return SampleMetrics(
            clip_id=clip_id,
            ground_truth=ground_truth,
            transcription=transcription,
            ground_truth_normalized=gt_norm,
            transcription_normalized=tr_norm,
            wer=0.0 if not tr_norm else 1.0,
            cer=0.0 if not tr_norm else 1.0,
            word_count=0,
            char_count=0,
            substitutions=0,
            deletions=0,
            insertions=len(tr_norm.split()) if tr_norm else 0,
        )

    # Compute WER
    wer_output = jiwer.process_words(gt_norm, tr_norm)
    wer = wer_output.wer

    # Compute CER
    cer_output = jiwer.process_characters(gt_norm, tr_norm)
    cer = cer_output.cer

    return SampleMetrics(
        clip_id=clip_id,
        ground_truth=ground_truth,
        transcription=transcription,
        ground_truth_normalized=gt_norm,
        transcription_normalized=tr_norm,
        wer=wer,
        cer=cer,
        word_count=len(gt_norm.split()),
        char_count=len(gt_norm),
        substitutions=wer_output.substitutions,
        deletions=wer_output.deletions,
        insertions=wer_output.insertions,
    )


def compute_aggregate_metrics(
    sample_metrics: list[SampleMetrics],
    durations_ms: list[float | None],
    inference_times_ms: list[float],
    failed_count: int,
    encode_times_ms: list[float | None] | None = None,
    decode_times_ms: list[float | None] | None = None,
) -> AggregateMetrics:
    """Compute aggregate metrics across all samples."""
    if not sample_metrics:
        return AggregateMetrics(
            corpus_wer=0.0,
            corpus_cer=0.0,
            mean_wer=0.0,
            mean_cer=0.0,
            median_wer=0.0,
            median_cer=0.0,
            p90_wer=0.0,
            p90_cer=0.0,
            p95_wer=0.0,
            p95_cer=0.0,
            total_samples=failed_count,
            successful_samples=0,
            failed_samples=failed_count,
            total_words=0,
            total_chars=0,
            total_duration_ms=0.0,
            total_inference_ms=sum(inference_times_ms),
            real_time_factor=0.0,
            encode_time_ms_mean=None,
            encode_time_ms_median=None,
            decode_time_ms_mean=None,
            decode_time_ms_median=None,
        )

    # Corpus-level WER/CER (weighted by length)
    total_words = sum(m.word_count for m in sample_metrics)
    total_chars = sum(m.char_count for m in sample_metrics)

    # Collect all normalized texts for corpus-level computation
    all_gt = [m.ground_truth_normalized for m in sample_metrics if m.ground_truth_normalized]
    all_tr = [m.transcription_normalized for m in sample_metrics if m.ground_truth_normalized]

    if all_gt:
        corpus_wer = jiwer.wer(all_gt, all_tr)
        corpus_cer = jiwer.cer(all_gt, all_tr)
    else:
        corpus_wer = 0.0
        corpus_cer = 0.0

    # Per-sample WER/CER for percentile calculations
    wers = [m.wer for m in sample_metrics]
    cers = [m.cer for m in sample_metrics]

    mean_wer = sum(wers) / len(wers)
    mean_cer = sum(cers) / len(cers)

    sorted_wers = sorted(wers)
    sorted_cers = sorted(cers)

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])

    # Total audio duration
    valid_durations = [d for d in durations_ms if d is not None]
    total_duration_ms = sum(valid_durations) if valid_durations else 0.0

    total_inference_ms = sum(inference_times_ms)

    # Real-time factor (inference time / audio duration)
    real_time_factor = (
        total_inference_ms / total_duration_ms if total_duration_ms > 0 else 0.0
    )

    # Calculate encoder/decoder time statistics
    encode_time_ms_mean = None
    encode_time_ms_median = None
    decode_time_ms_mean = None
    decode_time_ms_median = None

    if encode_times_ms:
        valid_encode_times = [t for t in encode_times_ms if t is not None]
        if valid_encode_times:
            encode_time_ms_mean = sum(valid_encode_times) / len(valid_encode_times)
            sorted_encode_times = sorted(valid_encode_times)
            encode_time_ms_median = percentile(sorted_encode_times, 50)

    if decode_times_ms:
        valid_decode_times = [t for t in decode_times_ms if t is not None]
        if valid_decode_times:
            decode_time_ms_mean = sum(valid_decode_times) / len(valid_decode_times)
            sorted_decode_times = sorted(valid_decode_times)
            decode_time_ms_median = percentile(sorted_decode_times, 50)

    return AggregateMetrics(
        corpus_wer=corpus_wer,
        corpus_cer=corpus_cer,
        mean_wer=mean_wer,
        mean_cer=mean_cer,
        median_wer=percentile(sorted_wers, 50),
        median_cer=percentile(sorted_cers, 50),
        p90_wer=percentile(sorted_wers, 90),
        p90_cer=percentile(sorted_cers, 90),
        p95_wer=percentile(sorted_wers, 95),
        p95_cer=percentile(sorted_cers, 95),
        total_samples=len(sample_metrics) + failed_count,
        successful_samples=len(sample_metrics),
        failed_samples=failed_count,
        total_words=total_words,
        total_chars=total_chars,
        total_duration_ms=total_duration_ms,
        total_inference_ms=total_inference_ms,
        real_time_factor=real_time_factor,
        encode_time_ms_mean=encode_time_ms_mean,
        encode_time_ms_median=encode_time_ms_median,
        decode_time_ms_mean=decode_time_ms_mean,
        decode_time_ms_median=decode_time_ms_median,
    )
