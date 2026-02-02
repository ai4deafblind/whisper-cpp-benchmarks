"""Common Voice dataset loading and sampling."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from .config import DatasetConfig, SamplingStrategy


@dataclass
class Sample:
    """A single audio sample from the dataset."""

    clip_id: str
    audio_path: Path
    sentence: str
    duration_ms: int | None = None


def load_durations(durations_path: Path) -> dict[str, int]:
    """Load clip durations from clip_durations.tsv."""
    durations: dict[str, int] = {}
    if not durations_path.exists():
        return durations

    with open(durations_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Handle both "clip" and "path" column names
            clip_name = row.get("clip") or row.get("path", "")
            duration_str = row.get("duration[ms]") or row.get("duration", "0")
            try:
                durations[clip_name] = int(float(duration_str))
            except (ValueError, TypeError):
                continue

    return durations


def load_samples(config: DatasetConfig) -> list[Sample]:
    """Load all samples from the dataset TSV."""
    durations = load_durations(config.durations_path)

    samples: list[Sample] = []
    with open(config.tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip_name = row.get("path", "")
            sentence = row.get("sentence", "")

            if not clip_name or not sentence:
                continue

            audio_path = config.clips_dir / clip_name
            if not audio_path.exists():
                continue

            samples.append(
                Sample(
                    clip_id=clip_name,
                    audio_path=audio_path,
                    sentence=sentence,
                    duration_ms=durations.get(clip_name),
                )
            )

    return samples


def sample_stratified(
    samples: list[Sample], n: int, seed: int
) -> list[Sample]:
    """Sample stratified by duration buckets (<3s, 3-7s, >7s)."""
    rng = random.Random(seed)

    # Split into buckets
    short: list[Sample] = []  # <3s
    medium: list[Sample] = []  # 3-7s
    long: list[Sample] = []  # >7s
    unknown: list[Sample] = []  # no duration info

    for sample in samples:
        if sample.duration_ms is None:
            unknown.append(sample)
        elif sample.duration_ms < 3000:
            short.append(sample)
        elif sample.duration_ms <= 7000:
            medium.append(sample)
        else:
            long.append(sample)

    # If no duration info, fall back to random sampling
    if not short and not medium and not long:
        return sample_random(samples, n, seed)

    # Calculate proportional allocation
    buckets = [b for b in [short, medium, long] if b]
    total_with_duration = sum(len(b) for b in buckets)

    result: list[Sample] = []
    remaining = n

    for i, bucket in enumerate(buckets):
        if i == len(buckets) - 1:
            # Last bucket gets remaining slots
            count = remaining
        else:
            # Proportional allocation
            count = round(n * len(bucket) / total_with_duration)
            count = min(count, remaining, len(bucket))

        rng.shuffle(bucket)
        result.extend(bucket[:count])
        remaining -= count

    # Fill remaining slots from unknown if needed
    if remaining > 0 and unknown:
        rng.shuffle(unknown)
        result.extend(unknown[:remaining])

    rng.shuffle(result)
    return result


def sample_random(samples: list[Sample], n: int, seed: int) -> list[Sample]:
    """Random sampling with seed."""
    rng = random.Random(seed)
    n = min(n, len(samples))
    return rng.sample(samples, n)


def sample_sequential(samples: list[Sample], n: int) -> list[Sample]:
    """Take first N samples."""
    return samples[:n]


def get_samples(config: DatasetConfig) -> list[Sample]:
    """Load and sample dataset according to configuration."""
    all_samples = load_samples(config)

    if not all_samples:
        raise ValueError(f"No valid samples found in {config.tsv_path}")

    strategy = config.sampling_strategy
    n = config.sample_size

    if strategy == "all":
        return all_samples
    elif strategy == "stratified":
        return sample_stratified(all_samples, n, config.seed)
    elif strategy == "random":
        return sample_random(all_samples, n, config.seed)
    elif strategy == "sequential":
        return sample_sequential(all_samples, n)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
