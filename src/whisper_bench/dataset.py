"""Common Voice dataset loading and sampling."""

import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from .config import DatasetConfig

logger = logging.getLogger(__name__)
console = Console()

INLINE_DURATION_COLUMNS = ("duration_ms", "duration[ms]", "duration")


def infer_language(config: DatasetConfig) -> str | None:
    """Infer language from the locale column of the data file.

    Opens the data file, reads the first row, and returns the locale
    column value if present and non-empty.
    """
    try:
        data_file = config.data_file_path
    except FileNotFoundError:
        return None

    try:
        with open(data_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=config.delimiter)
            for row in reader:
                locale = row.get("locale", "").strip()
                return locale if locale else None
    except (OSError, csv.Error):
        return None
    return None


@dataclass
class Sample:
    """A single audio sample from the dataset."""

    clip_id: str
    audio_path: Path
    sentence: str
    duration_ms: int | None = None


@dataclass
class DatasetValidation:
    """Result of dataset validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_rows: int = 0
    audio_found: int = 0
    missing_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_dataset(config: DatasetConfig, max_missing_shown: int = 10) -> DatasetValidation:
    """Validate a dataset configuration and return structured results."""
    result = DatasetValidation()

    # Check clips directory
    clips_dir = config.clips_dir
    if not clips_dir.exists():
        result.errors.append(f"Audio directory not found: {clips_dir}")
    elif not clips_dir.is_dir():
        result.errors.append(f"Audio path is not a directory: {clips_dir}")

    # Check data file
    try:
        data_file = config.data_file_path
    except FileNotFoundError as e:
        result.errors.append(str(e))
        return result

    # Check columns in header
    with open(data_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=config.delimiter)
        headers = reader.fieldnames or []

        if config.path_column not in headers:
            available = ", ".join(headers)
            result.errors.append(
                f"Path column '{config.path_column}' not found in {data_file.name}. "
                f"Available columns: {available}"
            )
        if config.sentence_column not in headers:
            available = ", ".join(headers)
            result.errors.append(
                f"Sentence column '{config.sentence_column}' not found in {data_file.name}. "
                f"Available columns: {available}"
            )

        # If fatal column errors, skip file counting
        if result.errors:
            return result

        # Count audio files
        for row in reader:
            result.total_rows += 1
            clip_name = row.get(config.path_column, "")
            if not clip_name:
                continue
            audio_path = clips_dir / clip_name
            if audio_path.exists():
                result.audio_found += 1
            elif len(result.missing_files) < max_missing_shown:
                result.missing_files.append(clip_name)

    missing_count = result.total_rows - result.audio_found
    if missing_count > 0:
        result.warnings.append(
            f"{missing_count}/{result.total_rows} audio files not found in {clips_dir}"
        )

    # Check duration file — suppress warning if inline duration column exists
    if not config.durations_path.exists():
        has_inline_duration = any(col in headers for col in INLINE_DURATION_COLUMNS)
        if not has_inline_duration:
            result.warnings.append(
                f"Duration file not found: {config.durations_path.name} "
                f"(stratified sampling will fall back to random)"
            )

    return result


def load_durations(
    durations_path: Path,
    delimiter: str = "\t",
    clip_id_column: str | None = None,
    duration_column: str | None = None,
) -> dict[str, int]:
    """Load clip durations from TSV or CSV file."""
    durations: dict[str, int] = {}
    parse_errors = 0
    if not durations_path.exists():
        return durations

    with open(durations_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            # Use override columns if provided, else fall back to defaults
            if clip_id_column:
                clip_name = row.get(clip_id_column, "")
            else:
                clip_name = row.get("clip") or row.get("path", "")

            if duration_column:
                duration_str = row.get(duration_column, "0")
            else:
                duration_str = row.get("duration[ms]") or row.get("duration", "0")

            try:
                durations[clip_name] = int(float(duration_str))
            except (ValueError, TypeError):
                parse_errors += 1
                continue

    if parse_errors > 0:
        console.print(
            f"[yellow]Warning:[/yellow] {parse_errors} unparseable duration values in {durations_path.name}"
        )

    return durations


def load_samples(config: DatasetConfig) -> list[Sample]:
    """Load all samples from the dataset (CSV or TSV)."""
    # Detect delimiter for durations file based on its extension
    durations_delimiter = "," if config.durations_path.suffix == ".csv" else "\t"
    durations = load_durations(
        config.durations_path,
        delimiter=durations_delimiter,
        clip_id_column=config.clip_id_column,
        duration_column=config.duration_column,
    )

    samples: list[Sample] = []
    skipped_empty = 0
    skipped_missing = 0
    with open(config.data_file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=config.delimiter)
        headers = reader.fieldnames or []

        # If no external durations, check for inline duration column
        inline_duration_col: str | None = None
        if not durations:
            for col in INLINE_DURATION_COLUMNS:
                if col in headers:
                    inline_duration_col = col
                    break

        for row in reader:
            clip_name = row.get(config.path_column, "")
            sentence = row.get(config.sentence_column, "")

            if not clip_name or not sentence:
                skipped_empty += 1
                continue

            audio_path = config.clips_dir / clip_name
            if not audio_path.exists():
                skipped_missing += 1
                continue

            # Resolve duration: external file first, then inline column
            duration_ms = durations.get(clip_name)
            if duration_ms is None and inline_duration_col:
                raw = row.get(inline_duration_col, "")
                if raw:
                    try:
                        val = float(raw)
                        duration_ms = int(val)
                    except (ValueError, TypeError):
                        pass

            samples.append(
                Sample(
                    clip_id=clip_name,
                    audio_path=audio_path,
                    sentence=sentence,
                    duration_ms=duration_ms,
                )
            )

    if skipped_empty > 0:
        console.print(
            f"[yellow]Warning:[/yellow] Skipped {skipped_empty} rows with empty path/sentence columns"
        )
    if skipped_missing > 0:
        console.print(
            f"[yellow]Warning:[/yellow] Skipped {skipped_missing} samples with missing audio files"
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
        console.print(
            "[yellow]Warning:[/yellow] No duration info available — falling back to random sampling"
        )
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
        raise ValueError(f"No valid samples found in {config.data_file_path}")

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
