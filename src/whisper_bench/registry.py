"""TOML-based dataset registry for named dataset references."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from .config import DatasetConfig

console = Console()


@dataclass
class DatasetEntry:
    """A dataset entry from the registry."""

    path: Path
    language: str | None = None
    format: str | None = None
    split: str | None = None
    path_column: str | None = None
    sentence_column: str | None = None
    audio_dir: str | None = None
    duration_file: str | None = None
    duration_column: str | None = None
    clip_id_column: str | None = None


REGISTRY_FILENAME = "datasets.toml"

SEARCH_PATHS = [
    Path.cwd(),
    Path.home() / ".config" / "whisper-bench",
]


def find_registry_file() -> Path | None:
    """Search for datasets.toml in known locations."""
    for directory in SEARCH_PATHS:
        candidate = directory / REGISTRY_FILENAME
        if candidate.exists():
            return candidate
    return None


def load_registry(path: Path) -> dict[str, DatasetEntry]:
    """Parse a datasets.toml file and return named entries."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    datasets_section = data.get("datasets", {})
    entries: dict[str, DatasetEntry] = {}

    for name, values in datasets_section.items():
        if not isinstance(values, dict):
            continue
        raw_path = values.get("path", "")
        if not raw_path:
            console.print(f"[yellow]Warning:[/yellow] Registry entry '{name}' has no path, skipping")
            continue
        entries[name] = DatasetEntry(
            path=Path(raw_path).expanduser(),
            language=values.get("language"),
            format=values.get("format"),
            split=values.get("split"),
            path_column=values.get("path_column"),
            sentence_column=values.get("sentence_column"),
            audio_dir=values.get("audio_dir"),
            duration_file=values.get("duration_file"),
            duration_column=values.get("duration_column"),
            clip_id_column=values.get("clip_id_column"),
        )

    return entries


def resolve_dataset(
    dataset_ref: str,
    split: str | None = None,
    language: str | None = None,
    path_column: str | None = None,
    sentence_column: str | None = None,
) -> tuple[dict, str | None]:
    """Resolve a dataset reference to a DatasetConfig and language.

    If dataset_ref is an existing directory path, use it directly (backward compatible).
    Otherwise, look it up in the registry.

    CLI overrides (split, language, path_column, sentence_column) take precedence
    over registry values.

    Returns (DatasetConfig kwargs dict, resolved language).
    """
    ref_path = Path(dataset_ref).expanduser()

    if ref_path.is_dir():
        # Direct path — backward compatible
        resolved_language = language
        config_kwargs: dict = {"dataset_path": ref_path}
        if split:
            config_kwargs["split"] = split
        if path_column:
            config_kwargs["path_column"] = path_column
        if sentence_column:
            config_kwargs["sentence_column"] = sentence_column
        return config_kwargs, resolved_language

    # Try registry lookup
    registry_file = find_registry_file()
    if registry_file is None:
        raise FileNotFoundError(
            f"'{dataset_ref}' is not a directory and no {REGISTRY_FILENAME} found.\n"
            f"Searched: {', '.join(str(p / REGISTRY_FILENAME) for p in SEARCH_PATHS)}"
        )

    registry = load_registry(registry_file)
    if dataset_ref not in registry:
        available = ", ".join(sorted(registry.keys())) or "(none)"
        raise KeyError(
            f"Dataset '{dataset_ref}' not found in {registry_file}.\n"
            f"Available datasets: {available}"
        )

    entry = registry[dataset_ref]
    if not entry.path.is_dir():
        raise FileNotFoundError(
            f"Registry entry '{dataset_ref}' points to '{entry.path}', "
            f"which does not exist or is not a directory."
        )

    # Build config kwargs — CLI overrides > registry values > defaults
    config_kwargs = {"dataset_path": entry.path}

    resolved_split = split or entry.split
    if resolved_split:
        config_kwargs["split"] = resolved_split

    resolved_language = language or entry.language

    # Column mappings: CLI override > registry > default
    pc = path_column or entry.path_column
    if pc:
        config_kwargs["path_column"] = pc
    sc = sentence_column or entry.sentence_column
    if sc:
        config_kwargs["sentence_column"] = sc
    if entry.audio_dir:
        config_kwargs["audio_dir"] = entry.audio_dir
    if entry.duration_file:
        config_kwargs["duration_file"] = entry.duration_file
    if entry.duration_column:
        config_kwargs["duration_column"] = entry.duration_column
    if entry.clip_id_column:
        config_kwargs["clip_id_column"] = entry.clip_id_column

    return config_kwargs, resolved_language
