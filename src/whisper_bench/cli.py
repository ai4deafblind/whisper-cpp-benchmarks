"""Click CLI interface for whisper-bench."""

import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .config import BenchmarkConfig, DatasetConfig, WhisperConfig
from .dataset import get_samples, infer_language, validate_dataset
from .metrics import (
    SampleMetrics,
    compute_aggregate_metrics,
    compute_sample_metrics,
)
from .registry import resolve_dataset
from .report import (
    print_summary,
    write_failed_result,
    write_sample_result,
    write_summary,
)
from .runner import run_whisper
from .system import (
    SampleHardwareMetrics,
    SystemInfo,
    compute_aggregate_hardware_metrics,
    get_system_info,
)

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Whisper.cpp benchmarking tool for Common Voice datasets."""
    pass


@main.command()
@click.option(
    "-m", "--model",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Whisper GGML model path",
)
@click.option(
    "-d", "--dataset",
    type=str,
    required=True,
    help="Dataset directory path or registry name from datasets.toml",
)
@click.option(
    "-l", "--language",
    default=None,
    help="Language code [default: from registry or 'id']",
)
@click.option(
    "-s", "--split",
    type=click.Choice(["test", "dev", "train", "validated"]),
    default=None,
    help="Dataset split [default: test]",
)
@click.option(
    "-n", "--samples",
    type=int,
    default=100,
    help="Number of samples [default: 100]",
)
@click.option(
    "--strategy",
    type=click.Choice(["stratified", "random", "sequential", "all"]),
    default="stratified",
    help="Sampling strategy [default: stratified]",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed [default: 42]",
)
@click.option(
    "-t", "--threads",
    type=int,
    default=None,
    help="Thread count [default: CPU/2]",
)
@click.option(
    "-bs", "--beam-size",
    type=int,
    default=None,
    help="Beam size for beam search [default: 5]",
)
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    default=Path("benchmarks"),
    help="Output directory [default: benchmarks]",
)
@click.option(
    "--no-gpu",
    is_flag=True,
    help="Disable GPU acceleration",
)
@click.option(
    "--run-name",
    type=str,
    default=None,
    help="Custom run name for output directory",
)
@click.option(
    "--no-monitoring",
    is_flag=True,
    help="Disable hardware monitoring",
)
@click.option(
    "--monitor-interval",
    type=int,
    default=100,
    help="Hardware monitoring sampling interval in ms [default: 100]",
)
@click.option(
    "--auto-detect-language",
    is_flag=True,
    help="Enable automatic language detection (passes -l auto)",
)
@click.option(
    "--suppress-non-speech",
    is_flag=True,
    help="Suppress non-speech tokens",
)
@click.option(
    "--no-speech-threshold",
    type=float,
    default=None,
    help="No-speech threshold [default: 0.60]",
)
@click.option(
    "--vad",
    is_flag=True,
    help="Enable Voice Activity Detection",
)
@click.option(
    "--vad-model",
    type=click.Path(exists=True, path_type=Path),
    help="VAD model path (required when --vad is enabled)",
)
@click.option(
    "--vad-threshold",
    type=float,
    default=None,
    help="VAD threshold [default: 0.50]",
)
@click.option(
    "--vad-min-speech-duration-ms",
    type=int,
    default=None,
    help="Min speech duration in ms [default: 250]",
)
@click.option(
    "--vad-min-silence-duration-ms",
    type=int,
    default=None,
    help="Min silence duration in ms [default: 100]",
)
@click.option(
    "--vad-max-speech-duration-s",
    type=float,
    default=None,
    help="Max speech duration in seconds",
)
@click.option(
    "--vad-speech-pad-ms",
    type=int,
    default=None,
    help="Speech padding in ms [default: 30]",
)
@click.option(
    "--vad-samples-overlap",
    type=float,
    default=None,
    help="Samples overlap [default: 0.10]",
)
@click.option(
    "--whisper-cli",
    type=str,
    default="whisper-cli",
    help="Path to whisper-cli binary [default: whisper-cli (uses PATH)]",
)
@click.option(
    "--path-column",
    type=str,
    default=None,
    help="Column name for audio file paths [default: path]",
)
@click.option(
    "--sentence-column",
    type=str,
    default=None,
    help="Column name for reference text [default: sentence]",
)
def run(
    model: Path,
    dataset: str,
    language: str | None,
    split: str | None,
    samples: int,
    strategy: str,
    seed: int,
    threads: int | None,
    beam_size: int | None,
    output: Path,
    no_gpu: bool,
    run_name: str | None,
    no_monitoring: bool,
    monitor_interval: int,
    auto_detect_language: bool,
    suppress_non_speech: bool,
    no_speech_threshold: float | None,
    vad: bool,
    vad_model: Path | None,
    vad_threshold: float | None,
    vad_min_speech_duration_ms: int | None,
    vad_min_silence_duration_ms: int | None,
    vad_max_speech_duration_s: float | None,
    vad_speech_pad_ms: int | None,
    vad_samples_overlap: float | None,
    whisper_cli: str,
    path_column: str | None,
    sentence_column: str | None,
) -> None:
    """Run benchmark against Common Voice dataset."""
    try:
        # Resolve dataset reference (path or registry name)
        try:
            dataset_kwargs, resolved_language = resolve_dataset(
                dataset,
                split=split,
                language=language,
                path_column=path_column,
                sentence_column=sentence_column,
            )
        except (FileNotFoundError, KeyError) as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)

        # Merge resolved kwargs with sampling options
        dataset_kwargs["sample_size"] = samples
        dataset_kwargs["sampling_strategy"] = strategy
        dataset_kwargs["seed"] = seed
        # Apply default split if not set by CLI or registry
        if "split" not in dataset_kwargs:
            dataset_kwargs["split"] = "test"

        dataset_config = DatasetConfig(**dataset_kwargs)

        # Infer language if not explicitly set (CLI -l or registry)
        if not resolved_language:
            resolved_language = infer_language(dataset_config)
            if not resolved_language:
                console.print(
                    "[yellow]Warning:[/yellow] Could not detect language from dataset, defaulting to 'id'"
                )
                resolved_language = "id"

        # Build whisper configuration
        whisper_kwargs: dict = {
            "model_path": model,
            "language": resolved_language,
            "no_gpu": no_gpu,
            "auto_detect_language": auto_detect_language,
            "suppress_non_speech": suppress_non_speech,
            "vad_enabled": vad,
            "whisper_cli": whisper_cli,
        }
        if threads is not None:
            whisper_kwargs["threads"] = threads
        if beam_size is not None:
            whisper_kwargs["beam_size"] = beam_size
        if no_speech_threshold is not None:
            whisper_kwargs["no_speech_threshold"] = no_speech_threshold
        if vad_model is not None:
            whisper_kwargs["vad_model_path"] = vad_model
        if vad_threshold is not None:
            whisper_kwargs["vad_threshold"] = vad_threshold
        if vad_min_speech_duration_ms is not None:
            whisper_kwargs["vad_min_speech_duration_ms"] = vad_min_speech_duration_ms
        if vad_min_silence_duration_ms is not None:
            whisper_kwargs["vad_min_silence_duration_ms"] = vad_min_silence_duration_ms
        if vad_max_speech_duration_s is not None:
            whisper_kwargs["vad_max_speech_duration_s"] = vad_max_speech_duration_s
        if vad_speech_pad_ms is not None:
            whisper_kwargs["vad_speech_pad_ms"] = vad_speech_pad_ms
        if vad_samples_overlap is not None:
            whisper_kwargs["vad_samples_overlap"] = vad_samples_overlap

        whisper_config = WhisperConfig(**whisper_kwargs)

        config = BenchmarkConfig(
            whisper=whisper_config,
            dataset=dataset_config,
            output_dir=output,
            run_name=run_name,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    # Pre-run validation
    validation = validate_dataset(dataset_config)
    for error in validation.errors:
        console.print(f"[red]Error:[/red] {error}")
    for warning in validation.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    if not validation.ok:
        raise SystemExit(1)
    if validation.total_rows > 0:
        console.print(
            f"Found [bold]{validation.audio_found}/{validation.total_rows}[/bold] "
            f"audio files in {dataset_config.clips_dir}"
        )

    # Create output directory
    run_dir = config.get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    # Collect system info if monitoring is enabled
    system_info: SystemInfo | None = None
    monitor_resources = not no_monitoring
    if monitor_resources:
        system_info = get_system_info()

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Dataset:[/bold] {dataset}")
    console.print(f"[bold]Split:[/bold] {dataset_config.split}")
    console.print(f"[bold]Strategy:[/bold] {strategy}")
    console.print(f"[bold]Output:[/bold] {run_dir}")
    if system_info:
        console.print(f"[bold]CPU:[/bold] {system_info.cpu.model}")
        console.print(f"[bold]Memory:[/bold] {system_info.memory_total_gb:.1f} GB")
        if system_info.gpu:
            console.print(f"[bold]GPU:[/bold] {system_info.gpu.name}")
    console.print()

    # Load samples
    with console.status("Loading dataset..."):
        try:
            dataset_samples = get_samples(dataset_config)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)

    loaded_count = len(dataset_samples)
    console.print(f"Loaded [bold]{loaded_count}[/bold] samples")
    if strategy == "all" and validation.audio_found > 0 and loaded_count < validation.audio_found:
        diff = validation.audio_found - loaded_count
        console.print(
            f"[yellow]Warning:[/yellow] {diff} fewer samples than expected "
            f"(empty columns or missing audio)"
        )
    console.print()

    # Run transcription
    sample_metrics: list[SampleMetrics] = []
    hardware_metrics_list: list[SampleHardwareMetrics] = []
    durations: list[float | None] = []
    inference_times: list[float] = []
    encode_times: list[float | None] = []
    decode_times: list[float | None] = []
    failed_count = 0

    with tempfile.TemporaryDirectory() as work_dir:
        work_path = Path(work_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Transcribing...", total=len(dataset_samples))

            for sample in dataset_samples:
                result = run_whisper(
                    sample,
                    whisper_config,
                    work_path,
                    monitor_resources=monitor_resources,
                    monitor_interval_ms=monitor_interval,
                )
                inference_times.append(result.inference_time_ms)
                durations.append(sample.duration_ms)
                encode_times.append(result.encode_time_ms)
                decode_times.append(result.decode_time_ms)

                # Collect hardware metrics if available
                if result.hardware_metrics:
                    hardware_metrics_list.append(result.hardware_metrics)

                if result.success:
                    metrics = compute_sample_metrics(
                        clip_id=sample.clip_id,
                        ground_truth=sample.sentence,
                        transcription=result.transcription,
                    )
                    sample_metrics.append(metrics)
                    write_sample_result(
                        results_path,
                        metrics,
                        result.inference_time_ms,
                        sample.duration_ms,
                        hardware_metrics=result.hardware_metrics,
                        encode_time_ms=result.encode_time_ms,
                        decode_time_ms=result.decode_time_ms,
                    )
                else:
                    failed_count += 1
                    write_failed_result(
                        results_path,
                        sample.clip_id,
                        result.error or "Unknown error",
                        result.inference_time_ms,
                    )

                progress.advance(task)

    # Compute aggregate metrics
    aggregate = compute_aggregate_metrics(
        sample_metrics,
        durations,
        inference_times,
        failed_count,
        encode_times,
        decode_times,
    )

    # Add hardware metrics to aggregate if available
    if hardware_metrics_list:
        aggregate.hardware = compute_aggregate_hardware_metrics(hardware_metrics_list)

    # Write summary
    write_summary(summary_path, config, aggregate, system_info=system_info)

    # Print summary
    print_summary(aggregate, console)

    console.print(f"[dim]Results saved to {run_dir}[/dim]")


@main.command()
@click.option(
    "-d", "--dataset",
    type=str,
    required=True,
    help="Dataset directory path or registry name from datasets.toml",
)
@click.option(
    "-s", "--split",
    type=click.Choice(["test", "dev", "train", "validated"]),
    default=None,
    help="Dataset split [default: test]",
)
@click.option(
    "--path-column",
    type=str,
    default=None,
    help="Column name for audio file paths [default: path]",
)
@click.option(
    "--sentence-column",
    type=str,
    default=None,
    help="Column name for reference text [default: sentence]",
)
def validate(
    dataset: str,
    split: str | None,
    path_column: str | None,
    sentence_column: str | None,
) -> None:
    """Validate a dataset configuration and report issues."""
    try:
        dataset_kwargs, resolved_language = resolve_dataset(
            dataset,
            split=split,
            language=None,
            path_column=path_column,
            sentence_column=sentence_column,
        )
    except (FileNotFoundError, KeyError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    # Apply default split if not set
    if "split" not in dataset_kwargs:
        dataset_kwargs["split"] = "test"

    try:
        dataset_config = DatasetConfig(**dataset_kwargs)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    # Infer language if not explicitly set (registry)
    if not resolved_language:
        resolved_language = infer_language(dataset_config)
        if not resolved_language:
            console.print(
                "[yellow]Warning:[/yellow] Could not detect language from dataset, defaulting to 'id'"
            )
            resolved_language = "id"

    console.print(f"[bold]Dataset path:[/bold] {dataset_config.dataset_path}")
    console.print(f"[bold]Language:[/bold] {resolved_language}")
    console.print(f"[bold]Split:[/bold] {dataset_config.split}")

    try:
        console.print(f"[bold]Data file:[/bold] {dataset_config.data_file_path}")
    except FileNotFoundError:
        pass  # validate_dataset will catch this

    console.print(f"[bold]Audio directory:[/bold] {dataset_config.clips_dir}")
    console.print(f"[bold]Path column:[/bold] {dataset_config.path_column}")
    console.print(f"[bold]Sentence column:[/bold] {dataset_config.sentence_column}")
    console.print()

    result = validate_dataset(dataset_config)

    # Print errors
    for error in result.errors:
        console.print(f"[red]FAIL[/red] {error}")

    # Print warnings
    for warning in result.warnings:
        console.print(f"[yellow]WARN[/yellow] {warning}")

    # Print audio coverage
    if result.total_rows > 0:
        console.print(
            f"\n[bold]Audio coverage:[/bold] {result.audio_found}/{result.total_rows} files found"
        )

    # Print missing files
    if result.missing_files:
        console.print(f"\n[bold]Missing files (first {len(result.missing_files)}):[/bold]")
        for f in result.missing_files:
            console.print(f"  {f}")

    # Duration file status
    if dataset_config.durations_path.exists():
        console.print(f"\n[bold]Duration file:[/bold] {dataset_config.durations_path} [green]found[/green]")
    else:
        # Check for inline duration column in data file
        from .dataset import INLINE_DURATION_COLUMNS
        try:
            with open(dataset_config.data_file_path, newline="", encoding="utf-8") as f:
                import csv as _csv
                reader = _csv.DictReader(f, delimiter=dataset_config.delimiter)
                data_headers = reader.fieldnames or []
            inline_col = next((c for c in INLINE_DURATION_COLUMNS if c in data_headers), None)
            if inline_col:
                console.print(f"\n[bold]Duration source:[/bold] inline column '{inline_col}' [green]found[/green]")
            else:
                console.print(f"\n[bold]Duration file:[/bold] {dataset_config.durations_path.name} [yellow]not found[/yellow]")
        except (FileNotFoundError, OSError):
            console.print(f"\n[bold]Duration file:[/bold] {dataset_config.durations_path.name} [yellow]not found[/yellow]")

    # Overall result
    console.print()
    if result.ok:
        console.print("[green bold]PASS[/green bold] Dataset validation passed.")
    else:
        console.print("[red bold]FAIL[/red bold] Dataset validation failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
