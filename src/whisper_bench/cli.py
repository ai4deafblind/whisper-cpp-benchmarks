"""Click CLI interface for whisper-bench."""

import os
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
from .dataset import get_samples
from .metrics import (
    AggregateMetrics,
    SampleMetrics,
    compute_aggregate_metrics,
    compute_sample_metrics,
)
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
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Common Voice dataset directory",
)
@click.option(
    "-l", "--language",
    default="id",
    help="Language code [default: id]",
)
@click.option(
    "-s", "--split",
    type=click.Choice(["test", "dev", "train", "validated"]),
    default="test",
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
def run(
    model: Path,
    dataset: Path,
    language: str,
    split: str,
    samples: int,
    strategy: str,
    seed: int,
    threads: int | None,
    output: Path,
    no_gpu: bool,
    run_name: str | None,
    no_monitoring: bool,
    monitor_interval: int,
) -> None:
    """Run benchmark against Common Voice dataset."""
    try:
        # Build configuration
        whisper_kwargs = {
            "model_path": model,
            "language": language,
            "no_gpu": no_gpu,
        }
        if threads is not None:
            whisper_kwargs["threads"] = threads

        whisper_config = WhisperConfig(**whisper_kwargs)
        dataset_config = DatasetConfig(
            dataset_path=dataset,
            split=split,
            sample_size=samples,
            sampling_strategy=strategy,
            seed=seed,
        )
        config = BenchmarkConfig(
            whisper=whisper_config,
            dataset=dataset_config,
            output_dir=output,
            run_name=run_name,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

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
    console.print(f"[bold]Split:[/bold] {split}")
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

    console.print(f"Loaded [bold]{len(dataset_samples)}[/bold] samples")
    console.print()

    # Run transcription
    sample_metrics: list[SampleMetrics] = []
    hardware_metrics_list: list[SampleHardwareMetrics] = []
    durations: list[float | None] = []
    inference_times: list[float] = []
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
    )

    # Add hardware metrics to aggregate if available
    if hardware_metrics_list:
        aggregate.hardware = compute_aggregate_hardware_metrics(hardware_metrics_list)

    # Write summary
    write_summary(summary_path, config, aggregate, system_info=system_info)

    # Print summary
    print_summary(aggregate, console)

    console.print(f"[dim]Results saved to {run_dir}[/dim]")


if __name__ == "__main__":
    main()
