# whisper-cpp-benchmarks

CLI tool to benchmark [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) transcription accuracy against [Mozilla Common Voice](https://commonvoice.mozilla.org/) datasets.

Measures Word Error Rate (WER) and Character Error Rate (CER) with configurable sampling strategies.

## Installation

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
whisper-bench run -m MODEL -d DATASET [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model PATH` | Whisper GGML model path | Required |
| `-d, --dataset PATH` | Common Voice dataset directory | Required |
| `-l, --language TEXT` | Language code | `id` |
| `-s, --split` | Dataset split: `test`, `dev`, `train`, `validated` | `test` |
| `-n, --samples INT` | Number of samples | `100` |
| `--strategy` | Sampling: `stratified`, `random`, `sequential`, `all` | `stratified` |
| `--seed INT` | Random seed | `42` |
| `-t, --threads INT` | Thread count | CPU/2 |
| `-o, --output PATH` | Output directory | `benchmarks` |
| `--no-gpu` | Disable GPU acceleration | |
| `--run-name TEXT` | Custom run name | |
| `--no-monitoring` | Disable hardware monitoring | |
| `--monitor-interval INT` | Hardware sampling interval (ms) | `100` |

### Examples

```bash
# Quick test with 10 samples
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/ -n 10

# Full benchmark with 100 stratified samples
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/

# Use all samples from test split
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/ --strategy all

# Custom output directory and run name
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/ -o results --run-name baseline
```

## Output

Each benchmark run creates a directory with:

- `results.jsonl` - Per-sample results (streaming, survives partial runs)
- `summary.json` - Aggregate metrics and configuration

### Sample Result (JSONL)

```json
{
  "clip_id": "common_voice_id_12345.mp3",
  "ground_truth": "Original text",
  "transcription": "Transcribed text",
  "wer": 0.25,
  "cer": 0.10,
  "inference_time_ms": 1234.5,
  "duration_ms": 3000,
  "hardware": {
    "cpu_percent_mean": 78.5,
    "cpu_percent_max": 95.2,
    "memory_rss_peak_mb": 1245.3,
    "cpu_temp_max_c": 65.0
  }
}
```

### Summary Metrics

- **Corpus WER/CER** - Weighted by sample length (standard metric)
- **Mean/Median WER/CER** - Per-sample statistics
- **P90/P95 WER** - Percentile metrics for outlier analysis
- **Real-time Factor** - Inference time / audio duration (lower is faster)

### Hardware Metrics

When monitoring is enabled (default), the summary includes:

- **System Info** - OS, CPU model, cores, memory, GPU (if NVIDIA)
- **CPU Usage** - Mean and peak utilization across all samples
- **Memory (RSS)** - Mean and peak resident set size
- **CPU Temperature** - Maximum temperature during inference
- **GPU Metrics** - Utilization, memory, temperature (NVIDIA only)

Hardware monitoring works on x86 Linux and Raspberry Pi 5. Use `--no-monitoring` to disable.

## Sampling Strategies

| Strategy | Description |
|----------|-------------|
| `stratified` | Balanced sampling across duration buckets (<3s, 3-7s, >7s) |
| `random` | Random sampling with seed |
| `sequential` | First N samples |
| `all` | Complete split |

## Dataset Structure

Expects Mozilla Common Voice format:

```
dataset/
├── clips/           # Audio files (.mp3)
├── test.tsv         # Tab-separated: path, sentence, ...
├── dev.tsv
├── train.tsv
├── validated.tsv
└── clip_durations.tsv  # Optional: clip, duration[ms]
```

## Text Normalization

For fair comparison, both ground truth and transcriptions are normalized:

- Unicode NFC normalization
- Lowercase
- Remove punctuation
- Collapse whitespace

## Dependencies

- [jiwer](https://github.com/jitsi/jiwer) - WER/CER calculation
- [click](https://click.palletsprojects.com/) - CLI framework
- [rich](https://rich.readthedocs.io/) - Progress bars and console output
- [psutil](https://github.com/giampaolo/psutil) - Hardware monitoring

### Optional

For NVIDIA GPU monitoring:

```bash
uv sync --extra nvidia
```
