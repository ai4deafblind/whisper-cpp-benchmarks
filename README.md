# whisper-cpp-benchmarks

CLI tool to benchmark [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) transcription accuracy against [Mozilla Common Voice](https://commonvoice.mozilla.org/) datasets.

Measures Word Error Rate (WER) and Character Error Rate (CER) with configurable sampling strategies.

## Prerequisites

### whisper.cpp

Clone and build whisper.cpp:

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build --config Release
```

The `whisper-cli` binary will be at `build/bin/whisper-cli`. Either add it to your PATH or pass the full path with `--whisper-cli`.

See the [whisper.cpp README](https://github.com/ggerganov/whisper.cpp#readme) for platform-specific build options (CUDA, Metal, OpenBLAS, etc.).

### Models

Download GGML models using the built-in script:

```bash
cd whisper.cpp
bash models/download-ggml-model.sh base
```

Or download directly from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main):

```bash
# Example: download ggml-base model
curl -L -o models/ggml-base.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

Available model sizes: `tiny`, `base`, `small`, `medium`, `large-v3`.

### Common Voice Dataset

1. Create an account at [Mozilla Common Voice](https://commonvoice.mozilla.org/)
2. Download a dataset for your target language from the [datasets page](https://commonvoice.mozilla.org/en/datasets)
3. Extract the archive — the expected directory layout depends on the version:

**Common Voice v19 (TSV format):**
```
cv-corpus-19.0-2024-09-13/id/
├── clips/              # Audio files (.mp3)
├── test.tsv            # Tab-separated: path, sentence, ...
├── dev.tsv
├── train.tsv
├── validated.tsv
└── clip_durations.tsv  # Optional: clip, duration[ms]
```

**Common Voice v24 (CSV format):**
```
commonvoice-v24/
├── audio_files/        # Audio files (.mp3)
├── test.csv            # Comma-separated
└── clip_durations.csv  # Optional
```

See [`dataset-examples/`](dataset-examples/) for example directory structures of each format with sample data files.

### Python & uv

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

## Installation

```bash
uv sync
```

For NVIDIA GPU monitoring:

```bash
uv sync --extra nvidia
```

## Quickstart

End-to-end example from zero to first benchmark result:

```bash
# 1. Install whisper-bench
cd whisper-cpp-benchmarks
uv sync

# 2. Validate your dataset
whisper-bench validate -d ~/datasets/cv-corpus-19.0-2024-09-13/id/

# 3. Run a quick benchmark (10 samples)
whisper-bench run \
  -m ~/whisper.cpp/models/ggml-base.bin \
  -d ~/datasets/cv-corpus-19.0-2024-09-13/id/ \
  -n 10

# 4. Check results
cat benchmarks/*/summary.json | python -m json.tool
```

## Dataset Configuration

### Direct path

Point directly to a dataset directory:

```bash
whisper-bench run -m MODEL -d /path/to/dataset/
```

### Dataset registry (`datasets.toml`)

For frequently used datasets, create a `datasets.toml` file to avoid typing long paths:

```bash
cp datasets.toml.example datasets.toml
# Edit with your dataset paths
```

The tool searches for `datasets.toml` in:
1. Current directory (`./datasets.toml`)
2. User config (`~/.config/whisper-bench/datasets.toml`)

Then reference datasets by name:

```bash
whisper-bench run -m MODEL -d cv19-id
whisper-bench validate -d cv24-id
```

**Registry format:**

```toml
[datasets.cv19-id]
path = "~/datasets/cv-corpus-19.0-2024-09-13/id"
language = "id"
split = "test"

[datasets.cv24-id]
path = "~/datasets/commonvoice-v24"
language = "id"
format = "csv"
path_column = "file_path"
sentence_column = "transcript"
audio_dir = "audio_files"
```

**Available fields:**

| Field | Description | Default |
|-------|-------------|---------|
| `path` | Dataset directory path (required) | — |
| `language` | Language code | `id` |
| `format` | `tsv` or `csv` | auto-detected |
| `split` | Dataset split | `test` |
| `path_column` | Column name for audio filenames | `path` |
| `sentence_column` | Column name for reference text | `sentence` |
| `audio_dir` | Audio files subdirectory | auto (`clips/` or `audio_files/`) |
| `duration_file` | Duration file name | auto (`clip_durations.tsv/.csv`) |
| `duration_column` | Duration column name | `duration[ms]` or `duration` |
| `clip_id_column` | Clip ID column in duration file | `clip` or `path` |

### Column mapping

For datasets with non-standard column names, use CLI options or registry fields:

```bash
# CLI override
whisper-bench run -m MODEL -d ./my-dataset/ \
  --path-column file_path --sentence-column transcript

# Or set in datasets.toml (see above)
```

## Dataset Validation

Check your dataset before running a benchmark:

```bash
whisper-bench validate -d ./datasets/id/
```

Output includes:
- Data file and audio directory paths
- Column availability check
- Audio file coverage (X/Y files found)
- First 10 missing file paths (if any)
- Duration file status

```bash
# Validate with custom columns
whisper-bench validate -d ./my-dataset/ \
  --path-column file_path --sentence-column transcript
```

## Usage

```bash
whisper-bench run -m MODEL -d DATASET [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model PATH` | Whisper GGML model path | Required |
| `-d, --dataset TEXT` | Dataset directory path or registry name | Required |
| `-l, --language TEXT` | Language code | from registry or `id` |
| `-s, --split` | Dataset split: `test`, `dev`, `train`, `validated` | `test` |
| `-n, --samples INT` | Number of samples | `100` |
| `--strategy` | Sampling: `stratified`, `random`, `sequential`, `all` | `stratified` |
| `--seed INT` | Random seed | `42` |
| `-t, --threads INT` | Thread count | CPU/2 |
| `-bs, --beam-size INT` | Beam size for beam search | `5` |
| `-o, --output PATH` | Output directory | `benchmarks` |
| `--no-gpu` | Disable GPU acceleration | |
| `--run-name TEXT` | Custom run name | |
| `--no-monitoring` | Disable hardware monitoring | |
| `--monitor-interval INT` | Hardware sampling interval (ms) | `100` |
| `--auto-detect-language` | Enable automatic language detection | |
| `--suppress-non-speech` | Suppress non-speech tokens | |
| `--no-speech-threshold FLOAT` | No-speech probability threshold | `0.60` |
| `--vad` | Enable Voice Activity Detection | |
| `--vad-model PATH` | VAD model path (required with `--vad`) | |
| `--vad-threshold FLOAT` | VAD threshold | `0.50` |
| `--vad-min-speech-duration-ms INT` | Minimum speech duration (ms) | `250` |
| `--vad-min-silence-duration-ms INT` | Minimum silence duration (ms) | `100` |
| `--vad-max-speech-duration-s FLOAT` | Maximum speech duration (s) | |
| `--vad-speech-pad-ms INT` | Speech padding (ms) | `30` |
| `--vad-samples-overlap FLOAT` | Samples overlap | `0.10` |
| `--whisper-cli TEXT` | Path to whisper-cli binary | `whisper-cli` |
| `--path-column TEXT` | Column name for audio file paths | `path` |
| `--sentence-column TEXT` | Column name for reference text | `sentence` |

### Examples

```bash
# Quick test with 10 samples
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/ -n 10

# Full benchmark with 100 stratified samples
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/

# Use a registry name
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d cv19-id

# Use all samples from test split
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/ --strategy all

# Custom output directory and run name
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/ -o results --run-name baseline

# Auto-detect language
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/ --auto-detect-language

# With non-speech suppression
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/ \
    --suppress-non-speech --no-speech-threshold 0.5

# With VAD (Voice Activity Detection)
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/ \
    --vad --vad-model ~/whisper.cpp/models/ggml-silero-v5.1.2.bin \
    --vad-threshold 0.6

# Custom beam size (default is 5, use 1 for greedy decoding)
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./datasets/id/ --beam-size 3

# Dataset with custom column names
whisper-bench run -m ~/whisper.cpp/models/ggml-base.bin -d ./my-dataset/ \
    --path-column file_path --sentence-column transcript
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
  "encode_time_ms": 858.63,
  "decode_time_ms": 375.87,
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
- **Encode/Decode Time** - Breakdown of encoder (audio → latent) and decoder (latent → text) timing

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

## Text Normalization

For fair comparison, both ground truth and transcriptions are normalized:

- Unicode NFC normalization
- Lowercase
- Remove punctuation
- Collapse whitespace

## Platform Notes

### macOS (Apple Silicon)

whisper.cpp uses Metal (GPU) acceleration by default on Apple Silicon. Use `--no-gpu` to force CPU-only inference. Hardware monitoring has limited support — CPU temperature is not available.

### Linux x86

For CUDA support, build whisper.cpp with `-DGGML_CUDA=ON`. Install `pynvml` (`uv sync --extra nvidia`) for GPU monitoring.

### Raspberry Pi 5

Build whisper.cpp with default settings. Use `--no-gpu` and consider smaller models (`tiny`, `base`). CPU temperature monitoring is supported via `/sys/class/thermal`.

## Troubleshooting

**`whisper-cli not found in PATH`**
- Build whisper.cpp first, then either add `build/bin/` to your PATH or pass `--whisper-cli /path/to/whisper-cli`.

**`No data file found for split 'test'`**
- Check that the split file (e.g., `test.tsv` or `test.csv`) exists in the dataset directory. Use `whisper-bench validate -d DATASET` to diagnose.

**`No valid samples found`**
- Audio files may be missing. Run `whisper-bench validate -d DATASET` to see how many files were found vs expected.

**`Path column 'path' not found`**
- Your dataset uses different column names. Run `whisper-bench validate` to see available columns, then use `--path-column` and `--sentence-column` to override.

**`Dataset 'xxx' not found in datasets.toml`**
- Check that `datasets.toml` exists in the current directory or `~/.config/whisper-bench/`. Run with `-d /full/path/` instead if you don't need the registry.

**Stratified sampling falls back to random**
- The duration file (`clip_durations.tsv` or `.csv`) is missing. This is optional — random sampling still works. Generate durations with ffprobe if needed.

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
