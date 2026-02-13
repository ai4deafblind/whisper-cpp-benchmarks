# Example Dataset Structures

Example directory layouts showing every supported dataset format. Audio directories contain `.gitkeep` placeholders — replace with real audio files to run benchmarks.

## Directory Layout

```
dataset-examples/
├── cv17-id/              # Common Voice v17 (TSV, clips/)
│   ├── clips/
│   │   └── .gitkeep
│   ├── test.tsv
│   ├── dev.tsv
│   ├── validated.tsv
│   └── clip_durations.tsv
│
├── cv19-id/              # Common Voice v19 (TSV, clips/, sentence_id)
│   ├── clips/
│   │   └── .gitkeep
│   ├── test.tsv
│   ├── dev.tsv
│   ├── train.tsv
│   ├── validated.tsv
│   └── clip_durations.tsv
│
├── cv24-en/              # Common Voice v24 (CSV, audio_files/, single file)
│   ├── audio_files/
│   │   └── .gitkeep
│   └── commonvoice-v24_en.csv
│
├── cv24-flat-csv/        # Common Voice v24 (CSV, audio_files/, split files)
│   ├── audio_files/
│   │   └── .gitkeep
│   ├── test.csv
│   ├── dev.csv
│   └── clip_durations.csv
│
└── custom-csv/           # Custom dataset (non-standard columns, wavs/)
    ├── wavs/
    │   └── .gitkeep
    ├── test.csv
    └── durations.csv
```

## Format Differences

| Format | File type | Audio dir | Columns | Duration |
|--------|-----------|-----------|---------|----------|
| **CV v17** | TSV | `clips/` | `path`, `sentence` | Separate `clip_durations.tsv` with `clip`, `duration[ms]` |
| **CV v19** | TSV | `clips/` | `path`, `sentence`, `sentence_id`, `sentence_domain`, `variant` | Separate `clip_durations.tsv` with `clip`, `duration[ms]` |
| **CV v24 (single)** | CSV | `audio_files/` | `path`, `sentence`, `duration_ms` (inline) | Embedded in main CSV |
| **CV v24 (split)** | CSV | `audio_files/` | `path`, `sentence`, `duration_ms` (inline) | Separate `clip_durations.csv` with `path`, `duration` |
| **Custom** | CSV | `wavs/` | `file_path`, `transcript` | Separate `durations.csv` with `filename`, `length_ms` |

## Usage with whisper-bench

```bash
# CV v17/v19 — auto-detected, works out of the box
whisper-bench validate -d dataset-examples/cv19-id/

# CV v24 single file — auto-detected
whisper-bench validate -d dataset-examples/cv24-en/

# CV v24 split files — auto-detected
whisper-bench validate -d dataset-examples/cv24-flat-csv/

# Custom — requires column mapping
whisper-bench validate -d dataset-examples/custom-csv/ \
  --path-column file_path --sentence-column transcript
```

Or configure in `datasets.toml`:

```toml
[datasets.custom-example]
path = "dataset-examples/custom-csv"
language = "id"
path_column = "file_path"
sentence_column = "transcript"
audio_dir = "wavs"
duration_file = "durations.csv"
duration_column = "length_ms"
clip_id_column = "filename"
```
