# Cloud Performance Monitor - Mini Project

Monitors machine utilisation from five sources (local PC, remote PC, AWS
CloudWatch, demo data, Kaggle VM dataset), flags unusual spikes with a
scikit-learn **Isolation Forest**, predicts **time until 90% utilisation**
with a linear regression on percentage metrics only, and logs every alert to
a local **SQLite** database. UI is **Streamlit**, plus a pure-CLI `debug.py`.

Every metric is displayed with its real unit - percentages (`%`), but also
the Kaggle dataset's absolute values: power (`W`), executed instructions
(`ops`), execution time (`ms`) and network throughput (`bytes/sec`).

## Team structure (separation of concerns)

| Part | Owner | Files |
|------|-------|-------|
| UI / CLI |  `app.py` (Streamlit), `debug.py` (CLI) |
| Data ingestion + metric registry |  `backend/ingestion/` - `local_pc.py` (psutil + HTTP agent), `aws_ingestion.py` (boto3), `datasets.py` (demo + Kaggle CSV), `__init__.py` (source dispatcher + unit registry) |
| ML engine + DB |  `backend/ml.py` (Isolation Forest + regression), `backend/storage.py` (SQLite) |

```
.
├── app.py                     # Streamlit UI
├── debug.py                   # CLI to test every function
├── backend/
│   ├── __init__.py
│   ├── ingestion/             #
│   │   ├── __init__.py        # source dispatcher + metric name/unit registry
│   │   ├── local_pc.py        # psutil collector + HTTP agent for remote PCs
│   │   ├── aws_ingestion.py   # boto3 CloudWatch (plug credentials later)
│   │   └── datasets.py        # demo generator + Kaggle vmCloud CSV reader
│   ├── ml.py                  # Isolation Forest + linear regression
│   └── storage.py             # SQLite alerts table
├── tests/                     # offline tests (no AWS needed)
├── requirements.txt
└── cloud_monitor.db           # created automatically
```

## Setup and run

With **uv** (what this repo is set up for):

```bash
uv sync
uv run streamlit run app.py
```

Or with plain **pip**:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

`pyproject.toml` is the source of truth for dependencies; regenerate the pip
requirements file after changing it with:

```bash
uv export --no-hashes --no-emit-project -o requirements.txt
```

Open http://localhost:8501, pick a **Data Source** (Local PC / AWS (Live) /
Demo / Kaggle Dataset (CSV) / Remote PC (HTTP)), a metric, then
**Fetch & Analyze**.

## Data sources

**Local PC** - `backend/ingestion/local_pc.py` polls `psutil` for CPU, RAM,
Disk and Network, and reads NVIDIA GPU utilisation through `GPUtil` when the
hardware/driver exists (otherwise the GPU metric is simply hidden).

**Remote PC** - on the other machine run:

```bash
python -m backend.ingestion.local_pc --port 8777
```

then in the UI choose `Remote PC (HTTP)` and enter that machine's IP. The
agent serves one JSON snapshot at `http://<ip>:8777/metrics`; the client polls
it to build the same series shape as every other source.

**AWS (Live)** - `backend/ingestion/aws_ingestion.py` calls
`cloudwatch:GetMetricStatistics` for `CPUUtilization`, `NetworkIn`,
`NetworkOut`. Credentials come from `aws configure` or the usual environment
variables. IAM needs `cloudwatch:GetMetricStatistics` (and
`ec2:DescribeInstances` only to fill the instance dropdown).

**Demo** - generated locally, no credentials:
- CPU: idle baseline with random periodic spikes
- RAM: slow upward leak
- Network In/Out: bursty, mostly quiet with sudden multipliers

**Kaggle Dataset (CSV)** - `backend/ingestion/datasets.py` reads the Kaggle
*VM Workload* dataset (`vmCloud_data.csv`, ~2M rows, gitignored) from the
project root. Six metrics are exposed with their real units:

| pipeline metric | CSV column | unit |
|-----------------|------------|------|
| `cpu` | `cpu_usage` | % |
| `ram` | `memory_usage` | % |
| `network` | `network_traffic` | bytes/sec |
| `power` | `power_consumption` | W |
| `instructions` | `num_executed_instructions` | ops |
| `execution_time` | `execution_time` | ms |

Timestamps are parsed, null values and bad rows are dropped, then the rows
are sorted chronologically and the most recent N are returned (the sidebar
sets the row count, default 100). The file can also be placed anywhere - the
sidebar takes a custom CSV path.

## Metric units

`METRIC_SPECS` in `backend/ingestion/__init__.py` is the single source of
truth for every metric's name and unit. Both the Streamlit metric cards and
every `debug.py` table read it from there, so units are shown consistently
everywhere. Values are displayed at full useful precision (integers with
thousands separators, fractions up to 6 decimals, never scientific
notation), so absolute metrics are not rounded away.

## ML

- **Anomaly detection** (`backend/ml.py`): Isolation Forest trained
  **in-memory on every call** and discarded - no joblib, no model file on disk.
  Roughly the most unusual 5% of points are flagged, their anomaly scores are
  shown in the UI/CLI for *every* metric (Watts, ops, ms included), and the
  flagged rows are written to SQLite. Min/avg/max stats likewise work for all
  metrics.
- **Trend prediction** (`backend/ml.py`): linear regression over the recent
  history of a **percentage** metric (CPU/RAM/Disk/GPU), returning the slope,
  R2 and an ETA for crossing 90% utilisation (or "not projected" when the
  trend is falling). Non-percentage metrics are safely bypassed: the
  predictor reports `supported: false` with a reason instead of inventing an
  ETA for Watts or instructions.

## CLI testing (no UI needed)

```bash
python debug.py metrics                          # metrics per source, with units
python debug.py local --duration 10              # psutil snapshot + series
python debug.py remote --host 192.168.1.50       # poll a friend's PC
python debug.py aws --instance i-0123456789abcdef0
python debug.py demo --metric cpu --hours 6
python debug.py kaggle --metric power --limit 100      # Watts from the CSV
python debug.py kaggle --metric instructions --limit 50 # ops from the CSV
python debug.py anomaly --source kaggle --metric power --save  # scores in W
python debug.py predict --source kaggle --metric power # bypass: not a %
python debug.py predict --source demo --metric ram
python debug.py db-test                          # insert + read back
python debug.py db-show --limit 10
python debug.py db-clear
```

## Tests

```bash
python -m pytest tests/ -q
```

## Out of scope

- Multi-cloud abstractions (AWS only, alongside local monitoring)
- Forecasting beyond the single linear-trend ETA
- Capacity planning reports
- Background schedulers / cron jobs
