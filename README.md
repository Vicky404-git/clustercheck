# Cloud Performance Monitor - Mini Project

Monitors machine utilisation from three sources (local PC, AWS CloudWatch, demo
data), flags unusual spikes with a scikit-learn **Isolation Forest**, predicts
**time until 90% utilisation** with a linear regression, and logs every alert to
a local **SQLite** database. UI is **Streamlit**, plus a pure-CLI `debug.py`.

## Team structure (separation of concerns)

| Part | Owner | Files |
|------|-------|-------|
| UI / CLI | Person A | `app.py` (Streamlit), `debug.py` (CLI) |
| Data ingestion | Person B | `backend/ingestion/` - `local_pc.py` (psutil + GPUtil), `agent.py` (HTTP agent), `aws_ingestion.py` (boto3), `demo_data.py` |
| ML engine + DB | Person C | `backend/ml/` - `anomaly.py`, `predictor.py`; `backend/storage/database.py` |

```
.
├── app.py                     # Streamlit UI (Person A)
├── debug.py                   # CLI to test every function (Person A)
├── backend/
│   ├── ingestion/             # Person B
│   │   ├── local_pc.py        # psutil: CPU, RAM, Disk, Network, GPU (GPUtil)
│   │   ├── agent.py           # HTTP endpoint to expose a PC to the network
│   │   ├── aws_ingestion.py   # boto3 CloudWatch (plug credentials later)
│   │   └── demo_data.py       # spiky CPU, leaking RAM, bursty network
│   ├── ml/                    # Person C
│   │   ├── anomaly.py         # Isolation Forest (in-memory, no joblib)
│   │   └── predictor.py       # linear regression -> time until 90%
│   └── storage/               # Person C
│       └── database.py        # SQLite alerts table
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
Demo / Remote PC (HTTP)), a metric, then **Fetch & Analyze**.

## Data sources

**Local PC** - `backend/ingestion/local_pc.py` polls `psutil` for CPU, RAM,
Disk and Network, and reads NVIDIA GPU utilisation through `GPUtil` when the
hardware/driver exists (otherwise the GPU metric is simply hidden).

**Remote PC** - on the other machine run:

```bash
python -m backend.ingestion.agent --port 8777
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

## ML

- **Anomaly detection** (`backend/ml/anomaly.py`): Isolation Forest trained
  **in-memory on every call** and discarded - no joblib, no model file on disk.
  Roughly the most unusual 5% of points are flagged and written to SQLite.
- **Trend prediction** (`backend/ml/predictor.py`): linear regression over the
  recent history of a 0-100% metric, returning the slope, R2 and an ETA for
  crossing 90% utilisation (or "not projected" when the trend is falling).

## CLI testing (no UI needed)

```bash
python debug.py metrics                          # metrics per source
python debug.py local --duration 10              # psutil snapshot + series
python debug.py remote --host 192.168.1.50       # poll a friend's PC
python debug.py aws --instance i-0123456789abcdef0
python debug.py demo --metric cpu --hours 6
python debug.py anomaly --source demo --metric cpu --save
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
