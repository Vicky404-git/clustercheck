import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args, env=None):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "debug.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=full_env,
    )


def test_debug_help_lists_every_layer():
    result = run("--help")
    assert result.returncode == 0
    for command in ("local", "remote", "aws", "demo", "anomaly", "predict", "db-test"):
        assert command in result.stdout


def test_debug_metrics_and_demo():
    metrics = run("metrics")
    assert metrics.returncode == 0
    assert "CPUUtilization" in metrics.stdout

    demo = run("demo", "--metric", "cpu", "--hours", "1")
    assert demo.returncode == 0
    assert "CPU Utilization" in demo.stdout

    predict = run("predict", "--source", "demo", "--metric", "ram")
    assert predict.returncode == 0
    assert "trend" in predict.stdout


def test_debug_db_flow(tmp_path):
    db = {"CLOUD_MONITOR_DB": str(tmp_path / "cli_test.db")}

    test = run("db-test", env=db)
    assert test.returncode == 0
    assert "inserted" in test.stdout

    show = run("db-show", "--limit", "5", env=db)
    assert show.returncode == 0
    assert "debug-db-test" in show.stdout

    clear = run("db-clear", env=db)
    assert clear.returncode == 0
    assert "0 remaining" in clear.stdout
