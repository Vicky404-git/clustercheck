"""EC2 metrics from AWS CloudWatch via boto3 .

Kept separate from the local/psutil collector so credentials can be plugged
in later without touching the rest of the pipeline.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

AWS_METRIC_NAMES = ["CPUUtilization", "NetworkIn", "NetworkOut"]

# short timeouts so the UI never hangs when the network or credentials
# are unavailable
CLIENT_CONFIG = Config(
    connect_timeout=5,
    read_timeout=10,
    retries={"max_attempts": 1},
)


def fetch_metric(
    instance_id: str,
    metric_name: str,
    region: str,
    hours: int = 6,
    period: int = 300,
) -> List[Dict[str, Any]]:
    """Pull one EC2 metric from CloudWatch for the given instance."""
    if metric_name not in AWS_METRIC_NAMES:
        raise RuntimeError(f"Unsupported metric: {metric_name}")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)

    try:
        client = boto3.client("cloudwatch", region_name=region, config=CLIENT_CONFIG)
        response = client.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName=metric_name,
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=["Average"],
        )
    except NoCredentialsError:
        raise RuntimeError(
            "AWS credentials not found. Run `aws configure` or set "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or use Demo/Local mode."
        )
    except (ClientError, BotoCoreError) as exc:
        logger.error("CloudWatch request failed: %s", exc)
        raise RuntimeError(f"CloudWatch request failed: {exc}") from exc

    points = [
        {"timestamp": dp["Timestamp"], "value": float(dp["Average"])}
        for dp in response.get("Datapoints", [])
        if "Average" in dp
    ]
    points.sort(key=lambda p: p["timestamp"])

    if not points:
        raise RuntimeError(
            f"No datapoints returned for {metric_name} on {instance_id} "
            f"in {region} over the last {hours} hour(s). Check the instance "
            "ID, region, and that the instance has been running."
        )
    return points


def list_instances(region: str) -> List[str]:
    """Best-effort list of EC2 instance IDs for the sidebar dropdown."""
    try:
        client = boto3.client("ec2", region_name=region, config=CLIENT_CONFIG)
        response = client.describe_instances()
    except Exception as exc:  # noqa: BLE001 - UI helper, never break the app
        logger.warning("Could not list instances: %s", exc)
        return []

    ids = []
    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            ids.append(instance["InstanceId"])
    return sorted(ids)
