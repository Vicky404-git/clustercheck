import concurrent.futures
import time
import requests

# CHANGE THESE IPs to your teammates' laptop IPs.
WORKERS = [
    "192.168.1.20:8000",  # RTX 4050 #1
    "192.168.1.21:8000",  # RTX 4050 #2
    "192.168.1.22:8000",  # RTX 5050
]

SIZE = 4096
TIMEOUT = 120


def get_info(worker):
    try:
        r = requests.get(f"http://{worker}/health", timeout=5)
        r.raise_for_status()
        return worker, r.json()
    except Exception as e:
        return worker, {"error": str(e)}


def run_job(worker):
    start = time.perf_counter()
    try:
        r = requests.post(
            f"http://{worker}/compute",
            json={"size": SIZE},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        result = r.json()
        result["network_seconds"] = round(time.perf_counter() - start, 4)
        return worker, result
    except Exception as e:
        return worker, {"error": str(e)}


def bar(seconds, max_seconds, width=24):
    filled = max(1, int((1 - seconds / max_seconds) * width)) if max_seconds else width
    filled = max(1, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def main():
    print("\n=== MINI DISTRIBUTED GPU CLUSTER ===\n")
    print(f"Matrix workload: {SIZE} x {SIZE}\n")

    print("Discovering workers...\n")
    online = []

    for worker in WORKERS:
        address, info = get_info(worker)
        if "error" in info:
            print(f"  ✗ {address:<22} OFFLINE  {info['error']}")
            continue

        online.append(address)
        print(
            f"  ✓ {address:<22} {info['gpu']} "
            f"({info['vram_gb']} GB VRAM)"
        )

    if not online:
        print("\nNo workers available.")
        return

    print(f"\nDispatching {len(online)} GPU jobs...\n")

    total_start = time.perf_counter()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(online)) as pool:
        futures = [pool.submit(run_job, worker) for worker in online]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    total = time.perf_counter() - total_start

    valid = [r for _, r in results if "error" not in r]

    print("Results:\n")
    if valid:
        max_time = max(r["seconds"] for r in valid)
    else:
        max_time = 1

    for worker, result in sorted(results):
        if "error" in result:
            print(f"  ✗ {worker:<22} FAILED: {result['error']}")
            continue

        print(
            f"  ✓ {worker:<22} "
            f"{result['gpu']:<20} "
            f"{bar(result['seconds'], max_time)} "
            f"{result['seconds']:.3f}s"
        )

    print("\n---------------------------------------------")
    print(f"Workers used:       {len(valid)}/{len(online)}")
    print(f"Distributed time:   {total:.3f}s")

    if valid:
        slowest = max(r["seconds"] for r in valid)
        print(f"Slowest GPU:        {slowest:.3f}s")
        print(f"Total GPU work:     {sum(r['seconds'] for r in valid):.3f}s")
        print("\nAll workers computed independently on their GPUs.")
        print("Run `nvidia-smi` on a worker while this runs to see GPU usage.")


if __name__ == "__main__":
    main()
