from fastapi import FastAPI
from pydantic import BaseModel
import socket
import time
import torch
import uvicorn

app = FastAPI(title="Mini Distributed GPU Worker")

HOSTNAME = socket.gethostname()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
else:
    GPU_NAME = "CPU"
    VRAM_GB = 0


class Job(BaseModel):
    size: int = 4096


@app.get("/health")
def health():
    return {
        "hostname": HOSTNAME,
        "device": DEVICE.type,
        "gpu": GPU_NAME,
        "vram_gb": VRAM_GB,
    }


@app.post("/compute")
def compute(job: Job):
    size = job.size

    if size < 256 or size > 16384:
        return {"error": "size must be between 256 and 16384"}

    # Force synchronization so timing measures actual GPU work.
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    a = torch.randn((size, size), device=DEVICE, dtype=torch.float32)
    b = torch.randn((size, size), device=DEVICE, dtype=torch.float32)
    c = a @ b

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Prevent the optimizer/compiler from making this look like unused work.
    checksum = float(c[0, 0].item())

    del a, b, c
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "hostname": HOSTNAME,
        "device": DEVICE.type,
        "gpu": GPU_NAME,
        "vram_gb": VRAM_GB,
        "size": size,
        "seconds": round(elapsed, 4),
        "checksum": round(checksum, 5),
    }


if __name__ == "__main__":
    print(f"Worker: {HOSTNAME}")
    print(f"Device: {DEVICE}")
    print(f"GPU:    {GPU_NAME}")
    if VRAM_GB:
        print(f"VRAM:   {VRAM_GB} GB")
    print("Listening on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
