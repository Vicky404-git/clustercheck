# Mini Distributed GPU Cluster

Tiny proof-of-concept: one coordinator sends matrix multiplication jobs to multiple GPU workers over the LAN.

## 1. On each GPU laptop

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install a CUDA-enabled PyTorch build appropriate for the machine if the default `torch` package does not provide CUDA.

Check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Start the worker:

```bash
python worker.py
```

The worker listens on port `8000`.

## 2. Find each worker's LAN IP

Windows:

```powershell
ipconfig
```

Linux:

```bash
ip addr
```

Use the IPv4 address on the same LAN/Wi-Fi.

## 3. On the coordinator laptop

Edit `coordinator.py` and replace the example IPs in `WORKERS` with the actual worker IPs.

Then:

```bash
pip install requests
python coordinator.py
```

## 4. Watch the GPU

On a worker with NVIDIA drivers:

```bash
nvidia-smi -l 1
```

Run the coordinator while this is open. You should see GPU memory/utilization increase during the matrix multiplication.

## Important

Windows Firewall may block inbound TCP port 8000. Allow Python/port 8000 on the private network.

This is task-level distributed computing, not splitting one neural-network model across GPUs.
