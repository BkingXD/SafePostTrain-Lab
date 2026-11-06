"""CUDA-only runtime selection and hardware preflight."""
import os
import platform
import sys

import torch


def resolve_device(requested="cuda"):
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError("CUDA GPU training is required. Supported devices: cuda or cuda:N")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA build of PyTorch and check nvidia-smi / GPU access. "
            "Training requires an accessible CUDA GPU."
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} does not exist")
    # Must be configured before CUDA BLAS initializes for deterministic matmul.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise RuntimeError("Deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    device = torch.device("cuda", index)
    # Detect unusable kernels/architecture, not just a visible device name.
    try:
        probe = torch.ones((2, 2), device=device)
        (probe @ probe).sum().item()
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError(f"CUDA preflight failed on {device}: {exc}") from exc
    return device


def configure_runtime(device, seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment(device):
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "platform": platform.system(), "device": str(device),
            "torch_threads": torch.get_num_threads(), "precision": "float32",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_runtime": torch.version.cuda, "tf32": False,
            "timing": "CUDA synchronized wall clock"}
    props = torch.cuda.get_device_properties(device)
    info.update({"gpu_name": props.name, "gpu_total_memory_bytes": props.total_memory,
                 "compute_capability": f"{props.major}.{props.minor}",
                 "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]})
    return info
